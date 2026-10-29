# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Physical PDF pages as revision-stable, retrievable evidence Sources."""

from __future__ import annotations

import hashlib
import io
import math
from collections import Counter
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.search import analyze_text
from powercontext.builtin.runtime.models import SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import SourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str


class PDFPageSource(Source):
    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class _PDFTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _PDFPageSourceDefinition:
    name = "pdf-page"
    version = "1"
    input_class = PDFPageInput
    source_class = PDFPageSource
    projections: tuple[SourceProjection[PDFPageSource], ...] = (_PDFTextEvidenceProjection(),)

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            definition_version=self.version,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = (
    _PDFPageSourceDefinition()
)

_PDF_QUERY_STOPWORDS = frozenset(
    "a an and are as at be by can for from give has have if in is it less not of on or only reported "
    "relying should shown state that the then these this to what when where which with within would".split()
)
_PDF_QUERY_EXPANSIONS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (
        frozenset({"geography", "geographies", "geographic", "operate", "operates", "operating"}),
        ("countries international markets regions north america united states"),
    ),
    (
        frozenset({"legal", "battle", "battles", "lawsuit", "lawsuits", "litigation"}),
        ("litigation legal proceedings claims contingencies"),
    ),
    (
        frozenset({"restructuring", "restructure"}),
        ("restructuring impairment charges productivity costs"),
    ),
    (
        frozenset({"ebitda", "capex", "capital", "depreciation", "amortization", "margin"}),
        ("operating income depreciation amortization capital expenditures cash flows"),
    ),
)


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int, /) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be non-empty")
    if not isinstance(document_sha256, str) or len(document_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in document_sha256
    ):
        raise ValueError("document_sha256 must be a lowercase SHA-256 digest")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"PDF page source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Extract every physical page, returning no partial result on failure."""

    if not isinstance(value.document_id, str) or not value.document_id.strip():
        raise ValueError("document_id must be non-empty")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("PDF requires a non-empty password")
        digest = hashlib.sha256(value.content).hexdigest()
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=index,
                text=page.extract_text() or "",
            )
            for index, page in enumerate(reader.pages, start=1)
        )
        return pages
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF could not be fully parsed") from error


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Search page evidence with lexical expansion and rank the full candidate pool."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    candidate_limit = min(max(limit * 16, 256), 512)
    queries = [question]
    question_terms = tuple(dict.fromkeys(analyze_text(question).split()))
    query_terms = set(question_terms)
    expanded_terms: Counter[str] = Counter()
    for triggers, expansion in _PDF_QUERY_EXPANSIONS:
        if query_terms.intersection(triggers):
            terms = expansion.split()
            expanded_terms.update(terms)

    retrieval_terms = tuple(
        term
        for term in (*question_terms, *expanded_terms.keys())
        if term not in _PDF_QUERY_STOPWORDS and len(term) > 1
    )
    retrieval_terms = tuple(dict.fromkeys(retrieval_terms))
    queries.extend(
        " ".join(retrieval_terms[index : index + 2])
        for index in range(0, len(retrieval_terms), 2)
    )

    candidates: dict[tuple[str, str, str, str], tuple[MemoryHit, int]] = {}
    for query in queries:
        result = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=candidate_limit))
        for rank, hit in enumerate(result.hits, start=1):
            key = (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id, hit.entry_version_id)
            previous = candidates.get(key)
            if previous is None:
                candidates[key] = (hit, rank)
            else:
                candidates[key] = (previous[0], min(previous[1], rank))

    ranked = _rank_pdf_candidates(question, expanded_terms, tuple(candidates.values()))
    return tuple(
        MemoryCitation(memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id)
        for hit in ranked[:limit]
    )


def _rank_pdf_candidates(
    question: str,
    expanded_terms: Counter[str],
    candidates: tuple[tuple[MemoryHit, int], ...],
) -> tuple[MemoryHit, ...]:
    if not candidates:
        return ()

    original_terms = tuple(
        term
        for term in analyze_text(question).split()
        if term not in _PDF_QUERY_STOPWORDS and len(term) > 1
    )
    weights = {term: float(frequency) for term, frequency in Counter(original_terms).items()}
    for term, frequency in expanded_terms.items():
        weights[term] = max(weights.get(term, 0.0), 0.45 * frequency)

    tokenized = [Counter(analyze_text(hit.text).split()) for hit, _rank in candidates]
    document_frequency = Counter(term for frequencies in tokenized for term in frequencies.keys())
    average_length = max(1.0, sum(sum(frequencies.values()) for frequencies in tokenized) / len(tokenized))
    total_documents = len(candidates)
    scored: list[tuple[float, int, MemoryHit]] = []
    for (hit, rank), frequencies in zip(candidates, tokenized, strict=True):
        document_length = sum(frequencies.values())
        score = 0.0
        matched_original: set[str] = set()
        for term, weight in weights.items():
            frequency = frequencies[term]
            if not frequency:
                continue
            if term in original_terms:
                matched_original.add(term)
            document_frequency_for_term = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (total_documents - document_frequency_for_term + 0.5) / (document_frequency_for_term + 0.5)
            )
            denominator = frequency + 1.2 * (0.25 + 0.75 * document_length / average_length)
            score += weight * inverse_document_frequency * frequency * 2.2 / denominator
        if original_terms:
            score += 0.35 * len(matched_original) / len(set(original_terms))
        scored.append((score, rank, hit))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(hit for _score, _rank, hit in scored)


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDF_PAGE_SOURCE_DEFINITION",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
