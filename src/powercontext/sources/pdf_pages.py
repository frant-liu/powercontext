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
import re
import unicodedata
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


class _PDFInputError(ValueError):
    def __init__(self, code: str) -> None:
        messages = {
            "document-id": "document_id must be non-empty",
            "digest": "document_sha256 must be a lowercase SHA-256 digest",
            "page": "page_number must be a positive integer",
            "identity": "PDF page number exceeds the identity size limit",
            "password": "PDF requires a non-empty password",
            "parse": "PDF could not be fully parsed",
            "question": "question must be non-empty",
            "limit": "limit must be a positive integer",
        }
        super().__init__(messages[code])


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


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = _PDFPageSourceDefinition()

_PDF_QUERY_STOPWORDS = frozenset([
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "for",
    "from",
    "give",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "its",
    "less",
    "not",
    "of",
    "on",
    "or",
    "only",
    "reported",
    "relying",
    "should",
    "shown",
    "state",
    "that",
    "the",
    "then",
    "these",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "within",
    "would",
    "answer",
    "respond",
    "response",
    "question",
    "calculate",
    "define",
    "assuming",
    "perspective",
    "investment",
    "analyst",
    "use",
    "detail",
    "directly",
    "explicitly",
    "outlined",
    "quantity",
    "usd",
    "million",
    "billion",
    "fiscal",
    "fy",
])
_PDF_QUERY_EXPANSIONS: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset({"geography", "geographic", "operate", "country", "region"}),
        "geographic geography countries international markets regions segments operations",
    ),
    (
        frozenset({"legal", "battle", "lawsuit", "litigation"}),
        "litigation legal proceedings claims contingencies",
    ),
    (
        frozenset({"restructuring", "restructure"}),
        "restructuring impairment charges productivity costs",
    ),
    (
        frozenset({"ebitda", "ebit", "profit", "income", "earning", "margin"}),
        "income earnings profit operating revenue sales",
    ),
    (
        frozenset({"ebitda", "depreciation", "amortization"}),
        "depreciation amortization cash flows",
    ),
    (
        frozenset({"capex", "expenditure", "spending"}),
        "capital expenditures spending purchases property plant equipment cash flows",
    ),
    (
        frozenset({"revenue", "sale"}),
        "revenue revenues sales income",
    ),
    (
        frozenset({"asset", "liability", "equity", "liquidity", "balance"}),
        "balance sheets financial position assets liabilities equity",
    ),
)
_PDF_STATEMENT_TITLES = {
    "income": re.compile(
        r"(?:(?:consolidated|combined|condensed|unaudited)\s+)*"
        r"(?:statements?\s+of\s+(?:income|operations|earnings|profit(?:\s+and\s+loss)?)|"
        r"(?:income|profit\s+and\s+loss)\s+statements?)"
    ),
    "cash": re.compile(
        r"(?:(?:consolidated|combined|condensed|unaudited)\s+)*"
        r"(?:statements?\s+of\s+cash\s+flows?|cash\s+flows?\s+statements?)"
    ),
    "balance": re.compile(
        r"(?:(?:consolidated|combined|condensed|unaudited)\s+)*"
        r"(?:balance\s+sheets?|statements?\s+of\s+financial\s+position)"
    ),
}
_PDF_SECTION_TITLES = (
    (
        frozenset({"geography", "geographic", "operate", "country", "region"}),
        re.compile(
            r"(?:our\s+)?(?:operations|business|segments|reportable segments|operating segments|"
            r"geographic(?:al)? (?:information|areas|segments)|company overview)"
        ),
    ),
    (
        frozenset({"legal", "litigation", "lawsuit", "battle"}),
        re.compile(r"(?:legal proceedings|litigation|commitments and contingencies|contingencies)"),
    ),
)


def _pdf_terms(text: str) -> tuple[str, ...]:
    """Normalize retrieval terms only; never change the captured page text."""

    text = re.sub(r"\bfy\s*(\d{4})\b", r"\1", unicodedata.normalize("NFKC", text).casefold())
    terms = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return tuple(
        term[:-3] + "y"
        if len(term) > 4 and term.endswith("ies")
        else term[:-1]
        if len(term) > 4 and term.endswith("s") and not term.endswith("ss")
        else term
        for term in terms
    )


def _requested_statements(terms: set[str]) -> set[str]:
    kinds = set()
    if terms & {"income", "earning", "profit", "ebitda", "ebit", "margin", "revenue", "sale"}:
        kinds.add("income")
    if terms & {"cash", "capex", "depreciation", "amortization"}:
        kinds.add("cash")
    if terms & {"asset", "liability", "equity", "liquidity", "balance"}:
        kinds.add("balance")
    return kinds


def _page_statements(text: str) -> set[str]:
    # Titles must be standalone headings near the top, rather than prose, an
    # exhibit reference or a contents entry ending in a printed page number.
    found = set()
    for line in text.splitlines()[1:16]:
        if re.search(r"\d", line):
            continue
        line = " ".join(re.findall(r"[a-z]+", line.casefold()))
        for kind, pattern in _PDF_STATEMENT_TITLES.items():
            match = pattern.fullmatch(line)
            if match is None:
                match = pattern.fullmatch(re.sub(r"\s+(?:continued|unaudited)$", "", line))
            if match is not None:
                found.add(kind)
    # A directory listing multiple statement families is not a statement page.
    return found if len(found) == 1 and len(re.findall(r"\d[\d,.]*", text)) >= 12 else set()


def _has_requested_section(text: str, terms: set[str]) -> bool:
    for triggers, pattern in _PDF_SECTION_TITLES:
        if not triggers.intersection(terms):
            continue
        for line in text.splitlines()[1:]:
            # SEC item/note numbers are headings, while a trailing page number
            # indicates a directory entry. Do not promote cross references.
            line = re.sub(r"^(?:item|note)\s+\d+[a-z]?\s*[.\-—:]?\s*", "", line.casefold().strip())
            if pattern.fullmatch(line.strip(" .:")):
                return True
    return False


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int, /) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise _PDFInputError("document-id")
    if (
        not isinstance(document_sha256, str)
        or len(document_sha256) != 64
        or any(char not in "0123456789abcdef" for char in document_sha256)
    ):
        raise _PDFInputError("digest")
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise _PDFInputError("page")
    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        # Keep revision and physical page explicit while bounding arbitrary
        # caller-supplied document names. The distinct prefix avoids ambiguity
        # with the reversible form used by existing citations.
        document_key = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
        source_id = f"pdf-sha256:{document_key}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise _PDFInputError("identity")
    return source_id


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Extract every physical page, returning no partial result on failure."""

    if not isinstance(value.document_id, str) or not value.document_id.strip():
        raise _PDFInputError("document-id")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise _PDFInputError("password")  # noqa: TRY301
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
    except ValueError:
        raise
    except Exception as error:
        raise _PDFInputError("parse") from error
    else:
        return pages


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Retrieve exact Memory citations, including tables needed by financial queries."""

    if not isinstance(question, str) or not question.strip():
        raise _PDFInputError("question")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise _PDFInputError("limit")
    candidate_limit = min(max(limit * 16, 256), 512)
    queries = [question]
    question_terms = tuple(dict.fromkeys(_pdf_terms(question)))
    query_terms = set(question_terms)
    expanded_terms: set[str] = set()
    expansion_words: list[str] = []
    for triggers, expansion in _PDF_QUERY_EXPANSIONS:
        if query_terms.intersection(triggers):
            expansion_words.extend(expansion.split())
            expanded_terms.update(_pdf_terms(expansion))

    retrieval_terms = tuple(
        term
        for term in (*analyze_text(question).split(), *question_terms, *expansion_words)
        if term not in _PDF_QUERY_STOPWORDS and len(term) > 1
    )
    retrieval_terms = tuple(dict.fromkeys(retrieval_terms))
    queries.extend(" ".join(retrieval_terms[index : index + 2]) for index in range(0, len(retrieval_terms), 2))

    # Short queries bypass long-query minimum coverage, which otherwise drops
    # tables using a different accounting label (e.g. profit versus income).
    for kind in sorted(_requested_statements(query_terms)):
        queries.extend(
            {
                "income": ("statement income", "statements operations", "statements earnings"),
                "cash": ("statement cash",),
                "balance": ("balance sheet", "financial position"),
            }[kind]
        )

    candidates: dict[tuple[str, int, str, str], tuple[MemoryHit, int]] = {}
    for query in dict.fromkeys(queries):
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
    expanded_terms: set[str],
    candidates: tuple[tuple[MemoryHit, int], ...],
) -> tuple[MemoryHit, ...]:
    if not candidates:
        return ()

    original_terms = {term for term in _pdf_terms(question) if term not in _PDF_QUERY_STOPWORDS and len(term) > 1}
    weights = dict.fromkeys(sorted(original_terms), 1.0)
    for term in sorted(expanded_terms):
        weights.setdefault(term, 0.35)

    scores = _pdf_lexical_scores(tuple(hit.text for hit, _rank in candidates), weights, original_terms)
    scale = max(scores) or 1.0
    requested_statements = _requested_statements(original_terms)
    # Document identifiers commonly carry the company and reporting year. Use
    # those terms as a prior, not a filter: an annual report also covers earlier
    # years and a question may require more than one report.
    query_years = {term for term in original_terms if re.fullmatch(r"(?:19|20)\d{2}", term)}
    identifiers = [set(_pdf_terms(hit.text.partition("\n")[0])) for hit, _rank in candidates]
    company_terms = set().union(*identifiers) & original_terms - query_years
    company_terms -= {"annual", "report", "10k", "10q", "form"}
    reranked = []
    for score, (hit, rank), identifier in zip(scores, candidates, identifiers, strict=True):
        if company_terms:
            score += scale * 0.6 * len(identifier & company_terms) / len(company_terms)
        if query_years & identifier:
            score += scale * 0.15
        if requested_statements & _page_statements(hit.text):
            score += scale * 0.9
        if _has_requested_section(hit.text, original_terms):
            score += scale * 0.75
        reranked.append((score, rank, hit))
    reranked.sort(key=lambda item: (-item[0], item[1], item[2].entry_id, item[2].entry_version_id))
    return tuple(hit for _score, _rank, hit in reranked)


def _pdf_lexical_scores(
    texts: tuple[str, ...], weights: dict[str, float], original_terms: set[str]
) -> tuple[float, ...]:
    tokenized = [Counter(_pdf_terms(text)) for text in texts]
    document_frequency = Counter(term for frequencies in tokenized for term in frequencies)
    average_length = max(1.0, sum(sum(frequencies.values()) for frequencies in tokenized) / len(tokenized))
    scores = []
    for frequencies in tokenized:
        document_length = sum(frequencies.values())
        score = 0.0
        for term, weight in weights.items():
            frequency = frequencies[term]
            if not frequency:
                continue
            count = document_frequency[term]
            inverse_document_frequency = math.log(1 + (len(texts) - count + 0.5) / (count + 0.5))
            denominator = frequency + 1.2 * (0.25 + 0.75 * document_length / average_length)
            score += weight * inverse_document_frequency * frequency * 2.2 / denominator
        if original_terms:
            score += 0.35 * len(original_terms.intersection(frequencies)) / len(original_terms)
        scores.append(score)
    return tuple(scores)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
