# Copyright (c) 2026 OceanBase.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PDF evidence remains complete and addressable through the real Runtime."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.artifacts.memory.errors import InvalidMemoryCandidateError
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    ReviseMemoryEntryRequest,
    SubmitSourceObservation,
    open_builtin_contexts,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import SourceDefinitionRegistry, manifest_for_definition, project_source_for_transport
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


class EvidencePipeline:
    async def extract(self, request):
        return tuple(
            MemoryEntryInput(
                kind="source-evidence",
                text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"],
                sources=(source,),
            )
            for source in request.sources
        )


async def store_pages(config, pages):
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    observations = []
    for page in pages:
        observations.append(project_source_for_transport(registry, await registry.resolve(page)))
    observations = tuple(observations)
    async with open_builtin_runtime(config) as runtime:
        scope = await runtime.scopes.create(ScopeDraft(title="Reports", summary="Reports", idempotency_key="reports"))
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        for observation in (*observations, observations[0]):
            await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scope.scope_id, observation=observation))
    async with open_builtin_runtime(config, candidate_pipeline=EvidencePipeline()) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        for _ in range(len(pages) + 1):
            if not (await memory.flush()).processed:
                break
        else:
            pytest.fail("Memory did not consume all page observations")
    return scope.scope_id, observations


async def test_long_document_identity_and_dense_pdf_survive_restart(tmp_path: Path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    body = "Revenue 123 inventory 456 operating profit 789. " * 300
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 9 Tf 20 750 Td ({body}) Tj ET".encode())
    page[NameObject("/Contents")] = stream
    writer.add_blank_page(width=600, height=800)
    buffer = io.BytesIO()
    writer.write(buffer)
    document_id = "财务报告:/" * 100
    pages = parse_pdf_document(PDFDocumentInput(document_id=document_id, content=buffer.getvalue()))
    assert len(pages) == 2 and pages[1].text == ""
    assert len(pages[0].text.encode()) > 8192 and pages[0].text.strip() == body.strip()
    assert pages[0].document_sha256 == hashlib.sha256(buffer.getvalue()).hexdigest()
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'dense.db'}"))
    scope_id, observations = await store_pages(config, pages)
    assert len({item.name for item in observations}) == 2
    assert all(len(item.name) <= 256 for item in observations)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope_id)
        citations = await search_pdf_pages(memory, "inventory operating profit", limit=1)
        assert len(citations) == 1
        record = await memory.get(GetMemoryEntryRequest(citation=citations[0]))
        assert record.entry.text == f"{document_id}\n{pages[0].text}".strip()
        assert record.entry.sources[0].source_id == observations[0].name
        with pytest.raises(InvalidMemoryCandidateError, match="8192 UTF-8 bytes"):
            await memory.revise(
                ReviseMemoryEntryRequest(
                    citation=citations[0], kind="source-evidence", text=record.entry.text + " revised"
                )
            )
        assert (await memory.get(GetMemoryEntryRequest(citation=citations[0]))).entry.text == record.entry.text
    async with open_builtin_contexts(config) as contexts:
        journal = await (await contexts.get(scope_id)).sources.journal.entries()
        assert len(journal) == 2
        assert {item.source.name for item in journal} == {item.name for item in observations}
        dense = next(item.source for item in journal if item.source.name == observations[0].name)
        assert dense.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"] == f"{document_id}\n{pages[0].text}"


async def test_statement_queries_keep_company_and_exact_page_citations(tmp_path: Path):
    def page(document, number, text):
        return PDFPageInput(document_id=document, document_sha256="a" * 64, page_number=number, text=text)

    numbers = "2024 2023 2022\n100 90 80\n70 60 50\n30 20 10"
    pages = (
        page("ALPHA_2024_10K", 1, "Consolidated Statements of Earnings\nNet revenue\nOperating profit\n" + numbers),
        page("ALPHA_2024_10K", 2, "Consolidated Statements of Cash Flows\nDepreciation amortization\n" + numbers),
        page("ALPHA_2024_10K", 3, "Item 3. Legal Proceedings.\nThere are no material pending claims."),
        page("BETA_2024_10K", 1, "Consolidated Statements of Income\nEBITDA operating income\n" + numbers),
        *(
            page("ALPHA_2024_10K", n + 4, "EBITDA operating income margin restructuring costs cash flow. " * 20)
            for n in range(25)
        ),
    )
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'reports.db'}"))
    scope_id, observations = await store_pages(config, pages)
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope_id)
        citations = await search_pdf_pages(
            memory, "What is ALPHA FY2024 EBITDA margin using the income and cash flow statements?", 2
        )
        entries = [(await memory.get(GetMemoryEntryRequest(citation=citation))).entry for citation in citations]
        sources = {ref.source_id for entry in entries for ref in entry.sources}
        assert {observations[0].name, observations[1].name} <= sources
        assert observations[3].name not in sources
        assert len(sources) == len(citations)
        legal = await search_pdf_pages(memory, "Has ALPHA any material legal battles?", 1)
        record = await memory.get(GetMemoryEntryRequest(citation=legal[0]))
        assert record.entry.sources[0].source_id == observations[2].name
