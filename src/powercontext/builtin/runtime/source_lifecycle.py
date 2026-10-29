# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Persisted scope-local dependency graphs for source availability."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, GetTopicMemoryRequest
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """A lifecycle node, dependency, or event violates its scope contract."""


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: int
    sequence: int
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle identifiers and reasons must be non-blank")
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lifecycle revision and sequence must be positive")
        return value


class SourceLifecycleApplication:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        if not isinstance(scope_id, str) or not scope_id.strip() or scope_id != scope_id.strip():
            raise SourceLifecycleError("scope_id must be non-empty and trimmed")
        return ScopedSourceLifecycle(self._runtime, self._database, scope_id)


class ScopedSourceLifecycle:
    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = scope_id

    async def register_source(self, node_id: str, reference: SourceRef, *, revision: int) -> None:
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("source revision must be positive")
        await self._runtime.records.for_scope(self.scope_id).get_source(reference.source_type, reference.source_id)
        await self._register(
            node_id,
            {"kind": "source", "reference": reference.model_dump(mode="json"), "revision": revision},
            parents=(),
            policy=None,
            available=True,
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        await self._register_derived(
            node_id,
            {"kind": "memory", "reference": citation.model_dump(mode="json")},
            parents,
            policy,
        )

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        if reference.family != "topic-memory":
            raise SourceLifecycleError("topic reference must identify Topic Memory")
        await self._runtime.topic_memory.for_scope(self.scope_id).get(GetTopicMemoryRequest(artifact=reference))
        await self._register_derived(
            node_id,
            {"kind": "topic", "reference": reference.model_dump(mode="json")},
            parents,
            policy,
        )

    async def _register_derived(
        self,
        node_id: str,
        identity: dict[str, object],
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        graph = await self._load()
        if not parents or len(set(parents)) != len(parents) or any(parent not in graph for parent in parents):
            raise SourceLifecycleError("derived nodes require unique, already registered parents")
        if policy not in {"all", "any"}:
            raise SourceLifecycleError("policy must be all or any")
        available = _combine(policy, [bool(graph[parent]["available"]) for parent in parents])
        await self._register(node_id, identity, parents=parents, policy=policy, available=available, graph=graph)

    async def _register(
        self,
        node_id: str,
        identity: dict[str, object],
        *,
        parents: tuple[str, ...],
        policy: str | None,
        available: bool,
        graph: dict[str, dict[str, object]] | None = None,
    ) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise SourceLifecycleError("node_id must be non-blank")
        graph = await self._load() if graph is None else graph
        candidate = {**identity, "parents": list(parents), "policy": policy, "available": available}
        existing = graph.get(node_id)
        if existing is not None:
            stable_keys = ("kind", "reference", "revision", "parents", "policy")
            if any(existing.get(key) != candidate.get(key) for key in stable_keys):
                raise SourceLifecycleError(f"node {node_id!r} has a conflicting registration")
            await self._reconcile_memory_states(graph)
            return
        graph[node_id] = candidate
        await self._save(graph)
        await self._reconcile_memory_states(graph)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        graph = await self._load()
        node = graph.get(event.node_id)
        if node is None or node.get("kind") != "source":
            raise SourceLifecycleError("events can only target a registered Source")
        if node.get("revision") != event.revision:
            raise SourceLifecycleError("event revision does not match the Source lifecycle revision")
        last = node.get("last_event")
        current_sequence = 0 if not isinstance(last, dict) else int(last["sequence"])
        incoming = event.model_dump(mode="json")
        if event.sequence < current_sequence:
            raise SourceLifecycleError("event sequence must increase")
        if event.sequence == current_sequence and last != incoming:
            raise SourceLifecycleError("same event sequence has conflicting content")
        if event.sequence > current_sequence:
            node["last_event"] = incoming
            node["available"] = event.available
            _recompute_availability(graph)
            await self._save(graph)
        await self._reconcile_memory_states(graph)
        return sorted(node_id for node_id, value in graph.items() if value["available"])

    async def resolve(self, node_id: str, *, purpose: Literal["generation", "audit"]):
        graph = await self._load()
        node = graph.get(node_id)
        if node is None or purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("unknown lifecycle node or purpose")
        if purpose == "generation" and not node["available"]:
            raise SourceLifecycleError("unavailable evidence cannot be used for generation")
        kind = node["kind"]
        if kind == "source":
            return SourceRef.model_validate(node["reference"])
        if kind == "memory":
            return MemoryCitation.model_validate(node["reference"])
        if kind == "topic":
            return ArtifactRef.model_validate(node["reference"])
        raise SourceLifecycleError("unknown lifecycle node kind")

    async def filter_topic_result(self, result):
        if self._database is None:
            return result
        graph = await self._load()
        refs = {
            (value["reference"]["artifact_id"], value["reference"]["revision"]): bool(value["available"])
            for value in graph.values()
            if value["kind"] == "topic"
        }
        hits = tuple(
            hit
            for hit in result.hits
            if refs.get((hit.artifact_ref.artifact_id, hit.artifact_ref.revision), True)
        )
        return result.model_copy(update={"hits": hits})

    async def _reconcile_memory_states(self, graph: dict[str, dict[str, object]]) -> None:
        memory_nodes = [(node_id, node) for node_id, node in graph.items() if node["kind"] == "memory"]
        if not memory_nodes:
            return
        runtime = self._runtime
        async with runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            entries = {entry.entry_id: entry for entry in await service.entries(current)}
            states = {entry.entry_id: entry.state for entry in current.content.manifest.entries}
            for _node_id, node in memory_nodes:
                citation = MemoryCitation.model_validate(node["reference"])
                entry = entries.get(citation.entry_id)
                if entry is None:
                    raise SourceLifecycleError("registered Memory entry is no longer present at the head")
                active = node["available"]
                if (states.get(citation.entry_id) == "active") == active:
                    continue
                if active:
                    await service.reactivate(current, entries=(entry,), reason="source lifecycle restored")
                else:
                    updated = await service.forget(current, entries=(entry,), reason="source lifecycle withdrawn")
                    current = updated
                if active:
                    current = await service.head(context.artifacts.memory_artifact_id)
                entries = {entry.entry_id: entry for entry in await service.entries(current)}
                states = {entry.entry_id: entry.state for entry in current.content.manifest.entries}

    async def _load(self) -> dict[str, dict[str, object]]:
        if self._database is None:
            raise SourceLifecycleError("source lifecycle persistence is unavailable")
        async with self._database.transaction() as connection:
            if connection.dialect.name == "sqlite":
                await connection.execute(
                    update(SOURCE_LIFECYCLE_TABLE)
                    .where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id)
                    .values(payload=SOURCE_LIFECYCLE_TABLE.c.payload)
                )
            payload = await connection.scalar(
                select(SOURCE_LIFECYCLE_TABLE.c.payload).where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id)
            )
        return {} if payload is None else json.loads(bytes(payload).decode("utf-8"))

    async def _save(self, graph: dict[str, dict[str, object]]) -> None:
        if self._database is None:
            raise SourceLifecycleError("source lifecycle persistence is unavailable")
        payload = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        async with self._database.transaction() as connection:
            result = await connection.execute(
                update(SOURCE_LIFECYCLE_TABLE)
                .where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id)
                .values(payload=payload)
            )
            if result.rowcount == 0:
                await connection.execute(insert(SOURCE_LIFECYCLE_TABLE).values(scope_id=self.scope_id, payload=payload))


def _combine(policy: str, parents: list[bool]) -> bool:
    return all(parents) if policy == "all" else any(parents)


def _recompute_availability(graph: dict[str, dict[str, object]]) -> None:
    changed = True
    while changed:
        changed = False
        for node in graph.values():
            parents = node.get("parents", [])
            if not parents:
                continue
            available = _combine(str(node["policy"]), [bool(graph[parent]["available"]) for parent in parents])
            if node["available"] != available:
                node["available"] = available
                changed = True


__all__ = ["SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent", "ScopedSourceLifecycle"]
