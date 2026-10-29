# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Pi v3 session history as immutable, branch-aware evidence Sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import SourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class _PiTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class _PiEventSourceDefinition:
    name = "pi-event"
    version = "1"
    input_class = PiEventInput
    source_class = PiEventSource
    projections: tuple[SourceProjection[PiEventSource], ...] = (_PiTextEvidenceProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            definition_version=self.version,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = (
    _PiEventSourceDefinition()
)


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    if not isinstance(session_id, str) or not session_id or not isinstance(event_id, str) or not event_id:
        raise ValueError("session and event IDs must be non-empty strings")
    result = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(result) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"Pi event source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return result


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    session_id, records, _ = _parse_archive(value.content)
    if value.leaf_id is None:
        if not records:
            raise ValueError("Pi session has no leaf event")
        leaf_id = next(reversed(records))
    else:
        leaf_id = value.leaf_id
    if leaf_id not in records:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")

    path: list[dict[str, Any]] = []
    current: str | None = leaf_id
    while current is not None:
        record = records[current]
        path.append(record)
        current = _parent_id(record.get("parentId"))
    path.reverse()
    events = []
    for record in path:
        visible = _visible_event(session_id, record)
        if visible is not None:
            events.append(visible)
    return tuple(events)


def _parse_archive(content: bytes) -> tuple[str, dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    try:
        text = content.decode("utf-8-sig", errors="strict")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Pi session is empty")
        header = json.loads(lines[0])
        if not isinstance(header, dict) or header.get("type") != "session" or header.get("version") != 3:
            raise ValueError("Pi session header must declare type=session and version=3")
        session_id = header.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Pi session id must be a non-empty string")

        records: dict[str, dict[str, Any]] = {}
        all_records: list[dict[str, Any]] = []
        for line in lines[1:]:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Pi session records must be objects")
            event_id = record.get("id")
            if not isinstance(event_id, str) or not event_id or event_id in records:
                raise ValueError("Pi event IDs must be unique non-empty strings")
            if "parentId" not in record:
                raise ValueError("Pi event must include parentId")
            parent = _parent_id(record["parentId"])
            if parent is not None and parent not in records:
                raise ValueError("Pi event parentId must refer to a previously seen event")
            records[event_id] = record
            all_records.append(record)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Pi session encoding or JSONL") from error
    return session_id, records, tuple(all_records)


def _parent_id(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Pi event parentId must be a string or null")
    return value


def _visible_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        if role == "toolResult":
            body = _tool_result_body(message)
        elif role == "bashExecution":
            command = message.get("command", "")
            output = message.get("output", "")
            if not isinstance(command, str) or not isinstance(output, str):
                body = ""
            else:
                body = f"$ {command}" + (f"\n{output}" if output else "")
        else:
            body = _message_content(message.get("content"))
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _message_content(record.get("content"))
    else:
        return None
    if not body:
        return None
    event_id = record["id"]
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=_parent_id(record.get("parentId")),
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            chunks.append(block["text"])
        elif block.get("type") == "toolCall":
            name, call_id = block.get("name", ""), block.get("id", "")
            arguments = block.get("arguments", {})
            chunks.append(
                f"[tool_call {name} {call_id}] "
                + json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
    return "\n".join(chunks)


def _tool_result_body(message: dict[str, Any]) -> str:
    name, call_id = message.get("toolName", ""), message.get("toolCallId", "")
    text = _message_content(message.get("content"))
    return f"[tool_result {name} {call_id}] {text}"


class PiSessionConnector:
    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        try:
            raw = self.path.read_bytes()
        except OSError as error:
            raise ValueError("could not read Pi session file") from error
        session_id, all_records, records = _parse_archive(raw)
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self.leaf_id))
        checkpoint = session.checkpoint
        if checkpoint is None:
            imported: dict[str, str] = {}
            prior_session = session_id
        elif isinstance(checkpoint, dict) and checkpoint.get("session_id") == session_id and isinstance(
            checkpoint.get("events"), dict
        ):
            imported = dict(checkpoint["events"])
            prior_session = session_id
        else:
            raise ValueError("Pi session checkpoint does not match this session")

        event_records = {record["id"]: record for record in records}
        for event_id, digest in imported.items():
            record = event_records.get(event_id)
            if record is None or _record_digest(record) != digest:
                raise ValueError(f"previously imported Pi event {event_id!r} was changed or removed")

        for event in events:
            digest = _record_digest(event.record)
            existing = imported.get(event.event_id)
            if existing is not None:
                if existing != digest:
                    raise ValueError(f"previously imported Pi event {event.event_id!r} was changed")
                continue
            await session.submit(pi_event_source_id(event.session_id, event.event_id), "pi-event", event)
            imported[event.event_id] = digest
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"session_id": prior_session, "events": imported},
        )


def _record_digest(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
