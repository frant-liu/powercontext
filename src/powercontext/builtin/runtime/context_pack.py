# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Self-validating wire format for full and prefix-delta prepared contexts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContextPackError(ValueError):
    """A pack or delta violates the public wire protocol."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        if not isinstance(items, tuple) or any(type(item) is not bytes for item in items):
            raise ContextPackError("items must be a tuple of bytes")
        for item in items:
            try:
                item.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ContextPackError("pack items must be valid UTF-8") from error
        digest = _pack_digest(items)
        byte_count = 0
        while True:
            encoded = _pack_json(items, digest, byte_count)
            actual = len(encoded)
            if actual == byte_count:
                return cls(items, digest, actual)
            byte_count = actual

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content must be text")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        try:
            raw = content.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ContextPackError("content must be valid UTF-8 text") from error
        items: list[bytes] = []
        offset = 0
        while offset < len(raw):
            end = min(offset + chunk_bytes, len(raw))
            while end > offset:
                piece = raw[offset:end]
                try:
                    piece.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    end -= 1
                else:
                    break
            if end == offset:
                raise ContextPackError("chunk_bytes cannot fit the next UTF-8 character")
            items.append(raw[offset:end])
            offset = end
        return cls.build(tuple(items))

    @classmethod
    def from_bytes(cls, payload: bytes) -> ContextPack:
        value = _decode_json(payload)
        if not isinstance(value, dict) or set(value) != {"schema", "items", "digest", "byte_count"}:
            raise ContextPackError("pack fields do not match the schema")
        if value["schema"] != _PACK_SCHEMA:
            raise ContextPackError("unsupported pack schema")
        encoded_items = value["items"]
        if not isinstance(encoded_items, list) or any(not isinstance(item, str) for item in encoded_items):
            raise ContextPackError("pack items must be a Base64 string array")
        items = tuple(_decode_base64(item) for item in encoded_items)
        result = cls.build(items)
        if (
            value["digest"] != result.digest
            or isinstance(value["byte_count"], bool)
            or not isinstance(value["byte_count"], int)
            or value["byte_count"] != result.byte_count
        ):
            raise ContextPackError("pack digest or byte_count does not match")
        return result

    def to_bytes(self) -> bytes:
        return _pack_json(self.items, self.digest, self.byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("between requires ContextPack values")
        prefix = 0
        while prefix < min(len(base.items), len(target.items)) and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        self._validate()
        payload = {
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [base64.b64encode(item).decode("ascii") for item in self.appended],
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, payload: bytes) -> ContextPackDelta:
        value = _decode_json(payload)
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "base_digest",
            "target_digest",
            "prefix_items",
            "appended",
        }:
            raise ContextPackError("delta fields do not match the schema")
        if value["schema"] != _DELTA_SCHEMA:
            raise ContextPackError("unsupported delta schema")
        appended = value["appended"]
        if not isinstance(appended, list) or any(not isinstance(item, str) for item in appended):
            raise ContextPackError("delta appended must be a Base64 string array")
        result = cls(
            base_digest=value["base_digest"],
            target_digest=value["target_digest"],
            prefix_items=value["prefix_items"],
            appended=tuple(_decode_base64(item) for item in appended),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if not isinstance(self.base_digest, str) or not _DIGEST_RE.fullmatch(self.base_digest):
            raise ContextPackError("invalid base digest")
        if not isinstance(self.target_digest, str) or not _DIGEST_RE.fullmatch(self.target_digest):
            raise ContextPackError("invalid target digest")
        if isinstance(self.prefix_items, bool) or not isinstance(self.prefix_items, int) or self.prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        if not isinstance(self.appended, tuple) or any(type(item) is not bytes for item in self.appended):
            raise ContextPackError("appended must be a tuple of bytes")
        for item in self.appended:
            try:
                item.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ContextPackError("delta items must be valid UTF-8") from error


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires a pack and delta")
    delta._validate()
    if base.digest != delta.base_digest:
        raise ContextPackError("base pack digest does not match")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("prefix_items exceeds the base pack length")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("reconstructed pack digest does not match")
    return target


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {
        "schema": _PACK_SCHEMA,
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    return "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _pack_json(items: tuple[bytes, ...], digest: str, byte_count: int) -> bytes:
    return _canonical_json(
        {
            "schema": _PACK_SCHEMA,
            "items": [base64.b64encode(item).decode("ascii") for item in items],
            "digest": digest,
            "byte_count": byte_count,
        }
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _decode_json(payload: bytes) -> object:
    if type(payload) is not bytes:
        raise ContextPackError("wire payload must be bytes")
    try:
        return json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("wire payload must be UTF-8 JSON") from error


def _decode_base64(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
        if base64.b64encode(decoded).decode("ascii") != value:
            raise ContextPackError("Base64 values must use canonical encoding")
        decoded.decode("utf-8", errors="strict")
        return decoded
    except (UnicodeEncodeError, binascii.Error, UnicodeDecodeError) as error:
        raise ContextPackError("invalid Base64 UTF-8 chunk") from error


__all__ = ["ContextPack", "ContextPackDelta", "ContextPackError", "apply_context_pack_delta"]
