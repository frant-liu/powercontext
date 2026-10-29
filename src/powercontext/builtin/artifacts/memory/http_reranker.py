# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""HTTP adapter for bounded, globally ranked Memory reranking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, _validate_bounds
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score candidate batches through a caller-owned asynchronous HTTP client."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(policy_id, str) or not policy_id or policy_id.strip() != policy_id:
            raise ValueError("policy_id must be non-empty and trimmed")
        if not isinstance(client, httpx.AsyncClient):
            raise ValueError("client must be an httpx.AsyncClient")
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not isinstance(query, str) or not query:
            raise ValueError("query must be non-empty")
        _validate_bounds(len(candidates), limit)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        scores: dict[int, int | float] = {}
        requests = 0
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            ranks = tuple(range(start + 1, start + len(batch) + 1))
            response = await self.client.post(
                self.endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text}
                        for rank, candidate in zip(ranks, batch, strict=True)
                    ],
                },
            )
            requests += 1
            response.raise_for_status()
            try:
                payload = response.json()
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("reranker response must be valid JSON") from error
            scores.update(_validate_batch_scores(payload, ranks))

        selected = tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=requests))


def _validate_batch_scores(payload: Any, expected_ranks: Sequence[int]) -> dict[int, float]:
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("reranker response must contain only scores")
    values = payload["scores"]
    if not isinstance(values, list) or len(values) != len(expected_ranks):
        raise ValueError("reranker scores must cover each batch rank exactly once")
    expected = set(expected_ranks)
    result: dict[int, float] = {}
    for item in values:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each score record must contain only rank and score")
        rank, score = item["rank"], item["score"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in expected or rank in result:
            raise ValueError("reranker returned an invalid, duplicate, or cross-batch rank")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or (
            isinstance(score, float) and not math.isfinite(score)
        ):
            raise ValueError("reranker score must be a finite number")
        result[rank] = score
    if set(result) != expected:
        raise ValueError("reranker scores did not cover the requested batch")
    return result


__all__ = ["HTTPMemoryReranker"]
