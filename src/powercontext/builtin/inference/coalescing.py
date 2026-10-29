# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Coalesce concurrent embedding work and retain a bounded successful LRU."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _EmbeddedText:
    vector: EmbeddingVector
    batch_id: int
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share same-loop embedding requests, batch unique texts, and cache successes."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[_EmbeddedText]] = {}
        self._pending: deque[tuple[str, asyncio.Future[_EmbeddedText]]] = deque()
        self._lock = asyncio.Lock()
        self._dispatcher: asyncio.Task[None] | None = None
        self._batch_sequence = 0

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not isinstance(texts, tuple) or any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must be a tuple of strings")
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[_EmbeddedText]] = {}
        cached: dict[str, EmbeddingVector] = {}
        async with self._lock:
            for text in dict.fromkeys(texts):
                vector = self._cache.get(text)
                if vector is not None:
                    self._cache.move_to_end(text)
                    cached[text] = vector
                    continue
                future = self._inflight.get(text)
                if future is None:
                    future = loop.create_future()
                    future.add_done_callback(_consume_future_exception)
                    self._inflight[text] = future
                    self._pending.append((text, future))
                futures[text] = future
            if self._pending and self._dispatcher is None:
                self._dispatcher = loop.create_task(self._dispatch())

        if not futures:
            return EmbeddingResult(
                vectors=tuple(cached[text] for text in texts),
                usage=InferenceUsage(requests=0),
            )
        resolved = await asyncio.gather(*(asyncio.shield(futures[text]) for text in futures))
        vectors = dict(cached)
        batches: dict[int, InferenceUsage] = {}
        for text, result in zip(futures, resolved, strict=True):
            vectors[text] = result.vector
            batches.setdefault(result.batch_id, result.usage)
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=_sum_usage(tuple(batches.values())),
        )

    async def _dispatch(self) -> None:
        if self._coalesce_seconds:
            await asyncio.sleep(self._coalesce_seconds)
        while True:
            async with self._lock:
                batch = [self._pending.popleft() for _ in range(min(self._max_batch_size, len(self._pending)))]
                if not batch:
                    self._dispatcher = None
                    return
                self._batch_sequence += 1
                batch_id = self._batch_sequence
            texts = tuple(text for text, _ in batch)
            try:
                result = await self._model.embed(texts)
                if len(result.vectors) != len(texts):
                    raise ValueError("embedding provider returned a different number of vectors")
            except BaseException as error:
                async with self._lock:
                    for text, future in batch:
                        if self._inflight.get(text) is future:
                            self._inflight.pop(text)
                        if not future.done():
                            future.set_exception(error)
                if isinstance(error, asyncio.CancelledError):
                    raise
            else:
                async with self._lock:
                    for (text, future), vector in zip(batch, result.vectors, strict=True):
                        if self._inflight.get(text) is future:
                            self._inflight.pop(text)
                        if self._cache_size:
                            self._cache[text] = vector
                            self._cache.move_to_end(text)
                            while len(self._cache) > self._cache_size:
                                self._cache.popitem(last=False)
                        if not future.done():
                            future.set_result(_EmbeddedText(vector, batch_id, result.usage))


def _sum_usage(usages: tuple[InferenceUsage, ...]) -> InferenceUsage:
    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=_sum_optional(usage.input_tokens for usage in usages),
        output_tokens=_sum_optional(usage.output_tokens for usage in usages),
    )


def _sum_optional(values) -> int | None:
    observed = tuple(value for value in values if value is not None)
    return None if not observed else sum(observed)


def _consume_future_exception(future: asyncio.Future[object]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["CoalescingEmbeddingModel"]
