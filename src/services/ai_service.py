"""
ai_service.py - SE-layer wrapper around the provided ai/ package.

Adds retries (tenacity), in-session embedding cache, and structured logging.
The vlm and embedder can be injected at construction time for offline testing.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import ai
from ai.providers.base import EmbeddingProvider, ProviderError, VLMProvider
from ai.schemas import ItemDescription
from src.config import settings

logger = logging.getLogger(__name__)


class AIService:
    """
    Dependency-injectable wrapper around ai.describe_item and ai.embed.
    Pass fake providers to the constructor for offline testing.
    """

    def __init__(
        self,
        vlm: VLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        max_attempts: int | None = None,
        wait_seconds: float | None = None,
    ) -> None:
        self._vlm = vlm
        self._embedder = embedder
        self._max_attempts = max_attempts if max_attempts is not None else settings.retry_max_attempts
        self._wait_seconds = wait_seconds if wait_seconds is not None else settings.retry_wait_seconds
        self._cache: dict[str, np.ndarray] = {}

    def _make_retry(self):
        return retry(
            retry=retry_if_exception_type(ProviderError),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(
                multiplier=self._wait_seconds,
                min=max(self._wait_seconds, 0.001),
                max=30.0,
            ),
            reraise=True,
        )

    def describe_item(self, image_path: str, user_text: str) -> ItemDescription:
        """Call ai.describe_item with exponential-backoff retries."""
        @self._make_retry()
        def _call() -> ItemDescription:
            t0 = time.perf_counter()
            result = ai.describe_item(image_path, user_text, vlm=self._vlm)
            elapsed = time.perf_counter() - t0
            logger.info(
                "describe_item OK image=%s elapsed=%.3fs class=%s confidence=%.2f",
                image_path, elapsed, result.object_class, result.confidence,
            )
            return result

        logger.debug("describe_item start image=%s", image_path)
        return _call()

    def embed(self, text: str) -> np.ndarray:
        """
        Return a unit-normalized embedding vector.
        Repeated calls with the same text hit the in-session dict cache.
        """
        if text in self._cache:
            logger.debug("embed cache_hit text_len=%d", len(text))
            return self._cache[text]

        @self._make_retry()
        def _call() -> np.ndarray:
            t0 = time.perf_counter()
            vec = ai.embed(text, embedder=self._embedder)
            elapsed = time.perf_counter() - t0
            logger.info(
                "embed OK text_len=%d dim=%d elapsed=%.3fs",
                len(text), vec.shape[0], elapsed,
            )
            return vec

        logger.debug("embed cache_miss text_len=%d", len(text))
        vec = _call()
        self._cache[text] = vec
        return vec

    def clear_cache(self) -> None:
        """Discard all cached embeddings (useful between test runs)."""
        self._cache.clear()
        logger.debug("embed cache cleared")


ai_service = AIService()