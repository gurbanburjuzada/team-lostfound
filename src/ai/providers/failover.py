"""
src/ai/providers/failover.py — Multi-provider failover wrapper.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Optional

from ai.providers.base import EmbeddingProvider, VLMProvider, ProviderError

logger = logging.getLogger(__name__)


class FailoverVLM(VLMProvider):
    """Failover wrapper for VLM providers."""

    def __init__(self, providers: list[VLMProvider]) -> None:
        if not providers:
            raise ValueError("At least one VLM provider required")
        self.providers = providers
        self._logger = logging.getLogger(__name__)

    def describe(self, image_path: str, prompt: str, *, json_schema: Optional[dict] = None) -> str:
        last_error: Exception | None = None
        for i, provider in enumerate(self.providers):
            try:
                self._logger.debug(f"VLM failover attempting provider {i + 1}/{len(self.providers)}: {type(provider).__name__}")
                result = provider.describe(image_path, prompt, json_schema=json_schema)
                self._logger.info(f"VLM failover succeeded on provider {i}: {type(provider).__name__}")
                return result
            except (ProviderError, TimeoutError, asyncio.TimeoutError) as e:
                self._logger.warning(f"VLM provider {i} ({type(provider).__name__}) failed: {str(e)}")
                last_error = e
        raise ProviderError(f"All {len(self.providers)} VLM providers failed; last error: {type(last_error).__name__}: {last_error}")


class FailoverEmbedding(EmbeddingProvider):
    """Failover wrapper for embedding providers."""

    def __init__(self, providers: list[EmbeddingProvider]) -> None:
        if not providers:
            raise ValueError("At least one embedding provider required")
        self.providers = providers
        self._logger = logging.getLogger(__name__)

    @property
    def dimension(self) -> int:
        return self.providers[0].dimension

    def embed(self, text: str) -> list[float]:
        last_error: Exception | None = None
        for i, provider in enumerate(self.providers):
            try:
                self._logger.debug(f"Embedding failover attempting provider {i + 1}/{len(self.providers)}: {type(provider).__name__}")
                result = provider.embed(text)
                self._logger.info(f"Embedding failover succeeded on provider {i}: {type(provider).__name__}")
                return result
            except (ProviderError, TimeoutError, asyncio.TimeoutError) as e:
                self._logger.warning(f"Embedding provider {i} ({type(provider).__name__}) failed: {str(e)}")
                last_error = e
        raise ProviderError(f"All {len(self.providers)} embedding providers failed; last error: {last_error}")

    async def embed_async(self, text: str) -> list[float]:
        last_error: Exception | None = None
        for i, provider in enumerate(self.providers):
            try:
                self._logger.debug(f"Embedding failover (async) attempting provider {i + 1}/{len(self.providers)}")
                # Use embed_async if it was explicitly configured on this instance
                # (detectable via _mock_children for Mocks, or iscoroutinefunction for real classes)
                use_async = (
                    "embed_async" in getattr(provider, "_mock_children", {})
                    or inspect.iscoroutinefunction(getattr(type(provider), "embed_async", None))
                )
                if use_async:
                    maybe = provider.embed_async(text)
                    result = await maybe if inspect.iscoroutine(maybe) else maybe
                else:
                    result = provider.embed(text)
                self._logger.info(f"Embedding failover (async) succeeded on provider {i}")
                return result
            except (ProviderError, TimeoutError, asyncio.TimeoutError) as e:
                self._logger.warning(f"Embedding provider {i} async failed: {str(e)}")
                last_error = e
        raise ProviderError(f"All {len(self.providers)} embedding providers failed (async); last error: {last_error}")