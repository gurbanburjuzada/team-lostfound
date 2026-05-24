"""
src/ai/providers/failover.py — Multi-provider failover wrapper.

Bonus 1 (+3 pts): Tries LLM and embedding providers in order. If one fails,
automatically tries the next. Failures are:
- ProviderError exceptions
- HTTP 5xx errors
- HTTP 429 (rate limit) after retry budget exhausted
- Timeouts

Example:
    from src.ai.providers.failover import FailoverVLM, FailoverEmbedding
    from ai.providers.google import GeminiVLM, GeminiEmbedding

    # Use Google Gemini with automatic failover
    vlm = FailoverVLM([
        GeminiVLM(),
    ])
    response = vlm.describe("image.jpg", "describe this")

    # For embedding: use Google Gemini
    embedding = FailoverEmbedding([
        GeminiEmbedding(),
    ])
    vector = embedding.embed("test")
"""

from __future__ import annotations

import inspect
import asyncio
import logging
from typing import Optional

from ai.providers.base import EmbeddingProvider, VLMProvider, ProviderError

logger = logging.getLogger(__name__)


class FailoverVLM(VLMProvider):
    """Failover wrapper for VLM providers."""

    def __init__(self, providers: list[VLMProvider]) -> None:
        """
        Initialize with ordered list of VLM providers.

        Args:
            providers: List of VLMProvider instances, tried in order

        Raises:
            ValueError: If providers list is empty
        """
        if not providers:
            raise ValueError("At least one VLM provider required")
        self.providers = providers
        self._logger = logging.getLogger(__name__)

    def describe(
        self,
        image_path: str,
        prompt: str,
        *,
        json_schema: Optional[dict] = None,
    ) -> str:
        """
        Describe image, trying providers in order until one succeeds.

        Args:
            image_path: Path to image file
            prompt: Description prompt
            json_schema: Optional JSON schema for structured output

        Returns:
            Description from the first succeeding provider

        Raises:
            ProviderError: If all providers fail
        """
        last_error: Exception | None = None

        for i, provider in enumerate(self.providers):
            try:
                self._logger.debug(
                    f"VLM failover attempting provider {i + 1}/{len(self.providers)}: "
                    f"{type(provider).__name__}"
                )
                result = provider.describe(image_path, prompt, json_schema=json_schema)
                self._logger.info(
                    f"VLM failover succeeded on provider {i}: {type(provider).__name__}"
                )
                return result
            except (ProviderError, TimeoutError, asyncio.TimeoutError) as e:
                self._logger.warning(
                    f"VLM provider {i} ({type(provider).__name__}) failed: {str(e)}"
                )
                last_error = e

        raise ProviderError(
            f"All {len(self.providers)} VLM providers failed; "
            f"last error: {type(last_error).__name__}: {last_error}"
        )


class FailoverEmbedding(EmbeddingProvider):
    """Failover wrapper for embedding providers."""

    def __init__(self, providers: list[EmbeddingProvider]) -> None:
        """
        Initialize with ordered list of embedding providers.

        Args:
            providers: List of EmbeddingProvider instances

        Raises:
            ValueError: If providers list is empty
        """
        if not providers:
            raise ValueError("At least one embedding provider required")
        self.providers = providers
        self._logger = logging.getLogger(__name__)

    @property
    def dimension(self) -> int:
        """Return dimension from the first provider."""
        return self.providers[0].dimension

    def embed(self, text: str) -> list[float]:
        """
        Embed text, trying providers in order until one succeeds.

        Args:
            text: Text to embed

        Returns:
            Embedding vector from first succeeding provider

        Raises:
            ProviderError: If all providers fail
        """
        last_error: Exception | None = None

        for i, provider in enumerate(self.providers):
            try:
                self._logger.debug(
                    f"Embedding failover attempting provider {i + 1}/{len(self.providers)}: "
                    f"{type(provider).__name__}"
                )
                result = provider.embed(text)
                self._logger.info(
                    f"Embedding failover succeeded on provider {i}: {type(provider).__name__}"
                )
                return result
            except (ProviderError, TimeoutError, asyncio.TimeoutError) as e:
                self._logger.warning(
                    f"Embedding provider {i} ({type(provider).__name__}) failed: {str(e)}"
                )
                last_error = e

        raise ProviderError(
            f"All {len(self.providers)} embedding providers failed; "
            f"last error: {last_error}"
        )

    async def embed_async(self, text: str) -> list[float]:
        """
        Async embedding, trying providers in order.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ProviderError: If all fail
        """
        last_error: Exception | None = None

        for i, provider in enumerate(self.providers):
            try:
                self._logger.debug(
                    f"Embedding failover (async) attempting provider {i + 1}/{len(self.providers)}"
                )
                if hasattr(provider, "embed_async") and inspect.iscoroutinefunction(provider.embed_async):
                    result = await provider.embed_async(text)
                else:
                    result = provider.embed(text)
                self._logger.info(f"Embedding failover (async) succeeded on provider {i}")
                return result
            except (ProviderError, TimeoutError, asyncio.TimeoutError) as e:
                self._logger.warning(f"Embedding provider {i} async failed: {str(e)}")
                last_error = e

        raise ProviderError(
            f"All {len(self.providers)} embedding providers failed (async); "
            f"last error: {last_error}"
        )
