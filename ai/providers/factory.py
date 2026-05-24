"""Factory functions that select the active provider based on environment variables.

Environment variables consulted
-------------------------------
LLM_PROVIDER     : "gemini"  (default: "gemini")
LLM_MODEL        : provider-specific model id          (optional)
EMBEDDING_PROVIDER : "gemini"  (default: "gemini")
EMBEDDING_MODEL  : provider-specific model id          (optional)

Plus the corresponding API_KEY variables. See each provider module for details.
"""

from __future__ import annotations

import os

from ai.providers.base import VLMProvider, EmbeddingProvider, ProviderError


def get_vlm() -> VLMProvider:
    """Return the configured VLM provider."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if provider in ("google", "gemini"):
        from ai.providers.google import GeminiVLM
        return GeminiVLM()
    raise ProviderError(
        f"Unknown LLM_PROVIDER={provider!r}. Only 'gemini' is supported."
    )


def get_embedder() -> EmbeddingProvider:
    """Return the configured embedding provider."""
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower().strip()
    if provider in ("google", "gemini"):
        from ai.providers.google import GeminiEmbedding
        return GeminiEmbedding()
    raise ProviderError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. Only 'gemini' is supported."
    )
