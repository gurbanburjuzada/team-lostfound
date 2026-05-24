"""
src/ai/streaming.py — Streaming response support for Google Gemini VLM provider.

Bonus 6 (+2 pts): Supports streaming output where available. Tokens are yielded
as they arrive, enabling real-time CLI feedback.

Example CLI usage:
    python -m src.cli vlm-stream image.jpg "describe this"

The streaming version writes tokens to stdout as they arrive rather than
buffering the entire response.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def stream_gemini_vlm(
    image_path: str,
    prompt: str,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream VLM response from Google Gemini.

    Args:
        image_path: Path to image file
        prompt: Prompt text
        model: Model name (e.g., 'gemini-2.0-flash')
        api_key: Google API key (uses env var if not provided)

    Yields:
        Token strings as they arrive
    """
    import base64
    import os
    from pathlib import Path

    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise ImportError("google-genai package required for streaming") from e

    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    # Load and encode image
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)

    # Upload file to Gemini
    from google.genai import types  # type: ignore
    
    try:
        uploaded = client.files.upload(file=str(path))
        
        # Stream from Gemini
        response = client.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(),
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        raise RuntimeError(f"Gemini streaming failed: {e}") from e


def stream_response(
    image_path: str,
    prompt: str,
    provider: str = "gemini",
    model: Optional[str] = None,
) -> Iterator[str]:
    """
    Universal streaming interface for VLM providers.

    Args:
        image_path: Path to image
        prompt: Prompt text
        provider: Provider name (currently only 'gemini' is supported)
        model: Model name (uses env var if not provided)

    Yields:
        Token strings from the streaming response
    """
    if provider not in ("gemini", "google"):
        raise ValueError(f"Unsupported provider for streaming: {provider}. Only 'gemini' is supported.")
    
    if not model:
        import os
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    
    yield from stream_gemini_vlm(image_path, prompt, model)
