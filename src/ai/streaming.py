"""
src/ai/streaming.py — Streaming response support for LLM providers.

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

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


def stream_openai_vlm(
    image_path: str,
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream VLM response from OpenAI.

    Args:
        image_path: Path to image file
        prompt: Prompt text
        model: Model name (e.g., 'gpt-4o-mini')
        api_key: OpenAI API key (uses env var if not provided)

    Yields:
        Token strings as they arrive
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package required for streaming")

    import base64
    import os
    from pathlib import Path

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    # Load and encode image
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)

    media_type = "image/jpeg" if path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    data_url = f"data:{media_type};base64,{b64}"

    # Stream from OpenAI
    with client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=1024,
        stream=True,  # Enable streaming
    ) as stream:
        for event in stream:
            chunk = event.choices[0].delta.content
            if chunk:
                yield chunk


def stream_anthropic_vlm(
    image_path: str,
    prompt: str,
    model: str = "claude-3-5-sonnet",
    api_key: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream VLM response from Anthropic.

    Args:
        image_path: Path to image file
        prompt: Prompt text
        model: Model name
        api_key: Anthropic API key

    Yields:
        Token strings as they arrive
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package required for streaming")

    import base64
    import os
    from pathlib import Path

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Load and encode image
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)

    media_type = "image/jpeg" if path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")

    # Stream from Anthropic
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_response(
    image_path: str,
    prompt: str,
    provider: str = "openai",
    model: Optional[str] = None,
) -> Iterator[str]:
    """
    Universal streaming interface for VLM providers.

    Args:
        image_path: Path to image
        prompt: Prompt text
        provider: Provider name ('openai' or 'anthropic')
        model: Model name (uses env var if not provided)

    Yields:
        Token strings from the streaming response
    """
    if provider == "openai":
        if not model:
            import os
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        yield from stream_openai_vlm(image_path, prompt, model)
    elif provider == "anthropic":
        if not model:
            import os
            model = os.getenv("LLM_MODEL", "claude-3-5-sonnet")
        yield from stream_anthropic_vlm(image_path, prompt, model)
    else:
        raise ValueError(f"Unsupported provider for streaming: {provider}")
