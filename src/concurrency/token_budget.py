"""
src/concurrency/token_budget.py — Token-aware rate limiter.

Bonus 7 (+2 pts): Tracks tokens-per-minute (TPM) against configured limits
and backs off before hitting the provider's rate limit. Uses a sliding window
to track consumed tokens over the past 60 seconds.

Example:
    budget = TokenBudget(tokens_per_minute=900_000)
    # Before each AI call:
    await budget.acquire(estimated_tokens=500)
    # Call proceeds; sleep if TPM headroom insufficient
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class TokenBudget:
    """Sliding-window token rate limiter."""

    def __init__(self, tokens_per_minute: int) -> None:
        """
        Initialize token budget.

        Args:
            tokens_per_minute: TPM limit (e.g., 900_000 for Claude 3.5 Sonnet)
        """
        if tokens_per_minute <= 0:
            raise ValueError(f"tokens_per_minute must be > 0, got {tokens_per_minute}")
        self.tpm = tokens_per_minute
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def acquire(self, estimated_tokens: int) -> None:
        """
        Acquire tokens from the budget. Sleeps if necessary to stay within TPM limit.

        Args:
            estimated_tokens: Number of tokens this call will consume (prompt + completion)

        Raises:
            ValueError: If estimated_tokens exceeds TPM
        """
        if estimated_tokens > self.tpm:
            raise ValueError(
                f"Single call {estimated_tokens} exceeds TPM limit {self.tpm}"
            )

        while True:
            wait_time = 0.0

            async with self._lock:
                now = time.monotonic()

                # Drop events older than 60 seconds (sliding window)
                while self._events and self._events[0][0] < now - 60:
                    self._events.popleft()

                # Check if we have headroom
                used_tokens = sum(t for _, t in self._events)

                if used_tokens + estimated_tokens <= self.tpm:
                    # Headroom available; acquire tokens and return
                    self._events.append((now, estimated_tokens))
                    self._logger.debug(
                        f"acquired {estimated_tokens} tokens; "
                        f"used {used_tokens + estimated_tokens}/{self.tpm}"
                    )
                    return

                # No headroom; calculate wait time
                if self._events:
                    oldest_event_time = self._events[0][0]
                    time_until_oldest_expires = 60 - (now - oldest_event_time)
                    wait_time = max(time_until_oldest_expires, 0.1)
                else:
                    # Shouldn't happen, but safety net
                    wait_time = 0.1

            self._logger.warning(
                f"TPM limit approaching ({used_tokens}/{self.tpm}); "
                f"backing off for {wait_time:.1f}s"
            )
            await asyncio.sleep(wait_time)

    def get_usage(self) -> tuple[int, int]:
        """
        Get current usage (tokens used, TPM limit).

        Returns:
            Tuple of (tokens_used_in_last_60s, tpm_limit)
        """
        now = time.monotonic()
        # Drop old events
        while self._events and self._events[0][0] < now - 60:
            self._events.popleft()
        used = sum(t for _, t in self._events)
        return (used, self.tpm)


# Pre-configured budgets for popular providers
PROVIDER_BUDGETS: dict[str, int] = {
    # Anthropic Claude models
    ("anthropic", "claude-opus-4-1"): 40_000,  # requests/min → ~tokens/min estimate
    ("anthropic", "claude-sonnet-4-6"): 40_000,
    ("anthropic", "claude-3-5-sonnet"): 40_000,
    # OpenAI GPT models
    ("openai", "gpt-4o"): 2_000_000,  # tokens per minute
    ("openai", "gpt-4o-mini"): 2_000_000,
    ("openai", "gpt-4-turbo"): 2_000_000,
    # Google Gemini
    ("gemini", "gemini-2.0-flash"): 4_000_000,
    ("gemini", "gemini-1.5-pro"): 4_000_000,
}


def get_budget_for_provider(provider: str, model: str) -> TokenBudget | None:
    """
    Get or create a TokenBudget for a provider+model pair.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-4o-mini')

    Returns:
        TokenBudget instance, or None if not configured
    """
    key = (provider, model)
    tpm = PROVIDER_BUDGETS.get(key)
    if tpm is None:
        logger.warning(f"No TPM budget configured for {provider}:{model}")
        return None
    return TokenBudget(tpm)
