"""
src/services/cost_meter.py — AI provider cost tracking.

Bonus 2 (+2 pts): Records token counts and estimated costs for every AI call.
Stores data in SQLite for later analysis via `cost-report` CLI command.

Pricing data from official provider documentation (as of May 2024):
- Anthropic Claude: https://www.anthropic.com/pricing
- OpenAI GPT: https://openai.com/pricing/
- Google Gemini: https://ai.google.dev/pricing
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Pricing per 1K tokens (prompt, completion)
PRICING_USD_PER_1K_TOKENS: dict[tuple[str, str], tuple[float, float]] = {
    # (provider, model): (prompt_price_per_1k, completion_price_per_1k)
    # Anthropic
    ("anthropic", "claude-opus-4-1"): (0.015, 0.075),
    ("anthropic", "claude-sonnet-4-6"): (0.003, 0.015),
    ("anthropic", "claude-3-5-sonnet"): (0.003, 0.015),
    ("anthropic", "claude-3-5-haiku"): (0.00080, 0.004),
    # OpenAI
    ("openai", "gpt-4o"): (0.0025, 0.010),
    ("openai", "gpt-4o-mini"): (0.00015, 0.0006),
    ("openai", "gpt-4-turbo"): (0.01, 0.03),
    ("openai", "gpt-3.5-turbo"): (0.0005, 0.0015),
    # Google Gemini
    ("gemini", "gemini-2.0-flash"): (0.000075, 0.0003),
    ("gemini", "gemini-1.5-pro"): (0.0035, 0.0105),
    ("gemini", "gemini-1.5-flash"): (0.0001875, 0.000375),
    # Embedding models
    ("openai", "text-embedding-3-small"): (0.00002, 0.0),
    ("openai", "text-embedding-3-large"): (0.00013, 0.0),
    ("gemini", "text-embedding-004"): (0.00001, 0.0),
}


@dataclass
class CostRecord:
    """A single AI provider call cost."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    dollars: float
    timestamp: str  # ISO format


class CostMeter:
    """Track and analyze AI provider costs."""

    def __init__(self, db_path: str | Path = "artifacts/cost_log.sqlite") -> None:
        """
        Initialize cost meter with SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create or open SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                dollars REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON cost_records(timestamp)"
        )
        conn.commit()
        conn.close()

    def estimate_cost(self, provider: str, model: str, prompt_t: int, completion_t: int) -> float:
        """
        Estimate cost for a provider call.

        Args:
            provider: Provider name (e.g., 'openai')
            model: Model name (e.g., 'gpt-4o-mini')
            prompt_t: Number of prompt tokens
            completion_t: Number of completion tokens

        Returns:
            Estimated cost in USD (0.0 if pricing unknown)
        """
        pricing = PRICING_USD_PER_1K_TOKENS.get((provider, model), (0.0, 0.0))
        prompt_price, completion_price = pricing
        cost = (prompt_t / 1000.0) * prompt_price + (completion_t / 1000.0) * completion_price
        return cost

    async def record(
        self, provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> CostRecord:
        """
        Record a cost entry for an AI call.

        Args:
            provider: Provider name
            model: Model name
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion

        Returns:
            CostRecord with estimated cost
        """
        dollars = self.estimate_cost(provider, model, prompt_tokens, completion_tokens)
        record = CostRecord(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            dollars=dollars,
            timestamp=datetime.utcnow().isoformat(),
        )

        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cost_records
                (provider, model, prompt_tokens, completion_tokens, dollars, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.provider,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.dollars,
                    record.timestamp,
                ),
            )
            conn.commit()
            conn.close()

        logger.info(f"Cost record: {provider}:{model} → ${dollars:.6f}")
        return record

    def get_report(self, since_hours: int = 24) -> dict:
        """
        Generate cost report for the last N hours.

        Args:
            since_hours: Hours to include (default 24)

        Returns:
            Report dict with totals, by-provider, and top-5 expensive calls
        """
        since_time = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total cost
        cursor.execute(
            "SELECT SUM(dollars) as total FROM cost_records WHERE timestamp > ?",
            (since_time,),
        )
        total_result = cursor.fetchone()
        total_cost = total_result["total"] or 0.0

        # By provider
        cursor.execute(
            """
            SELECT provider, SUM(dollars) as total, COUNT(*) as calls
            FROM cost_records
            WHERE timestamp > ?
            GROUP BY provider
            ORDER BY total DESC
            """,
            (since_time,),
        )
        by_provider = [dict(row) for row in cursor.fetchall()]

        # Top 5 most expensive calls
        cursor.execute(
            """
            SELECT provider, model, prompt_tokens, completion_tokens, dollars, timestamp
            FROM cost_records
            WHERE timestamp > ?
            ORDER BY dollars DESC
            LIMIT 5
            """,
            (since_time,),
        )
        top_expensive = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "period_hours": since_hours,
            "total_cost_usd": round(total_cost, 4),
            "total_calls": sum(p["calls"] for p in by_provider),
            "by_provider": by_provider,
            "top_5_expensive": top_expensive,
        }


# Global singleton
_cost_meter: CostMeter | None = None


def get_cost_meter(db_path: str | Path = "artifacts/cost_log.sqlite") -> CostMeter:
    """Get or create global cost meter."""
    global _cost_meter
    if _cost_meter is None:
        _cost_meter = CostMeter(db_path)
    return _cost_meter
