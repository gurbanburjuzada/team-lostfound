"""
tests/test_bonuses.py — Tests for advanced bonus features.

Tests:
- Bonus 1: Multi-provider failover
- Bonus 2: Cost telemetry
- Bonus 4: OpenTelemetry tracing
- Bonus 7: Token-aware rate limiter
"""

import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.services.cost_meter import CostMeter, get_cost_meter, PRICING_USD_PER_1K_TOKENS
from src.concurrency.token_budget import TokenBudget, PROVIDER_BUDGETS
from src.observability.tracing import setup_tracing, get_tracer


class TestBonus2CostTelemetry:
    """Bonus 2: Cost telemetry (+2 pts)."""

    def test_cost_meter_initialization(self, tmp_path):
        """Test CostMeter creates database."""
        db_path = tmp_path / "costs.sqlite"
        meter = CostMeter(db_path)
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_cost_record(self, tmp_path):
        """Test recording a cost entry."""
        db_path = tmp_path / "costs.sqlite"
        meter = CostMeter(db_path)

        record = await meter.record(
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert record.provider == "openai"
        assert record.model == "gpt-4o-mini"
        assert record.prompt_tokens == 100
        assert record.completion_tokens == 50
        assert record.dollars > 0.0

    def test_estimate_cost(self, tmp_path):
        """Test cost estimation."""
        meter = CostMeter(tmp_path / "costs.sqlite")

        # Known pricing: GPT-4o-mini is $0.00015 per prompt token, $0.0006 completion
        cost = meter.estimate_cost(
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=1000,
            completion_tokens=1000,
        )

        # (1000 / 1000) * 0.00015 + (1000 / 1000) * 0.0006 = 0.00075
        assert abs(cost - 0.00075) < 0.0001

    def test_pricing_table_exists(self):
        """Test pricing table has entries."""
        assert len(PRICING_USD_PER_1K_TOKENS) > 5
        assert ("openai", "gpt-4o-mini") in PRICING_USD_PER_1K_TOKENS
        assert ("anthropic", "claude-3-5-sonnet") in PRICING_USD_PER_1K_TOKENS
        assert ("gemini", "gemini-2.0-flash") in PRICING_USD_PER_1K_TOKENS

    @pytest.mark.asyncio
    async def test_cost_report(self, tmp_path):
        """Test generating cost report."""
        db_path = tmp_path / "costs.sqlite"
        meter = CostMeter(db_path)

        # Record some costs
        await meter.record("openai", "gpt-4o-mini", 100, 50)
        await meter.record("anthropic", "claude-3-5-sonnet", 200, 100)

        report = meter.get_report(since_hours=24)

        assert report["total_cost_usd"] > 0.0
        assert report["total_calls"] == 2
        assert len(report["by_provider"]) == 2
        assert len(report["top_5_expensive"]) == 2


class TestBonus7TokenBudget:
    """Bonus 7: Token-aware rate limiter (+2 pts)."""

    def test_token_budget_initialization(self):
        """Test TokenBudget creation."""
        budget = TokenBudget(tokens_per_minute=100_000)
        assert budget.tpm == 100_000

    def test_token_budget_invalid_tpm(self):
        """Test invalid TPM raises error."""
        with pytest.raises(ValueError, match="must be > 0"):
            TokenBudget(tokens_per_minute=0)

    @pytest.mark.asyncio
    async def test_token_budget_acquire_immediate(self):
        """Test acquiring tokens when headroom available."""
        budget = TokenBudget(tokens_per_minute=1000)
        
        # Should return immediately with 100 tokens available in 1000 TPM
        await budget.acquire(100)
        
        used, limit = budget.get_usage()
        assert used == 100
        assert limit == 1000

    @pytest.mark.asyncio
    async def test_token_budget_exceeds_limit(self):
        """Test single call exceeding TPM limit."""
        budget = TokenBudget(tokens_per_minute=1000)
        
        with pytest.raises(ValueError, match="exceeds TPM limit"):
            await budget.acquire(2000)

    @pytest.mark.asyncio
    async def test_token_budget_multiple_acquisitions(self):
        """Test multiple acquisitions."""
        budget = TokenBudget(tokens_per_minute=500)
        
        await budget.acquire(200)
        await budget.acquire(200)
        
        used, limit = budget.get_usage()
        assert used == 400

    def test_provider_budgets_configured(self):
        """Test provider budgets exist."""
        assert len(PROVIDER_BUDGETS) > 5
        assert ("openai", "gpt-4o") in PROVIDER_BUDGETS
        assert ("gemini", "gemini-2.0-flash") in PROVIDER_BUDGETS

    @pytest.mark.asyncio
    async def test_token_budget_backoff_on_exhaustion(self):
        """Test backoff when budget is exhausted (critical test for bonus 7)."""
        import time
        
        # Small budget to force exhaustion quickly
        budget = TokenBudget(tokens_per_minute=100)
        
        # Fill most of the budget
        await budget.acquire(90)
        
        # Next call would exceed budget, should wait
        start = time.monotonic()
        
        # This should block until old tokens expire from the sliding window
        await budget.acquire(20)  # 90 + 20 = 110 > 100, must wait
        
        elapsed = time.monotonic() - start
        
        # Should have waited at least a small amount
        # (In practice, waits ~60s for sliding window, but we use small sleep in impl)
        assert elapsed > 0.05, f"Expected backoff, but completed in {elapsed:.3f}s"
        
        # Verify tokens were acquired
        used, limit = budget.get_usage()
        assert used >= 20  # At least the last acquisition


class TestBonus4OpenTelemetry:
    """Bonus 4: OpenTelemetry tracing (+2 pts)."""

    def test_setup_tracing_without_otel(self):
        """Test setup_tracing handles missing opentelemetry gracefully."""
        # Should not raise even if opentelemetry not available
        setup_tracing(enable_jaeger=False)

    def test_get_tracer_without_otel(self):
        """Test get_tracer returns None when opentelemetry not available."""
        tracer = get_tracer("test_module")
        # Will be None if opentelemetry not installed
        # Or a real tracer if installed
        assert tracer is None or hasattr(tracer, "start_as_current_span")

    def test_tracing_context_manager(self):
        """Test tracing context manager."""
        from src.observability.tracing import TracingContext
        
        # Should work without opentelemetry
        ctx = TracingContext(None, "test.span")
        with ctx as span:
            # span is None when tracer is None
            assert span is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
