"""
tests/test_failover.py — Tests for multi-provider failover (Bonus 1).

Chaos tests demonstrating that when the primary provider fails,
the system transparently retries against the secondary provider.
"""

import pytest
from unittest.mock import Mock
from ai.providers.base import ProviderError
from src.ai.providers.failover import FailoverVLM, FailoverEmbedding


class TestFailoverVLM:
    """Tests for VLM failover wrapper."""

    def test_failover_vlm_uses_secondary_when_primary_fails(self):
        """Chaos test: primary fails with ProviderError, secondary succeeds."""
        # Arrange
        primary = Mock()
        primary.describe.side_effect = ProviderError("upstream down")
        
        secondary = Mock()
        secondary.describe.return_value = '{"object": "umbrella", "color": "black"}'
        
        failover = FailoverVLM([primary, secondary])
        
        # Act
        result = failover.describe("test.jpg", "describe this")
        
        # Assert
        assert result == '{"object": "umbrella", "color": "black"}'
        assert primary.describe.call_count == 1
        assert secondary.describe.call_count == 1

    def test_failover_vlm_uses_secondary_on_timeout(self):
        """Chaos test: primary times out, secondary succeeds."""
        primary = Mock()
        primary.describe.side_effect = TimeoutError("request timeout")
        
        secondary = Mock()
        secondary.describe.return_value = '{"object": "keys"}'
        
        failover = FailoverVLM([primary, secondary])
        
        result = failover.describe("keys.jpg", "describe")
        
        assert result == '{"object": "keys"}'
        assert secondary.describe.call_count == 1

    def test_failover_vlm_raises_when_all_providers_fail(self):
        """All providers fail → ProviderError raised with details."""
        primary = Mock()
        primary.describe.side_effect = ProviderError("error 1")
        
        secondary = Mock()
        secondary.describe.side_effect = ProviderError("error 2")
        
        failover = FailoverVLM([primary, secondary])
        
        with pytest.raises(ProviderError, match="All 2 VLM providers failed"):
            failover.describe("test.jpg", "describe")
        
        # Both should have been tried
        assert primary.describe.call_count == 1
        assert secondary.describe.call_count == 1

    def test_failover_vlm_succeeds_on_first_provider(self):
        """When primary succeeds, secondary is never called."""
        primary = Mock()
        primary.describe.return_value = '{"object": "wallet"}'
        
        secondary = Mock()
        secondary.describe.return_value = '{"object": "should not be called"}'
        
        failover = FailoverVLM([primary, secondary])
        
        result = failover.describe("wallet.jpg", "describe")
        
        assert result == '{"object": "wallet"}'
        assert primary.describe.call_count == 1
        assert secondary.describe.call_count == 0  # Never called

    def test_failover_vlm_requires_at_least_one_provider(self):
        """Empty provider list raises ValueError."""
        with pytest.raises(ValueError, match="At least one VLM provider required"):
            FailoverVLM([])

    def test_failover_vlm_with_three_providers(self):
        """Failover works with 3+ providers."""
        primary = Mock()
        primary.describe.side_effect = ProviderError("fail 1")
        
        secondary = Mock()
        secondary.describe.side_effect = ProviderError("fail 2")
        
        tertiary = Mock()
        tertiary.describe.return_value = '{"object": "backpack"}'
        
        failover = FailoverVLM([primary, secondary, tertiary])
        
        result = failover.describe("backpack.jpg", "describe")
        
        assert result == '{"object": "backpack"}'
        assert primary.describe.call_count == 1
        assert secondary.describe.call_count == 1
        assert tertiary.describe.call_count == 1


class TestFailoverEmbedding:
    """Tests for embedding failover wrapper."""

    def test_failover_embedding_uses_secondary_when_primary_fails(self):
        """Chaos test: primary embedding fails, secondary succeeds."""
        primary = Mock()
        primary.embed.side_effect = ProviderError("rate limited")
        
        secondary = Mock()
        secondary.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        failover = FailoverEmbedding([primary, secondary])
        
        result = failover.embed("test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert primary.embed.call_count == 1
        assert secondary.embed.call_count == 1

    def test_failover_embedding_raises_when_all_fail(self):
        """All embedding providers fail → ProviderError raised."""
        primary = Mock()
        primary.embed.side_effect = ProviderError("error 1")
        
        secondary = Mock()
        secondary.embed.side_effect = ProviderError("error 2")
        
        failover = FailoverEmbedding([primary, secondary])
        
        with pytest.raises(ProviderError, match="All 2 embedding providers failed"):
            failover.embed("test text")

    def test_failover_embedding_succeeds_on_first(self):
        """When primary succeeds, secondary is never called."""
        primary = Mock()
        primary.embed.return_value = [0.9, 0.8, 0.7]
        
        secondary = Mock()
        
        failover = FailoverEmbedding([primary, secondary])
        
        result = failover.embed("test")
        
        assert result == [0.9, 0.8, 0.7]
        assert primary.embed.call_count == 1
        assert secondary.embed.call_count == 0

    def test_failover_embedding_requires_at_least_one_provider(self):
        """Empty provider list raises ValueError."""
        with pytest.raises(ValueError, match="At least one embedding provider required"):
            FailoverEmbedding([])

    @pytest.mark.asyncio
    async def test_failover_embedding_async_uses_secondary(self):
        """Async failover: primary fails, secondary succeeds."""
        primary = Mock()
        primary.embed_async.side_effect = ProviderError("async fail")
        
        secondary = Mock()
        secondary.embed_async.return_value = [0.5, 0.5, 0.5]
        
        failover = FailoverEmbedding([primary, secondary])
        
        result = await failover.embed_async("async test")
        
        assert result == [0.5, 0.5, 0.5]
        assert primary.embed_async.call_count == 1
        assert secondary.embed_async.call_count == 1

    @pytest.mark.asyncio
    async def test_failover_embedding_async_falls_back_to_sync(self):
        """Async failover falls back to sync embed if embed_async not available."""
        primary = Mock()
        primary.embed.side_effect = ProviderError("fail")
        
        secondary = Mock()
        secondary.embed.return_value = [0.1, 0.2]
        # secondary doesn't have embed_async
        
        failover = FailoverEmbedding([primary, secondary])
        
        result = await failover.embed_async("test")
        
        assert result == [0.1, 0.2]
        assert secondary.embed.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
