"""
test_services.py - offline unit tests for src/services/ai_service.py.

No network, no API keys. Uses FakeVLM and FakeEmbedder from conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.providers.base import ProviderError
from src.services.ai_service import AIService


@pytest.fixture
def svc(fake_vlm, fake_embedder):
    return AIService(vlm=fake_vlm, embedder=fake_embedder, max_attempts=3, wait_seconds=0.001)


# -- describe_item ------------------------------------------------------------

def test_describe_item_returns_item_description(svc, sample_image):
    result = svc.describe_item(sample_image, "black umbrella")
    assert result.object_class == "umbrella"
    assert result.colors == ["black"]
    assert 0.0 <= result.confidence <= 1.0


def test_describe_item_retries_and_succeeds(fake_vlm, fake_embedder, sample_image):
    call_count = 0
    original = fake_vlm.describe

    def flaky(image_path, prompt, *, json_schema=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderError("transient")
        return original(image_path, prompt, json_schema=json_schema)

    fake_vlm.describe = flaky
    svc = AIService(vlm=fake_vlm, embedder=fake_embedder, max_attempts=3, wait_seconds=0.001)
    result = svc.describe_item(sample_image, "test")
    assert call_count == 3
    assert result.object_class == "umbrella"


def test_describe_item_raises_after_max_retries(fake_vlm, fake_embedder, sample_image):
    def always_fail(image_path, prompt, *, json_schema=None):
        raise ProviderError("permanent")

    fake_vlm.describe = always_fail
    svc = AIService(vlm=fake_vlm, embedder=fake_embedder, max_attempts=2, wait_seconds=0.001)
    with pytest.raises(ProviderError):
        svc.describe_item(sample_image, "test")


# -- embed + cache ------------------------------------------------------------

def test_embed_returns_unit_vector(svc):
    vec = svc.embed("hello world")
    assert isinstance(vec, np.ndarray)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_embed_cache_hit_returns_same_object(svc):
    v1 = svc.embed("same text")
    v2 = svc.embed("same text")
    assert v1 is v2


def test_embed_different_texts_are_different(svc):
    v1 = svc.embed("alpha")
    v2 = svc.embed("beta")
    assert not np.allclose(v1, v2)


def test_embed_provider_called_once_for_repeated_text(fake_vlm, fake_embedder):
    call_count = 0
    original = fake_embedder.embed

    def counting(text):
        nonlocal call_count
        call_count += 1
        return original(text)

    fake_embedder.embed = counting
    svc = AIService(vlm=fake_vlm, embedder=fake_embedder, max_attempts=1, wait_seconds=0.001)
    svc.embed("repeat")
    svc.embed("repeat")
    svc.embed("repeat")
    assert call_count == 1


def test_clear_cache_forces_re_embed(fake_vlm, fake_embedder):
    call_count = 0
    original = fake_embedder.embed

    def counting(text):
        nonlocal call_count
        call_count += 1
        return original(text)

    fake_embedder.embed = counting
    svc = AIService(vlm=fake_vlm, embedder=fake_embedder, max_attempts=1, wait_seconds=0.001)
    svc.embed("text")
    svc.clear_cache()
    svc.embed("text")
    assert call_count == 2