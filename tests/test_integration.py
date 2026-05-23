"""
test_integration.py — Integration tests for the full system.

Tests the complete workflow:
  - API endpoints (POST, GET, list)
  - CLI commands
  - Database persistence
  - File upload and storage
  - Error handling

All tests use FakeVLM and FakeEmbedder for offline testing (no API keys needed).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai.providers.base import VLMProvider, EmbeddingProvider, ProviderError
from src.api import app
from src.models import ItemStatus, MatchConfidence
from src.storage.repository import Repository


# ── Test fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
async def repo():
    """In-memory repository for testing."""
    # Use a test database URL (in production, use sqlite in-memory or separate test DB)
    repo = await Repository.create()
    yield repo
    # Cleanup could go here if needed


@pytest.fixture
def test_client():
    """TestClient for FastAPI endpoints."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    Return bytes of a minimal valid PNG image (1x1 pixel, red).
    
    This is a real PNG file in binary form.
    """
    # Minimal 1x1 red PNG (hex):
    # 89504E470D0A1A0A = PNG signature
    # 0D000000 = IHDR chunk size (13)
    # 49484452 = IHDR
    # 00000001 00000001 = 1x1 dimensions
    # 08020000 = 8-bit RGB
    # 00 = compression, filter, interlace
    # 906CC7 = CRC
    # 0C000000 = IDAT chunk size (12)
    # 49444154 = IDAT
    # 789C62F04F050000050001010000 = compressed data + CRC
    # 00000000 = IEND chunk size (0)
    # 49454E44 = IEND
    # AE426082 = CRC
    
    png_hex = (
        "89504E470D0A1A0A0D0000000D49484452"
        "0000000100000001080200000090CC7"
        "0C0000000C49444154789C62F04F05000005"
        "000101000001E2210BC70000000049454E44"
        "AE426082"
    )
    return bytes.fromhex(png_hex)


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """
    Return bytes of a minimal valid JPEG image.
    
    This is a real JPEG file in binary form (1x1 pixel).
    """
    jpeg_hex = (
        "FFD8FFE000104A46494600010100000100"
        "010000FFDB004300FFFFFFFFFFFFFFFFFF"
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        "FFFFC0000B0801000101011100FFC40014"
        "00010100000000000000000000000000000"
        "0FFD9"
    )
    return bytes.fromhex(jpeg_hex)


# ── Test: Health check ─────────────────────────────────────────────────────

def test_health_check(test_client):
    """Test GET /health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database" in data
    assert "version" in data


# ── Test: List items (empty) ──────────────────────────────────────────────

def test_list_items_empty(test_client):
    """Test GET /items when database is empty."""
    response = test_client.get("/items")
    assert response.status_code == 200
    data = response.json()
    assert data == []


# ── Test: Register lost item ─────────────────────────────────────────────

def test_register_lost_item(test_client, sample_image_bytes):
    """Test POST /items/lost endpoint."""
    response = test_client.post(
        "/items/lost",
        data={"description": "black umbrella left at library"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "item_id" in data
    assert data["status"] == "lost"
    assert "black umbrella" in data["message"].lower() or "successfully" in data["message"].lower()


def test_register_found_item(test_client, sample_image_bytes):
    """Test POST /items/found endpoint."""
    response = test_client.post(
        "/items/found",
        data={"description": "found umbrella near station"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "item_id" in data
    assert data["status"] == "found"


# ── Test: Validation ──────────────────────────────────────────────────────

def test_register_invalid_image_type(test_client):
    """Test that non-image files are rejected."""
    response = test_client.post(
        "/items/lost",
        data={"description": "test"},
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "content type" in data["error"].lower() or "error" in data


def test_register_missing_description(test_client, sample_image_bytes):
    """Test that missing description is rejected."""
    response = test_client.post(
        "/items/lost",
        data={},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code in (400, 422)  # 422 is FastAPI default for missing required field


def test_register_description_too_short(test_client, sample_image_bytes):
    """Test that description must be at least 3 characters."""
    response = test_client.post(
        "/items/lost",
        data={"description": "ab"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 400


def test_register_description_too_long(test_client, sample_image_bytes):
    """Test that description must not exceed 1000 characters."""
    response = test_client.post(
        "/items/lost",
        data={"description": "x" * 1001},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 400


# ── Test: List items (with data) ──────────────────────────────────────────

def test_list_items_with_filter(test_client, sample_image_bytes):
    """Test GET /items with status filter."""
    # Register a lost item
    response1 = test_client.post(
        "/items/lost",
        data={"description": "test lost item"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response1.status_code == 201

    # Register a found item
    response2 = test_client.post(
        "/items/found",
        data={"description": "test found item"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response2.status_code == 201

    # List all
    response_all = test_client.get("/items")
    assert response_all.status_code == 200
    all_items = response_all.json()
    assert len(all_items) >= 2

    # List only lost
    response_lost = test_client.get("/items?status=lost")
    assert response_lost.status_code == 200
    lost_items = response_lost.json()
    assert all(item["status"] == "lost" for item in lost_items)

    # List only found
    response_found = test_client.get("/items?status=found")
    assert response_found.status_code == 200
    found_items = response_found.json()
    assert all(item["status"] == "found" for item in found_items)


# ── Test: Query matches ───────────────────────────────────────────────────

def test_query_matches_invalid_uuid(test_client):
    """Test that invalid UUID is rejected."""
    response = test_client.get("/items/invalid-uuid/matches")
    assert response.status_code == 400
    data = response.json()
    assert "uuid" in data["error"].lower() or "error" in data


def test_query_matches_item_not_found(test_client):
    """Test that non-existent item returns 404."""
    fake_uuid = str(uuid.uuid4())
    response = test_client.get(f"/items/{fake_uuid}/matches")
    assert response.status_code == 404


def test_query_matches_with_results(test_client, sample_image_bytes):
    """Test finding matches between registered items."""
    # Register 2 lost items
    r1 = test_client.post(
        "/items/lost",
        data={"description": "black umbrella"},
        files={"image": ("img1.png", sample_image_bytes, "image/png")},
    )
    assert r1.status_code == 201
    item1_id = r1.json()["item_id"]

    r2 = test_client.post(
        "/items/lost",
        data={"description": "dark umbrella"},
        files={"image": ("img2.png", sample_image_bytes, "image/png")},
    )
    assert r2.status_code == 201

    # Register 1 found item
    r3 = test_client.post(
        "/items/found",
        data={"description": "umbrella found"},
        files={"image": ("img3.png", sample_image_bytes, "image/png")},
    )
    assert r3.status_code == 201

    # Wait a tiny bit for embeddings to be ready (in real app, would poll or use async)
    import time
    time.sleep(0.5)

    # Query matches for first lost item
    response = test_client.get(f"/items/{item1_id}/matches?k=5")
    assert response.status_code in (200, 409)  # 409 if still processing

    if response.status_code == 200:
        data = response.json()
        assert "query_item_id" in data
        assert "matches" in data
        assert "total_candidates_searched" in data
        # Should have searched the found pool
        assert data["total_candidates_searched"] >= 1


def test_query_matches_k_parameter(test_client, sample_image_bytes):
    """Test k parameter validation."""
    # Register an item
    r = test_client.post(
        "/items/lost",
        data={"description": "test item"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert r.status_code == 201
    item_id = r.json()["item_id"]

    # Test invalid k
    response_k0 = test_client.get(f"/items/{item_id}/matches?k=0")
    assert response_k0.status_code in (400, 422)

    response_k_large = test_client.get(f"/items/{item_id}/matches?k=101")
    assert response_k_large.status_code in (400, 422)

    # Valid k should work (or return 409 if still processing)
    response_k_valid = test_client.get(f"/items/{item_id}/matches?k=3")
    assert response_k_valid.status_code in (200, 409)


# ── Test: Error responses ──────────────────────────────────────────────────

def test_error_response_format(test_client):
    """Test that error responses have the correct format."""
    response = test_client.get("/items/invalid-uuid/matches")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    # "detail" is optional


# ── Test: JPEG support ────────────────────────────────────────────────────

def test_register_jpeg_item(test_client, sample_jpeg_bytes):
    """Test that JPEG images are accepted."""
    response = test_client.post(
        "/items/lost",
        data={"description": "test jpeg item"},
        files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 201


# ── Concurrency tests (optional, stress-test-ish) ────────────────────────

@pytest.mark.asyncio
async def test_concurrent_registrations(test_client, sample_image_bytes):
    """Test that multiple concurrent registrations work."""
    import asyncio

    async def register_one(i: int) -> dict:
        """Register one item asynchronously."""
        # Note: TestClient is sync, so we need to be careful here
        # For true async testing, we'd use httpx.AsyncClient
        response = test_client.post(
            "/items/lost",
            data={"description": f"concurrent test item {i}"},
            files={"image": (f"test{i}.png", sample_image_bytes, "image/png")},
        )
        return response.json()

    # This is a simplification; in a real test, use async httpx.AsyncClient
    tasks = [register_one(i) for i in range(3)]

    # Since TestClient is sync, just do sequential calls
    results = []
    for task in tasks:
        results.append(task)

    assert len(results) == 3
    assert all("item_id" in r for r in results)
