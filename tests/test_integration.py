"""
test_integration.py — Integration tests for the full system.

Tests the complete workflow:
  - API endpoints (POST, GET, list)
  - Database persistence
  - File upload and storage
  - Error handling

All tests use FakeVLM and FakeEmbedder for offline testing (no API keys needed).
The FakeRepository replaces PostgreSQL with an in-memory store, so no database
is required either.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai.providers.base import VLMProvider, EmbeddingProvider, ProviderError
from src.api import app, get_repo
from src.models import ItemRecord, ItemStatus, ItemSummary, MatchConfidence
from src.services.ai_service import AIService


# ---------------------------------------------------------------------------
# FakeRepository — in-memory replacement for PostgreSQL
# ---------------------------------------------------------------------------

class FakeRepository:
    """
    In-memory repository that mimics src.storage.repository.Repository.
    No database required; all data lives in a plain dict.
    """

    def __init__(self, tmp_path: Path) -> None:
        self._items: dict[uuid.UUID, ItemRecord] = {}
        self._tmp_path = tmp_path

    async def save_item(
        self,
        status: ItemStatus,
        description: str,
        source_image_path: str,
        vlm_description: dict | None = None,
        embedding: np.ndarray | None = None,
    ) -> ItemRecord:
        item_id = uuid.uuid4()
        record = ItemRecord(
            id=item_id,
            status=status,
            description=description,
            image_path=source_image_path,
            vlm_description=json.dumps(vlm_description) if vlm_description else None,
            embedding=embedding.tolist() if embedding is not None else None,
            created_at=datetime.now(timezone.utc),
        )
        self._items[item_id] = record
        return record

    async def update_embedding(self, item_id: uuid.UUID, embedding: np.ndarray) -> None:
        record = self._items.get(item_id)
        if record is None:
            raise ValueError(f"Item {item_id} not found")
        # Pydantic v2 models are immutable by default; use model_copy
        updated = record.model_copy(update={"embedding": embedding.tolist()})
        self._items[item_id] = updated

    async def get_item(self, item_id: uuid.UUID) -> Optional[ItemRecord]:
        return self._items.get(item_id)

    async def list_items(self, status: Optional[ItemStatus] = None) -> list[ItemSummary]:
        items = list(self._items.values())
        if status:
            items = [i for i in items if i.status == status]
        return [
            ItemSummary(
                id=i.id,
                status=i.status,
                description=i.description,
                image_path=i.image_path,
                created_at=i.created_at,
            )
            for i in items
        ]

    async def get_items_with_embeddings(
        self, status: ItemStatus
    ) -> list[tuple[ItemRecord, np.ndarray]]:
        result = []
        for record in self._items.values():
            if record.status == status and record.embedding is not None:
                result.append((record, np.array(record.embedding, dtype=np.float32)))
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_repo(tmp_path) -> FakeRepository:
    """In-memory repository; no database required."""
    return FakeRepository(tmp_path)


@pytest.fixture
def test_client(fake_vlm, fake_embedder, fake_repo, tmp_path):
    """
    TestClient for FastAPI endpoints.

    Patches:
      - Repository.create() → returns fake_repo (so startup doesn't hit Postgres)
      - src.api.ai_service   → AIService with fake VLM/embedder (no API keys)
      - get_repo dependency  → returns fake_repo
    """
    fake_svc = AIService(
        vlm=fake_vlm,
        embedder=fake_embedder,
        max_attempts=1,
        wait_seconds=0.001,
    )

    # Override the FastAPI dependency so endpoints use our in-memory repo
    app.dependency_overrides[get_repo] = lambda: fake_repo

    with (
        patch("src.storage.repository.Repository.create", AsyncMock(return_value=fake_repo)),
        patch("src.api.ai_service", fake_svc),
    ):
        with TestClient(app) as client:
            yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    Minimal valid 1×1 PNG image as bytes.

    These are the same bytes used in conftest.py's sample_image fixture,
    verified to be a correctly-formed PNG.
    """
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000"
        "00907753de0000000c4944415408d76360000000000004000146a13a"
        "020000000049454e44ae426082"
    )


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Minimal valid 1×1 grayscale JPEG image as bytes."""
    return bytes.fromhex(
        "FFD8FFE000104A46494600010100000100010000"
        "FFDB004300010101010101010101010101010101"
        "010101010101010101010101010101010101010101"
        "010101010101010101010101010101010101010101"
        "FFC0000801000001011100FFC4001F000001050101"
        "01010101010000000000000000010203040506070809"
        "0A0BFFDA00080101000003F00AFFD9"
    )


# ---------------------------------------------------------------------------
# Tests: Health check
# ---------------------------------------------------------------------------

def test_health_check(test_client):
    """GET /health should return status ok."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database" in data
    assert "version" in data


# ---------------------------------------------------------------------------
# Tests: List items
# ---------------------------------------------------------------------------

def test_list_items_empty(test_client):
    """GET /items on an empty repo returns an empty list."""
    response = test_client.get("/items")
    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# Tests: Register items
# ---------------------------------------------------------------------------

def test_register_lost_item(test_client, sample_image_bytes):
    """POST /items/lost registers a lost item and returns 201."""
    response = test_client.post(
        "/items/lost",
        data={"description": "black umbrella left at library"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "item_id" in data
    assert data["status"] == "lost"
    assert "message" in data


def test_register_found_item(test_client, sample_image_bytes):
    """POST /items/found registers a found item and returns 201."""
    response = test_client.post(
        "/items/found",
        data={"description": "found umbrella near station"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "item_id" in data
    assert data["status"] == "found"


def test_register_jpeg_item(test_client, sample_jpeg_bytes):
    """POST /items/lost with a JPEG image is accepted."""
    response = test_client.post(
        "/items/lost",
        data={"description": "test jpeg item"},
        files={"image": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 201


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------

def test_register_invalid_image_type(test_client):
    """Non-image content type is rejected with 400."""
    response = test_client.post(
        "/items/lost",
        data={"description": "test"},
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_register_missing_description(test_client, sample_image_bytes):
    """Missing description field yields 400 or 422."""
    response = test_client.post(
        "/items/lost",
        data={},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code in (400, 422)


def test_register_description_too_short(test_client, sample_image_bytes):
    """Description shorter than 3 characters is rejected (400 or 422)."""
    response = test_client.post(
        "/items/lost",
        data={"description": "ab"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code in (400, 422)


def test_register_description_too_long(test_client, sample_image_bytes):
    """Description longer than 1000 characters is rejected (400 or 422)."""
    response = test_client.post(
        "/items/lost",
        data={"description": "x" * 1001},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Tests: List items with filter
# ---------------------------------------------------------------------------

def test_list_items_with_filter(test_client, sample_image_bytes):
    """GET /items?status=lost returns only lost items."""
    test_client.post(
        "/items/lost",
        data={"description": "test lost item"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    test_client.post(
        "/items/found",
        data={"description": "test found item"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )

    resp_all = test_client.get("/items")
    assert resp_all.status_code == 200
    assert len(resp_all.json()) >= 2

    resp_lost = test_client.get("/items?status=lost")
    assert resp_lost.status_code == 200
    assert all(i["status"] == "lost" for i in resp_lost.json())

    resp_found = test_client.get("/items?status=found")
    assert resp_found.status_code == 200
    assert all(i["status"] == "found" for i in resp_found.json())


# ---------------------------------------------------------------------------
# Tests: Query matches
# ---------------------------------------------------------------------------

def test_query_matches_invalid_uuid(test_client):
    """GET /items/<invalid>/matches returns 400 with error key."""
    response = test_client.get("/items/not-a-uuid/matches")
    assert response.status_code == 400
    assert "error" in response.json()


def test_query_matches_item_not_found(test_client):
    """GET /items/<unknown-uuid>/matches returns 404."""
    response = test_client.get(f"/items/{uuid.uuid4()}/matches")
    assert response.status_code == 404


def test_query_matches_with_results(test_client, sample_image_bytes):
    """End-to-end: register lost + found items, then query matches."""
    r1 = test_client.post(
        "/items/lost",
        data={"description": "black umbrella"},
        files={"image": ("img1.png", sample_image_bytes, "image/png")},
    )
    assert r1.status_code == 201
    item1_id = r1.json()["item_id"]

    test_client.post(
        "/items/found",
        data={"description": "umbrella found near exit"},
        files={"image": ("img2.png", sample_image_bytes, "image/png")},
    )

    response = test_client.get(f"/items/{item1_id}/matches?k=5")
    # Either matches returned or item has no embedding yet (409)
    assert response.status_code in (200, 409)
    if response.status_code == 200:
        data = response.json()
        assert "query_item_id" in data
        assert "matches" in data
        assert "total_candidates_searched" in data


def test_query_matches_k_parameter(test_client, sample_image_bytes):
    """k=0 and k=101 are invalid; k=3 is valid."""
    r = test_client.post(
        "/items/lost",
        data={"description": "test item"},
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert r.status_code == 201
    item_id = r.json()["item_id"]

    assert test_client.get(f"/items/{item_id}/matches?k=0").status_code in (400, 422)
    assert test_client.get(f"/items/{item_id}/matches?k=101").status_code in (400, 422)
    assert test_client.get(f"/items/{item_id}/matches?k=3").status_code in (200, 409)


# ---------------------------------------------------------------------------
# Tests: Error response format
# ---------------------------------------------------------------------------

def test_error_response_format(test_client):
    """Error responses contain an 'error' key."""
    response = test_client.get("/items/bad-uuid/matches")
    assert response.status_code == 400
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# Tests: Concurrency (offline)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_registrations(fake_vlm, fake_embedder, fake_repo, tmp_path):
    """
    asyncio.gather over multiple register_batch calls should all succeed.

    This verifies the concurrency layer (pipeline.py) works offline
    without a real database or API.
    """
    import asyncio
    from src.concurrency.pipeline import register_batch

    # Write a tiny PNG to disk so the pipeline can open it
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000"
        "00907753de0000000c4944415408d76360000000000004000146a13a"
        "020000000049454e44ae426082"
    )
    img_path = str(tmp_path / "tiny.png")
    Path(img_path).write_bytes(png_bytes)

    svc = AIService(vlm=fake_vlm, embedder=fake_embedder, max_attempts=1, wait_seconds=0.001)

    # Run three registrations concurrently
    tasks = [
        register_batch(
            [(img_path, f"concurrent item {i}", ItemStatus.LOST)],
            repo=fake_repo,
            ai_svc=svc,
        )
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)

    # Each call should return exactly one record
    assert len(results) == 3
    for batch in results:
        assert len(batch) == 1
        assert batch[0].status == ItemStatus.LOST
