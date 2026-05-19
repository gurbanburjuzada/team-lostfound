"""
models.py — Pydantic data models for the SE (software-engineering) layer.

These are YOUR models for DB rows, API requests/responses, and CLI output.
They are separate from the ai/schemas.py models that the provided AI package owns.

Rule: no naked dicts cross module boundaries — always use one of these.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class ItemStatus(str, Enum):
    LOST = "lost"
    FOUND = "found"


class MatchConfidence(str, Enum):
    HIGH = "high"      # similarity >= 0.85
    MEDIUM = "medium"  # similarity >= 0.65
    LOW = "low"        # similarity < 0.65


# ── Item models ───────────────────────────────────────────────────────────────

class ItemCreate(BaseModel):
    """
    Input model: what a user provides when registering a lost or found item.
    Used by both the CLI and the HTTP API.
    """
    description: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Short text description from the user",
    )
    status: ItemStatus
    image_path: Path = Field(..., description="Path to the uploaded image file")

    @field_validator("description")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("image_path")
    @classmethod
    def image_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Image file not found: {v}")
        return v


class ItemRecord(BaseModel):
    """
    Full DB record for a registered item.
    Returned by the storage layer and the HTTP API GET endpoints.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    status: ItemStatus
    description: str
    image_path: str                          # stored as string in DB
    vlm_description: Optional[str] = None   # JSON-encoded ItemDescription from ai/
    embedding: Optional[list[float]] = None # stored as bytes in DB, decoded here
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}  # allows building from DB row objects


class ItemSummary(BaseModel):
    """
    Lightweight version of ItemRecord for list endpoints (no embedding bytes).
    """
    id: uuid.UUID
    status: ItemStatus
    description: str
    image_path: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Match models ──────────────────────────────────────────────────────────────

class MatchResult(BaseModel):
    """
    A single match returned by the similarity search.
    The 'query_item_id' is the item the user searched for;
    'matched_item' is a candidate from the opposite pool.
    """
    query_item_id: uuid.UUID
    matched_item: ItemSummary
    similarity_score: float = Field(ge=0.0, le=1.0)
    confidence: MatchConfidence
    vlm_reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation from the VLM description comparison",
    )

    @model_validator(mode="after")
    def set_confidence_from_score(self) -> "MatchResult":
        if self.similarity_score >= 0.85:
            self.confidence = MatchConfidence.HIGH
        elif self.similarity_score >= 0.65:
            self.confidence = MatchConfidence.MEDIUM
        else:
            self.confidence = MatchConfidence.LOW
        return self


class MatchResponse(BaseModel):
    """
    HTTP response model for GET /items/{id}/matches.
    """
    query_item_id: uuid.UUID
    query_description: str
    matches: list[MatchResult]
    total_candidates_searched: int


# ── API request/response wrappers ────────────────────────────────────────────

class RegisterResponse(BaseModel):
    """Returned after successfully registering a lost or found item."""
    item_id: uuid.UUID
    status: ItemStatus
    message: str


class ErrorResponse(BaseModel):
    """Standard error shape returned by the HTTP API."""
    error: str
    detail: Optional[str] = None


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    database: str = "connected"
    version: str = "1.0.0"
