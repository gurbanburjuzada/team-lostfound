"""
matcher.py - top-k matching between lost and found item pools.

A LOST item is matched against FOUND items, and vice versa.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

import ai
from src.models import (
    ItemRecord,
    ItemStatus,
    ItemSummary,
    MatchConfidence,
    MatchResponse,
    MatchResult,
)

logger = logging.getLogger(__name__)


def _confidence(score: float) -> MatchConfidence:
    if score >= 0.85:
        return MatchConfidence.HIGH
    if score >= 0.65:
        return MatchConfidence.MEDIUM
    return MatchConfidence.LOW


def _to_summary(record: ItemRecord) -> ItemSummary:
    return ItemSummary(
        id=record.id,
        status=record.status,
        description=record.description,
        image_path=record.image_path,
        created_at=record.created_at,
    )


def _reason(query: ItemRecord, match: ItemRecord, score: float) -> str:
    return (
        f"Similarity {score:.1%}. "
        f"Query: '{query.description}'. "
        f"Candidate: '{match.description}'."
    )


async def find_matches(
    item_id: uuid.UUID,
    k: int,
    repo,
) -> MatchResponse:
    """
    Retrieve the item, fetch the opposite-status pool, and run top-k similarity.

    Raises ValueError if the item is not found or has no embedding yet.
    """
    item = await repo.get_item(item_id)
    if item is None:
        raise ValueError(f"Item {item_id} not found")
    if item.embedding is None:
        raise ValueError(f"Item {item_id} has no embedding — still processing?")

    query_vec = np.array(item.embedding, dtype=np.float32)
    opposite = ItemStatus.FOUND if item.status == ItemStatus.LOST else ItemStatus.LOST
    candidates = await repo.get_items_with_embeddings(opposite)

    if not candidates:
        logger.info("find_matches id=%s no candidates in opposite pool", item_id)
        return MatchResponse(
            query_item_id=item_id,
            query_description=item.description,
            matches=[],
            total_candidates_searched=0,
        )

    records = [r for r, _ in candidates]
    vecs = [v for _, v in candidates]
    raw = ai.top_k(query_vec, vecs, k=k)

    results = [
        MatchResult(
            query_item_id=item_id,
            matched_item=_to_summary(records[m.candidate_id]),
            similarity_score=round(m.score, 4),
            confidence=_confidence(m.score),
            vlm_reason=_reason(item, records[m.candidate_id], m.score),
        )
        for m in raw
    ]

    logger.info(
        "find_matches id=%s status=%s candidates=%d results=%d",
        item_id, item.status.value, len(candidates), len(results),
    )
    return MatchResponse(
        query_item_id=item_id,
        query_description=item.description,
        matches=results,
        total_candidates_searched=len(candidates),
    )