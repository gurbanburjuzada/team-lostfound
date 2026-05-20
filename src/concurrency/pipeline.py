"""
pipeline.py - async batch registration using asyncio.gather + semaphore.

Each item runs describe_item -> embed -> save_item concurrently,
bounded by settings.semaphore_limit to stay within provider rate limits.
"""

from __future__ import annotations

import asyncio
import logging

from src.config import settings
from src.models import ItemRecord, ItemStatus
from src.services.ai_service import AIService

logger = logging.getLogger(__name__)


async def _process_one(
    image_path: str,
    description: str,
    status: ItemStatus,
    repo,
    ai_svc: AIService,
    sem: asyncio.Semaphore,
) -> ItemRecord:
    """Describe, embed, and persist a single item under the shared semaphore."""
    async with sem:
        loop = asyncio.get_running_loop()
        vlm_desc = await loop.run_in_executor(
            None, ai_svc.describe_item, image_path, description
        )
        search_text = vlm_desc.to_search_text()
        embedding = await loop.run_in_executor(None, ai_svc.embed, search_text)

    record = await repo.save_item(
        status=status,
        description=description,
        source_image_path=image_path,
        vlm_description=vlm_desc.to_dict(),
        embedding=embedding,
    )
    logger.info("pipeline item done id=%s status=%s", record.id, status.value)
    return record


async def register_batch(
    items: list[tuple[str, str, ItemStatus]],
    repo,
    ai_svc: AIService,
) -> list[ItemRecord]:
    """
    Register a list of (image_path, description, status) tuples concurrently.

    Failed items are logged and skipped; the returned list contains only
    successfully registered records.
    """
    sem = asyncio.Semaphore(settings.semaphore_limit)
    tasks = [
        _process_one(img, desc, status, repo, ai_svc, sem)
        for img, desc, status in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    records: list[ItemRecord] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("pipeline batch item %d failed: %s", i, result)
        else:
            records.append(result)

    logger.info("register_batch total=%d ok=%d", len(items), len(records))
    return records