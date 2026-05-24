"""
repository.py - Two storage backends:
  1. PostgreSQL (via SQLAlchemy async ORM) for item metadata.
  2. Filesystem for image blobs under settings.image_storage_dir.

ORM model (ItemORM) lives here; SE-layer Pydantic models live in src/models.py.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import ARRAY, DateTime, Float, String, Text, select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config import settings
from src.models import ItemRecord, ItemStatus, ItemSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ItemORM(Base):
    """SQLAlchemy ORM representation of an item row."""

    __tablename__ = "items"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    vlm_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Float(precision=32)), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


"""
# Repository
"""

class Repository:
    """
    Async repository wrapping SQLAlchemy ORM (PostgreSQL) and filesystem storage.

    Inherit or compose this class to swap the storage backend in tests.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @classmethod
    async def create(cls) -> "Repository":
        """Create engine, run DDL (CREATE TABLE IF NOT EXISTS), return instance."""
        engine = create_async_engine(settings.database_url, echo=False, pool_size=5)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        logger.info("db ready url=%s", settings.database_url.split("@")[-1])
        return cls(factory)

    # -- write ----------------------------------------------------------------

    async def save_item(
        self,
        status: ItemStatus,
        description: str,
        source_image_path: str,
        vlm_description: dict | None = None,
        embedding: np.ndarray | None = None,
    ) -> ItemRecord:
        """
        Copy the image to managed storage, then insert an ORM row.
        Returns the full ItemRecord (Pydantic) of the saved item.
        """
        item_id = uuid.uuid4()
        dest = self._store_image(source_image_path, item_id)

        orm_obj = ItemORM(
            id=item_id,
            status=status.value,
            description=description,
            image_path=str(dest),
            vlm_description=json.dumps(vlm_description) if vlm_description else None,
            embedding=embedding.tolist() if embedding is not None else None,
            created_at=datetime.utcnow(),
        )

        async with self._session_factory() as session:
            session.add(orm_obj)
            await session.commit()
            await session.refresh(orm_obj)

        logger.info("save_item id=%s status=%s", item_id, status.value)
        return _to_record(orm_obj)

    async def update_embedding(self, item_id: uuid.UUID, embedding: np.ndarray) -> None:
        """Patch only the embedding column (called after async AI processing)."""
        async with self._session_factory() as session:
            obj = await session.get(ItemORM, item_id)
            if obj is None:
                raise ValueError(f"Item {item_id} not found")
            obj.embedding = embedding.tolist()
            await session.commit()
        logger.debug("update_embedding id=%s dim=%d", item_id, len(embedding))

    # -- read -----------------------------------------------------------------

    async def get_item(self, item_id: uuid.UUID) -> Optional[ItemRecord]:
        async with self._session_factory() as session:
            obj = await session.get(ItemORM, item_id)
        return _to_record(obj) if obj else None

    async def list_items(self, status: Optional[ItemStatus] = None) -> list[ItemSummary]:
        async with self._session_factory() as session:
            stmt = select(ItemORM).order_by(ItemORM.created_at.desc())
            if status:
                stmt = stmt.where(ItemORM.status == status.value)
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [_to_summary(r) for r in rows]

    async def get_items_with_embeddings(
        self, status: ItemStatus
    ) -> list[tuple[ItemRecord, np.ndarray]]:
        """Return (record, embedding) for all items of status that have embeddings."""
        async with self._session_factory() as session:
            stmt = (
                select(ItemORM)
                .where(ItemORM.status == status.value)
                .where(ItemORM.embedding.isnot(None))
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [
            (_to_record(r), np.array(r.embedding, dtype=np.float32))
            for r in rows
        ]

    # -- filesystem (second storage backend) ----------------------------------

    def _store_image(self, source: str, item_id: uuid.UUID) -> Path:
        """
        Copy the source image into the managed image directory with a safe UUID-based name.
        
        Security: Uses UUID for filename to prevent path traversal attacks.
        User-provided filename is NOT used, only the extension is preserved.
        """
        src = Path(source)
        
        # Generate safe filename - NEVER trust user input for paths
        ext = src.suffix.lower()
        safe_filename = f"{uuid.uuid4().hex}{ext}"
        
        dest_dir = settings.image_storage_dir / str(item_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_filename
        
        # Defense in depth: verify destination doesn't escape storage dir
        if not str(dest.resolve()).startswith(str(settings.image_storage_dir.resolve())):
            logger.error("Path traversal attempt detected: %s", source)
            raise ValueError("Invalid path: traversal detected")
        
        shutil.copy2(src, dest)
        logger.debug("image stored src=%s dest=%s", src, dest)
        return dest


"""
ORM -> Pydantic converters
"""

def _to_record(obj: ItemORM) -> ItemRecord:
    return ItemRecord(
        id=obj.id,
        status=ItemStatus(obj.status),
        description=obj.description,
        image_path=obj.image_path,
        vlm_description=obj.vlm_description,
        embedding=list(obj.embedding) if obj.embedding else None,
        created_at=obj.created_at,
    )


def _to_summary(obj: ItemORM) -> ItemSummary:
    return ItemSummary(
        id=obj.id,
        status=ItemStatus(obj.status),
        description=obj.description,
        image_path=obj.image_path,
        created_at=obj.created_at,
    )