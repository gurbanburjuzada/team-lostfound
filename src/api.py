"""
api.py — FastAPI HTTP server for the Lost & Found service.

Endpoints:
  POST   /items/lost              — Register a lost item (multipart form)
  POST   /items/found             — Register a found item
  GET    /items/{id}/matches?k=N  — Get top-k matches for an item
  GET    /items?status=lost|found — List all items (optional filter)
  GET    /health                  — Health check
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query, status, Depends
from fastapi.responses import JSONResponse

import src.core.matcher as matcher
from src.config import settings
from src.concurrency.pipeline import register_batch
from src.models import (
    ErrorResponse,
    HealthResponse,
    ItemStatus,
    ItemSummary,
    MatchResponse,
    RegisterResponse,
)
from src.services.ai_service import ai_service
from src.storage.repository import Repository

logger = logging.getLogger(__name__)

# ── FastAPI app initialization ────────────────────────────────────────────────

app = FastAPI(
    title="Smart Lost & Found API",
    description="Match lost and found items using AI vision and embeddings",
    version="1.0.0",
)

# Global repo (initialized on startup)
_repo: Optional[Repository] = None


@app.on_event("startup")
async def startup_event():
    """Initialize database connection pool on app startup."""
    global _repo
    _repo = await Repository.create()
    logger.info("API startup: database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on app shutdown."""
    logger.info("API shutdown: cleaning up resources")


# ── Dependency: get repository ────────────────────────────────────────────────

def get_repo() -> Repository:
    """Dependency that provides the global repository to endpoints."""
    if _repo is None:
        raise RuntimeError("Repository not initialized. Did the startup event run?")
    return _repo


# ── Validation helpers ────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def validate_image_upload(file: UploadFile) -> tuple[str, bytes]:
    """
    Validate uploaded image: content-type, extension, and size.
    Returns (filename, content) if valid, raises HTTPException otherwise.
    """
    # Check content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Invalid content type: %s", file.content_type)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {file.content_type}. Must be image/jpeg or image/png.",
        )

    # Check extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Invalid file extension: %s", ext)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension: {ext}. Must be .jpg, .jpeg, or .png.",
        )

    # Read and validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        logger.warning("File too large: %.2f MB > %.2f MB", size_mb, settings.max_image_size_mb)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size_mb:.2f} MB exceeds max {settings.max_image_size_mb} MB.",
        )

    logger.debug("Image validated: %s (%.2f MB)", file.filename, size_mb)
    return file.filename or "image", content


async def save_uploaded_file(filename: str, content: bytes) -> str:
    """
    Save uploaded file to a temporary location.
    Returns the path to the saved file.
    """
    import tempfile
    import os

    # Create temp file with a reasonable name
    temp_dir = Path(tempfile.gettempdir()) / "lostfound_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename

    # Write to temp file
    with open(temp_path, "wb") as f:
        f.write(content)

    logger.debug("Uploaded file saved: %s", temp_path)
    return str(temp_path)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check(repo: Repository = Depends(get_repo)) -> HealthResponse:
    """Health check endpoint."""
    try:
        # Try a simple query to verify DB is accessible
        await repo.list_items()
        db_status = "connected"
    except Exception as e:
        logger.error("Health check: database error: %s", e)
        db_status = "disconnected"

    return HealthResponse(
        status="ok",
        database=db_status,
        version="1.0.0",
    )


# ── Register endpoints ────────────────────────────────────────────────────────

async def _register_item(
    status: ItemStatus,
    image: UploadFile,
    description: str,
    repo: Repository,
) -> RegisterResponse:
    """
    Shared logic for registering a lost or found item.

    Steps:
      1. Validate and save the uploaded image
      2. Call register_batch() to process (describe + embed + persist)
      3. Return RegisterResponse with the item ID
    """
    # Validate image
    filename, content = await validate_image_upload(image)
    temp_path = await save_uploaded_file(filename, content)

    # Validate description
    description = description.strip()
    if not description or len(description) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Description must be at least 3 characters.",
        )
    if len(description) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Description must not exceed 1000 characters.",
        )

    # Register via batch pipeline (concurrent describe + embed + save)
    try:
        records = await register_batch(
            items=[(temp_path, description, status)],
            repo=repo,
            ai_svc=ai_service,
        )
    except Exception as e:
        logger.error("register_item failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process image: {str(e)}",
        )
    finally:
        # Clean up temp file
        try:
            Path(temp_path).unlink()
        except Exception as e:
            logger.warning("Failed to clean up temp file %s: %s", temp_path, e)

    if not records:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register item (no records returned).",
        )

    record = records[0]
    logger.info(
        "Item registered: id=%s status=%s description_len=%d",
        record.id, status.value, len(description),
    )

    return RegisterResponse(
        item_id=record.id,
        status=record.status,
        message=f"Item registered successfully as {status.value}",
    )


@app.post("/items/lost", response_model=RegisterResponse, status_code=201)
async def register_lost_item(
    image: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    description: str = Form(..., min_length=3, max_length=1000),
    repo: Repository = Depends(get_repo),
) -> RegisterResponse:
    """
    Register a lost item.

    **Request body (multipart/form-data)**:
      - image: image file (JPEG or PNG, max 5 MB by default)
      - description: text description (3-1000 characters)

    **Response**: RegisterResponse with the item ID
    """
    return await _register_item(ItemStatus.LOST, image, description, repo)


@app.post("/items/found", response_model=RegisterResponse, status_code=201)
async def register_found_item(
    image: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    description: str = Form(..., min_length=3, max_length=1000),
    repo: Repository = Depends(get_repo),
) -> RegisterResponse:
    """
    Register a found item.

    **Request body (multipart/form-data)**:
      - image: image file (JPEG or PNG, max 5 MB by default)
      - description: text description (3-1000 characters)

    **Response**: RegisterResponse with the item ID
    """
    return await _register_item(ItemStatus.FOUND, image, description, repo)


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/items/{item_id}/matches", response_model=MatchResponse)
async def get_matches(
    item_id: str,
    k: int = Query(3, ge=1, le=100, description="Number of top matches to return"),
    repo: Repository = Depends(get_repo),
) -> MatchResponse:
    """
    Get top-k matches for a given item.

    **Path parameters**:
      - item_id: UUID of the item to search for

    **Query parameters**:
      - k: number of results (default 3, max 100)

    **Response**: MatchResponse with list of MatchResult

    **Errors**:
      - 404: Item not found
      - 409: Item has no embedding yet (still processing)
    """
    # Parse UUID
    import uuid
    try:
        item_uuid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID: {item_id}",
        )

    # Query matches
    try:
        result = await matcher.find_matches(item_uuid, k, repo)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            logger.warning("Item not found: %s", item_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Item {item_id} not found",
            )
        elif "no embedding" in error_msg.lower():
            logger.info("Item has no embedding yet: %s", item_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Item {item_id} is still being processed. Try again in a moment.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )

    logger.info(
        "get_matches: id=%s k=%d results=%d",
        item_id, k, len(result.matches),
    )
    return result


@app.get("/items", response_model=list[ItemSummary])
async def list_items(
    status: Optional[str] = Query(None, pattern="^(lost|found)$"),
    repo: Repository = Depends(get_repo),
) -> list[ItemSummary]:
    """
    List all items, optionally filtered by status.

    **Query parameters**:
      - status: optional, must be "lost" or "found"

    **Response**: List of ItemSummary
    """
    # Parse status
    item_status = None
    if status:
        try:
            item_status = ItemStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}. Must be 'lost' or 'found'.",
            )

    # Query items
    items = await repo.list_items(item_status)
    logger.info("list_items: status=%s count=%d", status or "all", len(items))
    return items


# ── Error handling ────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Format HTTPException as ErrorResponse."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail or "An error occurred",
            detail=None,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Format unexpected exceptions as 500 errors."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.log_level == "DEBUG" else None,
        ).model_dump(),
    )


# ── Middleware for logging ────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests and responses."""
    import time

    t0 = time.time()
    response = await call_next(request)
    elapsed = time.time() - t0

    logger.info(
        "%s %s %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
