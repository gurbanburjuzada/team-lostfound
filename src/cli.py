"""
cli.py — Click CLI for the Lost & Found service.

Commands:
  register-lost <image> <description>      — Register a lost item
  register-found <image> <description>     — Register a found item
  search-matches <item-id> --k <N>         — Get top-k matches for an item
  list [--status lost|found]               — List all items
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import uuid as uuid_lib

import src.core.matcher as matcher
from src.config import settings
from src.concurrency.pipeline import register_batch
from src.models import ItemStatus, MatchResponse
from src.services.ai_service import ai_service
from src.storage.repository import Repository

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def get_repo() -> Repository:
    """Initialize and return the repository."""
    return await Repository.create()


def validate_image_file(image_path: str) -> Path:
    """Validate that the image file exists and has a valid extension."""
    p = Path(image_path)
    if not p.exists():
        raise click.BadParameter(f"Image file not found: {image_path}")

    ext = p.suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise click.BadParameter(
            f"Image must be JPEG or PNG, got: {ext}"
        )

    # Check size
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise click.BadParameter(
            f"Image too large: {size_mb:.2f} MB > {settings.max_image_size_mb} MB"
        )

    return p


# ── Common group ──────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Smart Lost & Found — match lost and found items using AI."""
    pass


# ── Register commands ─────────────────────────────────────────────────────────

@cli.command("register-lost")
@click.argument("image", type=str)
@click.argument("description", type=str)
def register_lost(image: str, description: str) -> None:
    """
    Register a lost item.

    IMAGE: path to image file (JPEG or PNG)
    DESCRIPTION: text description (3-1000 characters)
    """
    asyncio.run(_register_lost_async(image, description))


async def _register_lost_async(image: str, description: str) -> None:
    """Async implementation of register-lost."""
    # Validate inputs
    image_path = validate_image_file(image)
    description = description.strip()

    if len(description) < 3:
        raise click.BadParameter("Description must be at least 3 characters")
    if len(description) > 1000:
        raise click.BadParameter("Description must not exceed 1000 characters")

    # Register
    try:
        repo = await get_repo()
        records = await register_batch(
            items=[(str(image_path), description, ItemStatus.LOST)],
            repo=repo,
            ai_svc=ai_service,
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if records:
        record = records[0]
        click.secho(
            f"✓ Lost item registered: {record.id}",
            fg="green",
        )
        click.echo(f"  Description: {record.description}")
        click.echo(f"  Image: {record.image_path}")
    else:
        click.secho("Error: Failed to register item", fg="red", err=True)
        sys.exit(1)


@cli.command("register-found")
@click.argument("image", type=str)
@click.argument("description", type=str)
def register_found(image: str, description: str) -> None:
    """
    Register a found item.

    IMAGE: path to image file (JPEG or PNG)
    DESCRIPTION: text description (3-1000 characters)
    """
    asyncio.run(_register_found_async(image, description))


async def _register_found_async(image: str, description: str) -> None:
    """Async implementation of register-found."""
    # Validate inputs
    image_path = validate_image_file(image)
    description = description.strip()

    if len(description) < 3:
        raise click.BadParameter("Description must be at least 3 characters")
    if len(description) > 1000:
        raise click.BadParameter("Description must not exceed 1000 characters")

    # Register
    try:
        repo = await get_repo()
        records = await register_batch(
            items=[(str(image_path), description, ItemStatus.FOUND)],
            repo=repo,
            ai_svc=ai_service,
        )
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if records:
        record = records[0]
        click.secho(
            f"✓ Found item registered: {record.id}",
            fg="green",
        )
        click.echo(f"  Description: {record.description}")
        click.echo(f"  Image: {record.image_path}")
    else:
        click.secho("Error: Failed to register item", fg="red", err=True)
        sys.exit(1)


# ── Search command ────────────────────────────────────────────────────────────

@cli.command("search-matches")
@click.argument("item_id", type=str)
@click.option(
    "--k",
    type=int,
    default=3,
    help="Number of top matches to return (default: 3)",
    show_default=True,
)
def search_matches(item_id: str, k: int) -> None:
    """
    Search for top-k matches for a given item.

    ITEM_ID: UUID of the item to search for
    """
    asyncio.run(_search_matches_async(item_id, k))


async def _search_matches_async(item_id: str, k: int) -> None:
    """Async implementation of search-matches."""
    # Validate UUID
    try:
        item_uuid = uuid_lib.UUID(item_id)
    except ValueError:
        raise click.BadParameter(f"Invalid UUID: {item_id}")

    # Validate k
    if k < 1 or k > 100:
        raise click.BadParameter("k must be between 1 and 100")

    # Search
    try:
        repo = await get_repo()
        result: MatchResponse = await matcher.find_matches(item_uuid, k, repo)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            click.secho(f"Error: Item {item_id} not found", fg="red", err=True)
        elif "no embedding" in error_msg.lower():
            click.secho(
                f"Error: Item {item_id} is still being processed. Try again in a moment.",
                fg="yellow",
                err=True,
            )
        else:
            click.secho(f"Error: {error_msg}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    # Print results
    click.secho(f"\nMatches for item {item_id}", fg="cyan", bold=True)
    click.echo(f"Query: {result.query_description}")
    click.echo(f"Searched: {result.total_candidates_searched} candidates\n")

    if not result.matches:
        click.secho("No matches found.", fg="yellow")
        return

    # Print table header
    click.echo("┌─────┬──────────────────────────────┬───────────┬────────────┐")
    click.echo("│ Rank│ Item ID                      │ Score     │ Confidence │")
    click.echo("├─────┼──────────────────────────────┼───────────┼────────────┤")

    # Print matches
    for i, match in enumerate(result.matches, start=1):
        item_id_str = str(match.matched_item.id)[:28].ljust(28)
        score_str = f"{match.similarity_score:.1%}".rjust(9)
        conf_str = match.confidence.value.capitalize().rjust(10)

        click.echo(
            f"│{i:5}│ {item_id_str} │{score_str} │ {conf_str} │"
        )

        # Print description and reason
        click.echo(f"  ├─ Desc: {match.matched_item.description[:60]}")
        if match.vlm_reason:
            click.echo(f"  └─ Reason: {match.vlm_reason}")

    click.echo("└─────┴──────────────────────────────┴───────────┴────────────┘")


# ── List command ──────────────────────────────────────────────────────────────

@cli.command("list")
@click.option(
    "--status",
    type=click.Choice(["lost", "found"]),
    default=None,
    help="Filter by status (optional)",
)
def list_items(status: Optional[str]) -> None:
    """List all items, optionally filtered by status."""
    asyncio.run(_list_items_async(status))


async def _list_items_async(status: Optional[str]) -> None:
    """Async implementation of list."""
    # Parse status
    item_status = None
    if status:
        item_status = ItemStatus(status)

    # Query items
    try:
        repo = await get_repo()
        items = await repo.list_items(item_status)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if not items:
        click.secho("No items found.", fg="yellow")
        return

    # Print results
    title = f"All items" if not status else f"{status.upper()} items"
    click.secho(f"\n{title} ({len(items)} total)", fg="cyan", bold=True)
    click.echo()

    # Print table header
    click.echo("┌────────────────────────────────┬─────────┬──────────────────────────────────────┬──────────────────┐")
    click.echo("│ ID (first 28 chars)            │ Status  │ Description (first 40 chars)        │ Created          │")
    click.echo("├────────────────────────────────┼─────────┼──────────────────────────────────────┼──────────────────┤")

    # Print items
    for item in items:
        id_str = str(item.id)[:28].ljust(28)
        status_str = item.status.value.upper().ljust(7)
        desc_str = item.description[:40].ljust(40)
        time_str = item.created_at.strftime("%Y-%m-%d %H:%M").ljust(16)

        click.echo(
            f"│ {id_str} │ {status_str} │ {desc_str} │ {time_str} │"
        )

    click.echo("└────────────────────────────────┴─────────┴──────────────────────────────────────┴──────────────────┘")


# ── Main entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
