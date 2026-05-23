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
from datetime import datetime
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


@cli.command()
@click.option("--since", type=int, default=24, help="Hours of history to include (default: 24)")
def cost_report(since: int) -> None:
    """
    Display AI provider cost report.

    Shows total spending, breakdown by provider, and top-5 most expensive calls.
    """
    from src.services.cost_meter import get_cost_meter
    from tabulate import tabulate

    meter = get_cost_meter()
    report = meter.get_report(since_hours=since)

    # Header
    click.secho(f"\n💰 AI Cost Report (last {since} hours)", fg="cyan", bold=True)
    click.secho(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fg="dim")
    click.echo()

    # Total cost
    total_usd = report["total_cost_usd"]
    total_calls = report["total_calls"]
    click.secho(f"Total Cost:  ${total_usd:.4f}", fg="green", bold=True)
    click.secho(f"Total Calls: {total_calls}", fg="green")
    click.echo()

    # By provider
    if report["by_provider"]:
        click.secho("By Provider:", fg="cyan", bold=True)
        provider_table = [
            [
                p["provider"],
                f"${p['total']:.4f}",
                p["calls"],
                f"${p['total'] / max(p['calls'], 1):.6f}",  # avg per call
            ]
            for p in report["by_provider"]
        ]
        click.echo(
            tabulate(
                provider_table,
                headers=["Provider", "Total ($)", "Calls", "Avg/Call ($)"],
                tablefmt="simple",
            )
        )
        click.echo()

    # Top 5 expensive
    if report["top_5_expensive"]:
        click.secho("Top 5 Most Expensive Calls:", fg="cyan", bold=True)
        expensive_table = [
            [
                r["provider"],
                r["model"],
                r["prompt_tokens"] + r["completion_tokens"],
                f"${r['dollars']:.6f}",
                r["timestamp"][:19],  # truncate to date time
            ]
            for r in report["top_5_expensive"]
        ]
        click.echo(
            tabulate(
                expensive_table,
                headers=["Provider", "Model", "Total Tokens", "Cost ($)", "Timestamp"],
                tablefmt="simple",
            )
        )
        click.echo()

    if total_calls == 0:
        click.secho("⚠️  No cost records found.", fg="yellow")


# ── Bonus 6: Streaming responses ──────────────────────────────────────────────

@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("prompt")
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic"], case_sensitive=False),
    default=None,
    help="Provider to use (default: from LLM_PROVIDER env var)",
)
@click.option(
    "--model",
    default=None,
    help="Model to use (default: from LLM_MODEL env var)",
)
def vlm_stream(
    image_path: str, prompt: str, provider: Optional[str], model: Optional[str]
) -> None:
    """
    Stream VLM response token-by-token (Bonus 6: Streaming +2 pts).
    
    Tokens are printed to stdout as they arrive from the provider,
    enabling real-time feedback instead of waiting for the full response.
    
    Example:
    
        python -m src.cli vlm-stream data/lost/umbrella_black.png "describe this item"
    
    Note: Streaming works best for prose descriptions. Structured JSON output
    may not stream cleanly due to schema enforcement.
    """
    import os
    from src.ai.streaming import stream_response
    
    # Use env vars as defaults
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL")
    
    click.secho(f"🔄 Streaming from {provider}", fg="cyan")
    if model:
        click.secho(f"   Model: {model}", fg="cyan")
    click.echo()
    
    try:
        token_count = 0
        for token in stream_response(image_path, prompt, provider, model):
            click.echo(token, nl=False)
            token_count += 1
        
        click.echo()  # Final newline
        click.echo()
        click.secho(f"✓ Streamed {token_count} tokens", fg="green")
        
    except ImportError as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
        click.secho(
            f"   Install required package: pip install {provider}",
            fg="yellow",
            err=True,
        )
        raise click.Abort()
    except FileNotFoundError:
        click.secho(f"❌ Image not found: {image_path}", fg="red", err=True)
        raise click.Abort()
    except ValueError as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
        raise click.Abort()
    except Exception as e:
        click.secho(f"❌ Unexpected error: {e}", fg="red", err=True)
        logger.exception("Streaming failed")
        raise click.Abort()


# ── Main entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
