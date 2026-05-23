"""
demo.py — End-to-end demo: register sample images, search for matches, save output.

Usage:
  python scripts/demo.py [--mode sequential|concurrent]

Output is saved to artifacts/demo_output.txt
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Literal

from src.concurrency.pipeline import register_batch
from src.models import ItemStatus
from src.services.ai_service import ai_service
from src.storage.repository import Repository
import src.core.matcher as matcher


async def main(mode: Literal["sequential", "concurrent"] = "concurrent") -> None:
    """
    Run the end-to-end demo.

    Steps:
      1. Create or connect to database
      2. Scan data/lost/ and data/found/ for sample images
      3. Register all items (sequentially or concurrently)
      4. Pick one lost item and find top-3 matches
      5. Print and save results
    """
    print("\n" + "=" * 70)
    print("Smart Lost & Found — End-to-End Demo")
    print("=" * 70 + "\n")

    # Initialize repository
    print("Initializing database...")
    repo = await Repository.create()
    print("✓ Database ready\n")

    # Scan sample images
    print("Scanning sample images...")
    lost_dir = Path("data/lost")
    found_dir = Path("data/found")

    lost_images = sorted(lost_dir.glob("*.png")) + sorted(lost_dir.glob("*.jpg"))
    found_images = sorted(found_dir.glob("*.png")) + sorted(found_dir.glob("*.jpg"))

    print(f"  Lost items: {len(lost_images)}")
    print(f"  Found items: {len(found_images)}")
    print()

    # Build registration list
    items_to_register = [
        (str(p), f"Sample lost item: {p.stem}") 
        for p in lost_images
    ] + [
        (str(p), f"Sample found item: {p.stem}")
        for p in found_images
    ]

    print(f"Registering {len(items_to_register)} items ({mode} mode)...\n")

    # Register items
    t0 = time.time()

    if mode == "sequential":
        # Sequential: one at a time
        records = []
        for i, (image_path, description) in enumerate(items_to_register, start=1):
            try:
                result = await register_batch(
                    items=[(image_path, description, ItemStatus.LOST if "lost" in description else ItemStatus.FOUND)],
                    repo=repo,
                    ai_svc=ai_service,
                )
                if result:
                    records.extend(result)
                    status_val = "LOST" if "lost" in description else "FOUND"
                    print(f"  [{i:2}/{len(items_to_register)}] ✓ {status_val}: {Path(image_path).stem}")
            except Exception as e:
                print(f"  [{i:2}/{len(items_to_register)}] ✗ {Path(image_path).stem}: {e}")
    else:
        # Concurrent: batch all at once
        records = await register_batch(
            items=[
                (image_path, description, ItemStatus.LOST if "lost" in description else ItemStatus.FOUND)
                for image_path, description in items_to_register
            ],
            repo=repo,
            ai_svc=ai_service,
        )
        for i, record in enumerate(records, start=1):
            status_val = record.status.value.upper()
            print(f"  [{i:2}/{len(items_to_register)}] ✓ {status_val}: {Path(record.image_path).stem}")

    elapsed = time.time() - t0
    print(f"\n✓ Registered {len(records)}/{len(items_to_register)} items in {elapsed:.2f}s")
    print(f"  ({len(records) / elapsed:.1f} items/sec)\n")

    # Pick one lost item for matching
    lost_records = [r for r in records if r.status == ItemStatus.LOST]
    if not lost_records:
        print("No lost items registered. Cannot demonstrate matching.")
        return

    query_item = lost_records[0]
    print(f"Searching for matches to: {query_item.description}")
    print(f"  Item ID: {query_item.id}\n")

    # Find matches
    try:
        result = await matcher.find_matches(query_item.id, k=3, repo=repo)
    except ValueError as e:
        print(f"Error finding matches: {e}")
        return

    # Format output
    output = _format_results(query_item, result, elapsed, len(records), len(items_to_register), mode)

    # Print to console
    print(output)

    # Save to artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_file = artifacts_dir / "demo_output.txt"
    output_file.write_text(output)
    print(f"\n✓ Output saved to: {output_file}")


def _format_results(
    query_item,
    result,
    registration_time: float,
    successful_registrations: int,
    total_items: int,
    mode: str,
) -> str:
    """Format results for display and file output."""
    lines = [
        "",
        "=" * 70,
        "RESULTS",
        "=" * 70,
        "",
        f"Mode: {mode}",
        f"Query item: {query_item.description}",
        f"Query ID: {query_item.id}",
        f"",
        f"Registration Summary:",
        f"  - Mode: {mode}",
        f"  - Total registered: {successful_registrations}/{total_items}",
        f"  - Time: {registration_time:.2f}s",
        f"  - Speed: {successful_registrations / registration_time:.1f} items/sec",
        f"",
        f"Matching Results:",
        f"  - Candidates searched: {result.total_candidates_searched}",
        f"  - Matches found: {len(result.matches)}",
        f"",
        "─" * 70,
    ]

    if result.matches:
        lines.append("Top Matches:")
        lines.append("")
        for i, match in enumerate(result.matches, start=1):
            lines.extend([
                f"{i}. Item ID: {match.matched_item.id}",
                f"   Status: {match.matched_item.status.value.upper()}",
                f"   Similarity: {match.similarity_score:.1%}",
                f"   Confidence: {match.confidence.value.upper()}",
                f"   Description: {match.matched_item.description}",
                f"   Reason: {match.vlm_reason or 'N/A'}",
                f"   Created: {match.matched_item.created_at}",
                "",
            ])
    else:
        lines.append("No matches found.")
        lines.append("")

    lines.extend([
        "=" * 70,
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end demo: register items and search for matches"
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent"],
        default="concurrent",
        help="Registration mode (default: concurrent)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(mode=args.mode))
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
