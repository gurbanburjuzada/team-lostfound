"""
ui/app.py — Streamlit Web UI for the Smart Lost & Found service.

Bonus 5 (Web UI, +2 pts): A single-file Streamlit frontend that exercises
the same service layer as the CLI and HTTP API.  It does NOT call providers
directly; every action goes through src.core.matcher, src.concurrency.pipeline,
and src.services.ai_service — exactly the same path as the API.

Run:
    streamlit run ui/app.py

Requires the DATABASE_URL (and AI provider keys if you want real embeddings)
to be set in the environment or .env file.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import uuid
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

st.set_page_config(
    page_title="Smart Lost & Found",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from src.concurrency.pipeline import register_batch
    from src.core import matcher
    from src.models import ItemStatus, ItemSummary
    from src.services.ai_service import ai_service
    from src.storage.repository import Repository

    _imports_ok = True
    _import_error: Optional[str] = None

except Exception as exc:
    _imports_ok = False
    _import_error = str(exc)

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_MB = 5


def _run(coro):
    """Run a coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@st.cache_resource(show_spinner="Connecting to database…")
def get_repo() -> Optional[Repository]:
    """Create (and cache) the Repository.  Returns None on failure."""
    if not _imports_ok:
        return None
    try:
        return _run(Repository.create())
    except Exception as exc:  # noqa: BLE001
        st.error(f"⚠️ Could not connect to the database: {exc}")
        return None


def _save_temp(uploaded_file) -> str:
    """Write Streamlit UploadedFile to a temp path; return the path string."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getbuffer())
        return f.name


def _confidence_badge(confidence: str) -> str:
    colours = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    return colours.get(confidence, "⚪")


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Lost & Found")
    st.caption("Smart matching powered by AI vision + embeddings")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Report Lost Item", "Report Found Item", "Find Matches", "Browse Items"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Items are matched using vision-language descriptions and embedding similarity.")

# ── import guard ─────────────────────────────────────────────────────────────

if not _imports_ok:
    st.error(f"Import error — make sure you run from the project root.\n\n`{_import_error}`")
    st.stop()

repo = get_repo()
if repo is None:
    st.warning("Database is not available. Start PostgreSQL and set DATABASE_URL.")
    st.stop()

# ── shared image preview helper ──────────────────────────────────────────────

def _show_register_form(status: ItemStatus):
    label = "lost" if status == ItemStatus.LOST else "found"
    icon  = "😢" if status == ItemStatus.LOST else "🎉"

    st.header(f"{icon} Report a {label.title()} Item")
    st.write(f"Upload a photo and describe the {label} item so we can match it automatically.")

    with st.form(f"register_{label}_form", clear_on_submit=True):
        uploaded = st.file_uploader(
            "Item photo (JPEG or PNG, max 5 MB)",
            type=ALLOWED_TYPES,
        )
        description = st.text_area(
            "Description",
            placeholder=f"e.g. 'Blue Nike backpack with a red keychain, lost near the library'",
            max_chars=1000,
            height=100,
        )
        submitted = st.form_submit_button(f"Register {label.title()} Item", use_container_width=True)

    if submitted:
        # Validate
        if uploaded is None:
            st.error("Please upload an image.")
            return
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > MAX_MB:
            st.error(f"Image is {size_mb:.1f} MB — max allowed is {MAX_MB} MB.")
            return
        description = description.strip()
        if len(description) < 3:
            st.error("Description must be at least 3 characters.")
            return

        # Show preview
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, caption="Your upload", use_container_width=True)

        temp_path = _save_temp(uploaded)
        with col2:
            with st.spinner("Analysing image and generating embedding… (this may take a few seconds)"):
                try:
                    records = _run(
                        register_batch(
                            items=[(temp_path, description, status)],
                            repo=repo,
                            ai_svc=ai_service,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Registration failed: {exc}")
                    Path(temp_path).unlink(missing_ok=True)
                    return
                finally:
                    Path(temp_path).unlink(missing_ok=True)

        if records:
            record = records[0]
            st.success(f"✅ {label.title()} item registered!")
            st.info(f"**Item ID** (save this to search for matches later):\n\n`{record.id}`")
            with st.expander("Full record"):
                st.json({
                    "id": str(record.id),
                    "status": record.status.value,
                    "description": record.description,
                    "created_at": str(record.created_at),
                })
        else:
            st.error("Registration pipeline returned no records. Check the logs.")


# ── pages ─────────────────────────────────────────────────────────────────────

if page == "Report Lost Item":
    _show_register_form(ItemStatus.LOST)

elif page == "Report Found Item":
    _show_register_form(ItemStatus.FOUND)

elif page == "Find Matches":
    st.header("🔎 Find Matches")
    st.write("Enter the ID of a registered item to see its top matches from the opposite pool.")

    with st.form("match_form"):
        item_id_str = st.text_input(
            "Item ID (UUID)",
            placeholder="e.g. 550e8400-e29b-41d4-a716-446655440000",
        )
        k = st.slider("Number of matches to return", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted:
        item_id_str = item_id_str.strip()
        try:
            item_uuid = uuid.UUID(item_id_str)
        except ValueError:
            st.error("Invalid UUID format.")
            st.stop()

        with st.spinner("Computing similarity scores…"):
            try:
                result = _run(matcher.find_matches(item_uuid, k, repo))
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")
                st.stop()

        st.subheader(f"Results for: _{result.query_description}_")
        st.caption(f"Searched {result.total_candidates_searched} candidate(s)")

        if not result.matches:
            st.info("No matches found yet — the opposite item pool may be empty.")
        else:
            for i, m in enumerate(result.matches, 1):
                badge = _confidence_badge(m.confidence.value)
                with st.expander(
                    f"{badge} Match #{i} — similarity {m.similarity_score:.1%} ({m.confidence.value})",
                    expanded=(i == 1),
                ):
                    cols = st.columns([2, 1])
                    with cols[0]:
                        st.write(f"**Description:** {m.matched_item.description}")
                        st.write(f"**Status:** {m.matched_item.status.value}")
                        st.write(f"**Registered:** {m.matched_item.created_at.strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Item ID:** `{m.matched_item.id}`")
                    with cols[1]:
                        # Try to show the matched item's image if it's accessible
                        img_path = Path(m.matched_item.image_path)
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.caption("_(image not on local disk)_")
                    if m.vlm_reason:
                        st.caption(f"ℹ️ {m.vlm_reason}")

elif page == "Browse Items":
    st.header("📋 Browse All Items")

    col1, col2 = st.columns([3, 1])
    with col1:
        filter_status = st.selectbox(
            "Filter by status",
            options=["All", "Lost", "Found"],
            index=0,
        )
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    status_filter: Optional[ItemStatus] = None
    if filter_status == "Lost":
        status_filter = ItemStatus.LOST
    elif filter_status == "Found":
        status_filter = ItemStatus.FOUND

    with st.spinner("Loading items…"):
        try:
            items: list[ItemSummary] = _run(repo.list_items(status_filter))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load items: {exc}")
            st.stop()

    if not items:
        st.info("No items registered yet.")
    else:
        st.caption(f"{len(items)} item(s) found")
        for item in items:
            icon = "😢" if item.status == ItemStatus.LOST else "🎉"
            with st.expander(f"{icon} [{item.status.value.upper()}] {item.description[:80]}…" if len(item.description) > 80 else f"{icon} [{item.status.value.upper()}] {item.description}"):
                cols = st.columns([2, 1])
                with cols[0]:
                    st.write(f"**ID:** `{item.id}`")
                    st.write(f"**Description:** {item.description}")
                    st.write(f"**Registered:** {item.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
                with cols[1]:
                    img_path = Path(item.image_path)
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.caption("_(image file not accessible)_")