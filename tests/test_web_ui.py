"""
tests/test_web_ui.py — Offline tests for Bonus 5 (Web UI, +2 pts).

These tests verify the UI module is importable and that all helper
functions behave correctly, without spinning up Streamlit or a database.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so the ui module can be imported without a running database
# or Streamlit server.
# ---------------------------------------------------------------------------

def _make_streamlit_stub():
    """Return a lightweight mock of the streamlit module."""
    st = types.ModuleType("streamlit")
    # Any attribute access → a callable no-op or a context-manager no-op
    class _Noop:
        def __call__(self, *a, **kw):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def __iter__(self):
            return iter([self, self])
    noop = _Noop()
    for attr in [
        "set_page_config", "header", "subheader", "write", "caption",
        "info", "success", "error", "warning", "stop", "rerun",
        "title", "divider", "sidebar", "form", "file_uploader",
        "text_area", "text_input", "slider", "selectbox", "radio",
        "form_submit_button", "button", "expander", "columns",
        "spinner", "image", "json",
    ]:
        setattr(st, attr, noop)
    st.cache_resource = lambda **kw: (lambda f: f)
    return st


# Inject stubs before importing the UI module
sys.modules.setdefault("streamlit", _make_streamlit_stub())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUiHelpers:
    """Unit-tests for pure helper functions inside ui/app.py."""

    def test_confidence_badge_high(self):
        from ui.app import _confidence_badge
        assert _confidence_badge("high") == "🟢"

    def test_confidence_badge_medium(self):
        from ui.app import _confidence_badge
        assert _confidence_badge("medium") == "🟡"

    def test_confidence_badge_low(self):
        from ui.app import _confidence_badge
        assert _confidence_badge("low") == "🔴"

    def test_confidence_badge_unknown(self):
        from ui.app import _confidence_badge
        assert _confidence_badge("whatever") == "⚪"

    def test_save_temp_creates_file(self, tmp_path):
        """_save_temp must write the upload bytes to disk and return a valid path."""
        from ui.app import _save_temp

        fake_upload = MagicMock()
        fake_upload.name = "photo.png"
        fake_upload.getbuffer.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result_path = _save_temp(fake_upload)
        p = Path(result_path)
        assert p.exists(), "temp file must exist after _save_temp"
        assert p.suffix == ".png"
        assert p.stat().st_size > 0
        p.unlink()  # clean up

    def test_save_temp_jpeg_suffix(self, tmp_path):
        from ui.app import _save_temp

        fake_upload = MagicMock()
        fake_upload.name = "shot.JPEG"
        fake_upload.getbuffer.return_value = b"\xff\xd8\xff" + b"\x00" * 50

        result_path = _save_temp(fake_upload)
        p = Path(result_path)
        assert p.suffix == ".JPEG"
        p.unlink()


class TestRunHelper:
    """The _run() helper must bridge sync → async correctly."""

    def test_run_simple_coroutine(self):
        from ui.app import _run

        async def _add(a, b):
            return a + b

        result = _run(_add(2, 3))
        assert result == 5

    def test_run_returns_none(self):
        from ui.app import _run

        async def _noop():
            pass

        assert _run(_noop()) is None


class TestImportContract:
    """The UI module must expose the names the tests and Docker CMD rely on."""

    def test_module_imports(self):
        import ui.app  # noqa: F401 — just check it doesn't raise

    def test_required_names_present(self):
        import ui.app as app
        for name in ["_confidence_badge", "_save_temp", "_run"]:
            assert hasattr(app, name), f"ui.app must export {name!r}"

    def test_allowed_types_constant(self):
        from ui.app import ALLOWED_TYPES
        assert "jpg" in ALLOWED_TYPES
        assert "png" in ALLOWED_TYPES

    def test_max_mb_constant(self):
        from ui.app import MAX_MB
        assert MAX_MB == 5