"""Renderer factory matrix + LegacyStreamRenderer contract tests (PR-5.1).

The factory must degrade to :class:`LegacyStreamRenderer` whenever any of
the five environment switches fires (Plan §3.5). The Rich dashboard lands
in PR-5.2; until then the default branch also falls back to legacy via the
``import .dashboard`` failure path.
"""

from __future__ import annotations

import os
import sys

import pytest

from checkmate.ui import (
    EventRenderer,
    LegacyStreamRenderer,
    NullRenderer,
    RenderEvent,
    choose_renderer,
)


# ---------------------------------------------------------------------------
# Factory fallback matrix
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_env(monkeypatch):
    """Strip the renderer-relevant env vars + force isatty=True so the
    default branch is the *only* difference under test."""
    for var in ("CI", "DEEPTRADE_NO_DASHBOARD", "TERM", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)
    # Pretend stdout is a TTY so we hit the dashboard-import branch.
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    return monkeypatch


def test_no_dashboard_flag_returns_legacy(isolated_env) -> None:
    r = choose_renderer(no_dashboard=True)
    assert isinstance(r, LegacyStreamRenderer)


def test_non_tty_returns_legacy(monkeypatch) -> None:
    for var in ("CI", "DEEPTRADE_NO_DASHBOARD", "TERM"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    r = choose_renderer()
    assert isinstance(r, LegacyStreamRenderer)


def test_ci_env_returns_legacy(isolated_env) -> None:
    isolated_env.setenv("CI", "1")
    assert isinstance(choose_renderer(), LegacyStreamRenderer)


def test_ci_truthy_variants_all_route_to_legacy(isolated_env) -> None:
    for val in ("1", "true", "TRUE", "yes", "on"):
        isolated_env.setenv("CI", val)
        assert isinstance(choose_renderer(), LegacyStreamRenderer), val


def test_deeptrade_no_dashboard_env_returns_legacy(isolated_env) -> None:
    isolated_env.setenv("DEEPTRADE_NO_DASHBOARD", "1")
    assert isinstance(choose_renderer(), LegacyStreamRenderer)


def test_term_dumb_returns_legacy(isolated_env) -> None:
    isolated_env.setenv("TERM", "dumb")
    assert isinstance(choose_renderer(), LegacyStreamRenderer)


def test_default_returns_rich_dashboard_when_available(isolated_env) -> None:
    """PR-5.2 ships ``checkmate.ui.dashboard``: with TTY + no opt-outs, the
    factory should hand back the Rich renderer rather than legacy."""
    from checkmate.ui.dashboard import RichDashboardRenderer  # noqa: PLC0415

    r = choose_renderer()
    assert isinstance(r, RichDashboardRenderer)


def test_renderer_satisfies_protocol() -> None:
    """LegacyStreamRenderer + NullRenderer satisfy the EventRenderer Protocol."""
    assert isinstance(LegacyStreamRenderer(), EventRenderer)
    assert isinstance(NullRenderer(), EventRenderer)


# ---------------------------------------------------------------------------
# LegacyStreamRenderer.on_event format contract
# ---------------------------------------------------------------------------


def test_legacy_on_event_format_info() -> None:
    sink: list[str] = []
    r = LegacyStreamRenderer(sink=sink.append)
    r.on_event(RenderEvent(type="STEP_STARTED", message="building universe"))
    assert sink == ["  ✔ [STEP_STARTED] building universe"]


def test_legacy_on_event_format_warn() -> None:
    sink: list[str] = []
    r = LegacyStreamRenderer(sink=sink.append)
    r.on_event(RenderEvent(type="ROW_SKIPPED", message="missing daily",
                           level="warn"))
    assert sink == ["  ⚠ [ROW_SKIPPED] missing daily"]


def test_legacy_on_event_format_error() -> None:
    sink: list[str] = []
    r = LegacyStreamRenderer(sink=sink.append)
    r.on_event(RenderEvent(type="RUN_FAILED", message="boom", level="error"))
    assert sink == ["  ✘ [RUN_FAILED] boom"]


def test_legacy_unknown_level_defaults_to_info_glyph() -> None:
    sink: list[str] = []
    r = LegacyStreamRenderer(sink=sink.append)
    r.on_event(RenderEvent(type="X", message="m", level="weird"))
    assert sink == ["  ✔ [X] m"]


def test_legacy_lifecycle_callbacks_are_silent() -> None:
    """on_run_start / on_run_finish / close must not emit any sink lines."""
    sink: list[str] = []
    r = LegacyStreamRenderer(sink=sink.append)
    r.on_run_start(run_id="x", mode="scan", params=None)
    r.on_run_finish(outcome=None)
    r.close()
    r.close()  # idempotent
    assert sink == []


def test_null_renderer_silently_consumes_events() -> None:
    r = NullRenderer()
    r.on_run_start(run_id="x", mode="scan", params=None)
    r.on_event(RenderEvent(type="X", message="m"))
    r.on_run_finish(outcome=None)
    r.close()  # all no-ops; assert no raise


# ---------------------------------------------------------------------------
# Mode plumbing (forwarded to the dashboard once it ships in PR-5.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["scan", "signals", "backtest"])
def test_choose_renderer_accepts_three_modes(isolated_env, mode) -> None:
    """The factory must accept all three long-running CLI commands' modes
    without raising. Until PR-5.2 every call still returns legacy."""
    r = choose_renderer(no_dashboard=True, mode=mode)
    assert isinstance(r, LegacyStreamRenderer)
