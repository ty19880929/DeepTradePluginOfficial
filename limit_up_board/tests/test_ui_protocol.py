"""PR-1 unit tests for the ``ui`` subsystem (protocol + legacy + dispatch).

Scope (see ``docs/limit-up-board/CLI_UI_Redesign_Plan.md`` §9.PR-1):

* ``LegacyStreamRenderer.on_event`` emits the v0.5.x stdout line format
  byte-for-byte (Plan §4.3, §10 — user scripts depend on
  ``[step.started]`` etc.).
* ``LubRunner._dispatch_to_renderer`` isolates the runner from renderer
  exceptions: a raise on event N → renderer is swapped to legacy and event N
  is still output (Plan §3.6.1, U-9).
* ``choose_renderer(no_dashboard=False)`` returns a working
  ``LegacyStreamRenderer`` and ``NullRenderer`` is a valid stand-in.

PR-2 will own the dashboard-side assertions (StageStack, layout, etc.).
"""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import MagicMock

import pytest
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from limit_up_board.ui import (
    EventRenderer,
    LegacyStreamRenderer,
    NullRenderer,
    choose_renderer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(
    type_: EventType, message: str, *, level: EventLevel = EventLevel.INFO
) -> StrategyEvent:
    return StrategyEvent(type=type_, level=level, message=message, payload={})


class _Buffer:
    """Capture stdout for byte-equality assertions."""

    def __init__(self) -> None:
        self.buf = StringIO()

    def write(self, s: str) -> int:
        return self.buf.write(s)

    def flush(self) -> None:
        return None

    @property
    def text(self) -> str:
        return self.buf.getvalue()


# ---------------------------------------------------------------------------
# choose_renderer / Protocol surface
# ---------------------------------------------------------------------------


class TestChooseRenderer:
    def test_returns_legacy_in_pr1(self) -> None:
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_no_dashboard_true_still_returns_legacy(self) -> None:
        # PR-1 has only legacy; both paths must resolve to it.
        r = choose_renderer(no_dashboard=True)
        assert isinstance(r, LegacyStreamRenderer)

    def test_legacy_satisfies_protocol(self) -> None:
        assert isinstance(LegacyStreamRenderer(), EventRenderer)

    def test_null_satisfies_protocol(self) -> None:
        assert isinstance(NullRenderer(), EventRenderer)


# ---------------------------------------------------------------------------
# LegacyStreamRenderer.on_event — byte format
# ---------------------------------------------------------------------------


class TestLegacyStreamRenderer:
    """v0.5.x compatibility: line format is frozen."""

    def test_info_event_uses_check_glyph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        r.on_event(_ev(EventType.STEP_STARTED, "Step 0: resolve trade date"))
        assert out.text == "  ✔ [step.started] Step 0: resolve trade date\n"

    def test_warn_event_uses_warn_glyph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        r.on_event(
            _ev(
                EventType.TUSHARE_FALLBACK,
                "cache fallback: daily",
                level=EventLevel.WARN,
            )
        )
        assert out.text == "  ⚠ [tushare.fallback] cache fallback: daily\n"

    def test_error_event_uses_x_glyph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        r.on_event(
            _ev(
                EventType.VALIDATION_FAILED,
                "batch 2 failed",
                level=EventLevel.ERROR,
            )
        )
        assert out.text == "  ✘ [validation.failed] batch 2 failed\n"

    def test_lifecycle_hooks_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """on_run_start / on_run_finish / close write nothing to stdout."""
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        # Minimal stand-in objects — legacy ignores both.
        r.on_run_start(run_id="rid", params=MagicMock(), debate=False)
        r.on_run_finish(MagicMock(run_id="rid"))
        r.close()
        assert out.text == ""


# ---------------------------------------------------------------------------
# Runner._dispatch_to_renderer — U-9 "raise → degrade to legacy"
# ---------------------------------------------------------------------------


class _RaisingRenderer:
    """Test fixture: raises on the 2nd on_event call. Tracks lifecycle."""

    def __init__(self, raise_on_call: int = 2) -> None:
        self.calls = 0
        self.raise_on_call = raise_on_call
        self.closed = False
        self.events_received: list[StrategyEvent] = []

    def on_run_start(
        self, *, run_id: str, params: Any, debate: bool
    ) -> None:
        return None

    def on_event(self, ev: StrategyEvent) -> None:
        self.calls += 1
        self.events_received.append(ev)
        if self.calls == self.raise_on_call:
            raise RuntimeError("simulated dashboard crash")

    def on_run_finish(self, outcome: Any) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _make_runner_with_renderer(renderer: Any) -> Any:
    """Build a LubRunner instance with the runtime mocked enough to test
    _dispatch_to_renderer in isolation. We don't go through ``execute()``."""
    from limit_up_board.runner import LubRunner

    rt = MagicMock()
    runner = LubRunner(rt, renderer=renderer)
    return runner


class TestDispatchToRenderer:
    def test_dispatches_to_active_renderer(self) -> None:
        captured: list[StrategyEvent] = []

        class _Spy:
            def on_run_start(self, **_: Any) -> None: ...
            def on_event(self, ev: StrategyEvent) -> None:
                captured.append(ev)
            def on_run_finish(self, _: Any) -> None: ...
            def close(self) -> None: ...

        spy = _Spy()
        runner = _make_runner_with_renderer(spy)
        ev = _ev(EventType.LOG, "hello")
        runner._dispatch_to_renderer(ev)
        assert captured == [ev]

    def test_raise_swaps_to_legacy_and_redispatches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """U-9: on_event raises → swap to LegacyStreamRenderer → re-dispatch.

        The first event N+1 (the one that raised) is *re-emitted* by legacy so
        the user sees it; subsequent events go straight to legacy.
        """
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        raising = _RaisingRenderer(raise_on_call=1)
        runner = _make_runner_with_renderer(raising)
        first = _ev(EventType.STEP_STARTED, "Step 0: resolve trade date")
        runner._dispatch_to_renderer(first)
        # Renderer was closed on fallback and replaced.
        assert raising.closed is True
        from limit_up_board.ui import LegacyStreamRenderer as L

        assert isinstance(runner._renderer, L)
        # The crashing event was re-emitted by legacy so the user sees it.
        assert (
            out.text == "  ✔ [step.started] Step 0: resolve trade date\n"
        )
        # Subsequent events keep going through legacy.
        second = _ev(EventType.LOG, "after the crash")
        runner._dispatch_to_renderer(second)
        assert "after the crash" in out.text

    def test_renderer_close_failure_is_swallowed(self) -> None:
        """Even if .close() raises on fallback, the runner doesn't crash."""

        class _BadRenderer:
            def on_run_start(self, **_: Any) -> None: ...
            def on_event(self, _: Any) -> None:
                raise RuntimeError("on_event boom")
            def on_run_finish(self, _: Any) -> None: ...
            def close(self) -> None:
                raise RuntimeError("close boom too")

        runner = _make_runner_with_renderer(_BadRenderer())
        # Should not raise; legacy renderer takes over.
        runner._dispatch_to_renderer(_ev(EventType.LOG, "boom"))


# ---------------------------------------------------------------------------
# NullRenderer
# ---------------------------------------------------------------------------


class TestNullRenderer:
    def test_consumes_full_lifecycle_silently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = NullRenderer()
        r.on_run_start(run_id="rid", params=MagicMock(), debate=True)
        for et in (EventType.STEP_STARTED, EventType.LOG, EventType.STEP_FINISHED):
            r.on_event(_ev(et, "x"))
        r.on_run_finish(MagicMock())
        r.close()
        assert out.text == ""
