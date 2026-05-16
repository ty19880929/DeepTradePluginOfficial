"""PR-1 unit tests for the ``ui`` subsystem (protocol + legacy + dispatch).

Scope (see ``docs/DeepTradePlugin/volume_anomaly/CLI_UI_Redesign_Plan.md``
§9.PR-1):

* :class:`LegacyStreamRenderer.on_event` emits the v0.7.x stdout line
  format byte-for-byte (Plan §4.3, §10 — user scripts depend on
  ``[step.started]`` etc.).
* :meth:`VaRunner._dispatch_to_renderer` isolates the runner from renderer
  exceptions: a raise on event N → renderer is swapped to legacy and event
  N is still output (Plan §3.6.1, U-8).
* :func:`choose_renderer(no_dashboard=False)` returns a working
  :class:`LegacyStreamRenderer` (PR-1 has no dashboard branch yet) and
  :class:`NullRenderer` is a valid stand-in.

PR-2 will own the dashboard-side assertions (StageStack, layout, funnel
card, etc.) — kept out of this file on purpose.
"""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import MagicMock

import pytest
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from volume_anomaly.ui import (
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
    def test_no_dashboard_true_returns_legacy(self) -> None:
        r = choose_renderer(no_dashboard=True)
        assert isinstance(r, LegacyStreamRenderer)

    def test_tty_no_fallbacks_returns_dashboard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR-2: a fake TTY with no env-var fallbacks → dashboard branch."""
        from volume_anomaly.ui.dashboard import RichDashboardRenderer

        class _FakeTtyStdout:
            def isatty(self) -> bool:
                return True

            def write(self, _s: str) -> int:
                return 0

            def flush(self) -> None:
                return None

        monkeypatch.setattr("sys.stdout", _FakeTtyStdout())
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DEEPTRADE_NO_DASHBOARD", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        r = choose_renderer(no_dashboard=False)
        try:
            assert isinstance(r, RichDashboardRenderer)
        finally:
            r.close()

    def test_non_tty_returns_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plan §3.5: non-TTY → legacy."""

        class _NonTtyStdout:
            def isatty(self) -> bool:
                return False

            def write(self, _s: str) -> int:
                return 0

            def flush(self) -> None:
                return None

        monkeypatch.setattr("sys.stdout", _NonTtyStdout())
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_ci_env_returns_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeTtyStdout:
            def isatty(self) -> bool:
                return True

            def write(self, _s: str) -> int:
                return 0

            def flush(self) -> None:
                return None

        monkeypatch.setattr("sys.stdout", _FakeTtyStdout())
        monkeypatch.setenv("CI", "true")
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_deeptrade_no_dashboard_env_returns_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeTtyStdout:
            def isatty(self) -> bool:
                return True

            def write(self, _s: str) -> int:
                return 0

            def flush(self) -> None:
                return None

        monkeypatch.setattr("sys.stdout", _FakeTtyStdout())
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("DEEPTRADE_NO_DASHBOARD", "1")
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_term_dumb_returns_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeTtyStdout:
            def isatty(self) -> bool:
                return True

            def write(self, _s: str) -> int:
                return 0

            def flush(self) -> None:
                return None

        monkeypatch.setattr("sys.stdout", _FakeTtyStdout())
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DEEPTRADE_NO_DASHBOARD", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_legacy_satisfies_protocol(self) -> None:
        assert isinstance(LegacyStreamRenderer(), EventRenderer)

    def test_null_satisfies_protocol(self) -> None:
        assert isinstance(NullRenderer(), EventRenderer)


# ---------------------------------------------------------------------------
# LegacyStreamRenderer.on_event — byte format
# ---------------------------------------------------------------------------


class TestLegacyStreamRenderer:
    """v0.7.x compatibility: line format is frozen."""

    def test_info_event_uses_check_glyph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        r.on_event(_ev(EventType.STEP_STARTED, "Step 0: 核对交易日期"))
        assert out.text == "  ✔ [step.started] Step 0: 核对交易日期\n"

    def test_warn_event_uses_warn_glyph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        r.on_event(
            _ev(
                EventType.TUSHARE_FALLBACK,
                "cache fallback: moneyflow",
                level=EventLevel.WARN,
            )
        )
        assert out.text == "  ⚠ [tushare.fallback] cache fallback: moneyflow\n"

    def test_error_event_uses_x_glyph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        r.on_event(
            _ev(
                EventType.VALIDATION_FAILED,
                "走势分析 批 2 失败",
                level=EventLevel.ERROR,
            )
        )
        assert (
            out.text == "  ✘ [validation.failed] 走势分析 批 2 失败\n"
        )

    def test_step2_prefix_pipeline_alignment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR-1 pipeline change: STEP_STARTED for 走势分析 now carries the
        'Step 2:' prefix so stage_id parsing aligns with LUB. Lock the
        rendered byte format so future regressions are caught immediately.
        """
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        r.on_event(
            _ev(
                EventType.STEP_STARTED,
                "Step 2: 走势分析（主升浪启动预测）",
            )
        )
        assert (
            out.text
            == "  ✔ [step.started] Step 2: 走势分析（主升浪启动预测）\n"
        )

    def test_lifecycle_hooks_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """on_run_start / on_run_finish / close write nothing to stdout."""
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        r = LegacyStreamRenderer()
        # Minimal stand-in objects — legacy ignores both.
        r.on_run_start(run_id="rid", mode="analyze", params=MagicMock())
        r.on_run_finish(MagicMock(run_id="rid"))
        r.close()
        assert out.text == ""


# ---------------------------------------------------------------------------
# Runner._dispatch_to_renderer — U-8 "raise → degrade to legacy"
# ---------------------------------------------------------------------------


class _RaisingRenderer:
    """Test fixture: raises on the Nth on_event call. Tracks lifecycle."""

    def __init__(self, raise_on_call: int = 1) -> None:
        self.calls = 0
        self.raise_on_call = raise_on_call
        self.closed = False
        self.events_received: list[StrategyEvent] = []

    def on_run_start(
        self, *, run_id: str, mode: str, params: Any
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
    """Build a VaRunner instance with the runtime mocked enough to test
    _dispatch_to_renderer in isolation. We don't go through ``execute_*()``."""
    from volume_anomaly.runner import VaRunner

    rt = MagicMock()
    runner = VaRunner(rt, renderer=renderer)
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
        """U-8: on_event raises → swap to LegacyStreamRenderer → re-dispatch.

        The crashing event is *re-emitted* by legacy so the user sees it;
        subsequent events keep going through legacy.
        """
        out = _Buffer()
        monkeypatch.setattr("sys.stdout", out)
        raising = _RaisingRenderer(raise_on_call=1)
        runner = _make_runner_with_renderer(raising)
        first = _ev(EventType.STEP_STARTED, "Step 0: 核对交易日期")
        runner._dispatch_to_renderer(first)
        # Renderer was closed on fallback and replaced.
        assert raising.closed is True
        assert isinstance(runner._renderer, LegacyStreamRenderer)
        # The crashing event was re-emitted by legacy so the user sees it.
        assert (
            out.text == "  ✔ [step.started] Step 0: 核对交易日期\n"
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
        r.on_run_start(run_id="rid", mode="screen", params=MagicMock())
        for et in (
            EventType.STEP_STARTED,
            EventType.LOG,
            EventType.STEP_FINISHED,
        ):
            r.on_event(_ev(et, "x"))
        r.on_run_finish(MagicMock())
        r.close()
        assert out.text == ""
