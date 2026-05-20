"""RichDashboardRenderer + StageStack tests (PR-5.2).

Strategy
--------
We do NOT start a real ``Live`` region in these tests — that requires a
terminal-capable console and would interleave with pytest's stdout capture.
Instead we construct the renderer with a ``Console(record=True)`` so the
final renderable can be inspected via ``console.capture()``. Key strings are
asserted as substrings, which keeps the tests robust to Rich version drift
without losing coverage of the layout decisions (which panels appear in
which mode, which payload fields surface, etc.).

For the orchestrator-side contract (events emitted by scan / signals /
backtest reach the renderer), see ``test_scan_smoke.py`` /
``test_signals_smoke.py`` / ``test_backtest_checkpoint.py`` — they already
cover the runtime path via the legacy stream renderer; the dashboard
shares the same on_event signature.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from checkmate.ui.dashboard import RichDashboardRenderer
from checkmate.ui.protocol import RenderEvent
from checkmate.ui.stage_model import StageStack


# ---------------------------------------------------------------------------
# StageStack
# ---------------------------------------------------------------------------


class TestStageStack:
    def test_scan_steps(self) -> None:
        s = StageStack.for_mode("scan")
        assert [step.no for step in s.steps] == [0, 1, 2]
        assert [step.label for step in s.steps] == ["universe", "features", "regime"]

    def test_signals_steps(self) -> None:
        s = StageStack.for_mode("signals")
        assert [step.no for step in s.steps] == [1, 2, 3]
        assert [step.label for step in s.steps] == ["entries", "exits", "risk filter"]

    def test_backtest_steps(self) -> None:
        s = StageStack.for_mode("backtest")
        assert len(s.steps) == 1
        assert s.steps[0].label == "session loop"

    def test_step_started_marks_running(self) -> None:
        s = StageStack.for_mode("scan")
        s.apply(RenderEvent(type="STEP_STARTED", message="Step 0: building universe"))
        assert s.get(0).state == "running"
        assert s.get(1).state == "pending"

    def test_step_finished_marks_done(self) -> None:
        s = StageStack.for_mode("scan")
        s.apply(RenderEvent(type="STEP_STARTED", message="Step 0: building universe"))
        s.apply(RenderEvent(type="STEP_FINISHED", message="Step 0 done — total=10"))
        assert s.get(0).state == "done"

    def test_run_failed_marks_running_step_as_failed(self) -> None:
        s = StageStack.for_mode("scan")
        s.apply(RenderEvent(type="STEP_STARTED", message="Step 1: computing features"))
        s.apply(RenderEvent(type="RUN_FAILED", message="boom", level="error"))
        assert s.get(1).state == "failed"

    def test_session_finished_cycles_backtest_stack(self) -> None:
        s = StageStack.for_mode("backtest")
        s.apply(RenderEvent(type="SESSION_FINISHED", message="20240329  fills=2"))
        assert s.steps[0].state == "done"
        assert "20240329" in s.steps[0].message

    def test_unknown_event_leaves_stack_unchanged(self) -> None:
        s = StageStack.for_mode("scan")
        s.apply(RenderEvent(type="WHATEVER", message="hi"))
        assert all(step.state == "pending" for step in s.steps)


# ---------------------------------------------------------------------------
# RichDashboardRenderer — structural snapshot via Console.capture()
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_console() -> Console:
    # width=160 keeps panel wrapping consistent across CI hosts.
    return Console(record=True, width=160, force_terminal=True)


def _capture(r: RichDashboardRenderer) -> str:
    with r.console.capture() as cap:
        r.console.print(r._render())
    return cap.get()


def _make_scan_renderer(console: Console) -> RichDashboardRenderer:
    return RichDashboardRenderer(mode="scan", console=console)


def _make_signals_renderer(console: Console) -> RichDashboardRenderer:
    return RichDashboardRenderer(mode="signals", console=console)


def _make_backtest_renderer(console: Console) -> RichDashboardRenderer:
    return RichDashboardRenderer(mode="backtest", console=console)


# --- scan mode ---


def test_scan_renderer_renders_header_and_stages(recording_console) -> None:
    r = _make_scan_renderer(recording_console)
    out = _capture(r)
    assert "checkmate" in out
    assert "scan" in out
    assert "Step 0 universe" in out
    assert "Step 1 features" in out
    assert "Step 2 regime" in out


def test_scan_renderer_shows_funnel_when_payload_present(recording_console) -> None:
    r = _make_scan_renderer(recording_console)
    r.on_event(RenderEvent(
        type="STEP_FINISHED",
        message="Step 0 done — total=100 eligible=80 excluded=20",
        payload={"total": 100, "eligible": 80, "excluded": 20},
    ))
    out = _capture(r)
    assert "筛选漏斗" in out
    assert "100" in out
    assert "80" in out
    assert "20" in out


def test_scan_renderer_skips_regime_card_in_scan_mode(recording_console) -> None:
    r = _make_scan_renderer(recording_console)
    # Even with regime payload, scan mode should not render the Regime card.
    r.on_event(RenderEvent(
        type="STEP_FINISHED",
        message="Step 2 done",
        payload={"regime": "strong", "exposure_cap": 1.0},
    ))
    out = _capture(r)
    assert "Regime" not in out


# --- signals mode ---


def test_signals_renderer_renders_regime_card_after_payload(recording_console) -> None:
    r = _make_signals_renderer(recording_console)
    r.on_event(RenderEvent(
        type="STEP_FINISHED",
        message="Step 2 done — exits=1",
        payload={"regime": "neutral", "exposure_cap": 0.6, "breadth_ma120": 0.55},
    ))
    out = _capture(r)
    assert "Regime" in out
    assert "neutral" in out


def test_signals_renderer_shows_three_signal_stages(recording_console) -> None:
    r = _make_signals_renderer(recording_console)
    out = _capture(r)
    assert "Step 1 entries" in out
    assert "Step 2 exits" in out
    assert "Step 3 risk filter" in out


# --- backtest mode ---


def test_backtest_renderer_shows_portfolio_after_session_event(recording_console) -> None:
    r = _make_backtest_renderer(recording_console)
    r.on_event(RenderEvent(
        type="SESSION_FINISHED",
        message="20240329 fills=2 equity=10,200,000 dd=2.10%",
        payload={
            "trade_date": "20240329",
            "equity": 10_200_000.0,
            "drawdown_pct": 0.021,
            "n_fills": 2,
            "regime": "strong",
        },
    ))
    out = _capture(r)
    assert "Portfolio" in out
    assert "20240329" in out
    assert "10,200,000" in out
    assert "strong" in out


def test_backtest_renderer_session_loop_step_advances(recording_console) -> None:
    r = _make_backtest_renderer(recording_console)
    out = _capture(r)
    # Before any SESSION_FINISHED, the step is pending.
    assert "session loop" in out
    r.on_event(RenderEvent(
        type="SESSION_FINISHED",
        message="20240329 fills=0",
        payload={"trade_date": "20240329", "equity": 1e7, "drawdown_pct": 0.0},
    ))
    out2 = _capture(r)
    assert "session loop" in out2  # step still labelled


# --- common: log panel + lifecycle + resilience ---


def test_log_panel_keeps_recent_events(recording_console) -> None:
    r = _make_scan_renderer(recording_console)
    for i in range(15):
        r.on_event(RenderEvent(type="STEP_FINISHED", message=f"Step {i % 3} tick {i}"))
    out = _capture(r)
    assert "Log" in out
    # Most recent tick should appear; an earlier one likely culled by max_lines=10.
    assert "tick 14" in out


def test_on_event_swallows_exceptions(recording_console, monkeypatch) -> None:
    """Force a renderer-internal failure and verify on_event doesn't raise."""
    r = _make_scan_renderer(recording_console)
    # Replace stack.apply with one that raises.
    def _boom(ev):
        raise RuntimeError("boom")
    monkeypatch.setattr(r.stack, "apply", _boom)
    # Must not raise
    r.on_event(RenderEvent(type="STEP_STARTED", message="Step 0: x"))


def test_close_is_idempotent(recording_console) -> None:
    r = _make_scan_renderer(recording_console)
    r.close()
    r.close()  # second call should be safe
    r.close()  # idempotent


def test_lifecycle_run_start_then_finish(recording_console) -> None:
    """on_run_start + on_event + on_run_finish flow with a real Live region.
    Use a recording console so terminal output is captured / discarded."""
    r = _make_scan_renderer(recording_console)
    r.on_run_start(run_id="abc12345-test", mode="scan", params=None)
    r.on_event(RenderEvent(type="STEP_STARTED", message="Step 0: building universe"))
    r.on_event(RenderEvent(type="STEP_FINISHED", message="Step 0 done — total=5"))
    r.on_run_finish(outcome=None)
    r.close()  # idempotent after finish
    # Render must still work for a separate snapshot pass
    out = _capture(r)
    assert "checkmate" in out
