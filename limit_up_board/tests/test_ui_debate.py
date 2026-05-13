"""PR-3 unit tests for the debate-mode summary grid + fallback matrix.

Scope (Plan §7.2 / §9.PR-3):

* **U-5** — 3 providers, all phases succeed → ``DebateGrid`` rows all
  show ✔ R1=N R2=M / R3 修订 K ; ``StageStack`` only carries Step 0/1.
* **U-6** — 1 provider's phase A worker raises → that row shows ✘ 失败
  and the note column captures the error.
* **U-8 / U-13 / U-14** — :func:`choose_renderer` fallback judges produce
  :class:`LegacyStreamRenderer` for non-TTY, ``DEEPTRADE_NO_DASHBOARD=1``,
  and ``TERM=dumb`` respectively.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from limit_up_board.ui import LegacyStreamRenderer, choose_renderer
from limit_up_board.ui.dashboard import RichDashboardRenderer
from limit_up_board.ui.debate_view import DebateGrid, DebateRow
from limit_up_board.ui.stage_model import StageStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(
    type_: EventType,
    message: str,
    *,
    level: EventLevel = EventLevel.INFO,
    payload: dict[str, Any] | None = None,
) -> StrategyEvent:
    return StrategyEvent(
        type=type_, level=level, message=message, payload=payload or {}
    )


def _make_debate_renderer(providers: list[str]) -> RichDashboardRenderer:
    """Build a dashboard primed for debate mode (no real Live entered)."""
    r = RichDashboardRenderer(no_color=True)
    # Mirror what on_run_start would do, sans Live.
    r._state.debate = True
    r._state.debate_grid = DebateGrid()
    # Seed via the same path the runtime uses (LOG event with payload).
    r.on_event(
        _ev(
            EventType.LOG,
            f"[辩论模式] 启用，参与 LLM = {providers}",
            payload={"providers": providers},
        )
    )
    return r


def _worker_event_burst_success(provider: str, *, r1: int, r2: int) -> list[StrategyEvent]:
    """The shape of events ``runner.result_events`` re-emits for a successful
    phase-A worker (tagged with provider + phase_a)."""
    base_payload = {"llm_provider": provider, "debate_phase": "phase_a"}
    return [
        _ev(
            EventType.STEP_STARTED,
            f"[{provider}] Step 2: R1 strong target analysis",
            payload={**base_payload, "n_batches": 1},
        ),
        _ev(
            EventType.LLM_BATCH_FINISHED,
            f"[{provider}] R1 batch 1/1 ok",
            payload=base_payload,
        ),
        _ev(
            EventType.STEP_FINISHED,
            f"[{provider}] Step 2: R1 strong target analysis",
            payload={**base_payload, "selected": r1, "success_batches": 1, "failed_batches": 0},
        ),
        _ev(
            EventType.STEP_STARTED,
            f"[{provider}] Step 4: R2 continuation prediction",
            payload={**base_payload, "n_batches": 1},
        ),
        _ev(
            EventType.LLM_BATCH_FINISHED,
            f"[{provider}] R2 batch 1/1 ok",
            payload=base_payload,
        ),
        _ev(
            EventType.STEP_FINISHED,
            f"[{provider}] Step 4: R2 continuation prediction",
            payload={**base_payload, "predictions": r2, "success_batches": 1, "failed_batches": 0},
        ),
    ]


def _worker_event_burst_phase_a_failed(provider: str, error: str) -> list[StrategyEvent]:
    """Worker future raised → ``runner.result_events`` yields just one
    LOG ERROR with ``[provider] worker failed: <type>: <msg>`` (no buffered
    pipeline events, since the worker died before yielding)."""
    payload = {"llm_provider": provider, "debate_phase": "phase_a"}
    return [
        _ev(
            EventType.LOG,
            f"[{provider}] worker failed: {error}",
            level=EventLevel.ERROR,
            payload=payload,
        )
    ]


def _worker_event_burst_phase_b_success(
    provider: str, *, revised: int
) -> list[StrategyEvent]:
    payload = {"llm_provider": provider, "debate_phase": "phase_b"}
    return [
        _ev(
            EventType.STEP_STARTED,
            f"[{provider}] Step 4.7: R3 debate revision",
            payload=payload,
        ),
        _ev(
            EventType.STEP_FINISHED,
            f"[{provider}] Step 4.7: R3 debate revision",
            payload={**payload, "success": True, "revised": revised},
        ),
    ]


# ---------------------------------------------------------------------------
# U-5 — 3 providers, all phases succeed
# ---------------------------------------------------------------------------


class TestU5DebateAllSuccess:
    def test_grid_rows_seeded_from_log_event(self) -> None:
        r = _make_debate_renderer(["deepseek", "kimi", "qwen"])
        grid = r._state.debate_grid
        assert grid is not None
        assert [row.provider for row in grid.rows] == [
            "deepseek",
            "kimi",
            "qwen",
        ]
        # All rows start WAITING; banner not yet set.
        for row in grid.rows:
            assert row.phase_a_status == StageStatus.WAITING
            assert row.phase_b_status == StageStatus.WAITING

    def test_phase_a_banner_then_success(self) -> None:
        r = _make_debate_renderer(["deepseek", "kimi", "qwen"])
        # Main-thread phase-A banner LIVE_STATUS.
        r.on_event(
            _ev(
                EventType.LIVE_STATUS,
                "[辩论模式] Phase A — 并行执行 R1+R2 (3 个 LLM)",
            )
        )
        # Worker bursts come back in completion order.
        for ev in _worker_event_burst_success("deepseek", r1=8, r2=8):
            r.on_event(ev)
        for ev in _worker_event_burst_success("kimi", r1=7, r2=7):
            r.on_event(ev)
        for ev in _worker_event_burst_success("qwen", r1=6, r2=6):
            r.on_event(ev)
        grid = r._state.debate_grid
        assert grid is not None
        for row in grid.rows:
            assert row.phase_a_status == StageStatus.SUCCESS
        assert grid.row_for("deepseek").r1_count == 8
        assert grid.row_for("deepseek").r2_count == 8
        assert grid.row_for("kimi").r1_count == 7
        assert grid.row_for("qwen").r2_count == 6

    def test_phase_b_banner_finalises_phase_a_and_runs_phase_b(self) -> None:
        r = _make_debate_renderer(["deepseek", "kimi"])
        r.on_event(_ev(EventType.LIVE_STATUS, "[辩论模式] Phase A — 并行执行 R1+R2 (2 个 LLM)"))
        for ev in _worker_event_burst_success("deepseek", r1=5, r2=5):
            r.on_event(ev)
        for ev in _worker_event_burst_success("kimi", r1=5, r2=5):
            r.on_event(ev)
        r.on_event(
            _ev(EventType.LIVE_STATUS, "[辩论模式] Phase B — 并行执行 R3 修订 (2 个 LLM)")
        )
        for ev in _worker_event_burst_phase_b_success("deepseek", revised=5):
            r.on_event(ev)
        for ev in _worker_event_burst_phase_b_success("kimi", revised=5):
            r.on_event(ev)
        # End-of-run report → "done" transition.
        r.on_event(
            _ev(
                EventType.RESULT_PERSISTED,
                "Report written: /tmp/r1/summary.md",
                payload={"report_dir": "/tmp/r1"},
            )
        )
        grid = r._state.debate_grid
        assert grid is not None
        assert grid.current_phase == "done"
        for row in grid.rows:
            assert row.phase_a_status == StageStatus.SUCCESS
            assert row.phase_b_status == StageStatus.SUCCESS
            assert row.revised_count == 5

    def test_step_0_and_1_still_appear_in_stagestack(self) -> None:
        r = _make_debate_renderer(["deepseek", "kimi"])
        r.on_event(_ev(EventType.STEP_STARTED, "Step 0: resolve trade date"))
        r.on_event(
            _ev(
                EventType.STEP_FINISHED,
                "Step 0: T=20260512 T+1=20260513",
                payload={"trade_date": "20260512", "next_trade_date": "20260513"},
            )
        )
        r.on_event(_ev(EventType.STEP_STARTED, "Step 1: data assembly"))
        r.on_event(
            _ev(
                EventType.STEP_FINISHED,
                "Step 1: 20 candidates",
                payload={"candidates": 20},
            )
        )
        # Step 0 and 1 land in the StageStack — the debate grid is only for
        # R1/R2/R3 fan-out (Plan §3.4.4 routing table).
        ids = [s.stage_id for s in r._state.stages.stages]
        assert ids == ["0", "1"]
        assert all(
            s.status == StageStatus.SUCCESS for s in r._state.stages.stages
        )


# ---------------------------------------------------------------------------
# U-6 — phase A failure for one provider
# ---------------------------------------------------------------------------


class TestU6DebatePhaseAFailure:
    def test_failed_worker_row_marked_failed_and_note_set(self) -> None:
        r = _make_debate_renderer(["deepseek", "kimi", "qwen"])
        r.on_event(
            _ev(EventType.LIVE_STATUS, "[辩论模式] Phase A — 并行执行 R1+R2 (3 个 LLM)")
        )
        for ev in _worker_event_burst_success("deepseek", r1=8, r2=8):
            r.on_event(ev)
        for ev in _worker_event_burst_success("kimi", r1=8, r2=8):
            r.on_event(ev)
        for ev in _worker_event_burst_phase_a_failed(
            "qwen", "TimeoutError: request timed out"
        ):
            r.on_event(ev)
        grid = r._state.debate_grid
        assert grid is not None
        assert grid.row_for("qwen").phase_a_status == StageStatus.FAILED
        assert "Timeout" in grid.row_for("qwen").note
        # Phase B should naturally skip the failed provider — verify the row
        # still has WAITING (will render as em-dash because phase A failed).
        assert grid.row_for("qwen").phase_b_status == StageStatus.WAITING
        # The other two still mark SUCCESS.
        assert grid.row_for("deepseek").phase_a_status == StageStatus.SUCCESS
        assert grid.row_for("kimi").phase_a_status == StageStatus.SUCCESS

    def test_render_grid_table_uses_em_dash_for_failed_phase_b(self) -> None:
        """When phase A failed, phase B cell shows em-dash regardless of
        downstream events."""
        from io import StringIO

        from rich.console import Console
        from deeptrade.theme import EVA_THEME

        from limit_up_board.ui.debate_view import render_grid_table

        grid = DebateGrid()
        grid.seed(["deepseek", "qwen"])
        grid.row_for("deepseek").phase_a_status = StageStatus.SUCCESS
        grid.row_for("deepseek").r1_count = 8
        grid.row_for("deepseek").r2_count = 8
        grid.row_for("deepseek").phase_b_status = StageStatus.SUCCESS
        grid.row_for("deepseek").revised_count = 7
        grid.row_for("qwen").phase_a_status = StageStatus.FAILED
        grid.row_for("qwen").note = "TimeoutError"

        buf = StringIO()
        Console(
            theme=EVA_THEME, file=buf, force_terminal=True, width=120
        ).print(render_grid_table(grid))
        text = buf.getvalue()
        assert "deepseek" in text
        assert "qwen" in text
        assert "TimeoutError" in text
        # qwen's R3 column should fall through to em-dash.
        # (Hard to assert exact placement, but the symbol should be present
        # on a non-success failed-A row.)
        assert "—" in text


# ---------------------------------------------------------------------------
# U-8 / U-13 / U-14 — choose_renderer fallback judges
# ---------------------------------------------------------------------------


class TestChooseRendererFallbacks:
    """Verifies Plan §3.5.2 — every fallback judge yields LegacyStreamRenderer.

    The :func:`choose_renderer` factory must short-circuit to legacy in
    each of these environments. ``monkeypatch`` is used because the test
    process itself runs under pytest's stdout-capture (which is non-TTY
    anyway) — we set the conditions explicitly so the test reads cleanly.
    """

    def _enable_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DEEPTRADE_NO_DASHBOARD", raising=False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")

    def test_no_dashboard_flag_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_tty(monkeypatch)
        r = choose_renderer(no_dashboard=True)
        assert isinstance(r, LegacyStreamRenderer)

    def test_u8_non_tty_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DEEPTRADE_NO_DASHBOARD", raising=False)
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_ci_env_var_truthy_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_tty(monkeypatch)
        for value in ("1", "true", "yes", "on", "TRUE"):
            monkeypatch.setenv("CI", value)
            assert isinstance(choose_renderer(no_dashboard=False), LegacyStreamRenderer), (
                f"CI={value!r} should disable dashboard"
            )

    def test_u13_deeptrade_no_dashboard_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_tty(monkeypatch)
        monkeypatch.setenv("DEEPTRADE_NO_DASHBOARD", "1")
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_u14_term_dumb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._enable_tty(monkeypatch)
        monkeypatch.setenv("TERM", "dumb")
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)

    def test_full_tty_yields_dashboard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_tty(monkeypatch)
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, RichDashboardRenderer)

    def test_falsy_envvar_values_dont_trigger_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An env var present but with falsy value must NOT force legacy —
        otherwise habitual ``CI=false`` in scripts would surprise users."""
        self._enable_tty(monkeypatch)
        for value in ("", "0", "false", "no", "off"):
            monkeypatch.setenv("CI", value)
            assert isinstance(
                choose_renderer(no_dashboard=False), RichDashboardRenderer
            ), f"CI={value!r} must not disable dashboard"

    def test_isatty_raising_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Some stdout implementations (e.g. wrapped streams) raise from
        isatty(). The factory must treat that as non-TTY."""

        def boom() -> bool:
            raise OSError("isatty exploded")

        monkeypatch.setattr("sys.stdout.isatty", boom)
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, LegacyStreamRenderer)
