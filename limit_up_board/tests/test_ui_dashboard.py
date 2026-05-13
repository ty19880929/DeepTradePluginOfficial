"""PR-2 unit tests for the single-LLM dashboard renderer.

Scope (Plan §7.2 — IDs U-1, U-2, U-4, U-10, U-11, U-12):

* StageStack transitions cover the 5 + Step 4.5 graph
* The dashboard renderer applies them when given pipeline-shaped events
* CANCELLED outcome closes the Live region cleanly and shows the banner
* TUSHARE_FALLBACK bumps the config-panel badge counter
* ``width < 80`` collapses into compact (border-less) mode
* ``NO_COLOR=1`` keeps the dashboard but disables Console colour

Snapshot tests are intentionally **not** here — Plan §10 calls them out as
fragile across rich versions, and PR-4 will add a contains-only snapshot
suite.
"""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import MagicMock

import pytest
from deeptrade.core.run_status import RunStatus
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent
from deeptrade.theme import EVA_THEME
from rich.console import Console

from limit_up_board.ui import choose_renderer
from limit_up_board.ui.dashboard import RichDashboardRenderer
from limit_up_board.ui.layout import (
    ConfigSummary,
    DashboardState,
    render_dashboard,
)
from limit_up_board.ui.mapping import parse_stage_id, title_for
from limit_up_board.ui.stage_model import StageStack, StageStatus


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


def _make_renderer() -> RichDashboardRenderer:
    """Build a renderer suitable for offline state assertions.

    We don't enter Live (no on_run_start) — handlers mutate state directly.
    This keeps the tests deterministic and avoids spinner refresh races.
    """
    return RichDashboardRenderer(no_color=True)


def _drive(renderer: RichDashboardRenderer, events: list[StrategyEvent]) -> None:
    for ev in events:
        renderer.on_event(ev)


def _single_llm_success_events() -> list[StrategyEvent]:
    """Reproduce the pipeline emit sequence for a clean single-LLM run
    (no R2 multi-batch, no failures). Order mirrors
    ``LubRunner._iter_pipeline`` in v0.5.7."""
    return [
        _ev(EventType.STEP_STARTED, "Step 0: resolve trade date"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 0: T=20260512 T+1=20260513",
            payload={"trade_date": "20260512", "next_trade_date": "20260513"},
        ),
        _ev(
            EventType.LOG,
            "运行配置: 40亿 < 流通市值 < 150亿、股价 < 20.0元",
            payload={
                "min_float_mv_yi": 40.0,
                "max_float_mv_yi": 150.0,
                "max_close_yuan": 20.0,
            },
        ),
        _ev(EventType.STEP_STARTED, "Step 1: data assembly"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 1: 20 candidates",
            payload={
                "candidates": 20,
                "lgb_model_id": "lgb_20260301_v1_abc",
                "lgb_scored": 20,
            },
        ),
        _ev(
            EventType.STEP_STARTED,
            "Step 2: R1 strong target analysis",
            payload={"n_candidates": 20, "n_batches": 1},
        ),
        _ev(EventType.LLM_BATCH_STARTED, "R1 batch 1/1"),
        _ev(EventType.LIVE_STATUS, "[强势标的分析] 已提交第 1/1 批 ..."),
        _ev(EventType.LLM_BATCH_FINISHED, "R1 batch 1/1 ok"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 2: R1 strong target analysis",
            payload={"selected": 12},
        ),
        _ev(
            EventType.STEP_STARTED,
            "Step 4: R2 continuation prediction",
            payload={"n_candidates": 12, "n_batches": 1},
        ),
        _ev(EventType.LLM_BATCH_FINISHED, "R2 batch 1/1 ok"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 4: R2 continuation prediction",
            payload={"predictions": 12},
        ),
        _ev(
            EventType.RESULT_PERSISTED,
            "Report written: /tmp/r1/summary.md",
            payload={"report_dir": "/tmp/r1", "selected": 12, "predictions": 12},
        ),
    ]


def _multi_batch_with_45_events() -> list[StrategyEvent]:
    """Force Step 4.5 by giving R2 two batches (success_batches > 1)."""
    return [
        _ev(EventType.STEP_STARTED, "Step 0: resolve trade date"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 0: T=20260512 T+1=20260513",
            payload={"trade_date": "20260512", "next_trade_date": "20260513"},
        ),
        _ev(EventType.STEP_STARTED, "Step 1: data assembly"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 1: 40 candidates",
            payload={"candidates": 40},
        ),
        _ev(
            EventType.STEP_STARTED,
            "Step 2: R1 strong target analysis",
            payload={"n_candidates": 40, "n_batches": 2},
        ),
        _ev(EventType.LLM_BATCH_FINISHED, "R1 batch 1/2 ok"),
        _ev(EventType.LLM_BATCH_FINISHED, "R1 batch 2/2 ok"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 2: R1 strong target analysis",
            payload={"selected": 25},
        ),
        _ev(
            EventType.STEP_STARTED,
            "Step 4: R2 continuation prediction",
            payload={"n_candidates": 25, "n_batches": 2},
        ),
        _ev(EventType.LLM_BATCH_FINISHED, "R2 batch 1/2 ok"),
        _ev(EventType.LLM_BATCH_FINISHED, "R2 batch 2/2 ok"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 4: R2 continuation prediction",
            payload={"predictions": 25},
        ),
        _ev(
            EventType.STEP_STARTED,
            "Step 4.5: final_ranking global reconciliation",
            payload={"n_finalists": 25},
        ),
        _ev(
            EventType.LLM_FINAL_RANK,
            "[全局重排] ok",
            payload={"input_tokens": 1234, "output_tokens": 567},
        ),
        _ev(
            EventType.STEP_FINISHED,
            "Step 4.5: final_ranking global reconciliation",
            payload={"success": True, "finalists": 25},
        ),
        _ev(
            EventType.RESULT_PERSISTED,
            "Report written: /tmp/r1/summary.md",
            payload={"report_dir": "/tmp/r1"},
        ),
    ]


# ---------------------------------------------------------------------------
# U-1: single-LLM five-stage all success
# ---------------------------------------------------------------------------


class TestU1SingleLLMAllSuccess:
    def test_stages_in_order(self) -> None:
        r = _make_renderer()
        _drive(r, _single_llm_success_events())
        ids = [s.stage_id for s in r._state.stages.stages]
        assert ids == ["0", "1", "2", "4", "5"]

    def test_all_stages_success(self) -> None:
        r = _make_renderer()
        _drive(r, _single_llm_success_events())
        for s in r._state.stages.stages:
            assert s.status == StageStatus.SUCCESS, (
                f"stage {s.stage_id} expected SUCCESS, got {s.status}"
            )

    def test_step_0_finished_event_populates_config(self) -> None:
        r = _make_renderer()
        _drive(r, _single_llm_success_events())
        assert r._state.config.trade_date == "20260512"
        assert r._state.config.next_trade_date == "20260513"
        assert r._state.config.min_float_mv_yi == 40.0
        assert r._state.config.max_float_mv_yi == 150.0
        assert r._state.config.max_close_yuan == 20.0
        assert r._state.config.lgb_enabled is True


# ---------------------------------------------------------------------------
# U-2: multi-batch R2 → Step 4.5 inserted between 4 and 5
# ---------------------------------------------------------------------------


class TestU2Step45Inserted:
    def test_45_appears_between_4_and_5(self) -> None:
        r = _make_renderer()
        _drive(r, _multi_batch_with_45_events())
        ids = [s.stage_id for s in r._state.stages.stages]
        assert ids == ["0", "1", "2", "4", "4.5", "5"]

    def test_45_finished_with_final_rank_detail(self) -> None:
        r = _make_renderer()
        _drive(r, _multi_batch_with_45_events())
        st45 = r._state.stages.get("4.5")
        assert st45 is not None
        assert st45.status == StageStatus.SUCCESS
        # LLM_FINAL_RANK detail merge happens before STEP_FINISHED for 4.5,
        # so the SUCCESS rendering keeps the token summary in detail.
        assert "全局重排完成" in st45.detail
        assert "in=1234" in st45.detail
        assert "out=567" in st45.detail

    def test_45_has_no_progress_total(self) -> None:
        r = _make_renderer()
        _drive(r, _multi_batch_with_45_events())
        st45 = r._state.stages.get("4.5")
        assert st45 is not None
        # final_ranking is a single LLM call — no batch progress bar.
        assert st45.progress_total is None


# ---------------------------------------------------------------------------
# U-4: CANCELLED outcome → banner + clean close
# ---------------------------------------------------------------------------


class TestU4Cancelled:
    def test_cancelled_outcome_sets_banner(self) -> None:
        r = _make_renderer()
        # Mid-R1 cancel: STEP_STARTED for 2 fires, then cancel.
        _drive(
            r,
            [
                _ev(EventType.STEP_STARTED, "Step 0: resolve trade date"),
                _ev(
                    EventType.STEP_FINISHED,
                    "Step 0: T=20260512 T+1=20260513",
                    payload={"trade_date": "20260512", "next_trade_date": "20260513"},
                ),
                _ev(
                    EventType.STEP_STARTED,
                    "Step 2: R1 strong target analysis",
                    payload={"n_batches": 1},
                ),
            ],
        )
        outcome = MagicMock()
        outcome.status = RunStatus.CANCELLED
        outcome.error = "KeyboardInterrupt"
        r.on_run_finish(outcome)
        assert r._state.banner is not None
        assert "CANCELLED" in r._state.banner
        # Running stage was force-failed.
        st2 = r._state.stages.get("2")
        assert st2 is not None
        assert st2.status == StageStatus.FAILED

    def test_close_is_idempotent_and_safe_without_live(self) -> None:
        # close() should be safe to call without a live Live region (e.g.
        # when on_run_start raised partway).
        r = _make_renderer()
        r.close()
        r.close()  # idempotent

    def test_close_after_run_start_exits_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """on_run_start → Live entered; close → Live exited cleanly."""
        r = _make_renderer()
        # Force the renderer's Console to write to a StringIO so the Live
        # region doesn't try to talk to a real terminal in the test process.
        r._console = Console(
            theme=EVA_THEME,
            no_color=True,
            file=StringIO(),
            force_terminal=True,
            width=120,
        )
        r.on_run_start(run_id="rid-1", params=MagicMock(), debate=False)
        assert r._live is not None
        r.close()
        assert r._live is None


# ---------------------------------------------------------------------------
# U-10: TUSHARE_FALLBACK badge
# ---------------------------------------------------------------------------


class TestU10TushareFallback:
    def test_badge_increments_per_event(self) -> None:
        r = _make_renderer()
        _drive(
            r,
            [
                _ev(
                    EventType.TUSHARE_FALLBACK,
                    "Tushare cache fallback: daily",
                    level=EventLevel.WARN,
                    payload={"api_name": "daily"},
                ),
                _ev(
                    EventType.TUSHARE_FALLBACK,
                    "Tushare cache fallback: moneyflow",
                    level=EventLevel.WARN,
                    payload={"api_name": "moneyflow"},
                ),
            ],
        )
        assert r._state.config.tushare_fallback_count == 2

    def test_render_includes_badge(self) -> None:
        r = _make_renderer()
        r._state.config.tushare_fallback_count = 3
        # Render at a wide terminal to ensure full panel layout.
        renderable = render_dashboard(r._state, width=120)
        buf = StringIO()
        Console(theme=EVA_THEME, file=buf, force_terminal=True, width=120).print(
            renderable
        )
        text = buf.getvalue()
        assert "缓存兜底" in text
        assert "×3" in text


# ---------------------------------------------------------------------------
# U-11: width < 80 → compact mode (no panel borders)
# ---------------------------------------------------------------------------


class TestU11CompactMode:
    def test_narrow_width_no_panel_borders(self) -> None:
        state = DashboardState(
            run_id="rid",
            plugin_version="0.6.0",
            config=ConfigSummary(trade_date="20260512", next_trade_date="20260513"),
        )
        narrow = render_dashboard(state, width=72)
        wide = render_dashboard(state, width=120)
        buf_narrow = StringIO()
        buf_wide = StringIO()
        Console(
            theme=EVA_THEME, file=buf_narrow, force_terminal=True, width=72
        ).print(narrow)
        Console(
            theme=EVA_THEME, file=buf_wide, force_terminal=True, width=120
        ).print(wide)
        # The narrow output should not include the panel border characters
        # ('╭' / '╰' / '─' panels). Rich uses ╭╮╰╯ for round borders.
        assert "╭" not in buf_narrow.getvalue()
        # The wide output uses panels and should include the panel chars.
        assert "╭" in buf_wide.getvalue() or "┌" in buf_wide.getvalue()


# ---------------------------------------------------------------------------
# U-12: NO_COLOR environment → dashboard enabled, Console no_color=True
# ---------------------------------------------------------------------------


class TestU12NoColor:
    def test_no_color_env_keeps_dashboard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        # Pretend stdout is a TTY so we don't fall through to legacy.
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        # Suppress the CI envvar in case the test runner sets it.
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DEEPTRADE_NO_DASHBOARD", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, RichDashboardRenderer)
        assert r._no_color is True

    def test_no_color_unset_means_colour_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DEEPTRADE_NO_DASHBOARD", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, RichDashboardRenderer)
        assert r._no_color is False


# ---------------------------------------------------------------------------
# Bonus: dashboard never raises out of on_event (contract from Plan §3.6.1)
# ---------------------------------------------------------------------------


class TestRendererNeverRaises:
    def test_bogus_payload_is_swallowed(self) -> None:
        r = _make_renderer()
        # Step started with non-dict payload (pydantic would reject, but
        # the handler should still not crash if a future code path emits
        # something unusual).
        ev = StrategyEvent(
            type=EventType.STEP_STARTED,
            level=EventLevel.INFO,
            message="Step 2: foo",
            payload={"n_batches": "not-an-int"},
        )
        r.on_event(ev)
        st2 = r._state.stages.get("2")
        assert st2 is not None
        # Total stays unset because the payload wasn't a positive int.
        assert st2.progress_total is None

    def test_unknown_step_id_falls_back_to_default_title(self) -> None:
        r = _make_renderer()
        r.on_event(
            _ev(EventType.STEP_STARTED, "Step 9.9: future ML stage")
        )
        st = r._state.stages.get("9.9")
        assert st is not None
        assert st.title == "Step 9.9"


# ---------------------------------------------------------------------------
# Pure parse_stage_id sanity (mapping module — cheap inclusion here)
# ---------------------------------------------------------------------------


class TestParseStageId:
    @pytest.mark.parametrize(
        ("msg", "expected"),
        [
            ("Step 0: resolve trade date", "0"),
            ("Step 1: data assembly", "1"),
            ("Step 2: R1 strong target analysis", "2"),
            ("Step 4: R2 continuation prediction", "4"),
            ("Step 4.5: final_ranking global reconciliation", "4.5"),
            ("Step 4.7: R3 debate revision", "4.7"),
            ("[强势标的分析] 已提交第 1/1 批 ...", None),
            ("Report written: /tmp", None),
            ("", None),
        ],
    )
    def test_extracts(self, msg: str, expected: str | None) -> None:
        assert parse_stage_id(msg) == expected

    def test_title_for_known_and_unknown(self) -> None:
        assert title_for("4.5") == "全局重排（多批合并）"
        assert title_for("99") == "Step 99"
