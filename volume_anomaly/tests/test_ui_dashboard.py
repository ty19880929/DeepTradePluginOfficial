"""PR-2 unit tests for ``RichDashboardRenderer`` (Plan §7.2 U-1..U-12).

Drives the dashboard through synthetic event streams and asserts both the
underlying ``DashboardState`` (StageStack / ConfigSummary / FunnelSummary)
and key substrings in the rendered frame. The full rich.Live region is
exercised via a captured Console so we never need a real TTY.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

import pytest
from deeptrade.core.run_status import RunStatus
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent
from deeptrade.theme import EVA_THEME
from rich.console import Console

from volume_anomaly.ui.dashboard import RichDashboardRenderer
from volume_anomaly.ui.layout import render_dashboard
from volume_anomaly.ui.stage_model import StageStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(
    type_: EventType,
    message: str,
    *,
    level: EventLevel = EventLevel.INFO,
    payload: dict | None = None,
) -> StrategyEvent:
    return StrategyEvent(
        type=type_, level=level, message=message, payload=payload or {}
    )


def _render_state_text(state, *, width: int = 120) -> str:
    """Render a DashboardState directly (no Live region) for substring tests."""
    buf = StringIO()
    console = Console(
        theme=EVA_THEME,
        file=buf,
        force_terminal=False,
        no_color=True,
        width=width,
    )
    console.print(render_dashboard(state, width=width))
    return buf.getvalue()


def _new_renderer(*, mode: str = "analyze") -> RichDashboardRenderer:
    """Build a dashboard with on_run_start called but no Live attached."""
    r = RichDashboardRenderer(no_color=True)
    # Skip Live.__enter__ — tests don't need terminal output, just state.
    r._live = MagicMock()
    r._state.run_id = "deadbeef-0000-0000-0000-000000000000"
    r._state.mode = mode
    from datetime import datetime
    r._state.started_at = datetime(2026, 5, 14, 18, 32, 0)
    if mode == "screen":
        from volume_anomaly.ui.funnel import FunnelSummary
        r._state.funnel = FunnelSummary()
    return r


# ---------------------------------------------------------------------------
# U-1: analyze multi-batch all success
# ---------------------------------------------------------------------------


class TestAnalyzeMultiBatchSuccess:
    def test_stagestack_contains_0_1_2_5_all_success(self) -> None:
        r = _new_renderer(mode="analyze")
        events = [
            _ev(EventType.LOG, "运行配置: lgb=on", payload={"lgb_enabled": True}),
            _ev(EventType.STEP_STARTED, "Step 0: 核对交易日期"),
            _ev(
                EventType.STEP_FINISHED,
                "Step 0: T=20260512 T+1=20260513",
                payload={
                    "trade_date": "20260512",
                    "next_trade_date": "20260513",
                },
            ),
            _ev(EventType.STEP_STARTED, "Step 1: 组装候选包"),
            _ev(
                EventType.STEP_FINISHED,
                "Step 1: 从候选池组装 35 只",
                payload={"candidates": 35},
            ),
            _ev(
                EventType.STEP_STARTED,
                "Step 2: 走势分析（主升浪启动预测）",
                payload={"n_batches": 3},
            ),
            _ev(
                EventType.LIVE_STATUS,
                "[走势分析] 已提交第 1/3 批 (15 只)，等待 LLM 响应...",
            ),
            _ev(
                EventType.LLM_BATCH_STARTED,
                "走势分析 批 1/3",
                payload={"batch_no": 1, "size": 15},
            ),
            _ev(
                EventType.LLM_BATCH_FINISHED,
                "走势分析 批 1/3 完成",
                payload={"batch_no": 1},
            ),
            _ev(
                EventType.LLM_BATCH_FINISHED,
                "走势分析 批 2/3 完成",
                payload={"batch_no": 2},
            ),
            _ev(
                EventType.LLM_BATCH_FINISHED,
                "走势分析 批 3/3 完成",
                payload={"batch_no": 3},
            ),
            _ev(
                EventType.STEP_FINISHED,
                "Step 2: 走势分析（主升浪启动预测）",
                payload={
                    "success_batches": 3,
                    "failed_batches": 0,
                    "predictions": 30,
                },
            ),
            _ev(
                EventType.RESULT_PERSISTED,
                "报告已生成: /tmp/run/report.md",
                payload={"predictions": 30},
            ),
        ]
        for ev in events:
            r.on_event(ev)

        ids = [s.stage_id for s in r._state.stages.stages]
        assert ids == ["0", "1", "2", "5"]
        statuses = {s.stage_id: s.status for s in r._state.stages.stages}
        assert all(
            v == StageStatus.SUCCESS for v in statuses.values()
        ), statuses
        # Trade dates harvested from Step 0 payload.
        assert r._state.config.trade_date == "20260512"
        assert r._state.config.next_trade_date == "20260513"
        # Progress reached 3/3.
        stage2 = r._state.stages.get("2")
        assert stage2 is not None
        assert stage2.progress_done == 3
        assert stage2.progress_total == 3
        r.close()


# ---------------------------------------------------------------------------
# U-2: VALIDATION_FAILED → PARTIAL + failed_batches list
# ---------------------------------------------------------------------------


class TestAnalyzeValidationFailed:
    def test_stage_2_marked_partial_with_failed_batch(self) -> None:
        r = _new_renderer(mode="analyze")
        for ev in (
            _ev(
                EventType.STEP_STARTED,
                "Step 2: 走势分析（主升浪启动预测）",
                payload={"n_batches": 3},
            ),
            _ev(
                EventType.LLM_BATCH_FINISHED,
                "走势分析 批 1/3 完成",
                payload={"batch_no": 1},
            ),
            _ev(
                EventType.VALIDATION_FAILED,
                "走势分析 批 2 失败: timeout",
                level=EventLevel.ERROR,
                payload={"batch_no": 2},
            ),
            _ev(
                EventType.LLM_BATCH_FINISHED,
                "走势分析 批 3/3 完成",
                payload={"batch_no": 3},
            ),
            _ev(
                EventType.STEP_FINISHED,
                "Step 2: 走势分析（主升浪启动预测）",
                payload={
                    "success_batches": 2,
                    "failed_batches": 1,
                    "predictions": 20,
                },
            ),
        ):
            r.on_event(ev)
        s = r._state.stages.get("2")
        assert s is not None
        assert s.status == StageStatus.PARTIAL
        assert len(s.failed_batches) == 1
        assert "批 2" in s.failed_batches[0]
        r.close()


# ---------------------------------------------------------------------------
# U-3: empty watchlist analyze
# ---------------------------------------------------------------------------


class TestAnalyzeEmptyWatchlist:
    def test_no_stage_2_when_only_step1_then_result_persisted(self) -> None:
        """Empty watchlist → pipeline emits STEP_FINISHED Step 1 with 0
        candidates, never starts Step 2, jumps to RESULT_PERSISTED."""
        r = _new_renderer(mode="analyze")
        for ev in (
            _ev(EventType.STEP_STARTED, "Step 0: 核对交易日期"),
            _ev(
                EventType.STEP_FINISHED,
                "Step 0: T=20260512 T+1=20260513",
                payload={"trade_date": "20260512", "next_trade_date": "20260513"},
            ),
            _ev(EventType.STEP_STARTED, "Step 1: 组装候选包"),
            _ev(
                EventType.STEP_FINISHED,
                "Step 1: 从候选池组装 0 只",
                payload={"candidates": 0},
            ),
            _ev(
                EventType.RESULT_PERSISTED,
                "无候选股，已生成空报告（empty watchlist）",
                payload={"reason": "empty watchlist"},
            ),
        ):
            r.on_event(ev)
        ids = [s.stage_id for s in r._state.stages.stages]
        # Stage 2 is not auto-inserted — it only appears if Step 2 fires.
        assert "2" not in ids
        assert "5" in ids
        assert r._state.stages.get("5").status == StageStatus.SUCCESS
        r.close()


# ---------------------------------------------------------------------------
# U-4: KeyboardInterrupt → CANCELLED banner + latest running marked FAILED
# ---------------------------------------------------------------------------


class TestCancelledOutcome:
    def test_cancelled_banner_and_failed_stage(self) -> None:
        r = _new_renderer(mode="analyze")
        for ev in (
            _ev(
                EventType.STEP_STARTED,
                "Step 2: 走势分析（主升浪启动预测）",
                payload={"n_batches": 3},
            ),
            _ev(
                EventType.LLM_BATCH_FINISHED,
                "走势分析 批 1/3 完成",
                payload={"batch_no": 1},
            ),
        ):
            r.on_event(ev)
        outcome = MagicMock(status=RunStatus.CANCELLED, error="KeyboardInterrupt")
        r.on_run_finish(outcome)
        assert "CANCELLED" in (r._state.banner or "")
        # Stage 2 was running → now FAILED.
        s = r._state.stages.get("2")
        assert s is not None
        assert s.status == StageStatus.FAILED
        r.close()


# ---------------------------------------------------------------------------
# U-5: screen mode funnel from DATA_SYNC_FINISHED payload
# ---------------------------------------------------------------------------


class TestScreenFunnel:
    def test_funnel_populated_from_payload(self) -> None:
        r = _new_renderer(mode="screen")
        for ev in (
            _ev(EventType.STEP_STARTED, "Step 0: 核对交易日期"),
            _ev(
                EventType.STEP_FINISHED,
                "Step 0: T=20260512 T+1=20260513",
                payload={"trade_date": "20260512", "next_trade_date": "20260513"},
            ),
            _ev(EventType.DATA_SYNC_STARTED, "Step 1: 异动筛选"),
            _ev(
                EventType.DATA_SYNC_FINISHED,
                "筛选漏斗: 3210 → 3187 → 412 → 248 → 35",
                payload={
                    "n_main_board": 3210,
                    "n_after_st_susp": 3187,
                    "n_after_t_day_rules": 412,
                    "n_after_turnover": 248,
                    "n_after_vol_rules": 35,
                },
            ),
            _ev(
                EventType.RESULT_PERSISTED,
                "异动筛选完成 — 新增 5 只，更新 30 只，候选池 85",
                payload={"n_new": 5, "n_updated": 30, "watchlist_total": 85},
            ),
        ):
            r.on_event(ev)
        assert r._state.funnel is not None
        assert r._state.funnel.n_main_board == 3210
        assert r._state.funnel.n_after_vol_rules == 35
        # Stage 1 should have been registered via DATA_SYNC_STARTED and
        # closed by DATA_SYNC_FINISHED.
        ids = [s.stage_id for s in r._state.stages.stages]
        assert "1" in ids
        assert r._state.stages.get("1").status == StageStatus.SUCCESS
        # Render the frame so we know the funnel goes into the output.
        text = _render_state_text(r._state, width=120)
        assert "3210" in text
        assert "35" in text
        assert "主板上市" in text
        r.close()


# ---------------------------------------------------------------------------
# U-9: TUSHARE_FALLBACK → counter + log
# ---------------------------------------------------------------------------


class TestTushareFallbackBadge:
    def test_counter_increments_and_log_recorded(self) -> None:
        r = _new_renderer(mode="analyze")
        r.on_event(
            _ev(
                EventType.TUSHARE_FALLBACK,
                "cache fallback: moneyflow",
                level=EventLevel.WARN,
            )
        )
        r.on_event(
            _ev(
                EventType.TUSHARE_FALLBACK,
                "cache fallback: daily",
                level=EventLevel.WARN,
            )
        )
        assert r._state.config.tushare_fallback_count == 2
        text = _render_state_text(r._state, width=120)
        assert "Tushare 缓存兜底" in text
        # Counter visible in config row.
        assert "×2" in text
        r.close()


# ---------------------------------------------------------------------------
# U-10: terminal width < 80 → compact mode (no panel borders)
# ---------------------------------------------------------------------------


class TestCompactWidth:
    def test_no_panel_borders_at_70_cols(self) -> None:
        r = _new_renderer(mode="analyze")
        r.on_event(_ev(EventType.STEP_STARTED, "Step 0: 核对交易日期"))
        text = _render_state_text(r._state, width=70)
        # Rich panel top borders look like `╭───`; compact mode omits them.
        assert "╭" not in text
        assert "╰" not in text
        # Stage glyph still present.
        assert "阶段 0" in text
        r.close()

    def test_funnel_compact_at_70_cols(self) -> None:
        r = _new_renderer(mode="screen")
        r.on_event(
            _ev(
                EventType.DATA_SYNC_FINISHED,
                "funnel",
                payload={
                    "n_main_board": 100,
                    "n_after_st_susp": 90,
                    "n_after_t_day_rules": 50,
                    "n_after_turnover": 30,
                    "n_after_vol_rules": 10,
                },
            )
        )
        text = _render_state_text(r._state, width=70)
        # Compact form uses → separator + 漏斗 prefix.
        assert "漏斗" in text
        assert "→" in text
        r.close()


# ---------------------------------------------------------------------------
# U-11: NO_COLOR=1 → dashboard with no_color=True
# ---------------------------------------------------------------------------


class TestNoColor:
    def test_no_color_flag_propagates(
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
        monkeypatch.delenv("TERM", raising=False)
        monkeypatch.setenv("NO_COLOR", "1")
        from volume_anomaly.ui import choose_renderer
        r = choose_renderer(no_dashboard=False)
        assert isinstance(r, RichDashboardRenderer)
        assert r._no_color is True
        r.close()


# ---------------------------------------------------------------------------
# U-12: cmd_prune / cmd_evaluate forced legacy — settings LOG not emitted
# ---------------------------------------------------------------------------


class TestPruneEvaluateForcedLegacy:
    """End-to-end check via the runner-level helper: legacy renderer +
    prune/evaluate mode must not emit the settings LOG (Plan §3.4.2)."""

    def test_va_settings_log_event_returns_none_for_prune(self) -> None:
        from volume_anomaly.runner import PruneParams, _va_settings_log_event
        ev = _va_settings_log_event(
            MagicMock(), "prune", PruneParams(days=30)
        )
        assert ev is None

    def test_va_settings_log_event_returns_none_for_evaluate(self) -> None:
        from volume_anomaly.runner import (
            EvaluateParams,
            _va_settings_log_event,
        )
        ev = _va_settings_log_event(
            MagicMock(), "evaluate", EvaluateParams()
        )
        assert ev is None

    def test_va_settings_log_event_emits_for_screen_with_payload(
        self,
    ) -> None:
        from volume_anomaly.runner import (
            ScreenParams,
            _va_settings_log_event,
        )
        rt = MagicMock()
        rt.config.get_app_config.return_value = MagicMock(
            app_profile="balanced"
        )
        rt.emit = lambda type_, message, **kw: StrategyEvent(
            type=type_,
            level=kw.get("level", EventLevel.INFO),
            message=message,
            payload=kw.get("payload") or {},
        )
        ev = _va_settings_log_event(rt, "screen", ScreenParams())
        assert ev is not None
        assert ev.type == EventType.LOG
        # Profile populated, lgb_enabled absent for screen mode.
        assert ev.payload.get("profile") == "balanced"
        assert "lgb_enabled" not in ev.payload
        # Rule numbers from ScreenRules defaults populated.
        assert ev.payload.get("pct_chg_min") is not None
        assert ev.payload.get("vol_ratio_5d_min") is not None

    def test_va_settings_log_event_emits_for_analyze_with_lgb(self) -> None:
        from volume_anomaly.runner import (
            AnalyzeParams,
            _va_settings_log_event,
        )
        rt = MagicMock()
        rt.config.get_app_config.return_value = MagicMock(
            app_profile="aggressive"
        )
        rt.emit = lambda type_, message, **kw: StrategyEvent(
            type=type_,
            level=kw.get("level", EventLevel.INFO),
            message=message,
            payload=kw.get("payload") or {},
        )
        ev = _va_settings_log_event(
            rt, "analyze", AnalyzeParams(lgb_enabled=False)
        )
        assert ev is not None
        assert ev.payload.get("lgb_enabled") is False


# ---------------------------------------------------------------------------
# LIVE_STATUS detail rendering
# ---------------------------------------------------------------------------


class TestLiveStatusDetail:
    def test_live_status_updates_running_stage_detail(self) -> None:
        r = _new_renderer(mode="analyze")
        r.on_event(
            _ev(
                EventType.STEP_STARTED,
                "Step 2: 走势分析（主升浪启动预测）",
                payload={"n_batches": 2},
            )
        )
        r.on_event(
            _ev(
                EventType.LIVE_STATUS,
                "[走势分析] 已提交第 1/2 批 (15 只)，等待 LLM 响应...",
            )
        )
        s = r._state.stages.get("2")
        assert s is not None
        assert "已提交第 1/2 批" in s.detail
        text = _render_state_text(r._state, width=120)
        # Running stage's detail row uses the └─ tree prefix.
        assert "└─" in text
        assert "已提交第 1/2 批" in text
        r.close()
