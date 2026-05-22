"""Renderer fallback + dashboard smoke — T4.5, T4.6, T4.9."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from accumulation_probe_washout.ui import choose_renderer
from accumulation_probe_washout.ui.dashboard import RichDashboardRenderer
from accumulation_probe_washout.ui.legacy import LegacyStreamRenderer
from accumulation_probe_washout.ui.layout import render_result_summary


class TestChooseRenderer:
    def test_no_dashboard_flag_forces_legacy(self):
        r = choose_renderer(no_dashboard=True, mode="screen")
        assert isinstance(r, LegacyStreamRenderer)

    def test_ci_env_forces_legacy(self):
        with patch.dict(os.environ, {"CI": "1"}, clear=False):
            r = choose_renderer(no_dashboard=False, mode="screen")
            assert isinstance(r, LegacyStreamRenderer)

    def test_deeptrade_no_dashboard_env_forces_legacy(self):
        with patch.dict(os.environ, {"DEEPTRADE_NO_DASHBOARD": "1"}, clear=False):
            r = choose_renderer(no_dashboard=False, mode="run")
            assert isinstance(r, LegacyStreamRenderer)

    def test_term_dumb_forces_legacy(self):
        with patch.dict(os.environ, {"TERM": "dumb"}, clear=False):
            r = choose_renderer(no_dashboard=False, mode="screen")
            assert isinstance(r, LegacyStreamRenderer)

    def test_non_tty_forces_legacy(self):
        # In pytest, sys.stdout.isatty() is False by default.
        r = choose_renderer(no_dashboard=False, mode="screen")
        assert isinstance(r, LegacyStreamRenderer)


class TestRichDashboardSmoke:
    """RichDashboardRenderer should at least render its layout without raising."""

    def test_renders_initial_layout(self):
        from rich.console import Console

        r = RichDashboardRenderer(mode="screen", console=Console(record=True))
        # _render must produce something Group-like (rich Renderable)
        out = r._render()
        # rich's Group has __rich_console__
        assert hasattr(out, "__rich_console__") or hasattr(out, "__rich__") or hasattr(out, "renderables")

    def test_funnel_appears_after_data_sync_payload(self):
        from rich.console import Console
        from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

        r = RichDashboardRenderer(mode="screen", console=Console(record=True))
        # Without starting Live, simulate the apply path.
        r._header = "header"
        ev = StrategyEvent(
            type=EventType.DATA_SYNC_FINISHED,
            level=EventLevel.INFO,
            message="ok",
            payload={
                "n_main_board": 1000,
                "n_after_st_susp": 950,
                "n_after_liquidity": 800,
                "n_after_accumulation": 200,
                "n_after_probe": 100,
                "n_after_washout": 50,
                "n_after_launch_ready": 10,
            },
        )
        r.on_event(ev)
        assert r.funnel_payload.get("n_after_launch_ready") == 10

    def test_result_summary_appears_after_step5_payload(self):
        from rich.console import Console
        from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

        r = RichDashboardRenderer(mode="run", console=Console(record=True))
        r._header = "header"
        ev = StrategyEvent(
            type=EventType.STEP_FINISHED,
            level=EventLevel.INFO,
            message="Step 5: 写入结果完成，写入 1 条",
            payload={
                "step": 5,
                "result_summary": [
                    {
                        "rank": 1,
                        "ts_code": "600000.SH",
                        "name": "测试股",
                        "current_price": 12.34,
                        "launch_score": 88.8,
                        "prediction": "launch_ready",
                        "confidence": "high",
                        "llm_opinion": "结构完整，等待放量突破。",
                    }
                ],
            },
        )
        r.on_event(ev)

        assert r.result_summary_rows[0]["current_price"] == 12.34
        assert r.result_summary_rows[0]["llm_opinion"] == "结构完整，等待放量突破。"
        out = r._render()
        assert hasattr(out, "__rich_console__") or hasattr(out, "__rich__") or hasattr(out, "renderables")

    def test_result_summary_uses_table_with_llm_opinion_column(self):
        from rich.console import Console

        console = Console(record=True, width=120)
        console.print(render_result_summary([
            {
                "rank": 1,
                "ts_code": "600000.SH",
                "name": "测试股",
                "current_price": 12.34,
                "launch_score": 88.8,
                "prediction": "launch_ready",
                "confidence": "high",
                "llm_opinion": "结构完整，等待放量突破。",
            }
        ]))

        text = console.export_text()
        assert "LLM意见" in text
        assert "结构完整，等待放量突破。" in text
        assert "当前价格" in text

    def test_result_summary_wraps_full_llm_opinion_without_ellipsis(self):
        from rich.console import Console

        opinion = (
            "结构完整，吸筹、试盘、洗盘链条清晰，当前价格贴近试盘高点，"
            "若下一交易日放量突破且不跌破洗盘低点，可继续观察启动确认。"
        )
        console = Console(record=True, width=72)
        console.print(render_result_summary([
            {
                "rank": 1,
                "ts_code": "600000.SH",
                "name": "测试股",
                "current_price": 12.34,
                "launch_score": 88.8,
                "prediction": "launch_ready",
                "confidence": "high",
                "llm_opinion": opinion,
            }
        ]))

        text = console.export_text()
        assert "..." not in text
        assert "…" not in text
        compact = "".join(ch for ch in text if ch not in " \n\r\t│┌┐└┘─")
        assert opinion in compact
