"""PR-3 snapshot tests for the two core dashboard scenarios.

Plan §9.PR-3 — lock down U-1 (analyze multi-batch all SUCCESS) and U-5
(screen with full funnel payload) against accidental regression.

Per the design's risk mitigation (§10 — *snapshots fragile across rich
versions*): we do **not** byte-compare against a reference file. Instead
we render each scenario, strip ANSI control sequences for stability, and
assert a list of "must contain" tokens. The full rendered text is
committed under ``tests/snapshots/`` as a human-readable reference, but
not asserted byte-for-byte.

To refresh the reference text after a deliberate visual change::

    UPDATE_SNAPSHOTS=1 pytest tests/test_ui_snapshots.py

The token-list assertions are what guards against regression; the .txt
files are documentation that survives ``rich`` upgrades.
"""

from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent
from deeptrade.theme import EVA_THEME
from rich.console import Console

from volume_anomaly.ui.dashboard import RichDashboardRenderer
from volume_anomaly.ui.funnel import FunnelSummary
from volume_anomaly.ui.layout import render_dashboard

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


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


def _render_to_text(
    renderer: RichDashboardRenderer, width: int = 120
) -> str:
    buf = StringIO()
    Console(
        theme=EVA_THEME,
        file=buf,
        force_terminal=True,
        width=width,
        no_color=True,
    ).print(render_dashboard(renderer._state, width=width))
    return buf.getvalue()


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _freeze_render_state(renderer: RichDashboardRenderer) -> None:
    """Replace runtime-fluctuating fields with stable values."""
    frozen = []
    for i, (_ts, level, msg) in enumerate(renderer._state.log_lines):
        frozen.append((f"12:00:{i:02d}", level, msg))
    renderer._state.log_lines.clear()
    for entry in frozen:
        renderer._state.log_lines.append(entry)


def _persist_snapshot(name: str, text: str) -> str:
    """Write or read the reference .txt under ``tests/snapshots/``."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS", "").strip() in {"1", "true", "yes"}:
        path.write_text(text, encoding="utf-8")
        return text
    if not path.is_file():
        # First-run convenience: capture as the reference, mark this
        # invocation as "freshly minted" so the user knows to commit it.
        path.write_text(text, encoding="utf-8")
        pytest.skip(
            f"Initialised snapshot {path.name}; re-run to assert against it. "
            "Commit the new file to bake in the reference."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _build_u1_analyze() -> RichDashboardRenderer:
    """U-1: analyze multi-batch (3 batches), all SUCCESS."""
    r = RichDashboardRenderer(no_color=True)
    r._state.mode = "analyze"
    events = [
        _ev(
            EventType.LOG,
            "运行配置: profile=balanced | 涨幅 5.0~8.0% | 换手 3.0~10.0% | 量比 ≥ 2.0 | LGB on",
            payload={
                "profile": "balanced",
                "main_board_only": True,
                "pct_chg_min": 5.0,
                "pct_chg_max": 8.0,
                "turnover_min": 3.0,
                "turnover_max": 10.0,
                "vol_ratio_5d_min": 2.0,
                "lgb_enabled": True,
            },
        ),
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
            payload={"n_candidates": 35, "n_batches": 3},
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
            payload={"predictions": 30, "imminent_launch": 5},
        ),
    ]
    for ev in events:
        r.on_event(ev)
    _freeze_render_state(r)
    r._state.started_at = None
    r._state.run_id = "run-u1-deterministic"
    r._state.plugin_version = "0.9.2"
    return r


def _build_u5_screen() -> RichDashboardRenderer:
    """U-5: screen mode with full 5-step funnel payload."""
    r = RichDashboardRenderer(no_color=True)
    r._state.mode = "screen"
    r._state.funnel = FunnelSummary()
    events = [
        _ev(
            EventType.LOG,
            "运行配置: profile=balanced | 涨幅 5.0~8.0% | 换手 3.0~10.0% | 量比 ≥ 2.0",
            payload={
                "profile": "balanced",
                "main_board_only": True,
                "pct_chg_min": 5.0,
                "pct_chg_max": 8.0,
                "turnover_min": 3.0,
                "turnover_max": 10.0,
                "vol_ratio_5d_min": 2.0,
            },
        ),
        _ev(EventType.STEP_STARTED, "Step 0: 核对交易日期"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 0: T=20260512 T+1=20260513",
            payload={
                "trade_date": "20260512",
                "next_trade_date": "20260513",
            },
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
            payload={
                "n_new": 5,
                "n_updated": 30,
                "watchlist_total": 85,
            },
        ),
    ]
    for ev in events:
        r.on_event(ev)
    _freeze_render_state(r)
    r._state.started_at = None
    r._state.run_id = "run-u5-deterministic"
    r._state.plugin_version = "0.9.2"
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnapshotU1Analyze:
    """analyze multi-batch all SUCCESS."""

    REQUIRED_TOKENS = [
        "DeepTrade · 成交量异动",
        "走势分析",
        "run-u1-",
        "运行配置",
        "20260512",
        "20260513",
        "主板",
        "涨幅 5.0~8.0%",
        "换手 3.0~10.0%",
        "量比 ≥ 2.0",
        "LGB: 已开启",
        "执行进度",
        "阶段 0: 核对交易日期",
        "阶段 1: 组装候选包（数据装配）",
        "阶段 2: 走势分析（主升浪启动预测）",
        "阶段 5: 生成走势分析报告",
        "完成",
        "日志",
    ]

    FORBIDDEN_TOKENS = [
        # screen-specific elements must not leak into analyze mode.
        "筛选漏斗",
        "异动筛选",
        "主板上市",
    ]

    def test_required_tokens_present(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u1_analyze()))
        for tok in self.REQUIRED_TOKENS:
            assert tok in text, (
                f"U-1 snapshot missing token: {tok!r}\n--- text ---\n{text}"
            )

    def test_forbidden_tokens_absent(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u1_analyze()))
        for tok in self.FORBIDDEN_TOKENS:
            assert tok not in text, (
                f"U-1 snapshot unexpectedly contains: {tok!r}"
            )

    def test_stages_ordered(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u1_analyze()))
        i0 = text.find("阶段 0:")
        i1 = text.find("阶段 1:")
        i2 = text.find("阶段 2:")
        i5 = text.find("阶段 5:")
        assert 0 < i0 < i1 < i2 < i5, (
            f"stage order broken (idx {i0} {i1} {i2} {i5})"
        )

    def test_snapshot_file_reflects_state(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u1_analyze()))
        ref = _persist_snapshot("U1_analyze_multibatch_success", text)
        for tok in self.REQUIRED_TOKENS:
            assert tok in ref


class TestSnapshotU5Screen:
    """screen mode with full funnel payload."""

    REQUIRED_TOKENS = [
        "DeepTrade · 成交量异动",
        "异动筛选",
        "run-u5-",
        "运行配置",
        "20260512",
        "执行进度",
        "阶段 0: 核对交易日期",
        "阶段 1: 异动筛选（主板量能漏斗）",
        "阶段 5: 生成筛选报告",
        "筛选漏斗",
        "主板上市",
        "排除 ST/停牌",
        "T 日规则",
        "换手率筛选",
        "量能筛选",
        "3210",
        "3187",
        "412",
        "248",
        "35",
        # Delta annotations on each funnel row.
        "-23",
        "-2775",
        "-164",
        "-213",
    ]

    FORBIDDEN_TOKENS = [
        # analyze-specific elements must not leak into screen mode.
        "走势分析",
        "LGB",  # screen mode has no LGB indicator.
    ]

    def test_required_tokens_present(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5_screen()))
        for tok in self.REQUIRED_TOKENS:
            assert tok in text, (
                f"U-5 snapshot missing token: {tok!r}\n--- text ---\n{text}"
            )

    def test_forbidden_tokens_absent(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5_screen()))
        for tok in self.FORBIDDEN_TOKENS:
            assert tok not in text, (
                f"U-5 snapshot unexpectedly contains: {tok!r}"
            )

    def test_funnel_ordered_top_down(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5_screen()))
        # Verify the 5 funnel rows appear in spec order (top → bottom).
        i_main = text.find("主板上市")
        i_st = text.find("排除 ST/停牌")
        i_t = text.find("T 日规则")
        i_turn = text.find("换手率筛选")
        i_vol = text.find("量能筛选")
        assert 0 < i_main < i_st < i_t < i_turn < i_vol, (
            f"funnel order broken (idx {i_main} {i_st} {i_t} {i_turn} {i_vol})"
        )

    def test_snapshot_file_reflects_state(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5_screen()))
        ref = _persist_snapshot("U5_screen_funnel_success", text)
        for tok in self.REQUIRED_TOKENS:
            assert tok in ref
