"""PR-4 snapshot tests for the three core dashboard scenarios.

Plan §9.PR-4 — lock down U-1 (single-LLM all success), U-2 (multi-batch
R2 with Step 4.5), and U-5 (3-provider debate all success) against
accidental regression.

Per the design's risk mitigation (§10 #7 — *snapshots fragile across rich
versions*): we do **not** byte-compare against a reference file. Instead
we render each scenario, strip the ANSI control sequences for stability,
and assert a list of "must contain" tokens. The full rendered text is
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

from limit_up_board.ui.dashboard import RichDashboardRenderer
from limit_up_board.ui.debate_view import DebateGrid
from limit_up_board.ui.layout import render_dashboard
from limit_up_board.ui.stage_model import StageStatus

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


def _render_to_text(renderer: RichDashboardRenderer, width: int = 120) -> str:
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
    """Replace runtime-fluctuating fields with stable values so the
    snapshot text doesn't change between runs."""
    # Log lines carry datetime.now() timestamps — rewrite to a fixed clock.
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
# Scenario builders (reused — same as test_ui_dashboard / test_ui_debate)
# ---------------------------------------------------------------------------


def _build_u1() -> RichDashboardRenderer:
    """U-1: single LLM, 5 stages, all SUCCESS, no Step 4.5."""
    r = RichDashboardRenderer(no_color=True)
    events = [
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
            payload={"candidates": 20, "lgb_model_id": "lgb_v1", "lgb_scored": 20},
        ),
        _ev(
            EventType.STEP_STARTED,
            "Step 2: R1 strong target analysis",
            payload={"n_candidates": 20, "n_batches": 1},
        ),
        _ev(EventType.LLM_BATCH_FINISHED, "R1 batch 1/1 ok"),
        _ev(
            EventType.STEP_FINISHED,
            "Step 2: R1 strong target analysis",
            payload={"selected": 12, "success_batches": 1, "failed_batches": 0},
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
            payload={"predictions": 12, "success_batches": 1, "failed_batches": 0},
        ),
        _ev(
            EventType.RESULT_PERSISTED,
            "Report written: /tmp/r1/summary.md",
            payload={"report_dir": "/tmp/r1", "selected": 12, "predictions": 12},
        ),
    ]
    for ev in events:
        r.on_event(ev)
    _freeze_render_state(r)
    r._state.started_at = None
    r._state.run_id = "run-u1-deterministic"
    r._state.plugin_version = "0.6.0"
    return r


def _build_u2() -> RichDashboardRenderer:
    """U-2: multi-batch R2 → Step 4.5 inserted, otherwise all SUCCESS."""
    r = RichDashboardRenderer(no_color=True)
    events = [
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
            payload={"selected": 25, "success_batches": 2, "failed_batches": 0},
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
            payload={"predictions": 25, "success_batches": 2, "failed_batches": 0},
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
    for ev in events:
        r.on_event(ev)
    _freeze_render_state(r)
    r._state.started_at = None
    r._state.run_id = "run-u2-deterministic"
    r._state.plugin_version = "0.6.0"
    return r


def _build_u5() -> RichDashboardRenderer:
    """U-5: 3 providers, all phases succeed."""
    r = RichDashboardRenderer(no_color=True)
    r._state.debate = True
    r._state.debate_grid = DebateGrid()
    providers = ["deepseek", "kimi", "qwen"]
    r.on_event(
        _ev(
            EventType.LOG,
            f"[辩论模式] 启用，参与 LLM = {providers}",
            payload={"providers": providers},
        )
    )
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
    r.on_event(
        _ev(EventType.LIVE_STATUS, "[辩论模式] Phase A — 并行执行 R1+R2 (3 个 LLM)")
    )
    for p, n in [("deepseek", 8), ("kimi", 7), ("qwen", 6)]:
        payload = {"llm_provider": p, "debate_phase": "phase_a"}
        r.on_event(
            _ev(
                EventType.STEP_FINISHED,
                f"[{p}] Step 2: R1 strong target analysis",
                payload={**payload, "selected": n, "success_batches": 1, "failed_batches": 0},
            )
        )
        r.on_event(
            _ev(
                EventType.STEP_FINISHED,
                f"[{p}] Step 4: R2 continuation prediction",
                payload={**payload, "predictions": n, "success_batches": 1, "failed_batches": 0},
            )
        )
    r.on_event(
        _ev(EventType.LIVE_STATUS, "[辩论模式] Phase B — 并行执行 R3 修订 (3 个 LLM)")
    )
    for p in providers:
        payload = {"llm_provider": p, "debate_phase": "phase_b"}
        r.on_event(
            _ev(
                EventType.STEP_FINISHED,
                f"[{p}] Step 4.7: R3 debate revision",
                payload={**payload, "success": True, "revised": 6},
            )
        )
    r.on_event(
        _ev(
            EventType.RESULT_PERSISTED,
            "Report written: /tmp/r1/summary.md",
            payload={"report_dir": "/tmp/r1"},
        )
    )
    _freeze_render_state(r)
    r._state.started_at = None
    r._state.run_id = "run-u5-deterministic"
    r._state.plugin_version = "0.6.0"
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnapshotU1:
    """Single LLM, 5 stages all SUCCESS."""

    REQUIRED_TOKENS = [
        "DeepTrade · 打板策略",
        "单 LLM 模式",
        "run-u1-deterministic"[:8],
        "运行配置",
        "20260512",
        "20260513",
        "40.0亿 < 流通市值 < 150.0亿",
        "股价 < 20.0元",
        "LGB: 已开启",
        "执行进度",
        "阶段 0: 核对交易日期",
        "阶段 1: 捕获基础标的",
        "阶段 2: R1 强势标的初筛",
        "阶段 4: R2 连板潜力预测",
        "阶段 5: 生成策略报告",
        "完成",
        "日志",
    ]

    FORBIDDEN_TOKENS = [
        "辩论",  # Single-LLM should not show the debate banner.
        "Step 4.5",
        "阶段 4.5",
    ]

    def test_required_tokens_present(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u1()))
        for tok in self.REQUIRED_TOKENS:
            assert tok in text, f"U-1 snapshot missing token: {tok!r}\n--- text ---\n{text}"

    def test_forbidden_tokens_absent(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u1()))
        for tok in self.FORBIDDEN_TOKENS:
            assert tok not in text, f"U-1 snapshot unexpectedly contains: {tok!r}"

    def test_snapshot_file_reflects_state(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u1()))
        # Persist the cleaned text so reviewers can eyeball it; assertion is
        # that the reference contains the same set of required tokens.
        ref = _persist_snapshot("U1_single_llm_success", text)
        for tok in self.REQUIRED_TOKENS:
            assert tok in ref


class TestSnapshotU2:
    """Multi-batch R2 → Step 4.5 inserted between 4 and 5."""

    REQUIRED_TOKENS = [
        "DeepTrade · 打板策略",
        "阶段 0: 核对交易日期",
        "阶段 1: 捕获基础标的",
        "阶段 2: R1 强势标的初筛",
        "阶段 4: R2 连板潜力预测",
        "阶段 4.5: 全局重排（多批合并）",
        "阶段 5: 生成策略报告",
        "全局重排完成",
        "in=1234",
        "out=567",
    ]

    def test_required_tokens_present(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u2()))
        for tok in self.REQUIRED_TOKENS:
            assert tok in text, f"U-2 snapshot missing token: {tok!r}"

    def test_45_ordered_between_4_and_5(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u2()))
        i4 = text.find("阶段 4: R2")
        i45 = text.find("阶段 4.5: 全局重排")
        i5 = text.find("阶段 5: 生成策略报告")
        assert 0 < i4 < i45 < i5, (
            f"stage 4.5 must appear between 4 and 5 (idx {i4} {i45} {i5})"
        )

    def test_snapshot_file_reflects_state(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u2()))
        ref = _persist_snapshot("U2_multibatch_step45", text)
        for tok in self.REQUIRED_TOKENS:
            assert tok in ref


class TestSnapshotU5:
    """3-provider debate, all phases SUCCESS."""

    REQUIRED_TOKENS = [
        "DeepTrade · 打板策略",
        "辩论模式",
        "阶段 0: 核对交易日期",
        "阶段 1: 捕获基础标的",
        "辩论汇总",
        "Provider",
        "Phase A (R1+R2)",
        "R3 修订",
        "deepseek",
        "kimi",
        "qwen",
        "R1=8 R2=8",
        "R1=7 R2=7",
        "R1=6 R2=6",
        "修订 6 只",
    ]

    FORBIDDEN_TOKENS = [
        # The single-LLM-mode header label must not leak in debate mode.
        "单 LLM 模式",
    ]

    def test_required_tokens_present(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5()))
        for tok in self.REQUIRED_TOKENS:
            assert tok in text, (
                f"U-5 snapshot missing token: {tok!r}\n--- text ---\n{text}"
            )

    def test_forbidden_tokens_absent(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5()))
        for tok in self.FORBIDDEN_TOKENS:
            assert tok not in text

    def test_step_45_absent_in_single_batch_debate(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5()))
        # Each provider ran one R1 + one R2 batch; no Step 4.5 should appear
        # in the StageStack (it only exists for main-thread emissions).
        assert "阶段 4.5" not in text

    def test_snapshot_file_reflects_state(self) -> None:
        text = _strip_ansi(_render_to_text(_build_u5()))
        ref = _persist_snapshot("U5_debate_3_providers_success", text)
        for tok in self.REQUIRED_TOKENS:
            assert tok in ref
