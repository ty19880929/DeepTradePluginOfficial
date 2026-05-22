"""LLM Review payload 构造 + 调用 + 持久化 + markdown 报告渲染。

PR #4 — 把 PR #2 计算出的胜率证据 + 当前策略配置摘要打包给 LLM，要求输出
结构化优化建议，再把整轮交互（payload + response）落到 ``lub_winrate_reviews``，
可选写一份人读 markdown 报告。

输入压缩三段（方案 §15.4）：
    strategy_context     — 稳定、短小、可版本化；自动从 LubConfig + 活跃 LGB
                            模型 + 输出口径常量生成，无需用户维护。
    performance_evidence — overall / by_prediction / by_rank_bucket，加 high-
                            score-failure / low-score-win 异常样本切片。
    review_task          — 期望字段、每条建议必须标注落点、安全约束。

落点（``landing``）受 Literal 约束，避免 LLM 输出 free-form 标签难以聚合。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database
    from deeptrade.core.llm_client import LLMClient
    from deeptrade.plugins_api import StageProfile

    from ..config import LubConfig
    from .resolver import ResolvedRecord
    from .stats import GroupStat, WinrateSummary


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response schema — LLM must return this shape
# ---------------------------------------------------------------------------


Landing = Literal[
    "filter_rule",
    "prompt_weighting",
    "lgb_usage",
    "classification_boundary",
    "risk_control",
    "validation_plan",
]


class LandedSuggestion(BaseModel):
    """One actionable suggestion with explicit落点 label.

    The Literal on ``landing`` forces the LLM to map each suggestion to one of
    six implementation surfaces — avoids vague "improve the strategy" output.
    """

    model_config = ConfigDict(extra="forbid")
    landing: Landing
    text: str = Field(..., min_length=1, max_length=500)


class WinrateLlmReview(BaseModel):
    """Top-level response — fields mirror 方案 §9.2 / §15.5."""

    model_config = ConfigDict(extra="forbid")
    diagnosis: str = Field(..., min_length=1, max_length=2000)
    prompt_adjustments: list[LandedSuggestion] = Field(default_factory=list, max_length=10)
    feature_suggestions: list[LandedSuggestion] = Field(default_factory=list, max_length=10)
    risk_controls: list[LandedSuggestion] = Field(default_factory=list, max_length=10)
    validation_plan: str = Field(..., min_length=1, max_length=1500)
    caveats: list[str] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def _strategy_context(
    lub_cfg: LubConfig, active_lgb_model_id: str | None
) -> dict[str, Any]:
    """Build the stable, code-derived strategy snapshot the LLM gets."""
    return {
        "objective": "基于 T 日涨停池预测 T+1 连板/高位溢价候选",
        "pipeline": [
            "数据同步", "候选过滤", "强势初筛", "连板预测", "可选全局重排",
        ],
        "candidate_filter": {
            "main_board_only": True,
            "float_market_cap_min_yi": lub_cfg.min_float_mv_yi,
            "float_market_cap_max_yi": lub_cfg.max_float_mv_yi,
            "close_price_max_yuan": lub_cfg.max_close_yuan,
            "exclude_st": True,
            "exclude_suspended": True,
        },
        "lgb": {
            "enabled": lub_cfg.lgb_enabled,
            "active_model_id": active_lgb_model_id,
            "min_score_floor": lub_cfg.lgb_min_score_floor,
            "decile_in_prompt": lub_cfg.lgb_decile_in_prompt,
            "label_threshold_pct": lub_cfg.lgb_label_threshold_pct,
            "usage": "作为次日最大溢价概率锚点供 LLM 参考，不直接替代 LLM 判断",
        },
        "output_classes": ["top_candidate", "watchlist", "avoid"],
        "outcome_definition": "T+1 open vs T close (限价口径)；> 胜, = 平, < 负",
        "known_limitations": [
            "仅单 LLM 模式记录预测结果",
            "辩论模式当前不纳入胜率样本",
            "胜负判定按统一口径（avoid 的胜负当前不反向定义）",
        ],
    }


def _slim_record(r: ResolvedRecord) -> dict[str, Any]:
    """One逐股 sample in compact form. Strip out raw_prediction_json + run_id
    to keep token budget under control."""
    rec = r.record
    return {
        "trade_date": rec.trade_date,
        "ts_code": rec.ts_code,
        "name": rec.name,
        "prediction": rec.prediction,
        "rank": rec.rank,
        "continuation_score": rec.continuation_score,
        "confidence": rec.confidence,
        "lgb_score": rec.lgb_score,
        "lgb_decile": rec.lgb_decile,
        "t_close_price": rec.t_close_price,
        "t1_open_price": r.t1_open_price,
        "open_vs_limit_pct": r.open_vs_limit_pct,
        "outcome": r.outcome,
    }


def _group_to_payload(g: GroupStat) -> dict[str, Any]:
    return {
        "key": g.key,
        "total": g.total,
        "resolved": g.resolved,
        "win": g.win,
        "flat": g.flat,
        "loss": g.loss,
        "strict_win_rate": g.strict_win_rate,
        "avg_open_vs_limit_pct": g.avg_open_vs_limit_pct,
    }


def _summary_to_payload(s: WinrateSummary) -> dict[str, Any]:
    return {
        "total": s.total,
        "resolved": s.resolved,
        "unresolved": s.unresolved,
        "win": s.win,
        "flat": s.flat,
        "loss": s.loss,
        "strict_win_rate": s.strict_win_rate,
        "non_loss_rate": s.non_loss_rate,
        "avg_open_vs_limit_pct": s.avg_open_vs_limit_pct,
    }


def _lgb_decile_breakdown(resolved: list[ResolvedRecord]) -> list[dict[str, Any]]:
    """Group resolved samples by lgb_decile (None bucket separate)."""
    from collections import defaultdict

    buckets: dict[Any, list[ResolvedRecord]] = defaultdict(list)
    for r in resolved:
        buckets[r.record.lgb_decile].append(r)

    out: list[dict[str, Any]] = []
    for decile, items in sorted(buckets.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        wins = sum(1 for x in items if x.outcome == "win")
        resolved_n = sum(1 for x in items if x.outcome != "unresolved")
        pcts = [x.open_vs_limit_pct for x in items if x.open_vs_limit_pct is not None]
        out.append(
            {
                "lgb_decile": decile,  # may be None
                "total": len(items),
                "resolved": resolved_n,
                "win": wins,
                "strict_win_rate": wins / resolved_n if resolved_n > 0 else None,
                "avg_open_vs_limit_pct": (sum(pcts) / len(pcts)) if pcts else None,
            }
        )
    return out


# Tunable cap on outlier sample lists — keeps token budget bounded; the
# slicing rule itself (sort by surprise then take K) is documented in 方案 §9.1.
MAX_OUTLIER_SAMPLES = 8


def _high_score_failures(resolved: list[ResolvedRecord]) -> list[dict[str, Any]]:
    """LLM 给了高 continuation_score 但 outcome=loss — most embarrassing failures."""
    cands = [
        r for r in resolved
        if r.outcome == "loss" and (r.record.continuation_score or 0) >= 70.0
    ]
    cands.sort(key=lambda r: -(r.record.continuation_score or 0))
    return [_slim_record(r) for r in cands[:MAX_OUTLIER_SAMPLES]]


def _low_score_wins(resolved: list[ResolvedRecord]) -> list[dict[str, Any]]:
    """LLM 给了低分但实际 win — 反向校准信号。"""
    cands = [
        r for r in resolved
        if r.outcome == "win" and (r.record.continuation_score or 100) <= 60.0
    ]
    cands.sort(key=lambda r: (r.record.continuation_score or 0))
    return [_slim_record(r) for r in cands[:MAX_OUTLIER_SAMPLES]]


def _top_candidate_worst_losses(resolved: list[ResolvedRecord]) -> list[dict[str, Any]]:
    """``prediction=top_candidate`` 里跌幅最深的样本。"""
    cands = [
        r for r in resolved
        if r.record.prediction == "top_candidate" and r.open_vs_limit_pct is not None
    ]
    cands.sort(key=lambda r: r.open_vs_limit_pct or 0)
    return [_slim_record(r) for r in cands[:MAX_OUTLIER_SAMPLES]]


def build_review_payload(
    *,
    window_start: str,
    window_end: str,
    resolved: list[ResolvedRecord],
    summary: WinrateSummary,
    by_prediction: list[GroupStat],
    by_rank: list[GroupStat],
    lub_cfg: LubConfig,
    active_lgb_model_id: str | None,
) -> dict[str, Any]:
    """Build the three-section payload sent to the LLM."""
    return {
        "strategy_context": _strategy_context(lub_cfg, active_lgb_model_id),
        "performance_evidence": {
            "window": {"start": window_start, "end": window_end},
            "overall": _summary_to_payload(summary),
            "by_prediction": [_group_to_payload(g) for g in by_prediction],
            "by_rank_bucket": [_group_to_payload(g) for g in by_rank],
            "lgb_decile_breakdown": _lgb_decile_breakdown(resolved),
            "high_score_failures": _high_score_failures(resolved),
            "low_score_wins": _low_score_wins(resolved),
            "top_candidate_worst_losses": _top_candidate_worst_losses(resolved),
        },
        "review_task": {
            "expected_fields": [
                "diagnosis",
                "prompt_adjustments",
                "feature_suggestions",
                "risk_controls",
                "validation_plan",
                "caveats",
            ],
            "each_suggestion_must_label_landing": [
                "filter_rule",
                "prompt_weighting",
                "lgb_usage",
                "classification_boundary",
                "risk_control",
                "validation_plan",
            ],
            "guardrails": [
                "样本不足（< 20 已解析）时输出 diagnosis 但不给出强 prompt/特征建议",
                "不允许编造未出现的事实",
                "仅用于策略研究，不构成投资建议",
            ],
        },
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """你是 A 股打板策略复盘助手。
你将收到一个 JSON payload，包含 strategy_context（当前策略配置摘要）、
performance_evidence（胜率统计与异常样本）、review_task（输出要求）。

任务：基于 evidence 对照 context，给出可落地的优化建议。
- 每条 prompt_adjustments / feature_suggestions / risk_controls 必须标注 landing
  字段（filter_rule / prompt_weighting / lgb_usage / classification_boundary /
  risk_control / validation_plan 之一）。
- 不要重复已经体现在 strategy_context 里的现状。
- 样本不足时只给出 diagnosis 与 caveats，不强出建议。
- 不要编造未出现在 evidence 里的事实。
- 输出严格遵守 schema，不要解释、不要附加自然语言。
"""


def _user_prompt(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _default_review_profile() -> StageProfile:
    """Conservative profile — moderate output, low temperature, reasoning on."""
    from deeptrade.plugins_api import StageProfile

    return StageProfile(
        thinking=True,
        reasoning_effort="high",
        temperature=0.2,
        max_output_tokens=8192,
    )


@dataclass
class ReviewOutcome:
    """All the artifacts of one llm-review call — useful for tests + persistence."""

    review_id: str
    llm_provider: str
    llm_model: str | None
    payload: dict[str, Any]
    response: WinrateLlmReview
    response_audit: dict[str, Any]  # raw audit from complete_json
    created_at: datetime


def call_llm_for_review(
    llm: LLMClient,
    payload: dict[str, Any],
    *,
    profile: StageProfile | None = None,
) -> tuple[WinrateLlmReview, dict[str, Any]]:
    """Invoke the LLM and decode the structured response. Lets exceptions
    propagate — caller is responsible for surfacing them."""
    prof = profile or _default_review_profile()
    response, audit = llm.complete_json(
        system=_SYSTEM_PROMPT,
        user=_user_prompt(payload),
        schema=WinrateLlmReview,
        profile=prof,
    )
    assert isinstance(response, WinrateLlmReview)
    return response, audit


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_INSERT_REVIEW_SQL = """
INSERT INTO lub_winrate_reviews (
    review_id, window_start, window_end,
    llm_provider, llm_model,
    sample_total, sample_resolved,
    strict_win_rate, non_loss_rate,
    payload_json, response_json, report_path
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def persist_review(
    db: Database,
    *,
    review_id: str,
    window_start: str,
    window_end: str,
    llm_provider: str,
    llm_model: str | None,
    summary: WinrateSummary,
    payload: dict[str, Any],
    response: WinrateLlmReview,
    report_path: str | None,
) -> None:
    """Persist one llm-review invocation to ``lub_winrate_reviews``.

    Caller wraps this in try/except — failures emit warnings but do not
    delete the report file already written.
    """
    db.execute(
        _INSERT_REVIEW_SQL,
        (
            review_id,
            window_start,
            window_end,
            llm_provider,
            llm_model,
            summary.total,
            summary.resolved,
            summary.strict_win_rate,
            summary.non_loss_rate,
            json.dumps(payload, ensure_ascii=False),
            json.dumps(response.model_dump(), ensure_ascii=False),
            report_path,
        ),
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown_report(
    *,
    review_id: str,
    window_start: str,
    window_end: str,
    llm_provider: str,
    summary: WinrateSummary,
    response: WinrateLlmReview,
) -> str:
    """Render LLM Review as a human-readable markdown doc."""
    lines: list[str] = []
    lines.append(f"# limit-up-board 胜率 LLM Review · {window_start}..{window_end}")
    lines.append("")
    lines.append(f"- review_id: `{review_id}`")
    lines.append(f"- llm_provider: `{llm_provider}`")
    lines.append(f"- 样本: total={summary.total} resolved={summary.resolved}")
    if summary.strict_win_rate is not None:
        lines.append(f"- 严格胜率: {summary.strict_win_rate * 100:.1f}%")
    if summary.non_loss_rate is not None:
        lines.append(f"- 非亏比例: {summary.non_loss_rate * 100:.1f}%")
    lines.append("")
    lines.append("## diagnosis")
    lines.append("")
    lines.append(response.diagnosis)
    lines.append("")

    def _emit_block(title: str, items: list[LandedSuggestion]) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.append("")
        for it in items:
            lines.append(f"- **[{it.landing}]** {it.text}")
        lines.append("")

    _emit_block("prompt_adjustments", response.prompt_adjustments)
    _emit_block("feature_suggestions", response.feature_suggestions)
    _emit_block("risk_controls", response.risk_controls)

    lines.append("## validation_plan")
    lines.append("")
    lines.append(response.validation_plan)
    lines.append("")

    if response.caveats:
        lines.append("## caveats")
        lines.append("")
        for c in response.caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> 本报告仅用于策略研究，不构成投资建议。")
    return "\n".join(lines)


def mint_review_id() -> str:
    """Short UUID-derived review_id — stable enough for cross-referencing."""
    return uuid.uuid4().hex[:16]
