"""Pydantic schemas for the LLM stages.

DESIGN §12.4-12.5 + the v0.3.1 fixes:
    F5 — 强势初筛 evidence max_length=4 (was 8); rationale length-capped via prompt
    M3 — extra='forbid' on every model
    M4 — FinalRankingResponse for multi-batch 连板预测 reconciliation
    S5 — final_rank field separated from batch_local_rank semantics

v0.12.4 — 空数组策略（P1-2）：``next_day_watch_points`` / ``failure_triggers``
被 prompt 明确禁止返回空数组，但偶发仍会发生。通过
``_empty_array_policy_ctx`` ContextVar 在三种行为间切换：
    - "repair"   → 抛 ValueError，让框架走 LLM repair-retry（默认）
    - "degraded" → 保留占位符 + 在 ``degraded_fields`` 标注命中字段名
    - "fallback" → 仅替换为占位符（v0.12.3 及之前的行为，已不再默认）
调用方在调用 ``LLMClient.complete_json`` 前用
``apply_empty_array_policy(...)`` 设定上下文。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_WATCH_FALLBACK = "（LLM 未给出观察点，人工复核）"
_TRIGGER_FALLBACK = "（LLM 未给出失败触发条件，人工复核）"

EmptyArrayPolicy = Literal["repair", "degraded", "fallback"]
_empty_array_policy_ctx: ContextVar[EmptyArrayPolicy] = ContextVar(
    "lub_empty_array_policy", default="repair"
)


@contextmanager
def apply_empty_array_policy(policy: EmptyArrayPolicy) -> Iterator[None]:
    """Set the active empty-array policy for the duration of the ``with`` block.

    Validators on :class:`ContinuationCandidate` read this ContextVar; the
    pipeline wraps each ``complete_json`` call in this manager so the policy
    follows the run config rather than being a static schema constant.
    """
    token = _empty_array_policy_ctx.set(policy)
    try:
        yield
    finally:
        _empty_array_policy_ctx.reset(token)


def _empty_fields(data: dict) -> list[str]:
    out: list[str] = []
    for fname in ("next_day_watch_points", "failure_triggers"):
        v = data.get(fname)
        if isinstance(v, list) and len(v) == 0:
            out.append(fname)
    return out


def _fallback_for(field_name: str) -> str:
    return _WATCH_FALLBACK if field_name == "next_day_watch_points" else _TRIGGER_FALLBACK

# ---------------------------------------------------------------------------
# Common evidence shape
# ---------------------------------------------------------------------------


class EvidenceItemStrict(BaseModel):
    """v0.12.4 (P2-1)：LLM 阶段使用的 strict evidence schema —— ``value`` 仅允许
    标量（与 ``prompts.py`` 中 \"严禁数组/对象\" 的硬性约束一致）。

    `field` MUST refer to a key actually present in the input prompt (e.g.
    `fd_amount_yi`, `up_stat`); `unit` is REQUIRED (F-M1 fix) so the prompt
    and the LLM speak the same units (亿/万/%/次/日/秒/none/...).
    Use literal "none" when the field genuinely has no unit (e.g. categorical).
    """

    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., min_length=1, max_length=64)
    value: str | int | float | None
    unit: str = Field(..., min_length=1, max_length=16)
    interpretation: str = Field(..., min_length=1, max_length=120)


class EvidenceItem(EvidenceItemStrict):
    """v0.12.3 及之前的宽松 schema，保留以兼容旧 ``lub_stage_results`` 反序列化。
    渲染层（``report.schema`` / winrate review）通过本类反序列化旧 run 历史，因为
    旧数据可能含 ``value: list[str]``。

    本类继承自 :class:`EvidenceItemStrict` 并将 ``value`` 字段加宽，从而：
    （a）``EvidenceItem`` 实例同时也是 ``EvidenceItemStrict`` 实例（isinstance 真），
        所以新 strict 字段类型仍能接收旧 fixture 构造的对象；
    （b）``EvidenceItemStrict(value=[...])`` 直接报 ValidationError，把 prompt 约束
        从文案升级为类型约束。
    """

    # pydantic v2 允许子类用更宽的注解覆盖父字段。
    value: str | int | float | list[str] | None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 强势初筛 — strong-target analysis
# ---------------------------------------------------------------------------


class StrongCandidate(BaseModel):
    """One 强势初筛 verdict per input candidate."""

    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    ts_code: str
    name: str
    selected: bool
    score: float = Field(ge=0, le=100)
    strength_level: Literal["high", "medium", "low"]
    rationale: str = Field(..., max_length=120)
    evidence: list[EvidenceItemStrict] = Field(min_length=1, max_length=4)
    risk_flags: list[str] = Field(default_factory=list, max_length=5)
    missing_data: list[str] = Field(default_factory=list)


class StrongAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["strong_target_analysis"]
    trade_date: str
    batch_no: int = Field(ge=1)
    batch_total: int = Field(ge=1)
    candidates: list[StrongCandidate]
    batch_summary: str


# ---------------------------------------------------------------------------
# 连板预测 — continuation prediction
# ---------------------------------------------------------------------------


class ContinuationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    ts_code: str
    name: str
    rank: int = Field(ge=1)
    continuation_score: float = Field(ge=0, le=100)
    confidence: Literal["high", "medium", "low"]
    prediction: Literal["top_candidate", "watchlist", "avoid"]
    rationale: str = Field(..., max_length=200)
    key_evidence: list[EvidenceItemStrict] = Field(min_length=1, max_length=5)
    next_day_watch_points: list[str] = Field(min_length=1, max_length=4)
    failure_triggers: list[str] = Field(min_length=1, max_length=4)
    missing_data: list[str] = Field(default_factory=list)
    # v0.12.4 (P1-2)：当 empty_array_policy="degraded" 时由 _apply_empty_array_policy
    # 自动写入，记录哪些字段被占位符兜底。"repair" / "fallback" 路径不写入。
    degraded_fields: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="before")
    @classmethod
    def _apply_empty_array_policy(cls, data):
        """Resolve empty ``next_day_watch_points`` / ``failure_triggers`` according
        to the active :data:`_empty_array_policy_ctx`. See module docstring."""
        if not isinstance(data, dict):
            return data
        empty_fields = _empty_fields(data)
        if not empty_fields:
            return data
        policy = _empty_array_policy_ctx.get()
        if policy == "repair":
            # 让 pydantic 抛 ValidationError；上游 LLMClient.complete_json 会捕获
            # 并触发 repair-retry。文案要让 LLM 知道哪些字段缺失。
            raise ValueError(
                "empty array(s) returned for required field(s): "
                f"{empty_fields}; prompt forbids empty arrays — regenerate "
                "with concrete观察点 / 失败触发条件 per candidate."
            )
        # degraded / fallback 都写占位符；degraded 额外记录命中字段名。
        # 不要原地修改入参（caller 可能复用同一 dict）；在副本上做替换。
        patched = dict(data)
        for fname in empty_fields:
            patched[fname] = [_fallback_for(fname)]
            logger.warning(
                "ContinuationCandidate.%s 返回空数组，自动填充占位符（policy=%s）",
                fname,
                policy,
            )
        if policy == "degraded":
            existing = list(patched.get("degraded_fields") or [])
            patched["degraded_fields"] = existing + empty_fields
        return patched


class ContinuationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["limit_up_continuation_prediction"]
    trade_date: str
    next_trade_date: str
    market_context_summary: str
    risk_disclaimer: str
    candidates: list[ContinuationCandidate]

    @field_validator("candidates")
    @classmethod
    def ranks_must_be_dense_1_to_n(
        cls, v: list[ContinuationCandidate]
    ) -> list[ContinuationCandidate]:
        """F-M1 — ranks must be a dense permutation 1..N (not just unique).
        E.g. [1,2,3] OK; [1,3,5] or [10,20,30] rejected."""
        ranks = sorted(c.rank for c in v)
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(f"candidate ranks must be a dense permutation 1..N; got {ranks}")
        return v


# ---------------------------------------------------------------------------
# 辩论修订 — multi-LLM peer revision (each LLM revises its own 连板预测 after
# seeing anonymised peer outputs)
# ---------------------------------------------------------------------------


class RevisedContinuationCandidate(ContinuationCandidate):
    """连板预测 fields + ``revision_note`` recording why the prediction shifted
    after reviewing peer LLM outputs."""

    model_config = ConfigDict(extra="forbid")
    revision_note: str = Field(..., max_length=120)


class RevisionResponse(BaseModel):
    """辩论修订 output. ``candidates`` keeps the same candidate_id set as the
    LLM's own 连板预测 (set-equality enforced by the pipeline); ranks are 1..N
    dense within this single batch."""

    model_config = ConfigDict(extra="forbid")
    stage: Literal["limit_up_continuation_revision"]
    trade_date: str
    next_trade_date: str
    revision_summary: str = Field(..., max_length=200)
    candidates: list[RevisedContinuationCandidate]

    @field_validator("candidates")
    @classmethod
    def ranks_must_be_dense_1_to_n(
        cls, v: list[RevisedContinuationCandidate]
    ) -> list[RevisedContinuationCandidate]:
        ranks = sorted(c.rank for c in v)
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(f"candidate ranks must be a dense permutation 1..N; got {ranks}")
        return v


# ---------------------------------------------------------------------------
# 全局重排 — global re-rank when 连板预测 was multi-batch (M4 + S5)
# ---------------------------------------------------------------------------


class FinalRankItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    ts_code: str
    final_rank: int = Field(ge=1)
    final_prediction: Literal["top_candidate", "watchlist", "avoid"]
    final_confidence: Literal["high", "medium", "low"]
    reason_vs_peers: str = Field(..., max_length=200)
    delta_vs_batch: Literal["upgraded", "kept", "downgraded"]


class FinalRankingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["final_ranking"]
    trade_date: str
    next_trade_date: str
    finalists: list[FinalRankItem]

    @field_validator("finalists")
    @classmethod
    def ranks_dense_and_unique(cls, v: list[FinalRankItem]) -> list[FinalRankItem]:
        ranks = sorted(c.final_rank for c in v)
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("final_rank must be a dense permutation 1..N")
        return v
