"""Pydantic schemas for the LLM stages.

DESIGN §12.4-12.5 + the v0.3.1 fixes:
    F5 — 强势初筛 evidence max_length=4 (was 8); rationale length-capped via prompt
    M3 — extra='forbid' on every model
    M4 — FinalRankingResponse for multi-batch 连板预测 reconciliation
    S5 — final_rank field separated from batch_local_rank semantics
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Common evidence shape
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """One field-level fact from the input data the LLM is using to reason.

    `field` MUST refer to a key actually present in the input prompt (e.g.
    `fd_amount_yi`, `up_stat`); `unit` is REQUIRED (F-M1 fix) so the prompt
    and the LLM speak the same units (亿/万/%/次/日/秒/none/...).
    Use literal "none" when the field genuinely has no unit (e.g. categorical).
    """

    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., min_length=1, max_length=64)
    value: str | int | float | list[str] | None
    unit: str = Field(..., min_length=1, max_length=16)
    interpretation: str = Field(..., min_length=1, max_length=120)


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
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=4)
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
    key_evidence: list[EvidenceItem] = Field(min_length=1, max_length=5)
    next_day_watch_points: list[str] = Field(min_length=1, max_length=4)
    failure_triggers: list[str] = Field(min_length=1, max_length=4)
    missing_data: list[str] = Field(default_factory=list)


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
