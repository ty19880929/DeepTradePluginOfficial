"""Pydantic schemas for the volume-anomaly LLM stage.

Hard constraints (mirrors limit_up_board conventions):
    * extra='forbid' on every model
    * candidate_id round-trips verbatim from input
    * rank is a dense permutation 1..N within each batch
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VAEvidenceItem(BaseModel):
    """One field-level fact the LLM is using to reason.

    `field` MUST refer to a key actually present in the input prompt; `unit`
    is REQUIRED so prompt and model speak the same units.
    """

    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., min_length=1, max_length=64)
    value: str | int | float | None
    unit: str = Field(..., min_length=1, max_length=16)
    interpretation: str = Field(..., min_length=1, max_length=120)


class VADimensionScores(BaseModel):
    """v0.6.0 P1-2 — per-dimension explicit scoring (0–100).

    `risk` is reverse-polarity — a higher score means greater risk; every
    other dimension is positive-polarity (higher = more bullish). The
    relationship between this sub-object and ``launch_score`` is left to the
    LLM's own consistency (F3 / F14 — soft constraint via prompt, no formula).
    """

    model_config = ConfigDict(extra="forbid")
    washout: int = Field(ge=0, le=100)        # 洗盘充分度
    pattern: int = Field(ge=0, le=100)        # 形态突破有效性
    capital: int = Field(ge=0, le=100)        # 资金验证（moneyflow / volume）
    sector: int = Field(ge=0, le=100)         # 板块强度 + 大盘相对（与 P1-1 合流）
    historical: int = Field(ge=0, le=100)     # 历史浪型位置（越早越好）
    risk: int = Field(ge=0, le=100)           # 风险（reverse-polarity — 高分 = 高风险）


class VATrendCandidate(BaseModel):
    """One LLM verdict per input candidate."""

    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    ts_code: str
    name: str
    rank: int = Field(ge=1)
    launch_score: float = Field(ge=0, le=100)
    confidence: Literal["high", "medium", "low"]
    prediction: Literal["imminent_launch", "watching", "not_yet"]
    pattern: Literal[
        "breakout",
        "consolidation_break",
        "first_wave",
        "second_leg",
        "unclear",
    ]
    washout_quality: Literal["sufficient", "partial", "insufficient", "unclear"]
    rationale: str = Field(..., max_length=200)
    # v0.6.0 P1-2 — required field (F8). Old persisted responses live in
    # va_stage_results.raw_response_json so the strict schema is safe.
    dimension_scores: VADimensionScores
    key_evidence: list[VAEvidenceItem] = Field(min_length=1, max_length=5)
    next_session_watch: list[str] = Field(min_length=1, max_length=4)
    invalidation_triggers: list[str] = Field(min_length=1, max_length=4)
    risk_flags: list[str] = Field(default_factory=list, max_length=5)
    missing_data: list[str] = Field(default_factory=list)


class VATrendResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["continuation_prediction"]
    trade_date: str
    next_trade_date: str
    batch_no: int = Field(ge=1)
    batch_total: int = Field(ge=1)
    market_context_summary: str = Field(..., max_length=200)
    risk_disclaimer: str = Field(..., max_length=160)
    candidates: list[VATrendCandidate]

    @field_validator("candidates")
    @classmethod
    def ranks_must_be_dense_1_to_n(cls, v: list[VATrendCandidate]) -> list[VATrendCandidate]:
        """Per-batch rank must be a dense permutation 1..N."""
        ranks = sorted(c.rank for c in v)
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(f"candidate ranks must be a dense permutation 1..N; got {ranks}")
        return v
