"""Pydantic schemas for the accumulation-probe-washout LLM stage.

Hard constraints (mirrors VA / limit_up_board conventions):
    * ``extra='forbid'`` on every model
    * ``candidate_id`` round-trips verbatim from input
    * ``rank`` is a dense permutation ``1..N`` within each batch
    * ``key_evidence[].field`` is a member of the per-candidate field whitelist
      (validated by ``check_response_against_inputs`` after model_validate —
      caller-side because the whitelist is dynamic per batch)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Phase state machine — single source of truth for runner + LLM
# ---------------------------------------------------------------------------


class APWPhase(str, Enum):
    NO_SETUP = "no_setup"
    ACCUMULATING = "accumulating"
    PROBE_SEEN = "probe_seen"
    WASHING_AFTER_PROBE = "washing_after_probe"
    LAUNCH_READY = "launch_ready"


# Top-level enum value tuples (also re-exported for tests / prompt rendering).
LLM_PREDICTION_VALUES = (
    "launch_ready",
    "watch_breakout",
    "still_washing",
    "probe_failed",
    "avoid",
)

LLM_MAIN_PATTERN_VALUES = (
    "probe_washout_breakout",
    "low_base_accumulation",
    "second_probe_after_washout",
    "failed_probe",
    "high_level_distribution",
    "unclear",
)

LLM_CONFIDENCE_VALUES = ("high", "medium", "low")

LLM_PHASE_VALUES = (
    "accumulating",
    "probe_seen",
    "washing_after_probe",
    "launch_ready",
    "unclear",
)


# ---------------------------------------------------------------------------
# Pydantic models — LLM I/O contract
# ---------------------------------------------------------------------------


class APWEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., min_length=1, max_length=64)
    value: str | int | float | None
    unit: str = Field(..., min_length=1, max_length=16)
    interpretation: str = Field(..., min_length=1, max_length=120)


class APWDimensionScores(BaseModel):
    """0–100 per dimension. ``risk`` is reverse-polarity (higher = worse)."""

    model_config = ConfigDict(extra="forbid")
    accumulation: int = Field(ge=0, le=100)
    probe: int = Field(ge=0, le=100)
    washout: int = Field(ge=0, le=100)
    launch_timing: int = Field(ge=0, le=100)
    capital_confirmation: int = Field(ge=0, le=100)
    risk: int = Field(ge=0, le=100)


class APWTrendCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    ts_code: str
    name: str
    rank: int = Field(ge=1)
    launch_score: float = Field(ge=0, le=100)
    confidence: Literal["high", "medium", "low"]
    prediction: Literal[
        "launch_ready",
        "watch_breakout",
        "still_washing",
        "probe_failed",
        "avoid",
    ]
    main_pattern: Literal[
        "probe_washout_breakout",
        "low_base_accumulation",
        "second_probe_after_washout",
        "failed_probe",
        "high_level_distribution",
        "unclear",
    ]
    phase: Literal[
        "accumulating",
        "probe_seen",
        "washing_after_probe",
        "launch_ready",
        "unclear",
    ]
    dimension_scores: APWDimensionScores
    rationale: str = Field(..., max_length=220)
    key_evidence: list[APWEvidenceItem] = Field(min_length=1, max_length=6)
    next_session_watch: list[str] = Field(min_length=1, max_length=5)
    invalidation_triggers: list[str] = Field(min_length=1, max_length=5)
    risk_flags: list[str] = Field(default_factory=list, max_length=6)
    missing_data: list[str] = Field(default_factory=list)


class APWTrendResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["accumulation_probe_washout_analysis"]
    trade_date: str
    next_trade_date: str
    batch_no: int = Field(ge=1)
    batch_total: int = Field(ge=1)
    market_context_summary: str = Field(..., max_length=220)
    risk_disclaimer: str = Field(..., max_length=180)
    candidates: list[APWTrendCandidate]

    @field_validator("candidates")
    @classmethod
    def _ranks_must_be_dense_1_to_n(cls, v: list[APWTrendCandidate]) -> list[APWTrendCandidate]:
        ranks = sorted(c.rank for c in v)
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(f"candidate ranks must be a dense permutation 1..N; got {ranks}")
        return v


# ---------------------------------------------------------------------------
# Helpers — caller-side validators
# ---------------------------------------------------------------------------


# Fields that exist on every input candidate; key_evidence.field MUST come
# from this set (or the per-candidate union if dynamic fields are added later).
INPUT_FIELD_WHITELIST: frozenset[str] = frozenset(
    [
        "candidate_id", "ts_code", "name", "trade_date", "phase",
        "listed_days", "close", "pct_chg", "turnover_rate", "amount_yi", "circ_mv_yi",
        # accumulation
        "accumulation_score", "accumulation_days", "accumulation_net_mf_yi",
        "accumulation_price_change_pct", "low_position_score",
        # probe
        "probe_date", "probe_days_ago", "probe_volume_ratio_5d",
        "probe_volume_ratio_20d", "probe_volume_rank_pct_60d",
        "probe_amount_ratio_20d",
        "probe_turnover_rate", "probe_amplitude_pct", "probe_upper_shadow_ratio",
        "probe_body_ratio",
        "probe_pct_chg", "probe_moneyflow_net_yi", "probe_quality_score",
        # washout
        "washout_days", "post_probe_max_drawdown_pct",
        "post_probe_volume_shrink_ratio", "post_probe_low_broken",
        "post_probe_ma20_broken", "post_probe_ma60_broken",
        "post_probe_moneyflow_net_yi", "washout_volatility_compression",
        "washout_score",
        # launch
        "launch_setup_score", "close_to_probe_high_pct", "break_probe_high",
        "break_washout_high", "current_volume_ratio_5d",
        "current_volume_ratio_20d", "current_moneyflow_net_yi",
        "above_ma5", "above_ma10", "above_ma20", "relative_strength_20d",
        "sector_strength_score",
        # v0.4.0 — VCP / 长周期阻力位 / alpha / MA / 涨停历史 / 量能事件
        "atr_10d", "atr_10d_pct", "atr_10d_quantile_in_60d",
        "bbw_20d", "bbw_compression_ratio",
        "dist_to_120d_high_pct", "dist_to_250d_high_pct",
        "is_above_120d_high", "is_above_250d_high", "pos_in_120d_range",
        "alpha_5d_pct", "alpha_20d_pct", "alpha_60d_pct", "alpha_leading",
        "ma5", "ma10", "ma20", "ma60",
        "close_to_ma5_pct", "close_to_ma10_pct",
        "close_to_ma20_pct", "close_to_ma60_pct",
        "volume_event_score",
        "prior_limit_up_count_60d", "days_since_last_limit_up",
        "probe_low", "probe_high",
        # meta
        "risk_flags_local", "missing_data",
    ]
)


class EvidenceFieldError(ValueError):
    """Raised when a key_evidence.field is not in the input field whitelist."""


def check_response_against_inputs(
    response: APWTrendResponse, input_ids: set[str]
) -> None:
    """Caller-side validators that need batch context (candidate_id set + whitelist).

    Raises ValueError on:
        * candidate_id set mismatch (handled by pipeline._complete_with_set_check,
          but a defensive double-check here keeps tests honest)
        * key_evidence.field not in INPUT_FIELD_WHITELIST
    """
    out_ids = {c.candidate_id for c in response.candidates}
    if out_ids != input_ids:
        missing = sorted(input_ids - out_ids)
        extra = sorted(out_ids - input_ids)
        raise ValueError(
            f"candidate_id set mismatch; missing={missing}, extra={extra}"
        )
    for cand in response.candidates:
        for ev in cand.key_evidence:
            if ev.field not in INPUT_FIELD_WHITELIST:
                raise EvidenceFieldError(
                    f"candidate {cand.candidate_id} key_evidence.field={ev.field!r} "
                    f"not in input whitelist"
                )
