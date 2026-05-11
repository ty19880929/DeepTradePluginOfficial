"""LLM pipeline for limit-up-board: R1 / R2 / final_ranking.

Implements:
    plan_r1_batches()                — F5 input + output token dual budget
    run_r1_batches()                 — yields events; collects StrongCandidate
    run_r2()                         — single-batch by default; auto multi-batch + final_ranking
    run_final_ranking()              — only when R2 multi-batch (M4)
    set_equality_check()             — strict candidate_id ⊆ ⊇ check
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from deeptrade.core.llm_client import (
    LLMClient,
    LLMTransportError,
    LLMValidationError,
)
from deeptrade.plugins_api import StageProfile
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from .data import Round1Bundle, SectorStrength
from .profiles import STAGE_FINAL, STAGE_R1, STAGE_R2, STAGE_R3, resolve_profile
from .prompts import (
    FINAL_RANKING_SYSTEM,
    R3_DEBATE_SYSTEM,
    build_r1_system,
    build_r2_system,
    final_ranking_user_prompt,
    r1_user_prompt,
    r2_user_prompt,
    r3_user_prompt,
)
from .schemas import (
    ContinuationCandidate,
    ContinuationResponse,
    FinalRankingResponse,
    RevisedContinuationCandidate,
    RevisionResponse,
    StrongAnalysisResponse,
    StrongCandidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token budgets / batching
# ---------------------------------------------------------------------------


# Conservative average per-candidate token estimates. Better-than-nothing
# defaults; production overrides via configure() if needed.
# avg_out raised from 600 → 800 after audit-log measurements showed actual
# per-candidate output (~250 tok with the old prompt, but the new full-schema
# prompt is more verbose) plus we want smaller batches: 32k * 0.85 / 800 ≈ 34
# candidates/batch instead of ~46. Smaller batches = shorter wall-clock per
# call and lower risk of hitting the per-call output cap.
DEFAULT_AVG_INPUT_TOKENS_PER_CANDIDATE = 350
DEFAULT_AVG_OUTPUT_TOKENS_PER_CANDIDATE = 800
DEFAULT_R1_INPUT_BUDGET = 80_000
DEFAULT_R2_INPUT_BUDGET = 200_000
SAFETY_RATIO = 0.85


@dataclass
class BatchPlan:
    batch_size: int
    n_batches: int


def plan_r1_batches(
    *,
    n_candidates: int,
    input_budget: int = DEFAULT_R1_INPUT_BUDGET,
    output_budget: int,  # = stage profile's max_output_tokens (R1 default 32k)
    overhead_input_tokens: int = 4_000,  # system prompt + market summary + sector
    avg_in: int = DEFAULT_AVG_INPUT_TOKENS_PER_CANDIDATE,
    avg_out: int = DEFAULT_AVG_OUTPUT_TOKENS_PER_CANDIDATE,
) -> BatchPlan:
    """Pick the largest batch_size satisfying BOTH input and output budgets.

    F5 fix: BOTH budgets matter — input alone won't catch the case where the
    LLM tries to write 30 candidates × 600 tokens > 8k output cap.
    """
    if n_candidates <= 0:
        return BatchPlan(batch_size=0, n_batches=0)

    in_room = max(0, input_budget - overhead_input_tokens)
    by_in = max(1, in_room // max(avg_in, 1))
    by_out = max(1, int(output_budget * SAFETY_RATIO) // max(avg_out, 1))
    batch_size = max(1, min(by_in, by_out, n_candidates))
    n_batches = (n_candidates + batch_size - 1) // batch_size
    return BatchPlan(batch_size=batch_size, n_batches=n_batches)


# ---------------------------------------------------------------------------
# Set equality check (M5 propagation)
# ---------------------------------------------------------------------------


def candidate_id_set_equal(
    inputs: list[dict[str, Any]] | list[StrongCandidate],
    outputs: list[StrongCandidate] | list[ContinuationCandidate],
) -> bool:
    """True iff input.candidate_id == output.candidate_id (as sets)."""
    in_ids = _ids(inputs)
    out_ids = {c.candidate_id for c in outputs}
    return in_ids == out_ids


def _ids(items: Any) -> set[str]:
    """Extract candidate_id set from a list of dicts or BaseModels."""
    return {(c["candidate_id"] if isinstance(c, dict) else c.candidate_id) for c in items}


def _set_mismatch_repair_hint(expected: set[str], actual: set[str]) -> str:
    """Build a corrective instruction appended to user prompt on retry."""
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    parts = ["\n\n⚠ 上一次响应集合不一致，请严格按照原 candidate_id 列表重新输出。"]
    if missing:
        parts.append(f"missing (你必须包含): {missing}")
    if extra:
        parts.append(f"extra (你不能包含): {extra}")
    parts.append("不可遗漏，不可新增，不可改名。")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline result containers
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    """Outcome of a single R1 / R2 / final_ranking phase."""

    success_batches: int = 0
    failed_batches: int = 0
    candidates_in: int = 0
    candidates_out: int = 0
    selected: list[StrongCandidate] = field(default_factory=list)
    predictions: list[ContinuationCandidate] = field(default_factory=list)
    final_items: list[Any] = field(default_factory=list)
    # F-L3 — concrete failed batch IDs (e.g. "r1.batch.3") for report banner
    failed_batch_ids: list[str] = field(default_factory=list)
    # F-M3 — record actual batch_size so finalists selection isn't hardcoded
    batch_size: int = 0


# ---------------------------------------------------------------------------
# F-H1 — set-mismatch repair retry
# ---------------------------------------------------------------------------


class _SetMismatchError(Exception):
    """Raised when LLM output's candidate_id set still doesn't equal the
    expected set after one repair retry."""


def _complete_with_set_check(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    schema: type[BaseModel],
    profile: StageProfile,
    expected_ids: set[str],
    output_attr: str = "candidates",
    repair_retries: int = 1,
    envelope_defaults: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Call LLM and verify the output's id-set matches. On mismatch, retry
    once with a corrective hint appended to the user prompt.

    Args:
        profile: caller-resolved StageProfile (see profiles.py).
        output_attr: name of the BaseModel field that holds the list of items
            (each having a ``.candidate_id`` attr). 'candidates' for R1/R2,
            'finalists' for final_ranking.
        envelope_defaults: caller-controlled top-level fields (e.g. ``stage``,
            ``trade_date``, ``batch_no``) injected when the LLM omits them.
    """
    current_user = user
    last_actual: set[str] = set()
    for attempt in range(repair_retries + 1):
        raw, meta = llm.complete_json(
            system=system,
            user=current_user,
            schema=schema,
            profile=profile,
            envelope_defaults=envelope_defaults,
        )
        obj = raw if isinstance(raw, schema) else schema.model_validate(raw)
        items = getattr(obj, output_attr)
        actual_ids = {item.candidate_id for item in items}
        if expected_ids == actual_ids:
            return obj, meta
        last_actual = actual_ids
        if attempt < repair_retries:
            current_user = user + _set_mismatch_repair_hint(expected_ids, actual_ids)
    raise _SetMismatchError(
        f"set mismatch after {repair_retries + 1} attempts; "
        f"missing={sorted(expected_ids - last_actual)}, "
        f"extra={sorted(last_actual - expected_ids)}"
    )


# ---------------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------------


def run_r1(
    *,
    llm: LLMClient,
    bundle: Round1Bundle,
    preset: str,
    input_budget: int = DEFAULT_R1_INPUT_BUDGET,
    lgb_min_score_floor: float | None = 30.0,
) -> Iterable[tuple[StrategyEvent, RoundResult | None]]:
    """Run all R1 batches, yielding (event, terminal_result_or_None).

    The caller (strategy.run) re-yields the events into the runner. The final
    iteration yields a result alongside the STEP_FINISHED event so the caller
    can hand it on to R2.
    """
    profile = resolve_profile(preset, STAGE_R1)
    candidates = bundle.candidates
    plan = plan_r1_batches(
        n_candidates=len(candidates),
        input_budget=input_budget,
        output_budget=profile.max_output_tokens,
    )
    r1_system = build_r1_system(lgb_min_score_floor=lgb_min_score_floor)
    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=(
                f"[强势标的分析] 待处理 {len(candidates)} 只，分 {plan.n_batches} 批提交..."
            ),
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_STARTED,
            message="Step 2: R1 strong target analysis",
            payload={"n_candidates": len(candidates), "n_batches": plan.n_batches},
        ),
        None,
    )

    result = RoundResult(candidates_in=len(candidates), batch_size=plan.batch_size)
    if plan.n_batches == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 2: R1 strong target analysis",
                payload={"selected": 0},
            ),
            result,
        )
        return

    for i in range(plan.n_batches):
        batch = candidates[i * plan.batch_size : (i + 1) * plan.batch_size]
        yield (
            StrategyEvent(
                type=EventType.LIVE_STATUS,
                message=(
                    f"[强势标的分析] 已提交第 {i + 1}/{plan.n_batches} 批 "
                    f"({len(batch)} 只)，等待 LLM 响应..."
                ),
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_STARTED,
                message=f"R1 batch {i + 1}/{plan.n_batches}",
                payload={"batch_no": i + 1, "size": len(batch)},
            ),
            None,
        )

        user = r1_user_prompt(
            trade_date=bundle.trade_date,
            batch_no=i + 1,
            batch_total=plan.n_batches,
            candidates=batch,
            market_summary=bundle.market_summary,
            sector_strength_source=bundle.sector_strength.source,
            sector_strength_data=bundle.sector_strength.data,
            data_unavailable=bundle.data_unavailable,
        )

        expected_ids = _ids(batch)
        try:
            obj, meta = _complete_with_set_check(
                llm,
                system=r1_system,
                user=user,
                schema=StrongAnalysisResponse,
                profile=profile,
                expected_ids=expected_ids,
                envelope_defaults={
                    "stage": STAGE_R1,
                    "trade_date": bundle.trade_date,
                    "batch_no": i + 1,
                    "batch_total": plan.n_batches,
                    "batch_summary": "",
                },
            )
        except (LLMValidationError, LLMTransportError, _SetMismatchError) as e:
            result.failed_batches += 1
            result.failed_batch_ids.append(f"r1.batch.{i + 1}")
            yield (
                StrategyEvent(
                    type=EventType.VALIDATION_FAILED,
                    level=EventLevel.ERROR,
                    message=f"R1 batch {i + 1} failed: {e}",
                    payload={"batch_no": i + 1},
                ),
                None,
            )
            continue

        result.success_batches += 1
        result.candidates_out += len(obj.candidates)
        result.selected.extend(c for c in obj.candidates if c.selected)
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_FINISHED,
                message=f"R1 batch {i + 1}/{plan.n_batches} ok",
                payload={
                    "batch_no": i + 1,
                    "input_tokens": meta["input_tokens"],
                    "output_tokens": meta["output_tokens"],
                },
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.LIVE_STATUS,
                message=(
                    f"[强势标的分析] 第 {i + 1}/{plan.n_batches} 批响应已收到 "
                    f"(累计入选 {sum(1 for c in result.selected)} 只)"
                ),
            ),
            None,
        )

    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=(
                f"[强势标的分析] 完成 — 入选 {len(result.selected)}/{result.candidates_in}"
            ),
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_FINISHED,
            message="Step 2: R1 strong target analysis",
            payload={
                "success_batches": result.success_batches,
                "failed_batches": result.failed_batches,
                "selected": len(result.selected),
            },
        ),
        result,
    )


# ---------------------------------------------------------------------------
# R2 + (optional) final_ranking
# ---------------------------------------------------------------------------


def run_r2(
    *,
    llm: LLMClient,
    selected: list[StrongCandidate],
    bundle: Round1Bundle,
    preset: str,
    input_budget: int = DEFAULT_R2_INPUT_BUDGET,
    lgb_min_score_floor: float | None = 30.0,
) -> Iterable[tuple[StrategyEvent, RoundResult | None]]:
    """Run R2; if the candidate set exceeds the input budget, multi-batch + final_ranking."""
    profile = resolve_profile(preset, STAGE_R2)
    plan = plan_r1_batches(
        n_candidates=len(selected),
        input_budget=input_budget,
        output_budget=profile.max_output_tokens,
    )
    r2_system = build_r2_system(lgb_min_score_floor=lgb_min_score_floor)
    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=f"[连板预测] 待处理 {len(selected)} 只，分 {plan.n_batches} 批提交...",
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_STARTED,
            message="Step 4: R2 continuation prediction",
            payload={"n_candidates": len(selected), "n_batches": plan.n_batches},
        ),
        None,
    )

    result = RoundResult(candidates_in=len(selected), batch_size=plan.batch_size)
    if plan.n_batches == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 4: R2 continuation prediction",
                payload={"predictions": 0},
            ),
            result,
        )
        return

    # Build candidate dicts for the prompt (R1 selected → minimal payload)
    payload_rows: list[dict[str, Any]] = [
        _r2_row_from_selected(c, bundle.candidates) for c in selected
    ]

    for i in range(plan.n_batches):
        batch_objs = selected[i * plan.batch_size : (i + 1) * plan.batch_size]
        batch_rows = payload_rows[i * plan.batch_size : (i + 1) * plan.batch_size]
        yield (
            StrategyEvent(
                type=EventType.LIVE_STATUS,
                message=(
                    f"[连板预测] 已提交第 {i + 1}/{plan.n_batches} 批 "
                    f"({len(batch_objs)} 只)，等待 LLM 响应..."
                ),
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_STARTED,
                message=f"R2 batch {i + 1}/{plan.n_batches}",
                payload={"batch_no": i + 1, "size": len(batch_objs)},
            ),
            None,
        )

        user = r2_user_prompt(
            trade_date=bundle.trade_date,
            next_trade_date=bundle.next_trade_date,
            candidates=batch_rows,
            market_context=bundle.market_summary,
            sector_strength_source=bundle.sector_strength.source,
            sector_strength_data=bundle.sector_strength.data,
            data_unavailable=bundle.data_unavailable,
        )

        expected_ids = _ids(batch_objs)
        try:
            obj, meta = _complete_with_set_check(
                llm,
                system=r2_system,
                user=user,
                schema=ContinuationResponse,
                profile=profile,
                expected_ids=expected_ids,
                envelope_defaults={
                    "stage": "limit_up_continuation_prediction",
                    "trade_date": bundle.trade_date,
                    "next_trade_date": bundle.next_trade_date,
                    "market_context_summary": "",
                    "risk_disclaimer": "",
                },
            )
        except (LLMValidationError, LLMTransportError, _SetMismatchError) as e:
            result.failed_batches += 1
            result.failed_batch_ids.append(f"r2.batch.{i + 1}")
            yield (
                StrategyEvent(
                    type=EventType.VALIDATION_FAILED,
                    level=EventLevel.ERROR,
                    message=f"R2 batch {i + 1} failed: {e}",
                    payload={"batch_no": i + 1},
                ),
                None,
            )
            continue

        result.success_batches += 1
        result.predictions.extend(obj.candidates)
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_FINISHED,
                message=f"R2 batch {i + 1}/{plan.n_batches} ok",
                payload={
                    "batch_no": i + 1,
                    "input_tokens": meta["input_tokens"],
                    "output_tokens": meta["output_tokens"],
                },
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.LIVE_STATUS,
                message=(
                    f"[连板预测] 第 {i + 1}/{plan.n_batches} 批响应已收到 "
                    f"(累计 {len(result.predictions)} 只预测)"
                ),
            ),
            None,
        )

    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=f"[连板预测] 完成 — 共预测 {len(result.predictions)} 只",
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_FINISHED,
            message="Step 4: R2 continuation prediction",
            payload={
                "success_batches": result.success_batches,
                "failed_batches": result.failed_batches,
                "predictions": len(result.predictions),
            },
        ),
        result,
    )


# ---------------------------------------------------------------------------
# Final ranking (only triggered when R2 was multi-batch)
# ---------------------------------------------------------------------------


def select_finalists(
    predictions: list[ContinuationCandidate],
    *,
    per_batch_top_ratio: float = 0.6,
    batch_size_hint: int | None = None,
) -> list[ContinuationCandidate]:
    """Pick finalists for the global re-rank (S5).

    F-M3 fix:
      * Sort top_candidate / watchlist by `continuation_score` DESC before
        truncation, so cap = "top by score", not "by batch order".
      * Boundary avoids are also sorted by score before sampling.
    """
    top_and_watch = sorted(
        (c for c in predictions if c.prediction in ("top_candidate", "watchlist")),
        key=lambda c: c.continuation_score,
        reverse=True,
    )
    avoids = sorted(
        (c for c in predictions if c.prediction == "avoid"),
        key=lambda c: c.continuation_score,
        reverse=True,
    )
    finalists = top_and_watch
    if batch_size_hint:
        cap = max(1, int(batch_size_hint * per_batch_top_ratio))
        finalists = finalists[:cap]
    # Boundary samples: top-scored avoids (up to 1/5 of total predictions)
    finalists.extend(avoids[: max(0, len(predictions) // 5)])
    return finalists


def run_final_ranking(
    *,
    llm: LLMClient,
    bundle: Round1Bundle,
    finalists: list[ContinuationCandidate],
    preset: str,
) -> Iterable[tuple[StrategyEvent, FinalRankingResponse | None]]:
    profile = resolve_profile(preset, STAGE_FINAL)
    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=(
                f"[全局重排] 合并 {len(finalists)} 只 finalists，等待 LLM 响应..."
            ),
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_STARTED,
            message="Step 4.5: final_ranking global reconciliation",
            payload={"n_finalists": len(finalists)},
        ),
        None,
    )

    finalist_payload = [_final_row_from_pred(c) for c in finalists]
    user = final_ranking_user_prompt(
        trade_date=bundle.trade_date,
        next_trade_date=bundle.next_trade_date,
        finalists=finalist_payload,
        market_context=bundle.market_summary,
    )
    expected_ids = {f.candidate_id for f in finalists}
    try:
        # F-H1 — final_ranking now also checks candidate_id set equality
        # (previously only `final_rank` 1..N denseness was validated).
        obj, meta = _complete_with_set_check(
            llm,
            system=FINAL_RANKING_SYSTEM,
            user=user,
            schema=FinalRankingResponse,
            profile=profile,
            expected_ids=expected_ids,
            output_attr="finalists",
            envelope_defaults={
                "stage": STAGE_FINAL,
                "trade_date": bundle.trade_date,
                "next_trade_date": bundle.next_trade_date,
            },
        )
    except (LLMValidationError, LLMTransportError, _SetMismatchError) as e:
        yield (
            StrategyEvent(
                type=EventType.VALIDATION_FAILED,
                level=EventLevel.ERROR,
                message=f"final_ranking failed: {e}",
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 4.5: final_ranking global reconciliation",
                payload={"success": False},
            ),
            None,
        )
        return

    yield (
        StrategyEvent(
            type=EventType.LLM_FINAL_RANK,
            message="final_ranking complete",
            payload={
                "input_tokens": meta["input_tokens"],
                "output_tokens": meta["output_tokens"],
            },
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=f"[全局重排] 完成 — {len(obj.finalists)} 只全局重排完毕",
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_FINISHED,
            message="Step 4.5: final_ranking global reconciliation",
            payload={"success": True, "finalists": len(obj.finalists)},
        ),
        obj,
    )


# ---------------------------------------------------------------------------
# Internal payload shapers
# ---------------------------------------------------------------------------


def _r2_row_from_selected(
    sc: StrongCandidate, all_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a minimal R2 input row from an R1 selected verdict + R1 raw fields.

    We re-attach the normalized fields so the LLM doesn't have to re-derive them.
    """
    base = next((r for r in all_candidates if r["candidate_id"] == sc.candidate_id), {})
    row: dict[str, Any] = {
        **base,
        "r1_score": sc.score,
        "r1_strength_level": sc.strength_level,
        "r1_themes": [],
        "r1_rationale": sc.rationale,
    }
    # Flatten list-valued source fields to scalar siblings so the LLM can cite
    # them in EvidenceItem.value (which is scalar-only: str|int|float|None).
    seats = row.pop("lhb_famous_seats", None)
    seats_list = seats if isinstance(seats, list) else []
    row["lhb_famous_seats_count"] = len(seats_list)
    row["lhb_famous_seats_text"] = "; ".join(str(s) for s in seats_list)
    return row


def _final_row_from_pred(p: ContinuationCandidate) -> dict[str, Any]:
    return {
        "candidate_id": p.candidate_id,
        "ts_code": p.ts_code,
        "name": p.name,
        "continuation_score": p.continuation_score,
        "confidence": p.confidence,
        "prediction": p.prediction,
        "rationale": p.rationale[:120],
        "key_evidence": [
            {"field": e.field, "value": e.value, "unit": e.unit, "interpretation": e.interpretation}
            for e in p.key_evidence[:3]
        ],
    }


# ---------------------------------------------------------------------------
# R3 — Debate-mode revision (each LLM revises its own R2 after seeing peers)
# ---------------------------------------------------------------------------


@dataclass
class DebateRoundResult:
    """Outcome of a single LLM's R3 (debate-revision) phase."""

    success: bool = False
    error: str | None = None
    candidates_in: int = 0
    revision_summary: str = ""
    revised: list[RevisedContinuationCandidate] = field(default_factory=list)


def run_r3_debate(
    *,
    llm: LLMClient,
    bundle: Round1Bundle,
    own_predictions: list[ContinuationCandidate],
    peers: list[tuple[str, list[ContinuationCandidate]]],
    preset: str,
    self_label: str = "you",
) -> Iterable[tuple[StrategyEvent, DebateRoundResult | None]]:
    """Single-batch R3 revision; yields events + a terminal result.

    The revising LLM receives its own ``own_predictions`` (full view) plus a
    trimmed view of every entry in ``peers`` (already anonymised). The output
    candidate_id set is enforced to equal the input ``own_predictions`` set.
    """
    profile = resolve_profile(preset, STAGE_R3)
    n = len(own_predictions)
    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=f"[辩论修订] 待修订 {n} 只，参考同行 {len(peers)} 位...",
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_STARTED,
            message="Step 4.7: R3 debate revision",
            payload={"n_candidates": n, "n_peers": len(peers), "self_label": self_label},
        ),
        None,
    )

    result = DebateRoundResult(candidates_in=n)
    if n == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 4.7: R3 debate revision",
                payload={"success": False, "reason": "empty own_predictions"},
            ),
            result,
        )
        return

    user = r3_user_prompt(
        trade_date=bundle.trade_date,
        next_trade_date=bundle.next_trade_date,
        own_predictions=own_predictions,
        peers=peers,
        market_context=bundle.market_summary,
    )
    expected_ids = {c.candidate_id for c in own_predictions}
    yield (
        StrategyEvent(
            type=EventType.LLM_BATCH_STARTED,
            message="R3 debate (single batch)",
            payload={"size": n},
        ),
        None,
    )
    try:
        obj, meta = _complete_with_set_check(
            llm,
            system=R3_DEBATE_SYSTEM,
            user=user,
            schema=RevisionResponse,
            profile=profile,
            expected_ids=expected_ids,
            envelope_defaults={
                "stage": "limit_up_continuation_revision",
                "trade_date": bundle.trade_date,
                "next_trade_date": bundle.next_trade_date,
                "revision_summary": "",
            },
        )
    except (LLMValidationError, LLMTransportError, _SetMismatchError) as e:
        result.error = f"{type(e).__name__}: {e}"
        yield (
            StrategyEvent(
                type=EventType.VALIDATION_FAILED,
                level=EventLevel.ERROR,
                message=f"R3 debate failed: {e}",
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 4.7: R3 debate revision",
                payload={"success": False},
            ),
            result,
        )
        return

    result.success = True
    result.revised = list(obj.candidates)
    result.revision_summary = obj.revision_summary
    yield (
        StrategyEvent(
            type=EventType.LLM_BATCH_FINISHED,
            message="R3 debate ok",
            payload={
                "input_tokens": meta["input_tokens"],
                "output_tokens": meta["output_tokens"],
            },
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=f"[辩论修订] 完成 — 修订 {len(result.revised)} 只",
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_FINISHED,
            message="Step 4.7: R3 debate revision",
            payload={"success": True, "revised": len(result.revised)},
        ),
        result,
    )


# Suppress unused-import warning for fields we re-export indirectly via tests
_ = SectorStrength
