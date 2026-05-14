"""LLM pipeline for the volume-anomaly走势分析阶段。

Single-stage strategy: no R2 / final_ranking. Multi-batch is supported but the
batch outputs are simply concatenated (no global re-rank), per the user's spec.

Borrows the dual input/output token budget pattern from limit-up-board, but
does NOT import from it (plugins are self-contained).
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

from .data import AnalyzeBundle
from .profiles import STAGE_TREND_ANALYSIS, resolve_profile
from .prompts import VA_TREND_SYSTEM, va_trend_user_prompt
from .schemas import VATrendCandidate, VATrendResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token budgets / batching
# ---------------------------------------------------------------------------


# Per-candidate cost is higher than limit-up-board's R1 because we ship 60d
# aggregates + 5d recent OHLCV + moneyflow summary per stock.
DEFAULT_AVG_INPUT_TOKENS_PER_CANDIDATE = 1_500
# v0.6.0 — bumped from 900 → 1_100 to absorb dimension_scores (6 ints) +
# alpha_*_pct fields without crowding output budgets at 25-candidate batches.
DEFAULT_AVG_OUTPUT_TOKENS_PER_CANDIDATE = 1_100
DEFAULT_INPUT_BUDGET = 200_000  # matches continuation_prediction stage
SAFETY_RATIO = 0.85


@dataclass
class BatchPlan:
    batch_size: int
    n_batches: int


def plan_batches(
    *,
    n_candidates: int,
    input_budget: int,
    output_budget: int,
    overhead_input_tokens: int = 5_000,
    avg_in: int = DEFAULT_AVG_INPUT_TOKENS_PER_CANDIDATE,
    avg_out: int = DEFAULT_AVG_OUTPUT_TOKENS_PER_CANDIDATE,
) -> BatchPlan:
    """Pick the largest batch_size satisfying BOTH input and output budgets."""
    if n_candidates <= 0:
        return BatchPlan(batch_size=0, n_batches=0)
    in_room = max(0, input_budget - overhead_input_tokens)
    by_in = max(1, in_room // max(avg_in, 1))
    by_out = max(1, int(output_budget * SAFETY_RATIO) // max(avg_out, 1))
    batch_size = max(1, min(by_in, by_out, n_candidates))
    n_batches = (n_candidates + batch_size - 1) // batch_size
    return BatchPlan(batch_size=batch_size, n_batches=n_batches)


# ---------------------------------------------------------------------------
# Set equality check (input candidate_id ⊆ ⊇ output)
# ---------------------------------------------------------------------------


def _ids(items: Any) -> set[str]:
    return {(c["candidate_id"] if isinstance(c, dict) else c.candidate_id) for c in items}


def _set_mismatch_repair_hint(expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    parts = ["\n\n⚠ 上一次响应集合不一致，请严格按照原 candidate_id 列表重新输出。"]
    if missing:
        parts.append(f"missing (你必须包含): {missing}")
    if extra:
        parts.append(f"extra (你不能包含): {extra}")
    parts.append("不可遗漏，不可新增，不可改名。")
    return "\n".join(parts)


class _SetMismatchError(Exception):
    """LLM output's candidate_id set still differs after repair retry."""


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
    """Call LLM once; on candidate_id set mismatch, append a repair hint and retry."""
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
# Pipeline result container
# ---------------------------------------------------------------------------


@dataclass
class AnalyzeResult:
    """Outcome of the走势分析 LLM phase."""

    success_batches: int = 0
    failed_batches: int = 0
    candidates_in: int = 0
    candidates_out: int = 0
    predictions: list[VATrendCandidate] = field(default_factory=list)
    failed_batch_ids: list[str] = field(default_factory=list)
    batch_size: int = 0
    market_context_summaries: list[str] = field(default_factory=list)
    risk_disclaimers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Run analyze stage
# ---------------------------------------------------------------------------


def run_analyze(
    *,
    llm: LLMClient,
    bundle: AnalyzeBundle,
    preset: str,
    input_budget: int = DEFAULT_INPUT_BUDGET,
) -> Iterable[tuple[StrategyEvent, AnalyzeResult | None]]:
    """Run all analyze batches, yielding (event, terminal_result_or_None).

    The final iteration emits a STEP_FINISHED event paired with the populated
    AnalyzeResult so the caller can hand it to render.
    """
    profile = resolve_profile(preset, STAGE_TREND_ANALYSIS)
    candidates = bundle.candidates
    plan = plan_batches(
        n_candidates=len(candidates),
        input_budget=input_budget,
        output_budget=profile.max_output_tokens,
    )
    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=(
                f"[走势分析] 待处理 {len(candidates)} 只，分 {plan.n_batches} 批提交..."
            ),
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_STARTED,
            message="Step 2: 走势分析（主升浪启动预测）",
            payload={"n_candidates": len(candidates), "n_batches": plan.n_batches},
        ),
        None,
    )

    result = AnalyzeResult(candidates_in=len(candidates), batch_size=plan.batch_size)
    if plan.n_batches == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 2: 走势分析（主升浪启动预测）",
                payload={"predictions": 0},
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
                    f"[走势分析] 已提交第 {i + 1}/{plan.n_batches} 批 "
                    f"({len(batch)} 只)，等待 LLM 响应..."
                ),
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_STARTED,
                message=f"analyze batch {i + 1}/{plan.n_batches}",
                payload={"batch_no": i + 1, "size": len(batch)},
            ),
            None,
        )

        user = va_trend_user_prompt(
            trade_date=bundle.trade_date,
            next_trade_date=bundle.next_trade_date,
            batch_no=i + 1,
            batch_total=plan.n_batches,
            candidates=batch,
            market_summary=bundle.market_summary,
            sector_strength_source=bundle.sector_strength_source,
            sector_strength_data=bundle.sector_strength_data,
            data_unavailable=bundle.data_unavailable,
        )
        expected_ids = _ids(batch)
        try:
            obj, meta = _complete_with_set_check(
                llm,
                system=VA_TREND_SYSTEM,
                user=user,
                schema=VATrendResponse,
                profile=profile,
                expected_ids=expected_ids,
                envelope_defaults={
                    "stage": STAGE_TREND_ANALYSIS,
                    "trade_date": bundle.trade_date,
                    "next_trade_date": bundle.next_trade_date,
                    "batch_no": i + 1,
                    "batch_total": plan.n_batches,
                    "market_context_summary": "",
                    "risk_disclaimer": "",
                },
            )
        except (LLMValidationError, LLMTransportError, _SetMismatchError) as e:
            result.failed_batches += 1
            result.failed_batch_ids.append(f"analyze.batch.{i + 1}")
            yield (
                StrategyEvent(
                    type=EventType.VALIDATION_FAILED,
                    level=EventLevel.ERROR,
                    message=f"analyze batch {i + 1} failed: {e}",
                    payload={"batch_no": i + 1},
                ),
                None,
            )
            continue

        result.success_batches += 1
        result.candidates_out += len(obj.candidates)
        result.predictions.extend(obj.candidates)
        result.market_context_summaries.append(obj.market_context_summary)
        result.risk_disclaimers.append(obj.risk_disclaimer)
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_FINISHED,
                message=f"analyze batch {i + 1}/{plan.n_batches} ok",
                payload={
                    "batch_no": i + 1,
                    "input_tokens": meta["input_tokens"],
                    "output_tokens": meta["output_tokens"],
                },
            ),
            None,
        )
        n_imminent = sum(1 for c in result.predictions if c.prediction == "imminent_launch")
        yield (
            StrategyEvent(
                type=EventType.LIVE_STATUS,
                message=(
                    f"[走势分析] 第 {i + 1}/{plan.n_batches} 批响应已收到 "
                    f"(累计 imminent_launch {n_imminent} 只)"
                ),
            ),
            None,
        )

    yield (
        StrategyEvent(
            type=EventType.LIVE_STATUS,
            message=(
                f"[走势分析] 完成 — 共 {len(result.predictions)}/{result.candidates_in} 个预测"
            ),
        ),
        None,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_FINISHED,
            message="Step 2: 走势分析（主升浪启动预测）",
            payload={
                "success_batches": result.success_batches,
                "failed_batches": result.failed_batches,
                "predictions": len(result.predictions),
            },
        ),
        result,
    )
