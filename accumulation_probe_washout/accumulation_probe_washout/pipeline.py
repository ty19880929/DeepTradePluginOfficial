"""LLM pipeline for the accumulation-probe-washout走势分析 stage.

Single stage, single batch round (no global re-rank). Multi-batch is supported
but each batch is independent — outputs are concatenated.

Contract (locked at v0.1):
    pack → batch → render prompt → call structured output → schema validate →
    repair retry ≤ 2 → write results
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from deeptrade.core.llm_client import (
    LLMClient,
    LLMTransportError,
    LLMValidationError,
)
from deeptrade.plugins_api.llm import StageProfile
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from .prompts import APW_SYSTEM, apw_user_prompt
from .schemas import (
    APWTrendCandidate,
    APWTrendResponse,
    EvidenceFieldError,
    check_response_against_inputs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token budgets / batching
# ---------------------------------------------------------------------------


DEFAULT_AVG_INPUT_TOKENS_PER_CANDIDATE = 1_400
DEFAULT_AVG_OUTPUT_TOKENS_PER_CANDIDATE = 1_100
DEFAULT_INPUT_BUDGET = 200_000
SAFETY_RATIO = 0.85


@dataclass
class BatchPlan:
    batch_size: int
    n_batches: int


def plan_batches(
    *,
    n_candidates: int,
    input_budget: int = DEFAULT_INPUT_BUDGET,
    output_budget: int = 64_000,
    overhead_input_tokens: int = 5_000,
    avg_in: int = DEFAULT_AVG_INPUT_TOKENS_PER_CANDIDATE,
    avg_out: int = DEFAULT_AVG_OUTPUT_TOKENS_PER_CANDIDATE,
    max_batch_size: int = 20,
) -> BatchPlan:
    """Pick the largest batch_size that satisfies BOTH input and output budgets.

    Capped at ``max_batch_size`` (default 20 per spec §4 in design doc).
    """
    if n_candidates <= 0:
        return BatchPlan(batch_size=0, n_batches=0)
    in_room = max(0, input_budget - overhead_input_tokens)
    by_in = max(1, in_room // max(avg_in, 1))
    by_out = max(1, int(output_budget * SAFETY_RATIO) // max(avg_out, 1))
    batch_size = max(1, min(by_in, by_out, max_batch_size, n_candidates))
    n_batches = (n_candidates + batch_size - 1) // batch_size
    return BatchPlan(batch_size=batch_size, n_batches=n_batches)


# ---------------------------------------------------------------------------
# Set equality + evidence whitelist repair loop
# ---------------------------------------------------------------------------


class _SetMismatchError(Exception):
    """LLM output's candidate_id set still differs after the repair budget."""


def _ids(items: Iterable[Any]) -> set[str]:
    return {(c["candidate_id"] if isinstance(c, dict) else c.candidate_id) for c in items}


def _repair_hint(*, missing: list[str], extra: list[str], evidence_err: str | None) -> str:
    parts = ["\n\n⚠ 上一次响应存在错误，请严格按照原 candidate_id 列表与字段白名单重新输出。"]
    if missing:
        parts.append(f"missing (你必须包含): {missing}")
    if extra:
        parts.append(f"extra (你不能包含): {extra}")
    if evidence_err:
        parts.append(f"key_evidence.field 错误: {evidence_err}")
    parts.append("不可遗漏、不可新增、不可改名；key_evidence.field 必须出自输入字段。")
    return "\n".join(parts)


def _complete_with_repair(
    llm: LLMClient,
    *,
    system: str,
    user: str,
    schema: type[BaseModel],
    profile: StageProfile,
    expected_ids: set[str],
    envelope_defaults: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> tuple[Any, dict[str, Any]]:
    """Call the LLM, validate, retry up to ``max_retries`` times on failure.

    Repair triggers (M3 T3.6):
      * candidate_id set mismatch (input ⊆/⊇ output)
      * key_evidence.field not in INPUT_FIELD_WHITELIST
      * Pydantic ValidationError (e.g. rank not 1..N, extra='forbid' violation,
        enum value mismatch)
    """
    current_user = user
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            raw, meta = llm.complete_json(
                system=system,
                user=current_user,
                schema=schema,
                profile=profile,
                envelope_defaults=envelope_defaults,
            )
            obj = raw if isinstance(raw, schema) else schema.model_validate(raw)
        except (LLMValidationError, ValidationError) as e:
            last_err = str(e)
            if attempt >= max_retries:
                raise
            current_user = user + _repair_hint(missing=[], extra=[], evidence_err=last_err)
            continue

        # Caller-side validators (set + whitelist)
        try:
            check_response_against_inputs(obj, expected_ids)
        except EvidenceFieldError as e:
            last_err = str(e)
            if attempt >= max_retries:
                raise
            current_user = user + _repair_hint(missing=[], extra=[], evidence_err=last_err)
            continue
        except ValueError as e:
            # candidate_id set mismatch
            actual_ids = {c.candidate_id for c in obj.candidates}
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            last_err = str(e)
            if attempt >= max_retries:
                raise _SetMismatchError(
                    f"set mismatch after {max_retries + 1} attempts; "
                    f"missing={missing}, extra={extra}"
                )
            current_user = user + _repair_hint(missing=missing, extra=extra, evidence_err=None)
            continue

        return obj, meta

    raise LLMValidationError(f"repair loop exhausted: {last_err}")


# ---------------------------------------------------------------------------
# Run analyze stage
# ---------------------------------------------------------------------------


@dataclass
class AnalyzeResult:
    success_batches: int = 0
    failed_batches: int = 0
    candidates_in: int = 0
    candidates_out: int = 0
    predictions: list[APWTrendCandidate] = field(default_factory=list)
    failed_batch_ids: list[str] = field(default_factory=list)
    batch_size: int = 0
    market_context_summaries: list[str] = field(default_factory=list)
    risk_disclaimers: list[str] = field(default_factory=list)


def default_profile() -> StageProfile:
    """Reasonable defaults for the APW走势分析 single-batch stage."""
    return StageProfile(
        thinking=True,
        reasoning_effort="medium",
        temperature=0.3,
        max_output_tokens=32_000,
    )


def run_analyze(
    *,
    llm: LLMClient,
    candidates: list[dict[str, Any]],
    trade_date: str,
    next_trade_date: str,
    market_summary: str = "",
    profile: StageProfile | None = None,
    max_batch_size: int = 20,
    max_repair_retries: int = 2,
) -> Iterable[tuple[StrategyEvent, AnalyzeResult | None]]:
    """Run all analyze batches as a generator of (event, terminal_result_or_None)."""
    profile = profile or default_profile()
    plan = plan_batches(
        n_candidates=len(candidates),
        output_budget=profile.max_output_tokens,
        max_batch_size=max_batch_size,
    )
    yield (
        StrategyEvent(
            type=EventType.STEP_STARTED,
            level=EventLevel.INFO,
            message=f"Step 3: 走势分析（待 {len(candidates)} 只，分 {plan.n_batches} 批）",
            payload={"step": 3, "n_candidates": len(candidates), "n_batches": plan.n_batches},
        ),
        None,
    )

    result = AnalyzeResult(candidates_in=len(candidates), batch_size=plan.batch_size)
    if plan.n_batches == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                level=EventLevel.INFO,
                message="Step 3: 走势分析 — 无候选",
                payload={"step": 3, "predictions": 0},
            ),
            result,
        )
        return

    for i in range(plan.n_batches):
        batch = candidates[i * plan.batch_size : (i + 1) * plan.batch_size]
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_STARTED,
                level=EventLevel.INFO,
                message=f"LLM batch {i + 1}/{plan.n_batches} ({len(batch)} 只)",
                payload={"batch_no": i + 1, "size": len(batch)},
            ),
            None,
        )

        user = apw_user_prompt(
            trade_date=trade_date,
            next_trade_date=next_trade_date,
            batch_no=i + 1,
            batch_total=plan.n_batches,
            candidates=batch,
            market_summary=market_summary,
        )
        expected_ids = _ids(batch)
        try:
            obj, meta = _complete_with_repair(
                llm,
                system=APW_SYSTEM,
                user=user,
                schema=APWTrendResponse,
                profile=profile,
                expected_ids=expected_ids,
                envelope_defaults={
                    "stage": "accumulation_probe_washout_analysis",
                    "trade_date": trade_date,
                    "next_trade_date": next_trade_date,
                    "batch_no": i + 1,
                    "batch_total": plan.n_batches,
                    "market_context_summary": "",
                    "risk_disclaimer": "",
                },
                max_retries=max_repair_retries,
            )
        except (LLMValidationError, LLMTransportError, _SetMismatchError, ValidationError) as e:
            logger.exception("走势分析 批 %d failed", i + 1)
            result.failed_batches += 1
            result.failed_batch_ids.append(str(i + 1))
            yield (
                StrategyEvent(
                    type=EventType.VALIDATION_FAILED,
                    level=EventLevel.ERROR,
                    message=f"走势分析 批 {i + 1} 失败: {e}",
                    payload={"batch_no": i + 1, "error": str(e)},
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
                level=EventLevel.INFO,
                message=f"LLM batch {i + 1}/{plan.n_batches} 完成 ({len(obj.candidates)} 条)",
                payload={
                    "batch_no": i + 1,
                    "input_tokens": meta.get("input_tokens"),
                    "output_tokens": meta.get("output_tokens"),
                    "latency_ms": meta.get("latency_ms"),
                },
            ),
            None,
        )

    yield (
        StrategyEvent(
            type=EventType.STEP_FINISHED,
            level=EventLevel.INFO,
            message=(
                f"Step 3: 走势分析完成 — 成功 {result.success_batches}/{plan.n_batches} 批，"
                f"产出 {result.candidates_out} 条"
            ),
            payload={
                "step": 3,
                "success_batches": result.success_batches,
                "failed_batches": result.failed_batches,
                "predictions": result.candidates_out,
            },
        ),
        result,
    )
