"""LLM pipeline for limit-up-board: 强势初筛 / 连板预测 / 全局重排 / 辩论修订.

Implements:
    plan_screening_batches()      — F5 input + output token dual budget
    run_screening()               — yields events; collects StrongCandidate
    run_prediction()              — single-batch by default; auto multi-batch + 全局重排
    run_final_ranking()           — only when 连板预测 was multi-batch (M4)
    run_debate_revision()         — multi-LLM peer revision (debate mode only)
    set_equality_check()          — strict candidate_id ⊆ ⊇ check
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
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
from .profiles import (
    STAGE_FINAL,
    STAGE_PREDICTION,
    STAGE_REVISION,
    STAGE_SCREENING,
    resolve_profile,
)
from .prompts import (
    FINAL_RANKING_SYSTEM,
    REVISION_SYSTEM,
    build_prediction_system,
    build_screening_system,
    final_ranking_user_prompt,
    prediction_user_prompt,
    revision_user_prompt,
    screening_user_prompt,
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
DEFAULT_SCREENING_INPUT_BUDGET = 80_000
DEFAULT_PREDICTION_INPUT_BUDGET = 200_000
SAFETY_RATIO = 0.85


@dataclass
class BatchPlan:
    batch_size: int
    n_batches: int


def plan_llm_batches(
    *,
    n_candidates: int,
    input_budget: int = DEFAULT_SCREENING_INPUT_BUDGET,
    output_budget: int,  # = stage profile's max_output_tokens (强势初筛 default 32k)
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
    """Outcome of a single 强势初筛 / 连板预测 / 全局重排 phase."""

    success_batches: int = 0
    failed_batches: int = 0
    candidates_in: int = 0
    candidates_out: int = 0
    selected: list[StrongCandidate] = field(default_factory=list)
    predictions: list[ContinuationCandidate] = field(default_factory=list)
    final_items: list[Any] = field(default_factory=list)
    # F-L3 — concrete failed batch ordinals (1-based, e.g. ``["3", "5"]``)
    # for report banner. The phase prefix (初筛/预测) is prepended by the runner
    # so the same RoundResult can be reused across stages without leaking the
    # stage tag into the persisted ids.
    failed_batch_ids: list[str] = field(default_factory=list)
    # F-M3 — record actual batch_size so finalists selection isn't hardcoded
    batch_size: int = 0


# ---------------------------------------------------------------------------
# F-H1 — set-mismatch repair retry
# ---------------------------------------------------------------------------


class _SetMismatchError(Exception):
    """Raised when LLM output's candidate_id set still doesn't equal the
    expected set after one repair retry."""


class _EvidenceValidationError(Exception):
    """Raised when LLM output's evidence references (e.g. ``evidence.field``)
    still violate the input-key whitelist after one repair retry. The
    ``errors`` attribute carries human-readable strings for the
    VALIDATION_FAILED event payload.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"evidence validation failed after retry; {len(errors)} error(s): "
            f"{errors[:3]}{'...' if len(errors) > 3 else ''}"
        )


# P0-2 v1 — fields the LLM is allowed to put in ``evidence.field`` /
# ``key_evidence.field``. Anything else is a fabrication risk and triggers a
# repair retry. ``risk_flags`` / ``next_day_watch_points`` 等自由文本字段不进
# 白名单（plan T-14 边界条款）—— 它们由 schema 单独承载。


# 额外允许引用的"派生 / 上下文"字段名（不出现在 candidate dict 但 prompt 文档
# 化承诺存在）：sector_strength.* 、market_summary.* 子字段、limit_step 全局摘要等。
# 这些放在白名单里避免误报；具体值由 prompt 上下文承载，validator 只检 key 名。
_EVIDENCE_CONTEXT_FIELDS_WHITELIST = frozenset(
    {
        "sector_strength",
        "sector_strength_source",
        "market_context",
        "market_summary",
        "limit_step",
        "limit_step_trend",
        "yesterday_failure_rate",
        "yesterday_winners_today",
        "limit_up_count",
        "data_unavailable",
        # 评分相关 派生字段
        "lgb_score",
        "lgb_decile",
    }
)


def _candidate_allowed_field_set(row: dict[str, Any]) -> set[str]:
    """Build the per-candidate allow-list = (dict keys) ∪ context allow-list.

    Picks up every key present in the candidate row (including derived /
    optional ones like ``cyq_winner_pct``, ``lgb_score`` when scoring ran),
    plus a small whitelist of prompt-documented context fields.
    """
    return set(row.keys()) | set(_EVIDENCE_CONTEXT_FIELDS_WHITELIST)


def _make_evidence_field_validator(
    allowed_fields_per_candidate: dict[str, set[str]],
    *,
    evidence_attr: str,
) -> Callable[[Any], list[str]]:
    """Build a validator closure that inspects ``obj.candidates[i].<evidence_attr>``.

    Returns a function: ``obj -> list[str]`` (empty list = pass). Each error
    string is of the form ``"{cid}: evidence.field={f!r} not in input keys"``.
    Unknown candidate_ids are silently skipped — the set-mismatch check
    catches those earlier.
    """

    def _validate(obj: Any) -> list[str]:
        errors: list[str] = []
        candidates = getattr(obj, "candidates", None) or []
        for c in candidates:
            cid = getattr(c, "candidate_id", None)
            allowed = allowed_fields_per_candidate.get(cid)
            if allowed is None:
                continue
            ev_items = getattr(c, evidence_attr, None) or []
            for ev in ev_items:
                field_name = getattr(ev, "field", None)
                if field_name is None:
                    continue
                if field_name not in allowed:
                    errors.append(
                        f"{cid}: {evidence_attr}.field={field_name!r} not in input keys"
                    )
        return errors

    return _validate


def _evidence_validation_repair_hint(errors: list[str]) -> str:
    """Corrective instruction appended to user prompt when evidence references
    fabricated fields. Lists up to 6 offending fields so the LLM can self-correct."""
    head = errors[:6]
    tail = "" if len(errors) <= 6 else f"\n（共 {len(errors)} 条，仅列前 6 条）"
    return (
        "\n\n⚠ 上一次响应中的 evidence 引用了输入数据中不存在的字段名，已被自动拒绝。"
        "请只引用候选股 JSON 中实际出现的键名（如 fd_amount_yi、turnover_ratio、"
        "lgb_score 等）；禁止编造或猜测字段。\n违规字段：\n" + "\n".join(head) + tail
    )


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
    evidence_validator: Callable[[Any], list[str]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Call LLM and verify the output's id-set matches. On mismatch, retry
    once with a corrective hint appended to the user prompt.

    Args:
        profile: caller-resolved StageProfile (see profiles.py).
        output_attr: name of the BaseModel field that holds the list of items
            (each having a ``.candidate_id`` attr). 'candidates' for 初筛/预测,
            'finalists' for final_ranking.
        envelope_defaults: caller-controlled top-level fields (e.g. ``stage``,
            ``trade_date``, ``batch_no``) injected when the LLM omits them.
        evidence_validator: P0-2 v1 — optional callable returning a list of
            human-readable error strings (empty = pass). Runs **after** the
            set-equality check passes; on non-empty errors it follows the
            same repair-retry contract as set-mismatch. On final attempt
            with errors → raise :class:`_EvidenceValidationError`.

    On retry, the meta dict carries ``evidence_validation_errors_first_attempt``
    (the first-attempt errors that the LLM self-corrected); callers can fold
    that into ``lub_stage_results.evidence_validation_errors_json`` if they
    want to record near-misses.
    """
    current_user = user
    last_actual: set[str] = set()
    first_attempt_evidence_errors: list[str] = []
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
        if expected_ids != actual_ids:
            last_actual = actual_ids
            if attempt < repair_retries:
                current_user = user + _set_mismatch_repair_hint(expected_ids, actual_ids)
                continue
            raise _SetMismatchError(
                f"set mismatch after {repair_retries + 1} attempts; "
                f"missing={sorted(expected_ids - last_actual)}, "
                f"extra={sorted(last_actual - expected_ids)}"
            )
        # set OK; now apply evidence-field whitelist if caller wired one in.
        if evidence_validator is None:
            return obj, meta
        ev_errors = evidence_validator(obj)
        if not ev_errors:
            # Stamp recovery info into meta when this run took retries.
            meta = {**meta, "evidence_validation_errors_first_attempt": first_attempt_evidence_errors}
            return obj, meta
        # Validator triggered; retry once with a corrective hint, then bail.
        if attempt == 0:
            first_attempt_evidence_errors = list(ev_errors)
        if attempt < repair_retries:
            current_user = user + _evidence_validation_repair_hint(ev_errors)
            continue
        raise _EvidenceValidationError(ev_errors)
    # Should be unreachable given the loop above always returns or raises;
    # added for type-checker satisfaction.
    raise _SetMismatchError("unreachable")


# ---------------------------------------------------------------------------
# 强势初筛
# ---------------------------------------------------------------------------


def run_screening(
    *,
    llm: LLMClient,
    bundle: Round1Bundle,
    preset: str,
    input_budget: int = DEFAULT_SCREENING_INPUT_BUDGET,
    lgb_min_score_floor: float | None = 30.0,
) -> Iterable[tuple[StrategyEvent, RoundResult | None]]:
    """Run all 强势初筛 batches, yielding (event, terminal_result_or_None).

    The caller (strategy.run) re-yields the events into the runner. The final
    iteration yields a result alongside the STEP_FINISHED event so the caller
    can hand it on to 连板预测.
    """
    profile = resolve_profile(preset, STAGE_SCREENING)
    candidates = bundle.candidates
    plan = plan_llm_batches(
        n_candidates=len(candidates),
        input_budget=input_budget,
        output_budget=profile.max_output_tokens,
    )
    screening_system = build_screening_system(lgb_min_score_floor=lgb_min_score_floor)
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
            message="Step 2: 强势初筛",
            payload={"n_candidates": len(candidates), "n_batches": plan.n_batches},
        ),
        None,
    )

    result = RoundResult(candidates_in=len(candidates), batch_size=plan.batch_size)
    if plan.n_batches == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 2: 强势初筛",
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
                message=f"初筛 批 {i + 1}/{plan.n_batches}",
                payload={"batch_no": i + 1, "size": len(batch)},
            ),
            None,
        )

        user = screening_user_prompt(
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
        # P0-2 v1 — evidence field whitelist (per candidate). 见
        # _make_evidence_field_validator 注释。
        allowed_fields_screening = {
            row.get("candidate_id"): _candidate_allowed_field_set(row)
            for row in batch
            if row.get("candidate_id")
        }
        ev_validator = _make_evidence_field_validator(
            allowed_fields_screening, evidence_attr="evidence"
        )
        try:
            obj, meta = _complete_with_set_check(
                llm,
                system=screening_system,
                user=user,
                schema=StrongAnalysisResponse,
                profile=profile,
                expected_ids=expected_ids,
                envelope_defaults={
                    "stage": STAGE_SCREENING,
                    "trade_date": bundle.trade_date,
                    "batch_no": i + 1,
                    "batch_total": plan.n_batches,
                    "batch_summary": "",
                },
                evidence_validator=ev_validator,
            )
        except (
            LLMValidationError,
            LLMTransportError,
            _SetMismatchError,
            _EvidenceValidationError,
        ) as e:
            result.failed_batches += 1
            result.failed_batch_ids.append(str(i + 1))
            payload: dict[str, Any] = {"batch_no": i + 1}
            if isinstance(e, _EvidenceValidationError):
                payload["evidence_validation_errors"] = e.errors
            yield (
                StrategyEvent(
                    type=EventType.VALIDATION_FAILED,
                    level=EventLevel.ERROR,
                    message=f"初筛 批 {i + 1} 失败: {e}",
                    payload=payload,
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
                message=f"初筛 批 {i + 1}/{plan.n_batches} 完成",
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
            message="Step 2: 强势初筛",
            payload={
                "success_batches": result.success_batches,
                "failed_batches": result.failed_batches,
                "selected": len(result.selected),
            },
        ),
        result,
    )


# ---------------------------------------------------------------------------
# 连板预测 + (optional) 全局重排
# ---------------------------------------------------------------------------


def run_prediction(
    *,
    llm: LLMClient,
    selected: list[StrongCandidate],
    bundle: Round1Bundle,
    preset: str,
    input_budget: int = DEFAULT_PREDICTION_INPUT_BUDGET,
    lgb_min_score_floor: float | None = 30.0,
) -> Iterable[tuple[StrategyEvent, RoundResult | None]]:
    """Run 连板预测; multi-batch + 全局重排 if the candidate set exceeds the budget."""
    profile = resolve_profile(preset, STAGE_PREDICTION)
    plan = plan_llm_batches(
        n_candidates=len(selected),
        input_budget=input_budget,
        output_budget=profile.max_output_tokens,
    )
    prediction_system = build_prediction_system(lgb_min_score_floor=lgb_min_score_floor)
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
            message="Step 4: 连板预测",
            payload={"n_candidates": len(selected), "n_batches": plan.n_batches},
        ),
        None,
    )

    result = RoundResult(candidates_in=len(selected), batch_size=plan.batch_size)
    if plan.n_batches == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 4: 连板预测",
                payload={"predictions": 0},
            ),
            result,
        )
        return

    # Build candidate dicts for the prompt (初筛 selected → minimal payload)
    payload_rows: list[dict[str, Any]] = [
        _prediction_row_from_selected(c, bundle.candidates) for c in selected
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
                message=f"预测 批 {i + 1}/{plan.n_batches}",
                payload={"batch_no": i + 1, "size": len(batch_objs)},
            ),
            None,
        )

        user = prediction_user_prompt(
            trade_date=bundle.trade_date,
            next_trade_date=bundle.next_trade_date,
            candidates=batch_rows,
            market_context=bundle.market_summary,
            sector_strength_source=bundle.sector_strength.source,
            sector_strength_data=bundle.sector_strength.data,
            data_unavailable=bundle.data_unavailable,
        )

        expected_ids = _ids(batch_objs)
        # P0-2 v1 — evidence field whitelist for 连板预测 (key_evidence on
        # ContinuationCandidate). 用 prompt 实际看到的 payload_rows 作为允许集。
        allowed_fields_prediction = {
            row.get("candidate_id"): _candidate_allowed_field_set(row)
            for row in batch_rows
            if row.get("candidate_id")
        }
        ev_validator = _make_evidence_field_validator(
            allowed_fields_prediction, evidence_attr="key_evidence"
        )
        try:
            obj, meta = _complete_with_set_check(
                llm,
                system=prediction_system,
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
                evidence_validator=ev_validator,
            )
        except (
            LLMValidationError,
            LLMTransportError,
            _SetMismatchError,
            _EvidenceValidationError,
        ) as e:
            result.failed_batches += 1
            result.failed_batch_ids.append(str(i + 1))
            payload: dict[str, Any] = {"batch_no": i + 1}
            if isinstance(e, _EvidenceValidationError):
                payload["evidence_validation_errors"] = e.errors
            yield (
                StrategyEvent(
                    type=EventType.VALIDATION_FAILED,
                    level=EventLevel.ERROR,
                    message=f"预测 批 {i + 1} 失败: {e}",
                    payload=payload,
                ),
                None,
            )
            continue

        result.success_batches += 1
        result.predictions.extend(obj.candidates)
        yield (
            StrategyEvent(
                type=EventType.LLM_BATCH_FINISHED,
                message=f"预测 批 {i + 1}/{plan.n_batches} 完成",
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
            message="Step 4: 连板预测",
            payload={
                "success_batches": result.success_batches,
                "failed_batches": result.failed_batches,
                "predictions": len(result.predictions),
            },
        ),
        result,
    )


# ---------------------------------------------------------------------------
# 全局重排 (only triggered when 连板预测 was multi-batch)
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
            message="Step 4.5: 全局重排（多批合并）",
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
                message="Step 4.5: 全局重排（多批合并）",
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
            message="Step 4.5: 全局重排（多批合并）",
            payload={"success": True, "finalists": len(obj.finalists)},
        ),
        obj,
    )


# ---------------------------------------------------------------------------
# Internal payload shapers
# ---------------------------------------------------------------------------


def _prediction_row_from_selected(
    sc: StrongCandidate, all_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a minimal 连板预测 input row from a 强势初筛 selected verdict + raw fields.

    We re-attach the normalized fields so the LLM doesn't have to re-derive them.
    Dict keys ``r1_*`` are kept verbatim — they appear in the prompt payload the
    LLM sees, and renaming them would silently change prompt semantics.
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
# 辩论修订 — each LLM revises its own 连板预测 after seeing peers
# ---------------------------------------------------------------------------


@dataclass
class DebateRoundResult:
    """Outcome of a single LLM's 辩论修订 phase."""

    success: bool = False
    error: str | None = None
    candidates_in: int = 0
    revision_summary: str = ""
    revised: list[RevisedContinuationCandidate] = field(default_factory=list)


def run_debate_revision(
    *,
    llm: LLMClient,
    bundle: Round1Bundle,
    own_predictions: list[ContinuationCandidate],
    peers: list[tuple[str, list[ContinuationCandidate]]],
    preset: str,
    self_label: str = "you",
) -> Iterable[tuple[StrategyEvent, DebateRoundResult | None]]:
    """Single-batch 辩论修订; yields events + a terminal result.

    The revising LLM receives its own ``own_predictions`` (full view) plus a
    trimmed view of every entry in ``peers`` (already anonymised). The output
    candidate_id set is enforced to equal the input ``own_predictions`` set.
    """
    profile = resolve_profile(preset, STAGE_REVISION)
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
            message="Step 4.7: 辩论修订",
            payload={"n_candidates": n, "n_peers": len(peers), "self_label": self_label},
        ),
        None,
    )

    result = DebateRoundResult(candidates_in=n)
    if n == 0:
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 4.7: 辩论修订",
                payload={"success": False, "reason": "empty own_predictions"},
            ),
            result,
        )
        return

    user = revision_user_prompt(
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
            message="辩论修订（单批）",
            payload={"size": n},
        ),
        None,
    )
    # P0-2 v1 — 辩论修订阶段 evidence validator 用 own_predictions 中每只
    # candidate 在 bundle.candidates 里对应的 dict keys 作为允许字段集。
    # prompt 给 LLM 看的是 own_predictions（精简版），但 key_evidence 仍可
    # 引用原始 bundle.candidates 中的字段名，故以 bundle 字典 keys 为准。
    bundle_lookup = {c["candidate_id"]: c for c in bundle.candidates if "candidate_id" in c}
    allowed_fields_revision = {
        c.candidate_id: _candidate_allowed_field_set(bundle_lookup.get(c.candidate_id, {}))
        for c in own_predictions
    }
    ev_validator = _make_evidence_field_validator(
        allowed_fields_revision, evidence_attr="key_evidence"
    )
    try:
        obj, meta = _complete_with_set_check(
            llm,
            system=REVISION_SYSTEM,
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
            evidence_validator=ev_validator,
        )
    except (
        LLMValidationError,
        LLMTransportError,
        _SetMismatchError,
        _EvidenceValidationError,
    ) as e:
        result.error = f"{type(e).__name__}: {e}"
        fail_payload: dict[str, Any] = {}
        if isinstance(e, _EvidenceValidationError):
            fail_payload["evidence_validation_errors"] = e.errors
        yield (
            StrategyEvent(
                type=EventType.VALIDATION_FAILED,
                level=EventLevel.ERROR,
                message=f"辩论修订 失败: {e}",
                payload=fail_payload,
            ),
            None,
        )
        yield (
            StrategyEvent(
                type=EventType.STEP_FINISHED,
                message="Step 4.7: 辩论修订",
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
            message="辩论修订 完成",
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
            message="Step 4.7: 辩论修订",
            payload={"success": True, "revised": len(result.revised)},
        ),
        result,
    )


# Suppress unused-import warning for fields we re-export indirectly via tests
_ = SectorStrength
