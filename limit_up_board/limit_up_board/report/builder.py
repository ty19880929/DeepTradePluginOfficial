"""``build_strategy_report`` —— 把 runner 完成态的数据组装成 ``StrategyReportSchema``。

输入与 ``write_report`` 接收的参数对齐：

    bundle:          Round1Bundle              # Step 1 数据
    selected:        list[StrongCandidate]     # Step 2 强势初筛
    predictions:     list[ContinuationCandidate]  # Step 4 连板预测
    final_ranking:   FinalRankingResponse | None   # Step 4.5 多批全局重排
    failed_batch_ids: list[str] | None         # partial_failed 失败批次

设计要点：
* 仅做数据装配 + 类型转换 + i18n，不做任何 I/O、不读 DB、不读文件。
* 输出 ``StrategyReportSchema`` pydantic 实例；调用方用
  ``.model_dump_json(by_alias=True, indent=2, ...)`` 落盘。
* 对 ``bundle.market_summary`` 里 schema 未显式映射的 key 全部塞进
  根模型的 ``_extras``，防止上游加字段静默丢失。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from deeptrade.core.run_status import RunStatus

from ..data import Round1Bundle
from ..schemas import (
    ContinuationCandidate,
    FinalRankingResponse,
    StrongCandidate,
)
from .i18n import to_confidence_zh, to_strength_zh
from .schema import (
    Counts,
    DataSource,
    FilteringLog,
    FilteringThresholds,
    HistogramBucket,
    LgbCell,
    LimitStepTrend,
    MarketCandidate,
    MarketSnapshot,
    PredictionCard,
    RejectedItem,
    ReportMeta,
    ScoreDistribution,
    ScoreStats,
    ScreeningItem,
    Step4Prediction,
    StrategyReportSchema,
    YesterdayFailureRate,
    YesterdayWinnersToday,
)

# ``market_summary`` 里被 MarketSnapshot / FilteringLog 显式消费掉的 key——
# 剩下的全部进 ``_extras``。
_KNOWN_MARKET_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "limit_up_count",
        "limit_step_distribution",
        "limit_step_distribution_prev",
        "limit_step_trend",
        "yesterday_failure_rate",
        "yesterday_winners_today",
        "candidate_filter_summary",
    }
)


def build_strategy_report(
    *,
    status: RunStatus,
    bundle: Round1Bundle,
    selected: list[StrongCandidate],
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None = None,
    failed_batch_ids: list[str] | None = None,
    run_id: str | None = None,
    generated_at: datetime | None = None,
) -> StrategyReportSchema:
    """主入口。所有参数与 ``render.write_report`` 对齐。"""
    generated_at = generated_at or datetime.now(timezone.utc).astimezone()
    cand_by_id = {c.get("candidate_id"): c for c in bundle.candidates}

    return StrategyReportSchema(
        meta=_build_meta(
            status=status,
            bundle=bundle,
            n_selected=len(selected),
            n_predictions=len(predictions),
            failed_batch_ids=failed_batch_ids,
            run_id=run_id,
            generated_at=generated_at,
        ),
        marketSnapshot=_build_market_snapshot(bundle),
        scoreDistribution=_build_score_distribution(bundle),
        step2_screening=_build_screening(selected, cand_by_id),
        step4_prediction=_build_predictions(predictions, final_ranking, cand_by_id),
        filteringDetails=_build_filtering(bundle),
        extras=_build_extras(bundle),
    )


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


def _build_meta(
    *,
    status: RunStatus,
    bundle: Round1Bundle,
    n_selected: int,
    n_predictions: int,
    failed_batch_ids: list[str] | None,
    run_id: str | None,
    generated_at: datetime,
) -> ReportMeta:
    return ReportMeta(
        title="打板策略报告",
        run_id=run_id or "",
        trade_date_t=bundle.trade_date,
        trade_date_t1=bundle.next_trade_date,
        status=status.value,  # type: ignore[arg-type]
        model_version=bundle.lgb_model_id or "disabled",
        counts=Counts(
            initial=len(bundle.candidates),
            selected=n_selected,
            predicted=n_predictions,
        ),
        dataSource=DataSource(
            themeStrength=bundle.sector_strength.source,  # type: ignore[arg-type]
            unavailable=list(bundle.data_unavailable or []),
        ),
        failedBatches=list(failed_batch_ids or []),
        generatedAt=generated_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# scoreDistribution
# ---------------------------------------------------------------------------


def _build_score_distribution(bundle: Round1Bundle) -> ScoreDistribution | None:
    """LGB 未启用 / 无候选 / 全候选都没分时 → None（前端整块不渲染）。"""
    if not bundle.lgb_model_id or not bundle.candidates:
        return None
    scores: list[float] = []
    for c in bundle.candidates:
        s = c.get("lgb_score")
        if s is None:
            continue
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    arr = sorted(scores)
    n = len(arr)
    stats = ScoreStats(
        n=n,
        min=arr[0],
        p25=_quantile(arr, 0.25),
        median=_quantile(arr, 0.50),
        p75=_quantile(arr, 0.75),
        max=arr[-1],
    )
    buckets = [0] * 10
    for s in arr:
        idx = min(9, max(0, int(s // 10)))
        buckets[idx] += 1
    histogram = [
        HistogramBucket(range=f"{i * 10}-{i * 10 + 9}", count=count)
        for i, count in enumerate(buckets)
    ]
    return ScoreDistribution(stats=stats, histogram=histogram)


def _quantile(sorted_arr: list[float], q: float) -> float:
    """线性插值分位数（沿用历史 HTML 报告中的实现，保证 stats 数值不漂移）。"""
    if not sorted_arr:
        return float("nan")
    if len(sorted_arr) == 1:
        return float(sorted_arr[0])
    pos = q * (len(sorted_arr) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_arr) - 1)
    frac = pos - lo
    return sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac


# ---------------------------------------------------------------------------
# step2_screening
# ---------------------------------------------------------------------------


def _build_screening(
    selected: list[StrongCandidate],
    cand_by_id: dict[str | None, dict[str, Any]],
) -> list[ScreeningItem]:
    items: list[ScreeningItem] = []
    for i, c in enumerate(selected, start=1):
        src = cand_by_id.get(c.candidate_id, {}) or {}
        theme = src.get("industry") or src.get("lu_desc") or ""
        items.append(
            ScreeningItem(
                rank=i,
                code=c.ts_code,
                name=c.name,
                close=_to_float(src.get("close_yuan"), default=0.0),
                score=float(c.score),
                lgb=_build_lgb_cell(src),
                level=to_strength_zh(c.strength_level),  # type: ignore[arg-type]
                theme=str(theme) if theme is not None else "",
                rationale=c.rationale,
                tags=list(c.risk_flags or []),
                evidence=list(c.evidence),
                missingData=list(c.missing_data or []),
            )
        )
    return items


# ---------------------------------------------------------------------------
# step4_prediction
# ---------------------------------------------------------------------------


def _build_predictions(
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None,
    cand_by_id: dict[str | None, dict[str, Any]],
) -> Step4Prediction:
    """单批用 ``ContinuationCandidate.rank``，多批用 ``FinalRankingResponse.finalists[].final_rank``。

    分组键：``final_prediction``（多批）或 ``prediction``（单批）。
    """
    final_map: dict[str, dict[str, Any]] = {}
    if final_ranking is not None:
        for fi in final_ranking.finalists:
            final_map[fi.candidate_id] = {
                "final_rank": fi.final_rank,
                "final_prediction": fi.final_prediction,
                "final_confidence": fi.final_confidence,
                "delta_vs_batch": fi.delta_vs_batch,
                "reason_vs_peers": fi.reason_vs_peers,
            }
    multi_batch = final_ranking is not None

    groups: dict[str, list[PredictionCard]] = {
        "top_candidate": [],
        "watchlist": [],
        "avoid": [],
    }

    for p in predictions:
        fr = final_map.get(p.candidate_id)
        pred_key = (fr["final_prediction"] if fr else p.prediction) or p.prediction
        conf_en = (fr["final_confidence"] if fr else p.confidence) or p.confidence
        effective_rank = (fr["final_rank"] if fr else p.rank) if multi_batch else p.rank
        src = cand_by_id.get(p.candidate_id, {}) or {}

        card = PredictionCard(
            rank=int(effective_rank),
            code=p.ts_code,
            name=p.name,
            prediction=pred_key,  # type: ignore[arg-type]
            confidence=to_confidence_zh(conf_en),  # type: ignore[arg-type]
            score=float(p.continuation_score),
            close=_to_float(src.get("close_yuan"), default=0.0),
            lgb=_build_lgb_cell(src),
            rationale=p.rationale,
            observationPoints=list(p.next_day_watch_points or []),
            failureConditions=list(p.failure_triggers or []),
            keyEvidence=list(p.key_evidence),
            missingData=list(p.missing_data or []),
            batchLocalRank=int(p.rank) if multi_batch else None,
            deltaVsBatch=(fr["delta_vs_batch"] if multi_batch and fr else None),
            reasonVsPeers=(fr["reason_vs_peers"] if multi_batch and fr else None),
        )
        groups.setdefault(pred_key, []).append(card)

    for g in groups.values():
        g.sort(key=lambda c: c.rank)

    return Step4Prediction(
        top_candidate=groups.get("top_candidate", []),
        watchlist=groups.get("watchlist", []),
        avoid=groups.get("avoid", []),
    )


# ---------------------------------------------------------------------------
# marketSnapshot
# ---------------------------------------------------------------------------


def _build_market_snapshot(bundle: Round1Bundle) -> MarketSnapshot:
    ms = bundle.market_summary or {}
    return MarketSnapshot(
        limit_up_count=int(ms.get("limit_up_count") or 0),
        limit_step_distribution=_coerce_step_distribution(
            ms.get("limit_step_distribution")
        ),
        limit_step_distribution_prev=_coerce_step_distribution(
            ms.get("limit_step_distribution_prev")
        ),
        limit_step_trend=_build_limit_step_trend(ms.get("limit_step_trend")),
        yesterday_failure_rate=_build_yesterday_failure(
            ms.get("yesterday_failure_rate")
        ),
        yesterday_winners_today=_build_yesterday_winners(
            ms.get("yesterday_winners_today")
        ),
        candidates=[_build_market_candidate(c) for c in bundle.candidates],
    )


def _coerce_step_distribution(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in value.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _build_limit_step_trend(value: Any) -> LimitStepTrend | None:
    if not isinstance(value, dict):
        return None
    try:
        return LimitStepTrend(
            max_height=int(value.get("max_height") or 0),
            max_height_prev=int(value.get("max_height_prev") or 0),
            high_board_delta=int(value.get("high_board_delta") or 0),
            total_limit_up_delta=int(value.get("total_limit_up_delta") or 0),
            interpretation=str(value.get("interpretation") or ""),
        )
    except (TypeError, ValueError):
        return None


def _build_yesterday_failure(value: Any) -> YesterdayFailureRate | None:
    if not isinstance(value, dict):
        return None
    try:
        return YesterdayFailureRate(
            trade_date_prev=str(value.get("trade_date_prev") or ""),
            u_count=int(value.get("u_count") or 0),
            z_count=int(value.get("z_count") or 0),
            rate_pct=_to_optional_float(value.get("rate_pct")),
            interpretation=_to_optional_str(value.get("interpretation")),
        )
    except (TypeError, ValueError):
        return None


def _build_yesterday_winners(value: Any) -> YesterdayWinnersToday | None:
    if not isinstance(value, dict):
        return None
    try:
        return YesterdayWinnersToday(
            trade_date_prev=str(value.get("trade_date_prev") or ""),
            n_winners=int(value.get("n_winners") or 0),
            n_continued_today=int(value.get("n_continued_today") or 0),
            continuation_rate_pct=_to_optional_float(value.get("continuation_rate_pct")),
            n_negative_today=int(value.get("n_negative_today") or 0),
            avg_pct_chg_today=_to_optional_float(value.get("avg_pct_chg_today")),
            interpretation=_to_optional_str(value.get("interpretation")),
        )
    except (TypeError, ValueError):
        return None


def _build_market_candidate(c: dict[str, Any]) -> MarketCandidate:
    theme = c.get("industry") or c.get("lu_desc") or ""
    return MarketCandidate(
        code=str(c.get("ts_code") or ""),
        name=str(c.get("name") or ""),
        close=_to_optional_float(c.get("close_yuan")),
        float_mv=_to_optional_float(c.get("float_mv_yi")),
        theme=str(theme) if theme is not None else "",
        lgb=_build_lgb_cell(c),
    )


# ---------------------------------------------------------------------------
# filteringDetails
# ---------------------------------------------------------------------------


def _build_filtering(bundle: Round1Bundle) -> FilteringLog:
    """没有 ``candidate_filter_summary`` 时退回到一份最小占位（entered=passed=候选总数）。"""
    fs = (bundle.market_summary or {}).get("candidate_filter_summary")
    if isinstance(fs, dict):
        before = int(fs.get("before") or 0)
        after = int(fs.get("after") or 0)
        thresholds = FilteringThresholds(
            min_float_mv_yi=_to_float(fs.get("min_float_mv_yi"), default=0.0),
            max_float_mv_yi=_to_float(fs.get("max_float_mv_yi"), default=0.0),
            max_close_yuan=_to_float(fs.get("max_close_yuan"), default=0.0),
        )
        rejected = [_build_rejected(d) for d in (fs.get("dropped_top3") or [])]
        return FilteringLog(
            entered=before,
            passed=after,
            rejected=max(before - after, 0),
            thresholds=thresholds,
            rejectedItems=rejected,
        )
    n = len(bundle.candidates)
    return FilteringLog(
        entered=n,
        passed=n,
        rejected=0,
        thresholds=FilteringThresholds(
            min_float_mv_yi=0.0, max_float_mv_yi=0.0, max_close_yuan=0.0
        ),
        rejectedItems=[],
    )


def _build_rejected(d: dict[str, Any]) -> RejectedItem:
    reasons = d.get("reasons") or []
    return RejectedItem(
        code=str(d.get("ts_code") or ""),
        name=str(d.get("name") or ""),
        float_mv=_to_optional_float(d.get("float_mv_yi")),
        close=_to_optional_float(d.get("close_yuan")),
        reason=", ".join(str(r) for r in reasons),
    )


# ---------------------------------------------------------------------------
# extras （兜底未来字段）
# ---------------------------------------------------------------------------


def _build_extras(bundle: Round1Bundle) -> dict[str, Any]:
    ms = bundle.market_summary or {}
    extras: dict[str, Any] = {}
    for k, v in ms.items():
        if k in _KNOWN_MARKET_SUMMARY_KEYS:
            continue
        extras[str(k)] = v
    return extras


# ---------------------------------------------------------------------------
# 共用辅助
# ---------------------------------------------------------------------------


def _build_lgb_cell(src: dict[str, Any]) -> LgbCell:
    if not isinstance(src, dict):
        return LgbCell(score=None, rank=None)
    score = _to_optional_float(src.get("lgb_score"))
    decile = src.get("lgb_decile")
    if score is None:
        return LgbCell(score=None, rank=None)
    rank_str: str | None = None
    if decile is not None:
        try:
            rank_str = f"d{int(decile)}"
        except (TypeError, ValueError):
            rank_str = None
    return LgbCell(score=score, rank=rank_str)


def _to_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    return s if s else None
