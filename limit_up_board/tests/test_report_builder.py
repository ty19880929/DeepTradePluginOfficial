"""单元测试：``limit_up_board.report.builder.build_strategy_report``。

覆盖目标（与 ``frontend_migration_notes.md`` 的对外契约一一对应）：

* meta：title / run_id / trade_date / status / model_version / counts / dataSource / failedBatches
* scoreDistribution：LGB 启用 → stats + 10 桶 histogram；LGB disabled / 无候选 → None
* step2_screening：英文 strength_level → 中文 level；risk_flags → tags；evidence / missingData 完整透传
* step4_prediction：三档分组；多批模式 batchLocalRank / deltaVsBatch / reasonVsPeers 注入；单批为 None
* marketSnapshot：bundle.market_summary 各子结构 + bundle.candidates 投影
* filteringDetails：candidate_filter_summary 字段平移 + dropped_top3 → rejectedItems
* extras：未知 market_summary key 兜底
* partial_failed：failedBatches 暴露失败批次
* pydantic 严格校验：未知字段会被 forbid
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from deeptrade.core.run_status import RunStatus

from limit_up_board.data import Round1Bundle, SectorStrength
from limit_up_board.report import build_strategy_report
from limit_up_board.report.schema import StrategyReportSchema
from limit_up_board.schemas import (
    ContinuationCandidate,
    EvidenceItem,
    EvidenceItemStrict,
    FinalRankItem,
    FinalRankingResponse,
    StrongCandidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bundle(*, with_lgb: bool = True, with_filter: bool = True) -> Round1Bundle:
    candidates = [
        {
            "candidate_id": "600519.SH",
            "ts_code": "600519.SH",
            "name": "茅台",
            "close_yuan": 12.5,
            "float_mv_yi": 45.0,
            "lgb_score": 73.0 if with_lgb else None,
            "lgb_decile": 8 if with_lgb else None,
            "industry": "电子",
        },
        {
            "candidate_id": "000001.SZ",
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "close_yuan": 10.0,
            "float_mv_yi": 80.0,
            "lgb_score": 21.0 if with_lgb else None,
            "lgb_decile": 2 if with_lgb else None,
            "industry": "金融",
        },
    ]
    market_summary: dict = {
        "limit_up_count": 42,
        "limit_step_distribution": {"1": 30, "2": 8, "3": 4},
        "limit_step_distribution_prev": {"1": 25, "2": 5},
        "limit_step_trend": {
            "max_height": 3,
            "max_height_prev": 2,
            "high_board_delta": 1,
            "total_limit_up_delta": 12,
            "interpretation": "spectrum_lifting",
        },
        "yesterday_failure_rate": {
            "trade_date_prev": "20260529",
            "u_count": 25,
            "z_count": 5,
            "rate_pct": 16.67,
            "interpretation": "moderate",
        },
        "yesterday_winners_today": {
            "trade_date_prev": "20260529",
            "n_winners": 25,
            "n_continued_today": 8,
            "continuation_rate_pct": 32.0,
            "n_negative_today": 4,
            "avg_pct_chg_today": 1.5,
            "interpretation": "neutral",
        },
        # 未知字段 → 应进 _extras
        "future_field_not_in_schema": {"experimental": True},
    }
    if with_filter:
        market_summary["candidate_filter_summary"] = {
            "before": 5,
            "after": 2,
            "min_float_mv_yi": 30.0,
            "max_float_mv_yi": 100.0,
            "max_close_yuan": 15.0,
            "dropped_top3": [
                {
                    "ts_code": "600000.SH",
                    "name": "浦发银行",
                    "float_mv_yi": 180.0,
                    "close_yuan": 8.5,
                    "reasons": ["float_mv>100.0"],
                },
                {
                    "ts_code": "600183.SH",
                    "name": "生益科技",
                    "float_mv_yi": 2586.30,
                    "close_yuan": 108.01,
                    "reasons": ["float_mv>100.0", "close>15.0"],
                },
            ],
        }
    return Round1Bundle(
        trade_date="20260530",
        next_trade_date="20260531",
        candidates=candidates,
        market_summary=market_summary,
        sector_strength=SectorStrength(
            source="limit_cpt_list", data={"top_sectors": []}
        ),
        data_unavailable=["cyq_perf_empty_response"],
        lgb_model_id="20260530_1_demo" if with_lgb else None,
        lgb_predictions=[],
    )


def _make_selected() -> list[StrongCandidate]:
    return [
        StrongCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name="茅台",
            selected=True,
            score=80.0,
            strength_level="high",
            rationale="封板早、量价配合好。",
            evidence=[
                EvidenceItem(
                    field="fd_amount_yi",
                    value=1.2,
                    unit="亿",
                    interpretation="封单 1.2 亿，强势封板",
                )
            ],
            risk_flags=["尾盘板", "振幅过大"],
            missing_data=["lhb"],
        )
    ]


def _make_predictions() -> list[ContinuationCandidate]:
    return [
        ContinuationCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name="茅台",
            rank=1,
            continuation_score=78.0,
            confidence="high",
            prediction="top_candidate",
            rationale="情绪强、模型分高、上量明确。",
            key_evidence=[
                EvidenceItem(
                    field="lgb_score",
                    value=73.0,
                    unit="none",
                    interpretation="分位 8 / 模型分 73",
                )
            ],
            next_day_watch_points=["盘口跟进", "板块联动"],
            failure_triggers=["开盘跌停"],
            missing_data=[],
        ),
        ContinuationCandidate(
            candidate_id="000001.SZ",
            ts_code="000001.SZ",
            name="平安银行",
            rank=2,
            continuation_score=42.0,
            confidence="low",
            prediction="avoid",
            rationale="模型分低、缺乏题材联动。",
            key_evidence=[
                EvidenceItem(
                    field="lgb_score",
                    value=21.0,
                    unit="none",
                    interpretation="模型分 21，分位 2",
                )
            ],
            next_day_watch_points=["不建议跟进"],
            failure_triggers=["开盘溢价"],
            missing_data=[],
        ),
    ]


def _make_final_ranking() -> FinalRankingResponse:
    return FinalRankingResponse(
        stage="final_ranking",
        trade_date="20260530",
        next_trade_date="20260531",
        finalists=[
            FinalRankItem(
                candidate_id="600519.SH",
                ts_code="600519.SH",
                final_rank=1,
                final_prediction="top_candidate",
                final_confidence="high",
                reason_vs_peers="跨批比对：模型分领先、题材独立性强。",
                delta_vs_batch="upgraded",
            ),
            FinalRankItem(
                candidate_id="000001.SZ",
                ts_code="000001.SZ",
                final_rank=2,
                final_prediction="avoid",
                final_confidence="low",
                reason_vs_peers="跨批比对：分数最低且无溢价空间。",
                delta_vs_batch="kept",
            ),
        ],
    )


def _build(
    *,
    status: RunStatus = RunStatus.SUCCESS,
    bundle: Round1Bundle | None = None,
    selected: list[StrongCandidate] | None = None,
    predictions: list[ContinuationCandidate] | None = None,
    final_ranking: FinalRankingResponse | None = None,
    failed_batch_ids: list[str] | None = None,
    run_id: str | None = "run-test",
) -> StrategyReportSchema:
    return build_strategy_report(
        status=status,
        bundle=bundle or _make_bundle(),
        selected=selected if selected is not None else _make_selected(),
        predictions=predictions if predictions is not None else _make_predictions(),
        final_ranking=final_ranking,
        failed_batch_ids=failed_batch_ids,
        run_id=run_id,
        generated_at=datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


def test_meta_basics() -> None:
    rpt = _build()
    assert rpt.meta.title == "打板策略报告"
    assert rpt.meta.run_id == "run-test"
    assert rpt.meta.trade_date_t == "20260530"
    assert rpt.meta.trade_date_t1 == "20260531"
    assert rpt.meta.status == "success"
    assert rpt.meta.model_version == "20260530_1_demo"
    assert rpt.meta.counts.initial == 2
    assert rpt.meta.counts.selected == 1
    assert rpt.meta.counts.predicted == 2
    assert rpt.meta.dataSource.themeStrength == "limit_cpt_list"
    assert rpt.meta.dataSource.unavailable == ["cyq_perf_empty_response"]
    assert rpt.meta.failedBatches == []
    assert "2026-05-30" in rpt.meta.generatedAt


def test_meta_model_version_disabled_when_lgb_off() -> None:
    rpt = _build(bundle=_make_bundle(with_lgb=False))
    assert rpt.meta.model_version == "disabled"


def test_meta_partial_failed_surfaces_failed_batches() -> None:
    rpt = _build(
        status=RunStatus.PARTIAL_FAILED,
        failed_batch_ids=["初筛#3", "预测#1"],
    )
    assert rpt.meta.status == "partial_failed"
    assert rpt.meta.failedBatches == ["初筛#3", "预测#1"]


# ---------------------------------------------------------------------------
# scoreDistribution
# ---------------------------------------------------------------------------


def test_score_distribution_with_lgb() -> None:
    rpt = _build()
    sd = rpt.scoreDistribution
    assert sd is not None
    assert sd.stats.n == 2
    assert sd.stats.min == 21.0
    assert sd.stats.max == 73.0
    # 10 桶覆盖 0-99
    assert len(sd.histogram) == 10
    ranges = [b.range for b in sd.histogram]
    assert ranges[0] == "0-9"
    assert ranges[2] == "20-29"
    assert ranges[7] == "70-79"
    assert ranges[9] == "90-99"
    # 21 → 桶 2，73 → 桶 7
    counts_by_range = {b.range: b.count for b in sd.histogram}
    assert counts_by_range["20-29"] == 1
    assert counts_by_range["70-79"] == 1
    assert sum(b.count for b in sd.histogram) == 2


def test_score_distribution_none_when_lgb_disabled() -> None:
    rpt = _build(bundle=_make_bundle(with_lgb=False))
    assert rpt.scoreDistribution is None


def test_score_distribution_none_when_no_candidates() -> None:
    bundle = _make_bundle()
    bundle.candidates = []
    rpt = _build(bundle=bundle, selected=[], predictions=[])
    assert rpt.scoreDistribution is None


# ---------------------------------------------------------------------------
# step2_screening
# ---------------------------------------------------------------------------


def test_screening_i18n_and_field_mapping() -> None:
    rpt = _build()
    assert len(rpt.step2_screening) == 1
    item = rpt.step2_screening[0]
    assert item.rank == 1
    assert item.code == "600519.SH"
    assert item.name == "茅台"
    assert item.close == 12.5
    assert item.score == 80.0
    assert item.lgb.score == 73.0
    assert item.lgb.rank == "d8"
    assert item.level == "强"  # high → 强
    assert item.theme == "电子"
    assert item.rationale == "封板早、量价配合好。"
    assert item.tags == ["尾盘板", "振幅过大"]  # risk_flags 映射
    assert item.missingData == ["lhb"]
    assert len(item.evidence) == 1
    assert item.evidence[0].field == "fd_amount_yi"


def test_screening_empty_when_no_selected() -> None:
    rpt = _build(selected=[])
    assert rpt.step2_screening == []


# ---------------------------------------------------------------------------
# step4_prediction
# ---------------------------------------------------------------------------


def test_predictions_single_batch_three_groups() -> None:
    rpt = _build()
    s4 = rpt.step4_prediction
    assert [c.code for c in s4.top_candidate] == ["600519.SH"]
    assert s4.watchlist == []
    assert [c.code for c in s4.avoid] == ["000001.SZ"]
    # 单批：batchLocalRank/deltaVsBatch/reasonVsPeers 必须为 None
    top = s4.top_candidate[0]
    assert top.batchLocalRank is None
    assert top.deltaVsBatch is None
    assert top.reasonVsPeers is None
    assert top.confidence == "高"
    assert top.prediction == "top_candidate"
    # rank 为 ContinuationCandidate.rank
    assert top.rank == 1


def test_predictions_multi_batch_uses_final_ranking() -> None:
    rpt = _build(final_ranking=_make_final_ranking())
    s4 = rpt.step4_prediction
    top = s4.top_candidate[0]
    avoid = s4.avoid[0]
    # rank 来自 final_rank（这里恰好和 p.rank 相同；用 deltaVsBatch 验证 final 注入路径）
    assert top.batchLocalRank == 1  # p.rank
    assert top.deltaVsBatch == "upgraded"
    assert top.reasonVsPeers == "跨批比对：模型分领先、题材独立性强。"
    assert avoid.batchLocalRank == 2
    assert avoid.deltaVsBatch == "kept"


def test_prediction_card_collects_observation_and_failure() -> None:
    rpt = _build()
    top = rpt.step4_prediction.top_candidate[0]
    assert top.observationPoints == ["盘口跟进", "板块联动"]
    assert top.failureConditions == ["开盘跌停"]
    assert top.keyEvidence[0].field == "lgb_score"


# ---------------------------------------------------------------------------
# marketSnapshot
# ---------------------------------------------------------------------------


def test_market_snapshot_fields() -> None:
    rpt = _build()
    ms = rpt.marketSnapshot
    assert ms.limit_up_count == 42
    assert ms.limit_step_distribution == {"1": 30, "2": 8, "3": 4}
    assert ms.limit_step_distribution_prev == {"1": 25, "2": 5}
    assert ms.limit_step_trend is not None
    assert ms.limit_step_trend.high_board_delta == 1
    assert ms.limit_step_trend.interpretation == "spectrum_lifting"
    assert ms.yesterday_failure_rate is not None
    assert ms.yesterday_failure_rate.rate_pct == 16.67
    assert ms.yesterday_winners_today is not None
    assert ms.yesterday_winners_today.continuation_rate_pct == 32.0
    # candidates 投影
    codes = [c.code for c in ms.candidates]
    assert codes == ["600519.SH", "000001.SZ"]
    assert ms.candidates[0].theme == "电子"
    assert ms.candidates[0].lgb.score == 73.0
    assert ms.candidates[0].lgb.rank == "d8"


def test_market_snapshot_lgb_cell_null_when_lgb_off() -> None:
    rpt = _build(bundle=_make_bundle(with_lgb=False))
    for c in rpt.marketSnapshot.candidates:
        assert c.lgb.score is None
        assert c.lgb.rank is None


# ---------------------------------------------------------------------------
# filteringDetails
# ---------------------------------------------------------------------------


def test_filtering_details_full() -> None:
    rpt = _build()
    fd = rpt.filteringDetails
    assert fd.entered == 5
    assert fd.passed == 2
    assert fd.rejected == 3
    assert fd.thresholds.min_float_mv_yi == 30.0
    assert fd.thresholds.max_float_mv_yi == 100.0
    assert fd.thresholds.max_close_yuan == 15.0
    assert len(fd.rejectedItems) == 2
    first = fd.rejectedItems[0]
    assert first.code == "600000.SH"
    assert first.reason == "float_mv>100.0"


def test_filtering_details_fallback_when_summary_absent() -> None:
    bundle = _make_bundle(with_filter=False)
    rpt = _build(bundle=bundle)
    fd = rpt.filteringDetails
    assert fd.entered == 2
    assert fd.passed == 2
    assert fd.rejected == 0
    assert fd.rejectedItems == []


# ---------------------------------------------------------------------------
# _extras （兜底未来字段）
# ---------------------------------------------------------------------------


def test_extras_collects_unknown_market_summary_keys() -> None:
    rpt = _build()
    assert "future_field_not_in_schema" in rpt.extras
    assert rpt.extras["future_field_not_in_schema"] == {"experimental": True}
    # 已知 key 不应被重复装进 _extras
    for known in (
        "limit_up_count",
        "limit_step_distribution",
        "limit_step_trend",
        "yesterday_failure_rate",
        "yesterday_winners_today",
        "candidate_filter_summary",
    ):
        assert known not in rpt.extras


# ---------------------------------------------------------------------------
# 序列化：JSON 输出契约（前端直接消费的形态）
# ---------------------------------------------------------------------------


def test_model_dump_json_round_trip_uses_extras_alias() -> None:
    rpt = _build()
    raw = rpt.model_dump_json(by_alias=True)
    data = json.loads(raw)
    # 顶层结构齐全
    assert set(data.keys()) >= {
        "meta",
        "marketSnapshot",
        "scoreDistribution",
        "step2_screening",
        "step4_prediction",
        "filteringDetails",
        "_extras",
    }
    # 三档分组在 JSON 中存在
    assert set(data["step4_prediction"].keys()) == {
        "top_candidate",
        "watchlist",
        "avoid",
    }
    # 反序列化也走 alias，确保前后端约定一致
    StrategyReportSchema.model_validate(data)


# ---------------------------------------------------------------------------
# pydantic 严格校验
# ---------------------------------------------------------------------------


def test_schema_forbids_unknown_fields_at_root() -> None:
    rpt = _build()
    data = rpt.model_dump(by_alias=True)
    data["unexpected_key"] = "boom"
    with pytest.raises(ValidationError):
        StrategyReportSchema.model_validate(data)


def test_schema_forbids_unknown_fields_inside_prediction_card() -> None:
    rpt = _build()
    data = rpt.model_dump(by_alias=True)
    data["step4_prediction"]["top_candidate"][0]["mystery"] = "no"
    with pytest.raises(ValidationError):
        StrategyReportSchema.model_validate(data)


# ---------------------------------------------------------------------------
# v0.16.1 回归：runner 真实喂给 builder 的 evidence 是 EvidenceItemStrict
# 实例（LLM strict 反序列化产物）。EvidenceItem 是 EvidenceItemStrict 的子类，
# IS-A 关系只在「子→父」方向 OK；report.schema 之前误把字段标成 list[EvidenceItem]
# （父类槽位），pydantic v2 拒绝接收 EvidenceItemStrict 实例，导致 summary.json
# 静默不落盘。这两个 case 直接用 strict 实例构造 fixture，覆盖生产路径。
# ---------------------------------------------------------------------------


def test_screening_accepts_evidence_item_strict() -> None:
    selected = [
        StrongCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name="茅台",
            selected=True,
            score=80.0,
            strength_level="high",
            rationale="封板早、量价配合好。",
            evidence=[
                EvidenceItemStrict(
                    field="fd_amount_yi",
                    value=1.2,
                    unit="亿",
                    interpretation="封单 1.2 亿，强势封板",
                )
            ],
            risk_flags=[],
            missing_data=[],
        )
    ]
    rpt = build_strategy_report(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(),
        selected=selected,
        predictions=[],
        final_ranking=None,
    )
    assert rpt.step2_screening[0].evidence[0].field == "fd_amount_yi"


def test_prediction_accepts_evidence_item_strict() -> None:
    predictions = [
        ContinuationCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name="茅台",
            rank=1,
            continuation_score=78.0,
            confidence="high",
            prediction="top_candidate",
            rationale="情绪强。",
            key_evidence=[
                EvidenceItemStrict(
                    field="lgb_score",
                    value=73.0,
                    unit="none",
                    interpretation="模型分 73",
                )
            ],
            next_day_watch_points=["盘口"],
            failure_triggers=["跌停"],
            missing_data=[],
        )
    ]
    rpt = build_strategy_report(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(),
        selected=[],
        predictions=predictions,
        final_ranking=None,
    )
    assert rpt.step4_prediction.top_candidate[0].keyEvidence[0].field == "lgb_score"
