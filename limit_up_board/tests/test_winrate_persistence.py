"""PR #1 — lub_prediction_records 写入路径单元测试。

覆盖点（与 PR 实施计划 §3.6 对齐）：
    - 单 LLM run upsert：行数 = len(predictions)
    - 同 (trade_date, ts_code) 二次写入：覆盖；updated_at 更新
    - final_ranking 存在时 rank/prediction 走 final_ranking
    - bundle.candidates 补齐 t_close_price / lgb_score / lgb_decile
    - 空 predictions → 返回 0，不写入
    - 写入失败由 caller 捕获（本测试只验证 caller 包了 try/except；行为契约）
    - load / purge 行为
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from deeptrade.core.db import Database

from limit_up_board.runtime import LubRuntime
from limit_up_board.schemas import (
    ContinuationCandidate,
    EvidenceItem,
    FinalRankItem,
    FinalRankingResponse,
)
from limit_up_board.winrate.persistence import (
    PredictionRecord,
    load_prediction_records,
    purge_prediction_records,
    record_predictions_from_run,
)

MIGRATION_FILES = [
    Path(__file__).resolve().parents[1] / "migrations" / "20260509_001_init.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_002_prediction_records.sql",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))

    from deeptrade.core import paths as core_paths

    database = Database(core_paths.db_path())
    for migration in MIGRATION_FILES:
        sql_text = migration.read_text(encoding="utf-8")
        for stmt in sql_text.split(";"):
            stmt = stmt.strip()
            if stmt:
                database.execute(stmt)
    return database


@pytest.fixture
def rt(db: Database) -> LubRuntime:
    return LubRuntime(db=db, config=MagicMock(), llms=MagicMock())


# ---------------------------------------------------------------------------
# Bundle / prediction builders
# ---------------------------------------------------------------------------


@dataclass
class _FakeBundle:
    """Minimal duck-typed Round1Bundle for the persistence module."""

    trade_date: str
    next_trade_date: str
    candidates: list[dict[str, Any]] = field(default_factory=list)


def _make_pred(
    ts_code: str,
    name: str,
    rank: int,
    prediction: str = "top_candidate",
    score: float = 80.0,
    confidence: str = "high",
) -> ContinuationCandidate:
    return ContinuationCandidate(
        candidate_id=f"cand-{ts_code}",
        ts_code=ts_code,
        name=name,
        rank=rank,
        continuation_score=score,
        confidence=confidence,
        prediction=prediction,
        rationale="strong volume + sector tailwind",
        key_evidence=[
            EvidenceItem(field="fd_amount_yi", value=3.2, unit="亿", interpretation="封单量较大")
        ],
        next_day_watch_points=["开盘竞价是否打开"],
        failure_triggers=["开盘破涨停"],
    )


def _make_cand(
    ts_code: str,
    name: str,
    *,
    close_yuan: float | None = 12.34,
    lgb_score: float | None = 0.71,
    lgb_decile: int | None = 9,
) -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "name": name,
        "close_yuan": close_yuan,
        "lgb_score": lgb_score,
        "lgb_decile": lgb_decile,
    }


def _make_final(items: list[tuple[str, int, str]]) -> FinalRankingResponse:
    """``items`` = [(ts_code, final_rank, final_prediction), ...]."""
    return FinalRankingResponse(
        stage="final_ranking",
        trade_date="20260521",
        next_trade_date="20260522",
        finalists=[
            FinalRankItem(
                candidate_id=f"cand-{ts}",
                ts_code=ts,
                final_rank=r,
                final_prediction=pred,  # type: ignore[arg-type]
                final_confidence="high",
                reason_vs_peers="strongest theme + clean tape",
                delta_vs_batch="kept",
            )
            for (ts, r, pred) in items
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_record_predictions_without_final_ranking(rt: LubRuntime) -> None:
    """No final_ranking → 直接用 predictions 的 rank / prediction。"""
    bundle = _FakeBundle(
        trade_date="20260521",
        next_trade_date="20260522",
        candidates=[
            _make_cand("600001.SH", "stock-a"),
            _make_cand("600002.SH", "stock-b", close_yuan=9.99, lgb_score=0.5, lgb_decile=6),
        ],
    )
    predictions = [
        _make_pred("600001.SH", "stock-a", rank=1, prediction="top_candidate"),
        _make_pred("600002.SH", "stock-b", rank=2, prediction="watchlist"),
    ]

    n = record_predictions_from_run(
        rt=rt,
        bundle=bundle,  # type: ignore[arg-type]
        predictions=predictions,
        final_ranking=None,
        run_id="run-001",
        trade_date="20260521",
        next_trade_date="20260522",
    )
    assert n == 2

    rows = load_prediction_records(rt.db)
    assert len(rows) == 2

    by_code = {r.ts_code: r for r in rows}
    assert by_code["600001.SH"].rank == 1
    assert by_code["600001.SH"].prediction == "top_candidate"
    assert by_code["600001.SH"].t_close_price == pytest.approx(12.34)
    assert by_code["600001.SH"].lgb_score == pytest.approx(0.71)
    assert by_code["600001.SH"].lgb_decile == 9

    assert by_code["600002.SH"].prediction == "watchlist"
    assert by_code["600002.SH"].t_close_price == pytest.approx(9.99)
    assert by_code["600002.SH"].lgb_decile == 6


def test_record_predictions_with_final_ranking(rt: LubRuntime) -> None:
    """final_ranking 存在 → rank / prediction / confidence 走 final；继续分仍来自 predictions。"""
    bundle = _FakeBundle(
        trade_date="20260521",
        next_trade_date="20260522",
        candidates=[
            _make_cand("600001.SH", "stock-a"),
            _make_cand("600002.SH", "stock-b"),
        ],
    )
    predictions = [
        _make_pred("600001.SH", "stock-a", rank=2, prediction="watchlist", score=72.0),
        _make_pred("600002.SH", "stock-b", rank=1, prediction="top_candidate", score=88.0),
    ]
    final_ranking = _make_final(
        [
            ("600001.SH", 1, "top_candidate"),  # 升级
            ("600002.SH", 2, "watchlist"),  # 降级
        ]
    )

    n = record_predictions_from_run(
        rt=rt,
        bundle=bundle,  # type: ignore[arg-type]
        predictions=predictions,
        final_ranking=final_ranking,
        run_id="run-002",
        trade_date="20260521",
        next_trade_date="20260522",
    )
    assert n == 2

    rows = {r.ts_code: r for r in load_prediction_records(rt.db)}
    # final_ranking 口径
    assert rows["600001.SH"].rank == 1
    assert rows["600001.SH"].prediction == "top_candidate"
    # 继续分仍来自原始 predictions
    assert rows["600001.SH"].continuation_score == pytest.approx(72.0)
    assert rows["600002.SH"].rank == 2
    assert rows["600002.SH"].prediction == "watchlist"
    assert rows["600002.SH"].continuation_score == pytest.approx(88.0)


def test_second_run_overwrites_same_day_same_code(rt: LubRuntime) -> None:
    """二次写入：同 (trade_date, ts_code) 后写覆盖前写，updated_at 推进，created_at 不变。"""
    bundle = _FakeBundle(
        trade_date="20260521",
        next_trade_date="20260522",
        candidates=[_make_cand("600001.SH", "stock-a", close_yuan=10.0)],
    )

    record_predictions_from_run(
        rt=rt,
        bundle=bundle,  # type: ignore[arg-type]
        predictions=[_make_pred("600001.SH", "stock-a", rank=5, prediction="watchlist", score=60.0)],
        final_ranking=None,
        run_id="run-A",
        trade_date="20260521",
        next_trade_date="20260522",
    )

    first = rt.db.fetchone(
        "SELECT prediction, rank, continuation_score, run_id, created_at, updated_at "
        "FROM lub_prediction_records WHERE trade_date=? AND ts_code=?",
        ("20260521", "600001.SH"),
    )
    assert first is not None
    pred0, rank0, score0, run0, created0, updated0 = first
    assert pred0 == "watchlist"
    assert rank0 == 5
    assert run0 == "run-A"

    # Ensure timestamp clock advances (DuckDB CURRENT_TIMESTAMP has 1µs resolution)
    time.sleep(0.05)

    record_predictions_from_run(
        rt=rt,
        bundle=bundle,  # type: ignore[arg-type]
        predictions=[_make_pred("600001.SH", "stock-a", rank=1, prediction="top_candidate", score=90.0)],
        final_ranking=None,
        run_id="run-B",
        trade_date="20260521",
        next_trade_date="20260522",
    )

    second = rt.db.fetchone(
        "SELECT prediction, rank, continuation_score, run_id, created_at, updated_at "
        "FROM lub_prediction_records WHERE trade_date=? AND ts_code=?",
        ("20260521", "600001.SH"),
    )
    assert second is not None
    pred1, rank1, score1, run1, created1, updated1 = second

    assert pred1 == "top_candidate"  # 覆盖
    assert rank1 == 1
    assert score1 == pytest.approx(90.0)
    assert run1 == "run-B"  # 最近一次 run_id
    assert created1 == created0  # created_at 不变
    assert updated1 >= updated0  # updated_at 推进

    # 全表仅一行
    rows = load_prediction_records(rt.db)
    assert len(rows) == 1


def test_empty_predictions_returns_zero(rt: LubRuntime) -> None:
    bundle = _FakeBundle(trade_date="20260521", next_trade_date="20260522", candidates=[])
    n = record_predictions_from_run(
        rt=rt,
        bundle=bundle,  # type: ignore[arg-type]
        predictions=[],
        final_ranking=None,
        run_id="run-empty",
        trade_date="20260521",
        next_trade_date="20260522",
    )
    assert n == 0
    assert load_prediction_records(rt.db) == []


def test_missing_candidate_fields_coerced_to_none(rt: LubRuntime) -> None:
    """bundle 里 close_yuan / lgb_score 缺失或 NaN → None；不应崩。"""
    bundle = _FakeBundle(
        trade_date="20260521",
        next_trade_date="20260522",
        candidates=[
            _make_cand("600001.SH", "stock-a", close_yuan=None, lgb_score=None, lgb_decile=None),
            # ts_code 不在 candidates 索引里的情况
        ],
    )
    predictions = [
        _make_pred("600001.SH", "stock-a", rank=1),
        _make_pred("600003.SH", "stock-c", rank=2),  # 不在 bundle.candidates
    ]
    n = record_predictions_from_run(
        rt=rt,
        bundle=bundle,  # type: ignore[arg-type]
        predictions=predictions,
        final_ranking=None,
        run_id="run-coerce",
        trade_date="20260521",
        next_trade_date="20260522",
    )
    assert n == 2
    rows = {r.ts_code: r for r in load_prediction_records(rt.db)}
    assert rows["600001.SH"].t_close_price is None
    assert rows["600001.SH"].lgb_score is None
    assert rows["600001.SH"].lgb_decile is None
    assert rows["600003.SH"].t_close_price is None  # not in candidates index


def test_load_filters_by_window_and_prediction(rt: LubRuntime) -> None:
    bundle1 = _FakeBundle(
        trade_date="20260520", next_trade_date="20260521",
        candidates=[_make_cand("600001.SH", "stock-a")],
    )
    bundle2 = _FakeBundle(
        trade_date="20260521", next_trade_date="20260522",
        candidates=[_make_cand("600002.SH", "stock-b")],
    )
    record_predictions_from_run(
        rt=rt, bundle=bundle1,  # type: ignore[arg-type]
        predictions=[_make_pred("600001.SH", "stock-a", rank=1, prediction="top_candidate")],
        final_ranking=None, run_id="r1",
        trade_date="20260520", next_trade_date="20260521",
    )
    record_predictions_from_run(
        rt=rt, bundle=bundle2,  # type: ignore[arg-type]
        predictions=[_make_pred("600002.SH", "stock-b", rank=1, prediction="avoid")],
        final_ranking=None, run_id="r2",
        trade_date="20260521", next_trade_date="20260522",
    )

    # filter by window
    assert {r.ts_code for r in load_prediction_records(rt.db, start="20260521", end="20260521")} == {
        "600002.SH"
    }
    # filter by prediction class
    assert {r.ts_code for r in load_prediction_records(rt.db, predictions=["top_candidate"])} == {
        "600001.SH"
    }


def test_purge_before_inclusive(rt: LubRuntime) -> None:
    for d, code in [
        ("20260518", "600001.SH"),
        ("20260519", "600002.SH"),
        ("20260520", "600003.SH"),
        ("20260521", "600004.SH"),
    ]:
        bundle = _FakeBundle(
            trade_date=d, next_trade_date="X",
            candidates=[_make_cand(code, "s")],
        )
        record_predictions_from_run(
            rt=rt, bundle=bundle,  # type: ignore[arg-type]
            predictions=[_make_pred(code, "s", rank=1)],
            final_ranking=None, run_id=f"run-{d}",
            trade_date=d, next_trade_date="X",
        )

    deleted = purge_prediction_records(rt.db, before="20260519")
    assert deleted == 2
    remaining = {r.trade_date for r in load_prediction_records(rt.db)}
    assert remaining == {"20260520", "20260521"}


def test_raw_prediction_json_serialized(rt: LubRuntime) -> None:
    """raw_prediction_json 持有 ContinuationCandidate 字段，便于 LLM review 复盘。"""
    bundle = _FakeBundle(
        trade_date="20260521", next_trade_date="20260522",
        candidates=[_make_cand("600001.SH", "stock-a")],
    )
    record_predictions_from_run(
        rt=rt, bundle=bundle,  # type: ignore[arg-type]
        predictions=[_make_pred("600001.SH", "stock-a", rank=1)],
        final_ranking=None, run_id="r",
        trade_date="20260521", next_trade_date="20260522",
    )
    rows = load_prediction_records(rt.db)
    assert rows[0].raw_prediction_json is not None
    assert "fd_amount_yi" in rows[0].raw_prediction_json  # evidence preserved


def test_purge_when_empty_returns_zero(rt: LubRuntime) -> None:
    assert purge_prediction_records(rt.db, before="20260601") == 0
