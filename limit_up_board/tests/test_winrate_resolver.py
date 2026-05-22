"""PR #2 — resolver: T+1 行情解析 + outcome 判定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from deeptrade.core.db import Database

from limit_up_board.winrate.persistence import PredictionRecord
from limit_up_board.winrate.resolver import (
    classify_outcome,
    resolve_records,
)

MIGRATION_FILES = [
    Path(__file__).resolve().parents[1] / "migrations" / "20260509_001_init.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_002_prediction_records.sql",
]


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))

    from deeptrade.core import paths as core_paths

    database = Database(core_paths.db_path())
    for mig in MIGRATION_FILES:
        sql_text = mig.read_text(encoding="utf-8")
        for stmt in sql_text.split(";"):
            stmt = stmt.strip()
            if stmt:
                database.execute(stmt)
    return database


def _seed_lub_daily(db: Database, ts_code: str, trade_date: str, open_price: float) -> None:
    db.execute(
        "INSERT INTO lub_daily (ts_code, trade_date, open, high, low, close, pre_close, "
        "change, pct_chg, vol, amount) VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
        (ts_code, trade_date, open_price),
    )


def _make_rec(
    ts_code: str = "600001.SH",
    *,
    trade_date: str = "20260521",
    next_trade_date: str = "20260522",
    t_close_price: float | None = 10.0,
    prediction: str = "top_candidate",
    rank: int = 1,
    name: str = "stock-a",
) -> PredictionRecord:
    return PredictionRecord(
        trade_date=trade_date,
        next_trade_date=next_trade_date,
        ts_code=ts_code,
        name=name,
        run_id="r1",
        prediction=prediction,
        rank=rank,
        continuation_score=80.0,
        confidence="high",
        t_close_price=t_close_price,
        lgb_score=0.7,
        lgb_decile=9,
        raw_prediction_json=None,
    )


# ---------------------------------------------------------------------------
# classify_outcome
# ---------------------------------------------------------------------------


def test_classify_win() -> None:
    assert classify_outcome(10.5, 10.0) == "win"


def test_classify_flat() -> None:
    assert classify_outcome(10.0, 10.0) == "flat"


def test_classify_loss() -> None:
    assert classify_outcome(9.5, 10.0) == "loss"


def test_classify_unresolved_missing_t1() -> None:
    assert classify_outcome(None, 10.0) == "unresolved"


def test_classify_unresolved_missing_close() -> None:
    assert classify_outcome(10.0, None) == "unresolved"


# ---------------------------------------------------------------------------
# resolve_records: local hit / tushare fallback / unresolved
# ---------------------------------------------------------------------------


def test_resolve_local_hit(db: Database) -> None:
    """lub_daily 命中 → outcome / pct 计算正确，不查 tushare。"""
    _seed_lub_daily(db, "600001.SH", "20260522", open_price=11.0)
    tushare = MagicMock()

    [out] = resolve_records([_make_rec()], db=db, tushare=tushare, force_sync=True)
    assert out.outcome == "win"
    assert out.t1_open_price == pytest.approx(11.0)
    assert out.open_vs_limit_pct == pytest.approx(10.0)
    tushare.call.assert_not_called()


def test_resolve_unresolved_local_only(db: Database) -> None:
    """lub_daily miss + tushare 未传 → unresolved。"""
    [out] = resolve_records([_make_rec()], db=db, tushare=None, force_sync=False)
    assert out.outcome == "unresolved"
    assert out.t1_open_price is None
    assert out.open_vs_limit_pct is None


def test_resolve_tushare_fallback(db: Database) -> None:
    """lub_daily miss + force_sync + tushare 可用 → tushare 回源。"""
    tushare = MagicMock()
    tushare.call.return_value = pd.DataFrame(
        {
            "ts_code": ["600001.SH"],
            "trade_date": ["20260522"],
            "open": [10.5],
            "close": [10.8],
        }
    )

    [out] = resolve_records([_make_rec()], db=db, tushare=tushare, force_sync=True)
    assert out.outcome == "win"
    assert out.t1_open_price == pytest.approx(10.5)
    tushare.call.assert_called_once()
    kwargs = tushare.call.call_args.kwargs
    assert kwargs["params"]["ts_code"] == "600001.SH"
    assert kwargs["params"]["start_date"] == "20260522"
    assert kwargs["params"]["end_date"] == "20260522"
    assert kwargs["force_sync"] is True


def test_resolve_tushare_no_force_sync(db: Database) -> None:
    """force_sync=False → 即使传了 tushare 也不调用。"""
    tushare = MagicMock()
    [out] = resolve_records([_make_rec()], db=db, tushare=tushare, force_sync=False)
    assert out.outcome == "unresolved"
    tushare.call.assert_not_called()


def test_resolve_tushare_returns_empty(db: Database) -> None:
    """T+1 行情仍未发布（tushare 返回空 df） → unresolved。"""
    tushare = MagicMock()
    tushare.call.return_value = pd.DataFrame()
    [out] = resolve_records([_make_rec()], db=db, tushare=tushare, force_sync=True)
    assert out.outcome == "unresolved"


def test_resolve_t_close_missing(db: Database) -> None:
    """t_close_price 缺失 → 即使 T+1 有也 unresolved。"""
    _seed_lub_daily(db, "600001.SH", "20260522", open_price=11.0)
    [out] = resolve_records([_make_rec(t_close_price=None)], db=db, tushare=None)
    assert out.outcome == "unresolved"
    assert out.t1_open_price == pytest.approx(11.0)
    assert out.open_vs_limit_pct is None


def test_resolve_tushare_exception_degrades(db: Database) -> None:
    """tushare 调用抛异常 → 静默降级为 unresolved，不影响其它行。"""
    _seed_lub_daily(db, "600002.SH", "20260522", open_price=12.0)
    tushare = MagicMock()
    tushare.call.side_effect = RuntimeError("network down")
    rec_fail = _make_rec("600001.SH")
    rec_ok = _make_rec("600002.SH", t_close_price=10.0, name="b")
    results = resolve_records([rec_fail, rec_ok], db=db, tushare=tushare, force_sync=True)
    assert results[0].outcome == "unresolved"
    assert results[1].outcome == "win"
