"""v0.4 — settings (流通市值 / 股价 上限) and the candidate market-filter helper.

Pure-function / direct-DB tests; no tushare or LLM in scope.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from deeptrade.core.db import Database
from limit_up_board.config import (
    LubConfig,
    list_for_show,
    load_config,
    save_config,
)
from limit_up_board.data import (
    _apply_market_filter,
)

# ---------------------------------------------------------------------------
# _apply_market_filter — pure pandas, no DB
# ---------------------------------------------------------------------------


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestApplyMarketFilter:
    def test_default_thresholds_keep_only_qualifying(self) -> None:
        # v0.6.4 P2-1：闭区间 (>=, <=, <=)。100 亿 / 15 元的标的现在通过。
        df = _df(
            [
                {"ts_code": "A", "float_mv": 50e8, "close": 10.0},   # keep
                {"ts_code": "B", "float_mv": 99e8, "close": 14.99},  # keep (just under)
                {"ts_code": "C", "float_mv": 100e8, "close": 10.0},  # keep (= 100亿 边界，闭区间)
                {"ts_code": "D", "float_mv": 50e8, "close": 15.0},   # keep (= 15元 边界，闭区间)
                {"ts_code": "E", "float_mv": 200e8, "close": 30.0},  # drop (both over)
            ]
        )
        out, summary = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0
        )
        assert list(out["ts_code"]) == ["A", "B", "C", "D"]
        # 子集断言：只看核心字段，避免 P2-1 新增 `dropped_top3` 字段影响其它用例
        assert summary["before"] == 5
        assert summary["after"] == 4
        assert summary["min_float_mv_yi"] == 0.0
        assert summary["max_float_mv_yi"] == 100.0
        assert summary["max_close_yuan"] == 15.0

    def test_min_float_mv_drops_too_small(self) -> None:
        # v0.6.4 P2-1：闭区间下，30 亿（== min）现在保留，25 亿（< min）剔除。
        df = _df(
            [
                {"ts_code": "A", "float_mv": 25e8, "close": 10.0},   # drop (< 30亿)
                {"ts_code": "B", "float_mv": 30e8, "close": 10.0},   # keep (= 30亿 边界，闭区间)
                {"ts_code": "C", "float_mv": 35e8, "close": 10.0},   # keep
                {"ts_code": "D", "float_mv": 99e8, "close": 10.0},   # keep
                {"ts_code": "E", "float_mv": 120e8, "close": 10.0},  # drop (> 100亿)
            ]
        )
        out, summary = _apply_market_filter(
            df,
            max_float_mv_yi=100.0,
            max_close_yuan=15.0,
            min_float_mv_yi=30.0,
        )
        assert list(out["ts_code"]) == ["B", "C", "D"]
        assert summary["min_float_mv_yi"] == 30.0
        assert summary["before"] == 5
        assert summary["after"] == 3

    def test_null_fields_filtered(self) -> None:
        df = _df(
            [
                {"ts_code": "A", "float_mv": 50e8, "close": 10.0},     # keep
                {"ts_code": "B", "float_mv": None, "close": 10.0},     # drop (null mv)
                {"ts_code": "C", "float_mv": 50e8, "close": None},     # drop (null close)
                {"ts_code": "D", "float_mv": float("nan"), "close": 10.0},  # drop (NaN)
            ]
        )
        out, summary = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0
        )
        assert list(out["ts_code"]) == ["A"]
        assert summary["before"] == 4
        assert summary["after"] == 1

    def test_custom_thresholds(self) -> None:
        df = _df(
            [
                {"ts_code": "A", "float_mv": 20e8, "close": 5.0},   # keep
                {"ts_code": "B", "float_mv": 35e8, "close": 5.0},   # drop (mv > 30 上限)
                {"ts_code": "C", "float_mv": 20e8, "close": 9.0},   # drop (close > 8 上限)
            ]
        )
        out, summary = _apply_market_filter(
            df, max_float_mv_yi=30.0, max_close_yuan=8.0
        )
        assert list(out["ts_code"]) == ["A"]
        assert summary["max_float_mv_yi"] == 30.0
        assert summary["max_close_yuan"] == 8.0

    def test_empty_input_returns_empty(self) -> None:
        df = _df([])
        out, summary = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0
        )
        assert out.empty
        # P2-1 后 summary 多了一个空 `dropped_top3` 字段；用 subset 校验更稳健。
        assert summary["before"] == 0
        assert summary["after"] == 0
        assert summary["min_float_mv_yi"] == 0.0
        assert summary["max_float_mv_yi"] == 100.0
        assert summary["max_close_yuan"] == 15.0
        assert summary["dropped_top3"] == []


# ---------------------------------------------------------------------------
# LubConfig persistence — uses lub_config table
# ---------------------------------------------------------------------------


@pytest.fixture
def lub_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.duckdb")
    # apply only the table this test cares about (avoid the whole plugin migration chain)
    db.execute(
        "CREATE TABLE lub_config ("
        "key VARCHAR PRIMARY KEY, value_json VARCHAR NOT NULL, "
        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    return db


class TestLubConfigPersistence:
    def test_load_returns_defaults_on_empty_table(self, lub_db: Database) -> None:
        cfg = load_config(lub_db)
        assert cfg.min_float_mv_yi == 30.0
        assert cfg.max_float_mv_yi == 100.0
        assert cfg.max_close_yuan == 15.0

    def test_save_then_load_round_trip(self, lub_db: Database) -> None:
        save_config(lub_db, LubConfig(max_float_mv_yi=80.0, max_close_yuan=12.5))
        cfg = load_config(lub_db)
        assert cfg.max_float_mv_yi == 80.0
        assert cfg.max_close_yuan == 12.5

    def test_save_overwrites_previous(self, lub_db: Database) -> None:
        save_config(lub_db, LubConfig(max_float_mv_yi=80.0, max_close_yuan=12.5))
        save_config(lub_db, LubConfig(max_float_mv_yi=50.0, max_close_yuan=8.0))
        cfg = load_config(lub_db)
        assert cfg.max_float_mv_yi == 50.0
        assert cfg.max_close_yuan == 8.0

    def test_list_for_show_marks_default_vs_persisted(self, lub_db: Database) -> None:
        rows = list_for_show(lub_db)
        sources = {key: source for key, _, source in rows}
        assert sources["lub.min_float_mv_yi"] == "default"
        assert sources["lub.max_float_mv_yi"] == "default"
        assert sources["lub.max_close_yuan"] == "default"
        values = {key: value for key, value, _ in rows}
        assert values["lub.min_float_mv_yi"] == 30.0

        save_config(
            lub_db,
            LubConfig(min_float_mv_yi=20.0, max_float_mv_yi=70.0, max_close_yuan=12.0),
        )
        rows = list_for_show(lub_db)
        sources = {key: source for key, _, source in rows}
        assert sources["lub.min_float_mv_yi"] == "persisted"
        assert sources["lub.max_float_mv_yi"] == "persisted"
        assert sources["lub.max_close_yuan"] == "persisted"
        values = {key: value for key, value, _ in rows}
        assert values["lub.min_float_mv_yi"] == 20.0
        assert values["lub.max_float_mv_yi"] == 70.0
        assert values["lub.max_close_yuan"] == 12.0
