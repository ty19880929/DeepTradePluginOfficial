"""Parameter-grid backtest tests (PR-6.1).

Three test layers:

1. **Unit**: ``expand_grid`` produces the right Cartesian count + each cell
   carries the right (section, field, value) tuples.
2. **Validation**: ``load_grid_yaml`` rejects unknown sections / fields /
   scalar-not-list values.
3. **Integration**: a 3×3 = 9-cell grid runs end-to-end on the synthetic
   fixture from ``test_backtest_checkpoint``; assert distinct config_hashes,
   ranking is sorted, and the aggregate JSON lands on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checkmate import paths
from checkmate.backtest import BacktestParams
from checkmate.config import EntryConfig, RegimeConfig, RiskConfig
from checkmate.grid import (
    apply_overrides,
    expand_grid,
    load_grid_yaml,
    run_grid,
)
from checkmate.runtime import CheckmateRuntime


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


# ---------------------------------------------------------------------------
# expand_grid + apply_overrides — pure helpers
# ---------------------------------------------------------------------------


def test_expand_grid_empty_returns_one_noop_cell() -> None:
    assert expand_grid({}) == [{}]


def test_expand_grid_single_dim() -> None:
    cells = expand_grid({"entry": {"breakout_lookback": [30, 40, 50]}})
    assert len(cells) == 3
    assert {c["entry"]["breakout_lookback"] for c in cells} == {30, 40, 50}


def test_expand_grid_cartesian_product() -> None:
    grid = {
        "entry": {"atr_stop_mult": [1.5, 2.0, 2.5]},
        "risk":  {"risk_per_trade": [0.005, 0.01, 0.02]},
    }
    cells = expand_grid(grid)
    assert len(cells) == 9
    seen = {(c["entry"]["atr_stop_mult"], c["risk"]["risk_per_trade"]) for c in cells}
    assert len(seen) == 9


def test_apply_overrides_replaces_subconfig() -> None:
    base = BacktestParams(start="20240101", end="20240131", initial_cash=1e6)
    out = apply_overrides(base, {"entry": {"breakout_lookback": 30, "atr_stop_mult": 1.5}})
    assert out.entry_cfg is not None
    assert out.entry_cfg.breakout_lookback == 30
    assert out.entry_cfg.atr_stop_mult == 1.5
    # Other fields on EntryConfig keep their defaults
    assert out.entry_cfg.continuation_rs60_min == EntryConfig().continuation_rs60_min


def test_apply_overrides_does_not_mutate_base() -> None:
    base = BacktestParams(start="20240101", end="20240131",
                          entry_cfg=EntryConfig(breakout_lookback=40))
    apply_overrides(base, {"entry": {"breakout_lookback": 30}})
    assert base.entry_cfg.breakout_lookback == 40  # unchanged


# ---------------------------------------------------------------------------
# load_grid_yaml — validation
# ---------------------------------------------------------------------------


def test_load_grid_yaml_happy(tmp_path) -> None:
    p = tmp_path / "grid.yaml"
    p.write_text(
        "entry:\n"
        "  breakout_lookback: [30, 40]\n"
        "risk:\n"
        "  risk_per_trade: [0.005, 0.01]\n",
        encoding="utf-8",
    )
    g = load_grid_yaml(p)
    assert g == {
        "entry": {"breakout_lookback": [30, 40]},
        "risk": {"risk_per_trade": [0.005, 0.01]},
    }


def test_load_grid_yaml_rejects_unknown_section(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("nonsense:\n  foo: [1,2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown section"):
        load_grid_yaml(p)


def test_load_grid_yaml_rejects_unknown_field(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("entry:\n  no_such_field: [1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_field"):
        load_grid_yaml(p)


def test_load_grid_yaml_rejects_scalar_value(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("entry:\n  breakout_lookback: 40\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        load_grid_yaml(p)


def test_load_grid_yaml_empty_file_returns_empty_dict(tmp_path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_grid_yaml(p) == {}


# ---------------------------------------------------------------------------
# Integration — 3×3 grid end-to-end with the synthetic backtest fixture
# ---------------------------------------------------------------------------


# Reuse the same synthetic data shape as test_backtest_checkpoint.
SYMBOLS = [
    ("600000.SH", "银行",  8.5,   0.005),
    ("600519.SH", "白酒",  1600.0, 0.50),
    ("000001.SZ", "银行",  12.0,  0.010),
    ("002415.SZ", "电子",  30.0,  0.020),
    ("300750.SZ", "电池",  200.0, 0.10),
    ("000333.SZ", "家电",  60.0,  0.015),
    ("600276.SH", "医药",  50.0,  0.030),
    ("600887.SH", "食品",  28.0,  -0.010),
]
END_DATE = "20240329"
N_DAYS = 250
BT_START = "20240325"
BT_END = "20240329"


def _make_qfq(ts_code: str, base: float, trend: float) -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    closes = np.array([base + trend * i for i in range(N_DAYS)], dtype=float)
    noise = np.linspace(-0.005, 0.005, N_DAYS) * base
    closes = closes + noise
    highs = closes * 1.015
    lows = closes * 0.985
    pre = np.empty_like(closes)
    pre[0] = closes[0] * 0.99
    pre[1:] = closes[:-1]
    df = pd.DataFrame({
        "ts_code": [ts_code] * N_DAYS,
        "trade_date": dates,
        "open": closes, "high": highs, "low": lows,
        "close": closes, "pre_close": pre,
        "adj_factor": [1.0] * N_DAYS,
    })
    for col in ("open", "high", "low", "close", "pre_close"):
        df[f"{col}_qfq"] = df[col]
    return df


def _make_daily_basic(ts_code: str) -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    return pd.DataFrame({
        "ts_code": [ts_code] * N_DAYS,
        "trade_date": dates,
        "amount": [500_000.0] * N_DAYS,
        "turnover_rate": [2.0] * N_DAYS,
        "total_mv": [1e10] * N_DAYS,
    })


def _make_stk_limit(trade_date: str) -> pd.DataFrame:
    rows = []
    for code, _, base, _ in SYMBOLS:
        rows.append({
            "ts_code": code, "trade_date": trade_date,
            "up_limit": base * 100.0, "down_limit": base / 100.0,
        })
    return pd.DataFrame(rows)


def _plant(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _plant_index_rising(index_code: str) -> None:
    n = 200
    dates = pd.bdate_range(end=END_DATE, periods=n).strftime("%Y%m%d").tolist()
    closes = [1000.0 * (1.0 + 0.002 * i) for i in range(n)]
    df = pd.DataFrame({
        "ts_code": [index_code] * n,
        "trade_date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
    })
    _plant(paths.index_daily_cache_dir() / f"{index_code}.parquet", df)


def _trade_cal_frame() -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    return pd.DataFrame({"cal_date": dates, "is_open": [1] * N_DAYS})


class _StubTushare:
    def call(self, api_name: str, **kwargs):
        if api_name == "trade_cal":
            return _trade_cal_frame()
        return pd.DataFrame()


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()
    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())

    for code, industry, base, trend in SYMBOLS:
        _plant(paths.daily_cache_dir() / f"{code}.parquet",
               _make_qfq(code, base, trend))
        _plant(paths.daily_basic_cache_dir() / f"{code}.parquet",
               _make_daily_basic(code))
        db.execute(
            """
            INSERT INTO checkmate_stock_status_history
                (ts_code, as_of_date, list_status, is_st, name, industry,
                 list_date, delist_date, raw_event_json, updated_at)
            VALUES (?, '20100101', 'L', FALSE, ?, ?, '20100101', NULL, ?,
                    CURRENT_TIMESTAMP)
            """,
            [code, code, industry, json.dumps({"src": "test"})],
        )
    _plant_index_rising(RegimeConfig().index_csi_code)
    _plant_index_rising(RegimeConfig().index_hs300_code)
    for trade_date in pd.bdate_range(BT_START, BT_END).strftime("%Y%m%d"):
        _plant(paths.stk_limit_cache_dir() / f"{trade_date}.parquet",
               _make_stk_limit(trade_date))

    rt = CheckmateRuntime(db=db, config=None, tushare=_StubTushare())  # type: ignore[arg-type]
    yield rt
    db.close()


def test_run_grid_3x3_executes_all_cells(rt) -> None:
    """3 atr_stop_mult × 3 risk_per_trade → 9 cells; each must produce a
    distinct config_hash and the aggregate JSON must land on disk."""
    base = BacktestParams(
        start=BT_START, end=BT_END,
        initial_cash=10_000_000.0, resume=False,
    )
    grid = {
        "entry": {"atr_stop_mult": [1.5, 2.0, 2.5]},
        "risk":  {"risk_per_trade": [0.005, 0.01, 0.015]},
    }
    log: list[str] = []
    aggregate = run_grid(rt, base, grid, rank_by="final_equity", echo=log.append)

    assert aggregate["n_cells"] == 9
    assert len(aggregate["cells"]) == 9
    # Distinct config_hashes per cell
    hashes = {c["config_hash"] for c in aggregate["cells"]}
    assert len(hashes) == 9

    # Ranking — final_equity desc; first row should have ≥ second's value.
    ranked = aggregate["ranked"]
    for i in range(len(ranked) - 1):
        assert ranked[i]["final_equity"] >= ranked[i + 1]["final_equity"], (
            f"ranking broken at cell {i}: {ranked[i]['final_equity']} < "
            f"{ranked[i+1]['final_equity']}"
        )

    # Aggregate JSON written
    out_path = Path(aggregate["output_path"])
    assert out_path.is_file()
    assert out_path.suffix == ".json"
    assert out_path.parent.name == "reports"
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["n_cells"] == 9
    assert "ranked" in parsed and len(parsed["ranked"]) == 9

    # Log captured cell progress
    joined = "\n".join(log)
    assert "[grid] running 9 cell" in joined
    assert "[grid] wrote" in joined


def test_run_grid_max_drawdown_ranks_ascending(rt) -> None:
    """For max_drawdown, lower is better — ranking must be ascending."""
    base = BacktestParams(start=BT_START, end=BT_END, initial_cash=1e7, resume=False)
    grid = {"entry": {"atr_stop_mult": [1.5, 2.0]}}
    aggregate = run_grid(rt, base, grid, rank_by="max_drawdown", echo=lambda _: None)
    ranked = aggregate["ranked"]
    for i in range(len(ranked) - 1):
        assert ranked[i]["max_drawdown"] <= ranked[i + 1]["max_drawdown"]


def test_run_grid_empty_grid_runs_single_default_cell(rt) -> None:
    """No grid → 1 cell with defaults, equivalent to a single backtest."""
    base = BacktestParams(start=BT_START, end=BT_END, initial_cash=1e7, resume=False)
    aggregate = run_grid(rt, base, {}, echo=lambda _: None)
    assert aggregate["n_cells"] == 1
    assert aggregate["cells"][0]["overrides"] == {}
