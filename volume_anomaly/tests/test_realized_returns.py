"""volume-anomaly v0.4.0 — T+N realized-return helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from volume_anomaly.calendar import (
    TradeCalendar,
)
from volume_anomaly.data import (
    EVALUATE_HORIZONS,
    _classify_data_status,
    _compute_realized_returns,
    _resolve_horizon_dates,
)


# ---------------------------------------------------------------------------
# Calendar fixture (Mon-Fri only; covers a few weeks)
# ---------------------------------------------------------------------------


def _make_cal() -> TradeCalendar:
    """A full June 2026 calendar with weekends closed."""
    rows = []
    cur = pd.Timestamp("2026-06-01")
    for _ in range(60):  # 60 days from 2026-06-01
        cal_date = cur.strftime("%Y%m%d")
        is_open = 1 if cur.dayofweek < 5 else 0
        rows.append({"cal_date": cal_date, "is_open": is_open})
        cur += pd.Timedelta(days=1)
    df = pd.DataFrame(rows)
    return TradeCalendar(df)


# ---------------------------------------------------------------------------
# _resolve_horizon_dates
# ---------------------------------------------------------------------------


class TestResolveHorizonDates:
    def test_horizons_skip_weekends(self) -> None:
        cal = _make_cal()
        # 2026-06-01 is Monday; T+1 = 2026-06-02 (Tue), T+5 = 2026-06-08 (Mon, skipping weekend)
        out = _resolve_horizon_dates(cal, "20260601")
        assert out[1] == "20260602"
        assert out[3] == "20260604"
        assert out[5] == "20260608"
        assert out[10] == "20260615"

    def test_horizons_partial_when_friday_anomaly(self) -> None:
        cal = _make_cal()
        # 2026-06-05 is Friday; T+1 = 2026-06-08 (skips weekend)
        out = _resolve_horizon_dates(cal, "20260605")
        assert out[1] == "20260608"
        assert out[3] == "20260610"

    def test_returns_dict_for_default_horizons(self) -> None:
        cal = _make_cal()
        out = _resolve_horizon_dates(cal, "20260601")
        assert set(out.keys()) == set(EVALUATE_HORIZONS)


# ---------------------------------------------------------------------------
# _compute_realized_returns
# ---------------------------------------------------------------------------


class TestComputeRealizedReturns:
    def test_full_complete(self) -> None:
        # T close = 100, T+1 = 105, T+3 = 110, T+5 = 108, T+10 = 95
        # Window 5d closes (T+1..T+5): 105, 102, 110, 108, 108
        # Window 10d closes (T+1..T+10): 105, 102, 110, 108, 108, 100, 100, 100, 99, 95
        out = _compute_realized_returns(
            t_close=100.0,
            horizon_closes={1: 105.0, 3: 110.0, 5: 108.0, 10: 95.0},
            window_5d_closes=[105, 102, 110, 108, 108],
            window_10d_closes=[105, 102, 110, 108, 108, 100, 100, 100, 99, 95],
        )
        assert out["ret_t1"] == 5.0
        assert out["ret_t3"] == 10.0
        assert out["ret_t5"] == 8.0
        assert out["ret_t10"] == -5.0
        assert out["max_close_5d"] == 110.0
        assert out["max_ret_5d"] == 10.0
        assert out["max_close_10d"] == 110.0
        assert out["max_ret_10d"] == 10.0
        # max_dd_5d: min(window_5d) = 102 → (102/100 - 1)*100 = 2.0
        # G2: positive in this case because the lowest 5d close is still above T close
        assert out["max_dd_5d"] == 2.0

    def test_t1_missing_returns_none(self) -> None:
        out = _compute_realized_returns(
            t_close=100.0,
            horizon_closes={1: None, 3: 110.0, 5: 108.0, 10: 95.0},
            window_5d_closes=[None, 102, 110, 108, 108],
            window_10d_closes=[None, 102, 110, 108, 108, 100, 100, 100, 99, 95],
        )
        assert out["ret_t1"] is None
        # max_close uses only non-None entries
        assert out["max_close_5d"] == 110.0

    def test_t_close_zero_yields_none(self) -> None:
        out = _compute_realized_returns(
            t_close=0.0,
            horizon_closes={1: 105.0, 3: 110.0, 5: 108.0, 10: 95.0},
            window_5d_closes=[105, 102, 110, 108, 108],
            window_10d_closes=[],
        )
        assert all(out[k] is None for k in ["ret_t1", "ret_t3", "ret_t5", "ret_t10"])

    def test_max_dd_negative_when_drop_below_t(self) -> None:
        out = _compute_realized_returns(
            t_close=100.0,
            horizon_closes={1: 95.0, 3: 90.0, 5: 92.0, 10: 95.0},
            window_5d_closes=[95, 90, 92, 92, 92],
            window_10d_closes=[95, 90, 92, 92, 92, 95, 95, 95, 95, 95],
        )
        # min over window5 = 90 → (90/100 - 1)*100 = -10
        assert out["max_dd_5d"] == -10.0


# ---------------------------------------------------------------------------
# _classify_data_status (G5 — strict three-state)
# ---------------------------------------------------------------------------


class TestClassifyDataStatus:
    def test_pending_when_t1_in_future(self) -> None:
        # today=20260601, T+1=20260602 → still future
        status = _classify_data_status(
            horizon_closes={1: None, 3: None, 5: None, 10: None},
            horizons=EVALUATE_HORIZONS,
            today="20260601",
            horizon_dates={1: "20260602", 3: "20260604", 5: "20260608", 10: "20260615"},
        )
        assert status == "pending"

    def test_partial_when_max_horizon_in_future(self) -> None:
        # today=20260610, T+10=20260615 → still future
        status = _classify_data_status(
            horizon_closes={1: 105.0, 3: 110.0, 5: 108.0, 10: None},
            horizons=EVALUATE_HORIZONS,
            today="20260610",
            horizon_dates={1: "20260602", 3: "20260604", 5: "20260608", 10: "20260615"},
        )
        assert status == "partial"

    def test_partial_when_horizon_close_missing(self) -> None:
        # All horizons in past, but T+5 missing close (suspended)
        status = _classify_data_status(
            horizon_closes={1: 105.0, 3: 110.0, 5: None, 10: 95.0},
            horizons=EVALUATE_HORIZONS,
            today="20260620",
            horizon_dates={1: "20260602", 3: "20260604", 5: "20260608", 10: "20260615"},
        )
        assert status == "partial"

    def test_complete(self) -> None:
        status = _classify_data_status(
            horizon_closes={1: 105.0, 3: 110.0, 5: 108.0, 10: 95.0},
            horizons=EVALUATE_HORIZONS,
            today="20260620",
            horizon_dates={1: "20260602", 3: "20260604", 5: "20260608", 10: "20260615"},
        )
        assert status == "complete"
