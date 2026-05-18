"""Shared pytest fixtures for APW tests."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig


@pytest.fixture
def default_cfg() -> ApwConfig:
    return ApwConfig()


def make_quotes(
    *,
    ts_code: str = "600000.SH",
    start_close: float = 10.0,
    n: int = 130,
    pattern: str = "flat",
    probe_index: int | None = None,
    probe_multiplier: float = 4.0,
) -> pd.DataFrame:
    """Build a daily quote frame with controlled pattern.

    pattern in {"flat", "uptrend", "downtrend", "v_shape"}.
    """
    rows: list[dict] = []
    base_vol = 1_000_000.0
    base_date = pd.Timestamp("2024-01-01")
    close = start_close
    for i in range(n):
        if pattern == "flat":
            close = start_close + (i % 5 - 2) * 0.05
        elif pattern == "uptrend":
            close = start_close * (1 + 0.005 * i)
        elif pattern == "downtrend":
            close = start_close * (1 - 0.005 * i)
        elif pattern == "v_shape":
            half = n // 2
            close = start_close * (1 - 0.005 * i) if i < half else start_close * (1 + 0.003 * (i - half))
        else:
            close = start_close

        prev_close = rows[-1]["close"] if rows else close
        pct_chg = (close - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0

        vol = base_vol * (1.0 + (i % 7 - 3) * 0.05)
        if probe_index is not None and i == probe_index:
            vol = base_vol * probe_multiplier
        high = close * 1.03
        low = close * 0.97
        open_p = (high + low) / 2.0
        # Make probe day's amplitude meaningful
        if probe_index is not None and i == probe_index:
            high = close * 1.08
            low = prev_close * 0.99
            open_p = prev_close
            close = high * 0.97  # 大阳线 with small upper shadow

        amount = close * vol / 100.0  # in 千元
        rows.append({
            "ts_code": ts_code,
            "trade_date": (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d"),
            "open": round(open_p, 3),
            "high": round(high, 3),
            "low": round(low, 3),
            "close": round(close, 3),
            "vol": round(vol, 1),
            "amount": round(amount, 2),
            "pct_chg": round(pct_chg, 2),
            "turnover_rate": 3.0 + (1.0 if probe_index is not None and i == probe_index else 0.0) * 6.0,
            "circ_mv": 5_000_000.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def quotes_flat():
    return make_quotes(pattern="flat", n=130)


@pytest.fixture
def quotes_with_probe():
    """130 trading days: 100 days of flat accumulation + a clear probe + 20 days of washout."""
    df = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
    # Make post-probe a slight pullback then sideways
    for i in range(111, 130):
        df.at[i, "close"] = df.at[110, "close"] * (0.95 + (i - 111) * 0.001)
        df.at[i, "high"] = df.at[i, "close"] * 1.02
        df.at[i, "low"] = df.at[i, "close"] * 0.98
        df.at[i, "vol"] = df.at[110, "vol"] * 0.4  # 缩量
    return df
