"""Tushare-backed ETF universe and daily cache for vwap-reversion.

This module intentionally stays outside the realtime execution path. It prepares
盤前/盤後 data that the strategy can use for liquidity, validity, and risk
filters without pretending daily data can reconstruct intraday VWAP paths.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database
    from deeptrade.core.tushare_client import TushareClient


@dataclass(frozen=True)
class SyncResult:
    rows: int
    message: str


def sync_etf_universe(
    db: Database,
    tushare: TushareClient,
    *,
    t0_whitelist: Iterable[str] = (),
) -> SyncResult:
    """Sync listed exchange-traded funds and mark user-declared T+0 codes."""
    whitelist = {c.strip().upper() for c in t0_whitelist if c.strip()}
    df = tushare.call(
        "fund_basic",
        params={"market": "E", "status": "L"},
    )
    if df is None or df.empty:
        return SyncResult(0, "fund_basic returned no listed exchange-traded funds")

    rows = 0
    with db.transaction():
        for _, row in df.iterrows():
            code = str(row.get("ts_code") or "").upper()
            if not code:
                continue
            db.execute(
                "INSERT OR REPLACE INTO vwr_etf_universe("
                "ts_code, name, fund_type, invest_type, market, status, list_date, "
                "delist_date, management, benchmark, t0_eligible, enabled, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "COALESCE((SELECT enabled FROM vwr_etf_universe WHERE ts_code = ?), 1), "
                "CURRENT_TIMESTAMP)",
                (
                    code,
                    _s(row, "name"),
                    _s(row, "fund_type"),
                    _s(row, "invest_type"),
                    _s(row, "market"),
                    _s(row, "status"),
                    _s(row, "list_date"),
                    _s(row, "delist_date"),
                    _s(row, "management"),
                    _s(row, "benchmark"),
                    1 if code in whitelist else 0,
                    code,
                ),
            )
            rows += 1
    return SyncResult(rows, f"synced {rows} ETF universe rows")


def sync_margin_eligibility(
    db: Database,
    tushare: TushareClient,
    *,
    trade_date: str,
) -> SyncResult:
    """Mark ETFs that appear in the margin securities list for the date."""
    total = 0
    with db.transaction():
        db.execute("UPDATE vwr_etf_universe SET margin_eligible = 0")
        for exchange in ("SSE", "SZSE"):
            df = tushare.call(
                "margin_secs",
                params={"trade_date": trade_date, "exchange": exchange},
            )
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                code = str(row.get("ts_code") or "").upper()
                if not code:
                    continue
                db.execute(
                    "UPDATE vwr_etf_universe SET margin_eligible = 1, "
                    "updated_at = CURRENT_TIMESTAMP WHERE ts_code = ?",
                    (code,),
                )
                total += 1
    return SyncResult(total, f"marked {total} margin-eligible securities")


def sync_etf_daily(
    db: Database,
    tushare: TushareClient,
    *,
    code: str,
    start: str,
    end: str,
) -> SyncResult:
    """Sync ETF daily market data plus available risk metadata for one ETF."""
    code = code.strip().upper()
    daily = _frame_by_date(
        tushare.call("fund_daily", params={"ts_code": code, "start_date": start, "end_date": end})
    )
    adj = _frame_by_date(
        tushare.call("fund_adj", params={"ts_code": code, "start_date": start, "end_date": end})
    )
    share = _frame_by_date(
        tushare.call("fund_share", params={"ts_code": code, "start_date": start, "end_date": end})
    )
    nav = _frame_by_date(
        tushare.call("fund_nav", params={"ts_code": code, "start_date": start, "end_date": end}),
        date_col="nav_date",
    )
    limits = _frame_by_date(
        tushare.call("stk_limit", params={"ts_code": code, "start_date": start, "end_date": end})
    )

    rows = 0
    with db.transaction():
        for trade_date, row in daily.items():
            payload = {
                "fund_daily": row,
                "fund_adj": adj.get(trade_date),
                "fund_share": share.get(trade_date),
                "fund_nav": nav.get(trade_date),
                "stk_limit": limits.get(trade_date),
            }
            db.execute(
                "INSERT OR REPLACE INTO vwr_etf_daily("
                "ts_code, trade_date, open, high, low, close, pre_close, pct_chg, "
                "vol, amount, adj_factor, fd_share, unit_nav, adj_nav, up_limit, "
                "down_limit, source_json, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "CURRENT_TIMESTAMP)",
                (
                    code,
                    trade_date,
                    _f(row, "open"),
                    _f(row, "high"),
                    _f(row, "low"),
                    _f(row, "close"),
                    _f(row, "pre_close"),
                    _f(row, "pct_chg"),
                    _f(row, "vol"),
                    _f(row, "amount"),
                    _f(adj.get(trade_date), "adj_factor"),
                    _f(share.get(trade_date), "fd_share"),
                    _f(nav.get(trade_date), "unit_nav"),
                    _f(nav.get(trade_date), "adj_nav"),
                    _f(limits.get(trade_date), "up_limit"),
                    _f(limits.get(trade_date), "down_limit"),
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
            rows += 1
    return SyncResult(rows, f"synced {rows} ETF daily rows for {code}")


def list_enabled_universe(db: Database, *, limit: int = 50) -> list[dict[str, Any]]:
    cols = (
        "ts_code",
        "name",
        "fund_type",
        "invest_type",
        "market",
        "status",
        "margin_eligible",
        "t0_eligible",
        "enabled",
    )
    rows = db.fetchall(
        f"SELECT {', '.join(cols)} FROM vwr_etf_universe "
        "ORDER BY t0_eligible DESC, margin_eligible DESC, ts_code LIMIT ?",
        (limit,),
    )
    return [dict(zip(cols, r)) for r in rows]


def build_daily_features(
    db: Database,
    *,
    code: str,
    start: str,
    end: str,
    min_amount_ma20: float = 200_000_000.0,
) -> SyncResult:
    """Build daily liquidity/volatility/regime features from cached ETF daily data."""
    code = code.strip().upper()
    cols = (
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "amount",
    )
    rows = db.fetchall(
        f"SELECT {', '.join(cols)} FROM vwr_etf_daily WHERE ts_code = ? "
        "AND trade_date <= ? ORDER BY trade_date",
        (code, end),
    )
    history = [dict(zip(cols, r)) for r in rows]
    rows_written = 0
    with db.transaction():
        for idx, row in enumerate(history):
            trade_date = str(row["trade_date"])
            if trade_date < start or trade_date > end:
                continue
            close = _num(row.get("close"))
            pre_close = _num(row.get("pre_close"))
            open_px = _num(row.get("open"))
            high = _num(row.get("high"))
            low = _num(row.get("low"))
            if close is None or pre_close is None or pre_close <= 0:
                continue

            ret_1d = close / pre_close - 1.0
            ret_window = _returns(history[: idx + 1])
            amount_window = [_num(r.get("amount")) for r in history[max(0, idx - 19): idx + 1]]
            amount_vals = [v for v in amount_window if v is not None]
            amount_ma20 = statistics.mean(amount_vals) if amount_vals else None
            amount_hist = [_num(r.get("amount")) for r in history[max(0, idx - 251): idx + 1]]
            amount_hist_vals = [v for v in amount_hist if v is not None]
            amount = _num(row.get("amount"))
            amount_pctile = _percentile_rank(amount_hist_vals, amount)
            rv_20d = _std(ret_window[-20:]) * math.sqrt(252.0) if len(ret_window[-20:]) >= 2 else None
            atr_pct = _atr_pct(history[max(0, idx - 19): idx + 1])
            ret_5d = _compound(ret_window[-5:]) if len(ret_window[-5:]) == 5 else None
            gap_pct = (open_px / pre_close - 1.0) if open_px is not None else None
            liquidity_ok = int(amount_ma20 is not None and amount_ma20 >= min_amount_ma20)
            volatility_regime = _vol_regime(rv_20d)
            trend_regime = _trend_regime(ret_5d)
            payload = {
                "min_amount_ma20": min_amount_ma20,
                "history_rows": idx + 1,
            }
            db.execute(
                "INSERT OR REPLACE INTO vwr_daily_features("
                "ts_code, trade_date, ret_1d, ret_5d, rv_20d, atr_pct_20d, "
                "amount_ma20, amount_pctile_252, gap_pct, liquidity_ok, "
                "volatility_regime, trend_regime, source_json, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (
                    code,
                    trade_date,
                    ret_1d,
                    ret_5d,
                    rv_20d,
                    atr_pct,
                    amount_ma20,
                    amount_pctile,
                    gap_pct,
                    liquidity_ok,
                    volatility_regime,
                    trend_regime,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            rows_written += 1
    return SyncResult(rows_written, f"built {rows_written} daily feature rows for {code}")


def _frame_by_date(df: Any, *, date_col: str = "trade_date") -> dict[str, dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        d = row.get(date_col)
        if d is None:
            continue
        out[str(d)] = {str(k): _jsonable(v) for k, v in row.to_dict().items()}
    return out


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _returns(rows: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        close = _num(row.get("close"))
        pre_close = _num(row.get("pre_close"))
        if close is not None and pre_close is not None and pre_close > 0:
            out.append(close / pre_close - 1.0)
    return out


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) >= 2 else None


def _compound(values: list[float]) -> float:
    acc = 1.0
    for v in values:
        acc *= 1.0 + v
    return acc - 1.0


def _percentile_rank(values: list[float], value: float | None) -> float | None:
    if value is None or not values:
        return None
    return sum(1 for v in values if v <= value) / len(values)


def _atr_pct(rows: list[dict[str, Any]]) -> float | None:
    vals: list[float] = []
    for row in rows:
        high = _num(row.get("high"))
        low = _num(row.get("low"))
        pre_close = _num(row.get("pre_close"))
        if high is None or low is None or pre_close is None or pre_close <= 0:
            continue
        tr = max(high - low, abs(high - pre_close), abs(low - pre_close))
        vals.append(tr / pre_close)
    return statistics.mean(vals) if vals else None


def _vol_regime(rv_20d: float | None) -> str:
    if rv_20d is None:
        return "unknown"
    if rv_20d >= 0.35:
        return "high"
    if rv_20d <= 0.12:
        return "low"
    return "normal"


def _trend_regime(ret_5d: float | None) -> str:
    if ret_5d is None:
        return "unknown"
    if ret_5d >= 0.03:
        return "up"
    if ret_5d <= -0.03:
        return "down"
    return "range"


def _jsonable(value: Any) -> Any:
    try:
        if value != value:
            return None
    except Exception:  # noqa: BLE001
        pass
    return value


def _s(row: Any, col: str) -> str | None:
    if row is None:
        return None
    val = row.get(col)
    if val is None:
        return None
    text = str(val)
    return None if text == "nan" else text


def _f(row: dict[str, Any] | None, col: str) -> float | None:
    if row is None:
        return None
    try:
        val = row.get(col)
        if val is None:
            return None
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if f != f else f
