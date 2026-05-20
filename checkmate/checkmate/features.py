"""Daily feature computation for Checkmate.

Five feature groups (development_plan §7.2 + iteration_tasks.md §3 PR-2.1):

1. **Trend** — ``ma20`` / ``ma60`` / ``ma120`` / ``ma_slope60``
   ``ma_slope60`` is the OLS slope of ``close_qfq`` over the trailing 60
   sessions, divided by the current close (fractional change per session).

2. **Volatility** — ``atr20`` (Wilder smoothing) / ``atr_pct = atr20 / close_qfq``

3. **Strength** — ``ret_60`` / ``ret_120`` (forward log-returns over the
   trailing window in percent) plus cross-sectional ranks ``rs60_pctile`` /
   ``rs120_pctile`` in ``[0, 1]`` (computed by :func:`compute_features_frame`
   across all symbols of the same ``trade_date``).

4. **Liquidity** — ``amount_20d_avg`` (yuan, normalised from Tushare's 千元)
   / ``turnover_20d_avg`` (percent) / ``limit_freq_60d`` (fraction of days in
   the 60-session window where ``|pct_chg| > cfg.limit_pct_threshold``).

5. **Pullback quality** — ``drawdown_60d_high`` (current close vs trailing
   60-session high, fractional, negative when below) / ``quiet_score`` (a
   [0, 100] measure of recent intraday amplitude compression) /
   ``above_ma20_days`` (count of trailing sessions with ``close_qfq > ma20``,
   over the same 60-session window).

The composite ``score`` is a [0, 100] weighted blend of the five components.
``score_breakdown`` carries both the component sub-scores and the weights so
the explain page can render them without re-deriving.

All per-symbol features come from the **qfq** view; raw ``close`` / ``pre_close``
are used only for the limit-day frequency (limits are exchange-imposed on
raw prices).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import data
from .config import FeaturesConfig
from .runtime import CheckmateRuntime

logger = logging.getLogger(__name__)


# Stable column order — checked in tests as a schema-drift guard.
FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_date", "ts_code",
    "close_qfq",
    "ma20", "ma60", "ma120", "ma_slope60",
    "atr20", "atr_pct",
    "ret_60", "ret_120",
    "rs60_pctile", "rs120_pctile",
    "amount_20d_avg", "turnover_20d_avg", "limit_freq_60d",
    "drawdown_60d_high", "quiet_score", "above_ma20_days",
    "score",
)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class FeaturesRow:
    trade_date: str
    ts_code: str
    close_qfq: float | None = None

    ma20: float | None = None
    ma60: float | None = None
    ma120: float | None = None
    ma_slope60: float | None = None

    atr20: float | None = None
    atr_pct: float | None = None

    ret_60: float | None = None
    ret_120: float | None = None
    rs60_pctile: float | None = None
    rs120_pctile: float | None = None

    amount_20d_avg: float | None = None
    turnover_20d_avg: float | None = None
    limit_freq_60d: float | None = None

    drawdown_60d_high: float | None = None
    quiet_score: float | None = None
    above_ma20_days: int | None = None

    score: float | None = None
    score_breakdown: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _tail(s: pd.Series, n: int) -> pd.Series:
    """Last ``n`` rows of a Series (preserves the original index)."""
    return s.iloc[-n:] if len(s) >= n else s


def _last_mean(s: pd.Series, n: int) -> float | None:
    if len(s) < n:
        return None
    val = float(_tail(s, n).mean())
    return val if math.isfinite(val) else None


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> float | None:
    """Standard Wilder ATR over window ``n``. Returns the latest ATR value.

    TR[t] = max(high-low, |high - prev_close|, |low - prev_close|)
    Initial ATR = simple mean of the first n TR values
    Subsequent  ATR[t] = (ATR[t-1] * (n-1) + TR[t]) / n
    """
    if len(close) < n + 1:
        return None
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.iloc[1:]  # drop the first row whose prev_close is NaN
    if len(tr) < n:
        return None
    atr = float(tr.iloc[:n].mean())
    for v in tr.iloc[n:]:
        atr = (atr * (n - 1) + float(v)) / n
    return atr if math.isfinite(atr) else None


def _ols_slope(y: pd.Series) -> float | None:
    """OLS slope of ``y`` vs an integer index. Returns None on degenerate input."""
    if len(y) < 2:
        return None
    x = np.arange(len(y), dtype=float)
    yv = y.astype(float).values
    if np.isnan(yv).any():
        return None
    x_mean = x.mean()
    y_mean = yv.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 0:
        return None
    slope = float(((x - x_mean) * (yv - y_mean)).sum() / denom)
    return slope if math.isfinite(slope) else None


# ---------------------------------------------------------------------------
# Per-symbol features
# ---------------------------------------------------------------------------


def compute_features_for_symbol(
    ts_code: str,
    trade_date: str,
    qfq_window: pd.DataFrame,
    daily_basic_window: pd.DataFrame,
    cfg: FeaturesConfig | None = None,
) -> FeaturesRow:
    """Pure function — derive a :class:`FeaturesRow` from pre-fetched windows.

    Does NOT compute ``rs60_pctile`` / ``rs120_pctile`` (cross-sectional, only
    knowable when the full trade-date cohort is available — see
    :func:`compute_features_frame`).

    Either window may be empty / shorter than the configured lookback; any
    feature that can't be computed degrades to ``None`` rather than raising.
    """
    cfg = cfg or FeaturesConfig()
    row = FeaturesRow(trade_date=trade_date, ts_code=ts_code)

    if qfq_window is None or qfq_window.empty:
        return row

    qf = qfq_window.copy()
    qf["trade_date"] = qf["trade_date"].astype(str)
    qf = qf[qf["trade_date"] <= trade_date].sort_values("trade_date").reset_index(drop=True)
    if qf.empty:
        return row

    # qfq columns: rely on '_qfq' suffix produced by data.fetch_daily_qfq.
    # Tests that build fixture frames directly can also pass raw columns —
    # fall back to those.
    def _col(name: str) -> pd.Series:
        qfq_name = f"{name}_qfq"
        if qfq_name in qf.columns:
            return qf[qfq_name].astype(float)
        if name in qf.columns:
            return qf[name].astype(float)
        return pd.Series(dtype=float)

    close = _col("close")
    high = _col("high")
    low = _col("low")
    if close.empty:
        return row
    row.close_qfq = float(close.iloc[-1]) if math.isfinite(float(close.iloc[-1])) else None

    # ---- trend
    row.ma20 = _last_mean(close, 20)
    row.ma60 = _last_mean(close, 60)
    row.ma120 = _last_mean(close, 120)
    if len(close) >= 60 and row.close_qfq:
        slope = _ols_slope(_tail(close, 60))
        if slope is not None:
            row.ma_slope60 = round(slope / row.close_qfq, 6)

    # ---- volatility (Wilder ATR over qfq high/low/close)
    if not high.empty and not low.empty and len(close) >= cfg.atr_window + 1:
        atr = _wilder_atr(high, low, close, cfg.atr_window)
        if atr is not None:
            row.atr20 = round(atr, 6)
            if row.close_qfq:
                row.atr_pct = round(atr / row.close_qfq, 6)

    # ---- strength (per-symbol returns; pctile lands in the frame-level pass)
    def _ret(n: int) -> float | None:
        if len(close) < n + 1:
            return None
        anchor = float(close.iloc[-(n + 1)])
        if anchor <= 0:
            return None
        last = float(close.iloc[-1])
        return round((last / anchor - 1.0) * 100.0, 4)

    row.ret_60 = _ret(60)
    row.ret_120 = _ret(120)

    # ---- liquidity (from daily_basic — Tushare amount is 千元)
    if daily_basic_window is not None and not daily_basic_window.empty:
        db = daily_basic_window.copy()
        db["trade_date"] = db["trade_date"].astype(str)
        db = db[db["trade_date"] <= trade_date].sort_values("trade_date")
        if not db.empty:
            win = db.tail(cfg.liquidity_window)
            if "amount" in win.columns:
                avg = float(win["amount"].astype(float).mean())
                row.amount_20d_avg = round(avg * 1000.0, 2)
            if "turnover_rate" in win.columns:
                row.turnover_20d_avg = round(float(win["turnover_rate"].astype(float).mean()), 4)

    # ---- limit_freq_60d uses raw close / pre_close (limits are on raw prices)
    if {"close", "pre_close"}.issubset(qf.columns) and len(qf) >= 1:
        win = qf.tail(cfg.limit_window)
        n = len(win)
        if n > 0:
            pre = win["pre_close"].astype(float).values
            cls = win["close"].astype(float).values
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(pre > 0, (cls / pre - 1.0), 0.0)
            n_limit = int((np.abs(pct) > cfg.limit_pct_threshold).sum())
            row.limit_freq_60d = round(n_limit / n, 4)

    # ---- pullback quality
    if len(close) >= 2:
        win = _tail(close, cfg.pullback_window)
        peak = float(win.max())
        if peak > 0 and row.close_qfq:
            row.drawdown_60d_high = round(row.close_qfq / peak - 1.0, 6)

    # Quiet score: low recent intraday amplitude → high score. We use the
    # qfq high/low, but only the *ratio* matters so the qfq scaling is fine.
    if not high.empty and not low.empty and row.close_qfq:
        win_h = _tail(high, cfg.liquidity_window)
        win_l = _tail(low, cfg.liquidity_window)
        win_c = _tail(close, cfg.liquidity_window)
        if len(win_h) == len(win_c) and len(win_h) > 0:
            amp = ((win_h.values - win_l.values) / win_c.values).mean()
            if math.isfinite(amp):
                # 0% amplitude → 100; 5% → ~50; 10% → 0.
                row.quiet_score = round(max(0.0, min(100.0, 100.0 - amp * 1000.0)), 4)

    # Above-ma20 days over the pullback window.
    if row.ma20 is not None and len(close) >= cfg.pullback_window:
        ma20_series = close.rolling(20, min_periods=20).mean()
        win_close = _tail(close, cfg.pullback_window)
        win_ma20 = _tail(ma20_series, cfg.pullback_window)
        valid = win_ma20.notna()
        row.above_ma20_days = int(((win_close[valid] > win_ma20[valid])).sum())

    return row


# ---------------------------------------------------------------------------
# Cross-sectional pass + scoring
# ---------------------------------------------------------------------------


def _component_trend(row: FeaturesRow) -> float:
    if row.close_qfq is None:
        return 0.0
    pts = 0.0
    if row.ma20 is not None and row.close_qfq > row.ma20:
        pts += 40.0
    if row.ma60 is not None and row.close_qfq > row.ma60:
        pts += 30.0
    if row.ma120 is not None and row.close_qfq > row.ma120:
        pts += 30.0
    return pts


def _component_volatility(row: FeaturesRow) -> float:
    if row.atr_pct is None:
        return 50.0
    # Peak around 2.5% atr_pct (mid-cap noisy-but-tradeable); falloff both sides.
    pct = row.atr_pct * 100.0  # convert to percent
    return max(0.0, 100.0 - abs(pct - 2.5) * 20.0)


def _component_strength(row: FeaturesRow) -> float:
    if row.rs60_pctile is None and row.rs120_pctile is None:
        return 50.0
    parts = [p for p in (row.rs60_pctile, row.rs120_pctile) if p is not None]
    return float(sum(parts) / len(parts)) * 100.0


def _component_liquidity(row: FeaturesRow) -> float:
    if not row.amount_20d_avg or row.amount_20d_avg <= 0:
        return 0.0
    # 5_000_万 (5e7) → 30; 5亿 (5e8) → 50; 50亿 (5e9) → 70; 500亿 (5e10) → 90.
    return max(0.0, min(100.0, 30.0 + 20.0 * math.log10(row.amount_20d_avg / 5e7)))


def _component_pullback(row: FeaturesRow) -> float:
    if row.drawdown_60d_high is None:
        return 50.0
    dd = row.drawdown_60d_high * 100.0  # to percent (negative number)
    if dd >= 0:
        return 60.0  # at the high — mediocre entry quality, but not bad
    abs_dd = -dd
    if abs_dd <= 5:
        return 100.0 - (5.0 - abs_dd) * 4.0  # 5% → 100; 0% → 80
    if abs_dd <= 20:
        return 100.0 - (abs_dd - 5.0) * 4.67  # 5% → 100; 20% → 30
    return max(0.0, 30.0 - (abs_dd - 20.0) * 1.5)


def compute_score(row: FeaturesRow, cfg: FeaturesConfig | None = None) -> tuple[float, dict[str, Any]]:
    """Return ``(score, breakdown)``. ``breakdown`` has components + weights."""
    cfg = cfg or FeaturesConfig()
    weights = {
        "trend": cfg.score_weight_trend,
        "volatility": cfg.score_weight_volatility,
        "strength": cfg.score_weight_strength,
        "liquidity": cfg.score_weight_liquidity,
        "pullback": cfg.score_weight_pullback,
    }
    components = {
        "trend": round(_component_trend(row), 4),
        "volatility": round(_component_volatility(row), 4),
        "strength": round(_component_strength(row), 4),
        "liquidity": round(_component_liquidity(row), 4),
        "pullback": round(_component_pullback(row), 4),
    }
    score = round(sum(weights[k] * components[k] for k in weights), 4)
    return score, {"components": components, "weights": weights}


def _to_frame(rows: list[FeaturesRow]) -> pd.DataFrame:
    return pd.DataFrame([
        {col: getattr(r, col) for col in FEATURE_COLUMNS}
        for r in rows
    ])


def compute_features_frame(
    rt: CheckmateRuntime,
    trade_date: str,
    ts_codes: list[str],
    cfg: FeaturesConfig | None = None,
) -> tuple[pd.DataFrame, list[FeaturesRow]]:
    """End-to-end: fetch windows for every ts_code + cross-sectional rank.

    Returns ``(frame, rows)`` so the caller can persist via
    :func:`upsert_features_daily` (rows form) or do further pandas work
    (frame form). The two views are byte-identical apart from columns
    that aren't part of ``FEATURE_COLUMNS`` (``score_breakdown`` is on the
    rows but not on the frame).
    """
    cfg = cfg or FeaturesConfig()
    rows: list[FeaturesRow] = []

    # Window: ``min_history_sessions`` calendar-day approximation (cache layer
    # rejects misses cheaply, so we ask for ~190 cal-days to comfortably
    # cover 130 sessions).
    from datetime import datetime, timedelta  # noqa: PLC0415

    window_start = (
        datetime.strptime(trade_date, "%Y%m%d")
        - timedelta(days=int(cfg.min_history_sessions * 1.7))
    ).strftime("%Y%m%d")

    for ts_code in ts_codes:
        try:
            qfq = data.fetch_daily_qfq(rt.tushare, ts_code, window_start, trade_date)
            db_win = data.fetch_daily_basic(rt.tushare, ts_code, window_start, trade_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("features fetch failed for %s: %s", ts_code, exc)
            rows.append(FeaturesRow(trade_date=trade_date, ts_code=ts_code))
            continue
        rows.append(compute_features_for_symbol(ts_code, trade_date, qfq, db_win, cfg))

    # Cross-sectional pctile ranks for ret_60 / ret_120.
    df = _to_frame(rows)
    if not df.empty:
        for src, dst in (("ret_60", "rs60_pctile"), ("ret_120", "rs120_pctile")):
            if src in df.columns:
                # pandas rank(pct=True) returns NaN for NaN inputs — exactly
                # what we want; the consumers treat None / NaN as missing.
                ranks = df[src].rank(pct=True, na_option="keep")
                df[dst] = ranks.round(6)
        # Propagate ranks back into rows so callers can keep using FeaturesRow.
        rs60 = df["rs60_pctile"].tolist()
        rs120 = df["rs120_pctile"].tolist()
        for i, r in enumerate(rows):
            v60 = rs60[i]
            v120 = rs120[i]
            r.rs60_pctile = None if pd.isna(v60) else float(v60)
            r.rs120_pctile = None if pd.isna(v120) else float(v120)
            r.score, r.score_breakdown = compute_score(r, cfg)
            # Re-sync the frame's score column with the recomputed value.
            df.at[i, "score"] = r.score
    return df, rows


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def upsert_features_daily(db: Any, rows: list[FeaturesRow]) -> int:
    """INSERT OR REPLACE one row per :class:`FeaturesRow` into the features table."""
    import json  # noqa: PLC0415

    n = 0
    for r in rows:
        db.execute(
            """
            INSERT OR REPLACE INTO checkmate_features_daily
                (trade_date, ts_code, close_qfq, ma20, ma60, ma120, ma_slope60,
                 atr20, atr_pct, ret_60, ret_120, rs60_pctile, rs120_pctile,
                 amount_20d_avg, turnover_20d_avg, limit_freq_60d,
                 drawdown_60d_high, quiet_score, above_ma20_days,
                 score, score_breakdown, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                r.trade_date, r.ts_code, r.close_qfq,
                r.ma20, r.ma60, r.ma120, r.ma_slope60,
                r.atr20, r.atr_pct, r.ret_60, r.ret_120,
                r.rs60_pctile, r.rs120_pctile,
                r.amount_20d_avg, r.turnover_20d_avg, r.limit_freq_60d,
                r.drawdown_60d_high, r.quiet_score, r.above_ma20_days,
                r.score, json.dumps(r.score_breakdown, ensure_ascii=False),
            ],
        )
        n += 1
    return n
