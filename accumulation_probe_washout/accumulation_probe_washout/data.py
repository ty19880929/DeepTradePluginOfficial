"""Data layer for the accumulation-probe-washout strategy.

Two layers — kept apart so unit tests can stay pure:
  * **Pure functions** — universe filter, scoring, state machine derivation.
    No I/O, fully deterministic over a pandas frame.
  * **Tushare thin wrappers** — narrow facades around TushareClient calls;
    optional-API failures degrade to (empty df, missing-data tag).

The scoring keeps every component on a 0-100 scale (matches the source spec
§6.3 weight blocks). All thresholds come from ``ApwConfig`` rather than being
inlined here so settings show / set can override at runtime.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from .calendar import TradeCalendar
from .config import ApwConfig
from .schemas import APWPhase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade-date resolution (mirrors VA helpers)
# ---------------------------------------------------------------------------


# Anchor for the latest-published-trade-date probe. The Shanghai Composite has
# been published every trading day since the API launched and is therefore the
# safest market-level signal for "what's the most recent trade day Tushare has
# data for". See limit_up_board/data.py for the same probe rationale.
LATEST_TRADE_DATE_PROBE_INDEX = "000001.SH"


def fetch_latest_trade_date(
    tushare: Any,
    *,
    index_code: str = LATEST_TRADE_DATE_PROBE_INDEX,
    lookback_days: int = 60,
) -> str:
    """Return the most recent published trade_date according to ``index_daily``.

    Previously T was derived from ``datetime.now()`` + ``trade_cal``, which
    broke whenever the machine's clock or timezone was off. T now comes from
    market data: Tushare's ``index_daily`` publishes on each trading day's
    close, so its ``max(trade_date)`` is the authoritative "latest trade day"
    regardless of local time. The local clock only bounds the query window
    (quota friendliness); grosser skew falls out of the window and raises here.

    ``force_sync=True`` is mandatory: TushareClient classifies daily-family
    APIs as ``trade_day_immutable``, so without it a stale cached window
    would be returned after the next trading day publishes.
    """
    now_local = datetime.now()
    start_date = (now_local - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_date = (now_local + timedelta(days=1)).strftime("%Y%m%d")
    df = tushare.call(
        "index_daily",
        params={"ts_code": index_code, "start_date": start_date, "end_date": end_date},
        force_sync=True,
    )
    if df is None or df.empty or "trade_date" not in df.columns:
        raise RuntimeError(
            f"index_daily({index_code}) probe returned no rows over "
            f"{start_date}..{end_date}; cannot resolve the latest trade date. "
            "Pass --trade-date <YYYYMMDD> to override, or check Tushare access "
            "(token / api permission / network)."
        )
    return str(df["trade_date"].astype(str).max())


def resolve_trade_date(
    calendar: TradeCalendar,
    *,
    latest_trade_date: str | None = None,
    user_specified: str | None = None,
) -> tuple[str, str]:
    """Return (T, T+1).

    Exactly one of ``user_specified`` (CLI override) or ``latest_trade_date``
    (typically from :func:`fetch_latest_trade_date`) must be supplied. T+1 is
    the first open day strictly after T per the trade calendar.

    No reliance on ``datetime.now()``: a machine with the wrong clock or
    timezone still gets the correct T, because T is grounded in either an
    explicit user value or published market data — never local-system time.
    """
    T = user_specified or latest_trade_date
    if not T:
        raise ValueError(
            "resolve_trade_date requires either user_specified or latest_trade_date"
        )
    return T, _safe_next_open(calendar, T)


def _safe_next_open(calendar: TradeCalendar, T: str) -> str:
    # next_open may raise when the loaded trade_cal doesn't extend past T
    # (e.g. unit-test fixture, or tushare returning a clipped window). Falling
    # back to T+1 calendar day keeps downstream schema/prompt fields populated
    # — callers that need a real trading day should rely on the runner pulling
    # the calendar with a forward buffer.
    try:
        return calendar.next_open(T)
    except ValueError:
        try:
            base = datetime.strptime(T, "%Y%m%d")
        except ValueError:
            return T
        return (base + timedelta(days=1)).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Pure functions — main board filter
# ---------------------------------------------------------------------------


def filter_main_board(
    stock_basic: pd.DataFrame,
    cfg: ApwConfig,
    *,
    trade_date: str | None = None,
) -> pd.DataFrame:
    """Keep listed Shanghai/Shenzhen main-board ordinary shares.

    Excludes STAR / ChiNext / BSE (688 / 300 / 8x / 4x), new stocks under
    ``cfg.listed_days_min`` (when trade_date is supplied), and unlisted.
    """
    if not {"market", "exchange", "ts_code"}.issubset(stock_basic.columns):
        raise ValueError("stock_basic missing market/exchange/ts_code columns")

    df = stock_basic.copy()
    df = df[(df["market"] == "主板") & (df["exchange"].isin(["SSE", "SZSE"]))]
    if "list_status" in df.columns:
        df = df[df["list_status"] == "L"]
    # ts_code prefix filter — narrow main-board to A-share ordinary by excluding
    # STAR (688), ChiNext (300), BSE (8x / 4x) and B-shares (Shenzhen 200xxx,
    # Shanghai 900xxx). Some of these would already be cut by market=主板, but
    # the prefix block is defense-in-depth against tushare schema drift.
    code_head = df["ts_code"].str.split(".").str[0]
    df = df[~code_head.str.startswith(("688", "300", "8", "4", "200", "900"))]

    if trade_date is not None and "list_date" in df.columns:
        try:
            t = datetime.strptime(trade_date, "%Y%m%d")
        except ValueError:
            t = None
        if t is not None:

            def _ld_days(ld: Any) -> int:
                try:
                    return (t - datetime.strptime(str(int(ld)), "%Y%m%d")).days
                except (ValueError, TypeError):
                    return 0

            df = df[df["list_date"].apply(_ld_days) >= cfg.listed_days_min]

    return df.reset_index(drop=True)


def filter_st_and_suspend(
    universe: pd.DataFrame,
    st_codes: set[str],
    suspended_codes: set[str],
) -> pd.DataFrame:
    """Drop ST and suspended-on-T rows."""
    mask = ~universe["ts_code"].isin(st_codes) & ~universe["ts_code"].isin(suspended_codes)
    return universe[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip_score(v: float) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    return max(0.0, min(100.0, float(v)))


def _last_n_open(window_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the n trailing rows sorted by trade_date asc."""
    df = window_df.sort_values("trade_date")
    return df.tail(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# accumulation_score — bottom accumulation detection
# ---------------------------------------------------------------------------


def compute_accumulation(
    window_df: pd.DataFrame,
    mf_df: pd.DataFrame | None,
    cfg: ApwConfig,
) -> dict[str, Any]:
    """Score the bottom-accumulation phase.

    Inputs:
      window_df — daily quotes (trade_date asc) for at least
        ``cfg.accumulation_lookback_trade_days`` rows + a broader window for
        low-position detection (up to ``cfg.base_lookback_trade_days``).
      mf_df     — moneyflow rows aligned by trade_date; None / empty triggers
        the degraded path (net_mf_yi=None and risk_flag).

    Returns flat dict with the scoring components specified in source §4.1.
    """
    lookback = cfg.accumulation_lookback_trade_days
    if window_df.empty:
        return {
            "accumulation_score": 0.0,
            "accumulation_days": 0,
            "accumulation_net_mf_yi": None,
            "accumulation_price_change_pct": 0.0,
            "accumulation_volume_trend": 0.0,
            "low_position_score": 0.0,
            "missing_data": ["window"],
        }

    acc = _last_n_open(window_df, lookback)
    if acc.empty:
        return {
            "accumulation_score": 0.0,
            "accumulation_days": 0,
            "accumulation_net_mf_yi": None,
            "accumulation_price_change_pct": 0.0,
            "accumulation_volume_trend": 0.0,
            "low_position_score": 0.0,
            "missing_data": ["window"],
        }

    days = len(acc)
    start_close = float(acc.iloc[0]["close"])
    end_close = float(acc.iloc[-1]["close"])
    price_change_pct = (
        ((end_close - start_close) / start_close) * 100.0 if start_close > 0 else 0.0
    )

    # ---- low position score: where the current close sits in the broader
    # base_lookback window. Lower = better.
    base = _last_n_open(window_df, cfg.base_lookback_trade_days)
    if base.empty:
        low_position_score = 0.0
    else:
        lo = float(base["low"].min())
        hi = float(base["high"].max())
        if hi <= lo:
            low_position_score = 50.0
        else:
            pos = (end_close - lo) / (hi - lo)  # 0 = at lowest, 1 = at top
            low_position_score = max(0.0, min(100.0, (1.0 - pos) * 100.0))

    # ---- range quality: price change in [-5, +15] is ideal (sideways/slow lift)
    pc_score = 0.0
    if -10.0 <= price_change_pct <= 25.0:
        # peak at +5
        delta = abs(price_change_pct - 5.0)
        pc_score = max(0.0, 100.0 - delta * 4.0)

    # ---- moneyflow accumulation (degraded path on missing data)
    missing: list[str] = []
    if mf_df is None or mf_df.empty:
        net_mf_yi = None
        mf_score = 0.0
        missing.append("moneyflow")
    else:
        # tushare moneyflow returns net_mf_amount in 万元 (10k yuan). We want 亿.
        recent = mf_df.sort_values("trade_date").tail(cfg.accumulation_moneyflow_days)
        if recent.empty:
            net_mf_yi = 0.0
            mf_score = 0.0
        else:
            col = (
                "net_mf_amount"
                if "net_mf_amount" in recent.columns
                else ("net_amount" if "net_amount" in recent.columns else None)
            )
            if col is None:
                net_mf_yi = None
                mf_score = 0.0
                missing.append("moneyflow_columns")
            else:
                total_wan = float(recent[col].sum())
                net_mf_yi = total_wan / 10000.0  # 万 → 亿
                # 0.5 亿 = 60 分；2 亿 = 100 分
                if net_mf_yi <= 0:
                    mf_score = 0.0
                else:
                    mf_score = min(100.0, 40.0 + net_mf_yi * 30.0)

    # ---- volume health: up-days relatively heavier than down-days
    vol_trend_score = 0.0
    if days >= 5:
        ups = acc[acc["pct_chg"] > 0]
        downs = acc[acc["pct_chg"] < 0]
        if not ups.empty and not downs.empty:
            up_avg = float(ups["vol"].mean())
            dn_avg = float(downs["vol"].mean())
            if dn_avg > 0:
                ratio = up_avg / dn_avg
                # ratio >=1 is healthy, peak at ~1.5
                vol_trend_score = max(0.0, min(100.0, (ratio - 0.5) * 80.0))

    # ---- penalty: huge daily spike during 'accumulation' window means it's
    # already been outed → not truly under-the-radar
    spike_penalty = 0.0
    if days > 1:
        avg_vol = float(acc["vol"].mean())
        if avg_vol > 0:
            top_vol = float(acc["vol"].max())
            if top_vol / avg_vol > 4.0:
                spike_penalty = min(20.0, (top_vol / avg_vol - 4.0) * 4.0)

    # Composition: 25% low position, 20% range, 25% moneyflow, 20% volume, 10% risk
    score = (
        0.25 * low_position_score
        + 0.20 * pc_score
        + 0.25 * mf_score
        + 0.20 * vol_trend_score
        + 0.10 * max(0.0, 100.0 - spike_penalty * 5.0)
    )

    return {
        "accumulation_score": _clip_score(score),
        "accumulation_days": int(days),
        "accumulation_net_mf_yi": (
            round(net_mf_yi, 4) if isinstance(net_mf_yi, float) else net_mf_yi
        ),
        "accumulation_price_change_pct": round(price_change_pct, 2),
        "accumulation_volume_trend": round(vol_trend_score, 2),
        "low_position_score": _clip_score(low_position_score),
        "missing_data": missing,
    }


# ---------------------------------------------------------------------------
# detect_probe_day — find the most recent天量试盘 within probe lookback
# ---------------------------------------------------------------------------


def detect_probe_day(
    window_df: pd.DataFrame,
    cfg: ApwConfig,
    mf_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Locate the latest probe-day (D2: scan trade_date desc, return first hit).

    Returns the flat ``probe_*`` field dict, or ``None`` if no day in the
    probe-lookback window qualifies (lets the caller short-circuit to
    ``no_setup`` / ``accumulating``).

    ``mf_df`` (moneyflow) is optional — when supplied the probe-day net inflow
    (亿元, from ``net_mf_amount`` / ``net_amount`` in 万元) is attached as
    ``probe_moneyflow_net_yi``; otherwise that field is ``None``.
    """
    if window_df.empty:
        return None

    lookback = cfg.probe_lookback_trade_days
    base_lookback = max(cfg.base_lookback_trade_days, 60)

    df = window_df.sort_values("trade_date").reset_index(drop=True)
    # Last day in the window is "today" relative to candidate scan.
    today_date = str(df.iloc[-1]["trade_date"])
    tail = df.tail(lookback).reset_index(drop=True)
    if tail.empty:
        return None

    # D2: scan latest → earliest, return first qualifier.
    candidates = tail.iloc[::-1]
    for _, row in candidates.iterrows():
        trade_date = str(row["trade_date"])
        vol = float(row["vol"])
        if vol <= 0:
            continue
        # Need 5d / 20d trailing windows ending the day BEFORE the probe.
        idx_in_df = df.index[df["trade_date"] == trade_date]
        if idx_in_df.empty:
            continue
        pos = int(idx_in_df[0])
        if pos < 20:
            continue  # not enough lookback for ratios

        prev5 = df.iloc[max(0, pos - 5) : pos]
        prev20 = df.iloc[max(0, pos - 20) : pos]
        avg5 = float(prev5["vol"].mean()) if not prev5.empty else 0.0
        avg20 = float(prev20["vol"].mean()) if not prev20.empty else 0.0
        if avg5 <= 0 or avg20 <= 0:
            continue
        vol_ratio_5d = vol / avg5
        vol_ratio_20d = vol / avg20

        # vol_rank_pct_60d: percentile within the base window ending AT (and
        # including) the candidate probe day. Slicing relative to ``pos`` keeps
        # later-day volumes out of the comparison; before round-2 P2 we used
        # ``df.tail(base_lookback)`` which leaked future sessions into the
        # ranking and made historic probes look weaker as the run progressed.
        base_start = max(0, pos - base_lookback + 1)
        base_vols = df.iloc[base_start : pos + 1]["vol"].astype(float).values
        rank_pct = float((base_vols < vol).sum()) / max(1, len(base_vols)) * 100.0

        turnover_rate = float(row.get("turnover_rate", 0.0) or 0.0)
        amplitude_pct = 0.0
        low_val = float(row.get("low", 0.0) or 0.0)
        high_val = float(row.get("high", 0.0) or 0.0)
        prev_close = float(prev5.iloc[-1]["close"]) if not prev5.empty else low_val
        if prev_close > 0:
            amplitude_pct = (high_val - low_val) / prev_close * 100.0

        # Threshold gate
        if vol_ratio_5d < cfg.probe_volume_ratio_5d_min:
            continue
        if vol_ratio_20d < cfg.probe_volume_ratio_20d_min:
            continue
        if rank_pct < cfg.probe_volume_rank_pct_60d_min:
            continue
        if turnover_rate < cfg.probe_turnover_rate_min:
            continue
        if amplitude_pct < cfg.probe_amplitude_pct_min:
            continue

        # Candle shape
        open_v = float(row.get("open", 0.0) or 0.0)
        close_v = float(row["close"])
        body = abs(close_v - open_v)
        rng = max(1e-9, high_val - low_val)
        body_ratio = body / rng if rng > 0 else 0.0
        upper_shadow = max(0.0, high_val - max(open_v, close_v))
        upper_shadow_ratio = upper_shadow / rng if rng > 0 else 0.0
        pct_chg = float(row.get("pct_chg", 0.0) or 0.0)

        # Quality scoring (weights from source §6.3):
        vol_score = min(100.0, (vol_ratio_5d - 1.0) * 25.0 + (rank_pct - 50.0))
        vol_score = max(0.0, vol_score)
        turnover_score = min(100.0, turnover_rate * 10.0)
        amp_score = min(100.0, amplitude_pct * 8.0)
        # Capital score from moneyflow proxy (left to caller via probe_moneyflow_net_yi);
        # without mf data we use pct_chg sign + volume conviction
        capital_score = max(0.0, min(100.0, 50.0 + pct_chg * 5.0))
        # Follow-through approximated by body ratio — short body or long upper
        # shadow signals failed试盘.
        follow_score = max(0.0, body_ratio * 100.0 - upper_shadow_ratio * 50.0)

        quality_score = (
            0.30 * vol_score
            + 0.20 * turnover_score
            + 0.20 * amp_score
            + 0.20 * capital_score
            + 0.10 * follow_score
        )

        # Days ago (relative to today_date — last day in window).
        idx_today = int(df.index[df["trade_date"] == today_date][0])
        probe_days_ago = idx_today - pos

        # probe_amount_ratio_20d — probe day amount vs prev-20d avg amount.
        # ``amount`` from tushare ``daily`` is in 千元; the ratio is unitless.
        amount_val = float(row.get("amount", 0.0) or 0.0)
        prev20_amt = (
            float(prev20["amount"].astype(float).mean())
            if "amount" in prev20.columns and not prev20.empty
            else 0.0
        )
        amount_ratio_20d = (amount_val / prev20_amt) if prev20_amt > 0 else 0.0

        # probe_moneyflow_net_yi — sum of net mf on probe day (亿元). Tushare
        # ``moneyflow`` exposes net amount in 万元 under ``net_mf_amount`` /
        # ``net_amount`` depending on API version. Missing mf data ⇒ None.
        probe_mf_net_yi: float | None = None
        if mf_df is not None and not mf_df.empty:
            mf_col = (
                "net_mf_amount"
                if "net_mf_amount" in mf_df.columns
                else ("net_amount" if "net_amount" in mf_df.columns else None)
            )
            if mf_col is not None:
                mf_row = mf_df[mf_df["trade_date"].astype(str) == trade_date]
                if not mf_row.empty:
                    probe_mf_net_yi = round(
                        float(mf_row[mf_col].astype(float).sum()) / 10000.0, 4
                    )

        return {
            "probe_date": trade_date,
            "probe_days_ago": int(probe_days_ago),
            "probe_volume_ratio_5d": round(vol_ratio_5d, 3),
            "probe_volume_ratio_20d": round(vol_ratio_20d, 3),
            "probe_volume_rank_pct_60d": round(rank_pct, 2),
            "probe_amount_ratio_20d": round(amount_ratio_20d, 3),
            "probe_turnover_rate": round(turnover_rate, 3),
            "probe_amplitude_pct": round(amplitude_pct, 2),
            "probe_upper_shadow_ratio": round(upper_shadow_ratio, 3),
            "probe_body_ratio": round(body_ratio, 3),
            "probe_pct_chg": round(pct_chg, 2),
            "probe_moneyflow_net_yi": probe_mf_net_yi,
            "probe_quality_score": _clip_score(quality_score),
            "probe_high": round(high_val, 3),
            "probe_low": round(low_val, 3),
            "probe_close": round(close_v, 3),
        }

    return None


# ---------------------------------------------------------------------------
# compute_washout — post-probe shake-out health
# ---------------------------------------------------------------------------


def _ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()


def compute_washout(
    window_df: pd.DataFrame,
    mf_df: pd.DataFrame | None,
    probe_info: dict[str, Any] | None,
    cfg: ApwConfig,
) -> dict[str, Any]:
    """Score the post-probe wash-out window."""
    if probe_info is None or window_df.empty:
        return {
            "washout_days": 0,
            "post_probe_max_drawdown_pct": 0.0,
            "post_probe_volume_shrink_ratio": 0.0,
            "post_probe_low_broken": False,
            "post_probe_ma20_broken": False,
            "post_probe_ma60_broken": False,
            "post_probe_moneyflow_net_yi": None,
            "washout_volatility_compression": 0.0,
            "washout_score": 0.0,
        }

    df = window_df.sort_values("trade_date").reset_index(drop=True)
    probe_date = probe_info["probe_date"]
    probe_high = float(probe_info.get("probe_high", probe_info.get("probe_close", 0.0)))
    probe_low = float(probe_info.get("probe_low", probe_info.get("probe_close", 0.0)))

    pos = df.index[df["trade_date"] == probe_date]
    if len(pos) == 0:
        return {
            "washout_days": 0,
            "post_probe_max_drawdown_pct": 0.0,
            "post_probe_volume_shrink_ratio": 0.0,
            "post_probe_low_broken": False,
            "post_probe_ma20_broken": False,
            "post_probe_ma60_broken": False,
            "post_probe_moneyflow_net_yi": None,
            "washout_volatility_compression": 0.0,
            "washout_score": 0.0,
        }

    after = df.iloc[int(pos[0]) + 1 :].reset_index(drop=True)
    washout_days = len(after)
    if washout_days == 0:
        return {
            "washout_days": 0,
            "post_probe_max_drawdown_pct": 0.0,
            "post_probe_volume_shrink_ratio": 0.0,
            "post_probe_low_broken": False,
            "post_probe_ma20_broken": False,
            "post_probe_ma60_broken": False,
            "post_probe_moneyflow_net_yi": None,
            "washout_volatility_compression": 0.0,
            "washout_score": 0.0,
        }

    # Drawdown from probe close
    probe_close = float(probe_info.get("probe_close", probe_high))
    min_low = float(after["low"].min())
    drawdown_pct = (
        ((probe_close - min_low) / probe_close) * 100.0 if probe_close > 0 else 0.0
    )

    # Volume shrink — avg post-probe vol vs avg pre-probe 20d vol
    probe_pos = int(pos[0])
    prev20 = df.iloc[max(0, probe_pos - 20) : probe_pos]
    pre_avg = float(prev20["vol"].mean()) if not prev20.empty else 0.0
    post_avg = float(after["vol"].mean()) if not after.empty else 0.0
    shrink_ratio = (post_avg / pre_avg) if pre_avg > 0 else 1.0

    # MA breaches — compute MAs on the full df up to each post-probe day
    closes = df["close"].astype(float)
    ma20 = _ma(closes, 20)
    ma60 = _ma(closes, 60)
    ma20_breaks = False
    ma60_breaks = False
    for i in range(int(pos[0]) + 1, len(df)):
        if float(df.iloc[i]["close"]) < float(ma20.iloc[i]):
            ma20_breaks = True
        if float(df.iloc[i]["close"]) < float(ma60.iloc[i]):
            ma60_breaks = True

    low_broken = min_low < probe_low

    # Volatility compression — late-half ATR vs early-half ATR
    vc_score = 0.0
    if washout_days >= 6:
        half = washout_days // 2
        early = after.iloc[:half]
        late = after.iloc[half:]
        atr_early = float((early["high"] - early["low"]).mean())
        atr_late = float((late["high"] - late["low"]).mean())
        if atr_early > 0:
            ratio = atr_late / atr_early
            if ratio < 1.0:
                vc_score = min(100.0, (1.0 - ratio) * 200.0)

    # Moneyflow net during washout
    mf_net_yi = None
    if mf_df is not None and not mf_df.empty:
        col = (
            "net_mf_amount"
            if "net_mf_amount" in mf_df.columns
            else ("net_amount" if "net_amount" in mf_df.columns else None)
        )
        if col is not None:
            sub = mf_df[mf_df["trade_date"] > probe_date]
            if not sub.empty:
                mf_net_yi = float(sub[col].sum()) / 10000.0

    # Component scoring
    dd_score = max(0.0, 100.0 - drawdown_pct * (100.0 / cfg.max_post_probe_drawdown_pct))
    shrink_score = (
        100.0 if shrink_ratio <= cfg.post_probe_volume_shrink_ratio_max
        else max(0.0, 100.0 - (shrink_ratio - cfg.post_probe_volume_shrink_ratio_max) * 200.0)
    )
    support_score = 100.0
    if low_broken:
        support_score -= 50.0
    if ma20_breaks:
        support_score -= 20.0
    if ma60_breaks:
        support_score -= 30.0
    support_score = max(0.0, support_score)

    # Time score: peak inside [washout_min, washout_max]
    if cfg.washout_min_trade_days <= washout_days <= cfg.washout_max_trade_days:
        time_score = 100.0
    elif washout_days < cfg.washout_min_trade_days:
        time_score = max(0.0, washout_days / cfg.washout_min_trade_days * 100.0)
    else:
        excess = washout_days - cfg.washout_max_trade_days
        time_score = max(0.0, 100.0 - excess * 5.0)

    mf_keep_score = 50.0
    if mf_net_yi is not None:
        mf_keep_score = max(0.0, min(100.0, 50.0 + mf_net_yi * 20.0))

    score = (
        0.25 * dd_score
        + 0.25 * shrink_score
        + 0.20 * support_score
        + 0.15 * time_score
        + 0.15 * mf_keep_score
    )

    return {
        "washout_days": int(washout_days),
        "post_probe_max_drawdown_pct": round(drawdown_pct, 2),
        "post_probe_volume_shrink_ratio": round(shrink_ratio, 3),
        "post_probe_low_broken": bool(low_broken),
        "post_probe_ma20_broken": bool(ma20_breaks),
        "post_probe_ma60_broken": bool(ma60_breaks),
        "post_probe_moneyflow_net_yi": (
            round(mf_net_yi, 4) if isinstance(mf_net_yi, float) else mf_net_yi
        ),
        "washout_volatility_compression": round(vc_score, 2),
        "washout_score": _clip_score(score),
    }


# ---------------------------------------------------------------------------
# compute_launch_setup — re-strengthening / breakout-prep
# ---------------------------------------------------------------------------


def compute_launch_setup(
    window_df: pd.DataFrame,
    probe_info: dict[str, Any] | None,
    washout_info: dict[str, Any],
    cfg: ApwConfig,
    *,
    index_df: pd.DataFrame | None = None,
    mf_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score launch-readiness at the latest row in window_df.

    ``mf_df`` (moneyflow) is optional — when supplied the net inflow over the
    last ``cfg.launch_moneyflow_days`` trade days (亿元, from
    ``net_mf_amount`` / ``net_amount`` in 万元) is attached as
    ``current_moneyflow_net_yi`` and folded into ``capital_score``; otherwise
    that field stays ``None`` and ``capital_score`` falls back to a neutral 50
    (round-2 P2 — was previously hard-coded ``None`` regardless of input).
    """
    if window_df.empty:
        return {
            "launch_setup_score": 0.0,
            "close_to_probe_high_pct": 0.0,
            "break_probe_high": False,
            "break_washout_high": False,
            "current_volume_ratio_5d": 0.0,
            "current_volume_ratio_20d": 0.0,
            "current_moneyflow_net_yi": None,
            "above_ma5": False,
            "above_ma10": False,
            "above_ma20": False,
            "relative_strength_20d": None,
            "sector_strength_score": None,
        }

    df = window_df.sort_values("trade_date").reset_index(drop=True)
    closes = df["close"].astype(float)
    ma5 = _ma(closes, 5)
    ma10 = _ma(closes, 10)
    ma20 = _ma(closes, 20)

    last = df.iloc[-1]
    last_close = float(last["close"])
    last_pos = len(df) - 1

    prev5 = df.iloc[max(0, last_pos - 5) : last_pos]
    prev20 = df.iloc[max(0, last_pos - 20) : last_pos]
    avg5 = float(prev5["vol"].mean()) if not prev5.empty else 0.0
    avg20 = float(prev20["vol"].mean()) if not prev20.empty else 0.0
    cur_vol = float(last["vol"])
    vr5 = cur_vol / avg5 if avg5 > 0 else 0.0
    vr20 = cur_vol / avg20 if avg20 > 0 else 0.0

    probe_high = float(probe_info.get("probe_high", 0.0)) if probe_info else 0.0
    close_to_probe_high_pct = (
        ((last_close - probe_high) / probe_high) * 100.0 if probe_high > 0 else 0.0
    )
    break_probe_high = bool(last_close > probe_high) if probe_high > 0 else False

    # washout high = max high between probe_date+1 and last - 1 (exclusive of
    # the current trade day). Including the current day would make
    # break_washout_high almost impossible to flip True, because last_high is
    # >= last_close (P1-2 in code review).
    washout_high = 0.0
    if probe_info and washout_info.get("washout_days", 0) > 0:
        probe_date = probe_info["probe_date"]
        pp = df.index[df["trade_date"] == probe_date]
        if len(pp) > 0:
            after = df.iloc[int(pp[0]) + 1 : last_pos]
            if not after.empty:
                washout_high = float(after["high"].max())
    break_washout_high = bool(last_close > washout_high) if washout_high > 0 else False

    above5 = bool(last_close > float(ma5.iloc[-1]))
    above10 = bool(last_close > float(ma10.iloc[-1]))
    above20 = bool(last_close > float(ma20.iloc[-1]))

    # relative strength: 20d return of stock minus baseline index (HS300 by
    # default — see cfg.baseline_index_code). When the caller doesn't supply
    # index_df we return None and rs_score falls back to neutral 50 so the
    # absolute return doesn't masquerade as a relative one (P2-1 in code
    # review).
    rs20: float | None = None
    if len(df) >= 21:
        start_close = float(df.iloc[-21]["close"])
        if start_close > 0:
            stock_ret = (last_close - start_close) / start_close * 100.0
            if index_df is not None and not index_df.empty and "close" in index_df.columns:
                idx = index_df.sort_values("trade_date").reset_index(drop=True)
                if len(idx) >= 21:
                    idx_start = float(idx.iloc[-21]["close"])
                    idx_last = float(idx.iloc[-1]["close"])
                    if idx_start > 0:
                        idx_ret = (idx_last - idx_start) / idx_start * 100.0
                        rs20 = stock_ret - idx_ret

    # Component scoring (source §6.3 — 20% each across 5 buckets)
    vol_score = max(0.0, min(100.0, (vr5 - 1.0) * 60.0))
    ma_strength_score = (40 if above5 else 0) + (30 if above10 else 0) + (30 if above20 else 0)
    near_probe_score = 0.0
    if probe_high > 0:
        if break_probe_high:
            near_probe_score = 100.0
        else:
            gap = -close_to_probe_high_pct  # positive gap from below
            near_probe_score = max(0.0, 100.0 - gap * 10.0)
    # capital_score from moneyflow over the last ``launch_moneyflow_days``
    # sessions. mf_df missing / empty / column-less → neutral 50 and the field
    # stays None so the runner's missing_data aggregator can flag it (acc-level
    # 'moneyflow' tag covers per-stock empty mf_df).
    current_mf_net_yi: float | None = None
    if mf_df is not None and not mf_df.empty:
        col = (
            "net_mf_amount"
            if "net_mf_amount" in mf_df.columns
            else ("net_amount" if "net_amount" in mf_df.columns else None)
        )
        if col is not None:
            recent_mf = mf_df.sort_values("trade_date").tail(
                max(1, cfg.launch_moneyflow_days)
            )
            if not recent_mf.empty:
                current_mf_net_yi = round(
                    float(recent_mf[col].astype(float).sum()) / 10000.0, 4
                )
    if current_mf_net_yi is None:
        capital_score = 50.0
    else:
        # Same scaling as compute_washout's mf_keep_score so the two dimensions
        # share calibration: ±2.5 亿 net flow ≈ ±50 score points.
        capital_score = max(0.0, min(100.0, 50.0 + current_mf_net_yi * 20.0))
    # rs20 None → neutral 50 (no index data to score against).
    rs_score = (
        max(0.0, min(100.0, 50.0 + rs20 * 2.0)) if rs20 is not None else 50.0
    )

    score = (
        0.20 * vol_score
        + 0.20 * ma_strength_score
        + 0.20 * near_probe_score
        + 0.20 * capital_score
        + 0.20 * rs_score
    )

    return {
        "launch_setup_score": _clip_score(score),
        "close_to_probe_high_pct": round(close_to_probe_high_pct, 2),
        "break_probe_high": break_probe_high,
        "break_washout_high": break_washout_high,
        "current_volume_ratio_5d": round(vr5, 3),
        "current_volume_ratio_20d": round(vr20, 3),
        "current_moneyflow_net_yi": current_mf_net_yi,
        "above_ma5": above5,
        "above_ma10": above10,
        "above_ma20": above20,
        "relative_strength_20d": round(rs20, 2) if rs20 is not None else None,
        "sector_strength_score": None,
    }


# ---------------------------------------------------------------------------
# derive_phase — state machine
# ---------------------------------------------------------------------------


def derive_phase(
    accumulation: dict[str, Any],
    probe: dict[str, Any] | None,
    washout: dict[str, Any],
    launch: dict[str, Any],
    cfg: ApwConfig,
) -> APWPhase:
    """Apply the state-machine rules from the spec (kept in sync with §3.2)."""
    acc_score = accumulation.get("accumulation_score", 0.0)
    if acc_score < cfg.accumulation_score_min:
        return APWPhase.NO_SETUP

    if probe is None:
        return APWPhase.ACCUMULATING

    if probe.get("probe_quality_score", 0.0) < cfg.probe_quality_score_min:
        return APWPhase.ACCUMULATING

    wash_ok = (
        washout.get("washout_score", 0.0) >= cfg.washout_score_min
        and not washout.get("post_probe_low_broken", True)
        and cfg.washout_min_trade_days
        <= washout.get("washout_days", 0)
        <= cfg.washout_max_trade_days
    )
    if not wash_ok:
        return APWPhase.PROBE_SEEN

    launch_ok = (
        launch.get("launch_setup_score", 0.0) >= cfg.launch_setup_score_min
        and launch.get("above_ma5", False)
        and launch.get("above_ma10", False)
        and launch.get("current_volume_ratio_5d", 0.0) >= cfg.launch_current_volume_ratio_5d_min
    )
    if launch_ok:
        return APWPhase.LAUNCH_READY
    return APWPhase.WASHING_AFTER_PROBE


# ---------------------------------------------------------------------------
# v0.4.0 — derived features that feed LightGBM (PR-3) and (some) LLM prompt
# ---------------------------------------------------------------------------


def compute_vcp_features(window_df: pd.DataFrame) -> dict[str, Any]:
    """ATR / BBW compression measures over the last 60 trade days.

    Mirrors the VA contract semantically but each implementation is owned
    locally so APW does not depend on volume_anomaly (PR-5 deletes VA).

    Returns NaN values when ``window_df`` lacks the required history; the
    LightGBM booster handles NaN natively (no special routing needed).
    """
    out: dict[str, Any] = {
        "atr_10d": None,
        "atr_10d_pct": None,
        "atr_10d_quantile_in_60d": None,
        "bbw_20d": None,
        "bbw_compression_ratio": None,
    }
    if window_df is None or window_df.empty:
        return out
    df = window_df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < 12:
        return out

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    prev_close = closes.shift(1)
    tr = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean()
    last_close = float(closes.iloc[-1])
    last_atr = float(atr.iloc[-1]) if not math.isnan(atr.iloc[-1]) else None
    if last_atr is not None and last_close > 0:
        out["atr_10d"] = round(last_atr, 4)
        out["atr_10d_pct"] = round(last_atr / last_close * 100.0, 4)
    # Quantile of the latest ATR within the last 60 ATR observations.
    window = atr.dropna().tail(60)
    if last_atr is not None and len(window) >= 20:
        rank = float((window <= last_atr).sum()) / float(len(window))
        out["atr_10d_quantile_in_60d"] = round(rank, 4)

    # Bollinger Band Width (20d) = (UB - LB) / MB = 4σ / mean(20d)
    if len(df) >= 20:
        roll_mean = closes.rolling(20).mean()
        roll_std = closes.rolling(20).std(ddof=0)
        mid = float(roll_mean.iloc[-1]) if not math.isnan(roll_mean.iloc[-1]) else None
        sd = float(roll_std.iloc[-1]) if not math.isnan(roll_std.iloc[-1]) else None
        if mid is not None and sd is not None and mid > 0:
            bbw = 4.0 * sd / mid
            out["bbw_20d"] = round(bbw, 4)
            # Compression ratio = current BBW / median BBW over the last 60
            # 20-day windows; values < 1 indicate compression vs baseline.
            bbw_series = (4.0 * roll_std / roll_mean).dropna().tail(60)
            if len(bbw_series) >= 20:
                median_bbw = float(bbw_series.median())
                if median_bbw > 0:
                    out["bbw_compression_ratio"] = round(bbw / median_bbw, 4)
    return out


def compute_long_range_features(window_df: pd.DataFrame) -> dict[str, Any]:
    """Distance to 120 / 250-day highs and position within the 120-day range."""
    out: dict[str, Any] = {
        "dist_to_120d_high_pct": None,
        "dist_to_250d_high_pct": None,
        "is_above_120d_high": None,
        "is_above_250d_high": None,
        "pos_in_120d_range": None,
    }
    if window_df is None or window_df.empty:
        return out
    df = window_df.sort_values("trade_date").reset_index(drop=True)
    last_close = float(df.iloc[-1]["close"]) if len(df) else 0.0
    if last_close <= 0:
        return out

    # Compare to the high of the PRIOR 120 / 250 sessions (excluding today),
    # so ``is_above_120d_high`` can legitimately flip True the moment today's
    # close pierces the historical resistance.
    if len(df) >= 121:
        sub = df.iloc[-121:-1]
        h120 = float(sub["high"].max())
        l120 = float(sub["low"].min())
        if h120 > 0:
            out["dist_to_120d_high_pct"] = round((last_close - h120) / h120 * 100.0, 4)
            out["is_above_120d_high"] = last_close > h120
        if h120 > l120:
            out["pos_in_120d_range"] = round((last_close - l120) / (h120 - l120), 4)

    if len(df) >= 251:
        sub = df.iloc[-251:-1]
        h250 = float(sub["high"].max())
        if h250 > 0:
            out["dist_to_250d_high_pct"] = round((last_close - h250) / h250 * 100.0, 4)
            out["is_above_250d_high"] = last_close > h250

    return out


def compute_alpha_features(
    window_df: pd.DataFrame,
    index_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Multi-horizon excess return vs ``baseline_index_code``.

    All three windows (5d / 20d / 60d) require ``len(window_df) > N`` AND a
    matching index frame; missing data → None and the feature stays NaN.
    ``alpha_leading`` is a coarse 3-state label (LEADING / NEUTRAL / LAGGING)
    derived from the 20d alpha for prompt readability.
    """
    out: dict[str, Any] = {
        "alpha_5d_pct": None,
        "alpha_20d_pct": None,
        "alpha_60d_pct": None,
        "alpha_leading": None,
    }
    if window_df is None or window_df.empty:
        return out
    df = window_df.sort_values("trade_date").reset_index(drop=True)
    if index_df is None or index_df.empty or "close" not in index_df.columns:
        return out
    idx = index_df.sort_values("trade_date").reset_index(drop=True)
    last_close = float(df.iloc[-1]["close"])
    last_idx = float(idx.iloc[-1]["close"]) if len(idx) else 0.0
    if last_close <= 0 or last_idx <= 0:
        return out

    for horizon, key in ((5, "alpha_5d_pct"), (20, "alpha_20d_pct"), (60, "alpha_60d_pct")):
        if len(df) < horizon + 1 or len(idx) < horizon + 1:
            continue
        start_close = float(df.iloc[-horizon - 1]["close"])
        idx_start = float(idx.iloc[-horizon - 1]["close"])
        if start_close <= 0 or idx_start <= 0:
            continue
        stock_ret = (last_close - start_close) / start_close * 100.0
        idx_ret = (last_idx - idx_start) / idx_start * 100.0
        out[key] = round(stock_ret - idx_ret, 4)

    a20 = out["alpha_20d_pct"]
    if a20 is not None:
        if a20 >= 5.0:
            out["alpha_leading"] = "LEADING"
        elif a20 <= -5.0:
            out["alpha_leading"] = "LAGGING"
        else:
            out["alpha_leading"] = "NEUTRAL"
    return out


def compute_volume_event_score(window_df: pd.DataFrame) -> float | None:
    """One-shot rating of T-day volume anomaly (0–100).

    Captures the gist of the legacy volume-anomaly screen (T-day yang body +
    volume ratio + amplitude) as a single auxiliary feature for APW LGB. Not
    used to filter candidates — it's a number on the candidate so the LLM and
    LGB can weigh it however they want.

    Components:
        * ``vol_ratio_5d`` — current vol / mean(prev 5 vol); clamped to 1..5
        * ``body_ratio`` — |close-open| / (high-low); 0..1
        * ``amplitude`` — (high-low) / prev_close * 100; clamped to 0..20%
    """
    if window_df is None or window_df.empty:
        return None
    df = window_df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < 6:
        return None
    prev_5 = df.iloc[-6:-1]
    if prev_5.empty:
        return None
    avg_vol = float(prev_5["vol"].mean())
    cur = df.iloc[-1]
    if avg_vol <= 0:
        return None

    vol_ratio_5d = float(cur["vol"]) / avg_vol
    range_ = max(float(cur["high"]) - float(cur["low"]), 1e-9)
    body = abs(float(cur["close"]) - float(cur.get("open", cur["close"])))
    body_ratio = min(1.0, body / range_)
    prev_close = float(df.iloc[-2]["close"])
    amplitude_pct = (
        (float(cur["high"]) - float(cur["low"])) / prev_close * 100.0
        if prev_close > 0
        else 0.0
    )

    vol_score = max(0.0, min(100.0, (min(vol_ratio_5d, 5.0) - 1.0) * 25.0))
    body_score = body_ratio * 100.0
    amp_score = max(0.0, min(100.0, amplitude_pct / 20.0 * 100.0))

    return round(0.5 * vol_score + 0.25 * body_score + 0.25 * amp_score, 2)


def compute_ma_distances(window_df: pd.DataFrame) -> dict[str, Any]:
    """Distance (%) between last close and MA5/10/20/60 + the MA60 value.

    The MA60 value is also surfaced standalone so the v0.3.0 ``prune`` rules
    (``close < MA60``) can consume it from the candidate JSON without a
    second pass.
    """
    out: dict[str, Any] = {
        "ma5": None,
        "ma10": None,
        "ma20": None,
        "ma60": None,
        "close_to_ma5_pct": None,
        "close_to_ma10_pct": None,
        "close_to_ma20_pct": None,
        "close_to_ma60_pct": None,
    }
    if window_df is None or window_df.empty:
        return out
    df = window_df.sort_values("trade_date").reset_index(drop=True)
    closes = df["close"].astype(float)
    last_close = float(closes.iloc[-1]) if len(closes) else 0.0
    if last_close <= 0:
        return out

    for n, key, key_pct in (
        (5, "ma5", "close_to_ma5_pct"),
        (10, "ma10", "close_to_ma10_pct"),
        (20, "ma20", "close_to_ma20_pct"),
        (60, "ma60", "close_to_ma60_pct"),
    ):
        if len(closes) < n:
            continue
        ma = float(closes.tail(n).mean())
        out[key] = round(ma, 4)
        if ma > 0:
            out[key_pct] = round((last_close - ma) / ma * 100.0, 4)
    return out


# ---------------------------------------------------------------------------
# pack_candidate — assemble the LLM-input dict
# ---------------------------------------------------------------------------


def pack_candidate(
    *,
    trade_date: str,
    ts_code: str,
    name: str,
    phase: APWPhase,
    basic: dict[str, Any],
    accumulation: dict[str, Any],
    probe: dict[str, Any] | None,
    washout: dict[str, Any],
    launch: dict[str, Any],
    sector_strength_score: float | None = None,
    missing_data: list[str] | None = None,
    # v0.4.0 — derived feature bundles. Defaults to ``None`` so the M2/M3
    # callers (which do not yet compute these) keep working byte-identically
    # to v0.3.0; the screen funnel passes the dicts in once the helpers are
    # wired up in ``runner.py``.
    vcp: dict[str, Any] | None = None,
    long_range: dict[str, Any] | None = None,
    alpha: dict[str, Any] | None = None,
    ma_distances: dict[str, Any] | None = None,
    volume_event_score: float | None = None,
    limit_up_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the flat candidate dict sent to the LLM (spec §8)."""
    candidate_id = f"{trade_date}_{ts_code}"
    out: dict[str, Any] = {
        "candidate_id": candidate_id,
        "ts_code": ts_code,
        "name": name,
        "trade_date": trade_date,
        "phase": phase.value,
    }
    out.update(
        {
            "listed_days": basic.get("listed_days"),
            "close": basic.get("close"),
            "pct_chg": basic.get("pct_chg"),
            "turnover_rate": basic.get("turnover_rate"),
            "amount_yi": basic.get("amount_yi"),
            "circ_mv_yi": basic.get("circ_mv_yi"),
        }
    )
    # Drop internal-only keys before exposing to LLM
    acc_keys = (
        "accumulation_score",
        "accumulation_days",
        "accumulation_net_mf_yi",
        "accumulation_price_change_pct",
        "low_position_score",
    )
    for k in acc_keys:
        out[k] = accumulation.get(k)

    if probe is not None:
        probe_keys = (
            "probe_date",
            "probe_days_ago",
            "probe_volume_ratio_5d",
            "probe_volume_ratio_20d",
            "probe_volume_rank_pct_60d",
            "probe_amount_ratio_20d",
            "probe_turnover_rate",
            "probe_amplitude_pct",
            "probe_upper_shadow_ratio",
            "probe_body_ratio",
            "probe_pct_chg",
            "probe_moneyflow_net_yi",
            "probe_quality_score",
        )
        for k in probe_keys:
            out[k] = probe.get(k)
    else:
        out["probe_date"] = None
        out["probe_days_ago"] = None
        out["probe_quality_score"] = None

    wash_keys = (
        "washout_days",
        "post_probe_max_drawdown_pct",
        "post_probe_volume_shrink_ratio",
        "post_probe_low_broken",
        "post_probe_ma20_broken",
        "post_probe_ma60_broken",
        "post_probe_moneyflow_net_yi",
        "washout_volatility_compression",
        "washout_score",
    )
    for k in wash_keys:
        out[k] = washout.get(k)

    launch_keys = (
        "launch_setup_score",
        "close_to_probe_high_pct",
        "break_probe_high",
        "break_washout_high",
        "current_volume_ratio_5d",
        "current_volume_ratio_20d",
        "current_moneyflow_net_yi",
        "above_ma5",
        "above_ma10",
        "above_ma20",
        "relative_strength_20d",
    )
    for k in launch_keys:
        out[k] = launch.get(k)

    out["sector_strength_score"] = sector_strength_score

    # v0.4.0 — extended derived features. Keys are emitted unconditionally
    # (None when the helper wasn't run) so feature_frame builders can rely on
    # a stable schema.
    vcp = vcp or {}
    for k in (
        "atr_10d",
        "atr_10d_pct",
        "atr_10d_quantile_in_60d",
        "bbw_20d",
        "bbw_compression_ratio",
    ):
        out[k] = vcp.get(k)

    long_range = long_range or {}
    for k in (
        "dist_to_120d_high_pct",
        "dist_to_250d_high_pct",
        "is_above_120d_high",
        "is_above_250d_high",
        "pos_in_120d_range",
    ):
        out[k] = long_range.get(k)

    alpha = alpha or {}
    for k in (
        "alpha_5d_pct",
        "alpha_20d_pct",
        "alpha_60d_pct",
        "alpha_leading",
    ):
        out[k] = alpha.get(k)

    ma_distances = ma_distances or {}
    for k in (
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "close_to_ma5_pct",
        "close_to_ma10_pct",
        "close_to_ma20_pct",
        "close_to_ma60_pct",
    ):
        out[k] = ma_distances.get(k)

    out["volume_event_score"] = volume_event_score

    limit_up_history = limit_up_history or {}
    out["prior_limit_up_count_60d"] = limit_up_history.get(
        "prior_limit_up_count_60d"
    )
    out["days_since_last_limit_up"] = limit_up_history.get(
        "days_since_last_limit_up"
    )

    # Surface probe_low alongside the existing probe dict so the v0.3.0 prune
    # rules (close < probe_low) can read it directly off the candidate JSON.
    if probe is not None and "probe_low" in probe and "probe_low" not in out:
        out["probe_low"] = probe.get("probe_low")
    if probe is not None and "probe_high" in probe and "probe_high" not in out:
        out["probe_high"] = probe.get("probe_high")

    out["risk_flags_local"] = []
    out["missing_data"] = list(missing_data or [])
    return out


# ---------------------------------------------------------------------------
# Tushare thin wrappers — kept narrow so unit tests can monkeypatch.
# ---------------------------------------------------------------------------


@dataclass
class FetchOutcome:
    df: pd.DataFrame
    missing: list[str] = field(default_factory=list)


def _try_optional(tushare: Any, api_name: str, **kwargs: Any) -> FetchOutcome:
    """Call an optional Tushare API; transient failure ⇒ empty df + missing tag."""
    try:
        from deeptrade.core.tushare_client import (  # noqa: PLC0415
            TushareRateLimitError,
            TushareServerError,
            TushareUnauthorizedError,
        )
    except ImportError:  # pragma: no cover - framework always supplies these
        TushareRateLimitError = TushareServerError = TushareUnauthorizedError = Exception

    try:
        df = tushare.call(api_name, **kwargs)
        if df is None:
            return FetchOutcome(pd.DataFrame(), [api_name])
        return FetchOutcome(df, [])
    except (TushareRateLimitError, TushareServerError, TushareUnauthorizedError) as exc:
        logger.warning("optional tushare api %s failed: %s", api_name, exc)
        return FetchOutcome(pd.DataFrame(), [api_name])
    except Exception as exc:  # noqa: BLE001
        logger.warning("unexpected error from tushare api %s: %s", api_name, exc)
        return FetchOutcome(pd.DataFrame(), [api_name])


def fetch_stock_basic(tushare: Any) -> pd.DataFrame:
    """Pull stock_basic (required API)."""
    return tushare.call(
        "stock_basic",
        list_status="L",
        fields="ts_code,name,industry,market,exchange,list_status,list_date",
    )


def fetch_trade_cal(tushare: Any, *, start: str, end: str) -> pd.DataFrame:
    return tushare.call("trade_cal", exchange="SSE", start_date=start, end_date=end)


def fetch_daily(tushare: Any, *, ts_codes: list[str], start: str, end: str) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    # 单次 query 上限 ~6000 rows；按需要分片由 caller 处理（runner 控制）
    return tushare.call(
        "daily",
        ts_code=",".join(ts_codes),
        start_date=start,
        end_date=end,
    )


def fetch_daily_basic(tushare: Any, *, ts_codes: list[str], start: str, end: str) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    return tushare.call(
        "daily_basic",
        ts_code=",".join(ts_codes),
        start_date=start,
        end_date=end,
    )


def fetch_moneyflow(tushare: Any, *, ts_codes: list[str], start: str, end: str) -> FetchOutcome:
    """Optional API. Returns FetchOutcome to track missing-data degradation."""
    if not ts_codes:
        return FetchOutcome(pd.DataFrame(), [])
    return _try_optional(
        tushare, "moneyflow", ts_code=",".join(ts_codes), start_date=start, end_date=end
    )


def fetch_st_codes(tushare: Any, *, trade_date: str) -> set[str]:
    try:
        df = tushare.call("stock_st", trade_date=trade_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("stock_st on %s failed: %s", trade_date, exc)
        return set()
    if df is None or df.empty or "ts_code" not in df.columns:
        return set()
    return set(df["ts_code"].astype(str).tolist())


def fetch_suspended_codes(tushare: Any, *, trade_date: str) -> set[str]:
    outcome = _try_optional(tushare, "suspend_d", trade_date=trade_date, suspend_type="S")
    if outcome.df is None or outcome.df.empty or "ts_code" not in outcome.df.columns:
        return set()
    return set(outcome.df["ts_code"].astype(str).tolist())


def fetch_index_daily(tushare: Any, *, index_code: str, start: str, end: str) -> FetchOutcome:
    return _try_optional(tushare, "index_daily", ts_code=index_code, start_date=start, end_date=end)


# ---------------------------------------------------------------------------
# M5 — realized-returns helpers
# ---------------------------------------------------------------------------


def fetch_realized_prices(
    tushare: Any,
    *,
    ts_codes: list[str],
    signal_date: str,
    max_horizon_calendar_days: int = 30,
) -> pd.DataFrame:
    """Pull T..T+max trade days of close/high/low for each ts_code.

    Caller does the per-row close-T₁/T₃/T₅/T₁₀ slicing — this just returns the
    raw window. Returns empty frame if the request is empty.
    """
    if not ts_codes:
        return pd.DataFrame()
    # Tushare daily takes start_date / end_date in YYYYMMDD; we widen by
    # max_horizon_calendar_days to give enough trade days for T+10.
    end_dt = datetime.strptime(signal_date, "%Y%m%d") + timedelta(
        days=max_horizon_calendar_days
    )
    end_str = end_dt.strftime("%Y%m%d")
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(ts_codes), 50):
        sub = ts_codes[i : i + 50]
        df = tushare.call(
            "daily",
            ts_code=",".join(sub),
            start_date=signal_date,
            end_date=end_str,
        )
        if df is not None and not df.empty:
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def compute_returns_and_labels(
    *,
    ts_code: str,
    signal_date: str,
    quotes: pd.DataFrame,
    label_t5_high_return_pct: float = 8.0,
    label_t5_max_drawdown_pct: float = 8.0,
    label_t10_high_return_pct: float = 12.0,
    label_t10_max_drawdown_pct: float = 10.0,
) -> dict[str, Any]:
    """Compute T+N returns + binary labels for one (ts_code, signal_date).

    Returns dict with status='complete' when enough trade days exist (T+10),
    or status='partial' when fewer. Missing-data signals stay non-fatal.
    """
    if quotes is None or quotes.empty:
        return {"data_status": "missing", "ts_code": ts_code, "signal_date": signal_date}

    sub = quotes[quotes["ts_code"] == ts_code].copy()
    sub["trade_date"] = sub["trade_date"].astype(str)
    sub = sub.sort_values("trade_date").reset_index(drop=True)
    # First trade_date >= signal_date is "T".
    sub = sub[sub["trade_date"] >= signal_date].reset_index(drop=True)
    if sub.empty:
        return {"data_status": "missing", "ts_code": ts_code, "signal_date": signal_date}

    close_t = float(sub.iloc[0]["close"])

    def _at(offset: int) -> float | None:
        if offset < len(sub):
            return float(sub.iloc[offset]["close"])
        return None

    close_t1 = _at(1)
    close_t3 = _at(3)
    close_t5 = _at(5)
    close_t10 = _at(10)

    def _pct(after: float | None) -> float | None:
        if after is None or close_t <= 0:
            return None
        return (after - close_t) / close_t * 100.0

    ret_t1 = _pct(close_t1)
    ret_t3 = _pct(close_t3)
    ret_t5 = _pct(close_t5)
    ret_t10 = _pct(close_t10)

    # Max-high / drawdown over T+1..T+N. Only return values when we have
    # enough rows to cover the full horizon — partial windows would mislead.
    def _max_high_dd(end_offset: int) -> tuple[float | None, float | None]:
        if len(sub) <= end_offset:
            return None, None
        rows = sub.iloc[1 : end_offset + 1]
        if rows.empty:
            return None, None
        max_high = float(rows["high"].max())
        min_low = float(rows["low"].min())
        max_high_pct = (max_high - close_t) / close_t * 100.0 if close_t > 0 else None
        max_dd_pct = (close_t - min_low) / close_t * 100.0 if close_t > 0 else None
        return max_high_pct, max_dd_pct

    max_high_t5_pct, max_dd_t5_pct = _max_high_dd(5)
    max_high_t10_pct, max_dd_t10_pct = _max_high_dd(10)

    def _label(high_pct: float | None, dd_pct: float | None,
               high_thresh: float, dd_thresh: float) -> int | None:
        if high_pct is None or dd_pct is None:
            return None
        return 1 if (high_pct >= high_thresh and dd_pct <= dd_thresh) else 0

    label_t5 = _label(max_high_t5_pct, max_dd_t5_pct,
                      label_t5_high_return_pct, label_t5_max_drawdown_pct)
    label_t10 = _label(max_high_t10_pct, max_dd_t10_pct,
                       label_t10_high_return_pct, label_t10_max_drawdown_pct)

    has_t10 = close_t10 is not None
    status = "complete" if has_t10 else "partial"
    return {
        "ts_code": ts_code,
        "signal_date": signal_date,
        "close_t": close_t,
        "close_t1": close_t1,
        "close_t3": close_t3,
        "close_t5": close_t5,
        "close_t10": close_t10,
        "ret_t1_pct": _round(ret_t1),
        "ret_t3_pct": _round(ret_t3),
        "ret_t5_pct": _round(ret_t5),
        "ret_t10_pct": _round(ret_t10),
        "max_high_t5_pct": _round(max_high_t5_pct),
        "max_high_t10_pct": _round(max_high_t10_pct),
        "max_drawdown_t5_pct": _round(max_dd_t5_pct),
        "max_drawdown_t10_pct": _round(max_dd_t10_pct),
        "label_launch_t5": label_t5,
        "label_launch_t10": label_t10,
        "data_status": status,
    }


def _round(v: float | None, ndigits: int = 2) -> float | None:
    return None if v is None else round(v, ndigits)
