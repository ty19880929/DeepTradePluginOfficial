"""Daily universe construction for Checkmate.

Inputs (read via ``CheckmateRuntime``):
  * ``checkmate_stock_status_history`` — survivorship snapshot keyed by
    (ts_code, as_of_date). Populated by ``sync``.
  * Per-ts_code parquet caches under ``cache/daily/`` and ``cache/daily_basic/``.

Output:
  * :class:`UniverseSnapshot` — one row per ts_code that has any status
    history at or before ``trade_date``, carrying eligibility flag,
    reason_codes list (failed rules), and a numeric ``liquidity_score``.

Reason codes (development_plan §14.1 + iteration_tasks.md §2 PR-1.3):
  * ``"st"`` — current ``is_st`` flag is True
  * ``"new_listing"`` — list_date within ``cfg.listed_days_min`` cal-days
  * ``"thin_trading"`` — fewer than ``cfg.thin_trading_min_days`` actual
    daily rows in the trailing 20-session window (suspensions count here)
  * ``"low_amount"`` — 20-day avg ``amount`` below the liquidity floor
  * ``"one_way_limit"`` — fraction of 一字 days (high == low) exceeds the
    ``cfg.one_way_limit_max_ratio`` threshold
  * ``"price_band"`` — last close outside the configured band

Delisted (``list_status`` ∈ {D, P}) names are silently dropped before
reason_code evaluation — they're not part of the tradable universe at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from . import data
from .calendar import load_trade_calendar
from .config import UniverseConfig
from .runtime import CheckmateRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class UniverseRow:
    ts_code: str
    eligible: bool
    reason_codes: list[str]
    liquidity_score: float
    amount_20d_avg: float            # in yuan (Tushare's 千元 normalised)
    turnover_20d_avg: float          # percent (Tushare daily_basic field)
    list_status: str
    is_st: bool
    name: str
    industry: str | None


@dataclass
class UniverseSnapshot:
    trade_date: str
    rows: list[UniverseRow] = field(default_factory=list)

    @property
    def eligible(self) -> list[UniverseRow]:
        return [r for r in self.rows if r.eligible]

    @property
    def excluded(self) -> list[UniverseRow]:
        return [r for r in self.rows if not r.eligible]

    def reason_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            for code in r.reason_codes:
                out[code] = out.get(code, 0) + 1
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calendar_days_between(start_yyyymmdd: str, end_yyyymmdd: str) -> int:
    """Inclusive calendar-day diff. Returns 0 if either bound is malformed."""
    try:
        a = datetime.strptime(start_yyyymmdd, "%Y%m%d")
        b = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    except ValueError:
        return 0
    return max(0, (b - a).days)


def _candidate_ts_codes(db: Any) -> list[str]:
    """All distinct ts_codes that have any status history row.

    Sorted ascending for deterministic ordering downstream (tests rely on
    stable iteration so :meth:`UniverseSnapshot.rows` is reproducible across
    runs without depending on Python dict / DB row order).
    """
    rows = db.execute(
        "SELECT DISTINCT ts_code FROM checkmate_stock_status_history ORDER BY ts_code"
    ).fetchall()
    return [r[0] for r in rows]


def _window_start(rt: CheckmateRuntime, trade_date: str, n_sessions: int) -> str:
    """Return the YYYYMMDD that's ``n_sessions`` open days before ``trade_date``.

    If the loaded trade calendar doesn't cover that far back, falls back to a
    calendar-day approximation (``n_sessions * 1.5`` cal-days) — the daily /
    daily_basic fetchers cache by trade_date so a slightly wider window is
    harmless.
    """
    try:
        cal = load_trade_calendar(rt.tushare)
        return cal.n_sessions_before(trade_date, n_sessions)
    except (ValueError, RuntimeError):
        # Defensive fallback so tests with sparse fixtures still produce a
        # plausible window.
        approx_days = int(n_sessions * 1.5) + 1
        anchor = datetime.strptime(trade_date, "%Y%m%d")
        return (anchor - pd.Timedelta(days=approx_days)).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Core: build_universe
# ---------------------------------------------------------------------------


def build_universe(
    rt: CheckmateRuntime,
    trade_date: str,
    cfg: UniverseConfig | None = None,
) -> UniverseSnapshot:
    """Construct the daily tradable universe for ``trade_date``.

    See module docstring for the rule set + reason_codes.
    """
    cfg = cfg or UniverseConfig()
    snapshot = UniverseSnapshot(trade_date=trade_date)

    window_start = _window_start(rt, trade_date, cfg.window_sessions)

    for ts_code in _candidate_ts_codes(rt.db):
        status = data.query_status_as_of(rt.db, ts_code, trade_date)
        if status is None:
            continue  # Not yet listed at trade_date
        if status["list_status"] in {"D", "P"}:
            continue  # Delisted / paused — not in universe at all

        reasons: list[str] = []

        # --- ST gate
        if status.get("is_st"):
            reasons.append("st")

        # --- new listing gate
        list_date = status.get("list_date")
        if list_date:
            listed_days = _calendar_days_between(str(list_date), trade_date)
            if listed_days < cfg.listed_days_min:
                reasons.append("new_listing")

        # --- 20-session daily_basic stats
        try:
            db_window = data.fetch_daily_basic(
                rt.tushare, ts_code, window_start, trade_date,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("daily_basic fetch failed for %s: %s", ts_code, exc)
            db_window = pd.DataFrame()

        n_actual = len(db_window) if not db_window.empty else 0
        if n_actual < cfg.thin_trading_min_days:
            reasons.append("thin_trading")

        amount_20d_avg_yuan = 0.0
        turnover_20d_avg = 0.0
        if not db_window.empty:
            # Tushare daily_basic.amount is in 千元 — convert to yuan for the
            # cfg comparison + persisted value (downstream consumers prefer
            # plain yuan to avoid silent unit confusion).
            if "amount" in db_window.columns:
                amount_20d_avg_yuan = float(db_window["amount"].astype(float).mean()) * 1000.0
            if "turnover_rate" in db_window.columns:
                turnover_20d_avg = float(db_window["turnover_rate"].astype(float).mean())

        if amount_20d_avg_yuan > 0 and amount_20d_avg_yuan < cfg.amount_20d_avg_min_yuan:
            reasons.append("low_amount")
        elif amount_20d_avg_yuan == 0 and n_actual >= cfg.thin_trading_min_days:
            # We have rows but no amount column / all zeros — treat as low.
            reasons.append("low_amount")

        # --- one-way limit detection (high == low → "一字")
        try:
            daily_window = data.fetch_daily_raw(
                rt.tushare, ts_code, window_start, trade_date,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("daily fetch failed for %s: %s", ts_code, exc)
            daily_window = pd.DataFrame()

        n_one_way = 0
        if not daily_window.empty and {"high", "low"}.issubset(daily_window.columns):
            highs = daily_window["high"].astype(float).values
            lows = daily_window["low"].astype(float).values
            n_one_way = int((abs(highs - lows) < 1e-6).sum())
        if n_actual > 0 and (n_one_way / max(1, n_actual)) > cfg.one_way_limit_max_ratio:
            reasons.append("one_way_limit")

        # --- price band
        last_close: float | None = None
        if not daily_window.empty and "close" in daily_window.columns:
            last_close = float(daily_window.iloc[-1]["close"])
        if last_close is not None:
            if cfg.price_band_low is not None and last_close < cfg.price_band_low:
                reasons.append("price_band")
            elif cfg.price_band_high is not None and last_close > cfg.price_band_high:
                reasons.append("price_band")

        # --- liquidity score (亿元 / 天, rounded for stable test assertions)
        liquidity_score = round(amount_20d_avg_yuan / 1e8, 4)

        snapshot.rows.append(UniverseRow(
            ts_code=ts_code,
            eligible=(len(reasons) == 0),
            reason_codes=reasons,
            liquidity_score=liquidity_score,
            amount_20d_avg=amount_20d_avg_yuan,
            turnover_20d_avg=turnover_20d_avg,
            list_status=str(status.get("list_status") or ""),
            is_st=bool(status.get("is_st", False)),
            name=str(status.get("name") or ""),
            industry=(str(status["industry"]) if status.get("industry") else None),
        ))

    # Deterministic order: descending liquidity, then ts_code for ties.
    snapshot.rows.sort(key=lambda r: (-r.liquidity_score, r.ts_code))
    return snapshot


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def upsert_universe_daily(db: Any, snapshot: UniverseSnapshot) -> int:
    """INSERT OR REPLACE every row in ``snapshot`` into ``checkmate_universe_daily``.

    Returns the number of rows written.
    """
    n = 0
    for r in snapshot.rows:
        db.execute(
            """
            INSERT OR REPLACE INTO checkmate_universe_daily
                (trade_date, ts_code, eligible, reason_codes,
                 liquidity_score, amount_20d_avg, turnover_20d_avg,
                 list_status, is_st, name, industry, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                snapshot.trade_date, r.ts_code, r.eligible,
                json.dumps(r.reason_codes, ensure_ascii=False),
                r.liquidity_score, r.amount_20d_avg, r.turnover_20d_avg,
                r.list_status, r.is_st, r.name, r.industry,
            ],
        )
        n += 1
    return n
