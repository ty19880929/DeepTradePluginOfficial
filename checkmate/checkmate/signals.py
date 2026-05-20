"""Entry-signal detection for Checkmate.

Three entry types (development_plan §8.1 + iteration_tasks.md §4 PR-3.1):

1. **Breakout (突破)** — close above the trailing ``breakout_lookback`` session
   high, today's ``amount`` ≥ ``breakout_amount_ratio`` × the prior
   ``breakout_amount_lookback``-day average, and today's pct_chg under the
   board-specific cap (8% main board, 11% ChiNext / STAR). The pct_chg cap
   filters out "chasing a limit-up" entries.

2. **Pullback (回踩)** — established uptrend (close > MA20, MA20 > MA60), low
   touched within ±``pullback_ma20_tol`` of MA20 in the last
   ``pullback_touch_window`` sessions, and today's close clears the
   ``pullback_platform_window`` short-term high.

3. **Continuation (趋势延续)** — cross-sectional strength rank
   (``rs60_pctile``) above ``continuation_rs60_min`` AND close clears the
   trailing ``continuation_breakout_lookback`` session high.

Every evaluator returns an :class:`EntryEval` with three fields:

* ``triggered`` — bool, gating Signal emission;
* ``hit`` — list[str], rule labels that passed;
* ``missed`` — list[str], rule labels that failed (with the numeric fact when
  helpful — e.g. ``"pct_chg<=8% (actual=9.10%)"``).

The explain CLI (PR-3.4) calls the evaluator directly on a chosen ts_code to
render the full hit / missed list regardless of whether a Signal would emit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from . import data
from .config import EntryConfig, ExitConfig
from .features import FeaturesRow
from .runtime import CheckmateRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EntryEval:
    """Result of evaluating one entry rule against a single (ts_code, date)."""

    signal_type: str
    triggered: bool
    hit: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Output row from the signal pipeline.

    Mirrors the ``checkmate_signals`` migration columns: ``action`` ∈
    {``enter`` / ``hold`` / ``defensive`` / ``exit``}; ``signal_type`` is the
    rule label (``breakout`` / ``pullback`` / ``continuation`` / etc.).
    """

    signal_date: str
    ts_code: str
    action: str
    signal_type: str
    score: float | None = None
    explain: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _board_pct_cap(ts_code: str, cfg: EntryConfig) -> tuple[float, str]:
    """Return ``(cap, board_label)`` for the given ts_code prefix."""
    prefix = ts_code.split(".", 1)[0]
    if prefix.startswith("300"):
        return cfg.pct_chg_cap_chinext, "chinext"
    if prefix.startswith("688"):
        return cfg.pct_chg_cap_star, "star"
    return cfg.pct_chg_cap_main_board, "main_board"


def _qfq_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Pick the qfq-adjusted version of ``name`` if present, else fall back."""
    qfq_name = f"{name}_qfq"
    if qfq_name in df.columns:
        return df[qfq_name].astype(float)
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(dtype=float)


def _normalise_window(df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """Sort by trade_date ascending, keep rows ≤ ``trade_date``."""
    if df is None or df.empty:
        return pd.DataFrame()
    s = df.copy()
    s["trade_date"] = s["trade_date"].astype(str)
    s = s[s["trade_date"] <= trade_date].sort_values("trade_date").reset_index(drop=True)
    return s


# ---------------------------------------------------------------------------
# Per-rule evaluators
# ---------------------------------------------------------------------------


def evaluate_breakout(
    ts_code: str,
    qfq_window: pd.DataFrame,
    daily_basic_window: pd.DataFrame,
    cfg: EntryConfig,
    *,
    trade_date: str,
) -> EntryEval:
    eval_ = EntryEval(signal_type="breakout", triggered=False)
    qf = _normalise_window(qfq_window, trade_date)
    if qf.empty or len(qf) < cfg.breakout_lookback + 1:
        eval_.missed.append("insufficient_history")
        return eval_

    close_q = _qfq_col(qf, "close")
    high_q = _qfq_col(qf, "high")
    if close_q.empty or high_q.empty:
        eval_.missed.append("insufficient_history")
        return eval_

    today_close = float(close_q.iloc[-1])
    prior_high = float(high_q.iloc[-(cfg.breakout_lookback + 1):-1].max())
    eval_.details["close"] = round(today_close, 4)
    eval_.details["prev_lookback_high"] = round(prior_high, 4)
    if today_close > prior_high:
        eval_.hit.append(f"close>{cfg.breakout_lookback}d_high")
    else:
        eval_.missed.append(
            f"close>{cfg.breakout_lookback}d_high (close={today_close:.2f}, high={prior_high:.2f})"
        )

    # Amount ratio (from daily_basic; Tushare daily_basic.amount is 千元 but
    # we only care about the ratio so units cancel)
    db = _normalise_window(daily_basic_window, trade_date)
    amount_ok = False
    if not db.empty and "amount" in db.columns and len(db) >= cfg.breakout_amount_lookback + 1:
        amounts = db["amount"].astype(float)
        today_amount = float(amounts.iloc[-1])
        prior_avg = float(amounts.iloc[-(cfg.breakout_amount_lookback + 1):-1].mean())
        ratio = today_amount / prior_avg if prior_avg > 0 else 0.0
        eval_.details["amount_today"] = round(today_amount, 2)
        eval_.details["amount_prior_avg"] = round(prior_avg, 2)
        eval_.details["amount_ratio"] = round(ratio, 4)
        if ratio >= cfg.breakout_amount_ratio:
            eval_.hit.append(f"amount_ratio>={cfg.breakout_amount_ratio}")
            amount_ok = True
        else:
            eval_.missed.append(
                f"amount_ratio>={cfg.breakout_amount_ratio} (actual={ratio:.2f})"
            )
    else:
        eval_.missed.append("amount_data_missing")

    # pct_chg cap (use RAW close / pre_close — limits are on raw prices)
    pct_chg = None
    if {"close", "pre_close"}.issubset(qf.columns):
        close_raw = float(qf["close"].astype(float).iloc[-1])
        pre_close = float(qf["pre_close"].astype(float).iloc[-1])
        if pre_close > 0:
            pct_chg = close_raw / pre_close - 1.0
    cap, board = _board_pct_cap(ts_code, cfg)
    eval_.details["pct_chg"] = round(pct_chg, 4) if pct_chg is not None else None
    eval_.details["board"] = board
    eval_.details["pct_chg_cap"] = cap
    if pct_chg is None:
        eval_.missed.append("pct_chg_data_missing")
    elif pct_chg <= cap:
        eval_.hit.append(f"pct_chg<={cap*100:.0f}% ({board})")
    else:
        eval_.missed.append(
            f"pct_chg<={cap*100:.0f}% ({board}) (actual={pct_chg*100:.2f}%)"
        )

    eval_.triggered = (
        (today_close > prior_high)
        and amount_ok
        and (pct_chg is not None and pct_chg <= cap)
    )
    return eval_


def evaluate_pullback(
    ts_code: str,  # noqa: ARG001
    qfq_window: pd.DataFrame,
    cfg: EntryConfig,
    *,
    trade_date: str,
) -> EntryEval:
    eval_ = EntryEval(signal_type="pullback", triggered=False)
    qf = _normalise_window(qfq_window, trade_date)
    # Need enough history to compute MA20 / MA60.
    if qf.empty or len(qf) < 60:
        eval_.missed.append("insufficient_history")
        return eval_
    close_q = _qfq_col(qf, "close")
    low_q = _qfq_col(qf, "low")
    if close_q.empty or low_q.empty:
        eval_.missed.append("insufficient_history")
        return eval_

    ma20 = float(close_q.tail(20).mean())
    ma60 = float(close_q.tail(60).mean())
    today_close = float(close_q.iloc[-1])
    eval_.details["close"] = round(today_close, 4)
    eval_.details["ma20"] = round(ma20, 4)
    eval_.details["ma60"] = round(ma60, 4)

    # Trend gates
    if today_close > ma20:
        eval_.hit.append("close>ma20")
    else:
        eval_.missed.append(f"close>ma20 (close={today_close:.2f}, ma20={ma20:.2f})")
    if ma20 > ma60 * cfg.pullback_trend_ma_ratio:
        eval_.hit.append("ma20>ma60")
    else:
        eval_.missed.append("ma20>ma60")

    # Touched MA20 ±tol within the recent window
    touch_window = low_q.tail(cfg.pullback_touch_window)
    tol = cfg.pullback_ma20_tol
    touched = bool(((touch_window >= ma20 * (1 - tol)) & (touch_window <= ma20 * (1 + tol))).any())
    eval_.details["pullback_touch_min_low"] = round(float(touch_window.min()), 4) if not touch_window.empty else None
    if touched:
        eval_.hit.append(f"touched_ma20_±{tol*100:.0f}%")
    else:
        eval_.missed.append(f"touched_ma20_±{tol*100:.0f}%")

    # Clear short-term platform high (excluding today)
    platform_window = close_q.iloc[-(cfg.pullback_platform_window + 1):-1]
    platform_high = float(platform_window.max()) if not platform_window.empty else float("inf")
    eval_.details[f"platform_{cfg.pullback_platform_window}d_high"] = round(platform_high, 4)
    if today_close > platform_high:
        eval_.hit.append(f"close>{cfg.pullback_platform_window}d_platform_high")
    else:
        eval_.missed.append(
            f"close>{cfg.pullback_platform_window}d_platform_high "
            f"(close={today_close:.2f}, high={platform_high:.2f})"
        )

    eval_.triggered = (
        today_close > ma20
        and ma20 > ma60 * cfg.pullback_trend_ma_ratio
        and touched
        and today_close > platform_high
    )
    return eval_


def evaluate_continuation(
    ts_code: str,  # noqa: ARG001
    qfq_window: pd.DataFrame,
    row: FeaturesRow,
    cfg: EntryConfig,
    *,
    trade_date: str,
) -> EntryEval:
    eval_ = EntryEval(signal_type="continuation", triggered=False)
    qf = _normalise_window(qfq_window, trade_date)
    if qf.empty or len(qf) < cfg.continuation_breakout_lookback + 1:
        eval_.missed.append("insufficient_history")
        return eval_

    close_q = _qfq_col(qf, "close")
    high_q = _qfq_col(qf, "high")
    today_close = float(close_q.iloc[-1])
    prior_high = float(high_q.iloc[-(cfg.continuation_breakout_lookback + 1):-1].max())
    eval_.details["close"] = round(today_close, 4)
    eval_.details[f"prev_{cfg.continuation_breakout_lookback}d_high"] = round(prior_high, 4)
    eval_.details["rs60_pctile"] = row.rs60_pctile

    # Strength gate
    if row.rs60_pctile is not None and row.rs60_pctile >= cfg.continuation_rs60_min:
        eval_.hit.append(f"rs60_pctile>={cfg.continuation_rs60_min}")
    else:
        actual = "None" if row.rs60_pctile is None else f"{row.rs60_pctile:.2f}"
        eval_.missed.append(
            f"rs60_pctile>={cfg.continuation_rs60_min} (actual={actual})"
        )

    # Breakout-of-recent gate
    if today_close > prior_high:
        eval_.hit.append(f"close>{cfg.continuation_breakout_lookback}d_high")
    else:
        eval_.missed.append(
            f"close>{cfg.continuation_breakout_lookback}d_high "
            f"(close={today_close:.2f}, high={prior_high:.2f})"
        )

    eval_.triggered = (
        row.rs60_pctile is not None
        and row.rs60_pctile >= cfg.continuation_rs60_min
        and today_close > prior_high
    )
    return eval_


# ---------------------------------------------------------------------------
# Composite per-symbol detection
# ---------------------------------------------------------------------------


def detect_entry_signals_for_symbol(
    row: FeaturesRow,
    qfq_window: pd.DataFrame,
    daily_basic_window: pd.DataFrame,
    cfg: EntryConfig | None = None,
) -> list[Signal]:
    """Run all three evaluators for one (ts_code, trade_date) and emit Signals.

    A symbol can trigger more than one entry type (e.g. breakout that also
    qualifies as continuation); callers de-dupe by priority if they need a
    single best signal per ts_code.
    """
    cfg = cfg or EntryConfig()
    sigs: list[Signal] = []
    out_evals: list[EntryEval] = []

    bo = evaluate_breakout(row.ts_code, qfq_window, daily_basic_window, cfg,
                           trade_date=row.trade_date)
    out_evals.append(bo)
    pb = evaluate_pullback(row.ts_code, qfq_window, cfg, trade_date=row.trade_date)
    out_evals.append(pb)
    co = evaluate_continuation(row.ts_code, qfq_window, row, cfg,
                               trade_date=row.trade_date)
    out_evals.append(co)

    for ev in out_evals:
        if ev.triggered:
            sigs.append(Signal(
                signal_date=row.trade_date,
                ts_code=row.ts_code,
                action="enter",
                signal_type=ev.signal_type,
                score=row.score,
                explain={
                    "signal_type": ev.signal_type,
                    "hit": list(ev.hit),
                    "missed": list(ev.missed),
                    "details": dict(ev.details),
                },
            ))
    return sigs


# ---------------------------------------------------------------------------
# Top-level entry: iterates a list of FeaturesRow objects
# ---------------------------------------------------------------------------


def detect_entry_signals(
    rt: CheckmateRuntime,
    trade_date: str,
    features_rows: list[FeaturesRow],
    cfg: EntryConfig | None = None,
) -> list[Signal]:
    """Iterate ``features_rows`` and emit entry signals.

    For each ts_code we re-fetch the qfq + daily_basic windows from the
    parquet cache. PR-2.3 has already populated those during ``scan``; this
    function is a thin orchestrator that consumes whatever is cached. If a
    cache is missing the per-symbol evaluator will mark
    ``insufficient_history`` and skip emission — never raises.
    """
    cfg = cfg or EntryConfig()
    out: list[Signal] = []

    # Pull a ~200 cal-day window to comfortably cover the longest lookback
    # (continuation 10d / breakout 40d / pullback 60d MA60).
    window_start = (
        datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=200)
    ).strftime("%Y%m%d")

    for row in features_rows:
        try:
            qfq = data.fetch_daily_qfq(rt.tushare, row.ts_code, window_start, trade_date)
            db_win = data.fetch_daily_basic(rt.tushare, row.ts_code, window_start, trade_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("signals fetch failed for %s: %s", row.ts_code, exc)
            continue
        out.extend(detect_entry_signals_for_symbol(row, qfq, db_win, cfg))
    return out


# ===========================================================================
# Exit-side: position state-machine driven
# ===========================================================================


@dataclass
class Position:
    """In-memory view over one ``checkmate_positions`` row.

    Field set is intentionally narrow — only what the exit evaluator and the
    risk module (PR-3.3) consume. ``risk_R`` is the per-share 1R dollar
    distance (entry - stop); ``peak_pnl_R`` is the high-water mark of
    unrealised profit in R units (None if the position never went positive).
    """

    ts_code: str
    entry_date: str
    entry_price_raw: float
    entry_price_qfq: float | None
    shares: int
    stop_price: float
    state: str  # pending / holding / defensive / closed
    risk_R: float | None  # per-share 1R (entry - stop), always > 0 in well-formed rows
    peak_pnl_R: float | None
    run_id: str | None = None


def _days_between(from_yyyymmdd: str, to_yyyymmdd: str) -> int:
    """Calendar days from ``from_`` to ``to_``. 0 on malformed input."""
    try:
        a = datetime.strptime(from_yyyymmdd, "%Y%m%d")
        b = datetime.strptime(to_yyyymmdd, "%Y%m%d")
    except ValueError:
        return 0
    return max(0, (b - a).days)


def evaluate_exit(
    position: Position,
    trade_date: str,
    *,
    today_close_raw: float | None,
    today_low_raw: float | None = None,
    regime: str | None = None,
    cfg: ExitConfig | None = None,
) -> EntryEval:
    """Evaluate the 5-rule exit cascade for ``position`` at ``trade_date``.

    Returns an :class:`EntryEval` whose ``signal_type`` is the first rule that
    fired (``hard_stop`` / ``risk_regime`` / ``defensive_profit`` /
    ``trailing_stop`` / ``time_exit``) or ``no_exit`` if none did.
    ``triggered`` mirrors that.

    T+1 contract: if ``position.entry_date == trade_date``, no exit signal is
    emitted no matter how dire the price action — the result carries
    ``t1_blocked=True`` in ``details`` so the runner can record a risk event
    without selling.
    """
    cfg = cfg or ExitConfig()
    ev = EntryEval(signal_type="no_exit", triggered=False)
    ev.details["state"] = position.state
    ev.details["entry_date"] = position.entry_date

    # ---- T+1 settlement gate
    if position.entry_date == trade_date:
        ev.signal_type = "t1_blocked"
        ev.details["t1_blocked"] = True
        ev.missed.append("t+1 settlement — cannot exit on entry day")
        return ev
    ev.details["t1_blocked"] = False

    # ---- numeric prep
    if today_close_raw is None:
        ev.missed.append("today_close_missing")
        return ev
    ev.details["close"] = round(today_close_raw, 4)
    ev.details["stop_price"] = round(position.stop_price, 4)
    pnl_per_share = today_close_raw - position.entry_price_raw
    pnl_R = (
        pnl_per_share / position.risk_R
        if (position.risk_R and position.risk_R > 0) else None
    )
    if pnl_R is not None:
        ev.details["pnl_R"] = round(pnl_R, 4)

    # ---- Rule 1: hard stop (priority highest)
    # Compare today's RAW close (and intraday low when available) to stop.
    triggered_hard = today_close_raw < position.stop_price
    if today_low_raw is not None and today_low_raw < position.stop_price:
        triggered_hard = True
    if triggered_hard:
        ev.signal_type = "hard_stop"
        ev.triggered = True
        ev.hit.append(
            f"close<stop ({today_close_raw:.2f} < {position.stop_price:.2f})"
        )
        ev.details["action"] = "exit"
        return ev
    else:
        ev.missed.append("hard_stop")

    # ---- Rule 2: risk regime
    if regime == cfg.risk_regime_tag:
        ev.signal_type = "risk_regime"
        ev.triggered = True
        ev.hit.append(f"regime={regime}")
        ev.details["regime"] = regime
        ev.details["action"] = "exit"
        return ev
    else:
        ev.missed.append(f"regime!=risk (current={regime})")

    # ---- Rule 3: defensive profit (state transition; only meaningful for
    # currently-holding positions — once in 'defensive' the trailing_stop
    # rule supersedes).
    if position.state == "holding" and position.peak_pnl_R is not None and pnl_R is not None:
        peak = position.peak_pnl_R
        ev.details["peak_pnl_R"] = round(peak, 4)
        if peak >= cfg.defensive_profit_peak_R and (peak - pnl_R) >= cfg.defensive_profit_retrace_R:
            ev.signal_type = "defensive_profit"
            ev.triggered = True
            ev.hit.append(
                f"peak_pnl_R>={cfg.defensive_profit_peak_R} AND retrace>={cfg.defensive_profit_retrace_R}R"
            )
            ev.details["action"] = "defensive"
            return ev
        else:
            ev.missed.append(
                f"defensive_profit (peak={peak:.2f}R, retrace={peak - pnl_R:.2f}R)"
            )

    # ---- Rule 4: trailing stop (uses peak in price terms, derived from
    # peak_pnl_R and 1R per-share dollar amount).
    if position.peak_pnl_R is not None and position.risk_R is not None:
        peak_price = position.entry_price_raw + position.peak_pnl_R * position.risk_R
        trail_stop = peak_price * (1.0 - cfg.trailing_pct)
        ev.details["peak_price"] = round(peak_price, 4)
        ev.details["trailing_stop"] = round(trail_stop, 4)
        if today_close_raw < trail_stop:
            ev.signal_type = "trailing_stop"
            ev.triggered = True
            ev.hit.append(
                f"close<peak*(1-{cfg.trailing_pct}) "
                f"({today_close_raw:.2f} < {trail_stop:.2f})"
            )
            ev.details["action"] = "exit"
            return ev
        else:
            ev.missed.append("trailing_stop")

    # ---- Rule 5: time exit
    days_held = _days_between(position.entry_date, trade_date)
    ev.details["days_held"] = days_held
    if days_held > cfg.max_hold_days:
        weak_pnl = pnl_R is None or pnl_R < cfg.time_exit_min_pnl_R
        if weak_pnl:
            ev.signal_type = "time_exit"
            ev.triggered = True
            ev.hit.append(
                f"days_held>{cfg.max_hold_days} AND pnl_R<{cfg.time_exit_min_pnl_R}"
            )
            ev.details["action"] = "exit"
            return ev
        else:
            ev.missed.append(
                f"time_exit_blocked (pnl_R={pnl_R:.2f} ≥ {cfg.time_exit_min_pnl_R})"
            )
    else:
        ev.missed.append(f"time_exit (days_held={days_held} ≤ {cfg.max_hold_days})")

    return ev


# ---------------------------------------------------------------------------
# DB shim — query active positions
# ---------------------------------------------------------------------------


_POSITION_COLS = (
    "ts_code", "entry_date", "entry_price_raw", "entry_price_qfq",
    "shares", "stop_price", "state", "risk_R", "peak_pnl_R", "run_id",
)


def query_active_positions(db: Any) -> list[Position]:
    """Return every position with ``state`` ∈ {holding, defensive}."""
    rows = db.execute(
        f"""
        SELECT {", ".join(_POSITION_COLS)}
        FROM checkmate_positions
        WHERE state IN ('holding', 'defensive')
        ORDER BY ts_code, entry_date
        """,
    ).fetchall()
    out: list[Position] = []
    for r in rows:
        d = dict(zip(_POSITION_COLS, r))
        out.append(Position(
            ts_code=str(d["ts_code"]),
            entry_date=str(d["entry_date"]),
            entry_price_raw=float(d["entry_price_raw"]),
            entry_price_qfq=(None if d["entry_price_qfq"] is None else float(d["entry_price_qfq"])),
            shares=int(d["shares"]),
            stop_price=float(d["stop_price"]),
            state=str(d["state"]),
            risk_R=(None if d["risk_R"] is None else float(d["risk_R"])),
            peak_pnl_R=(None if d["peak_pnl_R"] is None else float(d["peak_pnl_R"])),
            run_id=(None if d["run_id"] is None else str(d["run_id"])),
        ))
    return out


# ---------------------------------------------------------------------------
# Top-level exit detection
# ---------------------------------------------------------------------------


def detect_exit_signals(
    rt: CheckmateRuntime,
    trade_date: str,
    *,
    regime: str | None = None,
    cfg: ExitConfig | None = None,
) -> list[Signal]:
    """For each active position, pull today's raw close (from cached daily)
    and run :func:`evaluate_exit`. Emit a Signal for every triggered rule.

    T+1-blocked positions don't yield a Signal but the evaluator's result is
    available on demand via the explain CLI (PR-3.4) — runners that need to
    log them as risk events can call :func:`evaluate_exit` directly.
    """
    cfg = cfg or ExitConfig()
    out: list[Signal] = []
    positions = query_active_positions(rt.db)
    if not positions:
        return out

    window_start = (
        datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=7)
    ).strftime("%Y%m%d")

    for pos in positions:
        try:
            df = data.fetch_daily_raw(rt.tushare, pos.ts_code, window_start, trade_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("exit fetch failed for %s: %s", pos.ts_code, exc)
            continue
        if df is None or df.empty:
            continue
        today = df[df["trade_date"].astype(str) == trade_date]
        if today.empty:
            # No quote on trade_date — likely suspended. Leave the position
            # alone; defensive logic kicks in once trading resumes.
            continue
        close_raw = float(today.iloc[-1]["close"])
        low_raw = float(today.iloc[-1]["low"]) if "low" in today.columns else None

        ev = evaluate_exit(
            pos, trade_date,
            today_close_raw=close_raw,
            today_low_raw=low_raw,
            regime=regime,
            cfg=cfg,
        )
        if not ev.triggered:
            continue
        action = ev.details.get("action", "exit")
        out.append(Signal(
            signal_date=trade_date,
            ts_code=pos.ts_code,
            action=str(action),
            signal_type=ev.signal_type,
            score=None,
            explain={
                "signal_type": ev.signal_type,
                "hit": list(ev.hit),
                "missed": list(ev.missed),
                "details": dict(ev.details),
            },
        ))
    return out


# ===========================================================================
# End-to-end signals orchestrator (PR-3.4)
# ===========================================================================


_ENTRY_PRIORITY = ("breakout", "continuation", "pullback")


@dataclass
class SignalsParams:
    """Inputs to :func:`run_signals`."""

    trade_date: str | None = None
    portfolio_value: float = 1_000_000.0  # v0.1 default; live/backtest override
    entry_cfg: EntryConfig | None = None
    exit_cfg: ExitConfig | None = None
    risk_cfg: Any = None  # late import — RiskConfig from .config


@dataclass
class SignalsOutcome:
    run_id: str
    trade_date: str
    n_entry_proposals: int
    n_entry_accepted: int
    n_exits: int
    regime: str | None
    started_at: datetime
    finished_at: datetime


# ---------------------------------------------------------------------------
# DB readers
# ---------------------------------------------------------------------------


def _load_features_rows(db: Any, trade_date: str) -> list[FeaturesRow]:
    rows = db.execute(
        """
        SELECT trade_date, ts_code, close_qfq, ma20, ma60, ma120, ma_slope60,
               atr20, atr_pct, ret_60, ret_120, rs60_pctile, rs120_pctile,
               amount_20d_avg, turnover_20d_avg, limit_freq_60d,
               drawdown_60d_high, quiet_score, above_ma20_days, score
        FROM checkmate_features_daily
        WHERE trade_date = ?
        ORDER BY ts_code
        """,
        [trade_date],
    ).fetchall()
    cols = (
        "trade_date", "ts_code", "close_qfq", "ma20", "ma60", "ma120", "ma_slope60",
        "atr20", "atr_pct", "ret_60", "ret_120", "rs60_pctile", "rs120_pctile",
        "amount_20d_avg", "turnover_20d_avg", "limit_freq_60d",
        "drawdown_60d_high", "quiet_score", "above_ma20_days", "score",
    )
    out: list[FeaturesRow] = []
    for r in rows:
        d = dict(zip(cols, r))
        out.append(FeaturesRow(**d))
    return out


def _load_universe_eligible(db: Any, trade_date: str) -> dict[str, dict[str, Any]]:
    rows = db.execute(
        """
        SELECT ts_code, industry, name, eligible
        FROM checkmate_universe_daily
        WHERE trade_date = ?
        """,
        [trade_date],
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for ts_code, industry, name, eligible in rows:
        out[str(ts_code)] = {
            "industry": (None if industry is None else str(industry)),
            "name": str(name) if name is not None else "",
            "eligible": bool(eligible),
        }
    return out


def _load_regime(db: Any, trade_date: str) -> str | None:
    row = db.execute(
        "SELECT regime FROM checkmate_regime_daily WHERE trade_date = ?",
        [trade_date],
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def _today_close_raw(rt: CheckmateRuntime, ts_code: str, trade_date: str) -> float | None:
    """Read today's RAW close from the cached daily parquet."""
    window_start = (
        datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=10)
    ).strftime("%Y%m%d")
    try:
        df = data.fetch_daily_raw(rt.tushare, ts_code, window_start, trade_date)
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None
    today = df[df["trade_date"].astype(str) == trade_date]
    if today.empty:
        return None
    return float(today.iloc[-1]["close"])


# ---------------------------------------------------------------------------
# Proposal construction
# ---------------------------------------------------------------------------


def _dedupe_by_priority(sigs: list[Signal]) -> Signal | None:
    """Pick the highest-priority entry Signal among multiple for one symbol.

    Priority: breakout > continuation > pullback. The dropped types are
    recorded in the chosen signal's ``explain.alternative_types``.
    """
    if not sigs:
        return None
    by_type = {s.signal_type: s for s in sigs}
    chosen: Signal | None = None
    for t in _ENTRY_PRIORITY:
        if t in by_type:
            chosen = by_type[t]
            break
    if chosen is None:
        chosen = sigs[0]
    alternates = [t for t in by_type if t != chosen.signal_type]
    if alternates:
        chosen.explain["alternative_types"] = alternates
    return chosen


def _build_proposals(
    rt: CheckmateRuntime,
    trade_date: str,
    features_rows: list[FeaturesRow],
    universe: dict[str, dict[str, Any]],
    cfg: EntryConfig,
) -> list:  # list[ProposedEntry] — late import to avoid circular dep
    from .risk import ProposedEntry  # noqa: PLC0415

    proposals = []
    for row in features_rows:
        u = universe.get(row.ts_code)
        if not u or not u["eligible"]:
            continue
        # Build entry signals; aggregate to one per ts_code by priority.
        window_start = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=200)
        ).strftime("%Y%m%d")
        try:
            qfq = data.fetch_daily_qfq(rt.tushare, row.ts_code, window_start, trade_date)
            db_win = data.fetch_daily_basic(rt.tushare, row.ts_code, window_start, trade_date)
        except Exception:  # noqa: BLE001
            continue
        sigs = detect_entry_signals_for_symbol(row, qfq, db_win, cfg)
        chosen = _dedupe_by_priority(sigs)
        if chosen is None:
            continue
        # Entry price = today's RAW close (executor uses raw + stk_limit at fill).
        entry_price = _today_close_raw(rt, row.ts_code, trade_date)
        if entry_price is None or entry_price <= 0:
            continue
        # Stop placement: entry - atr_stop_mult * atr20 (atr20 is in qfq units;
        # for a non-split day raw == qfq so this is exact, otherwise close
        # enough for v0.1 sizing — refined when executor formalises the
        # raw/qfq mapping in Iter-4).
        if row.atr20 is None or row.atr20 <= 0:
            continue
        stop_price = max(0.0, entry_price - cfg.atr_stop_mult * row.atr20)
        if stop_price >= entry_price:
            continue
        proposals.append(ProposedEntry(
            ts_code=row.ts_code,
            entry_price=round(entry_price, 4),
            stop_price=round(stop_price, 4),
            industry=u["industry"],
            score=row.score,
            signal_type=chosen.signal_type,
            explain=dict(chosen.explain),
        ))
    return proposals


# ---------------------------------------------------------------------------
# Portfolio snapshot for the risk filter
# ---------------------------------------------------------------------------


def _position_views(
    rt: CheckmateRuntime, trade_date: str,
) -> list:  # list[PositionView]
    from .risk import PositionView  # noqa: PLC0415

    views: list = []
    for pos in query_active_positions(rt.db):
        close = _today_close_raw(rt, pos.ts_code, trade_date) or pos.entry_price_raw
        # industry from status_history at trade_date — cheaper than a join.
        status = data.query_status_as_of(rt.db, pos.ts_code, trade_date)
        industry = status.get("industry") if status else None
        views.append(PositionView(
            ts_code=pos.ts_code,
            industry=industry,
            market_value=close * pos.shares,
        ))
    return views


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def upsert_signals(db: Any, run_id: str, signals: list[Signal]) -> int:
    """INSERT OR REPLACE one row per Signal into ``checkmate_signals``."""
    import json as _json  # noqa: PLC0415

    n = 0
    for s in signals:
        db.execute(
            """
            INSERT OR REPLACE INTO checkmate_signals
                (signal_date, ts_code, action, signal_type, score, explain,
                 run_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                s.signal_date, s.ts_code, s.action, s.signal_type, s.score,
                _json.dumps(s.explain, ensure_ascii=False, default=str),
                run_id,
            ],
        )
        n += 1
    return n


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_signals(
    rt: CheckmateRuntime,
    params: SignalsParams,
    *,
    renderer: Any = None,  # EventRenderer
    echo: Callable[[str], None] | None = None,
) -> SignalsOutcome:
    """Run the full signals pipeline for one trade_date.

    Steps:
      0. Resolve trade_date + load features / universe / regime from DB.
         (Caller must have run ``scan`` for this date already.)
      1. Build :class:`ProposedEntry` rows from triggered entry signals.
      2. Detect exit signals against active positions.
      3. Apply portfolio constraints (single / industry / regime / daily cap).
      4. Persist accepted entries + all exits as Signal rows.
      5. Write checkmate_runs(mode='signals') + events.

    ``renderer`` is the preferred event sink (v0.2.0+). ``echo`` is the
    legacy back-compat hook — when provided alone it gets wrapped in a
    :class:`LegacyStreamRenderer` whose ``sink`` is ``echo``.
    """
    import json as _json  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    from .config import RiskConfig  # noqa: PLC0415
    from .risk import apply_portfolio_constraints  # noqa: PLC0415
    from .ui import LegacyStreamRenderer, RenderEvent  # noqa: PLC0415

    if renderer is None:
        renderer = LegacyStreamRenderer(sink=echo) if echo is not None else LegacyStreamRenderer()

    started_at = datetime.now()
    run_id = str(uuid.uuid4())
    rt.run_id = run_id

    entry_cfg = params.entry_cfg or EntryConfig()
    exit_cfg = params.exit_cfg or ExitConfig()
    risk_cfg = params.risk_cfg or RiskConfig()

    # ---- step 0: resolve trade_date + load context
    trade_date = (params.trade_date or "").replace("-", "")
    if not trade_date:
        from .calendar import load_trade_calendar  # noqa: PLC0415

        cal = load_trade_calendar(rt.tushare)
        today = datetime.now().strftime("%Y%m%d")
        trade_date = today if cal.is_trade_day(today) else cal.prev_session(today)

    rt.db.execute(
        """
        INSERT INTO checkmate_runs
            (run_id, mode, trade_date, status, started_at, params_json)
        VALUES (?, 'signals', ?, 'running', CURRENT_TIMESTAMP, ?)
        """,
        [run_id, trade_date,
         _json.dumps({"portfolio_value": params.portfolio_value}, ensure_ascii=False)],
    )

    def _emit(seq: list[int], event_type: str, message: str,
              level: str = "info", payload: dict | None = None) -> None:
        seq[0] += 1
        try:
            rt.db.execute(
                """
                INSERT INTO checkmate_events
                    (run_id, seq, event_time, level, event_type, message, payload_json)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
                """,
                [run_id, seq[0], level, event_type, message,
                 _json.dumps(payload or {}, ensure_ascii=False, default=str)],
            )
        except Exception:  # noqa: BLE001
            logger.exception("event persist failed")
        try:
            renderer.on_event(RenderEvent(
                type=event_type, message=message, level=level, payload=payload or {},
            ))
        except Exception:  # noqa: BLE001
            logger.exception("renderer on_event raised for %s", event_type)

    seq = [0]
    renderer.on_run_start(run_id=run_id, mode="signals", params=params)
    _emit(seq, "RUN_STARTED",
          f"signals run started for {trade_date}",
          payload={"run_id": run_id})

    features_rows = _load_features_rows(rt.db, trade_date)
    universe = _load_universe_eligible(rt.db, trade_date)
    regime = _load_regime(rt.db, trade_date)
    if not features_rows:
        _emit(seq, "STEP_FAILED",
              "no features rows for trade_date — run `scan` first",
              level="error",
              payload={"trade_date": trade_date})
        rt.db.execute(
            "UPDATE checkmate_runs SET status='failed', exit_code=2, "
            "finished_at=CURRENT_TIMESTAMP, error=? WHERE run_id=?",
            ["features_missing", run_id],
        )
        fail_outcome = SignalsOutcome(
            run_id=run_id, trade_date=trade_date,
            n_entry_proposals=0, n_entry_accepted=0, n_exits=0,
            regime=regime, started_at=started_at, finished_at=datetime.now(),
        )
        try:
            renderer.on_run_finish(fail_outcome)
        finally:
            try:
                renderer.close()
            except Exception:  # noqa: BLE001
                logger.exception("renderer close raised")
        return fail_outcome

    # ---- step 1: entry proposals
    _emit(seq, "STEP_STARTED", "Step 1: detect entry signals")
    proposals = _build_proposals(rt, trade_date, features_rows, universe, entry_cfg)
    _emit(seq, "STEP_FINISHED",
          f"Step 1 done — proposals={len(proposals)}",
          payload={"n_proposals": len(proposals)})

    # ---- step 2: exit signals
    _emit(seq, "STEP_STARTED", "Step 2: detect exit signals")
    exits = detect_exit_signals(rt, trade_date, regime=regime, cfg=exit_cfg)
    _emit(seq, "STEP_FINISHED",
          f"Step 2 done — exits={len(exits)}",
          payload={"n_exits": len(exits)})

    # ---- step 3: portfolio constraints
    _emit(seq, "STEP_STARTED", "Step 3: apply portfolio constraints")
    pos_views = _position_views(rt, trade_date)
    sized = apply_portfolio_constraints(
        proposals, pos_views,
        portfolio_value=params.portfolio_value,
        regime=regime, cfg=risk_cfg,
    )
    accepted = [o for o in sized if o.accepted]
    rejected = [o for o in sized if not o.accepted]
    _emit(seq, "STEP_FINISHED",
          f"Step 3 done — accepted={len(accepted)} rejected={len(rejected)}",
          payload={
              "n_accepted": len(accepted),
              "n_rejected": len(rejected),
              "regime": regime,
          })

    # ---- step 4: persist Signal rows
    signals: list[Signal] = []
    for o in accepted:
        signals.append(Signal(
            signal_date=trade_date,
            ts_code=o.ts_code,
            action="enter",
            signal_type=o.signal_type,
            score=o.score,
            explain={
                **o.explain,
                "shares": o.shares,
                "entry_price": o.entry_price,
                "stop_price": o.stop_price,
                "risk_R": o.risk_R,
                "weight": o.weight,
                "industry": o.industry,
            },
        ))
    # Persist rejected as Signal rows with action='rejected' (PK conflicts
    # with accepted are impossible — action differs). Useful for explain CLI
    # and post-mortem analysis without re-running the filter.
    for o in rejected:
        signals.append(Signal(
            signal_date=trade_date,
            ts_code=o.ts_code,
            action="rejected",
            signal_type=o.signal_type,
            score=o.score,
            explain={
                **o.explain,
                "cancel_reason": o.cancel_reason,
                "entry_price": o.entry_price,
                "stop_price": o.stop_price,
                "industry": o.industry,
            },
        ))
    signals.extend(exits)
    n_persisted = upsert_signals(rt.db, run_id, signals)
    _emit(seq, "RUN_FINISHED",
          f"signals run finished — persisted={n_persisted}",
          payload={"n_persisted": n_persisted})

    rt.db.execute(
        """
        UPDATE checkmate_runs
        SET status='success', exit_code=0, finished_at=CURRENT_TIMESTAMP,
            summary_json=?
        WHERE run_id=?
        """,
        [_json.dumps({
            "n_entry_proposals": len(proposals),
            "n_entry_accepted": len(accepted),
            "n_exits": len(exits),
            "regime": regime,
        }, ensure_ascii=False), run_id],
    )

    outcome = SignalsOutcome(
        run_id=run_id, trade_date=trade_date,
        n_entry_proposals=len(proposals),
        n_entry_accepted=len(accepted),
        n_exits=len(exits),
        regime=regime,
        started_at=started_at, finished_at=datetime.now(),
    )
    try:
        renderer.on_run_finish(outcome)
    except Exception:  # noqa: BLE001
        logger.exception("renderer on_run_finish raised")
    try:
        renderer.close()
    except Exception:  # noqa: BLE001
        logger.exception("renderer close raised")
    return outcome
