"""Backtest engine — trade_cal stepper + per-day checkpoint (PR-4.2).

Pipeline per ``trade_date`` (development_plan §11 + iteration_tasks §5):

  1. Resolve today's PendingOrders (= yesterday's accepted entries + carried
     deferred sells + yesterday's exit signals).
  2. Fetch today's raw daily + stk_limit, call :func:`executor.simulate_session`,
     update portfolio cash / positions from fills, persist
     ``checkmate_trades`` + ``checkmate_positions`` rows.
  3. Compute today's universe / features / regime (calls reuse the live
     scan path) → persist into the three daily tables.
  4. Detect entry + exit signals against today's features + portfolio.
  5. Apply :func:`risk.apply_portfolio_constraints` → SizedOrder list.
  6. Build tomorrow's PendingOrder list: accepted entries (buy) + exit
     signals (sell, T+1-safe) + carried-over deferred sells.
  7. Mark-to-market (cash + Σ positions × today's close) → equity & DD.
  8. Save per-day checkpoint shard.

Checkpoint files live at::

    ~/.deeptrade/checkmate/backtests/<config_hash>/days/<YYYYMMDD>.json

(JSON for v0.1 — the per-day payload is small + nested lists are easier to
round-trip than via parquet's flat-row constraint. The on-disk layout stays
parquet-compatible — switch the I/O helpers below when payload size demands
it. Spec note in iteration_tasks.md PR-4.2 calls out parquet; we hold the
directory structure invariant and only diverge on the extension.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import data, paths
from .calendar import load_trade_calendar
from .config import (
    EntryConfig,
    ExecutionConfig,
    ExitConfig,
    FeaturesConfig,
    RegimeConfig,
    RiskConfig,
    UniverseConfig,
)
from .executor import Cancel, Fill, PendingOrder, simulate_session
from .features import compute_features_frame, upsert_features_daily
from .regime import classify_regime, compute_breadth_limit_down_5d, upsert_regime_daily
from .risk import PositionView, ProposedEntry, apply_portfolio_constraints
from .runtime import CheckmateRuntime
from .signals import (
    Position,
    detect_entry_signals_for_symbol,
    detect_exit_signals,
)
from .ui import EventRenderer, LegacyStreamRenderer, RenderEvent
from .universe import build_universe, upsert_universe_daily

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BacktestParams:
    start: str  # YYYY-MM-DD or YYYYMMDD
    end: str
    initial_cash: float = 1_000_000.0
    resume: bool = True
    code_version: str = "checkmate@0.4.0"
    universe_cfg: UniverseConfig | None = None
    features_cfg: FeaturesConfig | None = None
    regime_cfg: RegimeConfig | None = None
    entry_cfg: EntryConfig | None = None
    exit_cfg: ExitConfig | None = None
    risk_cfg: RiskConfig | None = None
    execution_cfg: ExecutionConfig | None = None


@dataclass
class BacktestOutcome:
    run_id: str
    config_hash: str
    start: str
    end: str
    n_days: int
    n_fills: int
    final_equity: float
    final_cash: float
    max_drawdown: float


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------


def _cfg_to_dict(params: BacktestParams) -> dict[str, Any]:
    """Canonical JSON-serialisable view of a BacktestParams for hashing.

    Includes every config block so any tweak produces a distinct hash. Excludes
    ``resume`` (resumption mode doesn't change the simulation logic).
    """
    def _d(cfg: Any) -> Any:
        if cfg is None:
            return None
        return {k: v for k, v in cfg.__dict__.items()}

    return {
        "start": params.start.replace("-", ""),
        "end": params.end.replace("-", ""),
        "initial_cash": params.initial_cash,
        "code_version": params.code_version,
        "universe":  _d(params.universe_cfg  or UniverseConfig()),
        "features":  _d(params.features_cfg  or FeaturesConfig()),
        "regime":    _d(params.regime_cfg    or RegimeConfig()),
        "entry":     _d(params.entry_cfg     or EntryConfig()),
        "exit":      _d(params.exit_cfg      or ExitConfig()),
        "risk":      _d(params.risk_cfg      or RiskConfig()),
        "execution": _d(params.execution_cfg or ExecutionConfig()),
    }


def compute_config_hash(params: BacktestParams) -> str:
    """BLAKE2b-64 hex digest of the canonical config JSON."""
    payload = json.dumps(_cfg_to_dict(params), sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def _checkpoint_dir(config_hash: str) -> Path:
    return paths.backtests_dir() / config_hash / "days"


def _shard_path(config_hash: str, trade_date: str) -> Path:
    return _checkpoint_dir(config_hash) / f"{trade_date}.json"


def _list_shards(config_hash: str) -> list[str]:
    """Return YYYYMMDD strings present as shards, sorted ascending."""
    d = _checkpoint_dir(config_hash)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def _save_shard(config_hash: str, trade_date: str, payload: dict[str, Any]) -> None:
    p = _shard_path(config_hash, trade_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _load_shard(config_hash: str, trade_date: str) -> dict[str, Any] | None:
    p = _shard_path(config_hash, trade_date)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _clear_checkpoint(config_hash: str) -> None:
    root = paths.backtests_dir() / config_hash
    if root.is_dir():
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------


@dataclass
class BacktestState:
    cash: float
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)  # ts_code → fields
    pending_orders: list[PendingOrder] = field(default_factory=list)
    high_water_mark: float = 0.0
    max_drawdown: float = 0.0


def _initial_state(initial_cash: float) -> BacktestState:
    return BacktestState(
        cash=initial_cash,
        positions={},
        pending_orders=[],
        high_water_mark=initial_cash,
        max_drawdown=0.0,
    )


def _state_to_dict(state: BacktestState) -> dict[str, Any]:
    return {
        "cash": state.cash,
        "positions": state.positions,
        "pending_orders": [_pending_to_dict(o) for o in state.pending_orders],
        "high_water_mark": state.high_water_mark,
        "max_drawdown": state.max_drawdown,
    }


def _state_from_dict(d: dict[str, Any]) -> BacktestState:
    return BacktestState(
        cash=float(d["cash"]),
        positions=dict(d.get("positions", {})),
        pending_orders=[_pending_from_dict(o) for o in d.get("pending_orders", [])],
        high_water_mark=float(d.get("high_water_mark", 0.0)),
        max_drawdown=float(d.get("max_drawdown", 0.0)),
    )


def _pending_to_dict(o: PendingOrder) -> dict[str, Any]:
    return {
        "ts_code": o.ts_code, "side": o.side, "shares": o.shares,
        "signal_date": o.signal_date,
        "stop_price": o.stop_price, "risk_R": o.risk_R,
        "reason_code": o.reason_code, "defer_count": o.defer_count,
        "same_day_entry": o.same_day_entry,
        "amount_20d_avg": o.amount_20d_avg,
    }


def _pending_from_dict(d: dict[str, Any]) -> PendingOrder:
    return PendingOrder(
        ts_code=str(d["ts_code"]), side=str(d["side"]),
        shares=int(d["shares"]), signal_date=str(d["signal_date"]),
        stop_price=d.get("stop_price"), risk_R=d.get("risk_R"),
        reason_code=str(d.get("reason_code", "")),
        defer_count=int(d.get("defer_count", 0)),
        same_day_entry=bool(d.get("same_day_entry", False)),
        amount_20d_avg=d.get("amount_20d_avg"),
    )


# ---------------------------------------------------------------------------
# Per-day DB helpers
# ---------------------------------------------------------------------------


def _upsert_position(db: Any, run_id: str, ts_code: str, p: dict[str, Any]) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO checkmate_positions
            (ts_code, entry_date, entry_price_raw, entry_price_qfq, shares,
             stop_price, state, risk_R, peak_pnl_R,
             exit_date, exit_price_raw, exit_reason, run_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            ts_code, p["entry_date"],
            p.get("entry_price_raw"), p.get("entry_price_qfq"),
            int(p.get("shares", 0)),
            p.get("stop_price"),
            p.get("state", "holding"),
            p.get("risk_R"), p.get("peak_pnl_R"),
            p.get("exit_date"), p.get("exit_price_raw"), p.get("exit_reason"),
            run_id,
        ],
    )


def _insert_trade(db: Any, run_id: str, ts_code: str, fill: Fill) -> None:
    db.execute(
        """
        INSERT INTO checkmate_trades
            (trade_id, run_id, ts_code, side, order_date, fill_date,
             fill_price_raw, fill_price_qfq, shares, cost_breakdown,
             exit_reason, cancel_reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            str(uuid.uuid4()), run_id, ts_code, fill.side,
            fill.order_date, fill.fill_date,
            fill.fill_price_raw, None, fill.shares,
            json.dumps(fill.cost_breakdown, ensure_ascii=False),
            fill.reason_code if fill.side == "sell" else None,
            None,
        ],
    )


def _insert_cancel_trade(db: Any, run_id: str, c: Cancel) -> None:
    """Persist a cancelled-order row in checkmate_trades for audit. shares=0
    keeps the unique trade_id constraint happy and signals "no fill" in the
    ledger."""
    db.execute(
        """
        INSERT INTO checkmate_trades
            (trade_id, run_id, ts_code, side, order_date, fill_date,
             fill_price_raw, fill_price_qfq, shares, cost_breakdown,
             exit_reason, cancel_reason, created_at)
        VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, NULL, ?, CURRENT_TIMESTAMP)
        """,
        [
            str(uuid.uuid4()), run_id, c.ts_code, c.side,
            c.order_date, c.shares, "{}", c.cancel_reason,
        ],
    )


# ---------------------------------------------------------------------------
# Position state-machine update from fills
# ---------------------------------------------------------------------------


def _apply_fills(state: BacktestState, fills: list[Fill]) -> tuple[float, float]:
    """Mutate ``state.cash`` + ``state.positions`` from fills.

    Returns ``(realised_buy_value, realised_sell_value)`` for diagnostics.
    """
    buy_val = 0.0
    sell_val = 0.0
    for f in fills:
        gross = f.shares * f.fill_price_raw
        fees = sum(f.cost_breakdown.values())
        if f.side == "buy":
            state.cash -= gross + fees
            buy_val += gross
            risk_per_share = None
            if f.fill_price_raw and gross > 0:
                # stop_price is set on the PendingOrder; the executor stamps
                # reason_code but not the stop. The signal layer populates
                # entry_price/stop_price via explain when ProposedEntry was
                # constructed. We re-derive stop here from the in-memory
                # pending order (kept on the side via ``Position`` below);
                # if the upstream order had no stop we leave it None.
                risk_per_share = None  # filled below when caller adds stop
            state.positions[f.ts_code] = {
                "entry_date": f.fill_date,
                "entry_price_raw": f.fill_price_raw,
                "entry_price_qfq": None,
                "shares": f.shares,
                "stop_price": None,  # caller patches from PendingOrder
                "state": "holding",
                "risk_R": risk_per_share,
                "peak_pnl_R": 0.0,
                "exit_date": None,
                "exit_price_raw": None,
                "exit_reason": None,
            }
        elif f.side == "sell":
            state.cash += gross - fees
            sell_val += gross
            pos = state.positions.get(f.ts_code)
            if pos is not None:
                pos["state"] = "closed"
                pos["exit_date"] = f.fill_date
                pos["exit_price_raw"] = f.fill_price_raw
                pos["exit_reason"] = f.reason_code
    return buy_val, sell_val


def _patch_buy_stops(
    state: BacktestState, fills: list[Fill], pending: list[PendingOrder],
) -> None:
    """After ``_apply_fills`` creates a position from a buy fill, copy the
    original PendingOrder's ``stop_price`` / ``risk_R`` into it (the executor
    doesn't carry these fields, but the position state machine needs them)."""
    pending_lookup = {(o.ts_code, o.signal_date): o for o in pending if o.side == "buy"}
    for f in fills:
        if f.side != "buy":
            continue
        key = (f.ts_code, f.order_date)
        po = pending_lookup.get(key)
        if po is None:
            continue
        pos = state.positions.get(f.ts_code)
        if pos is None:
            continue
        pos["stop_price"] = po.stop_price
        pos["risk_R"] = po.risk_R


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------


def _mark_to_market(
    state: BacktestState, prices_raw: pd.DataFrame, trade_date: str,
) -> float:
    """Return total equity = cash + Σ shares × today's close (raw)."""
    if prices_raw is None or prices_raw.empty:
        return state.cash + sum(
            p["shares"] * float(p["entry_price_raw"] or 0.0)
            for p in state.positions.values() if p["state"] != "closed"
        )
    close_by_code = dict(zip(
        prices_raw["ts_code"].astype(str),
        prices_raw["close"].astype(float),
    ))
    total = state.cash
    for ts_code, p in state.positions.items():
        if p["state"] == "closed":
            continue
        close = close_by_code.get(ts_code, float(p["entry_price_raw"] or 0.0))
        total += p["shares"] * close
        # update peak_pnl_R while we have today's close in hand
        if p["risk_R"] and p["risk_R"] > 0:
            pnl_R = (close - p["entry_price_raw"]) / p["risk_R"]
            if pnl_R > (p.get("peak_pnl_R") or 0.0):
                p["peak_pnl_R"] = pnl_R
    return total


# ---------------------------------------------------------------------------
# Per-day fetchers (collapses runner I/O into one place for testability)
# ---------------------------------------------------------------------------


def _prices_raw_for_day(rt: CheckmateRuntime, ts_codes: list[str], trade_date: str) -> pd.DataFrame:
    """Pull today's open/high/low/close/pre_close for ``ts_codes`` from cache."""
    if not ts_codes:
        return pd.DataFrame()
    rows: list[dict] = []
    window_start = trade_date  # one-day window — cache slice is cheap
    for ts_code in ts_codes:
        try:
            df = data.fetch_daily_raw(rt.tushare, ts_code, window_start, trade_date)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        today = df[df["trade_date"].astype(str) == trade_date]
        if today.empty:
            continue
        rows.append(today.iloc[0].to_dict())
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _stk_limit_for_day(rt: CheckmateRuntime, trade_date: str) -> pd.DataFrame:
    try:
        return data.fetch_stk_limit(rt.tushare, trade_date)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tomorrow's PendingOrder construction
# ---------------------------------------------------------------------------


def _build_next_pending(
    accepted_entries: list[Any],         # SizedOrder
    exit_signals: list[Any],             # Signal[]
    carried_deferred: list[PendingOrder],
    today_trade_date: str,
    *,
    amount_lookup: dict[str, float] | None = None,
) -> list[PendingOrder]:
    """Build tomorrow's PendingOrder list.

    PR-7.1: ``amount_lookup`` is an optional ``{ts_code: amount_20d_avg}``
    map that the executor uses for dynamic-slippage bps. Carried-over
    (deferred) orders keep their original amount.
    """
    lookup = amount_lookup or {}

    out: list[PendingOrder] = []
    for o in accepted_entries:
        out.append(PendingOrder(
            ts_code=o.ts_code, side="buy", shares=o.shares,
            signal_date=today_trade_date,
            stop_price=o.stop_price, risk_R=o.risk_R,
            reason_code=o.signal_type,
            amount_20d_avg=lookup.get(o.ts_code),
        ))
    for s in exit_signals:
        if s.action != "exit":
            continue
        shares = int(s.explain.get("details", {}).get("shares") or 0)
        out.append(PendingOrder(
            ts_code=s.ts_code, side="sell", shares=shares,
            signal_date=today_trade_date,
            reason_code=s.signal_type,
            amount_20d_avg=lookup.get(s.ts_code),
        ))
    out.extend(carried_deferred)
    return out


# ---------------------------------------------------------------------------
# execute_day
# ---------------------------------------------------------------------------


def execute_day(
    rt: CheckmateRuntime,
    state: BacktestState,
    trade_date: str,
    run_id: str,
    params: BacktestParams,
    *,
    renderer: EventRenderer | None = None,
    echo: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one trade_date and return a serialisable shard payload."""
    if renderer is None:
        renderer = LegacyStreamRenderer(sink=echo) if echo is not None else LegacyStreamRenderer()
    exec_cfg = params.execution_cfg or ExecutionConfig()
    universe_cfg = params.universe_cfg or UniverseConfig()
    features_cfg = params.features_cfg or FeaturesConfig()
    regime_cfg = params.regime_cfg or RegimeConfig()
    entry_cfg = params.entry_cfg or EntryConfig()
    exit_cfg = params.exit_cfg or ExitConfig()
    risk_cfg = params.risk_cfg or RiskConfig()

    # Step 1: today's universe (drives downstream features + entry candidates).
    snapshot = build_universe(rt, trade_date, universe_cfg)
    upsert_universe_daily(rt.db, snapshot)
    eligible_codes = [r.ts_code for r in snapshot.eligible]
    universe_by_code = {r.ts_code: r for r in snapshot.eligible}

    # Step 2: gather every ts_code we need prices for this session:
    # pending orders + currently-held positions + eligible (for entry sizing).
    today_pending = list(state.pending_orders)
    ts_codes_today = sorted(
        {o.ts_code for o in today_pending}
        | set(state.positions.keys())
        | set(eligible_codes)
    )
    prices_raw = _prices_raw_for_day(rt, ts_codes_today, trade_date)
    stk_limit = _stk_limit_for_day(rt, trade_date)

    # Step 3: executor (uses prices/stk_limit for the pending order subset)
    report = simulate_session(today_pending, trade_date, prices_raw, stk_limit, exec_cfg)
    _apply_fills(state, report.fills)
    _patch_buy_stops(state, report.fills, today_pending)
    for f in report.fills:
        _insert_trade(rt.db, run_id, f.ts_code, f)
        pos = state.positions.get(f.ts_code)
        if pos is not None:
            _upsert_position(rt.db, run_id, f.ts_code, pos)
    for c in report.cancels:
        _insert_cancel_trade(rt.db, run_id, c)

    # Step 4: today's features + regime
    if eligible_codes:
        _, feat_rows = compute_features_frame(rt, trade_date, eligible_codes, features_cfg)
        upsert_features_daily(rt.db, feat_rows)
    else:
        feat_rows = []
    breadth_ld5 = compute_breadth_limit_down_5d(rt, trade_date, [r.ts_code for r in snapshot.rows])
    regime_row = classify_regime(rt, trade_date, regime_cfg, breadth_limit_down_5d=breadth_ld5)
    upsert_regime_daily(rt.db, regime_row)

    # Step 5: entry proposals — uses prices_raw for entry_price and atr20 for stop
    proposals: list[ProposedEntry] = []
    window_start = (
        datetime.strptime(trade_date, "%Y%m%d") - pd.Timedelta(days=200)
    ).strftime("%Y%m%d")
    close_by_code = (
        dict(zip(prices_raw["ts_code"].astype(str), prices_raw["close"].astype(float)))
        if not prices_raw.empty and "ts_code" in prices_raw.columns
        else {}
    )
    for row in feat_rows:
        if row.ts_code not in universe_by_code:
            continue
        try:
            qfq = data.fetch_daily_qfq(rt.tushare, row.ts_code, window_start, trade_date)
            db_win = data.fetch_daily_basic(rt.tushare, row.ts_code, window_start, trade_date)
        except Exception:  # noqa: BLE001
            continue
        sigs = detect_entry_signals_for_symbol(row, qfq, db_win, entry_cfg)
        if not sigs:
            continue
        chosen = sigs[0]
        for s in sigs:
            if s.signal_type == "breakout":
                chosen = s
                break
        entry_price = close_by_code.get(row.ts_code)
        if entry_price is None or entry_price <= 0:
            continue
        if row.atr20 is None or row.atr20 <= 0:
            continue
        stop_price = max(0.0, entry_price - entry_cfg.atr_stop_mult * row.atr20)
        if stop_price >= entry_price:
            continue
        proposals.append(ProposedEntry(
            ts_code=row.ts_code, entry_price=entry_price, stop_price=stop_price,
            industry=universe_by_code[row.ts_code].industry,
            score=row.score, signal_type=chosen.signal_type, explain=chosen.explain,
        ))

    exits = detect_exit_signals(rt, trade_date, regime=regime_row.regime, cfg=exit_cfg)

    # Step 6: risk filter — needs PositionView for industry concentration
    pos_views: list[PositionView] = []
    for ts_code, p in state.positions.items():
        if p["state"] == "closed":
            continue
        # industry from universe snapshot (most recent known)
        u = universe_by_code.get(ts_code)
        industry = u.industry if u else None
        close = close_by_code.get(ts_code, float(p["entry_price_raw"] or 0.0))
        pos_views.append(PositionView(
            ts_code=ts_code, industry=industry, market_value=close * p["shares"],
        ))
    today_equity = _mark_to_market(state, prices_raw, trade_date)
    sized = apply_portfolio_constraints(
        proposals, pos_views,
        portfolio_value=today_equity,
        regime=regime_row.regime, cfg=risk_cfg,
    )
    accepted_entries = [o for o in sized if o.accepted]

    # Patch exit signals' shares from current positions (signals.detect_exit_signals
    # doesn't know shares — it surfaces the rule + reason).
    enriched_exits = []
    for s in exits:
        pos = state.positions.get(s.ts_code)
        if pos is None or pos["state"] == "closed":
            continue
        s.explain = {**s.explain, "details": {**s.explain.get("details", {}),
                                              "shares": pos["shares"]}}
        enriched_exits.append(s)

    # Step 7: tomorrow's pending orders. PR-7.1: pre-build the amount_lookup
    # so the executor can pick a per-symbol dynamic slippage bps tomorrow.
    amount_lookup: dict[str, float] = {}
    for row in feat_rows:
        if row.amount_20d_avg is not None and row.amount_20d_avg > 0:
            amount_lookup[row.ts_code] = float(row.amount_20d_avg)
    new_pending = _build_next_pending(
        accepted_entries=accepted_entries,
        exit_signals=enriched_exits,
        carried_deferred=report.deferred,
        today_trade_date=trade_date,
        amount_lookup=amount_lookup,
    )
    state.pending_orders = new_pending

    # Step 8: equity & drawdown
    state.high_water_mark = max(state.high_water_mark, today_equity)
    dd = (state.high_water_mark - today_equity) / state.high_water_mark if state.high_water_mark > 0 else 0.0
    state.max_drawdown = max(state.max_drawdown, dd)

    payload = {
        "trade_date": trade_date,
        "state": _state_to_dict(state),
        "n_fills": len(report.fills),
        "n_cancels": len(report.cancels),
        "n_deferred": len(report.deferred),
        "regime": regime_row.regime,
        "equity": today_equity,
        "drawdown_pct": dd,
    }
    try:
        renderer.on_event(RenderEvent(
            type="SESSION_FINISHED",
            message=(
                f"{trade_date}  fills={len(report.fills)}  "
                f"cancels={len(report.cancels)}  regime={regime_row.regime}  "
                f"equity={today_equity:,.0f}  dd={dd*100:.2f}%"
            ),
            payload={
                "trade_date": trade_date,
                "n_fills": len(report.fills),
                "n_cancels": len(report.cancels),
                "regime": regime_row.regime,
                "equity": today_equity,
                "drawdown_pct": dd,
            },
        ))
    except Exception:  # noqa: BLE001
        logger.exception("renderer on_event raised for SESSION_FINISHED")
    return payload


# ---------------------------------------------------------------------------
# run_backtest orchestrator
# ---------------------------------------------------------------------------


def run_backtest(
    rt: CheckmateRuntime,
    params: BacktestParams,
    *,
    renderer: EventRenderer | None = None,
    echo: Callable[[str], None] | None = None,
) -> BacktestOutcome:
    """Drive the per-day pipeline from ``params.start`` to ``params.end``.

    ``renderer`` is the preferred event sink (v0.2.0+); ``echo`` is back-compat
    and gets wrapped in :class:`LegacyStreamRenderer` when used alone.
    """
    if renderer is None:
        renderer = LegacyStreamRenderer(sink=echo) if echo is not None else LegacyStreamRenderer()

    run_id = str(uuid.uuid4())
    rt.run_id = run_id
    config_hash = compute_config_hash(params)
    renderer.on_run_start(run_id=run_id, mode="backtest", params=params)
    renderer.on_event(RenderEvent(
        type="RUN_STARTED",
        message=f"run_id={run_id}  config_hash={config_hash}  resume={params.resume}",
        payload={"run_id": run_id, "config_hash": config_hash, "resume": params.resume},
    ))

    if not params.resume:
        _clear_checkpoint(config_hash)

    cal = load_trade_calendar(rt.tushare)
    start = params.start.replace("-", "")
    end = params.end.replace("-", "")
    trade_dates = cal.sessions_in_range(start, end)
    if not trade_dates:
        raise ValueError(f"no trade sessions in [{start}, {end}]")

    # Resume: replay state from latest shard <= end
    state = _initial_state(params.initial_cash)
    completed = set(_list_shards(config_hash))
    if params.resume and completed:
        latest = max(d for d in completed if d <= end)
        prev = _load_shard(config_hash, latest)
        if prev is not None:
            state = _state_from_dict(prev["state"])
            renderer.on_event(RenderEvent(
                type="RESUME",
                message=f"resuming from {latest}; cash={state.cash:,.0f}",
                payload={"resume_anchor": latest, "cash": state.cash},
            ))

    # Register the run row
    rt.db.execute(
        """
        INSERT INTO checkmate_backtest_runs
            (run_id, config_hash, code_version, start_date, end_date,
             started_at, status, config_json)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'running', ?)
        """,
        [run_id, config_hash, params.code_version, start, end,
         json.dumps(_cfg_to_dict(params), ensure_ascii=False, default=str)],
    )
    rt.db.execute(
        """
        INSERT INTO checkmate_runs
            (run_id, mode, trade_date, status, started_at, params_json)
        VALUES (?, 'backtest', ?, 'running', CURRENT_TIMESTAMP, ?)
        """,
        [run_id, end, json.dumps({"config_hash": config_hash}, ensure_ascii=False)],
    )

    n_fills_total = 0
    last_equity = state.cash

    try:
        for trade_date in trade_dates:
            if params.resume and trade_date in completed:
                shard = _load_shard(config_hash, trade_date)
                if shard is not None:
                    state = _state_from_dict(shard["state"])
                    n_fills_total += int(shard.get("n_fills", 0))
                    last_equity = float(shard.get("equity", last_equity))
                    continue
            payload = execute_day(rt, state, trade_date, run_id, params, renderer=renderer)
            _save_shard(config_hash, trade_date, payload)
            n_fills_total += int(payload["n_fills"])
            last_equity = float(payload["equity"])
    except Exception as exc:
        # Mark backtest_runs row failed; leave checkpoint intact for resume.
        rt.db.execute(
            "UPDATE checkmate_backtest_runs SET status='failed', finished_at=CURRENT_TIMESTAMP "
            "WHERE run_id=?", [run_id],
        )
        rt.db.execute(
            "UPDATE checkmate_runs SET status='failed', exit_code=1, "
            "finished_at=CURRENT_TIMESTAMP, error=? WHERE run_id=?",
            [str(exc), run_id],
        )
        raise

    # On success, finalise rows. Checkpoint stays unless caller wipes via --fresh
    # on the next run; tests that want a clean slate can call _clear_checkpoint.
    rt.db.execute(
        """
        UPDATE checkmate_backtest_runs
        SET status='success', finished_at=CURRENT_TIMESTAMP,
            metrics_json=?
        WHERE run_id=?
        """,
        [json.dumps({
            "final_equity": last_equity,
            "final_cash": state.cash,
            "max_drawdown": state.max_drawdown,
            "n_fills": n_fills_total,
            "n_days": len(trade_dates),
        }, ensure_ascii=False), run_id],
    )
    rt.db.execute(
        "UPDATE checkmate_runs SET status='success', exit_code=0, "
        "finished_at=CURRENT_TIMESTAMP, summary_json=? WHERE run_id=?",
        [json.dumps({"n_fills": n_fills_total, "n_days": len(trade_dates)},
                    ensure_ascii=False), run_id],
    )

    outcome = BacktestOutcome(
        run_id=run_id, config_hash=config_hash,
        start=start, end=end, n_days=len(trade_dates),
        n_fills=n_fills_total,
        final_equity=last_equity,
        final_cash=state.cash,
        max_drawdown=state.max_drawdown,
    )
    try:
        renderer.on_event(RenderEvent(
            type="RUN_FINISHED",
            message=(
                f"backtest done — equity={last_equity:,.2f}  "
                f"max_dd={state.max_drawdown*100:.2f}%  fills={n_fills_total}"
            ),
            payload={"n_fills": n_fills_total, "n_days": len(trade_dates),
                     "final_equity": last_equity, "max_drawdown": state.max_drawdown},
        ))
        renderer.on_run_finish(outcome)
    except Exception:  # noqa: BLE001
        logger.exception("renderer on_run_finish raised")
    try:
        renderer.close()
    except Exception:  # noqa: BLE001
        logger.exception("renderer close raised")
    return outcome


__all__ = [
    "BacktestParams",
    "BacktestOutcome",
    "BacktestState",
    "compute_config_hash",
    "execute_day",
    "run_backtest",
]
