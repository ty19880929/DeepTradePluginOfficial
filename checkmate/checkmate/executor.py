"""Backtest order executor — single-session matching against raw prices.

Contract (development_plan §11 + iteration_tasks.md §5 PR-4.1):

* **Inputs are pure dataclasses + pandas frames** — no DB / Tushare access.
  The runner pre-fetches daily + stk_limit for ``trade_date`` and hands them
  in. This keeps unit-testing deterministic and makes the function safe to
  call from multiple backtest workers in the future.

* **Raw prices everywhere.** Limits, gaps, and fills compare RAW close /
  open / pre_close to RAW up_limit / down_limit (Tushare ``stk_limit``).
  ``qfq`` views are recomputed by the report layer when ATR cross-checks
  are needed; the executor itself is exchange-truth.

* **T+1 rule.** Sell orders for ts_codes whose entry_date equals
  ``trade_date`` are rejected with ``cancel_reason='t1_blocked'`` and a
  risk-event entry. The caller (BacktestRunner) typically filters these
  upstream from signals.py; the duplicate check here is defence in depth.

* **No partial fills (v0.1).** A buy that can't take its full size in one
  trade gets cancelled entirely; downsizing happens upstream in
  :func:`checkmate.risk.size_position`. Partial fills + downsize-on-gap
  are tracked under Iter-7 (v0.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import ExecutionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PendingOrder:
    """Order awaiting execution on ``simulate_session``'s trade_date.

    Buys originate from yesterday's accepted entry signals; sells from
    yesterday's exit signals and any carried-over (deferred) prior sells.
    """

    ts_code: str
    side: str            # 'buy' / 'sell'
    shares: int
    signal_date: str     # the trade_date the signal was generated
    stop_price: float | None = None     # for buys, written to the new Position
    risk_R: float | None = None         # per-share 1R, ditto
    reason_code: str = ""               # signal_type for buys, exit_reason for sells
    defer_count: int = 0                # incremented each session this order was deferred
    # T+1 check: provided by caller for sells against a position opened today.
    same_day_entry: bool = False
    # PR-7.1 (v0.4.0): 20-session average trade amount (yuan). When
    # ``ExecutionConfig.slippage_model == "dynamic"`` the executor uses
    # this to pick a per-order bps via the configured curve. Left as None,
    # the executor falls back to the fixed ``cfg.slippage_bps``.
    amount_20d_avg: float | None = None


@dataclass
class Fill:
    """One executed trade row — maps to ``checkmate_trades`` schema."""

    ts_code: str
    side: str
    shares: int
    fill_price_raw: float
    fill_date: str
    order_date: str
    cost_breakdown: dict[str, float]
    reason_code: str = ""
    cancel_reason: str | None = None  # always None on a Fill (set on Cancel)


@dataclass
class Cancel:
    """An order that won't fill — held aside for the trade ledger / explain CLI."""

    ts_code: str
    side: str
    shares: int
    order_date: str
    cancel_reason: str
    defer_count: int = 0  # for distinguishing "tried for 5 sessions" from "instant cancel"


@dataclass
class RiskEvent:
    """A non-fillable observation worth logging (e.g. T+1 stop pierce)."""

    ts_code: str
    event_type: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionReport:
    """Output of :func:`simulate_session`."""

    trade_date: str
    fills: list[Fill] = field(default_factory=list)
    cancels: list[Cancel] = field(default_factory=list)
    deferred: list[PendingOrder] = field(default_factory=list)
    risk_events: list[RiskEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------


def impact_bps(
    order_value: float, amount_20d_avg: float | None, cfg: ExecutionConfig,
) -> float:
    """Market-impact basis points for an order of ``order_value`` yuan.

    PR-7.3: square-root model. Participation = order_value / amount_20d_avg.
    Returns 0 when ``cfg.impact_model == "none"`` (back-compat default), when
    no liquidity proxy is available, or when participation is below
    ``cfg.impact_min_participation`` (the bid-ask spread absorbs tiny
    orders without measurable price push).

    Formula (sqrt model)::

        bps = impact_coefficient * sqrt(participation) * 100

    Calibration anchors:
      participation =  1% → 10 bps
      participation =  5% → ~22 bps
      participation = 10% → ~32 bps
      participation = 25% → 50 bps (where ``impact_coefficient = 1.0``)
    """
    import math  # noqa: PLC0415

    if cfg.impact_model == "none":
        return 0.0
    if not amount_20d_avg or amount_20d_avg <= 0 or order_value <= 0:
        return 0.0
    participation = order_value / amount_20d_avg
    if participation < cfg.impact_min_participation:
        return 0.0
    return float(cfg.impact_coefficient) * math.sqrt(participation) * 100.0


def _effective_slippage_bps(
    cfg: ExecutionConfig, amount_20d_avg: float | None,
) -> float:
    """Return the per-order slippage in basis points.

    Routes via :func:`dynamic_slippage_bps` when ``cfg.slippage_model ==
    "dynamic"`` AND a non-zero ``amount_20d_avg`` is available; otherwise
    falls back to the fixed ``cfg.slippage_bps`` (v0.1.x behaviour).
    """
    if cfg.slippage_model == "dynamic" and amount_20d_avg and amount_20d_avg > 0:
        return dynamic_slippage_bps(amount_20d_avg, cfg)
    return float(cfg.slippage_bps)


def dynamic_slippage_bps(amount_20d_avg: float, cfg: ExecutionConfig) -> float:
    """Piecewise-linear interpolation in ``log10(amount)`` space.

    PR-7.1: micro-caps pay 30 bps, mega-caps 2 bps (defaults). The curve is
    five breakpoints; we linearly interpolate between them and clamp at the
    extremes. The function is pure and deterministic — same input always
    yields the same output.

    Why log space: tradable A-share daily amounts span six orders of magnitude
    (10万-100亿). A linear curve in raw amount would compress the mid-cap range
    to a tiny segment; log space gives roughly equal bandwidth per decade.
    """
    import math  # noqa: PLC0415

    curve = sorted(cfg.slippage_bps_curve)
    if not curve:
        return float(cfg.slippage_bps)

    x = math.log10(max(1e-9, amount_20d_avg))
    # Clamp below the lowest breakpoint
    if x <= curve[0][0]:
        return float(curve[0][1])
    # Clamp above the highest breakpoint
    if x >= curve[-1][0]:
        return float(curve[-1][1])
    # Linear interpolation between flanking breakpoints
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return float(y0 + t * (y1 - y0))
    # Defensive — should be unreachable given the clamps above.
    return float(cfg.slippage_bps)


def compute_costs(
    side: str,
    shares: int,
    fill_price: float,
    cfg: ExecutionConfig,
    *,
    amount_20d_avg: float | None = None,
) -> dict[str, float]:
    """Per-trade cost decomposition. Returns a dict with 5 keys (PR-7.3):

      * ``commission`` — broker commission, floored at ``commission_min``
      * ``stamp_tax`` — sell-side only (per A-share rules)
      * ``transfer_fee`` — uniform both sides (v0.1 simplification)
      * ``slippage`` — bps slippage on the notional (PR-7.1 dynamic, or
        :attr:`cfg.slippage_bps` in fixed mode)
      * ``impact`` — market-impact bps from :func:`impact_bps` (PR-7.3);
        ``0.0`` when ``cfg.impact_model == "none"`` or no liquidity proxy.

    Total dollar cost: sum of the dict's values.
    """
    notional = shares * fill_price
    commission = max(notional * cfg.commission_rate, cfg.commission_min)
    stamp_tax = notional * cfg.stamp_tax_rate if side == "sell" else 0.0
    transfer_fee = notional * cfg.transfer_fee_rate
    slip_bps = _effective_slippage_bps(cfg, amount_20d_avg)
    slippage = notional * (slip_bps / 10_000.0)
    impact_b = impact_bps(notional, amount_20d_avg, cfg)
    impact = notional * (impact_b / 10_000.0)
    return {
        "commission": round(commission, 4),
        "stamp_tax": round(stamp_tax, 4),
        "transfer_fee": round(transfer_fee, 4),
        "slippage": round(slippage, 4),
        "impact": round(impact, 4),
    }


def _slip_price(
    side: str,
    raw_open: float,
    cfg: ExecutionConfig,
    *,
    amount_20d_avg: float | None = None,
    order_value: float | None = None,
) -> float:
    """Apply slippage + impact to the fill price.

    PR-7.1: dynamic slippage bps drive direction-aware drift.
    PR-7.3: impact bps (when ``cfg.impact_model != "none"`` and
    ``order_value`` known) get added in the same direction — buys pay
    more, sells receive less. The two components surface separately in
    :func:`compute_costs`'s output dict but stack onto a single fill price
    here.
    """
    slip_b = _effective_slippage_bps(cfg, amount_20d_avg)
    imp_b = impact_bps(order_value or 0.0, amount_20d_avg, cfg)
    total_bps = slip_b + imp_b
    delta = raw_open * (total_bps / 10_000.0)
    return raw_open + delta if side == "buy" else raw_open - delta


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _row_by_code(df: pd.DataFrame, ts_code: str) -> dict | None:
    """Return the single row for ``ts_code`` as a dict, or None if missing."""
    if df is None or df.empty:
        return None
    sub = df[df["ts_code"].astype(str) == ts_code]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _is_at_up_limit(open_raw: float, up_limit: float | None, *, tol: float = 1e-4) -> bool:
    """True if open is at-or-above today's up_limit (limit-up open)."""
    if up_limit is None or up_limit <= 0:
        return False
    return open_raw >= up_limit - tol


def _is_at_down_limit(open_raw: float, down_limit: float | None, *, tol: float = 1e-4) -> bool:
    """True if open is at-or-below today's down_limit (limit-down open)."""
    if down_limit is None or down_limit <= 0:
        return False
    return open_raw <= down_limit + tol


# ---------------------------------------------------------------------------
# Core: simulate_session
# ---------------------------------------------------------------------------


def simulate_session(
    pending: list[PendingOrder],
    trade_date: str,
    prices_raw: pd.DataFrame,
    stk_limit: pd.DataFrame,
    cfg: ExecutionConfig | None = None,
) -> SessionReport:
    """Try to fill every order in ``pending`` against today's prices + limits.

    Decision matrix (development_plan §11):

      * BUY at next-session open. Cancel if open ≥ up_limit (limit-up open),
        if no quote (suspended), or if ``(open / pre_close - 1) >
        cfg.max_gap_up_pct`` (gap-up too large). Otherwise fill at
        ``open_raw * (1 + slippage)``.
      * SELL at next-session open. If open ≤ down_limit (limit-down open),
        defer to next session. After ``cfg.max_defer_days`` consecutive
        defers, cancel (the symbol is presumed wedged).
      * T+1: any sell carrying ``same_day_entry=True`` is rejected with
        ``cancel_reason='t1_blocked'`` and a RiskEvent gets recorded.
    """
    cfg = cfg or ExecutionConfig()
    report = SessionReport(trade_date=trade_date)

    for order in pending:
        price_row = _row_by_code(prices_raw, order.ts_code)
        limit_row = _row_by_code(stk_limit, order.ts_code)
        if price_row is None or "open" not in price_row or pd.isna(price_row.get("open")):
            report.cancels.append(Cancel(
                ts_code=order.ts_code, side=order.side, shares=order.shares,
                order_date=order.signal_date,
                cancel_reason="suspended_no_quote",
                defer_count=order.defer_count,
            ))
            continue

        open_raw = float(price_row["open"])
        pre_close = float(price_row.get("pre_close") or 0.0)
        up_limit = float(limit_row["up_limit"]) if limit_row and "up_limit" in limit_row else None
        down_limit = float(limit_row["down_limit"]) if limit_row and "down_limit" in limit_row else None

        if order.side == "buy":
            # 1) limit-up at open → cancel
            if _is_at_up_limit(open_raw, up_limit):
                report.cancels.append(Cancel(
                    ts_code=order.ts_code, side="buy", shares=order.shares,
                    order_date=order.signal_date,
                    cancel_reason="limit_up_open",
                ))
                continue
            # 2) gap-up too large → cancel
            if pre_close > 0:
                gap = open_raw / pre_close - 1.0
                if gap > cfg.max_gap_up_pct:
                    report.cancels.append(Cancel(
                        ts_code=order.ts_code, side="buy", shares=order.shares,
                        order_date=order.signal_date,
                        cancel_reason=f"gap_up_too_large ({gap*100:.2f}% > {cfg.max_gap_up_pct*100:.2f}%)",
                    ))
                    continue
            # 3) fill — use the pre-slippage notional as the impact-model
            # input (close enough; the post-slippage notional shifts by <1bps).
            est_notional = order.shares * open_raw
            fill_price = _slip_price("buy", open_raw, cfg,
                                      amount_20d_avg=order.amount_20d_avg,
                                      order_value=est_notional)
            costs = compute_costs("buy", order.shares, fill_price, cfg,
                                   amount_20d_avg=order.amount_20d_avg)
            report.fills.append(Fill(
                ts_code=order.ts_code, side="buy", shares=order.shares,
                fill_price_raw=round(fill_price, 4),
                fill_date=trade_date, order_date=order.signal_date,
                cost_breakdown=costs, reason_code=order.reason_code,
            ))
        elif order.side == "sell":
            # T+1: sells against same-day entries are illegal on A-share.
            if order.same_day_entry:
                report.cancels.append(Cancel(
                    ts_code=order.ts_code, side="sell", shares=order.shares,
                    order_date=order.signal_date,
                    cancel_reason="t1_blocked",
                ))
                report.risk_events.append(RiskEvent(
                    ts_code=order.ts_code,
                    event_type="t1_sell_blocked",
                    message="sell on entry day blocked by T+1 rule",
                    payload={"shares": order.shares, "reason_code": order.reason_code},
                ))
                continue

            # 1) limit-down at open → defer (or cancel after max attempts)
            if _is_at_down_limit(open_raw, down_limit):
                deferred = PendingOrder(
                    ts_code=order.ts_code, side="sell", shares=order.shares,
                    signal_date=order.signal_date,
                    stop_price=order.stop_price, risk_R=order.risk_R,
                    reason_code=order.reason_code,
                    defer_count=order.defer_count + 1,
                    same_day_entry=False,
                )
                if deferred.defer_count > cfg.max_defer_days:
                    report.cancels.append(Cancel(
                        ts_code=order.ts_code, side="sell", shares=order.shares,
                        order_date=order.signal_date,
                        cancel_reason=f"limit_down_wedged ({deferred.defer_count} sessions)",
                        defer_count=deferred.defer_count,
                    ))
                else:
                    report.deferred.append(deferred)
                continue

            # 2) fill (slippage + impact drag price *down* on sells)
            est_notional = order.shares * open_raw
            fill_price = _slip_price("sell", open_raw, cfg,
                                      amount_20d_avg=order.amount_20d_avg,
                                      order_value=est_notional)
            costs = compute_costs("sell", order.shares, fill_price, cfg,
                                   amount_20d_avg=order.amount_20d_avg)
            report.fills.append(Fill(
                ts_code=order.ts_code, side="sell", shares=order.shares,
                fill_price_raw=round(fill_price, 4),
                fill_date=trade_date, order_date=order.signal_date,
                cost_breakdown=costs, reason_code=order.reason_code,
            ))
        else:
            # Unknown side → log and skip; never crash the session.
            logger.warning("unknown order side %r for %s", order.side, order.ts_code)
            report.cancels.append(Cancel(
                ts_code=order.ts_code, side=order.side, shares=order.shares,
                order_date=order.signal_date,
                cancel_reason=f"unknown_side ({order.side})",
            ))

    return report


__all__ = [
    "ExecutionConfig",
    "PendingOrder",
    "Fill",
    "Cancel",
    "RiskEvent",
    "SessionReport",
    "simulate_session",
    "compute_costs",
    "dynamic_slippage_bps",
    "impact_bps",
]
