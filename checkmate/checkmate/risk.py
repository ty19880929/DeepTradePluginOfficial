"""Position sizing + portfolio constraints for Checkmate.

Pure-function module — never touches the DB or Tushare. Inputs and outputs
are plain dataclasses so the runner (backtest or live) can compose the
sizing/filter step in isolation and the tests can drive it without setting
up fixtures.

Functions
---------

* :func:`size_position` — translate a ``(entry_price, stop_price, portfolio_value)``
  tuple into an integer share count, honouring the 100-share A-share lot.
* :func:`apply_portfolio_constraints` — accept a list of ``ProposedEntry``
  rows and filter them against the four caps in :class:`~checkmate.config.RiskConfig`
  (single, industry, daily count, regime).

Inputs are sorted by score descending before evaluation so that, when caps
bite, the higher-quality proposals survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import RiskConfig


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProposedEntry:
    """A signal turned into a sizing input — the runner builds these from
    :class:`~checkmate.signals.Signal` plus the symbol's industry tag."""

    ts_code: str
    entry_price: float
    stop_price: float
    industry: str | None = None
    score: float | None = None
    signal_type: str = ""
    explain: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionView:
    """Slim projection of an existing position used by portfolio constraints.

    ``market_value`` is the caller's most recent mark-to-market dollar value
    of the position. Living separately from :class:`~checkmate.signals.Position`
    so the runner can compute it once per trade_date and reuse here without
    forcing the executor's column set on this module.
    """

    ts_code: str
    industry: str | None
    market_value: float


@dataclass
class SizedOrder:
    """Output row — one per :class:`ProposedEntry`, accepted or rejected.

    Reasonable runners persist every row (accepted + rejected) so the explain
    page can show "you proposed X but it was dropped because Y" without
    re-running the filter.
    """

    ts_code: str
    side: str
    shares: int
    entry_price: float
    stop_price: float
    risk_R: float
    weight: float
    industry: str | None
    score: float | None
    signal_type: str
    explain: dict[str, Any]
    accepted: bool
    cancel_reason: str | None = None


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def size_position(
    entry_price: float,
    stop_price: float,
    portfolio_value: float,
    cfg: RiskConfig | None = None,
) -> int:
    """Return the share count obeying the ATR/risk budget and lot rule.

    Returns ``0`` for any degenerate input (non-positive price, stop ≥ entry,
    non-positive portfolio value, zero risk budget). The caller treats 0 as
    "no order"; a downstream INSERT with shares=0 should be filtered.
    """
    cfg = cfg or RiskConfig()
    if entry_price <= 0 or portfolio_value <= 0 or cfg.lot_size <= 0:
        return 0
    if stop_price >= entry_price or stop_price < 0:
        return 0
    risk_per_share = entry_price - stop_price
    risk_dollars = portfolio_value * cfg.risk_per_trade
    if risk_dollars <= 0 or risk_per_share <= 0:
        return 0
    raw = int(risk_dollars // risk_per_share)
    lots = raw // cfg.lot_size
    return max(0, lots * cfg.lot_size)


# ---------------------------------------------------------------------------
# Portfolio constraints
# ---------------------------------------------------------------------------


def _industry_value_from_positions(positions: list[PositionView]) -> dict[str, float]:
    """Aggregate market value by industry. ``None`` industries collapse into
    a single ``"_unknown"`` bucket for cap accounting purposes."""
    agg: dict[str, float] = {}
    for p in positions:
        key = p.industry or "_unknown"
        agg[key] = agg.get(key, 0.0) + max(0.0, p.market_value)
    return agg


def _regime_cap(regime: str | None, cfg: RiskConfig) -> int:
    if regime is None:
        return cfg.max_new_entries_per_day
    return cfg.regime_entry_caps.get(regime, cfg.max_new_entries_per_day)


def apply_portfolio_constraints(
    proposals: list[ProposedEntry],
    current_positions: list[PositionView],
    *,
    portfolio_value: float,
    regime: str | None = None,
    cfg: RiskConfig | None = None,
) -> list[SizedOrder]:
    """Filter proposals through the 4-stage constraint cascade.

    Stage 1 — regime + daily caps: the effective new-entry budget is
        ``min(cfg.max_new_entries_per_day, cfg.regime_entry_caps[regime])``.
        Excess proposals get ``cancel_reason="daily_cap" / "regime_cap"``.

    Stage 2 — sizing: each surviving proposal is sized with
        :func:`size_position`. Zero shares (typically: stop too tight for the
        risk budget) yields ``cancel_reason="zero_shares"``.

    Stage 3 — single position cap: rejects if the new order's weight exceeds
        ``cfg.max_single_weight``.

    Stage 4 — industry cap: rejects if (current industry value + this
        order's value) / portfolio_value would exceed
        ``cfg.max_industry_weight``. Already-accepted orders in the same
        industry feed into the running tally so caps don't leak across
        proposals on the same day.

    Sort order: proposals are scored descending (``None`` scores trail) so
    higher-quality picks consume cap budget first.
    """
    cfg = cfg or RiskConfig()
    out: list[SizedOrder] = []

    sorted_props = sorted(
        proposals,
        key=lambda p: -(p.score if p.score is not None else float("-inf")),
    )

    # Running counters
    accepted_today = 0
    daily_cap = min(cfg.max_new_entries_per_day, _regime_cap(regime, cfg))
    industry_value = _industry_value_from_positions(current_positions)

    for prop in sorted_props:
        # Default: build a SizedOrder, then mark accepted / cancel_reason.
        risk_per_share = max(0.0, prop.entry_price - prop.stop_price)
        order = SizedOrder(
            ts_code=prop.ts_code,
            side="buy",
            shares=0,
            entry_price=prop.entry_price,
            stop_price=prop.stop_price,
            risk_R=risk_per_share,
            weight=0.0,
            industry=prop.industry,
            score=prop.score,
            signal_type=prop.signal_type,
            explain=dict(prop.explain),
            accepted=False,
            cancel_reason=None,
        )

        # --- Stage 1: daily / regime cap ----------------------------------
        if accepted_today >= daily_cap:
            order.cancel_reason = (
                "regime_cap" if daily_cap < cfg.max_new_entries_per_day else "daily_cap"
            )
            out.append(order)
            continue

        # --- Stage 2: sizing ----------------------------------------------
        shares = size_position(prop.entry_price, prop.stop_price, portfolio_value, cfg)
        if shares <= 0:
            order.cancel_reason = "zero_shares"
            out.append(order)
            continue
        order.shares = shares
        order_value = shares * prop.entry_price
        order.weight = order_value / portfolio_value if portfolio_value > 0 else 0.0

        # --- Stage 3: single-weight cap -----------------------------------
        if order.weight > cfg.max_single_weight:
            order.cancel_reason = (
                f"single_weight_cap (weight={order.weight:.4f} > "
                f"{cfg.max_single_weight:.4f})"
            )
            out.append(order)
            continue

        # --- Stage 4: industry cap ----------------------------------------
        industry_key = prop.industry or "_unknown"
        new_industry_value = industry_value.get(industry_key, 0.0) + order_value
        new_industry_weight = (
            new_industry_value / portfolio_value if portfolio_value > 0 else 0.0
        )
        if new_industry_weight > cfg.max_industry_weight:
            order.cancel_reason = (
                f"industry_cap (industry={industry_key}, "
                f"would_be={new_industry_weight:.4f} > {cfg.max_industry_weight:.4f})"
            )
            out.append(order)
            continue

        # --- Accepted ------------------------------------------------------
        order.accepted = True
        industry_value[industry_key] = new_industry_value
        accepted_today += 1
        out.append(order)

    return out
