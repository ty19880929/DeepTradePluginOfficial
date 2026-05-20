"""Checkmate strategy configuration.

Iter-1 PR-1.3 ships :class:`UniverseConfig` only. Iter-2/3 will add
``RegimeConfig`` / ``TrendConfig`` / ``EntryConfig`` / ``RiskConfig`` /
``ExitConfig`` / ``ExecutionConfig`` / ``ReportingConfig`` as the matching
pipeline stages land, plus a top-level :class:`CheckmateConfig` aggregate
loaded from ``ConfigService`` under the ``checkmate.*`` namespace.

Defaults follow development_plan §13.2 — main-board A-share long-only
trend follower with mid-cap-and-up liquidity preference. All thresholds are
plain numbers (no enums / Pydantic for v0.1) so tests can override with a
dataclass replacement in a single line.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionConfig:
    """Backtest executor parameters (development_plan §11).

    Costs are decomposed so ``checkmate_trades.cost_breakdown`` can store
    per-component figures rather than a single opaque "fees" number —
    drift detection between sims and live becomes a column-level diff.

    All comparisons happen on RAW (un-adjusted) prices: limits, gap-ups,
    and stop fills are exchange-truth events.
    """

    # Costs (fractional or 万分之 unless noted)
    commission_rate: float = 0.0003        # 万三 broker commission
    commission_min: float = 5.0            # 5 元 minimum per trade (yuan)
    stamp_tax_rate: float = 0.001          # 千一, sell-side only
    transfer_fee_rate: float = 0.00002     # 万分之二, both sides (上交所 only in reality;
                                            # we apply uniformly for v0.1 simplicity)
    slippage_bps: int = 5                  # fixed-mode slippage (used when model="fixed")

    # PR-7.1 (v0.4.0): dynamic slippage by liquidity bucket. ``"fixed"`` keeps
    # v0.1.x behaviour; ``"dynamic"`` swaps in a piecewise-linear curve in
    # log10(amount_20d_avg) space, so micro-caps pay more bps than mega-caps.
    # The curve is a list of ``(log10_amount, bps)`` breakpoints — caller can
    # tune by overriding this field on the dataclass.
    slippage_model: str = "fixed"          # "fixed" | "dynamic"
    slippage_bps_curve: tuple[tuple[float, float], ...] = (
        # (log10(amount yuan), slippage bps)
        (6.0, 30.0),   # 1e6   元/天   →  30 bps (micro)
        (7.0, 20.0),   # 1e7  →  20 bps (small)
        (8.0, 10.0),   # 1e8  →  10 bps (mid)
        (9.0,  5.0),   # 1e9  →   5 bps (large)
        (10.0, 2.0),   # 1e10 →   2 bps (mega)
    )

    # PR-7.3 (v0.4.0): market-impact cost. Default ``"none"`` keeps the
    # cost dict's ``impact`` field at 0.0 (back-compat with v0.3.x). The
    # ``"sqrt"`` model uses the classic Almgren-Chriss-style functional
    # form impact_bps = ``impact_coefficient * sqrt(participation) * 100``
    # where ``participation = order_value / amount_20d_avg``.
    # ``impact_min_participation`` zeroes out impact for tiny orders (the
    # bid-ask spread already absorbs them).
    impact_model: str = "none"             # "none" | "sqrt"
    impact_coefficient: float = 1.0
    impact_min_participation: float = 0.005  # 0.5% of daily amount

    # Order-cancellation thresholds
    max_gap_up_pct: float = 0.05           # cancel buy if open > prev_close*(1+5%)
    max_defer_days: int = 5                # sell stuck at limit-down for N sessions → cancel


@dataclass
class RiskConfig:
    """Risk budgeting + portfolio-level constraints (development_plan §9).

    The sizing branch uses a fixed-fractional risk budget — each new
    position can lose at most ``risk_per_trade`` × portfolio_value on a
    move to the stop. Per-share risk is ``entry_price - stop_price`` so
    shares = floor(risk_dollars / per_share_risk), then truncated to the
    A-share lot size (100).

    The portfolio-constraint branch sorts proposals by score desc and
    accepts them one by one, rejecting any that would violate:

      * ``max_single_weight``     — single ts_code's weight cap
      * ``max_industry_weight``   — sector concentration cap (uses the
        caller-supplied ``industry_value`` map)
      * ``max_new_entries_per_day`` — total new entries today
      * ``regime_entry_caps[regime]`` — regime-conditional override

    Defaults are conservative and tunable via ``ConfigService`` in v0.2+.
    """

    risk_per_trade: float = 0.01            # 1% of portfolio per trade
    max_single_weight: float = 0.10         # 10% cap per ts_code
    max_industry_weight: float = 0.30       # 30% cap per industry
    max_new_entries_per_day: int = 3
    lot_size: int = 100                     # A-share standard lot

    # Per-regime new-entry caps. ``risk`` cuts all new opens; ``weak`` allows
    # one defensive add; ``neutral`` two; ``strong`` matches the daily cap.
    regime_entry_caps: dict[str, int] = field(default_factory=lambda: {
        "strong":  3,
        "neutral": 2,
        "weak":    1,
        "risk":    0,
    })


@dataclass
class ExitConfig:
    """Exit rule thresholds (development_plan §10.3).

    Five rules evaluated in priority order:

    1. ``hard_stop`` — close < stop_price (RAW prices, exchange-truth)
    2. ``risk_regime`` — current regime tag is ``risk`` (broad capitulation)
    3. ``defensive_profit`` — gave back ``defensive_profit_retrace_R`` from a
       peak ≥ ``defensive_profit_peak_R``; transitions to ``defensive`` state
       (tighter stop) rather than full exit
    4. ``trailing_stop`` — close < peak × (1 - ``trailing_pct``)
    5. ``time_exit`` — held > ``max_hold_days`` AND pnl_R < ``time_exit_min_pnl_R``

    T+1 (settlement rule on A-share) blocks every exit signal on the same
    day the position was opened — the evaluator surfaces a ``t1_blocked``
    flag so the runner can record a risk event without actually selling.
    """

    # risk_regime
    risk_regime_tag: str = "risk"

    # defensive_profit (state transition holding → defensive)
    defensive_profit_peak_R: float = 3.0
    defensive_profit_retrace_R: float = 1.5

    # trailing_stop (used in defensive state, but also from holding if peak was high)
    trailing_pct: float = 0.15

    # time_exit
    max_hold_days: int = 120
    time_exit_min_pnl_R: float = 1.0


@dataclass
class EntryConfig:
    """Thresholds for the three entry signal types (development_plan §8.1).

    All ``pct_chg`` values are fractional (0.08 → 8%). ts_code prefix decides
    which board cap applies — see :func:`checkmate.signals._board_pct_cap`.
    """

    # ---- breakout (突破)
    breakout_lookback: int = 40          # session high to clear (exclusive of today)
    breakout_amount_ratio: float = 1.2   # today vs prior `breakout_amount_lookback` avg
    breakout_amount_lookback: int = 20
    pct_chg_cap_main_board: float = 0.08  # ≤ 8% today — don't chase post-spike
    pct_chg_cap_chinext: float = 0.11     # 创业板 (300.xx)
    pct_chg_cap_star: float = 0.11        # 科创板 (688.xx)

    # ---- pullback (回踩)
    pullback_ma20_tol: float = 0.03      # touched ±3% of MA20 within the window
    pullback_touch_window: int = 10      # session window to scan for the touch
    pullback_platform_window: int = 5    # close must clear this short-term high
    pullback_trend_ma_ratio: float = 1.0 # require ma20 > pullback_trend_ma_ratio × ma60

    # ---- continuation (趋势延续)
    continuation_rs60_min: float = 0.80  # rs60_pctile threshold (top quintile)
    continuation_breakout_lookback: int = 10  # close > recent N-session high

    # ---- stop placement (used by the signals orchestrator when building
    # ProposedEntry rows): stop_price = entry_price - atr_stop_mult * atr20.
    # 2.0 is the standard "two ATR" trend-follower default; widens stops on
    # high-vol names so risk_per_trade-sized positions are smaller.
    atr_stop_mult: float = 2.0


@dataclass
class RegimeConfig:
    """Market regime classifier thresholds.

    Two index inputs feed the index-trend half of the classifier (中证全指 +
    沪深300, both relative to their 120-session MA). The breadth half reads
    ``checkmate_features_daily`` for the trade_date and computes the share
    of names with ``close_qfq > ma120``.

    Default tag mapping:
      breadth >= ``breadth_strong`` AND both indices above → ``strong``
      breadth >= ``breadth_weak``  AND ≥ 1 index above   → ``neutral``
      breadth <  ``breadth_risk``                          → ``risk``
      otherwise                                            → ``weak``

    ``exposure_cap`` is the portfolio-level long exposure ceiling for the
    risk module (Iter-3) — 1.0 means full deployment, 0.0 means defensive
    cash.
    """

    index_csi_code: str = "000985.CSI"
    index_hs300_code: str = "000300.SH"
    ma_window: int = 120

    breadth_strong: float = 0.60
    breadth_weak: float = 0.40
    breadth_risk: float = 0.20

    exposure_strong: float = 1.0
    exposure_neutral: float = 0.6
    exposure_weak: float = 0.3
    exposure_risk: float = 0.0


@dataclass
class FeaturesConfig:
    """Feature-computation thresholds & windows.

    All windows count *trade sessions*, not calendar days.
    """

    # Window the caller must supply for per-symbol qfq daily input. ma120 +
    # one extra day for the ret_120 anchor → 121; we add a 10-day buffer so
    # the very first row can also be skipped if pre_close is missing.
    min_history_sessions: int = 130

    # Trailing window for amount / turnover stats — matches UniverseConfig.
    liquidity_window: int = 20

    # Trailing window for limit-day frequency.
    limit_window: int = 60

    # Trailing window for drawdown / quiet score / above_ma20_days.
    pullback_window: int = 60

    # ATR / Wilder smoothing window.
    atr_window: int = 20

    # Threshold above which |pct_chg| counts as a limit day. 0.097 catches
    # the standard ±10% main-board limit even after rounding.
    limit_pct_threshold: float = 0.097

    # Score component weights. Must sum to 1.0 (sanity-checked in tests).
    score_weight_trend: float = 0.25
    score_weight_volatility: float = 0.10
    score_weight_strength: float = 0.30
    score_weight_liquidity: float = 0.15
    score_weight_pullback: float = 0.20


@dataclass
class UniverseConfig:
    """Eligibility thresholds for daily universe construction."""

    # New-listing exclusion: ``listed_days_min`` calendar days from list_date
    # to trade_date. 250 ≈ one trading year + buffer; default is conservative
    # because newly listed names dominate Z-score outliers in early features.
    listed_days_min: int = 250

    # Liquidity floor: 20-day average ``amount`` (yuan) must clear this bar.
    # 50_000_000 元 / 天 (5 千万 / 5000 万) excludes the bottom decile of the
    # main board without touching mid caps. Tushare daily_basic ``amount`` is
    # 千元 — the orchestrator multiplies by 1000 before comparing.
    amount_20d_avg_min_yuan: float = 50_000_000.0

    # Thin-trading floor: number of actual daily rows in the trailing 20-session
    # window. 18/20 leaves room for one isolated suspension (e.g. shareholder
    # meeting) without disqualifying mid caps.
    thin_trading_min_days: int = 18

    # One-way (一字) limit detection: fraction of 20-day rows where
    # ``high == low`` (no intraday range, almost certainly a limit-stuck day).
    # 0.25 ≈ 5+ stuck days out of 20 → tradeability is suspect.
    one_way_limit_max_ratio: float = 0.25

    # Price band — exclude penny stocks (default low 2 yuan) and optionally
    # cap the high end. ``None`` on either side disables that bound.
    price_band_low: float | None = 2.0
    price_band_high: float | None = None

    # Trailing window size (sessions) for amount / turnover stats. The full
    # window is ``window_sessions`` sessions ending at trade_date inclusive.
    window_sessions: int = 20
