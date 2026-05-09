"""Data layer for the volume-anomaly strategy.

Two distinct data flows:
    screen_anomalies(...)  — apply local rules to find new anomaly hits on T
    collect_analyze_bundle(...) — read watchlist + assemble per-stock context for LLM

Reuses limit_up_board's main_board_filter / FIELD_UNITS_RAW conventions where
sensible but does NOT import from limit_up_board (plugins are self-contained).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

import pandas as pd

from deeptrade.core.tushare_client import (
    TushareClient,
    TushareUnauthorizedError,
)

from .calendar import TradeCalendar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 0 — resolve trade date (mirrors limit_up_board behaviour)
# ---------------------------------------------------------------------------


def resolve_trade_date(
    now_dt: datetime,
    calendar: TradeCalendar,
    *,
    user_specified: str | None = None,
    allow_intraday: bool = False,
    close_after: time = time(18, 0),
) -> tuple[str, str]:
    """Return (T, T+1).

    T defaults to the most recent CLOSED trade day:
      * if today is open AND now ≥ close_after  → today
      * if today is open AND allow_intraday      → today (intraday banner)
      * else                                     → pretrade_date(today)
    """
    if user_specified:
        T = user_specified
        return T, calendar.next_open(T)

    today = now_dt.strftime("%Y%m%d")
    today_is_open = calendar.is_open(today)
    if today_is_open and (now_dt.time() >= close_after or allow_intraday):
        T = today
    elif today_is_open:
        T = calendar.pretrade_date(today)
    else:
        T = calendar.pretrade_date(today)
    return T, calendar.next_open(T)


# ---------------------------------------------------------------------------
# Main board filter
# ---------------------------------------------------------------------------


def main_board_filter(stock_basic: pd.DataFrame) -> pd.DataFrame:
    """Keep only Shanghai/Shenzhen MAIN board, listed.

    Excludes ChiNext (300xxx), STAR (688xxx), BSE (8xxxxx), CDR.
    """
    if "market" not in stock_basic.columns or "exchange" not in stock_basic.columns:
        raise ValueError("stock_basic missing market/exchange columns")
    df = stock_basic[
        (stock_basic["market"] == "主板") & (stock_basic["exchange"].isin(["SSE", "SZSE"]))
    ].copy()
    if "list_status" in df.columns:
        df = df[df["list_status"] == "L"]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optional API wrapper (transient failure → empty df + reason string)
# ---------------------------------------------------------------------------


def _try_optional(
    tushare: TushareClient, api_name: str, **kwargs: Any
) -> tuple[pd.DataFrame, str | None]:
    """Call an optional tushare API; on transient failure return (empty df, err)."""
    from deeptrade.core.tushare_client import (  # noqa: PLC0415
        TushareRateLimitError,
        TushareServerError,
    )

    try:
        return tushare.call(api_name, **kwargs), None
    except TushareUnauthorizedError as e:
        return pd.DataFrame(), f"unauthorized: {e}"
    except TushareServerError as e:
        return pd.DataFrame(), f"server_error: {e}"
    except TushareRateLimitError as e:
        return pd.DataFrame(), f"rate_limited: {e}"


# ---------------------------------------------------------------------------
# Unit normalizers (per-field; tushare units are heterogeneous)
# ---------------------------------------------------------------------------


FIELD_UNITS_RAW: dict[str, str] = {
    # daily.amount is 千元; daily.vol is 手 (handled separately)
    "amount_daily": "千元",
    # daily_basic
    "circ_mv": "万元",
    "total_mv": "万元",
    "free_share": "万股",
    "float_share": "万股",
    "total_share": "万股",
    # moneyflow (all amounts in 万元)
    "net_mf_amount": "万元",
    "buy_lg_amount": "万元",
    "buy_elg_amount": "万元",
    "buy_md_amount": "万元",
    "buy_sm_amount": "万元",
    "sell_lg_amount": "万元",
    "sell_elg_amount": "万元",
}


def normalize_to_yi(field: str, raw_value: float | None) -> float | None:
    if raw_value is None or pd.isna(raw_value):
        return None
    unit = FIELD_UNITS_RAW.get(field, "元")
    if unit == "元":
        factor = 1e8
    elif unit == "万元":
        factor = 1e4
    elif unit == "千元":
        factor = 1e5
    else:
        return None
    return round(float(raw_value) / factor, 2)


def round2(v: float | None) -> float | None:
    if v is None or pd.isna(v):
        return None
    return round(float(v), 2)


def _opt_int(v: Any) -> int | None:
    if v is None or pd.isna(v):
        return None
    return int(v)


def _normalize_id_cols(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Coerce identifier columns to str so cross-frame sort/compare is stable.

    The tushare-on-disk JSON cache widens "20260428" → 20260428 on round-trip;
    if some rows come fresh from the SDK (str) and others from the cache (int),
    pandas .sort_values()/comparisons raise:
        TypeError: '<' not supported between instances of 'int' and 'str'
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in ("trade_date", "ts_code", "cal_date"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _shift_calendar_days(yyyymmdd: str, days: int) -> str:
    """Naive ±days shift on YYYYMMDD (calendar days, not trade days)."""
    d = datetime.strptime(yyyymmdd, "%Y%m%d") + timedelta(days=days)
    return d.strftime("%Y%m%d")


def _calendar_days_between(earlier: str, later: str) -> int:
    """Calendar-day diff (later - earlier) on YYYYMMDD strings; negative if reversed."""
    d1 = datetime.strptime(earlier, "%Y%m%d")
    d2 = datetime.strptime(later, "%Y%m%d")
    return (d2 - d1).days


# ---------------------------------------------------------------------------
# SCREEN MODE — anomaly screening rules
# ---------------------------------------------------------------------------


# Default lookback window (kept module-level so analyze mode can re-use it
# without depending on the screen-only ScreenRules dataclass).
RULE_LOOKBACK_TRADE_DAYS = 60  # ~3 months

# v0.4.0 P1-3 — T+N realized-return evaluation horizons. F6 decision: keep as a
# module-level constant rather than introducing a new `va_config` table.
EVALUATE_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)
EVALUATE_DEFAULT_LOOKBACK_DAYS: int = 30
EVALUATE_MAX_HORIZON: int = max(EVALUATE_HORIZONS)
EVALUATE_WINDOW_5D = 5
EVALUATE_WINDOW_10D = 10


# v0.3.0 P0-2 — default circ_mv-bucketed turnover thresholds.
# Each tuple is (circ_mv_yi_max, turnover_min, turnover_max). The first bucket
# whose `max` is ≥ the candidate's circ_mv_yi (亿元) wins; boundary values fall
# into the smaller bucket (E4 — `circ_mv_yi ≤ bucket_max`).
DEFAULT_TURNOVER_BUCKETS: list[tuple[float, float, float]] = [
    (50.0, 5.0, 15.0),       # ≤ 50亿 — 微盘
    (200.0, 3.5, 12.0),      # 50–200亿 — 中小盘
    (1000.0, 2.5, 9.0),      # 200–1000亿 — 中盘
    (math.inf, 1.5, 6.0),    # > 1000亿 — 大盘
]


def _bucket_label(bucket_max: float, prev_max: float) -> str:
    """Render a human-readable bucket label like "≤50亿" / "50-200亿" / ">1000亿"."""
    if prev_max <= 0:
        return f"≤{int(bucket_max)}亿"
    if math.isinf(bucket_max):
        return f">{int(prev_max)}亿"
    return f"{int(prev_max)}-{int(bucket_max)}亿"


def _resolve_turnover_bucket(
    circ_mv_yi: float, buckets: list[tuple[float, float, float]]
) -> tuple[int, str, float, float]:
    """Return (idx, label, t_min, t_max) for the first bucket where circ_mv_yi ≤ max."""
    prev_max = 0.0
    for idx, (b_max, t_min, t_max) in enumerate(buckets):
        if circ_mv_yi <= b_max:
            return idx, _bucket_label(b_max, prev_max), t_min, t_max
        prev_max = b_max
    # Past the last bucket (only possible if last bucket isn't math.inf —
    # ScreenRules.__post_init__ guards against that, but be defensive).
    last_max, t_min, t_max = buckets[-1]
    return len(buckets) - 1, _bucket_label(last_max, prev_max), t_min, t_max


@dataclass
class ScreenRules:
    """User-tunable screening thresholds.

    Plan A (v0.2): turnover_max raised 7 → 10 — empirically the dominant
    bottleneck on real funnel data.
    Plan B (v0.2): vol rule split into "short-window must be max" OR
    "long-window top-N", because strict 60d-max disqualifies any stock
    that happened to have a single大量 day in the past 3 months.
    Plan C (v0.2): all knobs collected by configure() at runtime.
    """

    pct_chg_min: float = 5.0
    pct_chg_max: float = 8.0
    body_ratio_min: float = 0.6
    turnover_min: float = 3.0
    turnover_max: float = 10.0  # Plan A — was 7.0
    vol_ratio_5d_min: float = 2.0
    # Plan B — vol passes if EITHER:
    #   (a) vol_t == max(vol over last `vol_max_short_window` trade days)  OR
    #   (b) vol_t is among the top `vol_top_n_long` over `lookback_trade_days`
    vol_max_short_window: int = 30
    vol_top_n_long: int = 3
    lookback_trade_days: int = RULE_LOOKBACK_TRADE_DAYS
    # P0 H2 — minimum fraction of `lookback_trade_days` a stock must have
    # in its history before vol-rule evaluation. Stocks with less are
    # surfaced in `insufficient_history` rather than silently passing through.
    min_history_coverage: float = 0.8
    # P2 L3 — apply adj_factor-based forward-volume adjustment to historical
    # vol so that vol_max comparisons stay valid across splits/送转 events.
    # Falls back to raw vol when adj_factor is unavailable (with a diagnostic).
    vol_adjust: bool = True
    # v0.3.0 P0-1 — drop hits whose upper shadow exceeds this fraction of the
    # day's range (避雷针 / 长上影). None disables the filter entirely.
    upper_shadow_ratio_max: float | None = 0.35
    # v0.3.0 P0-2 — circ_mv-bucketed (turnover_min, turnover_max). Each entry is
    # (circ_mv_yi_max, turnover_min, turnover_max); the first bucket where
    # circ_mv_yi ≤ max wins. None falls back to the global turnover_min/max.
    turnover_buckets: list[tuple[float, float, float]] | None = field(
        default_factory=lambda: list(DEFAULT_TURNOVER_BUCKETS)
    )

    def __post_init__(self) -> None:
        """P1 L2 — fail loud on impossible threshold combos.

        These checks run at construction (defaults / from_dict / explicit),
        so misconfigured runs surface a ValueError immediately rather than
        silently producing 0 hits.
        """
        if not (0 <= self.pct_chg_min <= self.pct_chg_max):
            raise ValueError(
                f"invalid pct_chg range [{self.pct_chg_min}, {self.pct_chg_max}] "
                "(require 0 ≤ min ≤ max)"
            )
        if not (0 <= self.turnover_min <= self.turnover_max):
            raise ValueError(
                f"invalid turnover range [{self.turnover_min}, {self.turnover_max}] "
                "(require 0 ≤ min ≤ max)"
            )
        if not (0.0 <= self.body_ratio_min <= 1.0):
            raise ValueError(
                f"body_ratio_min must be in [0, 1], got {self.body_ratio_min}"
            )
        if self.vol_ratio_5d_min < 0:
            raise ValueError(f"vol_ratio_5d_min must be ≥ 0, got {self.vol_ratio_5d_min}")
        if self.vol_max_short_window <= 0:
            raise ValueError(
                f"vol_max_short_window must be > 0, got {self.vol_max_short_window}"
            )
        if self.vol_top_n_long <= 0:
            raise ValueError(f"vol_top_n_long must be > 0, got {self.vol_top_n_long}")
        if self.lookback_trade_days < 6:
            # 6 = 5 prev-day window + the T row itself
            raise ValueError(
                f"lookback_trade_days must be ≥ 6 (5 prev + T), got {self.lookback_trade_days}"
            )
        if self.vol_max_short_window > self.lookback_trade_days:
            raise ValueError(
                f"vol_max_short_window ({self.vol_max_short_window}) must be ≤ "
                f"lookback_trade_days ({self.lookback_trade_days})"
            )
        if not (0.0 < self.min_history_coverage <= 1.0):
            raise ValueError(
                f"min_history_coverage must be in (0, 1], got {self.min_history_coverage}"
            )
        if self.upper_shadow_ratio_max is not None and not (
            0.0 < self.upper_shadow_ratio_max <= 1.0
        ):
            raise ValueError(
                f"upper_shadow_ratio_max must be in (0, 1] or None, "
                f"got {self.upper_shadow_ratio_max}"
            )
        if self.turnover_buckets is not None:
            if not self.turnover_buckets:
                raise ValueError("turnover_buckets, if set, must be non-empty")
            prev_max = float("-inf")
            for entry in self.turnover_buckets:
                if not isinstance(entry, tuple) or len(entry) != 3:
                    raise ValueError(
                        f"each turnover_buckets entry must be a 3-tuple "
                        f"(circ_mv_yi_max, turnover_min, turnover_max); got {entry}"
                    )
                b_max, t_min, t_max = entry
                if b_max <= prev_max:
                    raise ValueError(
                        f"turnover_buckets circ_mv_yi_max must be strictly increasing; "
                        f"{prev_max} → {b_max}"
                    )
                if not (0 <= t_min <= t_max):
                    raise ValueError(
                        f"turnover_buckets entry has invalid turnover range "
                        f"[{t_min}, {t_max}] (require 0 ≤ min ≤ max)"
                    )
                prev_max = b_max

    @classmethod
    def defaults(cls) -> ScreenRules:
        return cls()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ScreenRules:
        """Build from a partial dict (configure() output); missing keys → default."""
        if not d:
            return cls.defaults()
        type_hints: dict[str, type] = {
            "pct_chg_min": float,
            "pct_chg_max": float,
            "body_ratio_min": float,
            "turnover_min": float,
            "turnover_max": float,
            "vol_ratio_5d_min": float,
            "vol_max_short_window": int,
            "vol_top_n_long": int,
            "lookback_trade_days": int,
            "min_history_coverage": float,
        }
        defaults = cls.defaults()
        kwargs: dict[str, Any] = {}
        for name, ty in type_hints.items():
            v = d.get(name)
            kwargs[name] = ty(v) if v is not None else getattr(defaults, name)
        # vol_adjust handled separately so we don't rely on bool(str) (which
        # is True for non-empty strings — an easy footgun for "false").
        if "vol_adjust" in d and d["vol_adjust"] is not None:
            v = d["vol_adjust"]
            if isinstance(v, str):
                kwargs["vol_adjust"] = v.strip().lower() in {"1", "true", "t", "yes", "y"}
            else:
                kwargs["vol_adjust"] = bool(v)
        # v0.3.0 P0-1 — `upper_shadow_ratio_max`: explicit `null` → disable filter;
        # missing key → keep default (0.35).
        if "upper_shadow_ratio_max" in d:
            v = d["upper_shadow_ratio_max"]
            kwargs["upper_shadow_ratio_max"] = float(v) if v is not None else None
        # v0.3.0 P0-2 — `turnover_buckets`: accept list-of-list (JSON has no tuple);
        # explicit `null` → fall back to global turnover_min/max; missing key →
        # keep default DEFAULT_TURNOVER_BUCKETS. The first element of any entry
        # may be `null` to mean "no upper bound" (math.inf).
        if "turnover_buckets" in d:
            raw = d["turnover_buckets"]
            if raw is None:
                kwargs["turnover_buckets"] = None
            else:
                parsed: list[tuple[float, float, float]] = []
                for entry in raw:
                    if len(entry) != 3:
                        raise ValueError(
                            f"each turnover_buckets entry must have 3 elements, got {entry}"
                        )
                    b_max_raw, t_min, t_max = entry
                    b_max = math.inf if b_max_raw is None else float(b_max_raw)
                    parsed.append((b_max, float(t_min), float(t_max)))
                kwargs["turnover_buckets"] = parsed
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict as _asdict  # noqa: PLC0415

        out = _asdict(self)
        # JSON has no `inf`; round-trip-friendly form mirrors what `from_dict`
        # accepts (`null` for an unbounded last bucket).
        if self.turnover_buckets is not None:
            out["turnover_buckets"] = [
                [None if math.isinf(b_max) else b_max, t_min, t_max]
                for (b_max, t_min, t_max) in self.turnover_buckets
            ]
        return out


@dataclass
class ScreenDiagnostics:
    """P0 — observable data-completeness counters surfaced in the report.

    Populated by `screen_anomalies` regardless of outcome so the user can
    自证 that no silent degradation happened on this run.
    """

    # Step 1
    stock_basic_rows: int = 0
    main_board_rows: int = 0
    # Step 2
    stock_st_count: int = 0
    stock_st_status: str = "ok"  # 'ok' | 'empty' (suspicious) | 'error: ...'
    suspend_d_count: int = 0
    suspend_d_status: str = "ok"
    # Step 3
    daily_t_total_rows: int = 0
    daily_t_main_board_rows: int = 0      # ts_codes intersected with main_codes
    # Step 4
    daily_basic_t_total_rows: int = 0
    daily_basic_t_main_board_rows: int = 0
    daily_basic_status: str = "ok"
    turnover_missing_codes: list[str] = field(default_factory=list)
    n_turnover_missing: int = 0
    # Step 5 (history window)
    history_window_planned_days: int = 0
    history_window_actual_days: int = 0
    history_window_missing_dates: list[str] = field(default_factory=list)
    history_min_required_days: int = 0
    insufficient_history: list[dict[str, Any]] = field(default_factory=list)
    # P2 L3 — adj_factor coverage; surfaces whether vol-adjust ran on full data,
    # degraded to raw vol for some codes, or was disabled.
    vol_adjust_enabled: bool = False
    vol_adjust_status: str = "disabled"  # 'ok' | 'disabled' | 'degraded: ...'
    adj_factor_planned_days: int = 0
    adj_factor_actual_days: int = 0
    adj_factor_missing_dates: list[str] = field(default_factory=list)
    adj_factor_missing_codes: list[str] = field(default_factory=list)
    # v0.3.0 P0-1 — upper-shadow filter; `enabled=False` when rules disable it.
    upper_shadow_filter_enabled: bool = False
    upper_shadow_filter_threshold: float | None = None
    n_after_upper_shadow: int = 0
    # v0.3.0 P0-2 — circ_mv-bucketed turnover bookkeeping.
    turnover_buckets_enabled: bool = False
    turnover_bucket_hits: dict[str, int] = field(default_factory=dict)
    n_missing_circ_mv: int = 0
    circ_mv_missing_codes: list[str] = field(default_factory=list)


@dataclass
class ScreenResult:
    """Outcome of a screen pass."""

    trade_date: str
    n_main_board: int
    n_after_st_susp: int
    n_after_t_day_rules: int  # pct_chg + body_ratio
    n_after_upper_shadow: int  # v0.3.0 P0-1
    n_after_turnover: int
    n_after_vol_rules: int  # vol_ratio_5d + dual vol rule
    rules: ScreenRules = field(default_factory=ScreenRules.defaults)
    diagnostics: ScreenDiagnostics = field(default_factory=ScreenDiagnostics)
    hits: list[dict[str, Any]] = field(default_factory=list)
    data_unavailable: list[str] = field(default_factory=list)


def screen_anomalies(
    *,
    tushare: TushareClient,
    calendar: TradeCalendar,
    trade_date: str,
    rules: ScreenRules | None = None,
    force_sync: bool = False,
) -> ScreenResult:
    """Apply the local screening rules and return matched candidates.

    Pipeline (cheapest filter first):
        1. stock_basic → main board pool
        2. stock_st(T) → drop ST; suspend_d(T) → drop suspended
        3. daily(T) → keep阳线 + pct_chg in [pct_chg_min, pct_chg_max]
                    + body_ratio ≥ body_ratio_min
        4. daily_basic(T) → keep turnover_rate in [turnover_min, turnover_max]
        5. daily(N-trade-day window) → keep
              (vol_t == max(vol_max_short_window) OR
               vol_t in top vol_top_n_long over lookback_trade_days)
              AND vol_t ≥ vol_ratio_5d_min × mean(prev 5d)
    """
    rules = rules or ScreenRules.defaults()
    data_unavailable: list[str] = []
    diag = ScreenDiagnostics()
    # v0.3.0 P0-1 / P0-2 — surface whether each new filter is engaged this run.
    diag.upper_shadow_filter_enabled = rules.upper_shadow_ratio_max is not None
    diag.upper_shadow_filter_threshold = rules.upper_shadow_ratio_max
    diag.turnover_buckets_enabled = rules.turnover_buckets is not None

    # 1. main board pool
    stock_basic = tushare.call("stock_basic", force_sync=force_sync)
    diag.stock_basic_rows = int(len(stock_basic)) if stock_basic is not None else 0
    main_pool = main_board_filter(stock_basic)
    main_codes = set(main_pool["ts_code"].astype(str))
    n_main = len(main_codes)
    diag.main_board_rows = n_main

    # 2a. ST exclusion (REQUIRED — propagate auth failure)
    st_df = tushare.call("stock_st", trade_date=trade_date, force_sync=force_sync)
    st_codes = set(st_df["ts_code"].astype(str)) if not st_df.empty else set()
    diag.stock_st_count = len(st_codes)
    if not st_codes:
        # P0 M2 — A股每日 ST 数稳定在 100+；返空一定是数据异常，应警示
        diag.stock_st_status = "empty (suspicious — verify data freshness)"
        data_unavailable.append(
            "stock_st(T) returned 0 ST codes — abnormal for A股, "
            "ST stocks may have leaked into candidates; verify data freshness"
        )

    # 2b. suspended exclusion (OPTIONAL)
    susp_df, susp_err = _try_optional(
        tushare, "suspend_d", trade_date=trade_date, force_sync=force_sync
    )
    if susp_err:
        data_unavailable.append(f"suspend_d ({susp_err})")
        diag.suspend_d_status = susp_err
    susp_codes = set(susp_df["ts_code"].astype(str)) if susp_df is not None and not susp_df.empty else set()
    diag.suspend_d_count = len(susp_codes)

    eligible = main_codes - st_codes - susp_codes
    n_after_st = len(eligible)

    # 3. T-day daily — single API call returns all stocks for that date
    daily_t_full = tushare.call("daily", trade_date=trade_date, force_sync=force_sync)
    daily_t_full = _normalize_id_cols(daily_t_full)
    if daily_t_full is None or daily_t_full.empty:
        data_unavailable.append("daily(T) returned empty")
        return ScreenResult(
            trade_date=trade_date,
            n_main_board=n_main,
            n_after_st_susp=n_after_st,
            n_after_t_day_rules=0,
            n_after_upper_shadow=0,
            n_after_turnover=0,
            n_after_vol_rules=0,
            rules=rules,
            diagnostics=diag,
            data_unavailable=data_unavailable,
        )
    diag.daily_t_total_rows = int(len(daily_t_full))
    diag.daily_t_main_board_rows = int(
        daily_t_full["ts_code"].astype(str).isin(main_codes).sum()
    )
    daily_t = daily_t_full[daily_t_full["ts_code"].astype(str).isin(eligible)].copy()

    # T-day阳线 + 实体占比 + 涨幅区间
    daily_t["body"] = daily_t["close"] - daily_t["open"]
    daily_t["range"] = (daily_t["high"] - daily_t["low"]).clip(lower=1e-9)
    daily_t["body_ratio"] = daily_t["body"] / daily_t["range"]
    # v0.3.0 P0-1 — upper shadow as a fraction of the day's range.
    # = (high − max(open, close)) / range; pure upper wick → 1.0.
    daily_t["upper_shadow_ratio"] = (
        daily_t["high"] - daily_t[["open", "close"]].max(axis=1)
    ) / daily_t["range"]
    t_day_hits = daily_t[
        (daily_t["close"] > daily_t["open"])
        & (daily_t["body_ratio"] >= rules.body_ratio_min)
        & (daily_t["pct_chg"] >= rules.pct_chg_min)
        & (daily_t["pct_chg"] <= rules.pct_chg_max)
    ].copy()
    n_after_t_rules = len(t_day_hits)
    if t_day_hits.empty:
        return ScreenResult(
            trade_date=trade_date,
            n_main_board=n_main,
            n_after_st_susp=n_after_st,
            n_after_t_day_rules=0,
            n_after_upper_shadow=0,
            n_after_turnover=0,
            n_after_vol_rules=0,
            rules=rules,
            diagnostics=diag,
            data_unavailable=data_unavailable,
        )

    # v0.3.0 P0-1 — upper-shadow filter (skipped when threshold is None).
    if rules.upper_shadow_ratio_max is not None:
        t_day_hits = t_day_hits[
            t_day_hits["upper_shadow_ratio"] <= rules.upper_shadow_ratio_max
        ].copy()
    n_after_upper_shadow = len(t_day_hits)
    diag.n_after_upper_shadow = n_after_upper_shadow
    if t_day_hits.empty:
        return ScreenResult(
            trade_date=trade_date,
            n_main_board=n_main,
            n_after_st_susp=n_after_st,
            n_after_t_day_rules=n_after_t_rules,
            n_after_upper_shadow=0,
            n_after_turnover=0,
            n_after_vol_rules=0,
            rules=rules,
            diagnostics=diag,
            data_unavailable=data_unavailable,
        )

    # 4. daily_basic — turnover_rate (+ circ_mv for v0.3.0 bucketing) filter
    db_t = tushare.call("daily_basic", trade_date=trade_date, force_sync=force_sync)
    db_t = _normalize_id_cols(db_t)
    db_lookup: dict[str, dict[str, Any]] = {}
    if db_t is not None and not db_t.empty and "turnover_rate" in db_t.columns:
        cols = ["turnover_rate"]
        if "circ_mv" in db_t.columns:
            cols.append("circ_mv")
        db_lookup = db_t.set_index("ts_code")[cols].to_dict("index")
        diag.daily_basic_t_total_rows = int(len(db_t))
        diag.daily_basic_t_main_board_rows = int(
            db_t["ts_code"].astype(str).isin(main_codes).sum()
        )
    else:
        diag.daily_basic_status = "empty"
        data_unavailable.append("daily_basic.turnover_rate (frame empty)")
    t_day_hits["turnover_rate"] = t_day_hits["ts_code"].map(
        lambda c: db_lookup.get(c, {}).get("turnover_rate")
    )
    # v0.3.0 P0-2 — circ_mv lookup (亿元 via normalize_to_yi).
    t_day_hits["circ_mv_yi"] = t_day_hits["ts_code"].map(
        lambda c: normalize_to_yi("circ_mv", db_lookup.get(c, {}).get("circ_mv"))
    )

    # P0 M1 — surface candidates whose turnover_rate lookup returned NaN.
    # They will be silently dropped by the comparison below; we make that visible.
    missing_mask = t_day_hits["turnover_rate"].isna()
    n_missing_turnover = int(missing_mask.sum())
    diag.n_turnover_missing = n_missing_turnover
    if n_missing_turnover > 0:
        miss_codes = t_day_hits.loc[missing_mask, "ts_code"].astype(str).tolist()
        diag.turnover_missing_codes = miss_codes
        sample = miss_codes[:5]
        ellipsis = "..." if n_missing_turnover > 5 else ""
        data_unavailable.append(
            f"daily_basic.turnover_rate missing for {n_missing_turnover} candidates "
            f"(silently dropped at turnover step): {sample}{ellipsis}"
        )

    # v0.3.0 P0-2 — bucket lookup. circ_mv missing → fall back to global thresholds.
    buckets = rules.turnover_buckets
    bucket_label_per_row: dict[Any, str | None] = {}
    bucket_hit_counter: dict[str, int] = {}
    circ_mv_missing_codes: list[str] = []

    def _row_passes_turnover(row: Any) -> bool:
        tr = row.turnover_rate
        if pd.isna(tr):
            return False
        circ = row.circ_mv_yi
        if buckets is None or circ is None or pd.isna(circ):
            t_min, t_max = rules.turnover_min, rules.turnover_max
            label = None
            if buckets is not None and (circ is None or pd.isna(circ)):
                circ_mv_missing_codes.append(str(row.ts_code))
        else:
            _, label, t_min, t_max = _resolve_turnover_bucket(float(circ), buckets)
        bucket_label_per_row[row.Index] = label
        return t_min <= tr <= t_max

    # We need pandas Index access — use `itertuples(index=True)` and rebuild filter mask.
    keep_mask = []
    for row in t_day_hits.itertuples(index=True):
        keep_mask.append(_row_passes_turnover(row))
    turnover_hits = t_day_hits.loc[keep_mask].copy()
    turnover_hits["turnover_bucket"] = turnover_hits.index.map(
        lambda i: bucket_label_per_row.get(i)
    )
    # Tally bucket distribution among rows that PASSED the filter.
    for label in turnover_hits["turnover_bucket"].tolist():
        if label is None:
            bucket_hit_counter["fallback (no circ_mv)"] = (
                bucket_hit_counter.get("fallback (no circ_mv)", 0) + 1
            )
        else:
            bucket_hit_counter[label] = bucket_hit_counter.get(label, 0) + 1
    diag.turnover_bucket_hits = bucket_hit_counter
    diag.n_missing_circ_mv = len(circ_mv_missing_codes)
    diag.circ_mv_missing_codes = circ_mv_missing_codes
    if circ_mv_missing_codes:
        sample = circ_mv_missing_codes[:5]
        ellipsis = "..." if len(circ_mv_missing_codes) > 5 else ""
        data_unavailable.append(
            f"daily_basic.circ_mv missing for {len(circ_mv_missing_codes)} candidates "
            f"(fell back to global turnover thresholds): {sample}{ellipsis}"
        )

    n_after_turnover = len(turnover_hits)
    if turnover_hits.empty:
        return ScreenResult(
            trade_date=trade_date,
            n_main_board=n_main,
            n_after_st_susp=n_after_st,
            n_after_t_day_rules=n_after_t_rules,
            n_after_upper_shadow=n_after_upper_shadow,
            n_after_turnover=0,
            n_after_vol_rules=0,
            rules=rules,
            diagnostics=diag,
            data_unavailable=data_unavailable,
        )

    # 5. N-trade-day vol history for surviving codes (Plan B dual rule)
    survivor_codes = set(turnover_hits["ts_code"].astype(str))
    history_dates = _last_n_trade_dates(calendar, trade_date, rules.lookback_trade_days)
    diag.history_window_planned_days = len(history_dates)

    # P0 H1 — capture which planned dates returned empty (silent skip → visible).
    history_df, missing_history_dates = _fetch_daily_history_by_date(
        tushare, history_dates, survivor_codes, force_sync=force_sync
    )
    diag.history_window_actual_days = len(history_dates) - len(missing_history_dates)
    diag.history_window_missing_dates = missing_history_dates
    if missing_history_dates:
        sample = missing_history_dates[:5]
        ellipsis = "..." if len(missing_history_dates) > 5 else ""
        data_unavailable.append(
            f"daily history missing on {len(missing_history_dates)}/"
            f"{len(history_dates)} planned days "
            f"(vol_max comparison weakened): {sample}{ellipsis}"
        )

    # P2 L3 — fetch adj_factor over the same window so vol_max comparisons
    # stay valid across splits/送转. Falls back to raw vol when unavailable.
    adj_factor_lookup, adj_factor_T_lookup = _build_adj_factor_lookups(
        tushare,
        history_dates,
        survivor_codes,
        trade_date=trade_date,
        rules=rules,
        diag=diag,
        data_unavailable=data_unavailable,
        force_sync=force_sync,
    )

    # P1 L1 — pre-compute the strict 5 trade-dates immediately preceding T.
    # `prior.tail(5)` was permissive: it would happily take any 5 surviving rows,
    # so a stock with gaps could end up averaging vol over a span > 5 trade days.
    expected_prev5_dates = (
        history_dates[-6:-1] if len(history_dates) >= 6 else history_dates[:-1]
    )
    expected_prev5_set = set(expected_prev5_dates)

    # P0 H2 — enforce minimum history coverage; record stocks that fail
    required_days = max(6, int(rules.lookback_trade_days * rules.min_history_coverage))
    diag.history_min_required_days = required_days
    insufficient_history: list[dict[str, Any]] = []

    final_hits: list[dict[str, Any]] = []
    industry_lookup = main_pool.set_index("ts_code")[["name", "industry"]].to_dict(orient="index")
    for row in turnover_hits.itertuples(index=False):
        code = str(row.ts_code)
        h = history_df[history_df["ts_code"].astype(str) == code].sort_values("trade_date")
        if len(h) < required_days:
            insufficient_history.append(
                {
                    "ts_code": code,
                    "name": industry_lookup.get(code, {}).get("name"),
                    "available_days": int(len(h)),
                    "required_days": required_days,
                    "lookback_window": rules.lookback_trade_days,
                }
            )
            continue
        # Identify T row + prev 5 days (excluding T)
        t_row = h[h["trade_date"].astype(str) == trade_date]
        prior = h[h["trade_date"].astype(str) < trade_date]
        if t_row.empty or len(prior) < 5:
            insufficient_history.append(
                {
                    "ts_code": code,
                    "name": industry_lookup.get(code, {}).get("name"),
                    "available_days": int(len(h)),
                    "required_days": required_days,
                    "lookback_window": rules.lookback_trade_days,
                    "reason": "missing T-row or <5 prior days",
                }
            )
            continue

        # P1 L1 — strict prev-5 trade-day filter: require all 5 calendar
        # positions (history_dates[-6:-1]) to be present, else surface as
        # insufficient_history rather than averaging over a sparse span.
        prior_5d_strict = prior[prior["trade_date"].astype(str).isin(expected_prev5_set)]
        if len(prior_5d_strict) < 5:
            insufficient_history.append(
                {
                    "ts_code": code,
                    "name": industry_lookup.get(code, {}).get("name"),
                    "available_days": int(len(h)),
                    "required_days": required_days,
                    "lookback_window": rules.lookback_trade_days,
                    "reason": (
                        f"missing prev-5d trade dates "
                        f"(have {len(prior_5d_strict)}/5 of {sorted(expected_prev5_set)})"
                    ),
                }
            )
            continue

        # P2 L3 — adj_factor-aware vol values. When vol_adjust is enabled and
        # f_T is available, compute forward-adjusted vol so that a 1:N split
        # between d and T inflates pre-split vol by N (= adj_T / adj_d) and
        # historical vol stays comparable to T-day vol. Falls back to raw vol
        # silently per-row when adj_factor is missing for that row.
        f_T = adj_factor_T_lookup.get(code)
        if rules.vol_adjust and f_T is not None and f_T > 0:
            def _adj(d: str, raw: float) -> float:
                f_d = adj_factor_lookup.get((code, d))
                if f_d is None or f_d <= 0:
                    return raw
                return raw * (f_T / f_d)
            vol_t = float(t_row.iloc[0]["vol"])  # at T, f_d == f_T → no change
            vols_long = [
                _adj(str(td), float(v))
                for td, v in zip(
                    h["trade_date"].astype(str).tolist(),
                    h["vol"].astype(float).tolist(),
                    strict=False,
                )
            ]
            short_h = h.tail(rules.vol_max_short_window)
            vols_short = [
                _adj(str(td), float(v))
                for td, v in zip(
                    short_h["trade_date"].astype(str).tolist(),
                    short_h["vol"].astype(float).tolist(),
                    strict=False,
                )
            ]
            vol_mean_prev5 = float(
                pd.Series(
                    [
                        _adj(str(td), float(v))
                        for td, v in zip(
                            prior_5d_strict["trade_date"].astype(str).tolist(),
                            prior_5d_strict["vol"].astype(float).tolist(),
                            strict=False,
                        )
                    ]
                ).mean()
            )
        else:
            vol_t = float(t_row.iloc[0]["vol"])
            vols_long = h["vol"].astype(float).tolist()
            vols_short = [
                float(v) for v in h.tail(rules.vol_max_short_window)["vol"].tolist()
            ]
            vol_mean_prev5 = float(prior_5d_strict["vol"].astype(float).mean())
        vol_max_long = max(vols_long)
        vol_max_short = max(vols_short)

        # Plan B — vol passes if either condition holds
        short_window_max_pass = vol_t >= vol_max_short - 1e-9
        days_with_higher_vol = sum(1 for v in vols_long if v > vol_t + 1e-9)
        long_window_top_n_pass = days_with_higher_vol < rules.vol_top_n_long
        if not (short_window_max_pass or long_window_top_n_pass):
            continue

        # vol_ratio_5d ≥ rules.vol_ratio_5d_min
        if vol_mean_prev5 <= 0:
            continue
        vol_ratio_5d = vol_t / vol_mean_prev5
        if vol_ratio_5d < rules.vol_ratio_5d_min:
            continue

        meta = industry_lookup.get(code, {})
        final_hits.append(
            {
                "ts_code": code,
                "name": meta.get("name"),
                "industry": meta.get("industry"),
                "trade_date": trade_date,
                "pct_chg": round2(row.pct_chg),
                "open": round2(row.open),
                "high": round2(row.high),
                "low": round2(row.low),
                "close": round2(row.close),
                "vol": round2(row.vol),
                "amount": round2(row.amount),
                "body_ratio": round2(row.body_ratio),
                "upper_shadow_ratio": round2(getattr(row, "upper_shadow_ratio", None)),
                "turnover_rate": round2(row.turnover_rate),
                "circ_mv_yi": round2(getattr(row, "circ_mv_yi", None)),
                "turnover_bucket": getattr(row, "turnover_bucket", None),
                "vol_ratio_5d": round2(vol_ratio_5d),
                "vol_rank_in_long_window": days_with_higher_vol + 1,
                "max_vol_short_window": round2(vol_max_short),
                "max_vol_long_window": round2(vol_max_long),
                "history_days_used": int(len(h)),
                # Legacy-named column populated by upsert_watchlist /
                # append_anomaly_history. Holds the long-window max regardless
                # of the actual lookback_trade_days setting.
                "max_vol_60d": round2(vol_max_long),
            }
        )

    diag.insufficient_history = insufficient_history
    if insufficient_history:
        sample = [r["ts_code"] for r in insufficient_history[:5]]
        ellipsis = "..." if len(insufficient_history) > 5 else ""
        data_unavailable.append(
            f"insufficient history (<{required_days} of {rules.lookback_trade_days} days) "
            f"for {len(insufficient_history)} candidates (excluded from vol rule): "
            f"{sample}{ellipsis}"
        )

    return ScreenResult(
        trade_date=trade_date,
        n_main_board=n_main,
        n_after_st_susp=n_after_st,
        n_after_t_day_rules=n_after_t_rules,
        n_after_upper_shadow=n_after_upper_shadow,
        n_after_turnover=n_after_turnover,
        n_after_vol_rules=len(final_hits),
        rules=rules,
        diagnostics=diag,
        hits=final_hits,
        data_unavailable=data_unavailable,
    )


def _last_n_trade_dates(calendar: TradeCalendar, end_date: str, n: int) -> list[str]:
    """Return the last `n` open trade dates ending at (and including) end_date."""
    dates: list[str] = []
    cursor = end_date
    if calendar.is_open(cursor):
        dates.append(cursor)
    while len(dates) < n:
        cursor = calendar.pretrade_date(cursor)
        dates.append(cursor)
    dates.sort()
    return dates


def _fetch_daily_history_by_date(
    tushare: TushareClient,
    trade_dates: list[str],
    candidate_codes: set[str],
    *,
    force_sync: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch daily(trade_date=X) for each X in trade_dates and concat.

    Per-day calls are O(N) but each call is cached as ``trade_day_immutable`` in
    TushareClient, so subsequent runs hit the cache. Filtering by candidate_codes
    happens client-side.

    Returns:
        (concat_df, missing_dates) — `missing_dates` lists every planned
        trade_date for which the daily call returned None or an empty frame.
        Caller (P0 H1) MUST surface these so the user knows the vol_max
        comparison was computed on incomplete data.
    """
    frames: list[pd.DataFrame] = []
    missing_dates: list[str] = []
    for d in trade_dates:
        df = tushare.call("daily", trade_date=d, force_sync=force_sync)
        if df is None or df.empty:
            missing_dates.append(d)
            continue
        df = _normalize_id_cols(df)
        if df is None or df.empty:
            missing_dates.append(d)
            continue
        if candidate_codes:
            df = df[df["ts_code"].isin(candidate_codes)]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, missing_dates


def _build_adj_factor_lookups(
    tushare: TushareClient,
    history_dates: list[str],
    survivor_codes: set[str],
    *,
    trade_date: str,
    rules: ScreenRules,
    diag: ScreenDiagnostics,
    data_unavailable: list[str],
    force_sync: bool = False,
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Fetch adj_factor for the screening window and build (code, date)→f and code→f_T lookups.

    The two returned dicts let the per-stock loop compute forward-adjusted vol
    in O(1) per row without re-filtering the frame each iteration.

    Diagnostics fields populated (P2 L3):
        diag.vol_adjust_enabled       — whether the rule was on at all
        diag.vol_adjust_status         — 'ok' | 'disabled' | 'degraded: ...'
        diag.adj_factor_planned_days   — len(history_dates) when enabled
        diag.adj_factor_actual_days    — successful per-day fetches
        diag.adj_factor_missing_dates  — list of date strings that returned empty
        diag.adj_factor_missing_codes  — codes whose T-day adj_factor was missing
                                         (forces fallback to raw vol for that code)
    """
    if not rules.vol_adjust:
        diag.vol_adjust_enabled = False
        diag.vol_adjust_status = "disabled"
        return {}, {}

    diag.vol_adjust_enabled = True
    diag.adj_factor_planned_days = len(history_dates)
    adj_df, missing_adj_dates = _fetch_adj_factor_history_by_date(
        tushare, history_dates, survivor_codes, force_sync=force_sync
    )
    diag.adj_factor_actual_days = len(history_dates) - len(missing_adj_dates)
    diag.adj_factor_missing_dates = missing_adj_dates

    if adj_df.empty or "adj_factor" not in adj_df.columns:
        diag.vol_adjust_status = "degraded: adj_factor unavailable (raw vol used)"
        data_unavailable.append(
            "adj_factor unavailable for the entire window — vol-adjust disabled, "
            "raw vol used (splits/送转 in lookback may understate historical vol)"
        )
        return {}, {}

    # (code, date) → adj_factor and code → adj_factor at T
    pair_lookup: dict[tuple[str, str], float] = {}
    for r in adj_df.itertuples(index=False):
        try:
            f = float(r.adj_factor)
        except (TypeError, ValueError):
            continue
        if pd.isna(f):
            continue
        pair_lookup[(str(r.ts_code), str(r.trade_date))] = f

    f_T_lookup: dict[str, float] = {
        code: f
        for (code, d), f in pair_lookup.items()
        if d == str(trade_date)
    }

    missing_t_codes = sorted(survivor_codes - set(f_T_lookup.keys()))
    diag.adj_factor_missing_codes = missing_t_codes

    if missing_adj_dates and not missing_t_codes:
        diag.vol_adjust_status = (
            f"degraded: {len(missing_adj_dates)} historical day(s) missing adj_factor"
        )
        sample = missing_adj_dates[:5]
        ellipsis = "..." if len(missing_adj_dates) > 5 else ""
        data_unavailable.append(
            f"adj_factor missing on {len(missing_adj_dates)}/{len(history_dates)} "
            f"days (raw vol used for those rows): {sample}{ellipsis}"
        )
    elif missing_t_codes:
        sample = missing_t_codes[:5]
        ellipsis = "..." if len(missing_t_codes) > 5 else ""
        diag.vol_adjust_status = (
            f"degraded: T-day adj_factor missing for {len(missing_t_codes)} code(s)"
        )
        data_unavailable.append(
            f"adj_factor(T) missing for {len(missing_t_codes)} candidate(s) — "
            f"those codes use raw vol: {sample}{ellipsis}"
        )
    else:
        diag.vol_adjust_status = "ok"

    return pair_lookup, f_T_lookup


def _fetch_adj_factor_history_by_date(
    tushare: TushareClient,
    trade_dates: list[str],
    candidate_codes: set[str],
    *,
    force_sync: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch adj_factor(trade_date=X) per X — same per-day-batch pattern as daily.

    adj_factor is end-of-day immutable, so cache hits dominate after the first
    pass. Missing days are returned to the caller (P2 L3) so the diagnostic
    can record whether vol-adjust ran on complete data.

    Permission may be missing on free Tushare tiers — callers must handle
    TushareUnauthorizedError or wrap with `_try_optional`.
    """
    frames: list[pd.DataFrame] = []
    missing_dates: list[str] = []
    for d in trade_dates:
        df, _err = _try_optional(tushare, "adj_factor", trade_date=d, force_sync=force_sync)
        if df is None or df.empty:
            missing_dates.append(d)
            continue
        df = _normalize_id_cols(df)
        if df is None or df.empty:
            missing_dates.append(d)
            continue
        if candidate_codes:
            df = df[df["ts_code"].isin(candidate_codes)]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, missing_dates


# ---------------------------------------------------------------------------
# Watchlist persistence
# ---------------------------------------------------------------------------


def upsert_watchlist(db: Any, hits: list[dict[str, Any]], trade_date: str) -> tuple[int, int]:
    """Insert new hits / update existing rows. Returns (n_new, n_updated).

    Original tracked_since is PRESERVED on duplicate hits — that's the whole
    point of the追踪日数 metric (a stock that re-triggers shouldn't reset its
    tracking start).
    """
    if not hits:
        return 0, 0
    existing = {
        row[0]: row[1]
        for row in db.fetchall("SELECT ts_code, tracked_since FROM va_watchlist")
    }
    n_new = 0
    n_updated = 0
    for h in hits:
        code = h["ts_code"]
        if code in existing:
            db.execute(
                "UPDATE va_watchlist SET name=?, industry=?, last_screened=?, "
                "last_pct_chg=?, last_close=?, last_vol=?, last_amount=?, "
                "last_body_ratio=?, last_turnover_rate=?, last_vol_ratio_5d=?, "
                "last_max_vol_60d=? WHERE ts_code=?",
                (
                    h.get("name"),
                    h.get("industry"),
                    trade_date,
                    h.get("pct_chg"),
                    h.get("close"),
                    h.get("vol"),
                    h.get("amount"),
                    h.get("body_ratio"),
                    h.get("turnover_rate"),
                    h.get("vol_ratio_5d"),
                    h.get("max_vol_60d"),
                    code,
                ),
            )
            n_updated += 1
        else:
            db.execute(
                "INSERT INTO va_watchlist(ts_code, name, industry, tracked_since, "
                "last_screened, last_pct_chg, last_close, last_vol, last_amount, "
                "last_body_ratio, last_turnover_rate, last_vol_ratio_5d, last_max_vol_60d) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    code,
                    h.get("name"),
                    h.get("industry"),
                    trade_date,
                    trade_date,
                    h.get("pct_chg"),
                    h.get("close"),
                    h.get("vol"),
                    h.get("amount"),
                    h.get("body_ratio"),
                    h.get("turnover_rate"),
                    h.get("vol_ratio_5d"),
                    h.get("max_vol_60d"),
                ),
            )
            n_new += 1
    return n_new, n_updated


def append_anomaly_history(db: Any, hits: list[dict[str, Any]]) -> None:
    """Append every hit row to va_anomaly_history (audit log).

    Uses INSERT OR REPLACE semantics via DELETE-then-INSERT on (trade_date, ts_code)
    since DuckDB lacks ON CONFLICT for composite PKs in older versions.
    """
    if not hits:
        return
    for h in hits:
        db.execute(
            "DELETE FROM va_anomaly_history WHERE trade_date=? AND ts_code=?",
            (h["trade_date"], h["ts_code"]),
        )
        db.execute(
            "INSERT INTO va_anomaly_history(trade_date, ts_code, name, industry, "
            "pct_chg, close, open, high, low, vol, amount, body_ratio, turnover_rate, "
            "vol_ratio_5d, max_vol_60d, raw_metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                h["trade_date"],
                h["ts_code"],
                h.get("name"),
                h.get("industry"),
                h.get("pct_chg"),
                h.get("close"),
                h.get("open"),
                h.get("high"),
                h.get("low"),
                h.get("vol"),
                h.get("amount"),
                h.get("body_ratio"),
                h.get("turnover_rate"),
                h.get("vol_ratio_5d"),
                h.get("max_vol_60d"),
                json.dumps(h, ensure_ascii=False),
            ),
        )


def fetch_watchlist(db: Any) -> list[dict[str, Any]]:
    """Read all watchlist rows as dicts."""
    rows = db.fetchall(
        "SELECT ts_code, name, industry, tracked_since, last_screened, last_pct_chg, "
        "last_close, last_vol, last_amount, last_body_ratio, last_turnover_rate, "
        "last_vol_ratio_5d, last_max_vol_60d FROM va_watchlist ORDER BY tracked_since"
    )
    cols = [
        "ts_code",
        "name",
        "industry",
        "tracked_since",
        "last_screened",
        "last_pct_chg",
        "last_close",
        "last_vol",
        "last_amount",
        "last_body_ratio",
        "last_turnover_rate",
        "last_vol_ratio_5d",
        "last_max_vol_60d",
    ]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def prune_watchlist(db: Any, *, min_tracked_calendar_days: int, today: str) -> list[dict[str, Any]]:
    """Remove every watchlist row whose calendar-day age ≥ N. Return removed rows."""
    rows = fetch_watchlist(db)
    pruned: list[dict[str, Any]] = []
    for r in rows:
        age = _calendar_days_between(r["tracked_since"], today)
        if age >= min_tracked_calendar_days:
            r["tracked_days"] = age
            pruned.append(r)
    if pruned:
        codes = [r["ts_code"] for r in pruned]
        # DuckDB executemany via parameterized loop; safer than IN-clause stitching
        for code in codes:
            db.execute("DELETE FROM va_watchlist WHERE ts_code=?", (code,))
    return pruned


# ---------------------------------------------------------------------------
# ANALYZE MODE — assemble per-stock context for LLM
# ---------------------------------------------------------------------------


# v0.3.0 P0-3 — VCP feature windows (kept module-level so callers / tests can
# read them without instantiating AnalyzeBundle).
ATR_WINDOW = 10                   # 10-day ATR window (simple-average TR)
ATR_QUANTILE_LOOKBACK = 60        # rank current ATR within the trailing 60-day series
BBW_WINDOW = 20                   # 20-day Bollinger band width
BBW_COMPRESSION_LOOKBACK = 60     # current BBW vs trailing 60-day mean BBW
# v0.3.0 P0-4 — resistance windows (E3-A: only `low_120d`, no `low_250d`).
RESIST_120D = 120
RESIST_250D = 250
# Default extended history window for analyze mode (E2-A: single 250d fetch
# is sliced internally for verbatim / VCP / resistance consumers).
DEFAULT_EXTENDED_LOOKBACK_TRADE_DAYS = RESIST_250D


def _compute_atr_series(history: list[dict[str, Any]]) -> list[float | None]:
    """Per-row trailing-10-day simple-average True Range.

    TR_t = max(high_t − low_t, |high_t − close_{t-1}|, |low_t − close_{t-1}|)
    ATR_10_t = mean(TR over the trailing 10 days, t inclusive)

    Returns a list aligned to ``history``. Entries are ``None`` until enough
    rows are available or whenever any input value is missing in the window.
    """
    n = len(history)
    if n < 2:
        return [None] * n
    trs: list[float | None] = [None]
    for i in range(1, n):
        h = history[i].get("high")
        low = history[i].get("low")
        c_prev = history[i - 1].get("close")
        if h is None or low is None or c_prev is None:
            trs.append(None)
            continue
        try:
            h_f, low_f, cp_f = float(h), float(low), float(c_prev)
        except (TypeError, ValueError):
            trs.append(None)
            continue
        if any(pd.isna(v) for v in (h_f, low_f, cp_f)):
            trs.append(None)
            continue
        trs.append(max(h_f - low_f, abs(h_f - cp_f), abs(low_f - cp_f)))

    out: list[float | None] = []
    for i in range(n):
        start = i - ATR_WINDOW + 1
        if start < 0:
            out.append(None)
            continue
        slice_ = trs[start : i + 1]
        if any(t is None for t in slice_):
            out.append(None)
            continue
        out.append(sum(slice_) / ATR_WINDOW)  # type: ignore[arg-type]
    return out


def _compute_bbw_series(history: list[dict[str, Any]]) -> list[float | None]:
    """Per-row 20-day Bollinger Band Width as a percentage of the 20-day MA.

    BBW = 4 × stdev(close_20) / mean(close_20) × 100

    The factor 4 = upper(MA + 2σ) − lower(MA − 2σ) → 4σ. Returns ``None`` until
    20 rows are available, or whenever any close in the window is missing /
    the rolling mean is non-positive.
    """
    n = len(history)
    closes: list[float | None] = []
    for r in history:
        c = r.get("close")
        if c is None:
            closes.append(None)
            continue
        try:
            f = float(c)
        except (TypeError, ValueError):
            closes.append(None)
            continue
        closes.append(None if pd.isna(f) else f)

    out: list[float | None] = []
    for i in range(n):
        start = i - BBW_WINDOW + 1
        if start < 0:
            out.append(None)
            continue
        slice_ = closes[start : i + 1]
        if any(c is None for c in slice_):
            out.append(None)
            continue
        floats: list[float] = [c for c in slice_ if c is not None]
        mean = sum(floats) / BBW_WINDOW
        if mean <= 0:
            out.append(None)
            continue
        # Population std — same family as Bollinger's original (close enough
        # for our discrimination purposes; the choice is uniform across the
        # series so trend comparisons are unbiased).
        var = sum((c - mean) ** 2 for c in floats) / BBW_WINDOW
        std = math.sqrt(var)
        out.append(4 * std / mean * 100)
    return out


def _quantile_in_window(
    series: list[float | None], idx: int, lookback: int
) -> float | None:
    """Return the [0, 1] quantile of ``series[idx]`` within the trailing
    ``lookback`` non-None values (idx inclusive). 0 = historical min,
    1 = historical max. ``None`` when fewer than ``lookback`` non-None values
    in the window or the current value itself is None."""
    if idx < 0 or idx >= len(series):
        return None
    cur = series[idx]
    if cur is None:
        return None
    start = max(0, idx - lookback + 1)
    window = [v for v in series[start : idx + 1] if v is not None]
    if len(window) < lookback:
        return None
    less_or_eq = sum(1 for v in window if v <= cur)
    return (less_or_eq - 1) / (len(window) - 1) if len(window) > 1 else 0.0


@dataclass
class AnalyzeBundle:
    """Everything the走势分析 LLM stage needs."""

    trade_date: str
    next_trade_date: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    market_summary: dict[str, Any] = field(default_factory=dict)
    sector_strength_source: str = "industry_fallback"
    sector_strength_data: dict[str, Any] = field(default_factory=dict)
    data_unavailable: list[str] = field(default_factory=list)


# v0.5.0 P1-1 — RPS / 大盘相对 alpha 配置
DEFAULT_BASELINE_INDEX_CODE = "000300.SH"
ALPHA_LEADING_THRESHOLD = 5.0   # alpha_20d_pct > +5 → leading
ALPHA_LAGGING_THRESHOLD = -5.0  # alpha_20d_pct < -5 → lagging


def collect_analyze_bundle(
    *,
    tushare: TushareClient,
    db: Any,
    calendar: TradeCalendar,
    trade_date: str,
    next_trade_date: str,
    history_lookback: int = DEFAULT_EXTENDED_LOOKBACK_TRADE_DAYS,
    moneyflow_lookback: int = 5,
    baseline_index_code: str = DEFAULT_BASELINE_INDEX_CODE,
    force_sync: bool = False,
) -> AnalyzeBundle:
    """Read watchlist + pull historical windows + assemble compact LLM context.

    Per the design spec:
      * 60-trade-day window for OHLCV → compressed into MA/aggregate features
      * 5-day moneyflow → compressed into trend + cumulative net flow
      * 60-day limit_list_d → flag历史涨停 (optional)
      * sector_strength: limit_cpt_list (tier 1) / industry aggregation fallback
      * tracked_days: calendar days since first added to watchlist
    """
    bundle = AnalyzeBundle(trade_date=trade_date, next_trade_date=next_trade_date)
    data_unavailable: list[str] = []

    watchlist = fetch_watchlist(db)
    if not watchlist:
        return bundle

    candidate_codes = {w["ts_code"] for w in watchlist}

    # -------- historical OHLCV (extended trade-day window, batch by date) ---
    history_dates = _last_n_trade_dates(calendar, trade_date, history_lookback)
    daily_df, missing_history_dates = _fetch_daily_history_by_date(
        tushare, history_dates, candidate_codes, force_sync=force_sync
    )
    if daily_df.empty:
        data_unavailable.append(
            f"daily({history_lookback}d-window) returned empty"
        )
    elif missing_history_dates:
        sample = missing_history_dates[:5]
        ellipsis = "..." if len(missing_history_dates) > 5 else ""
        data_unavailable.append(
            f"daily history missing on {len(missing_history_dates)}/"
            f"{len(history_dates)} planned days: {sample}{ellipsis}"
        )

    # -------- baseline index daily (v0.5.0 P1-1 — alpha computation) --------
    # F1: 沪深 300; G1: 250d window matched to per-stock daily history.
    # G8: failures emit a WARN-level mention into data_unavailable; the runner
    # surfaces it as an EventLevel.WARN LOG instead of silently degrading.
    baseline_close_by_date: dict[str, float] = {}
    if history_dates:
        idx_df, idx_err = _try_optional(
            tushare,
            "index_daily",
            params={
                "ts_code": baseline_index_code,
                "start_date": history_dates[0],
                "end_date": history_dates[-1],
            },
            force_sync=force_sync,
        )
        if idx_err:
            data_unavailable.append(
                f"index_daily ({idx_err}) — alpha 字段降级为 None；"
                f"如需启用 alpha，请确认 Tushare 账户已开通 index_daily 权限"
            )
        else:
            idx_df = _normalize_id_cols(idx_df)
            if idx_df is not None and not idx_df.empty and "close" in idx_df.columns:
                for r in idx_df[["trade_date", "close"]].itertuples(index=False):
                    if r.close is not None:
                        baseline_close_by_date[str(r.trade_date)] = float(r.close)
            else:
                data_unavailable.append(
                    f"index_daily({baseline_index_code}) returned empty — alpha 字段降级为 None"
                )

    # -------- daily_basic on T (turnover, circ_mv, pe, pb) -------------------
    db_basic_t = tushare.call("daily_basic", trade_date=trade_date, force_sync=force_sync)
    db_basic_lookup: dict[str, dict[str, Any]] = {}
    if not db_basic_t.empty:
        for r in db_basic_t.itertuples(index=False):
            db_basic_lookup[str(r.ts_code)] = {
                "turnover_rate": getattr(r, "turnover_rate", None),
                "volume_ratio": getattr(r, "volume_ratio", None),
                "pe": getattr(r, "pe", None),
                "pb": getattr(r, "pb", None),
                "circ_mv": getattr(r, "circ_mv", None),
                "total_mv": getattr(r, "total_mv", None),
            }
    else:
        data_unavailable.append("daily_basic(T)")

    # -------- moneyflow (5-day per stock, optional) -------------------------
    mf_start = _shift_calendar_days(trade_date, -(moneyflow_lookback + 7))
    mf_df, mf_err = _try_optional(
        tushare,
        "moneyflow",
        params={"start_date": mf_start, "end_date": trade_date},
        force_sync=force_sync,
    )
    if mf_err:
        data_unavailable.append(f"moneyflow ({mf_err})")
    mf_df = _normalize_id_cols(mf_df)
    if mf_df is not None and not mf_df.empty:
        mf_df = mf_df[mf_df["ts_code"].isin(candidate_codes)]

    # -------- limit_list_d 60-day (flag stocks with prior涨停) ---------------
    lu_start = history_dates[0] if history_dates else trade_date
    lu_df, lu_err = _try_optional(
        tushare,
        "limit_list_d",
        params={"start_date": lu_start, "end_date": trade_date, "limit_type": "U"},
        force_sync=force_sync,
    )
    if lu_err:
        data_unavailable.append(f"limit_list_d ({lu_err})")
    lu_by_code: dict[str, list[str]] = {}
    if lu_df is not None and not lu_df.empty:
        for r in lu_df.itertuples(index=False):
            lu_by_code.setdefault(str(r.ts_code), []).append(str(r.trade_date))

    # -------- sector strength (tier 1: limit_cpt_list, fallback: industry agg)
    cpt_df, cpt_err = _try_optional(
        tushare, "limit_cpt_list", trade_date=trade_date, force_sync=force_sync
    )
    if cpt_err:
        data_unavailable.append(f"limit_cpt_list ({cpt_err})")
    if cpt_df is not None and not cpt_df.empty:
        bundle.sector_strength_source = "limit_cpt_list"
        top = cpt_df.sort_values("rank").head(10) if "rank" in cpt_df.columns else cpt_df.head(10)
        bundle.sector_strength_data = {"top_sectors": top.to_dict(orient="records")}
    else:
        # Industry fallback: aggregate watchlist by industry
        agg: dict[str, int] = {}
        for w in watchlist:
            ind = w.get("industry") or "未分类"
            agg[ind] = agg.get(ind, 0) + 1
        bundle.sector_strength_source = "industry_fallback"
        bundle.sector_strength_data = {
            "top_sectors": [
                {"sector": k, "watchlist_count": v}
                for k, v in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:10]
            ]
        }

    # -------- per-stock context assembly ------------------------------------
    daily_by_code = _index_daily_by_code(daily_df)
    mf_by_code = _index_moneyflow_by_code(mf_df)
    candidates: list[dict[str, Any]] = []
    for w in watchlist:
        code = w["ts_code"]
        history = daily_by_code.get(code, [])
        if not history:
            # No data for this stock — still include it so LLM sees missing_data
            candidates.append(
                {
                    "candidate_id": code,
                    "ts_code": code,
                    "name": w.get("name"),
                    "industry": w.get("industry"),
                    "tracked_since": w.get("tracked_since"),
                    "tracked_days": _calendar_days_between(w["tracked_since"], trade_date),
                    "_missing_history": True,
                }
            )
            continue

        rec = _build_candidate_row(
            watchlist_row=w,
            trade_date=trade_date,
            history=history,
            daily_basic=db_basic_lookup.get(code, {}),
            moneyflow_5d=mf_by_code.get(code, [])[-moneyflow_lookback:],
            limit_up_dates=sorted(lu_by_code.get(code, [])),
            baseline_index_code=baseline_index_code,
            baseline_close_by_date=baseline_close_by_date,
        )
        candidates.append(rec)

    # -------- market summary -------------------------------------------------
    bundle.market_summary = {
        "watchlist_total": len(watchlist),
        "history_lookback_trade_days": history_lookback,
        "moneyflow_lookback_days": moneyflow_lookback,
    }
    bundle.candidates = candidates
    bundle.data_unavailable = data_unavailable
    return bundle


def _index_daily_by_code(df: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    if df is None or df.empty or "ts_code" not in df.columns:
        return {}
    df = _normalize_id_cols(df)
    if df is None:
        return {}
    df = df.sort_values("trade_date") if "trade_date" in df.columns else df
    out: dict[str, list[dict[str, Any]]] = {}
    for code, group in df.groupby("ts_code"):
        out[str(code)] = group.to_dict(orient="records")
    return out


def _index_moneyflow_by_code(df: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    if df is None or df.empty or "ts_code" not in df.columns:
        return {}
    df = _normalize_id_cols(df)
    if df is None:
        return {}
    df = df.sort_values("trade_date") if "trade_date" in df.columns else df
    out: dict[str, list[dict[str, Any]]] = {}
    for code, group in df.groupby("ts_code"):
        out[str(code)] = group.to_dict(orient="records")
    return out


def _compute_alpha_pct(
    history: list[dict[str, Any]],
    baseline_close_by_date: dict[str, float],
    n: int,
) -> float | None:
    """alpha_n = stock_pct_chg_n − baseline_pct_chg_n (over the last n trade days).

    Both legs use simple compounded close-to-close return. Returns None when
    either leg can't be computed (insufficient history / baseline data missing
    on the required dates).
    """
    if len(history) <= n:
        return None
    end_row = history[-1]
    start_row = history[-1 - n]
    end_close = end_row.get("close")
    start_close = start_row.get("close")
    if end_close is None or start_close is None or start_close <= 0:
        return None
    end_date = str(end_row.get("trade_date") or "")
    start_date = str(start_row.get("trade_date") or "")
    base_end = baseline_close_by_date.get(end_date)
    base_start = baseline_close_by_date.get(start_date)
    if base_end is None or base_start is None or base_start <= 0:
        return None
    stock_ret = (float(end_close) / float(start_close) - 1.0) * 100.0
    base_ret = (float(base_end) / float(base_start) - 1.0) * 100.0
    return round(stock_ret - base_ret, 2)


def _classify_rel_strength(alpha_20d: float | None) -> str | None:
    if alpha_20d is None:
        return None
    if alpha_20d > ALPHA_LEADING_THRESHOLD:
        return "leading"
    if alpha_20d < ALPHA_LAGGING_THRESHOLD:
        return "lagging"
    return "in_line"


def _build_candidate_row(
    *,
    watchlist_row: dict[str, Any],
    trade_date: str,
    history: list[dict[str, Any]],
    daily_basic: dict[str, Any],
    moneyflow_5d: list[dict[str, Any]],
    limit_up_dates: list[str],
    baseline_index_code: str = DEFAULT_BASELINE_INDEX_CODE,
    baseline_close_by_date: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compress (up to) 250-day history → moving averages + base/washout +
    VCP / resistance features.

    Reduces token usage by emitting compact scalars; the recent 5 OHLCV rows
    are still passed verbatim for form reference. v0.3.0 (PR-2):
        * input window widened from 60 → 250 trading days (E2-A) and sliced
          internally — 60d for MAs / aggregates, full window for VCP and
          120d / 250d resistance.
        * new fields: atr_10d_pct / atr_10d_quantile_in_60d / bbw_20d /
          bbw_compression_ratio (P0-3) and high_120d / high_250d / low_120d /
          dist_to_120d_high_pct / dist_to_250d_high_pct / is_above_120d_high /
          is_above_250d_high / pos_in_120d_range (P0-4).
    """
    closes = [float(r["close"]) for r in history if r.get("close") is not None]
    if not closes:
        return {
            "candidate_id": watchlist_row["ts_code"],
            "ts_code": watchlist_row["ts_code"],
            "name": watchlist_row.get("name"),
            "tracked_since": watchlist_row["tracked_since"],
            "tracked_days": _calendar_days_between(watchlist_row["tracked_since"], trade_date),
            "_missing_history": True,
        }

    # The 60d "compressed feature" slice — preserve pre-v0.3.0 semantics for
    # ma60 / high_60d / low_60d / pct_chg_60d when the input history is now
    # 250d wide.
    closes_60 = closes[-60:]
    last_close = closes[-1]

    def _ma(n: int) -> float | None:
        if len(closes) < n:
            return None
        return round(sum(closes[-n:]) / n, 3)

    ma5, ma10, ma20, ma60 = _ma(5), _ma(10), _ma(20), _ma(60)
    above_ma60 = ma60 is not None and last_close > ma60
    above_ma20 = ma20 is not None and last_close > ma20

    # 60d aggregates (over the most-recent 60 closes)
    high_60d = round(max(closes_60), 3)
    low_60d = round(min(closes_60), 3)
    range_pct_60d = round((high_60d - low_60d) / max(low_60d, 1e-9) * 100, 2)
    pct_chg_60d = (
        round((last_close / closes_60[0] - 1) * 100, 2)
        if len(closes_60) >= 60 and closes_60[0] > 0
        else None
    )

    # v0.3.0 P0-3 — VCP波动率收敛指标
    atr_series = _compute_atr_series(history)
    bbw_series = _compute_bbw_series(history)
    last_idx = len(history) - 1
    atr_now = atr_series[last_idx] if atr_series else None
    bbw_now = bbw_series[last_idx] if bbw_series else None
    atr_10d_pct: float | None = None
    if atr_now is not None and last_close > 0:
        atr_10d_pct = round(atr_now / last_close * 100, 3)
    atr_10d_quantile_in_60d = _quantile_in_window(
        atr_series, last_idx, ATR_QUANTILE_LOOKBACK
    )
    if atr_10d_quantile_in_60d is not None:
        atr_10d_quantile_in_60d = round(atr_10d_quantile_in_60d, 3)
    bbw_20d = round(bbw_now, 3) if bbw_now is not None else None
    bbw_compression_ratio: float | None = None
    if bbw_now is not None:
        prior = [b for b in bbw_series[-BBW_COMPRESSION_LOOKBACK:] if b is not None]
        if len(prior) >= BBW_COMPRESSION_LOOKBACK:
            mean_prior = sum(prior) / len(prior)
            if mean_prior > 0:
                bbw_compression_ratio = round(bbw_now / mean_prior, 3)

    # v0.3.0 P0-4 — 120d / 250d 阻力位距离 (closes-based to match high_60d).
    # Compute the raw (unrounded) extremes for comparison against last_close so
    # boundary cases like "last close IS the 60d high" don't flip on rounding;
    # round only the emitted scalar field.
    def _window_extremes(n: int) -> tuple[float | None, float | None]:
        if len(closes) < n:
            return None, None
        sl = closes[-n:]
        return max(sl), min(sl)

    high_120d_raw, low_120d_raw = _window_extremes(RESIST_120D)
    high_250d_raw, _ = _window_extremes(RESIST_250D)
    high_120d = round(high_120d_raw, 3) if high_120d_raw is not None else None
    high_250d = round(high_250d_raw, 3) if high_250d_raw is not None else None
    low_120d = round(low_120d_raw, 3) if low_120d_raw is not None else None
    dist_to_120d_high_pct = (
        round((last_close - high_120d_raw) / high_120d_raw * 100, 2)
        if high_120d_raw not in (None, 0)
        else None
    )
    dist_to_250d_high_pct = (
        round((last_close - high_250d_raw) / high_250d_raw * 100, 2)
        if high_250d_raw not in (None, 0)
        else None
    )
    is_above_120d_high = high_120d_raw is not None and last_close > high_120d_raw
    is_above_250d_high = high_250d_raw is not None and last_close > high_250d_raw
    pos_in_120d_range: float | None = None
    if (
        high_120d_raw is not None
        and low_120d_raw is not None
        and high_120d_raw > low_120d_raw
    ):
        pos_in_120d_range = round(
            (last_close - low_120d_raw) / (high_120d_raw - low_120d_raw), 3
        )

    # v0.5.0 P1-1 — RPS / 大盘相对 alpha. F10 — 5d / 20d / 60d。
    baseline_close_by_date = baseline_close_by_date or {}
    alpha_5d_pct = _compute_alpha_pct(history, baseline_close_by_date, 5)
    alpha_20d_pct = _compute_alpha_pct(history, baseline_close_by_date, 20)
    alpha_60d_pct = _compute_alpha_pct(history, baseline_close_by_date, 60)
    rel_strength_label = _classify_rel_strength(alpha_20d_pct)

    # Base / washout features — find the最近 anomaly day (T) and the platform before it
    # The异动 day is `trade_date` itself (or the最近 row matching T). The "base" is
    # the period between the previous notable up-move and T.
    t_idx = next(
        (i for i, r in enumerate(history) if str(r.get("trade_date")) == trade_date),
        len(history) - 1,
    )
    # v0.3.0 PR-2 — keep the base/washout window at 60d even though `history`
    # is now up to 250d, so `base_*` field semantics stay backward-compatible.
    base_window_size = 60
    if t_idx > 0:
        base_start = max(0, t_idx - base_window_size)
        base_window_pre_t = history[base_start:t_idx]
    else:
        base_window_pre_t = []

    # base_days = consecutive days before T where pct_chg is moderate (|pct_chg| < 4)
    base_days = 0
    for r in reversed(base_window_pre_t):
        if abs(float(r.get("pct_chg") or 0)) < 4.0:
            base_days += 1
        else:
            break

    # Drawdown within base window: (max_close - min_close) / max_close * 100
    base_closes = [float(r["close"]) for r in base_window_pre_t if r.get("close") is not None]
    base_max_drawdown_pct = None
    base_avg_vol = None
    base_vol_shrink_ratio = None
    base_avg_turnover_rate = None
    if base_closes:
        bmax = max(base_closes)
        bmin = min(base_closes)
        if bmax > 0:
            base_max_drawdown_pct = round((bmax - bmin) / bmax * 100, 2)
    base_vols_pre = [float(r["vol"]) for r in base_window_pre_t if r.get("vol") is not None]
    if base_vols_pre:
        base_avg_vol = round(sum(base_vols_pre) / len(base_vols_pre), 2)
        # Compare平均 of整理后期 vs 整理前期 — shrinkage indicator
        if len(base_vols_pre) >= 10:
            half = len(base_vols_pre) // 2
            early = sum(base_vols_pre[:half]) / max(half, 1)
            late = sum(base_vols_pre[half:]) / max(len(base_vols_pre) - half, 1)
            if early > 0:
                base_vol_shrink_ratio = round(late / early, 2)

    # days_since_last_limit_up — strictly before T
    prior_limit_ups = [d for d in limit_up_dates if d < trade_date]
    days_since_last_limit_up: int | None = None
    if prior_limit_ups:
        days_since_last_limit_up = _calendar_days_between(prior_limit_ups[-1], trade_date)

    # Recent 5 days OHLCV (verbatim, for form reference)
    recent5 = [
        {
            "date": str(r.get("trade_date")),
            "open": round2(r.get("open")),
            "high": round2(r.get("high")),
            "low": round2(r.get("low")),
            "close": round2(r.get("close")),
            "pct_chg": round2(r.get("pct_chg")),
            "vol": _opt_int(r.get("vol")),
        }
        for r in history[-5:]
    ]

    # Moneyflow summary
    mf_summary: dict[str, Any] = {}
    if moneyflow_5d:
        net_amounts = [float(r.get("net_mf_amount") or 0) for r in moneyflow_5d]
        elg_amounts = [float(r.get("buy_elg_amount") or 0) for r in moneyflow_5d]
        lg_amounts = [float(r.get("buy_lg_amount") or 0) for r in moneyflow_5d]
        cum_net_yi = round(sum(net_amounts) / 1e4, 3)  # 万元 → 亿
        cum_elg_lg_yi = round(sum(elg_amounts + lg_amounts) / 1e4, 3)
        # trend: increasing if last3 > first2 mean
        trend = "flat"
        if len(net_amounts) >= 5:
            first2 = sum(net_amounts[:2]) / 2
            last3 = sum(net_amounts[-3:]) / 3
            if last3 > first2 * 1.2:
                trend = "rising"
            elif last3 < first2 * 0.8:
                trend = "falling"
        mf_summary = {
            "cum_net_mf_yi": cum_net_yi,
            "cum_elg_plus_lg_buy_yi": cum_elg_lg_yi,
            "net_mf_trend": trend,
            "rows_used": len(moneyflow_5d),
        }
    else:
        mf_summary = {"rows_used": 0}

    tracked_days = _calendar_days_between(watchlist_row["tracked_since"], trade_date)
    return {
        "candidate_id": watchlist_row["ts_code"],
        "ts_code": watchlist_row["ts_code"],
        "name": watchlist_row.get("name"),
        "industry": watchlist_row.get("industry"),
        "tracked_since": watchlist_row["tracked_since"],
        "tracked_days": tracked_days,
        # T-day snapshot (from watchlist row — the异动 day metrics)
        "anomaly_day": watchlist_row.get("last_screened"),
        "anomaly_pct_chg": watchlist_row.get("last_pct_chg"),
        "anomaly_body_ratio": watchlist_row.get("last_body_ratio"),
        "anomaly_turnover_rate": watchlist_row.get("last_turnover_rate"),
        "anomaly_vol_ratio_5d": watchlist_row.get("last_vol_ratio_5d"),
        # Latest market data
        "last_close": round2(last_close),
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "high_60d": high_60d,
        "low_60d": low_60d,
        "range_pct_60d": range_pct_60d,
        "pct_chg_60d": pct_chg_60d,
        # v0.3.0 P0-3 — VCP波动率收敛
        "atr_10d_pct": atr_10d_pct,
        "atr_10d_quantile_in_60d": atr_10d_quantile_in_60d,
        "bbw_20d": bbw_20d,
        "bbw_compression_ratio": bbw_compression_ratio,
        # v0.5.0 P1-1 — RPS / 大盘相对 alpha
        "alpha_5d_pct": alpha_5d_pct,
        "alpha_20d_pct": alpha_20d_pct,
        "alpha_60d_pct": alpha_60d_pct,
        "baseline_index_code": baseline_index_code,
        "rel_strength_label": rel_strength_label,
        # v0.3.0 P0-4 — 120d/250d 阻力位 (E3-A：不补 low_250d / pos_in_250d_range)
        "high_120d": high_120d,
        "high_250d": high_250d,
        "low_120d": low_120d,
        "dist_to_120d_high_pct": dist_to_120d_high_pct,
        "dist_to_250d_high_pct": dist_to_250d_high_pct,
        "is_above_120d_high": is_above_120d_high,
        "is_above_250d_high": is_above_250d_high,
        "pos_in_120d_range": pos_in_120d_range,
        # Washout / base features (the user's要求 #7 维度)
        "base_days": base_days,
        "base_max_drawdown_pct": base_max_drawdown_pct,
        "base_avg_vol": base_avg_vol,
        "base_vol_shrink_ratio": base_vol_shrink_ratio,
        "base_avg_turnover_rate": base_avg_turnover_rate,
        "days_since_last_limit_up": days_since_last_limit_up,
        "prior_limit_up_count_60d": len(prior_limit_ups),
        # Latest daily_basic
        "turnover_rate_t": round2(daily_basic.get("turnover_rate")),
        "volume_ratio_t": round2(daily_basic.get("volume_ratio")),
        "pe_t": round2(daily_basic.get("pe")),
        "pb_t": round2(daily_basic.get("pb")),
        "circ_mv_yi": normalize_to_yi("circ_mv", daily_basic.get("circ_mv")),
        "total_mv_yi": normalize_to_yi("total_mv", daily_basic.get("total_mv")),
        # Recent 5 OHLCV verbatim
        "recent_5d": recent5,
        # Moneyflow摘要 (5d)
        "moneyflow_5d_summary": mf_summary,
    }


# ---------------------------------------------------------------------------
# EVALUATE MODE — T+N realized-return computation (v0.4.0 P1-3)
# ---------------------------------------------------------------------------


def _resolve_horizon_dates(
    calendar: TradeCalendar,
    anomaly_date: str,
    horizons: tuple[int, ...] = EVALUATE_HORIZONS,
) -> dict[int, str]:
    """For each horizon n, resolve the trade_date that is n trade days AFTER
    ``anomaly_date`` (skipping non-open days, including holidays / weekends).

    Returns ``{n: yyyymmdd_string}`` for every requested horizon. Raises
    ``ValueError`` only when the calendar has no future trade days at all
    (which would indicate the calendar fixture is too short).
    """
    out: dict[int, str] = {}
    cursor = anomaly_date
    advanced = 0
    target_n = max(horizons)
    while advanced < target_n:
        cursor = calendar.next_open(cursor)
        advanced += 1
        if advanced in horizons:
            out[advanced] = cursor
    return out


def _compute_realized_returns(
    *,
    t_close: float | None,
    horizon_closes: dict[int, float | None],
    window_5d_closes: list[float | None],
    window_10d_closes: list[float | None],
) -> dict[str, float | None]:
    """Convert raw OHLCV inputs into the realised-return scalar metrics
    persisted in ``va_realized_returns``.

    Args:
        t_close: T-day close (basis for all percentage calcs).
        horizon_closes: ``{1: c1, 3: c3, 5: c5, 10: c10}``. Any missing horizon
            value is OK — it surfaces as ``None`` in the result.
        window_5d_closes: ordered closes for T+1..T+5 (length ≤ 5; may
            contain ``None`` for suspended days).
        window_10d_closes: same idea for T+1..T+10.

    Output keys: ``ret_t1`` ``ret_t3`` ``ret_t5`` ``ret_t10`` ``max_close_5d``
    ``max_close_10d`` ``max_ret_5d`` ``max_ret_10d`` ``max_dd_5d``.
    """

    def _pct(num: float | None) -> float | None:
        if num is None or t_close is None or t_close <= 0:
            return None
        return round((num / t_close - 1) * 100, 2)

    out: dict[str, float | None] = {
        "ret_t1": _pct(horizon_closes.get(1)),
        "ret_t3": _pct(horizon_closes.get(3)),
        "ret_t5": _pct(horizon_closes.get(5)),
        "ret_t10": _pct(horizon_closes.get(10)),
    }
    valid_5 = [c for c in window_5d_closes if c is not None]
    valid_10 = [c for c in window_10d_closes if c is not None]
    out["max_close_5d"] = round(max(valid_5), 3) if valid_5 else None
    out["max_close_10d"] = round(max(valid_10), 3) if valid_10 else None
    out["max_ret_5d"] = _pct(out["max_close_5d"])
    out["max_ret_10d"] = _pct(out["max_close_10d"])
    # G2 决策: max_dd from T = (min(close[T+1..T+5]) - t_close) / t_close × 100
    out["max_dd_5d"] = (
        _pct(min(valid_5)) if valid_5 else None
    )
    return out


def _classify_data_status(
    *,
    horizon_closes: dict[int, float | None],
    horizons: tuple[int, ...],
    today: str,
    horizon_dates: dict[int, str],
) -> str:
    """Determine ``data_status`` per the v3 G5 rule:

    * ``pending``  — T+1 trade_date is still in the future (no horizon column
                     can possibly be filled yet).
    * ``partial``  — max_horizon trade_date is in the future, OR any reachable
                     horizon row has missing close (suspension / data gap).
    * ``complete`` — max_horizon is in the past AND every horizon was filled.
    """
    if not horizon_dates:
        return "pending"
    h1_date = horizon_dates.get(min(horizons))
    if h1_date is not None and h1_date > today:
        return "pending"
    max_n = max(horizons)
    max_date = horizon_dates.get(max_n)
    max_reached = max_date is not None and max_date <= today
    all_filled = all(horizon_closes.get(n) is not None for n in horizons)
    if max_reached and all_filled:
        return "complete"
    return "partial"


def fetch_anomaly_dates_within_lookback(
    db: Any, *, today: str, lookback_days: int
) -> list[tuple[str, str]]:
    """Return ``[(anomaly_date, ts_code)]`` for every va_anomaly_history row
    whose ``anomaly_date`` is within the trailing ``lookback_days`` calendar
    days of ``today``."""
    cutoff = _shift_calendar_days(today, -int(lookback_days))
    rows = db.fetchall(
        "SELECT trade_date, ts_code FROM va_anomaly_history "
        "WHERE trade_date >= ? ORDER BY trade_date, ts_code",
        (cutoff,),
    )
    return [(str(r[0]), str(r[1])) for r in rows]


def upsert_realized_return(
    db: Any,
    *,
    anomaly_date: str,
    ts_code: str,
    t_close: float | None,
    horizon_closes: dict[int, float | None],
    metrics: dict[str, float | None],
    data_status: str,
) -> None:
    """UPSERT one row into ``va_realized_returns``."""
    db.execute(
        "DELETE FROM va_realized_returns WHERE anomaly_date=? AND ts_code=?",
        (anomaly_date, ts_code),
    )
    db.execute(
        "INSERT INTO va_realized_returns(anomaly_date, ts_code, t_close, "
        "t1_close, t3_close, t5_close, t10_close, "
        "ret_t1, ret_t3, ret_t5, ret_t10, "
        "max_close_5d, max_close_10d, max_ret_5d, max_ret_10d, max_dd_5d, "
        "data_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            anomaly_date,
            ts_code,
            t_close,
            horizon_closes.get(1),
            horizon_closes.get(3),
            horizon_closes.get(5),
            horizon_closes.get(10),
            metrics.get("ret_t1"),
            metrics.get("ret_t3"),
            metrics.get("ret_t5"),
            metrics.get("ret_t10"),
            metrics.get("max_close_5d"),
            metrics.get("max_close_10d"),
            metrics.get("max_ret_5d"),
            metrics.get("max_ret_10d"),
            metrics.get("max_dd_5d"),
            data_status,
        ),
    )


def fetch_completed_realized_keys(db: Any) -> set[tuple[str, str]]:
    """Return ``{(anomaly_date, ts_code)}`` for every row with
    ``data_status='complete'`` — used to skip work on subsequent evaluate
    runs (idempotency)."""
    rows = db.fetchall(
        "SELECT anomaly_date, ts_code FROM va_realized_returns "
        "WHERE data_status = 'complete'"
    )
    return {(str(r[0]), str(r[1])) for r in rows}
