"""Data layer for the limit-up-board strategy.

DESIGN §12.2 (T-resolution) + §11.3 (sector_strength fallback chain) + S2 (close_after config) +
S4 (zero candidates legal) + Q2 (main board only) + C5 (raw units in DB, normalized in prompt).

v0.5+ (lightgbm_design.md §7.2): when a non-None ``lgb_scorer`` is passed to
:func:`collect_round1`, each candidate dict gets ``lgb_score`` / ``lgb_decile`` /
``lgb_feature_missing`` and the bundle captures the model id + per-row audit
payloads for the runner to persist to ``lub_lgb_predictions``.

Key public entry points:
    resolve_trade_date(...)            — Step 0
    collect_round1(...)                — Step 1 (returns candidates + market summary +
                                          sector_strength + data_unavailable + LGB scores)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from deeptrade.core.tushare_client import (
    TushareClient,
    TushareUnauthorizedError,
)

from .calendar import TradeCalendar

if TYPE_CHECKING:  # pragma: no cover
    from .lgb.scorer import LgbScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 0 — resolve trade date
# ---------------------------------------------------------------------------


def resolve_trade_date(
    now_dt: datetime,
    calendar: TradeCalendar,
    *,
    user_specified: str | None = None,
    allow_intraday: bool = False,
    close_after: time = time(18, 0),
) -> tuple[str, str]:
    """Return (T, T+1) per DESIGN §12.2.

    T defaults to the most recent CLOSED trade day:
      * if today is open AND now ≥ close_after  → today
      * if today is open AND allow_intraday      → today (with intraday banner)
      * else                                     → pretrade_date(today)

    T+1 is the first open day strictly after T.
    """
    if user_specified:
        T = user_specified
        return T, calendar.next_open(T)

    today = now_dt.strftime("%Y%m%d")
    today_is_open = calendar.is_open(today)

    if today_is_open and (now_dt.time() >= close_after or allow_intraday):
        T = today
    elif today_is_open:
        # Today is a trade day but it's intraday and user has not opted in.
        T = calendar.pretrade_date(today)
    else:
        # Non-trading day (weekend/holiday). Walk back.
        T = calendar.pretrade_date(today)

    return T, calendar.next_open(T)


# ---------------------------------------------------------------------------
# Filters: main board / ST / suspended
# ---------------------------------------------------------------------------


def main_board_filter(stock_basic: pd.DataFrame) -> pd.DataFrame:
    """Keep only Shanghai/Shenzhen MAIN board (Q2 fix).

    Excludes ChiNext (300xxx), STAR (688xxx), BSE (8xxxxx), and CDR.
    Tushare ``stock_basic.market`` is a Chinese label like '主板'.
    """
    if "market" not in stock_basic.columns or "exchange" not in stock_basic.columns:
        raise ValueError("stock_basic missing market/exchange columns")
    df = stock_basic[
        (stock_basic["market"] == "主板") & (stock_basic["exchange"].isin(["SSE", "SZSE"]))
    ].copy()
    if "list_status" in df.columns:
        df = df[df["list_status"] == "L"]
    return df.reset_index(drop=True)


def exclude_st(df: pd.DataFrame, st_codes: set[str]) -> pd.DataFrame:
    """Drop rows whose ts_code is in the ST / *ST set."""
    if df.empty:
        return df
    return df[~df["ts_code"].isin(st_codes)].reset_index(drop=True)


def exclude_suspended(df: pd.DataFrame, suspended_codes: set[str]) -> pd.DataFrame:
    """Drop rows whose ts_code is suspended on T."""
    if df.empty:
        return df
    return df[~df["ts_code"].isin(suspended_codes)].reset_index(drop=True)


def _apply_market_filter(
    candidates_df: pd.DataFrame,
    *,
    max_float_mv_yi: float,
    max_close_yuan: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """v0.4 — keep only rows whose 流通市值 < ``max_float_mv_yi`` (亿) AND
    close < ``max_close_yuan`` (元). Null in either field → dropped (conservative;
    we cannot validate "small cap / low price" claims without the data).

    Returns ``(filtered_df, summary)`` where summary is the candidate_filter_summary
    payload that gets stored in ``bundle.market_summary``.
    """
    n_before = int(len(candidates_df))
    summary: dict[str, Any] = {
        "before": n_before,
        "after": n_before,
        "max_float_mv_yi": max_float_mv_yi,
        "max_close_yuan": max_close_yuan,
    }
    if n_before == 0:
        return candidates_df, summary
    fm_yi = pd.to_numeric(candidates_df.get("float_mv"), errors="coerce") / 1e8
    cl = pd.to_numeric(candidates_df.get("close"), errors="coerce")
    mask = (
        fm_yi.notna()
        & cl.notna()
        & (fm_yi < max_float_mv_yi)
        & (cl < max_close_yuan)
    )
    filtered = candidates_df[mask].reset_index(drop=True)
    summary["after"] = int(len(filtered))
    return filtered, summary


# ---------------------------------------------------------------------------
# Sector strength resolver — three-tier fallback (F2 fix + §11.3)
# ---------------------------------------------------------------------------


SectorStrengthSource = Literal["limit_cpt_list", "lu_desc_aggregation", "industry_fallback"]


@dataclass
class SectorStrength:
    """Sector heat / leadership data fed into the prompt.

    `source` is exposed verbatim to the LLM via ``sector_strength_source`` so
    the model can downweight confidence when it sees a fallback label.
    """

    source: SectorStrengthSource
    data: dict[str, Any]


def resolve_sector_strength(
    *,
    candidates: pd.DataFrame,
    limit_cpt_list: pd.DataFrame | None,
    limit_list_ths: pd.DataFrame | None,
) -> SectorStrength:
    """Pick the best available sector data and aggregate by candidate's sector tag.

    Priority: limit_cpt_list > limit_list_ths.lu_desc aggregation >
    stock_basic.industry aggregation.
    """
    # Tier 1: official concept rankings
    if limit_cpt_list is not None and not limit_cpt_list.empty:
        # Top-ranked sectors (rank ascending, take first ~10)
        top = limit_cpt_list.sort_values("rank").head(10)
        return SectorStrength(
            source="limit_cpt_list",
            data={
                "top_sectors": top.to_dict(orient="records"),
                "candidates_with_sector_tag": [],  # joined externally if needed
            },
        )

    # Tier 2: aggregate THS涨停原因
    if limit_list_ths is not None and not limit_list_ths.empty:
        agg = (
            limit_list_ths.groupby("lu_desc", dropna=True)
            .agg(up_nums=("ts_code", "count"))
            .reset_index()
            .sort_values("up_nums", ascending=False)
            .head(10)
        )
        return SectorStrength(
            source="lu_desc_aggregation",
            data={"top_sectors": agg.to_dict(orient="records")},
        )

    # Tier 3: aggregate by stock_basic.industry  (last resort)
    if candidates is not None and not candidates.empty and "industry" in candidates.columns:
        agg = (
            candidates.groupby("industry", dropna=True)
            .agg(up_nums=("ts_code", "count"))
            .reset_index()
            .sort_values("up_nums", ascending=False)
            .head(10)
        )
        return SectorStrength(
            source="industry_fallback",
            data={"top_sectors": agg.to_dict(orient="records")},
        )

    return SectorStrength(source="industry_fallback", data={"top_sectors": []})


# ---------------------------------------------------------------------------
# Normalizers (C5 fix: prompt uses normalized units; DB keeps raw)
# B3.1 (M6) fix: tushare fields have HETEROGENEOUS raw units; a simple
# `value / 1e8` is wrong for moneyflow.* (which is 万元) and daily_basic.circ_mv
# (also 万元). FIELD_UNITS_RAW is the source of truth.
# ---------------------------------------------------------------------------


# Per-field raw unit declarations, sourced from tushare official docs.
# Values absent from this map default to "元" (the most common unit).
FIELD_UNITS_RAW: dict[str, str] = {
    # limit_list_d (元)
    "fd_amount": "元",
    "limit_amount": "元",
    "amount": "元",
    "float_mv": "元",
    "total_mv": "元",
    # top_list (元)
    "net_amount": "元",
    # daily_basic (mixed: market values are 万元 in tushare!)
    "circ_mv": "万元",
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
    # daily (千元 for amount, 手 for vol)
    # Note: limit_list_d.amount is 元 but daily.amount is 千元 — context-dependent
    # callers must use normalize_field with the API context if they need disambiguation.
}


# B1 — known A-share 游资席位 substring hints. Match is verbatim against
# top_inst.exalter; on hit, the actual exalter string is written into
# lhb_famous_seats (we never expose the hint label to the LLM, preserving
# anonymity per DESIGN §12 R3 spirit).
FAMOUS_SEATS_HINTS: tuple[str, ...] = (
    "拉萨团结路",
    "拉萨东环路",
    "拉萨金融城南环路",
    "宁波桑田路",
    "宁波解放南路",
    "深圳益田路荣超商务中心",
    "中信证券上海溧阳路",
    "华泰证券厦门厦禾路",
    "国泰君安上海江苏路",
    "国泰君安顺德大良",
    "财通证券杭州体育场路",
    "光大证券宁波解放南路",
    "东方财富证券拉萨",
    "国金证券上海互联网金融",
    "招商证券深圳深南大道",
)


def normalize_to_yi(field: str, raw_value: float | None) -> float | None:
    """Convert a raw field value to 亿 based on its declared unit."""
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


def normalize_to_wan(field: str, raw_value: float | None) -> float | None:
    """Convert a raw field value to 万 based on its declared unit."""
    if raw_value is None or pd.isna(raw_value):
        return None
    unit = FIELD_UNITS_RAW.get(field, "元")
    if unit == "元":
        factor = 1e4
    elif unit == "万元":
        factor = 1.0
    elif unit == "千元":
        factor = 0.1
    else:
        return None
    return round(float(raw_value) / factor, 2)


def yi(value: float | None) -> float | None:
    """Legacy helper assuming raw='元'. Prefer ``normalize_to_yi(field, value)``."""
    if value is None or pd.isna(value):
        return None
    return round(float(value) / 1e8, 2)


def wan(value: float | None) -> float | None:
    """Legacy helper assuming raw='元'. Prefer ``normalize_to_wan(field, value)``."""
    if value is None or pd.isna(value):
        return None
    return round(float(value) / 1e4, 2)


def round2(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


# ---------------------------------------------------------------------------
# Round-1 collection
# ---------------------------------------------------------------------------


@dataclass
class Round1Bundle:
    """Everything the R1 LLM stage needs.

    v0.5+ — ``lgb_model_id`` captures which LightGBM booster produced the
    ``lgb_score`` values on each candidate dict; ``None`` 表示 LGB 未启用 /
    未加载（report 会显示 ``lgb_model_id: disabled``)。
    ``lgb_predictions`` 是 :mod:`limit_up_board.lgb.audit` 准备好的批量审计
    payload 列表（每行一只候选股 × 一次 run），由 runner 在 Step 1 之后
    INSERT 到 ``lub_lgb_predictions``。
    """

    trade_date: str
    next_trade_date: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    market_summary: dict[str, Any] = field(default_factory=dict)
    sector_strength: SectorStrength = field(
        default_factory=lambda: SectorStrength(source="industry_fallback", data={"top_sectors": []})
    )
    data_unavailable: list[str] = field(default_factory=list)
    lgb_model_id: str | None = None
    lgb_predictions: list[dict[str, Any]] = field(default_factory=list)


def collect_round1(
    *,
    tushare: TushareClient,
    trade_date: str,
    next_trade_date: str,
    prev_trade_date: str | None = None,
    daily_lookback: int = 30,
    moneyflow_lookback: int = 5,
    max_float_mv_yi: float = 100.0,
    max_close_yuan: float = 15.0,
    force_sync: bool = False,
    lgb_scorer: LgbScorer | None = None,
) -> Round1Bundle:
    """Assemble the R1 input bundle.

    The flow:
        1. stock_basic (static) → main_board_filter()
        2. limit_list_d(T, limit='U') → join main_board → DROP if 0 candidates
           (zero candidates is a LEGAL outcome — S4)
        2b. v0.4 — drop candidates whose 流通市值 ≥ ``max_float_mv_yi``
            or 当前股价 ≥ ``max_close_yuan``; null in either field → drop
            (conservative; thresholds owned by ``LubConfig``).
        3. stock_st(T) (REQUIRED) / suspend_d(T) (optional) → drop codes
        4. limit_list_ths(T) (optional) → bring in lu_desc, tag, suc_rate
        5. limit_cpt_list(T) (optional) → sector strength tier 1
        6. limit_step(T) (REQUIRED) — for global ladder distribution
        7. daily / daily_basic / moneyflow over T-N..T (B1.2): histories that
           let the LLM see trend, turnover, market value, capital flow
        8. Build normalized prompt fields per candidate (raw → normalized via FIELD_UNITS_RAW)
    """
    bundle = Round1Bundle(trade_date=trade_date, next_trade_date=next_trade_date)
    data_unavailable: list[str] = []

    # 1. main board pool
    stock_basic = tushare.call("stock_basic", force_sync=force_sync)
    main_pool = main_board_filter(stock_basic)

    # 2. limit-up rows (limit='U'); we filter by limit afterward in case the
    # transport returns the full list_d.
    limit_list_d = tushare.call(
        "limit_list_d",
        trade_date=trade_date,
        params={"limit_type": "U"},
        force_sync=force_sync,
    )
    if "limit" in limit_list_d.columns:
        limit_list_d = limit_list_d[limit_list_d["limit"] == "U"]

    # join on ts_code
    if limit_list_d.empty:
        bundle.candidates = []
        return bundle  # zero candidates: legal end state (S4)
    candidates_df = limit_list_d.merge(
        main_pool[["ts_code", "market", "exchange", "industry", "list_date"]].rename(
            columns={"industry": "industry_basic"}
        ),
        on="ts_code",
        how="inner",
    )
    if candidates_df.empty:
        bundle.candidates = []
        return bundle

    # 2b. v0.4 — 流通市值 / 股价上限筛选（null → 过滤）。
    candidates_df, market_filter_summary = _apply_market_filter(
        candidates_df,
        max_float_mv_yi=max_float_mv_yi,
        max_close_yuan=max_close_yuan,
    )
    bundle.market_summary["candidate_filter_summary"] = market_filter_summary
    if candidates_df.empty:
        bundle.candidates = []
        return bundle

    # B1 — LHB (top_list / top_inst) — REQUIRED. Unauthorized must propagate.
    # candidate 未上榜时 lhb_* 字段为 null（合法事实），不进 data_unavailable。
    top_list_df = tushare.call("top_list", trade_date=trade_date, force_sync=force_sync)
    top_inst_df = tushare.call("top_inst", trade_date=trade_date, force_sync=force_sync)

    # B2 — cyq_perf (chip distribution) — REQUIRED.
    # 单只 candidate 在返回中无记录 → 该 candidate.missing_data 写入 cyq 字段名（LLM 自动填）。
    cyq_perf_df = tushare.call("cyq_perf", trade_date=trade_date, force_sync=force_sync)

    # 3a. ST exclusion — REQUIRED. Unauthorized must propagate to the runner.
    # Per DESIGN §11.1 + B1.3 fix: stock_st is in metadata.required → cannot
    # be silently skipped; runner will mark the run failed.
    st_df = tushare.call("stock_st", trade_date=trade_date, force_sync=force_sync)
    st_codes = set(st_df["ts_code"].astype(str)) if not st_df.empty else set()
    candidates_df = exclude_st(candidates_df, st_codes)

    # 3b. Suspended exclusion — OPTIONAL. F-H3: catch all transient errors.
    susp_df, susp_err = _try_optional(
        tushare, "suspend_d", trade_date=trade_date, force_sync=force_sync
    )
    if susp_err:
        data_unavailable.append(f"suspend_d ({susp_err})")
        susp_codes: set[str] = set()
    else:
        susp_codes = set(susp_df["ts_code"].astype(str)) if not susp_df.empty else set()
    candidates_df = exclude_suspended(candidates_df, susp_codes)

    if candidates_df.empty:
        bundle.candidates = []
        return bundle

    # 4. THS涨停榜 (optional). F-H3: catch all transient errors.
    ths_df, ths_err = _try_optional(
        tushare,
        "limit_list_ths",
        trade_date=trade_date,
        params={"limit_type": "U"},
        force_sync=force_sync,
    )
    if ths_err:
        data_unavailable.append(f"limit_list_ths ({ths_err})")

    # 5. concept ranking (optional). F-H3: same.
    cpt_df, cpt_err = _try_optional(
        tushare, "limit_cpt_list", trade_date=trade_date, force_sync=force_sync
    )
    if cpt_err:
        data_unavailable.append(f"limit_cpt_list ({cpt_err})")

    sector = resolve_sector_strength(
        candidates=candidates_df,
        limit_cpt_list=cpt_df,
        limit_list_ths=ths_df,
    )
    bundle.sector_strength = sector

    # 6. limit_step (required) — for global ladder distribution
    step_df = tushare.call("limit_step", trade_date=trade_date, force_sync=force_sync)
    today_step = _summarize_limit_step(step_df)
    # update() (not reassign) to preserve candidate_filter_summary set in step 2b.
    bundle.market_summary.update(
        {
            "limit_up_count": int(len(candidates_df)),
            "limit_step_distribution": today_step,
        }
    )
    # A2 — yesterday context: three keys (limit_step_trend / yesterday_failure_rate /
    # yesterday_winners_today). Best-effort; sub-fetch failures degrade individual
    # sections to null rather than failing the run.
    if prev_trade_date is not None:
        yctx, yctx_err = _collect_yesterday_context(
            tushare,
            trade_date=trade_date,
            prev_trade_date=prev_trade_date,
            today_step=today_step,
            force_sync=force_sync,
        )
        bundle.market_summary.update(yctx)
        if yctx_err:
            data_unavailable.extend(yctx_err)

    # 7. B1.2 — REQUIRED histories: daily / daily_basic / moneyflow over a window.
    # Tushare returns ALL stocks for one trade_date in one call; we instead query
    # by trade_date range so each ts_code's history is one slice of the result.
    # Buffer ×2 (calendar-day basis) covers weekends/holidays so even a 30-day
    # lookback (= ma20 + up_count_30d) reliably yields ≥30 trade rows.
    candidate_codes = set(candidates_df["ts_code"].astype(str))
    start_date = _shift_date(trade_date, -(daily_lookback * 2))
    daily_df = _fetch_history_window(
        tushare, "daily", start_date, trade_date, candidate_codes, force_sync=force_sync
    )
    daily_basic_df = _fetch_history_window(
        tushare,
        "daily_basic",
        start_date,
        trade_date,
        candidate_codes,
        force_sync=force_sync,
    )
    mf_start = _shift_date(trade_date, -(moneyflow_lookback + 5))
    moneyflow_df = _fetch_history_window(
        tushare,
        "moneyflow",
        mf_start,
        trade_date,
        candidate_codes,
        force_sync=force_sync,
    )

    # 8. Build normalized rows
    bundle.candidates = _build_candidate_rows(
        candidates_df,
        ths_df,
        daily_df=daily_df,
        daily_basic_df=daily_basic_df,
        moneyflow_df=moneyflow_df,
        top_list_df=top_list_df,
        top_inst_df=top_inst_df,
        cyq_perf_df=cyq_perf_df,
        daily_lookback=daily_lookback,
        moneyflow_lookback=moneyflow_lookback,
    )
    bundle.data_unavailable = data_unavailable

    # 9. v0.5 LGB — annotate each candidate dict with lgb_score / lgb_decile /
    # lgb_feature_missing (None when scorer disabled or model not loaded; never
    # raises — see lightgbm_design.md §7.3 "core red line").
    _attach_lgb_scores(
        bundle,
        candidates_df=candidates_df,
        daily_df=daily_df,
        daily_basic_df=daily_basic_df,
        moneyflow_df=moneyflow_df,
        top_list_df=top_list_df,
        top_inst_df=top_inst_df,
        cyq_perf_df=cyq_perf_df,
        scorer=lgb_scorer,
    )

    # B2.3 + F-M4 — Persist to business tables (DuckDB is the persistence layer
    # per DESIGN). Errors don't fail the run (cache_blob still holds the data),
    # but they DO surface via data_unavailable so users see them in the report.
    materialize_errors = _materialize_business_tables(
        tushare,
        stock_basic=stock_basic,
        limit_list_d=limit_list_d,
        ths_df=ths_df,
        daily_df=daily_df,
        daily_basic_df=daily_basic_df,
        moneyflow_df=moneyflow_df,
        top_list_df=top_list_df,
        top_inst_df=top_inst_df,
        cyq_perf_df=cyq_perf_df,
    )
    if materialize_errors:
        bundle.data_unavailable.extend(materialize_errors)
    return bundle


def _materialize_business_tables(
    tushare: TushareClient,
    *,
    stock_basic: pd.DataFrame,
    limit_list_d: pd.DataFrame,
    ths_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    daily_basic_df: pd.DataFrame | None,
    moneyflow_df: pd.DataFrame | None,
    top_list_df: pd.DataFrame | None = None,
    top_inst_df: pd.DataFrame | None = None,
    cyq_perf_df: pd.DataFrame | None = None,
) -> list[str]:
    """B2.3 + F-M4 — write tushare frames into the named business tables.

    Returns a list of error strings for any tables that failed to materialize.
    Caller surfaces these via data_unavailable / events instead of silent log.
    """
    errors: list[str] = []

    def _safe(table: str, df: pd.DataFrame, key_cols: list[str]) -> None:
        if df is None or df.empty:
            return
        try:
            tushare.materialize(table, df, key_cols=key_cols)
        except Exception as e:  # noqa: BLE001
            msg = f"materialize:{table} ({type(e).__name__}: {e})"
            logger.warning(msg)
            errors.append(msg)

    # All tables live under the lub_* prefix — this plugin owns its own
    # copy of every tushare-derived business table (Plan A pure isolation).
    _safe("lub_stock_basic", stock_basic, ["ts_code"])
    _safe("lub_limit_list_d", limit_list_d, ["trade_date", "ts_code", "limit"])
    _safe(
        "lub_limit_ths",
        ths_df if ths_df is not None else pd.DataFrame(),
        ["trade_date", "ts_code", "limit_type"],
    )
    _safe(
        "lub_daily",
        daily_df if daily_df is not None else pd.DataFrame(),
        ["ts_code", "trade_date"],
    )
    _safe(
        "lub_daily_basic",
        daily_basic_df if daily_basic_df is not None else pd.DataFrame(),
        ["ts_code", "trade_date"],
    )
    _safe(
        "lub_moneyflow",
        moneyflow_df if moneyflow_df is not None else pd.DataFrame(),
        ["ts_code", "trade_date"],
    )
    _safe(
        "lub_top_list",
        top_list_df if top_list_df is not None else pd.DataFrame(),
        ["trade_date", "ts_code", "reason"],
    )
    _safe(
        "lub_top_inst",
        top_inst_df if top_inst_df is not None else pd.DataFrame(),
        ["trade_date", "ts_code", "exalter", "side", "reason"],
    )
    _safe(
        "lub_cyq_perf",
        cyq_perf_df if cyq_perf_df is not None else pd.DataFrame(),
        ["trade_date", "ts_code"],
    )
    return errors


def _shift_date(yyyymmdd: str, days: int) -> str:
    """Naive ±days shift on YYYYMMDD (calendar days, not trade days). Adequate for
    setting a tushare query upper bound; result is filtered by trade_cal anyway."""
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    d = _dt.strptime(yyyymmdd, "%Y%m%d") + _td(days=days)
    return d.strftime("%Y%m%d")


def _fetch_history_window(
    tushare: TushareClient,
    api_name: str,
    start_date: str,
    end_date: str,
    candidate_codes: set[str],
    *,
    force_sync: bool = False,
) -> pd.DataFrame:
    """Fetch (api_name) for [start_date, end_date]; filter to candidates."""
    # tushare daily/daily_basic/moneyflow accept start_date/end_date for batch fetch.
    df = tushare.call(
        api_name,
        params={"start_date": start_date, "end_date": end_date},
        force_sync=force_sync,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if "ts_code" in df.columns and candidate_codes:
        df = df[df["ts_code"].astype(str).isin(candidate_codes)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# F-H3 — optional API wrapper
# ---------------------------------------------------------------------------


def _try_optional(
    tushare: TushareClient, api_name: str, **kwargs: Any
) -> tuple[pd.DataFrame, str | None]:
    """Call an optional tushare API; on transient failure return (empty df, err msg).

    Catches: TushareUnauthorizedError, TushareServerError, TushareRateLimitError.
    Required APIs should NOT use this — they should propagate failure.
    """
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
# A2 — yesterday-context (market sentiment three-pack)
# ---------------------------------------------------------------------------


def _collect_yesterday_context(
    tushare: TushareClient,
    *,
    trade_date: str,
    prev_trade_date: str,
    today_step: dict[str, int],
    force_sync: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Fetch T-1 limit_step / limit_list_d + T daily, derive market sentiment summary.

    Returns (market_summary_patch, errors). Sub-fetch failures degrade gracefully
    (the corresponding section becomes null) and are reported in errors.
    """
    errors: list[str] = []

    step_prev_df, err = _try_optional(
        tushare, "limit_step", trade_date=prev_trade_date, force_sync=force_sync
    )
    if err:
        errors.append(f"limit_step[T-1] ({err})")
    step_prev = _summarize_limit_step(step_prev_df)

    ll_prev_df, err = _try_optional(
        tushare, "limit_list_d", trade_date=prev_trade_date, force_sync=force_sync
    )
    if err:
        errors.append(f"limit_list_d[T-1] ({err})")

    daily_t_df, err = _try_optional(
        tushare, "daily", trade_date=trade_date, force_sync=force_sync
    )
    if err:
        errors.append(f"daily[T] ({err})")

    return {
        "limit_step_distribution_prev": step_prev,
        "limit_step_trend": _limit_step_trend(today_step, step_prev),
        "yesterday_failure_rate": _yesterday_failure_rate(prev_trade_date, ll_prev_df),
        "yesterday_winners_today": _yesterday_winners_today(
            prev_trade_date, ll_prev_df, daily_t_df
        ),
    }, errors


def _max_height(step: dict[str, int]) -> int:
    if not step:
        return 0
    keys: list[int] = []
    for k in step:
        try:
            keys.append(int(k))
        except (TypeError, ValueError):
            continue
    return max(keys) if keys else 0


def _limit_step_trend(today: dict[str, int], prev: dict[str, int]) -> dict[str, Any]:
    today_max = _max_height(today)
    prev_max = _max_height(prev)
    today_total = sum(today.values())
    prev_total = sum(prev.values())
    high_delta = today_max - prev_max
    total_delta = today_total - prev_total
    if high_delta > 0 and total_delta > 0:
        interp = "spectrum_lifting"
    elif high_delta < 0 or total_delta < -10:
        interp = "spectrum_collapsing"
    else:
        interp = "stable"
    return {
        "max_height": today_max,
        "max_height_prev": prev_max,
        "high_board_delta": high_delta,
        "total_limit_up_delta": total_delta,
        "interpretation": interp,
    }


def _yesterday_failure_rate(
    prev_trade_date: str, ll_prev_df: pd.DataFrame | None
) -> dict[str, Any]:
    if ll_prev_df is None or ll_prev_df.empty or "limit" not in ll_prev_df.columns:
        return {
            "trade_date_prev": prev_trade_date,
            "u_count": 0,
            "z_count": 0,
            "rate_pct": None,
            "interpretation": None,
        }
    u = int((ll_prev_df["limit"] == "U").sum())
    z = int((ll_prev_df["limit"] == "Z").sum())
    total = u + z
    rate = round(z / total * 100, 2) if total > 0 else None
    if rate is None:
        interp: str | None = None
    elif rate >= 25:
        interp = "high"
    elif rate <= 10:
        interp = "low"
    else:
        interp = "moderate"
    return {
        "trade_date_prev": prev_trade_date,
        "u_count": u,
        "z_count": z,
        "rate_pct": rate,
        "interpretation": interp,
    }


def _yesterday_winners_today(
    prev_trade_date: str,
    ll_prev_df: pd.DataFrame | None,
    daily_t_df: pd.DataFrame | None,
) -> dict[str, Any]:
    if ll_prev_df is None or ll_prev_df.empty or "limit" not in ll_prev_df.columns:
        return {
            "trade_date_prev": prev_trade_date,
            "n_winners": 0,
            "n_continued_today": 0,
            "continuation_rate_pct": None,
            "n_negative_today": 0,
            "avg_pct_chg_today": None,
            "interpretation": None,
        }
    winners = ll_prev_df[ll_prev_df["limit"] == "U"]
    n_winners = int(len(winners))
    if n_winners == 0 or daily_t_df is None or daily_t_df.empty:
        return {
            "trade_date_prev": prev_trade_date,
            "n_winners": n_winners,
            "n_continued_today": 0,
            "continuation_rate_pct": None,
            "n_negative_today": 0,
            "avg_pct_chg_today": None,
            "interpretation": None,
        }
    winner_codes = set(winners["ts_code"].astype(str))
    today_rows = daily_t_df[daily_t_df["ts_code"].astype(str).isin(winner_codes)]
    if today_rows.empty:
        return {
            "trade_date_prev": prev_trade_date,
            "n_winners": n_winners,
            "n_continued_today": 0,
            "continuation_rate_pct": None,
            "n_negative_today": 0,
            "avg_pct_chg_today": None,
            "interpretation": None,
        }
    pct = today_rows["pct_chg"].dropna()
    n_continued = int((pct >= 9.8).sum())
    n_negative = int((pct < -2).sum())
    avg = round(float(pct.mean()), 2) if not pct.empty else None
    cont_rate = round(n_continued / n_winners * 100, 2) if n_winners > 0 else None

    if cont_rate is None or avg is None:
        interp: str | None = None
    elif cont_rate >= 50 and avg >= 3:
        interp = "strong_money_effect"
    elif cont_rate <= 25 or avg <= 0:
        interp = "weak_money_effect"
    else:
        interp = "neutral"

    return {
        "trade_date_prev": prev_trade_date,
        "n_winners": n_winners,
        "n_continued_today": n_continued,
        "continuation_rate_pct": cont_rate,
        "n_negative_today": n_negative,
        "avg_pct_chg_today": avg,
        "interpretation": interp,
    }


def _summarize_limit_step(step_df: pd.DataFrame) -> dict[str, int]:
    """Convert limit_step rows to a {board_height: count} mapping."""
    if step_df is None or step_df.empty:
        return {}
    if "nums" not in step_df.columns:
        return {}
    counts = step_df.groupby("nums").size().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# A1 derived factors (Phase A — pure compute, no new APIs)
# ---------------------------------------------------------------------------


def _amplitude_pct(daily_t_row: dict[str, Any] | None) -> float | None:
    if not daily_t_row:
        return None
    high = _to_float(daily_t_row.get("high"))
    low = _to_float(daily_t_row.get("low"))
    pre_close = _to_float(daily_t_row.get("pre_close"))
    if high is None or low is None or not pre_close:
        return None
    return round((high - low) / pre_close * 100, 2)


def _fd_amount_ratio(fd_amount: float | None, amount: float | None) -> float | None:
    fd = _to_float(fd_amount)
    amt = _to_float(amount)
    if fd is None or not amt:
        return None
    return round(fd / amt * 100, 2)


def _to_float(v: Any) -> float | None:
    if v is None or pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ma_metrics(closes: list[float]) -> dict[str, float | bool | None]:
    """Compute ma5/ma10/ma20 + ma_bull_aligned from a trailing-close list
    (ascending by date, last element = T-day close).
    Returns null for any window that has insufficient history."""
    out: dict[str, float | bool | None] = {
        "ma5": None,
        "ma10": None,
        "ma20": None,
        "ma_bull_aligned": None,
    }
    if not closes:
        return out

    def _ma(window: int) -> float | None:
        if len(closes) < window:
            return None
        return round(sum(closes[-window:]) / window, 2)

    out["ma5"] = _ma(5)
    out["ma10"] = _ma(10)
    out["ma20"] = _ma(20)
    if all(out[k] is not None for k in ("ma5", "ma10", "ma20")):
        latest = closes[-1]
        out["ma_bull_aligned"] = bool(
            latest > out["ma5"] > out["ma10"] > out["ma20"]  # type: ignore[operator]
        )
    return out


def _up_count_30d(d_hist: list[dict[str, Any]]) -> int | None:
    """Count of trade days in the last 30 with pct_chg ≥ 9.8 (10cm main board)."""
    if len(d_hist) < 30:
        return None
    recent = d_hist[-30:]
    return sum(1 for r in recent if (r.get("pct_chg") or 0) >= 9.8)


def _trailing_closes(d_hist: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for r in d_hist:
        c = r.get("close")
        if c is None or pd.isna(c):
            continue
        out.append(float(c))
    return out


def _build_cyq_lookup(cyq_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """Per-ts_code dict of derived chip-distribution fields."""
    out: dict[str, dict[str, Any]] = {}
    if cyq_df is None or cyq_df.empty or "ts_code" not in cyq_df.columns:
        return out
    for row in cyq_df.itertuples(index=False):
        ts = str(row.ts_code)
        weight_avg = _to_float(getattr(row, "weight_avg", None))
        winner_rate = _to_float(getattr(row, "winner_rate", None))
        cost_5 = _to_float(getattr(row, "cost_5pct", None))
        cost_95 = _to_float(getattr(row, "cost_95pct", None))
        out[ts] = {
            "cyq_winner_pct": round(winner_rate, 2) if winner_rate is not None else None,
            "cyq_avg_cost_yuan": round(weight_avg, 2) if weight_avg is not None else None,
            "cyq_top10_concentration": _cyq_concentration(cost_5, cost_95, weight_avg),
        }
    return out


def _cyq_concentration(
    cost_5: float | None, cost_95: float | None, weight_avg: float | None
) -> float | None:
    """Concentration score in [0, 100]; higher = chips more clustered around weight_avg.

    Definition: 100 − (cost_95pct − cost_5pct) / weight_avg × 100.
    A 90% chip-price spread of 30% of weight_avg yields concentration = 70.
    """
    if cost_5 is None or cost_95 is None or not weight_avg:
        return None
    spread_pct = (cost_95 - cost_5) / weight_avg * 100
    return round(max(0.0, min(100.0, 100.0 - spread_pct)), 2)


def _close_to_avg_cost_pct(
    close: float | None, weight_avg: float | None
) -> float | None:
    if close is None or not weight_avg:
        return None
    return round((close - weight_avg) / weight_avg * 100, 2)


def _famous_seats_hits(seats: list[str]) -> list[str]:
    """Return de-duplicated exalter strings whose substring matches any
    famous-seat hint (case-insensitive)."""
    out: list[str] = []
    seen: set[str] = set()
    hints_lower = tuple(h.lower() for h in FAMOUS_SEATS_HINTS)
    for s in seats:
        if not isinstance(s, str) or s in seen:
            continue
        sl = s.lower()
        if any(h in sl for h in hints_lower):
            out.append(s)
            seen.add(s)
    return out


def _build_lhb_rollup(
    top_list_df: pd.DataFrame | None,
    top_inst_df: pd.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    """Roll up top_list / top_inst into per-ts_code lhb_* fields.

    Returns ``{ts_code: {lhb_net_buy_yi, lhb_inst_count, lhb_famous_seats}}``.
    Candidates absent from this map → 未上榜（lhb_* = null in their row）。
    """
    rollup: dict[str, dict[str, Any]] = {}

    if top_list_df is not None and not top_list_df.empty and "ts_code" in top_list_df.columns:
        for row in top_list_df.itertuples(index=False):
            ts = str(row.ts_code)
            net = normalize_to_yi("net_amount", getattr(row, "net_amount", None))
            rollup.setdefault(ts, {})["lhb_net_buy_yi"] = net

    if top_inst_df is not None and not top_inst_df.empty and "ts_code" in top_inst_df.columns:
        for ts, group in top_inst_df.groupby("ts_code"):
            ts_str = str(ts)
            seats = [str(e) for e in group["exalter"].tolist()] if "exalter" in group.columns else []
            entry = rollup.setdefault(ts_str, {})
            entry["lhb_inst_count"] = int(len(set(seats)))
            entry["lhb_famous_seats"] = _famous_seats_hits(seats)

    return rollup


def _build_candidate_rows(
    candidates_df: pd.DataFrame,
    ths_df: pd.DataFrame | None,
    *,
    daily_df: pd.DataFrame | None = None,
    daily_basic_df: pd.DataFrame | None = None,
    moneyflow_df: pd.DataFrame | None = None,
    top_list_df: pd.DataFrame | None = None,
    top_inst_df: pd.DataFrame | None = None,
    cyq_perf_df: pd.DataFrame | None = None,
    daily_lookback: int = 30,
    moneyflow_lookback: int = 5,
) -> list[dict[str, Any]]:
    """Project candidates to a list of dicts with raw + normalized fields + history.

    B1.2 additions:
        prev_daily        — last N daily rows: [(date, close, pct_chg, vol), ...]
        prev_moneyflow    — last N moneyflow rows: [(date, net_mf_yi, buy_lg_yi, buy_elg_yi)]
        turnover_rate, volume_ratio, circ_mv_yi   — from daily_basic on T

    All numeric fields go through normalize_to_yi/wan with FIELD_UNITS_RAW for
    correct unit conversion (B3.1 / M6 fix).
    """
    if ths_df is not None and not ths_df.empty:
        ths_lookup = ths_df.set_index("ts_code").to_dict(orient="index")
    else:
        ths_lookup = {}

    daily_by_code = _index_by_code(daily_df)
    daily_basic_by_code = _index_by_code(daily_basic_df)
    moneyflow_by_code = _index_by_code(moneyflow_df)
    lhb_rollup = _build_lhb_rollup(top_list_df, top_inst_df)
    cyq_lookup = _build_cyq_lookup(cyq_perf_df)

    out: list[dict[str, Any]] = []
    for row in candidates_df.itertuples(index=False):
        ts_code = str(row.ts_code)
        fd_amount_raw = getattr(row, "fd_amount", None)
        amount_raw = getattr(row, "amount", None)
        rec = {
            "candidate_id": ts_code,
            "ts_code": ts_code,
            "name": getattr(row, "name", None),
            "industry": getattr(row, "industry_basic", None) or getattr(row, "industry", None),
            "first_time": getattr(row, "first_time", None),
            "last_time": getattr(row, "last_time", None),
            "open_times": _opt_int(getattr(row, "open_times", None)),
            "limit_times": _opt_int(getattr(row, "limit_times", None)),
            "up_stat": getattr(row, "up_stat", None),
            "pct_chg": round2(getattr(row, "pct_chg", None)),
            "close_yuan": round2(getattr(row, "close", None)),
            "turnover_ratio": round2(getattr(row, "turnover_ratio", None)),
            "fd_amount_yi": normalize_to_yi("fd_amount", fd_amount_raw),
            "limit_amount_yi": normalize_to_yi("limit_amount", getattr(row, "limit_amount", None)),
            "amount_yi": normalize_to_yi("amount", amount_raw),
            "total_mv_yi": normalize_to_yi("total_mv", getattr(row, "total_mv", None)),
            "float_mv_yi": normalize_to_yi("float_mv", getattr(row, "float_mv", None)),
            "fd_amount_ratio": _fd_amount_ratio(fd_amount_raw, amount_raw),
        }
        ths = ths_lookup.get(ts_code)
        if ths is not None:
            rec["lu_desc"] = ths.get("lu_desc")
            rec["tag"] = ths.get("tag")
            rec["limit_up_suc_rate"] = round2(ths.get("limit_up_suc_rate"))
            rec["free_float_yi"] = normalize_to_yi("free_float", ths.get("free_float"))

        # B1.2 history attachments
        d_hist = daily_by_code.get(ts_code, [])
        if d_hist:
            rec["prev_daily"] = [
                {
                    "date": r.get("trade_date"),
                    "close": round2(r.get("close")),
                    "pct_chg": round2(r.get("pct_chg")),
                    "vol": _opt_int(r.get("vol")),
                }
                for r in d_hist[-daily_lookback:]
            ]
            rec["amplitude_pct"] = _amplitude_pct(d_hist[-1])
            rec.update(_ma_metrics(_trailing_closes(d_hist)))
            rec["up_count_30d"] = _up_count_30d(d_hist)
        else:
            rec["amplitude_pct"] = None
            rec["ma5"] = rec["ma10"] = rec["ma20"] = None
            rec["ma_bull_aligned"] = None
            rec["up_count_30d"] = None
        db_hist = daily_basic_by_code.get(ts_code, [])
        if db_hist:
            latest = db_hist[-1]
            rec["turnover_rate"] = round2(latest.get("turnover_rate"))
            rec["volume_ratio"] = round2(latest.get("volume_ratio"))
            rec["circ_mv_yi"] = normalize_to_yi("circ_mv", latest.get("circ_mv"))
        mf_hist = moneyflow_by_code.get(ts_code, [])
        if mf_hist:
            rec["prev_moneyflow"] = [
                {
                    "date": r.get("trade_date"),
                    "net_mf_yi": normalize_to_yi("net_mf_amount", r.get("net_mf_amount")),
                    "buy_lg_yi": normalize_to_yi("buy_lg_amount", r.get("buy_lg_amount")),
                    "buy_elg_yi": normalize_to_yi("buy_elg_amount", r.get("buy_elg_amount")),
                }
                for r in mf_hist[-moneyflow_lookback:]
            ]
        # B1 LHB roll-up — null when candidate didn't make the day's top_list
        # (合法事实，不进 missing_data，由 LLM 通过 null 判断"未上榜")
        lhb = lhb_rollup.get(ts_code, {})
        rec["lhb_net_buy_yi"] = lhb.get("lhb_net_buy_yi")
        rec["lhb_inst_count"] = lhb.get("lhb_inst_count")
        rec["lhb_famous_seats"] = lhb.get("lhb_famous_seats") or []
        # B2 cyq_perf — null when no row for this ts_code (LLM puts cyq_* in
        # candidate.missing_data via the standard prompt rule)
        cyq = cyq_lookup.get(ts_code, {})
        rec["cyq_winner_pct"] = cyq.get("cyq_winner_pct")
        rec["cyq_top10_concentration"] = cyq.get("cyq_top10_concentration")
        rec["cyq_avg_cost_yuan"] = cyq.get("cyq_avg_cost_yuan")
        rec["cyq_close_to_avg_cost_pct"] = _close_to_avg_cost_pct(
            _to_float(getattr(row, "close", None)),
            cyq.get("cyq_avg_cost_yuan"),
        )
        out.append(rec)
    return out


def _index_by_code(df: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    """Group a DataFrame by ts_code into ascending-by-trade_date row lists."""
    if df is None or df.empty or "ts_code" not in df.columns:
        return {}
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date")
    out: dict[str, list[dict[str, Any]]] = {}
    for code, group in df.groupby("ts_code"):
        out[str(code)] = group.to_dict(orient="records")
    return out


def _opt_int(v: Any) -> int | None:
    if v is None or pd.isna(v):
        return None
    return int(v)


# ---------------------------------------------------------------------------
# v0.5 — LGB scoring attachment (PR-2.2; lightgbm_design.md §7.2)
# ---------------------------------------------------------------------------


def _attach_lgb_scores(
    bundle: Round1Bundle,
    *,
    candidates_df: pd.DataFrame,
    daily_df: pd.DataFrame | None,
    daily_basic_df: pd.DataFrame | None,
    moneyflow_df: pd.DataFrame | None,
    top_list_df: pd.DataFrame | None,
    top_inst_df: pd.DataFrame | None,
    cyq_perf_df: pd.DataFrame | None,
    scorer: LgbScorer | None,
) -> None:
    """Inject ``lgb_score`` / ``lgb_decile`` / ``lgb_feature_missing`` per candidate.

    * Scorer ``None`` or ``loaded=False`` → every candidate gets ``lgb_score=None``,
      ``lgb_decile=None``, ``lgb_feature_missing=[]``; ``bundle.lgb_model_id``
      stays ``None`` and ``bundle.data_unavailable`` is annotated with the
      ``lgb_model (…)`` reason from the scorer.
    * Any exception inside this path is logged and degrades to the "未启用"
      branch above—LGB must never block R1/R2 (设计 §7.3 红线)。

    The actual booster math + per-row diagnostics live in :class:`LgbScorer`;
    this function only marshals data between the strategy pipeline and the
    scorer, and decides how the model output is exposed to the LLM.
    """
    # Helper: write the "disabled" / "failed" fallback values into every candidate.
    def _fill_disabled(reason: str | None) -> None:
        for rec in bundle.candidates:
            rec.setdefault("lgb_score", None)
            rec.setdefault("lgb_decile", None)
            rec.setdefault("lgb_feature_missing", [])
        if reason:
            bundle.data_unavailable.append(f"lgb_model ({reason})")

    if scorer is None:
        _fill_disabled(None)  # user --no-lgb or framework opted out entirely
        return
    if not bundle.candidates:
        return

    # Lazy-load the booster on first call. The scorer swallows errors and
    # exposes them via ``load_error`` — we surface that to data_unavailable.
    try:
        scorer.warmup()
    except Exception as e:  # noqa: BLE001 — defensive, scorer should never raise
        logger.warning("LgbScorer.warmup raised unexpectedly: %s", e)
        _fill_disabled(f"warmup_raised: {type(e).__name__}")
        return

    if not scorer.loaded:
        _fill_disabled(scorer.load_error or "unloaded")
        return

    # Build the feature matrix from the same intermediate frames _build_candidate_rows
    # consumed. We re-derive the lookups (cheap groupby) so this stays a self-contained
    # path with no extra arguments threaded through _build_candidate_rows.
    try:
        from .lgb.features import build_feature_frame  # noqa: PLC0415
        from .lgb.scorer import attach_deciles  # noqa: PLC0415

        daily_by_code = _index_by_code(daily_df)
        daily_basic_by_code = _index_by_code(daily_basic_df)
        moneyflow_by_code = _index_by_code(moneyflow_df)
        lhb_rollup = _build_lhb_rollup(top_list_df, top_inst_df)
        cyq_lookup = _build_cyq_lookup(cyq_perf_df)

        feature_df = build_feature_frame(
            candidates_df=candidates_df,
            daily_by_code=daily_by_code,
            daily_basic_by_code=daily_basic_by_code,
            moneyflow_by_code=moneyflow_by_code,
            cyq_by_code=cyq_lookup,
            lhb_rollup=lhb_rollup,
            sector_strength=bundle.sector_strength,
            market_summary=bundle.market_summary,
            trade_date=bundle.trade_date,
        )
    except Exception as e:  # noqa: BLE001 — feature build must not crash the run
        logger.warning("build_feature_frame failed for LGB scoring: %s", e)
        _fill_disabled(f"feature_build_failed: {type(e).__name__}")
        return

    try:
        scored = scorer.score_batch(feature_df)
    except Exception as e:  # noqa: BLE001 — score_batch should not raise but be defensive
        logger.warning("score_batch raised unexpectedly: %s", e)
        _fill_disabled(f"score_raised: {type(e).__name__}")
        return

    # 计算 decile（< 10 个候选 → 全 NaN）
    deciles = attach_deciles(scored, n_buckets=10)

    bundle.lgb_model_id = scorer.model_id
    audit_rows: list[dict[str, Any]] = []
    score_lookup: dict[str, dict[str, Any]] = {}
    for ts_code in scored.index:
        ts = str(ts_code)
        row = scored.loc[ts_code]
        raw_score = row["lgb_score"]
        if pd.isna(raw_score):
            score_lookup[ts] = {"lgb_score": None, "lgb_decile": None, "missing": []}
            continue
        decile = deciles.loc[ts_code] if ts_code in deciles.index else None
        try:
            missing = json.loads(row["feature_missing_json"]) if row["feature_missing_json"] else []
        except (TypeError, ValueError):
            missing = []
        # Design §7.2: 报告 / candidate dict 展示 0–100 浮点（× 100 + round(_, 1)）
        display_score = round(float(raw_score) * 100.0, 1)
        score_lookup[ts] = {
            "lgb_score": display_score,
            "lgb_decile": (int(decile) if pd.notna(decile) else None),
            "missing": missing,
        }
        audit_rows.append(
            {
                "ts_code": ts,
                "lgb_score": float(raw_score),  # raw booster output ∈ [0,1] for audit
                "lgb_decile": (int(decile) if pd.notna(decile) else None),
                "feature_hash": str(row["feature_hash"]),
                "feature_missing_json": str(row["feature_missing_json"]),
            }
        )

    for rec in bundle.candidates:
        ts = rec.get("ts_code")
        info = score_lookup.get(ts) if ts else None
        if info is None:
            rec["lgb_score"] = None
            rec["lgb_decile"] = None
            rec["lgb_feature_missing"] = []
        else:
            rec["lgb_score"] = info["lgb_score"]
            rec["lgb_decile"] = info["lgb_decile"]
            rec["lgb_feature_missing"] = info["missing"]

    bundle.lgb_predictions = audit_rows


# ---------------------------------------------------------------------------
# v0.5 — public aliases for lgb.dataset reuse (lightgbm_iteration_plan.md PR-1.2)
# ---------------------------------------------------------------------------
#
# The underscore-prefixed helpers above were originally module-internal; the
# LightGBM training pipeline needs to call them from ``limit_up_board.lgb.dataset``.
# We expose public names without renaming the originals so that existing tests
# (which import the underscore names) continue to work unchanged.

apply_market_filter = _apply_market_filter
build_lhb_rollup = _build_lhb_rollup
build_cyq_lookup = _build_cyq_lookup
index_by_code = _index_by_code
build_candidate_rows = _build_candidate_rows
summarize_limit_step = _summarize_limit_step
fetch_history_window = _fetch_history_window
try_optional = _try_optional
shift_date = _shift_date
