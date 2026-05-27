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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from deeptrade.core.tushare_client import (
    TushareClient,
    TushareUnauthorizedError,
)

from .calendar import TradeCalendar

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api import ConceptRepository

    from .lgb.scorer import LgbScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 0 — resolve trade date
# ---------------------------------------------------------------------------


# Anchor for the latest-published-trade-date probe. The Shanghai Composite
# has been published every trading day since the API launched and is therefore
# the safest market-level signal for "what's the most recent trade day Tushare
# has data for".
LATEST_TRADE_DATE_PROBE_INDEX = "000001.SH"


def fetch_latest_trade_date(
    tushare: TushareClient,
    *,
    index_code: str = LATEST_TRADE_DATE_PROBE_INDEX,
    lookback_days: int = 60,
) -> str:
    """Return the most recent published trade_date according to ``index_daily``.

    The strategy used to derive T from ``datetime.now()`` + ``trade_cal``;
    that broke whenever the machine's clock or timezone was off. Now T is
    sourced from market data instead — Tushare's ``index_daily`` is published
    on each trading day's close, so its ``max(trade_date)`` is the authoritative
    "latest available trade day" regardless of local time.

    The local clock is still consulted to bound the query window
    (``[now-lookback_days, now+1d]``) — but only as a quota-friendly window
    bound. As long as the clock is within ``lookback_days`` of reality the
    probe returns the true latest trade_date; grosser skew falls out of the
    window and raises here, which is the correct failure mode (much better
    than silently anchoring T to a wrong "today").

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


_CANDIDATE_SORT_KEYS: tuple[str, ...] = (
    "trade_date",
    "first_time",
    "limit_times",
    "fd_amount",
    "ts_code",
)
_CANDIDATE_SORT_ASCENDING: tuple[bool, ...] = (
    True,   # trade_date asc — usually constant within a single run
    True,   # first_time asc — earliest 封板时间 first
    False,  # limit_times desc — 连板高度优先
    False,  # fd_amount desc — 封单金额优先
    True,   # ts_code asc — final tie-breaker
)


def _stable_sort_candidates_df(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """P1-B: stable sort the merged candidate DataFrame.

    Keys: ``trade_date asc, first_time asc, limit_times desc, fd_amount desc,
    ts_code asc``. Mergesort + ``na_position='last'`` ensure rows with null
    ``first_time`` are pushed to the bottom rather than reshuffled. Missing
    columns degrade to ``ts_code`` only (defensive — collect_round1 always
    merges with limit_list_d so all five keys should exist, but this keeps
    unit tests with stub DataFrames working).

    Sorting changes prompt input order; it does NOT change which candidates
    survive filtering (those are decided by ``_apply_market_filter`` etc.).
    """
    available_keys: list[str] = []
    available_asc: list[bool] = []
    for key, asc in zip(_CANDIDATE_SORT_KEYS, _CANDIDATE_SORT_ASCENDING, strict=True):
        if key in candidates_df.columns:
            available_keys.append(key)
            available_asc.append(asc)
    if not available_keys:
        return candidates_df.reset_index(drop=True)
    return candidates_df.sort_values(
        by=available_keys,
        ascending=available_asc,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _apply_market_filter(
    candidates_df: pd.DataFrame,
    *,
    max_float_mv_yi: float,
    max_close_yuan: float,
    min_float_mv_yi: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """v0.6.4 (P2-1) — 闭区间筛选：
    ``min_float_mv_yi <= 流通市值(亿) <= max_float_mv_yi`` AND ``close <= max_close_yuan``。

    边界从开 (``> < <``) 改为闭 (``>= <= <=``) —— 用户在 settings 里写
    ``max=100`` 时，100 亿的标的现在会通过；同理 ``max_close_yuan=15`` 时 15 元的
    标的也会通过。Null 仍然被剔除（保守，无法验证"小市值/低价"声明）。

    Returns ``(filtered_df, summary)``。除常规 before/after，还在
    ``summary["dropped"]`` 写入「**全部**被剔除标的的 ts_code + 剔除原因」
    （按流通市值降序），便于 render 报告 / summary.json 完整展示，而非只看前几只。
    """
    n_before = int(len(candidates_df))
    summary: dict[str, Any] = {
        "before": n_before,
        "after": n_before,
        "min_float_mv_yi": min_float_mv_yi,
        "max_float_mv_yi": max_float_mv_yi,
        "max_close_yuan": max_close_yuan,
        "dropped": [],
    }
    if n_before == 0:
        return candidates_df, summary
    fm_yi = pd.to_numeric(candidates_df.get("float_mv"), errors="coerce") / 1e8
    cl = pd.to_numeric(candidates_df.get("close"), errors="coerce")
    mask = (
        fm_yi.notna()
        & cl.notna()
        & (fm_yi >= min_float_mv_yi)
        & (fm_yi <= max_float_mv_yi)
        & (cl <= max_close_yuan)
    )
    filtered = candidates_df[mask].reset_index(drop=True)
    summary["after"] = int(len(filtered))

    # P2-1 / v0.16.3: 把**全部**被剔除标的（按 float_mv 降序）连同原因写进 summary，
    # 让 render 报告 / summary.json 能完整展示"为何排除"，而不再截断为前 3 只。
    # 空 / 全保留场景下保持空 list。
    dropped_df = candidates_df[~mask]
    if not dropped_df.empty:
        # ⚠ Bug fix (v0.16.3 regression): must sort ONLY the dropped rows.
        # ``fm_yi.where(~mask)`` keeps all rows (NaN-ing the passed ones) and
        # ``sort_values(na_position="last")`` does NOT drop those NaN rows — it
        # returns the full index, so the loop below also walked the *passed*
        # candidates, tagging them with reason ["unknown"]. Restrict to
        # ``dropped_df.index`` (genuine float_mv_null drops keep NaN → sort last).
        dropped_fm = fm_yi.loc[dropped_df.index]
        # 排序键：先按 float_mv_yi 降序；NaN 排到最后保证有数值的优先。
        ordered_idx = dropped_fm.sort_values(ascending=False, na_position="last").index
        dropped_items: list[dict[str, Any]] = []
        for idx in ordered_idx:
            row = candidates_df.loc[idx]
            mv_val = float(fm_yi.loc[idx]) if pd.notna(fm_yi.loc[idx]) else None
            close_val = float(cl.loc[idx]) if pd.notna(cl.loc[idx]) else None
            reasons: list[str] = []
            if mv_val is None:
                reasons.append("float_mv_null")
            else:
                if mv_val < min_float_mv_yi:
                    reasons.append(f"float_mv<{min_float_mv_yi}")
                if mv_val > max_float_mv_yi:
                    reasons.append(f"float_mv>{max_float_mv_yi}")
            if close_val is None:
                reasons.append("close_null")
            elif close_val > max_close_yuan:
                reasons.append(f"close>{max_close_yuan}")
            dropped_items.append(
                {
                    "ts_code": str(row.get("ts_code", "")),
                    "name": row.get("name"),
                    "float_mv_yi": round(mv_val, 2) if mv_val is not None else None,
                    "close_yuan": round(close_val, 2) if close_val is not None else None,
                    "reasons": reasons or ["unknown"],
                }
            )
        summary["dropped"] = dropped_items
    return filtered, summary


# ---------------------------------------------------------------------------
# Sector strength resolver — three-tier fallback (F2 fix + §11.3)
# ---------------------------------------------------------------------------


SectorStrengthSource = Literal["limit_cpt_list", "unavailable"]


@dataclass
class SectorStrength:
    """Sector heat / leadership data fed into the prompt.

    `source` is exposed verbatim to the LLM via ``sector_strength_source`` so
    the model can downweight confidence when it sees an ``unavailable`` label.

    v0.16.0 — 简化为只接受 ``limit_cpt_list``（官方概念涨停统计）；插件本地
    基于 ``lu_desc`` 或 ``stock_basic.industry`` 的兜底聚合已移除（其权威性
    远不及 Tushare 官方排名，且 LLM 拿到也很难真正用上）。题材归属信息已经
    由 ``ConceptRepository`` 在 candidate 行级别（concepts / industries / regions）
    全量暴露，板块强度只保留全局热度这一维度。
    """

    source: SectorStrengthSource
    data: dict[str, Any]


def resolve_sector_strength(
    *,
    limit_cpt_list: pd.DataFrame | None,
) -> SectorStrength:
    """Return SectorStrength from Tushare ``limit_cpt_list``.

    v0.16.0 起仅保留 Tushare 官方源；``limit_cpt_list`` 缺失时返回
    ``source="unavailable"`` 让 LLM 知道这次没有全市场板块热度排名可参考。
    """
    if limit_cpt_list is not None and not limit_cpt_list.empty:
        top = limit_cpt_list.sort_values("rank").head(10)
        return SectorStrength(
            source="limit_cpt_list",
            data={"top_sectors": top.to_dict(orient="records")},
        )
    return SectorStrength(source="unavailable", data={"top_sectors": []})


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
# anonymity per DESIGN §12 辩论修订 spirit).
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
    """Everything the 强势初筛 LLM stage needs.

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
        default_factory=lambda: SectorStrength(source="unavailable", data={"top_sectors": []})
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
    min_float_mv_yi: float = 0.0,
    force_sync: bool = False,
    lgb_scorer: LgbScorer | None = None,
    concept_repo: ConceptRepository | None = None,
) -> Round1Bundle:
    """Assemble the 强势初筛 input bundle.

    The flow:
        1. stock_basic (static) → main_board_filter()
        2. limit_list_d(T, limit='U') → join main_board → DROP if 0 candidates
           (zero candidates is a LEGAL outcome — S4)
        2b. v0.4 — drop candidates whose 流通市值 ≤ ``min_float_mv_yi``
            or ≥ ``max_float_mv_yi``, or 当前股价 ≥ ``max_close_yuan``;
            null in either field → drop (conservative; thresholds owned by
            ``LubConfig``).
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

    # Market-wide breadth = every 'U' limit-up that day, captured BEFORE the
    # main-board / 市值 / 价格 / ST / 停牌 filters below. This is what
    # ``limit_up_count`` (report 市场快照 + LGB ``f_mkt_total_limit_up``) must
    # reflect; using ``len(candidates_df)`` (post-filter) understated breadth by
    # >2× and made the market look far weaker than it was.
    market_limit_up_count = int(len(limit_list_d))

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
    # P1-B: stable sort — Tushare merge order is implementation-defined and can
    # change across cache hits / force_sync runs, which leaks into Prompt input
    # ordering. Sort by business priority (first_time asc, limit_times desc,
    # fd_amount desc) with ts_code as final tie-breaker. Mergesort is required
    # — pandas defaults to quicksort which is NOT stable when NaN is present.
    candidates_df = _stable_sort_candidates_df(candidates_df)

    # 2b. v0.4 — 流通市值 / 股价上限筛选（null → 过滤）。
    candidates_df, market_filter_summary = _apply_market_filter(
        candidates_df,
        max_float_mv_yi=max_float_mv_yi,
        max_close_yuan=max_close_yuan,
        min_float_mv_yi=min_float_mv_yi,
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

    # P1-3: required API 全表为空 → 不一定意味着 API 不可用，但「全市场无龙虎榜个股」
    # 或「全市场无 cyq 数据」都是值得在报告/事件流里提示用户的情景。把空响应写入
    # data_unavailable，由 LLM 与下游 _build_lhb_rollup 共同识别成 api_empty 状态。
    top_list_empty = top_list_df.empty
    top_inst_empty = top_inst_df.empty
    cyq_perf_empty = cyq_perf_df.empty
    if top_list_empty:
        data_unavailable.append("top_list_empty_response")
    if top_inst_empty:
        data_unavailable.append("top_inst_empty_response")
    if cyq_perf_empty:
        data_unavailable.append("cyq_perf_empty_response")

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

    sector = resolve_sector_strength(limit_cpt_list=cpt_df)
    bundle.sector_strength = sector

    # 6. limit_step (required) — for global ladder distribution
    step_df = tushare.call("limit_step", trade_date=trade_date, force_sync=force_sync)
    today_step = _summarize_limit_step(step_df)
    # update() (not reassign) to preserve candidate_filter_summary set in step 2b.
    bundle.market_summary.update(
        {
            "limit_up_count": market_limit_up_count,
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
    # ``daily`` feeds ma5/ma10/ma20 + up_count_30d, the most determinism-sensitive
    # features (a single missing trailing day shifts every MA window). Retry empty
    # days so the immutable cache is repopulated and reruns are stable; surface any
    # residual gap via data_unavailable (see _fetch_history_window docstring).
    daily_df, daily_missing = _fetch_history_window(
        tushare,
        "daily",
        start_date,
        trade_date,
        candidate_codes,
        force_sync=force_sync,
        retry_empty_days=True,
    )
    daily_basic_df, _ = _fetch_history_window(
        tushare,
        "daily_basic",
        start_date,
        trade_date,
        candidate_codes,
        force_sync=force_sync,
    )
    mf_start = _shift_date(trade_date, -(moneyflow_lookback + 5))
    moneyflow_df, _ = _fetch_history_window(
        tushare,
        "moneyflow",
        mf_start,
        trade_date,
        candidate_codes,
        force_sync=force_sync,
    )
    if daily_missing:
        # Loud + deterministic: the trailing-close window has a hole. After the
        # force_sync retry this should be rare (genuinely unpublished days only),
        # but when it happens the MAs/动量 features for the affected window are
        # unreliable — make it visible rather than letting the prompt drift silently.
        data_unavailable.append(
            "daily_window_gap: 日线历史窗口缺失交易日 "
            f"{','.join(daily_missing)}（force_sync 重取后仍空；"
            "均线/动量特征可能在不同运行间漂移）"
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
        # P1-3 — propagate "整体空" 信号到每个 candidate 的 lhb_data_quality 三态
        lhb_api_empty=top_list_empty and top_inst_empty,
        concept_repo=concept_repo,
    )
    # Data-quality guardrail — the prompt deliberately treats per-row null
    # ma5/up_count_30d/... as "legal facts" (occasional new-listing gaps), so a
    # SYSTEMATIC absence of multi-day history (e.g. a broken/truncated daily
    # window) would otherwise be invisible: missing_data stays [], nothing hits
    # data_unavailable, and only lgb_feature_missing in the raw snapshot betrays
    # it. Surface it explicitly so the LLM downweights and the report shows it.
    history_warn = _detect_incomplete_history(bundle.candidates)
    if history_warn:
        data_unavailable.append(history_warn)
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
    retry_empty_days: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch (api_name) over [start_date, end_date]; filter to candidates.

    Returns ``(frame, missing_days)`` where ``missing_days`` lists the calendar
    open-days whose **whole-market** per-date frame came back empty (an anomaly
    for settled days, distinct from a day that simply had no *candidate* rows).

    ⚠ History fetch must loop **per trade-date**, not issue a single all-market
    ``start_date/end_date`` query. Tushare's daily/daily_basic/moneyflow cap a
    single response at ~6000 rows and the framework transport does NOT paginate
    (``TushareSDKTransport.call`` = one SDK call). An all-market window spans
    ~5400 stocks × N days ≫ 6000, so the response silently truncates to the most
    recent ~1 trade-date — leaving every candidate with a single history row and
    nulling the entire 动量 / 5 日比率 feature block (ma5/ma10/ma20/up_count_30d/
    pct_chg_Nd_sum/mf_net_5d_sum/...). Querying by ``trade_date=d`` instead caps
    each call at one market-day (≤ market size < 6000) AND each per-date frame is
    ``trade_day_immutable``, so the cache is shared across runs and training days.

    Determinism (v0.18): a whole-market per-date frame that returns empty is NOT
    cached as ``ok``, so the framework re-fetches it on the next run. A transient
    upstream blip on a *single* open-day therefore silently dropped that day from
    one run but not the next — shifting the trailing ``closes[-N:]`` membership and
    making ma5/ma10/ma20 (and the LGB features / score derived from them, and thus
    the whole LLM prompt) drift across otherwise-identical reruns. ``retry_empty_days``
    re-fetches an empty day once with ``force_sync=True`` to repopulate the immutable
    cache (so subsequent runs hit a frozen row), and any day still empty after the
    retry is reported via ``missing_days`` so the caller can surface it loudly
    instead of letting the gap stay invisible.
    """
    cal = TradeCalendar(tushare.call("trade_cal", force_sync=force_sync))
    open_days = cal.range(start_date, end_date)
    if not open_days:
        return pd.DataFrame(), []
    frames: list[pd.DataFrame] = []
    missing_days: list[str] = []
    for d in open_days:
        df = tushare.call(api_name, trade_date=d, force_sync=force_sync)
        if (df is None or df.empty) and retry_empty_days and not force_sync:
            # Force a fresh upstream fetch so an ``ok`` row lands in the immutable
            # cache; future runs then read a frozen value rather than re-rolling
            # the dice on a transient empty.
            df = tushare.call(api_name, trade_date=d, force_sync=True)
        if df is None or df.empty:
            missing_days.append(d)
            continue
        if "ts_code" in df.columns and candidate_codes:
            df = df[df["ts_code"].astype(str).isin(candidate_codes)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), missing_days
    # _index_by_code re-sorts by (ts_code, trade_date) + dedups, so concat order
    # is irrelevant here — we keep ascending date order anyway for readability.
    return pd.concat(frames, ignore_index=True), missing_days


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


def _detect_incomplete_history(
    candidates: list[dict[str, Any]], *, min_fraction: float = 0.5
) -> str | None:
    """Return a ``data_unavailable`` marker when multi-day history is missing for
    a suspicious majority of candidates, else None.

    ``ma5`` needs ≥5 trailing daily rows. A handful of recent IPOs legitimately
    lack it, but if **most** of the day's limit-up leaders have no 5-day MA the
    daily history window almost certainly under-returned (the failure mode that
    nulled the entire 动量 block). We gate on candidates that DO have a close
    (so genuinely empty/zero-candidate days don't trip it) and require a
    majority, keeping false positives near-zero.
    """
    valued = [c for c in candidates if c.get("close_yuan") is not None]
    n = len(valued)
    if n < 2:
        return None
    missing = sum(1 for c in valued if c.get("ma5") is None)
    if missing / n >= min_fraction and missing >= 2:
        return (
            f"daily_history_incomplete: {missing}/{n} 候选缺失≥5日均线"
            "（疑似 daily 历史窗口拉取不足，动量/多日特征不可靠）"
        )
    return None


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
    famous-seat hint (case-insensitive).

    P1-E: output is sorted by seat name asc so the downstream
    ``lhb_famous_seats_text = "; ".join(...)`` is stable regardless of
    Tushare's row order in ``top_inst``. Previously the join inherited
    insertion order, leaking remote API ordering into the LLM prompt.
    """
    seen: set[str] = set()
    hits: list[str] = []
    hints_lower = tuple(h.lower() for h in FAMOUS_SEATS_HINTS)
    for s in seats:
        if not isinstance(s, str) or s in seen:
            continue
        sl = s.lower()
        if any(h in sl for h in hints_lower):
            hits.append(s)
            seen.add(s)
    hits.sort()
    return hits


def _aggregate_top_list_net(
    top_list_df: pd.DataFrame | None,
    *,
    reasons_text_max_chars: int = 80,
) -> dict[str, dict[str, Any]]:
    """v0.12.4 (P1-3) — per-ts_code aggregation of ``top_list``.

    ``lub_top_list`` 的主键含 ``reason``，同一 ts_code 可能因 \"日涨幅偏离7%\"、
    \"日涨幅偏离7%(成交额过亿)\"、\"机构专用\" 等多个 reason 同时上榜，每行
    各自带一份 ``net_amount``。v0.12.3 及之前用 per-row 循环 ``rollup[ts]['lhb_net_buy_yi'] = net``
    导致后到行覆盖先到行、丢失资金信息。修复后改为 groupby 求和。

    Per ts_code 返回：
        * ``lhb_net_buy_yi``       —— 全部 reason 净买入额之和（亿）
        * ``lhb_reason_count``     —— 该 ts_code 当日上榜原因数
        * ``lhb_reasons_text``     —— 按各 reason 净买入额降序拼接，逗号分隔；
                                      超过 ``reasons_text_max_chars`` 截断并附 "…"
    无 net_amount 列 / 全空时 ``lhb_net_buy_yi`` 为 ``None``（pandas sum() 默认
    会把空组聚成 0；显式判断更稳）。
    """
    if top_list_df is None or top_list_df.empty or "ts_code" not in top_list_df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    has_net = "net_amount" in top_list_df.columns
    has_reason = "reason" in top_list_df.columns
    for ts_raw, group in top_list_df.groupby("ts_code"):
        ts = str(ts_raw)
        entry: dict[str, Any] = {"lhb_reason_count": int(len(group))}
        # 净买入额：先按行 normalize 到亿，再求和；全 None → keep None。
        if has_net:
            per_row = [
                normalize_to_yi("net_amount", v) for v in group["net_amount"].tolist()
            ]
            non_null = [x for x in per_row if x is not None]
            entry["lhb_net_buy_yi"] = round(sum(non_null), 2) if non_null else None
        else:
            per_row = [None] * len(group)
            entry["lhb_net_buy_yi"] = None
        # reasons_text：按当行 net_amount 降序，None 视为 -inf；截断。
        # P1-D: 同净买入额时增加 reason 文本作为 tie-breaker —— 原实现仅按 net_amount
        # 排序，同金额时取决于 Tushare 返回顺序，会让相同输入产出不同 prompt。
        if has_reason:
            pairs: list[tuple[str, float]] = []
            reasons = group["reason"].tolist()
            for r, n in zip(reasons, per_row, strict=True):
                if r is None:
                    continue
                rs = str(r).strip()
                if not rs:
                    continue
                pairs.append((rs, n if n is not None else float("-inf")))
            # 主键：-net_amount（None → +inf 排到最后）；次键：reason 文本字典序升序。
            pairs.sort(key=lambda kv: (
                -kv[1] if kv[1] != float("-inf") else float("inf"),
                kv[0],
            ))
            joined = ", ".join(p[0] for p in pairs)
            if len(joined) > reasons_text_max_chars:
                joined = joined[: reasons_text_max_chars - 1].rstrip(", ") + "…"
            entry["lhb_reasons_text"] = joined or None
        else:
            entry["lhb_reasons_text"] = None
        out[ts] = entry
    return out


def _build_lhb_rollup(
    top_list_df: pd.DataFrame | None,
    top_inst_df: pd.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    """Roll up top_list / top_inst into per-ts_code lhb_* fields.

    Returns ``{ts_code: {lhb_net_buy_yi, lhb_reason_count, lhb_reasons_text,
    lhb_inst_count, lhb_famous_seats}}``. Candidates absent from this map →
    未上榜（lhb_* = null in their row）。
    """
    rollup: dict[str, dict[str, Any]] = {}

    # v0.12.4 (P1-3)：top_list 改为 groupby 聚合；同 ts_code 多 reason 行
    # 不再相互覆盖，net_buy_yi 是各 reason 净买入额之和。
    for ts, entry in _aggregate_top_list_net(top_list_df).items():
        rollup.setdefault(ts, {}).update(entry)

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
    daily_lookback: int = 30,  # noqa: ARG001 — v0.8.0 P1-2 删除 prev_daily 后保留签名以兼容
    moneyflow_lookback: int = 5,  # noqa: ARG001 — v0.8.0 P1-3 同上
    lhb_api_empty: bool = False,
    lhb_api_unavailable: bool = False,
    concept_repo: ConceptRepository | None = None,
) -> list[dict[str, Any]]:
    """Project candidates to a list of dicts with raw + normalized fields + summary derivations.

    v0.8.0 评审响应（LLM 噪音收敛 — P1-2 / P1-3 / P1-1）：
      * 删除 ``prev_daily`` / ``prev_moneyflow`` 原始数组（评审：让 LLM 自行从 30
        日明细归纳趋势的稳定性不如代码侧摘要；同时拉高 token 与注意力噪音）。
      * 新增 prev_daily 派生摘要：``max_upper_shadow_ratio_5d_pct`` 等（评审 P1-2）。
      * 新增资金流派生摘要：``mf_net_5d_sum_yi`` / ``mf_consecutive_positive_days`` /
        ``mf_net_to_amount_pct`` / ``mf_large_order_strength_pct`` /
        ``mf_divergence_flag``（评审 P1-3，与 LGB 同名 feature 同步派生，确保
        prompt 与模型口径一致）。
      * 新增题材内相对地位：``sec_intra_rank_by_limit_times`` /
        ``sec_first_to_limit_flag`` / ``sec_is_height_board`` /
        ``sec_fd_amount_rank_pct``（评审 P1-1）。

    All numeric fields go through normalize_to_yi/wan with FIELD_UNITS_RAW for
    correct unit conversion (B3.1 / M6 fix).
    """
    from .lgb.features import industry_intra_position, max_upper_shadow_ratio_5d  # noqa: PLC0415

    if ths_df is not None and not ths_df.empty:
        ths_lookup = ths_df.set_index("ts_code").to_dict(orient="index")
    else:
        ths_lookup = {}

    daily_by_code = _index_by_code(daily_df)
    daily_basic_by_code = _index_by_code(daily_basic_df)
    moneyflow_by_code = _index_by_code(moneyflow_df)
    lhb_rollup = _build_lhb_rollup(top_list_df, top_inst_df)
    cyq_lookup = _build_cyq_lookup(cyq_perf_df)
    # v0.8.0 P1-1 — batch-level 题材内相对地位（同 LGB feature 派生口径）
    intra_position = industry_intra_position(candidates_df)

    out: list[dict[str, Any]] = []
    for row in candidates_df.itertuples(index=False):
        ts_code = str(row.ts_code)
        fd_amount_raw = getattr(row, "fd_amount", None)
        amount_raw = getattr(row, "amount", None)
        # v0.16.0 — 概念 / 行业 / 地域板块（同花顺，全量暴露，不截断）。
        # ConceptRepository 在快照为空时返回 []，等价于「未注入」分支。
        concepts: list[dict[str, str]] = []
        industries_full: list[dict[str, str]] = []
        regions: list[dict[str, str]] = []
        if concept_repo is not None:
            for b in concept_repo.boards_by_stock(ts_code):
                entry = {"ts_code": b.ts_code, "name": b.name}
                if b.type == "N":
                    concepts.append(entry)
                elif b.type == "I":
                    industries_full.append(entry)
                elif b.type == "R":
                    regions.append(entry)
        rec = {
            "candidate_id": ts_code,
            "ts_code": ts_code,
            "name": getattr(row, "name", None),
            "industry": getattr(row, "industry_basic", None) or getattr(row, "industry", None),
            "industries": industries_full,
            "concepts": concepts,
            "regions": regions,
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

        # v0.8.0 P1-2 — daily 历史改为摘要字段（不再 dump prev_daily 数组）
        d_hist = daily_by_code.get(ts_code, [])
        if d_hist:
            rec["amplitude_pct"] = _amplitude_pct(d_hist[-1])
            rec.update(_ma_metrics(_trailing_closes(d_hist)))
            rec["up_count_30d"] = _up_count_30d(d_hist)
            rec["max_upper_shadow_ratio_5d_pct"] = round2(max_upper_shadow_ratio_5d(d_hist))
        else:
            rec["amplitude_pct"] = None
            rec["ma5"] = rec["ma10"] = rec["ma20"] = None
            rec["ma_bull_aligned"] = None
            rec["up_count_30d"] = None
            rec["max_upper_shadow_ratio_5d_pct"] = None

        db_hist = daily_basic_by_code.get(ts_code, [])
        if db_hist:
            latest = db_hist[-1]
            rec["turnover_rate"] = round2(latest.get("turnover_rate"))
            rec["volume_ratio"] = round2(latest.get("volume_ratio"))
            rec["circ_mv_yi"] = normalize_to_yi("circ_mv", latest.get("circ_mv"))

        # v0.8.0 P1-3 — 资金流改为摘要派生（不再 dump prev_moneyflow 数组）
        mf_hist = moneyflow_by_code.get(ts_code, [])
        rec.update(
            _moneyflow_summary(
                mf_hist,
                amount_yuan=_to_float(amount_raw),
                daily_amount_qian=(
                    _to_float(d_hist[-1].get("amount")) if d_hist else None
                ),
            )
        )

        # v0.8.0 P1-1 — 题材内相对地位（同 LGB feature 派生口径）
        intra = intra_position.get(ts_code) or {}
        rec["sec_intra_rank_by_limit_times"] = _opt_int(intra.get("rank_by_limit_times"))
        rec["sec_first_to_limit_flag"] = _opt_int(intra.get("first_to_limit_flag"))
        rec["sec_is_height_board"] = _opt_int(intra.get("is_height_board"))
        rec["sec_fd_amount_rank_pct"] = (
            round2(intra.get("fd_amount_rank_pct"))
            if intra.get("fd_amount_rank_pct") is not None
            else None
        )

        # B1 LHB roll-up — null when candidate didn't make the day's top_list
        # (合法事实，不进 missing_data，由 LLM 通过 null 判断"未上榜")
        # v0.8.0 P1-3 — 数组字段 ``lhb_famous_seats`` 已被替换为标量伴生字段
        # ``lhb_famous_seats_count`` / ``lhb_famous_seats_text``，以匹配 prompt 的
        # evidence.value 标量约束（旧实现在 R2 阶段才转换，导致 R1 prompt 引用
        # lhb_famous_seats_count 会被 evidence validator 拦截）。
        lhb = lhb_rollup.get(ts_code, {})
        rec["lhb_net_buy_yi"] = lhb.get("lhb_net_buy_yi")
        rec["lhb_inst_count"] = lhb.get("lhb_inst_count")
        seats_list = lhb.get("lhb_famous_seats") or []
        rec["lhb_famous_seats_count"] = int(len(seats_list))
        rec["lhb_famous_seats_text"] = "; ".join(str(s) for s in seats_list)
        # v0.12.4 (P1-3) — 多 reason 派生字段：原因数 + 按净买入额降序拼接文本。
        # 当 ts_code 未上榜时为 0 / 空串（LLM 通过 lhb_data_quality 区分）。
        rec["lhb_reason_count"] = int(lhb.get("lhb_reason_count") or 0)
        rec["lhb_reasons_text"] = lhb.get("lhb_reasons_text") or ""
        # P1-3 — 三态显式标记数据质量，让 LLM 区分「未上榜（事实）」与「接口异常」
        if lhb_api_unavailable:
            rec["lhb_data_quality"] = "api_unavailable"
        elif lhb_api_empty:
            rec["lhb_data_quality"] = "api_empty"
        elif lhb:
            rec["lhb_data_quality"] = "listed"
        else:
            rec["lhb_data_quality"] = "not_listed"
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
    # P1-B: ``out`` order reflects ``candidates_df`` iteration order. The
    # caller (collect_round1) already passes a stably-sorted DataFrame via
    # ``_stable_sort_candidates_df`` — business priority (first_time asc,
    # limit_times desc, fd_amount desc, ts_code asc). itertuples preserves
    # row order, so no extra sort is needed here.
    return out


def _moneyflow_summary(
    moneyflow_rows: list[dict[str, Any]],
    *,
    amount_yuan: float | None,
    daily_amount_qian: float | None,
) -> dict[str, Any]:
    """Build the v0.8.0 P1-3 moneyflow summary attached to candidate rows.

    Returns the 5 scalar fields the LLM prompt now references in lieu of the
    deprecated ``prev_moneyflow`` array. Field semantics match the LGB
    feature派生 in :func:`limit_up_board.lgb.features._mf_block` so the
    prompt and the booster see the same numbers — when their decisions
    disagree it is genuine signal, not unit mismatch.

    Unit conventions (see ``FIELD_UNITS_RAW``):
      * moneyflow.* 金额单位 ``万元``
      * limit_list_d.amount 单位 ``元``
      * daily.amount 单位 ``千元``
    """
    out: dict[str, Any] = {
        "mf_net_t_yi": None,
        "mf_net_5d_sum_yi": None,
        "mf_consecutive_positive_days": None,
        "mf_net_to_amount_pct": None,
        "mf_large_order_strength_pct": None,
        "mf_divergence_flag": 0,
    }
    if not moneyflow_rows:
        return out

    # Resolve T-day amount in yuan once.
    amt_yuan = amount_yuan
    if amt_yuan is None and daily_amount_qian is not None:
        amt_yuan = daily_amount_qian * 1e3

    last = moneyflow_rows[-1]
    net_wan = _to_float(last.get("net_mf_amount"))
    out["mf_net_t_yi"] = None if net_wan is None else round(net_wan / 1e4, 4)

    if len(moneyflow_rows) >= 5:
        vals = [_to_float(r.get("net_mf_amount")) for r in moneyflow_rows[-5:]]
        if all(v is not None for v in vals):
            out["mf_net_5d_sum_yi"] = round(
                sum(v for v in vals if v is not None) / 1e4, 4
            )

    consec_pos = 0
    for r in reversed(moneyflow_rows[-5:]):
        net = _to_float(r.get("net_mf_amount"))
        if net is not None and net > 0:
            consec_pos += 1
        else:
            break
    out["mf_consecutive_positive_days"] = consec_pos

    if amt_yuan and net_wan is not None:
        out["mf_net_to_amount_pct"] = round(net_wan * 1e4 / amt_yuan * 100, 2)

    buy_lg_wan = _to_float(last.get("buy_lg_amount"))
    buy_elg_wan = _to_float(last.get("buy_elg_amount"))
    if amt_yuan and (buy_lg_wan is not None or buy_elg_wan is not None):
        lg = buy_lg_wan or 0.0
        elg = buy_elg_wan or 0.0
        out["mf_large_order_strength_pct"] = round((lg + elg) * 1e4 / amt_yuan * 100, 2)

    # Divergence: large+xlarge buys ≥ 5% of amount yet net flow ≤ 0.
    strength = out["mf_large_order_strength_pct"]
    net_t_yi = out["mf_net_t_yi"]
    if (
        strength is not None
        and strength >= 5.0
        and net_t_yi is not None
        and net_t_yi <= 0
    ):
        out["mf_divergence_flag"] = 1

    return out


def _index_by_code(df: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    """Group a DataFrame by ts_code into ascending-by-trade_date row lists.

    P1-C: pandas' default quicksort is NOT stable when NaN is present, and
    Tushare history responses can occasionally include duplicate
    ``(ts_code, trade_date)`` rows after cache merges. Sort with mergesort
    on both keys, deduplicate keeping the last row per (ts_code, trade_date),
    then group. Without this, ``daily / daily_basic / moneyflow / cyq_perf``
    row order could subtly drift across reruns and break Prompt fingerprint
    stability even when the underlying cache hasn't changed.
    """
    if df is None or df.empty or "ts_code" not in df.columns:
        return {}
    if "trade_date" in df.columns:
        df = (
            df.sort_values(["ts_code", "trade_date"], kind="mergesort", na_position="last")
              .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
              .reset_index(drop=True)
        )
    else:
        df = df.sort_values(["ts_code"], kind="mergesort").reset_index(drop=True)
    out: dict[str, list[dict[str, Any]]] = {}
    for code, group in df.groupby("ts_code", sort=True):
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
      branch above—LGB must never block 初筛/预测 (设计 §7.3 红线)。

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
