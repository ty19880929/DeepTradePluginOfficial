"""LightGBM 特征工程 — 训练 + 推理共用入口。

本模块是 ``lightgbm_design.md`` §4 的实现。**单一来源**：`collect_round1`
（推理时）与 `dataset.collect_training_window`（训练时）都必须经此函数
生成特征矩阵，禁止任何重复实现。

设计原则
--------

1. **列名 / 顺序 / 数量** 由模块级常量 :data:`FEATURE_NAMES` 单一声明。
   训练落盘时 ``lgb_model.feature_name()`` 会被持久化；
   :func:`assert_columns` 在训练前 / 推理前校验一致性，发现 schema 漂移
   抛 :class:`FeatureSchemaMismatch` 并拒绝加载模型。
2. **NaN 原生处理**：除业务上"未上榜 / 未发生 → 0"的极少数特征外，
   缺失值一律返回 NaN，由 LightGBM 原生 missing-value handling 接收。
3. **数值健壮性**：比率类特征 clip 到 ``[-500, 500]``，防止脏数据（如
   `amount=0`）产生 inf；时间字段（HHMMSS）解析失败 → NaN。
4. **schema_version**：列定义变更（增删 / 重排序）必须 bump
   :data:`SCHEMA_VERSION`；旧模型加载时被拒绝，CLI 提示重训。

PR-1.1 范围
-----------
* :data:`FEATURE_NAMES` / :data:`SCHEMA_VERSION`
* :func:`build_feature_frame` — 完整 ~50 个特征
* :class:`FeatureSchemaMismatch` + :func:`assert_columns`

PR-2.x 才会接入 :class:`LgbScorer` 与 `collect_round1`。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

from ..data import SectorStrength

# ---------------------------------------------------------------------------
# Schema declaration — bump when列定义变更，旧模型必须重训
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 2  # v0.8.0 评审响应——见 lightgbm_design.md §SCHEMA-v2

# 列顺序就是 LightGBM 的特征序——LgbScorer.load 时会按此顺序对齐。
#
# v0.8.0 评审响应变更（FEATURE_NAMES 强制 schema_version 1 → 2）：
#   * 新增 ``f_vol_max_upper_shadow_ratio_5d``（P1-2 prev_daily 摘要补强）
#   * 新增 ``f_mf_net_to_amount_pct`` / ``f_mf_large_order_strength`` /
#     ``f_mf_divergence_flag``（P1-3 资金流派生因子，归一化 + 分歧标签）
#   * 新增 ``f_sec_intra_rank_by_limit_times`` /
#     ``f_sec_first_to_limit_flag`` / ``f_sec_is_height_board`` /
#     ``f_sec_fd_amount_rank_pct``（P1-1 题材内相对地位）
#   * 删除 ``f_sec_today_industry_up_ratio``（P2-1，恒为 1.0 的低区分度列）
FEATURE_NAMES: list[str] = [
    # ---- 涨停板属性 ----
    "f_lim_open_times",
    "f_lim_limit_times",
    "f_lim_first_time_seconds",
    "f_lim_last_time_seconds",
    "f_lim_first_to_last_seconds",
    "f_lim_fd_to_amount_pct",
    "f_lim_fd_to_float_mv_pct",
    "f_lim_limit_amount_to_float_mv_pct",
    "f_lim_up_stat_consecutive",
    "f_lim_up_stat_total_in_window",
    # ---- 量价 ----
    "f_vol_pct_chg_t",
    "f_vol_amplitude_pct",
    "f_vol_turnover_ratio",
    "f_vol_turnover_rate_t",
    "f_vol_volume_ratio_t",
    "f_vol_amount_yi_t",
    "f_vol_close_to_pre_close_pct",
    "f_vol_amount_ratio_5d",
    "f_vol_turnover_rate_ratio_5d",
    "f_vol_max_upper_shadow_ratio_5d",  # v0.8.0 P1-2
    # ---- 动量 ----
    "f_mom_close_to_ma5_bias",
    "f_mom_close_to_ma10_bias",
    "f_mom_close_to_ma20_bias",
    "f_mom_ma_bull_aligned",
    "f_mom_up_count_30d",
    "f_mom_pct_chg_5d_sum",
    "f_mom_pct_chg_10d_sum",
    "f_mom_high_to_close_pct_5d",
    # ---- 资金流 ----
    "f_mf_net_t_yi",
    "f_mf_net_5d_sum_yi",
    "f_mf_buy_lg_pct_t",
    "f_mf_buy_elg_pct_t",
    "f_mf_net_consecutive_pos_days",
    "f_mf_net_to_amount_pct",         # v0.8.0 P1-3 强度归一化
    "f_mf_large_order_strength_pct",  # v0.8.0 P1-3 (lg+elg) / amount × 100
    "f_mf_divergence_flag",           # v0.8.0 P1-3 大单买入强 + 总净流入弱
    # ---- 筹码 ----
    "f_chip_winner_pct",
    "f_chip_top10_concentration",
    "f_chip_close_to_avg_cost_pct",
    # ---- 龙虎榜 ----
    "f_lhb_appeared",
    "f_lhb_net_buy_yi",
    "f_lhb_inst_count",
    "f_lhb_famous_seats_count",
    # ---- 板块 (P1-1 题材内相对地位) ----
    "f_sec_strength_source_rank",
    "f_sec_today_industry_up_count",
    "f_sec_intra_rank_by_limit_times",  # v0.8.0 P1-1 同题材连板数排名（1-based）
    "f_sec_first_to_limit_flag",        # v0.8.0 P1-1 是否同题材最早封板
    "f_sec_is_height_board",            # v0.8.0 P1-1 是否同题材高度板
    "f_sec_fd_amount_rank_pct",         # v0.8.0 P1-1 同题材封单强度分位
    # ---- 市场环境 ----
    "f_mkt_total_limit_up",
    "f_mkt_max_height",
    "f_mkt_yesterday_failure_rate",
    "f_mkt_yesterday_continuation_rate",
    "f_mkt_high_board_delta",
    # ---- 静态 ----
    "f_st_float_mv_yi",
    "f_st_close_yuan",
    "f_st_listed_days",
]

# `f_lhb_appeared` 已经是 0/1 显式编码——不再 NaN。
# v0.8.0 新增的几个 0/1 标志位同样填 0.0（缺失 = 未触发）。
# 其余列遇缺失就保留 NaN，让模型自己学到 missingness 信号。
NA_FILLERS: dict[str, float] = {
    "f_lhb_appeared": 0.0,
    "f_mf_divergence_flag": 0.0,
    "f_sec_first_to_limit_flag": 0.0,
    "f_sec_is_height_board": 0.0,
}

# 板块强度来源 → 权威性排名（越小越权威）。
# v0.16.0 — 资源源缩窄为 limit_cpt_list / unavailable；保留 lu_desc_aggregation /
# industry_fallback 两个 legacy 键以保证历史训练数据落特征时数值不漂移
# （旧的 dataset.parquet 快照里仍可能出现这两个标签）。"unavailable" 对齐
# legacy 末档 = 3，避免对线上模型产生负向 drift。
_SECTOR_SOURCE_RANK: dict[str, int] = {
    "limit_cpt_list": 1,
    "lu_desc_aggregation": 2,
    "industry_fallback": 3,
    "unavailable": 3,
}


# ---------------------------------------------------------------------------
# Schema 自检
# ---------------------------------------------------------------------------


class FeatureSchemaMismatch(RuntimeError):
    """Feature matrix 的列名 / 顺序 / 数量与当前 :data:`FEATURE_NAMES` 不一致。

    在以下两个调用点抛出：
      * 训练前（防止历史特征列污染当前训练）；
      * :class:`LgbScorer.load` 加载落盘模型时（防止 train/infer skew）。
    """


def assert_columns(df: pd.DataFrame, *, expected: list[str] | None = None) -> None:
    """Validate ``df.columns`` exactly equals ``expected`` (default = FEATURE_NAMES)."""
    expected_list = list(expected if expected is not None else FEATURE_NAMES)
    actual = list(df.columns)
    if actual != expected_list:
        missing = [c for c in expected_list if c not in actual]
        extra = [c for c in actual if c not in expected_list]
        raise FeatureSchemaMismatch(
            f"feature columns mismatch (schema_version={SCHEMA_VERSION}): "
            f"missing={missing} extra={extra} "
            f"actual_count={len(actual)} expected_count={len(expected_list)}"
        )


# ---------------------------------------------------------------------------
# Number / time helpers
# ---------------------------------------------------------------------------


_RATIO_CLIP = 500.0


def _clip_ratio(v: float | None) -> float | None:
    if v is None:
        return None
    if not math.isfinite(v):
        # inf / nan from pathological inputs (amount=0 等) → 显式置 NaN，让模型走 missing 路径。
        return float("nan")
    if v > _RATIO_CLIP:
        return _RATIO_CLIP
    if v < -_RATIO_CLIP:
        return -_RATIO_CLIP
    return v


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _safe_div(num: float | None, den: float | None) -> float | None:
    """``num / den``；分母 0/None/NaN → None。结果不含 ±inf。"""
    if num is None or den is None or den == 0:
        return None
    out = num / den
    if not math.isfinite(out):
        return None
    return out


def _parse_time_to_seconds(v: Any) -> float | None:
    """Tushare ``first_time`` / ``last_time`` → 距离当日 00:00:00 的秒数。

    Tushare 实际返回 ``"HH:MM:SS"``（带冒号）；为稳健起见也兼容 ``"HHMMSS"``
    6 位数字以及浮点数（罕见）。无法解析 → NaN。
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 2:
            h, m, sec = nums[0], nums[1], 0
        elif len(nums) == 3:
            h, m, sec = nums
        else:
            return None
    else:
        # 期望 "HHMMSS"。允许 "93000"（缺前导 0）/ "093000"。
        digits = s.lstrip("0") or "0"
        n = int(digits) if digits.isdigit() else None
        if n is None:
            return None
        h = n // 10000
        m = (n // 100) % 100
        sec = n % 100
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
        return None
    return float(h * 3600 + m * 60 + sec)


def _parse_up_stat(v: Any) -> tuple[float, float] | None:
    """``"3/5"`` → ``(3.0, 5.0)``。解析失败 → None。"""
    if v is None:
        return None
    s = str(v).strip()
    if "/" not in s:
        return None
    a, b = s.split("/", 1)
    try:
        return float(int(a.strip())), float(int(b.strip()))
    except (TypeError, ValueError):
        return None


def _days_between(end_yyyymmdd: str, start_yyyymmdd: Any) -> float | None:
    """日历天数差；无法解析 start → None。"""
    if start_yyyymmdd is None:
        return None
    s = str(start_yyyymmdd).strip()
    if not s or not s.isdigit() or len(s) != 8:
        return None
    try:
        d_end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
        d_start = datetime.strptime(s, "%Y%m%d")
    except ValueError:
        return None
    return float((d_end - d_start).days)


# ---------------------------------------------------------------------------
# Per-candidate feature builders
# ---------------------------------------------------------------------------


def _limit_block(row: dict[str, Any]) -> dict[str, float | None]:
    """涨停板属性 (10 个)。"""
    fd = _to_float(row.get("fd_amount"))
    amt = _to_float(row.get("amount"))
    limit_amount = _to_float(row.get("limit_amount"))
    float_mv = _to_float(row.get("float_mv"))

    first_sec = _parse_time_to_seconds(row.get("first_time"))
    last_sec = _parse_time_to_seconds(row.get("last_time"))
    delta_sec = (
        (last_sec - first_sec)
        if first_sec is not None and last_sec is not None
        else None
    )

    up_stat = _parse_up_stat(row.get("up_stat"))
    if up_stat is None:
        consec, total = None, None
    else:
        consec, total = up_stat

    return {
        "f_lim_open_times": _to_float(row.get("open_times")),
        "f_lim_limit_times": _to_float(row.get("limit_times")),
        "f_lim_first_time_seconds": first_sec,
        "f_lim_last_time_seconds": last_sec,
        "f_lim_first_to_last_seconds": delta_sec,
        "f_lim_fd_to_amount_pct": _clip_ratio(
            (fd / amt * 100) if (fd is not None and amt) else None
        ),
        "f_lim_fd_to_float_mv_pct": _clip_ratio(
            (fd / float_mv * 100) if (fd is not None and float_mv) else None
        ),
        "f_lim_limit_amount_to_float_mv_pct": _clip_ratio(
            (limit_amount / float_mv * 100)
            if (limit_amount is not None and float_mv)
            else None
        ),
        "f_lim_up_stat_consecutive": consec,
        "f_lim_up_stat_total_in_window": total,
    }


def _vol_block(
    row: dict[str, Any],
    daily_rows: list[dict[str, Any]],
    daily_basic_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """量价 (10 个)。

    v0.8.0 P1-2 — 新增 ``f_vol_max_upper_shadow_ratio_5d``：近 5 日 (high - max(close, open))
    / close × 100 的最大值，作为冲高回落 / 上影长度的代理；负值（上影 ≤ 0）clip 到 0。
    """
    close = _to_float(row.get("close"))
    pre_close = _to_float(daily_rows[-1].get("pre_close")) if daily_rows else None
    pct_chg = _to_float(row.get("pct_chg"))
    if pct_chg is None and daily_rows:
        pct_chg = _to_float(daily_rows[-1].get("pct_chg"))

    amplitude = None
    if daily_rows:
        last_d = daily_rows[-1]
        high = _to_float(last_d.get("high"))
        low = _to_float(last_d.get("low"))
        pc = _to_float(last_d.get("pre_close"))
        if high is not None and low is not None and pc:
            amplitude = (high - low) / pc * 100

    # close/pre_close → pct_chg fallback
    close_to_pre = None
    if pre_close and close is not None:
        close_to_pre = (close - pre_close) / pre_close * 100

    amount_t = _to_float(daily_rows[-1].get("amount")) if daily_rows else None
    amount_5d_mean = None
    if len(daily_rows) >= 6:
        prev_5 = [
            _to_float(r.get("amount")) for r in daily_rows[-6:-1] if r.get("amount") is not None
        ]
        prev_5 = [v for v in prev_5 if v is not None and v > 0]
        if prev_5:
            amount_5d_mean = sum(prev_5) / len(prev_5)
    amount_ratio_5d = _safe_div(amount_t, amount_5d_mean)

    turnover_rate_t = (
        _to_float(daily_basic_rows[-1].get("turnover_rate"))
        if daily_basic_rows
        else None
    )
    volume_ratio = (
        _to_float(daily_basic_rows[-1].get("volume_ratio"))
        if daily_basic_rows
        else None
    )
    turnover_rate_5d_mean = None
    if len(daily_basic_rows) >= 6:
        prev_5 = [
            _to_float(r.get("turnover_rate"))
            for r in daily_basic_rows[-6:-1]
        ]
        prev_5 = [v for v in prev_5 if v is not None and v > 0]
        if prev_5:
            turnover_rate_5d_mean = sum(prev_5) / len(prev_5)
    turnover_ratio_5d = _safe_div(turnover_rate_t, turnover_rate_5d_mean)

    # v0.8.0 P1-2 — 近 5 日最大上影占收盘价比（冲高回落代理）。
    max_upper_shadow_pct_5d = max_upper_shadow_ratio_5d(daily_rows)

    return {
        "f_vol_pct_chg_t": _clip_ratio(pct_chg),
        "f_vol_amplitude_pct": _clip_ratio(amplitude),
        "f_vol_turnover_ratio": _clip_ratio(_to_float(row.get("turnover_ratio"))),
        "f_vol_turnover_rate_t": _clip_ratio(turnover_rate_t),
        "f_vol_volume_ratio_t": _clip_ratio(volume_ratio),
        "f_vol_amount_yi_t": (
            None if amount_t is None else round(amount_t / 1e5, 4)
            # daily.amount 单位 = 千元 → 亿 = ÷ 1e5
        ),
        "f_vol_close_to_pre_close_pct": _clip_ratio(close_to_pre),
        "f_vol_amount_ratio_5d": _clip_ratio(amount_ratio_5d),
        "f_vol_turnover_rate_ratio_5d": _clip_ratio(turnover_ratio_5d),
        "f_vol_max_upper_shadow_ratio_5d": _clip_ratio(max_upper_shadow_pct_5d),
    }


def max_upper_shadow_ratio_5d(daily_rows: list[dict[str, Any]]) -> float | None:
    """近 5 日 ``(high - max(open, close)) / close * 100`` 的最大值。

    评审 P1-2：作为冲高回落 / 上影长度的代理摘要字段。``daily_rows`` 不足 5
    天 / close=0 → None。Negative shadows (open/close ≥ high) clip 到 0。
    Exported so ``data._build_candidate_rows`` can reuse the same definition
    in the LLM candidate payload — keeping LGB feature and prompt summary
    consistent.
    """
    if len(daily_rows) < 1:
        return None
    window = daily_rows[-5:]
    out: list[float] = []
    for r in window:
        high = _to_float(r.get("high"))
        open_ = _to_float(r.get("open"))
        close = _to_float(r.get("close"))
        if high is None or close is None or close <= 0:
            continue
        body_top = max(o for o in (open_, close) if o is not None) if open_ is not None else close
        shadow = max(0.0, high - body_top)
        out.append(shadow / close * 100)
    if not out:
        return None
    return max(out)


def _mom_block(
    daily_rows: list[dict[str, Any]],
    up_count_30d: float | None,
) -> dict[str, float | None]:
    """动量 (8 个)。"""
    closes: list[float] = [
        _to_float(r.get("close")) for r in daily_rows if r.get("close") is not None
    ]
    closes = [c for c in closes if c is not None]

    def _ma(window: int) -> float | None:
        if len(closes) < window:
            return None
        return sum(closes[-window:]) / window

    ma5, ma10, ma20 = _ma(5), _ma(10), _ma(20)
    latest = closes[-1] if closes else None

    def _bias(latest: float | None, ma: float | None) -> float | None:
        if latest is None or ma is None or ma == 0:
            return None
        return (latest - ma) / ma * 100

    bull = None
    if all(v is not None for v in (latest, ma5, ma10, ma20)):
        # mypy 已通过 all-not-None 收敛类型，但运行时仍 cast
        bull = 1.0 if (latest > ma5 > ma10 > ma20) else 0.0  # type: ignore[operator]

    def _pct_sum(window: int) -> float | None:
        if len(daily_rows) < window:
            return None
        recent = daily_rows[-window:]
        vals = [_to_float(r.get("pct_chg")) for r in recent]
        if any(v is None for v in vals):
            return None
        return sum(v for v in vals if v is not None)

    high_to_close_5d = None
    if len(daily_rows) >= 5 and latest is not None and latest > 0:
        highs = [_to_float(r.get("high")) for r in daily_rows[-5:]]
        highs = [v for v in highs if v is not None]
        if highs:
            high_to_close_5d = (max(highs) - latest) / latest * 100

    return {
        "f_mom_close_to_ma5_bias": _clip_ratio(_bias(latest, ma5)),
        "f_mom_close_to_ma10_bias": _clip_ratio(_bias(latest, ma10)),
        "f_mom_close_to_ma20_bias": _clip_ratio(_bias(latest, ma20)),
        "f_mom_ma_bull_aligned": bull,
        "f_mom_up_count_30d": up_count_30d,
        "f_mom_pct_chg_5d_sum": _clip_ratio(_pct_sum(5)),
        "f_mom_pct_chg_10d_sum": _clip_ratio(_pct_sum(10)),
        "f_mom_high_to_close_pct_5d": _clip_ratio(high_to_close_5d),
    }


def _mf_block(
    moneyflow_rows: list[dict[str, Any]],
    candidate_row: dict[str, Any],
    daily_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """资金流 (8 个).

    单位说明（见 ``data.FIELD_UNITS_RAW``）：
      * moneyflow.* 金额均为 ``万元``；
      * limit_list_d.amount 单位 ``元``（来自当日候选行 ``candidate_row``）；
      * daily.amount 单位 ``千元`` （来自历史窗口）；
    比率计算时统一换算到 元 后再除。

    v0.8.0 P1-3 新增 3 个派生：
      * ``f_mf_net_to_amount_pct``      — 净流入 / T 日成交额 × 100（强度归一化）
      * ``f_mf_large_order_strength``   — (大单买 + 超大单买) / T 日成交额 × 100
      * ``f_mf_divergence_flag``        — 大单买入强（>5%）但净流入 ≤ 0 时置 1，
                                          反映「主力净流入与大单买入分歧」（评审 P1-3）
    """
    empty = {
        "f_mf_net_t_yi": None,
        "f_mf_net_5d_sum_yi": None,
        "f_mf_buy_lg_pct_t": None,
        "f_mf_buy_elg_pct_t": None,
        "f_mf_net_consecutive_pos_days": None,
        "f_mf_net_to_amount_pct": None,
        "f_mf_large_order_strength_pct": None,
        "f_mf_divergence_flag": 0.0,
    }
    if not moneyflow_rows:
        return empty

    last = moneyflow_rows[-1]
    net_wan = _to_float(last.get("net_mf_amount"))
    net_t_yi = None if net_wan is None else net_wan / 1e4  # 万元 → 亿

    if len(moneyflow_rows) >= 5:
        recent_5 = moneyflow_rows[-5:]
        vals = [_to_float(r.get("net_mf_amount")) for r in recent_5]
        if all(v is not None for v in vals):
            net_5d_sum_yi = sum(v for v in vals if v is not None) / 1e4  # 万元 → 亿
        else:
            net_5d_sum_yi = None
    else:
        net_5d_sum_yi = None

    buy_lg_wan = _to_float(last.get("buy_lg_amount"))
    buy_elg_wan = _to_float(last.get("buy_elg_amount"))
    # candidate_row.amount 单位 = 元；万元 × 1e4 = 元，再 / 元 = 无量纲
    amount_yuan = _to_float(candidate_row.get("amount"))
    if amount_yuan is None and daily_rows:
        # fallback to daily.amount (千元) × 1e3 = 元
        amount_qian = _to_float(daily_rows[-1].get("amount"))
        amount_yuan = None if amount_qian is None else amount_qian * 1e3

    def _pct(buy_wan: float | None) -> float | None:
        if buy_wan is None or not amount_yuan:
            return None
        return _clip_ratio(buy_wan * 1e4 / amount_yuan)

    buy_lg_pct = _pct(buy_lg_wan)
    buy_elg_pct = _pct(buy_elg_wan)

    # 强度归一化：净流入 / 成交额（消除市值大小差异；同样使用 万元 → 元）。
    net_to_amount_pct: float | None = None
    if net_wan is not None and amount_yuan:
        net_to_amount_pct = _clip_ratio(net_wan * 1e4 / amount_yuan * 100)

    # (大单买 + 超大单买) / 成交额 — 主力买入强度。
    large_order_strength: float | None = None
    if amount_yuan and (buy_lg_wan is not None or buy_elg_wan is not None):
        lg = buy_lg_wan or 0.0
        elg = buy_elg_wan or 0.0
        large_order_strength = _clip_ratio((lg + elg) * 1e4 / amount_yuan * 100)

    # 分歧标签：大单 + 超大单买入强（≥5% of amount）但当日净流入 ≤ 0。
    divergence_flag = 0.0
    if (
        large_order_strength is not None
        and large_order_strength >= 5.0
        and net_t_yi is not None
        and net_t_yi <= 0
    ):
        divergence_flag = 1.0

    consec_pos = 0
    for r in reversed(moneyflow_rows[-5:]):
        net = _to_float(r.get("net_mf_amount"))
        if net is not None and net > 0:
            consec_pos += 1
        else:
            break
    consec_pos_f: float = float(consec_pos)

    return {
        "f_mf_net_t_yi": net_t_yi,
        "f_mf_net_5d_sum_yi": net_5d_sum_yi,
        "f_mf_buy_lg_pct_t": buy_lg_pct,
        "f_mf_buy_elg_pct_t": buy_elg_pct,
        "f_mf_net_consecutive_pos_days": consec_pos_f,
        "f_mf_net_to_amount_pct": net_to_amount_pct,
        "f_mf_large_order_strength_pct": large_order_strength,
        "f_mf_divergence_flag": divergence_flag,
    }


def _chip_block(
    cyq: dict[str, Any] | None, close: float | None = None
) -> dict[str, float | None]:
    """筹码 (3 个)。

    ``f_chip_close_to_avg_cost_pct`` is derived here from ``close`` and the cyq
    weighted-average cost. The cyq lookup (``data._build_cyq_lookup``) only
    carries ``cyq_avg_cost_yuan`` — it never wrote a ``cyq_close_to_avg_cost_pct``
    key, so the previous ``cyq.get("cyq_close_to_avg_cost_pct")`` read was always
    None and this feature was permanently NaN (a dead feature in both train and
    serve). Compute it the same way ``data._close_to_avg_cost_pct`` does so the
    booster and the prompt see the same number.
    """
    if not cyq:
        return {
            "f_chip_winner_pct": None,
            "f_chip_top10_concentration": None,
            "f_chip_close_to_avg_cost_pct": None,
        }
    avg_cost = _to_float(cyq.get("cyq_avg_cost_yuan"))
    close_f = _to_float(close)
    close_to_avg: float | None = None
    if close_f is not None and avg_cost:
        close_to_avg = (close_f - avg_cost) / avg_cost * 100
    return {
        "f_chip_winner_pct": _to_float(cyq.get("cyq_winner_pct")),
        "f_chip_top10_concentration": _to_float(cyq.get("cyq_top10_concentration")),
        "f_chip_close_to_avg_cost_pct": _clip_ratio(close_to_avg),
    }


def _lhb_block(lhb: dict[str, Any] | None) -> dict[str, float | None]:
    """龙虎榜 (4 个)。``lhb`` 为 None / 空表示该候选未上榜。"""
    if not lhb:
        return {
            "f_lhb_appeared": 0.0,
            "f_lhb_net_buy_yi": None,
            "f_lhb_inst_count": None,
            "f_lhb_famous_seats_count": None,
        }
    seats = lhb.get("lhb_famous_seats") or []
    return {
        "f_lhb_appeared": 1.0,
        "f_lhb_net_buy_yi": _to_float(lhb.get("lhb_net_buy_yi")),
        "f_lhb_inst_count": _to_float(lhb.get("lhb_inst_count")),
        "f_lhb_famous_seats_count": float(len(seats)),
    }


# ---------------------------------------------------------------------------
# Batch-level feature derivation (sector + market)
# ---------------------------------------------------------------------------


def _resolve_industry(row: pd.Series) -> str | None:
    ib = row.get("industry_basic")
    if isinstance(ib, str) and ib.strip():
        return ib.strip()
    ind = row.get("industry")
    if isinstance(ind, str) and ind.strip():
        return ind.strip()
    return None


def _industry_aggregates(candidates_df: pd.DataFrame) -> dict[str, int]:
    """Return ``{industry: today_up_count}``.

    候选股本身就是当日涨停标的——所以 today_up_count == 同行业候选数。
    v0.8.0 P2-1 — 移除原本的 ratio (恒为 1.0)；只保留 count，更细的相对地位由
    :func:`industry_intra_position` 单独派生。
    """
    out: dict[str, int] = {}
    if candidates_df.empty:
        return out
    industries = candidates_df.apply(_resolve_industry, axis=1)
    for ind in industries.dropna():
        out[ind] = out.get(ind, 0) + 1
    return out


def industry_intra_position(
    candidates_df: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """Per-candidate intra-industry ranking signals (v0.8.0 P1-1).

    Returns ``{ts_code: {
        "rank_by_limit_times": int,         # 1-based, descending by limit_times
        "first_to_limit_flag": 0.0/1.0,     # 是否同题材最早封板 (first_time 最小)
        "is_height_board": 0.0/1.0,         # limit_times == industry max
        "fd_amount_rank_pct": float,        # 同题材 fd_amount 分位 [0, 100]
                                            # 100 = 最强；只有 1 人时记 100
    }}``.

    Single-candidate industries → flags 都置 1.0（自己就是龙头），rank_pct=100。
    Industry 无法识别 → 全部 None。
    """
    out: dict[str, dict[str, float | None]] = {}
    if candidates_df.empty:
        return out
    groups: dict[str, list[dict[str, Any]]] = {}
    for _, row in candidates_df.iterrows():
        ind = _resolve_industry(row)
        if ind is None:
            ts = str(row.get("ts_code") or "")
            if ts:
                out[ts] = {
                    "rank_by_limit_times": None,
                    "first_to_limit_flag": None,
                    "is_height_board": None,
                    "fd_amount_rank_pct": None,
                }
            continue
        groups.setdefault(ind, []).append(
            {
                "ts_code": str(row.get("ts_code") or ""),
                "limit_times": _to_float(row.get("limit_times")) or 0.0,
                "first_time": _parse_time_to_seconds(row.get("first_time")),
                "fd_amount": _to_float(row.get("fd_amount")),
            }
        )
    for ind, members in groups.items():
        # rank by limit_times desc (higher is stronger)
        sorted_members = sorted(
            members, key=lambda m: m["limit_times"], reverse=True
        )
        max_lt = sorted_members[0]["limit_times"]
        # first_time min wins (None → +inf so it loses)
        valid_first = [m for m in members if m["first_time"] is not None]
        earliest_first = (
            min(m["first_time"] for m in valid_first) if valid_first else None
        )
        # fd_amount percentile rank (higher = stronger).
        with_fd = [m for m in members if m["fd_amount"] is not None]
        fd_sorted = sorted(with_fd, key=lambda m: m["fd_amount"] or 0.0)
        fd_rank: dict[str, float] = {}
        if fd_sorted:
            n = len(fd_sorted)
            # 1-based ascending rank → percentile = rank / n * 100
            for idx, m in enumerate(fd_sorted, start=1):
                fd_rank[m["ts_code"]] = idx / n * 100.0

        for idx, m in enumerate(sorted_members, start=1):
            ts = m["ts_code"]
            out[ts] = {
                "rank_by_limit_times": float(idx),
                "first_to_limit_flag": (
                    1.0
                    if (m["first_time"] is not None and m["first_time"] == earliest_first)
                    else 0.0
                ),
                "is_height_board": (
                    1.0 if m["limit_times"] >= max_lt and max_lt > 0 else 0.0
                ),
                "fd_amount_rank_pct": fd_rank.get(ts),
            }
    return out


def _market_block(market_summary: dict[str, Any]) -> dict[str, float | None]:
    """市场环境 (5 个)。"""
    total_lu = market_summary.get("limit_up_count")
    step = market_summary.get("limit_step_distribution") or {}
    if isinstance(step, dict) and step:
        try:
            max_height = max(int(k) for k in step.keys() if str(k).lstrip("-").isdigit())
        except ValueError:
            max_height = None
    else:
        max_height = None

    yfr = market_summary.get("yesterday_failure_rate") or {}
    yfr_pct = _to_float(yfr.get("rate_pct"))
    ywt = market_summary.get("yesterday_winners_today") or {}
    cont_pct = _to_float(ywt.get("continuation_rate_pct"))
    trend = market_summary.get("limit_step_trend") or {}
    high_delta = _to_float(trend.get("high_board_delta"))

    return {
        "f_mkt_total_limit_up": _to_float(total_lu),
        "f_mkt_max_height": None if max_height is None else float(max_height),
        "f_mkt_yesterday_failure_rate": _clip_ratio(yfr_pct),
        "f_mkt_yesterday_continuation_rate": _clip_ratio(cont_pct),
        "f_mkt_high_board_delta": high_delta,
    }


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def build_feature_frame(
    *,
    candidates_df: pd.DataFrame,
    daily_by_code: dict[str, list[dict[str, Any]]],
    daily_basic_by_code: dict[str, list[dict[str, Any]]],
    moneyflow_by_code: dict[str, list[dict[str, Any]]],
    cyq_by_code: dict[str, dict[str, Any]],
    lhb_rollup: dict[str, dict[str, Any]],
    ths_lookup: dict[str, dict[str, Any]] | None = None,  # noqa: ARG001 (reserved for v0.6)
    sector_strength: SectorStrength | None = None,
    market_summary: dict[str, Any] | None = None,
    trade_date: str | None = None,
    up_count_30d_by_code: dict[str, float | None] | None = None,
    daily_lookback: int = 30,  # noqa: ARG001 (reserved for future window control)
    moneyflow_lookback: int = 5,  # noqa: ARG001
) -> pd.DataFrame:
    """Compute the ``(n_candidates × n_features)`` feature matrix.

    Parameters
    ----------
    candidates_df
        ``limit_list_d ⋈ stock_basic`` 后的当日候选；必须含 ``ts_code`` 列。
        以及 close / amount / fd_amount / limit_amount / float_mv / pct_chg /
        first_time / last_time / open_times / limit_times / up_stat /
        turnover_ratio / industry / industry_basic / list_date。
    daily_by_code / daily_basic_by_code / moneyflow_by_code
        每个 ts_code 的历史行列表（按 trade_date 升序，最后一行 = T 当日）。
        来自 ``data._index_by_code``。
    cyq_by_code
        ``data._build_cyq_lookup`` 的输出 ``{ts_code: {cyq_winner_pct, ...}}``。
    lhb_rollup
        ``data._build_lhb_rollup`` 的输出。candidate 不在 dict 中 → 未上榜。
    sector_strength
        ``data.SectorStrength`` 实例；用于 ``f_sec_strength_source_rank``。
    market_summary
        ``Round1Bundle.market_summary`` 的 dict（``limit_step_distribution``
        / ``yesterday_failure_rate`` / ``yesterday_winners_today`` /
        ``limit_step_trend``）。可以是空 dict，但会得到一批 NaN。
    trade_date
        ``YYYYMMDD`` 格式，仅用于计算 ``f_st_listed_days``。可省略，省略时
        从 ``candidates_df['trade_date']`` 取第一行（限 inference 推断；训练时
        建议显式传入）。
    up_count_30d_by_code
        可选预先算好的 ``{ts_code: up_count_30d}``。省略时本函数从 ``daily_rows``
        现算。

    Returns
    -------
    pd.DataFrame
        index = ``candidates_df['ts_code']``；列 = :data:`FEATURE_NAMES`。
        缺失值保留 NaN；比率类列已 clip 到 ``[-500, 500]``。
    """
    if "ts_code" not in candidates_df.columns:
        raise ValueError("candidates_df missing 'ts_code' column")

    if trade_date is None:
        td_col = candidates_df.get("trade_date")
        if td_col is None or td_col.empty:
            raise ValueError(
                "trade_date must be provided (candidates_df has no trade_date column either)"
            )
        trade_date = str(td_col.iloc[0])

    market_summary = market_summary or {}
    industry_agg = _industry_aggregates(candidates_df)
    intra_position = industry_intra_position(candidates_df)  # v0.8.0 P1-1

    if sector_strength is not None:
        source_rank = float(_SECTOR_SOURCE_RANK.get(sector_strength.source, 3))
    else:
        source_rank = float("nan")

    market_block = _market_block(market_summary)

    rows: list[dict[str, float | None]] = []
    ts_codes: list[str] = []
    for _, candidate in candidates_df.iterrows():
        ts = str(candidate["ts_code"])
        ts_codes.append(ts)
        cand_dict: dict[str, Any] = candidate.to_dict()

        daily_rows = daily_by_code.get(ts, [])
        daily_basic_rows = daily_basic_by_code.get(ts, [])
        moneyflow_rows = moneyflow_by_code.get(ts, [])
        cyq = cyq_by_code.get(ts)
        lhb = lhb_rollup.get(ts)

        # up_count_30d：优先用调用方预算结果（与 data._up_count_30d 一致语义）
        if up_count_30d_by_code is not None and ts in up_count_30d_by_code:
            up_count = up_count_30d_by_code[ts]
        elif len(daily_rows) >= 30:
            recent = daily_rows[-30:]
            up_count = float(
                sum(1 for r in recent if (_to_float(r.get("pct_chg")) or 0) >= 9.8)
            )
        else:
            up_count = None

        feat: dict[str, float | None] = {}
        feat.update(_limit_block(cand_dict))
        feat.update(_vol_block(cand_dict, daily_rows, daily_basic_rows))
        feat.update(_mom_block(daily_rows, up_count))
        feat.update(_mf_block(moneyflow_rows, cand_dict, daily_rows))
        feat.update(_chip_block(cyq, _to_float(cand_dict.get("close"))))
        feat.update(_lhb_block(lhb))

        # sector — v0.8.0 P1-1 题材内相对地位
        industry = _resolve_industry(candidate)
        up_cnt = industry_agg.get(industry, 0) if industry is not None else 0
        feat["f_sec_strength_source_rank"] = source_rank
        feat["f_sec_today_industry_up_count"] = float(up_cnt) if up_cnt else None
        intra = intra_position.get(ts) or {}
        feat["f_sec_intra_rank_by_limit_times"] = intra.get("rank_by_limit_times")
        feat["f_sec_first_to_limit_flag"] = intra.get("first_to_limit_flag")
        feat["f_sec_is_height_board"] = intra.get("is_height_board")
        feat["f_sec_fd_amount_rank_pct"] = intra.get("fd_amount_rank_pct")

        # market (constant per batch)
        feat.update(market_block)

        # static
        float_mv_yi = None
        fmv = _to_float(cand_dict.get("float_mv"))
        if fmv is not None:
            float_mv_yi = fmv / 1e8
        feat["f_st_float_mv_yi"] = float_mv_yi
        feat["f_st_close_yuan"] = _to_float(cand_dict.get("close"))
        feat["f_st_listed_days"] = _days_between(trade_date, cand_dict.get("list_date"))

        rows.append(feat)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES, index=pd.Index(ts_codes, name="ts_code"))

    # 显式 NA fillers（目前只有 f_lhb_appeared，本来就是 0/1 不会 NaN——保险起见再 fill）。
    for col, fill_value in NA_FILLERS.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    assert_columns(df)
    return df


def feature_missing_columns(feature_row: pd.Series) -> list[str]:
    """Return列名 list of NaN entries in a single row (for audit logging)."""
    return [c for c in FEATURE_NAMES if c in feature_row.index and pd.isna(feature_row[c])]
