"""P1-3：``_aggregate_top_list_net`` / ``_build_lhb_rollup`` 多 reason 聚合。

v0.12.3 及之前：``rollup[ts]['lhb_net_buy_yi'] = net`` per-row 直接覆盖，
同一 ts_code 多 reason 行时后到行覆盖先到行 → 数据丢失。
v0.12.4 改为 groupby + sum；新增 lhb_reason_count / lhb_reasons_text 派生。
"""

from __future__ import annotations

import pandas as pd

from limit_up_board.data import (
    _aggregate_top_list_net,
    _build_lhb_rollup,
)


# ---------------------------------------------------------------------------
# Multi-reason sum
# ---------------------------------------------------------------------------


def test_multi_reason_sums_net_amount_in_yi() -> None:
    """同一 ts_code 三种 reason 上榜，net_amount 应当 sum 而不是覆盖。"""
    df = pd.DataFrame(
        [
            # net_amount 单位 = 元（normalize_to_yi 除以 1e8）
            {"ts_code": "000001.SZ", "reason": "日涨幅偏离7%", "net_amount": 5e8},
            {"ts_code": "000001.SZ", "reason": "机构专用", "net_amount": 1e8},
            {"ts_code": "000001.SZ", "reason": "成交额异动", "net_amount": -3e8},
        ]
    )
    out = _aggregate_top_list_net(df)
    assert "000001.SZ" in out
    entry = out["000001.SZ"]
    # 5 + 1 + (-3) = 3 亿
    assert entry["lhb_net_buy_yi"] == 3.0
    assert entry["lhb_reason_count"] == 3
    # 按 net desc 排序：日涨幅偏离7%(5亿), 机构专用(1亿), 成交额异动(-3亿)
    assert entry["lhb_reasons_text"] == "日涨幅偏离7%, 机构专用, 成交额异动"


def test_single_reason_unchanged_behavior() -> None:
    """单 reason 行：行为应与历史等价（sum of 1 row = that row）。"""
    df = pd.DataFrame(
        [{"ts_code": "600001.SH", "reason": "机构专用", "net_amount": 7e8}]
    )
    out = _aggregate_top_list_net(df)
    entry = out["600001.SH"]
    assert entry["lhb_net_buy_yi"] == 7.0
    assert entry["lhb_reason_count"] == 1
    assert entry["lhb_reasons_text"] == "机构专用"


def test_reasons_text_truncated_to_max_chars() -> None:
    """同 ts_code 多个长 reason 字符串拼接超过上限 → 截断 + 省略号。"""
    long_reasons = [f"非常长的原因名称编号{i:02d}填充" for i in range(8)]
    df = pd.DataFrame(
        [
            {"ts_code": "300999.SZ", "reason": r, "net_amount": (10 - i) * 1e8}
            for i, r in enumerate(long_reasons)
        ]
    )
    out = _aggregate_top_list_net(df, reasons_text_max_chars=40)
    txt = out["300999.SZ"]["lhb_reasons_text"]
    assert txt.endswith("…")
    assert len(txt) <= 40
    # 高 net_amount 的 reason 在前
    assert txt.startswith("非常长的原因名称编号00填充")


def test_null_net_amount_keeps_net_buy_yi_none() -> None:
    df = pd.DataFrame(
        [
            {"ts_code": "002001.SZ", "reason": "x", "net_amount": None},
            {"ts_code": "002001.SZ", "reason": "y", "net_amount": None},
        ]
    )
    out = _aggregate_top_list_net(df)
    entry = out["002001.SZ"]
    assert entry["lhb_net_buy_yi"] is None
    assert entry["lhb_reason_count"] == 2


def test_partial_null_treats_none_as_skip_in_sum() -> None:
    """None 不计入 sum，但仍占 reason_count 一席。"""
    df = pd.DataFrame(
        [
            {"ts_code": "002002.SZ", "reason": "a", "net_amount": 2e8},
            {"ts_code": "002002.SZ", "reason": "b", "net_amount": None},
        ]
    )
    out = _aggregate_top_list_net(df)
    entry = out["002002.SZ"]
    assert entry["lhb_net_buy_yi"] == 2.0
    assert entry["lhb_reason_count"] == 2


def test_empty_df_returns_empty_dict() -> None:
    assert _aggregate_top_list_net(None) == {}
    assert _aggregate_top_list_net(pd.DataFrame()) == {}
    # missing ts_code column → empty
    assert _aggregate_top_list_net(pd.DataFrame({"reason": ["x"]})) == {}


# ---------------------------------------------------------------------------
# _build_lhb_rollup wires aggregation into the existing rollup
# ---------------------------------------------------------------------------


def test_build_lhb_rollup_carries_new_fields_through() -> None:
    top_list_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "reason": "rA", "net_amount": 4e8},
            {"ts_code": "000001.SZ", "reason": "rB", "net_amount": 2e8},
        ]
    )
    top_inst_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "exalter": "机构专用", "side": 0},
            {"ts_code": "000001.SZ", "exalter": "上海互联网金融", "side": 0},
        ]
    )
    rollup = _build_lhb_rollup(top_list_df, top_inst_df)
    entry = rollup["000001.SZ"]
    assert entry["lhb_net_buy_yi"] == 6.0
    assert entry["lhb_reason_count"] == 2
    assert entry["lhb_reasons_text"] == "rA, rB"
    assert entry["lhb_inst_count"] == 2  # exalter 个数（dedup 后）
