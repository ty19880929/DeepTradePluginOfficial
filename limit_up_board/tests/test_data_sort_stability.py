"""P1-B / P1-C / P1-D / P1-E: stable-sort tests for Prompt-input determinism.

Each test constructs deliberately-shuffled DataFrames and asserts that the
relevant data-layer function produces output whose order does NOT depend on
input row order. Without these guarantees, identical Tushare cache hits can
still produce different LLM Prompts across reruns.
"""

from __future__ import annotations

import pandas as pd

from limit_up_board.data import (
    _aggregate_top_list_net,
    _build_lhb_rollup,
    _famous_seats_hits,
    _index_by_code,
    _stable_sort_candidates_df,
)


# ---------------------------------------------------------------------------
# P1-B: _stable_sort_candidates_df
# ---------------------------------------------------------------------------


def test_stable_sort_candidates_df_orders_by_business_priority() -> None:
    """trade_date asc, first_time asc, limit_times desc, fd_amount desc, ts_code asc."""
    df = pd.DataFrame(
        [
            {"ts_code": "300001.SZ", "trade_date": "20260530",
             "first_time": "09:35", "limit_times": 1, "fd_amount": 1e8},
            {"ts_code": "000001.SZ", "trade_date": "20260530",
             "first_time": "09:30", "limit_times": 2, "fd_amount": 5e8},
            {"ts_code": "600000.SH", "trade_date": "20260530",
             "first_time": "09:30", "limit_times": 2, "fd_amount": 3e8},
        ]
    )
    out = _stable_sort_candidates_df(df)
    # 09:30 同时间，limit_times=2 同高度，fd_amount desc: 5亿 > 3亿
    # 然后 09:35 / limit_times=1 排末
    assert out["ts_code"].tolist() == ["000001.SZ", "600000.SH", "300001.SZ"]


def test_stable_sort_candidates_df_invariant_to_input_shuffle() -> None:
    base = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260530",
             "first_time": "09:30", "limit_times": 2, "fd_amount": 5e8},
            {"ts_code": "600000.SH", "trade_date": "20260530",
             "first_time": "09:30", "limit_times": 2, "fd_amount": 3e8},
            {"ts_code": "300001.SZ", "trade_date": "20260530",
             "first_time": "09:35", "limit_times": 1, "fd_amount": 1e8},
        ]
    )
    shuffled = base.iloc[::-1].reset_index(drop=True)
    out_a = _stable_sort_candidates_df(base)
    out_b = _stable_sort_candidates_df(shuffled)
    assert out_a["ts_code"].tolist() == out_b["ts_code"].tolist()


def test_stable_sort_candidates_df_null_first_time_goes_last() -> None:
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260530",
             "first_time": None, "limit_times": 1, "fd_amount": 9e8},
            {"ts_code": "600000.SH", "trade_date": "20260530",
             "first_time": "10:00", "limit_times": 1, "fd_amount": 1e8},
        ]
    )
    out = _stable_sort_candidates_df(df)
    # null first_time should land at bottom even though it has higher fd_amount
    assert out["ts_code"].tolist() == ["600000.SH", "000001.SZ"]


def test_stable_sort_candidates_df_falls_back_when_columns_missing() -> None:
    """Defensive: minimal DataFrame with only ts_code still sorts deterministically."""
    df = pd.DataFrame([{"ts_code": "600000.SH"}, {"ts_code": "000001.SZ"}])
    out = _stable_sort_candidates_df(df)
    assert out["ts_code"].tolist() == ["000001.SZ", "600000.SH"]


# ---------------------------------------------------------------------------
# P1-C: _index_by_code
# ---------------------------------------------------------------------------


def test_index_by_code_sorts_within_group_by_trade_date() -> None:
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260530", "close": 10.0},
            {"ts_code": "000001.SZ", "trade_date": "20260528", "close": 9.0},
            {"ts_code": "000001.SZ", "trade_date": "20260529", "close": 9.5},
        ]
    )
    out = _index_by_code(df)
    assert [r["trade_date"] for r in out["000001.SZ"]] == ["20260528", "20260529", "20260530"]


def test_index_by_code_dedupes_keeping_last() -> None:
    """Duplicate (ts_code, trade_date) rows — keep last (most recent fetch)."""
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260530", "close": 10.0},
            {"ts_code": "000001.SZ", "trade_date": "20260530", "close": 10.5},  # dup, keep this
        ]
    )
    out = _index_by_code(df)
    assert len(out["000001.SZ"]) == 1
    assert out["000001.SZ"][0]["close"] == 10.5


def test_index_by_code_invariant_to_input_shuffle() -> None:
    base = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260528", "close": 9.0},
            {"ts_code": "000001.SZ", "trade_date": "20260529", "close": 9.5},
            {"ts_code": "600000.SH", "trade_date": "20260528", "close": 5.0},
            {"ts_code": "600000.SH", "trade_date": "20260529", "close": 5.5},
        ]
    )
    shuffled = base.iloc[::-1].reset_index(drop=True)
    out_a = _index_by_code(base)
    out_b = _index_by_code(shuffled)
    assert list(out_a.keys()) == list(out_b.keys())
    for code in out_a:
        assert [r["trade_date"] for r in out_a[code]] == [r["trade_date"] for r in out_b[code]]


# ---------------------------------------------------------------------------
# P1-D: _aggregate_top_list_net reason tie-breaker
# ---------------------------------------------------------------------------


def test_aggregate_top_list_net_reason_tie_breaker_stable() -> None:
    """Same net_amount → reasons sorted asc by text (not by input order)."""
    df_a = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "reason": "Z_reason", "net_amount": 1e8},
            {"ts_code": "000001.SZ", "reason": "A_reason", "net_amount": 1e8},
            {"ts_code": "000001.SZ", "reason": "M_reason", "net_amount": 1e8},
        ]
    )
    df_b = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "reason": "A_reason", "net_amount": 1e8},
            {"ts_code": "000001.SZ", "reason": "M_reason", "net_amount": 1e8},
            {"ts_code": "000001.SZ", "reason": "Z_reason", "net_amount": 1e8},
        ]
    )
    out_a = _aggregate_top_list_net(df_a)["000001.SZ"]
    out_b = _aggregate_top_list_net(df_b)["000001.SZ"]
    # Same content, same order regardless of input shuffle
    assert out_a["lhb_reasons_text"] == out_b["lhb_reasons_text"]
    assert out_a["lhb_reasons_text"] == "A_reason, M_reason, Z_reason"


def test_aggregate_top_list_net_primary_sort_still_by_net_desc() -> None:
    """Tie-breaker must NOT override the primary net_amount-desc ordering."""
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "reason": "Z_low_net", "net_amount": 1e8},
            {"ts_code": "000001.SZ", "reason": "A_high_net", "net_amount": 5e8},
        ]
    )
    out = _aggregate_top_list_net(df)["000001.SZ"]
    assert out["lhb_reasons_text"] == "A_high_net, Z_low_net"


# ---------------------------------------------------------------------------
# P1-E: _famous_seats_hits sorted output
# ---------------------------------------------------------------------------


def test_famous_seats_hits_output_sorted() -> None:
    """Output is sorted by name regardless of input order.

    Uses real seat strings that match ``FAMOUS_SEATS_HINTS`` substrings.
    """
    seat_lasa = "东方财富证券拉萨团结路第二营业部"      # matches "拉萨团结路" / "东方财富证券拉萨"
    seat_ningbo = "宁波桑田路证券营业部"                # matches "宁波桑田路"
    seat_zhongxin = "中信证券上海溧阳路证券营业部"      # matches "中信证券上海溧阳路"
    seats_a = [seat_lasa, seat_ningbo, seat_zhongxin]
    seats_b = [seat_zhongxin, seat_lasa, seat_ningbo]
    out_a = _famous_seats_hits(seats_a)
    out_b = _famous_seats_hits(seats_b)
    assert len(out_a) == 3
    assert out_a == out_b
    assert out_a == sorted(out_a)


def test_famous_seats_hits_deduplicates() -> None:
    seat = "东方财富证券拉萨团结路第二营业部"
    out = _famous_seats_hits([seat, seat, seat])
    assert out == [seat]


def test_famous_seats_hits_filters_non_famous() -> None:
    """Seats that don't match any hint are dropped (regression guard for the
    earlier wrong assumption that '机构专用' was famous)."""
    famous = "宁波桑田路证券营业部"
    out = _famous_seats_hits(["机构专用", famous, "某不知名营业部"])
    assert out == [famous]


def test_build_lhb_rollup_famous_seats_stable() -> None:
    """End-to-end: rollup.lhb_famous_seats stable for the same seat set."""
    seat_a = "东方财富证券拉萨团结路第二营业部"
    seat_b = "宁波桑田路证券营业部"
    top_inst_a = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "exalter": seat_a},
            {"ts_code": "000001.SZ", "exalter": seat_b},
        ]
    )
    top_inst_b = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "exalter": seat_b},
            {"ts_code": "000001.SZ", "exalter": seat_a},
        ]
    )
    rollup_a = _build_lhb_rollup(None, top_inst_a)
    rollup_b = _build_lhb_rollup(None, top_inst_b)
    assert rollup_a["000001.SZ"]["lhb_famous_seats"] == rollup_b["000001.SZ"]["lhb_famous_seats"]
