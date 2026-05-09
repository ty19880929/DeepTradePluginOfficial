"""Phase B — LHB + cyq_perf derived-factor unit tests.

Covers:
    B1 — _famous_seats_hits / _build_lhb_rollup (top_list+top_inst → per-ts_code)
    B2 — _build_cyq_lookup / _cyq_concentration / _close_to_avg_cost_pct

The "未上榜不是数据缺失" semantic is asserted at the rollup boundary.
"""

from __future__ import annotations

import pandas as pd

from limit_up_board.data import (
    FAMOUS_SEATS_HINTS,
    _build_cyq_lookup,
    _build_lhb_rollup,
    _close_to_avg_cost_pct,
    _cyq_concentration,
    _famous_seats_hits,
)

# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------


class TestFamousSeatsHits:
    def test_substring_match_case_insensitive(self) -> None:
        seats = ["华泰证券厦门厦禾路营业部", "中信建投上海某营业部"]
        hits = _famous_seats_hits(seats)
        assert "华泰证券厦门厦禾路营业部" in hits
        assert len(hits) == 1

    def test_dedup(self) -> None:
        seats = ["拉萨团结路东路证券营业部", "拉萨团结路东路证券营业部"]
        hits = _famous_seats_hits(seats)
        assert hits == ["拉萨团结路东路证券营业部"]

    def test_no_match(self) -> None:
        assert _famous_seats_hits(["东吴证券苏州古亭路", "中泰证券济南"]) == []

    def test_skips_non_string(self) -> None:
        # Defensive: pandas may produce NaN or numeric in exalter slot
        assert _famous_seats_hits([None, 123, "拉萨团结路总部"]) == ["拉萨团结路总部"]  # type: ignore[list-item]

    def test_whitelist_size(self) -> None:
        # Spec: ~15 entries first version
        assert 10 <= len(FAMOUS_SEATS_HINTS) <= 25


class TestBuildLhbRollup:
    def test_full_rollup(self) -> None:
        top_list = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600519.SH"],
                "net_amount": [5e8, -3e8],  # 元 → yi after normalize
            }
        )
        top_inst = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ", "600519.SH"],
                "exalter": [
                    "拉萨团结路证券营业部",  # famous
                    "中泰证券济南",  # not famous
                    "深圳益田路荣超商务中心证券营业部",  # famous
                ],
                "side": [0, 0, 1],
            }
        )
        rollup = _build_lhb_rollup(top_list, top_inst)
        assert "000001.SZ" in rollup
        assert rollup["000001.SZ"]["lhb_net_buy_yi"] == 5.0
        assert rollup["000001.SZ"]["lhb_inst_count"] == 2
        assert rollup["000001.SZ"]["lhb_famous_seats"] == ["拉萨团结路证券营业部"]
        assert rollup["600519.SH"]["lhb_net_buy_yi"] == -3.0
        assert rollup["600519.SH"]["lhb_inst_count"] == 1

    def test_empty_inputs(self) -> None:
        # candidate didn't make top_list → 未上榜，rollup 不含其 ts_code
        rollup = _build_lhb_rollup(pd.DataFrame(), pd.DataFrame())
        assert rollup == {}

    def test_none_inputs(self) -> None:
        assert _build_lhb_rollup(None, None) == {}

    def test_top_list_only_no_inst(self) -> None:
        top_list = pd.DataFrame({"ts_code": ["000001.SZ"], "net_amount": [1e8]})
        rollup = _build_lhb_rollup(top_list, None)
        assert rollup["000001.SZ"]["lhb_net_buy_yi"] == 1.0
        assert "lhb_inst_count" not in rollup["000001.SZ"]


# ---------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------


class TestCyqConcentration:
    def test_tight_chips(self) -> None:
        # spread = 5% → concentration = 95
        assert _cyq_concentration(cost_5=9.75, cost_95=10.25, weight_avg=10.0) == 95.0

    def test_wide_chips(self) -> None:
        # spread = 50% → concentration = 50
        assert _cyq_concentration(cost_5=7.5, cost_95=12.5, weight_avg=10.0) == 50.0

    def test_clipped_at_zero(self) -> None:
        # spread > 100% → clipped to 0
        assert _cyq_concentration(cost_5=1.0, cost_95=20.0, weight_avg=10.0) == 0.0

    def test_none_returns_none(self) -> None:
        assert _cyq_concentration(None, 1.0, 1.0) is None
        assert _cyq_concentration(1.0, None, 1.0) is None
        assert _cyq_concentration(1.0, 1.0, None) is None
        assert _cyq_concentration(1.0, 1.0, 0) is None


class TestCloseToAvgCostPct:
    def test_positive(self) -> None:
        assert _close_to_avg_cost_pct(11.0, 10.0) == 10.0

    def test_negative(self) -> None:
        assert _close_to_avg_cost_pct(8.5, 10.0) == -15.0

    def test_zero_avg(self) -> None:
        assert _close_to_avg_cost_pct(10.0, 0) is None

    def test_none(self) -> None:
        assert _close_to_avg_cost_pct(None, 10.0) is None
        assert _close_to_avg_cost_pct(10.0, None) is None


class TestBuildCyqLookup:
    def test_full(self) -> None:
        df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "weight_avg": [10.0],
                "winner_rate": [65.5],
                "cost_5pct": [9.0],
                "cost_15pct": [9.5],
                "cost_50pct": [10.0],
                "cost_85pct": [10.5],
                "cost_95pct": [11.0],
            }
        )
        lookup = _build_cyq_lookup(df)
        assert lookup["000001.SZ"]["cyq_winner_pct"] == 65.5
        assert lookup["000001.SZ"]["cyq_avg_cost_yuan"] == 10.0
        # spread = 20% → concentration = 80
        assert lookup["000001.SZ"]["cyq_top10_concentration"] == 80.0

    def test_empty_or_none(self) -> None:
        assert _build_cyq_lookup(None) == {}
        assert _build_cyq_lookup(pd.DataFrame()) == {}

    def test_missing_columns_handled(self) -> None:
        df = pd.DataFrame({"ts_code": ["X"], "weight_avg": [10.0]})
        # Other cols missing → cyq_top10_concentration = None, winner = None
        lookup = _build_cyq_lookup(df)
        assert lookup["X"]["cyq_winner_pct"] is None
        assert lookup["X"]["cyq_avg_cost_yuan"] == 10.0
        assert lookup["X"]["cyq_top10_concentration"] is None
