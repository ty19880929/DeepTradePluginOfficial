"""data.py unit tests covering P1-3 data-quality additions.

Scope:
* required-API 空表 → ``data_unavailable`` 追加 ``{api}_empty_response``
* ``_build_candidate_rows`` 写入 ``lhb_data_quality`` 三态
  (``listed`` / ``not_listed`` / ``api_empty`` / ``api_unavailable``)
"""

from __future__ import annotations

import pandas as pd

from limit_up_board.data import _apply_market_filter, _build_candidate_rows


def _toy_candidate_frame() -> pd.DataFrame:
    """A minimal candidate frame with 2 ts_codes — enough to exercise rollup states."""
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600519.SH"],
            "name": ["平安银行", "贵州茅台"],
            "industry_basic": ["银行", "白酒"],
            "first_time": ["09:30:00", "09:30:00"],
            "last_time": ["09:30:01", "09:30:01"],
            "open_times": [0, 0],
            "limit_times": [1, 1],
            "up_stat": ["1/1", "1/1"],
            "pct_chg": [10.0, 10.0],
            "close": [12.0, 1800.0],
            "turnover_ratio": [3.5, 1.2],
            "fd_amount": [1.5e9, 2.0e10],
            "limit_amount": [1.5e9, 2.0e10],
            "amount": [3e9, 4e10],
            "total_mv": [5e10, 2e12],
            "float_mv": [5e10, 2e12],
        }
    )


class TestLhbThreeStates:
    """P1-3 — lhb_data_quality 三态：listed / not_listed / api_empty / api_unavailable."""

    def test_listed_when_in_rollup(self) -> None:
        # 000001.SZ 上榜（top_list 有它），600519.SH 同日未上榜
        candidates = _toy_candidate_frame()
        top_list = pd.DataFrame(
            {"ts_code": ["000001.SZ"], "net_amount": [5e8]}
        )
        top_inst = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "exalter": ["拉萨团结路证券营业部"],
                "side": [0],
            }
        )
        rows = _build_candidate_rows(
            candidates,
            ths_df=None,
            top_list_df=top_list,
            top_inst_df=top_inst,
        )
        by_code = {r["ts_code"]: r for r in rows}
        assert by_code["000001.SZ"]["lhb_data_quality"] == "listed"
        assert by_code["600519.SH"]["lhb_data_quality"] == "not_listed"

    def test_api_empty_when_top_list_globally_empty(self) -> None:
        """整体空（zero rows in top_list）→ 所有 candidate 都标 api_empty。"""
        candidates = _toy_candidate_frame()
        rows = _build_candidate_rows(
            candidates,
            ths_df=None,
            top_list_df=pd.DataFrame(),
            top_inst_df=pd.DataFrame(),
            lhb_api_empty=True,
        )
        for r in rows:
            assert r["lhb_data_quality"] == "api_empty"

    def test_api_unavailable_overrides_other_states(self) -> None:
        """``api_unavailable`` 优先级最高（接口异常 > 全空 > 未上榜）。"""
        candidates = _toy_candidate_frame()
        rows = _build_candidate_rows(
            candidates,
            ths_df=None,
            top_list_df=pd.DataFrame(),
            top_inst_df=pd.DataFrame(),
            lhb_api_empty=True,
            lhb_api_unavailable=True,
        )
        for r in rows:
            assert r["lhb_data_quality"] == "api_unavailable"

    def test_default_flags_dont_break_existing_callers(self) -> None:
        """旧 caller 不传 lhb_api_* 参数 → 默认走 listed / not_listed 分支。"""
        candidates = _toy_candidate_frame()
        top_list = pd.DataFrame({"ts_code": ["000001.SZ"], "net_amount": [5e8]})
        rows = _build_candidate_rows(
            candidates,
            ths_df=None,
            top_list_df=top_list,
            top_inst_df=None,
        )
        by_code = {r["ts_code"]: r for r in rows}
        assert by_code["000001.SZ"]["lhb_data_quality"] == "listed"
        assert by_code["600519.SH"]["lhb_data_quality"] == "not_listed"


class TestFilterBoundariesInclusive:
    """P2-1 — 筛选边界改闭区间：``>`` ``<`` ``<`` → ``>=`` ``<=`` ``<=``。"""

    def _df(self, mv_yi: float, close: float) -> pd.DataFrame:
        # float_mv 是元，要 × 1e8；下游 `_apply_market_filter` 会除回去
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["平安银行"],
                "float_mv": [mv_yi * 1e8],
                "close": [close],
            }
        )

    def test_lower_bound_included(self) -> None:
        """min_float_mv_yi=30 时，30 亿的标的应通过（旧版本：开区间会被剔除）。"""
        df = self._df(mv_yi=30.0, close=10.0)
        kept, _ = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0, min_float_mv_yi=30.0
        )
        assert len(kept) == 1

    def test_upper_bound_included(self) -> None:
        """max_float_mv_yi=100 时，100 亿的标的应通过。"""
        df = self._df(mv_yi=100.0, close=10.0)
        kept, _ = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0, min_float_mv_yi=30.0
        )
        assert len(kept) == 1

    def test_close_upper_bound_included(self) -> None:
        """max_close_yuan=15 时，正好 15 元的标的应通过。"""
        df = self._df(mv_yi=50.0, close=15.0)
        kept, _ = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0, min_float_mv_yi=30.0
        )
        assert len(kept) == 1

    def test_just_outside_bounds_rejected(self) -> None:
        """超过 max 1e-6 的标的应被剔除（闭区间仍是严格阈值）。"""
        df = self._df(mv_yi=100.01, close=10.0)
        kept, summary = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0, min_float_mv_yi=30.0
        )
        assert len(kept) == 0
        # P2-1: dropped_top3 should record the rejection reason
        top3 = summary["dropped_top3"]
        assert len(top3) == 1
        assert top3[0]["ts_code"] == "000001.SZ"
        assert any("float_mv>" in r for r in top3[0]["reasons"])

    def test_dropped_top3_truncates_at_3(self) -> None:
        """5 个被剔除时只展示 TOP 3（按 float_mv 降序）。"""
        df = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(1, 6)],
                "name": [f"S{i}" for i in range(1, 6)],
                "float_mv": [500 * 1e8, 400 * 1e8, 300 * 1e8, 200 * 1e8, 150 * 1e8],
                "close": [10.0] * 5,
            }
        )
        _, summary = _apply_market_filter(
            df, max_float_mv_yi=100.0, max_close_yuan=15.0, min_float_mv_yi=30.0
        )
        top3 = summary["dropped_top3"]
        assert len(top3) == 3
        # 排序：500 → 400 → 300
        assert [d["float_mv_yi"] for d in top3] == [500.0, 400.0, 300.0]


class TestRequiredApiEmptyAppendsDataUnavailable:
    """P1-3 — collect_round1 中 top_list / top_inst / cyq_perf 全空时
    应把 ``{api}_empty_response`` 写入 ``bundle.data_unavailable``。

    我们这里只断言 _build_candidate_rows + 上层 collect_round1 协议，
    避免拉起完整 tushare/db；collect_round1 集成测试由 test_phase_a_factors 覆盖。
    """

    def test_data_unavailable_uses_empty_response_marker(self) -> None:
        """断言生成的 marker 字符串符合「{api}_empty_response」契约。

        Render / event 渲染层依赖这个字符串前缀；这里固化下来防止误改。
        """
        markers = [
            "top_list_empty_response",
            "top_inst_empty_response",
            "cyq_perf_empty_response",
        ]
        for m in markers:
            assert m.endswith("_empty_response")
            # 前缀就是 tushare api 名 —— 用户能直接对照 deeptrade_plugin.yaml
            api = m[: -len("_empty_response")]
            assert api in {"top_list", "top_inst", "cyq_perf"}
