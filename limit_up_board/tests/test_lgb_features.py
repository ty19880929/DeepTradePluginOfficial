"""PR-1.1 — LightGBM 特征工程单元测试。

测试目标（lightgbm_iteration_plan.md §2.1）：
* 每类特征至少一个 happy path + 一个 NaN / edge case 用例
* 列数 / 列顺序与 FEATURE_NAMES 一致
* 同一输入两次调用幂等
* 比率 clip / 时间解析 / up_stat 解析等小函数边界覆盖
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from limit_up_board.data import SectorStrength
from limit_up_board.lgb.features import (
    FEATURE_NAMES,
    NA_FILLERS,
    SCHEMA_VERSION,
    FeatureSchemaMismatch,
    _clip_ratio,
    _days_between,
    _parse_time_to_seconds,
    _parse_up_stat,
    assert_columns,
    build_feature_frame,
    feature_missing_columns,
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_candidates(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_daily_history(
    ts_code: str,
    trade_date_end: str = "20260530",
    n: int = 30,
    base_close: float = 10.0,
    pct_chg_each: float = 0.5,
    high_premium: float = 0.4,
    amount_each: float = 5e5,  # 千元
) -> list[dict[str, Any]]:
    """Generate *n* sorted-ascending daily rows ending on trade_date_end.

    Closes drift up by pct_chg_each per day; high = close + premium; low = close - premium;
    pre_close = previous close.
    """
    from datetime import datetime, timedelta

    end = datetime.strptime(trade_date_end, "%Y%m%d")
    out: list[dict[str, Any]] = []
    prev_close = base_close
    close = base_close
    for i in range(n):
        d = end - timedelta(days=n - 1 - i)
        close = prev_close * (1 + pct_chg_each / 100.0)
        out.append(
            {
                "ts_code": ts_code,
                "trade_date": d.strftime("%Y%m%d"),
                "open": prev_close * 1.001,
                "high": close + high_premium,
                "low": close - high_premium,
                "close": close,
                "pre_close": prev_close,
                "pct_chg": (close - prev_close) / prev_close * 100,
                "amount": amount_each,  # 千元
                "vol": 12345,
            }
        )
        prev_close = close
    return out


def _make_daily_basic_history(
    ts_code: str,
    n: int = 30,
    turnover_rate_each: float = 2.0,
    volume_ratio_each: float = 1.5,
    circ_mv_wan: float = 5e6,  # 万元 → 5e10 元
) -> list[dict[str, Any]]:
    return [
        {
            "ts_code": ts_code,
            "trade_date": f"2026{i:04d}",
            "turnover_rate": turnover_rate_each,
            "volume_ratio": volume_ratio_each,
            "circ_mv": circ_mv_wan,
        }
        for i in range(n)
    ]


def _make_moneyflow_history(
    ts_code: str,
    n: int = 10,
    net_wan: float = 1500.0,
    buy_lg_wan: float = 800.0,
    buy_elg_wan: float = 500.0,
) -> list[dict[str, Any]]:
    return [
        {
            "ts_code": ts_code,
            "trade_date": f"2026{i:04d}",
            "net_mf_amount": net_wan,
            "buy_lg_amount": buy_lg_wan,
            "buy_elg_amount": buy_elg_wan,
        }
        for i in range(n)
    ]


def _full_candidate(
    ts_code: str = "000001.SZ",
    *,
    close: float = 11.0,
    fd_amount: float = 5e7,     # 元
    amount: float = 2e8,        # 元
    limit_amount: float = 4e8,  # 元
    float_mv: float = 5e9,      # 元 = 50 亿
    first_time: Any = "09:35:00",
    last_time: Any = "14:50:00",
    open_times: int = 1,
    limit_times: int = 1,
    up_stat: Any = "3/5",
    turnover_ratio: float = 4.5,
    pct_chg: float = 10.0,
    industry: str = "电子",
    industry_basic: str | None = "电子",
    list_date: str = "20180101",
    trade_date: str = "20260530",
    name: str = "Demo",
) -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "name": name,
        "trade_date": trade_date,
        "close": close,
        "fd_amount": fd_amount,
        "amount": amount,
        "limit_amount": limit_amount,
        "float_mv": float_mv,
        "first_time": first_time,
        "last_time": last_time,
        "open_times": open_times,
        "limit_times": limit_times,
        "up_stat": up_stat,
        "turnover_ratio": turnover_ratio,
        "pct_chg": pct_chg,
        "industry": industry,
        "industry_basic": industry_basic,
        "list_date": list_date,
    }


def _market_summary(*, total_lu: int = 80, max_height: int = 5) -> dict[str, Any]:
    return {
        "limit_up_count": total_lu,
        "limit_step_distribution": {str(max_height): 1, "1": 60, "2": 15, "3": 3, "4": 1},
        "limit_step_trend": {"high_board_delta": 1},
        "yesterday_failure_rate": {"rate_pct": 12.5},
        "yesterday_winners_today": {"continuation_rate_pct": 42.0},
    }


# ===========================================================================
# Tests
# ===========================================================================


class TestUtilHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("09:30:00", 9 * 3600 + 30 * 60),
            ("093000", 9 * 3600 + 30 * 60),
            ("14:55:01", 14 * 3600 + 55 * 60 + 1),
            ("145501", 14 * 3600 + 55 * 60 + 1),
            ("", None),
            ("nan", None),
            (None, None),
            ("99:99:99", None),
            ("garbage", None),
        ],
    )
    def test_parse_time_to_seconds(self, raw: Any, expected: int | None) -> None:
        out = _parse_time_to_seconds(raw)
        if expected is None:
            assert out is None
        else:
            assert out == float(expected)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("3/5", (3.0, 5.0)),
            ("1/2", (1.0, 2.0)),
            ("0/4", (0.0, 4.0)),
            ("foo", None),
            ("3", None),
            (None, None),
        ],
    )
    def test_parse_up_stat(self, raw: Any, expected: tuple[float, float] | None) -> None:
        assert _parse_up_stat(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            (10.0, 10.0),
            (1e9, 500.0),
            (-1e9, -500.0),
            (float("inf"), None),  # _to_float 不接受 inf → 走 NaN 路径
            (float("nan"), None),
        ],
    )
    def test_clip_ratio_bounds(self, raw: Any, expected: float | None) -> None:
        out = _clip_ratio(raw)
        if expected is None:
            assert out is None or (isinstance(out, float) and pd.isna(out))
        else:
            assert out == expected

    def test_days_between(self) -> None:
        assert _days_between("20260530", "20260520") == 10.0
        assert _days_between("20260530", "20180101") == (
            (pd.Timestamp("20260530") - pd.Timestamp("20180101")).days
        )
        assert _days_between("20260530", None) is None
        assert _days_between("20260530", "not-a-date") is None


class TestSchemaContract:
    def test_feature_names_unique_and_nonempty(self) -> None:
        assert len(FEATURE_NAMES) >= 45
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
        # 每个名字必须以 f_ 开头并属于已知 8 个一级分类
        prefixes = {"f_lim_", "f_vol_", "f_mom_", "f_mf_", "f_chip_", "f_lhb_", "f_sec_", "f_mkt_", "f_st_"}
        for n in FEATURE_NAMES:
            assert any(n.startswith(p) for p in prefixes), n

    def test_schema_version_int(self) -> None:
        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION >= 1

    def test_assert_columns_passthrough(self) -> None:
        df = pd.DataFrame({c: [0.0] for c in FEATURE_NAMES})
        assert_columns(df)  # 不抛

    def test_assert_columns_mismatch_raises(self) -> None:
        df = pd.DataFrame({c: [0.0] for c in FEATURE_NAMES[:-1]})  # 少一列
        with pytest.raises(FeatureSchemaMismatch):
            assert_columns(df)

    def test_na_fillers_contains_lhb_appeared(self) -> None:
        # design §4.3 — 唯一的显式编码"未发生"特征
        assert "f_lhb_appeared" in NA_FILLERS
        assert NA_FILLERS["f_lhb_appeared"] == 0.0


class TestBuildFeatureFrameHappyPath:
    @pytest.fixture
    def df_full(self) -> pd.DataFrame:
        cand = _full_candidate()
        candidates = _make_candidates([cand])
        return build_feature_frame(
            candidates_df=candidates,
            daily_by_code={cand["ts_code"]: _make_daily_history(cand["ts_code"])},
            daily_basic_by_code={cand["ts_code"]: _make_daily_basic_history(cand["ts_code"])},
            moneyflow_by_code={cand["ts_code"]: _make_moneyflow_history(cand["ts_code"])},
            cyq_by_code={
                cand["ts_code"]: {
                    "cyq_winner_pct": 72.0,
                    "cyq_top10_concentration": 65.0,
                    "cyq_close_to_avg_cost_pct": 8.0,
                }
            },
            lhb_rollup={
                cand["ts_code"]: {
                    "lhb_net_buy_yi": 0.85,
                    "lhb_inst_count": 3,
                    "lhb_famous_seats": ["拉萨团结路"],
                }
            },
            sector_strength=SectorStrength(source="limit_cpt_list", data={"top_sectors": []}),
            market_summary=_market_summary(),
            trade_date="20260530",
        )

    def test_columns_match_feature_names(self, df_full: pd.DataFrame) -> None:
        assert list(df_full.columns) == FEATURE_NAMES

    def test_index_is_ts_code(self, df_full: pd.DataFrame) -> None:
        assert df_full.index.name == "ts_code"
        assert list(df_full.index) == ["000001.SZ"]

    def test_limit_block_basic(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        assert row["f_lim_open_times"] == 1.0
        assert row["f_lim_limit_times"] == 1.0
        # 09:35:00 → 9*3600 + 35*60 = 34500
        assert row["f_lim_first_time_seconds"] == 9 * 3600 + 35 * 60
        # 14:50:00 → 14*3600 + 50*60 = 53400
        assert row["f_lim_last_time_seconds"] == 14 * 3600 + 50 * 60
        assert row["f_lim_first_to_last_seconds"] == (14 * 3600 + 50 * 60) - (9 * 3600 + 35 * 60)
        # fd/amount = 5e7 / 2e8 * 100 = 25%
        assert row["f_lim_fd_to_amount_pct"] == pytest.approx(25.0)
        # fd/float_mv = 5e7 / 5e9 * 100 = 1%
        assert row["f_lim_fd_to_float_mv_pct"] == pytest.approx(1.0)
        # limit_amount/float_mv = 4e8 / 5e9 * 100 = 8%
        assert row["f_lim_limit_amount_to_float_mv_pct"] == pytest.approx(8.0)
        # up_stat "3/5"
        assert row["f_lim_up_stat_consecutive"] == 3.0
        assert row["f_lim_up_stat_total_in_window"] == 5.0

    def test_vol_block_basic(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        # pct_chg 注入了 10.0
        assert row["f_vol_pct_chg_t"] == 10.0
        # turnover_ratio 注入 4.5
        assert row["f_vol_turnover_ratio"] == 4.5
        # turnover_rate_t 来自最后一个 daily_basic（2.0）
        assert row["f_vol_turnover_rate_t"] == 2.0
        # volume_ratio_t = 1.5
        assert row["f_vol_volume_ratio_t"] == 1.5
        # amount_yi_t: 5e5 千元 → 5e8 元 → 5 亿
        assert row["f_vol_amount_yi_t"] == pytest.approx(5.0)
        # 5d ratio：所有 amount 相同 → 比率 1.0
        assert row["f_vol_amount_ratio_5d"] == pytest.approx(1.0)
        # turnover_rate_ratio_5d：同上 → 1.0
        assert row["f_vol_turnover_rate_ratio_5d"] == pytest.approx(1.0)

    def test_mom_block_basic(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        # 30 个 daily 行，up_count_30d 在均匀温和上涨场景下 = 0
        assert row["f_mom_up_count_30d"] == 0.0
        # ma_bull_aligned: closes 单调递增；latest > ma5 > ma10 > ma20 应成立 → 1.0
        assert row["f_mom_ma_bull_aligned"] == 1.0
        # bias 应该都 > 0（上涨趋势）
        assert row["f_mom_close_to_ma5_bias"] > 0
        assert row["f_mom_close_to_ma10_bias"] > row["f_mom_close_to_ma5_bias"]
        assert row["f_mom_close_to_ma20_bias"] > row["f_mom_close_to_ma10_bias"]
        # high_to_close_pct_5d: 最近 5 天 high - close 都是 0.4
        assert row["f_mom_high_to_close_pct_5d"] > 0

    def test_mf_block_basic(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        # net_t_yi: 1500 万元 → 0.15 亿
        assert row["f_mf_net_t_yi"] == pytest.approx(0.15)
        # net_5d_sum_yi: 5 × 1500 万 = 7500 万 → 0.75 亿
        assert row["f_mf_net_5d_sum_yi"] == pytest.approx(0.75)
        # buy_lg_pct_t: 800 万元 × 1e4 = 8e6 元；除 amount_yuan(2e8 元) = 0.04 (4%)
        assert row["f_mf_buy_lg_pct_t"] == pytest.approx(0.04)
        # buy_elg_pct_t: 500 万元 × 1e4 = 5e6 元；除 2e8 = 0.025
        assert row["f_mf_buy_elg_pct_t"] == pytest.approx(0.025)
        # 全部为正 → consecutive_pos_days = 5（窗口上限）
        assert row["f_mf_net_consecutive_pos_days"] == 5.0

    def test_chip_block(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        assert row["f_chip_winner_pct"] == 72.0
        assert row["f_chip_top10_concentration"] == 65.0
        assert row["f_chip_close_to_avg_cost_pct"] == 8.0

    def test_lhb_block_present(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        assert row["f_lhb_appeared"] == 1.0
        assert row["f_lhb_net_buy_yi"] == 0.85
        assert row["f_lhb_inst_count"] == 3.0
        assert row["f_lhb_famous_seats_count"] == 1.0

    def test_sector_block(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        # source=limit_cpt_list → rank 1
        assert row["f_sec_strength_source_rank"] == 1.0
        # 当批只有 1 只 candidate，industry='电子' → up_count = 1
        assert row["f_sec_today_industry_up_count"] == 1.0
        # ratio: 1/1 = 1
        assert row["f_sec_today_industry_up_ratio"] == pytest.approx(1.0)

    def test_market_block(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        assert row["f_mkt_total_limit_up"] == 80.0
        assert row["f_mkt_max_height"] == 5.0
        assert row["f_mkt_yesterday_failure_rate"] == 12.5
        assert row["f_mkt_yesterday_continuation_rate"] == 42.0
        assert row["f_mkt_high_board_delta"] == 1.0

    def test_static_block(self, df_full: pd.DataFrame) -> None:
        row = df_full.iloc[0]
        # float_mv = 5e9 元 → 50 亿
        assert row["f_st_float_mv_yi"] == pytest.approx(50.0)
        assert row["f_st_close_yuan"] == 11.0
        # 上市天数：list_date=20180101, T=20260530
        assert row["f_st_listed_days"] == _days_between("20260530", "20180101")


class TestEdgeAndNaN:
    def test_no_history_yields_nan_blocks(self) -> None:
        cand = _full_candidate(ts_code="000002.SZ")
        df = build_feature_frame(
            candidates_df=_make_candidates([cand]),
            daily_by_code={},
            daily_basic_by_code={},
            moneyflow_by_code={},
            cyq_by_code={},
            lhb_rollup={},  # 未上榜
            sector_strength=SectorStrength(source="industry_fallback", data={"top_sectors": []}),
            market_summary={},
            trade_date="20260530",
        )
        row = df.iloc[0]
        # 历史缺失：动量 / 量价大半 / 资金流 / 筹码 → NaN
        assert pd.isna(row["f_mom_close_to_ma5_bias"])
        assert pd.isna(row["f_mom_up_count_30d"])
        assert pd.isna(row["f_vol_amount_yi_t"])
        assert pd.isna(row["f_mf_net_t_yi"])
        assert pd.isna(row["f_chip_winner_pct"])
        # 但 candidate 自带字段仍在
        assert row["f_lim_open_times"] == 1.0
        assert row["f_st_close_yuan"] == 11.0
        # 龙虎榜：未上榜 → appeared=0、其余 NaN
        assert row["f_lhb_appeared"] == 0.0
        assert pd.isna(row["f_lhb_net_buy_yi"])
        # sector source = industry_fallback → rank 3
        assert row["f_sec_strength_source_rank"] == 3.0

    def test_invalid_first_time_yields_nan(self) -> None:
        cand = _full_candidate(first_time="not-a-time", last_time=None)
        df = build_feature_frame(
            candidates_df=_make_candidates([cand]),
            daily_by_code={},
            daily_basic_by_code={},
            moneyflow_by_code={},
            cyq_by_code={},
            lhb_rollup={},
            sector_strength=None,
            market_summary={},
            trade_date="20260530",
        )
        row = df.iloc[0]
        assert pd.isna(row["f_lim_first_time_seconds"])
        assert pd.isna(row["f_lim_last_time_seconds"])
        assert pd.isna(row["f_lim_first_to_last_seconds"])

    def test_zero_amount_yields_nan_for_ratios(self) -> None:
        """amount=0 → 比率不应炸成 inf；走 NaN 路径。"""
        cand = _full_candidate(amount=0.0)
        df = build_feature_frame(
            candidates_df=_make_candidates([cand]),
            daily_by_code={cand["ts_code"]: _make_daily_history(cand["ts_code"])},
            daily_basic_by_code={cand["ts_code"]: _make_daily_basic_history(cand["ts_code"])},
            moneyflow_by_code={cand["ts_code"]: _make_moneyflow_history(cand["ts_code"])},
            cyq_by_code={},
            lhb_rollup={},
            sector_strength=None,
            market_summary={},
            trade_date="20260530",
        )
        row = df.iloc[0]
        assert pd.isna(row["f_lim_fd_to_amount_pct"])
        # daily.amount 仍存在，所以 f_vol_amount_yi_t 不是 NaN
        assert row["f_vol_amount_yi_t"] == pytest.approx(5.0)

    def test_extreme_ratio_clipped(self) -> None:
        # float_mv=1 元、fd_amount=1e9 元 → 比率超大，应 clip 到 500
        cand = _full_candidate(float_mv=1.0, fd_amount=1e9)
        df = build_feature_frame(
            candidates_df=_make_candidates([cand]),
            daily_by_code={},
            daily_basic_by_code={},
            moneyflow_by_code={},
            cyq_by_code={},
            lhb_rollup={},
            sector_strength=None,
            market_summary={},
            trade_date="20260530",
        )
        row = df.iloc[0]
        assert row["f_lim_fd_to_float_mv_pct"] == 500.0

    def test_industry_aggregation_two_industries(self) -> None:
        cand_a = _full_candidate(ts_code="A.SZ", industry="电子", industry_basic="电子")
        cand_b = _full_candidate(ts_code="B.SZ", industry="电子", industry_basic="电子")
        cand_c = _full_candidate(ts_code="C.SZ", industry="医药", industry_basic="医药")
        df = build_feature_frame(
            candidates_df=_make_candidates([cand_a, cand_b, cand_c]),
            daily_by_code={},
            daily_basic_by_code={},
            moneyflow_by_code={},
            cyq_by_code={},
            lhb_rollup={},
            sector_strength=SectorStrength(source="lu_desc_aggregation", data={}),
            market_summary={},
            trade_date="20260530",
        )
        # 两只电子 + 一只医药
        a_row = df.loc["A.SZ"]
        b_row = df.loc["B.SZ"]
        c_row = df.loc["C.SZ"]
        assert a_row["f_sec_today_industry_up_count"] == 2.0
        assert b_row["f_sec_today_industry_up_count"] == 2.0
        assert c_row["f_sec_today_industry_up_count"] == 1.0
        # rank = 2 (lu_desc_aggregation)
        assert a_row["f_sec_strength_source_rank"] == 2.0

    def test_idempotent_under_repeat(self) -> None:
        cand = _full_candidate()
        kwargs: dict[str, Any] = {
            "candidates_df": _make_candidates([cand]),
            "daily_by_code": {cand["ts_code"]: _make_daily_history(cand["ts_code"])},
            "daily_basic_by_code": {cand["ts_code"]: _make_daily_basic_history(cand["ts_code"])},
            "moneyflow_by_code": {cand["ts_code"]: _make_moneyflow_history(cand["ts_code"])},
            "cyq_by_code": {},
            "lhb_rollup": {},
            "sector_strength": SectorStrength(source="limit_cpt_list", data={"top_sectors": []}),
            "market_summary": _market_summary(),
            "trade_date": "20260530",
        }
        a = build_feature_frame(**kwargs)
        b = build_feature_frame(**kwargs)
        pd.testing.assert_frame_equal(a, b)

    def test_feature_missing_columns_helper(self) -> None:
        cand = _full_candidate(ts_code="X.SZ")
        df = build_feature_frame(
            candidates_df=_make_candidates([cand]),
            daily_by_code={},
            daily_basic_by_code={},
            moneyflow_by_code={},
            cyq_by_code={},
            lhb_rollup={},
            sector_strength=None,
            market_summary={},
            trade_date="20260530",
        )
        miss = feature_missing_columns(df.iloc[0])
        # 至少 ma_bias / 资金流 / 筹码列在 miss 中
        assert "f_mom_close_to_ma5_bias" in miss
        assert "f_mf_net_t_yi" in miss
        assert "f_chip_winner_pct" in miss
        # 但 f_lhb_appeared 一定不会在 miss 里（被 NA_FILLERS 兜底 0）
        assert "f_lhb_appeared" not in miss

    def test_missing_ts_code_column_raises(self) -> None:
        with pytest.raises(ValueError, match="ts_code"):
            build_feature_frame(
                candidates_df=pd.DataFrame([{"close": 10.0}]),
                daily_by_code={},
                daily_basic_by_code={},
                moneyflow_by_code={},
                cyq_by_code={},
                lhb_rollup={},
                trade_date="20260530",
            )

    def test_missing_trade_date_raises(self) -> None:
        df_no_td = pd.DataFrame([{"ts_code": "A.SZ"}])
        with pytest.raises(ValueError, match="trade_date"):
            build_feature_frame(
                candidates_df=df_no_td,
                daily_by_code={},
                daily_basic_by_code={},
                moneyflow_by_code={},
                cyq_by_code={},
                lhb_rollup={},
            )
