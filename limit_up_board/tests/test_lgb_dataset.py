"""PR-1.2 — collect_training_window 集成测试。

用 stub TushareClient 跑 5 个交易日的训练数据收集，验证：
* 样本数 = Σ 每日 candidate 数
* 列数 = len(FEATURE_NAMES)
* 标签分布合理（含 1 / 0 / <NA>，<NA> 仅出现在 T+1 daily 缺失的天）
* sample_index / split_groups 字段齐全
* filter_labeled 行为
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest

from limit_up_board.calendar import TradeCalendar
from limit_up_board.lgb.dataset import (
    LgbDataset,
    _enumerate_trade_dates,
    collect_training_window,
)
from limit_up_board.lgb.features import FEATURE_NAMES, SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Stub TushareClient — minimal interface used by dataset.py
# ---------------------------------------------------------------------------


class FakeTushareClient:
    """Returns pre-baked DataFrames per (api_name, key)."""

    def __init__(self, fixtures: dict[tuple[str, str], pd.DataFrame]) -> None:
        # key = (api_name, cache_key) where cache_key is trade_date or "start:end" or "*"
        self._fixtures = fixtures

    def call(
        self,
        api_name: str,
        *,
        trade_date: str | None = None,
        params: dict[str, Any] | None = None,
        fields: str | None = None,  # noqa: ARG002
        force_sync: bool = False,  # noqa: ARG002
    ) -> pd.DataFrame:
        params = dict(params or {})
        if trade_date is not None:
            key = str(trade_date)
        elif "trade_date" in params:
            key = str(params["trade_date"])
        elif "start_date" in params and "end_date" in params:
            key = f"{params['start_date']}:{params['end_date']}"
        else:
            key = "*"
        # default empty when fixture missing — matches tushare optional-API fallthrough
        return self._fixtures.get((api_name, key), pd.DataFrame()).copy()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


_TRADE_DATES = ["20260520", "20260521", "20260522", "20260525", "20260526"]
# T+1 of last day → "20260527"
_NEXT_OF_LAST = "20260527"
_ALL_DATES_INCL_T1 = _TRADE_DATES + [_NEXT_OF_LAST]


def _trade_cal_df(open_dates: Iterable[str], full_range_days: int = 20) -> pd.DataFrame:
    """Build a trade_cal frame: every day in May 20..29 is_open=1 (synthetic)."""
    base = pd.Timestamp("20260518")
    rows = []
    for i in range(full_range_days):
        d = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
        rows.append({"exchange": "SSE", "cal_date": d, "is_open": 1, "pretrade_date": None})
    df = pd.DataFrame(rows)
    open_set = set(open_dates)
    df["is_open"] = df["cal_date"].apply(lambda d: 1 if d in open_set else 0)
    # fill pretrade_date as the prior open cal_date
    last_open: str | None = None
    pretrades: list[str | None] = []
    for d, is_open in zip(df["cal_date"], df["is_open"], strict=False):
        pretrades.append(last_open)
        if is_open == 1:
            last_open = d
    df["pretrade_date"] = pretrades
    return df


def _stock_basic_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "Demo A",
                "market": "主板",
                "exchange": "SZSE",
                "industry": "电子",
                "list_date": "20180101",
                "list_status": "L",
            },
            {
                "ts_code": "600002.SH",
                "name": "Demo B",
                "market": "主板",
                "exchange": "SSE",
                "industry": "医药",
                "list_date": "20190101",
                "list_status": "L",
            },
            # 创业板 + 已退市，应被 main_board_filter 过滤掉
            {
                "ts_code": "300003.SZ",
                "name": "Demo C",
                "market": "创业板",
                "exchange": "SZSE",
                "industry": "电子",
                "list_date": "20200101",
                "list_status": "L",
            },
        ]
    )


def _limit_list_d_for(trade_date: str) -> pd.DataFrame:
    """Two main-board candidates limit-up on each day."""
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ts_code": "000001.SZ",
                "name": "Demo A",
                "industry": "电子",
                "close": 11.0,
                "pct_chg": 10.0,
                "amount": 2e8,
                "fd_amount": 5e7,
                "limit_amount": 4e8,
                "float_mv": 5e9,
                "total_mv": 8e9,
                "turnover_ratio": 4.2,
                "first_time": "09:35:00",
                "last_time": "14:50:00",
                "open_times": 1,
                "limit_times": 1,
                "up_stat": "2/3",
                "limit": "U",
            },
            {
                "trade_date": trade_date,
                "ts_code": "600002.SH",
                "name": "Demo B",
                "industry": "医药",
                "close": 9.0,
                "pct_chg": 10.0,
                "amount": 1.5e8,
                "fd_amount": 3e7,
                "limit_amount": 3e8,
                "float_mv": 4e9,
                "total_mv": 6e9,
                "turnover_ratio": 3.0,
                "first_time": "10:15:00",
                "last_time": "11:30:00",
                "open_times": 0,
                "limit_times": 1,
                "up_stat": "1/1",
                "limit": "U",
            },
        ]
    )


def _daily_window_df(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate daily rows for 000001.SZ + 600002.SH between start and end (inclusive)."""
    pd_start = pd.Timestamp(start_date)
    pd_end = pd.Timestamp(end_date)
    if pd_start > pd_end:
        return pd.DataFrame()
    days = pd.date_range(pd_start, pd_end, freq="D")
    rows = []
    for ts_code, base in (("000001.SZ", 10.0), ("600002.SH", 8.0)):
        close = base
        for d in days:
            pre = close
            close = pre * 1.005
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": d.strftime("%Y%m%d"),
                    "open": pre * 1.001,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "pre_close": pre,
                    "pct_chg": (close - pre) / pre * 100,
                    "amount": 5e5,
                    "vol": 1234,
                }
            )
    return pd.DataFrame(rows)


def _daily_t1_df(trade_date: str) -> pd.DataFrame:
    """Single-day daily frame for T+1 label lookup.

    Day chosen produces label=1 for A and label=0 for B (heterogeneous).
    """
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 11.5,
                "high": 12.21,        # (12.21-11)/11*100 = 11% → label 1
                "low": 11.0,
                "close": 12.0,
                "pre_close": 11.0,
                "pct_chg": 9.09,
                "amount": 6e5,
                "vol": 1500,
            },
            {
                "ts_code": "600002.SH",
                "trade_date": trade_date,
                "open": 9.0,
                "high": 9.5,           # (9.5-9)/9*100 = 5.55% → label 0
                "low": 8.8,
                "close": 9.3,
                "pre_close": 9.0,
                "pct_chg": 3.33,
                "amount": 4e5,
                "vol": 900,
            },
        ]
    )


def _daily_basic_window_df(start_date: str, end_date: str) -> pd.DataFrame:
    pd_start = pd.Timestamp(start_date)
    pd_end = pd.Timestamp(end_date)
    if pd_start > pd_end:
        return pd.DataFrame()
    days = pd.date_range(pd_start, pd_end, freq="D")
    rows = []
    for ts_code in ("000001.SZ", "600002.SH"):
        for d in days:
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": d.strftime("%Y%m%d"),
                    "turnover_rate": 2.0,
                    "volume_ratio": 1.4,
                    "circ_mv": 5e6,
                }
            )
    return pd.DataFrame(rows)


def _moneyflow_window_df(start_date: str, end_date: str) -> pd.DataFrame:
    pd_start = pd.Timestamp(start_date)
    pd_end = pd.Timestamp(end_date)
    if pd_start > pd_end:
        return pd.DataFrame()
    days = pd.date_range(pd_start, pd_end, freq="D")
    rows = []
    for ts_code in ("000001.SZ", "600002.SH"):
        for d in days:
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": d.strftime("%Y%m%d"),
                    "net_mf_amount": 1500.0,
                    "buy_lg_amount": 800.0,
                    "buy_elg_amount": 500.0,
                }
            )
    return pd.DataFrame(rows)


def _build_fixtures() -> dict[tuple[str, str], pd.DataFrame]:
    f: dict[tuple[str, str], pd.DataFrame] = {}

    # stock_basic — single static frame
    f[("stock_basic", "*")] = _stock_basic_df()

    # per-day limit_list_d + stock_st (empty) + cyq_perf (empty) + step / ths / cpt / lhb
    for T in _TRADE_DATES:
        f[("limit_list_d", T)] = _limit_list_d_for(T)
        # stock_st empty → no exclusions
        f[("stock_st", T)] = pd.DataFrame(columns=["ts_code"])
        f[("cyq_perf", T)] = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": T,
                    "winner_rate": 72.0,
                    "weight_avg": 10.3,
                    "cost_5pct": 9.0,
                    "cost_95pct": 11.5,
                },
                {
                    "ts_code": "600002.SH",
                    "trade_date": T,
                    "winner_rate": 60.0,
                    "weight_avg": 8.5,
                    "cost_5pct": 8.0,
                    "cost_95pct": 9.0,
                },
            ]
        )
        f[("top_list", T)] = pd.DataFrame()
        f[("top_inst", T)] = pd.DataFrame()
        f[("limit_list_ths", T)] = pd.DataFrame()
        f[("limit_cpt_list", T)] = pd.DataFrame()
        f[("limit_step", T)] = pd.DataFrame(
            [
                {"trade_date": T, "ts_code": "000001.SZ", "nums": 2},
                {"trade_date": T, "ts_code": "600002.SH", "nums": 1},
            ]
        )
        # T+1 daily for label lookup
        try:
            t1 = _next_day(T)
        except Exception:
            t1 = None
        if t1 is not None:
            f[("daily", t1)] = _daily_t1_df(t1)

    # Yesterday context — _collect_yesterday_context calls limit_step(prev) /
    # limit_list_d(prev) / daily(trade_date=T). Reuse the same fixtures we already
    # have; for the first T (20260520) prev = 20260519 (synthetic non-open day in
    # the trade_cal we build) — those lookups return empty, which is fine.
    for T in _TRADE_DATES:
        f.setdefault(("daily", T), _daily_t1_df(T))  # context daily(T) — single-day

    # Daily / daily_basic / moneyflow window queries — collect_day_samples uses
    # start_date / end_date range (cache_key = "start:end"). To keep the fixture
    # table small we just memoize one big window covering the full month.
    big_start = "20260301"
    big_end = "20260601"
    daily_full = _daily_window_df(big_start, big_end)
    daily_basic_full = _daily_basic_window_df(big_start, big_end)
    moneyflow_full = _moneyflow_window_df(big_start, big_end)

    for T in _TRADE_DATES:
        # mirror the cache_key computation in collect_day_samples._shift_yyyymmdd
        from limit_up_board.lgb.dataset import _shift_yyyymmdd as _shift

        daily_start = _shift(T, -60)  # daily_lookback * 2 = 30*2 = 60
        mf_start = _shift(T, -10)
        f[("daily", f"{daily_start}:{T}")] = daily_full[
            (daily_full["trade_date"] >= daily_start)
            & (daily_full["trade_date"] <= T)
        ]
        f[("daily_basic", f"{daily_start}:{T}")] = daily_basic_full[
            (daily_basic_full["trade_date"] >= daily_start)
            & (daily_basic_full["trade_date"] <= T)
        ]
        f[("moneyflow", f"{mf_start}:{T}")] = moneyflow_full[
            (moneyflow_full["trade_date"] >= mf_start)
            & (moneyflow_full["trade_date"] <= T)
        ]

    return f


def _next_day(yyyymmdd: str) -> str:
    return (pd.Timestamp(yyyymmdd) + pd.Timedelta(days=1)).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calendar() -> TradeCalendar:
    return TradeCalendar(_trade_cal_df(_ALL_DATES_INCL_T1))


@pytest.fixture(scope="module")
def tushare() -> FakeTushareClient:
    return FakeTushareClient(_build_fixtures())


class TestEnumerateTradeDates:
    def test_returns_open_only(self, calendar: TradeCalendar) -> None:
        days = _enumerate_trade_dates(calendar, "20260520", "20260526")
        assert days == _TRADE_DATES

    def test_empty_window(self, calendar: TradeCalendar) -> None:
        assert _enumerate_trade_dates(calendar, "20260530", "20260520") == []


class TestTradeCalendarRange:
    """Direct coverage for the new TradeCalendar.range() public API (P3-1)."""

    def test_inclusive_endpoints(self, calendar: TradeCalendar) -> None:
        days = calendar.range("20260520", "20260526")
        assert days == _TRADE_DATES
        # First/last open day in the window must be present (闭区间)
        assert days[0] == _TRADE_DATES[0]
        assert days[-1] == _TRADE_DATES[-1]

    def test_excludes_closed_days(self, calendar: TradeCalendar) -> None:
        # 20260523 (周六) / 20260524 (周日) 在 _ALL_DATES_INCL_T1 标 is_open=0；不应出现
        days = calendar.range("20260520", "20260526")
        assert "20260523" not in days
        assert "20260524" not in days

    def test_inverted_window_returns_empty(self, calendar: TradeCalendar) -> None:
        assert calendar.range("20260530", "20260520") == []


class TestCollectTrainingWindow:
    @pytest.fixture(scope="class")
    def dataset(
        self,
        tushare: FakeTushareClient,
        calendar: TradeCalendar,
    ) -> LgbDataset:
        return collect_training_window(
            tushare=tushare,
            calendar=calendar,
            start_date="20260520",
            end_date="20260526",
        )

    def test_n_samples_matches_per_day_candidates(self, dataset: LgbDataset) -> None:
        # 5 days × 2 candidates per day = 10 samples
        assert dataset.n_samples == 10

    def test_feature_matrix_shape_and_columns(self, dataset: LgbDataset) -> None:
        assert dataset.feature_matrix.shape == (10, len(FEATURE_NAMES))
        assert list(dataset.feature_matrix.columns) == FEATURE_NAMES

    def test_schema_version_attached(self, dataset: LgbDataset) -> None:
        assert dataset.schema_version == SCHEMA_VERSION

    def test_sample_index_columns(self, dataset: LgbDataset) -> None:
        assert list(dataset.sample_index.columns) == [
            "ts_code",
            "trade_date",
            "next_trade_date",
            "pct_chg_t1",
        ]
        # ts_code 取值是预期两只股票
        assert set(dataset.sample_index["ts_code"]) == {"000001.SZ", "600002.SH"}

    def test_split_groups_match_trade_date(self, dataset: LgbDataset) -> None:
        # 5 个不同的 split_group 值 = 5 个交易日
        unique_groups = set(int(g) for g in dataset.split_groups.dropna())
        assert unique_groups == {int(d) for d in _TRADE_DATES}

    def test_label_distribution(self, dataset: LgbDataset) -> None:
        # A 总是 label=1, B 总是 label=0；5 天 × 2 = 5 个 1，5 个 0；无 <NA>
        # （最后一天 20260526 的 T+1 = 20260527 也有 fixture）
        labels = dataset.labels
        assert labels.notna().all()
        ones = int((labels == 1).sum())
        zeros = int((labels == 0).sum())
        assert ones == 5
        assert zeros == 5

    def test_filter_labeled_returns_same_when_no_missing(self, dataset: LgbDataset) -> None:
        filt = dataset.filter_labeled()
        assert filt.n_samples == dataset.n_samples
        assert filt.n_labeled == dataset.n_labeled

    def test_n_positive_helper(self, dataset: LgbDataset) -> None:
        assert dataset.n_positive == 5

    def test_progress_callback_invoked(
        self,
        tushare: FakeTushareClient,
        calendar: TradeCalendar,
    ) -> None:
        events: list[tuple[str, int, int]] = []
        collect_training_window(
            tushare=tushare,
            calendar=calendar,
            start_date="20260520",
            end_date="20260526",
            on_day=lambda T, n, cum: events.append((T, n, cum)),
        )
        assert [e[0] for e in events] == _TRADE_DATES
        assert [e[1] for e in events] == [2, 2, 2, 2, 2]
        assert [e[2] for e in events] == [2, 4, 6, 8, 10]


class TestEmptyWindow:
    def test_empty_when_no_open_days(
        self,
        tushare: FakeTushareClient,
        calendar: TradeCalendar,
    ) -> None:
        # 选一段全部 is_open=0 的窗口（trade_cal 默认所有日子都有，但只有 _TRADE_DATES open）
        ds = collect_training_window(
            tushare=tushare,
            calendar=calendar,
            start_date="20260518",
            end_date="20260519",
        )
        assert ds.n_samples == 0
        assert ds.feature_matrix.empty
        assert list(ds.feature_matrix.columns) == FEATURE_NAMES  # schema 仍保留

    def test_invalid_window_raises(
        self,
        tushare: FakeTushareClient,
        calendar: TradeCalendar,
    ) -> None:
        with pytest.raises(ValueError, match="start_date"):
            collect_training_window(
                tushare=tushare,
                calendar=calendar,
                start_date="20260601",
                end_date="20260520",
            )


class TestMissingT1Label:
    def test_label_is_na_when_t1_daily_empty(
        self,
        calendar: TradeCalendar,
    ) -> None:
        # Build fixtures lacking T+1 daily for 20260526 (last day)
        fx = _build_fixtures()
        last_t1 = _next_day("20260526")
        # 移除最后一天 T+1 的 daily fixture
        fx.pop(("daily", last_t1), None)
        ts = FakeTushareClient(fx)
        ds = collect_training_window(
            tushare=ts,
            calendar=calendar,
            start_date="20260526",
            end_date="20260526",
        )
        assert ds.n_samples == 2
        # 缺 T+1 daily → 两个样本的 label 都是 <NA>
        assert ds.labels.isna().all()
        # filter_labeled 应该过滤掉所有样本
        filt = ds.filter_labeled()
        assert filt.n_samples == 0
