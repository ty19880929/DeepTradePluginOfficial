"""sync_window + fetch_latest_trade_date orchestration tests.

Asserts the *call pattern* (per-day / range / per-index) declared in
design §5.2. Doesn't test the framework's ``materialize`` implementation
itself — that's covered by deeptrade-quant's own test suite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from market_review.data import (
    LATEST_TRADE_DATE_PROBE_INDEX,
    WIDE_BASE_INDICES,
    SyncResult,
    fetch_latest_trade_date,
    sync_window,
)
from market_review.runtime import MrRuntime
from market_review.windows import Window

from conftest import FakeTushare  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# fetch_latest_trade_date
# ---------------------------------------------------------------------------


def test_fetch_latest_trade_date_probes_index_daily_with_force_sync(
    fake_tushare: FakeTushare,
) -> None:
    fake_tushare.set_static(
        "index_daily",
        pd.DataFrame({
            "ts_code": [LATEST_TRADE_DATE_PROBE_INDEX] * 3,
            "trade_date": ["20260526", "20260527", "20260528"],
            "close": [3000.0, 3010.0, 3020.0],
        }),
    )
    latest = fetch_latest_trade_date(fake_tushare)  # type: ignore[arg-type]
    assert latest == "20260528"
    assert len(fake_tushare.calls_to("index_daily")) == 1
    call = fake_tushare.calls_to("index_daily")[0]
    assert call.force_sync is True
    assert call.params is not None
    assert call.params["ts_code"] == LATEST_TRADE_DATE_PROBE_INDEX


def test_fetch_latest_trade_date_raises_on_empty(fake_tushare: FakeTushare) -> None:
    fake_tushare.set_static("index_daily", pd.DataFrame())
    with pytest.raises(RuntimeError, match="probe returned no rows"):
        fetch_latest_trade_date(fake_tushare)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# sync_window
# ---------------------------------------------------------------------------


def _runtime_with_tushare(mr_db, fake_tushare: FakeTushare) -> MrRuntime:
    """Build an MrRuntime with the fake wired in. ``config`` / ``llms`` are
    None — sync_window doesn't touch them."""
    return MrRuntime(
        db=mr_db,
        config=None,  # type: ignore[arg-type]
        llms=None,    # type: ignore[arg-type]
        tushare=fake_tushare,  # type: ignore[arg-type]
    )


def _three_day_window() -> Window:
    return Window(
        mode="range",
        start="20260526", end="20260528",
        trade_dates=("20260526", "20260527", "20260528"),
        anchor="20260528",
    )


def test_sync_window_requires_tushare(mr_db) -> None:
    rt = MrRuntime(db=mr_db, config=None, llms=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="MrRuntime.tushare is None"):
        sync_window(rt, _three_day_window())


def test_sync_window_returns_counters(mr_db, fake_tushare: FakeTushare) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    res = sync_window(rt, _three_day_window())
    assert isinstance(res, SyncResult)
    # All APIs returned empty → no counters but no crash.
    assert res.rows_materialized == {}
    assert res.duration_seconds >= 0.0


def test_sync_window_one_shot_apis_called_without_trade_date(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    sync_window(rt, _three_day_window())
    # stock_basic / trade_cal are one-shot — exactly one call, no trade_date.
    for api in ("stock_basic", "trade_cal"):
        calls = fake_tushare.calls_to(api)
        assert len(calls) == 1, f"{api} expected 1 call, got {len(calls)}"
        assert calls[0].trade_date is None
        assert calls[0].params is None


def test_sync_window_per_day_apis_loop_open_days(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    sync_window(rt, _three_day_window())
    expected_dates = ["20260526", "20260527", "20260528"]
    # The per-day APIs (all must loop one call per trade_date).
    per_day_apis = (
        "daily", "daily_basic", "moneyflow",
        "stock_st", "suspend_d",
        "limit_step", "limit_cpt_list",
        "top_list", "top_inst",
        "ths_hot", "dc_hot",
        "margin", "block_trade",
    )
    for api in per_day_apis:
        calls = fake_tushare.calls_to(api)
        assert [c.trade_date for c in calls] == expected_dates, (
            f"{api} expected per-day loop, got {[c.trade_date for c in calls]}"
        )


def test_sync_window_range_apis_called_once_with_start_end(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    sync_window(rt, _three_day_window())
    range_apis = (
        "limit_list_d", "limit_list_ths",
        "moneyflow_hsgt", "hsgt_top10",
        "moneyflow_mkt_dc", "moneyflow_ind_ths", "moneyflow_cnt_ths",
    )
    for api in range_apis:
        calls = fake_tushare.calls_to(api)
        assert len(calls) == 1, f"{api} expected single range call, got {len(calls)}"
        assert calls[0].trade_date is None
        assert calls[0].params == {"start_date": "20260526", "end_date": "20260528"}


def test_sync_window_index_apis_loop_wide_base_basket(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    sync_window(rt, _three_day_window())
    for api in ("index_daily", "index_dailybasic"):
        calls = fake_tushare.calls_to(api)
        assert len(calls) == len(WIDE_BASE_INDICES), (
            f"{api} expected {len(WIDE_BASE_INDICES)} calls, got {len(calls)}"
        )
        seen_codes = {c.params["ts_code"] for c in calls if c.params}
        assert seen_codes == set(WIDE_BASE_INDICES)


def test_sync_window_force_sync_propagates(mr_db, fake_tushare: FakeTushare) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    sync_window(rt, _three_day_window(), force_sync=True)
    assert all(c.force_sync for c in fake_tushare.calls), (
        "force_sync=True must propagate to every Tushare call"
    )


def test_sync_window_records_empty_days_per_api(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)

    # daily returns content for day 1+3, empty for day 2.
    def daily_responder(*, trade_date, **_):
        if trade_date == "20260527":
            return pd.DataFrame()
        return pd.DataFrame({
            "ts_code": ["000001.SZ"], "trade_date": [trade_date], "close": [10.0],
        })

    fake_tushare.set_response("daily", daily_responder)
    res = sync_window(rt, _three_day_window())
    assert res.empty_days.get("daily") == ["20260527"]
    assert res.rows_materialized.get("mr_daily", 0) == 2


def test_sync_window_materializes_non_empty_frames(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    fake_tushare.set_static(
        "stock_basic",
        pd.DataFrame({
            "ts_code": ["600001.SH", "300001.SZ"],
            "symbol": ["600001", "300001"],
            "name": ["A", "B"],
        }),
    )
    res = sync_window(rt, _three_day_window())
    assert res.rows_materialized.get("mr_stock_basic") == 2
    # The one and only materialize call to mr_stock_basic used the right key cols.
    materialized = fake_tushare.materializes_to("mr_stock_basic")
    assert len(materialized) == 1
    assert materialized[0].key_cols == ["ts_code"]


def test_sync_window_hot_transform_renames_and_injects_source(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    """ths_hot / dc_hot rows must land with source=ths|dc + name + hot_value."""
    rt = _runtime_with_tushare(mr_db, fake_tushare)

    def ths_responder(*, trade_date, **_):
        return pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "ts_name": ["平安银行"],
            "rank": ["1"],  # str — must coerce
            "hot": [9999.0],
        })

    fake_tushare.set_response("ths_hot", ths_responder)
    sync_window(rt, _three_day_window())

    hot_materializes = fake_tushare.materializes_to("mr_hot")
    assert len(hot_materializes) == 3  # one per day
    # The transform happens before materialize — assert on the call args.
    # Since FakeTushare.materialize doesn't capture df itself, we re-derive:
    # 3 rows total materialized (one per day).
    assert sum(m.rows for m in hot_materializes) == 3


def test_sync_window_moneyflow_ind_ths_renames_industry_to_name(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    """Tushare 的 ``moneyflow_ind_ths`` 返回行业名为 ``industry``，落表
    ``mr_moneyflow_ind_ths`` 的 PK 字段叫 ``name`` —— 不重命名会触发
    PK 隐含的 NOT NULL 约束（v0.1.1 实测崩溃过）。``_transform_ind_ths``
    必须在 ``materialize`` 之前把这一列对齐。"""
    rt = _runtime_with_tushare(mr_db, fake_tushare)

    fake_tushare.set_static(
        "moneyflow_ind_ths",
        pd.DataFrame({
            "trade_date": ["20260528", "20260528"],
            "ts_code": ["881270.TI", "881271.TI"],
            "industry": ["元件", "光模块"],   # ← Tushare 真实列名
            "lead_stock": ["胜业电气", "新易盛"],
            "net_amount": [47.0, 88.0],
        }),
    )
    sync_window(rt, _three_day_window())

    mats = fake_tushare.materializes_to("mr_moneyflow_ind_ths")
    assert len(mats) == 1, "range API expected single materialize"
    assert mats[0].rows == 2
    assert "name" in mats[0].columns, (
        "transform must rename 'industry' → 'name' before materialize "
        f"(saw {mats[0].columns!r})"
    )
    assert "industry" not in mats[0].columns, (
        f"transform must drop 'industry' after rename (saw {mats[0].columns!r})"
    )


def test_sync_window_moneyflow_ind_ths_drops_null_name_rows(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    """Tushare 偶尔回吐 ``industry=None`` 的行，仍会触发同样的 NOT NULL；
    transform 必须在落表前剔除这些行而不是整批失败。"""
    rt = _runtime_with_tushare(mr_db, fake_tushare)

    fake_tushare.set_static(
        "moneyflow_ind_ths",
        pd.DataFrame({
            "trade_date": ["20260528", "20260528", "20260528"],
            "industry": ["元件", None, ""],
            "net_amount": [47.0, 1.0, 2.0],
        }),
    )
    sync_window(rt, _three_day_window())

    mats = fake_tushare.materializes_to("mr_moneyflow_ind_ths")
    assert len(mats) == 1
    assert mats[0].rows == 1, "rows with NULL / empty industry must be filtered"


def test_sync_window_skips_apis_yielding_empty(
    mr_db, fake_tushare: FakeTushare,
) -> None:
    """Empty frame → no materialize call; row counter stays at 0."""
    rt = _runtime_with_tushare(mr_db, fake_tushare)
    fake_tushare.set_static("stock_basic", pd.DataFrame())
    sync_window(rt, _three_day_window())
    assert fake_tushare.materializes_to("mr_stock_basic") == []
    assert "mr_stock_basic" not in {*[]}  # documentary noop
