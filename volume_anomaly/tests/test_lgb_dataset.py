"""PR-1.2 — :func:`collect_training_window` end-to-end with stub Tushare.

Exercises the historical-replay loop over five anomaly_dates with a tiny
stub TushareClient and seeded ``va_anomaly_history`` / ``va_realized_returns``.

Coverage targets:
    * VaLgbDataset shape (sample count, feature column count)
    * Label distribution (mix of 0 / 1 / <NA> for pending rows)
    * sample_index columns (ts_code / anomaly_date / max_ret_5d / data_status)
    * filter_labeled drops <NA> rows
    * Sample order is anomaly_date-ascending

Shared fixtures (``stub_tushare`` / ``db_with_anomalies`` /
``isolated_plugin_data_dir`` / ``anomaly_dates``) live in
``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

from deeptrade.core.db import Database
from volume_anomaly.calendar import TradeCalendar
from volume_anomaly.lgb.checkpoint import META_COLUMNS
from volume_anomaly.lgb.dataset import (
    VaLgbDataset,
    collect_training_window,
    enumerate_anomaly_dates,
)
from volume_anomaly.lgb.features import FEATURE_NAMES

from conftest import FakeTushareClient, trade_cal_df  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# enumerate_anomaly_dates
# ---------------------------------------------------------------------------


def test_enumerate_anomaly_dates(
    db_with_anomalies: Database, anomaly_dates: list[str]
) -> None:
    dates = enumerate_anomaly_dates(
        db_with_anomalies, start_date="20260601", end_date="20260612"
    )
    assert dates == anomaly_dates


def test_enumerate_anomaly_dates_excludes_outside_window(
    db_with_anomalies: Database,
) -> None:
    dates = enumerate_anomaly_dates(
        db_with_anomalies, start_date="20260610", end_date="20260611"
    )
    assert dates == ["20260610", "20260611"]


# ---------------------------------------------------------------------------
# collect_training_window — end-to-end
# ---------------------------------------------------------------------------


def test_collect_training_window_basic_shape(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
) -> None:
    cal = TradeCalendar(trade_cal_df())
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=cal,
        start_date="20260601",
        end_date="20260612",
        label_source="max_ret_5d",
        label_threshold_pct=5.0,
        daily_lookback=5,
    )
    assert isinstance(ds, VaLgbDataset)
    # 9 samples seeded across 5 anomaly_dates; all have stub history → 9 rows.
    assert ds.n_samples == 9
    assert list(ds.feature_matrix.columns) == FEATURE_NAMES


def test_collect_training_window_label_distribution(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
) -> None:
    cal = TradeCalendar(trade_cal_df())
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=cal,
        start_date="20260601",
        end_date="20260612",
        label_source="max_ret_5d",
        label_threshold_pct=5.0,
        daily_lookback=5,
    )
    # pending row (20260611, 000001.SZ) → <NA>; others labeled.
    assert ds.n_labeled == 8
    # Positive labels: 6.5/5.0/8.0/9.5/6.0 ≥ 5 → 5 positives.
    assert ds.n_positive == 5


def test_filter_labeled_drops_pending(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
) -> None:
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=TradeCalendar(trade_cal_df()),
        start_date="20260601",
        end_date="20260612",
        label_source="max_ret_5d",
        label_threshold_pct=5.0,
        daily_lookback=5,
    )
    labeled = ds.filter_labeled()
    assert labeled.n_samples == 8
    assert labeled.labels.notna().all()


def test_sample_index_carries_required_columns(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
    anomaly_dates: list[str],
) -> None:
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=TradeCalendar(trade_cal_df()),
        start_date="20260601",
        end_date="20260612",
        daily_lookback=5,
    )
    assert set(ds.sample_index.columns) == set(META_COLUMNS)
    dates = set(ds.sample_index["anomaly_date"].astype(str).tolist())
    assert dates.issubset(set(anomaly_dates))


def test_threshold_8pct_drops_marginal_positives(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
) -> None:
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=TradeCalendar(trade_cal_df()),
        start_date="20260601",
        end_date="20260612",
        label_threshold_pct=8.0,
        daily_lookback=5,
    )
    # 8% threshold: only 8.0 / 9.5 qualify → 2 positives.
    assert ds.n_positive == 2


def test_label_source_ret_t3_uses_correct_column(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
) -> None:
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=TradeCalendar(trade_cal_df()),
        start_date="20260601",
        end_date="20260612",
        label_source="ret_t3",
        label_threshold_pct=5.0,
        daily_lookback=5,
    )
    # Fixture sets ret_t3 == max_ret_5d → same positive count under same threshold.
    assert ds.n_positive == 5


def test_empty_window_returns_empty_dataset(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
) -> None:
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=TradeCalendar(trade_cal_df()),
        start_date="20260101",
        end_date="20260105",
        daily_lookback=5,
    )
    assert ds.n_samples == 0
    assert list(ds.feature_matrix.columns) == FEATURE_NAMES


def test_dataset_metadata_records_label_config(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
    anomaly_dates: list[str],
) -> None:
    ds = collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=TradeCalendar(trade_cal_df()),
        start_date="20260601",
        end_date="20260612",
        label_threshold_pct=5.0,
        label_source="max_ret_5d",
        daily_lookback=5,
    )
    assert ds.label_threshold_pct == 5.0
    assert ds.label_source == "max_ret_5d"
    assert ds.anomaly_dates == anomaly_dates


def test_running_twice_is_idempotent_via_checkpoint(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
) -> None:
    cal = TradeCalendar(trade_cal_df())
    ds1 = collect_training_window(
        tushare=stub_tushare, db=db_with_anomalies, calendar=cal,
        start_date="20260601", end_date="20260612", daily_lookback=5,
    )
    ds2 = collect_training_window(
        tushare=stub_tushare, db=db_with_anomalies, calendar=cal,
        start_date="20260601", end_date="20260612", daily_lookback=5,
    )
    assert ds1.n_samples == ds2.n_samples
    a = ds1.sample_index[["ts_code", "anomaly_date"]].apply(tuple, axis=1).tolist()
    b = ds2.sample_index[["ts_code", "anomaly_date"]].apply(tuple, axis=1).tolist()
    assert sorted(a) == sorted(b)


def test_progress_callback_invoked_per_anomaly_date(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
    stub_tushare: FakeTushareClient,
    anomaly_dates: list[str],
) -> None:
    seen: list[tuple[str, int, int]] = []

    def cb(T: str, n: int, cum: int) -> None:
        seen.append((T, n, cum))

    collect_training_window(
        tushare=stub_tushare,
        db=db_with_anomalies,
        calendar=TradeCalendar(trade_cal_df()),
        start_date="20260601",
        end_date="20260612",
        daily_lookback=5,
        on_day=cb,
    )
    dates_seen = [t for t, _, _ in seen]
    assert dates_seen == anomaly_dates
    cums = [c for _, _, c in seen]
    assert cums == sorted(cums)
