"""Label construction for LightGBM training (设计 §5).

Labels come straight out of ``va_realized_returns`` — VA's existing T+N
backfill table — so the training pipeline performs **zero** additional Tushare
calls. ``evaluate`` populates the table; ``labels.fetch_labels_for_date``
just JOINs on (anomaly_date, ts_code) and applies the configured threshold.

The three supported sources (per design §5.1):

* ``max_ret_5d`` — T+1..T+5 任一日收盘价相对 T 日的最大涨幅 (default; matches
  the strategy's "main-launch" semantics).
* ``ret_t3``    — T+3 收盘价相对 T 日涨幅 (continuity-flavoured label).
* ``max_ret_10d``— 10 日窗口最大涨幅 (looser, more positives).

Threshold is in percent (5% → ``threshold_pct=5.0``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


VALID_LABEL_SOURCES = ("max_ret_5d", "ret_t3", "max_ret_10d")
VALID_DATA_STATUS = ("complete", "partial")


class LgbLabelError(ValueError):
    """Configuration error: bad label_source or non-positive threshold."""


def _validate_args(*, source: str, threshold_pct: float) -> None:
    if source not in VALID_LABEL_SOURCES:
        raise LgbLabelError(
            f"label_source must be one of {VALID_LABEL_SOURCES}, got {source!r}"
        )
    if threshold_pct <= 0:
        raise LgbLabelError(
            f"threshold_pct must be > 0, got {threshold_pct}"
        )


def fetch_labels_for_date(
    db: Database,
    *,
    anomaly_date: str,
    source: str = "max_ret_5d",
    threshold_pct: float = 5.0,
) -> pd.Series:
    """Return ``{ts_code: int label}`` for one anomaly_date.

    Empty Series when the table has no rows for that date (or all rows are
    ``data_status='pending'`` / NULL on the requested source).
    """
    _validate_args(source=source, threshold_pct=threshold_pct)
    rows = db.fetchall(
        f"SELECT ts_code, {source} FROM va_realized_returns "
        f"WHERE anomaly_date = ? AND data_status IN ('complete', 'partial') "
        f"AND {source} IS NOT NULL",
        (anomaly_date,),
    )
    return _rows_to_label_series(rows, threshold_pct=threshold_pct)


def fetch_labels_for_window(
    db: Database,
    *,
    start_date: str,
    end_date: str,
    source: str = "max_ret_5d",
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """Bulk variant: returns DataFrame[anomaly_date, ts_code, label].

    Useful for ``dataset.collect_training_window`` which needs labels across
    many anomaly_dates and can JOIN them onto the feature frame in one go.
    """
    _validate_args(source=source, threshold_pct=threshold_pct)
    rows = db.fetchall(
        f"SELECT anomaly_date, ts_code, {source} FROM va_realized_returns "
        f"WHERE anomaly_date BETWEEN ? AND ? "
        f"AND data_status IN ('complete', 'partial') "
        f"AND {source} IS NOT NULL",
        (start_date, end_date),
    )
    if not rows:
        return pd.DataFrame(columns=["anomaly_date", "ts_code", "label"])
    data = []
    for r in rows:
        anomaly_date, ts_code, value = r
        if value is None:
            continue
        label = 1 if float(value) >= threshold_pct else 0
        data.append((str(anomaly_date), str(ts_code), label))
    return pd.DataFrame(data, columns=["anomaly_date", "ts_code", "label"])


def _rows_to_label_series(
    rows: list[tuple], *, threshold_pct: float
) -> pd.Series:
    if not rows:
        return pd.Series(dtype="int8", name="label")
    codes: list[str] = []
    labels: list[int] = []
    for ts_code, value in rows:
        if value is None:
            continue
        codes.append(str(ts_code))
        labels.append(1 if float(value) >= threshold_pct else 0)
    return pd.Series(labels, index=codes, dtype="int8", name="label")


__all__ = [
    "LgbLabelError",
    "VALID_DATA_STATUS",
    "VALID_LABEL_SOURCES",
    "fetch_labels_for_date",
    "fetch_labels_for_window",
]
