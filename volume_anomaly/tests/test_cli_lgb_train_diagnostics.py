"""Tests for the `lgb train` under-threshold diagnostics helper.

Covers v0.8.1: when collected labeled samples fall below
``lgb_train_min_samples``, ``_emit_under_threshold_diagnostics`` must
distinguish three states and emit unambiguous next-step guidance instead of
the old generic "先运行 evaluate 回填" line that collided with the
``lgb evaluate`` subcommand name.

Three states:
    A. ``va_anomaly_history`` empty in window      → tell user to run screen / analyze
    B. anomaly rows exist but realized_returns empty / unusable
                                                   → tell user to run `volume-anomaly evaluate`
                                                     (and disambiguate from `lgb evaluate`)
    C. realized rows partially usable but still under threshold
                                                   → tell user T+N data still ramping up
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptrade.core.db import Database

from volume_anomaly.cli import _emit_under_threshold_diagnostics


# ---------------------------------------------------------------------------
# Fixture: a fresh DB with the two tables the helper queries.
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "lgb_train_diag.duckdb")
    db.execute(
        """
        CREATE TABLE va_anomaly_history (
            trade_date          VARCHAR NOT NULL,
            ts_code             VARCHAR NOT NULL,
            name                VARCHAR,
            industry            VARCHAR,
            pct_chg             DOUBLE,
            close               DOUBLE,
            open                DOUBLE,
            high                DOUBLE,
            low                 DOUBLE,
            vol                 DOUBLE,
            amount              DOUBLE,
            body_ratio          DOUBLE,
            turnover_rate       DOUBLE,
            vol_ratio_5d        DOUBLE,
            max_vol_60d         DOUBLE,
            raw_metrics_json    VARCHAR,
            PRIMARY KEY (trade_date, ts_code)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE va_realized_returns (
            anomaly_date  VARCHAR NOT NULL,
            ts_code       VARCHAR NOT NULL,
            t_close       DOUBLE,
            t1_close      DOUBLE,
            t3_close      DOUBLE,
            t5_close      DOUBLE,
            t10_close     DOUBLE,
            ret_t1        DOUBLE,
            ret_t3        DOUBLE,
            ret_t5        DOUBLE,
            ret_t10       DOUBLE,
            max_close_5d  DOUBLE,
            max_close_10d DOUBLE,
            max_ret_5d    DOUBLE,
            max_ret_10d   DOUBLE,
            max_dd_5d     DOUBLE,
            computed_at   TIMESTAMP,
            data_status   VARCHAR NOT NULL,
            PRIMARY KEY (anomaly_date, ts_code)
        )
        """
    )
    yield db
    db.close()


def _insert_anomaly(db: Database, *, trade_date: str, ts_code: str) -> None:
    db.execute(
        "INSERT INTO va_anomaly_history "
        "(trade_date, ts_code, name, industry, pct_chg, close, open, high, "
        "low, vol, amount, body_ratio, turnover_rate, vol_ratio_5d, max_vol_60d) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trade_date, ts_code, f"Co_{ts_code}", "测试",
            7.0, 12.0, 11.5, 12.5, 11.0, 200000.0, 250000.0,
            0.72, 8.5, 2.1, 250000.0,
        ),
    )


def _insert_realized(
    db: Database,
    *,
    anomaly_date: str,
    ts_code: str,
    max_ret_5d: float | None,
    data_status: str,
) -> None:
    db.execute(
        "INSERT INTO va_realized_returns "
        "(anomaly_date, ts_code, t_close, max_ret_5d, ret_t3, max_ret_10d, "
        "data_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (anomaly_date, ts_code, 12.0, max_ret_5d, max_ret_5d, max_ret_5d, data_status),
    )


# ---------------------------------------------------------------------------
# Branch A — va_anomaly_history empty in window
# ---------------------------------------------------------------------------


def test_branch_a_empty_anomaly_history(empty_db: Database, capsys) -> None:
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20250101",
        end="20260513",
        label_source="max_ret_5d",
        n_labeled=0,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "labeled samples = 0 < lgb_train_min_samples=1000" in out
    assert "anomaly_dates=0" in out
    assert "realized_rows=0" in out
    assert "va_anomaly_history 无异动记录" in out
    assert "screen" in out and "analyze" in out
    # Must NOT recommend evaluate in this state (would be misleading).
    assert "volume-anomaly evaluate" not in out


def test_branch_a_anomalies_outside_window(empty_db: Database, capsys) -> None:
    # Anomaly exists, but outside the requested window → still branch A.
    _insert_anomaly(empty_db, trade_date="20240101", ts_code="000001.SZ")
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20250101",
        end="20260513",
        label_source="max_ret_5d",
        n_labeled=0,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "anomaly_dates=0" in out
    assert "va_anomaly_history 无异动记录" in out


# ---------------------------------------------------------------------------
# Branch B — anomalies present but realized_returns has no usable rows
# ---------------------------------------------------------------------------


def test_branch_b_anomalies_no_realized_rows(empty_db: Database, capsys) -> None:
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000002.SZ")
    _insert_anomaly(empty_db, trade_date="20260602", ts_code="000001.SZ")
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20260101",
        end="20260630",
        label_source="max_ret_5d",
        n_labeled=0,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "anomaly_dates=2" in out
    assert "realized_rows=0 (其中可用=0)" in out
    assert "va_realized_returns 缺 T+N 实现收益" in out
    assert "deeptrade volume-anomaly evaluate" in out
    assert "--backfill-all" in out
    # Critical: must disambiguate from the lgb subcommand name collision.
    assert "不是 `lgb evaluate`" in out


def test_branch_b_all_pending_or_null(empty_db: Database, capsys) -> None:
    # Rows exist but max_ret_5d is NULL OR data_status='pending' — count as
    # unusable, still branch B.
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260601", ts_code="000001.SZ",
        max_ret_5d=None, data_status="pending",
    )
    _insert_anomaly(empty_db, trade_date="20260602", ts_code="000002.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260602", ts_code="000002.SZ",
        max_ret_5d=None, data_status="complete",
    )
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20260101",
        end="20260630",
        label_source="max_ret_5d",
        n_labeled=0,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "anomaly_dates=2" in out
    assert "realized_rows=2 (其中可用=0)" in out
    assert "deeptrade volume-anomaly evaluate" in out


# ---------------------------------------------------------------------------
# Branch C — usable realized rows exist but still under threshold
# ---------------------------------------------------------------------------


def test_branch_c_partial_realized_below_threshold(empty_db: Database, capsys) -> None:
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260601", ts_code="000001.SZ",
        max_ret_5d=6.0, data_status="complete",
    )
    _insert_anomaly(empty_db, trade_date="20260602", ts_code="000002.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260602", ts_code="000002.SZ",
        max_ret_5d=2.0, data_status="partial",
    )
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20260101",
        end="20260630",
        label_source="max_ret_5d",
        n_labeled=2,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "labeled samples = 2 < lgb_train_min_samples=1000" in out
    assert "anomaly_dates=2" in out
    assert "realized_rows=2 (其中可用=2)" in out
    assert "T+N 数据未就绪 或窗口太短" in out
    # Branch C must NOT push the user back to `evaluate` as the primary fix.
    assert "deeptrade volume-anomaly evaluate --backfill-all" not in out


def test_label_source_swap_uses_alternate_column(empty_db: Database, capsys) -> None:
    """`label_source=ret_t3` should be honored when counting usable rows."""
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
    # max_ret_5d is filled but ret_t3 is NULL → unusable for ret_t3 → branch B.
    empty_db.execute(
        "INSERT INTO va_realized_returns "
        "(anomaly_date, ts_code, t_close, max_ret_5d, ret_t3, max_ret_10d, "
        "data_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("20260601", "000001.SZ", 12.0, 6.5, None, 7.0, "complete"),
    )
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20260101",
        end="20260630",
        label_source="ret_t3",
        n_labeled=0,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "realized_rows=1 (其中可用=0)" in out
    assert "deeptrade volume-anomaly evaluate" in out
