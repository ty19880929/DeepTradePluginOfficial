"""Tests for the `lgb train` under-threshold diagnostics helper.

Covers v0.8.3: when collected labeled samples fall below
``lgb_train_min_samples``, ``_emit_under_threshold_diagnostics`` must
distinguish FOUR states and emit unambiguous next-step guidance. The v0.8.2
version collapsed "no rows yet" and "rows exist but all pending" into one
branch, which sent users into a loop ("run evaluate" → still pending → "run
evaluate" → …) when T+N trade days simply hadn't elapsed yet.

Four states:
    A.  ``va_anomaly_history`` empty in window     → run screen / analyze
    B1. anomaly rows exist, realized_returns empty → run `volume-anomaly evaluate`
                                                     (disambiguate from `lgb evaluate`)
    B2. realized rows present, some/all ``pending``→ wait for T+N OR accumulate
                                                     earlier anomaly_dates
                                                     (must NOT recommend evaluate)
    B3. realized rows present, none pending, label
        column NULL on every row                   → data gap / wrong label_source
    C.  usable rows exist but still under threshold→ expand window / accumulate
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
# Branch B1 — anomalies present, va_realized_returns completely empty
# ---------------------------------------------------------------------------


def test_branch_b1_anomalies_no_realized_rows(empty_db: Database, capsys) -> None:
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
    assert "va_realized_returns 在窗口内无任何行" in out
    assert "deeptrade volume-anomaly evaluate" in out
    assert "--backfill-all" in out
    # Critical: must disambiguate from the lgb subcommand name collision.
    assert "不是 `lgb evaluate`" in out


# ---------------------------------------------------------------------------
# Branch B2 — rows exist but data_status='pending' (T+N not elapsed yet).
# Critical regression: v0.8.2 sent these users back to `evaluate` in a loop.
# ---------------------------------------------------------------------------


def test_branch_b2_all_pending_does_not_recommend_evaluate(
    empty_db: Database, capsys
) -> None:
    # Two pending rows on a single recent trade_date — exactly the user's
    # bug-report shape (just smaller). Re-running evaluate cannot help.
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260601", ts_code="000001.SZ",
        max_ret_5d=None, data_status="pending",
    )
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000002.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260601", ts_code="000002.SZ",
        max_ret_5d=None, data_status="pending",
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
    assert "anomaly_dates=1" in out
    assert "realized_rows=2 (其中可用=0)" in out
    assert "data_status='pending'" in out
    assert "最近异动日=20260601" in out
    # Must NOT loop the user back into evaluate — that's the bug we fixed.
    assert "deeptrade volume-anomaly evaluate --backfill-all" not in out
    # Should surface the actionable alternatives.
    assert "screen" in out
    assert "更早" in out


def test_branch_b2_regression_seven_pending_one_date(
    empty_db: Database, capsys
) -> None:
    """Exact shape of the user-reported v0.8.2 bug: 7 rows pending on a single
    anomaly_date with today close to the anomaly_date so T+N can't be reached.
    `evaluate --backfill-all --force-recompute` already ran (rows are there)
    yet the diagnostic told the user to run it again. Must not happen again."""
    for i in range(1, 8):
        code = f"00000{i}.SZ"
        _insert_anomaly(empty_db, trade_date="20260513", ts_code=code)
        _insert_realized(
            empty_db, anomaly_date="20260513", ts_code=code,
            max_ret_5d=None, data_status="pending",
        )
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20250101",
        end="20260513",
        label_source="max_ret_5d",
        n_labeled=0,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "realized_rows=7 (其中可用=0)" in out
    assert "7/7 行 data_status='pending'" in out
    assert "最近异动日=20260513" in out
    # The loop-causing recommendation must be absent.
    assert "deeptrade volume-anomaly evaluate --backfill-all" not in out


def test_branch_b2_mixed_pending_and_complete_still_warns(
    empty_db: Database, capsys
) -> None:
    """If even one row is pending the wait/accumulate guidance still wins;
    re-running evaluate would still leave that pending row pending."""
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260601", ts_code="000001.SZ",
        max_ret_5d=8.0, data_status="complete",
    )
    _insert_anomaly(empty_db, trade_date="20260610", ts_code="000002.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260610", ts_code="000002.SZ",
        max_ret_5d=None, data_status="pending",
    )
    _emit_under_threshold_diagnostics(
        empty_db,
        start="20260101",
        end="20260630",
        label_source="max_ret_5d",
        n_labeled=1,
        min_samples=1000,
    )
    out = capsys.readouterr().out
    assert "1/2 行 data_status='pending'" in out
    # Must surface the most-recent anomaly_date (20260610), not the earliest.
    assert "最近异动日=20260610" in out
    assert "deeptrade volume-anomaly evaluate --backfill-all" not in out


# ---------------------------------------------------------------------------
# Branch B3 — no pending rows, but the chosen label column is NULL on every
# status-ok row. Either real data gap (suspension/delisting) or the user
# picked a label_source `evaluate` didn't populate.
# ---------------------------------------------------------------------------


def test_branch_b3_complete_rows_with_null_label(empty_db: Database, capsys) -> None:
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
    _insert_realized(
        empty_db, anomaly_date="20260601", ts_code="000001.SZ",
        max_ret_5d=None, data_status="complete",
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
    assert "realized_rows=2 (其中可用=0)" in out
    assert "max_ret_5d 为 NULL" in out
    assert "label_source" in out
    # Branch B3 must not loop the user into another evaluate.
    assert "deeptrade volume-anomaly evaluate --backfill-all" not in out
    # Pending-specific phrasing must NOT appear when nothing is pending.
    assert "data_status='pending'" not in out


def test_label_source_swap_uses_alternate_column(empty_db: Database, capsys) -> None:
    """`label_source=ret_t3` should be honored when counting usable rows. The
    row is status='complete' for max_ret_5d but ret_t3 is NULL — this is a
    label_source mismatch (Branch B3), not a pending state."""
    _insert_anomaly(empty_db, trade_date="20260601", ts_code="000001.SZ")
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
    assert "ret_t3 为 NULL" in out
    # B3 must point to the label_source as the fix, not back to evaluate.
    assert "max_ret_5d" in out  # default suggestion mentioned in message
    assert "deeptrade volume-anomaly evaluate --backfill-all" not in out


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
    assert "实现收益已就绪但样本数仍少于阈值" in out
    # Branch C must NOT push the user back to `evaluate` as the primary fix.
    assert "deeptrade volume-anomaly evaluate --backfill-all" not in out
