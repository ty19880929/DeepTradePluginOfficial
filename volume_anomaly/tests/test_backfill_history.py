"""Tests for ``volume-anomaly screen --backfill-history`` (v0.9.0).

Covers the LLM-free batch replay path that bootstraps the training corpus
for new users. Mocks ``screen_anomalies`` at the runner-level seam — the
screen-rule logic itself is tested by ``test_screen_rules``; here we only
verify the loop / resume / overwrite / error-isolation semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from deeptrade.core.db import Database

from volume_anomaly import runner as runner_mod
from volume_anomaly.calendar import TradeCalendar
from volume_anomaly.data import ScreenResult
from volume_anomaly.runner import BackfillHistoryParams, VaRunner


# ---------------------------------------------------------------------------
# Calendar helper
# ---------------------------------------------------------------------------


def _calendar_with_open_dates(open_dates: list[str]) -> TradeCalendar:
    rows = []
    cur = pd.Timestamp("2026-05-25")
    end = pd.Timestamp("2026-06-30")
    while cur <= end:
        d = cur.strftime("%Y%m%d")
        rows.append({"cal_date": d, "is_open": 1 if d in open_dates else 0})
        cur += pd.Timedelta(days=1)
    return TradeCalendar(pd.DataFrame(rows))


def test_open_dates_in_range_filters_closed_days() -> None:
    cal = _calendar_with_open_dates(
        ["20260601", "20260602", "20260605", "20260608"]
    )
    assert cal.open_dates_in_range("20260601", "20260610") == [
        "20260601", "20260602", "20260605", "20260608",
    ]
    # Endpoints inclusive.
    assert cal.open_dates_in_range("20260602", "20260605") == [
        "20260602", "20260605",
    ]
    # Range with no open days.
    assert cal.open_dates_in_range("20260603", "20260604") == []
    # Empty / inverted range.
    assert cal.open_dates_in_range("20260610", "20260601") == []


# ---------------------------------------------------------------------------
# Runner-level stub plumbing
# ---------------------------------------------------------------------------


class _StubTushareForBackfill:
    """Minimal TushareClient stub for backfill — only ``trade_cal`` is called
    once before the screen_anomalies monkeypatch takes over."""

    def __init__(self, cal_df: pd.DataFrame) -> None:
        self._cal_df = cal_df

    def call(self, api_name: str, **_kwargs: Any) -> pd.DataFrame:
        if api_name == "trade_cal":
            return self._cal_df.copy()
        return pd.DataFrame()


def _make_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a (VaRunner, db, monkeypatched_call_log) triple ready to run
    backfill. ``call_log`` records the trade_dates passed to the mocked
    ``screen_anomalies`` so tests can assert skip/overwrite behaviour."""
    db = Database(tmp_path / "backfill.duckdb")
    _create_anomaly_history_table(db)
    _create_va_runs_events(db)

    open_dates = [
        "20260601", "20260602", "20260603", "20260604", "20260605",
        "20260608",
    ]
    cal_rows = []
    cur = pd.Timestamp("2026-05-25")
    end = pd.Timestamp("2026-06-30")
    while cur <= end:
        d = cur.strftime("%Y%m%d")
        cal_rows.append({"cal_date": d, "is_open": 1 if d in open_dates else 0})
        cur += pd.Timedelta(days=1)
    cal_df = pd.DataFrame(cal_rows)

    # Bypass real Tushare client / runtime construction; we plug a tiny
    # runtime object that VaRunner only inspects for db / tushare / run_id.
    from deeptrade.plugins_api.events import EventLevel, StrategyEvent

    class _Cfg:
        def get_app_config(self) -> None:
            return None

    class _Rt:
        def __init__(self, db_handle: Database) -> None:
            self.plugin_id = "volume-anomaly"
            self.run_id: str | None = None
            self.is_intraday = False
            self.tushare = None
            self.llms = None
            self.db = db_handle
            self.config = _Cfg()

        @staticmethod
        def emit(event_type, message, level=None, payload=None):
            return StrategyEvent(
                type=event_type,
                level=level or EventLevel.INFO,
                message=message,
                payload=payload or {},
            )

    rt = _Rt(db)

    def _fake_build_tushare_client(_rt, *, intraday, event_cb):  # noqa: ARG001
        return _StubTushareForBackfill(cal_df)

    monkeypatch.setattr(
        runner_mod, "build_tushare_client", _fake_build_tushare_client
    )
    monkeypatch.setattr(
        runner_mod, "export_llm_calls", lambda *args, **kwargs: None
    )

    call_log: list[str] = []

    def _fake_screen(*, tushare, calendar, trade_date, rules, force_sync):  # noqa: ARG001
        call_log.append(trade_date)
        return ScreenResult(
            trade_date=trade_date,
            n_main_board=1500,
            n_after_st_susp=1400,
            n_after_t_day_rules=200,
            n_after_upper_shadow=180,
            n_after_turnover=80,
            n_after_vol_rules=2,
            hits=[
                {
                    "trade_date": trade_date,
                    "ts_code": "000001.SZ",
                    "name": "PA Bank",
                    "industry": "金融",
                    "pct_chg": 6.5,
                    "close": 12.0, "open": 11.5, "high": 12.5, "low": 11.0,
                    "vol": 1_000_000, "amount": 12_000_000,
                    "body_ratio": 0.7, "turnover_rate": 5.0,
                    "vol_ratio_5d": 2.0, "max_vol_60d": 1_500_000,
                },
                {
                    "trade_date": trade_date,
                    "ts_code": "600000.SH",
                    "name": "PA Bank",
                    "industry": "金融",
                    "pct_chg": 7.0,
                    "close": 14.0, "open": 13.4, "high": 14.5, "low": 13.0,
                    "vol": 2_000_000, "amount": 27_000_000,
                    "body_ratio": 0.6, "turnover_rate": 6.5,
                    "vol_ratio_5d": 2.5, "max_vol_60d": 2_400_000,
                },
            ],
        )

    monkeypatch.setattr(runner_mod, "screen_anomalies", _fake_screen)

    runner = VaRunner(rt)
    runner._renderer = runner_mod.LegacyStreamRenderer()  # silent enough
    return runner, db, call_log


def _create_anomaly_history_table(db: Database) -> None:
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


def _create_va_runs_events(db: Database) -> None:
    db.execute(
        """
        CREATE TABLE va_runs (
            run_id        VARCHAR PRIMARY KEY,
            mode          VARCHAR,
            trade_date    VARCHAR,
            status        VARCHAR,
            is_intraday   BOOLEAN,
            started_at    TIMESTAMP,
            finished_at   TIMESTAMP,
            params_json   VARCHAR,
            summary_json  VARCHAR,
            error         VARCHAR
        )
        """
    )
    db.execute(
        """
        CREATE TABLE va_events (
            run_id      VARCHAR,
            seq         INTEGER,
            level       VARCHAR,
            event_type  VARCHAR,
            message     VARCHAR,
            payload_json VARCHAR
        )
        """
    )


# ---------------------------------------------------------------------------
# Iterator semantics
# ---------------------------------------------------------------------------


def test_happy_path_processes_every_open_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, db, call_log = _make_runtime(tmp_path, monkeypatch)
    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start_date="20260601", end_date="20260605")
    )
    # 5 calendar dates, all open in the fixture.
    assert outcome.status.value == "success"
    assert call_log == ["20260601", "20260602", "20260603", "20260604", "20260605"]
    rows = db.fetchall("SELECT trade_date, ts_code FROM va_anomaly_history "
                       "ORDER BY trade_date, ts_code")
    assert len(rows) == 10  # 5 dates × 2 hits
    final = [e for e in outcome.seen_events if e.type.value == "result.persisted"]
    assert final and final[0].payload["n_processed"] == 5
    assert final[0].payload["n_skipped"] == 0
    assert final[0].payload["n_failed"] == 0
    assert final[0].payload["n_hits_added"] == 10
    db.close()


def test_resume_skips_dates_with_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, db, call_log = _make_runtime(tmp_path, monkeypatch)
    # Pre-seed 20260602 → backfill must skip it.
    db.execute(
        "INSERT INTO va_anomaly_history "
        "(trade_date, ts_code, name, industry, pct_chg, close, open, high, low, "
        "vol, amount, body_ratio, turnover_rate, vol_ratio_5d, max_vol_60d) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("20260602", "STALE.SZ", "Stale", "测试", 5.0, 10.0, 9.5, 10.5, 9.0,
         100.0, 1000.0, 0.5, 3.0, 1.5, 200.0),
    )

    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start_date="20260601", end_date="20260605")
    )
    # screen_anomalies must NOT have been called for the pre-seeded date.
    assert "20260602" not in call_log
    assert sorted(call_log) == ["20260601", "20260603", "20260604", "20260605"]
    # Stale row preserved (skip = don't touch).
    stale = db.fetchall(
        "SELECT ts_code FROM va_anomaly_history WHERE trade_date='20260602'"
    )
    assert stale == [("STALE.SZ",)]
    final = [e for e in outcome.seen_events if e.type.value == "result.persisted"][0]
    assert final.payload["n_skipped"] == 1
    assert final.payload["n_processed"] == 4
    db.close()


def test_overwrite_replaces_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, db, call_log = _make_runtime(tmp_path, monkeypatch)
    db.execute(
        "INSERT INTO va_anomaly_history "
        "(trade_date, ts_code, name, industry, pct_chg, close, open, high, low, "
        "vol, amount, body_ratio, turnover_rate, vol_ratio_5d, max_vol_60d) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("20260602", "STALE.SZ", "Stale", "测试", 5.0, 10.0, 9.5, 10.5, 9.0,
         100.0, 1000.0, 0.5, 3.0, 1.5, 200.0),
    )

    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(
            start_date="20260601", end_date="20260605", overwrite=True
        )
    )
    # Overwrite → screen_anomalies called for every date.
    assert sorted(call_log) == [
        "20260601", "20260602", "20260603", "20260604", "20260605",
    ]
    # Stale row replaced by fresh hits (STALE.SZ wasn't in our mocked hits).
    codes_on_0602 = db.fetchall(
        "SELECT ts_code FROM va_anomaly_history WHERE trade_date='20260602' "
        "ORDER BY ts_code"
    )
    assert ("STALE.SZ",) not in codes_on_0602
    assert ("000001.SZ",) in codes_on_0602
    final = [e for e in outcome.seen_events if e.type.value == "result.persisted"][0]
    assert final.payload["n_skipped"] == 0
    assert final.payload["overwrite"] is True
    db.close()


def test_calendar_skips_non_open_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Window spans a weekend; only is_open=1 dates get screened."""
    runner, db, call_log = _make_runtime(tmp_path, monkeypatch)
    # In our fixture 20260606 / 20260607 are weekend (closed).
    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start_date="20260605", end_date="20260608")
    )
    assert call_log == ["20260605", "20260608"]
    final = [e for e in outcome.seen_events if e.type.value == "result.persisted"][0]
    assert final.payload["n_open_dates"] == 2
    db.close()


def test_per_day_failure_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One date raises → other dates still process; outcome stays SUCCESS."""
    runner, db, call_log = _make_runtime(tmp_path, monkeypatch)

    real_screen = runner_mod.screen_anomalies

    def _flaky(*, tushare, calendar, trade_date, rules, force_sync):
        # Record the attempt for every date — including the failing one —
        # so the test can assert the loop didn't skip past 20260603.
        call_log.append(trade_date)
        if trade_date == "20260603":
            raise RuntimeError("simulated tushare hiccup")
        result = real_screen(
            tushare=tushare, calendar=calendar, trade_date=trade_date,
            rules=rules, force_sync=force_sync,
        )
        # real_screen is the captured _fake_screen which already appended;
        # remove its duplicate entry so the log stays single-entry per date.
        call_log.pop()
        return result

    monkeypatch.setattr(runner_mod, "screen_anomalies", _flaky)

    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start_date="20260601", end_date="20260605")
    )
    assert outcome.status.value == "success"
    # Every date was attempted including the failing one.
    assert call_log == [
        "20260601", "20260602", "20260603", "20260604", "20260605",
    ]
    # Failing date contributed no rows.
    rows_0603 = db.fetchall(
        "SELECT COUNT(*) FROM va_anomaly_history WHERE trade_date='20260603'"
    )
    assert rows_0603[0][0] == 0
    final = [e for e in outcome.seen_events if e.type.value == "result.persisted"][0]
    assert final.payload["n_failed"] == 1
    assert final.payload["n_processed"] == 4
    db.close()


def test_empty_window_no_open_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A range covering only weekends/holidays produces n_open_dates=0 and
    no exception."""
    runner, db, call_log = _make_runtime(tmp_path, monkeypatch)
    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start_date="20260606", end_date="20260607")
    )
    assert outcome.status.value == "success"
    assert call_log == []
    final = [e for e in outcome.seen_events if e.type.value == "result.persisted"][0]
    assert final.payload["n_processed"] == 0
    assert final.payload["n_open_dates"] == 0
    db.close()


# ---------------------------------------------------------------------------
# CLI-level validation
# ---------------------------------------------------------------------------


def test_cli_backfill_history_requires_start_and_end() -> None:
    """The cmd_screen handler bails out with a non-zero exit if the new flags
    are misused. We invoke typer's CliRunner to exercise the actual command
    surface without spinning up the runtime."""
    from typer.testing import CliRunner

    from volume_anomaly.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["screen", "--backfill-history"])
    assert res.exit_code != 0
    assert "--start" in res.stdout or "--start" in (res.stderr or "")


def test_cli_backfill_history_rejects_inverted_range() -> None:
    from typer.testing import CliRunner

    from volume_anomaly.cli import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["screen", "--backfill-history", "--start", "20260605", "--end", "20260601"],
    )
    assert res.exit_code != 0


def test_cli_backfill_history_rejects_intraday_combo() -> None:
    from typer.testing import CliRunner

    from volume_anomaly.cli import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["screen", "--backfill-history", "--start", "20260601", "--end",
         "20260605", "--allow-intraday"],
    )
    assert res.exit_code != 0


def test_cli_live_screen_rejects_backfill_flags() -> None:
    """Without --backfill-history, --start / --end / --overwrite must error
    rather than silently passing through."""
    from typer.testing import CliRunner

    from volume_anomaly.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["screen", "--start", "20260601"])
    assert res.exit_code != 0
    res2 = runner.invoke(app, ["screen", "--overwrite"])
    assert res2.exit_code != 0
