"""Unit tests for survivorship snapshot helpers in checkmate.data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from checkmate import data


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


@pytest.fixture
def fresh_db(tmp_path):
    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())
    yield db
    db.close()


# ---------------------------------------------------------------------------
# build_status_history_rows — pure logic
# ---------------------------------------------------------------------------


def _sb(rows: list[dict]) -> pd.DataFrame:
    """stock_basic frame builder with all expected columns."""
    cols = ("ts_code", "name", "industry", "market", "exchange",
            "list_status", "list_date", "delist_date")
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def _nc(rows: list[dict]) -> pd.DataFrame:
    cols = ("ts_code", "name", "start_date", "end_date", "ann_date", "change_reason")
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def test_listed_only_yields_single_synthetic_row() -> None:
    sb = _sb([{
        "ts_code": "600519.SH", "name": "贵州茅台", "industry": "白酒",
        "market": "主板", "exchange": "SSE", "list_status": "L",
        "list_date": "20010827", "delist_date": None,
    }])
    nc = _nc([])
    rows = data.build_status_history_rows(sb, nc)
    assert len(rows) == 1
    row = rows[0]
    assert row["ts_code"] == "600519.SH"
    assert row["as_of_date"] == "20010827"
    assert row["list_status"] == "L"
    assert row["is_st"] is False
    assert row["name"] == "贵州茅台"
    payload = json.loads(row["raw_event_json"])
    assert payload["source"] == "stock_basic"


def test_st_in_then_out_history() -> None:
    sb = _sb([{
        "ts_code": "000001.SZ", "name": "平安银行", "industry": "银行",
        "market": "主板", "exchange": "SZSE", "list_status": "L",
        "list_date": "19910403", "delist_date": None,
    }])
    nc = _nc([
        {"ts_code": "000001.SZ", "name": "ST平安", "start_date": "20100101"},
        {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20110101"},
        {"ts_code": "000001.SZ", "name": "*ST平安", "start_date": "20120101"},
    ])
    rows = data.build_status_history_rows(sb, nc)
    by_date = {r["as_of_date"]: r for r in rows}
    assert set(by_date) == {"19910403", "20100101", "20110101", "20120101"}
    assert by_date["19910403"]["is_st"] is False
    assert by_date["20100101"]["is_st"] is True
    assert by_date["20110101"]["is_st"] is False
    assert by_date["20120101"]["is_st"] is True


def test_delisted_stock_has_terminal_row() -> None:
    sb = _sb([{
        "ts_code": "600485.SH", "name": "*ST信威", "industry": "通信",
        "market": "主板", "exchange": "SSE", "list_status": "D",
        "list_date": "20111219", "delist_date": "20200716",
    }])
    nc = _nc([])
    rows = data.build_status_history_rows(sb, nc)
    by_date = {r["as_of_date"]: r for r in rows}
    assert "20200716" in by_date
    assert by_date["20200716"]["list_status"] == "D"
    assert by_date["20111219"]["list_status"] == "L"  # IPO row stays as L


def test_paused_stock_has_terminal_row() -> None:
    sb = _sb([{
        "ts_code": "000005.SZ", "name": "ST星源", "industry": "环保",
        "market": "主板", "exchange": "SZSE", "list_status": "P",
        "list_date": "19901210", "delist_date": "20220101",
    }])
    rows = data.build_status_history_rows(sb, _nc([]))
    by_date = {r["as_of_date"]: r for r in rows}
    assert by_date["20220101"]["list_status"] == "P"


def test_namechange_on_ipo_day_wins_over_synthetic() -> None:
    """If a namechange event is dated == list_date, the synthetic row is
    overwritten in the dedup pass."""
    sb = _sb([{
        "ts_code": "600999.SH", "name": "招商证券", "industry": "证券",
        "market": "主板", "exchange": "SSE", "list_status": "L",
        "list_date": "20091117", "delist_date": None,
    }])
    nc = _nc([
        {"ts_code": "600999.SH", "name": "招商证券A", "start_date": "20091117"},
    ])
    rows = data.build_status_history_rows(sb, nc)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "招商证券A"
    assert json.loads(row["raw_event_json"])["source"] == "namechange"


def test_st_detection_corner_cases() -> None:
    assert data._is_st_name("*ST信威") is True
    assert data._is_st_name("ST锐电") is True
    # Names that happen to start with "S" but are not ST shouldn't trip.
    assert data._is_st_name("SOHO中国") is False
    assert data._is_st_name("") is False
    assert data._is_st_name(None) is False
    # Edge: "ST" alone (no following char) — rare but technically prefix match
    assert data._is_st_name("ST") is True


# ---------------------------------------------------------------------------
# upsert_status_history + query_status_as_of — DB shim
# ---------------------------------------------------------------------------


def test_upsert_and_query_status_as_of(fresh_db) -> None:
    sb = _sb([{
        "ts_code": "000001.SZ", "name": "平安银行", "industry": "银行",
        "market": "主板", "exchange": "SZSE", "list_status": "L",
        "list_date": "19910403", "delist_date": None,
    }])
    nc = _nc([
        {"ts_code": "000001.SZ", "name": "ST平安", "start_date": "20100101"},
        {"ts_code": "000001.SZ", "name": "平安银行", "start_date": "20110101"},
    ])
    n = data.upsert_status_history(fresh_db, data.build_status_history_rows(sb, nc))
    assert n == 3

    # Before any event → IPO-day row
    asof_pre = data.query_status_as_of(fresh_db, "000001.SZ", "20050101")
    assert asof_pre is not None
    assert asof_pre["is_st"] is False
    assert asof_pre["name"] == "平安银行"

    # Inside the ST window → ST flag
    asof_st = data.query_status_as_of(fresh_db, "000001.SZ", "20100630")
    assert asof_st is not None
    assert asof_st["is_st"] is True
    assert asof_st["name"] == "ST平安"

    # After un-ST → flag clears
    asof_post = data.query_status_as_of(fresh_db, "000001.SZ", "20110201")
    assert asof_post is not None
    assert asof_post["is_st"] is False
    assert asof_post["name"] == "平安银行"

    # Unknown code → None
    assert data.query_status_as_of(fresh_db, "999999.SH", "20100101") is None


def test_upsert_is_idempotent(fresh_db) -> None:
    sb = _sb([{
        "ts_code": "600519.SH", "name": "贵州茅台", "industry": "白酒",
        "market": "主板", "exchange": "SSE", "list_status": "L",
        "list_date": "20010827", "delist_date": None,
    }])
    rows = data.build_status_history_rows(sb, _nc([]))
    data.upsert_status_history(fresh_db, rows)
    data.upsert_status_history(fresh_db, rows)  # re-apply
    total = fresh_db.execute(
        "SELECT COUNT(*) FROM checkmate_stock_status_history"
    ).fetchone()[0]
    assert total == 1
