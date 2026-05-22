"""PR #3 — exporter: JSON / CSV payload 构造 + 序列化。"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime

import pytest

from limit_up_board.winrate.exporter import (
    CSV_COLUMNS,
    build_payload,
    infer_format,
    serialize_csv,
    serialize_json,
)
from limit_up_board.winrate.persistence import PredictionRecord
from limit_up_board.winrate.resolver import ResolvedRecord
from limit_up_board.winrate.stats import group_by_prediction, summarize


def _rec(ts: str, rank: int = 1, prediction: str = "top_candidate") -> PredictionRecord:
    return PredictionRecord(
        trade_date="20260521",
        next_trade_date="20260522",
        ts_code=ts,
        name=f"name-{ts}",
        run_id="r1",
        prediction=prediction,
        rank=rank,
        continuation_score=80.0,
        confidence="high",
        t_close_price=10.0,
        lgb_score=0.7,
        lgb_decile=9,
        raw_prediction_json=None,
    )


def _res(ts: str, outcome: str, pct: float | None, **kw) -> ResolvedRecord:
    return ResolvedRecord(
        record=_rec(ts, **kw),
        t1_open_price=10.0 + (pct or 0) / 10.0 if pct is not None else None,
        open_vs_limit_pct=pct,
        outcome=outcome,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# infer_format
# ---------------------------------------------------------------------------


def test_infer_format_explicit_wins() -> None:
    assert infer_format("foo.json", "csv") == "csv"
    assert infer_format("foo.csv", "json") == "json"


def test_infer_format_extension() -> None:
    assert infer_format("reports/x.csv") == "csv"
    assert infer_format("reports/x.json") == "json"


def test_infer_format_default_json() -> None:
    assert infer_format("reports/x.txt") == "json"


def test_infer_format_invalid_raises() -> None:
    with pytest.raises(ValueError):
        infer_format("x.json", "yaml")


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def test_build_payload_full_shape() -> None:
    resolved = [
        _res("600001.SH", "win", 1.5),
        _res("600002.SH", "loss", -1.0, prediction="watchlist"),
    ]
    payload = build_payload(
        window_start="20260520",
        window_end="20260521",
        summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        resolved=resolved,
        generated_at=datetime(2026, 5, 22, 10, 0, 0),
    )
    assert payload.generated_at == "2026-05-22T10:00:00"
    assert payload.window == {"start": "20260520", "end": "20260521"}
    assert payload.summary["total"] == 2
    assert len(payload.by_prediction) == 2
    assert len(payload.records) == 2
    # Per-record fields
    r0 = payload.records[0]
    assert r0["ts_code"] == "600001.SH"
    assert r0["outcome"] == "win"
    assert r0["t1_open_price"] == pytest.approx(10.15)
    assert r0["open_vs_limit_pct"] == pytest.approx(1.5)


def test_build_payload_empty() -> None:
    payload = build_payload(
        window_start="20260521",
        window_end="20260521",
        summary=summarize([]),
        by_prediction=[],
        resolved=[],
        generated_at=datetime(2026, 5, 22, 10, 0, 0),
    )
    assert payload.summary["total"] == 0
    assert payload.records == []


# ---------------------------------------------------------------------------
# serialize_json
# ---------------------------------------------------------------------------


def test_serialize_json_roundtrip() -> None:
    resolved = [_res("600001.SH", "win", 1.5)]
    payload = build_payload(
        window_start="20260521", window_end="20260521",
        summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        resolved=resolved,
        generated_at=datetime(2026, 5, 22, 10, 0, 0),
    )
    text = serialize_json(payload)
    obj = json.loads(text)
    assert obj["window"]["start"] == "20260521"
    assert obj["summary"]["win"] == 1
    assert obj["records"][0]["ts_code"] == "600001.SH"
    # Ensure no float NaN sneaks in (json.dumps default allows it but breaks Excel)
    assert "NaN" not in text


def test_serialize_json_chinese_chars() -> None:
    """ensure_ascii=False → 中文不被转义。"""
    resolved = [_res("600001.SH", "win", 1.5)]
    payload = build_payload(
        window_start="20260521", window_end="20260521",
        summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        resolved=resolved,
    )
    # Inject name with Chinese
    payload.records[0]["name"] = "贵州茅台"
    text = serialize_json(payload)
    assert "贵州茅台" in text


# ---------------------------------------------------------------------------
# serialize_csv
# ---------------------------------------------------------------------------


def test_serialize_csv_columns_and_rows() -> None:
    resolved = [
        _res("600001.SH", "win", 1.5),
        _res("600002.SH", "loss", -1.0, prediction="watchlist"),
    ]
    payload = build_payload(
        window_start="20260521", window_end="20260521",
        summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        resolved=resolved,
    )
    text = serialize_csv(payload)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    # Header row
    assert tuple(rows[0]) == CSV_COLUMNS
    # 2 data rows
    assert len(rows) == 3
    # Find ts_code column index
    ts_idx = CSV_COLUMNS.index("ts_code")
    outcome_idx = CSV_COLUMNS.index("outcome")
    assert rows[1][ts_idx] == "600001.SH"
    assert rows[1][outcome_idx] == "win"
    assert rows[2][ts_idx] == "600002.SH"
    assert rows[2][outcome_idx] == "loss"


def test_serialize_csv_empty_records_still_has_header() -> None:
    payload = build_payload(
        window_start="20260521", window_end="20260521",
        summary=summarize([]),
        by_prediction=[],
        resolved=[],
    )
    text = serialize_csv(payload)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    assert len(rows) == 1
    assert tuple(rows[0]) == CSV_COLUMNS
