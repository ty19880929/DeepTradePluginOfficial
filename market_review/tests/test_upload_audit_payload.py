"""maybe_upload_summary — payload + skip / failure event audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from market_review.report.upload import PLUGIN_DISPLAY_NAME, maybe_upload_summary
from market_review.windows import Window


@dataclass
class _UploadResult:
    """Stand-in for framework's UploadResult — fields :func:`upload` reads."""

    status: str
    duration_ms: float = 0.0


class _FakeUploader:
    """Records every call into ``calls`` for assertion."""

    def __init__(self, *, result: _UploadResult | Exception):
        self.calls: list[dict] = []
        self._result = result

    def upload(self, json_path, *, plugin_name, trade_date, extra_fields=None):
        self.calls.append({
            "json_path": json_path,
            "plugin_name": plugin_name,
            "trade_date": trade_date,
            "extra_fields": extra_fields,
        })
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeCtx:
    def __init__(self, uploader: _FakeUploader | Exception):
        self._uploader = uploader

    def make_report_uploader(self, *, run_id=None):
        if isinstance(self._uploader, Exception):
            raise self._uploader
        return self._uploader


def _window() -> Window:
    return Window(mode="day", start="20260530", end="20260530",
                  trade_dates=("20260530",), anchor="20260530")


def _make_summary(tmp_path: Path, body: dict | None = None) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(body or {"meta": {"runId": "run-1"}}),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Success path — plugin name + trade_date arguments
# ---------------------------------------------------------------------------


def test_upload_invokes_uploader_with_design_args(tmp_path: Path) -> None:
    _make_summary(tmp_path)
    uploader = _FakeUploader(result=_UploadResult(status="ok", duration_ms=120.0))
    ctx = _FakeCtx(uploader)

    events = list(maybe_upload_summary(
        ctx, run_id="run-1", report_dir=tmp_path, window=_window(),
    ))

    assert len(uploader.calls) == 1
    call = uploader.calls[0]
    # Design §15.9 — plugin_name + trade_date are the contract.
    assert call["plugin_name"] == PLUGIN_DISPLAY_NAME == "市场复盘"
    assert call["trade_date"] == "20260530"
    assert call["json_path"].name == "summary.json"
    # OK event emitted.
    assert len(events) == 1
    assert events[0].level.name == "INFO"
    assert "上传成功" in events[0].message
    assert events[0].payload["status"] == "ok"
    assert events[0].payload["duration_ms"] == 120.0


def test_upload_uses_window_anchor_as_trade_date(tmp_path: Path) -> None:
    """Range mode → trade_date == window.anchor == window.end (design §15.9.3)."""
    _make_summary(tmp_path)
    uploader = _FakeUploader(result=_UploadResult(status="ok"))
    ctx = _FakeCtx(uploader)

    win = Window(
        mode="range", start="20260520", end="20260530",
        trade_dates=("20260520", "20260530"), anchor="20260530",
    )
    list(maybe_upload_summary(ctx, run_id="r", report_dir=tmp_path, window=win))
    assert uploader.calls[0]["trade_date"] == "20260530"


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------


def test_ctx_none_skips_with_log_event(tmp_path: Path) -> None:
    _make_summary(tmp_path)
    events = list(maybe_upload_summary(
        None, run_id="r", report_dir=tmp_path, window=_window(),
    ))
    assert len(events) == 1
    assert events[0].payload["status"] == "skipped_no_ctx"


def test_missing_summary_json_skips_with_log_event(tmp_path: Path) -> None:
    """No summary.json file → skipped_no_local_file (uploader never called)."""
    uploader = _FakeUploader(result=_UploadResult(status="ok"))
    ctx = _FakeCtx(uploader)
    events = list(maybe_upload_summary(
        ctx, run_id="r", report_dir=tmp_path, window=_window(),
    ))
    assert uploader.calls == []
    assert events[0].payload["status"] == "skipped_no_local_file"
    assert events[0].level.name == "INFO"


def test_skip_disabled_status_propagates(tmp_path: Path) -> None:
    _make_summary(tmp_path)
    uploader = _FakeUploader(result=_UploadResult(status="skipped_disabled"))
    ctx = _FakeCtx(uploader)
    events = list(maybe_upload_summary(
        ctx, run_id="r", report_dir=tmp_path, window=_window(),
    ))
    assert len(events) == 1
    assert events[0].level.name == "INFO"
    assert events[0].payload["status"] == "skipped_disabled"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_failed_http_status_emits_warn(tmp_path: Path) -> None:
    _make_summary(tmp_path)
    uploader = _FakeUploader(result=_UploadResult(status="failed_http"))
    ctx = _FakeCtx(uploader)
    events = list(maybe_upload_summary(
        ctx, run_id="r", report_dir=tmp_path, window=_window(),
    ))
    assert events[0].level.name == "WARN"
    assert events[0].payload["status"] == "failed_http"
    assert "上传失败" in events[0].message


def test_uploader_init_failure_emits_warn(tmp_path: Path) -> None:
    _make_summary(tmp_path)
    ctx = _FakeCtx(uploader=RuntimeError("framework setup broken"))
    events = list(maybe_upload_summary(
        ctx, run_id="r", report_dir=tmp_path, window=_window(),
    ))
    assert events[0].level.name == "WARN"
    assert events[0].payload["status"] == "skipped_uploader_init_failed"
    assert "framework setup broken" in events[0].message


def test_upload_call_raise_caught(tmp_path: Path) -> None:
    """ReportUploader.upload should never raise per framework contract, but
    defense-in-depth: when it does, we catch + emit WARN, never propagate."""
    _make_summary(tmp_path)
    uploader = _FakeUploader(result=ConnectionError("network down"))
    ctx = _FakeCtx(uploader)
    events = list(maybe_upload_summary(
        ctx, run_id="r", report_dir=tmp_path, window=_window(),
    ))
    assert events[0].level.name == "WARN"
    assert events[0].payload["status"] == "raised"
    assert "ConnectionError" in events[0].message


# ---------------------------------------------------------------------------
# Generator semantics
# ---------------------------------------------------------------------------


def test_emits_exactly_one_event_per_outcome(tmp_path: Path) -> None:
    _make_summary(tmp_path)
    uploader = _FakeUploader(result=_UploadResult(status="ok"))
    ctx = _FakeCtx(uploader)
    events = list(maybe_upload_summary(
        ctx, run_id="r", report_dir=tmp_path, window=_window(),
    ))
    # Exactly one event — the runner inserts it into its stream
    # without duplication.
    assert len(events) == 1
