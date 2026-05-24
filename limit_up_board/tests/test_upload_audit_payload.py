"""``_maybe_upload_summary`` 写入的事件 payload 必须包含与 v0.12.3 兼容的审计字段
（``enabled / url / status / duration_ms / public_url / public_path / error_class``），
且无 token / Authorization 明文。

v0.13.3 改造：``_maybe_upload_summary`` 不再持有 ``LubConfig``，改为通过
``self._ctx.make_report_uploader()`` 委派给框架 :class:`ReportUploader`。
测试通过 mock ``PluginContext`` + ``ReportUploader.upload`` 返回值断言 payload
字段结构。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from limit_up_board.runner import LubRunner


@dataclass
class _StubResult:
    """Minimal stand-in for framework v0.11 :class:`UploadResult`."""

    status: str
    public_url: str | None = None
    public_path: str | None = None
    public_index: int | None = None
    public_date: str | None = None
    duration_ms: float = 0.0
    error: str | None = None
    error_class: str | None = None


def _make_runner_with_mock_rt(
    ctx: Any = None,
) -> tuple[LubRunner, list[dict[str, Any]]]:
    captured_payloads: list[dict[str, Any]] = []

    def emit_stub(
        event_type, message, *, level=EventLevel.INFO, payload=None, **extra
    ) -> StrategyEvent:  # noqa: ANN001
        merged = dict(payload or {})
        merged.update(extra)
        captured_payloads.append(merged)
        return StrategyEvent(
            type=event_type, level=level, message=message, payload=merged
        )

    rt = MagicMock()
    rt.emit.side_effect = emit_stub
    rt.run_id = "test-run-id"
    runner = LubRunner.__new__(LubRunner)
    runner._rt = rt
    runner._pending = []
    runner._ctx = ctx
    return runner, captured_payloads


def _write_summary(tmp_path: Path) -> Path:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"meta": {"title": "ok"}}), encoding="utf-8")
    return p


def _ctx_with_uploader(upload_result: _StubResult) -> MagicMock:
    uploader = MagicMock()
    uploader.upload.return_value = upload_result
    ctx = MagicMock()
    ctx.make_report_uploader.return_value = uploader
    return ctx


def test_audit_payload_on_success(tmp_path: Path) -> None:
    _write_summary(tmp_path)
    result = _StubResult(
        status="ok",
        public_url="https://blob.example.com/reports/2026-05-22/1.json",
        public_path="reports/2026-05-22/1.json",
        public_index=1,
        public_date="2026-05-22",
        duration_ms=412.5,
    )
    ctx = _ctx_with_uploader(result)
    runner, payloads = _make_runner_with_mock_rt(ctx=ctx)

    list(runner._maybe_upload_summary(tmp_path, "20260522"))

    # uploader 被以正确入参调用
    ctx.make_report_uploader.assert_called_once_with(run_id="test-run-id")
    args, kwargs = ctx.make_report_uploader.return_value.upload.call_args
    assert args[0] == tmp_path / "summary.json"
    assert kwargs["plugin_name"] == "打板策略"
    assert kwargs["trade_date"] == "20260522"

    # 事件 payload 含完整的 v0.12.3 兼容字段
    assert len(payloads) == 1
    payload = payloads[0]
    expected_keys = {
        "enabled",
        "url",
        "status",
        "duration_ms",
        "public_url",
        "public_path",
        "public_index",
        "public_date",
        "trade_date",
        "json_path",
        "error_class",
    }
    assert expected_keys.issubset(payload.keys()), (
        f"missing audit fields: {expected_keys - set(payload.keys())}"
    )
    assert payload["status"] == "ok"
    assert payload["public_url"] == result.public_url
    assert payload["public_path"] == result.public_path
    assert payload["duration_ms"] == 412.5


def test_audit_payload_on_failure(tmp_path: Path) -> None:
    _write_summary(tmp_path)
    result = _StubResult(
        status="failed",
        duration_ms=12.0,
        error="HTTP 400: bad request",
        error_class="UploadError",
    )
    ctx = _ctx_with_uploader(result)
    runner, payloads = _make_runner_with_mock_rt(ctx=ctx)

    list(runner._maybe_upload_summary(tmp_path, "20260522"))

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["status"] == "failed"
    assert payload["error_class"] == "UploadError"
    assert payload["duration_ms"] == 12.0


def test_skipped_disabled_emits_no_event(tmp_path: Path) -> None:
    """框架返回 ``status="skipped_disabled"`` 时插件应静默返回，不写事件。"""
    _write_summary(tmp_path)
    result = _StubResult(status="skipped_disabled")
    ctx = _ctx_with_uploader(result)
    runner, payloads = _make_runner_with_mock_rt(ctx=ctx)

    list(runner._maybe_upload_summary(tmp_path, "20260522"))

    assert payloads == []


def test_skipped_missing_file_emits_no_event(tmp_path: Path) -> None:
    """框架返回 ``status="skipped_missing_file"`` 时也静默返回。"""
    result = _StubResult(status="skipped_missing_file")
    ctx = _ctx_with_uploader(result)
    runner, payloads = _make_runner_with_mock_rt(ctx=ctx)

    list(runner._maybe_upload_summary(tmp_path, "20260522"))

    assert payloads == []


def test_no_ctx_short_circuits(tmp_path: Path) -> None:
    """``ctx=None``（未注入 PluginContext）时直接返回，不触发上传。"""
    _write_summary(tmp_path)
    runner, payloads = _make_runner_with_mock_rt(ctx=None)

    list(runner._maybe_upload_summary(tmp_path, "20260522"))

    assert payloads == []
