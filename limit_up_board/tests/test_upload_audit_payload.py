"""P0-2：``_maybe_upload_summary`` 写入的事件 payload 必须包含完整审计字段
（``enabled / url / status / duration_ms / public_url / public_path / error_class``），
且**绝不出现 token / Authorization 明文**。

直接在 ``LubRunner._maybe_upload_summary`` 上做 unit-level 测试：
绕过 ``__init__`` 装配，注入一个 mock rt + 实时 mock ``upload_summary_json``，
捕获 ``rt.emit`` 的 ``payload`` 参数后做断言。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from limit_up_board.config import LubConfig
from limit_up_board.runner import LubRunner
from limit_up_board.uploader import UploadError


SECRET = "TOTALLY-SECRET-TOKEN-DO-NOT-LEAK"


def _make_runner_with_mock_rt() -> tuple[LubRunner, list[dict[str, Any]]]:
    captured_payloads: list[dict[str, Any]] = []

    def emit_stub(event_type, message, *, level=EventLevel.INFO, payload=None, **extra) -> StrategyEvent:  # noqa: ANN001
        merged = dict(payload or {})
        merged.update(extra)
        captured_payloads.append(merged)
        return StrategyEvent(type=event_type, level=level, message=message, payload=merged)

    rt = MagicMock()
    rt.emit.side_effect = emit_stub
    runner = LubRunner.__new__(LubRunner)
    runner._rt = rt
    runner._pending = []
    return runner, captured_payloads


def _write_summary(tmp_path: Path) -> Path:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"meta": {"title": "ok"}}), encoding="utf-8")
    return p


def _flatten_strings(obj: Any) -> str:
    """Serialize the payload (incl. nested values) so we can grep for token leaks."""
    return json.dumps(obj, default=str, ensure_ascii=False)


def test_audit_payload_excludes_token_on_success(tmp_path: Path) -> None:
    json_path = _write_summary(tmp_path)
    cfg = replace(
        LubConfig(),
        summary_upload_enabled=True,
        summary_upload_token=SECRET,
    )
    runner, payloads = _make_runner_with_mock_rt()
    with patch(
        "limit_up_board.runner.upload_summary_json",
        return_value={
            "success": True,
            "url": "https://blob.example.com/reports/2026-05-22/1.json",
            "pathname": "reports/2026-05-22/1.json",
            "index": 1,
            "date": "2026-05-22",
        },
    ) as mock_upload:
        list(runner._maybe_upload_summary(cfg, tmp_path, "20260522"))

    # uploader 拿到了 token（来源 = config）
    _, kwargs = mock_upload.call_args
    assert kwargs["token"] == SECRET

    # 事件 payload 只有 1 条且字段齐全
    assert len(payloads) == 1
    payload = payloads[0]
    expected_keys = {
        "enabled", "url", "json_path", "token_configured", "trade_date",
        "status", "duration_ms", "public_url", "public_path", "date", "index",
    }
    assert expected_keys.issubset(payload.keys()), (
        f"missing audit fields: {expected_keys - set(payload.keys())}"
    )
    assert payload["status"] == "ok"
    assert payload["token_configured"] is True
    assert isinstance(payload["duration_ms"], float) and payload["duration_ms"] >= 0
    # 关键反向断言：token 明文 / Authorization 不得出现在任何字段值里
    serialized = _flatten_strings(payload)
    assert SECRET not in serialized
    assert "authorization" not in serialized.lower()
    assert "token" not in {k.lower() for k in payload.keys() if k != "token_configured"}


def test_audit_payload_excludes_token_on_upload_error(tmp_path: Path) -> None:
    _write_summary(tmp_path)
    cfg = replace(
        LubConfig(),
        summary_upload_enabled=True,
        summary_upload_token=SECRET,
    )
    runner, payloads = _make_runner_with_mock_rt()
    with patch(
        "limit_up_board.runner.upload_summary_json",
        side_effect=UploadError("HTTP 400: bad request"),
    ):
        list(runner._maybe_upload_summary(cfg, tmp_path, "20260522"))

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["status"] == "failed"
    assert payload["error_class"] == "UploadError"
    assert "duration_ms" in payload
    serialized = _flatten_strings(payload)
    assert SECRET not in serialized


def test_audit_payload_skipped_when_upload_disabled(tmp_path: Path) -> None:
    _write_summary(tmp_path)
    cfg = LubConfig()  # 默认 enabled=False
    assert cfg.summary_upload_enabled is False
    runner, payloads = _make_runner_with_mock_rt()
    with patch("limit_up_board.runner.upload_summary_json") as mock_upload:
        list(runner._maybe_upload_summary(cfg, tmp_path, "20260522"))
    assert payloads == []
    mock_upload.assert_not_called()


def test_anonymous_upload_when_token_blank(tmp_path: Path) -> None:
    """token 为空时 uploader 收到 token=None，走匿名分支。"""
    _write_summary(tmp_path)
    cfg = replace(
        LubConfig(),
        summary_upload_enabled=True,
        summary_upload_token="",
    )
    runner, payloads = _make_runner_with_mock_rt()
    with patch(
        "limit_up_board.runner.upload_summary_json",
        return_value={"success": True, "url": "https://x/y", "pathname": "y"},
    ) as mock_upload:
        list(runner._maybe_upload_summary(cfg, tmp_path, "20260522"))

    _, kwargs = mock_upload.call_args
    assert kwargs["token"] is None
    assert payloads[0]["token_configured"] is False
