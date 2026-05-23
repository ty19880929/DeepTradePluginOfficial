"""Tests for ``limit_up_board.uploader``.

The uploader is a pure stdlib wrapper around ``urllib.request.urlopen``, so
we patch that single call to simulate success / HTTP error / network failure
without ever hitting the real endpoint.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from limit_up_board.uploader import (
    DEFAULT_UPLOAD_URL,
    UploadError,
    _build_multipart,
    upload_summary_json,
)


# ---------------------------------------------------------------------------
# Multipart body
# ---------------------------------------------------------------------------


def test_build_multipart_has_expected_structure() -> None:
    body = _build_multipart(
        "BOUNDARY-X",
        field_name="file",
        filename="summary.json",
        content=b'{"meta": {}}',
        mime="application/json",
    )
    text = body.decode("utf-8")
    assert text.startswith("--BOUNDARY-X\r\n")
    assert 'Content-Disposition: form-data; name="file"; filename="summary.json"' in text
    assert "Content-Type: application/json" in text
    assert '{"meta": {}}' in text
    assert text.endswith("\r\n--BOUNDARY-X--\r\n")


def test_build_multipart_embeds_extra_text_fields() -> None:
    """v0.12.2+ — plugin_name / trade_date 走 multipart text part。"""
    body = _build_multipart(
        "BOUNDARY-X",
        field_name="file",
        filename="summary.json",
        content=b'{"meta": {}}',
        mime="application/json",
        text_fields={"plugin_name": "打板策略", "trade_date": "20260522"},
    )
    text = body.decode("utf-8")
    # 两个 text part 各自有 boundary + Content-Disposition + 空行 + value
    assert 'Content-Disposition: form-data; name="plugin_name"\r\n\r\n打板策略\r\n' in text
    assert 'Content-Disposition: form-data; name="trade_date"\r\n\r\n20260522\r\n' in text
    # 文件 part 仍在尾部
    assert 'filename="summary.json"' in text
    assert text.endswith("\r\n--BOUNDARY-X--\r\n")


# ---------------------------------------------------------------------------
# Path / suffix validation
# ---------------------------------------------------------------------------


def test_upload_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(UploadError, match="file not found"):
        upload_summary_json(missing)


def test_upload_rejects_non_json_suffix(tmp_path: Path) -> None:
    p = tmp_path / "summary.md"
    p.write_text("# not json", encoding="utf-8")
    with pytest.raises(UploadError, match="only .json files"):
        upload_summary_json(p)


def test_upload_rejects_html_suffix_after_v012_switch(tmp_path: Path) -> None:
    """旧 ``.html`` 文件不再被接受 —— 防止误传遗留产物。"""
    p = tmp_path / "summary.html"
    p.write_text("<html></html>", encoding="utf-8")
    with pytest.raises(UploadError, match="only .json files"):
        upload_summary_json(p)


# ---------------------------------------------------------------------------
# HTTP flow — mocked urlopen
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._buf = io.BytesIO(payload)

    def read(self) -> bytes:
        return self._buf.read()

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _write_json(tmp_path: Path) -> Path:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"meta": {"title": "ok"}}), encoding="utf-8")
    return p


def test_upload_success_returns_decoded_json(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path)
    body = json.dumps(
        {
            "success": True,
            "url": "https://blob.example.com/reports/2026-05-22/1.json",
            "pathname": "reports/2026-05-22/1.json",
            "index": 1,
            "date": "2026-05-22",
        }
    ).encode("utf-8")

    captured_request: dict = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured_request["url"] = req.full_url
        captured_request["headers"] = dict(req.header_items())
        captured_request["data"] = req.data or b""
        captured_request["data_len"] = len(captured_request["data"])
        captured_request["method"] = req.get_method()
        return _FakeResp(200, body)

    with patch("limit_up_board.uploader.urlopen", side_effect=fake_urlopen):
        result = upload_summary_json(json_path, token="my-test-token")

    assert result["success"] is True
    assert result["url"].endswith("/reports/2026-05-22/1.json")
    assert captured_request["url"] == DEFAULT_UPLOAD_URL
    assert captured_request["method"] == "POST"
    auth = {k.lower(): v for k, v in captured_request["headers"].items()}
    assert auth["authorization"] == "Bearer my-test-token"
    assert auth["content-type"].startswith("multipart/form-data; boundary=")
    # Body should be larger than just the file bytes (multipart overhead).
    assert captured_request["data_len"] > json_path.stat().st_size
    # Body must carry the JSON mime in the file part so the server can branch.
    assert b"Content-Type: application/json" in captured_request["data"]
    assert b'filename="summary.json"' in captured_request["data"]


def test_upload_omits_authorization_header_when_token_blank(tmp_path: Path) -> None:
    """v0.12.3+：token=None / "" 时不写 Authorization header（匿名）。"""
    json_path = _write_json(tmp_path)
    captured: dict = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured["headers"] = dict(req.header_items())
        return _FakeResp(200, b'{"success": true}')

    with patch("limit_up_board.uploader.urlopen", side_effect=fake_urlopen):
        upload_summary_json(json_path)  # token 默认 None
    auth_keys = {k.lower() for k in captured["headers"]}
    assert "authorization" not in auth_keys

    with patch("limit_up_board.uploader.urlopen", side_effect=fake_urlopen):
        upload_summary_json(json_path, token="")
    auth_keys = {k.lower() for k in captured["headers"]}
    assert "authorization" not in auth_keys


def test_upload_no_default_token_constant_exported() -> None:
    """v0.12.3+：DEFAULT_UPLOAD_TOKEN 已下线，禁止再有源码硬编码 token。"""
    import limit_up_board.uploader as u

    assert not hasattr(u, "DEFAULT_UPLOAD_TOKEN")


def test_upload_forwards_extra_fields_in_multipart(tmp_path: Path) -> None:
    """v0.12.2+ — upload_summary_json 把 extra_fields 串成 multipart text part。"""
    json_path = _write_json(tmp_path)
    captured: dict = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured["data"] = req.data or b""
        return _FakeResp(200, b'{"success": true}')

    with patch("limit_up_board.uploader.urlopen", side_effect=fake_urlopen):
        upload_summary_json(
            json_path,
            extra_fields={"plugin_name": "打板策略", "trade_date": "20260522"},
        )

    body = captured["data"]
    assert '打板策略'.encode("utf-8") in body
    assert b"20260522" in body
    assert b'name="plugin_name"' in body
    assert b'name="trade_date"' in body
    assert b'filename="summary.json"' in body


def test_upload_http_error_maps_to_upload_error(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path)
    err_body = b'{"error":"Only JSON files are allowed"}'
    http_err = HTTPError(
        url=DEFAULT_UPLOAD_URL,
        code=400,
        msg="Bad Request",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(err_body),
    )
    with patch("limit_up_board.uploader.urlopen", side_effect=http_err):
        with pytest.raises(UploadError, match="HTTP 400"):
            upload_summary_json(json_path)


def test_upload_network_error_maps_to_upload_error(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(UploadError, match="network error"):
            upload_summary_json(json_path)


def test_upload_timeout_maps_to_upload_error(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        side_effect=TimeoutError("read timed out"),
    ):
        with pytest.raises(UploadError, match="network error"):
            upload_summary_json(json_path)


def test_upload_invalid_json_response_maps_to_upload_error(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        return_value=_FakeResp(200, b"<html>not json</html>"),
    ):
        with pytest.raises(UploadError, match="invalid JSON"):
            upload_summary_json(json_path)


def test_upload_non_object_json_response_maps_to_upload_error(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        return_value=_FakeResp(200, b'["unexpected", "list"]'),
    ):
        with pytest.raises(UploadError, match="unexpected JSON shape"):
            upload_summary_json(json_path)


def test_upload_unexpected_status_maps_to_upload_error(tmp_path: Path) -> None:
    """Some servers may reply 204 / 302 without raising HTTPError; reject those."""
    json_path = _write_json(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        return_value=_FakeResp(204, b""),
    ):
        with pytest.raises(UploadError, match="unexpected status"):
            upload_summary_json(json_path)
