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
    DEFAULT_UPLOAD_TOKEN,
    DEFAULT_UPLOAD_URL,
    UploadError,
    _build_multipart,
    upload_summary_html,
)


# ---------------------------------------------------------------------------
# Multipart body
# ---------------------------------------------------------------------------


def test_build_multipart_has_expected_structure() -> None:
    body = _build_multipart(
        "BOUNDARY-X",
        field_name="file",
        filename="report.html",
        content=b"<html>hi</html>",
        mime="text/html",
    )
    text = body.decode("utf-8")
    assert text.startswith("--BOUNDARY-X\r\n")
    assert 'Content-Disposition: form-data; name="file"; filename="report.html"' in text
    assert "Content-Type: text/html" in text
    assert "<html>hi</html>" in text
    assert text.endswith("\r\n--BOUNDARY-X--\r\n")


# ---------------------------------------------------------------------------
# Path / suffix validation
# ---------------------------------------------------------------------------


def test_upload_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.html"
    with pytest.raises(UploadError, match="file not found"):
        upload_summary_html(missing)


def test_upload_rejects_non_html_suffix(tmp_path: Path) -> None:
    p = tmp_path / "summary.md"
    p.write_text("# not html", encoding="utf-8")
    with pytest.raises(UploadError, match="only .html files"):
        upload_summary_html(p)


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


def _write_html(tmp_path: Path) -> Path:
    p = tmp_path / "summary.html"
    p.write_text("<!doctype html><html><body>ok</body></html>", encoding="utf-8")
    return p


def test_upload_success_returns_decoded_json(tmp_path: Path) -> None:
    html = _write_html(tmp_path)
    body = json.dumps(
        {
            "success": True,
            "url": "https://blob.example.com/reports/2026-05-22/1.html",
            "pathname": "reports/2026-05-22/1.html",
            "index": 1,
            "date": "2026-05-22",
        }
    ).encode("utf-8")

    captured_request: dict = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured_request["url"] = req.full_url
        captured_request["headers"] = dict(req.header_items())
        captured_request["data_len"] = len(req.data or b"")
        captured_request["method"] = req.get_method()
        return _FakeResp(200, body)

    with patch("limit_up_board.uploader.urlopen", side_effect=fake_urlopen):
        result = upload_summary_html(html)

    assert result["success"] is True
    assert result["url"].endswith("/reports/2026-05-22/1.html")
    assert captured_request["url"] == DEFAULT_UPLOAD_URL
    assert captured_request["method"] == "POST"
    # Headers case-insensitive in HTTP but urllib normalizes to Title-Case keys.
    auth = {k.lower(): v for k, v in captured_request["headers"].items()}
    assert auth["authorization"] == f"Bearer {DEFAULT_UPLOAD_TOKEN}"
    assert auth["content-type"].startswith("multipart/form-data; boundary=")
    # Body should be larger than just the file bytes (multipart overhead).
    assert captured_request["data_len"] > html.stat().st_size


def test_upload_http_error_maps_to_upload_error(tmp_path: Path) -> None:
    html = _write_html(tmp_path)
    err_body = b'{"error":"Only HTML files are allowed"}'
    http_err = HTTPError(
        url=DEFAULT_UPLOAD_URL,
        code=400,
        msg="Bad Request",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(err_body),
    )
    with patch("limit_up_board.uploader.urlopen", side_effect=http_err):
        with pytest.raises(UploadError, match="HTTP 400"):
            upload_summary_html(html)


def test_upload_network_error_maps_to_upload_error(tmp_path: Path) -> None:
    html = _write_html(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(UploadError, match="network error"):
            upload_summary_html(html)


def test_upload_timeout_maps_to_upload_error(tmp_path: Path) -> None:
    html = _write_html(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        side_effect=TimeoutError("read timed out"),
    ):
        with pytest.raises(UploadError, match="network error"):
            upload_summary_html(html)


def test_upload_invalid_json_maps_to_upload_error(tmp_path: Path) -> None:
    html = _write_html(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        return_value=_FakeResp(200, b"<html>not json</html>"),
    ):
        with pytest.raises(UploadError, match="invalid JSON"):
            upload_summary_html(html)


def test_upload_non_object_json_maps_to_upload_error(tmp_path: Path) -> None:
    html = _write_html(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        return_value=_FakeResp(200, b'["unexpected", "list"]'),
    ):
        with pytest.raises(UploadError, match="unexpected JSON shape"):
            upload_summary_html(html)


def test_upload_unexpected_status_maps_to_upload_error(tmp_path: Path) -> None:
    """Some servers may reply 204 / 302 without raising HTTPError; reject those."""
    html = _write_html(tmp_path)
    with patch(
        "limit_up_board.uploader.urlopen",
        return_value=_FakeResp(204, b""),
    ):
        with pytest.raises(UploadError, match="unexpected status"):
            upload_summary_html(html)
