"""Upload finished JSON reports to the DeepTrade server.

Stdlib-only multipart/form-data POST so the plugin does not gain a new
runtime dependency (e.g. requests). The endpoint contract (v0.12+):

    POST https://deeptrade.tiey.ai/api/reports/upload
    Authorization: Bearer <token>   # 可选；token 为空时不带此 header（匿名）
    Content-Type: multipart/form-data; boundary=...
    field "file"        — the JSON file (filename must end with .json)
    field "plugin_name" — 插件中文名（v0.12.2+）
    field "trade_date"  — 执行策略时的 T 日（YYYYMMDD，v0.12.2+）

On 200 OK the server returns JSON like::

    {
      "success": true,
      "url": "https://xxx.public.blob.vercel-storage.com/reports/2026-05-22/1.json",
      "pathname": "reports/2026-05-22/1.json",
      "index": 1,
      "date": "2026-05-22"
    }

v0.12.3 起 ``DEFAULT_UPLOAD_TOKEN`` 已下线；token 必须由调用方显式传入（从
``LubConfig.summary_upload_token`` 读取），空串/None 代表匿名上传。

All failure modes (bad path, HTTP non-200, network error, malformed JSON)
raise :class:`UploadError`; callers in ``runner.py`` catch it and degrade
to a WARN-level LOG event so an upload outage never fails a strategy run.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_UPLOAD_URL = "https://deeptrade.tiey.ai/api/reports/upload"
DEFAULT_UPLOAD_TIMEOUT = 30.0


class UploadError(Exception):
    """Any failure while uploading the report to the DeepTrade endpoint."""


def upload_summary_json(
    json_path: Path,
    *,
    url: str = DEFAULT_UPLOAD_URL,
    token: str | None = None,
    timeout: float = DEFAULT_UPLOAD_TIMEOUT,
    extra_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST *json_path* as form-data ``file`` to the reports endpoint.

    ``token`` (v0.12.3+) 为 None 或空串时不写 ``Authorization`` header，匿名上传
    由服务端决定是否接受。``extra_fields`` (v0.12.2+) adds additional
    ``multipart/form-data`` text parts alongside the file — used to send
    ``plugin_name`` / ``trade_date``.

    Returns the decoded JSON response on success. Raises :class:`UploadError`
    on any failure (missing file, non-.json suffix, HTTP non-200, network
    error, non-JSON body).
    """
    if not json_path.is_file():
        raise UploadError(f"file not found: {json_path}")
    if json_path.suffix.lower() != ".json":
        raise UploadError(f"only .json files are accepted (got {json_path.name!r})")

    payload = json_path.read_bytes()
    boundary = f"----DeepTradeBoundary{uuid.uuid4().hex}"
    body = _build_multipart(
        boundary,
        field_name="file",
        filename=json_path.name,
        content=payload,
        mime="application/json",
        text_fields=extra_fields,
    )

    headers: dict[str, str] = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(
        url,
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = getattr(resp, "status", None) or resp.getcode()
    except HTTPError as e:
        # Server replied with a non-2xx status; surface the body for the operator.
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            err_body = ""
        raise UploadError(
            f"HTTP {e.code}: {err_body[:200] if err_body else e.reason}"
        ) from e
    except (URLError, TimeoutError) as e:
        raise UploadError(f"network error: {e}") from e

    if status != 200:
        raise UploadError(f"unexpected status {status}: {raw[:200]!r}")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise UploadError(f"invalid JSON response: {raw[:200]!r}") from e
    if not isinstance(decoded, dict):
        raise UploadError(f"unexpected JSON shape (not an object): {decoded!r}")
    return decoded


def _build_multipart(
    boundary: str,
    *,
    field_name: str,
    filename: str,
    content: bytes,
    mime: str,
    text_fields: dict[str, str] | None = None,
) -> bytes:
    """Assemble a multipart/form-data body: text parts (if any) + file part."""
    parts: list[bytes] = []
    for name, value in (text_fields or {}).items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    file_head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")
    return b"".join(parts) + file_head + content + tail
