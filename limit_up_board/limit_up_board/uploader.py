"""Upload finished HTML reports to the DeepTrade server.

Stdlib-only multipart/form-data POST so the plugin does not gain a new
runtime dependency (e.g. requests). The endpoint contract:

    POST https://deeptrade.tiey.ai/api/reports/upload
    Authorization: Bearer deeptrade
    Content-Type: multipart/form-data; boundary=...
    field "file" — the HTML file (filename must end with .html)

On 200 OK the server returns JSON like::

    {
      "success": true,
      "url": "https://xxx.public.blob.vercel-storage.com/reports/2026-05-22/1.html",
      "pathname": "reports/2026-05-22/1.html",
      "index": 1,
      "date": "2026-05-22"
    }

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
DEFAULT_UPLOAD_TOKEN = "deeptrade"
DEFAULT_UPLOAD_TIMEOUT = 30.0


class UploadError(Exception):
    """Any failure while uploading the report to the DeepTrade endpoint."""


def upload_summary_html(
    html_path: Path,
    *,
    url: str = DEFAULT_UPLOAD_URL,
    token: str = DEFAULT_UPLOAD_TOKEN,
    timeout: float = DEFAULT_UPLOAD_TIMEOUT,
) -> dict[str, Any]:
    """POST *html_path* as form-data ``file`` to the reports endpoint.

    Returns the decoded JSON response on success. Raises :class:`UploadError`
    on any failure (missing file, non-.html suffix, HTTP non-200, network
    error, non-JSON body).
    """
    if not html_path.is_file():
        raise UploadError(f"file not found: {html_path}")
    if html_path.suffix.lower() != ".html":
        raise UploadError(f"only .html files are accepted (got {html_path.name!r})")

    payload = html_path.read_bytes()
    boundary = f"----DeepTradeBoundary{uuid.uuid4().hex}"
    body = _build_multipart(
        boundary,
        field_name="file",
        filename=html_path.name,
        content=payload,
        mime="text/html",
    )

    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
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
) -> bytes:
    """Assemble a minimal multipart/form-data body with a single file part."""
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")
    return head + content + tail
