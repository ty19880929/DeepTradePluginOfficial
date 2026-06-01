"""Best-effort upload of ``summary.json`` — design §15.9.

Mirror of lub v0.16.1 ``_maybe_upload_summary`` semantics:

- Looks for ``<report_dir>/summary.json`` (PR-5 builder always emits this).
- Routes through framework ``ctx.make_report_uploader(run_id=...).upload(...)``
  which honors the user's ``report.upload.enabled / url / token / timeout``
  config (the framework, not the plugin, owns those keys).
- Surfaces every upload outcome as a :class:`StrategyEvent` so the
  dashboard / event stream can show "uploading…" → "ok" / "skipped" /
  "failed" with no further plumbing on the runner side.
- Never raises; failures are encoded into ``UploadResult.status`` and
  emitted as a WARN-level event.

The standalone-function shape (vs lub's method on the runner) lets PR-6
import + call without first having a fully-constructed runner. The
``ctx`` argument is whatever ``PluginContext`` the framework hands the
plugin at install time; tests pass a fake context with a
``make_report_uploader`` callable.

This module is the *only* place market-review references the framework
upload primitive. Schema-level concerns stay in :mod:`.schema` and
:mod:`.builder`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Protocol

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

if TYPE_CHECKING:  # pragma: no cover
    from ..windows import Window

logger = logging.getLogger(__name__)


# Plugin display name — design §15.9 step 2 maps this to the website's
# per-plugin aggregation bucket. Mirrors lub's ``"打板策略"`` literal.
PLUGIN_DISPLAY_NAME = "市场复盘"


class _UploaderProtocol(Protocol):
    """The shape :func:`maybe_upload_summary` consumes (lub-aligned)."""

    def upload(
        self,
        json_path: Path,
        *,
        plugin_name: str,
        trade_date: str,
        extra_fields: dict[str, str] | None = None,
    ): ...  # noqa: ANN201 — actual return is UploadResult


class _CtxProtocol(Protocol):
    def make_report_uploader(self, *, run_id: str | None = None) -> _UploaderProtocol: ...


def maybe_upload_summary(
    ctx: _CtxProtocol | None,
    *,
    run_id: str,
    report_dir: Path,
    window: Window,
) -> Iterator[StrategyEvent]:
    """Yield :class:`StrategyEvent`(s) describing the upload outcome.

    Generator semantics chosen to mirror lub: PR-6 runner consumes events
    into its existing event stream via ``yield from``. Callers that don't
    care about events just iterate to completion (``list(...)``).

    ``ctx`` is the framework :class:`PluginContext`. Pass ``None`` for
    test runs that should skip upload entirely; we emit a single LOG
    event to make the skip visible in the audit.
    """
    if ctx is None:
        yield _log(
            "ctx 未注入，跳过 summary.json 上传（v0.1 测试 / 临时调用路径）",
            payload={"status": "skipped_no_ctx", "anchor": window.anchor},
        )
        return

    json_path = Path(report_dir) / "summary.json"
    if not json_path.is_file():
        yield _log(
            f"summary.json 未生成，跳过上传：{json_path}",
            payload={
                "status": "skipped_no_local_file",
                "json_path": str(json_path),
                "anchor": window.anchor,
            },
        )
        return

    try:
        uploader = ctx.make_report_uploader(run_id=run_id)
    except Exception as exc:  # noqa: BLE001 — framework setup mustn't kill upload path
        logger.warning("make_report_uploader failed: %s", exc, exc_info=True)
        yield _warn(
            f"make_report_uploader 异常，跳过上传：{type(exc).__name__}: {exc}",
            payload={"status": "skipped_uploader_init_failed", "anchor": window.anchor},
        )
        return

    try:
        result = uploader.upload(
            json_path,
            plugin_name=PLUGIN_DISPLAY_NAME,
            trade_date=window.anchor,
        )
    except Exception as exc:  # noqa: BLE001 — framework promises no raise, but be safe
        logger.warning("ReportUploader.upload raised: %s", exc, exc_info=True)
        yield _warn(
            f"summary.json 上传异常：{type(exc).__name__}: {exc}",
            payload={"status": "raised", "anchor": window.anchor},
        )
        return

    status = getattr(result, "status", "unknown")
    duration_ms = getattr(result, "duration_ms", None)

    if isinstance(status, str) and status.startswith("skipped"):
        yield _log(
            f"summary.json 上传跳过：{status}",
            payload={
                "status": status, "anchor": window.anchor,
                "json_path": str(json_path),
                "duration_ms": duration_ms,
            },
        )
        return
    if status == "ok":
        yield _log(
            "summary.json 上传成功",
            payload={
                "status": status, "anchor": window.anchor,
                "json_path": str(json_path),
                "duration_ms": duration_ms,
            },
        )
        return

    # Any other status (failed_http / failed_network / unknown) → WARN.
    yield _warn(
        f"summary.json 上传失败：{status}",
        payload={
            "status": status, "anchor": window.anchor,
            "json_path": str(json_path),
            "duration_ms": duration_ms,
        },
    )


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def _log(message: str, *, payload: dict) -> StrategyEvent:
    return StrategyEvent(
        type=EventType.LOG,
        level=EventLevel.INFO,
        message=message,
        payload=payload,
    )


def _warn(message: str, *, payload: dict) -> StrategyEvent:
    return StrategyEvent(
        type=EventType.LOG,
        level=EventLevel.WARN,
        message=message,
        payload=payload,
    )
