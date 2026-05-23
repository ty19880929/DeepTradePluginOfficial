"""P2-3 (v0.13.0)：运行级耗时 / 调用统计聚合器。

``RunMetrics`` 在 ``LubRunner.execute`` 期间观察事件流，被动累积：

* 阶段耗时（``stage_duration_ms``）：``STEP_STARTED`` → ``STEP_FINISHED``
* Tushare 调用：``EventType.TUSHARE_CALL`` / ``TUSHARE_FALLBACK`` 事件 payload
* LLM 批次：``LLM_BATCH_STARTED`` → ``LLM_BATCH_FINISHED`` 配对
* 上传：``_maybe_upload_summary`` 写入的 audit 字段
* LGB：通过 ``record_lgb`` 手动注入

收尾时 ``build_summary_payload`` 产出 ``OBSERVABILITY_SUMMARY`` 事件 / summary.json
``quality_metrics`` 共用的 dict。本模块不依赖 deeptrade 框架可变状态，便于单测。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

OBSERVABILITY_SUBTYPE = "observability_summary"
_LOG_PREFIX = "[observability]"


@dataclass
class _StageRecord:
    """One STEP_STARTED → STEP_FINISHED pair."""

    name: str
    started_at: float
    finished_at: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at) * 1000.0


@dataclass
class RunMetrics:
    """Mutable accumulator updated by ``observe_event`` on each pipeline event."""

    run_started_at: float = field(default_factory=time.monotonic)
    stages: list[_StageRecord] = field(default_factory=list)
    _stage_by_name: dict[str, _StageRecord] = field(default_factory=dict)
    tushare_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    _llm_open: dict[str, dict[str, Any]] = field(default_factory=dict)
    upload: dict[str, Any] | None = None
    lgb: dict[str, Any] | None = None
    validation_failed_count: int = 0

    def observe(self, event: StrategyEvent) -> None:
        """Tap each emitted event and update internal counters.

        Safe to call on every event — unknown types are ignored. Never
        raises (any error is treated as observability failure, which
        shouldn't blow up the run).
        """
        from deeptrade.plugins_api.events import EventType  # noqa: PLC0415

        try:
            now = time.monotonic()
            etype = event.type
            payload = event.payload or {}

            if etype == EventType.STEP_STARTED:
                rec = _StageRecord(name=event.message or "?", started_at=now)
                self.stages.append(rec)
                self._stage_by_name.setdefault(rec.name, rec)
            elif etype == EventType.STEP_FINISHED:
                # Match by message-prefix (e.g. "Step 1: ...") to the most-recent
                # not-yet-finished record sharing that prefix; fall back to the
                # last unfinished record.
                target: _StageRecord | None = None
                msg = event.message or ""
                prefix = msg.split(":", 1)[0].strip() if ":" in msg else msg.strip()
                for rec in reversed(self.stages):
                    rec_prefix = rec.name.split(":", 1)[0].strip() if ":" in rec.name else rec.name.strip()
                    if rec.finished_at is None and rec_prefix == prefix:
                        target = rec
                        break
                if target is None:
                    for rec in reversed(self.stages):
                        if rec.finished_at is None:
                            target = rec
                            break
                if target is not None:
                    target.finished_at = now
                    if payload:
                        target.payload.update(payload)

            elif etype in (EventType.TUSHARE_CALL, EventType.TUSHARE_FALLBACK):
                entry: dict[str, Any] = {
                    "event": etype.value,
                    "api": payload.get("api") or payload.get("api_name"),
                    "rows": payload.get("rows"),
                    "duration_ms": payload.get("duration_ms"),
                    "cache_hit": payload.get("cache_hit"),
                    "error": payload.get("error"),
                }
                self.tushare_calls.append(entry)

            elif etype == EventType.LLM_BATCH_STARTED:
                key = self._llm_key(payload)
                self._llm_open[key] = {
                    "provider": payload.get("provider"),
                    "model": payload.get("model"),
                    "batch_no": payload.get("batch_no"),
                    "stage": payload.get("stage"),
                    "started_at": now,
                }
            elif etype == EventType.LLM_BATCH_FINISHED:
                key = self._llm_key(payload)
                open_rec = self._llm_open.pop(key, None)
                if open_rec is None:
                    entry = {
                        "provider": payload.get("provider"),
                        "model": payload.get("model"),
                        "batch_no": payload.get("batch_no"),
                        "stage": payload.get("stage"),
                        "duration_ms": payload.get("duration_ms"),
                    }
                else:
                    entry = {
                        **{k: v for k, v in open_rec.items() if k != "started_at"},
                        "duration_ms": (now - open_rec["started_at"]) * 1000.0,
                    }
                # surface common extra fields if present
                for extra in ("repair_count", "candidates", "tokens_in", "tokens_out"):
                    if extra in payload:
                        entry[extra] = payload[extra]
                self.llm_calls.append(entry)

            elif etype == EventType.VALIDATION_FAILED:
                self.validation_failed_count += 1

            elif etype == EventType.LOG and isinstance(payload, dict) and "public_url" in payload:
                # _maybe_upload_summary emits the upload audit payload via LOG.
                self.upload = {
                    k: payload[k]
                    for k in (
                        "enabled", "url", "status", "duration_ms",
                        "public_url", "public_path", "error_class", "token_configured",
                    )
                    if k in payload
                }
        except Exception:  # noqa: BLE001 — observability never blocks a run
            pass

    @staticmethod
    def _llm_key(payload: dict[str, Any]) -> str:
        return (
            f"{payload.get('provider', '?')}::{payload.get('stage', '?')}::"
            f"{payload.get('batch_no', 0)}"
        )

    def record_lgb(self, *, model_id: str | None, coverage: float | None,
                   missing_rate: float | None) -> None:
        self.lgb = {
            "model_id": model_id,
            "coverage": coverage,
            "missing_rate": missing_rate,
        }

    def build_summary_payload(self) -> dict[str, Any]:
        """Produce the dict shipped in OBSERVABILITY_SUMMARY + summary.json."""
        return {
            "observability_summary": True,
            "run_duration_ms": (time.monotonic() - self.run_started_at) * 1000.0,
            "stage_duration_ms": {
                rec.name: rec.duration_ms for rec in self.stages
            },
            "tushare_api_calls": list(self.tushare_calls),
            "llm_calls": list(self.llm_calls),
            "lgb": self.lgb,
            "upload": self.upload,
            "validation_failed_count": self.validation_failed_count,
        }
