"""MrRunner — orchestrates a full ``deeptrade market-review run`` (design §5.1).

Steps in order (design §5.1):

- **Step 0** — resolve the :class:`Window` from ``RunParams`` + the
  ``mr_trade_cal`` calendar (probing :func:`data.fetch_latest_trade_date`
  when neither ``--trade-date`` nor ``--start/--end`` was given).
- **Step 1** — sync the 27 PR-2 APIs into ``mr_*`` tables via
  :func:`data.sync_window`; if any LLM section will need板块 quotes (i.e.,
  always in v0.1), also call :func:`data.sync_sector_quotes`.
- **Step 2** — compute the 7 PR-3 metric reviews on the synced DB.
- **Step 3** — compute the input fingerprint, run the 7 PR-4 LLM sections
  via :func:`pipeline.run_sections`.
- **Step 4** — build :class:`ReviewReportSchema` (PR-5), write
  ``summary.json`` + ``summary.md`` + per-section ``*.md`` + audit dumps.
- **Step 5** — best-effort upload via :func:`report.upload.maybe_upload_summary`.

Persistence happens incrementally:

- ``mr_runs`` — single row inserted at Step 0 (status="running"), updated
  to the final status + summary JSON when the run completes / fails.
- ``mr_events`` — every :class:`StrategyEvent` emitted is also written.
- ``mr_stage_results`` — one row per LLM section after Step 3.

Status taxonomy (design §11):

- ``success`` — all 7 sections returned valid schemas, no per-step errors.
- ``partial_failed`` — at least one LLM section failed; data layer / metrics
  succeeded.
- ``failed`` — fatal step-level error (Tushare unauthorized, calendar
  empty, configuration missing, …); the run yields the recorded error
  event and exits.
- ``cancelled`` — reserved for SIGINT marker integration (PR-7+).
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from . import data
from .calendar import TradeCalendar
from .config import load_mr_config
from .metrics.breadth import compute_breadth
from .metrics.capital import compute_capital
from .metrics.leaders import compute_leaders
from .metrics.risk import compute_risk
from .metrics.sectors import compute_sectors
from .metrics.sentiment import compute_sentiment
from .metrics.style import compute_style
from .pipeline import MetricsBundle, run_sections
from .render import (
    dump_llm_calls_audit,
    dump_metrics_json,
    write_section_files,
    write_summary_md,
)
from .report.builder import build_review_report, write_summary_json
from .report.upload import maybe_upload_summary
from .schemas import SECTION_ORDER, SectionName
from .universe import build_window_universes
from .windows import Window, WindowSpecError, resolve_window

if TYPE_CHECKING:  # pragma: no cover
    from .runtime import MrRuntime

logger = logging.getLogger(__name__)


PLUGIN_VERSION = "0.1.0"


class PreconditionError(RuntimeError):
    """Caller-facing preflight failure (bad CLI args, missing token, ...).

    CLI translates these into a user-readable ``✘ {msg}`` + exit code 2.
    Distinct from :class:`RuntimeError` so the CLI layer can choose not to
    print a stack trace for these — they're user-facing, not bug reports.
    """


@dataclass(frozen=True)
class RunParams:
    """CLI inputs in a frozen container — design §7.1."""

    trade_date: str | None = None
    start: str | None = None
    end: str | None = None
    force_sync: bool = False
    llm_provider: str | None = None
    no_llm: bool = False
    no_upload: bool = False
    # v0.1 reserves only the fields actually wired up below; future flags
    # like --sections / --replay-only land here in PR-7 polish.


@dataclass
class RunOutcome:
    """What :meth:`MrRunner.execute` returns to the CLI.

    ``status`` mirrors the value persisted to ``mr_runs.status``. The CLI
    maps ``success`` / ``partial_failed`` → exit 0; ``failed`` /
    ``cancelled`` → exit 1.

    ``report_dir`` is the per-run reports directory (always populated even
    on failure — the failure event audit + partial outputs go there).
    """

    run_id: str
    status: str
    report_dir: Path
    failed_sections: list[str] = field(default_factory=list)
    error: str | None = None


class MrRunner:
    """Stateless-ish orchestrator. One ``MrRunner`` per run is fine; reuse
    is technically safe but no one bothers — construct → execute → done.
    """

    def __init__(
        self,
        rt: MrRuntime,
        *,
        ctx=None,
        plugin_version: str = PLUGIN_VERSION,
        reports_root: Path | None = None,
    ) -> None:
        self._rt = rt
        self._ctx = ctx
        self._plugin_version = plugin_version
        # PR-7 will switch to framework ``paths.reports_dir()`` once that
        # helper exists; for now an explicit override + a sane default
        # under the user dir keeps integration testing trivial.
        self._reports_root = reports_root or (Path.home() / ".deeptrade" / "reports")

    # ------------------------------------------------------------------
    # Public entries
    # ------------------------------------------------------------------

    def execute(self, params: RunParams) -> RunOutcome:
        """Full pipeline. Run the 5 steps, return the final outcome.

        Events are persisted into ``mr_events`` as they fire; callers that
        want the live stream should look at the mr_events table. For PR-6
        we keep the simpler return-the-outcome shape — the runner doesn't
        expose a generator.
        """
        return self._run(params, full=True)

    def execute_sync_only(self, params: RunParams) -> RunOutcome:
        """Steps 0-1 only — no metrics, no LLM, no upload.

        Used by ``market-review sync`` to refresh DB tables without paying
        for an LLM round-trip. Status is always ``success`` unless a step
        raised; partial_failed is meaningless without the LLM stages.
        """
        return self._run(params, full=False)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _run(self, params: RunParams, *, full: bool) -> RunOutcome:
        run_id = str(uuid.uuid4())
        started_at = datetime.now()
        seq = [0]  # mutable counter so ``_emit`` can increment

        # Resolve window (Step 0). Failures here are PreconditionErrors that
        # we DO surface back to the caller — but we don't persist a run row
        # for window-spec errors (lub's policy: only commit a run when the
        # framework has agreed to spend resources on it).
        try:
            window = self._resolve_window(params)
        except WindowSpecError as exc:
            raise PreconditionError(str(exc)) from exc

        # Reports directory created up-front so failure paths can still
        # write their audit files.
        report_dir = self._reports_root / run_id
        report_dir.mkdir(parents=True, exist_ok=True)

        params_json = json.dumps(asdict(params), ensure_ascii=False, sort_keys=True)
        self._insert_run_row(
            run_id=run_id, window=window, status="running",
            started_at=started_at, params_json=params_json,
        )

        try:
            self._emit(run_id, seq, "STEP_STARTED", "Step 0: 窗口已解析", payload={
                "anchor": window.anchor, "mode": window.mode,
                "n_days": window.n_days, "snapped_from": list(window.snapped_from or ()),
            })

            # Step 1: data sync
            self._emit(run_id, seq, "STEP_STARTED", "Step 1: 开始同步数据")
            sync_res = data.sync_window(self._rt, window, force_sync=params.force_sync)
            self._emit(
                run_id, seq, "STEP_FINISHED",
                f"sync_window 完成（{sum(sync_res.rows_materialized.values())} 行写入）",
                payload={
                    "rows_materialized": dict(sync_res.rows_materialized),
                    "empty_days": dict(sync_res.empty_days),
                    "duration_seconds": sync_res.duration_seconds,
                },
            )
            sec_res = data.sync_sector_quotes(self._rt, window, force_sync=params.force_sync)
            self._emit(
                run_id, seq, "STEP_FINISHED",
                f"sync_sector_quotes 完成（{sum(sec_res.rows_materialized.values())} 行）",
                payload={
                    "rows_materialized": dict(sec_res.rows_materialized),
                    "empty_days": dict(sec_res.empty_days),
                },
            )

            if not full:
                return self._finalize_sync_only(
                    run_id=run_id, window=window, started_at=started_at,
                    report_dir=report_dir,
                )

            # Step 2: metrics
            self._emit(run_id, seq, "STEP_STARTED", "Step 2: 指标聚合")
            universes = build_window_universes(self._rt.db, window.trade_dates)
            breadth = compute_breadth(self._rt.db, window, universes)
            sentiment = compute_sentiment(self._rt.db, window, universes, breadth)
            capital_view = compute_capital(self._rt.db, window, universes)
            sectors_view = compute_sectors(self._rt.db, window)
            leaders_view = compute_leaders(
                self._rt.db, window, universes, sector_review=sectors_view,
            )
            style_view = compute_style(self._rt.db, window)
            risk_view = compute_risk(
                self._rt.db, window, universes,
                breadth=breadth, capital=capital_view,
            )
            self._emit(run_id, seq, "STEP_FINISHED", "指标聚合完成")

            bundle = MetricsBundle(
                window=window, breadth=breadth, sentiment=sentiment,
                capital=capital_view, sectors=sectors_view, leaders=leaders_view,
                style=style_view, risk=risk_view,
            )

            # Step 3: LLM sections
            self._emit(run_id, seq, "STEP_STARTED", "Step 3: LLM 7 节复盘")
            input_fp = _compute_input_fingerprint(bundle, plugin_version=self._plugin_version)
            self._rt.input_fingerprint = input_fp
            self._rt.run_id = run_id
            section_results = run_sections(
                self._rt, bundle,
                llm_provider=params.llm_provider,
                input_fingerprint=input_fp,
                replay=self._build_replay_policy(),
            )
            # ``SectorsSection.provider`` is data-source metadata (THS / DC),
            # not LLM output — it lives in MrConfig. We removed it from the
            # _SECTORS_SYSTEM prompt allow-list (v0.1.9 fix for qwen-plus
            # 把 provider 填成 "板块轮动分析师" 的幻觉)，so overwrite it from
            # config now to keep the persisted JSON + report metadata honest.
            self._inject_sector_provider(section_results)
            self._persist_stage_results(run_id, section_results, params.llm_provider)
            failed_sections: list[SectionName] = [
                name for name in SECTION_ORDER
                if section_results[name].error is not None
            ]
            self._emit(
                run_id, seq, "STEP_FINISHED",
                f"LLM 完成（{len(SECTION_ORDER) - len(failed_sections)} / {len(SECTION_ORDER)} 节成功）",
                payload={"failed_sections": list(failed_sections)},
            )

            # Step 4: report
            self._emit(run_id, seq, "STEP_STARTED", "Step 4: 渲染报告")
            sections_map = {name: r.schema for name, r in section_results.items()}
            run_status = (
                "success" if not failed_sections else "partial_failed"
            )
            report = build_review_report(
                status=run_status,
                window=window,
                breadth=breadth, sentiment=sentiment, capital=capital_view,
                sectors=sectors_view, leaders=leaders_view, style=style_view,
                risk=risk_view,
                sections=sections_map,
                failed_sections=failed_sections,
                run_id=run_id,
                llm_provider=params.llm_provider or "",
                plugin_version=self._plugin_version,
                input_fingerprint=input_fp,
            )
            summary_json_path = write_summary_json(report_dir, report)
            write_summary_md(
                report_dir, run_id=run_id, window=window,
                results=section_results,
            )
            write_section_files(report_dir, section_results)
            dump_metrics_json(report_dir, bundle)
            dump_llm_calls_audit(report_dir, section_results)
            self._emit(
                run_id, seq, "STEP_FINISHED", "报告写盘完成",
                payload={"report_dir": str(report_dir)},
            )

            # Step 5: upload (best-effort)
            if not params.no_upload:
                for ev in maybe_upload_summary(
                    self._ctx, run_id=run_id, report_dir=report_dir, window=window,
                ):
                    self._record_event(run_id, seq, ev)

            # Finalize
            self._finalize_run_row(
                run_id=run_id, status=run_status, started_at=started_at,
                summary_json=summary_json_path.read_text(encoding="utf-8"),
                input_fingerprint=input_fp, error=None,
            )
            return RunOutcome(
                run_id=run_id, status=run_status, report_dir=report_dir,
                failed_sections=[str(s) for s in failed_sections],
            )

        except PreconditionError:
            # Re-raise — caller handles user-facing error rendering.
            self._finalize_run_row(
                run_id=run_id, status="failed", started_at=started_at,
                summary_json=None, input_fingerprint=None,
                error="PreconditionError",
            )
            raise
        except Exception as exc:  # noqa: BLE001 — design §11.1 fatal path
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.exception("run %s failed: %s", run_id, err_msg)
            self._emit(run_id, seq, "LOG", f"run failed: {err_msg}", level="ERROR")
            self._finalize_run_row(
                run_id=run_id, status="failed", started_at=started_at,
                summary_json=None, input_fingerprint=None,
                error=err_msg,
            )
            return RunOutcome(
                run_id=run_id, status="failed", report_dir=report_dir,
                error=err_msg,
            )

    def _finalize_sync_only(
        self, *, run_id: str, window: Window, started_at: datetime,
        report_dir: Path,
    ) -> RunOutcome:
        self._finalize_run_row(
            run_id=run_id, status="success", started_at=started_at,
            summary_json=None, input_fingerprint=None, error=None,
        )
        return RunOutcome(
            run_id=run_id, status="success", report_dir=report_dir,
        )

    # ------------------------------------------------------------------
    # Step 0 — window resolution
    # ------------------------------------------------------------------

    def _resolve_window(self, params: RunParams) -> Window:
        """Probe latest trade_date if needed, build calendar, resolve."""
        # Lazy-import pandas via TradeCalendar — keeps startup tight.
        cal_rows = self._fetch_trade_cal(force_sync=False)
        if cal_rows.empty:
            # First-time run on a fresh DB: pull trade_cal once, write it,
            # try again. The framework cache class is "static" so subsequent
            # runs hit the cache.
            if self._rt.tushare is None:
                raise PreconditionError(
                    "MrRuntime.tushare 未初始化；CLI 需要先 build_tushare_client(rt)"
                )
            df = self._rt.tushare.call("trade_cal", force_sync=True)
            if df is None or df.empty:
                raise PreconditionError(
                    "trade_cal 返回为空，无法构造日历；请检查 Tushare token 与网络"
                )
            self._rt.tushare.materialize("mr_trade_cal", df, key_cols=["exchange", "cal_date"])
            cal_rows = self._fetch_trade_cal(force_sync=False)

        calendar = TradeCalendar(cal_rows)
        latest = None
        if params.trade_date is None and params.start is None and params.end is None:
            if self._rt.tushare is None:
                raise PreconditionError("无 Tushare client；无法探测最新交易日")
            latest = data.fetch_latest_trade_date(self._rt.tushare)
        return resolve_window(
            calendar,
            trade_date=params.trade_date,
            start=params.start, end=params.end,
            latest_trade_date=latest,
        )

    def _fetch_trade_cal(self, *, force_sync: bool):
        """Fetch ``mr_trade_cal`` rows into a DataFrame for TradeCalendar."""
        import pandas as pd  # noqa: PLC0415 — pandas already loaded by data.py
        rows = self._rt.db.fetchall(
            "SELECT exchange, cal_date, is_open, pretrade_date FROM mr_trade_cal"
        )
        if not rows:
            return pd.DataFrame(columns=["exchange", "cal_date", "is_open", "pretrade_date"])
        return pd.DataFrame(rows, columns=["exchange", "cal_date", "is_open", "pretrade_date"])

    def _build_replay_policy(self):
        """Build a :class:`LLMReplayPolicy` from framework config + env.

        Tests with ``rt.config=None`` get ``None`` back (cache disabled,
        matches pre-v0.1 behavior). Real-world runs go through
        :func:`policy_from_env(policy_from_app_config(...))` so framework
        keys ``llm.replay.enabled / .write / .ttl_days`` AND env overrides
        ``DEEPTRADE_FRESH_LLM`` / ``DEEPTRADE_NO_LLM_REPLAY`` /
        ``DEEPTRADE_REPLAY_ONLY`` both take effect.
        """
        if self._rt.config is None:
            return None
        try:
            from deeptrade.plugins_api.llm import (  # noqa: PLC0415
                policy_from_app_config, policy_from_env,
            )
            app_cfg = self._rt.config.get_app_config()
            return policy_from_env(policy_from_app_config(app_cfg))
        except Exception:  # noqa: BLE001 — never block a run on cache plumbing
            logger.warning("replay policy build failed; running cache-disabled",
                           exc_info=True)
            return None

    # ------------------------------------------------------------------
    # mr_runs / mr_events / mr_stage_results persistence
    # ------------------------------------------------------------------

    def _insert_run_row(
        self, *, run_id: str, window: Window, status: str,
        started_at: datetime, params_json: str,
    ) -> None:
        self._rt.db.execute(
            """INSERT INTO mr_runs
            (run_id, mode, start_date, end_date, anchor, status,
             started_at, params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [run_id, window.mode, window.start, window.end, window.anchor,
             status, started_at, params_json],
        )

    def _finalize_run_row(
        self, *, run_id: str, status: str, started_at: datetime,
        summary_json: str | None, input_fingerprint: str | None,
        error: str | None,
    ) -> None:
        finished = datetime.now()
        self._rt.db.execute(
            """UPDATE mr_runs SET status = ?, finished_at = ?, summary_json = ?,
                input_fingerprint = ?, error = ?
               WHERE run_id = ?""",
            [status, finished, summary_json, input_fingerprint, error, run_id],
        )

    def _inject_sector_provider(self, results: dict) -> None:
        """Overwrite ``SectorsSection.provider`` from :class:`MrConfig`.

        Why: ``provider`` is data-source metadata (THS vs DC), not an LLM
        decision, so the prompt forbids the LLM from emitting it (v0.1.9).
        Without this hook, the persisted ``response_json`` and the rendered
        report would always show the schema default ``"ths"`` regardless of
        ``mr_config.mr.sector_provider``. Errors loading config degrade
        silently to the schema default — a stale config row should never
        block report rendering.
        """
        sectors = results.get("sectors")
        if sectors is None or sectors.schema is None:
            return
        try:
            cfg = load_mr_config(self._rt.db)
        except Exception as exc:  # noqa: BLE001 — config-read robustness
            logger.warning("load_mr_config failed; leaving sectors.provider default: %s", exc)
            return
        sectors.schema.provider = cfg.sector_provider

    def _persist_stage_results(
        self, run_id: str, results: dict, llm_provider: str | None,
    ) -> None:
        for section, result in results.items():
            response = result.schema.model_dump_json(by_alias=True)
            self._rt.db.execute(
                """INSERT INTO mr_stage_results
                (run_id, section, llm_provider, response_json)
                VALUES (?, ?, ?, ?)""",
                [run_id, section, llm_provider, response],
            )

    def _emit(
        self, run_id: str, seq: list[int], event_type: str, message: str,
        *, payload: dict | None = None, level: str = "INFO",
    ) -> None:
        ev = StrategyEvent(
            type=getattr(EventType, event_type),
            level=getattr(EventLevel, level),
            message=message,
            payload=payload or {},
        )
        self._record_event(run_id, seq, ev)

    def _record_event(self, run_id: str, seq: list[int], ev: StrategyEvent) -> None:
        seq[0] += 1
        payload_json = json.dumps(
            ev.payload, ensure_ascii=False, default=str,
        ) if ev.payload else None
        self._rt.db.execute(
            """INSERT INTO mr_events
            (run_id, seq, level, event_type, message, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            [run_id, seq[0], ev.level.name, ev.type.name, ev.message, payload_json],
        )


# ---------------------------------------------------------------------------
# Input fingerprint — sha256 of canonical JSON over the metric bundle
# ---------------------------------------------------------------------------


def _compute_input_fingerprint(
    bundle: MetricsBundle, *, plugin_version: str,
) -> str:
    """Deterministic 64-char hex digest of (window + 7 metric views + version).

    The fingerprint feeds :meth:`LLMClient.complete_json` so the framework
    replay cache can dedupe identical inputs across reruns. For v0.1 we
    canonicalize via :func:`_safe_canonical` (recursive dataclass→dict,
    sorted keys, NaN/Inf → None) and hash UTF-8 bytes.
    """
    payload = {
        "pluginVersion": plugin_version,
        "schemaVersion": "1.0",
        "window": _safe_canonical(bundle.window),
        "breadth": _safe_canonical(bundle.breadth),
        "sentiment": _safe_canonical(bundle.sentiment),
        "capital": _safe_canonical(bundle.capital),
        "sectors": _safe_canonical(bundle.sectors),
        "leaders": _safe_canonical(bundle.leaders),
        "style": _safe_canonical(bundle.style),
        "risk": _safe_canonical(bundle.risk),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_canonical(obj):
    from dataclasses import is_dataclass  # noqa: PLC0415
    import math  # noqa: PLC0415

    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _safe_canonical(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _safe_canonical(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_safe_canonical(v) for v in obj]
    if isinstance(obj, frozenset):
        return sorted(_safe_canonical(v) for v in obj)
    if isinstance(obj, float):
        # NaN / Inf serialize as None for stable hashing.
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)
