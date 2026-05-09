"""Plugin-internal run lifecycle for volume-anomaly.

Three modes — screen / analyze / prune — each with its own execute method.
All write to ``va_runs`` (with ``mode`` column) and ``va_events``.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from deeptrade.core.run_status import RunStatus
from deeptrade.core.tushare_client import TushareUnauthorizedError
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from .calendar import TradeCalendar
from .data import (
    EVALUATE_DEFAULT_LOOKBACK_DAYS,
    EVALUATE_HORIZONS,
    EVALUATE_WINDOW_5D,
    EVALUATE_WINDOW_10D,
    AnalyzeBundle,
    ScreenResult,
    ScreenRules,
    _classify_data_status,
    _compute_realized_returns,
    _resolve_horizon_dates,
    append_anomaly_history,
    collect_analyze_bundle,
    fetch_anomaly_dates_within_lookback,
    fetch_completed_realized_keys,
    prune_watchlist,
    resolve_trade_date,
    screen_anomalies,
    upsert_realized_return,
    upsert_watchlist,
)
from .pipeline import run_analyze
from .render import (
    EvaluateOutcome,
    export_llm_calls,
    render_terminal_summary,
    write_analyze_report,
    write_evaluate_report,
    write_prune_report,
    write_screen_report,
)
from .runtime import VaRuntime, build_tushare_client, pick_llm_provider
from .schemas import VATrendCandidate

logger = logging.getLogger(__name__)

DEFAULT_PRUNE_DAYS = 30


@dataclass
class ScreenParams:
    trade_date: str | None = None
    allow_intraday: bool = False
    force_sync: bool = False
    screen_rules: dict[str, Any] | None = None


@dataclass
class AnalyzeParams:
    trade_date: str | None = None
    allow_intraday: bool = False
    force_sync: bool = False


@dataclass
class PruneParams:
    trade_date: str | None = None
    allow_intraday: bool = False
    days: int = DEFAULT_PRUNE_DAYS


@dataclass
class EvaluateParams:
    """v0.4.0 P1-3 — evaluate mode: compute T+N realised returns for past hits."""
    trade_date: str | None = None
    allow_intraday: bool = False
    lookback_days: int = EVALUATE_DEFAULT_LOOKBACK_DAYS
    backfill_all: bool = False         # F12 — when True, ignore lookback_days
    force_recompute: bool = False      # re-evaluate rows that are already 'complete'
    force_sync: bool = False


@dataclass
class RunOutcome:
    run_id: str
    status: RunStatus
    error: str | None
    seen_events: list[StrategyEvent]


class VaRunner:
    def __init__(self, rt: VaRuntime) -> None:
        self._rt = rt
        self._pending: list[StrategyEvent] = []

    # ----- entry points --------------------------------------------------

    def execute_screen(self, params: ScreenParams) -> RunOutcome:
        return self._drive("screen", params, self._iter_screen(params))

    def execute_analyze(self, params: AnalyzeParams) -> RunOutcome:
        return self._drive("analyze", params, self._iter_analyze(params))

    def execute_prune(self, params: PruneParams) -> RunOutcome:
        return self._drive("prune", params, self._iter_prune(params))

    def execute_evaluate(self, params: EvaluateParams) -> RunOutcome:
        # G10 — evaluate writes va_runs / va_events with mode='evaluate' so it
        # appears in `deeptrade volume-anomaly history` alongside other modes.
        return self._drive("evaluate", params, self._iter_evaluate(params))

    # ----- driver --------------------------------------------------------

    def _drive(
        self, mode: str, params: Any, iterator: Iterable[StrategyEvent]
    ) -> RunOutcome:
        run_id = str(uuid.uuid4())
        self._rt.run_id = run_id
        self._rt.is_intraday = bool(getattr(params, "allow_intraday", False))
        self._record_run_start(run_id, mode, params)

        events: list[StrategyEvent] = []
        seen_validation_failed = False
        terminal_status = RunStatus.SUCCESS
        terminal_error: str | None = None

        try:
            seq = 0
            for ev in iterator:
                seq += 1
                self._persist_event(run_id, seq, ev)
                events.append(ev)
                self._render_event(ev)
                if ev.type == EventType.VALIDATION_FAILED:
                    seen_validation_failed = True
        except KeyboardInterrupt:
            terminal_status = RunStatus.CANCELLED
            terminal_error = "KeyboardInterrupt"
        except Exception as e:  # noqa: BLE001
            terminal_status = RunStatus.FAILED
            terminal_error = f"{type(e).__name__}: {e}"
            logger.exception("volume-anomaly %s run %s raised", mode, run_id)

        if terminal_status == RunStatus.SUCCESS and seen_validation_failed:
            terminal_status = RunStatus.PARTIAL_FAILED

        self._record_run_finish(run_id, terminal_status, terminal_error, events)
        return RunOutcome(
            run_id=run_id, status=terminal_status, error=terminal_error, seen_events=events
        )

    # ----- screen --------------------------------------------------------

    def _iter_screen(self, params: ScreenParams) -> Iterable[StrategyEvent]:
        rt = self._rt
        rt.tushare = build_tushare_client(
            rt, intraday=params.allow_intraday, event_cb=self._on_tushare_event
        )
        cfg = rt.config.get_app_config()

        yield rt.emit(EventType.STEP_STARTED, "Step 0: resolve trade date")
        cal_df = rt.tushare.call("trade_cal")
        cal = TradeCalendar(cal_df)
        T, T1 = resolve_trade_date(
            datetime.now(),
            cal,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
            close_after=cfg.app_close_after if cfg is not None else time(18, 0),
        )
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 0: T={T} T+1={T1}",
            payload={"trade_date": T, "next_trade_date": T1},
        )

        rules = ScreenRules.from_dict(params.screen_rules)
        yield rt.emit(EventType.DATA_SYNC_STARTED, "Step 1: screen anomalies")
        try:
            result: ScreenResult = screen_anomalies(
                tushare=rt.tushare,
                calendar=cal,
                trade_date=T,
                rules=rules,
                force_sync=params.force_sync,
            )
        except TushareUnauthorizedError as e:
            yield rt.emit(
                EventType.LOG, f"required tushare api unauthorized: {e}", level=EventLevel.ERROR
            )
            raise
        yield from self._drain_pending()
        yield rt.emit(
            EventType.DATA_SYNC_FINISHED,
            f"funnel: {result.n_main_board} → {result.n_after_st_susp} → "
            f"{result.n_after_t_day_rules} → {result.n_after_turnover} → "
            f"{result.n_after_vol_rules}",
            payload={
                "n_main_board": result.n_main_board,
                "n_after_st_susp": result.n_after_st_susp,
                "n_after_t_day_rules": result.n_after_t_day_rules,
                "n_after_turnover": result.n_after_turnover,
                "n_after_vol_rules": result.n_after_vol_rules,
                "data_unavailable": result.data_unavailable,
            },
        )

        n_new, n_updated = upsert_watchlist(rt.db, result.hits, trade_date=T)
        append_anomaly_history(rt.db, result.hits)
        watchlist_total = int(rt.db.fetchone("SELECT COUNT(*) FROM va_watchlist")[0])

        report_path = write_screen_report(
            rt.run_id,
            status=RunStatus.SUCCESS,
            is_intraday=params.allow_intraday,
            result=result,
            n_new=n_new,
            n_updated=n_updated,
            watchlist_total=watchlist_total,
        )
        export_llm_calls(rt.run_id, rt.db)
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"screen done — {n_new} new, {n_updated} updated, pool={watchlist_total}",
            payload={
                "report_dir": str(report_path),
                "n_new": n_new,
                "n_updated": n_updated,
                "watchlist_total": watchlist_total,
                "n_hits": len(result.hits),
            },
        )

    # ----- analyze -------------------------------------------------------

    def _iter_analyze(self, params: AnalyzeParams) -> Iterable[StrategyEvent]:
        rt = self._rt
        rt.tushare = build_tushare_client(
            rt, intraday=params.allow_intraday, event_cb=self._on_tushare_event
        )
        from deeptrade.core import paths

        provider_name = pick_llm_provider(rt)
        reports_dir = paths.reports_dir() / rt.run_id if rt.run_id else None
        llm = rt.llms.get_client(
            provider_name,
            plugin_id=rt.plugin_id,
            run_id=rt.run_id,
            reports_dir=reports_dir,
        )
        cfg = rt.config.get_app_config()

        yield rt.emit(EventType.STEP_STARTED, "Step 0: resolve trade date")
        cal_df = rt.tushare.call("trade_cal")
        cal = TradeCalendar(cal_df)
        T, T1 = resolve_trade_date(
            datetime.now(),
            cal,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
            close_after=cfg.app_close_after if cfg is not None else time(18, 0),
        )
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 0: T={T} T+1={T1}",
            payload={"trade_date": T, "next_trade_date": T1},
        )

        yield rt.emit(EventType.STEP_STARTED, "Step 1: data assembly")
        try:
            bundle: AnalyzeBundle = collect_analyze_bundle(
                tushare=rt.tushare,
                db=rt.db,
                calendar=cal,
                trade_date=T,
                next_trade_date=T1,
                force_sync=params.force_sync,
            )
        except TushareUnauthorizedError as e:
            yield rt.emit(
                EventType.LOG, f"required tushare api unauthorized: {e}", level=EventLevel.ERROR
            )
            raise
        yield from self._drain_pending()
        # G8 — explicit WARN log when index_daily is unavailable so users notice
        # the alpha-field degradation rather than silently losing the signal.
        for entry in bundle.data_unavailable:
            if entry.startswith("index_daily"):
                yield rt.emit(
                    EventType.LOG,
                    entry,
                    level=EventLevel.WARN,
                )
                break
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 1: {len(bundle.candidates)} candidates from watchlist",
            payload={
                "candidates": len(bundle.candidates),
                "data_unavailable": bundle.data_unavailable,
                "sector_strength_source": bundle.sector_strength_source,
            },
        )

        if not bundle.candidates:
            yield from self._emit_empty_analyze_report(bundle, params, reason="empty watchlist")
            return

        preset = cfg.app_profile  # v0.7: per-stage tuning resolved by plugin
        analyze_result = None
        for ev, res in run_analyze(llm=llm, bundle=bundle, preset=preset):
            yield ev
            if res is not None:
                analyze_result = res

        predictions: list[VATrendCandidate] = (
            analyze_result.predictions if analyze_result else []
        )
        market_ctx_summary = (
            analyze_result.market_context_summaries[0]
            if analyze_result and analyze_result.market_context_summaries
            else None
        )
        risk_disclaimer = (
            analyze_result.risk_disclaimers[0]
            if analyze_result and analyze_result.risk_disclaimers
            else None
        )

        terminal_status = RunStatus.SUCCESS
        if analyze_result and analyze_result.failed_batches > 0:
            terminal_status = RunStatus.PARTIAL_FAILED

        _write_stage_results(rt, "analyze", predictions, bundle)
        failed_batches = list(analyze_result.failed_batch_ids) if analyze_result else []

        report_path = write_analyze_report(
            rt.run_id,
            status=terminal_status,
            is_intraday=params.allow_intraday,
            bundle=bundle,
            predictions=predictions,
            market_context_summary=market_ctx_summary,
            risk_disclaimer=risk_disclaimer,
            failed_batch_ids=failed_batches or None,
        )
        export_llm_calls(rt.run_id, rt.db)

        n_imminent = sum(1 for c in predictions if c.prediction == "imminent_launch")
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"Report written: {report_path}",
            payload={
                "report_dir": str(report_path),
                "predictions": len(predictions),
                "imminent_launch": n_imminent,
            },
        )

    def _emit_empty_analyze_report(
        self, bundle: AnalyzeBundle, params: AnalyzeParams, *, reason: str
    ) -> Iterable[StrategyEvent]:
        rt = self._rt
        report_path = write_analyze_report(
            rt.run_id,
            status=RunStatus.SUCCESS,
            is_intraday=params.allow_intraday,
            bundle=bundle,
            predictions=[],
            market_context_summary=None,
            risk_disclaimer=None,
        )
        export_llm_calls(rt.run_id, rt.db)
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"empty analyze report ({reason})",
            payload={"report_dir": str(report_path), "reason": reason},
        )

    # ----- prune ---------------------------------------------------------

    def _iter_prune(self, params: PruneParams) -> Iterable[StrategyEvent]:
        rt = self._rt
        rt.tushare = build_tushare_client(
            rt, intraday=params.allow_intraday, event_cb=self._on_tushare_event
        )
        cfg = rt.config.get_app_config()

        if params.days < 0:
            raise ValueError(f"days must be ≥ 0, got {params.days}")

        yield rt.emit(EventType.STEP_STARTED, "Step 0: resolve today")
        cal_df = rt.tushare.call("trade_cal")
        cal = TradeCalendar(cal_df)
        today, _ = resolve_trade_date(
            datetime.now(),
            cal,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
            close_after=cfg.app_close_after if cfg is not None else time(18, 0),
        )
        yield rt.emit(
            EventType.STEP_FINISHED, f"Step 0: today={today}", payload={"today": today}
        )

        before_total = int(rt.db.fetchone("SELECT COUNT(*) FROM va_watchlist")[0])
        yield rt.emit(EventType.STEP_STARTED, "Step 1: prune watchlist")
        pruned = prune_watchlist(rt.db, min_tracked_calendar_days=params.days, today=today)
        remaining = int(rt.db.fetchone("SELECT COUNT(*) FROM va_watchlist")[0])
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"pruned {len(pruned)}; remaining {remaining}",
            payload={
                "pruned": len(pruned),
                "before_total": before_total,
                "remaining": remaining,
                "min_tracked_days": params.days,
            },
        )

        report_path = write_prune_report(
            rt.run_id,
            status=RunStatus.SUCCESS,
            today=today,
            min_tracked_days=params.days,
            pruned=pruned,
            watchlist_remaining=remaining,
        )
        export_llm_calls(rt.run_id, rt.db)
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"prune done — removed {len(pruned)} / remaining {remaining}",
            payload={
                "report_dir": str(report_path),
                "pruned": len(pruned),
                "watchlist_remaining": remaining,
            },
        )

    # ----- evaluate (v0.4.0 P1-3) ----------------------------------------

    def _iter_evaluate(self, params: EvaluateParams) -> Iterable[StrategyEvent]:
        rt = self._rt
        rt.tushare = build_tushare_client(
            rt, intraday=params.allow_intraday, event_cb=self._on_tushare_event
        )
        cfg = rt.config.get_app_config()

        yield rt.emit(EventType.STEP_STARTED, "Step 0: resolve today")
        cal_df = rt.tushare.call("trade_cal")
        cal = TradeCalendar(cal_df)
        today, _ = resolve_trade_date(
            datetime.now(),
            cal,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
            close_after=cfg.app_close_after if cfg is not None else time(18, 0),
        )
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 0: today={today}",
            payload={"today": today},
        )

        # F12 — backfill_all overrides lookback_days; everyone gets re-evaluated.
        lookback = 365 * 10 if params.backfill_all else params.lookback_days
        anomaly_pairs = fetch_anomaly_dates_within_lookback(
            rt.db, today=today, lookback_days=lookback
        )
        completed = (
            set() if params.force_recompute else fetch_completed_realized_keys(rt.db)
        )
        # Skip already-complete rows unless --force-recompute.
        targets = [pair for pair in anomaly_pairs if pair not in completed]
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 1: {len(targets)} target hits "
            f"({len(anomaly_pairs)} total within lookback, "
            f"{len(anomaly_pairs) - len(targets)} already complete)",
            payload={
                "targets": len(targets),
                "total_in_lookback": len(anomaly_pairs),
                "skipped_complete": len(anomaly_pairs) - len(targets),
                "lookback_days": lookback,
            },
        )

        # Group target codes by anomaly_date so we can batch the tushare calls.
        by_date: dict[str, list[str]] = {}
        for adate, code in targets:
            by_date.setdefault(adate, []).append(code)

        # Resolve required future trade_dates for every distinct anomaly_date.
        # We'll fetch each unique trade_date (T + horizon) once via daily(td=...)
        # and cache the per-code close lookup.
        all_dates_to_fetch: set[str] = set()
        anomaly_horizon_dates: dict[str, dict[int, str]] = {}
        for adate in by_date:
            try:
                horizon_dates = _resolve_horizon_dates(cal, adate, EVALUATE_HORIZONS)
            except ValueError as e:
                logger.warning("evaluate: cannot resolve horizons for %s (%s)", adate, e)
                horizon_dates = {}
            anomaly_horizon_dates[adate] = horizon_dates
            all_dates_to_fetch.add(adate)
            all_dates_to_fetch.update(
                d for d in horizon_dates.values() if d is not None and d <= today
            )

        # Fetch all needed daily(trade_date=X) frames once. The TushareClient
        # already caches each call as trade_day_immutable so subsequent runs
        # are zero-incremental.
        close_by_code_date: dict[tuple[str, str], float] = {}
        n_fetched = 0
        for d in sorted(all_dates_to_fetch):
            df = rt.tushare.call("daily", trade_date=d, force_sync=params.force_sync)
            if df is None or df.empty or "close" not in df.columns:
                continue
            n_fetched += 1
            for r in df[["ts_code", "close"]].itertuples(index=False):
                if r.close is not None:
                    close_by_code_date[(str(r.ts_code), str(d))] = float(r.close)

        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 2: fetched daily for {n_fetched}/{len(all_dates_to_fetch)} unique dates",
            payload={
                "dates_fetched": n_fetched,
                "dates_planned": len(all_dates_to_fetch),
            },
        )
        yield from self._drain_pending()

        # 5d / 10d window dates for max_close / max_dd computation.
        # We need T+1..T+5 and T+1..T+10 trade dates.
        n_complete = 0
        n_partial = 0
        n_pending = 0
        for adate, codes in by_date.items():
            horizon_dates = anomaly_horizon_dates.get(adate, {})
            window5_dates = self._range_horizon_dates(cal, adate, EVALUATE_WINDOW_5D)
            window10_dates = self._range_horizon_dates(cal, adate, EVALUATE_WINDOW_10D)
            # Make sure those window dates are also fetched (they typically
            # overlap with horizon_dates so we just augment the cache lazily).
            extra = (set(window5_dates) | set(window10_dates)) - all_dates_to_fetch
            extra = {d for d in extra if d <= today}
            for d in extra:
                df = rt.tushare.call(
                    "daily", trade_date=d, force_sync=params.force_sync
                )
                if df is None or df.empty or "close" not in df.columns:
                    continue
                for r in df[["ts_code", "close"]].itertuples(index=False):
                    if r.close is not None:
                        close_by_code_date[(str(r.ts_code), str(d))] = float(r.close)
            for code in codes:
                t_close = close_by_code_date.get((code, adate))
                horizon_closes = {
                    n: close_by_code_date.get((code, d))
                    for n, d in horizon_dates.items()
                }
                window5_closes = [
                    close_by_code_date.get((code, d)) for d in window5_dates
                ]
                window10_closes = [
                    close_by_code_date.get((code, d)) for d in window10_dates
                ]
                metrics = _compute_realized_returns(
                    t_close=t_close,
                    horizon_closes=horizon_closes,
                    window_5d_closes=window5_closes,
                    window_10d_closes=window10_closes,
                )
                status = _classify_data_status(
                    horizon_closes=horizon_closes,
                    horizons=EVALUATE_HORIZONS,
                    today=today,
                    horizon_dates=horizon_dates,
                )
                upsert_realized_return(
                    rt.db,
                    anomaly_date=adate,
                    ts_code=code,
                    t_close=t_close,
                    horizon_closes=horizon_closes,
                    metrics=metrics,
                    data_status=status,
                )
                if status == "complete":
                    n_complete += 1
                elif status == "partial":
                    n_partial += 1
                else:
                    n_pending += 1

        outcome = EvaluateOutcome(
            today=today,
            n_targets=len(targets),
            n_skipped_complete=len(anomaly_pairs) - len(targets),
            n_complete=n_complete,
            n_partial=n_partial,
            n_pending=n_pending,
            lookback_days=lookback,
            backfill_all=params.backfill_all,
        )
        report_path = write_evaluate_report(rt.run_id, outcome=outcome)
        export_llm_calls(rt.run_id, rt.db)
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"evaluate done — complete={n_complete}, partial={n_partial}, "
            f"pending={n_pending}",
            payload={
                "report_dir": str(report_path),
                "n_complete": n_complete,
                "n_partial": n_partial,
                "n_pending": n_pending,
                "lookback_days": lookback,
                "backfill_all": params.backfill_all,
            },
        )

    @staticmethod
    def _range_horizon_dates(
        cal: TradeCalendar, anomaly_date: str, n: int
    ) -> list[str]:
        """Return the n trade dates immediately AFTER ``anomaly_date``
        (T+1..T+n). Stops early if the calendar runs out, returning shorter
        list (caller handles ``None`` for missing entries via the close map)."""
        out: list[str] = []
        cursor = anomaly_date
        for _ in range(n):
            try:
                cursor = cal.next_open(cursor)
            except ValueError:
                break
            out.append(cursor)
        return out

    # ----- helpers -------------------------------------------------------

    def _on_tushare_event(self, event_type: str, message: str, payload: dict) -> None:
        try:
            etype = EventType(event_type)
        except ValueError:
            logger.warning("unknown tushare event type: %s", event_type)
            return
        self._pending.append(
            StrategyEvent(type=etype, level=EventLevel.WARN, message=message, payload=payload)
        )

    def _drain_pending(self) -> Iterable[StrategyEvent]:
        while self._pending:
            yield self._pending.pop(0)

    def _render_event(self, ev: StrategyEvent) -> None:
        glyph = "✔" if ev.level == EventLevel.INFO else ("⚠" if ev.level == EventLevel.WARN else "✘")
        print(f"  {glyph} [{ev.type.value}] {ev.message}", flush=True)

    # ----- DB helpers ----------------------------------------------------

    def _record_run_start(self, run_id: str, mode: str, params: Any) -> None:
        self._rt.db.execute(
            "INSERT INTO va_runs(run_id, mode, trade_date, status, is_intraday, started_at, "
            "params_json) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
            (
                run_id,
                mode,
                getattr(params, "trade_date", None) or "",
                RunStatus.RUNNING.value,
                bool(getattr(params, "allow_intraday", False)),
                json.dumps(params.__dict__, ensure_ascii=False, default=str),
            ),
        )

    def _record_run_finish(
        self,
        run_id: str,
        status: RunStatus,
        error: str | None,
        events: list[StrategyEvent],
    ) -> None:
        summary = {
            "event_count": len(events),
            "validation_failed_count": sum(
                1 for e in events if e.type == EventType.VALIDATION_FAILED
            ),
        }
        self._rt.db.execute(
            "UPDATE va_runs SET status=?, finished_at=CURRENT_TIMESTAMP, "
            "summary_json=?, error=? WHERE run_id=?",
            (status.value, json.dumps(summary, ensure_ascii=False), error, run_id),
        )

    def _persist_event(self, run_id: str, seq: int, ev: StrategyEvent) -> None:
        self._rt.db.execute(
            "INSERT INTO va_events(run_id, seq, level, event_type, message, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                seq,
                ev.level.value,
                ev.type.value,
                ev.message,
                json.dumps(ev.payload, ensure_ascii=False, default=str),
            ),
        )


def _write_stage_results(
    rt: VaRuntime,
    stage: str,
    items: list[VATrendCandidate],
    bundle: AnalyzeBundle,
) -> None:
    if not items:
        return
    tracked_lookup = {
        c["candidate_id"]: int(c.get("tracked_days") or 0)
        for c in bundle.candidates
        if isinstance(c, dict)
    }
    for item in items:
        d = item.model_dump(mode="json")
        # v0.6.0 P1-2 — split dimension_scores into 6 dedicated DOUBLE columns
        # so `stats --by dimension_scores` can aggregate via plain SQL (G6).
        dim = d.get("dimension_scores") or {}
        rt.db.execute(
            "INSERT INTO va_stage_results(run_id, stage, batch_no, trade_date, ts_code, name, "
            "rank, launch_score, confidence, prediction, pattern, rationale, tracked_days, "
            "evidence_json, risk_flags_json, raw_response_json, "
            "dim_washout, dim_pattern, dim_capital, dim_sector, dim_historical, dim_risk) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rt.run_id,
                stage,
                d.get("batch_no", 0),
                bundle.trade_date,
                d.get("ts_code", ""),
                d.get("name"),
                d.get("rank"),
                d.get("launch_score"),
                d.get("confidence"),
                d.get("prediction"),
                d.get("pattern"),
                d.get("rationale"),
                tracked_lookup.get(d.get("candidate_id", ""), 0),
                json.dumps(d.get("key_evidence") or [], ensure_ascii=False),
                json.dumps(d.get("risk_flags") or [], ensure_ascii=False),
                json.dumps(d, ensure_ascii=False),
                dim.get("washout"),
                dim.get("pattern"),
                dim.get("capital"),
                dim.get("sector"),
                dim.get("historical"),
                dim.get("risk"),
            ),
        )


def render_finished_run(run_id: str) -> None:
    render_terminal_summary(run_id)
