"""Plugin-internal run lifecycle: drives the pipeline generator, persists
events to ``lub_events``, and writes the run record to ``lub_runs``.

Replaces the deleted framework-side ``core/strategy_runner.py``: each plugin
manages its own run history on Plan A's pure-isolation model.

v0.8 — debate mode (multi-LLM): when ``RunParams.debate`` is set, R1 + R2 +
optional final_ranking + R3 (debate revision) all fan out across configured
LLM providers with one worker thread per provider. Each worker uses an
isolated ``LubRuntime`` (private DB connection + LLMManager) so concurrent
``LLMClient.complete_json`` calls don't share lock/audit-write bookkeeping.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deeptrade.core.run_status import RunStatus
from deeptrade.core.tushare_client import TushareUnauthorizedError
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.config import ConfigService
    from deeptrade.core.llm_client import LLMClient

from .calendar import TradeCalendar
from .config import LubConfig, load_config
from .data import Round1Bundle, collect_round1, resolve_trade_date
from .lgb.audit import record_predictions as _record_lgb_predictions
from .lgb.scorer import LgbScorer
from .pipeline import (
    DebateRoundResult,
    RoundResult,
    run_final_ranking,
    run_r1,
    run_r2,
    run_r3_debate,
    select_finalists,
)
from .prompts import assign_peer_labels
from .render import export_llm_calls, render_terminal_summary, write_report
from .runtime import (
    LubRuntime,
    build_tushare_client,
    open_worker_runtime,
    pick_llm_provider,
)
from .schemas import (
    ContinuationCandidate,
    FinalRankingResponse,
    RevisedContinuationCandidate,
)
from .ui import EventRenderer, LegacyStreamRenderer, NullRenderer

logger = logging.getLogger(__name__)


def _safe_prev_trade_date(cal: TradeCalendar, trade_date: str) -> str | None:
    try:
        return cal.pretrade_date(trade_date)
    except ValueError:
        return None


def _settings_log_event(rt: LubRuntime, lub_cfg: LubConfig) -> StrategyEvent:
    """LOG event announcing the active settings before Step 1."""
    return rt.emit(
        EventType.LOG,
        (
            f"运行配置: {lub_cfg.min_float_mv_yi}亿 < 流通市值 < "
            f"{lub_cfg.max_float_mv_yi}亿、股价 < {lub_cfg.max_close_yuan}元"
        ),
        payload={
            "min_float_mv_yi": lub_cfg.min_float_mv_yi,
            "max_float_mv_yi": lub_cfg.max_float_mv_yi,
            "max_close_yuan": lub_cfg.max_close_yuan,
        },
    )


# ---------------------------------------------------------------------------
# Run params (replaces deleted StrategyParams)
# ---------------------------------------------------------------------------


@dataclass
class RunParams:
    trade_date: str | None = None
    allow_intraday: bool = False
    force_sync: bool = False
    daily_lookback: int = 30
    moneyflow_lookback: int = 5
    debate: bool = False
    debate_llms: list[str] | None = None
    # v0.5 LGB 开关：用户传 --no-lgb 时设为 False（一次性覆盖 LubConfig.lgb_enabled）。
    # PR-0.3 仅落字段，pipeline 接入在 PR-2.2。
    lgb_enabled: bool = True


# ---------------------------------------------------------------------------
# Debate-mode per-provider results
# ---------------------------------------------------------------------------


@dataclass
class ProviderDebateResult:
    """Aggregated per-provider state across debate phases A and B."""

    provider: str
    r1_result: RoundResult | None = None
    r2_result: RoundResult | None = None
    final_initial: FinalRankingResponse | None = None
    final_attempted: bool = False
    revision: DebateRoundResult | None = None
    error: str | None = None

    @property
    def initial_predictions(self) -> list[ContinuationCandidate]:
        return self.r2_result.predictions if self.r2_result else []

    @property
    def revised_predictions(self) -> list[RevisedContinuationCandidate]:
        if self.revision and self.revision.success:
            return self.revision.revised
        return []


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    run_id: str
    status: RunStatus
    error: str | None
    seen_events: list[StrategyEvent]
    debate_results: list[ProviderDebateResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PreconditionError(RuntimeError):
    """Run cannot start because user-facing preconditions are not met
    (e.g. insufficient configured LLM providers for debate mode).

    Plugin-internal contract: raise BEFORE ``_record_run_start`` so no run
    row is persisted. ``cli.main`` renders these as ``✘ {message}`` without
    a traceback or type prefix — they are user-config errors, not runtime
    crashes.
    """


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LubRunner:
    """Drives the pipeline generator and persists run / events."""

    def __init__(
        self, rt: LubRuntime, *, renderer: EventRenderer | None = None
    ) -> None:
        self._rt = rt
        # Buffer for events emitted by sub-systems (currently TushareClient)
        # and drained between yields in the pipeline.
        self._pending: list[StrategyEvent] = []
        # Selected LLM client for the current run. Bound at execute() entry
        # via rt.llms.get_client(provider_name, ...). Stays None for
        # execute_sync_only() and for debate mode.
        self._llm: LLMClient | None = None
        # Sequence counter used by both single-LLM and debate paths.
        self._seq = 0
        # UI: stays NullRenderer when callers don't inject one (defensive —
        # cli.py always passes a renderer via choose_renderer). The runner is
        # responsible for the EventRenderer lifecycle (on_run_start /
        # on_event / on_run_finish / close); see _dispatch_to_renderer for
        # the contract-isolation wrapper.
        self._renderer: EventRenderer = renderer or NullRenderer()

    # ----- public --------------------------------------------------------

    def execute(self, params: RunParams) -> RunOutcome:
        run_id = str(uuid.uuid4())
        self._rt.run_id = run_id
        self._rt.is_intraday = params.allow_intraday
        self._rt.tushare = build_tushare_client(
            self._rt, intraday=params.allow_intraday, event_cb=self._on_tushare_event
        )

        # v0.5 — construct the LGB scorer once per run. Loading is lazy (first
        # score_batch call), errors degrade to lgb_score=None on every candidate
        # (lightgbm_design.md §7.3). When the user passed --no-lgb (or the
        # config flag is off), we skip construction entirely to keep the run
        # path identical to v0.4.
        self._rt.lgb_scorer = self._maybe_build_scorer(params)

        # v0.6 — renderer lifecycle. on_run_start is called once before any
        # event; the finally block guarantees on_run_finish + close even on
        # KeyboardInterrupt or unhandled exception (Plan §3.1, §3.6).
        self._renderer.on_run_start(
            run_id=run_id, params=params, debate=params.debate
        )
        try:
            if params.debate:
                outcome = self._execute_debate(run_id, params)
            else:
                outcome = self._execute_single(run_id, params)
        finally:
            try:
                # Pass the best outcome we have; if the runner crashed before
                # building one, hand back a synthetic FAILED outcome so the
                # renderer can finalise its UI cleanly. The runner re-raises
                # below — outcome assembly does not swallow user-visible
                # errors.
                outcome_for_render = locals().get("outcome") or RunOutcome(
                    run_id=run_id,
                    status=RunStatus.FAILED,
                    error="runner aborted before outcome",
                    seen_events=[],
                )
                self._renderer.on_run_finish(outcome_for_render)
            finally:
                self._renderer.close()
        return outcome

    def _maybe_build_scorer(self, params: RunParams) -> LgbScorer | None:
        """Build the scorer iff the user hasn't disabled LGB for this run.

        ``RunParams.lgb_enabled`` defaults to True; ``--no-lgb`` flips it.
        ``LubConfig.lgb_enabled`` is the persistent default; either being
        False short-circuits to ``None`` (zero-cost path).
        """
        if not params.lgb_enabled:
            return None
        try:
            cfg = load_config(self._rt.db)
        except Exception as e:  # noqa: BLE001 — config table missing → degrade silently
            logger.warning("load_config failed during scorer construction: %s", e)
            return None
        if not cfg.lgb_enabled:
            return None
        try:
            return LgbScorer(self._rt.db)
        except Exception as e:  # noqa: BLE001 — defensive; constructor is trivial
            logger.warning("LgbScorer construction failed: %s", e)
            return None

    def _execute_single(self, run_id: str, params: RunParams) -> RunOutcome:
        from deeptrade.core import paths

        provider_name = pick_llm_provider(self._rt)
        self._llm = self._rt.llms.get_client(
            provider_name,
            plugin_id=self._rt.plugin_id,
            run_id=run_id,
            reports_dir=paths.reports_dir() / run_id,
        )

        self._record_run_start(run_id, params)

        events: list[StrategyEvent] = []
        seen_validation_failed = False
        terminal_status = RunStatus.SUCCESS
        terminal_error: str | None = None

        try:
            for ev in self._iter_pipeline(params):
                self._seq += 1
                self._persist_event(run_id, self._seq, ev)
                events.append(ev)
                self._dispatch_to_renderer(ev)
                if ev.type == EventType.VALIDATION_FAILED:
                    seen_validation_failed = True
        except KeyboardInterrupt:
            terminal_status = RunStatus.CANCELLED
            terminal_error = "KeyboardInterrupt"
        except Exception as e:  # noqa: BLE001
            terminal_status = RunStatus.FAILED
            terminal_error = f"{type(e).__name__}: {e}"
            logger.exception("limit-up-board run %s raised", run_id)

        if terminal_status == RunStatus.SUCCESS and seen_validation_failed:
            terminal_status = RunStatus.PARTIAL_FAILED

        self._record_run_finish(run_id, terminal_status, terminal_error, events)
        return RunOutcome(
            run_id=run_id, status=terminal_status, error=terminal_error, seen_events=events
        )

    def execute_sync_only(self, params: RunParams) -> RunOutcome:
        """Data-only path: same lifecycle as execute() but yields via _iter_sync."""
        run_id = str(uuid.uuid4())
        self._rt.run_id = run_id
        self._rt.is_intraday = params.allow_intraday
        self._rt.tushare = build_tushare_client(
            self._rt, intraday=params.allow_intraday, event_cb=self._on_tushare_event
        )

        self._renderer.on_run_start(run_id=run_id, params=params, debate=False)
        self._record_run_start(run_id, params)
        events: list[StrategyEvent] = []
        terminal_status = RunStatus.SUCCESS
        terminal_error: str | None = None

        try:
            for ev in self._iter_sync(params):
                self._seq += 1
                self._persist_event(run_id, self._seq, ev)
                events.append(ev)
                self._dispatch_to_renderer(ev)
        except KeyboardInterrupt:
            terminal_status = RunStatus.CANCELLED
            terminal_error = "KeyboardInterrupt"
        except Exception as e:  # noqa: BLE001
            terminal_status = RunStatus.FAILED
            terminal_error = f"{type(e).__name__}: {e}"
            logger.exception("limit-up-board sync %s raised", run_id)

        self._record_run_finish(run_id, terminal_status, terminal_error, events)
        outcome = RunOutcome(
            run_id=run_id, status=terminal_status, error=terminal_error, seen_events=events
        )
        try:
            self._renderer.on_run_finish(outcome)
        finally:
            self._renderer.close()
        return outcome

    # ----- pipeline iteration -------------------------------------------

    def _iter_sync(self, params: RunParams) -> Iterable[StrategyEvent]:
        """Data-only iteration (no LLM stages)."""
        rt = self._rt
        cfg = rt.config.get_app_config()

        yield rt.emit(EventType.STEP_STARTED, "Step 0: resolve trade date")
        cal_df = rt.tushare.call("trade_cal")  # type: ignore[union-attr]
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

        lub_cfg = load_config(rt.db)
        yield _settings_log_event(rt, lub_cfg)
        yield rt.emit(EventType.DATA_SYNC_STARTED, "Step 1: data assembly")
        # sync_data path does NOT use the scorer — keeping data sync free of
        # model inference matches the "data-only" contract.
        bundle = collect_round1(
            tushare=rt.tushare,  # type: ignore[arg-type]
            trade_date=T,
            next_trade_date=T1,
            prev_trade_date=_safe_prev_trade_date(cal, T),
            daily_lookback=params.daily_lookback,
            moneyflow_lookback=params.moneyflow_lookback,
            max_float_mv_yi=lub_cfg.max_float_mv_yi,
            max_close_yuan=lub_cfg.max_close_yuan,
            min_float_mv_yi=lub_cfg.min_float_mv_yi,
            force_sync=params.force_sync,
        )
        yield from self._drain_pending()
        yield rt.emit(
            EventType.DATA_SYNC_FINISHED,
            f"synced {len(bundle.candidates)} candidates",
            payload={"candidates": len(bundle.candidates), "data_unavailable": bundle.data_unavailable},
        )

    def _iter_pipeline(self, params: RunParams) -> Iterable[StrategyEvent]:
        """Full pipeline: Step 0..5."""
        rt = self._rt
        cfg = rt.config.get_app_config()

        # Step 0
        yield rt.emit(EventType.STEP_STARTED, "Step 0: resolve trade date")
        cal_df = rt.tushare.call("trade_cal")  # type: ignore[union-attr]
        cal = TradeCalendar(cal_df)
        now = datetime.now()
        T, T1 = resolve_trade_date(
            now,
            cal,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
            close_after=cfg.app_close_after if cfg is not None else time(18, 0),
        )
        today_str = now.strftime("%Y%m%d")
        auto_resolved_to_today_after_close = (
            params.trade_date is None and not params.allow_intraday and T == today_str
        )
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 0: T={T} T+1={T1}",
            payload={"trade_date": T, "next_trade_date": T1},
        )

        # Step 1
        lub_cfg = load_config(rt.db)
        yield _settings_log_event(rt, lub_cfg)
        yield rt.emit(EventType.STEP_STARTED, "Step 1: data assembly")
        try:
            bundle = collect_round1(
                tushare=rt.tushare,  # type: ignore[arg-type]
                trade_date=T,
                next_trade_date=T1,
                prev_trade_date=_safe_prev_trade_date(cal, T),
                daily_lookback=params.daily_lookback,
                moneyflow_lookback=params.moneyflow_lookback,
                max_float_mv_yi=lub_cfg.max_float_mv_yi,
                max_close_yuan=lub_cfg.max_close_yuan,
                min_float_mv_yi=lub_cfg.min_float_mv_yi,
                force_sync=params.force_sync,
                lgb_scorer=rt.lgb_scorer,
            )
        except TushareUnauthorizedError as e:
            yield rt.emit(
                EventType.LOG, f"required tushare api unauthorized: {e}", level=EventLevel.ERROR
            )
            raise
        yield from self._drain_pending()
        self._persist_lgb_predictions(bundle)
        yield rt.emit(
            EventType.STEP_FINISHED,
            f"Step 1: {len(bundle.candidates)} candidates",
            payload={
                "candidates": len(bundle.candidates),
                "data_unavailable": bundle.data_unavailable,
                "sector_strength_source": bundle.sector_strength.source,
                "lgb_model_id": bundle.lgb_model_id,
                "lgb_scored": sum(1 for c in bundle.candidates if c.get("lgb_score") is not None),
            },
        )

        if not bundle.candidates and auto_resolved_to_today_after_close:
            raise RuntimeError(
                f"limit_list_d({T}) returned 0 rows after close_after — tushare "
                "data may not be published yet. Try again later, or use "
                "`--trade-date <YYYYMMDD>` to specify a known historical day."
            )

        if not bundle.candidates:
            yield from self._emit_empty_report(bundle, params)
            return

        # Step 2 — R1
        preset = cfg.app_profile  # v0.7: per-stage tuning resolved by plugin
        # v0.5 LGB: thread the configured min_score_floor into the prompts;
        # when LGB is fully disabled we pass None so the prompt drops the
        # numeric threshold sentence (the rest of the LGB guidance survives).
        lgb_floor = lub_cfg.lgb_min_score_floor if rt.lgb_scorer is not None else None
        r1_result = None
        for ev, res in run_r1(
            llm=self._llm,
            bundle=bundle,
            preset=preset,
            lgb_min_score_floor=lgb_floor,
        ):
            yield ev
            if res is not None:
                r1_result = res
        selected = r1_result.selected if r1_result else []
        if not selected:
            yield from self._emit_empty_report(bundle, params, reason="no R1 selected")
            return

        # Step 4 — R2
        r2_result = None
        for ev, res in run_r2(
            llm=self._llm,
            selected=selected,
            bundle=bundle,
            preset=preset,
            lgb_min_score_floor=lgb_floor,
        ):
            yield ev
            if res is not None:
                r2_result = res
        predictions = r2_result.predictions if r2_result else []

        # Step 4.5 — final_ranking when R2 was multi-batch
        final_obj: FinalRankingResponse | None = None
        final_ranking_attempted = False
        if r2_result and r2_result.success_batches > 1 and predictions:
            final_ranking_attempted = True
            finalists = select_finalists(predictions, batch_size_hint=r2_result.batch_size or 20)
            for ev, fr_obj in run_final_ranking(
                llm=self._llm,
                bundle=bundle,
                finalists=finalists,
                preset=preset,
            ):
                yield ev
                if fr_obj is not None:
                    final_obj = fr_obj

        # Step 5 — finalize
        terminal_status = RunStatus.SUCCESS
        if r1_result and r1_result.failed_batches > 0:
            terminal_status = RunStatus.PARTIAL_FAILED
        if r2_result and r2_result.failed_batches > 0:
            terminal_status = RunStatus.PARTIAL_FAILED
        if final_ranking_attempted and final_obj is None:
            terminal_status = RunStatus.PARTIAL_FAILED

        _write_stage_results(rt, "r1", selected)
        _write_stage_results(rt, "r2", predictions)
        if final_obj is not None:
            _write_stage_results(rt, "final_ranking", final_obj.finalists)

        failed_batches: list[str] = []
        if r1_result and r1_result.failed_batch_ids:
            failed_batches.extend(f"R1#{b}" for b in r1_result.failed_batch_ids)
        if r2_result and r2_result.failed_batch_ids:
            failed_batches.extend(f"R2#{b}" for b in r2_result.failed_batch_ids)
        if final_ranking_attempted and final_obj is None:
            failed_batches.append("final_ranking")

        report_path = write_report(
            rt.run_id,
            status=terminal_status,
            is_intraday=params.allow_intraday,
            bundle=bundle,
            selected=selected,
            predictions=predictions,
            final_ranking=final_obj,
            failed_batch_ids=failed_batches or None,
        )
        export_llm_calls(rt.run_id, rt.db)
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"Report written: {report_path}",
            payload={
                "report_dir": str(report_path),
                "selected": len(selected),
                "predictions": len(predictions),
                "final_ranking_used": final_obj is not None,
            },
        )

    # ====================================================================
    # Debate mode (multi-LLM)
    # ====================================================================

    def _execute_debate(self, run_id: str, params: RunParams) -> RunOutcome:
        """Multi-LLM debate flow.

        Step 0/1 stay on the main thread; R1/R2/(final_ranking) fan out across
        providers in phase A; R3 fans out across the same providers in phase B
        with peer outputs cross-fed and anonymised.
        """
        from deeptrade.core import paths

        rt = self._rt

        # Precondition check: must run BEFORE _record_run_start so that a
        # config error never persists a "failed" run row. PreconditionError
        # propagates up to cli.main and renders as ``✘ {message}``.
        providers = self._select_debate_providers(params)

        self._record_run_start(run_id, params)

        events: list[StrategyEvent] = []
        terminal_status = RunStatus.SUCCESS
        terminal_error: str | None = None
        provider_results: list[ProviderDebateResult] = []

        # Helper: emit + persist + render in one shot, append to events.
        def emit(ev: StrategyEvent) -> None:
            self._seq += 1
            self._persist_event(run_id, self._seq, ev)
            events.append(ev)
            self._dispatch_to_renderer(ev)

        seen_validation_failed = False
        try:
            emit(
                rt.emit(
                    EventType.LOG,
                    f"[辩论模式] 启用，参与 LLM = {providers}",
                    level=EventLevel.INFO,
                    providers=providers,
                )
            )

            # Step 0/1 (main thread, single)
            bundle = self._do_step_0_and_1(params, emit)
            if bundle is None:
                # _do_step_0_and_1 already emitted the empty report
                self._record_run_finish(run_id, RunStatus.SUCCESS, None, events)
                return RunOutcome(
                    run_id=run_id, status=RunStatus.SUCCESS, error=None, seen_events=events
                )

            cfg = rt.config.get_app_config()
            preset = cfg.app_profile
            reports_dir = paths.reports_dir() / run_id

            # v0.5 — resolve the LGB floor once per run; workers all share it.
            lub_cfg = load_config(rt.db)
            lgb_floor = lub_cfg.lgb_min_score_floor if rt.lgb_scorer is not None else None

            # ----- Phase A: parallel R1 + R2 + (final_ranking) ---------------
            emit(
                rt.emit(
                    EventType.LIVE_STATUS,
                    f"[辩论模式] Phase A — 并行执行 R1+R2 ({len(providers)} 个 LLM)",
                )
            )
            with ThreadPoolExecutor(max_workers=len(providers)) as pool:
                futures = {
                    pool.submit(
                        _worker_phase_a,
                        provider,
                        bundle,
                        preset,
                        rt.plugin_id,
                        run_id,
                        reports_dir,
                        params.allow_intraday,
                        rt.config,
                        lgb_floor,
                    ): provider
                    for provider in providers
                }
                for fut in as_completed(futures):
                    provider = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:  # noqa: BLE001
                        result = ProviderDebateResult(
                            provider=provider, error=f"{type(e).__name__}: {e}"
                        )
                        logger.exception("debate phase A worker %s failed", provider)
                    provider_results.append(result)
                    for ev in result_events(result, "phase_a"):
                        emit(ev)
                        if ev.type == EventType.VALIDATION_FAILED:
                            seen_validation_failed = True

            # Persist phase-A stage results
            for r in provider_results:
                if r.r1_result and r.r1_result.selected:
                    _write_stage_results(
                        rt, f"r1:{r.provider}", r.r1_result.selected,
                        llm_provider=r.provider,
                    )
                if r.r2_result and r.r2_result.predictions:
                    _write_stage_results(
                        rt, f"r2_initial:{r.provider}", r.r2_result.predictions,
                        llm_provider=r.provider,
                    )
                if r.final_initial is not None:
                    _write_stage_results(
                        rt, f"r2_final_initial:{r.provider}", r.final_initial.finalists,
                        llm_provider=r.provider,
                    )

            # Filter survivors (must have non-empty initial predictions)
            survivors = [r for r in provider_results if r.initial_predictions]
            if len(survivors) < 2:
                emit(
                    rt.emit(
                        EventType.LOG,
                        f"[辩论模式] 有效产出 LLM 数 = {len(survivors)} < 2，"
                        "跳过 R3 修订阶段，按现有结果出报告",
                        level=EventLevel.WARN,
                    )
                )
                terminal_status = RunStatus.PARTIAL_FAILED
            else:
                # ----- Phase B: parallel R3 debate revisions -----------------
                emit(
                    rt.emit(
                        EventType.LIVE_STATUS,
                        f"[辩论模式] Phase B — 并行执行 R3 修订 ({len(survivors)} 个 LLM)",
                    )
                )
                surviving_providers = [r.provider for r in survivors]
                survivor_map = {r.provider: r for r in survivors}
                with ThreadPoolExecutor(max_workers=len(survivors)) as pool_b:
                    futures_b = {
                        pool_b.submit(
                            _worker_phase_b,
                            r.provider,
                            bundle,
                            preset,
                            rt.plugin_id,
                            run_id,
                            reports_dir,
                            params.allow_intraday,
                            r.initial_predictions,
                            [
                                (
                                    assign_peer_labels(r.provider, surviving_providers)[
                                        peer.provider
                                    ],
                                    peer.initial_predictions,
                                )
                                for peer in survivors
                                if peer.provider != r.provider
                            ],
                            rt.config,
                        ): r.provider
                        for r in survivors
                    }
                    for fut_b in as_completed(futures_b):
                        provider = futures_b[fut_b]
                        evs_b: list[StrategyEvent]
                        debate_result: DebateRoundResult
                        try:
                            evs_b, debate_result = fut_b.result()
                        except Exception as e:  # noqa: BLE001
                            evs_b = []
                            debate_result = DebateRoundResult(
                                error=f"{type(e).__name__}: {e}"
                            )
                            logger.exception("debate phase B worker %s failed", provider)
                        survivor_map[provider].revision = debate_result
                        for ev in evs_b:
                            tagged = _tag_event(ev, provider, "phase_b")
                            emit(tagged)
                            if tagged.type == EventType.VALIDATION_FAILED:
                                seen_validation_failed = True

                # Persist phase-B stage results
                for r in survivors:
                    if r.revision and r.revision.success and r.revision.revised:
                        _write_stage_results(
                            rt,
                            f"r2_revised:{r.provider}",
                            r.revision.revised,
                            llm_provider=r.provider,
                        )
                    elif r.revision and not r.revision.success:
                        # Mark partial fail; revised view falls back to initial
                        terminal_status = RunStatus.PARTIAL_FAILED

            # Aggregate failed batch ids across providers for the banner
            failed_batches: list[str] = []
            for r in provider_results:
                tag = r.provider
                if r.error:
                    failed_batches.append(f"{tag}:phase_a")
                if r.r1_result and r.r1_result.failed_batch_ids:
                    failed_batches.extend(f"{tag}:R1#{b}" for b in r.r1_result.failed_batch_ids)
                if r.r2_result and r.r2_result.failed_batch_ids:
                    failed_batches.extend(f"{tag}:R2#{b}" for b in r.r2_result.failed_batch_ids)
                if r.final_attempted and r.final_initial is None:
                    failed_batches.append(f"{tag}:final_ranking")
                if r.revision and not r.revision.success:
                    failed_batches.append(f"{tag}:R3")

            if failed_batches:
                terminal_status = RunStatus.PARTIAL_FAILED

            # Write report (debate-aware)
            report_path = write_report(
                run_id,
                status=terminal_status,
                is_intraday=params.allow_intraday,
                bundle=bundle,
                selected=[],  # main report tables are replaced by debate sections
                predictions=[],
                final_ranking=None,
                failed_batch_ids=failed_batches or None,
                debate_results=provider_results,
            )
            export_llm_calls(run_id, rt.db)
            emit(
                rt.emit(
                    EventType.RESULT_PERSISTED,
                    f"Report written: {report_path}",
                    payload={
                        "report_dir": str(report_path),
                        "providers": [r.provider for r in provider_results],
                        "survivors": [r.provider for r in provider_results if r.initial_predictions],
                    },
                )
            )

        except KeyboardInterrupt:
            terminal_status = RunStatus.CANCELLED
            terminal_error = "KeyboardInterrupt"
        except Exception as e:  # noqa: BLE001
            terminal_status = RunStatus.FAILED
            terminal_error = f"{type(e).__name__}: {e}"
            logger.exception("limit-up-board debate run %s raised", run_id)

        if terminal_status == RunStatus.SUCCESS and seen_validation_failed:
            terminal_status = RunStatus.PARTIAL_FAILED

        self._record_run_finish(run_id, terminal_status, terminal_error, events)
        return RunOutcome(
            run_id=run_id,
            status=terminal_status,
            error=terminal_error,
            seen_events=events,
            debate_results=provider_results,
        )

    def _select_debate_providers(self, params: RunParams) -> list[str]:
        available = self._rt.llms.list_providers()
        if params.debate_llms:
            requested = list(dict.fromkeys(params.debate_llms))  # dedup, preserve order
            missing = [p for p in requested if p not in available]
            if missing:
                raise PreconditionError(
                    f"--debate-llms 包含未配置或缺 api_key 的 provider: {missing}; "
                    f"当前可用: {available}"
                )
            providers = requested
        else:
            providers = list(available)
        if len(providers) < 2:
            raise PreconditionError(
                f"辩论模式需要至少 2 个已配置 LLM；当前可用 {len(providers)} 个"
                + (f": {providers}" if providers else "")
                + "。请运行 `deeptrade config set-llm` 配置至少 2 个 provider。"
            )
        return providers

    def _do_step_0_and_1(
        self, params: RunParams, emit: Any
    ) -> Round1Bundle | None:
        rt = self._rt
        cfg = rt.config.get_app_config()

        emit(rt.emit(EventType.STEP_STARTED, "Step 0: resolve trade date"))
        cal_df = rt.tushare.call("trade_cal")  # type: ignore[union-attr]
        cal = TradeCalendar(cal_df)
        now = datetime.now()
        T, T1 = resolve_trade_date(
            now,
            cal,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
            close_after=cfg.app_close_after if cfg is not None else time(18, 0),
        )
        today_str = now.strftime("%Y%m%d")
        auto_resolved_to_today_after_close = (
            params.trade_date is None and not params.allow_intraday and T == today_str
        )
        emit(
            rt.emit(
                EventType.STEP_FINISHED,
                f"Step 0: T={T} T+1={T1}",
                payload={"trade_date": T, "next_trade_date": T1},
            )
        )

        lub_cfg = load_config(rt.db)
        emit(_settings_log_event(rt, lub_cfg))
        emit(rt.emit(EventType.STEP_STARTED, "Step 1: data assembly"))
        try:
            bundle = collect_round1(
                tushare=rt.tushare,  # type: ignore[arg-type]
                trade_date=T,
                next_trade_date=T1,
                prev_trade_date=_safe_prev_trade_date(cal, T),
                daily_lookback=params.daily_lookback,
                moneyflow_lookback=params.moneyflow_lookback,
                max_float_mv_yi=lub_cfg.max_float_mv_yi,
                max_close_yuan=lub_cfg.max_close_yuan,
                min_float_mv_yi=lub_cfg.min_float_mv_yi,
                force_sync=params.force_sync,
                lgb_scorer=rt.lgb_scorer,
            )
        except TushareUnauthorizedError as e:
            emit(
                rt.emit(
                    EventType.LOG,
                    f"required tushare api unauthorized: {e}",
                    level=EventLevel.ERROR,
                )
            )
            raise
        for ev in self._drain_pending():
            emit(ev)
        self._persist_lgb_predictions(bundle)
        emit(
            rt.emit(
                EventType.STEP_FINISHED,
                f"Step 1: {len(bundle.candidates)} candidates",
                payload={
                    "candidates": len(bundle.candidates),
                    "data_unavailable": bundle.data_unavailable,
                    "sector_strength_source": bundle.sector_strength.source,
                    "lgb_model_id": bundle.lgb_model_id,
                    "lgb_scored": sum(1 for c in bundle.candidates if c.get("lgb_score") is not None),
                },
            )
        )

        if not bundle.candidates and auto_resolved_to_today_after_close:
            raise RuntimeError(
                f"limit_list_d({T}) returned 0 rows after close_after — tushare "
                "data may not be published yet. Try again later, or use "
                "`--trade-date <YYYYMMDD>` to specify a known historical day."
            )
        if not bundle.candidates:
            for ev in self._emit_empty_report(bundle, params):
                emit(ev)
            return None
        return bundle

    # ----- helpers ------------------------------------------------------

    def _persist_lgb_predictions(self, bundle: Round1Bundle) -> None:
        """Insert this run's LGB scores into ``lub_lgb_predictions``.

        ``bundle.lgb_predictions`` is the per-row payload list ``data._attach_lgb_scores``
        prepared; ``bundle.lgb_model_id`` is the active model id. No model →
        no rows; the audit helper itself swallows DB errors so a broken
        audit insert never blocks the LLM stages.
        """
        if not bundle.lgb_predictions or not bundle.lgb_model_id or not self._rt.run_id:
            return
        try:
            _record_lgb_predictions(
                self._rt.db,
                run_id=self._rt.run_id,
                trade_date=bundle.trade_date,
                model_id=bundle.lgb_model_id,
                rows=bundle.lgb_predictions,
            )
        except Exception as e:  # noqa: BLE001 — audit must not block run
            logger.warning("persist_lgb_predictions raised: %s", e)

    def _emit_empty_report(
        self, bundle: Round1Bundle, params: RunParams, *, reason: str = "zero candidates"
    ) -> Iterable[StrategyEvent]:
        rt = self._rt
        report_path = write_report(
            rt.run_id,
            status=RunStatus.SUCCESS,
            is_intraday=params.allow_intraday,
            bundle=bundle,
            selected=[],
            predictions=[],
            final_ranking=None,
        )
        export_llm_calls(rt.run_id, rt.db)
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"empty report ({reason})",
            payload={"report_dir": str(report_path), "reason": reason},
        )

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

    def _dispatch_to_renderer(self, ev: StrategyEvent) -> None:
        """Hand ``ev`` to the active renderer, with contract isolation.

        Plan §3.6.1 — *UI failure ≠ run failure*: the renderer is contractually
        forbidden from raising out of ``on_event``, but we install a safety
        net here anyway. If a renderer does raise, we log a WARN, close it,
        and **swap to** :class:`LegacyStreamRenderer` for the rest of the
        run. Already-emitted events are not replayed; legacy resumes from
        the current event onward (matches the design's "don't backfill"
        rule — backfilling would risk further crashes).
        """
        try:
            self._renderer.on_event(ev)
        except Exception as e:  # noqa: BLE001 — renderer must never crash a run
            logger.warning(
                "renderer.on_event raised; degrading to legacy: %s", e
            )
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001 — close() is best-effort
                pass
            self._renderer = LegacyStreamRenderer()
            try:
                self._renderer.on_event(ev)
            except Exception:  # noqa: BLE001 — legacy print failed → give up silently
                logger.warning(
                    "legacy renderer also raised after fallback; suppressing"
                )

    # ----- DB helpers ---------------------------------------------------

    def _record_run_start(self, run_id: str, params: RunParams) -> None:
        self._rt.db.execute(
            "INSERT INTO lub_runs(run_id, trade_date, status, is_intraday, started_at, "
            "params_json) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
            (
                run_id,
                params.trade_date or "",
                RunStatus.RUNNING.value,
                params.allow_intraday,
                json.dumps(params.__dict__, ensure_ascii=False),
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
            "UPDATE lub_runs SET status=?, finished_at=CURRENT_TIMESTAMP, "
            "summary_json=?, error=? WHERE run_id=?",
            (status.value, json.dumps(summary, ensure_ascii=False), error, run_id),
        )

    def _persist_event(self, run_id: str, seq: int, ev: StrategyEvent) -> None:
        self._rt.db.execute(
            "INSERT INTO lub_events(run_id, seq, level, event_type, message, payload_json) "
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


# ---------------------------------------------------------------------------
# Debate worker functions (run in ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def _worker_phase_a(
    provider: str,
    bundle: Round1Bundle,
    preset: str,
    plugin_id: str,
    run_id: str,
    reports_dir: Path,
    is_intraday: bool,
    config: ConfigService,
    lgb_min_score_floor: float | None = 30.0,
) -> ProviderDebateResult:
    """One provider's R1 + R2 + (optional) final_ranking. Tagged events are
    attached to the returned ProviderDebateResult; the main thread will emit
    them in completion order."""
    db, wrt = open_worker_runtime(
        plugin_id, run_id, config=config, is_intraday=is_intraday
    )
    out = ProviderDebateResult(provider=provider)
    try:
        llm = wrt.llms.get_client(
            provider, plugin_id=plugin_id, run_id=run_id, reports_dir=reports_dir
        )

        events: list[StrategyEvent] = []

        for ev, res in run_r1(
            llm=llm, bundle=bundle, preset=preset,
            lgb_min_score_floor=lgb_min_score_floor,
        ):
            events.append(ev)
            if res is not None:
                out.r1_result = res
        selected = out.r1_result.selected if out.r1_result else []

        if selected:
            for ev, res in run_r2(
                llm=llm, selected=selected, bundle=bundle, preset=preset,
                lgb_min_score_floor=lgb_min_score_floor,
            ):
                events.append(ev)
                if res is not None:
                    out.r2_result = res

        if out.r2_result and out.r2_result.success_batches > 1 and out.r2_result.predictions:
            out.final_attempted = True
            finalists = select_finalists(
                out.r2_result.predictions, batch_size_hint=out.r2_result.batch_size or 20
            )
            for ev, fr_obj in run_final_ranking(
                llm=llm, bundle=bundle, finalists=finalists, preset=preset
            ):
                events.append(ev)
                if fr_obj is not None:
                    out.final_initial = fr_obj

        # Attach events to the result via a sidecar attribute. Cleaner than
        # widening the dataclass since these are only used during emit.
        out._events = events  # type: ignore[attr-defined]
    finally:
        db.close()
    return out


def _worker_phase_b(
    provider: str,
    bundle: Round1Bundle,
    preset: str,
    plugin_id: str,
    run_id: str,
    reports_dir: Path,
    is_intraday: bool,
    own_predictions: list[ContinuationCandidate],
    peers: list[tuple[str, list[ContinuationCandidate]]],
    config: ConfigService,
) -> tuple[list[StrategyEvent], DebateRoundResult]:
    """One provider's R3 debate revision."""
    db, wrt = open_worker_runtime(
        plugin_id, run_id, config=config, is_intraday=is_intraday
    )
    try:
        llm = wrt.llms.get_client(
            provider, plugin_id=plugin_id, run_id=run_id, reports_dir=reports_dir
        )
        events: list[StrategyEvent] = []
        result: DebateRoundResult | None = None
        for ev, res in run_r3_debate(
            llm=llm,
            bundle=bundle,
            own_predictions=own_predictions,
            peers=peers,
            preset=preset,
        ):
            events.append(ev)
            if res is not None:
                result = res
        if result is None:
            result = DebateRoundResult(error="run_r3_debate yielded no terminal result")
        return events, result
    finally:
        db.close()


def _tag_event(ev: StrategyEvent, provider: str, phase: str) -> StrategyEvent:
    """Return a copy of ``ev`` with ``[provider]`` prefixed in the message and
    the provider name added to payload (so persisted JSON is queryable)."""
    payload = dict(ev.payload)
    payload["llm_provider"] = provider
    payload["debate_phase"] = phase
    return StrategyEvent(
        type=ev.type,
        level=ev.level,
        message=f"[{provider}] {ev.message}",
        payload=payload,
    )


def result_events(result: ProviderDebateResult, phase: str) -> Iterable[StrategyEvent]:
    """Drain the events buffered on a phase-A worker result, tagged with the
    provider name."""
    raw = getattr(result, "_events", []) or []
    for ev in raw:
        yield _tag_event(ev, result.provider, phase)
    if result.error:
        yield StrategyEvent(
            type=EventType.LOG,
            level=EventLevel.ERROR,
            message=f"[{result.provider}] worker failed: {result.error}",
            payload={"llm_provider": result.provider, "debate_phase": phase},
        )


# ---------------------------------------------------------------------------
# Stage results
# ---------------------------------------------------------------------------


def _write_stage_results(
    rt: LubRuntime,
    stage: str,
    items: list[Any],
    *,
    llm_provider: str | None = None,
) -> None:
    """Persist R1/R2/final_ranking/R3 outputs to lub_stage_results.

    In debate mode, ``stage`` is suffixed with the provider (e.g.
    ``r1:deepseek``) to keep the (run_id, stage, ts_code) PK unique across
    providers; the explicit ``llm_provider`` column lets queries filter by
    provider without parsing the stage string.
    """
    if not items:
        return
    for i, item in enumerate(items):
        d = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        rt.db.execute(
            "INSERT INTO lub_stage_results(run_id, stage, batch_no, trade_date, ts_code, "
            "name, score, rank, decision, rationale, evidence_json, risk_flags_json, "
            "raw_response_json, llm_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rt.run_id,
                stage,
                d.get("batch_no", 0),
                d.get("trade_date", ""),
                d.get("ts_code", ""),
                d.get("name"),
                d.get("score") or d.get("continuation_score"),
                d.get("rank") or d.get("final_rank") or i + 1,
                d.get("decision") or d.get("prediction") or d.get("final_prediction"),
                d.get("rationale") or d.get("reason_vs_peers"),
                json.dumps(d.get("evidence") or d.get("key_evidence") or [], ensure_ascii=False),
                json.dumps(d.get("risk_flags") or [], ensure_ascii=False),
                json.dumps(d, ensure_ascii=False),
                llm_provider,
            ),
        )


def render_finished_run(run_id: str) -> None:
    """Re-render a finished run's terminal summary (used by `report` subcommand)."""
    render_terminal_summary(run_id)
