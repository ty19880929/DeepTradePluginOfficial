"""Plugin-internal run lifecycle: drives the pipeline generator, persists
events to ``lub_events``, and writes the run record to ``lub_runs``.

Replaces the deleted framework-side ``core/strategy_runner.py``: each plugin
manages its own run history on Plan A's pure-isolation model.

v0.8 — debate mode (multi-LLM): when ``RunParams.debate`` is set, 强势初筛 +
连板预测 + optional 全局重排 + 辩论修订 all fan out across configured LLM
providers with one worker thread per provider. Each worker uses an isolated
``LubRuntime`` (private DB connection + LLMManager) so concurrent
``LLMClient.complete_json`` calls don't share lock/audit-write bookkeeping.
"""

from __future__ import annotations

import json
import logging
import signal
import traceback
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deeptrade.core.run_status import RunStatus
from deeptrade.core.tushare_client import TushareUnauthorizedError
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.config import ConfigService
    from deeptrade.core.llm_client import LLMClient
    from deeptrade.plugins_api import PluginContext, ReportUploader

from .calendar import TradeCalendar
from .cancellation import cancel_requested
from .config import LubConfig, load_config
from .observability import RunMetrics
from .data import (
    Round1Bundle,
    collect_round1,
    fetch_latest_trade_date,
    resolve_trade_date,
)
from .lgb.audit import record_predictions as _record_lgb_predictions
from .lgb.scorer import LgbScorer
from .pipeline import (
    DebateRoundResult,
    RoundResult,
    run_debate_revision,
    run_final_ranking,
    run_prediction,
    run_screening,
)
from .prompts import assign_peer_labels
from .render import export_llm_calls, render_terminal_summary, write_report
from .schemas import apply_empty_array_policy
from .runtime import (
    LubRuntime,
    ProviderConfigSnapshot,
    build_provider_config_snapshot,
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


def _run_providers_ordered(
    providers: list[str],
    submit_worker,
    on_complete=None,
) -> list[Any]:
    """P1-H: run one worker per provider in parallel; return results in canonical
    ``providers`` input order regardless of completion order.

    ``submit_worker(provider)`` is called once per provider inside a fresh
    ``ThreadPoolExecutor`` and must return a value that downstream code
    consumes (worker exceptions are NOT caught here — callers wrap them into
    their own per-provider error result types).

    ``on_complete(provider, result)`` is invoked from the main thread as each
    worker finishes (completion order, not canonical order). Use it to emit
    real-time UI events; the returned list is what gets persisted / iterated
    downstream and is ALWAYS in canonical order.

    Why: previous ``as_completed`` accumulation made downstream persistence,
    peer inputs, and report sections depend on network timing — two runs with
    identical inputs could swap provider order. Holding events in completion
    order keeps the dashboard responsive while pinning artifacts to a
    deterministic provider sequence.
    """
    if not providers:
        return []
    result_by_provider: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {pool.submit(submit_worker, p): p for p in providers}
        for fut in as_completed(futures):
            provider = futures[fut]
            # Caller's submit_worker is responsible for wrapping per-worker
            # exceptions into a result type; if it raises here it's a bug
            # we want to surface immediately rather than swallow.
            result = fut.result()
            result_by_provider[provider] = result
            if on_complete is not None:
                on_complete(provider, result)
    return [result_by_provider[p] for p in providers]


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
    force_sync: bool = False
    daily_lookback: int = 30
    moneyflow_lookback: int = 5
    debate: bool = False
    debate_llms: list[str] | None = None
    # v0.6.8 — 非辩论模式下用户用 ``--llm <name>`` 钉死本次 run 的 provider；
    # None 表示走框架默认（``LLMManager.get_client(name=None)`` 走 is_default 行）。
    # 与 ``debate`` 互斥，CLI 层提前校验；落到 ``lub_runs.params_json`` 用于复盘。
    llm_provider: str | None = None
    # v0.5 LGB 开关：用户传 --no-lgb 时设为 False（一次性覆盖 LubConfig.lgb_enabled）。
    # PR-0.3 仅落字段，pipeline 接入在 PR-2.2。
    lgb_enabled: bool = True
    # v0.15.0 (P3-B) — LLM 响应重放 CLI 三件套（互斥；CLI 层已校验）。
    # 落 ``lub_runs.params_json`` 便于复盘。运行时由 LubRunner 与 LubConfig
    # 合并成 LLMReplayPolicy；框架未合并 Phase 2 时这些 flag 退化为 no-op
    # （--replay-only 会显式报错，因为用户主动要求 replay）。
    fresh_llm: bool = False
    no_llm_replay: bool = False
    replay_only: bool = False


# ---------------------------------------------------------------------------
# Debate-mode per-provider results
# ---------------------------------------------------------------------------


@dataclass
class ProviderDebateResult:
    """Aggregated per-provider state across debate phases A and B."""

    provider: str
    screening_result: RoundResult | None = None
    prediction_result: RoundResult | None = None
    final_initial: FinalRankingResponse | None = None
    final_attempted: bool = False
    revision: DebateRoundResult | None = None
    error: str | None = None

    @property
    def initial_predictions(self) -> list[ContinuationCandidate]:
        return self.prediction_result.predictions if self.prediction_result else []

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
# Per-run plugin-local log file
# ---------------------------------------------------------------------------


def _plugin_logs_dir() -> Path:
    """Return ``<framework data root>/limit_up_board/logs/`` (mkdir on use).

    Sits alongside the LGB models/datasets/checkpoints dirs so all per-plugin
    state lives under one tree. We don't reuse the framework's
    ``~/.deeptrade/logs/deeptrade.log`` because that's a shared rotating sink;
    the per-run file here is keyed by ``run_id`` so support copy-paste is easy.
    """
    from deeptrade.core import paths as _fw_paths  # noqa: PLC0415

    p = _fw_paths.home_dir() / "limit_up_board" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _attach_run_logfile(run_id: str) -> tuple[logging.Handler, Path] | tuple[None, None]:
    """Attach a per-run FileHandler to the root logger.

    Returns ``(handler, log_path)`` on success; ``(None, None)`` if the file
    can't be opened (read-only home, encoding issues, …). The caller MUST
    detach in a ``finally`` via :func:`_detach_run_logfile`; otherwise every
    run leaks a file descriptor + an in-memory handler that catches unrelated
    log records from subsequent runs.
    """
    try:
        log_path = _plugin_logs_dir() / f"run-{run_id}.log"
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        # Tag so _detach_run_logfile can identify ours unambiguously even
        # if other code touches the root logger handlers list.
        handler._lub_run_id = run_id  # type: ignore[attr-defined]
        root = logging.getLogger()
        root.addHandler(handler)
        # If the root logger's level is higher than DEBUG (default WARNING),
        # bump it down so DEBUG / INFO records from this plugin reach the
        # file. We restore on detach.
        root._lub_prev_level = root.level  # type: ignore[attr-defined]
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        return handler, log_path
    except Exception:  # noqa: BLE001 — never block a run because logging failed
        logger.warning("failed to attach per-run log file for %s", run_id, exc_info=True)
        return None, None


def _detach_run_logfile(handler: logging.Handler | None) -> None:
    if handler is None:
        return
    root = logging.getLogger()
    try:
        root.removeHandler(handler)
    except ValueError:
        pass
    prev = getattr(root, "_lub_prev_level", None)
    if isinstance(prev, int):
        root.setLevel(prev)
        try:
            delattr(root, "_lub_prev_level")
        except AttributeError:
            pass
    try:
        handler.close()
    except Exception:  # noqa: BLE001 — best-effort
        pass


def _format_traceback(e: BaseException, *, limit: int = 12) -> list[str]:
    """Render an exception's traceback as a list of stripped lines.

    Used by the runner to feed each line into the renderer as a separate LOG
    ERROR event, so the dashboard's log panel can show the stack frame-by-frame
    instead of one wrapped blob. ``limit`` caps the number of stack frames to
    keep the dashboard from being flooded by a deep recursion.
    """
    tb_text = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    raw_lines = tb_text.splitlines()
    if len(raw_lines) > limit * 2:
        head = raw_lines[: limit]
        tail = raw_lines[-limit:]
        return head + [f"... ({len(raw_lines) - 2 * limit} more lines elided) ..."] + tail
    return raw_lines


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LubRunner:
    """Drives the pipeline generator and persists run / events."""

    def __init__(
        self,
        rt: LubRuntime,
        *,
        renderer: EventRenderer | None = None,
        ctx: PluginContext | None = None,
    ) -> None:
        self._rt = rt
        # v0.13.3 — framework v0.11+ PluginContext (carries make_report_uploader).
        # ``None`` keeps legacy-test paths working; ``_maybe_upload_summary`` is
        # the only consumer and bails out cleanly when missing (skipped_disabled).
        self._ctx = ctx
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
        # Path of the per-run plugin-local log file (set in execute() /
        # execute_sync_only()). ``None`` outside of a run, or when attaching
        # the FileHandler failed (read-only home etc.); the failure mode is
        # observable in the framework's ~/.deeptrade/logs/deeptrade.log only.
        self._log_file_path: Path | None = None
        # v0.13.0 (P2-3)：耗时 / 调用 / 校验失败计数器，被动观察事件流；
        # 收尾时聚合成 OBSERVABILITY_SUMMARY 事件 + summary.json.quality_metrics。
        self._metrics: RunMetrics = RunMetrics()

    # ----- public --------------------------------------------------------

    def execute(self, params: RunParams) -> RunOutcome:
        # v0.15.0 (P3-B) — --replay-only is a hard precondition: it only makes
        # sense once the framework's LLM replay cache has landed. Reject early
        # (before _record_run_start so no failed run row leaks). The other two
        # flags (--fresh-llm / --no-llm-replay) silently no-op on pre-Phase-2
        # framework — they at worst force a fresh LLM call which is the
        # existing behaviour.
        from .replay_policy import complete_json_supports_replay  # noqa: PLC0415

        if params.replay_only and not complete_json_supports_replay():
            raise PreconditionError(
                "--replay-only 需要框架 Phase 2（LLM replay cache）已合并；"
                "当前 deeptrade-quant 的 LLMClient.complete_json 未支持 replay 形参。"
                "请升级框架，或移除 --replay-only。"
            )

        run_id = str(uuid.uuid4())
        self._rt.run_id = run_id
        self._rt.tushare = build_tushare_client(
            self._rt, event_cb=self._on_tushare_event
        )

        # v0.5 — construct the LGB scorer once per run. Loading is lazy (first
        # score_batch call), errors degrade to lgb_score=None on every candidate
        # (lightgbm_design.md §7.3). When the user passed --no-lgb (or the
        # config flag is off), we skip construction entirely to keep the run
        # path identical to v0.4.
        self._rt.lgb_scorer = self._maybe_build_scorer(params)

        # v0.6.5 — attach a per-run plugin-local FileHandler so any
        # logger.exception(...) traceback (e.g. the bottom-of-execute catch
        # block, debate worker failures) lands on disk. The framework's shared
        # ~/.deeptrade/logs/deeptrade.log gets the same record from the
        # framework-level StreamHandler+RotatingFileHandler (when
        # setup_logging() has been called by cli.main), so the per-run file
        # is for support copy-paste rather than primary storage.
        log_handler, self._log_file_path = _attach_run_logfile(run_id)

        # v0.6 — renderer lifecycle. on_run_start is called once before any
        # event; the finally block guarantees on_run_finish + close even on
        # KeyboardInterrupt or unhandled exception (Plan §3.1, §3.6).
        self._renderer.on_run_start(
            run_id=run_id, params=params, debate=params.debate
        )
        try:
            # Surface the log file path right at the top of the run so users
            # can find it before anything goes wrong (dashboard log panel /
            # legacy stream both pick this up).
            if self._log_file_path is not None:
                self._dispatch_to_renderer(
                    StrategyEvent(
                        type=EventType.LOG,
                        level=EventLevel.INFO,
                        message=f"运行日志: {self._log_file_path}",
                        payload={"log_file": str(self._log_file_path)},
                    )
                )
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
                _detach_run_logfile(log_handler)
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

        # v0.6.8 — precondition check BEFORE _record_run_start so a bad --llm
        # never persists a "failed" run row. Mirrors _select_debate_providers
        # in debate mode; raises PreconditionError → cli.main → ✘ {msg}.
        provider_name = self._validate_single_provider(params)
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

        # Surface the chosen provider as the first persisted event of the run
        # (symmetric to debate mode's "[辩论模式] 启用，参与 LLM = ..." LOG).
        # None means we deferred to the framework default; show that verbatim
        # so the log is unambiguous.
        provider_display = provider_name if provider_name is not None else "(framework default)"
        provider_event = self._rt.emit(
            EventType.LOG,
            f"LLM provider = {provider_display}",
            payload={"llm_provider": provider_name},
        )
        self._seq += 1
        self._persist_event(run_id, self._seq, provider_event)
        events.append(provider_event)
        self._dispatch_to_renderer(provider_event)

        try:
            for ev in self._iter_pipeline(params):
                self._seq += 1
                self._persist_event(run_id, self._seq, ev)
                events.append(ev)
                self._metrics.observe(ev)
                self._dispatch_to_renderer(ev)
                if ev.type == EventType.VALIDATION_FAILED:
                    seen_validation_failed = True
        except KeyboardInterrupt:
            terminal_status = RunStatus.CANCELLED
            terminal_error = "用户手动中断"
            self._emit_cancelled_log()
        except Exception as e:  # noqa: BLE001
            if cancel_requested():
                terminal_status = RunStatus.CANCELLED
                terminal_error = "用户手动中断"
                logger.info(
                    "limit-up-board run %s cancelled (derived %s: %s)",
                    run_id, type(e).__name__, e,
                )
                self._emit_cancelled_log()
            else:
                terminal_status = RunStatus.FAILED
                terminal_error = self._handle_runtime_exception(e, run_id, "run")

        if terminal_status == RunStatus.SUCCESS and seen_validation_failed:
            terminal_status = RunStatus.PARTIAL_FAILED

        self._emit_observability_summary(run_id, events)
        self._shielded_record_run_finish(run_id, terminal_status, terminal_error, events)
        return RunOutcome(
            run_id=run_id, status=terminal_status, error=terminal_error, seen_events=events
        )

    def execute_sync_only(self, params: RunParams) -> RunOutcome:
        """Data-only path: same lifecycle as execute() but yields via _iter_sync."""
        run_id = str(uuid.uuid4())
        self._rt.run_id = run_id
        self._rt.tushare = build_tushare_client(
            self._rt, event_cb=self._on_tushare_event
        )

        log_handler, self._log_file_path = _attach_run_logfile(run_id)

        self._renderer.on_run_start(run_id=run_id, params=params, debate=False)
        if self._log_file_path is not None:
            self._dispatch_to_renderer(
                StrategyEvent(
                    type=EventType.LOG,
                    level=EventLevel.INFO,
                    message=f"运行日志: {self._log_file_path}",
                    payload={"log_file": str(self._log_file_path)},
                )
            )
        self._record_run_start(run_id, params)
        events: list[StrategyEvent] = []
        terminal_status = RunStatus.SUCCESS
        terminal_error: str | None = None

        try:
            for ev in self._iter_sync(params):
                self._seq += 1
                self._persist_event(run_id, self._seq, ev)
                events.append(ev)
                self._metrics.observe(ev)
                self._dispatch_to_renderer(ev)
        except KeyboardInterrupt:
            terminal_status = RunStatus.CANCELLED
            terminal_error = "用户手动中断"
            self._emit_cancelled_log()
        except Exception as e:  # noqa: BLE001
            if cancel_requested():
                terminal_status = RunStatus.CANCELLED
                terminal_error = "用户手动中断"
                logger.info(
                    "limit-up-board sync %s cancelled (derived %s: %s)",
                    run_id, type(e).__name__, e,
                )
                self._emit_cancelled_log()
            else:
                terminal_status = RunStatus.FAILED
                terminal_error = self._handle_runtime_exception(e, run_id, "sync")

        self._emit_observability_summary(run_id, events)
        self._shielded_record_run_finish(run_id, terminal_status, terminal_error, events)
        outcome = RunOutcome(
            run_id=run_id, status=terminal_status, error=terminal_error, seen_events=events
        )
        try:
            self._renderer.on_run_finish(outcome)
        finally:
            self._renderer.close()
            _detach_run_logfile(log_handler)
        return outcome

    # ----- pipeline iteration -------------------------------------------

    def _iter_sync(self, params: RunParams) -> Iterable[StrategyEvent]:
        """Data-only iteration (no LLM stages)."""
        rt = self._rt

        yield rt.emit(EventType.STEP_STARTED, "Step 0: resolve trade date")
        cal_df = rt.tushare.call("trade_cal")  # type: ignore[union-attr]
        cal = TradeCalendar(cal_df)
        latest = (
            None
            if params.trade_date
            else fetch_latest_trade_date(rt.tushare)  # type: ignore[arg-type]
        )
        T, T1 = resolve_trade_date(
            cal,
            latest_trade_date=latest,
            user_specified=params.trade_date,
        )
        # P1-A: 回填 lub_runs.trade_date —— sync 路径未传 --trade-date 时同样会
        # 在 _record_run_start 里落空字符串，必须在解析出真实 T 后立即更新。
        if rt.run_id:
            self._backfill_run_trade_date(rt.run_id, T)
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
            concept_repo=rt.concept_repo,
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
        latest = (
            None
            if params.trade_date
            else fetch_latest_trade_date(rt.tushare)  # type: ignore[arg-type]
        )
        T, T1 = resolve_trade_date(
            cal,
            latest_trade_date=latest,
            user_specified=params.trade_date,
        )
        # P1-A: 回填 lub_runs.trade_date —— 单 LLM 路径与辩论 _do_step_0_and_1 对齐，
        # 避免 history / report 按 trade_date 聚合时漏掉单 LLM run。
        if rt.run_id:
            self._backfill_run_trade_date(rt.run_id, T)
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
                concept_repo=rt.concept_repo,
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

        if not bundle.candidates:
            yield from self._emit_empty_report(bundle, params)
            return

        # P1-I/K: compute input_fingerprint once Step 1 has produced the
        # full bundle. Stored on the runtime so report writers, future LLM
        # replay-cache lookups, and the summary header can read it without
        # rebuilding. Failure to compute the fingerprint MUST NOT block the
        # run — degrade to None and log so observability still surfaces it.
        self._set_input_fingerprint(bundle, params, lub_cfg)

        # P3-A/B: resolve replay policy from CLI flags + LubConfig defaults,
        # activate the ContextVar so _complete_with_set_check picks it up.
        # The single-LLM path stays on the main thread for all LLM calls, so
        # one set() (no token) survives until Step 5 finalize. Workers in
        # debate mode enter their own apply_replay_context() before LLM
        # calls (ContextVars do NOT auto-propagate across ThreadPoolExecutor).
        from .replay_policy import (  # noqa: PLC0415
            _replay_policy_ctx, _stage_fingerprint_ctx,
        )
        from .profiles import (  # noqa: PLC0415
            STAGE_FINAL, STAGE_PREDICTION, STAGE_REVISION, STAGE_SCREENING,
        )
        _replay_policy_ctx.set(self._resolve_replay_policy(params, lub_cfg))
        _stage_fingerprint_ctx.set({
            STAGE_SCREENING: self._rt.input_fingerprint,
            STAGE_PREDICTION: self._rt.input_fingerprint,
            STAGE_FINAL: self._rt.input_fingerprint,
            STAGE_REVISION: self._rt.input_fingerprint,
        })

        # Step 2 — 强势初筛
        preset = cfg.app_profile  # v0.7: per-stage tuning resolved by plugin
        # v0.5 LGB: thread the configured min_score_floor into the prompts;
        # when LGB is fully disabled we pass None so the prompt drops the
        # numeric threshold sentence (the rest of the LGB guidance survives).
        # v0.7 LGB: lgb_decile_in_prompt controls whether lgb_decile reaches
        # the LLM (P2-2). Audit/render layers always keep the value.
        lgb_floor = lub_cfg.lgb_min_score_floor if rt.lgb_scorer is not None else None
        include_decile = lub_cfg.lgb_decile_in_prompt
        screening_result = None
        for ev, res in run_screening(
            llm=self._llm,
            bundle=bundle,
            preset=preset,
            lgb_min_score_floor=lgb_floor,
            include_decile=include_decile,
        ):
            yield ev
            if res is not None:
                screening_result = res
        # v0.18 — 强势分析不再过滤：``selected`` 仅作"强势推荐"建议标签，连板预测
        # 对 ``analyzed``（全部候选）运行，使候选集在重复运行间稳定。
        analyzed = screening_result.analyzed if screening_result else []
        selected = screening_result.selected if screening_result else []
        if not analyzed:
            yield from self._emit_empty_report(bundle, params, reason="强势分析后无候选股")
            return

        # Step 4 — 连板预测（对全部已分析候选运行）
        # v0.12.4 (P1-2)：空数组兜底策略由 lub_cfg 驱动；schema 内的
        # ``model_validator`` 通过 ContextVar 读取此策略，决定 repair / degraded /
        # fallback 三种行为。
        prediction_result = None
        with apply_empty_array_policy(lub_cfg.empty_array_policy):
            for ev, res in run_prediction(
                llm=self._llm,
                candidates=analyzed,
                bundle=bundle,
                preset=preset,
                lgb_min_score_floor=lgb_floor,
                include_decile=include_decile,
            ):
                yield ev
                if res is not None:
                    prediction_result = res
        predictions = prediction_result.predictions if prediction_result else []

        # Step 4.5 — 确定性全局重排 when 连板预测 was multi-batch (v0.18：不再调用 LLM)
        final_obj: FinalRankingResponse | None = None
        final_ranking_attempted = False
        if prediction_result and prediction_result.success_batches > 1 and predictions:
            final_ranking_attempted = True
            for ev, fr_obj in run_final_ranking(
                bundle=bundle,
                predictions=predictions,
            ):
                yield ev
                if fr_obj is not None:
                    final_obj = fr_obj

        # Step 5 — finalize
        terminal_status = RunStatus.SUCCESS
        if screening_result and screening_result.failed_batches > 0:
            terminal_status = RunStatus.PARTIAL_FAILED
        if prediction_result and prediction_result.failed_batches > 0:
            terminal_status = RunStatus.PARTIAL_FAILED
        if final_ranking_attempted and final_obj is None:
            terminal_status = RunStatus.PARTIAL_FAILED

        _write_stage_results(rt, "r1", analyzed)
        _write_stage_results(rt, "r2", predictions)
        if final_obj is not None:
            _write_stage_results(rt, "final_ranking", final_obj.finalists)

        failed_batches: list[str] = []
        if screening_result and screening_result.failed_batch_ids:
            failed_batches.extend(f"初筛#{b}" for b in screening_result.failed_batch_ids)
        if prediction_result and prediction_result.failed_batch_ids:
            failed_batches.extend(f"预测#{b}" for b in prediction_result.failed_batch_ids)
        if final_ranking_attempted and final_obj is None:
            failed_batches.append("全局重排")

        report_path, json_error = write_report(
            rt.run_id,
            status=terminal_status,
            bundle=bundle,
            selected=selected,
            predictions=predictions,
            final_ranking=final_obj,
            failed_batch_ids=failed_batches or None,
            input_fingerprint=rt.input_fingerprint,
            analyzed=analyzed,
        )
        export_llm_calls(rt.run_id, rt.db)
        json_path = report_path / "summary.json"
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"Report written: {report_path}",
            payload={
                "report_dir": str(report_path),
                "report_json": str(json_path) if json_path.is_file() else None,
                "selected": len(selected),
                "predictions": len(predictions),
                "final_ranking_used": final_obj is not None,
            },
        )
        if json_error is not None:
            yield from self._emit_json_build_failed(report_path, json_error)

        # PR #1 — 单 LLM 模式 T 日预测留痕（胜率分析样本来源）。
        # 失败只 warning：胜率分析是辅助产物，不能影响 run 状态或后续上传。
        if predictions:
            try:
                from .winrate.persistence import record_predictions_from_run

                saved = record_predictions_from_run(
                    rt=rt,
                    bundle=bundle,
                    predictions=predictions,
                    final_ranking=final_obj,
                    run_id=rt.run_id or "",
                    trade_date=bundle.trade_date,
                    next_trade_date=bundle.next_trade_date,
                )
                yield rt.emit(
                    EventType.RESULT_PERSISTED,
                    f"Prediction records saved: {saved}",
                    payload={"prediction_records": saved},
                )
            except Exception as exc:  # noqa: BLE001 — degrade, never fail the run
                logger.warning("winrate record save failed: %s", exc, exc_info=True)
                yield rt.emit(
                    EventType.LOG,
                    f"prediction records skipped: {exc}",
                    level=EventLevel.WARNING,
                )

        yield from self._maybe_upload_summary(report_path, bundle.trade_date)

    # ====================================================================
    # Debate mode (multi-LLM)
    # ====================================================================

    def _execute_debate(self, run_id: str, params: RunParams) -> RunOutcome:
        """Multi-LLM debate flow.

        Step 0/1 stay on the main thread; 强势初筛 + 连板预测 + (final_ranking)
        fan out across providers in phase A; 辩论修订 fans out across the same
        providers in phase B with peer outputs cross-fed and anonymised.
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
            # v0.7 — lgb_decile_in_prompt likewise shared across debate workers.
            lub_cfg = load_config(rt.db)
            lgb_floor = lub_cfg.lgb_min_score_floor if rt.lgb_scorer is not None else None
            include_decile = lub_cfg.lgb_decile_in_prompt

            # v0.13.0 (P1-4)：辩论 worker 接收 frozen ProviderConfigSnapshot，
            # 不再共享主线程 ConfigService / Database。snapshot 仅构造一次。
            config_snapshot = build_provider_config_snapshot(rt.config)

            # ----- Phase A: parallel 强势初筛 + 连板预测 + (final_ranking) ---
            emit(
                rt.emit(
                    EventType.LIVE_STATUS,
                    f"[辩论模式] Phase A — 并行执行 初筛+预测 ({len(providers)} 个 LLM)",
                )
            )
            # P1-H: collect by provider key, then materialise the final list in
            # canonical ``providers`` input order. Events still stream as
            # workers complete (dashboard UX unchanged); persistence / peer
            # inputs / reports iterate canonical order so a slow provider
            # never reshuffles downstream artifacts.
            # P3-A: hand the resolved policy + fingerprint to each worker so
            # apply_replay_context can be entered inside the thread.
            replay_policy = self._resolve_replay_policy(params, lub_cfg)
            result_by_provider: dict[str, ProviderDebateResult] = {}
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
                        config_snapshot,
                        lgb_floor,
                        include_decile,
                        lub_cfg.empty_array_policy,
                        replay_policy,
                        rt.input_fingerprint,
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
                    result_by_provider[provider] = result
                    for ev in result_events(result, "phase_a"):
                        emit(ev)
                        if ev.type == EventType.VALIDATION_FAILED:
                            seen_validation_failed = True
            # Reorder by canonical providers list (user --debate-llms order
            # or LLMManager.list_providers fallback). All downstream consumers
            # iterate this list, so Phase-B survivors and report sections
            # inherit canonical order automatically.
            provider_results.extend(result_by_provider[p] for p in providers)

            # Persist phase-A stage results
            for r in provider_results:
                if r.screening_result and r.screening_result.analyzed:
                    _write_stage_results(
                        rt, f"r1:{r.provider}", r.screening_result.analyzed,
                        llm_provider=r.provider,
                    )
                if r.prediction_result and r.prediction_result.predictions:
                    _write_stage_results(
                        rt, f"r2_initial:{r.provider}", r.prediction_result.predictions,
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
                        "跳过辩论修订阶段，按现有结果出报告",
                        level=EventLevel.WARN,
                    )
                )
                terminal_status = RunStatus.PARTIAL_FAILED
            else:
                # ----- Phase B: parallel 辩论修订 -----------------------------
                emit(
                    rt.emit(
                        EventType.LIVE_STATUS,
                        f"[辩论模式] Phase B — 并行执行 辩论修订 ({len(survivors)} 个 LLM)",
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
                            config_snapshot,
                            lub_cfg.empty_array_policy,
                            replay_policy,
                            rt.input_fingerprint,
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
                if r.screening_result and r.screening_result.failed_batch_ids:
                    failed_batches.extend(f"{tag}:初筛#{b}" for b in r.screening_result.failed_batch_ids)
                if r.prediction_result and r.prediction_result.failed_batch_ids:
                    failed_batches.extend(f"{tag}:预测#{b}" for b in r.prediction_result.failed_batch_ids)
                if r.final_attempted and r.final_initial is None:
                    failed_batches.append(f"{tag}:全局重排")
                if r.revision and not r.revision.success:
                    failed_batches.append(f"{tag}:修订")

            if failed_batches:
                terminal_status = RunStatus.PARTIAL_FAILED

            # Write report (debate-aware)
            # ``json_error`` 在辩论模式下恒为 None（render.write_report 不为辩论
            # 模式生成 summary.json），无需 emit 失败事件。即使如此还是把变量解
            # 出来，未来 PR-X 给辩论 schema 时直接复用同一通路。
            report_path, json_error = write_report(
                run_id,
                status=terminal_status,
                bundle=bundle,
                selected=[],  # main report tables are replaced by debate sections
                predictions=[],
                final_ranking=None,
                failed_batch_ids=failed_batches or None,
                debate_results=provider_results,
                input_fingerprint=rt.input_fingerprint,
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
            if json_error is not None:
                for ev in self._emit_json_build_failed(report_path, json_error):
                    emit(ev)
            # v0.16.1 — 辩论模式之前不调上传链路（C 修复）。即使当前 debate 模式
            # 没有 summary.json 落盘，也走 _maybe_upload_summary：Fix B 的 is_file
            # 兜底会优雅跳过并 emit 一条 INFO，把"为何不上传"显式告诉用户，而不
            # 是过去那种神秘静默。一旦 debate-mode JSON schema (PR-X) 合并就立刻
            # 自动启用上传，无需再改 runner。
            for ev in self._maybe_upload_summary(report_path, bundle.trade_date):
                emit(ev)

        except KeyboardInterrupt:
            terminal_status = RunStatus.CANCELLED
            terminal_error = "用户手动中断"
            self._emit_cancelled_log()
        except Exception as e:  # noqa: BLE001
            if cancel_requested():
                terminal_status = RunStatus.CANCELLED
                terminal_error = "用户手动中断"
                logger.info(
                    "limit-up-board debate %s cancelled (derived %s: %s)",
                    run_id, type(e).__name__, e,
                )
                self._emit_cancelled_log()
            else:
                terminal_status = RunStatus.FAILED
                terminal_error = self._handle_runtime_exception(e, run_id, "debate")

        if terminal_status == RunStatus.SUCCESS and seen_validation_failed:
            terminal_status = RunStatus.PARTIAL_FAILED

        self._emit_observability_summary(run_id, events)
        self._shielded_record_run_finish(run_id, terminal_status, terminal_error, events)
        return RunOutcome(
            run_id=run_id,
            status=terminal_status,
            error=terminal_error,
            seen_events=events,
            debate_results=provider_results,
        )

    def _validate_single_provider(self, params: RunParams) -> str | None:
        """Resolve & validate the non-debate-mode provider override.

        Returns the provider name to pass to ``LLMManager.get_client`` —
        either ``params.llm_provider`` (when the user pinned one via
        ``--llm``) or ``None`` (defer to framework default).

        Raises:
            PreconditionError: when ``--llm`` named a provider that is not
                in ``LLMManager.list_providers()`` (i.e. not configured or
                missing api_key). Surface BEFORE ``_record_run_start`` so no
                run row is persisted; cli.main renders as ``✘ {message}``.
        """
        override = pick_llm_provider(self._rt, params.llm_provider)
        if override is None:
            return None
        available = self._rt.llms.list_providers()
        if override not in available:
            raise PreconditionError(
                f"--llm 指定的 provider {override!r} 未配置或缺 api_key; "
                f"当前可用: {available}。"
                "请运行 `deeptrade config set-llm` 配置后重试。"
            )
        return override

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

        emit(rt.emit(EventType.STEP_STARTED, "Step 0: resolve trade date"))
        cal_df = rt.tushare.call("trade_cal")  # type: ignore[union-attr]
        cal = TradeCalendar(cal_df)
        latest = (
            None
            if params.trade_date
            else fetch_latest_trade_date(rt.tushare)  # type: ignore[arg-type]
        )
        T, T1 = resolve_trade_date(
            cal,
            latest_trade_date=latest,
            user_specified=params.trade_date,
        )
        # P3-2: 回填 lub_runs.trade_date —— _record_run_start 时 params.trade_date 可能为 None
        # （CLI 未传 --trade-date），导致表里落了 ""，对 history / report join 不友好。
        # Step 0 解析出真实 T 后立即更新，保证 lub_runs.trade_date 始终非空。
        if rt.run_id:
            self._backfill_run_trade_date(rt.run_id, T)
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
                concept_repo=rt.concept_repo,
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

        if not bundle.candidates:
            for ev in self._emit_empty_report(bundle, params):
                emit(ev)
            return None
        # P1-I/K: same fingerprint contract as the single-LLM path.
        self._set_input_fingerprint(bundle, params, lub_cfg)
        # P3-A/B: activate replay policy on the main thread. Workers each
        # call apply_replay_context() themselves with the policy + fp passed
        # in as arguments — ContextVars don't propagate across pool threads.
        from .replay_policy import (  # noqa: PLC0415
            _replay_policy_ctx, _stage_fingerprint_ctx,
        )
        from .profiles import (  # noqa: PLC0415
            STAGE_FINAL, STAGE_PREDICTION, STAGE_REVISION, STAGE_SCREENING,
        )
        _replay_policy_ctx.set(self._resolve_replay_policy(params, lub_cfg))
        _stage_fingerprint_ctx.set({
            STAGE_SCREENING: self._rt.input_fingerprint,
            STAGE_PREDICTION: self._rt.input_fingerprint,
            STAGE_FINAL: self._rt.input_fingerprint,
            STAGE_REVISION: self._rt.input_fingerprint,
        })
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
        lub_cfg = load_config(rt.db)
        report_path, json_error = write_report(
            rt.run_id,
            status=RunStatus.SUCCESS,
            bundle=bundle,
            selected=[],
            predictions=[],
            final_ranking=None,
            input_fingerprint=rt.input_fingerprint,
        )
        export_llm_calls(rt.run_id, rt.db)
        json_path = report_path / "summary.json"
        yield rt.emit(
            EventType.RESULT_PERSISTED,
            f"empty report ({reason})",
            payload={
                "report_dir": str(report_path),
                "report_json": str(json_path) if json_path.is_file() else None,
                "reason": reason,
            },
        )
        if json_error is not None:
            yield from self._emit_json_build_failed(report_path, json_error)
        yield from self._maybe_upload_summary(report_path, bundle.trade_date)

    def _emit_json_build_failed(
        self, report_path: Path, json_error: str
    ) -> Iterable[StrategyEvent]:
        """v0.16.1 (Fix A) — emit a single WARN event when ``build_strategy_report``
        raised inside :func:`render.write_report`. Before this, the failure
        only landed in ``logger.warning`` (per-run log file) and the missing
        ``summary.json`` then cascaded into a silent upload skip — terminal
        looked normal end-to-end. The event payload carries the exception
        ``repr`` so users can paste it back without trawling log files.
        """
        rt = self._rt
        yield rt.emit(
            EventType.LOG,
            f"⚠ summary.json 生成失败：{json_error}",
            level=EventLevel.WARN,
            payload={
                "report_dir": str(report_path),
                "json_error": json_error,
                "stage": "build_strategy_report",
            },
        )

    def _maybe_upload_summary(
        self, report_path: Path, trade_date: str
    ) -> Iterable[StrategyEvent]:
        """Best-effort POST ``summary.json`` via 框架 v0.11 ``ReportUploader``.

        上传开关 / URL / 超时 / token 全部从框架 ``report.upload.*`` 读取；
        插件只负责传文件路径 + ``plugin_name`` + ``trade_date``。
        框架返回 ``status="skipped_*"`` 时静默返回（保留旧行为：用户未开启就不打扰）。

        v0.16.1 (Fix B) — 入口加 ``json_path.is_file()`` 兜底：文件不存在时直接
        emit INFO 级 skip 事件并 return，不再让框架 uploader 拿着不存在的路径
        走一遍 HTTP 准备栈。两种情况会命中：
        （a）单 LLM 模式 ``build_strategy_report`` 抛异常（v0.16.1 Fix A 已经
            emit 过一条 WARN，这里再补一条 INFO 描述"上传跳过的具体原因"，
            两条事件分工清晰：WARN 说 JSON 没写出来、INFO 说所以上传跳了）；
        （b）辩论模式（debate-mode summary.json 暂未实现，render.write_report
            主动跳过；现在 _execute_debate 也走 _maybe_upload_summary，本兜底
            给出可见信号而不是过去的静默）。
        """
        if self._ctx is None:
            # 没有 PluginContext（旧测试 / 未注入）时按"上传未启用"对待。
            return
        json_path = report_path / "summary.json"
        rt = self._rt
        if not json_path.is_file():
            yield rt.emit(
                EventType.LOG,
                f"summary.json 未生成，跳过上传：{json_path}",
                level=EventLevel.INFO,
                payload={
                    "enabled": True,
                    "status": "skipped_no_local_file",
                    "json_path": str(json_path),
                    "trade_date": trade_date,
                },
            )
            return
        uploader: ReportUploader = self._ctx.make_report_uploader(run_id=rt.run_id)
        result = uploader.upload(
            json_path,
            plugin_name="打板策略",
            trade_date=trade_date,
        )

        if result.status.startswith("skipped"):
            return

        # v0.12.3 事件 payload 兼容：沿用 enabled / url / status / duration_ms /
        # public_url / public_path / error_class 字段名，便于前端 / 日志无感升级。
        payload: dict[str, Any] = {
            "enabled": True,
            "url": result.public_url,
            "status": result.status,
            "duration_ms": result.duration_ms,
            "public_url": result.public_url,
            "public_path": result.public_path,
            "public_index": result.public_index,
            "public_date": result.public_date,
            "trade_date": trade_date,
            "json_path": str(json_path),
            "error_class": result.error_class,
        }
        if result.status == "ok":
            yield rt.emit(
                EventType.LOG,
                f"📤 报告已同步至官网：{result.public_url}",
                payload=payload,
            )
        else:
            yield rt.emit(
                EventType.LOG,
                f"⚠ summary.json 上传失败：{result.error}",
                level=EventLevel.WARN,
                payload=payload,
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

    def _emit_cancelled_log(self) -> None:
        """Push two short LOG events explaining the user cancel.

        Replaces the line-by-line traceback fan-out of
        :meth:`_handle_runtime_exception` for cancel-class outcomes. WARN
        level (not ERROR) so the dashboard renders these in amber rather
        than red, matching the "user intent, not failure" semantic.
        Renderer failures are swallowed — the run is already ending.
        """
        for msg in ("用户手动中断运行，正在停止当前任务", "运行已取消"):
            try:
                self._dispatch_to_renderer(
                    StrategyEvent(
                        type=EventType.LOG,
                        level=EventLevel.WARN,
                        message=msg,
                    )
                )
            except Exception:  # noqa: BLE001
                pass

    def _emit_observability_summary(
        self, run_id: str, events: list[StrategyEvent]
    ) -> None:
        """v0.13.0 (P2-3) — emit a LOG event with the run's aggregated
        ``stage_duration_ms / tushare_api_calls / llm_calls / lgb / upload``
        right before persistence. Failure is logged but never raised — the
        rest of the finalize path must still run."""
        try:
            summary_payload = self._metrics.build_summary_payload()
            obs_event = StrategyEvent(
                type=EventType.LOG,
                level=EventLevel.INFO,
                message="[observability] run summary",
                payload=summary_payload,
            )
            self._seq += 1
            self._persist_event(run_id, self._seq, obs_event)
            events.append(obs_event)
            self._dispatch_to_renderer(obs_event)
        except Exception:  # noqa: BLE001
            logger.warning("failed to emit OBSERVABILITY_SUMMARY", exc_info=True)

    def _shielded_record_run_finish(
        self,
        run_id: str,
        terminal_status: RunStatus,
        terminal_error: str | None,
        events: list[StrategyEvent],
    ) -> None:
        """Run ``_record_run_finish`` with SIGINT temporarily ignored.

        Without this, a user mashing Ctrl+C during the finally block can
        skip the ``UPDATE lub_runs SET status = ...`` write and strand the
        row at ``RunStatus.RUNNING``. The shield is the whole UPDATE — a
        sub-millisecond window — after which SIGINT semantics are restored
        immediately. If signal manipulation isn't available (off main
        thread / embedded interpreter), we run unshielded; the prior
        behaviour was the same.
        """
        prev = None
        installed = False
        try:
            prev = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            installed = True
        except (ValueError, OSError):
            installed = False
        try:
            self._record_run_finish(run_id, terminal_status, terminal_error, events)
        finally:
            if installed and prev is not None:
                try:
                    signal.signal(signal.SIGINT, prev)
                except (ValueError, OSError):
                    pass

    def _handle_runtime_exception(
        self, e: BaseException, run_id: str, mode: str
    ) -> str:
        """Format a runtime exception for the outcome, surface it to the user.

        v0.6.5 — previously the bottom-of-execute catch block only set
        ``terminal_error = "ExcType: msg"`` and called ``logger.exception``.
        With the rich dashboard owning stdout (and Python's last-resort log
        handler going to stderr), tracebacks frequently never reached the
        user. This helper:

          1. emits each traceback line as a ``LOG ERROR`` event → the active
             renderer's log panel surfaces them in order;
          2. appends a single LOG event pointing at the per-run log file so
             users have a copy-pasteable path even when the dashboard scrolls;
          3. delegates to ``logger.exception`` so the FileHandler attached by
             :func:`_attach_run_logfile` records the full traceback on disk.

        ``mode`` distinguishes the call sites in the framework log ("run" /
        "debate" / "sync"); the user-facing summary is the same for all
        three.
        """
        summary = f"{type(e).__name__}: {e}"
        logger.exception("limit-up-board %s %s raised", mode, run_id)

        # Feed each traceback line individually so the dashboard's log ring
        # buffer captures the stack frame-by-frame (the buffer's maxlen was
        # bumped to 12 in v0.6.5 to make room for this).
        try:
            for line in _format_traceback(e):
                if not line.strip():
                    continue
                self._dispatch_to_renderer(
                    StrategyEvent(
                        type=EventType.LOG,
                        level=EventLevel.ERROR,
                        message=line,
                    )
                )
        except Exception:  # noqa: BLE001 — never let surfacing the error fail the run
            logger.warning("surfacing traceback to renderer failed", exc_info=True)

        # Point the user at the log file (banner / log panel both pick this
        # up). If FileHandler attach failed earlier, ``_log_file_path`` is
        # None and we omit the hint rather than print a misleading path.
        if self._log_file_path is not None:
            try:
                self._dispatch_to_renderer(
                    StrategyEvent(
                        type=EventType.LOG,
                        level=EventLevel.ERROR,
                        message=f"完整 traceback 见 {self._log_file_path}",
                        payload={"log_file": str(self._log_file_path)},
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            return f"{summary}  (完整 traceback 见 {self._log_file_path})"
        return summary

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
        # ``is_intraday`` column is retained for migration-immutability; we
        # always write FALSE now that the intraday opt-in flag is gone.
        self._rt.db.execute(
            "INSERT INTO lub_runs(run_id, trade_date, status, is_intraday, started_at, "
            "params_json) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
            (
                run_id,
                params.trade_date or "",
                RunStatus.RUNNING.value,
                False,
                json.dumps(params.__dict__, ensure_ascii=False),
            ),
        )

    def _backfill_run_trade_date(self, run_id: str, trade_date: str) -> None:
        """更新 lub_runs.trade_date —— Step 0 解析真实 T 后调用。

        ``_record_run_start`` 写入的 trade_date 可能为 ""（用户未传 ``--trade-date``）。
        Step 0 解析出真实 T 后必须立即回填，否则 history / report join 不到。
        """
        self._rt.db.execute(
            "UPDATE lub_runs SET trade_date=? WHERE run_id=?",
            (trade_date, run_id),
        )

    def _resolve_replay_policy(
        self,
        params: RunParams,
        lub_cfg: LubConfig,
    ):
        """P3-B: turn RunParams CLI flags + LubConfig defaults into an
        ``LLMReplayPolicy``. Returns the policy regardless of framework
        support — pipeline-level feature detection short-circuits the
        per-call pass-through.
        """
        from .replay_policy import _ReplayCLIFlags, build_replay_policy  # noqa: PLC0415

        cli = _ReplayCLIFlags(
            fresh_llm=params.fresh_llm,
            no_llm_replay=params.no_llm_replay,
            replay_only=params.replay_only,
        )
        return build_replay_policy(
            cli=cli,
            cfg_enabled=lub_cfg.llm_replay_enabled,
            cfg_write=lub_cfg.llm_replay_write,
            cfg_ttl_days=lub_cfg.llm_replay_ttl_days,
        )

    def _set_input_fingerprint(
        self,
        bundle: Round1Bundle,
        params: RunParams,
        lub_cfg: LubConfig,
    ) -> None:
        """P1-I/K: compute & stash the run-level input_fingerprint.

        Called once after Step 1 completes, before any LLM stage runs. The
        fingerprint hashes the canonical-JSON of Round1Bundle (candidates +
        market_summary + sector_strength + lgb_model_id), the full
        :class:`LubConfig`, the four StageProfiles in use, and the
        ``LLM_SCHEMA_VERSION`` / ``PROMPT_TEMPLATE_VERSION`` sentinels.

        Failure here MUST NOT abort the run — degrades to ``None`` and logs.
        The fingerprint is observability + Phase 3 cache-key input; missing
        it just means the summary header shows ``unknown`` for that run.
        """
        from .fingerprint import build_input_fingerprint  # noqa: PLC0415
        from .profiles import (  # noqa: PLC0415
            LLM_SCHEMA_VERSION,
            PROFILES,
            PROMPT_TEMPLATE_VERSION,
            STAGE_FINAL,
            STAGE_PREDICTION,
            STAGE_REVISION,
            STAGE_SCREENING,
        )

        try:
            preset = self._rt.config.get_app_config().app_profile
            stage_profiles = {
                STAGE_SCREENING: PROFILES[preset][STAGE_SCREENING],
                STAGE_PREDICTION: PROFILES[preset][STAGE_PREDICTION],
                STAGE_FINAL: PROFILES[preset][STAGE_FINAL],
                STAGE_REVISION: PROFILES[preset][STAGE_REVISION],
            }
            digest, _payload = build_input_fingerprint(
                trade_date=bundle.trade_date,
                next_trade_date=bundle.next_trade_date,
                daily_lookback=params.daily_lookback,
                moneyflow_lookback=params.moneyflow_lookback,
                lub_config=lub_cfg,
                bundle=bundle,
                stage_profiles=stage_profiles,
                llm_schema_version=LLM_SCHEMA_VERSION,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            )
            self._rt.input_fingerprint = digest
        except Exception:  # noqa: BLE001 — fingerprint is observability-only
            logger.exception("input_fingerprint computation failed; surfacing as None")
            self._rt.input_fingerprint = None

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
    config_snapshot: ProviderConfigSnapshot,
    lgb_min_score_floor: float | None = 30.0,
    include_decile: bool = True,
    empty_array_policy: str = "repair",
    replay_policy: Any = None,
    input_fingerprint: str | None = None,
) -> ProviderDebateResult:
    """One provider's 强势初筛 + 连板预测 + (optional) final_ranking. Tagged
    events are attached to the returned ProviderDebateResult; the main thread
    will emit them in completion order.

    v0.12.4 (P1-2): ``empty_array_policy`` is applied via the schemas
    ContextVar inside this worker thread; ThreadPoolExecutor doesn't
    auto-propagate ContextVars so callers pass the policy explicitly.

    v0.13.0 (P1-4): receives a frozen :class:`ProviderConfigSnapshot` instead
    of a live :class:`ConfigService`. The worker never reads from the main
    thread's ``Database._conn`` / ``SecretStore``.

    v0.15.0 (P3-A): ``replay_policy`` + ``input_fingerprint`` are activated
    via :func:`apply_replay_context` inside this thread (ContextVars don't
    propagate across ThreadPoolExecutor). When ``replay_policy`` is None
    (caller used a pre-Phase-3 codepath) the default "off" policy applies.
    """
    from .replay_policy import LLMReplayPolicy, apply_replay_context  # noqa: PLC0415
    from .profiles import (  # noqa: PLC0415
        STAGE_FINAL, STAGE_PREDICTION, STAGE_REVISION, STAGE_SCREENING,
    )

    db, wrt = open_worker_runtime(plugin_id, run_id, config_snapshot=config_snapshot)
    out = ProviderDebateResult(provider=provider)
    effective_policy = replay_policy or LLMReplayPolicy(read_enabled=False, write_enabled=False)
    stage_fps = {
        STAGE_SCREENING: input_fingerprint,
        STAGE_PREDICTION: input_fingerprint,
        STAGE_FINAL: input_fingerprint,
        STAGE_REVISION: input_fingerprint,
    }
    try:
        llm = wrt.llms.get_client(
            provider, plugin_id=plugin_id, run_id=run_id, reports_dir=reports_dir
        )

        events: list[StrategyEvent] = []

        with apply_replay_context(effective_policy, stage_to_fingerprint=stage_fps):
            for ev, res in run_screening(
                llm=llm, bundle=bundle, preset=preset,
                lgb_min_score_floor=lgb_min_score_floor,
                include_decile=include_decile,
            ):
                events.append(ev)
                if res is not None:
                    out.screening_result = res
            # v0.18 — 连板预测对全部已分析候选运行（不再过滤到 selected 子集）。
            analyzed = out.screening_result.analyzed if out.screening_result else []

            if analyzed:
                with apply_empty_array_policy(empty_array_policy):  # type: ignore[arg-type]
                    for ev, res in run_prediction(
                        llm=llm, candidates=analyzed, bundle=bundle, preset=preset,
                        lgb_min_score_floor=lgb_min_score_floor,
                        include_decile=include_decile,
                    ):
                        events.append(ev)
                        if res is not None:
                            out.prediction_result = res

            if out.prediction_result and out.prediction_result.success_batches > 1 and out.prediction_result.predictions:
                out.final_attempted = True
                for ev, fr_obj in run_final_ranking(
                    bundle=bundle, predictions=out.prediction_result.predictions
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
    own_predictions: list[ContinuationCandidate],
    peers: list[tuple[str, list[ContinuationCandidate]]],
    config_snapshot: ProviderConfigSnapshot,
    empty_array_policy: str = "repair",
    replay_policy: Any = None,
    input_fingerprint: str | None = None,
) -> tuple[list[StrategyEvent], DebateRoundResult]:
    """One provider's 辩论修订 (peer-aware revision).

    v0.12.4 (P1-2): ``empty_array_policy`` applied via ContextVar; see
    :func:`_worker_phase_a` for rationale.

    v0.13.0 (P1-4): receives a frozen :class:`ProviderConfigSnapshot` instead
    of a live :class:`ConfigService` (worker isolation).

    v0.15.0 (P3-A): replay context activated inside the thread; see
    :func:`_worker_phase_a` docstring.
    """
    from .replay_policy import LLMReplayPolicy, apply_replay_context  # noqa: PLC0415
    from .profiles import STAGE_REVISION  # noqa: PLC0415

    db, wrt = open_worker_runtime(plugin_id, run_id, config_snapshot=config_snapshot)
    effective_policy = replay_policy or LLMReplayPolicy(read_enabled=False, write_enabled=False)
    try:
        llm = wrt.llms.get_client(
            provider, plugin_id=plugin_id, run_id=run_id, reports_dir=reports_dir
        )
        events: list[StrategyEvent] = []
        result: DebateRoundResult | None = None
        with apply_replay_context(
            effective_policy,
            stage_to_fingerprint={STAGE_REVISION: input_fingerprint},
        ):
            with apply_empty_array_policy(empty_array_policy):  # type: ignore[arg-type]
                for ev, res in run_debate_revision(
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
            result = DebateRoundResult(error="run_debate_revision yielded no terminal result")
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
    """Persist 强势初筛/连板预测/全局重排/辩论修订 outputs to lub_stage_results.

    In debate mode, ``stage`` is suffixed with the provider (e.g.
    ``r1:deepseek``) to keep the (run_id, stage, ts_code) PK unique across
    providers; the explicit ``llm_provider`` column lets queries filter by
    provider without parsing the stage string.

    P1-G: defensively stable-sort ``items`` by ``(rank or final_rank, ts_code)``
    before inserting. The data-layer / pipeline already produce
    deterministically-ordered outputs (P1-B candidates → R1; P1-F finalists),
    but this defensive sort guarantees row order in ``lub_stage_results``
    stays stable even if a future refactor adds a non-deterministic step.
    StrongCandidate has no ``rank`` field — those items fall through to
    ts_code asc.
    """
    if not items:
        return

    def _sort_key(it: Any) -> tuple[int, str]:
        d_ = it.model_dump(mode="json") if hasattr(it, "model_dump") else dict(it)
        rank_val = d_.get("rank") or d_.get("final_rank")
        # Missing rank → sentinel ``10**9`` so unranked StrongCandidate rows
        # collapse to ts_code-asc among themselves.
        rank_int = int(rank_val) if rank_val is not None else 10**9
        ts = str(d_.get("ts_code") or "")
        return (rank_int, ts)

    items_sorted = sorted(items, key=_sort_key)

    for i, item in enumerate(items_sorted):
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
