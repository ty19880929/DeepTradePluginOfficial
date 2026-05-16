"""Rich-based dashboard renderer for the ``limit-up-board`` runner.

This is the user-facing renderer when the terminal supports it (Plan §3.5
fallback matrix → :func:`limit_up_board.ui.choose_renderer`). It owns a
single :class:`rich.live.Live` region and redraws the full frame on every
incoming event. Refresh is throttled to 8 fps by ``Live`` to keep flicker
out of high-event-rate phases.

Event → state-mutation logic lives here (Plan §3.2.3); the actual
*rendering* is delegated to :mod:`limit_up_board.ui.layout`. The dashboard
must **never raise** from ``on_event`` (Plan §3.6.1); we wrap every
handler in ``try / except`` and log warnings instead.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from deeptrade.plugins_api.events import EventLevel, EventType
from deeptrade.theme import EVA_THEME
from rich.console import Console
from rich.live import Live

from .debate_view import DebateGrid
from .layout import ConfigSummary, DashboardState, render_dashboard
from .mapping import parse_stage_id, title_for
from .stage_model import StageStatus

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

    from limit_up_board.runner import RunOutcome, RunParams


logger = logging.getLogger(__name__)


def _plugin_version() -> str:
    """Best-effort plugin version probe; never raises."""
    from pathlib import Path

    yaml_path = (
        Path(__file__).resolve().parent.parent.parent / "deeptrade_plugin.yaml"
    )
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        return str(data.get("version", "?"))
    except Exception:  # noqa: BLE001 — version is decorative only
        return "?"


class RichDashboardRenderer:
    """Renderer that paints a rich.Live dashboard frame per event.

    Implements the :class:`EventRenderer` Protocol; see ``protocol.py`` for
    the lifecycle contract.
    """

    def __init__(self, *, no_color: bool = False) -> None:
        # Console writes to stdout; the post-run ``status:`` line printed by
        # cli.py composes naturally after Live.__exit__ restores the stream.
        self._console = Console(
            theme=EVA_THEME,
            no_color=no_color,
            highlight=False,
            soft_wrap=False,
        )
        self._no_color = no_color  # exposed for tests
        self._live: Live | None = None
        self._state = DashboardState(plugin_version=_plugin_version())
        self._closed = False
        # v0.6.7 — root-logger stream handlers we silenced for the duration of
        # the run. Restored in close(). Empty outside of a run.
        # See _silence_stream_handlers / _restore_stream_handlers for why.
        self._silenced_handlers: list[tuple[logging.Handler, int]] = []

    # ----- lifecycle -----------------------------------------------------

    def on_run_start(
        self, *, run_id: str, params: RunParams, debate: bool
    ) -> None:
        self._state.run_id = run_id
        self._state.started_at = datetime.now()
        self._state.debate = debate
        if debate:
            self._state.debate_grid = DebateGrid()
        # v0.6.7 — silence the framework's stderr StreamHandler for the run.
        # cli.main calls deeptrade.core.logging_config.setup_logging() which
        # attaches one; concurrent stderr writes during Live._refresh shift
        # the cursor out from under Rich's "move up N lines" bookkeeping and
        # leak header/panel fragments above the live region (issue surfaced
        # in v0.6.6 after setup_logging() was wired up). The per-run
        # FileHandler attached by runner._attach_run_logfile and the
        # framework's RotatingFileHandler keep capturing everything to disk.
        self._silence_stream_handlers()
        self._live = Live(
            self._render_frame(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
            redirect_stdout=False,
            # ``redirect_stderr=True`` is a safety net: even with handlers
            # silenced, any third-party C extension / direct ``sys.stderr``
            # write gets serialized through Rich instead of corrupting the
            # live region.
            redirect_stderr=True,
        )
        self._live.__enter__()

    def on_event(self, ev: StrategyEvent) -> None:
        try:
            self._handle_event(ev)
        except Exception as e:  # noqa: BLE001 — contract: never raise out
            logger.warning("dashboard handler raised on %s: %s", ev.type, e)
            return
        self._safe_update()

    def on_run_finish(self, outcome: RunOutcome) -> None:
        try:
            self._finalise_state(outcome)
            self._safe_update()
        except Exception as e:  # noqa: BLE001
            logger.warning("dashboard on_run_finish raised: %s", e)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception as e:  # noqa: BLE001 — best-effort terminal restore
                logger.warning("Live.__exit__ raised: %s", e)
            self._live = None
        # Restore stream handler levels we silenced in on_run_start. Done
        # AFTER Live.__exit__ so any tail-end stderr writes during teardown
        # also stay clear of the live region.
        self._restore_stream_handlers()

    # ----- root-logger stream handler silencing --------------------------

    def _silence_stream_handlers(self) -> None:
        """Quiet stderr/stdout ``StreamHandler``s on the root logger.

        ``FileHandler`` subclasses are excluded by an ``isinstance`` check so
        the per-run + rotating disk logs keep flowing. Levels are saved into
        ``self._silenced_handlers`` for :meth:`_restore_stream_handlers`.
        """
        if self._silenced_handlers:
            # Defensive: re-entry should not double-silence.
            return
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                continue
            if not isinstance(h, logging.StreamHandler):
                continue
            stream = getattr(h, "stream", None)
            if stream not in (sys.stderr, sys.stdout):
                continue
            self._silenced_handlers.append((h, h.level))
            # CRITICAL+10 is unreachable in practice (logging.CRITICAL = 50),
            # so this both drops INFO/WARN/ERROR records and leaves room for
            # someone bumping the root level even higher.
            h.setLevel(logging.CRITICAL + 10)

    def _restore_stream_handlers(self) -> None:
        for h, prev_level in self._silenced_handlers:
            try:
                h.setLevel(prev_level)
            except Exception:  # noqa: BLE001 — never crash on teardown
                logger.warning("failed to restore handler level on %r", h)
        self._silenced_handlers = []

    # ----- event handlers ------------------------------------------------

    def _handle_event(self, ev: StrategyEvent) -> None:
        et = ev.type
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        provider = payload.get("llm_provider")

        # Debate routing (Plan §3.4.4): events tagged by ``runner._tag_event``
        # carry ``llm_provider`` in payload — route them to the grid, never
        # to the StageStack.
        if self._state.debate_grid is not None and isinstance(provider, str):
            self._route_to_debate_grid(ev, provider, payload)
            # LOG events still belong in the log panel — fall through after.
            if et is EventType.LOG:
                self._on_log(ev)
            return

        if et is EventType.STEP_STARTED:
            self._on_step_started(ev)
        elif et is EventType.STEP_FINISHED:
            self._on_step_finished(ev)
        elif et is EventType.LIVE_STATUS:
            self._on_live_status(ev)
        elif et is EventType.LLM_BATCH_STARTED:
            # The matching LIVE_STATUS covers detail; avoid double-update.
            return
        elif et is EventType.LLM_BATCH_FINISHED:
            self._on_batch_finished()
        elif et is EventType.LLM_FINAL_RANK:
            self._on_final_rank(ev)
        elif et is EventType.VALIDATION_FAILED:
            self._on_validation_failed(ev)
        elif et is EventType.RESULT_PERSISTED:
            self._on_result_persisted(ev)
        elif et is EventType.LOG:
            self._on_log(ev)
        elif et is EventType.TUSHARE_FALLBACK:
            self._on_tushare_fallback(ev)
        elif et is EventType.TUSHARE_UNAUTH:
            self._on_tushare_unauth(ev)
        # TUSHARE_CALL / DATA_SYNC_* are intentionally not rendered in the
        # ``run`` path (Plan §3.3 — DATA_SYNC_* belongs to the ``sync`` path).

    def _on_step_started(self, ev: StrategyEvent) -> None:
        sid = parse_stage_id(ev.message)
        if sid is None:
            return
        st = self._state.stages.push_or_get(sid, title_for(sid))
        n_batches = ev.payload.get("n_batches") if isinstance(ev.payload, dict) else None
        if isinstance(n_batches, int) and n_batches > 0:
            self._state.stages.set_running(sid, total=n_batches)
        else:
            self._state.stages.set_running(sid)
        # Reset prior detail/progress when a stage restarts (defensive — the
        # current pipeline never re-enters a stage, but it's cheap insurance).
        st.progress_done = 0
        st.detail = ""

    def _on_step_finished(self, ev: StrategyEvent) -> None:
        sid = parse_stage_id(ev.message)
        if sid is None:
            return
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        # Harvest config-panel facts opportunistically from Step 0 / Step 1.
        if sid == "0":
            td = payload.get("trade_date")
            ntd = payload.get("next_trade_date")
            if isinstance(td, str):
                self._state.config.trade_date = td
            if isinstance(ntd, str):
                self._state.config.next_trade_date = ntd
        elif sid == "1":
            if "lgb_model_id" in payload:
                self._state.config.lgb_enabled = (
                    payload.get("lgb_model_id") is not None
                )
        existing = self._state.stages.get(sid)
        failed = int(payload.get("failed_batches", 0) or 0)
        partial = failed > 0 or bool(existing and existing.failed_batches)
        self._state.stages.mark_finished(sid, partial=partial)

    def _on_live_status(self, ev: StrategyEvent) -> None:
        # Debate-phase banners route to the grid rather than the StageStack.
        grid = self._state.debate_grid
        if grid is not None and "[辩论模式]" in ev.message:
            grid.banner = ev.message
            if "Phase B" in ev.message:
                grid.transition_to_phase("phase_b")
            elif "Phase A" in ev.message:
                grid.current_phase = "phase_a"
            return

        latest = self._state.stages.latest_running()
        if latest is None:
            return
        latest.detail = ev.message

    def _on_batch_finished(self) -> None:
        latest = self._state.stages.latest_running()
        if latest is None:
            return
        self._state.stages.tick_progress(latest.stage_id)

    def _on_final_rank(self, ev: StrategyEvent) -> None:
        latest = self._state.stages.latest_running()
        if latest is None:
            return
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        in_tok = payload.get("input_tokens", "?")
        out_tok = payload.get("output_tokens", "?")
        latest.detail = f"✔ 全局重排完成 (in={in_tok} / out={out_tok})"

    def _on_validation_failed(self, ev: StrategyEvent) -> None:
        # v0.6.7 — VALIDATION_FAILED is now ALSO surfaced in the log panel
        # (untruncated) so the user sees the full error text; the stage
        # stack still gets the 96-char one-liner so the progress panel
        # doesn't blow up vertically. Banner trips error_mode in
        # layout.render_dashboard, which widens the log window to 8/12 rows.
        ts = datetime.now().strftime("%H:%M:%S")
        self._state.log_lines.append((ts, "ERROR", ev.message))
        if self._state.banner is None:
            self._state.banner = f"✘ {ev.message}"
            self._state.banner_style = "status.error"

        latest = self._state.stages.latest_running()
        if latest is None:
            return
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        batch_no = payload.get("batch_no") or payload.get("batch_id") or "?"
        label = f"批 {batch_no}: {ev.message}"
        if len(label) > 96:
            label = label[:93] + "…"
        self._state.stages.append_failed_batch(latest.stage_id, label)

    def _on_result_persisted(self, ev: StrategyEvent) -> None:
        # Synthesise Step 5 even though pipeline never emits a STEP_STARTED
        # for it — the report-writing happens inside ``_iter_pipeline`` after
        # all other steps complete, and the user wants to see the row land.
        st = self._state.stages.push_or_get("5", title_for("5"))
        self._state.stages.set_running("5")
        st.detail = ev.message
        self._state.stages.mark_finished("5")
        # In debate mode, RESULT_PERSISTED marks the end of phase B (and the
        # whole run). Any provider still in RUNNING transitions to SUCCESS.
        if self._state.debate_grid is not None:
            self._state.debate_grid.transition_to_phase("done")

    def _route_to_debate_grid(
        self,
        ev: StrategyEvent,
        provider: str,
        payload: dict[str, object],
    ) -> None:
        grid = self._state.debate_grid
        if grid is None:
            return
        row = grid.row_for(provider)
        phase = str(payload.get("debate_phase") or "phase_a")

        # First-event-of-phase transition: WAITING → RUNNING.
        if phase == "phase_a" and row.phase_a_status == StageStatus.WAITING:
            row.phase_a_status = StageStatus.RUNNING
        elif phase == "phase_b" and row.phase_b_status == StageStatus.WAITING:
            row.phase_b_status = StageStatus.RUNNING

        et = ev.type
        # Worker-failed sentinel is emitted by ``runner.result_events`` as a
        # LOG ERROR with message ``[provider] worker failed: ...`` when the
        # phase-A future raised. (Phase B has a similar fallback path.)
        if et is EventType.LOG and ev.level == EventLevel.ERROR and "worker failed" in ev.message:
            if phase == "phase_a":
                row.phase_a_status = StageStatus.FAILED
            else:
                row.phase_b_status = StageStatus.FAILED
            # Strip the `[provider] worker failed: ` prefix for the cell.
            note = ev.message.split("worker failed:", 1)[-1].strip()
            row.note = note or row.note
            return

        if et is EventType.VALIDATION_FAILED:
            row.note = f"validation: {ev.message[:60]}"
            return

        if et is EventType.STEP_FINISHED:
            stage_id = parse_stage_id(self._strip_provider_prefix(ev.message))
            if stage_id == "2":
                selected = payload.get("selected")
                if isinstance(selected, int):
                    row.screening_count = selected
                failed = int(payload.get("failed_batches", 0) or 0)
                if failed > 0:
                    row.phase_a_status = StageStatus.PARTIAL
            elif stage_id == "4":
                preds = payload.get("predictions")
                if isinstance(preds, int):
                    row.prediction_count = preds
                failed = int(payload.get("failed_batches", 0) or 0)
                if failed > 0 and row.phase_a_status != StageStatus.FAILED:
                    row.phase_a_status = StageStatus.PARTIAL
                # End of phase A for this provider in the no-multi-batch case.
                if row.phase_a_status == StageStatus.RUNNING:
                    row.phase_a_status = StageStatus.SUCCESS
            elif stage_id == "4.5":
                success = bool(payload.get("success", False))
                if not success and row.phase_a_status != StageStatus.FAILED:
                    row.phase_a_status = StageStatus.PARTIAL
                if row.phase_a_status == StageStatus.RUNNING:
                    row.phase_a_status = StageStatus.SUCCESS
            elif stage_id == "4.7":
                success = bool(payload.get("success", False))
                revised = payload.get("revised")
                if isinstance(revised, int):
                    row.revised_count = revised
                if success:
                    row.phase_b_status = StageStatus.SUCCESS
                else:
                    row.phase_b_status = StageStatus.FAILED
                    reason = payload.get("reason")
                    if isinstance(reason, str) and reason:
                        row.note = reason[:60]

    @staticmethod
    def _strip_provider_prefix(message: str) -> str:
        """``[deepseek] Step 2: ...`` → ``Step 2: ...``."""
        if message.startswith("["):
            close = message.find("]")
            if close > 0:
                return message[close + 1 :].lstrip()
        return message

    def _on_log(self, ev: StrategyEvent) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        if ev.level == EventLevel.WARN:
            level = "WARN"
        elif ev.level == EventLevel.ERROR:
            level = "ERROR"
        else:
            level = "INFO"
        self._state.log_lines.append((ts, level, ev.message))
        # Mine the settings LOG payload for the config panel (Plan §3.3 ⊕
        # _settings_log_event in runner.py:72).
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        cfg = self._state.config
        for key in ("min_float_mv_yi", "max_float_mv_yi", "max_close_yuan"):
            if key in payload:
                setattr(cfg, key, payload[key])
        # Seed the debate grid from the "[辩论模式] 启用，参与 LLM = ..." LOG
        # (runner.py:556 attaches the providers list in payload).
        grid = self._state.debate_grid
        if grid is not None and not grid.rows:
            providers = payload.get("providers")
            if isinstance(providers, list):
                grid.seed([str(p) for p in providers])
        if ev.level == EventLevel.ERROR:
            # Show the worst error as a banner; later errors overwrite it.
            self._state.banner = f"✘ {ev.message}"
            self._state.banner_style = "status.error"

    def _on_tushare_fallback(self, ev: StrategyEvent) -> None:
        self._state.config.tushare_fallback_count += 1
        ts = datetime.now().strftime("%H:%M:%S")
        self._state.log_lines.append((ts, "WARN", f"⚠ {ev.message}"))

    def _on_tushare_unauth(self, ev: StrategyEvent) -> None:
        self._state.banner = f"✘ Tushare 未授权: {ev.message}"
        self._state.banner_style = "status.error"
        ts = datetime.now().strftime("%H:%M:%S")
        self._state.log_lines.append((ts, "ERROR", ev.message))

    # ----- frame rendering ---------------------------------------------

    def _render_frame(self):  # type: ignore[no-untyped-def]
        # ``shutil.get_terminal_size`` can raise OSError under some shells;
        # Plan §4.4 says: assume 80 cols.
        try:
            cols = shutil.get_terminal_size((80, 24)).columns
        except OSError:
            cols = 80
        return render_dashboard(self._state, width=cols)

    def _safe_update(self) -> None:
        if self._live is None:
            return
        try:
            self._live.update(self._render_frame())
        except Exception as e:  # noqa: BLE001 — never crash the runner
            logger.warning("Live.update raised: %s", e)

    def _finalise_state(self, outcome: RunOutcome) -> None:
        # Resolve trade dates from Step 0 STEP_FINISHED payload if present.
        st0 = self._state.stages.get("0")
        if st0 is None:
            return
        # No-op for now — the existing handlers update state in-flight. We
        # only need to make sure CANCELLED / FAILED shows a banner.
        from deeptrade.core.run_status import RunStatus

        if outcome.status == RunStatus.CANCELLED:
            self._state.banner = "⏹ CANCELLED — 用户中断"
            self._state.banner_style = "status.error"
            latest = self._state.stages.latest_running()
            if latest is not None:
                self._state.stages.mark_failed(latest.stage_id)
        elif outcome.status == RunStatus.FAILED:
            self._state.banner = f"✘ 运行失败: {outcome.error or '未知'}"
            self._state.banner_style = "status.error"
            latest = self._state.stages.latest_running()
            if latest is not None:
                self._state.stages.mark_failed(latest.stage_id)


__all__ = ["RichDashboardRenderer"]
