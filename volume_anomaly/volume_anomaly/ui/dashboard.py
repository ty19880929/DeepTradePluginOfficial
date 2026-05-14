"""Rich-based dashboard renderer for the ``volume-anomaly`` runner.

This is the user-facing renderer when the terminal supports it (Plan §3.5
fallback matrix → :func:`volume_anomaly.ui.choose_renderer`). It owns a
single :class:`rich.live.Live` region and redraws the full frame on every
incoming event. Refresh is throttled to 8 fps by ``Live`` to keep flicker
out of high-event-rate phases.

Event → state-mutation logic lives here (Plan §3.2.3); the actual
*rendering* is delegated to :mod:`volume_anomaly.ui.layout`. The dashboard
must **never raise** from ``on_event`` (Plan §3.6.1); we wrap every
handler in ``try / except`` and log warnings instead.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

from deeptrade.plugins_api.events import EventLevel, EventType
from deeptrade.theme import EVA_THEME
from rich.console import Console
from rich.live import Live

from .funnel import FunnelSummary
from .layout import DashboardState, render_dashboard
from .mapping import parse_stage_id, title_for
from .stage_model import StageStatus

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

    from volume_anomaly.runner import RunOutcome
    from volume_anomaly.ui.protocol import RunParams


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


# Funnel field keys the runner attaches to ``DATA_SYNC_FINISHED.payload``
# (runner.py:209 — _iter_screen). Centralised here so the dashboard and
# the runner can't accidentally drift.
_FUNNEL_KEYS = (
    "n_main_board",
    "n_after_st_susp",
    "n_after_t_day_rules",
    "n_after_turnover",
    "n_after_vol_rules",
)

# Config-panel keys the runner emits via ``_va_settings_log_event`` (LOG with
# payload). Anything matching one of these names gets ``setattr``-ed onto
# :class:`ConfigSummary` opportunistically (see :meth:`_on_log`). Names
# mirror ``ScreenRules`` / ``AppConfig.app_profile`` field names exactly.
_CONFIG_KEYS = (
    "profile",
    "main_board_only",
    "pct_chg_min",
    "pct_chg_max",
    "turnover_min",
    "turnover_max",
    "vol_ratio_5d_min",
    "lgb_enabled",
)


class RichDashboardRenderer:
    """Renderer that paints a rich.Live dashboard frame per event.

    Implements the :class:`EventRenderer` Protocol; see ``protocol.py`` for
    the lifecycle contract.
    """

    def __init__(self, *, no_color: bool = False) -> None:
        # Console: stderr is intentionally left alone (the framework's logger
        # writes there). The dashboard draws to stdout so it composes with
        # the post-run ``status:`` line printed by cli.py.
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

    # ----- lifecycle ----------------------------------------------------

    def on_run_start(
        self, *, run_id: str, mode: str, params: RunParams
    ) -> None:
        self._state.run_id = run_id
        self._state.started_at = datetime.now()
        self._state.mode = mode
        # Funnel is screen-only (Plan §4.2). Instantiating it elsewhere
        # would force the layout to render an empty "(等待异动筛选完成…)"
        # placeholder during analyze runs — confusing and ugly.
        if mode == "screen":
            self._state.funnel = FunnelSummary()
        self._live = Live(
            self._render_frame(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
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
            except Exception as e:  # noqa: BLE001 — best-effort restore
                logger.warning("Live.__exit__ raised: %s", e)
            self._live = None

    # ----- event handlers ----------------------------------------------

    def _handle_event(self, ev: StrategyEvent) -> None:
        et = ev.type

        if et is EventType.STEP_STARTED:
            self._on_step_started(ev)
        elif et is EventType.STEP_FINISHED:
            self._on_step_finished(ev)
        elif (
            et is EventType.DATA_SYNC_STARTED
            and self._state.mode == "screen"
        ):
            # Plan §3.2.2: screen's anomaly funnel uses DATA_SYNC events but
            # carries the "Step 1: ..." message prefix — handle as a stage
            # event so it shows up under "执行进度".
            self._on_step_started(ev)
        elif (
            et is EventType.DATA_SYNC_FINISHED
            and self._state.mode == "screen"
        ):
            self._on_screen_funnel(ev)
            self._on_data_sync_finished(ev)
        elif et is EventType.LIVE_STATUS:
            self._on_live_status(ev)
        elif et is EventType.LLM_BATCH_STARTED:
            # The matching LIVE_STATUS covers detail; avoid double-update.
            return
        elif et is EventType.LLM_BATCH_FINISHED:
            self._on_batch_finished()
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
        # TUSHARE_CALL is intentionally ignored (noise, Plan §3.3).

    def _on_step_started(self, ev: StrategyEvent) -> None:
        sid = parse_stage_id(ev.message)
        if sid is None:
            return
        st = self._state.stages.push_or_get(
            sid, title_for(sid, self._state.mode)
        )
        n_batches = (
            ev.payload.get("n_batches")
            if isinstance(ev.payload, dict)
            else None
        )
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
        existing = self._state.stages.get(sid)
        failed = int(payload.get("failed_batches", 0) or 0)
        partial = failed > 0 or bool(existing and existing.failed_batches)
        self._state.stages.mark_finished(sid, partial=partial)

    def _on_data_sync_finished(self, ev: StrategyEvent) -> None:
        """Close the current screen-stage on DATA_SYNC_FINISHED.

        Plan §3.2.2 — the runner emits DATA_SYNC_FINISHED with the funnel
        summary message (``"funnel: 3210 → ..."``) which has *no* "Step X:"
        prefix. Falling back to ``latest_running`` keeps the stage closing
        even when stage_id can't be parsed. The "funnel: ..." text is still
        useful as the closing detail line.
        """
        sid = parse_stage_id(ev.message)
        if sid is None:
            latest = self._state.stages.latest_running()
            if latest is None:
                return
            sid = latest.stage_id
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        existing = self._state.stages.get(sid)
        failed = int(payload.get("failed_batches", 0) or 0)
        partial = failed > 0 or bool(existing and existing.failed_batches)
        self._state.stages.mark_finished(sid, partial=partial)

    def _on_screen_funnel(self, ev: StrategyEvent) -> None:
        """Pull the 5 funnel counts off DATA_SYNC_FINISHED.payload."""
        if self._state.funnel is None:
            return
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        for key in _FUNNEL_KEYS:
            if key in payload:
                try:
                    setattr(self._state.funnel, key, int(payload[key]))
                except (TypeError, ValueError):
                    # Garbage payload — leave the field as-is (None).
                    continue

    def _on_live_status(self, ev: StrategyEvent) -> None:
        latest = self._state.stages.latest_running()
        if latest is None:
            return
        latest.detail = ev.message

    def _on_batch_finished(self) -> None:
        latest = self._state.stages.latest_running()
        if latest is None:
            return
        self._state.stages.tick_progress(latest.stage_id)

    def _on_validation_failed(self, ev: StrategyEvent) -> None:
        latest = self._state.stages.latest_running()
        if latest is None:
            return
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        batch_no = (
            payload.get("batch_no") or payload.get("batch_id") or "?"
        )
        label = f"批 {batch_no}: {ev.message}"
        # Trim long messages to keep one screen line.
        if len(label) > 96:
            label = label[:93] + "…"
        self._state.stages.append_failed_batch(latest.stage_id, label)

    def _on_result_persisted(self, ev: StrategyEvent) -> None:
        # Synthesise Step 5 even though pipeline never emits a STEP_STARTED
        # for it — the report-writing happens inside ``_iter_*`` after all
        # other steps complete, and the user wants to see the row land.
        title = title_for("5", self._state.mode)
        st = self._state.stages.push_or_get("5", title)
        self._state.stages.set_running("5")
        st.detail = ev.message
        self._state.stages.mark_finished("5")

    def _on_log(self, ev: StrategyEvent) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        if ev.level == EventLevel.WARN:
            level = "WARN"
        elif ev.level == EventLevel.ERROR:
            level = "ERROR"
        else:
            level = "INFO"
        self._state.log_lines.append((ts, level, ev.message))
        # Mine the settings LOG payload for the config panel (Plan §3.3
        # ⊕ _va_settings_log_event in runner.py). Keys are listed in
        # _CONFIG_KEYS up top — anything else is ignored.
        payload = ev.payload if isinstance(ev.payload, dict) else {}
        cfg = self._state.config
        for key in _CONFIG_KEYS:
            if key in payload:
                setattr(cfg, key, payload[key])
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
        from deeptrade.core.run_status import RunStatus

        if outcome.status == RunStatus.CANCELLED:
            self._state.banner = "⏹ CANCELLED — 用户中断"
            self._state.banner_style = "status.error"
            latest = self._state.stages.latest_running()
            if latest is not None:
                self._state.stages.mark_failed(latest.stage_id)
        elif outcome.status == RunStatus.FAILED:
            self._state.banner = (
                f"✘ 运行失败: {outcome.error or '未知'}"
            )
            self._state.banner_style = "status.error"
            latest = self._state.stages.latest_running()
            if latest is not None:
                self._state.stages.mark_failed(latest.stage_id)


__all__ = ["RichDashboardRenderer"]
