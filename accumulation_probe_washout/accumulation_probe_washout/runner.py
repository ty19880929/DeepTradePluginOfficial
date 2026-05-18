"""Plugin-internal run lifecycle for accumulation-probe-washout.

Modes (v0.1):
    screen   — local rules only (no LLM); writes apw_signal_history + apw_watchlist
    analyze  — read watchlist → LLM batches → apw_stage_results (M3)
    run      — screen → analyze串联 (M4)
    evaluate — T+N realised returns into apw_realized_returns (M5)

All modes upsert one ``apw_runs`` row + a stream of ``apw_events``.
"""

from __future__ import annotations

import json
import logging
import signal
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any

import pandas as pd

from deeptrade.core.run_status import RunStatus
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from .calendar import TradeCalendar
from .cancellation import cancel_requested
from .config import ApwConfig, ApwConfigStore
from .data import (
    compute_accumulation,
    compute_launch_setup,
    compute_returns_and_labels,
    compute_washout,
    derive_phase,
    detect_probe_day,
    fetch_daily,
    fetch_daily_basic,
    fetch_index_daily,
    fetch_moneyflow,
    fetch_realized_prices,
    fetch_stock_basic,
    fetch_st_codes,
    fetch_suspended_codes,
    fetch_trade_cal,
    filter_main_board,
    filter_st_and_suspend,
    pack_candidate,
    resolve_trade_date,
)
from .runtime import ApwRuntime, build_tushare_client
from .schemas import APWPhase
from .ui.protocol import EventRenderer, NullRenderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScreenParams:
    trade_date: str | None = None
    allow_intraday: bool = False
    force_sync: bool = False
    max_candidates: int | None = None


@dataclass
class AnalyzeParams:
    trade_date: str | None = None
    allow_intraday: bool = False
    max_candidates: int | None = None
    llm_provider: str | None = None  # --llm <provider>
    prediction_filter: str | None = None  # e.g. "launch_ready"


@dataclass
class RunParams:
    """run = screen → analyze."""
    trade_date: str | None = None
    allow_intraday: bool = False
    force_sync: bool = False
    max_candidates: int | None = None
    llm_provider: str | None = None


@dataclass
class EvaluateParams:
    from_date: str | None = None
    to_date: str | None = None
    horizons: str = "1,3,5,10"
    include_early_phases: bool = False
    force_recompute: bool = False


@dataclass
class RunOutcome:
    run_id: str
    status: RunStatus
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ApwRunner:
    """Encapsulates the screen / analyze / run / evaluate lifecycle."""

    def __init__(
        self,
        rt: ApwRuntime,
        *,
        renderer: EventRenderer | None = None,
    ) -> None:
        self.rt = rt
        self.renderer = renderer or NullRenderer()
        self._seq = 0

    # ---- event plumbing ----

    def _emit(
        self,
        event_type: EventType,
        message: str,
        *,
        level: EventLevel = EventLevel.INFO,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ev = self.rt.emit(event_type, message, level=level, payload=payload or {})
        self._write_event(ev)
        self._dispatch_to_renderer(ev)

    def _write_event(self, ev: StrategyEvent) -> None:
        if self.rt.run_id is None:
            return
        self._seq += 1
        try:
            self.rt.db.execute(
                """
                INSERT INTO apw_events
                (run_id, seq, event_time, level, event_type, message, payload_json)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
                """,
                [
                    self.rt.run_id,
                    self._seq,
                    ev.level.value,
                    ev.type.value,
                    ev.message,
                    json.dumps(ev.payload or {}, ensure_ascii=False),
                ],
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed to persist event %s", ev.type)

    def _dispatch_to_renderer(self, ev: StrategyEvent) -> None:
        try:
            self.renderer.on_event(ev)
        except Exception:  # noqa: BLE001
            logger.exception("renderer crashed on event %s; degrading to null", ev.type)
            self.renderer = NullRenderer()

    # ---- run lifecycle ----

    def _start_run(self, mode: str, trade_date: str, params: Any) -> str:
        run_id = str(uuid.uuid4())
        self.rt.run_id = run_id
        now = datetime.now(timezone.utc)
        self.rt.db.execute(
            """
            INSERT INTO apw_runs
            (run_id, mode, trade_date, status, is_intraday, started_at, params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                mode,
                trade_date,
                RunStatus.RUNNING.value,
                self.rt.is_intraday,
                now,
                json.dumps(_dc_to_dict(params), ensure_ascii=False),
            ],
        )
        try:
            self.renderer.on_run_started(
                mode=mode, trade_date=trade_date, run_id=run_id, params=_dc_to_dict(params)
            )
        except Exception:  # noqa: BLE001
            logger.exception("renderer crashed on run_started; degrading to null")
            self.renderer = NullRenderer()
        return run_id

    def _finish_run(
        self,
        status: RunStatus,
        *,
        error: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        if self.rt.run_id is None:
            return
        now = datetime.now(timezone.utc)
        self.rt.db.execute(
            """
            UPDATE apw_runs
            SET status = ?, finished_at = ?, summary_json = ?, error = ?
            WHERE run_id = ?
            """,
            [
                status.value,
                now,
                json.dumps(summary or {}, ensure_ascii=False),
                error,
                self.rt.run_id,
            ],
        )
        try:
            self.renderer.on_run_finished(status=status, error=error, summary=summary or {})
        except Exception:  # noqa: BLE001
            logger.exception("renderer crashed on run_finished; ignoring")

    # ---- cancel surfacing ---------------------------------------------------

    def _emit_cancelled_log(self) -> None:
        """Push two short LOG events explaining the user cancel.

        Replaces the line-by-line traceback fan-out for cancel-class
        outcomes. WARN level (not ERROR) — "user intent, not failure".
        """
        for msg in ("用户手动中断运行，正在停止当前任务", "运行已取消"):
            try:
                self._emit(EventType.LOG, msg, level=EventLevel.WARN)
            except Exception:  # noqa: BLE001
                pass

    def _shielded_finish_run(
        self,
        status: RunStatus,
        *,
        error: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Run :meth:`_finish_run` with SIGINT temporarily ignored.

        Without this, a user mashing Ctrl+C during the finally window can
        skip the ``UPDATE apw_runs SET status = ...`` write and strand the
        row at ``RunStatus.RUNNING``. Off main thread / embedded falls
        through unshielded; the prior behaviour was the same.
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
            self._finish_run(status, error=error, summary=summary)
        finally:
            if installed and prev is not None:
                try:
                    signal.signal(signal.SIGINT, prev)
                except (ValueError, OSError):
                    pass

    def _make_cancel_outcome(
        self, run_id: str, mode: str, _owns_run: bool
    ) -> "RunOutcome":
        """Build a CANCELLED RunOutcome and persist the run row (shielded).

        Used by every ``except KeyboardInterrupt`` branch in this runner.
        Mirrors the same call shape as the original failure branches so
        callers don't need to change return semantics.
        """
        logger.info("apw %s %s cancelled by user", mode, run_id)
        self._emit_cancelled_log()
        if _owns_run:
            self._shielded_finish_run(RunStatus.CANCELLED, error="用户手动中断")
        return RunOutcome(
            run_id=run_id, status=RunStatus.CANCELLED, error="用户手动中断"
        )

    # ---- screen (M2) ----

    def execute_screen(self, params: ScreenParams, *, _owns_run: bool = True) -> RunOutcome:
        cfg_store = ApwConfigStore(self.rt.db)
        cfg = cfg_store.load()
        max_candidates = params.max_candidates or cfg.max_llm_candidates

        # ---- trade date resolution
        if self.rt.tushare is None:
            self.rt.tushare = build_tushare_client(
                self.rt, intraday=params.allow_intraday
            )

        tushare = self.rt.tushare
        now_utc = datetime.now(timezone.utc)
        today_iso = now_utc.strftime("%Y%m%d")
        # Pull a wide trade_cal window — last 24 months back, +90 calendar days
        # forward so that next_open(T) succeeds even when T == latest trade day.
        cal_start = (now_utc.replace(day=1)
                     .replace(year=now_utc.year - 2)).strftime("%Y%m%d")
        cal_end = (now_utc + timedelta(days=90)).strftime("%Y%m%d")
        cal_df = fetch_trade_cal(tushare, start=cal_start, end=cal_end)
        calendar = TradeCalendar(cal_df)

        T, _ = resolve_trade_date(
            datetime.now(),
            calendar,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
        )
        self.rt.is_intraday = params.allow_intraday and calendar.is_open(T) and (
            datetime.now().time() < time(18, 0)
        )

        if _owns_run:
            run_id = self._start_run("screen", T, params)
        else:
            # Parent (execute_run) already opened a 'run' row; reuse it so the
            # whole screen + analyze flow shares one run_id and one event seq.
            assert self.rt.run_id is not None, (
                "_owns_run=False requires the caller to have started a run"
            )
            run_id = self.rt.run_id

        try:
            self._emit(EventType.DATA_SYNC_STARTED, f"开始数据同步 (T={T})")

            # ---- Step 0: universe sync
            stock_basic = fetch_stock_basic(tushare)
            n_total = len(stock_basic)
            main_board = filter_main_board(stock_basic, cfg, trade_date=T)
            n_main_board = len(main_board)

            st = fetch_st_codes(tushare, trade_date=T)
            sus = fetch_suspended_codes(tushare, trade_date=T)
            uni = filter_st_and_suspend(main_board, st, sus)
            n_after_st_susp = len(uni)

            # ---- daily quotes for the whole universe — base lookback window
            # We need cfg.base_lookback_trade_days of trade days ending at T.
            window_start = _date_n_calendar_days_before(T, cfg.base_lookback_trade_days * 2 + 60)
            quotes = _fetch_daily_in_batches(
                tushare, ts_codes=uni["ts_code"].tolist(), start=window_start, end=T,
                batch_size=50,
            )
            basic_extra = _fetch_daily_basic_in_batches(
                tushare, ts_codes=uni["ts_code"].tolist(), start=window_start, end=T,
                batch_size=50,
            )
            mf_outcome = fetch_moneyflow(
                tushare, ts_codes=uni["ts_code"].tolist(), start=window_start, end=T
            )

            # ---- index_daily for relative_strength_20d (cfg.baseline_index_code).
            # Empty string disables the baseline (rs20 falls back to None +
            # missing_data). Tushare server / auth errors land in
            # outcome.missing without breaking the screen.
            index_df = pd.DataFrame()
            index_missing: list[str] = []
            if cfg.baseline_index_code:
                index_outcome = fetch_index_daily(
                    tushare,
                    index_code=cfg.baseline_index_code,
                    start=window_start,
                    end=T,
                )
                index_df = index_outcome.df
                index_missing = list(index_outcome.missing)
                if index_df.empty and not index_missing:
                    # API succeeded but returned nothing — surface explicitly.
                    index_missing = ["index_daily"]
            else:
                index_missing = ["index_daily"]

            # ---- Liquidity filter on T-day amount
            day_amount = _amount_on_date(quotes, T)
            liquidity_mask = uni["ts_code"].map(
                lambda c: day_amount.get(c, 0.0) >= cfg.min_amount_yi
            )
            after_liquidity = uni[liquidity_mask].reset_index(drop=True)
            n_after_liquidity = len(after_liquidity)

            self._emit(
                EventType.DATA_SYNC_FINISHED,
                f"数据同步完成: 主板池={n_main_board}, ST/停牌后={n_after_st_susp}, 流动性后={n_after_liquidity}",
                payload={
                    "n_total": int(n_total),
                    "n_main_board": int(n_main_board),
                    "n_after_st_susp": int(n_after_st_susp),
                    "n_after_liquidity": int(n_after_liquidity),
                },
            )

            # ---- Step 1: per-stock funnel
            self._emit(EventType.STEP_STARTED, "Step 1: 漏斗筛选", payload={"step": 1})

            quotes_by_code = (
                {} if quotes.empty else dict(list(quotes.groupby("ts_code")))
            )
            basic_extra_by_code = (
                {} if basic_extra.empty else dict(list(basic_extra.groupby("ts_code")))
            )
            mf_by_code = (
                {}
                if mf_outcome.df.empty
                else dict(list(mf_outcome.df.groupby("ts_code")))
            )

            n_after_accumulation = 0
            n_after_probe = 0
            n_after_washout = 0
            n_after_launch_ready = 0

            hits: list[dict[str, Any]] = []
            for _, row in after_liquidity.iterrows():
                code = row["ts_code"]
                qdf = quotes_by_code.get(code, pd.DataFrame())
                if qdf.empty:
                    continue
                qdf = _normalize_quotes(qdf, basic_extra_by_code.get(code, pd.DataFrame()))
                mfd = mf_by_code.get(code, pd.DataFrame())

                acc = compute_accumulation(qdf, mfd, cfg)
                if acc["accumulation_score"] >= cfg.accumulation_score_min:
                    n_after_accumulation += 1

                probe = detect_probe_day(qdf, cfg, mfd)
                if probe is not None and probe.get("probe_quality_score", 0.0) >= cfg.probe_quality_score_min:
                    n_after_probe += 1

                wash = compute_washout(qdf, mfd, probe, cfg)
                if (
                    probe is not None
                    and wash["washout_score"] >= cfg.washout_score_min
                    and not wash["post_probe_low_broken"]
                    and cfg.washout_min_trade_days
                    <= wash["washout_days"]
                    <= cfg.washout_max_trade_days
                ):
                    n_after_washout += 1

                launch = compute_launch_setup(
                    qdf, probe, wash, cfg,
                    index_df=index_df if not index_df.empty else None,
                )
                phase = derive_phase(acc, probe, wash, launch, cfg)
                if phase == APWPhase.LAUNCH_READY:
                    n_after_launch_ready += 1

                if phase == APWPhase.NO_SETUP:
                    continue

                last_row = qdf.iloc[-1]
                basic = {
                    "listed_days": _listed_days(row, T),
                    "close": float(last_row["close"]),
                    "pct_chg": float(last_row.get("pct_chg", 0.0) or 0.0),
                    "turnover_rate": float(last_row.get("turnover_rate", 0.0) or 0.0),
                    "amount_yi": float(last_row.get("amount", 0.0) or 0.0) / 100000.0,
                    "circ_mv_yi": float(last_row.get("circ_mv", 0.0) or 0.0) / 10000.0,
                }
                missing = list(set(
                    acc.get("missing_data", [])
                    + (mf_outcome.missing or [])
                    + index_missing
                ))
                cand = pack_candidate(
                    trade_date=T,
                    ts_code=code,
                    name=str(row.get("name", "")),
                    phase=phase,
                    basic=basic,
                    accumulation=acc,
                    probe=probe,
                    washout=wash,
                    launch=launch,
                    missing_data=missing,
                )
                hits.append(cand)

            self._emit(
                EventType.STEP_FINISHED,
                f"漏斗筛选完成: 命中 {len(hits)} 只",
                payload={
                    "step": 1,
                    "n_after_accumulation": n_after_accumulation,
                    "n_after_probe": n_after_probe,
                    "n_after_washout": n_after_washout,
                    "n_after_launch_ready": n_after_launch_ready,
                },
            )

            # ---- write hits — apw_signal_history (all non-no_setup; D3)
            self._emit(EventType.STEP_STARTED, "Step 2: 持久化命中", payload={"step": 2})
            persisted_history = 0
            persisted_watch = 0
            for cand in hits:
                self._upsert_signal_history(cand)
                persisted_history += 1
                if cand["phase"] in {
                    APWPhase.WASHING_AFTER_PROBE.value,
                    APWPhase.LAUNCH_READY.value,
                }:
                    self._upsert_watchlist(cand)
                    persisted_watch += 1

            self._emit(
                EventType.STEP_FINISHED,
                f"持久化完成: history={persisted_history} watchlist={persisted_watch}",
                payload={
                    "step": 2,
                    "n_signal_history": persisted_history,
                    "n_watchlist": persisted_watch,
                },
            )

            summary = {
                "n_total": int(n_total),
                "n_main_board": int(n_main_board),
                "n_after_st_susp": int(n_after_st_susp),
                "n_after_liquidity": int(n_after_liquidity),
                "n_after_accumulation": int(n_after_accumulation),
                "n_after_probe": int(n_after_probe),
                "n_after_washout": int(n_after_washout),
                "n_after_launch_ready": int(n_after_launch_ready),
                "n_signal_history": int(persisted_history),
                "n_watchlist": int(persisted_watch),
            }
            if _owns_run:
                self._finish_run(RunStatus.SUCCESS, summary=summary)
            return RunOutcome(run_id=run_id, status=RunStatus.SUCCESS, summary=summary)
        except KeyboardInterrupt:
            return self._make_cancel_outcome(run_id, "screen", _owns_run)
        except Exception as exc:  # noqa: BLE001
            if cancel_requested():
                logger.info(
                    "apw screen %s cancelled (derived %s: %s)",
                    run_id, type(exc).__name__, exc,
                )
                return self._make_cancel_outcome(run_id, "screen", _owns_run)
            tb = traceback.format_exc()
            self._emit(EventType.LOG, f"run failed: {exc}", level=EventLevel.ERROR,
                       payload={"traceback": tb})
            if _owns_run:
                self._shielded_finish_run(RunStatus.FAILED, error=str(exc))
            return RunOutcome(run_id=run_id, status=RunStatus.FAILED, error=str(exc))

    # ---- evaluate (M5) ----

    def execute_evaluate(self, params: EvaluateParams) -> RunOutcome:
        from .render import render_evaluate_report

        cfg_store = ApwConfigStore(self.rt.db)
        cfg = cfg_store.load()
        horizons = [int(x) for x in params.horizons.split(",") if x.strip()]
        max_h = max(horizons) if horizons else 10

        if self.rt.tushare is None:
            self.rt.tushare = build_tushare_client(self.rt, intraday=False)
        tushare = self.rt.tushare

        # Pick the trade_date range — D3 default keeps only meaningful phases.
        phase_filter = (
            None if params.include_early_phases
            else ("washing_after_probe", "launch_ready")
        )

        # ---- gather all signal_date/ts_code pairs to evaluate
        if params.from_date and params.to_date:
            sql = (
                "SELECT trade_date, ts_code, phase, "
                "  json_extract_string(raw_candidate_json, '$.launch_setup_score') AS launch_score "
                "FROM apw_signal_history "
                "WHERE trade_date BETWEEN ? AND ?"
            )
            bind: list[Any] = [params.from_date, params.to_date]
        else:
            sql = (
                "SELECT trade_date, ts_code, phase, "
                "  json_extract_string(raw_candidate_json, '$.launch_setup_score') AS launch_score "
                "FROM apw_signal_history"
            )
            bind = []
        if phase_filter:
            placeholders = ",".join("?" * len(phase_filter))
            sql += f" AND phase IN ({placeholders})" if bind else f" WHERE phase IN ({placeholders})"
            bind.extend(phase_filter)

        rows = self.rt.db.fetchall(sql, bind) or []

        run_id = self._start_run("evaluate", params.from_date or "all", params)

        try:
            self._emit(
                EventType.STEP_STARTED,
                f"Step 1: 收集候选 ({len(rows)} 条)",
                payload={"step": 1, "n_candidates": len(rows)},
            )

            # Optionally join LLM predictions to enrich rows
            pred_lookup: dict[tuple[str, str], dict[str, Any]] = {}
            for r in self.rt.db.fetchall(
                "SELECT trade_date, ts_code, prediction, launch_score FROM apw_stage_results"
            ) or []:
                pred_lookup[(r[0], r[1])] = {"prediction": r[2], "launch_score": r[3]}

            self._emit(
                EventType.STEP_FINISHED,
                f"Step 1 完成 — 已加载 LLM 预测 {len(pred_lookup)} 条",
                payload={"step": 1, "n_predictions": len(pred_lookup)},
            )

            # ---- fetch realized prices in chunks; group rows by signal_date
            self._emit(EventType.STEP_STARTED, "Step 2: 拉取 T..T+N 行情", payload={"step": 2})
            by_date: dict[str, list[Any]] = {}
            for r in rows:
                by_date.setdefault(r[0], []).append(r)

            evaluated: list[dict[str, Any]] = []
            n_complete = 0
            n_partial = 0
            n_missing = 0

            for signal_date, batch in by_date.items():
                ts_codes = sorted({r[1] for r in batch})
                # If force_recompute=False, skip already-complete rows.
                if not params.force_recompute:
                    existing = self.rt.db.fetchall(
                        "SELECT ts_code FROM apw_realized_returns "
                        "WHERE signal_date = ? AND data_status = 'complete'",
                        [signal_date],
                    ) or []
                    done_codes = {row[0] for row in existing}
                    ts_codes = [c for c in ts_codes if c not in done_codes]
                if not ts_codes:
                    continue

                quotes = fetch_realized_prices(
                    tushare,
                    ts_codes=ts_codes,
                    signal_date=signal_date,
                    max_horizon_calendar_days=max_h * 3,
                )
                for r in batch:
                    if r[1] not in ts_codes:
                        continue
                    computed = compute_returns_and_labels(
                        ts_code=r[1],
                        signal_date=signal_date,
                        quotes=quotes,
                        label_t5_high_return_pct=cfg.label_t5_high_return_pct,
                        label_t5_max_drawdown_pct=cfg.label_t5_max_drawdown_pct,
                        label_t10_high_return_pct=cfg.label_t10_high_return_pct,
                        label_t10_max_drawdown_pct=cfg.label_t10_max_drawdown_pct,
                    )
                    # Decorate with phase / prediction / launch_score for grouping.
                    computed["phase"] = r[2]
                    pred = pred_lookup.get((r[0], r[1]), {})
                    computed["prediction"] = pred.get("prediction")
                    try:
                        computed["launch_score"] = (
                            float(pred["launch_score"]) if pred.get("launch_score") is not None
                            else (float(r[3]) if r[3] else None)
                        )
                    except (ValueError, TypeError):
                        computed["launch_score"] = None

                    self._upsert_realized_return(computed)

                    if computed["data_status"] == "complete":
                        n_complete += 1
                    elif computed["data_status"] == "partial":
                        n_partial += 1
                    else:
                        n_missing += 1
                    evaluated.append(computed)

            self._emit(
                EventType.STEP_FINISHED,
                f"Step 2 完成 — complete={n_complete} partial={n_partial} missing={n_missing}",
                payload={
                    "step": 2,
                    "n_complete": n_complete,
                    "n_partial": n_partial,
                    "n_missing": n_missing,
                },
            )

            # Re-read all rows in scope to include previously-completed ones
            sql_read = "SELECT * FROM apw_realized_returns"
            if params.from_date and params.to_date:
                sql_read += " WHERE signal_date BETWEEN ? AND ?"
                all_rows = self.rt.db.fetchall(sql_read, [params.from_date, params.to_date])
            else:
                all_rows = self.rt.db.fetchall(sql_read)
            col_names = [
                "signal_date", "ts_code", "probe_date", "prediction", "launch_score", "phase",
                "close_t", "close_t1", "close_t3", "close_t5", "close_t10",
                "ret_t1_pct", "ret_t3_pct", "ret_t5_pct", "ret_t10_pct",
                "max_high_t5_pct", "max_high_t10_pct",
                "max_drawdown_t5_pct", "max_drawdown_t10_pct",
                "label_launch_t5", "label_launch_t10", "data_status", "computed_at",
            ]
            report_rows = [dict(zip(col_names, row)) for row in (all_rows or [])]
            if phase_filter:
                report_rows = [r for r in report_rows if r.get("phase") in phase_filter]

            render_evaluate_report(
                report_rows,
                horizons=horizons,
                include_early_phases=params.include_early_phases,
            )

            summary = {
                "n_candidates": len(rows),
                "n_complete": n_complete,
                "n_partial": n_partial,
                "n_missing": n_missing,
            }
            self._finish_run(RunStatus.SUCCESS, summary=summary)
            return RunOutcome(run_id=run_id, status=RunStatus.SUCCESS, summary=summary)
        except KeyboardInterrupt:
            return self._make_cancel_outcome(run_id, "evaluate", True)
        except Exception as exc:  # noqa: BLE001
            if cancel_requested():
                logger.info(
                    "apw evaluate %s cancelled (derived %s: %s)",
                    run_id, type(exc).__name__, exc,
                )
                return self._make_cancel_outcome(run_id, "evaluate", True)
            tb = traceback.format_exc()
            self._emit(EventType.LOG, f"evaluate failed: {exc}",
                       level=EventLevel.ERROR, payload={"traceback": tb})
            self._shielded_finish_run(RunStatus.FAILED, error=str(exc))
            return RunOutcome(run_id=run_id, status=RunStatus.FAILED, error=str(exc))

    def _upsert_realized_return(self, row: dict[str, Any]) -> None:
        self.rt.db.execute(
            "DELETE FROM apw_realized_returns WHERE signal_date = ? AND ts_code = ?",
            [row["signal_date"], row["ts_code"]],
        )
        self.rt.db.execute(
            """
            INSERT INTO apw_realized_returns
            (signal_date, ts_code, probe_date, prediction, launch_score, phase,
             close_t, close_t1, close_t3, close_t5, close_t10,
             ret_t1_pct, ret_t3_pct, ret_t5_pct, ret_t10_pct,
             max_high_t5_pct, max_high_t10_pct,
             max_drawdown_t5_pct, max_drawdown_t10_pct,
             label_launch_t5, label_launch_t10, data_status)
            VALUES (?, ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?, ?,
                    ?, ?,  ?, ?,  ?, ?,  ?)
            """,
            [
                row["signal_date"], row["ts_code"], row.get("probe_date"),
                row.get("prediction"), _f(row.get("launch_score")), row.get("phase"),
                _f(row.get("close_t")), _f(row.get("close_t1")), _f(row.get("close_t3")),
                _f(row.get("close_t5")), _f(row.get("close_t10")),
                _f(row.get("ret_t1_pct")), _f(row.get("ret_t3_pct")),
                _f(row.get("ret_t5_pct")), _f(row.get("ret_t10_pct")),
                _f(row.get("max_high_t5_pct")), _f(row.get("max_high_t10_pct")),
                _f(row.get("max_drawdown_t5_pct")), _f(row.get("max_drawdown_t10_pct")),
                row.get("label_launch_t5"), row.get("label_launch_t10"),
                row.get("data_status", "missing"),
            ],
        )

    # ---- run (M4) — screen + analyze chained sharing the same run_id ----

    def execute_run(self, params: RunParams) -> RunOutcome:
        """Run = screen → analyze under a single ``mode='run'`` audit row.

        Implements detailed-design T4.7: callers see one ``run_id``, one
        monotonic event sequence in ``apw_events``, and one row in ``apw_runs``
        with ``mode='run'``. Each sub-stage emits its own events; failures
        propagate the sub-stage status to the parent run.
        """
        # ---- resolve T once so both sub-stages key off the same trade day
        if self.rt.tushare is None:
            self.rt.tushare = build_tushare_client(
                self.rt, intraday=params.allow_intraday
            )
        tushare = self.rt.tushare
        now_utc = datetime.now(timezone.utc)
        cal_start = (now_utc.replace(day=1)
                     .replace(year=now_utc.year - 2)).strftime("%Y%m%d")
        cal_end = (now_utc + timedelta(days=90)).strftime("%Y%m%d")
        cal_df = fetch_trade_cal(tushare, start=cal_start, end=cal_end)
        calendar = TradeCalendar(cal_df)
        T, _ = resolve_trade_date(
            datetime.now(),
            calendar,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
        )
        self.rt.is_intraday = params.allow_intraday and calendar.is_open(T) and (
            datetime.now().time() < time(18, 0)
        )

        run_id = self._start_run("run", T, params)

        try:
            # Pin T into the sub-params so screen/analyze resolve to the same day.
            screen_params = ScreenParams(
                trade_date=T,
                allow_intraday=params.allow_intraday,
                force_sync=params.force_sync,
                max_candidates=params.max_candidates,
            )
            screen_outcome = self.execute_screen(screen_params, _owns_run=False)
            if screen_outcome.status not in {RunStatus.SUCCESS, RunStatus.PARTIAL_FAILED}:
                summary = {"screen": screen_outcome.summary}
                self._finish_run(
                    screen_outcome.status,
                    error=screen_outcome.error,
                    summary=summary,
                )
                return RunOutcome(
                    run_id=run_id,
                    status=screen_outcome.status,
                    error=screen_outcome.error,
                    summary=summary,
                )

            analyze_params = AnalyzeParams(
                trade_date=T,
                allow_intraday=params.allow_intraday,
                max_candidates=params.max_candidates,
                llm_provider=params.llm_provider,
            )
            analyze_outcome = self.execute_analyze(analyze_params, _owns_run=False)
            summary = {
                "screen": screen_outcome.summary,
                "analyze": analyze_outcome.summary,
            }
            self._finish_run(
                analyze_outcome.status,
                error=analyze_outcome.error,
                summary=summary,
            )
            return RunOutcome(
                run_id=run_id,
                status=analyze_outcome.status,
                error=analyze_outcome.error,
                summary=summary,
            )
        except KeyboardInterrupt:
            return self._make_cancel_outcome(run_id, "run", True)
        except Exception as exc:  # noqa: BLE001
            if cancel_requested():
                logger.info(
                    "apw run %s cancelled (derived %s: %s)",
                    run_id, type(exc).__name__, exc,
                )
                return self._make_cancel_outcome(run_id, "run", True)
            tb = traceback.format_exc()
            self._emit(
                EventType.LOG,
                f"run failed: {exc}",
                level=EventLevel.ERROR,
                payload={"traceback": tb},
            )
            self._shielded_finish_run(RunStatus.FAILED, error=str(exc))
            return RunOutcome(run_id=run_id, status=RunStatus.FAILED, error=str(exc))

    # ---- analyze (M3) ----

    def execute_analyze(self, params: AnalyzeParams, *, _owns_run: bool = True) -> RunOutcome:
        from .pipeline import default_profile, run_analyze

        cfg_store = ApwConfigStore(self.rt.db)
        cfg = cfg_store.load()
        max_candidates = params.max_candidates or cfg.max_llm_candidates

        # ---- date resolution (mirrors VA — read calendar for next_trade_date)
        if self.rt.tushare is None:
            self.rt.tushare = build_tushare_client(
                self.rt, intraday=params.allow_intraday
            )
        tushare = self.rt.tushare
        now_utc = datetime.now(timezone.utc)
        cal_start = (now_utc.replace(day=1)
                     .replace(year=now_utc.year - 2)).strftime("%Y%m%d")
        cal_end = (now_utc + timedelta(days=90)).strftime("%Y%m%d")
        cal_df = fetch_trade_cal(tushare, start=cal_start, end=cal_end)
        calendar = TradeCalendar(cal_df)

        T, next_T = resolve_trade_date(
            datetime.now(),
            calendar,
            user_specified=params.trade_date,
            allow_intraday=params.allow_intraday,
        )

        if _owns_run:
            run_id = self._start_run("analyze", T, params)
        else:
            assert self.rt.run_id is not None, (
                "_owns_run=False requires the caller to have started a run"
            )
            run_id = self.rt.run_id

        try:
            # ---- read watchlist (≥ washing_after_probe by construction)
            self._emit(EventType.STEP_STARTED, "Step 1: 读取 watchlist", payload={"step": 1})
            rows = self.rt.db.fetchall(
                """
                SELECT ts_code, name, phase, raw_candidate_json
                FROM apw_watchlist
                WHERE phase IN ('washing_after_probe', 'launch_ready')
                ORDER BY launch_setup_score DESC NULLS LAST,
                         washout_score DESC NULLS LAST
                """
            )
            candidates: list[dict[str, Any]] = []
            for row in rows or []:
                try:
                    cand = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    continue
                # Apply optional prediction filter — only available pre-LLM via phase.
                if params.prediction_filter and cand.get("phase") != params.prediction_filter:
                    continue
                candidates.append(cand)

            candidates = candidates[: max_candidates]
            self._emit(
                EventType.STEP_FINISHED,
                f"Step 1: 读取 watchlist 完成，候选 {len(candidates)}",
                payload={"step": 1, "n_candidates": len(candidates)},
            )

            if not candidates:
                if _owns_run:
                    self._finish_run(
                        RunStatus.SUCCESS,
                        summary={"n_candidates": 0, "n_predictions": 0},
                    )
                return RunOutcome(
                    run_id=run_id, status=RunStatus.SUCCESS,
                    summary={"n_candidates": 0, "n_predictions": 0},
                )

            # ---- pick LLM provider (None = framework default)
            llm = self.rt.llms.get_client(
                params.llm_provider,
                plugin_id=self.rt.plugin_id,
                run_id=run_id,
            )

            # ---- run analyze pipeline
            profile = default_profile()
            terminal_result = None
            for ev, terminal in run_analyze(
                llm=llm,
                candidates=candidates,
                trade_date=T,
                next_trade_date=next_T,
                market_summary="",
                profile=profile,
                max_batch_size=cfg.llm_batch_size,
                max_repair_retries=cfg.llm_max_repair_retries,
            ):
                self._write_event(ev)
                self._dispatch_to_renderer(ev)
                if terminal is not None:
                    terminal_result = terminal

            assert terminal_result is not None  # generator always yields terminal

            # ---- persist
            n_persisted = 0
            for cand in terminal_result.predictions:
                self._upsert_stage_result(cand, run_id=run_id, trade_date=T)
                n_persisted += 1
                # Refresh watchlist row with latest LLM verdict
                self.rt.db.execute(
                    """
                    UPDATE apw_watchlist SET
                        latest_launch_score = ?,
                        latest_prediction = ?,
                        latest_confidence = ?,
                        updated_at = ?
                    WHERE ts_code = ?
                    """,
                    [
                        float(cand.launch_score),
                        cand.prediction,
                        cand.confidence,
                        datetime.now(timezone.utc),
                        cand.ts_code,
                    ],
                )

            summary = {
                "n_candidates": len(candidates),
                "n_predictions": n_persisted,
                "n_failed_batches": terminal_result.failed_batches,
                "failed_batch_ids": terminal_result.failed_batch_ids,
                "trade_date": T,
            }
            status = (
                RunStatus.PARTIAL_FAILED
                if terminal_result.failed_batches > 0 and terminal_result.success_batches > 0
                else (RunStatus.FAILED if terminal_result.success_batches == 0 else RunStatus.SUCCESS)
            )
            if _owns_run:
                self._finish_run(status, summary=summary)
            return RunOutcome(run_id=run_id, status=status, summary=summary)
        except KeyboardInterrupt:
            return self._make_cancel_outcome(run_id, "analyze", _owns_run)
        except Exception as exc:  # noqa: BLE001
            if cancel_requested():
                logger.info(
                    "apw analyze %s cancelled (derived %s: %s)",
                    run_id, type(exc).__name__, exc,
                )
                return self._make_cancel_outcome(run_id, "analyze", _owns_run)
            tb = traceback.format_exc()
            self._emit(EventType.LOG, f"analyze failed: {exc}",
                       level=EventLevel.ERROR, payload={"traceback": tb})
            if _owns_run:
                self._shielded_finish_run(RunStatus.FAILED, error=str(exc))
            return RunOutcome(run_id=run_id, status=RunStatus.FAILED, error=str(exc))

    def _upsert_stage_result(
        self, cand: Any, *, run_id: str, trade_date: str
    ) -> None:
        now = datetime.now(timezone.utc)
        self.rt.db.execute(
            "DELETE FROM apw_stage_results WHERE run_id = ? AND ts_code = ?",
            [run_id, cand.ts_code],
        )
        self.rt.db.execute(
            """
            INSERT INTO apw_stage_results
            (run_id, trade_date, ts_code, candidate_id, rank, launch_score,
             confidence, prediction, main_pattern, phase,
             dimension_scores_json, key_evidence_json, rationale,
             next_session_watch_json, invalidation_triggers_json,
             risk_flags_json, missing_data_json, raw_response_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                trade_date,
                cand.ts_code,
                cand.candidate_id,
                int(cand.rank),
                float(cand.launch_score),
                cand.confidence,
                cand.prediction,
                cand.main_pattern,
                cand.phase,
                json.dumps(cand.dimension_scores.model_dump(), ensure_ascii=False),
                json.dumps(
                    [e.model_dump() for e in cand.key_evidence],
                    ensure_ascii=False,
                ),
                cand.rationale,
                json.dumps(list(cand.next_session_watch), ensure_ascii=False),
                json.dumps(list(cand.invalidation_triggers), ensure_ascii=False),
                json.dumps(list(cand.risk_flags), ensure_ascii=False),
                json.dumps(list(cand.missing_data), ensure_ascii=False),
                cand.model_dump_json(),
                now,
            ],
        )

    # ---- persistence helpers ----

    def _upsert_signal_history(self, cand: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        # Composite PK (trade_date, ts_code) — DELETE + INSERT is portable across
        # SQLite / DuckDB without depending on ON CONFLICT syntax variants.
        self.rt.db.execute(
            "DELETE FROM apw_signal_history WHERE trade_date = ? AND ts_code = ?",
            [cand["trade_date"], cand["ts_code"]],
        )
        self.rt.db.execute(
            """
            INSERT INTO apw_signal_history
            (trade_date, ts_code, name, phase, probe_date,
             accumulation_score, probe_quality_score, washout_score, launch_setup_score,
             raw_candidate_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                cand["trade_date"],
                cand["ts_code"],
                cand.get("name"),
                cand["phase"],
                cand.get("probe_date"),
                _f(cand.get("accumulation_score")),
                _f(cand.get("probe_quality_score")),
                _f(cand.get("washout_score")),
                _f(cand.get("launch_setup_score")),
                json.dumps(cand, ensure_ascii=False),
                now,
            ],
        )

    def _upsert_watchlist(self, cand: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        existing = self.rt.db.fetchone(
            "SELECT first_seen_date FROM apw_watchlist WHERE ts_code = ?",
            [cand["ts_code"]],
        )
        first_seen = existing[0] if existing else cand["trade_date"]
        self.rt.db.execute("DELETE FROM apw_watchlist WHERE ts_code = ?", [cand["ts_code"]])
        self.rt.db.execute(
            """
            INSERT INTO apw_watchlist
            (ts_code, name, first_seen_date, last_seen_date, phase, probe_date,
             accumulation_score, probe_quality_score, washout_score, launch_setup_score,
             latest_launch_score, latest_prediction, latest_confidence,
             raw_candidate_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
            """,
            [
                cand["ts_code"],
                cand.get("name"),
                first_seen,
                cand["trade_date"],
                cand["phase"],
                cand.get("probe_date"),
                _f(cand.get("accumulation_score")),
                _f(cand.get("probe_quality_score")),
                _f(cand.get("washout_score")),
                _f(cand.get("launch_setup_score")),
                json.dumps(cand, ensure_ascii=False),
                now,
            ],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dc_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    try:
        return {k: v for k, v in obj.__dict__.items()}
    except AttributeError:
        return {}


def _date_n_calendar_days_before(date: str, n: int) -> str:
    """Naive calendar-day arithmetic, good enough for fetching wide windows."""
    from datetime import datetime as _dt, timedelta as _td
    dt = _dt.strptime(date, "%Y%m%d") - _td(days=n)
    return dt.strftime("%Y%m%d")


def _fetch_daily_in_batches(
    tushare: Any, *, ts_codes: list[str], start: str, end: str, batch_size: int = 50
) -> pd.DataFrame:
    """Split ts_codes by chunk size to dodge Tushare per-call row caps."""
    if not ts_codes:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(ts_codes), batch_size):
        sub = ts_codes[i : i + batch_size]
        df = fetch_daily(tushare, ts_codes=sub, start=start, end=end)
        if df is not None and not df.empty:
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _fetch_daily_basic_in_batches(
    tushare: Any, *, ts_codes: list[str], start: str, end: str, batch_size: int = 50
) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(ts_codes), batch_size):
        sub = ts_codes[i : i + batch_size]
        df = fetch_daily_basic(tushare, ts_codes=sub, start=start, end=end)
        if df is not None and not df.empty:
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _amount_on_date(quotes: pd.DataFrame, trade_date: str) -> dict[str, float]:
    """tushare daily.amount is in 千元 — convert to 亿元 for the threshold check."""
    if quotes is None or quotes.empty:
        return {}
    sub = quotes[quotes["trade_date"] == trade_date]
    out: dict[str, float] = {}
    for _, r in sub.iterrows():
        amt = float(r.get("amount", 0.0) or 0.0)
        # daily.amount is in 千元 → divide by 100000 to get 亿元
        out[r["ts_code"]] = amt / 100000.0
    return out


def _listed_days(row: pd.Series, trade_date: str) -> int:
    ld = row.get("list_date")
    if not ld:
        return 0
    try:
        return (
            datetime.strptime(trade_date, "%Y%m%d")
            - datetime.strptime(str(int(ld)), "%Y%m%d")
        ).days
    except (ValueError, TypeError):
        return 0


def _normalize_quotes(qdf: pd.DataFrame, basic_extra: pd.DataFrame) -> pd.DataFrame:
    """Merge daily + daily_basic on (ts_code, trade_date), sorted by date asc.

    daily_basic provides turnover_rate + circ_mv that the scorers need.
    """
    if qdf.empty:
        return qdf
    df = qdf.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").reset_index(drop=True)
    if not basic_extra.empty:
        b = basic_extra.copy()
        b["trade_date"] = b["trade_date"].astype(str)
        keep_cols = [c for c in ("ts_code", "trade_date", "turnover_rate", "circ_mv") if c in b.columns]
        df = df.merge(b[keep_cols], on=["ts_code", "trade_date"], how="left")
    return df
