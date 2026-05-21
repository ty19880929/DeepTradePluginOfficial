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
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from deeptrade.core.run_status import RunStatus
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from .calendar import TradeCalendar
from .cancellation import cancel_requested
from .config import ApwConfig, ApwConfigStore
from .data import (
    compute_accumulation,
    compute_alpha_features,
    compute_launch_setup,
    compute_long_range_features,
    compute_ma_distances,
    compute_returns_and_labels,
    compute_vcp_features,
    compute_volume_event_score,
    compute_washout,
    derive_phase,
    detect_probe_day,
    fetch_daily,
    fetch_daily_basic,
    fetch_daily_basic_on,
    fetch_index_daily,
    fetch_latest_trade_date,
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
from .runtime import ApwRuntime, build_tushare_client, pick_llm_provider
from .schemas import APWPhase
from .ui.protocol import EventRenderer, NullRenderer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Param dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScreenParams:
    trade_date: str | None = None
    force_sync: bool = False
    max_candidates: int | None = None
    # v0.3.0 — backfill-history knobs. Default ``False`` keeps single-day
    # ``screen`` behaviour byte-identical to v0.2.0; ``execute_backfill_history``
    # flips both to true on its per-day inner calls.
    skip_watchlist: bool = False
    overwrite_history: bool = False
    skip_if_history_exists: bool = False


@dataclass
class AnalyzeParams:
    trade_date: str | None = None
    max_candidates: int | None = None
    llm_provider: str | None = None  # --llm <provider>
    prediction_filter: str | None = None  # e.g. "launch_ready"
    # v0.6.0 — one-shot LGB disable. Persistent default lives in
    # ApwConfig.lgb_enabled (apw_config table).
    disable_lgb: bool = False


@dataclass
class RunParams:
    """run = screen → analyze."""
    trade_date: str | None = None
    force_sync: bool = False
    max_candidates: int | None = None
    llm_provider: str | None = None
    disable_lgb: bool = False


@dataclass
class EvaluateParams:
    from_date: str | None = None
    to_date: str | None = None
    horizons: str = "1,3,5,10"
    include_early_phases: bool = False
    force_recompute: bool = False


@dataclass
class BackfillHistoryParams:
    """``screen --backfill-history`` — LLM-free batch replay across [start, end].

    Writes only ``apw_signal_history``; never touches ``apw_watchlist``.
    Resumes by default (skips dates with existing rows); ``overwrite`` does a
    wholesale DELETE-by-date before re-inserting that date's hits.
    """

    start: str
    end: str
    overwrite: bool = False
    force_sync: bool = False


@dataclass
class PruneParams:
    """``prune`` — phase-aware watchlist cleanup.

    See §3.1.5 of the migration plan for the trigger table. ``dry_run`` only
    surfaces the candidates that would be deleted (renderer log) without
    touching ``apw_watchlist``.
    """

    dry_run: bool = False
    trade_date: str | None = None  # overrides "today" anchor for reproducibility


@dataclass
class RunOutcome:
    run_id: str
    status: RunStatus
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PreconditionError(RuntimeError):
    """Run cannot start because user-facing preconditions are not met
    (e.g. ``--llm`` named a provider that is not configured).

    Plugin-internal contract: raise BEFORE ``_start_run`` so no run row is
    persisted. ``cli.main`` renders these as ``✘ {message}`` without a
    traceback or type prefix — they are user-config errors, not runtime
    crashes.
    """


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
        # ``is_intraday`` column retained for migration-immutability; always
        # FALSE now that the intraday opt-in flag is gone.
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
                False,
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

    def _validate_single_provider(self, override: str | None) -> str | None:
        """Resolve & validate the ``--llm`` override.

        Returns the provider name to pass to ``LLMManager.get_client`` —
        either the user's override (when ``--llm`` named a configured
        provider) or ``None`` (defer to framework default).

        Raises:
            PreconditionError: when ``--llm`` named a provider that is not
                in ``LLMManager.list_providers()`` (i.e. not configured or
                missing api_key). Surface BEFORE ``_start_run`` so no run
                row is persisted; ``cli.main`` renders as ``✘ {message}``.
        """
        resolved = pick_llm_provider(self.rt, override)
        if resolved is None:
            return None
        try:
            available = self.rt.llms.list_providers()
        except Exception:  # noqa: BLE001 — degraded LLMManager → defer to caller
            available = []
        if resolved not in available:
            raise PreconditionError(
                f"--llm 指定的 provider {resolved!r} 未配置或缺 api_key; "
                f"当前可用: {available}。"
                "请运行 `deeptrade config set-llm` 配置后重试。"
            )
        return resolved

    def _emit_llm_provider_log(self, provider: str | None) -> None:
        """Emit the audit LOG event announcing the resolved provider.

        Mirrors limit-up-board's first persisted event of the run. ``None``
        means we deferred to the framework default; show that verbatim.
        """
        display = provider if provider is not None else "(framework default)"
        self._emit(
            EventType.LOG,
            f"LLM provider = {display}",
            payload={"llm_provider": provider},
        )

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
            self.rt.tushare = build_tushare_client(self.rt)

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

        latest = None if params.trade_date else fetch_latest_trade_date(tushare)
        T, _ = resolve_trade_date(
            calendar,
            latest_trade_date=latest,
            user_specified=params.trade_date,
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
            # daily_basic must be pulled per trade_date — multi-code
            # ts_code lists return 0 rows from Tushare (see helper docstring).
            uni_codes = uni["ts_code"].astype(str).tolist()
            trade_dates_in_window = calendar.open_dates_in_range(window_start, T)
            basic_extra = _fetch_daily_basic_by_day(
                tushare,
                trade_dates=trade_dates_in_window,
                universe=set(uni_codes),
            )
            # moneyflow caps ts_code lists at 1000; fetch_moneyflow batches internally.
            mf_outcome = fetch_moneyflow(
                tushare, ts_codes=uni_codes, start=window_start, end=T
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

            # ---- Liquidity filter on T-day amount, then market-cap band.
            # circ_mv comes from daily_basic in 万元 → convert to 亿元 to compare
            # against cfg.min_circ_mv_yi / max_circ_mv_yi (round-2 P3 — the
            # band was exposed in settings but had no effect on the universe).
            day_amount = _amount_on_date(quotes, T)
            day_circ_mv_yi = _circ_mv_yi_on_date(basic_extra, T)
            liquidity_mask = uni["ts_code"].map(
                lambda c: day_amount.get(c, 0.0) >= cfg.min_amount_yi
            )
            after_liquidity = uni[liquidity_mask].reset_index(drop=True)
            n_after_liquidity = len(after_liquidity)

            # ---- daily_basic coverage guard.
            # When circ_mv is missing for the bulk of the liquid universe we
            # silently sink n_after_mv to 0 and the whole run looks like
            # "today nothing matched" — but the real cause is a data outage.
            # Fail fast so the user gets an actionable error instead of an
            # empty success row.
            if basic_extra is None or basic_extra.empty:
                msg = (
                    f"daily_basic returned no rows for window {window_start}..{T}; "
                    "cannot apply market-cap filter (check Tushare access / "
                    "fetch_daily_basic_on shape)"
                )
                raise RuntimeError(msg)
            if n_after_liquidity > 0:
                covered = sum(
                    1 for code in after_liquidity["ts_code"].astype(str)
                    if day_circ_mv_yi.get(code) is not None
                )
                coverage_ratio = covered / float(n_after_liquidity)
                if coverage_ratio < 0.5:
                    msg = (
                        f"daily_basic returned circ_mv for only {covered}/"
                        f"{n_after_liquidity} (≈{coverage_ratio:.0%}) of the liquid "
                        f"universe on T={T}; cannot apply market-cap filter "
                        "reliably (likely Tushare data gap)"
                    )
                    raise RuntimeError(msg)

            def _in_mv_band(code: str) -> bool:
                cm = day_circ_mv_yi.get(code)
                # Missing daily_basic for this code on T → exclude to honour
                # the contract (and avoid feeding through a candidate whose
                # basic.circ_mv_yi would be 0).
                if cm is None:
                    return False
                if cm < cfg.min_circ_mv_yi:
                    return False
                if cfg.max_circ_mv_yi > 0 and cm > cfg.max_circ_mv_yi:
                    return False
                return True

            mv_mask = after_liquidity["ts_code"].map(_in_mv_band)
            after_mv = after_liquidity[mv_mask].reset_index(drop=True)
            n_after_mv = len(after_mv)

            self._emit(
                EventType.DATA_SYNC_FINISHED,
                f"数据同步完成: 主板池={n_main_board}, ST/停牌后={n_after_st_susp}, "
                f"流动性后={n_after_liquidity}, 市值后={n_after_mv}",
                payload={
                    "n_total": int(n_total),
                    "n_main_board": int(n_main_board),
                    "n_after_st_susp": int(n_after_st_susp),
                    "n_after_liquidity": int(n_after_liquidity),
                    "n_after_mv": int(n_after_mv),
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
            for _, row in after_mv.iterrows():
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
                    mf_df=mfd if not mfd.empty else None,
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
                # v0.4.0 — extended derived features (VCP / long-range
                # resistance / alpha / MA distances / volume_event_score).
                # All are NaN-safe; on too-little history each helper emits
                # ``None`` values and the LGB booster handles them natively.
                vcp_features = compute_vcp_features(qdf)
                long_range = compute_long_range_features(qdf)
                alpha_features = compute_alpha_features(
                    qdf, index_df if not index_df.empty else None
                )
                ma_distances = compute_ma_distances(qdf)
                volume_event = compute_volume_event_score(qdf)
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
                    vcp=vcp_features,
                    long_range=long_range,
                    alpha=alpha_features,
                    ma_distances=ma_distances,
                    volume_event_score=volume_event,
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
            # v0.3.0 — backfill-history can DELETE the whole T day's rows up
            # front so the per-cand DELETE+INSERT inside _upsert_signal_history
            # behaves like a clean replace (and not just per-(date, ts_code)).
            if params.overwrite_history:
                self.rt.db.execute(
                    "DELETE FROM apw_signal_history WHERE trade_date = ?", [T]
                )
            for cand in hits:
                if params.skip_if_history_exists:
                    existing = self.rt.db.fetchone(
                        "SELECT 1 FROM apw_signal_history "
                        "WHERE trade_date = ? AND ts_code = ?",
                        [cand["trade_date"], cand["ts_code"]],
                    )
                    if existing is not None:
                        continue
                self._upsert_signal_history(cand)
                persisted_history += 1
                if params.skip_watchlist:
                    continue
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
            self.rt.tushare = build_tushare_client(self.rt)
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

    # ---- backfill-history (v0.3.0) ------------------------------------------

    def execute_backfill_history(self, params: BackfillHistoryParams) -> RunOutcome:
        """LLM-free batch replay of screen rules across ``[start, end]``.

        Owns its own ``apw_runs`` row with ``mode='backfill_history'`` and
        delegates the per-day funnel to :meth:`execute_screen` via
        ``_owns_run=False``. Writes only ``apw_signal_history``; never touches
        ``apw_watchlist`` (the ``skip_watchlist`` flag is forced on for every
        inner call).

        Resume policy:
            * default — skip dates whose ``apw_signal_history`` already has any
              row (DB-resident resume; cheap "WHERE trade_date = ? LIMIT 1");
            * ``--overwrite`` — DELETE the full day's rows first, then refill.
        """
        if self.rt.tushare is None:
            self.rt.tushare = build_tushare_client(self.rt)
        tushare = self.rt.tushare
        now_utc = datetime.now(timezone.utc)
        cal_start = (now_utc.replace(day=1).replace(year=now_utc.year - 4)).strftime(
            "%Y%m%d"
        )
        cal_end = (now_utc + timedelta(days=30)).strftime("%Y%m%d")
        cal_df = fetch_trade_cal(tushare, start=cal_start, end=cal_end)
        calendar = TradeCalendar(cal_df)

        if not (params.start and params.end):
            raise ValueError("backfill-history requires both --start and --end")
        if params.start > params.end:
            raise ValueError(
                f"backfill-history: --start ({params.start}) must be <= --end ({params.end})"
            )

        all_dates = calendar.open_dates_in_range(params.start, params.end)
        if not all_dates:
            raise ValueError(
                f"backfill-history: no open trade dates in [{params.start}, {params.end}]"
            )

        run_id = self._start_run("backfill_history", params.start, params)
        try:
            self._emit(
                EventType.LOG,
                f"backfill-history: scanning {len(all_dates)} trade dates "
                f"[{all_dates[0]} .. {all_dates[-1]}] overwrite={params.overwrite}",
                payload={"n_dates": len(all_dates), "overwrite": params.overwrite},
            )

            n_processed = 0
            n_skipped = 0
            n_failed = 0
            n_history_total = 0
            for T in all_dates:
                if cancel_requested():
                    raise KeyboardInterrupt
                # Resume: skip dates with any existing apw_signal_history row,
                # unless --overwrite was passed.
                if not params.overwrite:
                    existing = self.rt.db.fetchone(
                        "SELECT 1 FROM apw_signal_history "
                        "WHERE trade_date = ? LIMIT 1",
                        [T],
                    )
                    if existing is not None:
                        n_skipped += 1
                        continue

                self._emit(
                    EventType.STEP_STARTED,
                    f"backfill T={T} ({n_processed + n_skipped + n_failed + 1}/{len(all_dates)})",
                    payload={"step": 1, "trade_date": T},
                )
                inner = ScreenParams(
                    trade_date=T,
                    force_sync=params.force_sync,
                    skip_watchlist=True,
                    overwrite_history=params.overwrite,
                    skip_if_history_exists=False,
                )
                # Share the same run_id / event stream with the inner call.
                outcome = self.execute_screen(inner, _owns_run=False)
                if outcome.status == RunStatus.SUCCESS:
                    n_processed += 1
                    n_history_total += int(
                        outcome.summary.get("n_signal_history", 0) or 0
                    )
                else:
                    n_failed += 1
                    self._emit(
                        EventType.LOG,
                        f"backfill T={T} failed: {outcome.error}",
                        level=EventLevel.WARN,
                        payload={"trade_date": T, "error": outcome.error},
                    )

            summary = {
                "n_dates_requested": len(all_dates),
                "n_dates_processed": n_processed,
                "n_dates_skipped": n_skipped,
                "n_dates_failed": n_failed,
                "n_signal_history_rows": n_history_total,
                "overwrite": bool(params.overwrite),
            }
            self._finish_run(RunStatus.SUCCESS, summary=summary)
            return RunOutcome(run_id=run_id, status=RunStatus.SUCCESS, summary=summary)
        except KeyboardInterrupt:
            return self._make_cancel_outcome(run_id, "backfill_history", True)
        except Exception as exc:  # noqa: BLE001
            if cancel_requested():
                return self._make_cancel_outcome(run_id, "backfill_history", True)
            tb = traceback.format_exc()
            self._emit(
                EventType.LOG,
                f"backfill-history failed: {exc}",
                level=EventLevel.ERROR,
                payload={"traceback": tb},
            )
            self._shielded_finish_run(RunStatus.FAILED, error=str(exc))
            return RunOutcome(run_id=run_id, status=RunStatus.FAILED, error=str(exc))

    # ---- prune (v0.3.0) -----------------------------------------------------

    def execute_prune(self, params: PruneParams) -> RunOutcome:
        """Phase-aware watchlist cleanup. See §3.1.5 of the migration plan.

        Emits one ``WATCHLIST_PRUNE_HIT`` event per candidate that matched a
        deletion rule (payload carries ``ts_code`` / ``reason`` / ``phase``).
        With ``dry_run=True`` events still fire but no rows are deleted.
        """
        cfg_store = ApwConfigStore(self.rt.db)
        cfg = cfg_store.load()

        # Resolve "today" — defaults to the latest open trade_date so the
        # idle-days math is calendar-aware.
        if self.rt.tushare is None:
            self.rt.tushare = build_tushare_client(self.rt)
        tushare = self.rt.tushare
        now_utc = datetime.now(timezone.utc)
        cal_start = (now_utc.replace(day=1).replace(year=now_utc.year - 1)).strftime(
            "%Y%m%d"
        )
        cal_end = (now_utc + timedelta(days=30)).strftime("%Y%m%d")
        cal_df = fetch_trade_cal(tushare, start=cal_start, end=cal_end)
        calendar = TradeCalendar(cal_df)
        # Only probe for the latest trade date when the caller didn't pin one
        # — saves the index_daily round-trip in tests/scripted contexts.
        latest = None if params.trade_date else fetch_latest_trade_date(tushare)
        T, _ = resolve_trade_date(
            calendar,
            latest_trade_date=latest,
            user_specified=params.trade_date,
        )

        run_id = self._start_run("prune", T, params)
        try:
            rows = (
                self.rt.db.fetchall(
                    """
                    SELECT ts_code, name, phase, probe_date, first_seen_date,
                           last_seen_date, raw_candidate_json
                    FROM apw_watchlist
                    ORDER BY last_seen_date DESC, ts_code ASC
                    """
                )
                or []
            )
            self._emit(
                EventType.STEP_STARTED,
                f"prune: scanning {len(rows)} watchlist rows (T={T}, dry_run={params.dry_run})",
                payload={"step": 1, "n_watchlist": len(rows), "trade_date": T,
                         "dry_run": bool(params.dry_run)},
            )

            to_delete: list[tuple[str, str, str]] = []  # (ts_code, reason, phase)
            for r in rows:
                ts_code, _name, phase, probe_date, _first_seen, last_seen, raw_json = r
                reason = self._prune_reason(
                    cfg=cfg,
                    phase=str(phase or ""),
                    last_seen=str(last_seen or ""),
                    probe_date=str(probe_date or ""),
                    raw_json=str(raw_json or "{}"),
                    today=T,
                    calendar=calendar,
                )
                if reason is not None:
                    to_delete.append((ts_code, reason, str(phase or "")))

            for ts_code, reason, phase in to_delete:
                self._emit(
                    EventType.LOG,
                    f"prune hit: {ts_code} [{phase}] — {reason}",
                    payload={
                        "ts_code": ts_code,
                        "phase": phase,
                        "reason": reason,
                        "trade_date": T,
                        "dry_run": bool(params.dry_run),
                    },
                )
                if not params.dry_run:
                    self.rt.db.execute(
                        "DELETE FROM apw_watchlist WHERE ts_code = ?", [ts_code]
                    )

            summary = {
                "n_watchlist": len(rows),
                "n_deleted": 0 if params.dry_run else len(to_delete),
                "n_would_delete": len(to_delete),
                "dry_run": bool(params.dry_run),
                "trade_date": T,
            }
            self._emit(
                EventType.STEP_FINISHED,
                f"prune 完成: n_deleted={summary['n_deleted']} "
                f"n_would_delete={summary['n_would_delete']}",
                payload={"step": 1, **summary},
            )
            self._finish_run(RunStatus.SUCCESS, summary=summary)
            return RunOutcome(run_id=run_id, status=RunStatus.SUCCESS, summary=summary)
        except KeyboardInterrupt:
            return self._make_cancel_outcome(run_id, "prune", True)
        except Exception as exc:  # noqa: BLE001
            if cancel_requested():
                return self._make_cancel_outcome(run_id, "prune", True)
            tb = traceback.format_exc()
            self._emit(
                EventType.LOG,
                f"prune failed: {exc}",
                level=EventLevel.ERROR,
                payload={"traceback": tb},
            )
            self._shielded_finish_run(RunStatus.FAILED, error=str(exc))
            return RunOutcome(run_id=run_id, status=RunStatus.FAILED, error=str(exc))

    def _prune_reason(
        self,
        *,
        cfg: ApwConfig,
        phase: str,
        last_seen: str,
        probe_date: str,
        raw_json: str,
        today: str,
        calendar: TradeCalendar,
    ) -> str | None:
        """Return a human-readable deletion reason or ``None`` to keep the row.

        Rules (mirroring §3.1.5 of the plan):
          1. ``launch_ready`` idle ≥ prune_idle_days_launch_ready trade days
          2. ``washing_after_probe`` past washout_max_trade_days w/o transition
          3. close on T below the probe-day low
          4. close on T below MA60
        """
        # Rule 1: launch_ready idle too long.
        if phase == APWPhase.LAUNCH_READY.value and last_seen:
            idle = calendar.trade_days_between(last_seen, today)
            if idle is not None and idle >= cfg.prune_idle_days_launch_ready:
                return (
                    f"launch_ready idle {idle} trade days "
                    f">= prune_idle_days_launch_ready={cfg.prune_idle_days_launch_ready}"
                )

        # Rule 2: washing_after_probe past max window.
        if phase == APWPhase.WASHING_AFTER_PROBE.value and probe_date:
            elapsed = calendar.trade_days_between(probe_date, today)
            if elapsed is not None and elapsed > cfg.washout_max_trade_days:
                return (
                    f"washout_after_probe elapsed {elapsed} trade days "
                    f"> washout_max_trade_days={cfg.washout_max_trade_days}"
                )

        # Parse the raw candidate payload once for rules 3/4.
        try:
            cand = json.loads(raw_json) if raw_json else {}
        except (json.JSONDecodeError, TypeError):
            cand = {}

        # Rule 3: close below probe-day low.
        if cfg.prune_drop_on_probe_low_break:
            close = _f(cand.get("close"))
            probe_low = _f(cand.get("probe_low"))
            if close is not None and probe_low is not None and close < probe_low:
                return (
                    f"close {close:.2f} fell below probe_low {probe_low:.2f}"
                )

        # Rule 4: close below MA60.
        if cfg.prune_drop_on_ma60_break:
            close = _f(cand.get("close"))
            ma60 = _f(cand.get("ma60"))
            if close is not None and ma60 is not None and close < ma60:
                return f"close {close:.2f} fell below MA60 {ma60:.2f}"

        return None

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
            self.rt.tushare = build_tushare_client(self.rt)
        tushare = self.rt.tushare
        now_utc = datetime.now(timezone.utc)
        cal_start = (now_utc.replace(day=1)
                     .replace(year=now_utc.year - 2)).strftime("%Y%m%d")
        cal_end = (now_utc + timedelta(days=90)).strftime("%Y%m%d")
        cal_df = fetch_trade_cal(tushare, start=cal_start, end=cal_end)
        calendar = TradeCalendar(cal_df)
        latest = None if params.trade_date else fetch_latest_trade_date(tushare)
        T, _ = resolve_trade_date(
            calendar,
            latest_trade_date=latest,
            user_specified=params.trade_date,
        )

        # Precondition check BEFORE _start_run so a bad --llm never persists a
        # RUNNING/FAILED row. PreconditionError → cli.main → ✘ {msg}.
        provider_name = self._validate_single_provider(params.llm_provider)

        run_id = self._start_run("run", T, params)
        self._emit_llm_provider_log(provider_name)

        try:
            # Pin T into the sub-params so screen/analyze resolve to the same day.
            screen_params = ScreenParams(
                trade_date=T,
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
                max_candidates=params.max_candidates,
                llm_provider=params.llm_provider,
                disable_lgb=params.disable_lgb,
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
            self.rt.tushare = build_tushare_client(self.rt)
        tushare = self.rt.tushare
        now_utc = datetime.now(timezone.utc)
        cal_start = (now_utc.replace(day=1)
                     .replace(year=now_utc.year - 2)).strftime("%Y%m%d")
        cal_end = (now_utc + timedelta(days=90)).strftime("%Y%m%d")
        cal_df = fetch_trade_cal(tushare, start=cal_start, end=cal_end)
        calendar = TradeCalendar(cal_df)

        latest = None if params.trade_date else fetch_latest_trade_date(tushare)
        T, next_T = resolve_trade_date(
            calendar,
            latest_trade_date=latest,
            user_specified=params.trade_date,
        )

        # Precondition check BEFORE _start_run so a bad --llm never persists a
        # RUNNING/FAILED row. When called from execute_run (_owns_run=False),
        # the parent already validated and emitted the audit LOG; skip both
        # to avoid duplicate noise.
        if _owns_run:
            provider_name = self._validate_single_provider(params.llm_provider)
        else:
            provider_name = params.llm_provider

        if _owns_run:
            run_id = self._start_run("analyze", T, params)
            self._emit_llm_provider_log(provider_name)
        else:
            assert self.rt.run_id is not None, (
                "_owns_run=False requires the caller to have started a run"
            )
            run_id = self.rt.run_id

        try:
            # ---- read watchlist (≥ washing_after_probe by construction).
            # Only rows refreshed by today's screen (last_seen_date = T) are
            # eligible; stale rows from prior trade days would otherwise be
            # re-analysed and persisted under the current T (review round 2 P1).
            # In run mode (screen → analyze in one process) screen already
            # owns step 1 ("漏斗筛选"); reuse step=1 here would overwrite its
            # done state in the dashboard. Use step=2.5 instead — the run
            # mode StageStack template has a matching slot, while standalone
            # analyze keeps the historical step=1.
            read_step: float = 1 if _owns_run else 2.5
            read_label_prefix = (
                "Step 1" if _owns_run else "Analyze Step 2.5"
            )
            self._emit(
                EventType.STEP_STARTED,
                f"{read_label_prefix}: 读取 watchlist",
                payload={"step": read_step},
            )
            rows = self.rt.db.fetchall(
                """
                SELECT ts_code, name, phase, raw_candidate_json
                FROM apw_watchlist
                WHERE phase IN ('washing_after_probe', 'launch_ready')
                  AND last_seen_date = ?
                ORDER BY launch_setup_score DESC NULLS LAST,
                         washout_score DESC NULLS LAST
                """,
                [T],
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
                f"{read_label_prefix}: 读取 watchlist 完成，候选 {len(candidates)}",
                payload={"step": read_step, "n_candidates": len(candidates)},
            )

            # v0.6.0 — LGB scoring before LLM. Failure paths are wholly
            # absorbed by LgbScorer (5-branch fallback emits LGB_DEGRADE_*
            # events and leaves lgb_score = None); analyze never aborts.
            lgb_scored = 0
            lgb_persisted = 0
            lgb_degrade = None
            lgb_model_id: str | None = None
            if candidates and cfg.lgb_enabled and not params.disable_lgb:
                from .lgb.scorer import build_lgb_scorer  # noqa: PLC0415

                self._emit(
                    EventType.STEP_STARTED,
                    "Step 1.5: LGB 评分",
                    payload={"step": 1.5},
                )
                self.rt.lgb_scorer = build_lgb_scorer(self.rt.db)
                outcome = self.rt.lgb_scorer.score_batch(candidates)
                lgb_model_id = outcome.model_id
                if outcome.degrade_reason:
                    lgb_degrade = outcome.degrade_reason
                    self._emit(
                        EventType.LOG,
                        f"LGB degraded: {outcome.degrade_reason}",
                        level=EventLevel.WARN,
                        payload={
                            "lgb_degrade_reason": outcome.degrade_reason,
                            "model_id": outcome.model_id,
                        },
                    )
                else:
                    lgb_persisted = self.rt.lgb_scorer.persist_predictions(
                        self.rt.db, outcome,
                        run_id=run_id, trade_date=T,
                    )
                # Inject score / decile into each candidate dict (None when
                # the row degraded). The prompt builder consumes them; the
                # whitelist already includes lgb_score / lgb_decile.
                by_code = {s.ts_code: s for s in outcome.scores}
                for cand in candidates:
                    s = by_code.get(str(cand.get("ts_code", "")))
                    if s is None:
                        continue
                    cand["lgb_score"] = s.lgb_score
                    if cfg.lgb_decile_in_prompt:
                        cand["lgb_decile"] = s.lgb_decile
                    if (
                        s.lgb_score is not None
                        and cfg.lgb_min_score_floor is not None
                        and s.lgb_score < cfg.lgb_min_score_floor
                    ):
                        # Tag — visible to LLM, doesn't filter the candidate.
                        flags = cand.setdefault("risk_flags_local", [])
                        if "low_lgb_score" not in flags:
                            flags.append("low_lgb_score")
                    if s.lgb_score is not None:
                        lgb_scored += 1
                self._emit(
                    EventType.STEP_FINISHED,
                    f"Step 1.5: LGB 评分完成 ({lgb_scored}/{len(candidates)})",
                    payload={
                        "step": 1.5,
                        "n_scored": lgb_scored,
                        "n_persisted": lgb_persisted,
                        "model_id": lgb_model_id,
                        "degrade_reason": lgb_degrade,
                    },
                )

            if not candidates:
                # Probe whether stale rows exist so standalone analyze callers
                # know they may need to run screen first today.
                stale_row = self.rt.db.fetchone(
                    """
                    SELECT COUNT(*) FROM apw_watchlist
                    WHERE phase IN ('washing_after_probe', 'launch_ready')
                      AND last_seen_date <> ?
                    """,
                    [T],
                )
                n_stale = int(stale_row[0]) if stale_row else 0
                if n_stale > 0:
                    self._emit(
                        EventType.LOG,
                        f"watchlist 当日候选为 0 (T={T})；存在 {n_stale} 条非当日行，"
                        "请先运行 screen 刷新当日命中。",
                        level=EventLevel.WARN,
                    )
                if _owns_run:
                    self._finish_run(
                        RunStatus.SUCCESS,
                        summary={"n_candidates": 0, "n_predictions": 0},
                    )
                return RunOutcome(
                    run_id=run_id, status=RunStatus.SUCCESS,
                    summary={"n_candidates": 0, "n_predictions": 0},
                )

            # ---- pick LLM provider (None = framework default).
            # ``provider_name`` was resolved+validated above (or inherited
            # from the parent run when ``_owns_run=False``).
            llm = self.rt.llms.get_client(
                provider_name,
                plugin_id=self.rt.plugin_id,
                run_id=run_id,
            )

            # ---- run analyze pipeline
            profile = default_profile()
            terminal_result = None

            def emit_pipeline_progress(ev: StrategyEvent) -> None:
                self._write_event(ev)
                self._dispatch_to_renderer(ev)

            for ev, terminal in run_analyze(
                llm=llm,
                candidates=candidates,
                trade_date=T,
                next_trade_date=next_T,
                market_summary="",
                profile=profile,
                max_batch_size=cfg.llm_batch_size,
                max_repair_retries=cfg.llm_max_repair_retries,
                event_sink=emit_pipeline_progress,
            ):
                self._write_event(ev)
                self._dispatch_to_renderer(ev)
                if terminal is not None:
                    terminal_result = terminal

            assert terminal_result is not None  # generator always yields terminal

            # ---- persist
            result_summary = _build_result_summary_rows(
                terminal_result.predictions,
                candidates,
            )
            self._emit(
                EventType.STEP_STARTED,
                "Step 5: 写入结果",
                payload={
                    "step": 5,
                    "n_predictions": terminal_result.candidates_out,
                },
            )
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
            self._emit(
                EventType.STEP_FINISHED,
                f"Step 5: 写入结果完成，写入 {n_persisted} 条",
                payload={
                    "step": 5,
                    "n_predictions": n_persisted,
                    "n_failed_batches": terminal_result.failed_batches,
                    "result_summary": result_summary,
                    "result_summary_total": len(terminal_result.predictions),
                    "result_summary_displayed": len(result_summary),
                },
            )

            summary = {
                "n_candidates": len(candidates),
                "n_predictions": n_persisted,
                "n_failed_batches": terminal_result.failed_batches,
                "failed_batch_ids": terminal_result.failed_batch_ids,
                "trade_date": T,
                "lgb_enabled": bool(cfg.lgb_enabled and not params.disable_lgb),
                "lgb_model_id": lgb_model_id,
                "lgb_n_scored": lgb_scored,
                "lgb_n_persisted": lgb_persisted,
                "lgb_degrade_reason": lgb_degrade,
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
        # v0.3.0 — dual-write the 6 dim_* DOUBLE columns alongside the legacy
        # json blob (migration 20260520_002 introduces the columns; rollback
        # = drop columns, json remains the source of truth).
        ds = cand.dimension_scores.model_dump()
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
             risk_flags_json, missing_data_json, raw_response_json, created_at,
             dim_accumulation, dim_probe, dim_washout, dim_launch_timing,
             dim_capital_confirmation, dim_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?)
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
                json.dumps(ds, ensure_ascii=False),
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
                _f(ds.get("accumulation")),
                _f(ds.get("probe")),
                _f(ds.get("washout")),
                _f(ds.get("launch_timing")),
                _f(ds.get("capital_confirmation")),
                _f(ds.get("risk")),
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


def _build_result_summary_rows(
    predictions: list[Any],
    input_candidates: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    by_candidate_id = {
        str(c.get("candidate_id")): c
        for c in input_candidates
        if c.get("candidate_id") is not None
    }
    by_code = {
        str(c.get("ts_code")): c
        for c in input_candidates
        if c.get("ts_code") is not None
    }
    rows: list[dict[str, Any]] = []
    for pred in sorted(predictions, key=lambda p: int(getattr(p, "rank", 0) or 0)):
        src = by_candidate_id.get(str(pred.candidate_id)) or by_code.get(str(pred.ts_code)) or {}
        rows.append(
            {
                "rank": int(pred.rank),
                "ts_code": pred.ts_code,
                "name": pred.name,
                "current_price": _f(src.get("close")),
                "launch_score": float(pred.launch_score),
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "llm_opinion": pred.rationale,
            }
        )
    return rows[:limit]


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


def _fetch_daily_basic_by_day(
    tushare: Any,
    *,
    trade_dates: list[str],
    universe: set[str],
) -> pd.DataFrame:
    """Pull ``daily_basic`` per trade day, then intersect with ``universe``.

    Tushare's ``daily_basic`` returns 0 rows when called with a multi-code
    ``ts_code='a,b,...'`` list (verified by direct probe), so the legacy
    chunked-by-ts_code helper silently produced empty frames and erased
    ``circ_mv`` for every code — sinking the market-cap filter and turning
    successful runs into ``n_after_mv=0``. Querying by ``trade_date`` is
    the only reliable shape; we filter to the active universe in memory
    so we don't haul unrelated codes around.

    Only the columns the runner consumes downstream are retained
    (``ts_code``, ``trade_date``, ``turnover_rate``, ``circ_mv``).
    """
    if not trade_dates or not universe:
        return pd.DataFrame()
    keep_cols = ("ts_code", "trade_date", "turnover_rate", "circ_mv")
    chunks: list[pd.DataFrame] = []
    for td in trade_dates:
        df = fetch_daily_basic_on(tushare, trade_date=td)
        if df is None or df.empty or "ts_code" not in df.columns:
            continue
        sub = df[df["ts_code"].astype(str).isin(universe)]
        if sub.empty:
            continue
        cols = [c for c in keep_cols if c in sub.columns]
        chunks.append(sub[cols].copy())
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out["trade_date"] = out["trade_date"].astype(str)
    return out


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


def _circ_mv_yi_on_date(
    basic_extra: pd.DataFrame, trade_date: str
) -> dict[str, float]:
    """tushare daily_basic.circ_mv is in 万元 — convert to 亿元 (÷10000).

    Returns ``{}`` if basic_extra is missing the column entirely (older
    Tushare snapshots), letting the caller decide how to degrade.
    """
    if basic_extra is None or basic_extra.empty:
        return {}
    if "circ_mv" not in basic_extra.columns:
        return {}
    sub = basic_extra[basic_extra["trade_date"].astype(str) == trade_date]
    out: dict[str, float] = {}
    for _, r in sub.iterrows():
        cm = r.get("circ_mv")
        if cm is None or pd.isna(cm):
            continue
        try:
            out[r["ts_code"]] = float(cm) / 10000.0
        except (TypeError, ValueError):
            continue
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
