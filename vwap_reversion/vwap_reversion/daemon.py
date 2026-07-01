"""CollectDaemon — 常驻事件循环（设计 §10.1，P1 = 只采不交易）.

生命周期：
    startup 决策（§4.3）→ vwr_runs(standby) → [待机心跳] → running →
    采集循环（轮询/差分/引擎/落库/事件）→ 午休 sleep → 15:00 收尾 → done
    任何阶段 Ctrl-C → aborted（exit 130）

崩溃恢复（§14）：重启同 (code, trade_date) 时，从 vwr_bars 重放重建
VwapEngine 的 V/A/Q，并用最后一条 vwr_snapshots prime BarBuilder 差分基线，
避免首条 bar 双计开盘以来的量。

时间纪律（§4）：一切「现在/今天/会话」判断走 MarketClock；sleep 目标先转
epoch 再求差。``sleep_fn`` 可注入（测试用假时钟推进）。
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from deeptrade.plugins_api.events import EventLevel, EventType

from .clock import (
    AFTERNOON_OPEN,
    MORNING_OPEN,
    MarketClock,
    SessionPhase,
    StartupAction,
    decide_startup,
    parse_hhmm,
)
from .engine.vwap import VwapEngine
from .feed.base import BarBuilder, CumulativeRegression, SnapshotSource
from .persistence import (
    EventWriter,
    TradeCalendarStore,
    create_run,
    insert_signal,
    insert_trade,
    last_snapshot,
    load_bars,
    update_run_status,
    upsert_bar,
    upsert_daily_summary,
    upsert_snapshot,
)
from .schemas import Snapshot
from .trading import BarOutcome, TradingSession
from .ui import LegacyStreamRenderer, RunMeta

if TYPE_CHECKING:  # pragma: no cover
    from .config import VwrConfig
    from .runtime import VwrRuntime
    from .ui.protocol import EventRenderer


class PreconditionError(Exception):
    """启动前置条件不满足（非交易日/收盘后/日历缺失等）—— exit 2，不留 running 残骸。"""


@dataclass
class DaemonOutcome:
    run_id: str | None
    status: str            # done / aborted / exit
    exit_code: int
    message: str = ""


# 连续失败到该阈值时把 WARN 升级为 ERROR（仍继续重试，不崩；设计 §14）。
_FAIL_ESCALATE = 5
_BACKOFF_BASE = 5.0
_BACKOFF_CAP = 60.0


class CollectDaemon:
    """P1 采集 daemon。P2 在采集循环里挂 signal/risk/execution。"""

    def __init__(
        self,
        rt: VwrRuntime,
        cfg: VwrConfig,
        *,
        source: SnapshotSource,
        calendar: TradeCalendarStore,
        renderer: EventRenderer,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.rt = rt
        self.cfg = cfg
        self.source = source
        self.calendar = calendar
        self.renderer: EventRenderer = renderer
        self._sleep = sleep_fn or _time.sleep
        self._events: EventWriter | None = None
        self._renderer_dead = False
        self._session: TradingSession | None = None
        self._run_id: str | None = None
        self._trade_count = 0

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def execute(self, *, code: str) -> DaemonOutcome:
        clock = self.rt.clock
        decision = decide_startup(
            clock,
            is_trading_day=self.calendar.is_trading_day,
            standby_across_days=self.cfg.standby_across_days,
        )
        if decision.action is StartupAction.EXIT:
            # 今日无会话：不留 run 残骸，直接前置失败（§4.3 EXIT 行）。
            raise PreconditionError(decision.reason)

        trade_date = decision.trade_date or clock.today_str()
        params = self._params_snapshot(code)
        run_id = create_run(
            self.rt.db, mode="paper", code=code, trade_date=trade_date,
            status="standby", params=params, initial_cash=self.cfg.initial_cash,
        )
        self.rt.run_id = run_id
        self._run_id = run_id
        self._events = EventWriter(self.rt.db, run_id)

        meta = RunMeta(
            run_id=run_id, code=code, trade_date=trade_date, mode="paper",
            params_summary=(
                f"poll={self.cfg.poll_interval_seconds}s "
                f"k=({self.cfg.band_k_entry}/{self.cfg.band_k_exit}/{self.cfg.band_k_stop}) "
                f"warmup={self.cfg.warmup_minutes}m tz={self.cfg.market_timezone}"
            ),
        )
        try:
            self.renderer.start(meta, clock)
        except Exception:  # noqa: BLE001 — UI 失败绝不崩 run（§14）
            self._degrade_renderer(meta, clock)

        try:
            if decision.action is StartupAction.STANDBY:
                assert decision.standby_until is not None
                self._standby(run_id, decision.standby_until)
            update_run_status(self.rt.db, run_id, "running")
            self._emit(EventType.STEP_STARTED, "进入采集+交易循环", status="running")
            n_samples, n_bars = self._collect_loop(code, trade_date)

            # ---- 收盘汇总（vwr_daily_summary；P3 交易报告之源）----
            final_cash = self.cfg.initial_cash
            session = self._session
            if session is not None:
                row = session.summary(run_id)
                upsert_daily_summary(self.rt.db, row)
                final_cash = float(row["final_cash"] or self.cfg.initial_cash)
                self._emit(
                    EventType.RESULT_PERSISTED,
                    f"当日汇总：成交 {row['n_trades']} 笔，净盈亏 "
                    f"{row['net_pnl']:+,.2f} 元，期末权益 {final_cash:,.2f} 元",
                    kind="summary", **{k: v for k, v in row.items() if k != "run_id"},
                )
            # 先置 done（报告会读 run 行的 status/final_cash），再生成双报告。
            update_run_status(
                self.rt.db, run_id, "done", final_cash=final_cash, finished=True,
            )
            # ---- 收盘双报告（§12）：失败仅 WARN，绝不影响 run 收尾 ----
            try:
                from .persistence import set_run_report_dir  # noqa: PLC0415
                from .reporting import generate_run_reports  # noqa: PLC0415

                report_dir = generate_run_reports(self.rt.db, run_id)
                set_run_report_dir(self.rt.db, run_id, str(report_dir))
                self._emit(
                    EventType.RESULT_PERSISTED,
                    f"收盘双报告已生成：{report_dir}",
                    kind="reports", report_dir=str(report_dir),
                )
            except Exception as e:  # noqa: BLE001
                self._emit(
                    EventType.LOG, f"报告生成失败（不影响 run）：{e}",
                    level=EventLevel.WARN, kind="report_error",
                )
            self._emit(
                EventType.STEP_FINISHED,
                f"收盘收尾：采样 {n_samples} 次，有效 bar {n_bars} 根",
                status="done", n_samples=n_samples, n_bars=n_bars,
            )
            return DaemonOutcome(run_id, "done", 0)
        except KeyboardInterrupt:
            self._emit(
                EventType.LOG, "用户手动中断，运行标记为 aborted",
                level=EventLevel.WARN, status="aborted",
            )
            update_run_status(self.rt.db, run_id, "aborted", finished=True)
            return DaemonOutcome(run_id, "aborted", 130, "用户手动中断")
        finally:
            try:
                self.renderer.finish()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 待机（§4.3）
    # ------------------------------------------------------------------

    def _standby(self, run_id: str, until) -> None:
        clock = self.rt.clock
        self._emit(
            EventType.LIVE_STATUS,
            f"待机中，{until.strftime('%Y-%m-%d %H:%M')}（市场时区）自动开始",
            status="standby", kind="standby",
            countdown_s=max(0, int(clock.seconds_until(until))),
        )
        while True:
            remaining = clock.seconds_until(until)
            if remaining <= 0:
                return
            self._sleep(min(self.cfg.standby_heartbeat_seconds, remaining))
            # 心跳重检：时钟漂移 / 临时休市公告（日历重读）都在此自愈（§4.3）。
            redecide = decide_startup(
                clock,
                is_trading_day=self.calendar.is_trading_day,
                standby_across_days=self.cfg.standby_across_days,
            )
            if redecide.action is StartupAction.RUN_NOW:
                return
            if redecide.action is StartupAction.EXIT:
                raise PreconditionError(redecide.reason)
            if redecide.standby_until is not None:
                until = redecide.standby_until
            self._emit(
                EventType.LIVE_STATUS,
                f"距开盘 {_fmt_mmss(clock.seconds_until(until))}",
                status="standby", kind="standby",
                countdown_s=max(0, int(clock.seconds_until(until))),
            )

    # ------------------------------------------------------------------
    # 采集循环
    # ------------------------------------------------------------------

    def _collect_loop(self, code: str, trade_date: str) -> tuple[int, int]:
        clock = self.rt.clock
        db = self.rt.db
        engine = VwapEngine(band_k=self.cfg.band_k_entry)
        builder = BarBuilder()
        session = TradingSession(self.cfg, code=code, trade_date=trade_date)
        self._session = session
        eod_epoch = clock.epoch_of(
            _date_of(trade_date), parse_hhmm(self.cfg.eod_flat_time)
        )

        # ---- 崩溃恢复（§14）----
        resumed_bars = load_bars(db, code, trade_date)
        if resumed_bars:
            n = engine.rebuild(resumed_bars)
            self._emit(
                EventType.LOG,
                f"恢复运行：重放 {n} 根已落库 bar（V={engine.cum_vol:,.0f}）",
                level=EventLevel.WARN, kind="resume", n_bars=n,
            )
        prev_snap = last_snapshot(db, code, trade_date)
        if prev_snap is not None:
            builder.prime(prev_snap)

        warmup_end = clock.epoch_of(_date_of(trade_date), MORNING_OPEN) + \
            self.cfg.warmup_minutes * 60
        n_samples, n_bars, consec_fail = 0, len(resumed_bars), 0
        last_progress_epoch = clock.now_epoch()
        stale_warned = False

        while True:
            phase = clock.phase()
            if phase is SessionPhase.AFTER_CLOSE or clock.today_str() != trade_date:
                break
            if phase is SessionPhase.LUNCH:
                resume_at = clock.wall(_date_of(trade_date), AFTERNOON_OPEN)
                self._emit(
                    EventType.LIVE_STATUS,
                    f"午休，{AFTERNOON_OPEN.strftime('%H:%M')} 恢复采集",
                    kind="lunch",
                )
                self._sleep(max(0.0, clock.seconds_until(resume_at)))
                continue
            if phase is SessionPhase.PRE_OPEN:  # 心跳误差兜底
                self._sleep(max(0.0, clock.seconds_until(
                    clock.wall(_date_of(trade_date), MORNING_OPEN))))
                continue

            tick_start = clock.now_epoch()
            try:
                snap = self.source.fetch()
            except Exception as e:  # noqa: BLE001 — 接口异常退避重试，不崩（§14）
                consec_fail += 1
                level = EventLevel.ERROR if consec_fail >= _FAIL_ESCALATE else EventLevel.WARN
                self._emit(
                    EventType.TUSHARE_CALL,
                    f"行情拉取失败（连续 {consec_fail} 次）：{e}",
                    level=level, kind="fetch_error", consec_fail=consec_fail,
                )
                self._sleep(min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** (consec_fail - 1))))
                continue
            consec_fail = 0
            quality_error = _snapshot_quality_error(snap, self.cfg.limit_price_guard_bps)
            if quality_error is not None:
                self._emit(
                    EventType.LOG,
                    f"实时快照异常，跳过本次采样：{quality_error}",
                    level=EventLevel.WARN,
                    kind="data_quality",
                    last=snap.last,
                    cum_vol=snap.cum_vol,
                    cum_amount=snap.cum_amount,
                    trade_time=snap.trade_time,
                )
                self._sleep(self.cfg.poll_interval_seconds)
                continue
            n_samples += 1
            prev_before = builder.prev
            upsert_snapshot(db, snap)
            progressed = _snapshot_progressed(prev_before, snap)
            if progressed:
                last_progress_epoch = snap.ts
                if session.risk.data_halted:
                    session.risk.data_halted = False
                    self._emit(
                        EventType.LOG,
                        "实时行情恢复推进，解除 data_stale 新开仓抑制",
                        kind="stale_recovered",
                    )
                stale_warned = False
            elif clock.now_epoch() - last_progress_epoch >= self.cfg.stale_quote_seconds:
                session.risk.data_halted = True
                if not stale_warned:
                    stale_warned = True
                    self._emit(
                        EventType.LOG,
                        f"实时行情 {self.cfg.stale_quote_seconds}s 未推进，暂停新开仓",
                        level=EventLevel.WARN,
                        kind="stale_quote",
                        stale_seconds=int(clock.now_epoch() - last_progress_epoch),
                    )

            # ---- EOD 强平（有最新价即可触发，不依赖新 bar；§13）----
            if not session.eod_done and clock.now_epoch() >= eod_epoch:
                out = session.eod_flat(snap.last, int(clock.now_epoch()))
                self._emit(
                    EventType.LIVE_STATUS,
                    f"EOD {self.cfg.eod_flat_time} 强平：回到锚点持仓，"
                    "此后只采集不交易",
                    kind="eod", **session.live_payload(snap.last),
                )
                self._apply_trading_outcome(out, session, snap.last)

            try:
                bar = builder.push(snap)
            except CumulativeRegression as e:
                self._emit(
                    EventType.LOG, f"累计量回退，丢弃本条快照：{e}",
                    level=EventLevel.WARN, kind="regression",
                )
                bar = None

            in_warmup = clock.now_epoch() < warmup_end
            if bar is None:
                self._emit(
                    EventType.STEP_PROGRESS,
                    f"采样#{n_samples} 无新成交（last={snap.last}）",
                    kind="sample", n_samples=n_samples, n_bars=n_bars,
                    last=snap.last, warmup=in_warmup,
                    **session.live_payload(snap.last),
                )
            else:
                metrics = engine.push(bar)
                n_bars += 1
                upsert_bar(db, bar, metrics)
                # ---- P2：signal → risk → execution（模拟撮合）----
                out = session.on_bar(bar, metrics, in_warmup=in_warmup)
                z_repr = "—" if metrics.z is None else f"{metrics.z:+.2f}"
                self._emit(
                    EventType.STEP_PROGRESS,
                    f"采样#{n_samples} last={bar.last:.4f} vwap={metrics.vwap:.4f} "
                    f"z={z_repr}" + ("（预热）" if in_warmup else ""),
                    kind="sample", n_samples=n_samples, n_bars=n_bars,
                    last=bar.last, vwap=metrics.vwap, sigma=metrics.sigma,
                    z=metrics.z, band_upper=metrics.band_upper,
                    band_lower=metrics.band_lower, cum_vol=bar.cum_vol,
                    cum_amount=bar.cum_amount, warmup=in_warmup,
                    **session.live_payload(bar.last),
                )
                self._apply_trading_outcome(out, session, bar.last)

            elapsed = clock.now_epoch() - tick_start
            self._sleep(max(0.0, self.cfg.poll_interval_seconds - elapsed))

        # 兜底强平：EOD 时段行情拉取若一直失败，循环结束（15:00）时持仓
        # 仍未平 —— 用最后已知价强平，绝不带腿过收盘汇总。
        if not session.eod_done and session.last_price is not None:
            out = session.eod_flat(session.last_price, int(clock.now_epoch()))
            if out.trades:
                self._emit(
                    EventType.LIVE_STATUS,
                    "收盘兜底强平（EOD 时段未取到行情，按最后已知价平仓）",
                    level=EventLevel.WARN, kind="eod",
                    **session.live_payload(session.last_price),
                )
            self._apply_trading_outcome(out, session, session.last_price)

        return n_samples, n_bars

    # ------------------------------------------------------------------
    # 交易产出：落库 + emit（P2）
    # ------------------------------------------------------------------

    def _apply_trading_outcome(
        self, out: BarOutcome, session: TradingSession, price: float
    ) -> None:
        db = self.rt.db
        assert self._run_id is not None
        if out.circuit_tripped:
            self._emit(
                EventType.LOG,
                f"⛔ 日亏熔断触发（亏损达 {self.cfg.daily_loss_limit_pct}% 初始资金）："
                "平仓并停止开新仓，今日只采集",
                level=EventLevel.ERROR, kind="circuit_break",
                **session.live_payload(price),
            )
        for sig in out.signals:
            insert_signal(db, self._run_id, sig)
            if sig.suppressed_by:
                self._emit(
                    EventType.LOG,
                    f"信号被抑制 [{sig.suppressed_by}]：{sig.side.value} "
                    f"reason={sig.reason} z={sig.z:+.2f}",
                    level=EventLevel.WARN, kind="signal_suppressed",
                    reason=sig.reason, suppressed_by=sig.suppressed_by, z=sig.z,
                )
            else:
                self._emit(
                    EventType.LOG,
                    f"信号 {sig.side.value} reason={sig.reason} z={sig.z:+.2f} "
                    f"price={sig.price:.4f}",
                    kind="signal", reason=sig.reason, z=sig.z, price=sig.price,
                )
        for tr in out.trades:
            self._trade_count += 1
            insert_trade(db, self._run_id, self._trade_count, tr)
            pnl_repr = "" if tr.side.value == "buy" and tr.realized_pnl == 0 else (
                f" pnl={tr.realized_pnl:+.2f}"
            )
            self._emit(
                EventType.LOG,
                f"{tr.side.value.upper()} {tr.qty} @ {tr.price:.4f} "
                f"fee={tr.fee:.2f}{pnl_repr} → 持仓 {tr.position_after}",
                kind="trade", side=tr.side.value, qty=tr.qty, price=tr.price,
                fee=tr.fee, realized_pnl=tr.realized_pnl,
                position_after=tr.position_after, cash_after=tr.cash_after,
                **session.live_payload(price),
            )

    # ------------------------------------------------------------------
    # 事件：落库 + 渲染（渲染失败降级 legacy，绝不崩 run）
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: EventType,
        message: str,
        *,
        level: EventLevel = EventLevel.INFO,
        **payload: Any,
    ) -> None:
        event = self.rt.emit(event_type, message, level=level, payload=payload)
        if self._events is not None:
            self._events.write(
                int(self.rt.clock.now_epoch()), event.type.value,
                str(event.level.value), message, payload,
            )
        try:
            self.renderer.handle(event)
        except Exception:  # noqa: BLE001 — §14：UI 异常 → 中途降级 legacy
            if not self._renderer_dead:
                self._degrade_renderer(None, self.rt.clock)
                try:
                    self.renderer.handle(event)
                except Exception:  # noqa: BLE001
                    pass

    def _degrade_renderer(self, meta: RunMeta | None, clock: MarketClock) -> None:
        self._renderer_dead = True
        try:
            self.renderer.finish()
        except Exception:  # noqa: BLE001
            pass
        fallback = LegacyStreamRenderer()
        if meta is not None:
            fallback.start(meta, clock)
        else:
            fallback._clock = clock  # 中途降级：跳过 start 头部，仅续流
        self.renderer = fallback
        self._renderer_dead = False

    def _params_snapshot(self, code: str) -> dict[str, Any]:
        from dataclasses import asdict  # noqa: PLC0415

        return {"code": code, **asdict(self.cfg)}


def _date_of(trade_date: str):
    from datetime import date  # noqa: PLC0415

    return date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:8]))


def _fmt_mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def _snapshot_progressed(prev: Snapshot | None, snap: Snapshot) -> bool:
    if prev is None:
        return True
    if snap.cum_vol > prev.cum_vol or snap.cum_amount > prev.cum_amount:
        return True
    if snap.trade_time and prev.trade_time and snap.trade_time != prev.trade_time:
        return True
    return False


def _snapshot_quality_error(snap: Snapshot, guard_bps: float) -> str | None:
    if snap.last <= 0:
        return f"last 非正数: {snap.last}"
    if snap.cum_vol < 0 or snap.cum_amount < 0:
        return f"累计量额为负: vol={snap.cum_vol}, amount={snap.cum_amount}"
    guard = guard_bps / 10_000.0
    if snap.high is not None and snap.high > 0 and snap.last > snap.high * (1.0 + guard):
        return f"last 高于 high 保护阈值: last={snap.last}, high={snap.high}"
    if snap.low is not None and snap.low > 0 and snap.last < snap.low * (1.0 - guard):
        return f"last 低于 low 保护阈值: last={snap.last}, low={snap.low}"
    if snap.cum_vol > 0 and snap.cum_amount <= 0:
        return f"有成交量但成交额非正: vol={snap.cum_vol}, amount={snap.cum_amount}"
    return None
