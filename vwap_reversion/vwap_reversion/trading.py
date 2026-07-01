"""TradingSession — signal → risk → execution 编排 + 腿账务（设计 §10.1 P2）.

每个 run 一个实例。daemon 在每根有效 bar 上调 :meth:`on_bar`，到
``eod_flat_time`` 调 :meth:`eod_flat`，收盘调 :meth:`summary` 出
``vwr_daily_summary`` 行。

账务口径：
* 策略盈亏 = 权益（cash + position×price）相对**初始权益**的变化。
  初始权益在锚定后确定：round_trip = initial_cash；base_position_t =
  initial_cash + base_shares×首个有效价（底仓本身的持有盈亏因此**计入**
  策略盈亏 —— 做 T 模式下底仓涨跌与做 T 收益共同构成当日结果；buy-and-hold
  基准对比留给报告呈现）。
* 每腿已实现盈亏 = (出场价 − 入场价) × qty × direction − 入场费 − 出场费
  （滑点已含在成交价里）。
* 熔断阈值 = −daily_loss_limit_pct% × initial_cash（绝对额，相对初始现金）。

已知限制（paper 阶段）：崩溃恢复（P1 §14）只重建 VWAP 引擎状态；上一个
aborted run 的虚拟持仓**不跨 run 延续**，新 run 一律从锚点（空仓/底仓）
起步。接真实 broker（P5）时改为查询真实持仓。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from .clock import parse_hhmm
from .engine.signal import (
    Intent,
    IntentKind,
    LegState,
    evaluate_flat,
    evaluate_leg,
)
from .execution.base import ExecutionRejected, Fill
from .execution.paper import PaperBroker
from .risk import RiskManager
from .schemas import IntervalBar, Side, Signal, Trade

if TYPE_CHECKING:  # pragma: no cover
    from .config import VwrConfig
    from .engine.vwap import BarMetrics


@dataclass
class BarOutcome:
    """一根 bar 的交易产出（daemon 据此落库 + emit 事件）。"""

    signals: list[Signal] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    circuit_tripped: bool = False   # 本根 bar 首次触发熔断


@dataclass
class ArmState:
    """V2 两段式入场的候选状态。direction=+1 低吸，-1 高抛底仓。"""

    direction: int
    armed_z: float
    extreme_price: float
    armed_ts: int


class TradingSession:
    def __init__(self, cfg: VwrConfig, *, code: str, trade_date: str) -> None:
        self.cfg = cfg
        self.code = code
        self.trade_date = trade_date
        self.broker = PaperBroker(
            cash=cfg.initial_cash,
            fee_bps=cfg.fee_bps,
            min_fee_per_trade=cfg.min_fee_per_trade,
            slippage_bps=cfg.slippage_bps,
        )
        self.risk = RiskManager(
            max_trades_per_day=cfg.max_trades_per_day,
            min_holding_seconds=cfg.min_holding_seconds,
            cooldown_seconds=cfg.cooldown_seconds,
            max_consecutive_losses=cfg.max_consecutive_losses,
            new_entry_cutoff_ts=_epoch_of_hhmm(
                trade_date, cfg.new_entry_cutoff_time, cfg.market_timezone
            ),
            kill_switch_enabled=cfg.kill_switch_enabled,
        )
        self.leg: LegState | None = None
        self.arm: ArmState | None = None
        self.eod_done = False

        self._anchor_qty = (
            cfg.base_shares if cfg.position_mode == "base_position_t" else 0
        )
        self._anchored = self._anchor_qty == 0
        self._initial_equity = cfg.initial_cash
        self._entry_fee = 0.0
        self._first_price: float | None = None
        self._last_price: float | None = None
        self._peak_equity = cfg.initial_cash
        self._max_dd_pct = 0.0
        self._closed_pnls: list[float] = []
        self._holding_secs: list[int] = []
        self._total_fee = 0.0
        self._total_slip = 0.0
        self._turnover = 0.0          # Σ 成交额（元）
        self._trade_seq = 0
        self._prev_vwap: float | None = None

    # ------------------------------------------------------------------
    # 对 daemon 的只读快照（实时面板 payload）
    # ------------------------------------------------------------------

    @property
    def n_fills(self) -> int:
        return self.risk.fills_today

    @property
    def last_price(self) -> float | None:
        return self._last_price

    def equity(self, price: float) -> float:
        return self.broker.cash + self.broker.position * price

    def strategy_pnl(self, price: float) -> float:
        return self.equity(price) - self._initial_equity

    def live_payload(self, price: float) -> dict[str, Any]:
        return {
            "position": self.broker.position,
            "cash": round(self.broker.cash, 2),
            "net_pnl": round(self.strategy_pnl(price), 2),
            "n_fills": self.n_fills,
            "circuit_broken": self.risk.circuit_broken,
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def on_bar(self, bar: IntervalBar, metrics: BarMetrics, *, in_warmup: bool) -> BarOutcome:
        out = BarOutcome()
        price, ts = bar.last, bar.ts
        self._ensure_anchor(price)
        self._last_price = price
        self._update_drawdown(price)
        if self.eod_done:
            return out

        # ---- 日亏熔断（先于一切信号；warmup 内也生效）（§13）----
        if not self.risk.circuit_broken and self.strategy_pnl(price) <= (
            -self.cfg.daily_loss_limit_pct / 100.0 * self.cfg.initial_cash
        ):
            self.risk.circuit_broken = True
            out.circuit_tripped = True
            if self.leg is not None:
                self._close_leg(out, metrics, price, ts, "circuit_break")
            return out

        if in_warmup or metrics.z is None:
            return out
        z = metrics.z
        vwap_slope_bps = self._vwap_slope_bps(metrics.vwap)

        if self.leg is None:
            intent = self._evaluate_flat(metrics, price, ts, z, vwap_slope_bps)
            if intent is None:
                return out
            self._open_leg(out, intent, metrics, price, ts, z)
        else:
            intent = evaluate_leg(
                self.leg, z, price,
                k_exit=self.cfg.band_k_exit, k_stop=self.cfg.band_k_stop,
                per_trade_stop_pct=self.cfg.per_trade_stop_pct,
            )
            if (
                intent is None
                and self.cfg.max_holding_seconds > 0
                and ts - self.leg.entry_ts >= self.cfg.max_holding_seconds
            ):
                side = Side.SELL if self.leg.direction > 0 else Side.BUY
                intent = Intent(IntentKind.CLOSE, side, "time_exit")
            if intent is None:
                return out
            suppressed = self.risk.check_close(intent.reason, self.leg, ts)
            sig = self._mk_signal(ts, intent, metrics, price, z, suppressed)
            out.signals.append(sig)
            if suppressed is None:
                self._close_leg(out, metrics, price, ts, intent.reason, signal_done=True)
        return out

    def eod_flat(self, price: float, ts: int) -> BarOutcome:
        """EOD 强平：平掉偏离腿、回到锚点，此后只采不交易（§13）。"""
        out = BarOutcome()
        self.risk.eod_halted = True
        if self.eod_done:
            return out
        self.eod_done = True
        self._ensure_anchor(price)
        self._last_price = price
        if self.leg is not None:
            self._close_leg(out, None, price, ts, "eod_flat")
        return out

    # ------------------------------------------------------------------
    # 收盘汇总（vwr_daily_summary 一行；P3 报告之源）
    # ------------------------------------------------------------------

    def summary(self, run_id: str) -> dict[str, Any]:
        last = self._last_price
        first = self._first_price
        closed = self._closed_pnls
        wins = [p for p in closed if p > 0]
        losses = [p for p in closed if p < 0]
        gross_gain = sum(wins)
        gross_loss = -sum(losses)
        return {
            "code": self.code,
            "trade_date": self.trade_date,
            "run_id": run_id,
            "n_trades": self.n_fills,
            "n_wins": len(wins),
            "win_rate": (len(wins) / len(closed)) if closed else None,
            "profit_factor": (gross_gain / gross_loss) if gross_loss > 0 else None,
            "gross_pnl": sum(closed),
            "net_pnl": self.strategy_pnl(last) if last is not None else 0.0,
            "total_fee": self._total_fee,
            "total_slippage": self._total_slip,
            "turnover": self._turnover,
            "max_drawdown": self._max_dd_pct,   # 峰值回撤百分比（% of peak equity）
            "avg_holding_seconds": (
                sum(self._holding_secs) / len(self._holding_secs)
                if self._holding_secs else None
            ),
            "final_cash": self.equity(last) if last is not None else self.broker.cash,
            "buy_hold_pnl": (
                self.cfg.initial_cash * (last / first - 1.0)
                if (first and last) else None
            ),
            "circuit_broken": int(self.risk.circuit_broken),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_anchor(self, price: float) -> None:
        if not self._anchored:
            # 底仓做 T：底仓视为既有资产注入，初始权益含底仓市值（模块 docstring）。
            self.broker.seed_position(self._anchor_qty)
            self._initial_equity = self.broker.cash + self._anchor_qty * price
            self._peak_equity = self._initial_equity
            self._anchored = True
        if self._first_price is None:
            self._first_price = price

    def _allow_short_leg(self) -> bool:
        return (
            self.cfg.position_mode == "base_position_t"
            and self.broker.position >= self.cfg.order_qty
        )

    def _evaluate_flat(
        self,
        metrics: BarMetrics,
        price: float,
        ts: int,
        z: float,
        vwap_slope_bps: float | None,
    ) -> Intent | None:
        if self.cfg.signal_version == "v1":
            return evaluate_flat(
                z, k_entry=self.cfg.band_k_entry, allow_short_leg=self._allow_short_leg()
            )

        k_entry = self._dynamic_entry_k(metrics)
        allow_short = self._allow_short_leg()
        if self.arm is None:
            if z <= -k_entry and not self._trend_blocks(direction=1, slope_bps=vwap_slope_bps):
                self.arm = ArmState(direction=1, armed_z=z, extreme_price=price, armed_ts=ts)
            elif (
                allow_short
                and z >= k_entry
                and not self._trend_blocks(direction=-1, slope_bps=vwap_slope_bps)
            ):
                self.arm = ArmState(direction=-1, armed_z=z, extreme_price=price, armed_ts=ts)
            return None

        arm = self.arm
        if arm.direction > 0:
            arm.extreme_price = min(arm.extreme_price, price)
            if z > -self.cfg.band_k_exit:
                self.arm = None
                return None
            if self._trend_blocks(direction=1, slope_bps=vwap_slope_bps):
                return None
            rebound_ok = price >= arm.extreme_price * (1.0 + self.cfg.min_rebound_bps / 10_000.0)
            z_ok = z >= arm.armed_z + self.cfg.confirm_z_recover
            if rebound_ok and z_ok:
                self.arm = None
                return Intent(IntentKind.OPEN, Side.BUY, "entry_long_confirmed")
            return None

        arm.extreme_price = max(arm.extreme_price, price)
        if z < self.cfg.band_k_exit:
            self.arm = None
            return None
        if not allow_short or self._trend_blocks(direction=-1, slope_bps=vwap_slope_bps):
            return None
        pullback_ok = price <= arm.extreme_price * (1.0 - self.cfg.min_rebound_bps / 10_000.0)
        z_ok = z <= arm.armed_z - self.cfg.confirm_z_recover
        if pullback_ok and z_ok:
            self.arm = None
            return Intent(IntentKind.OPEN, Side.SELL, "entry_short_t_confirmed")
        return None

    def _dynamic_entry_k(self, metrics: BarMetrics) -> float:
        k = self.cfg.band_k_entry
        if metrics.vwap > 0:
            sigma_bps = metrics.sigma / metrics.vwap * 10_000.0
            if sigma_bps >= self.cfg.high_vol_sigma_bps:
                k *= self.cfg.high_vol_entry_multiplier
        return k

    def _vwap_slope_bps(self, vwap: float) -> float | None:
        prev = self._prev_vwap
        self._prev_vwap = vwap
        if prev is None or prev <= 0:
            return None
        return (vwap / prev - 1.0) * 10_000.0

    def _trend_blocks(self, *, direction: int, slope_bps: float | None) -> bool:
        if slope_bps is None:
            return False
        guard = self.cfg.trend_guard_vwap_slope_bps
        if guard <= 0:
            return False
        if direction > 0:
            return slope_bps <= -guard
        return slope_bps >= guard

    def _open_leg(
        self, out: BarOutcome, intent: Intent, metrics: BarMetrics,
        price: float, ts: int, z: float,
    ) -> None:
        suppressed = self.risk.check_open(ts)
        if suppressed is None:
            try:
                fill = self.broker.fill(intent.side, self.cfg.order_qty, price, ts)
            except ExecutionRejected as e:
                suppressed = f"execution_rejected:{e}"
                fill = None
        else:
            fill = None
        out.signals.append(self._mk_signal(ts, intent, metrics, price, z, suppressed))
        if fill is None:
            return
        direction = 1 if intent.side is Side.BUY else -1
        self.arm = None
        self.leg = LegState(
            direction=direction, qty=fill.qty, entry_price=fill.price,
            entry_z=z, entry_ts=ts,
        )
        self._entry_fee = fill.fee
        self._book_fill(out, fill, realized=0.0)

    def _close_leg(
        self, out: BarOutcome, metrics: BarMetrics | None,
        price: float, ts: int, reason: str, *, signal_done: bool = False,
    ) -> None:
        leg = self.leg
        assert leg is not None
        side = Side.SELL if leg.direction > 0 else Side.BUY
        try:
            fill = self.broker.fill(side, leg.qty, price, ts)
        except ExecutionRejected as e:  # pragma: no cover — 腿账务保证不会发生
            out.signals.append(Signal(
                ts=ts, side=side, z=metrics.z if metrics else 0.0,
                vwap=metrics.vwap if metrics else 0.0,
                sigma=metrics.sigma if metrics else 0.0,
                price=price, reason=reason, suppressed_by=f"execution_rejected:{e}",
            ))
            return
        realized = (
            (fill.price - leg.entry_price) * leg.qty * leg.direction
            - self._entry_fee - fill.fee
        )
        if not signal_done:
            z_val = metrics.z if metrics is not None and metrics.z is not None else 0.0
            out.signals.append(Signal(
                ts=ts, side=side, z=z_val,
                vwap=metrics.vwap if metrics else 0.0,
                sigma=metrics.sigma if metrics else 0.0,
                price=price, reason=reason, suppressed_by=None,
            ))
        self._closed_pnls.append(realized)
        self._holding_secs.append(ts - leg.entry_ts)
        self.risk.note_close(ts, realized)
        self.leg = None
        self._entry_fee = 0.0
        self._book_fill(out, fill, realized=realized)

    def _book_fill(self, out: BarOutcome, fill: Fill, *, realized: float) -> None:
        self.risk.note_fill()
        self._total_fee += fill.fee
        self._total_slip += fill.slippage_cost
        self._turnover += fill.price * fill.qty
        self._trade_seq += 1
        out.trades.append(Trade(
            ts=fill.ts, side=fill.side, qty=fill.qty, price=fill.price,
            fee=fill.fee, slippage=fill.slippage_cost, realized_pnl=realized,
            cash_after=fill.cash_after, position_after=fill.position_after,
        ))

    def _mk_signal(
        self, ts: int, intent: Intent, metrics: BarMetrics,
        price: float, z: float, suppressed: str | None,
    ) -> Signal:
        return Signal(
            ts=ts, side=intent.side, z=z, vwap=metrics.vwap, sigma=metrics.sigma,
            price=price, reason=intent.reason, suppressed_by=suppressed,
        )

    def _update_drawdown(self, price: float) -> None:
        eq = self.equity(price)
        if eq > self._peak_equity:
            self._peak_equity = eq
        elif self._peak_equity > 0:
            dd = (self._peak_equity - eq) / self._peak_equity * 100.0
            if dd > self._max_dd_pct:
                self._max_dd_pct = dd

    @property
    def trade_seq(self) -> int:
        return self._trade_seq


def _epoch_of_hhmm(trade_date: str, hhmm: str, market_timezone: str) -> int:
    d = date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:8]))
    return int(datetime.combine(d, parse_hhmm(hhmm), tzinfo=ZoneInfo(market_timezone)).timestamp())
