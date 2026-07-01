"""TradingSession 信号→风控→撮合 编排（P2 验收项核心）.

直接构造 IntervalBar + BarMetrics 喂 on_bar，逐场景验证状态机与账务。
fee 5bps / slip 10bps 取好手算的值。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from vwap_reversion.config import VwrConfig
from vwap_reversion.engine.vwap import BarMetrics
from vwap_reversion.schemas import IntervalBar, Side
from vwap_reversion.trading import TradingSession

DAY = "20260603"
CODE = "159518.SZ"
SH = ZoneInfo("Asia/Shanghai")


def sh_epoch(hh: int, mm: int, ss: int = 0) -> int:
    return int(datetime(2026, 6, 3, hh, mm, ss, tzinfo=SH).timestamp())


def cfg(**kw) -> VwrConfig:
    base = dict(
        initial_cash=10_000.0, order_qty=100, fee_bps=5.0, slippage_bps=10.0,
        band_k_entry=2.0, band_k_exit=0.3, band_k_stop=3.5,
        min_holding_seconds=60, cooldown_seconds=60, max_trades_per_day=10,
        per_trade_stop_pct=0.8, daily_loss_limit_pct=1.5,
    )
    base.update(kw)
    return replace(VwrConfig(), **base)


def bar(ts: int, price: float) -> IntervalBar:
    return IntervalBar(
        code=CODE, trade_date=DAY, ts=ts, interval_vol=1000,
        interval_amount=1000 * price, last=price,
        cum_vol=ts * 100.0, cum_amount=ts * 100.0,
    )


def m(z: float | None, vwap: float = 1.0, sigma: float = 0.01) -> BarMetrics:
    return BarMetrics(
        vwap=vwap, sigma=sigma,
        band_upper=vwap + 2 * sigma, band_lower=vwap - 2 * sigma, z=z,
    )


def session(**kw) -> TradingSession:
    return TradingSession(cfg(**kw), code=CODE, trade_date=DAY)


def open_leg(s: TradingSession, *, ts: int = 1000, price: float = 1.0,
             z: float = -2.5):
    out = s.on_bar(bar(ts, price), m(z), in_warmup=False)
    assert s.leg is not None, "前置：开腿失败"
    return out


# ---------------------------------------------------------------------------
# 入场
# ---------------------------------------------------------------------------


def test_entry_long_books_signal_and_trade() -> None:
    s = session()
    out = open_leg(s)
    assert [sig.reason for sig in out.signals] == ["entry_long"]
    assert out.signals[0].suppressed_by is None
    assert len(out.trades) == 1
    tr = out.trades[0]
    assert tr.side is Side.BUY and tr.qty == 100
    assert tr.price == pytest.approx(1.001)   # 含滑点
    assert tr.realized_pnl == 0.0
    assert s.broker.position == 100


def test_no_entry_in_warmup_or_z_none_or_small_z() -> None:
    s = session()
    assert s.on_bar(bar(1, 1.0), m(-5.0), in_warmup=True).signals == []
    assert s.on_bar(bar(2, 1.0), m(None), in_warmup=False).signals == []
    assert s.on_bar(bar(3, 1.0), m(-1.9), in_warmup=False).signals == []
    assert s.leg is None


def test_signal_v2_arms_then_confirms_long_entry() -> None:
    s = session(
        signal_version="v2",
        confirm_z_recover=0.3,
        min_rebound_bps=5.0,
        trend_guard_vwap_slope_bps=0.0,
        high_vol_sigma_bps=9999.0,
    )
    out = s.on_bar(bar(1000, 0.9900), m(-2.5), in_warmup=False)
    assert out.signals == [] and out.trades == []
    assert s.arm is not None

    out = s.on_bar(bar(1060, 0.9908), m(-2.1), in_warmup=False)
    assert [sig.reason for sig in out.signals] == ["entry_long_confirmed"]
    assert out.trades and s.leg is not None


def test_signal_v2_blocks_adverse_vwap_trend() -> None:
    s = session(
        signal_version="v2",
        trend_guard_vwap_slope_bps=1.0,
        high_vol_sigma_bps=9999.0,
    )
    # Prime prev VWAP.
    assert s.on_bar(bar(1000, 1.0), m(-1.0, vwap=1.0000), in_warmup=False).signals == []
    out = s.on_bar(bar(1060, 0.99), m(-2.5, vwap=0.9990), in_warmup=False)
    assert out.signals == [] and out.trades == []
    assert s.arm is None


def test_signal_v2_high_vol_requires_wider_entry() -> None:
    s = session(
        signal_version="v2",
        high_vol_sigma_bps=50.0,
        high_vol_entry_multiplier=1.5,
        trend_guard_vwap_slope_bps=0.0,
    )
    out = s.on_bar(bar(1000, 0.99), m(-2.5, vwap=1.0, sigma=0.01), in_warmup=False)
    assert out.signals == [] and s.arm is None  # dynamic k = 3.0
    out = s.on_bar(bar(1060, 0.98), m(-3.2, vwap=1.0, sigma=0.01), in_warmup=False)
    assert out.signals == [] and s.arm is not None


def test_round_trip_never_shorts_on_high_z() -> None:
    s = session()  # round_trip：z 高位空仓不开空腿
    out = s.on_bar(bar(1, 1.02), m(+5.0), in_warmup=False)
    assert out.signals == [] and out.trades == []


def test_held_leg_no_pyramiding() -> None:
    s = session()
    open_leg(s)
    out = s.on_bar(bar(1030, 0.999), m(-2.6), in_warmup=False)  # 仍深负
    assert out.trades == []  # 不加仓、不重复入场
    assert s.broker.position == 100


def test_max_holding_seconds_time_exit() -> None:
    s = session(max_holding_seconds=60, min_holding_seconds=0)
    open_leg(s, ts=1000, price=1.0)
    out = s.on_bar(bar(1061, 0.999), m(-1.0), in_warmup=False)
    assert [sig.reason for sig in out.signals] == ["time_exit"]
    assert out.trades and s.leg is None


# ---------------------------------------------------------------------------
# 平仓三优先级
# ---------------------------------------------------------------------------


def test_revert_exit_pnl_exact() -> None:
    s = session()
    open_leg(s, ts=1000, price=1.0)                      # buy @1.001, fee .05005
    out = s.on_bar(bar(1100, 1.01), m(-0.2), in_warmup=False)
    assert [sig.reason for sig in out.signals] == ["revert_exit"]
    tr = out.trades[0]
    assert tr.side is Side.SELL
    expected = (1.00899 - 1.001) * 100 - 0.05005 - 1.00899 * 100 * 5e-4
    assert tr.realized_pnl == pytest.approx(expected)
    assert s.leg is None and s.broker.position == 0


def test_hard_stop_fires_within_min_holding() -> None:
    s = session()
    open_leg(s, ts=1000, price=1.0)  # entry 1.001；硬止损线 = ×(1−0.8%) ≈ 0.99299
    out = s.on_bar(bar(1010, 0.992), m(-3.0), in_warmup=False)  # 仅持 10s
    assert [sig.reason for sig in out.signals] == ["stop_hard"]
    assert out.signals[0].suppressed_by is None  # 止损不受 min_holding 防抖
    assert out.trades and out.trades[0].realized_pnl < 0
    assert s.leg is None


def test_band_stop_requires_worse_than_entry_z() -> None:
    s = session()
    open_leg(s, ts=1000, price=1.0, z=-4.0)  # 跳空入场 z=-4（已 < −k_stop）
    # z=-3.6 ≤ −k_stop 但未劣于入场 −4.0 → 不止损（防「入场即止损」）
    out = s.on_bar(bar(1100, 0.998), m(-3.6), in_warmup=False)
    assert out.signals == []
    # z=-4.5 劣于入场 → 带止损
    out = s.on_bar(bar(1200, 0.998), m(-4.5), in_warmup=False)
    assert [sig.reason for sig in out.signals] == ["stop_band"]
    assert s.leg is None


def test_min_holding_suppresses_revert_only() -> None:
    s = session()
    open_leg(s, ts=1000, price=1.0)
    out = s.on_bar(bar(1030, 1.005), m(-0.1), in_warmup=False)  # 持仓 30s < 60s
    assert out.signals[0].suppressed_by == "min_holding"
    assert out.trades == [] and s.leg is not None
    out = s.on_bar(bar(1061, 1.005), m(-0.1), in_warmup=False)  # 61s → 放行
    assert out.signals[0].suppressed_by is None
    assert s.leg is None


# ---------------------------------------------------------------------------
# 风控闸门
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_reentry() -> None:
    s = session()
    open_leg(s, ts=1000, price=1.0)
    s.on_bar(bar(1100, 1.01), m(-0.2), in_warmup=False)         # 平腿 @1100
    out = s.on_bar(bar(1130, 1.0), m(-2.5), in_warmup=False)    # 30s 后想再开
    assert out.signals[0].suppressed_by == "cooldown"
    out = s.on_bar(bar(1161, 1.0), m(-2.5), in_warmup=False)    # 61s 后放行
    assert out.signals[0].suppressed_by is None and s.leg is not None


def test_max_trades_leaves_quota_for_close() -> None:
    s = session(max_trades_per_day=3, cooldown_seconds=0, min_holding_seconds=0)
    open_leg(s, ts=1000)                                        # fill #1
    s.on_bar(bar(1010, 1.01), m(-0.2), in_warmup=False)         # fill #2（平）
    # fills=2 ≥ max−1=2 → 不再开新腿（保证已开腿永远有名额平掉）
    out = s.on_bar(bar(1020, 1.0), m(-2.5), in_warmup=False)
    assert out.signals[0].suppressed_by == "max_trades"
    assert s.n_fills == 2


def test_circuit_breaker_flattens_and_halts() -> None:
    s = session(order_qty=5000, daily_loss_limit_pct=1.5)  # 阈值 = −150 元
    out = s.on_bar(bar(1000, 1.0), m(-2.5), in_warmup=False)
    assert s.broker.position == 5000
    out = s.on_bar(bar(1060, 0.96), m(-6.0), in_warmup=False)  # 浮亏 ≈ −205
    assert out.circuit_tripped
    assert [sig.reason for sig in out.signals] == ["circuit_break"]
    assert s.leg is None and s.broker.position == 0
    # 此后开仓一律被抑制
    out = s.on_bar(bar(1200, 0.96), m(-5.0), in_warmup=False)
    assert out.signals[0].suppressed_by == "circuit_breaker"
    assert int(s.risk.circuit_broken) == 1


def test_kill_switch_suppresses_new_open_only() -> None:
    s = session(kill_switch_enabled=True)
    out = s.on_bar(bar(1000, 1.0), m(-2.5), in_warmup=False)
    assert out.signals[0].suppressed_by == "kill_switch"
    assert out.trades == [] and s.leg is None


def test_entry_cutoff_suppresses_new_open_only() -> None:
    s = session(new_entry_cutoff_time="14:40")
    out = s.on_bar(bar(sh_epoch(14, 41), 1.0), m(-2.5), in_warmup=False)
    assert out.signals[0].suppressed_by == "entry_cutoff"
    assert out.trades == []


def test_consecutive_losses_suppress_new_open_after_threshold() -> None:
    s = session(
        max_consecutive_losses=2,
        cooldown_seconds=0,
        min_holding_seconds=0,
        per_trade_stop_pct=0.5,
    )
    open_leg(s, ts=sh_epoch(10, 0), price=1.0)
    s.on_bar(bar(sh_epoch(10, 1), 0.99), m(-4.0), in_warmup=False)
    open_leg(s, ts=sh_epoch(10, 2), price=1.0)
    s.on_bar(bar(sh_epoch(10, 3), 0.99), m(-4.0), in_warmup=False)

    out = s.on_bar(bar(sh_epoch(10, 4), 1.0), m(-2.5), in_warmup=False)
    assert out.signals[0].suppressed_by == "consecutive_losses"
    assert out.trades == []


def test_insufficient_cash_suppressed_not_crash() -> None:
    s = session(initial_cash=50.0)
    out = s.on_bar(bar(1000, 1.0), m(-2.5), in_warmup=False)
    assert out.signals[0].suppressed_by.startswith("execution_rejected")
    assert out.trades == [] and s.leg is None


# ---------------------------------------------------------------------------
# EOD / 底仓做 T / 汇总
# ---------------------------------------------------------------------------


def test_eod_flat_closes_and_halts() -> None:
    s = session()
    open_leg(s, ts=1000, price=1.0)
    out = s.eod_flat(1.002, ts=2000)
    assert [sig.reason for sig in out.signals] == ["eod_flat"]
    assert s.broker.position == 0 and s.eod_done
    # EOD 后 on_bar 完全静默
    out = s.on_bar(bar(2100, 0.95), m(-9.0), in_warmup=False)
    assert out.signals == [] and out.trades == []


def test_base_position_t_short_leg_round_trip() -> None:
    s = session(position_mode="base_position_t", base_shares=1000)
    # 首 bar 锚定底仓；z 高位 → 高抛 100
    out = s.on_bar(bar(1000, 1.02), m(+2.5), in_warmup=False)
    assert [sig.reason for sig in out.signals] == ["entry_short_t"]
    assert out.trades[0].side is Side.SELL
    assert s.broker.position == 900
    # 回归 → 低吸回补，回到底仓；卖高买低 → 盈利
    out = s.on_bar(bar(1100, 1.00), m(+0.2), in_warmup=False)
    assert [sig.reason for sig in out.signals] == ["revert_exit"]
    assert out.trades[0].side is Side.BUY
    assert s.broker.position == 1000
    assert out.trades[0].realized_pnl > 0


def test_round_trip_anchor_is_zero_no_seed() -> None:
    s = session()
    s.on_bar(bar(1000, 1.0), m(0.0), in_warmup=False)
    assert s.broker.position == 0
    assert s.equity(1.0) == pytest.approx(10_000.0)


def test_summary_fields() -> None:
    s = session()
    open_leg(s, ts=1000, price=1.0)
    s.on_bar(bar(1100, 1.01), m(-0.2), in_warmup=False)  # 盈利平腿
    row = s.summary("run-x")
    assert row["n_trades"] == 2
    assert row["n_wins"] == 1 and row["win_rate"] == 1.0
    assert row["profit_factor"] is None        # 无亏损腿 → NULL
    assert row["gross_pnl"] > 0
    assert row["net_pnl"] == pytest.approx(s.strategy_pnl(1.01))
    assert row["total_fee"] > 0 and row["total_slippage"] > 0
    assert row["turnover"] == pytest.approx(1.001 * 100 + 1.00899 * 100)
    assert row["final_cash"] == pytest.approx(s.equity(1.01))
    assert row["circuit_broken"] == 0
    assert row["avg_holding_seconds"] == pytest.approx(100)
    assert row["buy_hold_pnl"] == pytest.approx(10_000 * (1.01 / 1.0 - 1))
