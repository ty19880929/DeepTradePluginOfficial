"""信号意图（设计 §2.3）— 纯函数，无副作用、无框架依赖.

腿（leg）抽象：把两种持仓模式统一成「锚点 + 偏离腿」——
* ``round_trip``      锚点 = 0（空仓），只允许多头腿（A 股不能裸卖空）
* ``base_position_t`` 锚点 = base_shares，允许多头腿（低吸加仓）与
  空头腿（高抛减仓，实际是卖出既有持仓，不是裸空）

平腿优先级：硬止损 > 带止损 > 回归止盈。

**带止损细化**（相对设计 §2.3 的工程修正）：30s 采样下入场时 z 可能已深于
−k_stop（跳空进入），若按裸 ``z ≤ −k_stop`` 判会「入场即止损」。因此带止损
要求**偏离相对入场继续恶化**：``z ≤ −k_stop 且 z < entry_z``（空头腿对称）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..schemas import Side


class IntentKind(Enum):
    OPEN = "open"
    CLOSE = "close"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    side: Side           # 本次撮合方向（开多/平空=BUY；开空腿/平多=SELL）
    reason: str          # entry_long / entry_short_t / revert_exit / stop_hard / stop_band


@dataclass(frozen=True)
class LegState:
    """一条已开偏离腿。direction=+1 多头腿（先买后卖）/ −1 空头腿（先卖后买回）。"""

    direction: int
    qty: int
    entry_price: float   # 含滑点的实际入场价
    entry_z: float
    entry_ts: int


def evaluate_flat(
    z: float,
    *,
    k_entry: float,
    allow_short_leg: bool,
) -> Intent | None:
    """无腿时的开仓意图。``allow_short_leg`` 仅 base_position_t 且底仓足够时为真。"""
    if z <= -k_entry:
        return Intent(IntentKind.OPEN, Side.BUY, "entry_long")
    if allow_short_leg and z >= k_entry:
        return Intent(IntentKind.OPEN, Side.SELL, "entry_short_t")
    return None


def evaluate_leg(
    leg: LegState,
    z: float,
    price: float,
    *,
    k_exit: float,
    k_stop: float,
    per_trade_stop_pct: float,
) -> Intent | None:
    """持腿时的平仓意图。优先级：硬止损 > 带止损 > 回归止盈。"""
    stop_frac = per_trade_stop_pct / 100.0
    if leg.direction > 0:
        if price <= leg.entry_price * (1.0 - stop_frac):
            return Intent(IntentKind.CLOSE, Side.SELL, "stop_hard")
        if z <= -k_stop and z < leg.entry_z:
            return Intent(IntentKind.CLOSE, Side.SELL, "stop_band")
        if z >= -k_exit:
            return Intent(IntentKind.CLOSE, Side.SELL, "revert_exit")
        return None
    # 空头腿（高抛待低吸回补）
    if price >= leg.entry_price * (1.0 + stop_frac):
        return Intent(IntentKind.CLOSE, Side.BUY, "stop_hard")
    if z >= k_stop and z > leg.entry_z:
        return Intent(IntentKind.CLOSE, Side.BUY, "stop_band")
    if z <= k_exit:
        return Intent(IntentKind.CLOSE, Side.BUY, "revert_exit")
    return None
