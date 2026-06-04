"""Pure data structures（设计 §3.2 / §9）.

纯 stdlib dataclass —— ``validate_static`` 在 install 阶段 import 本模块以
触发字段校验，因此这里禁止引入 pandas / rich / deeptrade 等重依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class BarSource(Enum):
    REALTIME = "realtime"   # run daemon 累计快照差分
    BACKFILL = "backfill"   # （后续）tushare 历史分钟线


@dataclass(frozen=True)
class Snapshot:
    """rt_etf_k 一次轮询的原始累计读数（vwr_snapshots 一行）。

    字段名 ←→ rt_etf_k 列名映射：last←close（最新价）、cum_vol←vol（股）、
    cum_amount←amount（元）、num_trades←num（开盘以来成交笔数）。
    ts 取本地轮询时刻的 epoch 秒（采样主时间轴）；trade_time 为交易所侧
    时间字符串，仅审计留痕，不参与计算。
    """

    code: str
    trade_date: str          # YYYYMMDD（市场时区日历日）
    ts: int                  # epoch 秒（轮询时刻）
    last: float              # close — 最新价
    cum_vol: float           # vol — 当日累计成交量（股）
    cum_amount: float        # amount — 当日累计成交额（元）
    num_trades: float | None = None   # num — 开盘以来成交笔数
    pre_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    bid_volume1: float | None = None  # 委托买盘（股）
    ask_volume1: float | None = None  # 委托卖盘（股）
    trade_time: str | None = None     # 交易所侧时间（原样留痕）


@dataclass(frozen=True)
class IntervalBar:
    """归一化最小单元 —— 引擎唯一认识的输入（设计 §3.2）。

    实盘：相邻两条 Snapshot 差分构造；回测：直接读 vwr_bars。
    VWAP = cum_amount / cum_vol（精确）；带的方差累加器吃 interval_* 增量。
    """

    code: str
    trade_date: str
    ts: int                  # 区间结束时刻（epoch 秒）
    interval_vol: float      # Δv_i（股）
    interval_amount: float   # Δa_i（元）
    last: float              # 区间末最新价（信号 & 撮合参考）
    cum_vol: float           # V_t
    cum_amount: float        # A_t


@dataclass(frozen=True)
class Signal:
    """引擎产出的一次交易意图（vwr_signals 一行）。suppressed_by 由风控回填。"""

    ts: int
    side: Side
    z: float
    vwap: float
    sigma: float
    price: float
    reason: str
    suppressed_by: str | None = None


@dataclass(frozen=True)
class Trade:
    """一笔模拟成交（vwr_trades 一行）。"""

    ts: int
    side: Side
    qty: int
    price: float             # 含滑点的成交价
    fee: float
    slippage: float
    realized_pnl: float      # 卖出时结转；买入为 0
    cash_after: float
    position_after: int
