"""rt_etf_k 实盘快照源（tushare 文档 doc_id=400）.

字段映射（逐字）：close→last（最新价）、vol→cum_vol（股）、amount→cum_amount
（元）、num→num_trades（开盘以来成交笔数）、pre_close/open/high/low 原名、
bid_volume1/ask_volume1（委托买/卖盘，股）、trade_time（交易所侧时间，留痕）。

调用约定：
* **每次 force_sync=True** —— 实时数据绝不吃缓存（即便 hot_or_anns TTL 内）。
* 沪市 ETF（``.SH``）必须传 ``topic='HQ_FND_TICK'``；深市不传（官方示例）。
* ``ts`` 取本地轮询时刻 epoch（统一采样轴）；交易所时间仅存 trade_time。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..clock import MarketClock
from ..schemas import Snapshot

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.tushare_client import TushareClient


class EmptyQuote(Exception):
    """rt_etf_k 返回空表或找不到目标代码行。"""


def _f(row: Any, col: str) -> float | None:
    """容错取浮点：列缺失 / NaN / 非数值 → None。"""
    try:
        v = row[col]
    except (KeyError, IndexError):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN check


class TushareRealtimeSource:
    """单标的 rt_etf_k 轮询源。"""

    def __init__(self, tushare: TushareClient, code: str, clock: MarketClock) -> None:
        self._tushare = tushare
        self.code = code
        self._clock = clock
        self._params: dict[str, Any] = {"ts_code": code}
        if code.upper().endswith(".SH"):
            # 官方文档：取上海 ETF 时 topic 必须为 HQ_FND_TICK；深市示例不传。
            self._params["topic"] = "HQ_FND_TICK"

    def fetch(self) -> Snapshot:
        df = self._tushare.call("rt_etf_k", params=dict(self._params), force_sync=True)
        if df is None or df.empty:
            raise EmptyQuote(f"rt_etf_k 返回空表（ts_code={self.code}）")
        # 单标的查询正常只有一行；防御 ts_code 通配/多行的情况。
        if "ts_code" in df.columns:
            hit = df[df["ts_code"] == self.code]
            if hit.empty:
                raise EmptyQuote(f"rt_etf_k 结果中无 {self.code} 行（共 {len(df)} 行）")
            row = hit.iloc[0]
        else:
            row = df.iloc[0]

        last = _f(row, "close")
        cum_vol = _f(row, "vol")
        cum_amount = _f(row, "amount")
        if last is None or cum_vol is None or cum_amount is None:
            raise EmptyQuote(
                f"rt_etf_k 关键列缺失/非法：close={last} vol={cum_vol} amount={cum_amount}"
            )

        trade_time = row["trade_time"] if "trade_time" in row else None
        return Snapshot(
            code=self.code,
            trade_date=self._clock.today_str(),
            ts=int(self._clock.now_epoch()),
            last=last,
            cum_vol=cum_vol,
            cum_amount=cum_amount,
            num_trades=_f(row, "num"),
            pre_close=_f(row, "pre_close"),
            open=_f(row, "open"),
            high=_f(row, "high"),
            low=_f(row, "low"),
            bid_volume1=_f(row, "bid_volume1"),
            ask_volume1=_f(row, "ask_volume1"),
            trade_time=None if trade_time is None else str(trade_time),
        )
