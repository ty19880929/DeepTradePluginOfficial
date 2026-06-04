"""Feed 抽象 + 快照差分器（设计 §3.1 / §3.2）.

* :class:`SnapshotSource` —— 实盘数据源协议：``fetch()`` 返回一条当日累计
  快照（失败抛异常，由 daemon 统一退避重试）。
* :class:`BarBuilder` —— 相邻快照差分出 :class:`IntervalBar`：
    - 首条快照：区间 = 开盘至今（含集合竞价量）；
    - ``Δv == 0``（无新成交/午休/停牌）：返回 None，不产 bar；
    - 累计量回退（理论上盘中不会发生）：丢弃本条并要求重新对齐，防止
      跨日残留/上游异常污染引擎；
    - 崩溃恢复：:meth:`prime` 用最后一条已落库快照对齐基线，避免重启后
      首条 bar 双计开盘以来的量。

纯 stdlib，无框架依赖。
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import IntervalBar, Snapshot

_EPS = 1e-9


class SnapshotSource(Protocol):
    """实盘快照源。实现：feed/tushare_realtime.py。"""

    def fetch(self) -> Snapshot:
        """拉取一条当日累计快照。网络/接口失败直接抛异常。"""
        ...


class CumulativeRegression(Exception):
    """累计量回退（cum_vol/cum_amount 比基线小）—— 疑似跨日或上游异常。"""


class BarBuilder:
    """相邻累计快照 → 区间 bar 差分器。每个交易日一个实例。"""

    def __init__(self) -> None:
        self._prev: Snapshot | None = None

    @property
    def prev(self) -> Snapshot | None:
        return self._prev

    def prime(self, snapshot: Snapshot) -> None:
        """崩溃恢复：以最后一条已落库快照为差分基线（不产 bar）。"""
        self._prev = snapshot

    def push(self, snap: Snapshot) -> IntervalBar | None:
        prev = self._prev
        if prev is not None and (
            snap.cum_vol < prev.cum_vol - _EPS or snap.cum_amount < prev.cum_amount - _EPS
        ):
            # 不更新基线、不产 bar —— 让调用方决定（emit WARN 后丢弃本条）。
            raise CumulativeRegression(
                f"cum_vol {prev.cum_vol}→{snap.cum_vol}, "
                f"cum_amount {prev.cum_amount}→{snap.cum_amount}"
            )

        if prev is None:
            dv, da = snap.cum_vol, snap.cum_amount  # 首条：开盘至今
        else:
            dv = snap.cum_vol - prev.cum_vol
            da = snap.cum_amount - prev.cum_amount

        self._prev = snap
        if dv <= _EPS:
            return None  # 无新成交（午休/停牌/数据未动）
        return IntervalBar(
            code=snap.code,
            trade_date=snap.trade_date,
            ts=snap.ts,
            interval_vol=dv,
            interval_amount=da,
            last=snap.last,
            cum_vol=snap.cum_vol,
            cum_amount=snap.cum_amount,
        )
