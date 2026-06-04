"""VWAP + 成交量加权 σ 在线累加器（设计 §2.1 / §2.2）.

全部状态只有三个标量：

* ``V_t`` —— 当日累计成交量（直接取 bar.cum_vol，**精确**）
* ``A_t`` —— 当日累计成交额（直接取 bar.cum_amount，**精确**）
* ``Q_t = Σ_i Δa_i²/Δv_i`` —— 方差累加器（吃区间增量；漏采样仅损带精度）

    VWAP_t = A_t / V_t
    Var_t  = Q_t / V_t − VWAP_t²
    σ_t    = sqrt(max(0, Var_t))
    z_t    = (last − VWAP_t) / σ_t      （σ≤0 时为 None）

崩溃恢复：:meth:`rebuild` 用已落库的 bars 重放即可还原全部状态（设计 §14）。
纯 stdlib，无框架依赖。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..schemas import IntervalBar

# 累计量出现负增长视为数据异常的容差（股）。集合竞价修正/接口抖动给一点余量。
_EPS = 1e-9


@dataclass(frozen=True)
class BarMetrics:
    """一根 bar 推进后的引擎读数（落 vwr_bars 的 vwap/sigma/band/z 列）。"""

    vwap: float
    sigma: float
    band_upper: float
    band_lower: float
    z: float | None          # σ≤0（如首根 bar / 单一价位）时为 None


class VwapEngine:
    """当日 session 锚定的 VWAP/σ 引擎。每个交易日一个实例（不跨日）。"""

    def __init__(self, *, band_k: float) -> None:
        if band_k <= 0:
            raise ValueError(f"band_k 必须 > 0（当前 {band_k}）")
        self.band_k = band_k
        self._v: float = 0.0   # V_t
        self._a: float = 0.0   # A_t
        self._q: float = 0.0   # Q_t
        self._n_bars: int = 0

    # ---- 推进 -----------------------------------------------------------

    def push(self, bar: IntervalBar) -> BarMetrics:
        """吃进一根区间 bar，返回该时点的引擎读数。

        约定（由 BarBuilder 保证）：``bar.interval_vol > 0``；``cum_*`` 单调
        不减。这里再防御一层 —— 违约 bar 直接拒绝，避免污染 Q_t。
        """
        if bar.interval_vol <= _EPS:
            raise ValueError(f"interval_vol 必须 > 0（当前 {bar.interval_vol}）")
        if bar.cum_vol < self._v - _EPS or bar.cum_amount < self._a - _EPS:
            raise ValueError(
                f"累计量回退：cum_vol {self._v}→{bar.cum_vol}, "
                f"cum_amount {self._a}→{bar.cum_amount}（疑似跨日/数据异常）"
            )
        self._q += (bar.interval_amount * bar.interval_amount) / bar.interval_vol
        self._v = bar.cum_vol
        self._a = bar.cum_amount
        self._n_bars += 1
        return self.metrics(bar.last)

    def rebuild(self, bars: Iterable[IntervalBar]) -> int:
        """崩溃恢复：按 ts 顺序重放已落库 bars，返回重放根数。"""
        n = 0
        for bar in bars:
            self.push(bar)
            n += 1
        return n

    # ---- 读数 -----------------------------------------------------------

    @property
    def n_bars(self) -> int:
        return self._n_bars

    @property
    def cum_vol(self) -> float:
        return self._v

    @property
    def cum_amount(self) -> float:
        return self._a

    @property
    def vwap(self) -> float:
        if self._v <= _EPS:
            raise ValueError("尚无成交量，VWAP 未定义（先 push 至少一根 bar）")
        return self._a / self._v

    @property
    def sigma(self) -> float:
        if self._v <= _EPS:
            return 0.0
        var = self._q / self._v - self.vwap * self.vwap
        return math.sqrt(var) if var > 0 else 0.0

    def z(self, price: float) -> float | None:
        s = self.sigma
        if s <= 0:
            return None
        return (price - self.vwap) / s

    def metrics(self, price: float) -> BarMetrics:
        vwap = self.vwap
        sigma = self.sigma
        return BarMetrics(
            vwap=vwap,
            sigma=sigma,
            band_upper=vwap + self.band_k * sigma,
            band_lower=vwap - self.band_k * sigma,
            z=self.z(price),
        )
