"""risk — 风险信号 (design §5.3.7 + Appendix A).

Eight signals per the design table; each tracked as a :class:`RiskSignal`
with ``triggered`` / ``severity`` / ``detail`` / sample list. The full
review is :class:`RiskReview` (just the signal list — easy to feed to PR-4
LLM prompts and PR-5 :class:`RiskOutlookSection`).

The "高位回撤" signal needs a 60-day cumulative return baseline that PR-3
doesn't have; this PR computes "within-window cumulative" as the closest
honest substitute, and surfaces that in the ``detail`` string. PR-6 will
re-wire to a proper lookback once the data layer carries 60+ trailing days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .breadth import BreadthReview, _bind_in
from .capital import CapitalReview

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..universe import UniverseSnapshot
    from ..windows import Window


_HIGH_DROP_PCT = -7.0          # 单日跌幅阈值
_RECENT_CUM_PCT_TRIGGER = 80.0  # 区间累计涨幅触发位
_VOL_RATIO_TRIGGER = 2.0       # 量比阈值
_STAGNATION_PCT_TRIGGER = 1.0  # 滞涨 |pct_chg|
_INDEX_VOLUME_DROP_PCT = 15.0  # 大盘量价背离 缩量阈值
_BLOCK_TRADE_DISCOUNT_PCT = -5.0   # 大宗折价阈值（per row）
_MARGIN_BALANCE_PCT = 3.0      # 融资骤变
_WAN_PER_YI = 10_000.0


@dataclass(frozen=True)
class RiskSignal:
    name: str
    triggered: bool
    severity: str  # info / warning / critical
    detail: str
    affected_samples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RiskReview:
    signals: list[RiskSignal] = field(default_factory=list)


def compute_risk(
    db: Database,
    window: Window,
    universes: dict[str, UniverseSnapshot],
    *,
    breadth: BreadthReview,
    capital: CapitalReview,
) -> RiskReview:
    """Evaluate all 8 signals over the window."""
    anchor = window.anchor
    snap = universes.get(anchor)
    universe = snap.ts_codes if snap is not None else frozenset()
    signals: list[RiskSignal] = [
        _high_drop_after_rally(db, window=window, universe=universe),
        _stagnant_on_volume(db, anchor=anchor, universe=universe),
        _index_volume_divergence(breadth=breadth),
        _north_outflow(capital=capital),
        _limit_down_spread(breadth=breadth),
        _block_trade_discount(db, anchor=anchor),
        _margin_swing(db, window=window),
        _yaog_top(db, anchor=anchor, universe=universe),
    ]
    return RiskReview(signals=signals)


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------


def _high_drop_after_rally(
    db: Database, *, window: Window, universe: frozenset[str]
) -> RiskSignal:
    """单日大跌 + 窗内累涨过高 (proxy for design §5.3.7 高位回撤)."""
    name = "high_position_drop"
    if not universe or not window.trade_dates:
        return RiskSignal(name=name, triggered=False, severity="info",
                          detail="universe 或 window 为空，跳过")

    in_clause, params = _bind_in(universe)
    anchor = window.anchor

    # 当日下跌幅 ≤ _HIGH_DROP_PCT 的个股
    drop_rows = db.fetchall(
        f"""
        SELECT ts_code, pct_chg FROM mr_daily
        WHERE trade_date = ? AND ts_code IN {in_clause}
          AND pct_chg IS NOT NULL AND pct_chg <= ?
        """,
        [anchor, *params, _HIGH_DROP_PCT],
    )
    if not drop_rows:
        return RiskSignal(name=name, triggered=False, severity="info",
                          detail=f"anchor 日（{anchor}）无跌幅 ≤ {_HIGH_DROP_PCT}% 的样本")
    drop_codes = {str(r[0]) for r in drop_rows}

    # 窗内累计涨幅（用首日开盘价 → anchor 收盘价的简易回撤估算）
    cum_rows = db.fetchall(
        f"""
        WITH first_open AS (
            SELECT ts_code, open AS p0
            FROM mr_daily
            WHERE trade_date = ? AND ts_code IN {in_clause}
        ),
        last_close AS (
            SELECT ts_code, close AS p1
            FROM mr_daily
            WHERE trade_date = ? AND ts_code IN {in_clause}
        )
        SELECT f.ts_code, ((l.p1 - f.p0) / NULLIF(f.p0, 0)) * 100.0 AS cum_pct
        FROM first_open f JOIN last_close l USING (ts_code)
        WHERE ((l.p1 - f.p0) / NULLIF(f.p0, 0)) * 100.0 >= ?
        """,
        [
            window.trade_dates[0], *params,
            anchor, *params,
            _RECENT_CUM_PCT_TRIGGER,
        ],
    )
    hot_codes = {str(r[0]) for r in cum_rows}
    overlap = drop_codes & hot_codes
    triggered = len(overlap) > 0
    severity = "warning" if triggered and len(overlap) >= 3 else "info"
    detail = (
        f"窗内累计涨幅 ≥ {_RECENT_CUM_PCT_TRIGGER}% 且 anchor 日跌幅 ≤ "
        f"{_HIGH_DROP_PCT}% 的高位股：{len(overlap)} 只"
    )
    return RiskSignal(
        name=name, triggered=triggered, severity=severity, detail=detail,
        affected_samples=sorted(overlap)[:5],
    )


def _stagnant_on_volume(
    db: Database, *, anchor: str, universe: frozenset[str]
) -> RiskSignal:
    """放量滞涨 — anchor 日 volume_ratio > 2 且 |pct_chg| < 1%。"""
    name = "stagnant_on_high_volume"
    if not universe:
        return RiskSignal(name=name, triggered=False, severity="info", detail="universe 为空")
    in_clause, params = _bind_in(universe)
    rows = db.fetchall(
        f"""
        SELECT db.ts_code FROM mr_daily_basic db
        JOIN mr_daily d ON db.ts_code = d.ts_code AND db.trade_date = d.trade_date
        WHERE db.trade_date = ? AND db.ts_code IN {in_clause}
          AND db.volume_ratio IS NOT NULL AND db.volume_ratio > ?
          AND d.pct_chg IS NOT NULL AND ABS(d.pct_chg) < ?
        """,
        [anchor, *params, _VOL_RATIO_TRIGGER, _STAGNATION_PCT_TRIGGER],
    )
    n = len(rows)
    triggered = n >= 5
    severity = "warning" if n >= 20 else "info"
    return RiskSignal(
        name=name,
        triggered=triggered,
        severity=severity,
        detail=f"anchor 日 量比>{_VOL_RATIO_TRIGGER} 且 |pct_chg|<{_STAGNATION_PCT_TRIGGER}% 的样本：{n}",
        affected_samples=[str(r[0]) for r in rows[:5]],
    )


def _index_volume_divergence(*, breadth: BreadthReview) -> RiskSignal:
    """大盘量价背离 — anchor 日 上证涨且总成交额环比缩量 > 15%。"""
    name = "index_volume_divergence"
    if len(breadth.series) < 2:
        return RiskSignal(name=name, triggered=False, severity="info",
                          detail="窗口少于 2 个交易日，无法判定环比")
    today = breadth.series[-1]
    yesterday = breadth.series[-2]
    shsz_change = today.index_returns.get("000001.SH", 0.0)
    if today.total_amount_yi <= 0 or yesterday.total_amount_yi <= 0:
        return RiskSignal(name=name, triggered=False, severity="info",
                          detail="成交额数据不全")
    delta_pct = (today.total_amount_yi - yesterday.total_amount_yi) / yesterday.total_amount_yi * 100.0
    triggered = (shsz_change > 0 and delta_pct < -_INDEX_VOLUME_DROP_PCT) or (
        shsz_change < 0 and delta_pct > _INDEX_VOLUME_DROP_PCT
    )
    return RiskSignal(
        name=name,
        triggered=triggered,
        severity="warning" if triggered else "info",
        detail=(
            f"上证 {shsz_change:+.2f}%；总成交额环比 {delta_pct:+.1f}%"
        ),
    )


def _north_outflow(*, capital: CapitalReview) -> RiskSignal:
    name = "north_capital_outflow"
    if not capital.north_series:
        return RiskSignal(name=name, triggered=False, severity="info",
                          detail="mr_moneyflow_hsgt 无数据")
    anchor_row = capital.north_series[-1]
    triggered_today = (anchor_row.north_money_yi or 0.0) < 0
    triggered_window = capital.north_total_yi < 0
    triggered = triggered_today or triggered_window
    severity = "warning" if (triggered_today and triggered_window) else "info"
    return RiskSignal(
        name=name,
        triggered=triggered,
        severity=severity,
        detail=(
            f"anchor 日北向 {anchor_row.north_money_yi if anchor_row.north_money_yi is not None else 'N/A'} 亿；"
            f"窗内累计 {capital.north_total_yi:+.1f} 亿"
        ),
    )


def _limit_down_spread(*, breadth: BreadthReview) -> RiskSignal:
    name = "limit_down_spread"
    if not breadth.series:
        return RiskSignal(name=name, triggered=False, severity="info", detail="无 breadth 数据")
    anchor = breadth.series[-1]
    triggered = anchor.n_limit_down >= 10
    severity = "critical" if anchor.n_limit_down >= 30 else (
        "warning" if anchor.n_limit_down >= 15 else "info"
    )
    return RiskSignal(
        name=name,
        triggered=triggered,
        severity=severity,
        detail=f"anchor 日跌停家数：{anchor.n_limit_down}",
    )


def _block_trade_discount(db: Database, *, anchor: str) -> RiskSignal:
    """大宗折价 — anchor 日大宗交易折价（成交价 < pre_close）总额。"""
    name = "block_trade_discount"
    rows = db.fetchall(
        """
        SELECT bt.ts_code, bt.price, bt.amount, d.pre_close
        FROM mr_block_trade bt
        LEFT JOIN mr_daily d ON bt.ts_code = d.ts_code AND bt.trade_date = d.trade_date
        WHERE bt.trade_date = ? AND bt.price IS NOT NULL AND bt.amount IS NOT NULL
        """,
        [anchor],
    )
    discounted_yi = 0.0
    sample: list[str] = []
    for r in rows:
        ts_code = str(r[0])
        price = float(r[1] or 0)
        amount_wan = float(r[2] or 0)
        pre_close = r[3]
        if pre_close is None or pre_close == 0:
            continue
        discount_pct = (price - float(pre_close)) / float(pre_close) * 100.0
        if discount_pct <= _BLOCK_TRADE_DISCOUNT_PCT:
            discounted_yi += amount_wan / _WAN_PER_YI
            if len(sample) < 5:
                sample.append(ts_code)
    triggered = discounted_yi >= 5.0
    return RiskSignal(
        name=name,
        triggered=triggered,
        severity="warning" if discounted_yi >= 20.0 else "info",
        detail=f"anchor 日折价 ≥{abs(_BLOCK_TRADE_DISCOUNT_PCT)}% 的大宗成交合计：{discounted_yi:.1f} 亿",
        affected_samples=sample,
    )


def _margin_swing(db: Database, *, window: Window) -> RiskSignal:
    """融资骤变 — anchor 日 vs 窗内前一交易日 融资余额变化幅度。"""
    name = "margin_balance_swing"
    if len(window.trade_dates) < 2:
        return RiskSignal(name=name, triggered=False, severity="info",
                          detail="窗口少于 2 个交易日")
    today, yesterday = window.trade_dates[-1], window.trade_dates[-2]
    rows = db.fetchall(
        """
        SELECT trade_date, SUM(COALESCE(rzye, 0)) FROM mr_margin
        WHERE trade_date IN (?, ?)
        GROUP BY trade_date
        """,
        [today, yesterday],
    )
    by_date = {str(r[0]): float(r[1] or 0) for r in rows}
    if today not in by_date or yesterday not in by_date or by_date[yesterday] == 0:
        return RiskSignal(name=name, triggered=False, severity="info",
                          detail="融资余额数据不全")
    delta_pct = (by_date[today] - by_date[yesterday]) / by_date[yesterday] * 100.0
    triggered = abs(delta_pct) >= _MARGIN_BALANCE_PCT
    return RiskSignal(
        name=name,
        triggered=triggered,
        severity="warning" if triggered else "info",
        detail=f"融资余额环比 {delta_pct:+.2f}%",
    )


def _yaog_top(
    db: Database, *, anchor: str, universe: frozenset[str]
) -> RiskSignal:
    """妖股见顶 — anchor 日 5+ 连板 且当日存在炸板 ('Z')。"""
    name = "yaog_topping"
    if not universe:
        return RiskSignal(name=name, triggered=False, severity="info", detail="universe 为空")
    in_clause, params = _bind_in(universe)
    rows = db.fetchall(
        f"""
        SELECT s.ts_code FROM mr_limit_step s
        JOIN mr_limit_list_d l
          ON s.ts_code = l.ts_code AND s.trade_date = l.trade_date
        WHERE s.trade_date = ? AND s.ts_code IN {in_clause}
          AND s.nums >= 5 AND l."limit" = 'Z'
        """,
        [anchor, *params],
    )
    n = len(rows)
    triggered = n > 0
    return RiskSignal(
        name=name,
        triggered=triggered,
        severity="warning" if n >= 1 else "info",
        detail=f"anchor 日 5+ 连板 + 炸板个股：{n}",
        affected_samples=[str(r[0]) for r in rows[:5]],
    )
