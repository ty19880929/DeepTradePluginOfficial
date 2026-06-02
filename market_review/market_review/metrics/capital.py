"""capital — 多口径资金流 (design §5.3.3 + Appendix A).

Six calibers in one :class:`CapitalReview` result:

1. **北向** — :class:`NorthFlowRow` (daily series + window total)
2. **大盘** — :class:`MktFlowRow` (daily 主力 / 散户净额)
3. **行业** — :class:`IndustryFlowRow` (THS 行业 Top / Bottom over the window)
4. **概念** — :class:`IndustryFlowRow` (THS 概念 Top / Bottom over the window)
5. **个股** — :class:`StockFlowRow` (Top 20 / Bottom 20 in universe)
6. **龙虎榜** — :class:`LhbFlowRow` (per-day stock count + total net buy)

**Unit convention**: Tushare's moneyflow APIs report amounts in 万元 (10k
yuan). All ``*_yi`` fields here are converted to 亿 by dividing by 10,000.
PR-6 integration tests will validate this assumption against real data;
mismatches surface immediately as values 4-5 orders of magnitude off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .breadth import _bind_in

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..universe import UniverseSnapshot
    from ..windows import Window


_WAN_PER_YI = 10_000.0
_STOCK_TOP_K = 20
_INDUSTRY_TOP_K = 10


def _yi(amount_wan: object) -> float:
    """Convert a (万元) amount cell into 亿. Treat None as 0.0."""
    if amount_wan is None:
        return 0.0
    return float(amount_wan) / _WAN_PER_YI


@dataclass(frozen=True)
class NorthFlowRow:
    trade_date: str
    north_money_yi: float | None


@dataclass(frozen=True)
class MktFlowRow:
    trade_date: str
    main_net_yi: float            # 大单 + 特大单 净流入
    retail_net_yi: float          # 中单 + 小单 净流入


@dataclass(frozen=True)
class IndustryFlowRow:
    name: str
    net_inflow_yi: float          # window total
    pct_change_avg: float         # window mean pct_change


@dataclass(frozen=True)
class StockFlowRow:
    ts_code: str
    name: str
    net_inflow_yi: float


@dataclass(frozen=True)
class HsgtTopRow:
    trade_date: str
    ts_code: str
    name: str
    net_amount_yi: float


@dataclass(frozen=True)
class LhbFlowRow:
    trade_date: str
    n_stocks: int
    net_buy_yi: float


@dataclass(frozen=True)
class CapitalReview:
    north_series: list[NorthFlowRow] = field(default_factory=list)
    north_total_yi: float = 0.0
    north_top10_anchor: list[HsgtTopRow] = field(default_factory=list)
    mkt_series: list[MktFlowRow] = field(default_factory=list)
    industry_top: list[IndustryFlowRow] = field(default_factory=list)
    industry_bottom: list[IndustryFlowRow] = field(default_factory=list)
    concept_top: list[IndustryFlowRow] = field(default_factory=list)
    concept_bottom: list[IndustryFlowRow] = field(default_factory=list)
    stock_top_inflow: list[StockFlowRow] = field(default_factory=list)
    stock_top_outflow: list[StockFlowRow] = field(default_factory=list)
    lhb_series: list[LhbFlowRow] = field(default_factory=list)


def compute_capital(
    db: Database,
    window: Window,
    universes: dict[str, UniverseSnapshot],
    *,
    industry_top_k: int = _INDUSTRY_TOP_K,
    stock_top_k: int = _STOCK_TOP_K,
) -> CapitalReview:
    """Compute all six caliber tables for the window."""
    trade_dates = list(window.trade_dates)
    if not trade_dates:
        return CapitalReview()

    north_series = _north_series(db, trade_dates)
    north_total = sum((r.north_money_yi or 0.0) for r in north_series)
    north_top10_anchor = _hsgt_top10(db, window.anchor)
    mkt_series = _mkt_series(db, trade_dates)
    industry_top, industry_bottom = _industry_or_concept(
        db, "mr_moneyflow_ind_ths", trade_dates, key="name", top_k=industry_top_k,
    )
    concept_top, concept_bottom = _industry_or_concept(
        db, "mr_moneyflow_cnt_ths", trade_dates, key="name", top_k=industry_top_k,
    )
    stock_top_inflow, stock_top_outflow = _stock_flows(
        db, trade_dates, universes, top_k=stock_top_k,
    )
    lhb_series = _lhb_series(db, trade_dates, universes)
    return CapitalReview(
        north_series=north_series,
        north_total_yi=north_total,
        north_top10_anchor=north_top10_anchor,
        mkt_series=mkt_series,
        industry_top=industry_top,
        industry_bottom=industry_bottom,
        concept_top=concept_top,
        concept_bottom=concept_bottom,
        stock_top_inflow=stock_top_inflow,
        stock_top_outflow=stock_top_outflow,
        lhb_series=lhb_series,
    )


# ---------------------------------------------------------------------------
# Per-caliber fetchers
# ---------------------------------------------------------------------------


def _north_series(db: Database, trade_dates: list[str]) -> list[NorthFlowRow]:
    in_clause = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    rows = db.fetchall(
        f"""
        SELECT trade_date, north_money FROM mr_moneyflow_hsgt
        WHERE trade_date IN {in_clause}
        ORDER BY trade_date
        """,
        list(trade_dates),
    )
    by_date = {str(r[0]): r[1] for r in rows}
    return [
        NorthFlowRow(
            trade_date=td,
            north_money_yi=_yi(by_date[td]) if td in by_date and by_date[td] is not None else None,
        )
        for td in trade_dates
    ]


def _hsgt_top10(db: Database, trade_date: str) -> list[HsgtTopRow]:
    rows = db.fetchall(
        """
        SELECT ts_code, name, COALESCE(net_amount, 0) AS net
        FROM mr_hsgt_top10
        WHERE trade_date = ?
        ORDER BY net DESC NULLS LAST
        LIMIT 10
        """,
        [trade_date],
    )
    return [
        HsgtTopRow(
            trade_date=trade_date,
            ts_code=str(r[0]),
            name=str(r[1] or ""),
            net_amount_yi=_yi(r[2]),
        )
        for r in rows
    ]


def _mkt_series(db: Database, trade_dates: list[str]) -> list[MktFlowRow]:
    in_clause = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    rows = db.fetchall(
        f"""
        SELECT trade_date,
               COALESCE(buy_lg_amount, 0) + COALESCE(buy_elg_amount, 0)
                   - COALESCE(sell_lg_amount, 0) - COALESCE(sell_elg_amount, 0) AS main_net,
               COALESCE(buy_sm_amount, 0) + COALESCE(buy_md_amount, 0)
                   - COALESCE(sell_sm_amount, 0) - COALESCE(sell_md_amount, 0) AS retail_net
        FROM mr_moneyflow_mkt
        WHERE trade_date IN {in_clause}
        ORDER BY trade_date
        """,
        list(trade_dates),
    )
    return [
        MktFlowRow(
            trade_date=str(r[0]),
            main_net_yi=_yi(r[1]),
            retail_net_yi=_yi(r[2]),
        )
        for r in rows
    ]


def _industry_or_concept(
    db: Database,
    table: str,
    trade_dates: list[str],
    *,
    key: str,
    top_k: int,
) -> tuple[list[IndustryFlowRow], list[IndustryFlowRow]]:
    in_clause = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    rows = db.fetchall(
        f"""
        SELECT {key} AS name,
               SUM(COALESCE(net_amount, 0)) AS net_total,
               AVG(COALESCE(pct_change, 0)) AS pct_avg
        FROM {table}
        WHERE trade_date IN {in_clause}
        GROUP BY {key}
        """,
        list(trade_dates),
    )
    parsed = [
        IndustryFlowRow(
            name=str(r[0] or ""),
            net_inflow_yi=_yi(r[1]),
            pct_change_avg=float(r[2] or 0.0),
        )
        for r in rows
        if r[0]
    ]
    # Sort by net_inflow_yi desc → top; asc → bottom.
    by_inflow = sorted(parsed, key=lambda x: x.net_inflow_yi, reverse=True)
    return by_inflow[:top_k], by_inflow[-top_k:][::-1]


def _stock_flows(
    db: Database,
    trade_dates: list[str],
    universes: dict[str, UniverseSnapshot],
    *,
    top_k: int,
) -> tuple[list[StockFlowRow], list[StockFlowRow]]:
    """Per-stock window-total net moneyflow, filtered by union of universes.

    The "union of per-day universes" is what the design implies — a stock
    that's listed for any day in the window contributes. Stocks suspended
    every day across the window get filtered out implicitly.
    """
    union = frozenset().union(*[
        u.ts_codes for u in universes.values()
    ])
    if not union:
        return [], []
    in_clause_dates = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    in_clause_codes, code_params = _bind_in(union)
    # LEFT JOIN mr_stock_basic so each row carries a stock name. Fall back to
    # ts_code when stock_basic is missing the row (keeps schemas.CapitalLeader's
    # name min_length=1 satisfied without inventing data).
    rows = db.fetchall(
        f"""
        SELECT m.ts_code,
               COALESCE(NULLIF(b.name, ''), m.ts_code) AS name,
               SUM(COALESCE(m.net_mf_amount, 0)) AS net_total
        FROM mr_moneyflow m
        LEFT JOIN mr_stock_basic b ON b.ts_code = m.ts_code
        WHERE m.trade_date IN {in_clause_dates} AND m.ts_code IN {in_clause_codes}
        GROUP BY m.ts_code, b.name
        """,
        [*trade_dates, *code_params],
    )
    parsed = [
        StockFlowRow(ts_code=str(r[0]), name=str(r[1]), net_inflow_yi=_yi(r[2]))
        for r in rows
    ]
    by_inflow = sorted(parsed, key=lambda x: x.net_inflow_yi, reverse=True)
    return by_inflow[:top_k], by_inflow[-top_k:][::-1]


def _lhb_series(
    db: Database,
    trade_dates: list[str],
    universes: dict[str, UniverseSnapshot],
) -> list[LhbFlowRow]:
    """Per-day LHB stock count + net buy total (亿)."""
    out: list[LhbFlowRow] = []
    for td in trade_dates:
        snap = universes.get(td)
        if snap is None or not snap.ts_codes:
            out.append(LhbFlowRow(trade_date=td, n_stocks=0, net_buy_yi=0.0))
            continue
        in_clause, params = _bind_in(snap.ts_codes)
        row = db.fetchone(
            f"""
            SELECT COUNT(DISTINCT ts_code), COALESCE(SUM(net_amount), 0)
            FROM mr_top_list
            WHERE trade_date = ? AND ts_code IN {in_clause}
            """,
            [td, *params],
        )
        n_stocks = int(row[0] or 0) if row else 0
        net = _yi(row[1]) if row else 0.0
        out.append(LhbFlowRow(trade_date=td, n_stocks=n_stocks, net_buy_yi=net))
    return out
