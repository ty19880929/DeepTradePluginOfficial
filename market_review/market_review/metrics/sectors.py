"""sectors — 板块轮动与持续性 (design §5.3.4 + Appendix A).

Reads ``mr_ths_daily`` (synced via :func:`market_review.data.sync_sector_quotes`)
to produce:

- 当日 Top 10 涨幅榜 — :class:`SectorEntry` × top_k
- 区间 Top 10 累计涨幅榜 — :class:`SectorEntry` × top_k
- 持续性 — 每个板块在区间内进入当日 Top 10 的天数
- 强度矩阵 — :class:`SectorMatrix` (板块 × 日期 → pct_chg)
- 三栏分类 — new_mainline / relay / fading

The "新主线 / 接力 / 退潮" classification splits the window in half: a board
in today's Top 10 that wasn't in the earlier-half's Top 10 is **new_mainline**;
in both halves' Top 10s is **relay**; in earlier-half's Top 10 but not today's
is **fading**. Empty halves degrade gracefully (single-day windows produce
an empty earlier-half so everything in today's Top is **new_mainline**).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..windows import Window


_DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class SectorEntry:
    ts_code: str
    name: str
    pct_chg: float            # 当日（today_top） or 区间累计（range_top）
    persistence_days: int     # 区间内进入当日 Top K 的天数


@dataclass(frozen=True)
class SectorMatrix:
    sectors: list[str]              # 行：sector ts_code
    sector_names: list[str]         # 与 sectors 等长，便于前端展示
    trade_dates: list[str]          # 列
    values: list[list[float]]       # rows × days；空格用 0.0
    cum_pct_chg: list[float]        # 区间累计涨幅 (与 sectors 等长)
    persistence_days: list[int]     # 区间内出现在当日 Top K 的天数


@dataclass(frozen=True)
class SectorReview:
    today_top: list[SectorEntry] = field(default_factory=list)
    range_top: list[SectorEntry] = field(default_factory=list)
    new_mainline: list[SectorEntry] = field(default_factory=list)
    relay: list[SectorEntry] = field(default_factory=list)
    fading: list[SectorEntry] = field(default_factory=list)
    matrix: SectorMatrix | None = None


def compute_sectors(
    db: Database,
    window: Window,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> SectorReview:
    """Build the sector review from already-synced ``mr_ths_daily``."""
    trade_dates = list(window.trade_dates)
    if not trade_dates:
        return SectorReview()

    in_clause_dates = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    rows = db.fetchall(
        f"""
        SELECT ts_code, trade_date, COALESCE(pct_change, 0)
        FROM mr_ths_daily
        WHERE trade_date IN {in_clause_dates}
        ORDER BY ts_code, trade_date
        """,
        list(trade_dates),
    )
    if not rows:
        return SectorReview()

    # ----- shape per-(sector, date) into a {ts_code: {date: pct_chg}} map ---
    daily: dict[str, dict[str, float]] = {}
    for ts_code, trade_date, pct in rows:
        daily.setdefault(str(ts_code), {})[str(trade_date)] = float(pct or 0.0)

    sector_names = _sector_names(db, daily.keys())

    # ----- persistence: per-sector, count days in that day's Top K ---------
    per_day_top: dict[str, list[str]] = {}
    for td in trade_dates:
        day_rows = [(code, daily[code].get(td, 0.0)) for code in daily]
        day_rows.sort(key=lambda x: x[1], reverse=True)
        per_day_top[td] = [code for code, _ in day_rows[:top_k]]
    persistence: dict[str, int] = {code: 0 for code in daily}
    for td, top_list in per_day_top.items():
        for code in top_list:
            persistence[code] = persistence.get(code, 0) + 1

    # ----- range cumulative ------------------------------------------------
    cum: dict[str, float] = {}
    for code, series in daily.items():
        # Geometric chaining of daily pct_chg (more accurate over multi-day
        # windows than simple sum). Treat missing dates as 0% — i.e., no
        # change — which is the cheapest defensible imputation.
        factor = 1.0
        for td in trade_dates:
            factor *= 1.0 + series.get(td, 0.0) / 100.0
        cum[code] = (factor - 1.0) * 100.0

    today_anchor = trade_dates[-1]
    today_pairs = sorted(
        ((code, daily[code].get(today_anchor, 0.0)) for code in daily),
        key=lambda x: x[1],
        reverse=True,
    )
    range_pairs = sorted(cum.items(), key=lambda x: x[1], reverse=True)

    today_top = [
        SectorEntry(
            ts_code=code,
            name=sector_names.get(code, code),
            pct_chg=pct,
            persistence_days=persistence.get(code, 0),
        )
        for code, pct in today_pairs[:top_k]
    ]
    range_top = [
        SectorEntry(
            ts_code=code,
            name=sector_names.get(code, code),
            pct_chg=pct,
            persistence_days=persistence.get(code, 0),
        )
        for code, pct in range_pairs[:top_k]
    ]

    # ----- 三栏分类（new_mainline / relay / fading）-------------------------
    earlier_half = trade_dates[: max(1, len(trade_dates) // 2)]
    earlier_top_set: set[str] = set()
    if len(trade_dates) >= 2:
        earlier_top_set = {
            code
            for td in earlier_half
            for code in per_day_top.get(td, [])
        }
    today_top_set = {e.ts_code for e in today_top}

    new_mainline = [e for e in today_top if e.ts_code not in earlier_top_set]
    relay = [e for e in today_top if e.ts_code in earlier_top_set]
    fading = [
        SectorEntry(
            ts_code=code,
            name=sector_names.get(code, code),
            pct_chg=cum[code],
            persistence_days=persistence.get(code, 0),
        )
        for code in (earlier_top_set - today_top_set)
        if code in cum
    ]
    fading.sort(key=lambda e: e.pct_chg, reverse=True)
    fading = fading[:top_k]

    matrix = _build_matrix(
        sectors=list(daily.keys()),
        sector_names=sector_names,
        trade_dates=trade_dates,
        daily=daily,
        cum=cum,
        persistence=persistence,
    )
    return SectorReview(
        today_top=today_top,
        range_top=range_top,
        new_mainline=new_mainline,
        relay=relay,
        fading=fading,
        matrix=matrix,
    )


def _build_matrix(
    *,
    sectors: list[str],
    sector_names: dict[str, str],
    trade_dates: list[str],
    daily: dict[str, dict[str, float]],
    cum: dict[str, float],
    persistence: dict[str, int],
) -> SectorMatrix:
    # Order rows by descending window-cumulative — keeps the strongest at
    # the top, mirroring the design's "排序板块强度" intent.
    ordered = sorted(sectors, key=lambda c: cum.get(c, 0.0), reverse=True)
    values: list[list[float]] = []
    for code in ordered:
        series = daily.get(code, {})
        values.append([series.get(td, 0.0) for td in trade_dates])
    return SectorMatrix(
        sectors=ordered,
        sector_names=[sector_names.get(c, c) for c in ordered],
        trade_dates=list(trade_dates),
        values=values,
        cum_pct_chg=[cum.get(c, 0.0) for c in ordered],
        persistence_days=[persistence.get(c, 0) for c in ordered],
    )


def _sector_names(db: Database, ts_codes) -> dict[str, str]:
    """Best-effort ts_code → 板块中文名 lookup.

    Primary source: ``mr_ths_index`` —— Tushare ``ths_index`` catalog 全量覆盖
    所有 THS 指数（行业 ``.TI`` / 概念 ``.CI`` / 风格 / 主题 / 宽基 ...）。
    Fallback: ``mr_moneyflow_cnt_ths`` —— 仅覆盖概念板块，但 catalog 落表前的
    历史库可能没有 ``mr_ths_index`` 行，作为退路保证至少概念板块仍有名字。
    都查不到时返回代码本身（最终 render 层会显示成 ``883422.TI`` 等字面）。
    """
    codes = list(ts_codes)
    if not codes:
        return {}
    in_clause = "(" + ",".join(["?"] * len(codes)) + ")"
    out: dict[str, str] = {}
    catalog_rows = db.fetchall(
        f"SELECT ts_code, name FROM mr_ths_index WHERE ts_code IN {in_clause}",
        codes,
    )
    for r in catalog_rows:
        if r[1]:
            out[str(r[0])] = str(r[1])
    missing = [c for c in codes if c not in out]
    if missing:
        in_clause2 = "(" + ",".join(["?"] * len(missing)) + ")"
        fb_rows = db.fetchall(
            f"SELECT ts_code, name FROM mr_moneyflow_cnt_ths WHERE ts_code IN {in_clause2}",
            missing,
        )
        for r in fb_rows:
            if r[1] and str(r[0]) not in out:
                out[str(r[0])] = str(r[1])
    return out
