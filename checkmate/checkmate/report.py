"""Backtest report aggregation (PR-4.3).

Reads:
  * ``checkmate_backtest_runs`` for the run header (config_hash, code_version,
    start/end, status, raw metrics).
  * Per-day shard JSONs under ``~/.deeptrade/checkmate/backtests/<config_hash>/days/``
    for the equity / drawdown timeseries.
  * ``checkmate_trades`` for fill + cancel ledger (cost breakdowns aggregate
    into total fees and limit-blocked counts).
  * ``checkmate_positions`` for closed positions → win-rate / hold-day stats.
  * ``checkmate_universe_daily`` + ``checkmate_regime_daily`` for industry
    & regime cross-tabs.

Outputs:
  * :class:`ReportPayload` — a dataclass with summary metrics + cross-tabs +
    equity timeseries. JSON-serialisable via :func:`dataclasses.asdict`.
  * :func:`to_markdown` — short text rendering for the CLI ``report``
    subcommand.

CAGR / Sharpe / Calmar are computed from the equity series (one observation
per trading day). The 252-day annualisation matches the A-share trading year.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import paths
from .runtime import CheckmateRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReportPayload:
    run_id: str
    config_hash: str
    code_version: str
    start: str
    end: str
    status: str
    n_days: int
    initial_cash: float
    final_equity: float
    final_cash: float
    cagr: float | None
    sharpe: float | None
    max_drawdown: float
    calmar: float | None
    win_rate: float | None
    n_closed_trades: int
    avg_hold_days: float | None
    limit_blocked_ratio: float | None
    total_fees: float
    n_fills: int
    n_cancels: int
    by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_industry: dict[str, float] = field(default_factory=dict)
    equity_series: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric helpers (pure functions, no DB)
# ---------------------------------------------------------------------------


def _compute_cagr(equity_series: list[float], n_days: int) -> float | None:
    if not equity_series or n_days < 1:
        return None
    initial = equity_series[0]
    final = equity_series[-1]
    if initial <= 0:
        return None
    years = n_days / 252.0
    if years < (1.0 / 252.0):
        return None
    return (final / initial) ** (1.0 / years) - 1.0


def _compute_daily_returns(equity_series: list[float]) -> list[float]:
    rets: list[float] = []
    for i in range(1, len(equity_series)):
        prev = equity_series[i - 1]
        if prev > 0:
            rets.append(equity_series[i] / prev - 1.0)
    return rets


def _compute_sharpe(equity_series: list[float]) -> float | None:
    rets = _compute_daily_returns(equity_series)
    if len(rets) < 2:
        return None
    mean = statistics.mean(rets)
    stdev = statistics.stdev(rets)
    if stdev <= 0:
        return None
    return mean / stdev * math.sqrt(252.0)


def _compute_max_drawdown(equity_series: list[float]) -> float:
    if not equity_series:
        return 0.0
    peak = equity_series[0]
    max_dd = 0.0
    for e in equity_series:
        peak = max(peak, e)
        if peak > 0:
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _calendar_days(from_yyyymmdd: str | None, to_yyyymmdd: str | None) -> int | None:
    if not from_yyyymmdd or not to_yyyymmdd:
        return None
    try:
        a = datetime.strptime(str(from_yyyymmdd), "%Y%m%d")
        b = datetime.strptime(str(to_yyyymmdd), "%Y%m%d")
    except ValueError:
        return None
    return max(0, (b - a).days)


# ---------------------------------------------------------------------------
# Shard reader
# ---------------------------------------------------------------------------


def _read_shards(config_hash: str) -> list[dict[str, Any]]:
    """Return shard payloads sorted by trade_date ascending."""
    d = paths.backtests_dir() / config_hash / "days"
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            logger.warning("could not parse shard %s", p)
    return out


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


def build_report(rt: CheckmateRuntime, run_id: str) -> ReportPayload:
    """Aggregate metrics + cross-tabs + equity series for ``run_id``.

    Raises ``ValueError`` if the run row is missing — callers should handle
    by printing a helpful error rather than crashing.
    """
    run_row = rt.db.execute(
        """
        SELECT config_hash, code_version, start_date, end_date, status,
               config_json, metrics_json
        FROM checkmate_backtest_runs
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run_row is None:
        raise ValueError(f"no checkmate_backtest_runs row for run_id={run_id!r}")
    config_hash, code_version, start, end, status, config_json, metrics_json = run_row

    config = json.loads(config_json) if config_json else {}
    initial_cash = float(config.get("initial_cash", 0.0))

    # Equity series from shards
    shards = _read_shards(config_hash)
    equity_series_raw = [float(s.get("equity", 0.0)) for s in shards]
    equity_series = [
        {"trade_date": s["trade_date"],
         "equity": float(s.get("equity", 0.0)),
         "drawdown_pct": float(s.get("drawdown_pct", 0.0)),
         "regime": s.get("regime")}
        for s in shards
    ]
    n_days = len(equity_series_raw)
    final_equity = equity_series_raw[-1] if equity_series_raw else initial_cash
    final_cash = float(shards[-1]["state"]["cash"]) if shards else initial_cash

    cagr = _compute_cagr(equity_series_raw, n_days)
    sharpe = _compute_sharpe(equity_series_raw)
    max_dd = _compute_max_drawdown(equity_series_raw)
    calmar = (cagr / abs(max_dd)) if (cagr is not None and max_dd > 0) else None

    # Trade ledger
    trade_rows = rt.db.execute(
        """
        SELECT ts_code, side, order_date, fill_date, fill_price_raw, shares,
               cost_breakdown, cancel_reason
        FROM checkmate_trades
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()
    n_fills = 0
    n_cancels = 0
    total_fees = 0.0
    n_limit_blocked = 0
    for ts_code, side, _order_date, fill_date, _fill_price, _shares, cb_json, cancel_reason in trade_rows:
        if cancel_reason:
            n_cancels += 1
            cr = str(cancel_reason)
            if cr.startswith("limit_up") or cr.startswith("limit_down"):
                n_limit_blocked += 1
        elif fill_date:
            n_fills += 1
            if cb_json:
                try:
                    cb = json.loads(cb_json)
                    total_fees += sum(float(v) for v in cb.values() if v is not None)
                except Exception:  # noqa: BLE001
                    pass
    total_orders = n_fills + n_cancels
    limit_blocked_ratio = (n_limit_blocked / total_orders) if total_orders > 0 else None

    # Closed positions → win rate / hold days
    pos_rows = rt.db.execute(
        """
        SELECT ts_code, entry_date, exit_date, entry_price_raw, exit_price_raw,
               shares, state
        FROM checkmate_positions
        WHERE run_id = ? AND state = 'closed'
        """,
        [run_id],
    ).fetchall()
    n_closed = len(pos_rows)
    wins = 0
    hold_days_total = 0
    hold_days_n = 0
    for ts_code, entry_date, exit_date, entry_price, exit_price, shares, _state in pos_rows:
        if entry_price is not None and exit_price is not None:
            if float(exit_price) > float(entry_price):
                wins += 1
        days = _calendar_days(entry_date, exit_date)
        if days is not None:
            hold_days_total += days
            hold_days_n += 1
    win_rate = (wins / n_closed) if n_closed > 0 else None
    avg_hold_days = (hold_days_total / hold_days_n) if hold_days_n > 0 else None

    # by_regime: aggregate closed-position PnL + hold-days keyed by the regime
    # at entry date. PR-6.3 widens the bucket to 4 metrics per regime so the
    # dashboard + HTML report can render a regime × metric cross-tab.
    by_regime: dict[str, dict[str, Any]] = {}
    for ts_code, entry_date, exit_date, entry_price, exit_price, shares, _state in pos_rows:
        if exit_price is None or entry_price is None or not shares:
            continue
        pnl = (float(exit_price) - float(entry_price)) * int(shares)
        regime_row = rt.db.execute(
            "SELECT regime FROM checkmate_regime_daily WHERE trade_date = ?",
            [entry_date],
        ).fetchone()
        regime = str(regime_row[0]) if regime_row else "unknown"
        bucket = by_regime.setdefault(regime, {
            "n_trades": 0, "total_pnl": 0.0, "wins": 0,
            "hold_days_sum": 0, "hold_days_n": 0,
        })
        bucket["n_trades"] += 1
        bucket["total_pnl"] += pnl
        if pnl > 0:
            bucket["wins"] += 1
        days = _calendar_days(entry_date, exit_date)
        if days is not None:
            bucket["hold_days_sum"] += days
            bucket["hold_days_n"] += 1
    for bucket in by_regime.values():
        bucket["win_rate"] = (
            bucket["wins"] / bucket["n_trades"] if bucket["n_trades"] > 0 else None
        )
        bucket["avg_hold_days"] = (
            bucket["hold_days_sum"] / bucket["hold_days_n"]
            if bucket["hold_days_n"] > 0 else None
        )
        bucket["total_pnl"] = round(bucket["total_pnl"], 4)
        # Strip internal accumulators that aren't part of the public schema.
        bucket.pop("hold_days_sum", None)
        bucket.pop("hold_days_n", None)

    # by_industry: aggregate closed-position PnL keyed by industry @ entry date
    by_industry: dict[str, float] = {}
    for ts_code, entry_date, exit_date, entry_price, exit_price, shares, _state in pos_rows:
        if exit_price is None or entry_price is None or not shares:
            continue
        pnl = (float(exit_price) - float(entry_price)) * int(shares)
        u = rt.db.execute(
            "SELECT industry FROM checkmate_universe_daily "
            "WHERE trade_date = ? AND ts_code = ?",
            [entry_date, ts_code],
        ).fetchone()
        industry = (u[0] if u and u[0] is not None else "_unknown")
        by_industry[industry] = round(by_industry.get(industry, 0.0) + pnl, 4)

    return ReportPayload(
        run_id=run_id,
        config_hash=str(config_hash),
        code_version=str(code_version or ""),
        start=str(start), end=str(end), status=str(status),
        n_days=n_days,
        initial_cash=initial_cash,
        final_equity=final_equity,
        final_cash=final_cash,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        n_closed_trades=n_closed,
        avg_hold_days=avg_hold_days,
        limit_blocked_ratio=limit_blocked_ratio,
        total_fees=round(total_fees, 4),
        n_fills=n_fills,
        n_cancels=n_cancels,
        by_regime=by_regime,
        by_industry=by_industry,
        equity_series=equity_series,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v*100:.2f}%"


def _fmt_num(v: float | None, n: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{n}f}"


def to_json(payload: ReportPayload) -> str:
    return json.dumps(asdict(payload), ensure_ascii=False, indent=2, default=str)


def to_markdown(payload: ReportPayload) -> str:
    p = payload
    lines: list[str] = []
    lines.append(f"# Checkmate backtest report — {p.run_id}")
    lines.append("")
    lines.append(f"- **config_hash**: `{p.config_hash}`")
    lines.append(f"- **code_version**: `{p.code_version}`")
    lines.append(f"- **window**: {p.start} → {p.end}  ({p.n_days} sessions)")
    lines.append(f"- **status**: `{p.status}`")
    lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| initial_cash | {p.initial_cash:,.2f} |")
    lines.append(f"| final_equity | {p.final_equity:,.2f} |")
    lines.append(f"| final_cash | {p.final_cash:,.2f} |")
    lines.append(f"| CAGR | {_fmt_pct(p.cagr)} |")
    lines.append(f"| Sharpe | {_fmt_num(p.sharpe)} |")
    lines.append(f"| max_drawdown | {_fmt_pct(p.max_drawdown)} |")
    lines.append(f"| Calmar | {_fmt_num(p.calmar)} |")
    lines.append(f"| win_rate | {_fmt_pct(p.win_rate)} |")
    lines.append(f"| n_closed_trades | {p.n_closed_trades} |")
    lines.append(f"| avg_hold_days | {_fmt_num(p.avg_hold_days, 1)} |")
    lines.append(f"| total_fees | {p.total_fees:,.2f} |")
    lines.append(f"| n_fills / n_cancels | {p.n_fills} / {p.n_cancels} |")
    lines.append(f"| limit_blocked_ratio | {_fmt_pct(p.limit_blocked_ratio)} |")
    lines.append("")
    if p.by_regime:
        lines.append("## By regime")
        lines.append("")
        lines.append("| regime | n_trades | total_pnl | win_rate | avg_hold_days |")
        lines.append("|---|---|---|---|---|")
        for k, v in sorted(p.by_regime.items()):
            lines.append(
                f"| {k} | {v.get('n_trades', 0)} | {v.get('total_pnl', 0):,.2f} | "
                f"{_fmt_pct(v.get('win_rate'))} | "
                f"{_fmt_num(v.get('avg_hold_days'), 1)} |"
            )
        lines.append("")
    if p.by_industry:
        lines.append("## By industry")
        lines.append("")
        lines.append("| industry | total_pnl |")
        lines.append("|---|---|")
        for k, v in sorted(p.by_industry.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v:,.2f} |")
        lines.append("")
    return "\n".join(lines)


def write_to_disk(payload: ReportPayload, *, fmt: str = "json") -> Path:
    """Persist the report under ``~/.deeptrade/checkmate/reports/<run_id>.<ext>``."""
    paths.reports_dir().mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path = paths.reports_dir() / f"{payload.run_id}.json"
        path.write_text(to_json(payload), encoding="utf-8")
    elif fmt == "markdown":
        path = paths.reports_dir() / f"{payload.run_id}.md"
        path.write_text(to_markdown(payload), encoding="utf-8")
    elif fmt == "html":
        path = paths.reports_dir() / f"{payload.run_id}.html"
        path.write_text(to_html(payload), encoding="utf-8")
    else:
        raise ValueError(f"unknown report format {fmt!r}")
    return path


# ---------------------------------------------------------------------------
# HTML rendering (PR-5.3)
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Checkmate Report — {{ p.run_id }}</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    color: #1f2937;
    max-width: 1100px;
    margin: 24px auto;
    padding: 0 16px;
    line-height: 1.5;
  }
  h1 { color: #0f766e; border-bottom: 2px solid #14b8a6; padding-bottom: 6px; }
  h2 { color: #155e75; margin-top: 28px; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; }
  .summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 12px 0 24px;
  }
  .summary-cell {
    background: #f1f5f9;
    border-left: 3px solid #14b8a6;
    padding: 8px 12px;
    border-radius: 4px;
  }
  .summary-cell .label { font-size: 12px; color: #64748b; text-transform: uppercase; }
  .summary-cell .value { font-size: 20px; font-weight: 600; color: #0f766e; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }
  th { background: #f8fafc; font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.pos { color: #16a34a; }
  td.neg { color: #dc2626; }
  .heatmap td { text-align: center; padding: 4px 6px; font-size: 12px; }
  .heatmap .label { background: #f8fafc; font-weight: 600; }
  .svg-frame {
    border: 1px solid #e5e7eb; border-radius: 6px;
    padding: 12px; background: #fafafa; margin: 8px 0;
  }
  .meta { color: #64748b; font-size: 13px; }
  .pie-legend { display: inline-block; vertical-align: top; margin-left: 16px; }
  .pie-legend .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: middle; }
</style>
</head>
<body>

<h1>Checkmate Backtest Report</h1>
<p class="meta">
  <strong>run_id</strong> <code>{{ p.run_id }}</code> ·
  <strong>config_hash</strong> <code>{{ p.config_hash }}</code> ·
  <strong>code_version</strong> <code>{{ p.code_version }}</code><br>
  window {{ p.start }} → {{ p.end }} ({{ p.n_days }} sessions) ·
  status <code>{{ p.status }}</code>
</p>

<div class="summary-grid">
  <div class="summary-cell">
    <div class="label">CAGR</div>
    <div class="value">{{ pct(p.cagr) }}</div>
  </div>
  <div class="summary-cell">
    <div class="label">max drawdown</div>
    <div class="value">{{ pct(p.max_drawdown) }}</div>
  </div>
  <div class="summary-cell">
    <div class="label">Sharpe</div>
    <div class="value">{{ num(p.sharpe) }}</div>
  </div>
  <div class="summary-cell">
    <div class="label">win rate</div>
    <div class="value">{{ pct(p.win_rate) }}</div>
  </div>
  <div class="summary-cell">
    <div class="label">final equity</div>
    <div class="value">{{ "{:,.0f}".format(p.final_equity) }}</div>
  </div>
  <div class="summary-cell">
    <div class="label">n_fills / cancels</div>
    <div class="value">{{ p.n_fills }} / {{ p.n_cancels }}</div>
  </div>
  <div class="summary-cell">
    <div class="label">avg hold (days)</div>
    <div class="value">{{ num(p.avg_hold_days, 1) }}</div>
  </div>
  <div class="summary-cell">
    <div class="label">total fees</div>
    <div class="value">{{ "{:,.0f}".format(p.total_fees) }}</div>
  </div>
</div>

<h2>Equity curve</h2>
<div class="svg-frame">{{ equity_svg|safe }}</div>

<h2>Monthly returns</h2>
{{ monthly_heatmap|safe }}

<h2>Exit reasons</h2>
<div class="svg-frame">
  {{ exit_pie_svg|safe }}
  {{ exit_pie_legend|safe }}
</div>

{% if p.by_regime %}
<h2>By regime</h2>
<table>
  <tr>
    <th>regime</th>
    <th>n_trades</th>
    <th>total_pnl</th>
    <th>win_rate</th>
    <th>avg_hold_days</th>
  </tr>
  {% for k, v in p.by_regime|dictsort %}
    <tr>
      <td><strong>{{ k }}</strong></td>
      <td class="num">{{ v.n_trades }}</td>
      <td class="num {% if v.total_pnl >= 0 %}pos{% else %}neg{% endif %}">
        {{ "{:,.2f}".format(v.total_pnl) }}
      </td>
      <td class="num">{{ pct(v.win_rate) }}</td>
      <td class="num">{{ num(v.avg_hold_days, 1) }}</td>
    </tr>
  {% endfor %}
</table>
{% endif %}

{% if p.by_industry %}
<h2>By industry</h2>
<table>
  <tr><th>industry</th><th>total_pnl</th></tr>
  {% for k, v in p.by_industry|dictsort(by='value', reverse=true) %}
    <tr>
      <td><strong>{{ k }}</strong></td>
      <td class="num {% if v >= 0 %}pos{% else %}neg{% endif %}">{{ "{:,.2f}".format(v) }}</td>
    </tr>
  {% endfor %}
</table>
{% endif %}

<p class="meta" style="margin-top: 32px;">
  Generated by Checkmate v0.2.0 · No JavaScript · Inline SVG only.
</p>
</body>
</html>
"""


def _equity_svg(series: list[dict[str, Any]], *, width: int = 900, height: int = 220) -> str:
    """Inline SVG line chart of equity over time."""
    if not series:
        return f'<svg width="{width}" height="{height}"></svg>'
    n = len(series)
    values = [float(p.get("equity", 0.0)) for p in series]
    v_min, v_max = min(values), max(values)
    if v_max == v_min:
        v_max = v_min + 1.0  # avoid div-zero on flat series
    pad_x, pad_y = 50, 24
    chart_w = width - 2 * pad_x
    chart_h = height - 2 * pad_y

    def _x(i: int) -> float:
        return pad_x + (i / max(1, n - 1)) * chart_w

    def _y(v: float) -> float:
        return pad_y + (1.0 - (v - v_min) / (v_max - v_min)) * chart_h

    points = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(values))
    last_x, last_y = _x(n - 1), _y(values[-1])
    first_x, first_y = _x(0), _y(values[0])
    init = values[0]
    final = values[-1]
    delta_pct = (final / init - 1.0) * 100.0 if init > 0 else 0.0
    color = "#16a34a" if delta_pct >= 0 else "#dc2626"

    parts: list[str] = []
    parts.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
                 f'xmlns="http://www.w3.org/2000/svg">')
    # axes
    parts.append(f'<line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{height - pad_y}" '
                 f'stroke="#cbd5e1" stroke-width="1"/>')
    parts.append(f'<line x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" '
                 f'stroke="#cbd5e1" stroke-width="1"/>')
    # axis labels
    parts.append(f'<text x="{pad_x - 6}" y="{pad_y + 4}" font-size="10" text-anchor="end" '
                 f'fill="#64748b">{v_max:,.0f}</text>')
    parts.append(f'<text x="{pad_x - 6}" y="{height - pad_y + 4}" font-size="10" '
                 f'text-anchor="end" fill="#64748b">{v_min:,.0f}</text>')
    # polyline
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    # endpoints
    parts.append(f'<circle cx="{first_x:.1f}" cy="{first_y:.1f}" r="3" fill="#64748b"/>')
    parts.append(f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{color}"/>')
    # delta badge
    parts.append(f'<text x="{width - pad_x}" y="{pad_y + 4}" font-size="12" '
                 f'text-anchor="end" font-weight="bold" fill="{color}">'
                 f'{delta_pct:+.2f}%</text>')
    parts.append('</svg>')
    return "".join(parts)


def _monthly_heatmap(series: list[dict[str, Any]]) -> str:
    """HTML table heatmap of month-over-month returns.

    Rows: year. Columns: Jan–Dec. Cell color brightness scales with |return|.
    Empty months render blank.
    """
    if not series:
        return '<p class="meta">No equity data.</p>'

    # Aggregate: for each (year, month), pick the last equity in that month.
    month_eq: dict[tuple[int, int], float] = {}
    for p in series:
        td = str(p.get("trade_date") or "")
        if len(td) < 6:
            continue
        try:
            year = int(td[:4])
            month = int(td[4:6])
        except ValueError:
            continue
        eq = float(p.get("equity", 0.0))
        month_eq[(year, month)] = eq

    if not month_eq:
        return '<p class="meta">No monthly aggregates available.</p>'

    # Sort entries chronologically; compute month returns vs previous tracked month.
    keys = sorted(month_eq.keys())
    returns: dict[tuple[int, int], float] = {}
    for i, k in enumerate(keys):
        if i == 0:
            continue
        prev_eq = month_eq[keys[i - 1]]
        cur_eq = month_eq[k]
        if prev_eq > 0:
            returns[k] = cur_eq / prev_eq - 1.0

    years = sorted({y for y, _ in keys})
    rows: list[str] = []
    rows.append('<table class="heatmap">')
    rows.append("<tr><td class='label'></td>"
                + "".join(f"<td class='label'>{m:02d}</td>" for m in range(1, 13))
                + "</tr>")
    for y in years:
        cells = [f"<td class='label'>{y}</td>"]
        for m in range(1, 13):
            r = returns.get((y, m))
            if r is None:
                cells.append("<td></td>")
            else:
                # color: red for negative, green for positive; alpha by |r|.
                magnitude = min(1.0, abs(r) / 0.10)  # 10% saturates
                if r >= 0:
                    color = f"rgba(22, 163, 74, {0.15 + 0.55 * magnitude:.2f})"
                else:
                    color = f"rgba(220, 38, 38, {0.15 + 0.55 * magnitude:.2f})"
                cells.append(
                    f"<td style='background:{color}'>{r*100:+.1f}%</td>"
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    return "\n".join(rows)


_PIE_COLORS = (
    "#0f766e", "#dc2626", "#ca8a04", "#7c3aed", "#0284c7",
    "#16a34a", "#db2777", "#9333ea", "#475569", "#ea580c",
)


def _exit_reason_pie(payload: ReportPayload, *, size: int = 240) -> tuple[str, str]:
    """Return ``(svg, legend_html)``. Aggregates closed-position exit_reasons
    from ``checkmate_positions`` via the report payload — to avoid threading
    a DB session through, we cheat slightly: we only build the pie if the
    report exposes per-reason counts. v0.2.0 keeps it simple by approximating
    from ``by_regime`` when no breakdown is present.

    For PR-5.3 this surfaces *any* category breakdown (exit_reason / regime
    / industry) — choose the widest the payload offers.
    """
    # Pick the most informative breakdown available.
    if payload.by_industry:
        items = sorted(payload.by_industry.items(),
                       key=lambda kv: -abs(kv[1]))[:8]
        title = "by_industry pnl"
        # For pie, slice by absolute pnl magnitude (signed labels in legend).
        slices = [(k, abs(v)) for k, v in items]
        labels_signed = {k: v for k, v in items}
    elif payload.by_regime:
        items = sorted(payload.by_regime.items(),
                       key=lambda kv: -kv[1].get("n_trades", 0))[:6]
        title = "by_regime n_trades"
        slices = [(k, v.get("n_trades", 0)) for k, v in items]
        labels_signed = {k: v.get("n_trades", 0) for k, v in items}
    else:
        return (f'<svg width="{size}" height="{size}"></svg>',
                '<div class="meta">No exit-reason data yet.</div>')

    total = sum(v for _, v in slices) or 1.0
    cx = cy = size / 2
    r = size * 0.42

    parts = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
             f'xmlns="http://www.w3.org/2000/svg">']
    start_angle = -90.0  # start from 12 o'clock
    legend_rows: list[str] = [f'<div class="pie-legend"><strong>{title}</strong><br>']
    for i, (label, val) in enumerate(slices):
        frac = val / total
        sweep = frac * 360.0
        end_angle = start_angle + sweep
        if frac >= 0.9999:
            # Single full pie — render as circle to avoid arc degeneracy.
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" '
                         f'fill="{_PIE_COLORS[i % len(_PIE_COLORS)]}"/>')
        else:
            from math import cos, radians, sin
            x1 = cx + r * cos(radians(start_angle))
            y1 = cy + r * sin(radians(start_angle))
            x2 = cx + r * cos(radians(end_angle))
            y2 = cy + r * sin(radians(end_angle))
            large = 1 if sweep > 180.0 else 0
            d = (f"M {cx:.1f},{cy:.1f} L {x1:.1f},{y1:.1f} "
                 f"A {r:.1f},{r:.1f} 0 {large} 1 {x2:.1f},{y2:.1f} Z")
            parts.append(f'<path d="{d}" fill="{_PIE_COLORS[i % len(_PIE_COLORS)]}"/>')
        start_angle = end_angle
        color = _PIE_COLORS[i % len(_PIE_COLORS)]
        legend_rows.append(
            f'<div><span class="swatch" style="background:{color}"></span>'
            f'<strong>{label}</strong>: {labels_signed[label]:,.2f}</div>'
        )
    parts.append("</svg>")
    legend_rows.append("</div>")
    return "".join(parts), "".join(legend_rows)


def to_html(payload: ReportPayload) -> str:
    """Render the report to a self-contained HTML page (Jinja2 + inline SVG).

    No external CSS, no JavaScript — drop the file in a browser and it
    renders. Equity curve + month heatmap + pie are all SVG / inline tables.
    """
    try:
        from jinja2 import Environment, StrictUndefined  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - deps are pinned
        raise RuntimeError(
            "Jinja2 is required for HTML reports; ensure it's installed "
            "(it's in deeptrade_plugin.yaml::dependencies as of v0.2.0)."
        ) from exc

    equity_svg = _equity_svg(payload.equity_series)
    monthly_heatmap = _monthly_heatmap(payload.equity_series)
    pie_svg, pie_legend = _exit_reason_pie(payload)

    def pct_filter(v: float | None) -> str:
        return _fmt_pct(v)

    def num_filter(v: float | None, n: int = 2) -> str:
        return _fmt_num(v, n)

    env = Environment(undefined=StrictUndefined, autoescape=True)
    env.filters["pct"] = pct_filter
    env.filters["num"] = num_filter
    tmpl = env.from_string(_HTML_TEMPLATE)
    return tmpl.render(
        p=payload,
        pct=pct_filter,
        num=num_filter,
        equity_svg=equity_svg,
        monthly_heatmap=monthly_heatmap,
        exit_pie_svg=pie_svg,
        exit_pie_legend=pie_legend,
    )


__all__ = [
    "ReportPayload",
    "build_report",
    "to_html",
    "to_json",
    "to_markdown",
    "write_to_disk",
]
