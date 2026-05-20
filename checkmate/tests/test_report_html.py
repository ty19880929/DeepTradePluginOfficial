"""HTML report rendering tests (PR-5.3).

These tests build a minimal :class:`ReportPayload` directly (no DB round-trip)
and check that :func:`to_html` produces a self-contained, parseable HTML page
with the expected SVG / heatmap / pie sections.

Why this is a structural test rather than a screenshot diff: pixel-level
diffs against an HTML artefact are fragile across CSS engines and Rich
upstream changes. Asserting "the equity SVG is present + contains a
polyline + the heatmap has month labels" catches the broken cases (template
typo, missing data wiring) without locking the visual to a specific style.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from checkmate import paths
from checkmate.report import (
    ReportPayload,
    _equity_svg,
    _exit_reason_pie,
    _monthly_heatmap,
    to_html,
    write_to_disk,
)


def _make_payload(
    *,
    n_days: int = 6,
    with_industry: bool = True,
    with_regime: bool = True,
) -> ReportPayload:
    """Tiny synthetic payload — equity curve from 1.0M to 1.05M over 6 sessions."""
    series = []
    for i in range(n_days):
        eq = 1_000_000.0 + i * 10_000.0
        series.append({
            "trade_date": f"2024030{(i % 9) + 1}",
            "equity": eq,
            "drawdown_pct": max(0.0, 0.001 * i),
            "regime": "strong" if i % 2 == 0 else "neutral",
        })
    return ReportPayload(
        run_id="abc12345-test",
        config_hash="deadbeef",
        code_version="checkmate@0.2.0",
        start="20240301", end="20240329", status="success",
        n_days=n_days,
        initial_cash=1_000_000.0,
        final_equity=series[-1]["equity"],
        final_cash=900_000.0,
        cagr=0.12,
        sharpe=1.4,
        max_drawdown=0.04,
        calmar=3.0,
        win_rate=0.55,
        n_closed_trades=11,
        avg_hold_days=14.0,
        limit_blocked_ratio=0.02,
        total_fees=1234.56,
        n_fills=20, n_cancels=3,
        by_regime=(
            {"strong": {"n_trades": 7, "total_pnl": 2500.0, "wins": 4,
                        "win_rate": 4 / 7, "avg_hold_days": 12.5},
             "neutral": {"n_trades": 4, "total_pnl": -300.0, "wins": 2,
                         "win_rate": 0.5, "avg_hold_days": 8.0}}
            if with_regime else {}
        ),
        by_industry=({"电子": 1500.0, "白酒": 800.0, "银行": -100.0}
                      if with_industry else {}),
        equity_series=series,
    )


# ---------------------------------------------------------------------------
# Equity SVG
# ---------------------------------------------------------------------------


def test_equity_svg_renders_polyline_with_points() -> None:
    payload = _make_payload()
    svg = _equity_svg(payload.equity_series)
    assert svg.startswith("<svg ")
    assert "<polyline" in svg
    # 6 points → 6 "x,y" coordinate pairs in the points attribute
    m = re.search(r'points="([^"]+)"', svg)
    assert m is not None
    coords = m.group(1).split()
    assert len(coords) == 6


def test_equity_svg_empty_input_yields_blank_svg() -> None:
    svg = _equity_svg([])
    assert svg == '<svg width="900" height="220"></svg>'


def test_equity_svg_handles_flat_series_without_div_zero() -> None:
    series = [{"trade_date": f"2024030{i+1}", "equity": 1_000_000.0} for i in range(5)]
    # Should not raise (v_max == v_min protection)
    svg = _equity_svg(series)
    assert "<polyline" in svg


# ---------------------------------------------------------------------------
# Monthly heatmap
# ---------------------------------------------------------------------------


def test_monthly_heatmap_renders_table_with_month_labels() -> None:
    series = [
        {"trade_date": "20240131", "equity": 1_000_000.0},
        {"trade_date": "20240229", "equity": 1_050_000.0},  # +5%
        {"trade_date": "20240329", "equity": 1_080_000.0},  # +2.86%
        {"trade_date": "20240430", "equity": 1_020_000.0},  # -5.56%
    ]
    html = _monthly_heatmap(series)
    assert "<table" in html and "</table>" in html
    # All 12 month labels in the header row
    for m in range(1, 13):
        assert f">{m:02d}<" in html
    # Year label "2024" appears
    assert ">2024<" in html
    # +5% bucket (Feb) should be color-coded green; -5.56% (Apr) red.
    assert "rgba(22, 163, 74" in html  # green
    assert "rgba(220, 38, 38" in html   # red


def test_monthly_heatmap_no_data_returns_meta_message() -> None:
    assert "No equity" in _monthly_heatmap([])


# ---------------------------------------------------------------------------
# Exit reason pie
# ---------------------------------------------------------------------------


def test_exit_reason_pie_uses_industry_breakdown_when_available() -> None:
    payload = _make_payload()
    svg, legend = _exit_reason_pie(payload)
    assert svg.startswith("<svg ")
    # 3 industries → 3 path slices (each industry gets a wedge)
    assert svg.count("<path") == 3
    # Legend lists each industry by name
    for industry in ("电子", "白酒", "银行"):
        assert industry in legend


def test_exit_reason_pie_falls_back_to_regime() -> None:
    payload = _make_payload(with_industry=False)
    svg, legend = _exit_reason_pie(payload)
    assert "<svg " in svg
    assert "strong" in legend or "neutral" in legend


def test_exit_reason_pie_no_data_returns_placeholder() -> None:
    payload = _make_payload(with_industry=False, with_regime=False)
    svg, legend = _exit_reason_pie(payload)
    assert "<svg " in svg
    assert "No exit-reason data" in legend


# ---------------------------------------------------------------------------
# Top-level to_html — Jinja2 template
# ---------------------------------------------------------------------------


def test_to_html_self_contained_doc() -> None:
    payload = _make_payload()
    html = to_html(payload)
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html and "</style>" in html
    # No JS, no remote resources (SVG xmlns URI is fine — it's a static
    # namespace identifier, not a network fetch).
    assert "<script" not in html.lower()
    assert "src=\"http" not in html
    assert "href=\"http" not in html
    # Headline metrics surface
    assert payload.run_id in html
    assert payload.config_hash in html


def test_to_html_contains_all_three_visual_sections() -> None:
    html = to_html(_make_payload())
    # Equity / monthly / pie sections
    assert "Equity curve" in html
    assert "Monthly returns" in html
    assert "Exit reasons" in html
    # Each renders SVG + table where applicable
    assert "<polyline" in html  # equity
    assert 'class="heatmap"' in html  # monthly
    assert "<path" in html  # pie


def test_to_html_renders_with_regime_and_industry_tables() -> None:
    html = to_html(_make_payload())
    assert "By regime" in html
    assert "By industry" in html
    # Specific row values land in the markup
    assert ">strong<" in html
    assert ">电子<" in html


def test_to_html_drops_optional_tables_when_payload_empty() -> None:
    html = to_html(_make_payload(with_industry=False, with_regime=False))
    assert "By regime" not in html
    assert "By industry" not in html


# ---------------------------------------------------------------------------
# write_to_disk html path
# ---------------------------------------------------------------------------


def test_write_to_disk_html_lands_at_run_id_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()
    payload = _make_payload()
    path = write_to_disk(payload, fmt="html")
    assert path.is_file()
    assert path.suffix == ".html"
    assert path.name == f"{payload.run_id}.html"
    contents = path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in contents
    assert payload.run_id in contents
