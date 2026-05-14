"""PR-2 unit tests for the screen-mode funnel widget (Plan §4.2, G5, U-5).

The funnel is VA-exclusive: 5 surviving counts (main board → ST/suspension
→ T-day rules → turnover → volume) rendered as a horizontal bar chart.
This test file locks both the data model and the renderable so future UI
tweaks don't silently drop the bar layer.
"""

from __future__ import annotations

from io import StringIO

import pytest
from deeptrade.theme import EVA_THEME
from rich.console import Console

from volume_anomaly.ui.funnel import (
    FunnelSummary,
    render_funnel_compact,
    render_funnel_full,
)


def _render_to_text(renderable, *, width: int = 120) -> str:
    """Render a rich object to plain text (no ANSI) for substring asserts."""
    buf = StringIO()
    console = Console(
        theme=EVA_THEME,
        file=buf,
        force_terminal=False,
        no_color=True,
        width=width,
    )
    console.print(renderable)
    return buf.getvalue()


class TestFunnelSummary:
    def test_default_all_none(self) -> None:
        f = FunnelSummary()
        assert f.n_main_board is None
        assert f.n_after_st_susp is None
        assert f.n_after_t_day_rules is None
        assert f.n_after_turnover is None
        assert f.n_after_vol_rules is None

    def test_populated(self) -> None:
        f = FunnelSummary(
            n_main_board=3210,
            n_after_st_susp=3187,
            n_after_t_day_rules=412,
            n_after_turnover=248,
            n_after_vol_rules=35,
        )
        assert f.n_main_board == 3210
        assert f.n_after_vol_rules == 35


class TestRenderFunnelFull:
    """U-5: full funnel renders all 5 rows with counts + deltas."""

    def test_all_five_steps_present(self) -> None:
        f = FunnelSummary(
            n_main_board=3210,
            n_after_st_susp=3187,
            n_after_t_day_rules=412,
            n_after_turnover=248,
            n_after_vol_rules=35,
        )
        text = _render_to_text(render_funnel_full(f), width=120)
        assert "主板上市" in text
        assert "排除 ST/停牌" in text
        assert "T 日规则" in text
        assert "换手率筛选" in text
        assert "量能筛选" in text

    def test_counts_visible(self) -> None:
        f = FunnelSummary(
            n_main_board=3210,
            n_after_st_susp=3187,
            n_after_t_day_rules=412,
            n_after_turnover=248,
            n_after_vol_rules=35,
        )
        text = _render_to_text(render_funnel_full(f), width=120)
        for n in ("3210", "3187", "412", "248", "35"):
            assert n in text

    def test_deltas_shown(self) -> None:
        f = FunnelSummary(
            n_main_board=3210,
            n_after_st_susp=3187,
            n_after_t_day_rules=412,
            n_after_turnover=248,
            n_after_vol_rules=35,
        )
        text = _render_to_text(render_funnel_full(f), width=120)
        # Deltas are negative (filter removes survivors).
        assert "-23" in text
        assert "-2775" in text
        assert "-164" in text
        assert "-213" in text

    def test_partial_population_renders_waiting_for_missing(self) -> None:
        """Only the first 3 steps known → rows 4-5 show ``(等待)``."""
        f = FunnelSummary(
            n_main_board=3210,
            n_after_st_susp=3187,
            n_after_t_day_rules=412,
            n_after_turnover=None,
            n_after_vol_rules=None,
        )
        text = _render_to_text(render_funnel_full(f), width=120)
        assert "3210" in text
        assert "412" in text
        assert "(等待)" in text

    def test_empty_funnel_does_not_divide_by_zero(self) -> None:
        """All None should still render without raising."""
        f = FunnelSummary()
        text = _render_to_text(render_funnel_full(f), width=120)
        assert "(等待)" in text


class TestRenderFunnelCompact:
    """Compact mode (< 80 cols) collapses to one line."""

    def test_arrow_separated_counts(self) -> None:
        f = FunnelSummary(
            n_main_board=3210,
            n_after_st_susp=3187,
            n_after_t_day_rules=412,
            n_after_turnover=248,
            n_after_vol_rules=35,
        )
        text = _render_to_text(render_funnel_compact(f), width=72)
        assert "3210" in text
        assert "→" in text
        assert "35" in text
        # Compact form should fit on a single line (no '\n' inside the
        # renderable other than the trailing console newline).
        lines = [
            l for l in text.splitlines() if l.strip() and "漏斗" in l
        ]
        assert len(lines) == 1

    def test_missing_counts_become_question_mark(self) -> None:
        f = FunnelSummary(n_main_board=3210)
        text = _render_to_text(render_funnel_compact(f), width=72)
        assert "3210" in text
        assert "?" in text
