"""The 5-step screening funnel — VA-exclusive UI element (Plan §4.2, G5).

The ``screen`` mode passes its watchlist through five sequential filters
(main board → ST/suspension → T-day rules → turnover → volume). Each
filter's surviving count is captured in the ``DATA_SYNC_FINISHED.payload``
the runner emits at end of step 1 (``runner.py:209``).

This module owns the small data model + the rich renderable that turns
those five counts into a horizontal bar chart. The bar widths scale to the
largest count (always ``n_main_board``) so users can eyeball the
"survivor ratio" at each stage.

analyze mode skips this entirely (``DashboardState.funnel is None`` →
``layout._format_funnel_panel`` returns ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text

if TYPE_CHECKING:  # pragma: no cover
    from rich.console import RenderableType


@dataclass
class FunnelSummary:
    """5 surviving counts, populated by ``DATA_SYNC_FINISHED.payload``.

    Field names mirror the runner's payload keys so the dashboard can
    splat them in via ``setattr(funnel, key, value)`` (Plan §3.2.3).
    Any field left ``None`` is still rendered (with a "(等待)" placeholder)
    so the user sees the filter list even before sync finishes.
    """

    n_main_board: int | None = None
    n_after_st_susp: int | None = None
    n_after_t_day_rules: int | None = None
    n_after_turnover: int | None = None
    n_after_vol_rules: int | None = None


# Display order + Chinese label per step. Tuple position is the field-name
# (must match :class:`FunnelSummary` attrs) so attribute lookup stays
# dynamic when the payload arrives.
_STEPS: tuple[tuple[str, str], ...] = (
    ("n_main_board", "主板上市"),
    ("n_after_st_susp", "排除 ST/停牌"),
    ("n_after_t_day_rules", "T 日规则"),
    ("n_after_turnover", "换手率筛选"),
    ("n_after_vol_rules", "量能筛选"),
)


def _max_count(funnel: FunnelSummary) -> int:
    """The denominator for bar-width ratios. Defensive against partial fills:
    if no field is populated yet, fall back to 1 so we don't divide by zero.
    """
    counts = [
        getattr(funnel, name)
        for name, _ in _STEPS
        if getattr(funnel, name) is not None
    ]
    if not counts:
        return 1
    return max(int(c) for c in counts) or 1


def render_funnel_full(funnel: FunnelSummary, *, bar_width: int = 32) -> Group:
    """Full layout — one row per filter step with a horizontal bar.

    Used in standard (≥ 100 cols) and medium (80–99 cols) terminals; the
    compact mode (< 80 cols) calls :func:`render_funnel_compact` instead.
    """
    max_cnt = _max_count(funnel)
    rows: list[RenderableType] = []
    prev: int | None = None
    label_w = max(len(label) for _, label in _STEPS)
    for attr, label in _STEPS:
        count = getattr(funnel, attr)
        line = Text()
        line.append(f"{label:<{label_w}}  ", style="k.label")
        if count is None:
            line.append("(等待)", style="dim")
        else:
            n = int(count)
            filled = max(1, int(round(n / max_cnt * bar_width)))
            bar = "█" * filled + " " * (bar_width - filled)
            line.append(bar, style="status.success")
            line.append(f"  {n:>6}", style="status.success")
            if prev is not None:
                delta = n - prev
                if delta:
                    line.append(
                        f" ({delta:+d})",
                        style="status.error" if delta < 0 else "dim",
                    )
            prev = n
        rows.append(line)
    return Group(*rows)


def render_funnel_compact(funnel: FunnelSummary) -> Text:
    """Single-line fallback for terminals < 80 cols.

    Renders as ``3210 → 3187 → 412 → 248 → 35`` with N/A placeholders for
    not-yet-arrived steps.
    """
    parts: list[str] = []
    for attr, _ in _STEPS:
        count = getattr(funnel, attr)
        parts.append("?" if count is None else str(int(count)))
    out = Text()
    out.append("漏斗 ", style="k.label")
    out.append(" → ".join(parts), style="status.success")
    return out


__all__ = [
    "FunnelSummary",
    "render_funnel_compact",
    "render_funnel_full",
]
