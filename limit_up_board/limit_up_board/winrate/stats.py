"""统计聚合 — 把 ResolvedRecord 列表压成 summary / by_prediction / by_rank_bucket。

PR #2 — 纯函数，不接 DB / 不接终端格式化。CLI 层和 LLM Review payload 都从
这里取数。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .resolver import ResolvedRecord


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WinrateSummary:
    """Overall window-level statistics."""

    total: int
    resolved: int
    unresolved: int
    win: int
    flat: int
    loss: int
    strict_win_rate: float | None       # win / resolved
    non_loss_rate: float | None         # (win + flat) / resolved
    avg_open_vs_limit_pct: float | None # mean across resolved


@dataclass(frozen=True)
class GroupStat:
    """Per-group aggregate (e.g. by prediction class, by rank bucket)."""

    key: str
    total: int
    resolved: int
    win: int
    flat: int
    loss: int
    strict_win_rate: float | None
    avg_open_vs_limit_pct: float | None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate(resolved_list: list[ResolvedRecord], key: str) -> GroupStat:
    total = len(resolved_list)
    counts = {"win": 0, "flat": 0, "loss": 0, "unresolved": 0}
    pcts: list[float] = []
    for r in resolved_list:
        counts[r.outcome] += 1
        if r.outcome != "unresolved" and r.open_vs_limit_pct is not None:
            pcts.append(r.open_vs_limit_pct)
    resolved = total - counts["unresolved"]
    strict = counts["win"] / resolved if resolved > 0 else None
    avg = sum(pcts) / len(pcts) if pcts else None
    return GroupStat(
        key=key,
        total=total,
        resolved=resolved,
        win=counts["win"],
        flat=counts["flat"],
        loss=counts["loss"],
        strict_win_rate=strict,
        avg_open_vs_limit_pct=avg,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def summarize(resolved: list[ResolvedRecord]) -> WinrateSummary:
    """Window-level summary."""
    g = _aggregate(resolved, key="__all__")
    non_loss = (
        (g.win + g.flat) / g.resolved if g.resolved > 0 else None
    )
    return WinrateSummary(
        total=g.total,
        resolved=g.resolved,
        unresolved=g.total - g.resolved,
        win=g.win,
        flat=g.flat,
        loss=g.loss,
        strict_win_rate=g.strict_win_rate,
        non_loss_rate=non_loss,
        avg_open_vs_limit_pct=g.avg_open_vs_limit_pct,
    )


def group_by_prediction(resolved: list[ResolvedRecord]) -> list[GroupStat]:
    """Group by `prediction` class. Returns stats in canonical class order."""
    buckets: dict[str, list[ResolvedRecord]] = defaultdict(list)
    for r in resolved:
        buckets[r.record.prediction].append(r)

    # Canonical order for deterministic display + JSON output
    canonical = ["top_candidate", "watchlist", "avoid"]
    out: list[GroupStat] = []
    for k in canonical:
        if k in buckets:
            out.append(_aggregate(buckets[k], key=k))
    # Append any non-canonical class (defensive — schemas enforce 3 classes)
    for k, items in buckets.items():
        if k not in canonical:
            out.append(_aggregate(items, key=k))
    return out


def group_by_rank_bucket(resolved: list[ResolvedRecord]) -> list[GroupStat]:
    """Group by rank bucket: 1-3 / 4-10 / 11+."""
    buckets: dict[str, list[ResolvedRecord]] = defaultdict(list)
    for r in resolved:
        rank = r.record.rank
        if rank <= 3:
            key = "Top 1-3"
        elif rank <= 10:
            key = "Top 4-10"
        else:
            key = "Top 11+"
        buckets[key].append(r)
    out: list[GroupStat] = []
    for k in ["Top 1-3", "Top 4-10", "Top 11+"]:
        if k in buckets:
            out.append(_aggregate(buckets[k], key=k))
    return out
