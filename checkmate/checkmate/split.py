"""Train / val / OOS split helpers + cherry-pick guard (PR-6.2).

``--split`` syntax (CLI flag, parsed here)::

    --split "train=2014-2020 val=2021-2023 oos=2024-2026"

or with explicit YYYY-MM-DD bounds::

    --split "train=2014-01-01:2020-12-31 val=2021-01-01:2023-12-31 oos=2024-01-01:2026-04-30"

The output is a :class:`Split` dataclass; each segment is a ``(start, end)``
YYYYMMDD pair. The caller (``cli.cmd_backtest``) uses it for:

  * **3-segment backtest reporting** — sequentially run + render each
    segment so cherry-picking is hard.
  * **Grid OOS protection** — :func:`forbid_rank_on_oos` raises if a grid
    is being ranked against the OOS window (the "test set"). The protocol
    is: tune on ``train``, validate on ``val``, *report* on ``oos``.

Year syntax (``2014-2020``) auto-fills January 1 / December 31. Mixed
``2014:2020-06-30`` etc. is rejected to keep parsing predictable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_YEAR_RE = re.compile(r"^(\d{4})-(\d{4})$")
_RANGE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}):(\d{4})-(\d{2})-(\d{2})$")
_YYYYMMDD_RE = re.compile(r"^\d{8}$")


@dataclass
class Segment:
    name: str   # 'train' / 'val' / 'oos' (lowercase)
    start: str  # YYYYMMDD
    end: str    # YYYYMMDD


@dataclass
class Split:
    """Container for the three segments. ``oos`` may be omitted."""
    train: Segment
    val: Segment | None = None
    oos: Segment | None = None

    @property
    def segments(self) -> list[Segment]:
        out = [self.train]
        if self.val is not None:
            out.append(self.val)
        if self.oos is not None:
            out.append(self.oos)
        return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_one(text: str) -> tuple[str, str]:
    """Convert ``"2014-2020"`` or ``"2014-01-01:2020-12-31"`` → ``(start, end)``."""
    text = text.strip()
    m = _YEAR_RE.match(text)
    if m:
        y1, y2 = m.group(1), m.group(2)
        if y1 > y2:
            raise ValueError(f"start year > end year in {text!r}")
        return f"{y1}0101", f"{y2}1231"
    m = _RANGE_RE.match(text)
    if m:
        start = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        end = f"{m.group(4)}{m.group(5)}{m.group(6)}"
        if start > end:
            raise ValueError(f"start > end in {text!r}")
        return start, end
    raise ValueError(
        f"unrecognised segment range {text!r}; "
        "use 'YYYY-YYYY' or 'YYYY-MM-DD:YYYY-MM-DD'"
    )


_VALID_NAMES = ("train", "val", "oos")


def parse_split(spec: str) -> Split:
    """Parse the ``--split`` CLI string into a :class:`Split`.

    Format: space-separated ``name=range`` pairs. ``train`` is required;
    ``val`` / ``oos`` are optional but, if present, must follow ``train``
    chronologically and not overlap.
    """
    if not spec or not spec.strip():
        raise ValueError("split spec is empty")
    parts = spec.strip().split()
    seen: dict[str, Segment] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"expected name=range, got {part!r}")
        name, _, range_text = part.partition("=")
        name = name.strip().lower()
        if name not in _VALID_NAMES:
            raise ValueError(
                f"unknown segment {name!r}; valid: {_VALID_NAMES}"
            )
        if name in seen:
            raise ValueError(f"duplicate segment {name!r}")
        start, end = _parse_one(range_text)
        seen[name] = Segment(name=name, start=start, end=end)
    if "train" not in seen:
        raise ValueError("split must include a 'train' segment")

    # Chronological non-overlap check across the segments present.
    ordered = sorted(seen.values(), key=lambda s: s.start)
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        if cur.start <= prev.end:
            raise ValueError(
                f"segments overlap: {prev.name} ends {prev.end}, "
                f"{cur.name} starts {cur.start}"
            )

    return Split(
        train=seen["train"],
        val=seen.get("val"),
        oos=seen.get("oos"),
    )


# ---------------------------------------------------------------------------
# Cherry-pick guard
# ---------------------------------------------------------------------------


def forbid_rank_on_oos(
    split: Split,
    *,
    grid_start: str | None,
    grid_end: str | None,
) -> None:
    """Raise ``ValueError`` if a grid run would be ranked on the OOS window.

    "Ranked on OOS" means the grid's backtest window falls inside (or
    substantially overlaps with) the OOS segment. Tuning parameters via
    grid search on the OOS window leaks future information and inflates
    apparent performance — :func:`run_grid` should be restricted to
    ``train`` (and validation).

    No-op if no OOS segment was declared.
    """
    if split.oos is None or grid_start is None or grid_end is None:
        return
    g_start = grid_start.replace("-", "")
    g_end = grid_end.replace("-", "")
    oos_start = split.oos.start
    oos_end = split.oos.end

    # Detect any overlap between [g_start, g_end] and [oos_start, oos_end].
    overlaps = not (g_end < oos_start or g_start > oos_end)
    if overlaps:
        raise ValueError(
            f"refusing to grid-search on the OOS window "
            f"({oos_start}..{oos_end}); restrict the grid to the train+val "
            f"segments to avoid cherry-picking. "
            f"(grid window: {g_start}..{g_end})"
        )


__all__ = [
    "Segment",
    "Split",
    "forbid_rank_on_oos",
    "parse_split",
]
