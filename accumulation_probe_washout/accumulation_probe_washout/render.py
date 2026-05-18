"""Terminal output helpers — evaluate report + analyze summary.

Plain ``typer.echo`` rendering for legacy-mode reports; the rich dashboard
handles run-time visualization separately.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable

import typer


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(pct / 100 * (len(values) - 1)))))
    return values[k]


def _summarize(rows: list[dict[str, Any]], horizons: list[int]) -> dict[str, Any]:
    """Aggregate stats for one group of rows."""
    n = len(rows)
    out: dict[str, Any] = {"n": n}
    for h in horizons:
        key = f"ret_t{h}_pct"
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            out[f"mean_ret_t{h}"] = round(statistics.mean(vals), 2)
            out[f"median_ret_t{h}"] = round(statistics.median(vals), 2)
            out[f"hit_rate_t{h}"] = round(sum(1 for v in vals if v > 0) / len(vals) * 100, 2)
        else:
            out[f"mean_ret_t{h}"] = None
            out[f"median_ret_t{h}"] = None
            out[f"hit_rate_t{h}"] = None
    dd = [r.get("max_drawdown_t5_pct") for r in rows if r.get("max_drawdown_t5_pct") is not None]
    out["max_dd_t5"] = round(max(dd), 2) if dd else None
    label5 = [r["label_launch_t5"] for r in rows if r.get("label_launch_t5") is not None]
    out["label_launch_t5_rate"] = round(sum(label5) / len(label5) * 100, 2) if label5 else None
    return out


def _bucket_launch_score(score: float | None) -> str:
    if score is None:
        return "n/a"
    if score >= 80:
        return "≥80"
    if score >= 70:
        return "70-79"
    if score >= 60:
        return "60-69"
    return "<60"


def render_evaluate_report(
    rows: list[dict[str, Any]],
    *,
    horizons: Iterable[int] = (1, 3, 5, 10),
    include_early_phases: bool = False,
) -> None:
    """Print a structured terminal report.

    rows must include at least: ts_code, signal_date, phase, prediction,
    launch_score, ret_t{h}_pct, max_drawdown_t5_pct, label_launch_t5.
    """
    horizons_list = list(horizons)
    typer.echo("\n=== APW evaluate 报告 ===")
    typer.echo(f"样本数: {len(rows)}  | horizons: {horizons_list}  "
               f"| include_early_phases={include_early_phases}")

    if not rows:
        typer.echo("(无样本)\n")
        return

    typer.echo("\n[overall]")
    s = _summarize(rows, horizons_list)
    _print_summary("overall", s, horizons_list)

    # ---- group by prediction
    typer.echo("\n[by prediction]")
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r.get("prediction", "n/a")].append(r)
    for key, sub in sorted(groups.items()):
        _print_summary(key, _summarize(sub, horizons_list), horizons_list)

    # ---- group by launch_score bucket
    typer.echo("\n[by launch_score bucket]")
    groups.clear()
    for r in rows:
        groups[_bucket_launch_score(r.get("launch_score"))].append(r)
    for key in ("≥80", "70-79", "60-69", "<60", "n/a"):
        sub = groups.get(key, [])
        if sub:
            _print_summary(key, _summarize(sub, horizons_list), horizons_list)

    # ---- group by phase
    typer.echo("\n[by phase]")
    groups.clear()
    for r in rows:
        groups[r.get("phase", "n/a")].append(r)
    for key, sub in sorted(groups.items()):
        _print_summary(key, _summarize(sub, horizons_list), horizons_list)


def _print_summary(label: str, s: dict[str, Any], horizons: list[int]) -> None:
    parts = [f"{label} (n={s['n']})"]
    for h in horizons:
        parts.append(
            f"T+{h}: mean={s[f'mean_ret_t{h}']} med={s[f'median_ret_t{h}']} "
            f"hit={s[f'hit_rate_t{h}']}%"
        )
    parts.append(f"label_t5_rate={s['label_launch_t5_rate']}%")
    typer.echo("  " + "  ".join(parts))
