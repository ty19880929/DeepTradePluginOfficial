"""Parameter-grid backtesting (PR-6.1).

Grid file format (YAML)::

    universe:
      amount_20d_avg_min_yuan: [50000000, 100000000]
      listed_days_min: [180, 250]
    entry:
      breakout_lookback: [30, 40, 50]
      atr_stop_mult: [1.5, 2.0]
    risk:
      risk_per_trade: [0.005, 0.01]

Each (section, field, value) tuple becomes a Cartesian dimension. The
example above yields 2 × 2 × 3 × 2 × 2 = 48 cells, each with its own
``BacktestParams`` clone + distinct ``config_hash`` + checkpoint directory.

Lifecycle:

  1. :func:`load_grid_yaml` reads the file → ``{section: {field: [values]}}``.
  2. :func:`expand_grid` returns a list of ``overrides`` dicts (one per cell).
  3. :func:`apply_overrides` builds a fresh :class:`BacktestParams` per cell
     by ``dataclasses.replace``-ing each config subobject.
  4. :func:`run_grid` iterates cells, calls :func:`run_backtest` (which
     reuses existing checkpoints via ``--resume``), collects the metrics
     subset on which we rank, and dumps the aggregate to
     ``~/.deeptrade/checkmate/reports/grid_<timestamp>.json``.

Each cell's ``run_id`` differs across grid invocations, but its
``config_hash`` is stable — so a grid run interrupted halfway recovers
exactly the same per-cell shards on the next launch (PR-4.2 contract).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from . import paths
from .backtest import BacktestOutcome, BacktestParams, run_backtest
from .config import (
    EntryConfig,
    ExecutionConfig,
    ExitConfig,
    FeaturesConfig,
    RegimeConfig,
    RiskConfig,
    UniverseConfig,
)
from .runtime import CheckmateRuntime

logger = logging.getLogger(__name__)


# Section → BacktestParams attribute name + config class. Adding a new
# section is a one-line addition here.
_SECTION_MAP: dict[str, tuple[str, type]] = {
    "universe":  ("universe_cfg",  UniverseConfig),
    "features":  ("features_cfg",  FeaturesConfig),
    "regime":    ("regime_cfg",    RegimeConfig),
    "entry":     ("entry_cfg",     EntryConfig),
    "exit":      ("exit_cfg",      ExitConfig),
    "risk":      ("risk_cfg",      RiskConfig),
    "execution": ("execution_cfg", ExecutionConfig),
}


# ---------------------------------------------------------------------------
# YAML loading + Cartesian expansion
# ---------------------------------------------------------------------------


def load_grid_yaml(path: Path) -> dict[str, dict[str, list[Any]]]:
    """Parse and validate the grid YAML file.

    Raises ``ValueError`` for unknown sections / non-list values.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"grid yaml must be a mapping at the top level, got {type(raw).__name__}")
    out: dict[str, dict[str, list[Any]]] = {}
    for section, fields in raw.items():
        if section not in _SECTION_MAP:
            raise ValueError(
                f"unknown section {section!r}; known: {sorted(_SECTION_MAP)}"
            )
        if not isinstance(fields, dict):
            raise ValueError(f"section {section!r} must be a mapping of field→list")
        cfg_class = _SECTION_MAP[section][1]
        cls_fields = set(cfg_class.__dataclass_fields__)
        for field_name, values in fields.items():
            if field_name not in cls_fields:
                raise ValueError(
                    f"{section}.{field_name!r} is not a field of {cfg_class.__name__}; "
                    f"valid: {sorted(cls_fields)}"
                )
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"{section}.{field_name} must be a non-empty list of values"
                )
        out[section] = dict(fields)
    return out


def expand_grid(grid: dict[str, dict[str, list[Any]]]) -> list[dict[str, dict[str, Any]]]:
    """Cartesian product over every (section, field, value).

    Returns one dict per cell, e.g.::

        {"universe": {"listed_days_min": 250}, "entry": {"breakout_lookback": 40}}

    Empty grid → ``[{}]`` (single no-op cell), matching the "no grid → just
    run with defaults" contract.
    """
    if not grid:
        return [{}]
    # Flatten to a list of (section, field, value) options.
    dims: list[list[tuple[str, str, Any]]] = []
    for section, fields in grid.items():
        for field_name, values in fields.items():
            dims.append([(section, field_name, v) for v in values])
    cells: list[dict[str, dict[str, Any]]] = []
    for combo in product(*dims):
        cell: dict[str, dict[str, Any]] = {}
        for section, field_name, value in combo:
            cell.setdefault(section, {})[field_name] = value
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Applying overrides to BacktestParams
# ---------------------------------------------------------------------------


def apply_overrides(
    base: BacktestParams, overrides: dict[str, dict[str, Any]],
) -> BacktestParams:
    """Return a new ``BacktestParams`` with the per-section overrides applied.

    Each section's config object is replaced via :func:`dataclasses.replace`
    so the resulting params has independent per-cell config blocks (mutating
    one cell's params does not affect another's).
    """
    kwargs: dict[str, Any] = {}
    for section, fields in (overrides or {}).items():
        attr_name, cfg_class = _SECTION_MAP[section]
        current = getattr(base, attr_name) or cfg_class()
        # field validation has already happened in load_grid_yaml.
        kwargs[attr_name] = replace(current, **fields)
    return replace(base, **kwargs)


# ---------------------------------------------------------------------------
# Ranking + serialisation
# ---------------------------------------------------------------------------


def _extract_rank_value(outcome: BacktestOutcome, rank_by: str) -> float | None:
    """Pull the ranking metric off a BacktestOutcome."""
    if rank_by == "cagr":
        # CAGR is not on the outcome directly; approximate from final/initial.
        # Future versions can wire the full Report's metrics in here.
        if outcome.n_days < 1:
            return None
        # Use the same definition as report._compute_cagr.
        initial = max(outcome.final_cash, 1.0)  # rough proxy; cells have same init cash
        years = outcome.n_days / 252.0
        if years < 1.0 / 252.0:
            return None
        # Approx CAGR from equity growth — exact value reproduced via report
        # later; this is just for sorting.
        if outcome.final_equity <= 0:
            return None
        # `outcome.final_cash` is whatever's left in cash; we want equity / initial_cash.
        # The cell's BacktestParams.initial_cash is in scope at the call site;
        # callers pass it through `_run_one_cell`.
        return None
    if rank_by == "final_equity":
        return float(outcome.final_equity)
    if rank_by == "max_drawdown":
        return float(outcome.max_drawdown)
    if rank_by == "n_fills":
        return float(outcome.n_fills)
    return None


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------


def _run_one_cell(
    rt: CheckmateRuntime,
    base: BacktestParams,
    overrides: dict[str, dict[str, Any]],
    *,
    echo: Callable[[str], None],
) -> tuple[BacktestOutcome, BacktestParams]:
    cell_params = apply_overrides(base, overrides)
    echo(f"[grid] cell overrides={overrides}")
    outcome = run_backtest(rt, cell_params, echo=echo)
    return outcome, cell_params


# ---------------------------------------------------------------------------
# Top-level grid runner
# ---------------------------------------------------------------------------


def run_grid(
    rt: CheckmateRuntime,
    base: BacktestParams,
    grid: dict[str, dict[str, list[Any]]],
    *,
    rank_by: str = "final_equity",
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every cell of ``grid`` and return an aggregate dict.

    The aggregate is also persisted to
    ``~/.deeptrade/checkmate/reports/grid_<UTC-timestamp>.json``. Resume
    semantics are inherited from each cell's :func:`run_backtest` call —
    cells whose checkpoint exists skip immediately.
    """
    cells = expand_grid(grid)
    echo(f"[grid] running {len(cells)} cell(s) ; rank_by={rank_by}")

    results: list[dict[str, Any]] = []
    for i, overrides in enumerate(cells, start=1):
        echo(f"[grid] [{i}/{len(cells)}] starting cell")
        outcome, cell_params = _run_one_cell(rt, base, overrides, echo=echo)
        initial = float(cell_params.initial_cash) or 1.0
        cagr = None
        if outcome.n_days >= 1 and outcome.final_equity > 0:
            years = outcome.n_days / 252.0
            if years >= 1.0 / 252.0:
                cagr = (outcome.final_equity / initial) ** (1.0 / years) - 1.0
        rec = {
            "cell_index": i - 1,
            "overrides": overrides,
            "run_id": outcome.run_id,
            "config_hash": outcome.config_hash,
            "start": outcome.start,
            "end": outcome.end,
            "n_days": outcome.n_days,
            "n_fills": outcome.n_fills,
            "final_equity": outcome.final_equity,
            "final_cash": outcome.final_cash,
            "max_drawdown": outcome.max_drawdown,
            "cagr": cagr,
        }
        results.append(rec)

    # Rank — higher is better for cagr/final_equity; lower is better for drawdown.
    def _sort_key(r: dict[str, Any]) -> Any:
        v = r.get(rank_by)
        if v is None:
            # NaN/None pushed to the bottom regardless of direction.
            return (1, 0.0) if _is_higher_better(rank_by) else (1, 0.0)
        return (0, -float(v)) if _is_higher_better(rank_by) else (0, float(v))

    ranked = sorted(results, key=_sort_key)
    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rank_by": rank_by,
        "n_cells": len(results),
        "grid": grid,
        "base_start": base.start,
        "base_end": base.end,
        "initial_cash": base.initial_cash,
        "cells": results,
        "ranked": ranked,
    }

    paths.reports_dir().mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = paths.reports_dir() / f"grid_{ts}.json"
    out_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    aggregate["output_path"] = str(out_path)
    echo(f"[grid] wrote {out_path}")
    return aggregate


def _is_higher_better(rank_by: str) -> bool:
    """Whether bigger values of the metric are 'better' for ranking purposes."""
    return rank_by not in {"max_drawdown"}


__all__ = [
    "apply_overrides",
    "expand_grid",
    "load_grid_yaml",
    "run_grid",
]
