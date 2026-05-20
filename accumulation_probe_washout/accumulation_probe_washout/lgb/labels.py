"""Label construction for APW LightGBM training.

Labels come straight from ``apw_realized_returns`` — APW's existing T+N
backfill table — so the training pipeline performs **zero** additional
Tushare calls. ``evaluate`` populates the table; ``labels.fetch_labels_for_window``
just JOINs on ``(signal_date, ts_code)`` and applies the configured label.

Three supported sources:

* ``label_launch_t5``  — APW's pre-computed t5 启动标签 (uses cfg.label_t5_*
  for the underlying return/drawdown thresholds). Default.
* ``label_launch_t10`` — t10 启动标签.
* ``custom_t5``        — ``max_high_t5_pct >= threshold_pct AND
  max_drawdown_t5_pct <= drawdown_threshold_pct``. Caller-supplied
  thresholds let researchers override the default t5 envelope without
  re-running ``evaluate``.

Returned labels are ``int8`` 0/1; rows with NULL underlying columns drop
out (the trainer filters down to labeled rows).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


VALID_LABEL_SOURCES = ("label_launch_t5", "label_launch_t10", "custom_t5")


class LgbLabelError(ValueError):
    """Bad label_source or non-positive thresholds."""


def _validate(
    *,
    source: str,
    threshold_pct: float | None,
    drawdown_threshold_pct: float | None,
) -> None:
    if source not in VALID_LABEL_SOURCES:
        raise LgbLabelError(
            f"label_source must be one of {VALID_LABEL_SOURCES}, got {source!r}"
        )
    if source == "custom_t5":
        if threshold_pct is None or threshold_pct <= 0:
            raise LgbLabelError(
                f"custom_t5 requires --label-threshold > 0; got {threshold_pct}"
            )
        if drawdown_threshold_pct is None or drawdown_threshold_pct <= 0:
            raise LgbLabelError(
                f"custom_t5 requires --label-drawdown-threshold > 0; got "
                f"{drawdown_threshold_pct}"
            )


def fetch_labels_for_window(
    db: Database,
    *,
    start_date: str,
    end_date: str,
    source: str = "label_launch_t5",
    threshold_pct: float | None = None,
    drawdown_threshold_pct: float | None = None,
) -> pd.DataFrame:
    """``DataFrame[signal_date, ts_code, label]`` for the window.

    ``data_status`` is filtered to ``complete | partial`` so half-evaluated
    rows don't poison the trainer. Caller filters NaN labels downstream.
    """
    _validate(
        source=source,
        threshold_pct=threshold_pct,
        drawdown_threshold_pct=drawdown_threshold_pct,
    )
    if source in ("label_launch_t5", "label_launch_t10"):
        rows = db.fetchall(
            f"SELECT signal_date, ts_code, {source} FROM apw_realized_returns "
            f"WHERE signal_date BETWEEN ? AND ? "
            f"AND data_status IN ('complete', 'partial') "
            f"AND {source} IS NOT NULL",
            (start_date, end_date),
        ) or []
        return pd.DataFrame(
            [(str(r[0]), str(r[1]), int(r[2])) for r in rows if r[2] is not None],
            columns=["signal_date", "ts_code", "label"],
        )

    # custom_t5 derived inline so it lives next to the canonical columns.
    rows = db.fetchall(
        "SELECT signal_date, ts_code, max_high_t5_pct, max_drawdown_t5_pct "
        "FROM apw_realized_returns "
        "WHERE signal_date BETWEEN ? AND ? "
        "AND data_status IN ('complete', 'partial') "
        "AND max_high_t5_pct IS NOT NULL "
        "AND max_drawdown_t5_pct IS NOT NULL",
        (start_date, end_date),
    ) or []
    out: list[tuple[str, str, int]] = []
    for sd, ts, hi, dd in rows:
        positive = float(hi) >= float(threshold_pct) and float(dd) <= float(
            drawdown_threshold_pct
        )
        out.append((str(sd), str(ts), 1 if positive else 0))
    return pd.DataFrame(out, columns=["signal_date", "ts_code", "label"])
