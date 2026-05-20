"""accumulation-probe-washout v0.3.0 — Stats query layer.

Pure read-only SQL aggregation over ``apw_signal_history`` JOIN
``apw_realized_returns`` (optionally joining ``apw_stage_results`` for the
LLM-prediction / pattern axes). Consumed by the ``stats`` CLI subcommand and
later by any LGB / multi-agent feature pipeline.

The shape returned to the renderer is:
    [{bucket: str,
      n_samples: int,
      ret_t5_mean: float | None,
      max_high_t5_mean: float | None,
      max_drawdown_t5_mean: float | None,
      label_launch_t5_winrate: float | None}, ...]
"""

from __future__ import annotations

from typing import Any

# Default bucket schedules — kept stable so the renderer / CLI tables can
# memo the column count without inspecting payloads.
LAUNCH_SCORE_BINS: list[tuple[str, float, float]] = [
    ("0-40", 0.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.0),
]
SCORE_BINS_GENERIC: list[tuple[str, float, float]] = [
    ("0-40", 0.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.0),
]
LGB_SCORE_BINS: list[tuple[str, float, float]] = [
    ("0-30", 0.0, 30.0),
    ("30-50", 30.0, 50.0),
    ("50-70", 50.0, 70.0),
    ("70-100", 70.0, 100.0),
]

# 6 dim_* columns physicalised in migration 20260520_002.
DIMENSION_COLS: tuple[str, ...] = (
    "dim_accumulation",
    "dim_probe",
    "dim_washout",
    "dim_launch_timing",
    "dim_capital_confirmation",
    "dim_risk",
)

# Score columns living on ``apw_signal_history`` that can be bucketed via
# ``--by <col>_bin``.
_SIGNAL_HISTORY_SCORE_BIN_COLS: dict[str, str] = {
    "accumulation_score_bin": "accumulation_score",
    "probe_quality_score_bin": "probe_quality_score",
    "washout_score_bin": "washout_score",
    "launch_setup_score_bin": "launch_setup_score",
}

# Recognised --by axes; ordering kept stable for help-text readability.
ALLOWED_BY: tuple[str, ...] = (
    "phase",
    "prediction",
    "main_pattern",
    "launch_score_bin",
    "accumulation_score_bin",
    "probe_quality_score_bin",
    "washout_score_bin",
    "launch_setup_score_bin",
    "dimension_scores",
    "lgb_score_bin",
)


class StatsQueryError(ValueError):
    """Raised when --by is invalid or a downstream requirement is missing."""


def run_stats_query(
    db: Any,
    *,
    from_date: str | None,
    to_date: str | None,
    by: str,
) -> tuple[list[dict[str, Any]], str]:
    """Execute the aggregation; return ``(rows, table_title)``."""
    if by not in ALLOWED_BY:
        raise StatsQueryError(
            f"unknown --by={by!r}; choose from {', '.join(ALLOWED_BY)}"
        )

    title_window = f"{from_date or '*'}–{to_date or '*'}"

    if by == "lgb_score_bin":
        # apw_lgb_predictions ships in PR-4 — surface a clear "not yet" error.
        try:
            db.fetchone("SELECT 1 FROM apw_lgb_predictions LIMIT 1")
        except Exception:  # noqa: BLE001 — DuckDB raises CatalogException
            raise StatsQueryError(
                "lgb_score_bin requires apw_lgb_predictions (PR-4). "
                "Re-run after the PR-4 migration lands."
            ) from None
        return (
            _by_lgb_score_bin(db, from_date=from_date, to_date=to_date),
            f"按 lgb_score_bin 维度（{title_window}）",
        )

    if by == "dimension_scores":
        return (
            _by_dimension_scores(db, from_date=from_date, to_date=to_date),
            f"6 个维度分与 ret_t5 的 Pearson 相关系数（{title_window}）",
        )

    if by == "launch_score_bin":
        return (
            _by_score_bin_on_stage_results(
                db,
                from_date=from_date,
                to_date=to_date,
                col="launch_score",
                bins=LAUNCH_SCORE_BINS,
            ),
            f"按 launch_score_bin 维度（{title_window}）",
        )

    if by in _SIGNAL_HISTORY_SCORE_BIN_COLS:
        return (
            _by_score_bin_on_signal_history(
                db,
                from_date=from_date,
                to_date=to_date,
                col=_SIGNAL_HISTORY_SCORE_BIN_COLS[by],
                bins=SCORE_BINS_GENERIC,
            ),
            f"按 {by} 维度（{title_window}）",
        )

    # phase / prediction / main_pattern → group-by on apw_stage_results.
    return (
        _by_categorical_on_stage_results(
            db, from_date=from_date, to_date=to_date, col=by
        ),
        f"按 {by} 分组（{title_window}）",
    )


# ---------------------------------------------------------------------------
# Group-by helpers
# ---------------------------------------------------------------------------


def _date_filter(
    *, from_date: str | None, to_date: str | None, col: str = "r.signal_date"
) -> tuple[str, list[Any]]:
    """Return ``" AND <col> BETWEEN ? AND ?"`` style fragment + bind list."""
    clauses: list[str] = []
    binds: list[Any] = []
    if from_date:
        clauses.append(f"{col} >= ?")
        binds.append(from_date)
    if to_date:
        clauses.append(f"{col} <= ?")
        binds.append(to_date)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), binds


def _by_categorical_on_stage_results(
    db: Any, *, from_date: str | None, to_date: str | None, col: str
) -> list[dict[str, Any]]:
    date_sql, binds = _date_filter(
        from_date=from_date, to_date=to_date, col="r.signal_date"
    )
    sql = f"""
        SELECT s.{col} AS bucket,
               COUNT(*) AS n_samples,
               AVG(r.ret_t5_pct) AS ret_t5_mean,
               AVG(r.max_high_t5_pct) AS max_high_t5_mean,
               AVG(r.max_drawdown_t5_pct) AS max_drawdown_t5_mean,
               AVG(CASE WHEN r.label_launch_t5 = 1 THEN 100.0
                        WHEN r.label_launch_t5 = 0 THEN 0.0
                        ELSE NULL END) AS label_launch_t5_winrate
        FROM apw_stage_results s
        JOIN apw_realized_returns r
          ON s.ts_code = r.ts_code AND s.trade_date = r.signal_date
        WHERE r.ret_t5_pct IS NOT NULL{date_sql}
        GROUP BY s.{col}
        ORDER BY s.{col}
    """
    return _materialise_buckets(db, sql, binds)


def _by_score_bin_on_stage_results(
    db: Any,
    *,
    from_date: str | None,
    to_date: str | None,
    col: str,
    bins: list[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    return _by_score_bin(
        db,
        from_date=from_date,
        to_date=to_date,
        col=f"s.{col}",
        bins=bins,
        from_join=(
            "FROM apw_stage_results s "
            "JOIN apw_realized_returns r "
            "  ON s.ts_code = r.ts_code AND s.trade_date = r.signal_date"
        ),
    )


def _by_score_bin_on_signal_history(
    db: Any,
    *,
    from_date: str | None,
    to_date: str | None,
    col: str,
    bins: list[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    return _by_score_bin(
        db,
        from_date=from_date,
        to_date=to_date,
        col=f"h.{col}",
        bins=bins,
        from_join=(
            "FROM apw_signal_history h "
            "JOIN apw_realized_returns r "
            "  ON h.ts_code = r.ts_code AND h.trade_date = r.signal_date"
        ),
    )


def _by_score_bin(
    db: Any,
    *,
    from_date: str | None,
    to_date: str | None,
    col: str,
    bins: list[tuple[str, float, float]],
    from_join: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date_sql, base_binds = _date_filter(
        from_date=from_date, to_date=to_date, col="r.signal_date"
    )
    for label, lo, hi in bins:
        sql = f"""
            SELECT '{label}' AS bucket,
                   COUNT(*) AS n_samples,
                   AVG(r.ret_t5_pct) AS ret_t5_mean,
                   AVG(r.max_high_t5_pct) AS max_high_t5_mean,
                   AVG(r.max_drawdown_t5_pct) AS max_drawdown_t5_mean,
                   AVG(CASE WHEN r.label_launch_t5 = 1 THEN 100.0
                            WHEN r.label_launch_t5 = 0 THEN 0.0
                            ELSE NULL END) AS label_launch_t5_winrate
            {from_join}
            WHERE r.ret_t5_pct IS NOT NULL
              AND {col} >= ? AND {col} < ?{date_sql}
        """
        binds = [lo, hi, *base_binds]
        agg = db.fetchone(sql, binds) or (label, 0, None, None, None, None)
        rows.append(_row_to_dict(agg))
    return rows


def _by_lgb_score_bin(
    db: Any, *, from_date: str | None, to_date: str | None
) -> list[dict[str, Any]]:
    # NOTE: only reached in PR-4+; PR-1 surfaces the friendly "not yet" error
    # from run_stats_query before getting here.
    rows: list[dict[str, Any]] = []
    date_sql, base_binds = _date_filter(
        from_date=from_date, to_date=to_date, col="r.signal_date"
    )
    for label, lo, hi in LGB_SCORE_BINS:
        sql = f"""
            SELECT '{label}' AS bucket,
                   COUNT(*) AS n_samples,
                   AVG(r.ret_t5_pct) AS ret_t5_mean,
                   AVG(r.max_high_t5_pct) AS max_high_t5_mean,
                   AVG(r.max_drawdown_t5_pct) AS max_drawdown_t5_mean,
                   AVG(CASE WHEN r.label_launch_t5 = 1 THEN 100.0
                            WHEN r.label_launch_t5 = 0 THEN 0.0
                            ELSE NULL END) AS label_launch_t5_winrate
            FROM apw_lgb_predictions p
            JOIN apw_realized_returns r
              ON p.ts_code = r.ts_code AND p.trade_date = r.signal_date
            WHERE r.ret_t5_pct IS NOT NULL
              AND p.lgb_score >= ? AND p.lgb_score < ?{date_sql}
        """
        binds = [lo, hi, *base_binds]
        agg = db.fetchone(sql, binds) or (label, 0, None, None, None, None)
        rows.append(_row_to_dict(agg))
    return rows


def _by_dimension_scores(
    db: Any, *, from_date: str | None, to_date: str | None
) -> list[dict[str, Any]]:
    """Pearson r between each dim_* column and ret_t5_pct, +1 sample count.

    DuckDB has a native ``corr(x, y)`` window/aggregate function that yields
    the same statistic as scipy.stats.pearsonr.
    """
    date_sql, binds = _date_filter(
        from_date=from_date, to_date=to_date, col="r.signal_date"
    )
    rows: list[dict[str, Any]] = []
    for col in DIMENSION_COLS:
        sql = f"""
            SELECT '{col}' AS bucket,
                   COUNT(*) AS n_samples,
                   AVG(r.ret_t5_pct) AS ret_t5_mean,
                   corr(s.{col}, r.ret_t5_pct) AS pearson_r_ret_t5,
                   corr(s.{col}, r.max_high_t5_pct) AS pearson_r_max_high_t5,
                   AVG(CASE WHEN r.label_launch_t5 = 1 THEN 100.0
                            WHEN r.label_launch_t5 = 0 THEN 0.0
                            ELSE NULL END) AS label_launch_t5_winrate
            FROM apw_stage_results s
            JOIN apw_realized_returns r
              ON s.ts_code = r.ts_code AND s.trade_date = r.signal_date
            WHERE s.{col} IS NOT NULL AND r.ret_t5_pct IS NOT NULL{date_sql}
        """
        agg = db.fetchone(sql, binds) or (col, 0, None, None, None, None)
        rows.append(
            {
                "bucket": str(agg[0]),
                "n_samples": int(agg[1] or 0),
                "ret_t5_mean": _to_float(agg[2]),
                "pearson_r_ret_t5": _to_float(agg[3]),
                "pearson_r_max_high_t5": _to_float(agg[4]),
                "label_launch_t5_winrate": _to_float(agg[5]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _materialise_buckets(db: Any, sql: str, binds: list[Any]) -> list[dict[str, Any]]:
    return [_row_to_dict(r) for r in (db.fetchall(sql, binds) or [])]


def _row_to_dict(r: Any) -> dict[str, Any]:
    return {
        "bucket": str(r[0]),
        "n_samples": int(r[1] or 0),
        "ret_t5_mean": _to_float(r[2]),
        "max_high_t5_mean": _to_float(r[3]),
        "max_drawdown_t5_mean": _to_float(r[4]),
        "label_launch_t5_winrate": _to_float(r[5]),
    }


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
