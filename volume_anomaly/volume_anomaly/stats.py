"""volume-anomaly v0.4.0 — Stats query layer.

Pure read-only SQL aggregation over ``va_stage_results`` JOIN
``va_realized_returns``. Consumed by the ``stats`` CLI subcommand and any
future LightGBM / Multi-Agent feature pipeline.

The shape returned to the renderer is:
    [{bucket: str, n_samples: int, t3_mean: float | None,
      t3_winrate: float | None, t5_max_ret_mean: float | None}, ...]
"""

from __future__ import annotations

from typing import Any

# G4 — default launch_score bins. v0.7+ may expose a `--bins` flag.
LAUNCH_SCORE_BINS: list[tuple[str, float, float]] = [
    ("0-40", 0.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.0),
]


DIMENSION_COLS: tuple[str, ...] = (
    "washout", "pattern", "capital", "sector", "historical", "risk",
)


def run_stats_query(
    db: Any,
    *,
    from_date: str | None,
    to_date: str | None,
    by: str,
) -> tuple[list[dict[str, Any]], str]:
    """Execute the aggregation; return (rows, table_title)."""
    if by not in {"prediction", "pattern", "launch_score_bin", "dimension_scores"}:
        raise ValueError(
            f"unknown --by={by!r}; choose from prediction | pattern | "
            f"launch_score_bin | dimension_scores"
        )

    # Build WHERE clause for date range. We bind via parameters to avoid SQL
    # injection even though these are user-supplied YYYYMMDD strings.
    where_clauses = ["s.trade_date = r.anomaly_date"]
    params: list[Any] = []
    if from_date:
        where_clauses.append("r.anomaly_date >= ?")
        params.append(from_date)
    if to_date:
        where_clauses.append("r.anomaly_date <= ?")
        params.append(to_date)
    where_sql = " AND ".join(where_clauses)

    if by == "launch_score_bin":
        return _by_launch_score_bin(db, where_sql=where_sql, params=params), (
            f"按 launch_score_bin 维度（{from_date or '*'}–{to_date or '*'}）"
        )

    if by == "dimension_scores":
        return _by_dimension_scores(db, where_sql=where_sql, params=params), (
            f"按 dimension_scores 维度与 ret_t3 的 Pearson 相关系数"
            f"（{from_date or '*'}–{to_date or '*'}）"
        )

    # Generic group-by for prediction / pattern.
    group_col = f"s.{by}"
    sql = f"""
        SELECT {group_col} AS bucket,
               COUNT(*) AS n_samples,
               AVG(r.ret_t3) AS t3_mean,
               AVG(CASE WHEN r.ret_t3 > 0 THEN 100.0 ELSE 0.0 END) AS t3_winrate,
               AVG(r.max_ret_5d) AS t5_max_ret_mean
        FROM va_stage_results s
        JOIN va_realized_returns r
          ON s.ts_code = r.ts_code AND {where_sql}
        WHERE r.ret_t3 IS NOT NULL
        GROUP BY {group_col}
        ORDER BY {group_col}
    """
    rows = db.fetchall(sql, tuple(params))
    title = f"按 {by} 维度（{from_date or '*'}–{to_date or '*'}）"
    return [_row_to_dict(r) for r in rows], title


def _by_launch_score_bin(
    db: Any, *, where_sql: str, params: list[Any]
) -> list[dict[str, Any]]:
    """Bin launch_score into LAUNCH_SCORE_BINS and aggregate per bucket.

    Implemented with a CASE expression to keep the query fully SQL-side
    (DuckDB executes faster than fetching everything to Python and binning).
    """
    case_lines = []
    for label, lo, hi in LAUNCH_SCORE_BINS:
        # Half-open low, closed high — inclusive of upper bound for the last bin
        is_last = label == LAUNCH_SCORE_BINS[-1][0]
        op_hi = "<=" if is_last else "<"
        case_lines.append(
            f"WHEN s.launch_score >= {lo} AND s.launch_score {op_hi} {hi} THEN '{label}'"
        )
    case_expr = "CASE " + " ".join(case_lines) + " ELSE 'oob' END"

    sql = f"""
        SELECT {case_expr} AS bucket,
               COUNT(*) AS n_samples,
               AVG(r.ret_t3) AS t3_mean,
               AVG(CASE WHEN r.ret_t3 > 0 THEN 100.0 ELSE 0.0 END) AS t3_winrate,
               AVG(r.max_ret_5d) AS t5_max_ret_mean
        FROM va_stage_results s
        JOIN va_realized_returns r
          ON s.ts_code = r.ts_code AND {where_sql}
        WHERE r.ret_t3 IS NOT NULL AND s.launch_score IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    """
    rows = db.fetchall(sql, tuple(params))
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(r: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "bucket": r[0],
        "n_samples": int(r[1] or 0),
        "t3_mean": float(r[2]) if r[2] is not None else None,
        "t3_winrate": float(r[3]) if r[3] is not None else None,
        "t5_max_ret_mean": float(r[4]) if r[4] is not None else None,
    }


def _by_dimension_scores(
    db: Any, *, where_sql: str, params: list[Any]
) -> list[dict[str, Any]]:
    """Compute Pearson correlation between each dim_<X> column and ret_t3.

    Pearson formula in pure SQL:
        corr(x, y) = (n*Σxy − Σx Σy) /
                     √((n*Σx² − (Σx)²)(n*Σy² − (Σy)²))

    DuckDB has a built-in `CORR(x, y)` aggregate which is even simpler — use it
    when available. (DuckDB ≥ 0.7.0 supports CORR.)
    """
    rows: list[dict[str, Any]] = []
    for dim in DIMENSION_COLS:
        col = f"dim_{dim}"
        sql = f"""
            SELECT COUNT(*) AS n_samples,
                   AVG(CASE WHEN s.{col} > 50 AND r.ret_t3 > 0 THEN 100.0 ELSE 0.0 END)
                       AS hit_rate_above_50,
                   CORR(s.{col}, r.ret_t3) AS corr_t3,
                   AVG(r.ret_t3) AS t3_mean,
                   AVG(r.max_ret_5d) AS t5_max_ret_mean
            FROM va_stage_results s
            JOIN va_realized_returns r
              ON s.ts_code = r.ts_code AND {where_sql}
            WHERE r.ret_t3 IS NOT NULL AND s.{col} IS NOT NULL
        """
        row = db.fetchone(sql, tuple(params))
        if not row:
            continue
        n, _hit, corr, t3_mean, t5_max = row
        rows.append(
            {
                "bucket": dim,
                "n_samples": int(n or 0),
                # Repurpose t3_mean column to show the correlation coefficient
                # so the existing renderer doesn't need a special-case branch.
                "t3_mean": float(corr) if corr is not None else None,
                "t3_winrate": None,
                "t5_max_ret_mean": float(t5_max) if t5_max is not None else None,
            }
        )
    return rows
