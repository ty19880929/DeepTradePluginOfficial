"""Persistence layer for ``lub_prediction_records``.

PR #1 — 单 LLM run 收尾时把"展示给用户的最终预测口径"upsert 到本表，作为后续
胜率分析的样本来源。主键 ``(trade_date, ts_code)``，后写覆盖前写。

口径回退链（与 report 一致）：
    1) final_ranking 存在 → final_rank / final_prediction，
                            从 predictions 补 continuation_score / confidence / rationale
    2) final_ranking 缺失 → 直接用 predictions 的 rank / prediction
    3) 从 bundle.candidates 按 ts_code 补 close_yuan / lgb_score / lgb_decile

辩论模式 (``_execute_debate``) 不调用本模块，避免多 provider 聚合口径未定义前
污染胜率样本。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..data import Round1Bundle
    from ..runtime import LubRuntime
    from ..schemas import ContinuationCandidate, FinalRankingResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row dataclass — what callers (CLI summary/export/llm-review) get back
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    """One row of ``lub_prediction_records``, decoded from DB."""

    trade_date: str
    next_trade_date: str
    ts_code: str
    name: str
    run_id: str
    prediction: str
    rank: int
    continuation_score: float | None
    confidence: str | None
    t_close_price: float | None
    lgb_score: float | None
    lgb_decile: int | None
    raw_prediction_json: str | None


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def _index_candidates(bundle: Round1Bundle) -> dict[str, dict[str, Any]]:
    """Build ts_code → candidate-dict index from bundle.candidates."""
    out: dict[str, dict[str, Any]] = {}
    for cand in bundle.candidates:
        ts_code = cand.get("ts_code")
        if ts_code:
            out[str(ts_code)] = cand
    return out


def _index_predictions(
    predictions: list[ContinuationCandidate],
) -> dict[str, ContinuationCandidate]:
    """Build ts_code → ContinuationCandidate index for fast lookups."""
    return {p.ts_code: p for p in predictions}


def _build_rows(
    *,
    bundle: Round1Bundle,
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None,
    run_id: str,
    trade_date: str,
    next_trade_date: str,
) -> list[dict[str, Any]]:
    """Produce one row-dict per prediction record to be upserted.

    Chooses口径 (prediction/rank source) based on final_ranking presence and
    fills in candidate-side fields (t_close_price / lgb_score / lgb_decile)
    from bundle.candidates.
    """
    cand_idx = _index_candidates(bundle)
    pred_idx = _index_predictions(predictions)

    rows: list[dict[str, Any]] = []

    if final_ranking is not None and final_ranking.finalists:
        # final_ranking 口径：rank / prediction 来自 final；继续分 / 置信度 / rationale 仍来自 predictions
        for item in final_ranking.finalists:
            pred = pred_idx.get(item.ts_code)
            cand = cand_idx.get(item.ts_code, {})
            name = (pred.name if pred else None) or str(cand.get("name") or "")
            rows.append(
                {
                    "trade_date": trade_date,
                    "next_trade_date": next_trade_date,
                    "ts_code": item.ts_code,
                    "name": name,
                    "run_id": run_id,
                    "prediction": item.final_prediction,
                    "rank": int(item.final_rank),
                    "continuation_score": float(pred.continuation_score) if pred else None,
                    "confidence": item.final_confidence or (pred.confidence if pred else None),
                    "t_close_price": _coerce_float(cand.get("close_yuan")),
                    "lgb_score": _coerce_float(cand.get("lgb_score")),
                    "lgb_decile": _coerce_int(cand.get("lgb_decile")),
                    "raw_prediction_json": _serialize_raw(pred, item),
                }
            )
        return rows

    # No final_ranking → use predictions verbatim
    for pred in predictions:
        cand = cand_idx.get(pred.ts_code, {})
        rows.append(
            {
                "trade_date": trade_date,
                "next_trade_date": next_trade_date,
                "ts_code": pred.ts_code,
                "name": pred.name or str(cand.get("name") or ""),
                "run_id": run_id,
                "prediction": pred.prediction,
                "rank": int(pred.rank),
                "continuation_score": float(pred.continuation_score),
                "confidence": pred.confidence,
                "t_close_price": _coerce_float(cand.get("close_yuan")),
                "lgb_score": _coerce_float(cand.get("lgb_score")),
                "lgb_decile": _coerce_int(cand.get("lgb_decile")),
                "raw_prediction_json": _serialize_raw(pred, None),
            }
        )
    return rows


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # NaN guard — pandas may surface NaN via dict.get; we want SQL NULL.
    if f != f:  # noqa: PLR0124
        return None
    return f


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        # Float → int round-trip is fine for decile (1..10); bail on NaN.
        f = float(v)
        if f != f:  # noqa: PLR0124
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def _serialize_raw(
    pred: ContinuationCandidate | None,
    final_item: Any,
) -> str | None:
    """JSON-serialize the original prediction object for audit + LLM review.

    Length-capped at 8KB so a runaway LLM rationale doesn't bloat the table.
    """
    if pred is None:
        return None
    try:
        payload = pred.model_dump()
        if final_item is not None:
            payload["_final_ranking"] = {
                "final_rank": final_item.final_rank,
                "final_prediction": final_item.final_prediction,
                "final_confidence": final_item.final_confidence,
                "delta_vs_batch": final_item.delta_vs_batch,
                "reason_vs_peers": final_item.reason_vs_peers,
            }
        s = json.dumps(payload, ensure_ascii=False)
        if len(s) > 8192:
            s = s[:8192]
        return s
    except Exception:  # noqa: BLE001 — best-effort audit field
        logger.warning("raw_prediction_json serialize failed", exc_info=True)
        return None


_UPSERT_SQL = """
INSERT INTO lub_prediction_records (
    trade_date, next_trade_date, ts_code, name, run_id,
    prediction, rank, continuation_score, confidence,
    t_close_price, lgb_score, lgb_decile, raw_prediction_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (trade_date, ts_code) DO UPDATE SET
    next_trade_date     = EXCLUDED.next_trade_date,
    name                = EXCLUDED.name,
    run_id              = EXCLUDED.run_id,
    prediction          = EXCLUDED.prediction,
    rank                = EXCLUDED.rank,
    continuation_score  = EXCLUDED.continuation_score,
    confidence          = EXCLUDED.confidence,
    t_close_price       = EXCLUDED.t_close_price,
    lgb_score           = EXCLUDED.lgb_score,
    lgb_decile          = EXCLUDED.lgb_decile,
    raw_prediction_json = EXCLUDED.raw_prediction_json,
    updated_at          = NOW()
"""


def record_predictions_from_run(
    *,
    rt: LubRuntime,
    bundle: Round1Bundle,
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None,
    run_id: str,
    trade_date: str,
    next_trade_date: str,
) -> int:
    """Upsert T 日预测留痕。返回写入条数（0 表示无 predictions 跳过）。

    本函数被 ``LubRunner._iter_pipeline`` 在 Step 5 (write_report) 之后、
    upload_summary 之前调用。失败由 caller 用 try/except 包裹后降级成
    WARNING 事件——本函数自己不做 swallow，便于单测断言。
    """
    if not predictions:
        return 0

    rows = _build_rows(
        bundle=bundle,
        predictions=predictions,
        final_ranking=final_ranking,
        run_id=run_id,
        trade_date=trade_date,
        next_trade_date=next_trade_date,
    )

    with rt.db.transaction():
        for r in rows:
            rt.db.execute(
                _UPSERT_SQL,
                (
                    r["trade_date"],
                    r["next_trade_date"],
                    r["ts_code"],
                    r["name"],
                    r["run_id"],
                    r["prediction"],
                    r["rank"],
                    r["continuation_score"],
                    r["confidence"],
                    r["t_close_price"],
                    r["lgb_score"],
                    r["lgb_decile"],
                    r["raw_prediction_json"],
                ),
            )

    return len(rows)


# ---------------------------------------------------------------------------
# Read path (used by PR #2 summary / PR #3 export / PR #4 llm-review)
# ---------------------------------------------------------------------------


_LOAD_BASE_SQL = """
SELECT trade_date, next_trade_date, ts_code, name, run_id,
       prediction, rank, continuation_score, confidence,
       t_close_price, lgb_score, lgb_decile, raw_prediction_json
FROM lub_prediction_records
"""


def load_prediction_records(
    db: Database,
    *,
    start: str | None = None,
    end: str | None = None,
    predictions: list[str] | None = None,
) -> list[PredictionRecord]:
    """Read records, optionally filtered by trade_date range / prediction class.

    ``start`` / ``end`` are inclusive YYYYMMDD strings (compared as text — works
    because trade_date is always zero-padded). ``predictions`` filters the
    ``prediction`` column when non-empty.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append("trade_date >= ?")
        params.append(start)
    if end is not None:
        clauses.append("trade_date <= ?")
        params.append(end)
    if predictions:
        placeholders = ",".join("?" * len(predictions))
        clauses.append(f"prediction IN ({placeholders})")
        params.extend(predictions)

    sql = _LOAD_BASE_SQL
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY trade_date ASC, rank ASC"

    rows = db.fetchall(sql, tuple(params) if params else None)
    return [
        PredictionRecord(
            trade_date=r[0],
            next_trade_date=r[1],
            ts_code=r[2],
            name=r[3],
            run_id=r[4],
            prediction=r[5],
            rank=int(r[6]),
            continuation_score=float(r[7]) if r[7] is not None else None,
            confidence=r[8],
            t_close_price=float(r[9]) if r[9] is not None else None,
            lgb_score=float(r[10]) if r[10] is not None else None,
            lgb_decile=int(r[11]) if r[11] is not None else None,
            raw_prediction_json=r[12],
        )
        for r in rows
    ]


def purge_prediction_records(db: Database, *, before: str) -> int:
    """Delete rows where ``trade_date <= before``. Returns deleted count.

    ``before`` is an inclusive YYYYMMDD string.
    """
    pre = db.fetchone(
        "SELECT COUNT(*) FROM lub_prediction_records WHERE trade_date <= ?",
        (before,),
    )
    n = int(pre[0]) if pre else 0
    if n > 0:
        db.execute("DELETE FROM lub_prediction_records WHERE trade_date <= ?", (before,))
    return n
