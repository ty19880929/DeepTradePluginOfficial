"""推理结果落 ``lub_lgb_predictions``。

设计文档 §3.5：每次 run 的 LGB 评分一行 = 一只候选股 × 一次 run，含特征摘要
hash + 缺失字段 JSON。复盘工具与离线评估（PR-3.x）都基于此表查询。

API 单一入口 :func:`record_predictions` — 由 ``runner._execute_single`` /
``runner._execute_debate`` 在 R1 collect 之后调用。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)


def record_predictions(
    db: Database,
    *,
    run_id: str,
    trade_date: str,
    model_id: str,
    rows: Iterable[dict[str, Any]],
) -> int:
    """Batch-INSERT into ``lub_lgb_predictions``.

    ``rows`` 元素期望含以下键（缺则跳过该行；不抛异常）:

    * ``ts_code``               (str)
    * ``lgb_score``             (float; 已 ``× 100`` 归一到 0–100 也可，
                                 但本表保存原始 booster.predict 输出 ∈ [0,1])
    * ``lgb_decile``            (int | None)
    * ``feature_hash``          (str; 空字符串视为占位，仍写入)
    * ``feature_missing_json``  (str; 默认 "[]")

    Returns
    -------
    int
        实际写入的行数。
    """
    inserted = 0
    payloads: list[tuple[Any, ...]] = []
    for r in rows:
        ts_code = r.get("ts_code")
        score = r.get("lgb_score")
        if ts_code is None or score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        decile = r.get("lgb_decile")
        decile_i: int | None
        if decile is None:
            decile_i = None
        else:
            try:
                decile_i = int(decile)
            except (TypeError, ValueError):
                decile_i = None
        payloads.append(
            (
                run_id,
                trade_date,
                str(ts_code),
                model_id,
                score_f,
                decile_i,
                str(r.get("feature_hash") or ""),
                str(r.get("feature_missing_json") or "[]"),
            )
        )

    if not payloads:
        return 0

    try:
        with db.transaction():
            for payload in payloads:
                db.execute(
                    "INSERT INTO lub_lgb_predictions("
                    "run_id, trade_date, ts_code, model_id, lgb_score, "
                    "lgb_decile, feature_hash, feature_missing_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
                inserted += 1
    except Exception as e:  # noqa: BLE001 — 审计失败不阻塞主流程
        logger.warning(
            "record_predictions failed after %d/%d rows: %s",
            inserted,
            len(payloads),
            e,
        )
        return inserted

    return inserted
