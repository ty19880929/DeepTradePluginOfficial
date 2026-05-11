"""离线评估 — 复用 :func:`dataset.collect_training_window` 拉同窗口的
特征 + 真实 T+1 outcome，按指定模型评分后计算 AUC / logloss / Top-K 命中率。

设计文档 §9.3：
* AUC / logloss 走 sklearn metrics（与训练阶段同口径）。
* Top-K 命中率 = 当日 candidate 中 LGB 分前 K 的 ``label=1`` 比例。
* Baseline = 当日所有 candidate 的 ``label=1`` 比例（等价于"随机 K 抽取"的期望命中率，
  避免引入伪随机种子带来的不可复现）。
* 实际 T+1 max upside（``pct_chg_t1``）按同一分组聚合，做"top-K 平均涨幅 vs 全体
  平均涨幅"对比，给出 evidence 是 LGB 是否选到了"真的能涨"的标的。

PR-3.1 范围
-----------
* :class:`EvaluateResult`
* :func:`evaluate_model`
* :func:`format_evaluate_table` — 给 CLI 渲染 stdout 表格用
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .dataset import collect_training_window
from .features import FEATURE_NAMES
from .labels import DEFAULT_LABEL_THRESHOLD_PCT
from .scorer import LgbScorer

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database
    from deeptrade.core.tushare_client import TushareClient

    from ..calendar import TradeCalendar

logger = logging.getLogger(__name__)


DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TopKMetrics:
    """Per-K aggregate snapshot."""

    k: int
    n_days_evaluated: int          # how many days had ≥ K candidates with score
    hit_count: int                  # sum of label=1 across all top-K picks
    pick_count: int                 # total picks (Σ min(K, n_day))
    hit_rate_pct: float | None      # hit_count / pick_count * 100
    avg_upside_pct: float | None    # mean of pct_chg_t1 over picks
    baseline_hit_rate_pct: float | None  # day-wise mean label rate (∀ candidates)
    baseline_avg_upside_pct: float | None  # day-wise mean pct_chg_t1
    delta_hit_rate_pct: float | None
    delta_avg_upside_pct: float | None


@dataclass
class EvaluateResult:
    """评估完整产出。可直接 :func:`json.dumps(asdict(result))` 落盘。"""

    model_id: str
    window_start: str
    window_end: str
    label_threshold_pct: float
    n_samples: int                 # rows in feature matrix
    n_labeled: int                 # rows with label∈{0,1}
    n_positive: int
    n_trade_dates: int             # unique trade_dates in the evaluated window
    auc: float | None
    logloss: float | None
    topk: list[TopKMetrics] = field(default_factory=list)
    feature_count: int = len(FEATURE_NAMES)
    schema_version_match: bool = True
    schema_mismatch_detail: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Friendly dict for `json.dumps`."""
        data = asdict(self)
        return data


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def evaluate_model(
    *,
    tushare: TushareClient,
    calendar: TradeCalendar,
    db: Database,
    start_date: str,
    end_date: str,
    model_id: str | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    label_threshold_pct: float = DEFAULT_LABEL_THRESHOLD_PCT,
    max_float_mv_yi: float = 100.0,
    max_close_yuan: float = 15.0,
    force_sync: bool = False,
    on_day: Callable[[str, int, int], None] | None = None,
) -> EvaluateResult:
    """Run离线评估 over ``[start_date, end_date]``.

    Steps
    -----
    1. :func:`collect_training_window` to get feature matrix + labels +
       sample_index (which carries ``pct_chg_t1``).
    2. Load model via :class:`LgbScorer` (``model_id=None`` → use active).
       Schema mismatch → result.schema_version_match=False; metrics all None.
    3. Score; compute AUC / logloss on labeled subset.
    4. For each K in ``k_values``: group by trade_date, sort by score desc,
       take min(K, n_day), aggregate hits + upsides; compare to per-day
       baseline (mean over all candidates).
    """
    scorer = LgbScorer(db, model_id=model_id)
    scorer.warmup()
    if not scorer.loaded:
        # We can still gather dataset stats but no scoring.
        logger.warning(
            "evaluate: model not loaded (%s) — returning empty metrics",
            scorer.load_error,
        )

    ds = collect_training_window(
        tushare=tushare,
        calendar=calendar,
        start_date=start_date,
        end_date=end_date,
        max_float_mv_yi=max_float_mv_yi,
        max_close_yuan=max_close_yuan,
        label_threshold_pct=label_threshold_pct,
        force_sync=force_sync,
        on_day=on_day,
    )

    result_model_id = scorer.model_id or (model_id or "")
    result = EvaluateResult(
        model_id=result_model_id,
        window_start=start_date,
        window_end=end_date,
        label_threshold_pct=label_threshold_pct,
        n_samples=ds.n_samples,
        n_labeled=ds.n_labeled,
        n_positive=ds.n_positive,
        n_trade_dates=int(ds.sample_index["trade_date"].nunique()) if not ds.sample_index.empty else 0,
        auc=None,
        logloss=None,
        topk=[],
    )

    if ds.n_samples == 0:
        result.notes.append("collect_training_window returned 0 samples")
        return result

    if not scorer.loaded:
        result.schema_version_match = False
        result.schema_mismatch_detail = scorer.load_error
        result.notes.append("scorer not loaded — metrics omitted")
        return result

    # ---- score every sample -------------------------------------------
    scored = scorer.score_batch(ds.feature_matrix.copy())
    # Align scored frame back to the sample_index
    score_series = pd.Series(scored["lgb_score"].to_numpy(), name="lgb_score")

    eval_df = pd.concat(
        [
            ds.sample_index.reset_index(drop=True),
            score_series.reset_index(drop=True),
            ds.labels.reset_index(drop=True).rename("label"),
        ],
        axis=1,
    )

    # ---- AUC / logloss on the labeled subset --------------------------
    labeled = eval_df.dropna(subset=["label", "lgb_score"]).copy()
    if not labeled.empty:
        result.auc = _safe_auc(
            labeled["label"].astype(int).to_numpy(),
            labeled["lgb_score"].astype(float).to_numpy(),
        )
        result.logloss = _safe_logloss(
            labeled["label"].astype(int).to_numpy(),
            labeled["lgb_score"].astype(float).to_numpy(),
        )

    # ---- Top-K aggregates --------------------------------------------
    result.topk = _compute_topk_metrics(eval_df, k_values=k_values)

    return result


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import log_loss  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None
    if len(y_true) == 0:
        return None
    eps = 1e-7
    clipped = np.clip(y_score.astype(float), eps, 1.0 - eps)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def _compute_topk_metrics(
    eval_df: pd.DataFrame, *, k_values: tuple[int, ...]
) -> list[TopKMetrics]:
    """Per-K aggregation. For each day, sort by lgb_score desc, take the first
    min(K, n_day) candidates; sum hits and upsides; compare to the per-day
    baseline (mean across all candidates that day)."""
    out: list[TopKMetrics] = []
    if eval_df.empty:
        for k in k_values:
            out.append(
                TopKMetrics(
                    k=k,
                    n_days_evaluated=0,
                    hit_count=0,
                    pick_count=0,
                    hit_rate_pct=None,
                    avg_upside_pct=None,
                    baseline_hit_rate_pct=None,
                    baseline_avg_upside_pct=None,
                    delta_hit_rate_pct=None,
                    delta_avg_upside_pct=None,
                )
            )
        return out

    # Per-day score-sorted scaffolding — sort once per group.
    grouped = list(eval_df.groupby("trade_date", sort=False))

    for k in k_values:
        hit_count = 0
        pick_count = 0
        upside_sum = 0.0
        upside_pick_count = 0  # for upside avg (some rows have null upside)
        baseline_hit_rates: list[float] = []
        baseline_upsides: list[float] = []
        n_days_evaluated = 0
        for _td, group in grouped:
            # Drop NaN scores so they don't end up in top-K via stable sort.
            scored = group.dropna(subset=["lgb_score"]).copy()
            if scored.empty:
                continue
            n_days_evaluated += 1

            top = scored.sort_values("lgb_score", ascending=False).head(k)
            # Hits only counted on labeled rows
            top_labeled = top.dropna(subset=["label"])
            hit_count += int(top_labeled["label"].astype(int).sum())
            pick_count += int(len(top_labeled))

            upsides = pd.to_numeric(top["pct_chg_t1"], errors="coerce").dropna()
            if not upsides.empty:
                upside_sum += float(upsides.sum())
                upside_pick_count += int(len(upsides))

            # Day-wise baseline rate over labeled rows
            group_labeled = scored.dropna(subset=["label"])
            if not group_labeled.empty:
                baseline_hit_rates.append(
                    float(group_labeled["label"].astype(int).mean())
                )
            group_upsides = pd.to_numeric(
                scored["pct_chg_t1"], errors="coerce"
            ).dropna()
            if not group_upsides.empty:
                baseline_upsides.append(float(group_upsides.mean()))

        hit_rate_pct = (
            (hit_count / pick_count * 100.0) if pick_count > 0 else None
        )
        avg_upside_pct = (
            (upside_sum / upside_pick_count) if upside_pick_count > 0 else None
        )
        baseline_hit_rate_pct = (
            (float(np.mean(baseline_hit_rates)) * 100.0)
            if baseline_hit_rates
            else None
        )
        baseline_avg_upside_pct = (
            float(np.mean(baseline_upsides)) if baseline_upsides else None
        )

        out.append(
            TopKMetrics(
                k=k,
                n_days_evaluated=n_days_evaluated,
                hit_count=hit_count,
                pick_count=pick_count,
                hit_rate_pct=_round_or_none(hit_rate_pct),
                avg_upside_pct=_round_or_none(avg_upside_pct),
                baseline_hit_rate_pct=_round_or_none(baseline_hit_rate_pct),
                baseline_avg_upside_pct=_round_or_none(baseline_avg_upside_pct),
                delta_hit_rate_pct=_safe_delta(
                    hit_rate_pct, baseline_hit_rate_pct
                ),
                delta_avg_upside_pct=_safe_delta(
                    avg_upside_pct, baseline_avg_upside_pct
                ),
            )
        )
    return out


def _round_or_none(v: float | None, ndigits: int = 2) -> float | None:
    return None if v is None else round(float(v), ndigits)


def _safe_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 2)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def format_evaluate_table(result: EvaluateResult) -> str:
    """Plain-text summary fit for ``typer.echo`` / log capture."""
    lines: list[str] = []
    lines.append(
        f"LGB evaluate · model={result.model_id or '<none>'} · "
        f"{result.window_start}..{result.window_end}"
    )
    lines.append(
        f"samples={result.n_samples}  labeled={result.n_labeled}  "
        f"positive={result.n_positive}  trade_dates={result.n_trade_dates}  "
        f"threshold={result.label_threshold_pct}%"
    )
    if not result.schema_version_match:
        lines.append(
            f"⚠ scorer not loaded: {result.schema_mismatch_detail or 'unknown'}"
        )
        return "\n".join(lines)

    auc_str = f"{result.auc:.4f}" if result.auc is not None else "—"
    logloss_str = f"{result.logloss:.4f}" if result.logloss is not None else "—"
    lines.append(f"AUC = {auc_str}   logloss = {logloss_str}")

    if not result.topk:
        return "\n".join(lines)

    lines.append("")
    lines.append("Top-K vs baseline (hit_rate% · avg_upside%):")
    lines.append(
        " K  | hit%   | up%   | base_hit% | base_up% | Δhit%   | Δup%"
    )
    lines.append("----+--------+-------+-----------+----------+---------+--------")
    for tk in result.topk:
        lines.append(
            f" {tk.k:2d} | "
            f"{_fmt(tk.hit_rate_pct, '5.1f'):>6} | "
            f"{_fmt(tk.avg_upside_pct, '4.2f'):>5} | "
            f"{_fmt(tk.baseline_hit_rate_pct, '5.1f'):>9} | "
            f"{_fmt(tk.baseline_avg_upside_pct, '4.2f'):>8} | "
            f"{_fmt(tk.delta_hit_rate_pct, '+5.1f'):>7} | "
            f"{_fmt(tk.delta_avg_upside_pct, '+5.2f'):>6}"
        )
    if result.notes:
        lines.append("")
        for n in result.notes:
            lines.append(f"note: {n}")
    return "\n".join(lines)


def _fmt(v: float | None, spec: str) -> str:
    if v is None:
        return "—"
    return format(v, spec)
