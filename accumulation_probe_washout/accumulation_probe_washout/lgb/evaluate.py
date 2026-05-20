"""Offline model evaluation + per-feature PSI drift.

Two entrypoints (both consumed by ``lgb evaluate`` CLI):

* :func:`evaluate_model` — re-score ``apw_signal_history`` over [start, end]
  using a registered booster, join with ``apw_realized_returns`` labels,
  emit AUC + log-loss + Top-K hit-rate (per-day baseline) JSON.
* :func:`evaluate_drift` — Population Stability Index (PSI) between a
  baseline model's training matrix snapshot and a candidate model's
  training snapshot, sorted PSI desc with {stable | moderate | shift}
  labels per feature.

Both write a JSON dump under ``reports_dir()/`` so the result is
auditable / re-renderable.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, SCHEMA_VERSION
from .labels import fetch_labels_for_window
from .paths import datasets_dir, reports_dir
from . import registry as _registry

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    model_id: str
    start_date: str
    end_date: str
    n_samples: int
    n_positive: int
    auc: float | None
    logloss: float | None
    topk: int
    topk_hit_rate: float | None       # share of top-k per-day predictions that hit positive label
    per_day_topk: list[dict[str, Any]] = field(default_factory=list)


class LgbEvalError(RuntimeError):
    """Wrong model id / no data in window / etc."""


def evaluate_model(
    db: Database,
    *,
    start_date: str,
    end_date: str,
    model_id: str | None = None,
    k: int = 10,
) -> tuple[EvalResult, Path]:
    """Re-score the window with one booster + write a JSON report.

    When ``model_id`` is None, the active model is used. Raises
    :class:`LgbEvalError` when the model is missing or the window has no
    labeled samples.
    """
    record = (
        _registry.get_model(db, model_id)
        if model_id
        else _registry.get_active(db)
    )
    if record is None:
        raise LgbEvalError(
            f"model not found: {'<no active>' if model_id is None else model_id}"
        )
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as e:
        raise LgbEvalError("lightgbm not installed") from e

    booster_path = Path(record.file_path)
    if not booster_path.exists():
        raise LgbEvalError(f"booster file missing: {booster_path}")
    booster = lgb.Booster(model_file=str(booster_path))

    # Pull labels for the window from apw_realized_returns using the
    # *model*'s label_source (so we evaluate against the same definition the
    # booster was trained on).
    label_df = fetch_labels_for_window(
        db,
        start_date=start_date,
        end_date=end_date,
        source=record.label_source,
        threshold_pct=record.label_threshold_pct,
        drawdown_threshold_pct=record.label_threshold_pct,  # same default
    )
    if label_df.empty:
        raise LgbEvalError(
            f"no labeled rows in {start_date}..{end_date} for label_source="
            f"{record.label_source!r}"
        )

    # Collect candidate dicts for those (signal_date, ts_code) pairs.
    from .dataset import _candidates_for_signal_date  # noqa: PLC0415

    distinct_dates = sorted(label_df["signal_date"].unique())
    score_rows: list[dict[str, Any]] = []
    per_day: list[dict[str, Any]] = []
    from .features import build_feature_frame  # noqa: PLC0415

    for d in distinct_dates:
        cands = _candidates_for_signal_date(db, d)
        if not cands:
            continue
        feat = build_feature_frame(candidate_rows=cands)
        if feat.empty or list(feat.columns) != list(FEATURE_NAMES):
            continue
        try:
            probs = booster.predict(feat.values)
        except Exception:  # noqa: BLE001
            logger.exception("predict failed on %s — skipping day", d)
            continue
        day_df = pd.DataFrame(
            {
                "signal_date": d,
                "ts_code": list(feat.index),
                "prob": list(probs),
            }
        )
        day_df = day_df.merge(
            label_df[label_df["signal_date"] == d][["ts_code", "label"]],
            on="ts_code",
            how="inner",
        )
        if day_df.empty:
            continue
        score_rows.append(day_df)
        # Per-day Top-K hit rate.
        topk_df = day_df.sort_values("prob", ascending=False).head(k)
        per_day.append(
            {
                "signal_date": d,
                "n_labeled": int(len(day_df)),
                "topk": int(min(k, len(topk_df))),
                "topk_hit_rate": float(topk_df["label"].mean())
                if not topk_df.empty
                else None,
                "auc_day": _safe_auc(day_df["label"], day_df["prob"]),
            }
        )

    if not score_rows:
        raise LgbEvalError("no scoreable candidates in window")

    big = pd.concat(score_rows, ignore_index=True)
    auc = _safe_auc(big["label"], big["prob"])
    ll = _safe_logloss(big["label"], big["prob"])
    topk_per_day_rates = [r["topk_hit_rate"] for r in per_day if r["topk_hit_rate"] is not None]
    topk_mean = (
        round(float(np.mean(topk_per_day_rates)), 4)
        if topk_per_day_rates
        else None
    )

    result = EvalResult(
        model_id=record.model_id,
        start_date=start_date,
        end_date=end_date,
        n_samples=int(len(big)),
        n_positive=int((big["label"] == 1).sum()),
        auc=auc,
        logloss=ll,
        topk=k,
        topk_hit_rate=topk_mean,
        per_day_topk=per_day,
    )
    out_path = _dump_report(
        result, kind="evaluate", start=start_date, end=end_date
    )
    return result, out_path


# ---------------------------------------------------------------------------
# evaluate_drift — per-feature PSI
# ---------------------------------------------------------------------------


PSI_BIN_COUNT = 10
PSI_STABLE = 0.10
PSI_SHIFT = 0.25


@dataclass
class DriftEntry:
    feature: str
    psi: float
    status: str   # 'stable' | 'moderate' | 'shift'


@dataclass
class DriftResult:
    baseline_model_id: str
    candidate_model_id: str
    entries: list[DriftEntry] = field(default_factory=list)


def evaluate_drift(
    db: Database, *, baseline_model_id: str, candidate_model_id: str
) -> tuple[DriftResult, Path]:
    """PSI between two models' dataset.parquet snapshots."""
    base = _registry.get_model(db, baseline_model_id)
    cand = _registry.get_model(db, candidate_model_id)
    if base is None or cand is None:
        raise LgbEvalError("baseline or candidate model not found")
    base_df = _load_dataset_snapshot(base.model_id)
    cand_df = _load_dataset_snapshot(cand.model_id)
    if base_df is None or cand_df is None:
        raise LgbEvalError(
            "training-matrix snapshot missing for baseline or candidate"
        )
    entries: list[DriftEntry] = []
    for feat in FEATURE_NAMES:
        if feat not in base_df.columns or feat not in cand_df.columns:
            continue
        psi = _psi_two_samples(base_df[feat].dropna(), cand_df[feat].dropna())
        if psi is None:
            continue
        entries.append(DriftEntry(feature=feat, psi=psi, status=_psi_status(psi)))
    entries.sort(key=lambda e: e.psi, reverse=True)
    result = DriftResult(
        baseline_model_id=baseline_model_id,
        candidate_model_id=candidate_model_id,
        entries=entries,
    )
    out_path = _dump_report(
        result, kind="drift",
        start=base.train_start_date, end=cand.train_end_date,
    )
    return result, out_path


def _load_dataset_snapshot(model_id: str) -> pd.DataFrame | None:
    p = datasets_dir() / f"{model_id}.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None


def _psi_two_samples(baseline: pd.Series, candidate: pd.Series) -> float | None:
    """PSI from baseline → candidate using ``PSI_BIN_COUNT`` quantile bins.

    Returns None when either side is too small / degenerate. PSI = Σ
    (p_cand - p_base) * ln(p_cand / p_base) over the bins; small smoothing
    (``+ 1e-6``) prevents log-of-zero.
    """
    if len(baseline) < PSI_BIN_COUNT or len(candidate) < PSI_BIN_COUNT:
        return None
    quantiles = np.linspace(0, 1, PSI_BIN_COUNT + 1)
    try:
        bins = baseline.quantile(quantiles).unique()
    except Exception:  # noqa: BLE001
        return None
    if len(bins) < 3:
        return None
    bins[0] = -math.inf
    bins[-1] = math.inf
    base_counts, _ = np.histogram(baseline, bins=bins)
    cand_counts, _ = np.histogram(candidate, bins=bins)
    base_pct = (base_counts / max(1, base_counts.sum())) + 1e-6
    cand_pct = (cand_counts / max(1, cand_counts.sum())) + 1e-6
    psi = float(np.sum((cand_pct - base_pct) * np.log(cand_pct / base_pct)))
    return round(psi, 4)


def _psi_status(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SHIFT:
        return "moderate"
    return "shift"


# ---------------------------------------------------------------------------
# Metrics + report dump
# ---------------------------------------------------------------------------


def _safe_auc(y, p) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415

        return round(float(roc_auc_score(y, p)), 4)
    except Exception:  # noqa: BLE001 — single-class window etc.
        return None


def _safe_logloss(y, p) -> float | None:
    try:
        from sklearn.metrics import log_loss  # noqa: PLC0415

        return round(float(log_loss(y, p, labels=[0, 1])), 4)
    except Exception:  # noqa: BLE001
        return None


def _dump_report(payload: Any, *, kind: str, start: str, end: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = reports_dir() / f"lgb_{kind}_{start}_{end}_{ts}.json"
    serialisable = (
        asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
    )
    out.write_text(
        json.dumps(
            serialisable, ensure_ascii=False, indent=2, default=str
        ),
        encoding="utf-8",
    )
    return out
