"""P2-2 (v0.13.1)：LightGBM 输出校准 — Isotonic / Platt + Brier 评估 + reliability table.

设计概要
--------

1. ``train_isotonic_calibrator(y_true, y_pred_oof)`` —— 在 CV out-of-fold 预测
   上拟合 :class:`sklearn.isotonic.IsotonicRegression`（``out_of_bounds="clip"``），
   返回校准器 + 校准前/后 Brier。OOF 预测来自 :func:`trainer._crossval_metrics`
   —— **不能**用 final fit 的 in-sample 预测，否则校准器会"看见"自己的训练集。

2. ``apply_calibrator(calibrator, raw)`` —— 把 booster 原始 sigmoid 输出
   ``∈ [0, 1]`` 转成校准后的"经验概率" ``∈ [0, 1]``。NaN 原样返回。

3. ``save_calibrator`` / ``load_calibrator`` —— pickle 文件落在
   :func:`lgb.paths.calibrator_file_name`（与 booster 同目录，文件名约定
   ``lgb_calibrator_<model_id>.pkl``）。加载失败（文件缺失 / pickle 损坏 /
   sklearn 缺包）返回 ``None``，调用方降级到原始排序分。

4. ``brier_score(y_true, y_pred)`` —— ``mean((y_pred - y_true)**2)``，NaN 过滤。

5. ``reliability_table(y_true, y_pred, n_bins=10)`` —— 等频分桶产出 ``[(bin_lo,
   bin_hi, n, mean_pred, observed)]``，画 calibration curve 用。

校准器不入 booster 文件，独立存储；这样 ``lgb scorer`` 在文件缺失时可以
不报错降级（设计 §7.3 红线）。
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_CALIBRATION_METHODS = ("isotonic", "platt")


# ---------------------------------------------------------------------------
# Calibrator training
# ---------------------------------------------------------------------------


def train_isotonic_calibrator(
    y_true: np.ndarray, y_pred_oof: np.ndarray
) -> tuple[Any | None, float | None, float | None, int]:
    """Fit isotonic regression on OOF preds.

    Parameters
    ----------
    y_true : array-like int / float (0 / 1)
    y_pred_oof : array-like float (raw booster preds)

    Returns
    -------
    (calibrator, brier_pre, brier_post, n_samples)
        ``calibrator`` is None when training is skipped (无样本 / 全单类 /
        sklearn 缺包)；上层据此把 ``calibration_method`` 写 None / ``"none"``。
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
    except ImportError:  # pragma: no cover — sklearn 是 deps
        logger.warning("sklearn missing — skip isotonic calibration")
        return None, None, None, 0

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred_oof, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return None, None, None, 0
    if len(np.unique(yt)) < 2:
        # 单一类别时 isotonic 退化；Brier 仍可算但校准器无意义。
        logger.warning(
            "calibrator skipped — only one class in OOF labels (n=%d)", len(yt)
        )
        return None, _safe_brier(yt, yp), None, int(len(yt))

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(yp, yt)
    yp_calibrated = iso.predict(yp)
    brier_pre = _safe_brier(yt, yp)
    brier_post = _safe_brier(yt, yp_calibrated)
    return iso, brier_pre, brier_post, int(len(yt))


def apply_calibrator(calibrator: Any, raw: np.ndarray) -> np.ndarray:
    """Apply *calibrator* to *raw* booster preds; NaN passthrough.

    NaN positions are preserved (caller already uses NaN for "未评分");
    isotonic's ``predict`` would otherwise raise on NaN input.
    """
    arr = np.asarray(raw, dtype="float64")
    if arr.size == 0:
        return arr
    out = np.full_like(arr, np.nan, dtype="float64")
    mask = np.isfinite(arr)
    if mask.any():
        try:
            out[mask] = calibrator.predict(arr[mask])
        except Exception as e:  # noqa: BLE001 — 推理失败完全降级
            logger.warning("calibrator.predict failed (%s); returning raw", e)
            out = arr
    return out


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def calibrator_file_name(model_id: str) -> str:
    """Naming convention: ``lgb_calibrator_<model_id>.pkl``."""
    return f"lgb_calibrator_{model_id}.pkl"


def save_calibrator(path: Path, calibrator: Any) -> None:
    """Pickle the calibrator to *path*. Parent dir must exist."""
    with path.open("wb") as fh:
        pickle.dump(calibrator, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_calibrator(path: Path) -> Any | None:
    """Best-effort load; missing file / corrupt pickle / missing sklearn → None.

    Callers (``LgbScorer``) treat None as "no calibrator → return raw scores",
    matching the 5/6-branch降级 contract.
    """
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as e:  # noqa: BLE001 — broken pickle / sklearn missing
        logger.warning("load_calibrator(%s) failed: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return None
    return float(np.mean((yp[mask] - yt[mask]) ** 2))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Public alias for :func:`_safe_brier`."""
    return _safe_brier(y_true, y_pred)


def reliability_table(
    y_true: np.ndarray, y_pred: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Equal-frequency reliability table.

    Returns a list of ``{bin_lo, bin_hi, n, mean_pred, observed, gap}`` —
    one row per bucket. Buckets with 0 samples are emitted with ``mean_pred``
    / ``observed`` = NaN so downstream renderers can still see the bin range.
    Empty input → ``[]``.
    """
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return []
    # Quantile cut: if all preds identical, fall back to evenly-spaced edges.
    edges = np.unique(np.quantile(yp, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Make leftmost / rightmost inclusive.
    edges[0] = min(edges[0], -np.inf)
    edges[-1] = max(edges[-1], np.inf)

    out: list[dict[str, float | int]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            bucket_mask = (yp >= lo) & (yp <= hi)
        else:
            bucket_mask = (yp >= lo) & (yp < hi)
        n = int(bucket_mask.sum())
        if n == 0:
            out.append({
                "bin_lo": float(lo) if np.isfinite(lo) else -1.0,
                "bin_hi": float(hi) if np.isfinite(hi) else 1.0,
                "n": 0,
                "mean_pred": float("nan"),
                "observed": float("nan"),
                "gap": float("nan"),
            })
            continue
        mean_pred = float(yp[bucket_mask].mean())
        observed = float(yt[bucket_mask].mean())
        out.append({
            "bin_lo": float(lo) if np.isfinite(lo) else -1.0,
            "bin_hi": float(hi) if np.isfinite(hi) else 1.0,
            "n": n,
            "mean_pred": round(mean_pred, 4),
            "observed": round(observed, 4),
            "gap": round(observed - mean_pred, 4),
        })
    return out
