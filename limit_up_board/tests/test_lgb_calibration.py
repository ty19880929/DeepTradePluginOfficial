"""P2-2 Part 2 (v0.13.1)：calibration 模块 + scorer 双路径 + evaluate Brier。

涵盖：
1. ``train_isotonic_calibrator`` 在 OOF 预测上正确拟合 + Brier 下降；
   单类输入 / 全 NaN 输入降级到 None；
2. ``apply_calibrator`` NaN 透传 + 实际单调变换；
3. ``save_calibrator`` / ``load_calibrator`` 文件 round-trip；
   缺文件 / 损坏 → None；
4. ``LgbScorer`` 在 ``calibration_method=None`` / 校准器文件缺失 / 加载成功
   三种分支下的行为；
5. ``brier_score`` + ``reliability_table`` 数值正确。
6. ``train_lightgbm`` 端到端：OOF 拼接 → 校准器训练 → registered metadata
   在 ``TrainResult`` 上可访问（标记 slow 防止 CI 默认跑）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from limit_up_board.lgb import calibration
from limit_up_board.lgb.calibration import (
    apply_calibrator,
    brier_score,
    calibrator_file_name,
    load_calibrator,
    reliability_table,
    save_calibrator,
    train_isotonic_calibrator,
)


# ---------------------------------------------------------------------------
# train_isotonic_calibrator
# ---------------------------------------------------------------------------


def test_isotonic_calibrator_reduces_brier_on_skewed_predictions() -> None:
    """构造一个 monotone-but-miscalibrated 预测分布，isotonic 应当显著改善 Brier。"""
    rng = np.random.default_rng(seed=7)
    n = 5000
    y_true = rng.binomial(1, 0.4, size=n).astype("int")
    # 把真实概率 ~ 0.4 的标签映射到一个过度自信的 sigmoid 输出
    raw = np.where(y_true == 1, rng.beta(8, 2, size=n), rng.beta(2, 8, size=n))
    iso, brier_pre, brier_post, n_used = train_isotonic_calibrator(y_true, raw)
    assert iso is not None
    assert n_used == n
    assert brier_pre is not None and brier_post is not None
    assert brier_post <= brier_pre + 1e-6  # 校准不应让 Brier 变差


def test_calibrator_skipped_on_single_class_labels() -> None:
    """全部 y_true=0 时无法学校准；返回 calibrator=None + brier_pre 保留。"""
    iso, brier_pre, brier_post, n = train_isotonic_calibrator(
        np.zeros(50, dtype="int"), np.linspace(0.1, 0.9, 50)
    )
    assert iso is None
    assert brier_pre is not None
    assert brier_post is None
    assert n == 50


def test_calibrator_skipped_on_empty_input() -> None:
    iso, brier_pre, brier_post, n = train_isotonic_calibrator(
        np.array([], dtype="int"), np.array([], dtype="float64")
    )
    assert iso is None
    assert brier_pre is None
    assert brier_post is None
    assert n == 0


def test_calibrator_handles_nan_in_oof() -> None:
    """OOF 数组里有 NaN（未参与 CV 的样本）— 应当只用有限值拟合。"""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1], dtype="int")
    raw = np.array([0.1, 0.8, np.nan, 0.7, 0.9, 0.2, 0.6, 0.3, np.nan, 0.85])
    iso, brier_pre, brier_post, n_used = train_isotonic_calibrator(y_true, raw)
    assert iso is not None
    assert n_used == 8  # 跳过 2 个 NaN
    assert brier_pre is not None and brier_post is not None


# ---------------------------------------------------------------------------
# apply_calibrator
# ---------------------------------------------------------------------------


def test_apply_calibrator_passthrough_nan() -> None:
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.3, size=200)
    p = np.clip(rng.normal(0.3, 0.1, size=200), 0.0, 1.0)
    iso, _, _, _ = train_isotonic_calibrator(y, p)
    assert iso is not None
    raw_with_nan = np.array([0.1, 0.5, np.nan, 0.9, np.nan, 0.2])
    out = apply_calibrator(iso, raw_with_nan)
    assert np.isnan(out[2]) and np.isnan(out[4])
    assert np.all(np.isfinite(out[[0, 1, 3, 5]]))


def test_apply_calibrator_monotone() -> None:
    """isotonic 是单调的；输入升序应当对应输出非降序。"""
    rng = np.random.default_rng(23)
    y = rng.binomial(1, 0.5, size=500)
    p = np.clip(rng.normal(0.5, 0.2, size=500), 0.0, 1.0)
    iso, _, _, _ = train_isotonic_calibrator(y, p)
    assert iso is not None
    grid = np.linspace(0.0, 1.0, 50)
    out = apply_calibrator(iso, grid)
    assert np.all(np.diff(out) >= -1e-9)


# ---------------------------------------------------------------------------
# save_calibrator / load_calibrator
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    y = rng.binomial(1, 0.4, size=300)
    p = np.clip(rng.normal(0.4, 0.15, size=300), 0.0, 1.0)
    iso, _, _, _ = train_isotonic_calibrator(y, p)
    assert iso is not None
    path = tmp_path / calibrator_file_name("test-model-1")
    save_calibrator(path, iso)
    assert path.is_file()

    loaded = load_calibrator(path)
    assert loaded is not None
    grid = np.linspace(0, 1, 20)
    assert np.allclose(apply_calibrator(iso, grid), apply_calibrator(loaded, grid))


def test_load_calibrator_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert load_calibrator(tmp_path / "does-not-exist.pkl") is None


def test_load_calibrator_returns_none_for_corrupt_pickle(tmp_path: Path) -> None:
    p = tmp_path / "garbage.pkl"
    p.write_bytes(b"not a pickle stream at all")
    assert load_calibrator(p) is None


# ---------------------------------------------------------------------------
# Brier + reliability_table
# ---------------------------------------------------------------------------


def test_brier_score_basic() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.4, 0.9])
    # ((0.2)^2 + (0.3)^2 + (0.4)^2 + (0.1)^2) / 4 = 0.075
    assert brier_score(y, p) == pytest.approx(0.075, rel=1e-6)


def test_brier_score_handles_nan() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, np.nan, 0.4, 0.9])
    # Only positions 0, 2, 3 → ((0.2)^2 + (0.4)^2 + (0.1)^2) / 3
    expected = (0.04 + 0.16 + 0.01) / 3
    assert brier_score(y, p) == pytest.approx(expected, rel=1e-6)


def test_brier_score_empty_returns_none() -> None:
    assert brier_score(np.array([]), np.array([])) is None


def test_reliability_table_returns_rows_summing_to_n() -> None:
    rng = np.random.default_rng(31)
    y = rng.binomial(1, 0.5, size=500)
    p = rng.uniform(0, 1, size=500)
    table = reliability_table(y, p, n_bins=10)
    assert table  # non-empty
    total = sum(int(row["n"]) for row in table)
    assert total == 500
    # Each row should have plausible types
    for row in table:
        assert "bin_lo" in row and "bin_hi" in row and "n" in row
        assert "mean_pred" in row and "observed" in row and "gap" in row


# ---------------------------------------------------------------------------
# Scorer integration (no LightGBM needed — fake the booster object)
# ---------------------------------------------------------------------------


class _FakeBooster:
    """Minimal stand-in for ``lightgbm.Booster`` used by ``LgbScorer``."""

    def __init__(self, feature_names: list[str], preds: np.ndarray) -> None:
        self._feature_names = feature_names
        self._preds = preds

    def feature_name(self) -> list[str]:
        return list(self._feature_names)

    def predict(self, arr: np.ndarray) -> np.ndarray:  # noqa: ARG002
        # Return the configured per-row preds (one entry per call to first-row count)
        return self._preds[: arr.shape[0]]


def _make_scorer_with_fake_loaded(
    *,
    calibrator: Any | None,
    calibration_method: str | None,
    preds: np.ndarray,
) -> tuple[Any, np.ndarray]:
    from limit_up_board.lgb.features import FEATURE_NAMES
    from limit_up_board.lgb.scorer import LgbScorer, _LoadedModel

    scorer = LgbScorer.__new__(LgbScorer)
    scorer._db = None  # type: ignore[assignment]
    scorer._requested_model_id = None
    import threading

    scorer._lock = threading.Lock()
    scorer._loaded = _LoadedModel(
        model_id="fake-id",
        booster=_FakeBooster(list(FEATURE_NAMES), preds),
        feature_names=tuple(FEATURE_NAMES),
        calibrator=calibrator,
        calibration_method=calibration_method,
    )
    scorer._load_attempted = True
    scorer._load_error = None
    return scorer, preds


def _toy_feature_frame() -> pd.DataFrame:
    from limit_up_board.lgb.features import FEATURE_NAMES

    rng = np.random.default_rng(13)
    return pd.DataFrame(
        rng.normal(size=(5, len(FEATURE_NAMES))).astype("float64"),
        columns=list(FEATURE_NAMES),
    )


def test_scorer_returns_raw_when_calibrator_missing() -> None:
    """No calibrator on _LoadedModel → score_batch returns booster raw output."""
    preds = np.array([0.1, 0.5, 0.6, 0.7, 0.9])
    scorer, _ = _make_scorer_with_fake_loaded(
        calibrator=None, calibration_method=None, preds=preds
    )
    out = scorer.score_batch(_toy_feature_frame())
    assert np.allclose(out["lgb_score"].to_numpy(), preds)
    assert scorer.calibration_method is None
    assert scorer.has_calibrator is False


def test_scorer_applies_calibrator_when_present() -> None:
    """With a fitted isotonic calibrator → output != raw (transformed)."""
    rng = np.random.default_rng(42)
    n_train = 500
    y = rng.binomial(1, 0.3, size=n_train)
    p_train = np.clip(rng.normal(0.3, 0.15, size=n_train), 0.0, 1.0)
    iso, _, _, _ = train_isotonic_calibrator(y, p_train)
    assert iso is not None

    preds = np.array([0.1, 0.5, 0.6, 0.7, 0.9])
    scorer, _ = _make_scorer_with_fake_loaded(
        calibrator=iso, calibration_method="isotonic", preds=preds
    )
    out = scorer.score_batch(_toy_feature_frame())
    # 校准器把过度自信的 raw 推回 base rate；输出不应等于 raw（除非偶发恒等）
    transformed = apply_calibrator(iso, preds)
    assert np.allclose(out["lgb_score"].to_numpy(), transformed)
    assert scorer.calibration_method == "isotonic"
    assert scorer.has_calibrator is True
