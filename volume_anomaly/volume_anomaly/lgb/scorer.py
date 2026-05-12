"""LightGBM 推理评分 (``LgbScorer``).

设计文档 §7。本模块负责：

* **加载** ``is_active=TRUE`` 的模型 booster（lazy，遇错降级）；
* **批量评分** 一组候选股的特征矩阵 → ``lgb_score / feature_hash /
  feature_missing_json``；
* **错误降级**：缺模型、文件缺失、schema mismatch、推理异常、lightgbm 包未装——
  任何一种都不应让 LLM 走势分析跑不下去（设计 §7.3 红线）。

VA 与 ``limit-up-board`` 的同名模块同构；VA 不存在 debate-mode worker，
scorer 始终主线程使用，但仍保留 lock 以便日后扩展。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import paths as lgb_paths
from . import registry as lgb_registry
from .features import FEATURE_NAMES, LgbFeatureSchemaError, assert_columns

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)


class LgbScorerError(RuntimeError):
    """Base for LgbScorer-side failures."""


class LgbModelMissingError(LgbScorerError):
    """``va_lgb_models`` row exists but the on-disk file is gone."""


class LgbModelSchemaMismatch(LgbScorerError):
    """Booster ``feature_name()`` disagrees with current FEATURE_NAMES."""


@dataclass(frozen=True)
class _LoadedModel:
    model_id: str
    booster: Any
    feature_names: tuple[str, ...]


def _empty_scored_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lgb_score": [float("nan")] * len(index),
            "feature_hash": [""] * len(index),
            "feature_missing_json": ["[]"] * len(index),
        },
        index=index,
    )


class LgbScorer:
    """Inference scorer; cheap to construct, lazy load on first call.

    Parameters
    ----------
    db
        ``deeptrade.core.db.Database`` instance. Only read ``va_lgb_models``.
    model_id
        Explicit model_id override; ``None`` → load whichever row is active.
        No active row → ``loaded=False`` (legal degradation).
    """

    def __init__(self, db: Database, *, model_id: str | None = None) -> None:
        self._db = db
        self._requested_model_id = model_id
        self._lock = threading.Lock()
        self._loaded: _LoadedModel | None = None
        self._load_attempted: bool = False
        self._load_error: str | None = None

    @property
    def model_id(self) -> str | None:
        return self._loaded.model_id if self._loaded is not None else None

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def warmup(self) -> None:
        """Force lazy load + a single empty predict to confirm the booster
        is usable. Never raises — diagnostics land on :attr:`load_error`."""
        with self._lock:
            self._ensure_loaded_locked()
            if self._loaded is None:
                return
            try:
                self._loaded.booster.predict(
                    np.full(
                        (1, len(self._loaded.feature_names)),
                        np.nan,
                        dtype="float64",
                    )
                )
            except Exception as e:  # noqa: BLE001
                self._load_error = (
                    f"predict_warmup_failed: {type(e).__name__}: {e}"
                )
                self._loaded = None
                logger.warning("LgbScorer.warmup predict probe failed: %s", e)

    def score_batch(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Score a ``(n_candidates × n_features)`` matrix.

        Always returns a DataFrame with ``lgb_score / feature_hash /
        feature_missing_json``, indexed by the input. Any failure path
        returns NaN scores — never raises.
        """
        if feature_df.empty:
            return _empty_scored_frame(feature_df.index)

        try:
            aligned = feature_df.reindex(columns=list(FEATURE_NAMES))
            assert_columns(aligned)
        except LgbFeatureSchemaError as e:
            logger.warning("LgbScorer.score_batch schema mismatch: %s", e)
            return _empty_scored_frame(feature_df.index)
        except Exception as e:  # noqa: BLE001
            logger.warning("LgbScorer.score_batch reindex failure: %s", e)
            return _empty_scored_frame(feature_df.index)

        with self._lock:
            self._ensure_loaded_locked()
            loaded = self._loaded

        missing_payload = _compute_missing_payload(aligned)
        hashes = _compute_row_hashes(aligned)

        if loaded is None:
            return pd.DataFrame(
                {
                    "lgb_score": [float("nan")] * len(aligned),
                    "feature_hash": hashes,
                    "feature_missing_json": missing_payload,
                },
                index=feature_df.index,
            )

        try:
            raw = aligned.to_numpy(dtype="float64", na_value=np.nan)
            preds = loaded.booster.predict(raw)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "LgbScorer.score_batch predict failed (%s); NaN fallback.", e
            )
            return pd.DataFrame(
                {
                    "lgb_score": [float("nan")] * len(aligned),
                    "feature_hash": hashes,
                    "feature_missing_json": missing_payload,
                },
                index=feature_df.index,
            )

        preds_array = np.asarray(preds, dtype="float64").reshape(-1)
        if preds_array.shape[0] != len(aligned):
            logger.warning(
                "LgbScorer prediction shape mismatch (%d vs %d).",
                preds_array.shape[0],
                len(aligned),
            )
            return pd.DataFrame(
                {
                    "lgb_score": [float("nan")] * len(aligned),
                    "feature_hash": hashes,
                    "feature_missing_json": missing_payload,
                },
                index=feature_df.index,
            )

        # design §7: LightGBM binary predict ∈ [0, 1]; ×100 for 0-100 score
        # happens in the data-layer attach_lgb_scores helper (PR-2.2).
        return pd.DataFrame(
            {
                "lgb_score": preds_array,
                "feature_hash": hashes,
                "feature_missing_json": missing_payload,
            },
            index=feature_df.index,
        )

    def _ensure_loaded_locked(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            record = self._lookup_record()
        except Exception as e:  # noqa: BLE001
            self._load_error = (
                f"registry_lookup_failed: {type(e).__name__}: {e}"
            )
            logger.warning("LgbScorer registry lookup failed: %s", e)
            return

        if record is None:
            self._load_error = (
                "no_active_model"
                if self._requested_model_id is None
                else f"model_id_not_found: {self._requested_model_id}"
            )
            return

        model_file = lgb_paths.plugin_data_dir() / record.file_path
        if not model_file.is_file():
            self._load_error = f"model_file_missing: {model_file}"
            logger.warning(
                "LgbScorer: model %s row exists but file missing at %s",
                record.model_id,
                model_file,
            )
            return

        try:
            booster = _load_booster(str(model_file))
        except Exception as e:  # noqa: BLE001
            self._load_error = f"booster_load_failed: {type(e).__name__}: {e}"
            logger.warning("LgbScorer booster load failed: %s", e)
            return

        try:
            feat_names = tuple(booster.feature_name())
        except Exception as e:  # noqa: BLE001
            self._load_error = (
                f"booster_introspect_failed: {type(e).__name__}: {e}"
            )
            logger.warning("LgbScorer booster introspection failed: %s", e)
            return

        if list(feat_names) != list(FEATURE_NAMES):
            self._load_error = (
                f"schema_mismatch: model_features({len(feat_names)}) != "
                f"FEATURE_NAMES({len(FEATURE_NAMES)})"
            )
            logger.warning(
                "LgbScorer: schema mismatch — refusing to load %s.",
                record.model_id,
            )
            return

        self._loaded = _LoadedModel(
            model_id=record.model_id,
            booster=booster,
            feature_names=feat_names,
        )
        logger.info(
            "LgbScorer loaded model_id=%s (features=%d)",
            record.model_id,
            len(feat_names),
        )

    def _lookup_record(self) -> Any:
        if self._requested_model_id is not None:
            return lgb_registry.get_model(self._db, self._requested_model_id)
        return lgb_registry.get_active(self._db)


def _compute_missing_payload(aligned: pd.DataFrame) -> list[str]:
    out: list[str] = []
    if aligned.empty:
        return out
    isna = aligned.isna().to_numpy()
    cols = list(aligned.columns)
    for row_mask in isna:
        missing_cols = [cols[i] for i, m in enumerate(row_mask) if bool(m)]
        out.append(json.dumps(missing_cols, ensure_ascii=False))
    return out


def _compute_row_hashes(aligned: pd.DataFrame) -> list[str]:
    if aligned.empty:
        return []
    out: list[str] = []
    arr = aligned.to_numpy(dtype="float64", na_value=np.nan)
    for row in arr:
        h = hashlib.blake2b(digest_size=8)
        h.update(row.tobytes())
        out.append(h.hexdigest())
    return out


def _load_booster(model_file: str) -> Any:
    try:
        import lightgbm as lgb_mod  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "lightgbm 未安装：pip install 'lightgbm>=4.3' 才能加载 LGB 模型。"
        ) from e
    return lgb_mod.Booster(model_file=model_file)


def attach_deciles(scored: pd.DataFrame, *, n_buckets: int = 10) -> pd.Series:
    """1..``n_buckets`` decile bucket per finite lgb_score.

    Batches with < ``n_buckets`` finite scores get NaN deciles (设计 §3.1).
    """
    if "lgb_score" not in scored.columns:
        return pd.Series(
            [pd.NA] * len(scored), index=scored.index, dtype="Int64"
        )
    s = pd.to_numeric(scored["lgb_score"], errors="coerce")
    finite = s.dropna()
    if len(finite) < n_buckets:
        return pd.Series(
            [pd.NA] * len(scored), index=scored.index, dtype="Int64"
        )
    try:
        codes = pd.qcut(finite, q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(
            [pd.NA] * len(scored), index=scored.index, dtype="Int64"
        )
    out = pd.Series([pd.NA] * len(scored), index=scored.index, dtype="Int64")
    out.loc[finite.index] = codes.astype("Int64") + 1
    return out


__all__ = [
    "LgbModelMissingError",
    "LgbModelSchemaMismatch",
    "LgbScorer",
    "LgbScorerError",
    "attach_deciles",
]
