"""LightGBM 推理评分（``LgbScorer``）。

设计文档 §7。本模块负责：

* **加载** ``is_active=TRUE`` 的模型 booster（lazy，遇错降级）；
* **批量评分** 一组候选股的特征矩阵 → ``lgb_score / feature_hash /
  feature_missing_json``；
* **错误降级**：缺模型、文件缺失、schema mismatch、推理异常、lightgbm 包未装——
  任何一种都不应让 强势初筛 / 连板预测 跑不下去（设计 §7.3 红线）。

关键实现细节
------------

1. **lazy + thread-safe load**：构造时只读 ``lub_lgb_models`` 元信息，不立即
   touch booster 文件。首次 :meth:`score_batch` / :meth:`warmup` 时再走
   实际加载，加锁防止 debate-mode 多 worker 并发加载。LightGBM ``Booster``
   实例本身被设计为线程安全只读，加载完成后多 worker 可共享只读引用调用
   ``predict``——这是 :func:`open_worker_runtime` 共享 scorer 的前提。
2. **特征列对齐**：每次评分前都 ``df.reindex(columns=FEATURE_NAMES)``，再
   ``assert_columns``。模型 booster 的 ``feature_name()`` 与 :data:`FEATURE_NAMES`
   不一致时直接拒绝（设计 §3.4：schema_version 是单一真相，旧 schema 须重训）。
3. **NaN 原生处理**：保持训练管线的 NaN 语义；不在此处 ``fillna``。
4. **feature_hash**：每行特征向量经 BLAKE2b-8B 摘要后 hex 化——审计表使用，
   便于复盘"为什么这只股得了 0.73 分"。
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
from .features import FEATURE_NAMES, FeatureSchemaMismatch, assert_columns

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors — kept as subclasses of RuntimeError so callers can catch generically.
# ---------------------------------------------------------------------------


class LgbScorerError(RuntimeError):
    """Base for all LgbScorer-side failures."""


class LgbModelMissingError(LgbScorerError):
    """``lub_lgb_models`` 行存在但磁盘文件缺失。"""


class LgbModelSchemaMismatch(LgbScorerError):
    """模型 ``feature_name()`` 与当前 :data:`FEATURE_NAMES` 不一致。"""


# ---------------------------------------------------------------------------
# Per-batch result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LoadedModel:
    model_id: str
    booster: Any  # lightgbm.Booster
    feature_names: tuple[str, ...]


_SCORE_COLUMNS: tuple[str, ...] = ("lgb_score", "feature_hash", "feature_missing_json")


def _empty_scored_frame(index: pd.Index) -> pd.DataFrame:
    """Result frame for the unloaded / failed path: lgb_score = NaN everywhere."""
    return pd.DataFrame(
        {
            "lgb_score": [float("nan")] * len(index),
            "feature_hash": [""] * len(index),
            "feature_missing_json": ["[]"] * len(index),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# LgbScorer
# ---------------------------------------------------------------------------


class LgbScorer:
    """推理评分器；构造便宜，加载延迟，可被多 worker 只读共享。

    Parameters
    ----------
    db
        ``deeptrade.core.db.Database`` 实例。仅用于读 ``lub_lgb_models``。
    model_id
        显式指定模型 ID 时绕过 ``is_active`` 查找；常用于 :command:`lgb evaluate`
        或在线 A/B（v0.6+）。``None`` → 自动选 ``is_active=TRUE`` 行；零行
        → ``loaded=False``（合法降级）。

    Attributes
    ----------
    model_id
        实际加载到的 model_id；未加载或失败时为 ``None``。
    loaded
        True 表示 booster 在内存里、:meth:`score_batch` 会真的算分。
    load_error
        加载失败时的简短诊断字符串，用于 ``data_unavailable`` 报告。
    """

    def __init__(self, db: Database, *, model_id: str | None = None) -> None:
        self._db = db
        self._requested_model_id = model_id
        self._lock = threading.Lock()
        self._loaded: _LoadedModel | None = None
        self._load_attempted: bool = False
        self._load_error: str | None = None

    # ----- public surface -----------------------------------------------

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
        """显式触发 lazy load + 一次空 predict，确认模型可用。

        失败不抛异常——把诊断记入 :attr:`load_error`，调用方据此决定是否
        放进 ``Round1Bundle.data_unavailable``。
        """
        with self._lock:
            self._ensure_loaded_locked()
            if self._loaded is None:
                return
            try:
                # 单行空 predict 验证 booster 可用 + numpy/lightgbm 链路完整。
                self._loaded.booster.predict(
                    np.full((1, len(self._loaded.feature_names)), np.nan, dtype="float64")
                )
            except Exception as e:  # noqa: BLE001
                self._load_error = f"predict_warmup_failed: {type(e).__name__}: {e}"
                self._loaded = None
                logger.warning("LgbScorer.warmup predict probe failed: %s", e)

    def score_batch(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """对一个 ``(n_candidates × n_features)`` 矩阵打分。

        返回 DataFrame 列：

        * ``lgb_score``            — float ∈ [0, 1] 或 NaN（缺模型/失败）
        * ``feature_hash``         — 8 字节 BLAKE2b 的 16 进制；空字符串表示
          未参与评分（缺模型 / 预测失败时仍写空，便于上游统一处理）
        * ``feature_missing_json`` — 每行缺失的列名 JSON 数组

        Index 与输入 :attr:`feature_df` 完全一致。

        加载失败 → 全 NaN；评分异常 → 全 NaN（不抛异常，符合设计 §7.3 红线）。
        """
        if feature_df.empty:
            return _empty_scored_frame(feature_df.index)

        # 列对齐 + schema 校验（永远在主调用前做，避免 booster.predict 直接崩）
        try:
            aligned = feature_df.reindex(columns=list(FEATURE_NAMES))
            assert_columns(aligned)
        except FeatureSchemaMismatch as e:
            logger.warning("LgbScorer.score_batch input schema mismatch: %s", e)
            return _empty_scored_frame(feature_df.index)
        except Exception as e:  # noqa: BLE001
            logger.warning("LgbScorer.score_batch reindex failure: %s", e)
            return _empty_scored_frame(feature_df.index)

        with self._lock:
            self._ensure_loaded_locked()
            loaded = self._loaded

        # ---- per-row diagnostics 总是计算（便于上游 data_unavailable 提示）----
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

        # ---- 真正推理 ----
        try:
            raw = aligned.to_numpy(dtype="float64", na_value=np.nan)
            preds = loaded.booster.predict(raw)
        except Exception as e:  # noqa: BLE001 — predict 失败必须降级
            logger.warning(
                "LgbScorer.score_batch predict failed (%s); returning NaN row.", e
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
                "LgbScorer prediction shape mismatch (%d vs %d); returning NaN.",
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

        return pd.DataFrame(
            {
                "lgb_score": preds_array,
                "feature_hash": hashes,
                "feature_missing_json": missing_payload,
            },
            index=feature_df.index,
        )

    # ----- internals -----------------------------------------------------

    def _ensure_loaded_locked(self) -> None:
        """First-call path: pull row, locate file, load booster, validate schema.

        Caller must hold :attr:`_lock`. Safe to call multiple times: once
        :attr:`_load_attempted` is True the method短路。
        """
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            record = self._lookup_record()
        except Exception as e:  # noqa: BLE001 — table missing / DB broken
            self._load_error = f"registry_lookup_failed: {type(e).__name__}: {e}"
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
        except Exception as e:  # noqa: BLE001 — ImportError / file corrupt
            self._load_error = f"booster_load_failed: {type(e).__name__}: {e}"
            logger.warning("LgbScorer booster load failed: %s", e)
            return

        try:
            feat_names = tuple(booster.feature_name())
        except Exception as e:  # noqa: BLE001 — pathological booster
            self._load_error = f"booster_introspect_failed: {type(e).__name__}: {e}"
            logger.warning("LgbScorer booster introspection failed: %s", e)
            return

        if list(feat_names) != list(FEATURE_NAMES):
            self._load_error = (
                f"schema_mismatch: model_features({len(feat_names)}) != "
                f"FEATURE_NAMES({len(FEATURE_NAMES)})"
            )
            logger.warning(
                "LgbScorer: schema mismatch — refusing to load %s. "
                "Model file features=%s, current FEATURE_NAMES=%s",
                record.model_id,
                feat_names,
                tuple(FEATURE_NAMES),
            )
            return

        self._loaded = _LoadedModel(
            model_id=record.model_id, booster=booster, feature_names=feat_names
        )
        logger.info(
            "LgbScorer loaded model_id=%s (features=%d)",
            record.model_id,
            len(feat_names),
        )

    def _lookup_record(self) -> Any:
        """Return ``ModelRecord`` for the active (or explicitly requested) row."""
        if self._requested_model_id is not None:
            return lgb_registry.get_model(self._db, self._requested_model_id)
        return lgb_registry.get_active(self._db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_missing_payload(aligned: pd.DataFrame) -> list[str]:
    """每行的 NaN 列名 JSON 数组（与 ``feature_missing_json`` 列写库格式一致）。"""
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
    """每行特征向量的 8 字节 BLAKE2b hex；NaN 用 sentinel bytes 占位以保稳定。"""
    if aligned.empty:
        return []
    out: list[str] = []
    arr = aligned.to_numpy(dtype="float64", na_value=np.nan)
    for row in arr:
        h = hashlib.blake2b(digest_size=8)
        # 直接 hash 行字节序列；NaN 在 float64 下有稳定 bit pattern，跨平台一致。
        h.update(row.tobytes())
        out.append(h.hexdigest())
    return out


def _load_booster(model_file: str) -> Any:
    """Lazy-import lightgbm + load a booster from disk. Raises on import error."""
    try:
        import lightgbm as lgb_mod  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover — soft dep
        raise RuntimeError(
            "lightgbm 未安装：pip install 'lightgbm>=4.3' 才能加载 LGB 模型。"
        ) from e
    return lgb_mod.Booster(model_file=model_file)


# ---------------------------------------------------------------------------
# Decile attachment helper — used by data.py to fill in lgb_decile per batch
# ---------------------------------------------------------------------------


def attach_deciles(scored: pd.DataFrame, *, n_buckets: int = 10) -> pd.Series:
    """Compute the 1..``n_buckets`` decile bucket per candidate score.

    Returns a series aligned to ``scored.index``; positions with NaN
    ``lgb_score`` get NaN. When the batch has fewer than ``n_buckets``
    finite scores, every row gets NaN (设计 §3.1：candidate 数量 < 10 时
    decile = None)。
    """
    if "lgb_score" not in scored.columns:
        return pd.Series([float("nan")] * len(scored), index=scored.index, dtype="Float64")
    s = pd.to_numeric(scored["lgb_score"], errors="coerce")
    finite = s.dropna()
    if len(finite) < n_buckets:
        return pd.Series([pd.NA] * len(scored), index=scored.index, dtype="Int64")
    try:
        # qcut returns 0..n_buckets-1 → +1 for 1..n_buckets
        codes = pd.qcut(finite, q=n_buckets, labels=False, duplicates="drop")
    except ValueError:
        # all-equal scores → qcut fails; degrade to NaN
        return pd.Series([pd.NA] * len(scored), index=scored.index, dtype="Int64")
    out = pd.Series([pd.NA] * len(scored), index=scored.index, dtype="Int64")
    out.loc[finite.index] = (codes.astype("Int64") + 1)
    return out
