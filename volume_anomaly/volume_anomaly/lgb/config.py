"""VaLgbConfig — plugin-local LightGBM settings (v0.7+).

Persisted in the ``va_config`` table. Defaults live on :class:`VaLgbConfig` and
are re-applied automatically when a row is missing — DB rows only override.

v0.7 LGB 字段（lightgbm_design.md §10）：
    * ``lgb_enabled``                  — 全局开关
    * ``lgb_min_score_floor``          — analyze prompt 中提示 LLM 的分数下限
    * ``lgb_decile_in_prompt``         — 是否注入 lgb_decile
    * ``lgb_label_threshold_pct``      — 标签阈值（默认 5.0）
    * ``lgb_label_source``             — 'max_ret_5d' | 'ret_t3' | 'max_ret_10d'
    * ``lgb_train_lookback_days``      — train CLI 默认窗口
    * ``lgb_train_min_samples``        — 训练样本量下限
    * ``lgb_max_models_to_keep``       — prune 默认保留模型数
    * ``lgb_max_datasets_to_keep``     — datasets/ 快照保留数

不走框架 ``ConfigService.set``：ConfigService 只允许 AppConfig 已知 key；
插件私有配置通过 ``va_config`` 自有表落库，与 limit-up-board 的 ``lub_config``
完全同构。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


_VALID_LABEL_SOURCES = ("max_ret_5d", "ret_t3", "max_ret_10d")


@dataclass
class VaLgbConfig:
    """User-tunable LightGBM knobs. Defaults reflect the design baseline."""

    lgb_enabled: bool = True
    lgb_min_score_floor: float | None = 25.0
    lgb_decile_in_prompt: bool = True
    lgb_label_threshold_pct: float = 5.0
    lgb_label_source: str = "max_ret_5d"
    lgb_train_lookback_days: int = 365
    lgb_train_min_samples: int = 1000
    lgb_max_models_to_keep: int = 5
    lgb_max_datasets_to_keep: int = 3

    def validate(self) -> None:
        """Sanity checks that the dataclass alone can't enforce."""
        if self.lgb_label_source not in _VALID_LABEL_SOURCES:
            raise ValueError(
                f"lgb_label_source must be one of {_VALID_LABEL_SOURCES}, "
                f"got {self.lgb_label_source!r}"
            )
        if self.lgb_label_threshold_pct <= 0:
            raise ValueError(
                f"lgb_label_threshold_pct must be > 0, got {self.lgb_label_threshold_pct}"
            )


_KEY_PREFIX = "va."


def _full_key(field_name: str) -> str:
    return f"{_KEY_PREFIX}{field_name}"


def load_config(db: Database) -> VaLgbConfig:
    """Materialize a :class:`VaLgbConfig` from ``va_config``; missing rows fall
    through to the dataclass default. Returns defaults when the table itself
    is absent (caller may be running against a pre-migration DB)."""
    overrides: dict[str, Any] = {}
    try:
        rows = db.fetchall("SELECT key, value_json FROM va_config")
    except Exception:  # noqa: BLE001 — table missing → use defaults
        return VaLgbConfig()
    by_key = {str(k): v for k, v in rows}
    for f in fields(VaLgbConfig):
        v = by_key.get(_full_key(f.name))
        if v is not None:
            overrides[f.name] = json.loads(v)
    return VaLgbConfig(**overrides)


def save_config(db: Database, cfg: VaLgbConfig) -> None:
    """Upsert every field of *cfg* into ``va_config``."""
    cfg.validate()
    with db.transaction():
        for f in fields(VaLgbConfig):
            key = _full_key(f.name)
            value = getattr(cfg, f.name)
            payload = json.dumps(value)
            db.execute("DELETE FROM va_config WHERE key = ?", (key,))
            db.execute(
                "INSERT INTO va_config(key, value_json) VALUES (?, ?)",
                (key, payload),
            )


def list_for_show(db: Database) -> list[tuple[str, Any, str]]:
    """``[(key, value, source)]`` for a future ``settings show`` table.

    ``source`` is ``"persisted"`` if the field has a row in ``va_config``,
    otherwise ``"default"``.
    """
    out: list[tuple[str, Any, str]] = []
    defaults = VaLgbConfig()
    for f in fields(VaLgbConfig):
        key = _full_key(f.name)
        try:
            row = db.fetchone("SELECT value_json FROM va_config WHERE key = ?", (key,))
        except Exception:  # noqa: BLE001
            row = None
        if row is not None:
            out.append((key, json.loads(row[0]), "persisted"))
        else:
            out.append((key, getattr(defaults, f.name), "default"))
    return out
