"""Plugin-local settings (v0.4 + v0.5 LGB 扩展).

Persisted in the ``lub_config`` table. Defaults live on :class:`LubConfig` and
are re-applied automatically when a row is missing — DB rows only override.

v0.4 字段（已沿用）：
    * ``max_float_mv_yi``  — 流通市值上限（亿）
    * ``max_close_yuan``   — 当前股价上限（元）

v0.5 LGB 字段（lightgbm_design.md §10）：
    * ``lgb_enabled``                  — 全局开关
    * ``lgb_min_score_floor``          — R1 prompt 中提示 LLM 的分数下限
    * ``lgb_decile_in_prompt``         — 是否注入 lgb_decile
    * ``lgb_label_threshold_pct``      — T+1 阈值（默认 9.7）
    * ``lgb_train_lookback_days``      — train CLI 默认窗口
    * ``lgb_train_min_samples``        — 训练样本量下限
    * ``lgb_max_models_to_keep``       — prune 默认保留模型数
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


@dataclass
class LubConfig:
    """User-tunable run filters + LGB knobs. Defaults reflect a typical 打板 watchlist."""

    max_float_mv_yi: float = 100.0
    max_close_yuan: float = 15.0

    # ---- v0.5 LightGBM 评分相关 ----
    lgb_enabled: bool = True
    lgb_min_score_floor: float | None = 30.0
    lgb_decile_in_prompt: bool = True
    lgb_label_threshold_pct: float = 9.7
    lgb_train_lookback_days: int = 730
    lgb_train_min_samples: int = 1500
    lgb_max_models_to_keep: int = 5


_KEY_PREFIX = "lub."


def _full_key(field_name: str) -> str:
    return f"{_KEY_PREFIX}{field_name}"


def load_config(db: Database) -> LubConfig:
    """Materialize a :class:`LubConfig` from ``lub_config``; missing rows fall
    through to the dataclass default."""
    overrides: dict[str, Any] = {}
    for f in fields(LubConfig):
        row = db.fetchone("SELECT value_json FROM lub_config WHERE key = ?", (_full_key(f.name),))
        if row is not None:
            overrides[f.name] = json.loads(row[0])
    return LubConfig(**overrides)


def save_config(db: Database, cfg: LubConfig) -> None:
    """Upsert every field of *cfg* into ``lub_config``."""
    with db.transaction():
        for f in fields(LubConfig):
            key = _full_key(f.name)
            value = getattr(cfg, f.name)
            payload = json.dumps(value)
            db.execute("DELETE FROM lub_config WHERE key = ?", (key,))
            db.execute(
                "INSERT INTO lub_config(key, value_json) VALUES (?, ?)",
                (key, payload),
            )


def list_for_show(db: Database) -> list[tuple[str, Any, str]]:
    """``[(key, value, source)]`` for the ``settings show`` table.

    ``source`` is ``"persisted"`` if the field has a row in ``lub_config``,
    otherwise ``"default"``.
    """
    out: list[tuple[str, Any, str]] = []
    defaults = LubConfig()
    for f in fields(LubConfig):
        key = _full_key(f.name)
        row = db.fetchone("SELECT value_json FROM lub_config WHERE key = ?", (key,))
        if row is not None:
            out.append((key, json.loads(row[0]), "persisted"))
        else:
            out.append((key, getattr(defaults, f.name), "default"))
    return out
