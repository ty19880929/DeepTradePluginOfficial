"""Plugin-local settings (v0.4 + v0.5 LGB 扩展).

Persisted in the ``lub_config`` table. Defaults live on :class:`LubConfig` and
are re-applied automatically when a row is missing — DB rows only override.

v0.4 字段（已沿用）：
    * ``min_float_mv_yi``  — 流通市值下限（亿）
    * ``max_float_mv_yi``  — 流通市值上限（亿）
    * ``max_close_yuan``   — 当前股价上限（元）

v0.5 LGB 字段（lightgbm_design.md §10）：
    * ``lgb_enabled``                  — 全局开关
    * ``lgb_min_score_floor``          — 强势初筛 prompt 中提示 LLM 的分数下限
    * ``lgb_decile_in_prompt``         — 是否注入 lgb_decile
    * ``lgb_label_threshold_pct``      — 次日最大溢价概率阈值（%）：T+1.high
                                         >= pre_close * (1 + 阈值%) 的样本标为正例。
                                         **注意**：此标签为「次日最高价是否触及阈值溢价」，
                                         并非「能否成交」、也非「真实可实现收益」（详见 P1-1
                                         讨论）。默认 9.7 接近涨停。
    * ``lgb_train_lookback_days``      — train CLI 默认窗口
    * ``lgb_train_min_samples``        — 训练样本量下限
    * ``lgb_max_models_to_keep``       — prune 默认保留模型数
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

EmptyArrayPolicy = Literal["repair", "degraded", "fallback"]
_VALID_EMPTY_ARRAY_POLICIES: frozenset[str] = frozenset({"repair", "degraded", "fallback"})


@dataclass
class LubConfig:
    """User-tunable run filters + LGB knobs. Defaults reflect a typical 打板 watchlist."""

    min_float_mv_yi: float = 30.0
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

    # ---- summary.json 上传到 DeepTrade 官网（v0.12+；v0.10/0.11 是 summary.html）----
    # v0.12.3 起 **默认关闭**（隐私优先）；需要分享报告时显式：
    #     deeptrade limit-up-board settings set summary_upload_enabled true
    # 失败时仅记 WARN 日志、不阻断 run。
    summary_upload_enabled: bool = False
    summary_upload_url: str = "https://deeptrade.tiey.ai/api/reports/upload"
    summary_upload_timeout: float = 30.0
    # 上传鉴权 token。v0.12.3+：从源码硬编码（"deeptrade"）改为可配置；空串表示匿名上传，
    # 不写 Authorization header，由服务端决定是否拒绝。
    summary_upload_token: str = ""

    # ---- v0.12.4 (P1-2)：连板预测 / 辩论修订 阶段的空数组兜底策略 ----
    # "repair"   → LLM 重新生成（默认；与 prompt 文案一致：禁止空数组）
    # "degraded" → 保留占位符 + 在 candidate.degraded_fields 标注命中字段
    # "fallback" → 仅替换为占位符（v0.12.3 及之前的行为）
    empty_array_policy: EmptyArrayPolicy = "repair"


_KEY_PREFIX = "lub."


def _full_key(field_name: str) -> str:
    return f"{_KEY_PREFIX}{field_name}"


def validate_config(cfg: LubConfig) -> None:
    """Strict validation for a :class:`LubConfig` instance.

    Called by ``load_config`` (catches corrupt DB values) / ``save_config``
    (rejects bad CLI input) so violations surface as early as possible
    rather than silently produce 0-candidate runs or model training errors.

    Raises ``ValueError`` (not ``AssertionError``) with a human-readable
    message identifying the offending field — typer will surface the message
    verbatim to CLI users via Exit(2).
    """
    if cfg.min_float_mv_yi < 0:
        raise ValueError(
            f"min_float_mv_yi 必须 >= 0（当前 {cfg.min_float_mv_yi}）"
        )
    if cfg.min_float_mv_yi >= cfg.max_float_mv_yi:
        raise ValueError(
            f"min_float_mv_yi（{cfg.min_float_mv_yi}）必须 < "
            f"max_float_mv_yi（{cfg.max_float_mv_yi}）"
        )
    if cfg.max_close_yuan <= 0:
        raise ValueError(
            f"max_close_yuan 必须 > 0（当前 {cfg.max_close_yuan}）"
        )
    if not (0 < cfg.lgb_label_threshold_pct < 20):
        raise ValueError(
            f"lgb_label_threshold_pct 必须落在 (0, 20) 之间"
            f"（当前 {cfg.lgb_label_threshold_pct}）"
        )
    if cfg.lgb_min_score_floor is not None and not (
        0 <= cfg.lgb_min_score_floor <= 100
    ):
        raise ValueError(
            f"lgb_min_score_floor 必须为 None 或 [0, 100] 内的数值"
            f"（当前 {cfg.lgb_min_score_floor}）"
        )
    if cfg.lgb_train_lookback_days < 30:
        raise ValueError(
            f"lgb_train_lookback_days 必须 >= 30（当前 {cfg.lgb_train_lookback_days}）"
        )
    if cfg.lgb_train_min_samples < 100:
        raise ValueError(
            f"lgb_train_min_samples 必须 >= 100（当前 {cfg.lgb_train_min_samples}）"
        )
    if cfg.lgb_max_models_to_keep < 1:
        raise ValueError(
            f"lgb_max_models_to_keep 必须 >= 1（当前 {cfg.lgb_max_models_to_keep}）"
        )
    if cfg.summary_upload_enabled and not (
        cfg.summary_upload_url.startswith("http://")
        or cfg.summary_upload_url.startswith("https://")
    ):
        raise ValueError(
            "summary_upload_url 必须是 http(s):// 开头的有效 URL"
            f"（当前 {cfg.summary_upload_url!r}）"
        )
    if cfg.summary_upload_timeout <= 0:
        raise ValueError(
            f"summary_upload_timeout 必须 > 0（当前 {cfg.summary_upload_timeout}）"
        )
    if cfg.empty_array_policy not in _VALID_EMPTY_ARRAY_POLICIES:
        raise ValueError(
            f"empty_array_policy 必须为 'repair' / 'degraded' / 'fallback' 之一"
            f"（当前 {cfg.empty_array_policy!r}）"
        )


def load_config(db: Database) -> LubConfig:
    """Materialize a :class:`LubConfig` from ``lub_config``; missing rows fall
    through to the dataclass default. ``validate_config`` runs on the
    assembled object so corrupt DB values surface immediately."""
    overrides: dict[str, Any] = {}
    for f in fields(LubConfig):
        row = db.fetchone("SELECT value_json FROM lub_config WHERE key = ?", (_full_key(f.name),))
        if row is not None:
            overrides[f.name] = json.loads(row[0])
    cfg = LubConfig(**overrides)
    validate_config(cfg)
    return cfg


def save_config(db: Database, cfg: LubConfig) -> None:
    """Upsert every field of *cfg* into ``lub_config``. Validates first."""
    validate_config(cfg)
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
