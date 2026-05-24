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
    from deeptrade.core.config import ConfigService
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

    # v0.13.3 起：summary.json 上传链路已下沉到框架 ``report.upload.*`` 配置族
    # （需要 ``deeptrade-quant>=0.11.0``）。旧的 ``lub.summary_upload_*`` 行由
    # :func:`migrate_legacy_upload_config` 在首次 dispatch 时一次性搬到框架配置
    # （URL / 超时只在框架仍是 default 时覆盖；token 非空则一律写入 secret_store；
    # enabled=True 写一次性开关），完成即清，无需用户介入。

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


def migrate_legacy_upload_config(db: Database, config: ConfigService) -> bool:
    """One-shot：把 ``lub.summary_upload_*`` 搬到框架 v0.11 的 ``report.upload.*``。

    幂等：只要 ``lub_config`` 中还有 ``lub.summary_upload_*`` 行就尝试迁移；迁移
    结束后 ``DELETE`` 这些行；下一次再调用直接 no-op。返回是否实际搬运过（供
    事件日志 / 测试断言用）。

    搬运策略：
      * ``url`` / ``timeout`` —— 仅当框架侧仍是 ``default`` 来源时才覆盖（不抢
        用户已手动设置的值）。
      * ``token`` —— 非空则一律写 ``secret_store``（迁移之前没地方存）。
      * ``enabled=True`` —— 一次性写 ``report.upload.enabled=True``；
        False / 未设 → 不动框架开关。
    """
    legacy_rows = dict(
        db.fetchall(
            "SELECT key, value_json FROM lub_config WHERE key LIKE 'lub.summary_upload_%'"
        )
    )
    if not legacy_rows:
        return False

    def _get(key: str, default: Any = None) -> Any:
        raw = legacy_rows.get(key)
        return json.loads(raw) if raw is not None else default

    legacy_url = _get("lub.summary_upload_url")
    legacy_timeout = _get("lub.summary_upload_timeout")
    legacy_token = _get("lub.summary_upload_token")
    legacy_enabled = bool(_get("lub.summary_upload_enabled", False))

    if (
        isinstance(legacy_url, str)
        and legacy_url
        and config.source_of("report.upload.url") == "default"
    ):
        config.set("report.upload.url", legacy_url)
    if (
        isinstance(legacy_timeout, (int, float))
        and legacy_timeout > 0
        and config.source_of("report.upload.timeout") == "default"
    ):
        config.set("report.upload.timeout", float(legacy_timeout))
    if isinstance(legacy_token, str) and legacy_token:
        config.set("report.upload.token", legacy_token)
    if legacy_enabled:
        config.set("report.upload.enabled", True)

    db.execute("DELETE FROM lub_config WHERE key LIKE 'lub.summary_upload_%'")
    return True


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
