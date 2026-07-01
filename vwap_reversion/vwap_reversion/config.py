"""Plugin-local settings（设计 §8）.

Persisted in the ``vwr_config`` table（key, value_json）。Defaults live on
:class:`VwrConfig` and are re-applied automatically when a row is missing —
DB rows only override. 与 limit_up_board.config 同构。

CLI 一次性覆盖（``run --k-entry`` 等）不落库；本表是持久默认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

BandMode = Literal["vol_weighted", "time_std"]
PositionMode = Literal["round_trip", "base_position_t"]
SignalVersion = Literal["v1", "v2"]
_VALID_BAND_MODES: frozenset[str] = frozenset({"vol_weighted", "time_std"})
_VALID_POSITION_MODES: frozenset[str] = frozenset({"round_trip", "base_position_t"})
_VALID_SIGNAL_VERSIONS: frozenset[str] = frozenset({"v1", "v2"})


@dataclass
class VwrConfig:
    """User-tunable knobs. Defaults按设计 §8；band_k_* / warmup 等待 P2/P3
    用真实采样回标后调整。"""

    # ---- 时区 / 待机（§4）----
    market_timezone: str = "Asia/Shanghai"
    standby_heartbeat_seconds: int = 60
    standby_across_days: bool = False

    # ---- 采集 ----
    poll_interval_seconds: int = 30

    # ---- VWAP 带 / 信号（§2）----
    band_mode: BandMode = "vol_weighted"
    band_k_entry: float = 2.0
    band_k_exit: float = 0.3
    band_k_stop: float = 3.5
    warmup_minutes: int = 15
    signal_version: SignalVersion = "v1"
    confirm_z_recover: float = 0.35
    min_rebound_bps: float = 3.0
    max_holding_seconds: int = 1800
    high_vol_sigma_bps: float = 60.0
    high_vol_entry_multiplier: float = 1.2
    trend_guard_vwap_slope_bps: float = 3.0

    # ---- 持仓模式 ----
    position_mode: PositionMode = "round_trip"
    base_shares: int = 0
    order_qty: int = 100

    # ---- 风控（§13）----
    max_trades_per_day: int = 10
    min_holding_seconds: int = 60
    cooldown_seconds: int = 60
    per_trade_stop_pct: float = 0.8
    daily_loss_limit_pct: float = 1.5
    kill_switch_enabled: bool = False
    max_consecutive_losses: int = 2
    stale_quote_seconds: int = 90
    limit_price_guard_bps: float = 20.0
    new_entry_cutoff_time: str = "14:40"
    eod_flat_time: str = "14:55"

    # ---- 模拟账户 / 成本 ----
    initial_cash: float = 100_000.0
    fee_bps: float = 0.5
    min_fee_per_trade: float = 0.0
    slippage_bps: float = 1.0


_KEY_PREFIX = "vwr."


def _full_key(field_name: str) -> str:
    return f"{_KEY_PREFIX}{field_name}"


def validate_config(cfg: VwrConfig) -> None:
    """Strict validation. Raises ``ValueError``（typer 会原样透传给 CLI 用户）。
    由 ``load_config``（拦截脏 DB 值）与 ``save_config``（拒绝坏 CLI 输入）调用。
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # noqa: PLC0415

    try:
        ZoneInfo(cfg.market_timezone)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f"market_timezone 无法解析: {cfg.market_timezone!r}（{e}）") from e
    if cfg.standby_heartbeat_seconds < 5:
        raise ValueError(
            f"standby_heartbeat_seconds 必须 >= 5（当前 {cfg.standby_heartbeat_seconds}）"
        )
    if cfg.poll_interval_seconds < 5:
        raise ValueError(
            f"poll_interval_seconds 必须 >= 5（当前 {cfg.poll_interval_seconds}）"
        )
    if cfg.band_mode not in _VALID_BAND_MODES:
        raise ValueError(
            f"band_mode 必须为 'vol_weighted' / 'time_std' 之一（当前 {cfg.band_mode!r}）"
        )
    if cfg.band_k_entry <= 0:
        raise ValueError(f"band_k_entry 必须 > 0（当前 {cfg.band_k_entry}）")
    if cfg.band_k_exit < 0:
        raise ValueError(f"band_k_exit 必须 >= 0（当前 {cfg.band_k_exit}）")
    if cfg.band_k_exit >= cfg.band_k_entry:
        raise ValueError(
            f"band_k_exit（{cfg.band_k_exit}）必须 < band_k_entry（{cfg.band_k_entry}）"
        )
    if cfg.band_k_stop <= cfg.band_k_entry:
        raise ValueError(
            f"band_k_stop（{cfg.band_k_stop}）必须 > band_k_entry（{cfg.band_k_entry}）"
        )
    if cfg.warmup_minutes < 0:
        raise ValueError(f"warmup_minutes 必须 >= 0（当前 {cfg.warmup_minutes}）")
    if cfg.signal_version not in _VALID_SIGNAL_VERSIONS:
        raise ValueError(
            f"signal_version 必须为 'v1' / 'v2' 之一（当前 {cfg.signal_version!r}）"
        )
    if cfg.confirm_z_recover < 0:
        raise ValueError(f"confirm_z_recover 必须 >= 0（当前 {cfg.confirm_z_recover}）")
    if cfg.min_rebound_bps < 0:
        raise ValueError(f"min_rebound_bps 必须 >= 0（当前 {cfg.min_rebound_bps}）")
    if cfg.max_holding_seconds < 0:
        raise ValueError(f"max_holding_seconds 必须 >= 0（当前 {cfg.max_holding_seconds}）")
    if cfg.high_vol_sigma_bps < 0:
        raise ValueError(f"high_vol_sigma_bps 必须 >= 0（当前 {cfg.high_vol_sigma_bps}）")
    if cfg.high_vol_entry_multiplier < 1.0:
        raise ValueError(
            f"high_vol_entry_multiplier 必须 >= 1（当前 {cfg.high_vol_entry_multiplier}）"
        )
    if cfg.trend_guard_vwap_slope_bps < 0:
        raise ValueError(
            f"trend_guard_vwap_slope_bps 必须 >= 0（当前 {cfg.trend_guard_vwap_slope_bps}）"
        )
    if cfg.position_mode not in _VALID_POSITION_MODES:
        raise ValueError(
            f"position_mode 必须为 'round_trip' / 'base_position_t' 之一"
            f"（当前 {cfg.position_mode!r}）"
        )
    if cfg.base_shares < 0 or cfg.base_shares % 100 != 0:
        raise ValueError(f"base_shares 必须为 >= 0 的 100 整数倍（当前 {cfg.base_shares}）")
    if cfg.position_mode == "base_position_t" and cfg.base_shares == 0:
        raise ValueError("position_mode=base_position_t 时 base_shares 必须 > 0")
    if cfg.order_qty <= 0 or cfg.order_qty % 100 != 0:
        raise ValueError(f"order_qty 必须为正的 100 整数倍（当前 {cfg.order_qty}）")
    if cfg.max_trades_per_day < 1:
        raise ValueError(f"max_trades_per_day 必须 >= 1（当前 {cfg.max_trades_per_day}）")
    if cfg.min_holding_seconds < 0:
        raise ValueError(f"min_holding_seconds 必须 >= 0（当前 {cfg.min_holding_seconds}）")
    if cfg.cooldown_seconds < 0:
        raise ValueError(f"cooldown_seconds 必须 >= 0（当前 {cfg.cooldown_seconds}）")
    if cfg.per_trade_stop_pct <= 0:
        raise ValueError(f"per_trade_stop_pct 必须 > 0（当前 {cfg.per_trade_stop_pct}）")
    if cfg.daily_loss_limit_pct <= 0:
        raise ValueError(f"daily_loss_limit_pct 必须 > 0（当前 {cfg.daily_loss_limit_pct}）")
    if cfg.max_consecutive_losses < 1:
        raise ValueError(
            f"max_consecutive_losses 必须 >= 1（当前 {cfg.max_consecutive_losses}）"
        )
    if cfg.stale_quote_seconds < cfg.poll_interval_seconds:
        raise ValueError(
            "stale_quote_seconds 必须 >= poll_interval_seconds"
            f"（当前 {cfg.stale_quote_seconds} < {cfg.poll_interval_seconds}）"
        )
    if cfg.limit_price_guard_bps < 0:
        raise ValueError(
            f"limit_price_guard_bps 必须 >= 0（当前 {cfg.limit_price_guard_bps}）"
        )
    # eod_flat_time 必须可解析且落在下午连续竞价段内（否则强平永远不会触发）。
    from .clock import AFTERNOON_CLOSE, AFTERNOON_OPEN, MORNING_OPEN, parse_hhmm  # noqa: PLC0415

    try:
        eod = parse_hhmm(cfg.eod_flat_time)
    except ValueError as e:
        raise ValueError(f"eod_flat_time 无法解析为 HH:MM: {cfg.eod_flat_time!r}") from e
    if not (AFTERNOON_OPEN <= eod < AFTERNOON_CLOSE):
        raise ValueError(
            f"eod_flat_time 必须落在 13:00–15:00 之间（当前 {cfg.eod_flat_time!r}）"
        )
    try:
        cutoff = parse_hhmm(cfg.new_entry_cutoff_time)
    except ValueError as e:
        raise ValueError(
            f"new_entry_cutoff_time 无法解析为 HH:MM: {cfg.new_entry_cutoff_time!r}"
        ) from e
    if not (MORNING_OPEN <= cutoff < eod):
        raise ValueError(
            "new_entry_cutoff_time 必须落在 09:30 之后且早于 eod_flat_time"
            f"（当前 {cfg.new_entry_cutoff_time!r}, eod={cfg.eod_flat_time!r}）"
        )
    if cfg.initial_cash <= 0:
        raise ValueError(f"initial_cash 必须 > 0（当前 {cfg.initial_cash}）")
    if cfg.fee_bps < 0:
        raise ValueError(f"fee_bps 必须 >= 0（当前 {cfg.fee_bps}）")
    if cfg.min_fee_per_trade < 0:
        raise ValueError(
            f"min_fee_per_trade 必须 >= 0（当前 {cfg.min_fee_per_trade}）"
        )
    if cfg.slippage_bps < 0:
        raise ValueError(f"slippage_bps 必须 >= 0（当前 {cfg.slippage_bps}）")


def load_config(db: Database) -> VwrConfig:
    """Materialize from ``vwr_config``; missing rows fall through to defaults.
    ``validate_config`` runs on the assembled object so corrupt DB values
    surface immediately."""
    overrides: dict[str, Any] = {}
    for f in fields(VwrConfig):
        row = db.fetchone(
            "SELECT value_json FROM vwr_config WHERE key = ?", (_full_key(f.name),)
        )
        if row is not None:
            overrides[f.name] = json.loads(row[0])
    cfg = VwrConfig(**overrides)
    validate_config(cfg)
    return cfg


def save_config(db: Database, cfg: VwrConfig) -> None:
    """Upsert every field into ``vwr_config``. Validates first."""
    validate_config(cfg)
    with db.transaction():
        for f in fields(VwrConfig):
            key = _full_key(f.name)
            payload = json.dumps(getattr(cfg, f.name))
            db.execute("DELETE FROM vwr_config WHERE key = ?", (key,))
            db.execute(
                "INSERT INTO vwr_config(key, value_json) VALUES (?, ?)", (key, payload)
            )


def reset_config(db: Database) -> None:
    """Delete all persisted rows → everything falls back to dataclass defaults."""
    db.execute("DELETE FROM vwr_config WHERE key LIKE ?", (f"{_KEY_PREFIX}%",))


def list_for_show(db: Database) -> list[tuple[str, Any, str]]:
    """``[(key, value, source)]`` for ``settings show``；source =
    persisted / default。"""
    out: list[tuple[str, Any, str]] = []
    defaults = VwrConfig()
    for f in fields(VwrConfig):
        key = _full_key(f.name)
        row = db.fetchone("SELECT value_json FROM vwr_config WHERE key = ?", (key,))
        if row is not None:
            out.append((key, json.loads(row[0]), "persisted"))
        else:
            out.append((key, getattr(defaults, f.name), "default"))
    return out


def set_one(db: Database, field_name: str, raw_value: str) -> VwrConfig:
    """``settings set <key> <value>`` 的实现：解析（JSON 优先，回退字符串）、
    替换、整体校验、落库。返回新配置。"""
    valid_names = {f.name for f in fields(VwrConfig)}
    if field_name not in valid_names:
        raise ValueError(
            f"未知配置项: {field_name!r}（可用项: {', '.join(sorted(valid_names))}）"
        )
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value  # bare string（如 Asia/Shanghai、14:55、round_trip）
    cur = load_config(db)
    from dataclasses import replace  # noqa: PLC0415

    new_cfg = replace(cur, **{field_name: value})
    save_config(db, new_cfg)  # validates
    return new_cfg
