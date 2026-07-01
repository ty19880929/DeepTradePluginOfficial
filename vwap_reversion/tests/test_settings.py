"""VwrConfig 持久化 + 校验单测（P0 验收项：settings 链路）.

用临时 DuckDB 文件 + 真实迁移 SQL 建表 —— 同时验证迁移文件本身可被 DuckDB
执行。需要 deeptrade 框架可导入（Database）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptrade.core.db import Database

from vwap_reversion.config import (
    VwrConfig,
    list_for_show,
    load_config,
    reset_config,
    save_config,
    set_one,
    validate_config,
)

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "20260603_001_init.sql"


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "test.duckdb")
    sql = MIGRATION.read_text(encoding="utf-8")
    for stmt in sql.split(";"):
        if stmt.strip():
            database.execute(stmt)
    yield database
    database.close()


def test_defaults_load_and_validate(db: Database) -> None:
    cfg = load_config(db)
    assert cfg == VwrConfig()
    assert cfg.market_timezone == "Asia/Shanghai"
    assert cfg.poll_interval_seconds == 30
    assert cfg.standby_across_days is False


def test_save_and_reload_roundtrip(db: Database) -> None:
    cfg = VwrConfig(band_k_entry=2.5, poll_interval_seconds=60, standby_across_days=True)
    save_config(db, cfg)
    assert load_config(db) == cfg


def test_list_for_show_sources(db: Database) -> None:
    rows = {k: (v, src) for k, v, src in list_for_show(db)}
    assert rows["vwr.band_k_entry"] == (2.0, "default")
    save_config(db, VwrConfig())
    rows = {k: (v, src) for k, v, src in list_for_show(db)}
    assert rows["vwr.band_k_entry"] == (2.0, "persisted")


def test_set_one_parses_json_and_bare_string(db: Database) -> None:
    new_cfg = set_one(db, "band_k_entry", "2.5")
    assert new_cfg.band_k_entry == 2.5
    new_cfg = set_one(db, "standby_across_days", "true")
    assert new_cfg.standby_across_days is True
    new_cfg = set_one(db, "eod_flat_time", "14:50")  # 裸字符串（非 JSON）
    assert new_cfg.eod_flat_time == "14:50"
    new_cfg = set_one(db, "market_timezone", "Asia/Shanghai")
    assert new_cfg.market_timezone == "Asia/Shanghai"


def test_set_one_unknown_key_rejected(db: Database) -> None:
    with pytest.raises(ValueError, match="未知配置项"):
        set_one(db, "no_such_key", "1")


def test_reset_clears_persisted_rows(db: Database) -> None:
    save_config(db, VwrConfig(band_k_entry=3.0))
    reset_config(db)
    assert load_config(db) == VwrConfig()
    rows = {k: src for k, _v, src in list_for_show(db)}
    assert set(rows.values()) == {"default"}


# ---------------------------------------------------------------------------
# validate_config — 拒绝坏值
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("market_timezone", "Not/AZone", "market_timezone"),
        ("poll_interval_seconds", 1, "poll_interval_seconds"),
        ("standby_heartbeat_seconds", 0, "standby_heartbeat_seconds"),
        ("band_mode", "bogus", "band_mode"),
        ("band_k_entry", 0.0, "band_k_entry"),
        ("band_k_exit", 2.0, "band_k_exit"),          # >= k_entry
        ("band_k_stop", 1.0, "band_k_stop"),          # <= k_entry
        ("warmup_minutes", -1, "warmup_minutes"),
        ("signal_version", "v3", "signal_version"),
        ("confirm_z_recover", -0.1, "confirm_z_recover"),
        ("min_rebound_bps", -0.1, "min_rebound_bps"),
        ("max_holding_seconds", -1, "max_holding_seconds"),
        ("high_vol_sigma_bps", -0.1, "high_vol_sigma_bps"),
        ("high_vol_entry_multiplier", 0.9, "high_vol_entry_multiplier"),
        ("trend_guard_vwap_slope_bps", -0.1, "trend_guard_vwap_slope_bps"),
        ("position_mode", "short", "position_mode"),
        ("base_shares", 150, "base_shares"),          # 非 100 整数倍
        ("order_qty", 0, "order_qty"),
        ("order_qty", 130, "order_qty"),
        ("max_trades_per_day", 0, "max_trades_per_day"),
        ("per_trade_stop_pct", 0.0, "per_trade_stop_pct"),
        ("daily_loss_limit_pct", 0.0, "daily_loss_limit_pct"),
        ("max_consecutive_losses", 0, "max_consecutive_losses"),
        ("stale_quote_seconds", 1, "stale_quote_seconds"),
        ("limit_price_guard_bps", -0.1, "limit_price_guard_bps"),
        ("new_entry_cutoff_time", "15:01", "new_entry_cutoff_time"),
        ("new_entry_cutoff_time", "abc", "new_entry_cutoff_time"),
        ("eod_flat_time", "15:30", "eod_flat_time"),  # 不在 13:00–15:00
        ("eod_flat_time", "abc", "eod_flat_time"),
        ("initial_cash", 0.0, "initial_cash"),
        ("fee_bps", -1.0, "fee_bps"),
        ("min_fee_per_trade", -0.1, "min_fee_per_trade"),
        ("slippage_bps", -0.1, "slippage_bps"),
    ],
)
def test_validate_rejects_bad_values(field: str, value, match: str) -> None:
    from dataclasses import replace

    cfg = replace(VwrConfig(), **{field: value})
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_validate_base_position_t_requires_base_shares() -> None:
    from dataclasses import replace

    cfg = replace(VwrConfig(), position_mode="base_position_t", base_shares=0)
    with pytest.raises(ValueError, match="base_shares"):
        validate_config(cfg)
    ok = replace(VwrConfig(), position_mode="base_position_t", base_shares=1000)
    validate_config(ok)  # 不抛


def test_corrupt_db_value_surfaces_on_load(db: Database) -> None:
    db.execute(
        "INSERT INTO vwr_config(key, value_json) VALUES ('vwr.band_k_entry', '-1.0')"
    )
    with pytest.raises(ValueError, match="band_k_entry"):
        load_config(db)
