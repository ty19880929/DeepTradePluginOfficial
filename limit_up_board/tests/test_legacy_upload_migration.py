"""v0.13.3 — ``migrate_legacy_upload_config`` 幂等迁移测试。

旧 ``lub.summary_upload_*`` 行 → 框架 ``report.upload.*``（部分走 secret_store）。
完成后清除旧行；再次调用应直接返回 ``False`` 且不产生副作用。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeptrade.core.db import Database
from limit_up_board.config import migrate_legacy_upload_config


@pytest.fixture
def db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "migrate.duckdb")
    db.execute(
        "CREATE TABLE lub_config ("
        "key VARCHAR PRIMARY KEY, value_json VARCHAR NOT NULL, "
        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    return db


class _FakeConfig:
    """Minimal ConfigService stand-in: only ``set`` / ``source_of`` are touched."""

    def __init__(self) -> None:
        self.sets: list[tuple[str, object]] = []
        self.sources: dict[str, str] = {}

    def source_of(self, key: str) -> str:
        return self.sources.get(key, "default")

    def set(self, key: str, value: object) -> None:
        self.sets.append((key, value))


def _seed_legacy_rows(db: Database, *, enabled: bool, url: str, timeout: float, token: str) -> None:
    rows = {
        "lub.summary_upload_enabled": json.dumps(enabled),
        "lub.summary_upload_url": json.dumps(url),
        "lub.summary_upload_timeout": json.dumps(timeout),
        "lub.summary_upload_token": json.dumps(token),
    }
    for k, v in rows.items():
        db.execute("INSERT INTO lub_config(key, value_json) VALUES (?, ?)", (k, v))


def _remaining_keys(db: Database) -> list[str]:
    return [r[0] for r in db.fetchall("SELECT key FROM lub_config")]


def test_no_legacy_rows_is_noop(db: Database) -> None:
    cfg = _FakeConfig()
    assert migrate_legacy_upload_config(db, cfg) is False
    assert cfg.sets == []


def test_migrates_all_fields_then_cleans_up(db: Database) -> None:
    _seed_legacy_rows(
        db,
        enabled=True,
        url="https://my.custom.endpoint/api/upload",
        timeout=45.0,
        token="my-secret-token",
    )
    cfg = _FakeConfig()  # 全部框架键来源 == default

    assert migrate_legacy_upload_config(db, cfg) is True

    written = dict(cfg.sets)
    assert written["report.upload.url"] == "https://my.custom.endpoint/api/upload"
    assert written["report.upload.timeout"] == 45.0
    assert written["report.upload.token"] == "my-secret-token"
    assert written["report.upload.enabled"] is True

    # 旧行已清空
    assert _remaining_keys(db) == []


def test_skips_url_when_framework_already_set(db: Database) -> None:
    _seed_legacy_rows(
        db,
        enabled=False,
        url="https://my.legacy/upload",
        timeout=30.0,
        token="",
    )
    cfg = _FakeConfig()
    # 框架已被用户手动设置过 URL；不应被覆盖。
    cfg.sources["report.upload.url"] = "persisted"
    cfg.sources["report.upload.timeout"] = "persisted"

    assert migrate_legacy_upload_config(db, cfg) is True
    written = dict(cfg.sets)
    assert "report.upload.url" not in written
    assert "report.upload.timeout" not in written
    # token 为空 → 不写
    assert "report.upload.token" not in written
    # enabled=False → 不写
    assert "report.upload.enabled" not in written
    # 旧行已清空
    assert _remaining_keys(db) == []


def test_token_non_empty_always_overrides(db: Database) -> None:
    _seed_legacy_rows(
        db,
        enabled=False,
        url="",
        timeout=30.0,
        token="legacy-token",
    )
    cfg = _FakeConfig()
    # 即使框架 token 已被手动设置过，旧 token 也应覆盖（迁移意图：最新意图保留）
    cfg.sources["report.upload.token"] = "persisted"

    assert migrate_legacy_upload_config(db, cfg) is True
    written = dict(cfg.sets)
    assert written["report.upload.token"] == "legacy-token"


def test_idempotent_second_call_is_noop(db: Database) -> None:
    _seed_legacy_rows(
        db,
        enabled=True,
        url="https://x/y",
        timeout=30.0,
        token="t",
    )
    cfg1 = _FakeConfig()
    assert migrate_legacy_upload_config(db, cfg1) is True
    assert cfg1.sets  # 第一次有副作用

    cfg2 = _FakeConfig()
    assert migrate_legacy_upload_config(db, cfg2) is False
    assert cfg2.sets == []
