"""PR-1.3 — ``lub_lgb_models`` registry CRUD 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptrade.core.db import Database

from limit_up_board.lgb.registry import (
    ModelRecord,
    delete_model,
    ensure_unique_model_id,
    get_active,
    get_model,
    insert_model,
    list_models,
    mint_model_id,
    set_active,
)

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_001_lgb_tables.sql"
)


@pytest.fixture
def lgb_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.duckdb")
    # 只应用本 PR 关心的迁移；不引入前置 init 迁移以保持测试隔离。
    sql_text = MIGRATION_FILE.read_text(encoding="utf-8")
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    return db


def _make_record(model_id: str = "20260530_1_abc123", *, is_active: bool = False) -> ModelRecord:
    return ModelRecord(
        model_id=model_id,
        schema_version=1,
        train_start_date="20260101",
        train_end_date="20260530",
        n_samples=2000,
        n_positive=420,
        cv_auc_mean=0.71,
        cv_auc_std=0.02,
        cv_logloss_mean=0.61,
        feature_count=50,
        feature_list_json="[]",
        hyperparams_json="{}",
        framework_version="0.2.0",
        plugin_version="0.5.0-alpha.2",
        git_commit="abc123",
        file_path="models/lgb_model_20260530_1_abc123.txt",
        is_active=is_active,
    )


class TestInsertAndQuery:
    def test_insert_activates_by_default(self, lgb_db: Database) -> None:
        insert_model(lgb_db, _make_record())
        active = get_active(lgb_db)
        assert active is not None
        assert active.model_id == "20260530_1_abc123"
        assert active.is_active is True

    def test_insert_no_activate(self, lgb_db: Database) -> None:
        insert_model(lgb_db, _make_record("first"), activate=False)
        assert get_active(lgb_db) is None
        # but the row is there
        assert get_model(lgb_db, "first") is not None

    def test_second_insert_flips_active(self, lgb_db: Database) -> None:
        insert_model(lgb_db, _make_record("first"))
        insert_model(lgb_db, _make_record("second"))
        active = get_active(lgb_db)
        assert active is not None and active.model_id == "second"
        # 旧的 active 被翻回 False
        first = get_model(lgb_db, "first")
        assert first is not None and first.is_active is False

    def test_list_models_ordered_by_created_desc(self, lgb_db: Database) -> None:
        insert_model(lgb_db, _make_record("oldest"))
        insert_model(lgb_db, _make_record("middle"))
        insert_model(lgb_db, _make_record("newest"))
        ids = [r.model_id for r in list_models(lgb_db)]
        assert ids == ["newest", "middle", "oldest"]


class TestSetActive:
    def test_atomic_switch(self, lgb_db: Database) -> None:
        insert_model(lgb_db, _make_record("a"))
        insert_model(lgb_db, _make_record("b"))
        # 现在 active = b
        assert set_active(lgb_db, "a") is True
        active = get_active(lgb_db)
        assert active is not None and active.model_id == "a"
        # b 翻 False
        b = get_model(lgb_db, "b")
        assert b is not None and b.is_active is False

    def test_set_active_unknown_id(self, lgb_db: Database) -> None:
        assert set_active(lgb_db, "ghost") is False


class TestDelete:
    def test_delete_existing(self, lgb_db: Database) -> None:
        insert_model(lgb_db, _make_record("doomed"))
        assert delete_model(lgb_db, "doomed") is True
        assert get_model(lgb_db, "doomed") is None

    def test_delete_missing(self, lgb_db: Database) -> None:
        assert delete_model(lgb_db, "never-existed") is False


class TestModelIdMinting:
    def test_mint_with_commit(self) -> None:
        out = mint_model_id(train_end_date="20260530", schema_version=1, git_commit="abc123")
        assert out == "20260530_1_abc123"

    def test_mint_without_commit(self) -> None:
        out = mint_model_id(train_end_date="20260530", schema_version=1, git_commit=None)
        assert out == "20260530_1_nogit"

    def test_ensure_unique_when_collision(self, lgb_db: Database) -> None:
        insert_model(lgb_db, _make_record("20260530_1_abc"))
        assert ensure_unique_model_id(lgb_db, "20260530_1_abc") == "20260530_1_abc-2"
        insert_model(lgb_db, _make_record("20260530_1_abc-2"))
        assert ensure_unique_model_id(lgb_db, "20260530_1_abc") == "20260530_1_abc-3"

    def test_ensure_unique_when_no_collision(self, lgb_db: Database) -> None:
        assert ensure_unique_model_id(lgb_db, "fresh-id") == "fresh-id"
