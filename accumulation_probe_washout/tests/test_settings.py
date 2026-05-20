"""ApwConfigStore + settings commands — T6.1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accumulation_probe_washout.config import (
    ALLOWED_KEYS,
    ApwConfig,
    ApwConfigStore,
    from_dict,
    to_dict,
)


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def fresh_db(tmp_path):
    from deeptrade.core.db import Database
    db = Database(tmp_path / "apw.duckdb")
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        for stmt in path.read_text(encoding="utf-8").split(";"):
            if stmt.strip():
                db.execute(stmt.strip())
    yield db
    db.close()


class TestApwConfigStore:
    def test_load_returns_defaults_when_empty(self, fresh_db):
        cfg = ApwConfigStore(fresh_db).load()
        assert cfg.accumulation_score_min == 55.0
        assert cfg.washout_max_trade_days == 25
        assert cfg.llm_batch_size == 20

    def test_set_overrides_value(self, fresh_db):
        store = ApwConfigStore(fresh_db)
        store.set("accumulation_score_min", 60.0)
        cfg = store.load()
        assert cfg.accumulation_score_min == 60.0

    def test_set_overrides_persist_across_loads(self, fresh_db):
        store = ApwConfigStore(fresh_db)
        store.set("max_llm_candidates", 50)
        store2 = ApwConfigStore(fresh_db)
        assert store2.load().max_llm_candidates == 50

    def test_set_unknown_key_raises(self, fresh_db):
        with pytest.raises(ValueError):
            ApwConfigStore(fresh_db).set("not_a_real_key", 1)

    def test_to_dict_and_from_dict_roundtrip(self):
        cfg = ApwConfig()
        d = to_dict(cfg)
        cfg2 = from_dict(d)
        assert to_dict(cfg2) == d

    def test_from_dict_ignores_unknown_keys(self):
        cfg = from_dict({"accumulation_score_min": 70.0, "unknown_key": "ignored"})
        assert cfg.accumulation_score_min == 70.0


class TestSettingsCLI:
    def test_settings_show_runs_clean(self, fresh_db, monkeypatch, tmp_path):
        from accumulation_probe_washout.cli import main
        from deeptrade.core import paths

        monkeypatch.setattr(paths, "db_path", lambda: tmp_path / "apw.duckdb")
        rc = main(["settings", "show"])
        assert rc == 0

    def test_settings_set_persists(self, fresh_db, monkeypatch, tmp_path):
        from accumulation_probe_washout.cli import main
        from deeptrade.core import paths

        # Point the CLI at our fresh_db location.
        monkeypatch.setattr(paths, "db_path", lambda: fresh_db._path)
        rc = main(["settings", "set", "max_llm_candidates", "55"])
        assert rc == 0
        # Confirm the row landed
        row = fresh_db.fetchone(
            "SELECT value_json FROM apw_config WHERE key = 'max_llm_candidates'"
        )
        assert row is not None
        assert json.loads(row[0]) == 55

    def test_settings_set_unknown_key_rejected(self, monkeypatch, tmp_path):
        from accumulation_probe_washout.cli import main
        from deeptrade.core import paths

        monkeypatch.setattr(paths, "db_path", lambda: tmp_path / "apw.duckdb")
        rc = main(["settings", "set", "not_a_real_key", "1"])
        assert rc == 2


class TestAllowedKeys:
    def test_allowed_keys_matches_dataclass(self):
        # Every field in ApwConfig must be in ALLOWED_KEYS exactly.
        from dataclasses import fields
        expected = {f.name for f in fields(ApwConfig)}
        assert set(ALLOWED_KEYS) == expected
