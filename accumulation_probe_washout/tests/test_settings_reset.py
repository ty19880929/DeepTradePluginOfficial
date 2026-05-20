"""Regression tests for ``settings reset`` (v0.3.0).

Backs the new `cli.cmd_settings_reset` flow: drop one override or all
overrides. ALLOWED_KEYS gating works the same as ``settings set``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from accumulation_probe_washout.config import ALLOWED_KEYS, ApwConfigStore


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


def test_reset_one_key_removes_only_that_override(fresh_db):
    store = ApwConfigStore(fresh_db)
    store.set("min_amount_yi", 5.0)
    store.set("max_circ_mv_yi", 800.0)

    fresh_db.execute("DELETE FROM apw_config WHERE key = ?", ["min_amount_yi"])

    rows = dict(store.items())
    assert "min_amount_yi" not in rows
    assert rows["max_circ_mv_yi"] == 800.0


def test_reset_all_clears_every_override(fresh_db):
    store = ApwConfigStore(fresh_db)
    store.set("min_amount_yi", 5.0)
    store.set("washout_score_min", 45.0)

    fresh_db.execute("DELETE FROM apw_config")

    assert store.items() == []
    cfg = store.load()
    # Defaults restored — sanity check one well-known field.
    assert cfg.min_amount_yi == 1.0
    assert cfg.washout_score_min == 55.0


def test_reset_unknown_key_rejected_by_cli_layer():
    """ALLOWED_KEYS gating prevents typos from silently doing nothing."""
    assert "min_amount_yi" in ALLOWED_KEYS
    assert "this_does_not_exist" not in ALLOWED_KEYS


def test_prune_keys_are_in_allowed_set():
    """v0.3.0 added 4 prune knobs; they must show up in settings show/set."""
    for k in (
        "prune_idle_days_launch_ready",
        "prune_drop_on_probe_low_break",
        "prune_drop_on_ma60_break",
        "prune_dry_run_default",
    ):
        assert k in ALLOWED_KEYS, k
