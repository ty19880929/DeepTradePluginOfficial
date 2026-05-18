"""M1 smoke tests: plugin class loads and validate_static doesn't raise."""

from __future__ import annotations


def test_plugin_class_importable() -> None:
    from accumulation_probe_washout.plugin import AccumulationProbeWashoutPlugin

    plugin = AccumulationProbeWashoutPlugin()
    assert plugin.metadata is None  # framework injects later


def test_validate_static_does_not_raise() -> None:
    from accumulation_probe_washout.plugin import AccumulationProbeWashoutPlugin

    plugin = AccumulationProbeWashoutPlugin()
    # ctx is unused in the M1 stub; pass a sentinel.
    plugin.validate_static(ctx=object())


def test_schemas_phase_enum_complete() -> None:
    from accumulation_probe_washout.schemas import APWPhase

    expected = {
        "no_setup",
        "accumulating",
        "probe_seen",
        "washing_after_probe",
        "launch_ready",
    }
    assert {p.value for p in APWPhase} == expected


def test_cli_dispatch_settings_show(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from accumulation_probe_washout.cli import main
    from deeptrade.core import paths
    from deeptrade.core.db import Database

    # Redirect DB and apply the migration so apw_config exists.
    db_file = tmp_path / "apw.duckdb"
    db = Database(db_file)
    sql_path = (
        Path(__file__).resolve().parent.parent
        / "migrations" / "20260601_001_init.sql"
    )
    for stmt in sql_path.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())
    db.close()

    monkeypatch.setattr(paths, "db_path", lambda: db_file)
    rc = main(["settings", "show"])
    assert rc == 0
