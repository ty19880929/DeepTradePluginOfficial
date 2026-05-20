"""Iter-0 PR-0.1 smoke test: package + plugin entry are importable."""

from __future__ import annotations


def test_plugin_module_imports() -> None:
    from checkmate import plugin  # noqa: PLC0415

    assert plugin.CheckmatePlugin.metadata is None


def test_validate_static_runs_without_context() -> None:
    from checkmate.plugin import CheckmatePlugin  # noqa: PLC0415

    # ctx is unused in the Iter-0 stub; pass None to mirror the install-time
    # contract (framework injects a PluginContext at runtime).
    CheckmatePlugin().validate_static(None)  # type: ignore[arg-type]


def test_dispatch_forwards_to_cli_main() -> None:
    from checkmate.plugin import CheckmatePlugin  # noqa: PLC0415

    # Invoke with --help: typer/click prints help and exits with 0 even under
    # standalone_mode=False (Click raises a SystemExit(0) which cli.main maps
    # to rc=0). The point is that dispatch() returns an int and never raises.
    rc = CheckmatePlugin().dispatch(["--help"])
    assert isinstance(rc, int)
    assert rc == 0
