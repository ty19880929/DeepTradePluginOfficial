"""Iter-0 PR-0.3 contract: every CLI surface command is registered and emits
an Iter-0 stub message with exit code 2.

When subsequent iters wire up real bodies, the affected test will need to
move to that iter's test file (and assert on real behaviour). The wholesale
"every subcommand returns 2" assertion intentionally lives here so it can
shrink one row at a time.
"""

from __future__ import annotations

import pytest

from checkmate.cli import main

# (argv, label) — every flag-required command gets just enough args to pass
# Click's parameter validation, so the body (which echoes the stub and exits 2)
# actually runs.
_CASES: list[tuple[list[str], str]] = [
    # ``sync`` graduated out of the stub set in Iter-1 PR-1.2; see
    # ``test_sync_smoke.py`` for its real-behaviour tests.
    # ``scan`` (universe-only) graduated in Iter-1 PR-1.3; see
    # ``test_universe.py`` for its real-behaviour tests.
    # ``signals`` / ``explain`` graduated in Iter-3 PR-3.4; see
    # ``test_signals_smoke.py`` for end-to-end coverage.
    # ``backtest`` / ``report`` graduated in Iter-4 PR-4.3.
    (["settings", "show"], "settings show"),
    (["settings", "reset"], "settings reset"),
]


@pytest.mark.parametrize("argv,label", _CASES, ids=[c[1] for c in _CASES])
def test_subcommand_is_iter0_stub(argv: list[str], label: str, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: ARG001
    rc = main(argv)
    assert rc == 2, f"{label!r} should exit 2 in Iter-0, got {rc}"
    captured = capsys.readouterr()
    assert "not yet implemented in Iter-0" in captured.out, (
        f"{label!r} did not print the Iter-0 stub message: {captured.out!r}"
    )


def test_root_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    # Sanity: help text mentions every public subcommand
    for name in ("sync", "scan", "signals", "backtest", "explain", "report", "settings"):
        assert name in out, f"help missing {name!r}"
