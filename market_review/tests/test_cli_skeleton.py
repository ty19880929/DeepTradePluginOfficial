"""PR-1 CLI skeleton smoke tests.

Locks in the exit-code contract between ``MarketReviewPlugin.dispatch`` and
typer/click — specifically the click 8.3+ behavior where ``standalone_mode``
``False`` returns ``exit_code`` as the function return value instead of
raising :class:`typer.Exit`. Without this guard, a future click upgrade can
silently turn every stub's ``raise typer.Exit(code=2)`` into ``exit 0``.

These tests do NOT exercise any subcommand business logic — that lives in
PR-2..6.
"""

from __future__ import annotations

import pytest

from market_review.plugin import MarketReviewPlugin


def test_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = MarketReviewPlugin().dispatch(["--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "市场复盘" in captured.out
    # The 7 subcommands declared in the skeleton must surface in --help.
    for sub in ("run", "sync", "history", "report", "settings"):
        assert sub in captured.out, f"--help missing subcommand {sub!r}"


@pytest.mark.parametrize(
    "argv",
    [["run"], ["sync"], ["history"], ["report", "00000000-0000-0000-0000-000000000000"]],
)
def test_unimplemented_subcommands_exit_two(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    rc = MarketReviewPlugin().dispatch(argv)
    out = capsys.readouterr().out
    assert rc == 2, f"argv={argv!r} expected exit=2, got {rc}"
    assert "尚未实现" in out


def test_settings_callback_stubs_when_no_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = MarketReviewPlugin().dispatch(["settings"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "尚未实现" in out


def test_no_args_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``no_args_is_help=True`` on the root app prints --help and exits 0/2.

    typer/click prints help to stderr when triggered via ``no_args_is_help``
    and exits with the usage error code (2) on click 8+. We assert the help
    text appears in *either* stream and the exit code is one of the documented
    values; both are stable behaviors users can rely on.
    """
    rc = MarketReviewPlugin().dispatch([])
    cap = capsys.readouterr()
    assert rc in (0, 2)
    combined = cap.out + cap.err
    assert "Usage" in combined or "市场复盘" in combined
