"""P3-B / P3-A: CLI --fresh-llm / --no-llm-replay / --replay-only flag plumbing.

Mirrors :mod:`tests.test_cli_llm_select` but for the replay-cache triple.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from limit_up_board.cli import app
from limit_up_board.runner import RunOutcome, RunParams, RunStatus


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_mock_outcome() -> RunOutcome:
    return RunOutcome(
        run_id="22222222-2222-2222-2222-222222222222",
        status=RunStatus.SUCCESS,
        error=None,
        seen_events=[],
    )


def _invoke_cmd_run(
    runner: CliRunner, args: list[str]
) -> tuple[int, str, RunParams | None]:
    """Run cmd_run with a fake LubRunner; return (exit_code, output, captured params)."""
    captured: dict[str, RunParams] = {}

    class _FakeRunner:
        def __init__(self, rt, renderer=None, ctx=None) -> None:  # noqa: ARG002
            pass

        def execute(self, params: RunParams) -> RunOutcome:
            captured["params"] = params
            return _make_mock_outcome()

    with (
        patch("limit_up_board.cli._open_runtime") as open_rt,
        patch("limit_up_board.cli.LubRunner", _FakeRunner),
        patch("limit_up_board.cli.render_finished_run") as _render,
        patch("limit_up_board.cli.choose_renderer", lambda **_: MagicMock()),
    ):
        open_rt.return_value = (MagicMock(close=MagicMock()), MagicMock(), MagicMock())
        _render.return_value = None
        result = runner.invoke(app, ["run", *args])
    return result.exit_code, result.output or "", captured.get("params")


class TestReplayFlagFlow:
    """Each flag flows into the matching RunParams field."""

    def test_no_flag_defaults(self, runner: CliRunner) -> None:
        exit_code, _, params = _invoke_cmd_run(runner, [])
        assert exit_code == 0
        assert params is not None
        assert params.fresh_llm is False
        assert params.no_llm_replay is False
        assert params.replay_only is False

    def test_fresh_llm_sets_field(self, runner: CliRunner) -> None:
        exit_code, _, params = _invoke_cmd_run(runner, ["--fresh-llm"])
        assert exit_code == 0
        assert params is not None
        assert params.fresh_llm is True
        assert params.no_llm_replay is False
        assert params.replay_only is False

    def test_no_llm_replay_sets_field(self, runner: CliRunner) -> None:
        exit_code, _, params = _invoke_cmd_run(runner, ["--no-llm-replay"])
        assert exit_code == 0
        assert params is not None
        assert params.no_llm_replay is True

    def test_replay_only_sets_field(self, runner: CliRunner) -> None:
        """--replay-only is allowed at the parse layer; execute() does the
        framework-support precondition check (covered by runner tests).
        Here we just verify the flag reaches RunParams."""
        exit_code, _, params = _invoke_cmd_run(runner, ["--replay-only"])
        # exit code may be non-zero if execute() rejects via PreconditionError
        # on pre-Phase-2 framework; but params should still be captured BEFORE
        # _FakeRunner.execute returns (it's set as the first call).
        # If FakeRunner short-circuits, params is captured. Real runner exit
        # path is exercised in test_runner.py.
        assert params is not None
        assert params.replay_only is True


class TestReplayFlagMutex:
    """At most one of the three flags may be set."""

    def test_fresh_and_no_llm_replay_are_mutex(self, runner: CliRunner) -> None:
        exit_code, output, params = _invoke_cmd_run(
            runner, ["--fresh-llm", "--no-llm-replay"]
        )
        assert exit_code == 2
        assert params is None
        assert "三者最多只能选一个" in output

    def test_fresh_and_replay_only_are_mutex(self, runner: CliRunner) -> None:
        exit_code, output, params = _invoke_cmd_run(
            runner, ["--fresh-llm", "--replay-only"]
        )
        assert exit_code == 2
        assert params is None
        assert "三者最多只能选一个" in output

    def test_no_replay_and_replay_only_are_mutex(self, runner: CliRunner) -> None:
        exit_code, output, params = _invoke_cmd_run(
            runner, ["--no-llm-replay", "--replay-only"]
        )
        assert exit_code == 2
        assert params is None

    def test_all_three_at_once_rejected(self, runner: CliRunner) -> None:
        exit_code, output, params = _invoke_cmd_run(
            runner, ["--fresh-llm", "--no-llm-replay", "--replay-only"]
        )
        assert exit_code == 2
        assert params is None
