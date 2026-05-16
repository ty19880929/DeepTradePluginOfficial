"""CLI tests for --llm provider override (v0.6.8).

Covers the non-debate-mode ``--llm <provider>`` flag:

* Value flows into ``RunParams.llm_provider``.
* Default (no flag) keeps ``llm_provider=None`` (framework default path).
* Mutually exclusive with ``--debate`` / ``--debate-llms``.
* Empty value rejected by the CLI parser.
* ``LubRunner._validate_single_provider`` raises ``PreconditionError`` when
  the requested provider isn't in ``LLMManager.list_providers()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from limit_up_board.cli import app
from limit_up_board.runner import LubRunner, PreconditionError, RunOutcome, RunParams, RunStatus
from limit_up_board.runtime import LubRuntime, pick_llm_provider


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_mock_outcome() -> RunOutcome:
    return RunOutcome(
        run_id="11111111-1111-1111-1111-111111111111",
        status=RunStatus.SUCCESS,
        error=None,
        seen_events=[],
    )


def _invoke_cmd_run_with_captured_params(
    runner: CliRunner, args: list[str]
) -> tuple[int, str, RunParams | None]:
    """Run cmd_run with a fake LubRunner; return (exit_code, output, params).

    ``params`` is None when the CLI rejected the command before constructing
    the runner (e.g. mutex / parse failure).
    """
    captured: dict[str, RunParams] = {}

    class _FakeRunner:
        def __init__(self, rt, renderer=None) -> None:  # noqa: ARG002
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
        open_rt.return_value = (MagicMock(close=MagicMock()), MagicMock())
        _render.return_value = None
        result = runner.invoke(app, ["run", *args])
    return result.exit_code, result.output or "", captured.get("params")


class TestLlmFlagFlow:
    """Single-LLM mode: --llm value reaches RunParams.llm_provider."""

    def test_llm_flag_sets_provider(self, runner: CliRunner) -> None:
        exit_code, _, params = _invoke_cmd_run_with_captured_params(
            runner, ["--llm", "deepseek"]
        )
        assert exit_code == 0
        assert params is not None
        assert params.llm_provider == "deepseek"
        assert params.debate is False

    def test_no_llm_flag_defaults_to_none(self, runner: CliRunner) -> None:
        exit_code, _, params = _invoke_cmd_run_with_captured_params(runner, [])
        assert exit_code == 0
        assert params is not None
        assert params.llm_provider is None

    def test_llm_flag_value_is_stripped(self, runner: CliRunner) -> None:
        exit_code, _, params = _invoke_cmd_run_with_captured_params(
            runner, ["--llm", "  kimi  "]
        )
        assert exit_code == 0
        assert params is not None
        assert params.llm_provider == "kimi"


class TestLlmDebateMutex:
    """--llm is rejected under --debate (and vice versa)."""

    def test_llm_and_debate_are_mutex(self, runner: CliRunner) -> None:
        exit_code, output, params = _invoke_cmd_run_with_captured_params(
            runner, ["--llm", "deepseek", "--debate"]
        )
        assert exit_code == 2
        # FakeRunner.execute should never have been reached.
        assert params is None
        assert "仅适用于非辩论模式" in output

    def test_llm_and_debate_llms_are_mutex(self, runner: CliRunner) -> None:
        exit_code, output, params = _invoke_cmd_run_with_captured_params(
            runner, ["--llm", "deepseek", "--debate", "--debate-llms", "kimi,qwen"]
        )
        assert exit_code == 2
        assert params is None
        assert "仅适用于非辩论模式" in output

    def test_empty_llm_value_rejected(self, runner: CliRunner) -> None:
        exit_code, output, params = _invoke_cmd_run_with_captured_params(
            runner, ["--llm", "   "]
        )
        assert exit_code == 2
        assert params is None
        assert "--llm 解析后为空" in output


class TestValidateSingleProvider:
    """Unit tests for LubRunner._validate_single_provider (no CLI roundtrip)."""

    def _make_runner(self, available: list[str]) -> LubRunner:
        rt = LubRuntime(
            db=MagicMock(),
            config=MagicMock(),
            llms=MagicMock(),
        )
        rt.llms.list_providers = MagicMock(return_value=available)
        return LubRunner(rt)

    def test_none_returns_none(self) -> None:
        """``llm_provider=None`` defers to framework default (returns None)."""
        runner = self._make_runner(available=["deepseek", "kimi"])
        result = runner._validate_single_provider(RunParams(llm_provider=None))
        assert result is None
        # list_providers should not be called when we're deferring.
        runner._rt.llms.list_providers.assert_not_called()

    def test_configured_provider_returns_name(self) -> None:
        runner = self._make_runner(available=["deepseek", "kimi"])
        result = runner._validate_single_provider(RunParams(llm_provider="kimi"))
        assert result == "kimi"

    def test_unconfigured_provider_raises_precondition(self) -> None:
        runner = self._make_runner(available=["deepseek"])
        with pytest.raises(PreconditionError) as exc:
            runner._validate_single_provider(RunParams(llm_provider="kimi"))
        msg = str(exc.value)
        assert "'kimi'" in msg
        assert "未配置或缺 api_key" in msg
        # Should surface the available list so users can self-correct.
        assert "deepseek" in msg

    def test_no_providers_configured_raises_precondition(self) -> None:
        runner = self._make_runner(available=[])
        with pytest.raises(PreconditionError):
            runner._validate_single_provider(RunParams(llm_provider="deepseek"))


class TestPickLlmProvider:
    """``pick_llm_provider(rt, override)`` is a pure pass-through after v0.6.8."""

    def test_override_none_returns_none(self) -> None:
        rt = MagicMock()
        assert pick_llm_provider(rt, None) is None

    def test_override_value_returns_verbatim(self) -> None:
        rt = MagicMock()
        assert pick_llm_provider(rt, "deepseek") == "deepseek"

    def test_default_arg_is_none(self) -> None:
        """No override = None (backward compat with v0.6.7- callers)."""
        rt = MagicMock()
        assert pick_llm_provider(rt) is None
