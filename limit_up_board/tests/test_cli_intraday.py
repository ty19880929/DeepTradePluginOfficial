"""CLI tests for intraday-mode behaviors (P0-3, v0.6.4).

Covers:
* ``--allow-intraday`` auto-disables LGB (lgb_enabled flips to False);
  ``intraday_lgb_auto_disabled=True`` flows down to RunParams.
* ``--force-lgb`` reverses the auto-disable while keeping ``allow_intraday``.
* Without ``--allow-intraday`` the existing semantics remain (LGB on by default).
* ``render_banners(is_intraday=True)`` mentions the unstable-fields list.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from limit_up_board.cli import app
from limit_up_board.render import render_banners
from limit_up_board.runner import RunParams, RunStatus, RunOutcome


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_mock_outcome() -> RunOutcome:
    """Cheap stand-in so CLI's `cmd_run` can finish without a real pipeline."""
    return RunOutcome(
        run_id="11111111-1111-1111-1111-111111111111",
        status=RunStatus.SUCCESS,
        error=None,
        seen_events=[],
    )


class TestIntradayAutoDisablesLgb:
    """P0-3: ``--allow-intraday`` 时 RunParams.lgb_enabled 必须为 False，
    且 ``intraday_lgb_auto_disabled=True``。"""

    def _invoke_cmd_run_with_captured_params(
        self,
        runner: CliRunner,
        args: list[str],
    ) -> RunParams:
        """Run cmd_run, capturing the RunParams the runner was given."""
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
        assert result.exit_code == 0, result.output
        return captured["params"]

    def test_intraday_auto_disables_lgb(self, runner: CliRunner) -> None:
        params = self._invoke_cmd_run_with_captured_params(
            runner, ["--allow-intraday"]
        )
        assert params.allow_intraday is True
        assert params.lgb_enabled is False
        assert params.intraday_lgb_auto_disabled is True

    def test_force_lgb_reverses_auto_disable(self, runner: CliRunner) -> None:
        params = self._invoke_cmd_run_with_captured_params(
            runner, ["--allow-intraday", "--force-lgb"]
        )
        assert params.allow_intraday is True
        assert params.lgb_enabled is True
        assert params.intraday_lgb_auto_disabled is False

    def test_no_intraday_keeps_lgb_on(self, runner: CliRunner) -> None:
        params = self._invoke_cmd_run_with_captured_params(runner, [])
        assert params.allow_intraday is False
        assert params.lgb_enabled is True
        assert params.intraday_lgb_auto_disabled is False

    def test_no_lgb_wins_over_force_lgb(self, runner: CliRunner) -> None:
        """``--no-lgb`` 显式禁用 LGB 时，``--force-lgb`` 不应反向覆盖。"""
        params = self._invoke_cmd_run_with_captured_params(
            runner, ["--allow-intraday", "--no-lgb", "--force-lgb"]
        )
        assert params.lgb_enabled is False


class TestIntradayBannerListsUnstableFields:
    """P0-3 render: INTRADAY MODE 横幅下应列出时点不稳定字段。"""

    def test_intraday_banner_lists_unstable_fields(self) -> None:
        banner = render_banners(status=RunStatus.SUCCESS, is_intraday=True)
        assert "INTRADAY MODE" in banner
        # 关键字段都应被点名，方便用户对照
        for fragment in (
            "daily.amount",
            "daily_basic.turnover_rate",
            "moneyflow.*",
            "top_list",
            "cyq_perf",
            "limit_step",
            "--force-lgb",
        ):
            assert fragment in banner, f"missing fragment: {fragment!r}"

    def test_non_intraday_banner_omits_unstable_block(self) -> None:
        banner = render_banners(status=RunStatus.SUCCESS, is_intraday=False)
        assert "INTRADAY MODE" not in banner
        assert "daily.amount" not in banner
