"""CLI-level tests for the `lgb` sub-command group.

These tests don't spin up a real LightGBM run — they just exercise the
guards / preflight checks at the typer command entry. Heavier integration
is covered by tests/test_lgb_*.py.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest
from typer.testing import CliRunner

from limit_up_board.cli import app


@pytest.fixture
def runner() -> CliRunner:
    # mix_stderr defaults differ across typer/click versions; let CliRunner
    # surface stderr in result.output for plain `assert ... in result.output`.
    return CliRunner()


class TestLgbTrainPyarrowPreflight:
    """P2-6 — `lgb train` 入口必须在缺 pyarrow 时给出友好 UsageError。

    放在 `lgb evaluate / info / list` 等读路径上的硬自检会阻断排查问题，
    所以只对 train 检测。
    """

    def test_lgb_train_without_pyarrow_raises_usage_error(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force `import pyarrow` to raise ImportError, even though the test
        # environment has pyarrow installed. Other imports keep working.
        real_import = builtins.__import__

        def _fake_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if name == "pyarrow":
                raise ImportError("simulated pyarrow missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        result = runner.invoke(
            app, ["lgb", "train", "--start", "20260101", "--end", "20260131"]
        )
        # typer.BadParameter exits with code 2 and prints "Error: ... pyarrow"
        # to the captured output. The exact wording matches the CLI message.
        assert result.exit_code != 0
        assert "pyarrow" in result.output

    def test_lgb_train_help_works_without_runtime(self, runner: CliRunner) -> None:
        """烟雾测：--help 不应触发 pyarrow check（避免影响离线排查）。"""
        result = runner.invoke(app, ["lgb", "train", "--help"])
        assert result.exit_code == 0
        assert "训练新的 LightGBM 模型" in result.output
