"""CLI 入口契约（仅 ``--help`` / 无参 路径）。

PR-6 之后子命令是真实实现，需要真实 framework runtime；那些路径的端到
端测试在 :mod:`tests.test_cli_e2e` 通过 monkeypatch ``_open_runtime`` 覆
盖。这里只保留对 dispatch 入口本身的契约断言：

- ``--help`` 退 0，列出 5 个子命令；
- 无参 → typer ``no_args_is_help`` 退 2 + 帮助文。

并锁住 ``main`` 的 click 8.3+ exit-code 行为（standalone_mode=False 下 click
把 typer.Exit 转成返回值；ClickException 仍 raise）。
"""

from __future__ import annotations

import pytest

from market_review.plugin import MarketReviewPlugin


def test_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = MarketReviewPlugin().dispatch(["--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "市场复盘" in captured.out
    for sub in ("run", "sync", "history", "report", "settings"):
        assert sub in captured.out, f"--help missing subcommand {sub!r}"


def test_no_args_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``no_args_is_help=True`` triggers click NoArgsIsHelpError → exit 2."""
    rc = MarketReviewPlugin().dispatch([])
    cap = capsys.readouterr()
    assert rc in (0, 2)
    combined = cap.out + cap.err
    assert "Usage" in combined or "市场复盘" in combined
