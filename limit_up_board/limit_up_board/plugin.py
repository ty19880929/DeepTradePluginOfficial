"""LimitUpBoardPlugin — Plugin Protocol entry for the打板策略.

Satisfies the framework's minimal :class:`deeptrade.plugins_api.Plugin`
contract: ``metadata`` + ``validate_static`` + ``dispatch``.

Everything else (run lifecycle, history, report, tushare/llm wiring) lives
inside the plugin: see ``cli.py`` and ``runner.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import cli as _cli

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.base import PluginContext


class LimitUpBoardPlugin:
    """Framework entry class for the limit-up-board plugin."""

    metadata = None  # injected by framework after install

    def validate_static(self, ctx: PluginContext) -> None:  # noqa: ARG002
        # No network. Light import-only sanity check.
        from . import schemas  # noqa: F401, PLC0415

    def dispatch(self, argv: list[str]) -> int:
        return _cli.main(argv)
