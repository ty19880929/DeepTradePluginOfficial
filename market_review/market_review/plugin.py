"""MarketReviewPlugin — Plugin Protocol entry for the市场复盘 strategy.

Satisfies the framework's minimal :class:`deeptrade.plugins_api.Plugin`
contract: ``metadata`` + ``validate_static`` + ``dispatch``.

Everything else (run lifecycle, history, report, tushare/llm wiring) lives
inside the plugin: see ``cli.py`` (and later ``runner.py`` / ``runtime.py``).

PR-1 scope (v0.1.0 skeleton)
----------------------------
This module only loads light-weight modules at install / validate time:

- :mod:`market_review.config` — :class:`MrConfig` dataclass (syntax check).
- :mod:`market_review.schemas` — schema-version constant + Literal aliases.

Heavy runtime dependencies (typer / rich / pandas / tushare / pydantic
section models) must NOT be imported here; they belong in :mod:`cli`
(loaded lazily by :meth:`dispatch`) per design §2.2 + lub v0.12.3 contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.base import PluginContext


class MarketReviewPlugin:
    """Framework entry class for the market-review plugin."""

    metadata = None  # injected by framework after install

    def validate_static(self, ctx: PluginContext) -> None:  # noqa: ARG002
        # No network. Light import-only sanity check.
        # Mirrors lub v0.12.3+ contract: keep validate_static cheap so the
        # framework's install-time hook never pulls typer / rich / pandas /
        # tushare into sys.modules. See tests/test_plugin_validate_static.py
        # for the regression guard.
        from . import config as _config  # noqa: F401, PLC0415
        from . import schemas as _schemas  # noqa: F401, PLC0415

    def dispatch(self, argv: list[str]) -> int:
        from . import cli  # noqa: PLC0415 — 运行时再加载完整 CLI 依赖

        return cli.main(argv)
