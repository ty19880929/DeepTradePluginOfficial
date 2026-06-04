"""VwapReversionPlugin — Plugin Protocol entry for the VWAP 带回归日内策略.

Satisfies the framework's minimal :class:`deeptrade.plugins_api.Plugin`
contract: ``metadata`` + ``validate_static`` + ``dispatch``.

Everything else (daemon lifecycle, engine, paper execution, reporting) lives
inside the plugin: see ``cli.py`` / ``daemon.py`` (P1+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.base import PluginContext


class VwapReversionPlugin:
    """Framework entry class for the vwap-reversion plugin."""

    metadata = None  # injected by framework after install

    def validate_static(self, ctx: PluginContext) -> None:  # noqa: ARG002
        # No network. Light import-only sanity check — touch the two pure
        # modules (no typer / rich / pandas import chain) so any field-name /
        # default-value error surfaces at install time. Third-party runtime
        # deps are guaranteed by yaml::dependencies before this runs.
        from . import clock  # noqa: F401, PLC0415
        from . import config as _config  # noqa: F401, PLC0415
        from . import schemas  # noqa: F401, PLC0415

    def dispatch(self, argv: list[str]) -> int:
        from . import cli  # noqa: PLC0415 — defer heavy CLI deps to runtime
        return cli.main(argv)
