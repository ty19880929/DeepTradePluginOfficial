"""VolumeAnomalyPlugin — Plugin Protocol entry for the成交量异动策略."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import cli as _cli

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.base import PluginContext


class VolumeAnomalyPlugin:
    """Framework entry class for the volume-anomaly plugin."""

    metadata = None  # injected by framework after install

    def validate_static(self, ctx: PluginContext) -> None:  # noqa: ARG002
        from . import schemas  # noqa: F401, PLC0415

    def dispatch(self, argv: list[str]) -> int:
        return _cli.main(argv)
