"""LimitUpBoardPlugin — Plugin Protocol entry for the打板策略.

Satisfies the framework's minimal :class:`deeptrade.plugins_api.Plugin`
contract: ``metadata`` + ``validate_static`` + ``dispatch``.

Everything else (run lifecycle, history, report, tushare/llm wiring) lives
inside the plugin: see ``cli.py`` and ``runner.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.base import PluginContext


class LimitUpBoardPlugin:
    """Framework entry class for the limit-up-board plugin."""

    metadata = None  # injected by framework after install

    def validate_static(self, ctx: PluginContext) -> None:  # noqa: ARG002
        # No network. Light import-only sanity check.
        # v0.12.3+：避免在 install 阶段把 cli / runner / runtime / lgb / winrate
        # 等重型依赖（typer / rich / questionary / lightgbm）拉进 import 链。
        # 仅 import 两个轻量纯数据模块以触发 pydantic / dataclass 字段校验，
        # 任何字段名 / 默认值的语法错误都会在此抛出。
        # 第三方运行时依赖（pandas / lightgbm / scikit-learn / tushare）由
        # `deeptrade_plugin.yaml::dependencies` 在 install 阶段保证已装。
        from . import config as _config  # noqa: F401, PLC0415
        from . import schemas  # noqa: F401, PLC0415

    def dispatch(self, argv: list[str]) -> int:
        from . import cli  # noqa: PLC0415 — 运行时再加载完整 CLI 依赖
        return cli.main(argv)
