"""ui — EventRenderer 双实现 + 选择逻辑（设计 §11，仿 limit_up_board.ui）.

降级条件（任一命中即 legacy）：``--no-dashboard``、stdout 非 TTY（管道/重定向/
pytest capture）、``CI`` 真值、``DEEPTRADE_NO_DASHBOARD`` 真值、``TERM=dumb``。
``NO_COLOR=1`` 保留 dashboard 但去色（rich 自行处理）。
"""

from __future__ import annotations

import os
import sys

from .legacy import LegacyStreamRenderer
from .protocol import EventRenderer, RunMeta

__all__ = ["EventRenderer", "RunMeta", "LegacyStreamRenderer", "choose_renderer"]

_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def choose_renderer(*, no_dashboard: bool = False) -> EventRenderer:
    if (
        no_dashboard
        or not sys.stdout.isatty()
        or _env_truthy("CI")
        or _env_truthy("DEEPTRADE_NO_DASHBOARD")
        or os.environ.get("TERM", "").lower() == "dumb"
    ):
        return LegacyStreamRenderer()
    from .dashboard import RichDashboardRenderer  # noqa: PLC0415 — 延迟重依赖

    return RichDashboardRenderer()
