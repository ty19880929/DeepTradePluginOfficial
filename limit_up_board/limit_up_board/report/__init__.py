"""JSON 报告生成子包（v0.12+）。

模块组织：
* :mod:`limit_up_board.report.schema`  — pydantic 模型镜像 ``StrategyReportSchema``
* :mod:`limit_up_board.report.builder` — ``build_strategy_report`` 把 runner 结果装配成 schema
* :mod:`limit_up_board.report.i18n`    — high/medium/low ↔ 强/中/弱 ↔ 高/中/低 映射常量

向上层暴露最小入口：
"""

from __future__ import annotations

from .builder import build_strategy_report
from .schema import StrategyReportSchema

__all__ = ["StrategyReportSchema", "build_strategy_report"]
