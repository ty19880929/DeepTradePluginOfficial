"""中文语义映射常量。

插件 LLM schema 用 ``high/medium/low`` 这种英文枚举做内部传输，但前端
``StrategyReportSchema`` 要求出口直接是中文（强/中/弱、高/中/低），减少前端 i18n 心智。
"""

from __future__ import annotations

from typing import Final, Literal

StrengthLevelEn = Literal["high", "medium", "low"]
ConfidenceLevelEn = Literal["high", "medium", "low"]

STRENGTH_ZH: Final[dict[StrengthLevelEn, str]] = {
    "high": "强",
    "medium": "中",
    "low": "弱",
}

CONFIDENCE_ZH: Final[dict[ConfidenceLevelEn, str]] = {
    "high": "高",
    "medium": "中",
    "low": "低",
}

# 不在映射表中的取值就原样回吐（防御 schema 变更时不至于直接崩）。
def to_strength_zh(level: str) -> str:
    return STRENGTH_ZH.get(level, level)  # type: ignore[arg-type]


def to_confidence_zh(level: str) -> str:
    return CONFIDENCE_ZH.get(level, level)  # type: ignore[arg-type]
