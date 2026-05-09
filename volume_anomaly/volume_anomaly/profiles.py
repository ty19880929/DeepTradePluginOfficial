"""LLM stage profile presets for volume-anomaly.

v0.7 — stage 调参档归插件维护。本插件单 stage（``trend_analysis``），名称
自然回归语义对应位（之前因为框架硬编码 stage 表，借用了 limit-up-board 的
``continuation_prediction`` 名）。

Preset 语义沿用 v0.6 ``PROFILES_DEFAULT`` 中 R2 阶段对应档（balanced/quality
开 thinking，fast 关）。
"""

from __future__ import annotations

from deeptrade.plugins_api import StageProfile

STAGE_TREND_ANALYSIS = "trend_analysis"


PROFILES: dict[str, dict[str, StageProfile]] = {
    "fast": {
        STAGE_TREND_ANALYSIS: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.2, max_output_tokens=32768
        ),
    },
    "balanced": {
        STAGE_TREND_ANALYSIS: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
    },
    "quality": {
        STAGE_TREND_ANALYSIS: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
    },
}


def resolve_profile(preset: str, stage: str) -> StageProfile:
    """Look up the StageProfile for ``preset × stage``."""
    if preset not in PROFILES:
        raise KeyError(
            f"unknown profile preset {preset!r}; expected one of {sorted(PROFILES)}"
        )
    table = PROFILES[preset]
    if stage not in table:
        raise KeyError(
            f"unknown stage {stage!r} for preset {preset!r}; "
            f"expected one of {sorted(table)}"
        )
    return table[stage]
