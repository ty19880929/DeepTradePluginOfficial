"""LLM stage profile presets for limit-up-board.

v0.7 — stage 调参档归插件维护。preset 名（``fast/balanced/quality``）仍是
框架级用户配置（``app.profile``），但每个 preset 在本插件三个 stage 上的
具体 tuning 由本模块持有。

Preset → stage tuning 表沿用 v0.6 ``PROFILES_DEFAULT`` 的语义：
  * fast     — 全程关 thinking，低成本
  * balanced — R1 关 thinking，R2 / final_ranking 开
  * quality  — 全程开 thinking
"""

from __future__ import annotations

from deeptrade.plugins_api import StageProfile

STAGE_R1 = "strong_target_analysis"
STAGE_R2 = "continuation_prediction"
STAGE_FINAL = "final_ranking"
STAGE_R3 = "continuation_revision"


PROFILES: dict[str, dict[str, StageProfile]] = {
    "fast": {
        STAGE_R1: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.1, max_output_tokens=32768
        ),
        STAGE_R2: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_R3: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.2, max_output_tokens=32768
        ),
    },
    "balanced": {
        STAGE_R1: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.1, max_output_tokens=32768
        ),
        STAGE_R2: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_R3: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
    },
    "quality": {
        STAGE_R1: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_R2: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_R3: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
    },
}


def resolve_profile(preset: str, stage: str) -> StageProfile:
    """Look up the StageProfile for ``preset × stage``.

    Raises:
        KeyError — unknown preset or stage name (caller decides how to surface).
    """
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
