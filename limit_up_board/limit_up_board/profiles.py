"""LLM stage profile presets for limit-up-board.

v0.7 — stage 调参档归插件维护。preset 名（``fast/balanced/quality``）仍是
框架级用户配置（``app.profile``），但每个 preset 在本插件三个 stage 上的
具体 tuning 由本模块持有。

Preset → stage tuning 表沿用 v0.6 ``PROFILES_DEFAULT`` 的语义：
  * fast     — 全程关 thinking，低成本
  * balanced — 强势初筛关 thinking，连板预测 / 全局重排开
  * quality  — 全程开 thinking
"""

from __future__ import annotations

from deeptrade.plugins_api import StageProfile

# Stage tag strings — values are persisted into lub_stage_results / lub_runs
# rows and into LLM audit JSON, so they MUST NOT be renamed without a DB
# migration. The Python symbol names (STAGE_SCREENING / STAGE_PREDICTION /
# STAGE_REVISION) are the user-visible refactor; the string values stay.
STAGE_SCREENING = "strong_target_analysis"
STAGE_PREDICTION = "continuation_prediction"
STAGE_FINAL = "final_ranking"
STAGE_REVISION = "continuation_revision"


PROFILES: dict[str, dict[str, StageProfile]] = {
    "fast": {
        STAGE_SCREENING: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.1, max_output_tokens=32768
        ),
        STAGE_PREDICTION: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_REVISION: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.2, max_output_tokens=32768
        ),
    },
    "balanced": {
        STAGE_SCREENING: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.1, max_output_tokens=32768
        ),
        STAGE_PREDICTION: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_REVISION: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
    },
    "quality": {
        STAGE_SCREENING: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_PREDICTION: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.2, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_REVISION: StageProfile(
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
