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

# P1-J: 复现性版本号 —— 用于框架 replay cache key 与 input fingerprint。
#
# LLM_SCHEMA_VERSION: 当 schemas.py 里的 Pydantic 模型字段、empty-array policy
# 或 set-check 行为发生变化时 bump。旧 cache 会自然 miss，避免反序列化失败。
#
# PROMPT_TEMPLATE_VERSION: 当 prompts.py 的模板文本（含 system / user 段、JSON
# 示例、注释规则、empty-array 指示）发生语义变化时 bump。Cosmetic 文案修改
# 不必 bump；任何会改变 LLM 输入语义的改动都必须 bump。
LLM_SCHEMA_VERSION = "lub-llm-schema-v1"
PROMPT_TEMPLATE_VERSION = "lub-prompts-v1"


# P1-L: All stage temperatures default to 0.0.
#
# Previously screening / prediction / revision used 0.1~0.2 to encourage
# diverse reasoning. In practice this leaked LLM sampling noise into the
# Prompt-output pipeline and broke "same inputs → same outputs" — even with
# identical prompt_hash, providers returned slightly different scores /
# ordering / rationales across reruns.
#
# Temperature alone does not guarantee determinism (provider-side sampling,
# load balancing, and quiet model upgrades can still drift), so this is
# necessary but not sufficient — the framework LLM replay cache (Phase 3)
# is the durable fix. Lowering temp to 0.0 narrows the noise floor that
# replay then makes deterministic.
PROFILES: dict[str, dict[str, StageProfile]] = {
    "fast": {
        STAGE_SCREENING: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=32768
        ),
        STAGE_PREDICTION: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_REVISION: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=32768
        ),
    },
    "balanced": {
        STAGE_SCREENING: StageProfile(
            thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=32768
        ),
        STAGE_PREDICTION: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_REVISION: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=32768
        ),
    },
    "quality": {
        STAGE_SCREENING: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=32768
        ),
        STAGE_PREDICTION: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=32768
        ),
        STAGE_FINAL: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=8192
        ),
        STAGE_REVISION: StageProfile(
            thinking=True, reasoning_effort="high", temperature=0.0, max_output_tokens=32768
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
