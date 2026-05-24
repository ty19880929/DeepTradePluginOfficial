"""P3-PRE / P3-B: ``replay_policy`` adapter — build_replay_policy decision
table, ContextVar plumbing, framework feature detection."""

from __future__ import annotations

import inspect

import pytest

from limit_up_board.replay_policy import (
    LLMReplayPolicy,
    _ReplayCLIFlags,
    apply_replay_context,
    build_replay_policy,
    complete_json_supports_replay,
    get_active_policy,
    get_stage_fingerprint,
)


# ---------------------------------------------------------------------------
# build_replay_policy — decision table
# ---------------------------------------------------------------------------


def _flags(**kw) -> _ReplayCLIFlags:
    return _ReplayCLIFlags(**kw)


def test_build_replay_policy_replay_only_wins_over_config() -> None:
    p = build_replay_policy(
        cli=_flags(replay_only=True),
        cfg_enabled=False,
        cfg_write=True,
        cfg_ttl_days=None,
    )
    assert p.read_enabled is True
    assert p.write_enabled is False
    assert p.replay_only is True


def test_build_replay_policy_no_llm_replay_disables_both() -> None:
    p = build_replay_policy(
        cli=_flags(no_llm_replay=True),
        cfg_enabled=True,   # config tries to enable, CLI overrides
        cfg_write=True,
        cfg_ttl_days=30,
    )
    assert p.read_enabled is False
    assert p.write_enabled is False
    assert p.replay_only is False


def test_build_replay_policy_fresh_llm_skips_read_keeps_write() -> None:
    p = build_replay_policy(
        cli=_flags(fresh_llm=True),
        cfg_enabled=True,
        cfg_write=True,
        cfg_ttl_days=7,
    )
    assert p.read_enabled is False
    assert p.write_enabled is True
    assert p.ttl_days == 7


def test_build_replay_policy_fresh_llm_respects_cfg_write_false() -> None:
    p = build_replay_policy(
        cli=_flags(fresh_llm=True),
        cfg_enabled=False,
        cfg_write=False,
        cfg_ttl_days=None,
    )
    assert p.read_enabled is False
    assert p.write_enabled is False


def test_build_replay_policy_cfg_enabled_gives_default_on() -> None:
    p = build_replay_policy(
        cli=_flags(),
        cfg_enabled=True,
        cfg_write=True,
        cfg_ttl_days=14,
    )
    assert p.read_enabled is True
    assert p.write_enabled is True
    assert p.ttl_days == 14


def test_build_replay_policy_all_off_falls_through_to_disabled() -> None:
    p = build_replay_policy(
        cli=_flags(),
        cfg_enabled=False,
        cfg_write=True,
        cfg_ttl_days=None,
    )
    assert p.read_enabled is False
    assert p.write_enabled is False
    assert p.replay_only is False


# ---------------------------------------------------------------------------
# apply_replay_context — ContextVar plumbing
# ---------------------------------------------------------------------------


def test_apply_replay_context_sets_and_restores_policy() -> None:
    default_before = get_active_policy()
    custom = LLMReplayPolicy(read_enabled=True, write_enabled=True, replay_only=False)
    with apply_replay_context(custom):
        assert get_active_policy() is custom
    assert get_active_policy() is default_before  # restored


def test_apply_replay_context_registers_stage_fingerprints() -> None:
    fp_a = "deadbeef" * 8
    fp_b = "cafe" * 16
    with apply_replay_context(
        LLMReplayPolicy(),
        stage_to_fingerprint={"strong_target_analysis": fp_a, "continuation_prediction": fp_b},
    ):
        assert get_stage_fingerprint("strong_target_analysis") == fp_a
        assert get_stage_fingerprint("continuation_prediction") == fp_b
        assert get_stage_fingerprint("unknown_stage") is None


def test_apply_replay_context_nesting_restores_outer() -> None:
    outer = LLMReplayPolicy(read_enabled=True)
    inner = LLMReplayPolicy(replay_only=True)
    with apply_replay_context(outer):
        with apply_replay_context(inner):
            assert get_active_policy() is inner
        assert get_active_policy() is outer


# ---------------------------------------------------------------------------
# Framework feature detection
# ---------------------------------------------------------------------------


def test_complete_json_supports_replay_returns_bool() -> None:
    """Smoke: returns a bool. True/False depends on installed framework."""
    out = complete_json_supports_replay()
    assert isinstance(out, bool)


def test_complete_json_supports_replay_matches_actual_signature() -> None:
    """Detector agrees with direct inspect.signature inspection (single source of truth)."""
    try:
        from deeptrade.core.llm_client import LLMClient
    except ImportError:
        # No framework installed → detector should report False.
        assert complete_json_supports_replay() is False
        return
    params = inspect.signature(LLMClient.complete_json).parameters
    assert complete_json_supports_replay() is ("replay" in params)
