"""P3-D: ``_complete_with_set_check`` stamps attempt audit fields on meta.

Verifies that the returned meta dict gains:
  * ``attempt_count``       — 1 on first-try success, 2 after one retry
  * ``first_error_class``   — None / "set_mismatch" / "evidence_validation"
  * ``repair_hint_hash``    — sha256 of the corrective hint, None if no retry
  * ``final_prompt_hash``   — sha256 of the user prompt that succeeded
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from deeptrade.plugins_api import StageProfile

from limit_up_board.fingerprint import hash_text
from limit_up_board.pipeline import (
    _SetMismatchError,
    _complete_with_set_check,
)


class _StubItem(BaseModel):
    candidate_id: str


class _StubResponse(BaseModel):
    candidates: list[_StubItem]


_DUMMY_PROFILE = StageProfile(
    thinking=False, reasoning_effort="medium", temperature=0.0, max_output_tokens=1024
)


def _make_llm(*responses: dict[str, Any]) -> MagicMock:
    """Fake LLMClient.complete_json that returns ``responses`` in order; each
    is wrapped as the (obj_dict, meta) tuple shape."""
    llm = MagicMock()
    counter = {"i": 0}

    def _side_effect(*args, **kwargs):
        idx = counter["i"]
        counter["i"] += 1
        if idx >= len(responses):
            pytest.fail(f"LLM called {idx + 1} times; only {len(responses)} stubbed")
        return responses[idx]

    llm.complete_json.side_effect = _side_effect
    return llm


def test_no_retry_attempt_count_is_1_first_error_class_none() -> None:
    """First-try success → attempt_count=1, first_error_class=None,
    repair_hint_hash=None."""
    expected = {"a", "b"}
    raw = _StubResponse(candidates=[_StubItem(candidate_id="a"), _StubItem(candidate_id="b")])
    llm = _make_llm((raw, {"tokens_total": 1234}))

    obj, meta = _complete_with_set_check(
        llm,
        system="sys",
        user="usr",
        schema=_StubResponse,
        profile=_DUMMY_PROFILE,
        expected_ids=expected,
    )

    assert meta["attempt_count"] == 1
    assert meta["first_error_class"] is None
    assert meta["repair_hint_hash"] is None
    assert meta["final_prompt_hash"] == hash_text("usr")
    # original meta fields preserved
    assert meta["tokens_total"] == 1234


def test_retry_after_set_mismatch_stamps_attempt_meta() -> None:
    """First attempt returns wrong id set; second attempt fixes it.

    attempt_count=2, first_error_class="set_mismatch",
    repair_hint_hash matches the appended hint, final_prompt_hash matches
    the corrected user prompt.
    """
    expected = {"a", "b"}
    bad = _StubResponse(candidates=[_StubItem(candidate_id="a"), _StubItem(candidate_id="c")])
    good = _StubResponse(candidates=[_StubItem(candidate_id="a"), _StubItem(candidate_id="b")])
    llm = _make_llm((bad, {"tokens_total": 100}), (good, {"tokens_total": 200}))

    obj, meta = _complete_with_set_check(
        llm,
        system="sys",
        user="usr",
        schema=_StubResponse,
        profile=_DUMMY_PROFILE,
        expected_ids=expected,
    )

    assert meta["attempt_count"] == 2
    assert meta["first_error_class"] == "set_mismatch"
    assert meta["repair_hint_hash"] is not None
    assert len(meta["repair_hint_hash"]) == 64  # sha256 hex
    # final_prompt_hash should match user + repair hint (not raw user)
    assert meta["final_prompt_hash"] != hash_text("usr")
    # second-call's meta passes through
    assert meta["tokens_total"] == 200


def test_unrecoverable_set_mismatch_raises() -> None:
    """Two consecutive mismatches → _SetMismatchError; no meta returned."""
    expected = {"a", "b"}
    bad = _StubResponse(candidates=[_StubItem(candidate_id="c")])
    llm = _make_llm((bad, {}), (bad, {}))

    with pytest.raises(_SetMismatchError):
        _complete_with_set_check(
            llm,
            system="sys",
            user="usr",
            schema=_StubResponse,
            profile=_DUMMY_PROFILE,
            expected_ids=expected,
        )


def test_no_retry_when_stage_kwarg_omitted_skips_replay_forwarding() -> None:
    """When caller doesn't pass ``stage=``, _complete_with_set_check must not
    forward replay/stage/schema_version/input_fingerprint to complete_json
    (pre-Phase-3 callers continue to work unchanged).
    """
    raw = _StubResponse(candidates=[_StubItem(candidate_id="a")])
    llm = _make_llm((raw, {}))

    _complete_with_set_check(
        llm,
        system="sys",
        user="usr",
        schema=_StubResponse,
        profile=_DUMMY_PROFILE,
        expected_ids={"a"},
        # stage NOT passed
    )
    call = llm.complete_json.call_args
    kwargs = call.kwargs
    assert "replay" not in kwargs
    assert "stage" not in kwargs
    assert "schema_version" not in kwargs
    assert "input_fingerprint" not in kwargs


def test_stage_kwarg_does_not_forward_when_framework_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller passes stage=..., but framework doesn't accept replay kwargs.
    _complete_with_set_check must still NOT forward (avoids TypeError)."""
    raw = _StubResponse(candidates=[_StubItem(candidate_id="a")])
    llm = _make_llm((raw, {}))

    monkeypatch.setattr(
        "limit_up_board.pipeline.complete_json_supports_replay", lambda: False
    )

    _complete_with_set_check(
        llm,
        system="sys",
        user="usr",
        schema=_StubResponse,
        profile=_DUMMY_PROFILE,
        expected_ids={"a"},
        stage="strong_target_analysis",
    )
    kwargs = llm.complete_json.call_args.kwargs
    assert "replay" not in kwargs


def test_stage_kwarg_forwards_when_framework_supports_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When framework supports replay, the four extra kwargs are forwarded
    to complete_json with values from ContextVars + LLM_SCHEMA_VERSION."""
    from limit_up_board.profiles import LLM_SCHEMA_VERSION
    from limit_up_board.replay_policy import (
        LLMReplayPolicy,
        apply_replay_context,
    )

    raw = _StubResponse(candidates=[_StubItem(candidate_id="a")])
    llm = _make_llm((raw, {}))

    monkeypatch.setattr(
        "limit_up_board.pipeline.complete_json_supports_replay", lambda: True
    )

    policy = LLMReplayPolicy(read_enabled=True, write_enabled=True)
    fp = "abc123" * 10
    with apply_replay_context(
        policy, stage_to_fingerprint={"strong_target_analysis": fp}
    ):
        _complete_with_set_check(
            llm,
            system="sys",
            user="usr",
            schema=_StubResponse,
            profile=_DUMMY_PROFILE,
            expected_ids={"a"},
            stage="strong_target_analysis",
        )

    kwargs = llm.complete_json.call_args.kwargs
    assert kwargs["replay"] is policy
    assert kwargs["stage"] == "strong_target_analysis"
    assert kwargs["schema_version"] == LLM_SCHEMA_VERSION
    assert kwargs["input_fingerprint"] == fp
