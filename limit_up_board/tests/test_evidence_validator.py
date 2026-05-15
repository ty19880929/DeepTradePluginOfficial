"""P0-2 v1 — evidence-field whitelist validator tests.

Scope:
* whitelist passes on legitimate ``evidence.field`` references
* unknown / fabricated fields produce errors → trigger repair retry
* free-text fields (``risk_flags`` / ``next_day_watch_points``) stay out of
  the whitelist (they are LLM-authored copy, not data-derived references)
* ``_complete_with_set_check`` integration: retries once on validator
  failure, raises ``_EvidenceValidationError`` after retries exhausted
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from limit_up_board.pipeline import (
    _EvidenceValidationError,
    _candidate_allowed_field_set,
    _complete_with_set_check,
    _make_evidence_field_validator,
)


# ---------------------------------------------------------------------------
# Validator unit tests — no LLM
# ---------------------------------------------------------------------------


class _FakeEvidence(BaseModel):
    field: str
    value: Any = None
    unit: str = "无"
    interpretation: str = "x"


class _FakeCandidate(BaseModel):
    candidate_id: str
    evidence: list[_FakeEvidence] = []
    risk_flags: list[str] = []
    next_day_watch_points: list[str] = []


class _FakeResponse(BaseModel):
    candidates: list[_FakeCandidate]


class TestCandidateAllowedFieldSet:
    """``_candidate_allowed_field_set`` 应取 candidate dict keys + 上下文白名单。"""

    def test_includes_dict_keys(self) -> None:
        allowed = _candidate_allowed_field_set(
            {"candidate_id": "X", "fd_amount_yi": 1.0, "turnover_ratio": 3.5}
        )
        assert "fd_amount_yi" in allowed
        assert "turnover_ratio" in allowed

    def test_includes_context_whitelist(self) -> None:
        allowed = _candidate_allowed_field_set({"candidate_id": "X"})
        # 这几个字段不在 candidate dict 里，但 prompt 文档化承诺存在
        assert "lgb_score" in allowed
        assert "lgb_decile" in allowed
        assert "sector_strength" in allowed
        assert "market_summary" in allowed


class TestEvidenceFieldValidator:
    """验证 _make_evidence_field_validator 返回的 closure 行为正确。"""

    def test_field_whitelist_passes(self) -> None:
        """field 全部命中 allow-list → 0 错误。"""
        allowed = {"X": {"fd_amount_yi", "lgb_score"}}
        validator = _make_evidence_field_validator(allowed, evidence_attr="evidence")
        resp = _FakeResponse(
            candidates=[
                _FakeCandidate(
                    candidate_id="X",
                    evidence=[
                        _FakeEvidence(field="fd_amount_yi"),
                        _FakeEvidence(field="lgb_score"),
                    ],
                )
            ]
        )
        assert validator(resp) == []

    def test_unknown_field_triggers_error(self) -> None:
        allowed = {"X": {"fd_amount_yi"}}
        validator = _make_evidence_field_validator(allowed, evidence_attr="evidence")
        resp = _FakeResponse(
            candidates=[
                _FakeCandidate(
                    candidate_id="X",
                    evidence=[
                        _FakeEvidence(field="fd_amount_yi"),  # ok
                        _FakeEvidence(field="market_buzz"),  # fabricated
                    ],
                )
            ]
        )
        errors = validator(resp)
        assert len(errors) == 1
        assert "X" in errors[0]
        assert "market_buzz" in errors[0]
        assert "not in input keys" in errors[0]

    def test_unknown_candidate_ignored(self) -> None:
        """validator 不为未知 candidate_id 报错（set-check 会先处理）。"""
        validator = _make_evidence_field_validator(
            {"X": {"f1"}}, evidence_attr="evidence"
        )
        resp = _FakeResponse(
            candidates=[
                _FakeCandidate(
                    candidate_id="Y",  # not in allowed
                    evidence=[_FakeEvidence(field="anything")],
                )
            ]
        )
        assert validator(resp) == []

    def test_free_text_fields_excluded(self) -> None:
        """``risk_flags`` / ``next_day_watch_points`` 是 LLM 自由文本，不应被检。

        以 ``evidence_attr="evidence"`` 实例化时，validator 只看 evidence；
        risk_flags 中放任何字符串都不应触发错误。
        """
        allowed = {"X": {"fd_amount_yi"}}
        validator = _make_evidence_field_validator(allowed, evidence_attr="evidence")
        resp = _FakeResponse(
            candidates=[
                _FakeCandidate(
                    candidate_id="X",
                    evidence=[_FakeEvidence(field="fd_amount_yi")],
                    risk_flags=["高位连续一字", "亏钱效应", "free_text_anything"],
                    next_day_watch_points=["明早是否跳空", "板块情绪"],
                )
            ]
        )
        assert validator(resp) == []


# ---------------------------------------------------------------------------
# Integration with _complete_with_set_check
# ---------------------------------------------------------------------------


class _FakeProfile:
    max_output_tokens = 32_000


class _FakeMeta(dict):
    pass


def _make_meta() -> dict:
    return {"input_tokens": 1, "output_tokens": 1}


class _FakeLLM:
    """Drives _complete_with_set_check responses in a controlled way.

    ``responses`` is a list of (raw_dict, expected_user_contains) tuples; each
    invocation of ``complete_json`` pops the next one.
    """

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, *, system, user, schema, profile, envelope_defaults=None):
        self.calls.append({"user": user, "schema": schema})
        raw = self.responses.pop(0)
        return raw, _make_meta()


class TestCompleteWithSetCheckEvidenceRetry:
    """`_complete_with_set_check` 在 validator 失败后必须 retry 一次，再失败抛错。"""

    def _good_resp(self) -> dict:
        return {
            "candidates": [
                {
                    "candidate_id": "X",
                    "evidence": [{"field": "fd_amount_yi"}],
                }
            ]
        }

    def _bad_resp(self) -> dict:
        return {
            "candidates": [
                {
                    "candidate_id": "X",
                    "evidence": [{"field": "market_buzz_fabricated"}],
                }
            ]
        }

    def _validator(self) -> Any:
        return _make_evidence_field_validator(
            {"X": {"fd_amount_yi"}}, evidence_attr="evidence"
        )

    def test_passes_first_attempt(self) -> None:
        llm = _FakeLLM([self._good_resp()])
        obj, _meta = _complete_with_set_check(
            llm,
            system="sys",
            user="orig",
            schema=_FakeResponse,
            profile=_FakeProfile(),
            expected_ids={"X"},
            evidence_validator=self._validator(),
        )
        assert len(obj.candidates) == 1
        assert len(llm.calls) == 1  # no retry needed

    def test_retries_on_first_attempt_bad_then_passes(self) -> None:
        """First call: bad evidence → validator fails → retry with hint;
        second call: good evidence → returns object."""
        llm = _FakeLLM([self._bad_resp(), self._good_resp()])
        obj, meta = _complete_with_set_check(
            llm,
            system="sys",
            user="orig",
            schema=_FakeResponse,
            profile=_FakeProfile(),
            expected_ids={"X"},
            evidence_validator=self._validator(),
        )
        assert len(llm.calls) == 2
        # Retry's user prompt must include the corrective hint
        assert "evidence 引用" in llm.calls[1]["user"] or "字段名" in llm.calls[1]["user"]
        # Meta carries the first-attempt error trail (recovery info)
        first_errs = meta.get("evidence_validation_errors_first_attempt") or []
        assert any("market_buzz_fabricated" in e for e in first_errs)
        assert obj.candidates[0].evidence[0].field == "fd_amount_yi"

    def test_raises_after_retry_still_bad(self) -> None:
        llm = _FakeLLM([self._bad_resp(), self._bad_resp()])
        with pytest.raises(_EvidenceValidationError) as ei:
            _complete_with_set_check(
                llm,
                system="sys",
                user="orig",
                schema=_FakeResponse,
                profile=_FakeProfile(),
                expected_ids={"X"},
                evidence_validator=self._validator(),
            )
        assert ei.value.errors
        assert any("market_buzz_fabricated" in e for e in ei.value.errors)
