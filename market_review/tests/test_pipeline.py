"""pipeline — 7-section orchestration, theme passing, error isolation.

Uses a :class:`_FakeLLM` stub in place of the framework's
:class:`LLMClient`. The fake records every call and returns
caller-configured schema instances per stage, so we can assert on the
sequence + the data flowing between sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from market_review.metrics.breadth import BreadthReview
from market_review.metrics.capital import CapitalReview
from market_review.metrics.leaders import LeaderReview
from market_review.metrics.risk import RiskReview
from market_review.metrics.sectors import SectorReview
from market_review.metrics.sentiment import SentimentReview
from market_review.metrics.style import StyleReview
from market_review.pipeline import MetricsBundle, SectionResult, run_sections
from market_review.schemas import (
    SECTION_ORDER,
    SECTION_SCHEMAS,
    CapitalSection,
    HeadlineMetric,
    LeadersSection,
    OutlookHypothesis,
    OverviewSection,
    RiskOutlookSection,
    SectorsSection,
    SentimentSection,
    StyleSection,
)
from market_review.windows import Window


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    stage: str
    system: str
    user: str
    schema_name: str
    input_fingerprint: str | None


class _FakeLLM:
    """Minimum subset of LLMClient that pipeline.py actually uses."""

    def __init__(self, responder=None):
        self.calls: list[_Call] = []
        self._responder = responder or _default_responder

    def complete_json(
        self, *, system, user, schema, profile, stage=None,
        schema_version=None, input_fingerprint=None,
        envelope_defaults=None, replay=None,
    ):
        self.calls.append(_Call(
            stage=stage or "", system=system, user=user,
            schema_name=schema.__name__, input_fingerprint=input_fingerprint,
        ))
        validated = self._responder(stage, schema, user=user)
        return validated, {"latency_ms": 1, "prompt_hash": f"hash:{stage}"}


class _FakeLLMManager:
    def __init__(self, client: _FakeLLM):
        self._client = client

    def get_client(self, name=None, *, plugin_id, run_id=None, reports_dir=None):
        return self._client


def _default_responder(stage, schema_cls, *, user=None):  # noqa: ARG001
    """Return a minimal-valid schema instance for each section."""
    if stage == "overview":
        return OverviewSection(
            market_tone="震荡分化",
            headline_metrics=[
                HeadlineMetric(label=f"m{i}", value=i, unit="个") for i in range(3)
            ],
            theme_tags=["主线A", "主线B"],
            narrative_md="结构性上涨。",
        )
    if stage == "risk_outlook":
        return RiskOutlookSection(
            hypotheses=[OutlookHypothesis(
                title="震荡延续",
                rationale="量能温和",
                watch_points=["5日线"],
                fail_triggers=["跌破60日均线"],
            )],
            narrative_md="风险可控。",
        )
    # Default: vanilla SectionBase subclass (no required fields beyond base).
    return schema_cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_bundle() -> MetricsBundle:
    return MetricsBundle(
        window=Window(mode="day", start="20260530", end="20260530",
                      trade_dates=("20260530",), anchor="20260530"),
        breadth=BreadthReview(), sentiment=SentimentReview(),
        capital=CapitalReview(), sectors=SectorReview(),
        leaders=LeaderReview(), style=StyleReview(), risk=RiskReview(),
    )


def _runtime_with_llm(client: _FakeLLM):
    """Build a Mock-like MrRuntime carrying the fake LLM manager."""
    from market_review.runtime import MrRuntime  # noqa: PLC0415
    return MrRuntime(
        db=None,  # type: ignore[arg-type] — pipeline doesn't touch db
        config=None,  # type: ignore[arg-type]
        llms=_FakeLLMManager(client),  # type: ignore[arg-type]
        run_id="run-test",
        plugin_id="market-review",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_sections_executes_all_seven_in_order() -> None:
    client = _FakeLLM()
    rt = _runtime_with_llm(client)
    results = run_sections(rt, _empty_bundle())
    assert list(results) == list(SECTION_ORDER)
    assert [c.stage for c in client.calls] == list(SECTION_ORDER)


def test_each_section_uses_its_own_schema() -> None:
    client = _FakeLLM()
    rt = _runtime_with_llm(client)
    run_sections(rt, _empty_bundle())
    for call, section in zip(client.calls, SECTION_ORDER):
        assert call.schema_name == SECTION_SCHEMAS[section].__name__


def test_input_fingerprint_propagates_to_every_call() -> None:
    client = _FakeLLM()
    rt = _runtime_with_llm(client)
    run_sections(rt, _empty_bundle(), input_fingerprint="abc" * 21 + "a")
    assert all(c.input_fingerprint == "abc" * 21 + "a" for c in client.calls)


def test_overview_theme_tags_injected_into_subsequent_sections() -> None:
    client = _FakeLLM()
    rt = _runtime_with_llm(client)
    run_sections(rt, _empty_bundle())
    # Find the sectors call's user prompt and verify prevContext was included.
    sectors_call = next(c for c in client.calls if c.stage == "sectors")
    import json  # noqa: PLC0415
    decoded = json.loads(sectors_call.user)
    assert decoded["prevContext"] == {
        "marketTone": "震荡分化",
        "themeTags": ["主线A", "主线B"],
    }


def test_overview_failure_lets_pipeline_continue_with_placeholder() -> None:
    """Design §11.4 — overview failure degrades to default tone, other sections run."""
    def responder(stage, schema_cls, *, user=None):  # noqa: ARG001
        if stage == "overview":
            raise RuntimeError("LLM transport timeout")
        return _default_responder(stage, schema_cls, user=user)

    client = _FakeLLM(responder=responder)
    rt = _runtime_with_llm(client)
    results = run_sections(rt, _empty_bundle())
    # All sections present even though overview failed.
    assert len(results) == len(SECTION_ORDER)
    assert results["overview"].error is not None
    assert "LLM transport timeout" in results["overview"].error
    # Placeholder gets market_tone="未知".
    assert isinstance(results["overview"].schema, OverviewSection)
    assert results["overview"].schema.market_tone == "未知"
    # Subsequent sections still ran.
    assert results["sectors"].error is None


def test_single_section_failure_is_isolated() -> None:
    """A mid-pipeline failure should not poison later sections."""
    def responder(stage, schema_cls, *, user=None):  # noqa: ARG001
        if stage == "capital":
            raise ValueError("capital probe failed")
        return _default_responder(stage, schema_cls, user=user)

    client = _FakeLLM(responder=responder)
    rt = _runtime_with_llm(client)
    results = run_sections(rt, _empty_bundle())
    assert results["capital"].error is not None
    assert "capital probe failed" in results["capital"].error
    # leaders / style / risk_outlook still completed.
    assert results["leaders"].error is None
    assert results["style"].error is None
    assert results["risk_outlook"].error is None


def test_failed_section_carries_default_schema_instance() -> None:
    """A failed section's schema must still validate (placeholder fields).

    PR-5 builder expects to consume ``result.schema`` directly; a None
    instance would crash the chain. We require a valid placeholder schema
    instance so downstream rendering doesn't need to special-case None.
    """
    def responder(stage, schema_cls, *, user=None):  # noqa: ARG001
        if stage == "sectors":
            raise RuntimeError("boom")
        return _default_responder(stage, schema_cls, user=user)

    client = _FakeLLM(responder=responder)
    rt = _runtime_with_llm(client)
    results = run_sections(rt, _empty_bundle())
    placeholder = results["sectors"].schema
    assert isinstance(placeholder, SectorsSection)
    # Placeholder has no narrative; error carries the message.
    assert placeholder.narrative_md == ""
    assert placeholder.error is not None


def test_meta_propagates_for_success_calls() -> None:
    client = _FakeLLM()
    rt = _runtime_with_llm(client)
    results = run_sections(rt, _empty_bundle())
    for r in results.values():
        if r.error is None:
            assert "latency_ms" in r.meta
            assert "prompt_hash" in r.meta


def test_section_result_carries_section_name() -> None:
    client = _FakeLLM()
    rt = _runtime_with_llm(client)
    results = run_sections(rt, _empty_bundle())
    for section, result in results.items():
        assert isinstance(result, SectionResult)
        assert result.section == section
