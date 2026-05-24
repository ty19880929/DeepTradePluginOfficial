"""P1-I: Tests for limit_up_board.fingerprint."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from limit_up_board.config import LubConfig
from limit_up_board.fingerprint import (
    build_input_fingerprint,
    canonical_json,
    hash_json,
    hash_text,
)
from limit_up_board.profiles import (
    LLM_SCHEMA_VERSION,
    PROMPT_TEMPLATE_VERSION,
    PROFILES,
    STAGE_FINAL,
    STAGE_PREDICTION,
    STAGE_REVISION,
    STAGE_SCREENING,
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys() -> None:
    a = canonical_json({"b": 1, "a": 2})
    b = canonical_json({"a": 2, "b": 1})
    assert a == b == '{"a":2,"b":1}'


def test_canonical_json_normalises_nan_and_inf() -> None:
    out = canonical_json({"x": math.nan, "y": math.inf, "z": -math.inf})
    # NaN/Inf become null; key order preserved (sort_keys)
    assert out == '{"x":null,"y":null,"z":null}'


def test_canonical_json_handles_nested_set() -> None:
    out = canonical_json({"tags": {"b", "a", "c"}})
    # Sets are sorted for determinism
    assert '"tags":["a","b","c"]' in out


def test_hash_json_is_deterministic_across_dict_order() -> None:
    h1 = hash_json({"a": 1, "b": [1, 2, 3]})
    h2 = hash_json({"b": [1, 2, 3], "a": 1})
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_text_basic() -> None:
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("hello!")


# ---------------------------------------------------------------------------
# build_input_fingerprint
# ---------------------------------------------------------------------------


@dataclass
class _StubSectorStrength:
    source: str = "industry_fallback"
    data: dict = field(default_factory=lambda: {"top_sectors": []})


@dataclass
class _StubBundle:
    candidates: list[dict] = field(default_factory=list)
    market_summary: dict = field(default_factory=dict)
    sector_strength: _StubSectorStrength = field(default_factory=_StubSectorStrength)
    data_unavailable: list[str] = field(default_factory=list)
    lgb_model_id: str | None = None


def _stage_profiles() -> dict:
    return {
        STAGE_SCREENING: PROFILES["balanced"][STAGE_SCREENING],
        STAGE_PREDICTION: PROFILES["balanced"][STAGE_PREDICTION],
        STAGE_FINAL: PROFILES["balanced"][STAGE_FINAL],
        STAGE_REVISION: PROFILES["balanced"][STAGE_REVISION],
    }


def _build(**overrides):
    """Convenience wrapper with sensible defaults."""
    defaults = dict(
        trade_date="20260530",
        next_trade_date="20260531",
        daily_lookback=30,
        moneyflow_lookback=5,
        lub_config=LubConfig(),
        bundle=_StubBundle(
            candidates=[
                {"ts_code": "000001.SZ", "name": "平安银行", "limit_times": 2, "lgb_score": 50.0},
                {"ts_code": "600000.SH", "name": "浦发银行", "limit_times": 1, "lgb_score": 60.0},
            ],
            market_summary={"limit_up_count": 50},
            data_unavailable=["moneyflow_dc"],
            lgb_model_id="model-abc",
        ),
        stage_profiles=_stage_profiles(),
        llm_schema_version=LLM_SCHEMA_VERSION,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    defaults.update(overrides)
    return build_input_fingerprint(**defaults)


def test_input_fingerprint_stable_across_candidate_input_order() -> None:
    """Lab guarantee: shuffling input candidates must not change the hash."""
    h1, p1 = _build()

    shuffled_bundle = _StubBundle(
        candidates=[
            {"ts_code": "600000.SH", "name": "浦发银行", "limit_times": 1, "lgb_score": 60.0},
            {"ts_code": "000001.SZ", "name": "平安银行", "limit_times": 2, "lgb_score": 50.0},
        ],
        market_summary={"limit_up_count": 50},
        data_unavailable=["moneyflow_dc"],
        lgb_model_id="model-abc",
    )
    h2, p2 = _build(bundle=shuffled_bundle)
    assert h1 == h2
    assert p1["candidates"] == p2["candidates"]


def test_input_fingerprint_changes_when_lgb_model_id_changes() -> None:
    h1, _ = _build()
    other = _StubBundle(
        candidates=[
            {"ts_code": "000001.SZ", "name": "平安银行", "limit_times": 2, "lgb_score": 50.0},
            {"ts_code": "600000.SH", "name": "浦发银行", "limit_times": 1, "lgb_score": 60.0},
        ],
        market_summary={"limit_up_count": 50},
        data_unavailable=["moneyflow_dc"],
        lgb_model_id="model-xyz",  # changed
    )
    h2, _ = _build(bundle=other)
    assert h1 != h2


def test_input_fingerprint_changes_when_lgb_score_changes() -> None:
    h1, _ = _build()
    bumped = _StubBundle(
        candidates=[
            {"ts_code": "000001.SZ", "name": "平安银行", "limit_times": 2, "lgb_score": 51.0},  # 50→51
            {"ts_code": "600000.SH", "name": "浦发银行", "limit_times": 1, "lgb_score": 60.0},
        ],
        market_summary={"limit_up_count": 50},
        data_unavailable=["moneyflow_dc"],
        lgb_model_id="model-abc",
    )
    h2, _ = _build(bundle=bumped)
    assert h1 != h2


def test_input_fingerprint_changes_when_config_changes() -> None:
    h1, _ = _build()
    cfg2 = LubConfig(max_close_yuan=20.0)  # was 15
    h2, _ = _build(lub_config=cfg2)
    assert h1 != h2


def test_input_fingerprint_changes_when_schema_version_changes() -> None:
    h1, _ = _build()
    h2, _ = _build(llm_schema_version="lub-llm-schema-v999")
    assert h1 != h2


def test_input_fingerprint_payload_data_unavailable_sorted() -> None:
    bundle = _StubBundle(
        candidates=[],
        data_unavailable=["z_dc", "a_dc", "m_dc"],
        lgb_model_id=None,
    )
    _, payload = _build(bundle=bundle)
    assert payload["data_unavailable"] == ["a_dc", "m_dc", "z_dc"]


def test_input_fingerprint_payload_filters_candidate_fields() -> None:
    """Allowlist: only declared fields enter the payload (extra fields stripped)."""
    bundle = _StubBundle(
        candidates=[
            {
                "ts_code": "000001.SZ",
                "name": "test",
                "limit_times": 1,
                # Noise that must NOT enter the fingerprint:
                "_internal_debug_blob": "some-large-string",
                "display_only_color": "#ff0000",
            },
        ],
    )
    _, payload = _build(bundle=bundle)
    cand = payload["candidates"][0]
    assert "ts_code" in cand
    assert "_internal_debug_blob" not in cand
    assert "display_only_color" not in cand
