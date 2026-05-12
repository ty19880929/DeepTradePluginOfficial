"""PR-2.3 — prompt now mentions the lgb_score quantitative anchor section."""

from __future__ import annotations

import json

from volume_anomaly.prompts import VA_TREND_SYSTEM, va_trend_user_prompt


def test_system_prompt_contains_quant_anchor_section() -> None:
    # The new "G. 量化锚点" block must reach the LLM.
    assert "量化锚点" in VA_TREND_SYSTEM
    assert "lgb_score" in VA_TREND_SYSTEM
    assert "lgb_decile" in VA_TREND_SYSTEM
    # Floor threshold mention so LLM knows what "low score" means.
    assert "25" in VA_TREND_SYSTEM


def test_system_prompt_keeps_existing_dimensions() -> None:
    # Regression: the original six dimensions (A..F) must still be present.
    for letter in ("A.", "B.", "C.", "D.", "E.", "F."):
        assert letter in VA_TREND_SYSTEM


def test_user_prompt_serializes_lgb_score_when_present() -> None:
    candidate = {
        "candidate_id": "000001.SZ",
        "ts_code": "000001.SZ",
        "name": "测试股",
        "lgb_score": 68.7,
        "lgb_decile": 8,
        "lgb_feature_missing": [],
    }
    prompt = va_trend_user_prompt(
        trade_date="20260601",
        next_trade_date="20260602",
        batch_no=1,
        batch_total=1,
        candidates=[candidate],
        market_summary={},
        sector_strength_source="industry_fallback",
        sector_strength_data={"top_sectors": []},
        data_unavailable=[],
    )
    assert "lgb_score" in prompt
    assert "68.7" in prompt
    assert "lgb_decile" in prompt


def test_user_prompt_omits_lgb_field_when_absent() -> None:
    candidate = {
        "candidate_id": "000001.SZ",
        "ts_code": "000001.SZ",
        "name": "测试股",
        # No lgb_* keys at all.
    }
    prompt = va_trend_user_prompt(
        trade_date="20260601",
        next_trade_date="20260602",
        batch_no=1,
        batch_total=1,
        candidates=[candidate],
        market_summary={},
        sector_strength_source="industry_fallback",
        sector_strength_data={"top_sectors": []},
        data_unavailable=[],
    )
    # The user prompt is JSON-dumped from the candidates dict; if the keys
    # aren't on the dict, they shouldn't be in the prompt.
    # (Note: the system prompt still mentions lgb_score as a concept; we
    # only assert here that the per-candidate JSON didn't fabricate one.)
    assert '"lgb_score": null' not in prompt
    assert '"lgb_score":' not in prompt
