"""Prompt template snapshots — T3.2."""

from __future__ import annotations

from accumulation_probe_washout.data import pack_candidate
from accumulation_probe_washout.prompts import APW_SYSTEM, apw_user_prompt
from accumulation_probe_washout.schemas import APWPhase, INPUT_FIELD_WHITELIST


def test_system_prompt_mentions_forbidden_behaviors() -> None:
    """System prompt must spell out the禁止编造 contract (源 §7.2)."""
    for phrase in (
        "外部搜索",
        "新闻、公告",
        "candidate_id",
        "key_evidence",
        "missing_data",
        "JSON",
    ):
        assert phrase in APW_SYSTEM

    for enum_val in (
        "launch_ready",
        "watch_breakout",
        "still_washing",
        "probe_failed",
        "avoid",
        "probe_washout_breakout",
        "low_base_accumulation",
        "high_level_distribution",
    ):
        assert enum_val in APW_SYSTEM


def test_user_prompt_contains_batch_metadata() -> None:
    candidates = [
        {"candidate_id": "20260515_600000.SH", "ts_code": "600000.SH",
         "name": "x", "trade_date": "20260515", "phase": "launch_ready"}
    ]
    user = apw_user_prompt(
        trade_date="20260515",
        next_trade_date="20260516",
        batch_no=1,
        batch_total=2,
        candidates=candidates,
    )
    assert "20260515" in user
    assert "20260516" in user
    assert "1/2" in user or "1/2)" in user
    assert "20260515_600000.SH" in user


# ---------------------------------------------------------------------------
# Fix 4 / P2-2 — pack_candidate ↔ INPUT_FIELD_WHITELIST must stay in sync.
# Without this guard the LLM evidence validator can reject perfectly valid
# field names just because the whitelist drifted from the candidate payload.
# ---------------------------------------------------------------------------


def _make_packed_candidate() -> dict:
    """Build a pack_candidate output that exercises every field branch."""
    basic = {
        "listed_days": 800,
        "close": 10.80,
        "pct_chg": 3.5,
        "turnover_rate": 4.2,
        "amount_yi": 5.6,
        "circ_mv_yi": 90.0,
    }
    accumulation = {
        "accumulation_score": 75.0,
        "accumulation_days": 40,
        "accumulation_net_mf_yi": 1.2,
        "accumulation_price_change_pct": 6.0,
        "low_position_score": 70.0,
    }
    probe = {
        "probe_date": "20260420",
        "probe_days_ago": 15,
        "probe_volume_ratio_5d": 3.4,
        "probe_volume_ratio_20d": 2.9,
        "probe_volume_rank_pct_60d": 98.0,
        "probe_amount_ratio_20d": 3.1,
        "probe_turnover_rate": 8.0,
        "probe_amplitude_pct": 7.5,
        "probe_upper_shadow_ratio": 0.18,
        "probe_body_ratio": 0.62,
        "probe_pct_chg": 4.2,
        "probe_moneyflow_net_yi": 0.45,
        "probe_quality_score": 80.0,
    }
    washout = {
        "washout_days": 12,
        "post_probe_max_drawdown_pct": 5.2,
        "post_probe_volume_shrink_ratio": 0.55,
        "post_probe_low_broken": False,
        "post_probe_ma20_broken": False,
        "post_probe_ma60_broken": False,
        "post_probe_moneyflow_net_yi": 0.1,
        "washout_volatility_compression": 60.0,
        "washout_score": 70.0,
    }
    launch = {
        "launch_setup_score": 65.0,
        "close_to_probe_high_pct": -1.2,
        "break_probe_high": False,
        "break_washout_high": True,
        "current_volume_ratio_5d": 1.4,
        "current_volume_ratio_20d": 1.2,
        "current_moneyflow_net_yi": 0.3,
        "above_ma5": True,
        "above_ma10": True,
        "above_ma20": True,
        "relative_strength_20d": 4.5,
    }
    return pack_candidate(
        trade_date="20260515",
        ts_code="600000.SH",
        name="浦发银行",
        phase=APWPhase.LAUNCH_READY,
        basic=basic,
        accumulation=accumulation,
        probe=probe,
        washout=washout,
        launch=launch,
        sector_strength_score=55.0,
    )


def test_pack_candidate_keys_match_whitelist() -> None:
    """Every key in the packed candidate must be on the LLM evidence whitelist.

    Regression for P2-2 in the round-one code review: fields like
    ``probe_body_ratio`` / ``washout_volatility_compression`` were being
    computed but never landed in the candidate payload, while ``probe_*``
    fields like ``probe_moneyflow_net_yi`` were referenced by the LLM prompt
    but missing from the whitelist. Drift in either direction is a bug.
    """
    cand = _make_packed_candidate()
    leaked = set(cand.keys()) - set(INPUT_FIELD_WHITELIST)
    assert not leaked, (
        f"pack_candidate emitted fields not on INPUT_FIELD_WHITELIST: "
        f"{sorted(leaked)}"
    )


def test_pack_candidate_carries_designed_evidence_fields() -> None:
    """Authoritative evidence-field checklist from the design doc §4 / §8."""
    cand = _make_packed_candidate()
    required = {
        # probe
        "probe_body_ratio",
        "probe_amount_ratio_20d",
        "probe_moneyflow_net_yi",
        # washout
        "post_probe_ma60_broken",
        "washout_volatility_compression",
        # launch
        "current_volume_ratio_20d",
        "break_washout_high",
        "relative_strength_20d",
    }
    missing = required - set(cand.keys())
    assert not missing, f"pack_candidate missing designed evidence fields: {sorted(missing)}"
    # And the same fields must be on the whitelist so LLM evidence using them
    # passes ``check_response_against_inputs``.
    assert required.issubset(set(INPUT_FIELD_WHITELIST))
