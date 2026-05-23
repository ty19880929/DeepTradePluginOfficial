"""P2-3 (v0.13.0)：``RunMetrics`` 聚合事件流并 emit 出 OBSERVABILITY_SUMMARY 内容。"""

from __future__ import annotations

import time

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from limit_up_board.observability import RunMetrics


def _ev(etype: EventType, msg: str = "", payload: dict | None = None) -> StrategyEvent:
    return StrategyEvent(type=etype, level=EventLevel.INFO, message=msg, payload=payload or {})


def test_stage_duration_pairs_step_started_with_finished() -> None:
    m = RunMetrics()
    m.observe(_ev(EventType.STEP_STARTED, "Step 1: data assembly"))
    time.sleep(0.01)
    m.observe(_ev(EventType.STEP_FINISHED, "Step 1: 60 candidates"))
    payload = m.build_summary_payload()
    durations = payload["stage_duration_ms"]
    # Started name is the key
    assert "Step 1: data assembly" in durations
    assert durations["Step 1: data assembly"] is not None
    assert durations["Step 1: data assembly"] >= 5.0


def test_multiple_stages_keep_independent_durations() -> None:
    m = RunMetrics()
    for name in ["Step 0", "Step 1", "Step 2"]:
        m.observe(_ev(EventType.STEP_STARTED, name))
        m.observe(_ev(EventType.STEP_FINISHED, name))
    payload = m.build_summary_payload()
    assert set(payload["stage_duration_ms"].keys()) == {"Step 0", "Step 1", "Step 2"}


def test_tushare_calls_collected_from_event_payloads() -> None:
    m = RunMetrics()
    m.observe(_ev(
        EventType.TUSHARE_CALL,
        "stock_basic",
        payload={"api": "stock_basic", "rows": 4500, "duration_ms": 320.0, "cache_hit": False},
    ))
    m.observe(_ev(
        EventType.TUSHARE_FALLBACK,
        "limit_list_ths",
        payload={"api": "limit_list_ths", "error": "unauthorized"},
    ))
    payload = m.build_summary_payload()
    apis = [c["api"] for c in payload["tushare_api_calls"]]
    assert apis == ["stock_basic", "limit_list_ths"]
    assert payload["tushare_api_calls"][0]["rows"] == 4500
    assert payload["tushare_api_calls"][1]["error"] == "unauthorized"


def test_llm_batch_pairs_started_and_finished() -> None:
    m = RunMetrics()
    m.observe(_ev(
        EventType.LLM_BATCH_STARTED,
        "screening batch 1",
        payload={"provider": "deepseek", "model": "v3", "batch_no": 1, "stage": "screening"},
    ))
    time.sleep(0.005)
    m.observe(_ev(
        EventType.LLM_BATCH_FINISHED,
        "screening batch 1 done",
        payload={
            "provider": "deepseek", "model": "v3", "batch_no": 1, "stage": "screening",
            "repair_count": 0, "tokens_in": 12345, "tokens_out": 5678,
        },
    ))
    payload = m.build_summary_payload()
    calls = payload["llm_calls"]
    assert len(calls) == 1
    rec = calls[0]
    assert rec["provider"] == "deepseek"
    assert rec["model"] == "v3"
    assert rec["batch_no"] == 1
    assert rec["repair_count"] == 0
    assert rec["tokens_in"] == 12345
    assert rec["duration_ms"] is not None and rec["duration_ms"] >= 2.0


def test_validation_failed_count() -> None:
    m = RunMetrics()
    for _ in range(3):
        m.observe(_ev(EventType.VALIDATION_FAILED, "set mismatch"))
    payload = m.build_summary_payload()
    assert payload["validation_failed_count"] == 3


def test_upload_audit_captured_from_log_payload() -> None:
    m = RunMetrics()
    m.observe(_ev(
        EventType.LOG,
        "📤 report uploaded",
        payload={
            "enabled": True,
            "url": "https://example.com/upload",
            "status": "ok",
            "duration_ms": 412.0,
            "public_url": "https://blob/r/1.json",
            "public_path": "r/1.json",
            "token_configured": True,
        },
    ))
    payload = m.build_summary_payload()
    up = payload["upload"]
    assert up is not None
    assert up["status"] == "ok"
    assert up["public_url"] == "https://blob/r/1.json"
    # token 不应出现
    assert "token" not in up


def test_record_lgb_appears_in_summary() -> None:
    m = RunMetrics()
    m.record_lgb(model_id="lgb-2026-05-23", coverage=0.86, missing_rate=0.04)
    payload = m.build_summary_payload()
    assert payload["lgb"]["model_id"] == "lgb-2026-05-23"
    assert payload["lgb"]["coverage"] == 0.86


def test_run_duration_ms_is_positive() -> None:
    m = RunMetrics()
    time.sleep(0.002)
    payload = m.build_summary_payload()
    assert payload["run_duration_ms"] >= 1.0


def test_unknown_event_does_not_raise() -> None:
    m = RunMetrics()
    # Construct a synthetic event with a normal EventType but garbage payload
    m.observe(_ev(EventType.LIVE_STATUS, "ignore me", payload={"k": object()}))
    payload = m.build_summary_payload()
    assert payload["observability_summary"] is True


def test_summary_payload_marker_is_present() -> None:
    """The marker key drives downstream filtering (summary.json::quality_metrics)."""
    m = RunMetrics()
    assert m.build_summary_payload()["observability_summary"] is True
