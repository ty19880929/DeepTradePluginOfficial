"""PR #4 — LLM review payload 构造 + 持久化 + markdown 渲染。

LLM 调用本身用 mock 客户端验证；不打真实网络。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deeptrade.core.db import Database

from limit_up_board.config import LubConfig
from limit_up_board.winrate.llm_review import (
    LandedSuggestion,
    WinrateLlmReview,
    build_review_payload,
    mint_review_id,
    persist_review,
    render_markdown_report,
)
from limit_up_board.winrate.persistence import PredictionRecord
from limit_up_board.winrate.resolver import ResolvedRecord
from limit_up_board.winrate.stats import (
    group_by_prediction,
    group_by_rank_bucket,
    summarize,
)

MIGRATION_FILES = [
    Path(__file__).resolve().parents[1] / "migrations" / "20260509_001_init.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_002_prediction_records.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_003_winrate_reviews.sql",
]


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))

    from deeptrade.core import paths as core_paths

    database = Database(core_paths.db_path())
    for mig in MIGRATION_FILES:
        sql_text = mig.read_text(encoding="utf-8")
        for stmt in sql_text.split(";"):
            stmt = stmt.strip()
            if stmt:
                database.execute(stmt)
    return database


def _rec(ts: str, *, rank: int = 1, prediction: str = "top_candidate", score: float = 80.0) -> PredictionRecord:
    return PredictionRecord(
        trade_date="20260521",
        next_trade_date="20260522",
        ts_code=ts,
        name=f"name-{ts}",
        run_id="r1",
        prediction=prediction,
        rank=rank,
        continuation_score=score,
        confidence="high",
        t_close_price=10.0,
        lgb_score=0.7,
        lgb_decile=9,
        raw_prediction_json=None,
    )


def _res(ts: str, outcome: str, pct: float | None, *, rank: int = 1, prediction: str = "top_candidate", score: float = 80.0) -> ResolvedRecord:
    rec = _rec(ts, rank=rank, prediction=prediction, score=score)
    t1 = 10.0 + (pct or 0) / 10.0 if pct is not None else None
    return ResolvedRecord(
        record=rec,
        t1_open_price=t1,
        open_vs_limit_pct=pct,
        outcome=outcome,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# build_review_payload — structure + safety
# ---------------------------------------------------------------------------


def test_build_payload_three_top_level_sections() -> None:
    resolved = [_res("a.SH", "win", 1.0), _res("b.SH", "loss", -1.0)]
    cfg = LubConfig()
    payload = build_review_payload(
        window_start="20260520", window_end="20260521",
        resolved=resolved,
        summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        by_rank=group_by_rank_bucket(resolved),
        lub_cfg=cfg,
        active_lgb_model_id="lgb-2026-04-01",
    )
    assert set(payload.keys()) == {"strategy_context", "performance_evidence", "review_task"}


def test_payload_strategy_context_reflects_config() -> None:
    cfg = LubConfig(min_float_mv_yi=20.0, max_float_mv_yi=80.0, max_close_yuan=12.0)
    payload = build_review_payload(
        window_start="20260520", window_end="20260521",
        resolved=[], summary=summarize([]),
        by_prediction=[], by_rank=[],
        lub_cfg=cfg,
        active_lgb_model_id="lgb-xyz",
    )
    ctx = payload["strategy_context"]
    assert ctx["candidate_filter"]["float_market_cap_min_yi"] == 20.0
    assert ctx["candidate_filter"]["float_market_cap_max_yi"] == 80.0
    assert ctx["candidate_filter"]["close_price_max_yuan"] == 12.0
    assert ctx["lgb"]["active_model_id"] == "lgb-xyz"
    assert ctx["lgb"]["enabled"] == cfg.lgb_enabled


def test_payload_strategy_context_lgb_none() -> None:
    payload = build_review_payload(
        window_start="20260520", window_end="20260521",
        resolved=[], summary=summarize([]),
        by_prediction=[], by_rank=[],
        lub_cfg=LubConfig(),
        active_lgb_model_id=None,
    )
    assert payload["strategy_context"]["lgb"]["active_model_id"] is None


def test_payload_does_not_leak_secrets() -> None:
    """payload 不应包含本地路径、token、prompt 原文等敏感字段。"""
    cfg = LubConfig()
    payload = build_review_payload(
        window_start="20260520", window_end="20260521",
        resolved=[_res("a.SH", "win", 1.0)], summary=summarize([_res("a.SH", "win", 1.0)]),
        by_prediction=[], by_rank=[],
        lub_cfg=cfg,
        active_lgb_model_id=None,
    )
    text = json.dumps(payload, ensure_ascii=False)
    # 不出现明显敏感词汇
    for forbidden in ["api_key", "token", "secret", "/Users/", "C:\\"]:
        assert forbidden not in text


def test_payload_high_score_failures_filtered() -> None:
    """high_score_failures 应只挑出 continuation_score >= 70 且 outcome=loss。"""
    resolved = [
        _res("a.SH", "loss", -1.0, score=80.0),  # 入选
        _res("b.SH", "loss", -1.0, score=50.0),  # 分数低，不入选
        _res("c.SH", "win", 1.0, score=80.0),    # 不是 loss
    ]
    payload = build_review_payload(
        window_start="20260520", window_end="20260521",
        resolved=resolved, summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        by_rank=group_by_rank_bucket(resolved),
        lub_cfg=LubConfig(),
        active_lgb_model_id=None,
    )
    failures = payload["performance_evidence"]["high_score_failures"]
    codes = [r["ts_code"] for r in failures]
    assert codes == ["a.SH"]


def test_payload_low_score_wins_filtered() -> None:
    resolved = [
        _res("a.SH", "win", 1.0, score=40.0),    # 入选
        _res("b.SH", "win", 1.0, score=90.0),    # 高分不算 surprise
        _res("c.SH", "loss", -1.0, score=40.0),  # 不是 win
    ]
    payload = build_review_payload(
        window_start="20260520", window_end="20260521",
        resolved=resolved, summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        by_rank=group_by_rank_bucket(resolved),
        lub_cfg=LubConfig(),
        active_lgb_model_id=None,
    )
    wins = payload["performance_evidence"]["low_score_wins"]
    assert [r["ts_code"] for r in wins] == ["a.SH"]


def test_payload_review_task_contains_landings() -> None:
    payload = build_review_payload(
        window_start="20260520", window_end="20260521",
        resolved=[], summary=summarize([]),
        by_prediction=[], by_rank=[],
        lub_cfg=LubConfig(),
        active_lgb_model_id=None,
    )
    landings = payload["review_task"]["each_suggestion_must_label_landing"]
    assert set(landings) == {
        "filter_rule", "prompt_weighting", "lgb_usage",
        "classification_boundary", "risk_control", "validation_plan",
    }


# ---------------------------------------------------------------------------
# Response schema enforcement
# ---------------------------------------------------------------------------


def test_landed_suggestion_rejects_unknown_landing() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LandedSuggestion(landing="invalid_label", text="...")  # type: ignore[arg-type]


def test_winrate_review_minimal_fields_required() -> None:
    rev = WinrateLlmReview(
        diagnosis="样本不足，但 top_candidate 表现弱于 watchlist。",
        validation_plan="再积累 10 个交易日后重新评估。",
    )
    assert rev.prompt_adjustments == []
    assert rev.caveats == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _sample_response() -> WinrateLlmReview:
    return WinrateLlmReview(
        diagnosis="top_candidate 严格胜率仅 30%，分类边界过宽。",
        prompt_adjustments=[
            LandedSuggestion(landing="prompt_weighting", text="提升封单评分权重"),
        ],
        feature_suggestions=[
            LandedSuggestion(landing="filter_rule", text="加入开板次数过滤"),
        ],
        risk_controls=[
            LandedSuggestion(landing="risk_control", text="T 日尾盘炸板的剔除"),
        ],
        validation_plan="下一个 10 日窗口对比 strict_win_rate 是否提升。",
        caveats=["样本量 < 50", "未区分指数环境"],
    )


def test_persist_review_writes_row(db: Database) -> None:
    resolved = [_res("a.SH", "win", 1.0)]
    summary = summarize(resolved)
    payload = {"foo": "bar"}
    persist_review(
        db,
        review_id="rev-001",
        window_start="20260520",
        window_end="20260521",
        llm_provider="deepseek",
        llm_model="deepseek-reasoner",
        summary=summary,
        payload=payload,
        response=_sample_response(),
        report_path="/tmp/out.md",
    )

    row = db.fetchone(
        "SELECT review_id, window_start, window_end, llm_provider, llm_model, "
        "sample_total, sample_resolved, strict_win_rate, report_path "
        "FROM lub_winrate_reviews WHERE review_id=?",
        ("rev-001",),
    )
    assert row is not None
    assert row[0] == "rev-001"
    assert row[1] == "20260520"
    assert row[3] == "deepseek"
    assert row[4] == "deepseek-reasoner"
    assert row[5] == 1
    assert row[6] == 1
    assert row[7] == pytest.approx(1.0)
    assert row[8] == "/tmp/out.md"


def test_persist_review_response_json_decodable(db: Database) -> None:
    summary = summarize([_res("a.SH", "win", 1.0)])
    persist_review(
        db,
        review_id="rev-002",
        window_start="20260520",
        window_end="20260521",
        llm_provider="kimi",
        llm_model=None,
        summary=summary,
        payload={},
        response=_sample_response(),
        report_path=None,
    )
    row = db.fetchone(
        "SELECT response_json FROM lub_winrate_reviews WHERE review_id=?",
        ("rev-002",),
    )
    assert row is not None
    decoded = json.loads(row[0])
    assert decoded["diagnosis"].startswith("top_candidate")
    assert decoded["prompt_adjustments"][0]["landing"] == "prompt_weighting"


def test_mint_review_id_uniqueness() -> None:
    ids = {mint_review_id() for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_includes_all_sections() -> None:
    summary = summarize([_res("a.SH", "win", 1.0), _res("b.SH", "loss", -1.0)])
    md = render_markdown_report(
        review_id="rev-md-1",
        window_start="20260520",
        window_end="20260521",
        llm_provider="deepseek",
        summary=summary,
        response=_sample_response(),
    )
    assert "rev-md-1" in md
    assert "20260520..20260521" in md
    assert "## diagnosis" in md
    assert "## prompt_adjustments" in md
    assert "## feature_suggestions" in md
    assert "## risk_controls" in md
    assert "## validation_plan" in md
    assert "## caveats" in md
    # 落点标签存在
    assert "[prompt_weighting]" in md
    assert "[filter_rule]" in md
    assert "不构成投资建议" in md


def test_render_markdown_skips_empty_blocks() -> None:
    """无 caveats / prompt_adjustments → 对应章节不输出。"""
    rev = WinrateLlmReview(
        diagnosis="样本不足无法给出建议。",
        validation_plan="待样本积累。",
    )
    summary = summarize([])
    md = render_markdown_report(
        review_id="rev-empty",
        window_start="20260520",
        window_end="20260521",
        llm_provider="deepseek",
        summary=summary,
        response=rev,
    )
    assert "## prompt_adjustments" not in md
    assert "## caveats" not in md
    assert "## diagnosis" in md
    assert "## validation_plan" in md
