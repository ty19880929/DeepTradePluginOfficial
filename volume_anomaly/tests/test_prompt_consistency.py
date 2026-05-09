"""volume-anomaly v0.3.0 — Few-Shot 示例字段名一致性测试。

防御目的：示例（prompts_examples.py::VA_TREND_FEWSHOT）中每一处
``"field": "<X>"`` 都必须能在以下两个集合之一中找到：
  1. _build_candidate_row 实际输出的 dict 键（analyze 阶段 LLM 看到的字段）
  2. screen_anomalies 输出的 hit 行字段（screen 报告中可见，且部分通过
     watchlist 持久化）

任何 data.py 中的字段重命名都会让此测试红，避免 prompt 和实际数据脱钩。
"""

from __future__ import annotations

import re
from typing import Any

from volume_anomaly.data import (
    _build_candidate_row,
)
from volume_anomaly.prompts_examples import (
    VA_TREND_FEWSHOT,
)


# ---------------------------------------------------------------------------
# Build the legal field set
# ---------------------------------------------------------------------------


def _make_synthetic_history() -> list[dict[str, Any]]:
    """Construct a 250-row OHLCV history sufficient to exercise every field
    branch in ``_build_candidate_row``."""
    from datetime import datetime, timedelta

    end = datetime.strptime("20261231", "%Y%m%d")
    out: list[dict[str, Any]] = []
    close = 10.0
    for i in range(250):
        date = (end - timedelta(days=249 - i)).strftime("%Y%m%d")
        rng = close * 0.02
        sign = 1.0 if (i % 2 == 0) else -1.0
        new_close = close + 0.01 + sign * rng / 4
        out.append(
            {
                "trade_date": date,
                "open": close,
                "high": max(close, new_close) + rng / 2,
                "low": min(close, new_close) - rng / 2,
                "close": new_close,
                "pct_chg": (new_close - close) / close * 100,
                "vol": 1_000_000 + i * 100,
            }
        )
        close = new_close
    return out


def _build_candidate_keys() -> set[str]:
    rec = _build_candidate_row(
        watchlist_row={
            "ts_code": "000001.SZ",
            "name": "测试",
            "industry": "测试",
            "tracked_since": "20260601",
            "last_screened": "20260601",
            "last_pct_chg": 6.5,
            "last_close": 10.0,
            "last_vol": 1_000_000,
            "last_amount": 10_000_000,
            "last_body_ratio": 0.7,
            "last_turnover_rate": 5.0,
            "last_vol_ratio_5d": 2.5,
            "last_max_vol_60d": 1_500_000,
        },
        trade_date="20261231",
        history=_make_synthetic_history(),
        daily_basic={
            "turnover_rate": 5.0,
            "volume_ratio": 1.5,
            "pe": 20.0,
            "pb": 2.0,
            "circ_mv": 800_000.0,
            "total_mv": 1_000_000.0,
        },
        moneyflow_5d=[],
        limit_up_dates=[],
    )
    return set(rec.keys())


# Hit-row schema fields populated by screen_anomalies (mirrored from data.py).
# Hardcoded per the design's instruction: "也可硬编码 hit schema 字段集合".
HIT_FIELDS: set[str] = {
    "ts_code",
    "name",
    "industry",
    "trade_date",
    "pct_chg",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "body_ratio",
    "upper_shadow_ratio",
    "turnover_rate",
    "circ_mv_yi",
    "turnover_bucket",
    "vol_ratio_5d",
    "vol_rank_in_long_window",
    "max_vol_short_window",
    "max_vol_long_window",
    "history_days_used",
    "max_vol_60d",
}


CANDIDATE_FIELDS = _build_candidate_keys()
LEGAL_FIELDS = CANDIDATE_FIELDS | HIT_FIELDS

# Allow special-cased candidate_id placeholders used in the few-shot's
# top-level fields (these aren't `key_evidence.field` references and live
# at the response root).
ALLOWED_TOP_LEVEL = {
    "candidate_id",
    "ts_code",
    "name",
    "rank",
    "launch_score",
    "confidence",
    "prediction",
    "pattern",
    "washout_quality",
    "rationale",
    "key_evidence",
    "next_session_watch",
    "invalidation_triggers",
    "risk_flags",
    "missing_data",
}


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_few_shot_field_names_are_legal() -> None:
    """Every `"field": "<X>"` in the few-shot must exist in candidate-row
    output OR screen-hit schema."""
    pattern = re.compile(r'"field":\s*"([^"]+)"')
    cited_fields = pattern.findall(VA_TREND_FEWSHOT)
    assert cited_fields, "few-shot block contains no `\"field\": ...` references"

    illegal = [f for f in cited_fields if f not in LEGAL_FIELDS]
    assert not illegal, (
        f"few-shot cites fields not in candidate-row or hit schema: {illegal}; "
        f"either rename in prompts_examples.py or add to data.py"
    )


def test_few_shot_block_appended_to_system_prompt() -> None:
    from volume_anomaly.prompts import (
        VA_TREND_SYSTEM,
    )

    assert "【参考示例】" in VA_TREND_SYSTEM
    # Sanity check: the body of FEWSHOT must be a substring of SYSTEM (i.e.
    # the concatenation must have happened — guards against a future refactor
    # that defines them separately but forgets to compose).
    assert "示例 A — 教科书式 VCP" in VA_TREND_SYSTEM
    assert "示例 B — 高位长上影线" in VA_TREND_SYSTEM


def test_dimension_a_mentions_volatility_convergence() -> None:
    """PR-3 also added a maintenance hint to dimension A — make sure the new
    field names referenced there are real."""
    from volume_anomaly.prompts import (
        VA_TREND_SYSTEM,
    )

    assert "atr_10d_quantile_in_60d" in VA_TREND_SYSTEM
    assert "bbw_compression_ratio" in VA_TREND_SYSTEM
    assert "atr_10d_quantile_in_60d" in CANDIDATE_FIELDS
    assert "bbw_compression_ratio" in CANDIDATE_FIELDS
