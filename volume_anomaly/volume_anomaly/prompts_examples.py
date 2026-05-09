"""Few-shot anchoring examples for the volume-anomaly trend-prediction stage.

Two synthetic candidates are wired into ``VA_TREND_SYSTEM`` (see prompts.py)
to anchor LLMs onto a consistent ``launch_score`` scale and to demonstrate the
expected ``key_evidence`` field-citation discipline.

CRITICAL CONSTRAINT: every ``"field": "<X>"`` referenced below MUST appear in
either the ``hits`` rows produced by ``screen_anomalies`` or the candidate
rows produced by ``_build_candidate_row``. The
``test_prompt_consistency.py`` test enforces this — any field rename in
``data.py`` will turn that test red, preventing silent prompt-schema drift.
"""

from __future__ import annotations

VA_TREND_FEWSHOT = """\

【参考示例】（仅展示判断尺度与字段引用规范，不是输入的一部分）

示例 A — 教科书式 VCP 缩量充分洗盘后放量突破
{
  "candidate_id": "000XXX.SZ",
  "ts_code": "000XXX.SZ",
  "name": "示例 A",
  "rank": 1,
  "launch_score": 78,
  "confidence": "high",
  "prediction": "imminent_launch",
  "pattern": "breakout",
  "washout_quality": "sufficient",
  "rationale": "整理 24 日，回撤 12%，波动率分位 0.08；T 日放量站上 MA20；moneyflow 5 日累计净流入；板块为当日主线。三维 VCP 齐降 + 主线共振 → 启动概率高。",
  "dimension_scores": {
    "washout": 80,
    "pattern": 75,
    "capital": 70,
    "sector": 75,
    "historical": 60,
    "risk": 25
  },
  "key_evidence": [
    {"field": "base_days",                  "value": 24,    "unit": "日",  "interpretation": "整理周期较长，洗盘相对充分"},
    {"field": "atr_10d_quantile_in_60d",    "value": 0.08,  "unit": "无",  "interpretation": "波动率处于近 60 日 8% 分位，VCP 收敛"},
    {"field": "anomaly_vol_ratio_5d",       "value": 2.4,   "unit": "倍",  "interpretation": "异动当日相对前 5 日放量 2.4 倍"},
    {"field": "dist_to_250d_high_pct",      "value": -3.5,  "unit": "%",   "interpretation": "距 250 日新高仅 3.5%，临近年线突破口"},
    {"field": "alpha_20d_pct",              "value": 8.2,   "unit": "%",   "interpretation": "近 20 日相对沪深 300 跑赢 8.2%，主线领涨"}
  ],
  "next_session_watch": ["次日开盘是否站稳 MA10", "板块强度是否延续主线地位"],
  "invalidation_triggers": ["收盘跌破 MA10 且 moneyflow 转为净流出", "板块跌出主线前 5"],
  "risk_flags": [],
  "missing_data": []
}

示例 B — 高位长上影线 + 资金外流
{
  "candidate_id": "600YYY.SH",
  "ts_code": "600YYY.SH",
  "name": "示例 B",
  "rank": 2,
  "launch_score": 22,
  "confidence": "medium",
  "prediction": "not_yet",
  "pattern": "unclear",
  "washout_quality": "insufficient",
  "rationale": "异动当天上影线占振幅 0.42，body_ratio 仅 0.55；过去 60 日已两次涨停且距 120 日新高 < 1%；moneyflow 5 日累计净流出。高位放量诱多概率高。",
  "dimension_scores": {
    "washout": 30,
    "pattern": 35,
    "capital": 25,
    "sector": 50,
    "historical": 30,
    "risk": 75
  },
  "key_evidence": [
    {"field": "upper_shadow_ratio",         "value": 0.42,  "unit": "无",  "interpretation": "上影线偏长，疑似冲高回落"},
    {"field": "prior_limit_up_count_60d",   "value": 2,     "unit": "次",  "interpretation": "近 60 日已 2 次涨停，浪型偏后"},
    {"field": "dist_to_120d_high_pct",      "value": -0.8,  "unit": "%",   "interpretation": "贴近 120 日新高，套牢盘压力大"},
    {"field": "alpha_20d_pct",              "value": -3.0,  "unit": "%",   "interpretation": "近 20 日相对沪深 300 跑输，板块支撑弱"}
  ],
  "next_session_watch": ["放量回踩 MA10 是否守住", "moneyflow 是否转为净流入"],
  "invalidation_triggers": ["再放量阴线击穿 MA20"],
  "risk_flags": ["high_bull_trap_risk", "late_stage_pattern"],
  "missing_data": []
}
"""
