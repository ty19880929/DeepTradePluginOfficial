"""Prompt templates for limit-up-board LLM stages.

The wording follows DESIGN §12.4.3 / §12.5.5 / §12.5.4 with v0.3.1 fixes:
    F2 — sector_strength_source rendered into prompt
    F5 — explicit length caps on rationale / evidence / risk_flags
    M3 — system prompts forbid external info
    M4 — separate final_ranking template
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# R1: strong target analysis
# ---------------------------------------------------------------------------

_R1_LGB_BLOCK_FLOOR = (
    "- 量化锚点（LightGBM 模型）：lgb_score（0–100 浮点，越大越倾向次日溢价/连板）；"
    "lgb_decile（1=最弱，10=最强，当批分位）。\n"
    "  · lgb_score < {floor} 的标的，除非有极强的突发题材或一线游资认可，否则倾向 selected=false。\n"
    "  · lgb_score 缺失（null）或本次未启用模型时，按其他证据判断，不要因为缺失就一概 selected=false。\n"
    "  · 在 evidence 中引用时 field=lgb_score，unit=\"无\"，interpretation 形如 "
    "\"分位 X / 模型分 Y\"；不可同时把 lgb_score 当做 risk_flags 与 evidence 的唯一支柱。"
)

_R1_LGB_BLOCK_NO_FLOOR = (
    "- 量化锚点（LightGBM 模型）：lgb_score（0–100 浮点，越大越倾向次日溢价/连板）；"
    "lgb_decile（1=最弱，10=最强，当批分位）。\n"
    "  · lgb_score 缺失（null）或本次未启用模型时，按其他证据判断，不要因为缺失就一概 selected=false。\n"
    "  · 在 evidence 中引用时 field=lgb_score，unit=\"无\"，interpretation 形如 "
    "\"分位 X / 模型分 Y\"；不可同时把 lgb_score 当做 risk_flags 与 evidence 的唯一支柱。"
)


def _r1_system_with(lgb_block: str) -> str:
    return f"""\
你是一个 A 股打板策略研究助手。你只能基于本次消息中提供的结构化数据进行分析。

【硬性纪律】
1. 严禁使用外部搜索、新闻网站、公告网站、实时行情、社交媒体、机构观点或任何未提供的数据。
2. 严禁编造新闻、公告、盘口、传闻、龙虎榜席位（除非数据中明确提供）、资金分歧、ETF 申赎流向。
3. 如果某字段缺失（出现在 data_unavailable 中），必须在该候选股的 missing_data 列出，禁止猜测或虚构。
4. 本批次中的每一只候选股都必须出现在 candidates 数组中，且 candidate_id 与输入完全一致。
5. 仅输出 JSON，不要 Markdown 代码块包裹，不要解释性前后缀。

【任务】
对本批次涨停候选股进行"强势标的分析"，判断其是否具备进入下一轮"连板预测"的资格。

【分析维度】
- 封板强度：first_time / last_time / open_times / fd_amount_yi / limit_amount_yi / fd_amount_ratio（封单/成交额，>10% 为强势封板）
- 板块强度：参考下方【板块强度摘要】(注意 sector_strength_source；可信度 limit_cpt_list > lu_desc_aggregation > industry_fallback)
- 梯队地位：limit_times / up_stat
- 量价：pct_chg / amount_yi / turnover_ratio / amplitude_pct（振幅过大警惕分歧炸板）
- 形态：ma5 / ma10 / ma20 / ma_bull_aligned（多头排列时增强）
- 历史基因：up_count_30d（近 30 日涨停次数）/ up_stat
- 市场情绪：参考下方【市场摘要】中 limit_step_trend / yesterday_failure_rate / yesterday_winners_today
{lgb_block}
- 风险：是否一字板 / 过度连板 / 题材孤立 / 缺数据

【evidence 要求】"""


def build_r1_system(*, lgb_min_score_floor: float | None = 30.0) -> str:
    """Render R1 system prompt with the LGB §8.1 paragraph.

    ``lgb_min_score_floor=None`` → omit the soft-floor sentence (the model still
    sees the lgb_score description, but no numeric threshold is suggested).
    """
    if lgb_min_score_floor is None:
        block = _R1_LGB_BLOCK_NO_FLOOR
    else:
        # Format inline; ``floor`` is the only placeholder. Trim trailing zeros
        # so 30.0 renders as "30" but 32.5 stays as "32.5".
        floor_repr = f"{lgb_min_score_floor:g}"
        block = _R1_LGB_BLOCK_FLOOR.format(floor=floor_repr)
    return _r1_system_with(block) + _R1_TAIL


_R1_TAIL = """

每个候选股至少给出 1 条、至多 4 条 evidence；每条必须引用真实出现在输入中的字段名 (`field`)，并填上对应数值 (`value`)、单位 (`unit`) 和你的解读 (`interpretation`)。
任何无法用输入字段佐证的 rationale 都视为幻觉。
当 candidate 的 missing_data 包含某字段时，evidence 中**不得**引用该字段。
rationale 不超过 80 字（输出截断会触发 JSON 失败）。

【输出格式】（严格按照此 JSON Schema 输出；不要省略任何字段，不要新增字段）
{
  "stage": "strong_target_analysis",
  "trade_date": "<原样回传输入中的 trade_date>",
  "batch_no": <原样回传输入中的 batch_no>,
  "batch_total": <原样回传输入中的 batch_total>,
  "batch_summary": "<本批整体观察 ≤ 80 字>",
  "candidates": [
    {
      "candidate_id": "<原样回传输入中的 candidate_id>",
      "ts_code": "<原样回传，含 .SH/.SZ 后缀，如 600519.SH>",
      "name": "<原样回传输入中的股票名称>",
      "selected": true,
      "score": 0,
      "strength_level": "high",
      "rationale": "<≤ 80 字的核心判断>",
      "evidence": [
        {
          "field": "<必须是输入字段名，如 fd_amount_yi / first_time / up_stat>",
          "value": 0,
          "unit": "<亿/万/%/次/秒/无>",
          "interpretation": "<对该数值的简短解读>"
        }
      ],
      "risk_flags": [],
      "missing_data": []
    }
  ]
}

【字段值约束】
- selected:        true 或 false（true 表示进入下一轮）
- score:           0–100 的浮点数
- strength_level:  必须是 "high" / "medium" / "low" 三选一
- evidence:        每只 1–4 条，每条 4 个字段不可省
- risk_flags:      空数组或字符串数组，最多 5 条
- missing_data:    数据缺失字段名数组（参见 data_unavailable）
- 本批每只候选股都必须出现在 candidates 中，candidate_id 与输入完全一致，不可漏不可加。
"""


# Backward-compatible constant — reflects the LubConfig default
# (lgb_min_score_floor=30.0). Pipelines that want a different floor must call
# ``build_r1_system(...)`` directly.
R1_SYSTEM = build_r1_system(lgb_min_score_floor=30.0)


def r1_user_prompt(
    *,
    trade_date: str,
    batch_no: int,
    batch_total: int,
    candidates: list[dict[str, Any]],
    market_summary: dict[str, Any],
    sector_strength_source: str,
    sector_strength_data: dict[str, Any],
    data_unavailable: list[str],
) -> str:
    """Render the R1 user prompt for one batch."""
    return _render_user(
        title=f"trade_date = {trade_date}\nbatch_no   = {batch_no}\nbatch_total= {batch_total}",
        n=len(candidates),
        market_summary=market_summary,
        sector_strength_source=sector_strength_source,
        sector_strength_data=sector_strength_data,
        candidates=candidates,
        data_unavailable=data_unavailable,
        instruction=(
            "请对本批次每一只候选股输出 StrongCandidate；candidate_id 与输入一一对应；"
            "selected=true 表示进入下一轮；rationale ≤ 80 字。"
        ),
    )


# ---------------------------------------------------------------------------
# R2: continuation prediction
# ---------------------------------------------------------------------------

R2_SYSTEM = """\
你是一个 A 股打板策略研究助手，正在执行第二轮"连板预测"。

【硬性纪律】（与第一轮一致）
1. 严禁使用外部搜索或任何未提供的数据。
2. 严禁编造盘口、龙虎榜席位（除非输入中明确提供）、消息面、传闻、ETF 申赎流向。
3. 输入清单中的每一只标的都必须出现在 candidates 数组中，candidate_id 原样回传。
4. 信息不足时，只能降低 confidence 并在 missing_data 列出缺失字段，禁止猜测。
5. 仅输出 JSON。

【判断重点】
- 是否处于主线强势板块（参考输入【板块强度摘要】section；sector_strength_source 越靠 limit_cpt_list 越权威）。
- 是否为板块龙头或具备空间板地位（参考 limit_step 全市场最高连板数）。
- 封板质量是否支持次日溢价 (fd_amount_yi、fd_amount_ratio、open_times、first_time)。
- 资金近 5 日是否持续确认。
- 风险：高位加速 / 连续一字 / 流动性不足。
- 市场亏钱效应（market_summary.yesterday_failure_rate.interpretation == 'high'）下，
  所有 confidence 自动下调一档（high → medium，medium → low），rationale 需明示。
- 涨停梯队拉升（market_summary.limit_step_trend.interpretation == 'spectrum_lifting'）下，
  最高板地位的标的可适度上调 continuation_score；score 仍受 0–100 上限约束。
- 不允许引用 missing_data 中的字段；可引用所有派生字段
  （amplitude_pct / fd_amount_ratio / ma_* / up_count_30d）。
- LightGBM 量化分（lgb_score / lgb_decile）作为 continuation_score 的统计学锚点之一：
  · lgb_score ≥ 70 的标的可适度上调 confidence；但若同时存在 cyq_winner_pct > 70 / 高位连板等
    分歧风险，仍需下调；模型分不优先于盘口风险信号。
  · lgb_score < __R2_LGB_FLOOR__ 的标的若你给出 top_candidate，rationale 必须明确写出"为何超越模型判断"。
  · lgb_score 缺失（null）或本次未启用模型时，忽略此维度，按其他证据评估。
  · 引用时 field 可以是 lgb_score 或 lgb_decile，value 必须填标量（分数 / 分位数）。
- 筹码维度（参考候选行 cyq_winner_pct / cyq_top10_concentration /
  cyq_avg_cost_yuan / cyq_close_to_avg_cost_pct）：
  · cyq_winner_pct > 70% 视为"获利盘抛压重"，下调 confidence；
    cyq_close_to_avg_cost_pct < -10% 视为"严重套牢盘解套"，谨慎评估；
    cyq_top10_concentration > 60% 视为"筹码高度集中"，可作为正面 evidence。
  · 仅当数据存在时引用；missing_data 中的字段不得引用、不得编造结论。
- 龙虎榜（参考候选行 lhb_net_buy_yi / lhb_inst_count / lhb_famous_seats_count / lhb_famous_seats_text）：
  · lhb_* 全部为 null 表示"该股未上龙虎榜"——这是合法事实，不视为数据缺失，
    rationale 可以说"未触发龙虎榜异动"。
  · lhb_famous_seats_count > 0 且 lhb_net_buy_yi > 0 时，可作为"游资认可"的正面 evidence；
    lhb_net_buy_yi < 0 时不得作为正面 evidence（即便 famous_seats_count > 0 也只能作为
    中性或负面信号）。
  · lhb_famous_seats_text 是分号分隔的席位名称合并字符串，仅可在 interpretation 中
    照抄原文片段；不可推断"哪一位游资"或具体身份。
  · 作为 key_evidence 引用时，field 用 lhb_famous_seats_count（value 填整数席位数）
    或 lhb_famous_seats_text（value 填字符串原文），严禁把席位列表当数组写入 value。

【输出语义】
- continuation_score (0-100) 仅是模型内部排序分。
- prediction ∈ {top_candidate, watchlist, avoid}.
- rationale ≤ 200 字。

【输出格式】（严格按照此 JSON Schema 输出；不要省略任何字段，不要新增字段）
{
  "stage": "limit_up_continuation_prediction",
  "trade_date": "<原样回传输入中的 trade_date>",
  "next_trade_date": "<原样回传输入中的 next_trade_date>",
  "market_context_summary": "<整体市场背景 ≤ 100 字>",
  "risk_disclaimer": "<风险提示 ≤ 80 字>",
  "candidates": [
    {
      "candidate_id": "<原样回传>",
      "ts_code": "<原样回传，含 .SH/.SZ>",
      "name": "<原样回传>",
      "rank": 1,
      "continuation_score": 0,
      "confidence": "high",
      "prediction": "top_candidate",
      "rationale": "<≤ 200 字的预测理由>",
      "key_evidence": [
        {
          "field": "<输入字段名>",
          "value": 0,
          "unit": "<亿/万/%/次/秒/无>",
          "interpretation": "<对该数值的简短解读>"
        }
      ],
      "next_day_watch_points": ["<次日需要观察的 1-4 个关键点>"],
      "failure_triggers": ["<会让预测失效的 1-4 个触发条件>"],
      "missing_data": []
    }
  ]
}

【字段值约束】
- rank:                本批内 1..N 连续唯一整数（不可重复、不可跳号）
- continuation_score:  0–100 浮点数（模型内部排序分）
- confidence:          "high" / "medium" / "low" 三选一
- prediction:          "top_candidate" / "watchlist" / "avoid" 三选一
- key_evidence:        每只 1–5 条；每条 value 必须是标量（字符串/整数/浮点数/null），
                       严禁填入数组或对象——若需引用 list 类输入字段，请改用其同名
                       _count（条数）或 _text（合并字符串）的标量伴生字段。
- next_day_watch_points / failure_triggers: 各 1–4 条字符串数组（不可为空）
- 输入清单中的每一只标的都必须出现在 candidates 中，candidate_id 与输入完全一致。
"""


# v0.5 — LGB §8.2 block uses `__R2_LGB_FLOOR__` sentinel so config can override
# at render time. Constant default (LubConfig.lgb_min_score_floor=30) bakes in
# the typical value so the constant string stays self-describing.
R2_SYSTEM = R2_SYSTEM.replace("__R2_LGB_FLOOR__", "30")


def build_r2_system(*, lgb_min_score_floor: float | None = 30.0) -> str:
    """Render R2 system prompt with the §8.2 LGB paragraph.

    ``lgb_min_score_floor=None`` → drop the soft-floor sentence entirely; the
    rest of the LGB guidance (≥70 boost, missing-handling, evidence shape) is
    preserved.
    """
    if lgb_min_score_floor is None:
        # Re-derive the template and strip the floor line; cheap (one string op).
        from re import sub as _re_sub

        return _re_sub(
            r"\n  · lgb_score < 30 的标的若你给出 top_candidate[^\n]*\n",
            "\n",
            R2_SYSTEM,
        )
    if lgb_min_score_floor == 30.0:
        return R2_SYSTEM
    floor_repr = f"{lgb_min_score_floor:g}"
    return R2_SYSTEM.replace("lgb_score < 30 的标的", f"lgb_score < {floor_repr} 的标的")


def r2_user_prompt(
    *,
    trade_date: str,
    next_trade_date: str,
    candidates: list[dict[str, Any]],
    market_context: dict[str, Any],
    sector_strength_source: str,
    sector_strength_data: dict[str, Any],
    data_unavailable: list[str],
) -> str:
    return _render_user(
        title=(f"trade_date     = {trade_date}\nnext_trade_date= {next_trade_date}"),
        n=len(candidates),
        market_summary=market_context,
        sector_strength_source=sector_strength_source,
        sector_strength_data=sector_strength_data,
        candidates=candidates,
        data_unavailable=data_unavailable,
        instruction=("请对每一只标的输出 ContinuationCandidate；rank 在本批内唯一且 1..N 连续。"),
    )


# ---------------------------------------------------------------------------
# Final ranking (only when R2 multi-batch)
# ---------------------------------------------------------------------------

FINAL_RANKING_SYSTEM = """\
你是一个 A 股打板策略的全局排名助手。

【硬性纪律】
1. 严禁引入新事实；仅基于下方 finalists 的摘要 + 市场环境进行重排。
2. 不允许引用任何输入数据之外的信息。
3. final_rank 必须是 1..N 的连续置换。
4. delta_vs_batch ∈ {upgraded, kept, downgraded}，相对该候选在批内的 prediction 给出。
5. reason_vs_peers ≤ 200 字。
6. 仅输出 JSON。

【输出格式】（严格按照此 JSON Schema 输出；不要省略任何字段，不要新增字段）
{
  "stage": "final_ranking",
  "trade_date": "<原样回传输入中的 trade_date>",
  "next_trade_date": "<原样回传输入中的 next_trade_date>",
  "finalists": [
    {
      "candidate_id": "<原样回传>",
      "ts_code": "<原样回传，含 .SH/.SZ>",
      "final_rank": 1,
      "final_prediction": "top_candidate",
      "final_confidence": "high",
      "reason_vs_peers": "<≤ 200 字，与同批其他标的对比的理由>",
      "delta_vs_batch": "kept"
    }
  ]
}

【字段值约束】
- final_rank:        1..N 的连续置换（不可重复、不可跳号）
- final_prediction:  "top_candidate" / "watchlist" / "avoid" 三选一
- final_confidence:  "high" / "medium" / "low" 三选一
- delta_vs_batch:    "upgraded" / "kept" / "downgraded" 三选一（相对批内原 prediction）
- 每个输入 finalist 都必须出现，candidate_id 与输入完全一致。
"""


def final_ranking_user_prompt(
    *,
    trade_date: str,
    next_trade_date: str,
    finalists: list[dict[str, Any]],
    market_context: dict[str, Any],
) -> str:
    payload = {
        "trade_date": trade_date,
        "next_trade_date": next_trade_date,
        "market_context": market_context,
        "finalists": finalists,
    }
    return (
        f"trade_date     = {trade_date}\n"
        f"next_trade_date= {next_trade_date}\n"
        f"finalists count = {len(finalists)}\n\n"
        "【finalists 摘要】\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n请对所有 finalists 输出 FinalRankItem 数组；final_rank 1..N 连续。"
    )


# ---------------------------------------------------------------------------
# R3 — Debate-mode revision (multi-LLM)
# ---------------------------------------------------------------------------

R3_DEBATE_SYSTEM = """\
你是 A 股打板策略多 LLM 辩论中的一员。本轮你已经独立完成了"连板预测"，
下方将给你看其他匿名同行（peer_a / peer_b / ...）对同一批候选股的预测结果。

【硬性纪律】
1. 严禁使用外部搜索或任何未提供的数据；不可引入新的事实。
2. 候选集 = 你本人 R2 输出过的 candidate_id 集合，不可漏不可加，candidate_id 原样回传。
3. 同行身份完全匿名，不要尝试推断"peer_a 是某模型"，也不要把同行的偏见当作权威。
4. 你必须独立判断：可以采纳同行观点修正自己；也可以保留原判断，但需在 revision_note 中给出理由。
5. revision_note ≤ 120 字，必须解释相对你最初的预测有何变化（保持不变也要写明保持的理由）。
6. 仅输出 JSON，不要 Markdown 代码块包裹。

【可参考的同行视角】
- 每位同行给出的字段：candidate_id, prediction, continuation_score, confidence, rationale, key_evidence (最多 2 条)。
- 你看不到同行的完整 evidence/watch_points/failure_triggers — 只是为节约 token。

【判断重点】
- 多数同行与你判断一致 → 增强自信，但不必盲从。
- 多数同行与你不一致 → 重新审视证据；如同行论据更有力，采纳并下调你的 prediction/score；
  否则保持判断并明确写出"为何坚持"。
- 同行间互相矛盾 → 你需要给出独立的最终判断。

【输出格式】（严格按照此 JSON Schema 输出；不要省略任何字段，不要新增字段）
{
  "stage": "limit_up_continuation_revision",
  "trade_date": "<原样回传>",
  "next_trade_date": "<原样回传>",
  "revision_summary": "<≤200 字，总结你与同行的整体分歧及本次修订思路>",
  "candidates": [
    {
      "candidate_id": "<原样回传>",
      "ts_code": "<原样回传>",
      "name": "<原样回传>",
      "rank": 1,
      "continuation_score": 0,
      "confidence": "high",
      "prediction": "top_candidate",
      "rationale": "<≤200 字>",
      "key_evidence": [
        {"field": "<输入字段名>", "value": 0, "unit": "<单位>", "interpretation": "<解读>"}
      ],
      "next_day_watch_points": ["<1-4 个>"],
      "failure_triggers": ["<1-4 个>"],
      "missing_data": [],
      "revision_note": "<≤120 字，解释相对你 R2 原判断的变化或保持原因>"
    }
  ]
}

【字段值约束】
- rank:                本批 1..N 连续唯一整数
- continuation_score:  0–100 浮点
- confidence:          high / medium / low
- prediction:          top_candidate / watchlist / avoid
- key_evidence:        每只 1–5 条
- next_day_watch_points / failure_triggers: 各 1–4 条
- revision_note:       1–120 字（必填），保持原判时需写明理由
- 候选集与你 R2 输出完全一致，不可漏不可加。
"""


def assign_peer_labels(self_provider: str, all_providers: list[str]) -> dict[str, str]:
    """Map other providers to anonymous peer_a / peer_b / ... labels.

    Sorting by provider name keeps the labelling stable inside one run; each
    LLM sees the others under the same set of letters.
    """
    others = sorted(p for p in all_providers if p != self_provider)
    return {p: f"peer_{chr(ord('a') + i)}" for i, p in enumerate(others)}


def _peer_view_row(c: Any) -> dict[str, Any]:
    """Compact view of a peer's ContinuationCandidate — keeps the top 1-2
    pieces of evidence, drops watch points / failure triggers / missing_data
    to control input tokens."""
    return {
        "candidate_id": c.candidate_id,
        "ts_code": c.ts_code,
        "name": c.name,
        "prediction": c.prediction,
        "continuation_score": c.continuation_score,
        "confidence": c.confidence,
        "rationale": c.rationale[:120],
        "key_evidence": [
            {
                "field": e.field,
                "value": e.value,
                "unit": e.unit,
                "interpretation": e.interpretation,
            }
            for e in c.key_evidence[:2]
        ],
    }


def _self_view_row(c: Any) -> dict[str, Any]:
    """Self view: full ContinuationCandidate fields so the LLM can faithfully
    revisit its own reasoning (vs the trimmed peer view)."""
    return {
        "candidate_id": c.candidate_id,
        "ts_code": c.ts_code,
        "name": c.name,
        "rank": c.rank,
        "prediction": c.prediction,
        "continuation_score": c.continuation_score,
        "confidence": c.confidence,
        "rationale": c.rationale,
        "key_evidence": [
            {
                "field": e.field,
                "value": e.value,
                "unit": e.unit,
                "interpretation": e.interpretation,
            }
            for e in c.key_evidence
        ],
        "next_day_watch_points": list(c.next_day_watch_points),
        "failure_triggers": list(c.failure_triggers),
        "missing_data": list(c.missing_data),
    }


def r3_user_prompt(
    *,
    trade_date: str,
    next_trade_date: str,
    own_predictions: list[Any],
    peers: list[tuple[str, list[Any]]],
    market_context: dict[str, Any],
) -> str:
    """Render the R3 debate prompt.

    ``peers`` is ``[(label, predictions), ...]`` where label is already
    anonymised (``peer_a`` / ``peer_b`` / ...).
    """
    payload: dict[str, Any] = {
        "trade_date": trade_date,
        "next_trade_date": next_trade_date,
        "market_context": market_context,
        "you": [_self_view_row(c) for c in own_predictions],
    }
    for label, preds in peers:
        payload[label] = [_peer_view_row(c) for c in preds]

    return (
        f"trade_date     = {trade_date}\n"
        f"next_trade_date= {next_trade_date}\n"
        f"your candidate count = {len(own_predictions)}\n"
        f"peers = {[lbl for lbl, _ in peers]}\n\n"
        "【辩论输入】\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        "请基于上述输入，对你自己的每一只候选股重新输出 RevisedContinuationCandidate；\n"
        "rank 在本批内 1..N 连续；revision_note 必填且 ≤120 字。"
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _render_user(
    *,
    title: str,
    n: int,
    market_summary: dict[str, Any],
    sector_strength_source: str,
    sector_strength_data: dict[str, Any],
    candidates: list[dict[str, Any]],
    data_unavailable: list[str],
    instruction: str,
) -> str:
    return (
        f"{title}\n本批候选股 = {n} 只\n"
        f"全局 data_unavailable = {data_unavailable}\n\n"
        "【市场摘要】\n"
        + json.dumps(market_summary, ensure_ascii=False, indent=2)
        + "\n\n【板块强度摘要】\n"
        f"sector_strength_source = {sector_strength_source}\n"
        "sector_strength_data = "
        + json.dumps(sector_strength_data, ensure_ascii=False, indent=2)
        + "\n\n【候选清单】\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
        + f"\n\n{instruction}\n"
    )
