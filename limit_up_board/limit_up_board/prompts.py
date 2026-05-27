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
# 强势初筛 (strong target analysis)
# ---------------------------------------------------------------------------

# v0.7.0 — LGB §8.1 block 拆成 3 段：score 描述、可选 floor 例外、可选 decile 段。
# 三段独立拼装让 ``build_screening_system`` 同时支持 (1) 关闭 floor、(2) 关闭 decile
# 输入两种正交开关，对应 LubConfig.lgb_min_score_floor / lgb_decile_in_prompt。
_SCREENING_LGB_HEAD = (
    "- 量化锚点（LightGBM 模型）：lgb_score（0–100 浮点，**未校准的模型排序分**——"
    "数值越大越倾向次日触及「T+1 最高价 ≥ T 收盘价 × (1+阈值%)」，但**不是概率**（v0.13.0 起明确语义）；越大越倾向次日高位溢价/连板，但 ≠ 必涨停 / "
    "≠ 可实现收益；属于盘后统计学锚点，盘口风险信号仍优先）。"
)

_SCREENING_LGB_DECILE_LINE = (
    "lgb_decile（1=最弱，10=最强，当批分位）。"
)

# P1-5 — 例外字段化：仅允许引用输入字段，避免 LLM 自由补全"突发题材/一线游资"。
_SCREENING_LGB_FLOOR_BLOCK = (
    "\n  · lgb_score < {floor} 的标的倾向 selected=false（不列为强势推荐）；仅当输入字段满足以下"
    "任一条件时才宜 selected=true，且 rationale 必须明确写出引用的字段名与数值：\n"
    "    (a) lhb_famous_seats_count > 0 且 lhb_net_buy_yi > 0（一线游资明确净买入）；\n"
    "    (b) lu_desc / tag 含明显主线题材关键词，或候选 concepts 数组中至少一个名称"
    "出现在 sector_strength_data.top_sectors（题材可信度足够，且 sector_strength_source = "
    "limit_cpt_list 时尤其可靠）。\n"
    "    （注：selected 只是推荐标签，selected=false 的标的仍会进入下一轮，score 请如实给出。）"
)

_SCREENING_LGB_TAIL = (
    "\n  · lgb_score 缺失（null）或本次未启用模型时，按其他证据判断，不要因为缺失就一概 selected=false。"
    "\n  · 在 evidence 中引用时 field=lgb_score，unit=\"无\"，interpretation 形如 "
    "\"分位 X / 模型分 Y\"；不可同时把 lgb_score 当做 risk_flags 与 evidence 的唯一支柱。"
)

# P1-4 — R1 阶段筹码/LHB 字段使用边界（候选行已带这些字段，但 R1 目标是高召回）。
_SCREENING_CYQ_LHB_BLOCK = (
    "- 筹码 / 龙虎榜（cyq_winner_pct / cyq_top10_concentration / lhb_net_buy_yi / "
    "lhb_famous_seats_count / lhb_data_quality）：**R1 仅作为风险或正向加分信号，"
    "不作为主要筛选维度**——\n"
    "  · cyq_winner_pct > 70% 视为「获利盘抛压重」→ 降分但不直接 selected=false；\n"
    "  · lhb_famous_seats_count > 0 且 lhb_net_buy_yi > 0 视为「游资认可」→ 可作为正向加分；\n"
    "  · lhb_data_quality=\"not_listed\" 是合法事实（未上龙虎榜），不视为缺失；\n"
    "  · 详细分歧/解套口径在下一轮（连板预测）评估，R1 不必深挖。"
)


def _screening_system_with(lgb_block: str) -> str:
    return f"""\
你是一个 A 股盘后涨停复盘研究助手。你只能基于本次消息中提供的结构化数据进行分析。

【场景前提】
- 当前是【盘后】基于 T 日已经收盘的涨停池，对【T+1 次日连板/高位溢价】候选进行预测。
- 输入均为盘后结构化数据；**不会**提供集合竞价、盘中实时盘口、实时封单撤单、新闻公告等。
- 因此判断标的强弱时，请围绕 T 日涨停盘的复盘特征（封板质量/题材/梯队/量价/资金/历史基因）展开。

【硬性纪律】
1. 严禁使用外部搜索、新闻网站、公告网站、实时行情、社交媒体、机构观点或任何未提供的数据。
2. 严禁编造新闻、公告、盘口、传闻、龙虎榜席位（除非数据中明确提供）、资金分歧、ETF 申赎流向。
3. missing_data 字段语义【严格定义，禁止泛化】：
   a) missing_data 必须是输入中 data_unavailable 列表的**子集**——只能从 data_unavailable 中的字符串原样挑选，**严禁**填入任何未出现在 data_unavailable 中的字段名（包括但不限于 ma5/ma10/ma20/ma_bull_aligned/up_count_30d/limit_amount_yi/mf_net_5d_sum_yi 等候选行内派生字段）。
   b) 当 data_unavailable 为 [] 时，所有候选股的 missing_data **必须**为 []。
   c) 候选行内某字段值为 null **不是**"数据缺失"，而是合法事实（如新股不足 30 日历史导致 up_count_30d=null、Tushare 未返回 limit_amount 导致 limit_amount_yi=null、未上龙虎榜导致 lhb_*=null 等）；这种情况**不写入 missing_data**，仅在 evidence 中避免引用该字段即可，rationale 也不要描述为"缺数据"。
   d) 严禁猜测、虚构或编造数据。
4. 本批次中的每一只候选股都必须出现在 candidates 数组中，且 candidate_id 与输入完全一致。
5. 仅输出 JSON，不要 Markdown 代码块包裹，不要解释性前后缀。

【任务】
对本批次 T 日涨停候选股进行"强势标的分析"：逐只给出强弱评分（score）、强弱等级
（strength_level）与核心判断（rationale + evidence）。
**重要：本批每一只候选股都会进入下一轮"次日连板/高位溢价预测"，强势分析不再做筛选淘汰。**
`selected` 仅是你给出的**建议性「强势推荐」标签**（true=你认为值得重点关注的强势标的），
用于报告高亮与人工参考，**不决定**任何标的是否进入下一轮——请如实评分，不要为了"让它进入
下一轮"而抬高 selected/score。

【分析维度】
- 封板强度：first_time / last_time / open_times / fd_amount_yi / limit_amount_yi / fd_amount_ratio（封单/成交额，>10% 为强势封板）
- 板块强度：参考下方【板块强度摘要】(注意 sector_strength_source；
  ∈ {{limit_cpt_list, unavailable}}；v0.16 起插件本地兜底聚合已移除，
  unavailable 即代表当日 Tushare 概念板块涨停统计缺失，需结合其他维度判断)
- 个股板块归属（v0.16 新增）：每个候选行带 industries / concepts / regions 三个数组，
  分别是同花顺行业 / 概念 / 地域板块，元素形如 {{ts_code, name}}。一只股票常常同时挂多个
  概念，全部如实列出；判断"是否处于主线板块"时应该把 concepts 与
  sector_strength_data.top_sectors 做名称交集，而不是仅看单一 industry 字段。
- 题材内相对地位（v0.8 新增）：sec_intra_rank_by_limit_times（同题材内连板数排名，1=最高）、
  sec_first_to_limit_flag（是否同题材最早封板，1=是）、sec_is_height_board（是否同题材高度板，1=是）、
  sec_fd_amount_rank_pct（同题材封单强度分位，越大越强）。
- 梯队地位：limit_times / up_stat
- 量价：pct_chg / amount_yi / turnover_ratio / amplitude_pct（振幅过大警惕分歧炸板）；
  max_upper_shadow_ratio_5d_pct（近 5 日最大上影占收盘价比，>3% 警惕冲高回落）。
- 形态：ma5 / ma10 / ma20 / ma_bull_aligned（多头排列时增强）
- 历史基因：up_count_30d（近 30 日涨停次数）/ up_stat
- 资金流摘要（v0.8 新增，替代原 prev_moneyflow 数组）：mf_net_t_yi（T 日净流入亿）、
  mf_net_5d_sum_yi、mf_consecutive_positive_days、mf_net_to_amount_pct（净流入/成交额归一化，
  > 3% 为偏强）、mf_large_order_strength_pct（大单+超大单买入/成交额）、mf_divergence_flag
  （=1 表示大单买入强但净流入 ≤ 0，警惕主力分歧）。
- 市场情绪：参考下方【市场摘要】中 limit_step_trend / yesterday_failure_rate / yesterday_winners_today
{lgb_block}
{_SCREENING_CYQ_LHB_BLOCK}
- 风险：是否一字板 / 过度连板 / 题材孤立 / 缺数据

【evidence 要求】"""


def _build_screening_lgb_block(
    *, lgb_min_score_floor: float | None, include_decile: bool
) -> str:
    """Compose the LGB block for the screening system prompt.

    Two orthogonal switches:
      * ``lgb_min_score_floor=None`` → drop the floor-exception sentence.
      * ``include_decile=False`` → drop the ``lgb_decile`` mention; LLM only
        sees ``lgb_score`` references. Must match ``data._attach_lgb_scores``
        which actually omits the ``lgb_decile`` field from candidate rows when
        the same flag is off.
    """
    if include_decile:
        head = _SCREENING_LGB_HEAD[:-1] + "；" + _SCREENING_LGB_DECILE_LINE
    else:
        head = _SCREENING_LGB_HEAD
    if lgb_min_score_floor is not None:
        floor_repr = f"{lgb_min_score_floor:g}"
        floor = _SCREENING_LGB_FLOOR_BLOCK.format(floor=floor_repr)
    else:
        floor = ""
    return head + floor + _SCREENING_LGB_TAIL


def build_screening_system(
    *,
    lgb_min_score_floor: float | None = 30.0,
    include_decile: bool = True,
) -> str:
    """Render 强势初筛 system prompt with the LGB §8.1 paragraph.

    ``lgb_min_score_floor=None`` → omit the soft-floor sentence (the model still
    sees the lgb_score description, but no numeric threshold is suggested).
    ``include_decile=False`` → omit the ``lgb_decile`` reference entirely; pair
    with ``_attach_lgb_scores`` dropping the field from candidate rows.
    """
    block = _build_screening_lgb_block(
        lgb_min_score_floor=lgb_min_score_floor,
        include_decile=include_decile,
    )
    return _screening_system_with(block) + _SCREENING_TAIL


_SCREENING_TAIL = """

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
- selected:        true 或 false（建议性「强势推荐」标签；true=重点关注。**不影响**是否进入下一轮——全部候选都会进入连板预测）
- score:           0–100 的浮点数
- strength_level:  必须是 "high" / "medium" / "low" 三选一
- evidence:        每只 1–4 条，每条 4 个字段不可省
- risk_flags:      空数组或字符串数组，最多 5 条
- missing_data:    必须是 data_unavailable 的**子集**（原样选取字符串）；data_unavailable 为 [] 时本字段必须为 []；候选行内 null 值不算缺失，不得写入
- 本批每只候选股都必须出现在 candidates 中，candidate_id 与输入完全一致，不可漏不可加。
"""


# Backward-compatible constant — reflects the LubConfig defaults
# (lgb_min_score_floor=30.0, lgb_decile_in_prompt=True). Pipelines that want a
# different floor / decile-toggle must call ``build_screening_system(...)``
# directly.
SCREENING_SYSTEM = build_screening_system(
    lgb_min_score_floor=30.0,
    include_decile=True,
)


def screening_user_prompt(
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
    """Render the 强势初筛 user prompt for one batch."""
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
            "selected=true 仅表示「强势推荐」建议标签（不影响是否进入下一轮，全部候选都会进入"
            "连板预测）；rationale ≤ 80 字。"
        ),
    )


# ---------------------------------------------------------------------------
# 连板预测 (continuation prediction)
# ---------------------------------------------------------------------------

_PREDICTION_SYSTEM_TEMPLATE = """\
你是一个 A 股盘后涨停复盘研究助手，正在执行第二轮"次日连板/高位溢价预测"。

【场景前提】（与第一轮一致）
- 当前是【盘后】基于 T 日已经收盘的涨停池，对【T+1 次日连板/高位溢价】候选进行预测。
- 输入均为盘后结构化数据；**不会**提供集合竞价、盘中实时盘口、实时封单撤单、新闻公告等。
- 你的任务是给每只候选股输出一个「次日重点关注 / 观察 / 回避」的预测决策。

【硬性纪律】（与第一轮一致）
1. 严禁使用外部搜索或任何未提供的数据。
2. 严禁编造盘口、龙虎榜席位（除非输入中明确提供）、消息面、传闻、ETF 申赎流向。
3. 输入清单中的每一只标的都必须出现在 candidates 数组中，candidate_id 原样回传。
4. missing_data 字段语义【严格定义，禁止泛化】：
   a) missing_data 必须是输入中 data_unavailable 列表的**子集**——只能从 data_unavailable 中的字符串原样挑选，**严禁**填入任何未出现在 data_unavailable 中的字段名（包括但不限于 ma5/ma10/ma20/ma_bull_aligned/up_count_30d/limit_amount_yi/mf_net_5d_sum_yi 等候选行内派生字段）。
   b) 当 data_unavailable 为 [] 时，missing_data **必须**为 []。
   c) 候选行内某字段值为 null **不是**"数据缺失"，而是合法事实（如新股不足 30 日历史、未上龙虎榜的 lhb_*、Tushare 未返回 limit_amount 等）；这种情况**不写入 missing_data**，仅在 key_evidence 中避免引用该字段即可。
   d) 信息不足时通过下调 confidence、在 rationale 中说明来表达不确定性，**不要**用 missing_data 表达"我不确定"。严禁猜测。
5. 仅输出 JSON。

【判断重点】
- 是否处于主线强势板块：sector_strength_source ∈ {limit_cpt_list, unavailable}
  （v0.16 起仅保留 Tushare 官方源；unavailable 时此维度无信号，须依赖个股板块归属判断）。
  把候选行 concepts 数组（同花顺概念，v0.16 新增）的 name 与
  sector_strength_data.top_sectors 求交集；命中即可判定"处于主线"。
  industries / regions（v0.16 新增）作为辅助上下文，单独命中权重低于 concepts。
- 是否为板块龙头或具备空间板地位（参考 limit_step 全市场最高连板数）；候选行的
  sec_intra_rank_by_limit_times=1 / sec_is_height_board=1 / sec_first_to_limit_flag=1
  都是同题材龙头/高度板的硬证据。
- 封板质量是否支持次日溢价 (fd_amount_yi、fd_amount_ratio、open_times、first_time、
  sec_fd_amount_rank_pct)。
- 资金近 5 日是否持续确认：参考 mf_consecutive_positive_days（连续净流入天数）、
  mf_net_5d_sum_yi（5 日净流入合计，亿）、mf_net_to_amount_pct（T 日净流入/成交额，归一化）。
  关注 mf_divergence_flag=1（大单买入强但当日净流入 ≤ 0，主力可能在分歧出货）。
- 风险：高位加速 / 连续一字 / 流动性不足 / max_upper_shadow_ratio_5d_pct 过大（冲高回落）。
- 市场亏钱效应（market_summary.yesterday_failure_rate.interpretation == 'high'）下，
  所有 confidence 自动下调一档（high → medium，medium → low），rationale 需明示。
- 涨停梯队拉升（market_summary.limit_step_trend.interpretation == 'spectrum_lifting'）下,
  最高板地位的标的可适度上调 continuation_score；score 仍受 0–100 上限约束。
- 不允许引用 missing_data 中的字段；可引用所有派生字段
  （amplitude_pct / fd_amount_ratio / ma_* / up_count_30d）。
- LightGBM 量化分（lgb_score__PREDICTION_DECILE_REF__）作为 continuation_score 的统计学锚点之一：
  · lgb_score = LightGBM 的**未校准模型排序分**（数值高 = 模型相对更看好「T+1 最高价 ≥ T 收盘价 × (1+阈值%)」触发），
    v0.13.0 起不再用"概率"二字描述（未做 Platt / Isotonic 校准，数值绝对水平不可信）；30/70 等阈值仅作排序参考；
    数值越大越倾向次日高位溢价/连板，但**不等价于"必涨停"或"可实现收益"**——盘口风险信号仍优先。
  · lgb_score ≥ 70 的标的可适度上调 confidence；但若同时存在 cyq_winner_pct > 70 / 高位连板等
    分歧风险，仍需下调。
  · lgb_score < __PREDICTION_LGB_FLOOR__ 的标的若你给出 top_candidate，rationale 必须明确写出"为何超越模型判断"，
    且该理由只能引用输入字段（如 lhb_famous_seats_count / lhb_net_buy_yi / lu_desc / tag /
    sector_strength_source 等），不得引用任何未提供数据。
  · lgb_score 缺失（null）或本次未启用模型时，忽略此维度，按其他证据评估。
  · 引用时 field 可以是 lgb_score__PREDICTION_DECILE_FIELD__，value 必须填标量（分数 / 分位数）。
- 筹码维度（参考候选行 cyq_winner_pct / cyq_top10_concentration /
  cyq_avg_cost_yuan / cyq_close_to_avg_cost_pct）：
  · cyq_winner_pct > 70% 视为"获利盘抛压重"，下调 confidence；
    cyq_close_to_avg_cost_pct < -10% 视为"严重套牢盘解套"，谨慎评估；
    cyq_top10_concentration > 60% 视为"筹码高度集中"，可作为正面 evidence。
  · 仅当数据存在时引用；missing_data 中的字段不得引用、不得编造结论。
- 龙虎榜（参考候选行 lhb_net_buy_yi / lhb_inst_count / lhb_famous_seats_count /
  lhb_famous_seats_text / lhb_reason_count / lhb_reasons_text，v0.12.4+ 新增后两个字段）：
  · lhb_* 全部为 null/0 表示"该股未上龙虎榜"——这是合法事实，不视为数据缺失，
    rationale 可以说"未触发龙虎榜异动"。
  · lhb_net_buy_yi 是该股当日**全部上榜原因合计**的净买入额（亿，v0.12.4 前版本只保留
    最后一行 reason 的值，已修复为 sum）。lhb_reason_count 是该股同日上榜的原因数；
    > 1 表示触发了多类型异动（如「日涨幅偏离 7%」+「机构专用」共同上榜），
    资金来源更分散。lhb_reasons_text 按 reason 净买入额降序逗号拼接（≤80 字）。
  · lhb_famous_seats_count > 0 且 lhb_net_buy_yi > 0 时，可作为"游资认可"的正面 evidence；
    lhb_net_buy_yi < 0 时不得作为正面 evidence（即便 famous_seats_count > 0 也只能作为
    中性或负面信号）。
  · lhb_famous_seats_text 是分号分隔的席位名称合并字符串，仅可在 interpretation 中
    照抄原文片段；不可推断"哪一位游资"或具体身份。lhb_reasons_text 同理仅照抄原文。
  · 作为 key_evidence 引用时，field 用 lhb_famous_seats_count / lhb_reason_count
    （value 填整数）或 lhb_famous_seats_text / lhb_reasons_text（value 填字符串原文），
    严禁把席位列表 / 原因列表当数组写入 value。

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
      "next_day_watch_points": ["<次日需要观察的 1-4 个关键点；至少 1 条，禁止 []>"],
      "failure_triggers": ["<会让预测失效的 1-4 个触发条件；至少 1 条，禁止 []>"],
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
  ★ 硬性约束：**每一个** candidate（含 prediction="avoid" / confidence="low" 的弱势标的）都必须给出
  至少 1 条 next_day_watch_points 与 1 条 failure_triggers，禁止返回 `[]`。
  对 avoid 候选，可写如「跌破前低/MA20 即确认失败」「分时反包失败/缩量阴跌」等通用要点，
  也不可省略。**返回空数组会导致整批响应被拒绝、整轮重试，请逐条自查再输出。**
- 输入清单中的每一只标的都必须出现在 candidates 中，candidate_id 与输入完全一致。
"""


def build_prediction_system(
    *,
    lgb_min_score_floor: float | None = 30.0,
    include_decile: bool = True,
) -> str:
    """Render 连板预测 system prompt with the §8.2 LGB paragraph.

    ``lgb_min_score_floor=None`` → drop the soft-floor sentence entirely; the
    rest of the LGB guidance (≥70 boost, missing-handling, evidence shape) is
    preserved.
    ``include_decile=False`` → strip every ``lgb_decile`` mention; pair with
    ``data._attach_lgb_scores`` dropping the field from candidate rows.
    """
    # v0.7 — re-derive from the raw template each call so the decile / floor
    # switches compose cleanly. Using the pre-rendered module constant would
    # make the substitutions overlap.
    raw = (
        _PREDICTION_SYSTEM_TEMPLATE.replace(
            "__PREDICTION_LGB_FLOOR__",
            f"{lgb_min_score_floor:g}" if lgb_min_score_floor is not None else "30",
        )
        .replace(
            "__PREDICTION_DECILE_REF__",
            " / lgb_decile" if include_decile else "",
        )
        .replace(
            "__PREDICTION_DECILE_FIELD__",
            " 或 lgb_decile" if include_decile else "",
        )
    )
    if lgb_min_score_floor is None:
        # Drop the floor-exception bullet (and its second wrapped line) so the
        # rest of the LGB block stays intact. The two-line shape was just
        # rendered above so the literal numeric value to strip is "30" or the
        # caller's float — we cover both.
        from re import sub as _re_sub

        raw = _re_sub(
            r"\n  · lgb_score < [^\n]+\n    且该理由只能引用输入字段[^\n]*\n",
            "\n",
            raw,
        )
    return raw


# Backward-compatible module-level constant — reflects the LubConfig defaults
# (lgb_min_score_floor=30.0, lgb_decile_in_prompt=True).
PREDICTION_SYSTEM = build_prediction_system(
    lgb_min_score_floor=30.0,
    include_decile=True,
)


def prediction_user_prompt(
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
# 全局重排 (only when 连板预测 was multi-batch)
# ---------------------------------------------------------------------------

FINAL_RANKING_SYSTEM = """\
你是一个 A 股盘后涨停复盘策略的全局排名助手——基于已经完成的次日连板/高位溢价预测结果，给出跨批次的最终排名。

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
# 辩论修订 (debate-mode revision, multi-LLM)
# ---------------------------------------------------------------------------

REVISION_SYSTEM = """\
你是 A 股盘后涨停复盘策略多 LLM 辩论中的一员。本轮你已经独立完成了"次日连板/高位溢价预测"，
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
      "next_day_watch_points": ["<1-4 个；至少 1 条，禁止 []>"],
      "failure_triggers": ["<1-4 个；至少 1 条，禁止 []>"],
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
- next_day_watch_points / failure_triggers: 各 1–4 条，**禁止为空数组 `[]`**；
  即便修订后判定为 avoid / low 的候选，也必须各保留至少 1 条（可沿用 R2 内容或写通用要点）。
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


def revision_user_prompt(
    *,
    trade_date: str,
    next_trade_date: str,
    own_predictions: list[Any],
    peers: list[tuple[str, list[Any]]],
    market_context: dict[str, Any],
) -> str:
    """Render the 辩论修订 prompt.

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
