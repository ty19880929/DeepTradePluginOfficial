# market-review — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## v0.1.13 — 2026-06-03 — 修复 CapitalSection.northSummary[*] schema-prompt 契约缺口

### Fixed

- `run --llm qwen-plus` 在 §4 capital section 上踩 `LLMValidationError`：
  ```
  northSummary.0.net_inflow_yi
    Extra inputs are not permitted [type=extra_forbidden, input_value=41.47]
  ```

  根因是第 5 例 schema-prompt 契约缺口（前 4 例：v0.1.8 evidence.unit /
  evidence.value、v0.1.9 sectors.provider、v0.1.10 SectorEntry.tsCode、
  v0.1.11 SentimentSnapshotJson.series），且**三条裂缝叠加**：

  1. **schema 自身不一致 — 局部正则投毒**。`CapitalSection` 的 6 个并列字段
     里 5 个 (``northTop10Today`` / ``industryTop`` / ``conceptTop`` /
     ``stockTop`` / ``stockBottom``) 的每行都用 ``netInflowYi`` 表示净流入，
     只有 ``northSummary[*]`` (CapitalDailyRow) 反其道用 ``northMoneyYi``。
     LLM 在章节内推断"局部正则"：6 个并列槽 5 个叫 ``netInflowYi`` → 第 6 个
     也该叫 ``netInflowYi``。qwen-plus 不是失控，是"忠实于章节惯例"。
  2. **user prompt 没预整形，key 名错位强迫 LLM 改名**。
     ``build_capital_user_prompt`` 之前裸 ``_serialize(capital)``，
     ``CapitalReview`` 暴露 ``north_series`` / ``mkt_series`` /
     ``north_top10_anchor`` / ``stock_top_inflow`` / ``stock_top_outflow``
     一堆与 schema 字段名不对齐的 key (输出 schema 对应叫 ``north_summary`` /
     合并 ``north_summary[*]`` / ``north_top10_today`` / ``stock_top`` /
     ``stock_bottom``)；item 内部 ``NorthFlowRow.north_money_yi`` /
     ``MktFlowRow.main_net_yi`` / ``HsgtTopRow.net_amount_yi`` /
     ``IndustryFlowRow.pct_change_avg`` 也都需要重命名才匹配 schema。一旦
     LLM 进入"改名模式"，第 1 条的局部正则就会牵引它把 ``northMoneyYi`` 也
     "统一化"为 ``netInflowYi``。
  3. **system prompt 没锁字段表，且主动 prime 错的字段名**。原 ``_CAPITAL_SYSTEM``
     只一句 ``"north_summary 直接列每日北向净流入序列"`` —— "净流入" 直接
     camelCase → ``netInflowYi``，再加上没有任何字段允许 / 禁止清单，相当于
     提示语主动把 LLM 推向错的字段名。对照 ``_SENTIMENT_SYSTEM`` 在 v0.1.11
     后用一整段明文锁定 8 字段，capital 章节**完全没有这一段**。

### Changed

- `prompts.py::_serialize_capital_for_prompt(capital)` 新增帮手：与 v0.1.10
  ``_serialize_sectors_for_prompt`` / v0.1.11 ``_serialize_sentiment_for_prompt``
  同款"输入侧预整形"模式。按 ``trade_date`` zip ``north_series`` +
  ``mkt_series`` → 输出 ``north_summary[*]`` (``tradeDate`` /
  ``northMoneyYi`` / ``mainNetInflowYi``)；``margin_balance_yi`` /
  ``margin_delta_yi`` v0.1 无上游数据，直接省略键（schema 允许 None，
  HARD_DISCIPLINE 2 严禁引用不在输入的字段）。同时把 ``north_top10_anchor`` →
  ``north_top10_today``（``net_amount_yi`` → ``net_inflow_yi``）、
  ``stock_top_inflow`` → ``stock_top``、``stock_top_outflow`` →
  ``stock_bottom``、``industry_top`` / ``concept_top`` 每行
  ``pct_change_avg`` → ``pct_chg`` 全部预映射好。``mkt_series.retail_net_yi``
  / ``industry_bottom`` / ``concept_bottom`` / ``north_total_yi`` /
  ``lhb_series`` 输出 schema 没槽位，直接 drop——LLM 工作退化为 verbatim 复制。
- `prompts.py::_CAPITAL_SYSTEM` 新增 ``northSummary[*]`` 5 字段清单段：
  明文列允许字段 (``tradeDate`` / ``northMoneyYi`` / ``mainNetInflowYi`` /
  ``marginBalanceYi`` / ``marginDeltaYi``)，**显式 blacklist ``netInflowYi``
  及 8 个输入侧字段名** (``northSeries`` / ``mktSeries`` / ``northTotal`` /
  ``netAmountYi`` / ``mainNetYi`` / ``pctChange`` / ``pctChangeAvg``)，并
  专段说明"capital 章节其它 5 个并列字段确实都叫 ``netInflowYi``，但
  ``northSummary[*]`` 是**唯一例外**（CapitalDailyRow schema 设计如此）"——
  把"局部正则"诱因反向锁死。后 3 个 margin / mainNetInflowYi 无数据时**直接
  省略**该键的指令也明文（呼应 HARD_DISCIPLINE 2）。

### Tests

- `tests/test_prompts.py` 新增 3 条回归 + 一个 `_make_capital_fixture` 帮手：
  - `test_capital_user_prompt_north_summary_matches_output_schema_shape` —
    断言 ``north_summary[0]`` 恰含 ``trade_date`` / ``north_money_yi`` /
    ``main_net_inflow_yi`` 3 key（``margin_*`` 因无数据省略），且**不含**
    ``net_inflow_yi``；
  - `test_capital_user_prompt_strips_input_side_north_keys` — 断言
    ``capital`` 顶层 6-key allowlist + 9 个输入侧 key 被剥；
    ``north_top10_today[*]`` 已是 CapitalLeader 形状 (``net_inflow_yi`` 而非
    ``net_amount_yi``)；``industry_top[*]`` 已是 IndustryFlowRowJson 形状
    (``pct_chg`` 而非 ``pct_change_avg``)；
  - `test_capital_system_prompt_pins_north_summary_field_list` —
    ``_CAPITAL_SYSTEM`` 含 5 必填字段名 + ``netInflowYi`` 禁词 +
    "唯一例外" 反向锁定语 + 引用 ``CapitalDailyRow`` 类名锚定。

无迁移变更；纯 schema-prompt 契约修复。0.1.12 → 0.1.13 升级时框架不会执行
任何 SQL。

---

## v0.1.12 — 2026-06-03 — 修复 sectors user prompt 矩阵巨型化导致流式连接中断

### Fixed

- `run --llm qwen-plus` 在 §2 sectors section 上踩 `LLMValidationError`
  的另一种走法：``httpx.RemoteProtocolError: peer closed connection without
  sending complete message body`` —— qwen-plus 流式网关在生成中途强制断流。
  该异常**不继承** openai SDK 的 ``(APITimeoutError, APIError)``，框架
  transport 层 except 接不住，tenacity 0 次重试，section 立即 fail。

  根因是 ``SectorReview.matrix`` (~全 THS 板块 = 400–500 行 × 窗口交易日 列
  的 ``pct_chg`` 二维浮点格 + ``sector_names`` + ``sectors`` 两条 500-长度
  并列轴) 被 ``_serialize(sectors)`` 原样塞进 sectors 段 user prompt，单段
  payload 膨胀到 50–100 KB，流式网关在 chunked-transfer 中途主动关连接。
  这不是 v0.1.11 的"LLM 输出错"，是**输入端**就已经把请求体撑爆。

### Changed

- `prompts.py::_serialize_sectors_for_prompt` 在 v0.1.10 的 ``ts_code`` 剥离
  之外，额外 ``raw.pop("matrix", None)``。输出 :class:`SectorsSection` schema
  不消费 ``matrix`` 字段，其内容已聚合进 ``today_top`` / ``range_top`` /
  ``classification.*`` 的 ``pct_chg`` + ``persistence_days``，对 LLM 契约无损。
- `prompts.py::_SECTORS_SYSTEM` 把 "+ 板块强度矩阵" 改为 "+ 板块持续性天数"，
  避免 prompt 承诺 matrix 输入而实际不再投喂——否则会变成 schema-prompt
  契约的反向缺口（prompt 说有，data 实际没有）。

### Tests

- `tests/test_prompts.py` 新增 3 条用例：
  - `test_sectors_user_prompt_drops_matrix` — ``matrix`` 不在 ``decoded["sectors"]``
    里；
  - `test_sectors_user_prompt_size_under_budget` — 500 boards × 5 days
    fixture 序列化后 < 8 KB（修复前同 fixture > 50 KB）；
  - `test_sectors_system_prompt_no_longer_references_matrix` — 系统 prompt
    不再出现 "板块强度矩阵"。

无迁移变更；纯 prompt-builder 数据流瘦身。0.1.11 → 0.1.12 升级时框架不会
执行任何 SQL。

> 框架侧 transport 异常过滤过窄（``httpx.RemoteProtocolError`` 没被
> ``(APITimeoutError, APIError)`` 接住）是兜底层问题，另案处理。

---

## v0.1.11 — 2026-06-03 — 修复 SentimentSection.series[*] schema-prompt 契约缺口 + error 字段类型保护

### Fixed

- `run --llm qwen-plus` 在 §3 sentiment section 上踩 `LLMValidationError`，
  16 条 pydantic 错误堆成一份"看起来在乱跑"的响应，实际只有两个根因：

  1. **`series[*]` 输入侧 / 输出侧 schema 错位**——`SentimentSnapshotJson`
     输出 schema 要求 `tradeDate` / `scoreOf100` / `nUp` / `nDown` /
     `medianPctChg` / `nLimitUp` / `nLimitDown` / `nZhaban` 共 8 个字段，
     其中 5 个计数字段 (`nUp` / `nDown` / `nLimitUp` / `nLimitDown` /
     `nZhaban`) **只存在于 `BreadthSnapshot`**，但 `build_sentiment_user_prompt`
     只 `_serialize(sentiment)`，把 `SentimentReview.series[*]` 原样塞进
     user prompt——其字段是 `pos_ratio` / `top_ratio` / `crash_ratio` /
     `limit_up_intensity` / `connection_health` / `n_lhb` / `north_money_yi` /
     `mean_pct_chg` / `score_0_100` / `median_pct_chg` / `trade_date`。
     qwen-plus 撞上**契约真空**（HARD_DISCIPLINE 2「不在输入中的字段一律不引用」
     vs schema 必填这 5 个），选择最保险的退路：把 user prompt 里看到的
     `sentiment.series[*]` 原样回写。结果：8× `extra_forbidden`（输入侧字段
     被回写）+ 6× `Field required`（输出字段缺失）。注意 `score_0_100` ≠
     `score_of_100`——输入侧用数字分隔、schema 用 of 分隔，`populate_by_name`
     也救不了。
  2. **顶层 `error: []` 幻觉**——`SectionBase.error` schema 是 `str | None`，
     LLM 把"没有错误"幻觉成空数组而不是 `null`。这是 v0.1.4 ~ v0.1.10 修过
     6 次的顶层字段幻觉同一族——`HARD_DISCIPLINE` 之前未明令禁过 `error`
     的类型。

  设计层面这是第四例 schema-prompt 契约缺口（前三例：v0.1.8 evidence.unit /
  evidence.value、v0.1.9 sectors.provider、v0.1.10 SectorEntry.tsCode）。
  qwen-plus 之所以反复触发，是因为它**更忠实于 prompt**——更强的 LLM 偶尔
  能蒙混过去，反而掩盖了契约缺口。

### Changed

- `prompts.py::_serialize_sentiment_for_prompt(sentiment, breadth)` 新增帮手：
  按 `trade_date` zip `BreadthSnapshot` + `SentimentSnapshot`，emit 出与
  `SentimentSnapshotJson` 完全对齐的 8 字段 `series[*]`——LLM 的工作退化为
  verbatim 复制。`avg_score` / `strongest_day` / `weakest_day` 仍透传到顶层。
  breadth 缺该日时 counters fallback 到 0（实际罕见，breadth 是 sentiment 的
  上游）。与 v0.1.10 `_serialize_sectors_for_prompt` 同款"输入侧预整形"模式。
- `prompts.py::build_sentiment_user_prompt` 签名新增 `breadth` 入参；
  `pipeline.py::_build_user_prompt` sentiment 路由同步把 `bundle.breadth`
  传进去。
- `prompts.py::_SENTIMENT_SYSTEM` 新增 `series[*]` 字段清单段落：明文列出
  8 个必填字段名 + 7 个禁止回写的输入侧字段（`posRatio` / `topRatio` /
  `crashRatio` / `limitUpIntensity` / `connectionHealth` / `nLhb` /
  `northMoneyYi` / `meanPctChg` / `score_0_100`）。特别加注 `scoreOf100`
  的"字母 O 不是数字 0"歧义——qwen-plus 之前把 `score_of_100` 错抄成
  `score_0_100` 就是栽在这。Belt-and-suspenders 与 v0.1.10 同款。
- `prompts.py::HARD_DISCIPLINE` 新增第 10 条 `error` 字段类型保护：明文
  规定 `error` 只能是字符串或 `null`，严禁 `[]` / `{}` / `""` 等任何空容器。
  覆盖全 7 个 section（`SectionBase` 共享 `error` 字段）。无错误时直接省略
  键即可，不要为凑字段硬塞空容器。

### Tests

- `tests/test_prompts.py` 新增 5 条回归测试，覆盖：
  - `test_sentiment_user_prompt_series_matches_output_schema_shape` —
    断言 `series[0]` 恰含 8 个 key，counters 来自 breadth；
  - `test_sentiment_user_prompt_strips_input_side_ratios` —
    断言 `pos_ratio` / `connection_health` / `score_0_100` 等 9 个输入侧字段
    不出现在 `series[0]`；
  - `test_sentiment_user_prompt_preserves_top_level_aggregates` —
    `avgScore` / `strongestDay` / `weakestDay` 仍透传到 `sentiment` payload 顶层；
  - `test_sentiment_user_prompt_tolerates_missing_breadth_day` —
    breadth 缺日时 counters fallback 到 0、不 KeyError；
  - `test_sentiment_system_prompt_pins_series_field_list` —
    `_SENTIMENT_SYSTEM` 含 8 必填字段名 + 3 个核心禁词 + 数字/字母歧义提示；
  - `test_hard_discipline_rule_10_pins_error_field_type` —
    HARD_DISCIPLINE 含 `error` / `null` / `空数组` / `空对象` / `空字符串`。
- 原 `test_sentiment_user_prompt_serializes_series` 改名为
  `_make_sentiment_breadth_pair` 帮手 + 三条更具体的断言。

无迁移变更；纯 schema-prompt 契约修复。0.1.10 → 0.1.11 升级时框架不会执行
任何 SQL。

---

## v0.1.10 — 2026-06-03 — 修复 SectorsSection.SectorEntry.tsCode schema-prompt 契约缺口

### Fixed

- `run --llm qwen-plus` 在 §2 sectors section 上踩 `LLMValidationError`，
  30 条 `extra_forbidden` 全部落在 `today_top` / `range_top` /
  `classification.new_mainline` / `classification.relay` / `classification.fading`
  的每一个 `SectorEntry` 上，``tsCode`` 字段被原样写入。根因不是 LLM 凭空
  乱跑——内部 `metrics.sectors.SectorEntry` 数据类带 `ts_code`（板块自身的
  THS 指数代码，如 `883424.TI`）用于 DB / matrix 索引，但 LLM 输出的
  `schemas.SectorEntry` **故意不带**——板块由 `name` 标识，``leader_ts_code``
  是板块内**龙头股票**代码（不同概念）。`build_sectors_user_prompt` 直接
  `_serialize(sectors)` → `asdict()` 把 `ts_code` 原样喂给 LLM，qwen-plus
  忠实回写为 `tsCode`，schema 以 30× `extra_forbidden` 拒收。

### Changed

- `prompts.py::_serialize_sectors_for_prompt(sectors)` 新增帮手：在交给 LLM
  之前从 `today_top` / `range_top` / `new_mainline` / `relay` / `fading` 五栏
  每一项里剥除 `ts_code`。`matrix` 故意保留——其 `sectors` 轴是 2-D 索引轴，
  与 `sector_names` + `values` 并列，LLM 不会误当成 SectorEntry 数组。
- `prompts.py::_SECTORS_SYSTEM` 明文说明 SectorEntry 无 `tsCode` 字段，
  区分 `leaderTsCode`（龙头股票）vs 板块自身 ts_code（不同概念），并警告
  即便在矩阵 / prevContext 偶然看到 ts_code 也**绝不**许可回写。

无迁移变更；纯 schema-prompt 契约 + prompt builder 数据流修复。0.1.9 →
0.1.10 升级时框架不会执行任何 SQL。

---

## v0.1.9 — 2026-06-02 — 修复 SectorsSection.provider 数据源元字段被 LLM 幻觉填充

### Fixed

- `run --llm qwen-plus` 在 §2 sectors section 上踩 `LLMValidationError`：
  ```
  SectorsSection.provider
    Input should be 'ths' or 'dc' [type=literal_error,
    input_value='板块轮动分析师', input_type=str]
  ```
  根因不是 LLM 不听话，而是 schema / prompt **角色错位**：
  - `SectorsSection.provider: Literal["ths", "dc"]` 语义上是「板块行情数据
    来自同花顺 (THS) 还是东方财富 (DC)」的**数据源元信息**，唯一权威来源是
    `MrConfig.sector_provider`，**从未经过 LLM**；
  - 但它被放进了 LLM 响应 schema 的允许字段，并且 `_SECTORS_SYSTEM` 的「8
    字段白名单」把它**列为必填**；
  - prompt 既没解释 `provider` 含义、也没给合法取值，user prompt 里也没有
    任何 `provider` 字段可供回声；
  - prompt 开头第一句正好是 "你是板块轮动分析师"。LLM 在「必填 + 无说明 +
    无回声源」真空里，把 `provider` 投射成"服务提供者/角色名"，于是直接写
    `"provider": "板块轮动分析师"`。这与 v0.1.4 – v0.1.8 修过的"顶层字段
    幻觉"是同一类**契约真空导致的虚构填充**，只是这次被填的字段是元信息
    而非业务字段。
- 副作用清理：之前即便 LLM 偶然填对 `"ths"`，渲染到报告里的 `_板块体系：…_`
  也只反映 schema 默认值，对 `mr_config.mr.sector_provider` 的用户覆盖
  视而不见 —— `settings set sector_provider dc` 后的 run 仍写 "ths"。本次
  一并修掉。

### Changed

- `prompts.py::_SECTORS_SYSTEM` 字段白名单从 8 字段（含 `provider`）改为 7
  字段（不含 `provider`），并把 `provider` 加入「严禁额外加」黑名单，同时
  说明它是「数据源元信息（THS / DC），由调用方根据 MrConfig 填入，**绝对
  不属于 LLM 输出**」。让 LLM 知道这字段为何不该出现，而不是默念字段名。
- `config.py` 新增 `load_mr_config(db)` 助手：以 `MrConfig()` 默认为底，套上
  `mr_config` 表的 `mr.<field>` 覆盖；`runner.py` 之前的 7 个 `compute_*`
  及 LLM 阶段从未实际读过 `mr_config`，这个助手填上了空缺，也复用了 cli
  `_settings_show` 的 override 加载逻辑。Malformed JSON / 未知 key 静默忽略。
- `runner.py::MrRunner._inject_sector_provider` 在 `run_sections` 返回之后、
  `_persist_stage_results` 写库之前，从 `load_mr_config(self._rt.db)` 读出
  实际 `sector_provider` 并覆盖 `section_results["sectors"].schema.provider`。
  `load_mr_config` 异常静默降级（保留 schema 默认 `"ths"`），不会因配置表
  问题阻塞报告渲染。

无迁移变更；纯 schema-prompt 契约 + runner 数据流修复。0.1.8 → 0.1.9 升级时
框架不会执行任何 SQL。

---

## v0.1.8 — 2026-06-02 — 修复 OverviewSection.findings[*].evidence schema-prompt 契约缺口

### Fixed

- `run --llm qwen-plus` 在 §1 overview section 上踩 `LLMValidationError`，同一
  响应里同时违反两条 `EvidenceItem` 约束：
  - `findings.{1,2}.evidence.{*}.unit  Input should be a valid string`
    (`input_value=None`)
  - `findings.3.evidence.0.value.{str,int,float}  Input should be a valid …`
    (`input_value=['stagnant_on_high_volume', 'limit_down_spread']`)
  根因不是 LLM 凭空乱跑，而是 `EvidenceItem` 的 schema 与 `HARD_DISCIPLINE`
  之间存在两处契约缺口（v0.1.4 ~ v0.1.7 都只补了顶层字段幻觉，没人下探到
  `findings[*].evidence[*]` 内部）：

  1. **`unit` 没有逃生通道** — `EvidenceItem.unit` 在 schema 里是强制非空字符串
     (`Field(min_length=1)`)，但 prompt 第 3 条只声明了 4 元组形状，**没**告诉
     LLM 分类型 evidence（risk_signal 名 / themeTag / marketTone 标签 /
     ts_code 等）该填什么。LLM 看到 ``marketTone="结构性上涨"`` 这种纯标签
     evidence 时只能选 `null`，撞 schema。讽刺的是 `HeadlineMetric.unit` 同样
     约束下 docstring 写了 `"none" allowed for categorical labels`，但 docstring
     进不了 prompt，且 `EvidenceItem` 连这行 docstring 都没有。
  2. **`value` 没有"拆条"逃生通道** — schema 限定 value 为标量
     (`str|int|float|None`)、prompt 第 6 条也说"严禁数组或对象"，但当 LLM
     想在**一条 finding** 里引用**多个并列的 signal 名 / ts_code**作为
     evidence 时，规则只说"不能用数组"，没说"应该拆成多条 evidence 项"——
     在"信息完整"和"schema 合规"二选一时，LLM 挑了前者，硬塞列表进去。
  3. §1 Overview 是唯一**横跨全部 7 个指标域**的 section，evidence 天然引用
     最多分类型聚合 (`triggered_risk_signals` / `theme_tags` …)，所以这两个
     缺口同时炸在它身上，§2..§7 都是单域、evidence 以数值为主，撞不上。

### Changed

- `schemas.py::EvidenceItem.unit` 从 `str` 改为 `str | None`（默认 `None`，
  保留 `max_length=16`、删除 `min_length=1`）。schema 与现实对齐：分类型
  evidence 本就没单位，不该被迫硬编。Numeric evidence 仍**应当**填具体单位，
  prompt 强制了这条；schema 上允许 `None` 是给 LLM 的"诚实退路"，不是默认行为。
- `prompts.py::HARD_DISCIPLINE` 第 3 条拆分**数值型** vs **分类型** evidence：
  数值型必须填具体单位符号（`"%"` / `"亿"` / `"家"` / `"分"` … ≤ 16 字），
  分类型 unit 必须填 `null` 或省略键、严禁空字符串、严禁伪单位（``"标签"`` /
  ``"个"`` 凑数会扭曲 interpretation 语义）。把"如何合规"翻译给 LLM。
- `prompts.py::HARD_DISCIPLINE` 第 6 条新增"拆条"逃生通道：明文说明若一条
  finding 需要引用多个并列对象，**必须拆成多条 evidence 项**——每项各引用
  一个具体值；并引用 `Finding.evidence` 上限 5 项的约束让 LLM 主动取舍。
- `render.py::_render_evidence` 同步把单位 falsy 判定从 `== "none"` 改为
  `in (None, "none")`，避免 `unit=None` 时 f-string 打印字面 `"None"`。
- `tests/test_schemas.py` 新增 `test_evidence_item_unit_nullable_for_categorical`
  / `test_evidence_item_unit_still_length_capped` / `test_evidence_item_disallows_dict_value`
  覆盖新约束；`tests/test_prompts.py::test_hard_discipline_lists_six_rules`
  关键字断言扩展 v0.1.8 rule 3/6 新增措辞 + docstring 同步加注 v0.1.8 来历。

无迁移变更；纯 schema + prompt 修复。0.1.7 → 0.1.8 升级时框架不会执行任何 SQL。

---

## v0.1.7 — 2026-06-02 — 修复 SentimentSection prevContext 回声幻觉 + 全章节顶层 allow-list

### Fixed

- `run --llm qwen-plus` 在 sentiment section 上踩 `LLMValidationError`：
  - `SentimentSection.market_tone  Extra inputs are not permitted` (`input_value='结构性上涨'`)
  - `SentimentSection.theme_tags   Extra inputs are not permitted` (`input_value=['指数失眞','AI算力','小盘承压','结构分化']`)
  报错中的 `'结构性上涨'` / `['指数失眞', …]` 就是同一 run 中 §1 OverviewSection
  刚输出、又通过 `_inject_prev_context` 注入回 §3 sentiment user prompt 的
  `prevContext.marketTone` / `prevContext.themeTags` —— 这是 **prompt 上下文
  回声 (context echo)**，不是 LLM 凭空编造。

  根因是 v0.1.3 / v0.1.4 / v0.1.5 同一类 "LLM 在顶层添字段" 幻觉的第四个形态，
  这次靶子是 prev_context 注入的字段被原样回吐到响应顶层：
  1. `SentimentSection` schema 顶层只有 `series / avgScore / strongestDay /
     weakestDay / moneyEffect / losingEffect / narrativeMd / findings / error`
     —— `marketTone` / `themeTags` 仅在 §1 `OverviewSection` 中存在。
  2. §3 sentiment system prompt 末句"保持 theme_tags + market_tone 论调
     一致"在表述上语义二义：LLM 在 user prompt 里看到 `prevContext.marketTone` /
     `prevContext.themeTags` 具体取值 + system 中点名"保持一致"，把"一致"
     误解为"包含/复述"。
  3. `HARD_DISCIPLINE` 第 8 条（v0.1.5 加固）禁的是 `section` / `sectionName` /
     `type` 这类**章节标识符**，没明确禁"prevContext 字段不得回写"——
     模糊指令斗不过 user prompt 里贴脸的具体取值。
  4. §3 / §4 / §6 / §7 缺像 §5 leaders 那样的**显式顶层 allow-list**，
     语义模糊处仍然存在。

### Changed

- `prompts.py::HARD_DISCIPLINE` 新增第 9 条 prev_context 回声禁令：明文禁止
  `prevContext` / `marketTone` / `themeTags` / `window` / `*Summary` /
  `*Anchor` / `sectorsContext` 等**输入侧字段**回吐到响应顶层（除非本节
  schema 显式定义同名字段，仅 §1 OverviewSection 如此）。从穷举式禁令升级
  为"输入侧 vs 输出侧"二元契约，治本。
- `prompts.py::_SECTORS_SYSTEM` / `_SENTIMENT_SYSTEM` 把含糊的"保持
  theme_tags + market_tone 论调一致"改写成"论调要与 user prompt 中的
  `prevContext.marketTone` / `prevContext.themeTags` 保持一致；prevContext
  仅用于语气校准，严禁把这些字段写入响应顶层（参见硬性纪律 9）"。
- §2 / §3 / §4 / §6 / §7 全部新增显式顶层 allow-list 段落（与 §5 leaders
  v0.1.5 同款），逐 section 列出本节 schema 顶层允许的 camelCase 字段名 +
  显式禁止 `marketTone` / `themeTags` / `prevContext` / `window` /
  `*Summary` / `*Context` 等。§5 leaders 原 allow-list 同步补齐 `marketTone` /
  `themeTags` / `prevContext` / `sectorsContext` 禁止例 + 引用纪律 8 / 9。
- `tests/test_prompts.py::test_hard_discipline_lists_six_rules` 关键字断言
  扩展 `prevContext` / `marketTone` / `themeTags` / `语气校准`；docstring
  新增 v0.1.7 rule 9 的来历。

无迁移变更；纯 prompt 修复。0.1.6 → 0.1.7 升级时框架不会执行任何 SQL。

---

## v0.1.6 — 2026-06-02 — 修复板块章节渲染显示代码而非名称

### Fixed

- 报告 sectors 章节里行业指数（``.TI`` 后缀，如 ``883422.TI``）渲染为代码字面
  而非中文名（"光模块"）。根因：``metrics.sectors._sector_names`` 只查
  ``mr_moneyflow_cnt_ths`` 一张表，但该表只覆盖**同花顺概念板块**，对
  ``.TI`` 行业指数没有任何映射；查不到时 ``SectorEntry.name`` fallback 到
  ``ts_code`` 字面，渲染层再原样输出。``mr_moneyflow_ind_ths`` 虽然有
  行业名但落表时 schema 没保留 ``ts_code`` 列（PK ``(trade_date, name)``），
  也无法用于反查。

### Changed

- 新增 ``mr_ths_index`` catalog 表（``ts_code`` PK + ``name`` / ``type`` /
  ``exchange`` / ``list_date`` / ``count``），数据来自 Tushare ``ths_index``
  接口（catalog API，``static`` cache 类别）。``data.sync_sector_quotes``
  入口处一次性拉取目录并 materialize，覆盖所有 THS 指数类型（``.TI`` 行业
  / ``.CI`` 概念 / 风格 / 主题 / 宽基）。
- ``_sector_names`` 改为「``mr_ths_index`` 主源 + ``mr_moneyflow_cnt_ths``
  退路 + 代码字面兜底」三段查询：catalog 表能覆盖到的板块直接取目录中文
  名；catalog 还没回流的（旧库或首次升级）继续用概念资金流表的反查；都
  查不到才退到代码字面。
- ``deeptrade_plugin.yaml``：``ths_index`` 加入 ``permissions.tushare_apis.required``
  + ``cache_overrides`` (``static``)；``tables`` 列表新增 ``mr_ths_index``
  条目（``purge_on_uninstall: true``）。

### Migration

- 新增 ``migrations/20260602_002_ths_index_catalog.sql``。SQL 只有一个
  ``CREATE TABLE IF NOT EXISTS mr_ths_index``；幂等、零数据迁移。0.1.5 →
  0.1.6 升级时框架自动执行。

### Tests

- ``tests/test_metrics_sectors.py`` 新增三个回归测试：``.TI`` 行业指数从
  ``mr_ths_index`` 取名、目录缺时退到 ``mr_moneyflow_cnt_ths``、两边都
  缺时退到代码字面。原有 ``_name(...)`` helper 改写为插 ``mr_ths_index``，
  并保留 ``_cnt_name(...)`` 助测函数复用 ``mr_moneyflow_cnt_ths``。
- ``tests/test_data.py`` 新增 ``test_sync_sector_quotes_calls_ths_index_catalog_once``
  —— ``sync_sector_quotes`` 一次 sync 只调一次 ``ths_index``，并用
  ``key_cols=["ts_code"]`` materialize 到 ``mr_ths_index``。
- ``tests/conftest.py::MIGRATION_FILES`` 把新迁移文件加入测试库 bootstrap
  序列。

---

## v0.1.5 — 2026-06-02 — 修复 LeadersSection 顶层 ``section`` 字段幻觉

### Fixed

- `run --llm qwen-plus` 在 v0.1.4 之后又踩到新的 `LLMValidationError`，
  失败点完全独立于前两次：
  - `LeadersSection.section  Extra inputs are not permitted`
  - `input_value='leader_identification'`
  LLM 在 LeadersSection 的 JSON 顶层凭空多塞了一个
  `"section": "leader_identification"` 的"章节标识符"字段，`LeadersSection`
  → `SectionBase` → `_StrictModel`（`ConfigDict(extra="forbid")`）以
  `extra_forbidden` 拒收。

  根因属于 v0.1.3 / v0.1.4 同一类幻觉的下一个形态——LLM 自行发明 schema
  未声明的顶层键：
  1. **`HARD_DISCIPLINE` 缺通用"顶层 extras 禁令"**：现有 7 条规则全是
     针对**已知幻觉**（top-level evidence / finding 内 narrativeMd）的
     穷举式条款，没有一条说"schema 未声明的顶层字段一律不许加"。每次
     LLM 想出新花样就要补一条规则，治标不治本。
  2. **prompt 文本里 "section" 一词被高频使用**（"section 顶层"在
     HARD_DISCIPLINE rule 5 / 7 出现三次，`_LEADERS_SYSTEM` 也提到
     "section"），LLM 把"section"误理解为 JSON 中应有的结构性
     discriminator（类似 OpenAPI 里的 `type` / `kind`）。
  3. **`LeadersSection` 顶层零必填字段**（`primary` / `secondary` /
     `min_score` / `sector_map` 全部带默认值），框架
     `LLMClient._retry_hint_for` 在零 required 时退化成无 field-name
     的通用提示，重试只是把同一份错误形状再发一遍。
  4. `_LEADERS_SYSTEM` 没像 v0.1.3 加固后的 `_SECTORS_SYSTEM` 那样**枚举
     顶层允许字段**，留下了二义性。

### Changed

- `prompts.py::HARD_DISCIPLINE` 新增第 8 条通用顶层 extras 禁令：明文
  禁止 `section` / `sectionName` / `sectionType` / `type` / `kind` /
  `name` / `title` / `id` / `category` 等任何"章节标识符"或元数据字段
  出现在 section 顶层。从穷举式禁令升级为白名单式契约，避免下一种新形态
  幻觉再次破壳。
- `prompts.py::_LEADERS_SYSTEM` 末尾新增一段，显式列出 7 个允许的顶层
  字段（`primary` / `secondary` / `minScore` / `sectorMap` /
  `narrativeMd` / `findings` / `error`），并明示与 HARD_DISCIPLINE rule 8
  的关联。
- `tests/test_prompts.py::test_hard_discipline_lists_six_rules` 关键词
  列表扩展 `章节标识符` / `sectionName` / `type` / `kind`，未来若有人把
  rule 8 误删立刻回归失败。

无迁移变更；纯 prompt 修复。0.1.4 → 0.1.5 升级时框架不会执行任何 SQL。

---

## v0.1.4 — 2026-06-02 — 修复 SectorsSection.findings[*] LLM 幻觉

### Fixed

- `run` 在 v0.1.3 之后仍然踩到 `LLMValidationError`，但这次的失败点与 v0.1.3
  修过的两处完全不同——sectors section 报：
  - `findings.0.headline   Field required`
  - `findings.0.detail     Field required`
  - `findings.0.narrativeMd Extra inputs are not permitted`
  LLM 把 section 顶层的 `narrativeMd` 错放进了每一条 `finding`，同时漏掉
  `Finding` 的两个必填字段 `headline` / `detail`。

  根因是 **prompt 从没描述过 `Finding` 的形状**：
  1. `HARD_DISCIPLINE` 第 3 条只说"evidence 放在 findings[*].evidence"，反而
     在暗示 LLM 去填 findings 数组，却没说明 finding 的其他必填字段。
  2. 第 5 条提到 `narrativeMd` 字段名，但 sectors 的 system body 没像
     leaders / style / risk_outlook 一样明写 narrativeMd 该写在 section 顶层；
     LLM 自行把它"配对"进了每条 finding，形成 `{evidence, narrativeMd}` 的
     幻觉形状。
  3. 框架 `LLMClient._retry_hint_for` 在 `ValidationError` 时调用
     `_required_field_names(schema)` —— 该函数只读 schema **顶层**
     `model_fields`，`SectorsSection` 顶层全部字段都有默认值，重试 hint
     退化成无 field-name 的通用提示，LLM 第二次重写同样的错误形状。

### Changed

- `prompts.py::HARD_DISCIPLINE` 新增第 7 条，显式钉死 `Finding` 形状：
  `{headline, detail, evidence, severity?}`，并明确禁止在 finding 内出现
  `narrativeMd` / `narrative` / `prose` / `commentary` / `text` 等叙事字段
  （schema 以 `extra_forbidden` 报错）。第 5 条同步加一句"narrativeMd 只能
  作为 section 顶层字段"。
- `prompts.py::_SECTORS_SYSTEM` 末尾补一条 `narrativeMd` 200~1200 字 /
  3~6 段在 section 顶层串联结论的说明，与 leaders / style / risk_outlook
  prompt 对齐。
- `tests/test_prompts.py::test_hard_discipline_lists_six_rules` 关键词列表
  扩展 `headline` / `detail` / `severity` / `extra_forbidden`，未来回归
  会立刻失败。

无迁移变更；纯 prompt 修复。0.1.3 → 0.1.4 升级时框架不会执行任何 SQL。

---

## v0.1.3 — 2026-06-02 — 修复 sectors / capital section LLM 校验失败

### Fixed

- `run` 在 sectors / sentiment / capital 三个 section 上稳定踩到
  `LLMValidationError`，整条 pipeline 走 placeholder 路径。诊断后定位到三个
  正交根因（一处数据层 bug + 两处 prompt 契约 bug）：
  1. **CapitalSection.stock_top 输入数据形状 ≠ 输出 schema 形状**。
     `metrics.capital.StockFlowRow` 用 `ts_code` + `net_mf_yi` 两个字段，没有
     `name`；而输出 schema `CapitalLeader` 要求 `{ts_code, name, netInflowYi}`
     且 `name` `min_length=1`。LLM 忠实复制输入即被 schema 以 `missing` +
     `extra_forbidden` 同时拒收（HARD_DISCIPLINE 第 2 条又禁止编造 `name`）。
  2. **SectorsSection.classification.* 嵌套对象结构在 system prompt 里没声明**。
     框架 `complete_json` 不会自动把 pydantic JSON Schema 注入 prompt（只有
     retry 时列一遍**顶层**必填字段名），原 prompt 只用散文说"列…的板块"，
     LLM 自然把 `new_mainline` 当成 `list[str]`（ts_code 列表）输出。
  3. **HARD_DISCIPLINE 第 3 条引导出"顶层 evidence"幻觉**。原文只说 evidence
     必须是 `{field, value, unit, interpretation}` 四元组，没说位置；schema 里
     `evidence` 只是 `Finding.evidence` 子字段，LLM 在 section 顶层另起
     `evidence: [...]` 即触发 `extra_forbidden`。

### Changed

- `metrics/capital.py`：`StockFlowRow` 字段重命名 `net_mf_yi → net_inflow_yi`
  并新增 `name`（与 `schemas.CapitalLeader` 对齐）。`_stock_flows()` SQL 改成
  `LEFT JOIN mr_stock_basic` 取名，`name` 缺失时回退到 `ts_code`
  （满足 `min_length=1` 而不编造数据）。
- `prompts.py::_SECTORS_SYSTEM`：显式声明 `today_top` / `range_top` /
  `classification.{new_mainline,relay,fading}` 每一项均为同形状的完整对象
  `{name, pctChg, ...}`，**严禁只填 ts_code 字符串**。
- `prompts.py::HARD_DISCIPLINE` 第 3 条补一句限定：evidence **只能**作为
  `findings[*].evidence` 数组项出现；section 顶层另起 `evidence` 会触发
  `extra_forbidden`。

无迁移变更；纯代码 / prompt 修复。0.1.2 → 0.1.3 升级时框架不会执行任何 SQL。

---

## v0.1.2 — 2026-06-02 — 修复 mr_moneyflow_ind_ths.name NOT NULL 约束错误

### Fixed

- `run` / `sync` 在拉取行业资金流当天整体崩溃：Tushare `moneyflow_ind_ths`
  返回的行业名字段实际叫 `industry`，而落表 `mr_moneyflow_ind_ths` 的列
  叫 `name`、PK 又是 `(trade_date, name)`（DuckDB PK 隐含 NOT NULL）。
  框架 `tushare_client.materialize` 只 INSERT「DataFrame ∩ 目标表」的列，
  `industry` 进不去、`name` 被 DuckDB 默认成 NULL → `ConstraintException:
  NOT NULL constraint failed: mr_moneyflow_ind_ths.name`。整条 run 立刻失败。

### Changed

- `data.py`：`_range` 与 `_per_day` 对齐补上 `transform=` 参数，签名为
  `transform(df, *, start, end) -> df`；新增 `_transform_ind_ths()` 工厂在
  `materialize` 之前把 `industry` 重命名为 `name`，同时剔除 `name` 为
  NULL / 空串的行（Tushare 偶发回吐）以避免再次触发 NOT NULL。
- `tests/conftest.py`：`_Materialize` 记录新增 `columns: tuple[str, ...]`
  字段，`FakeTushare.materialize` 落表前抓取 `df.columns` —— 历史的「fake 只
  记录 rows」让本 bug 在单测里完全隐形（API 没配 responder 返回空 DF → early
  return → 永远走不到 `materialize`）。新增 `_Materialize.columns` 让回归
  测试可以断言 transform 后的 schema。
- `tests/test_data.py`：新增两条 `moneyflow_ind_ths` 回归 ——
  `test_sync_window_moneyflow_ind_ths_renames_industry_to_name`
  （正向：`industry` → `name`），
  `test_sync_window_moneyflow_ind_ths_drops_null_name_rows`
  （边缘：NULL / 空 `industry` 行被过滤）。

无迁移变更；纯代码 / 测试修复。0.1.1 → 0.1.2 升级时框架不会执行任何 SQL。

---

## v0.1.1 — 2026-06-02 — 修复 mr_block_trade PK 约束错误

### Fixed

- `run` / `sync` 在含活跃大宗交易的交易日整体崩溃：`mr_block_trade` 原 PK
  `(trade_date, ts_code, buyer, seller)` 与 Tushare `block_trade` 数据语义
  不匹配 —— 该接口返回的明细没有 row-id，买/卖席位字段经常是 `机构专用`
  这类通用占位名，同一对手方同一天同一标的常出现多笔不同价 / 不同量的成交，
  原始 payload 内部就已经违反 PK 假设，DuckDB 在 INSERT 阶段抛
  `ConstraintException: PRIMARY KEY or UNIQUE constraint violation`。
  v0.1.1 通过新迁移 `20260602_001_block_trade_drop_pk.sql` 重建该表去除
  PK 约束，并加一个非唯一索引 `idx_mr_block_trade_date_code
  (trade_date, ts_code)` 供 `metrics.risk._block_trade_discount` 命中。

### Changed

- `data.py`：`block_trade` 不再走通用 `_per_day(key_cols=…)` 路径（无 PK
  后 `materialize` 的逐行 DELETE 已无意义）；改走新增的
  `_per_day_replace_by_date(db, …)` —— 每个 open day 先
  `DELETE FROM mr_block_trade WHERE trade_date=?` 再 `materialize(key_cols=None)`
  纯 INSERT，重 sync 幂等性由本路径承担。

### Migration

- 新增 `20260602_001_block_trade_drop_pk.sql`：CREATE/INSERT/DROP/RENAME
  四步重建 `mr_block_trade`（DuckDB 不支持 ALTER TABLE DROP PRIMARY KEY），
  保留任何已有数据。0.1.0 → 0.1.1 升级时由框架按 `migrations` 顺序自动应用。

---

## v0.1.0 — 2026-06-01 — 首个正式版本（PR-1 ~ PR-7）

`market-review` —— A 股市场单日 / 区间复盘插件的 MVP 释出。覆盖设计文档
[MARKET_REVIEW_DESIGN.md](../MARKET_REVIEW_DESIGN.md) 的全部 v0.1 MVP 范围
（§1.3 "v0.1 必做" 七项全部交付）。

### Highlights

- **单日 / 区间复盘**：3 种入口（隐式探最近交易日 / `--trade-date` / `--start
  ... --end`），区间长度上限 `MrConfig.max_window_days`（默认 60、硬上限 252）。
- **7 个 LLM section**：overview / sectors / sentiment / capital / leaders /
  style / risk_outlook —— 顺序编排 + `theme_tags` 传递，per-section 失败隔
  离自动回落到 `partial_failed` 而不中断后续。
- **27 个 Tushare API 全量落库**：所有 §5.2 落表矩阵的 API 一次性 sync 到
  `mr_*` 表（单日 / 区间两个粒度），`force_sync` 透传到框架 cache class 层。
- **7 个确定性 metrics 模块**：市场宽度 / 情绪温度计 (0-100) / 多口径资金流 /
  板块轮动 (geometric chain 累计 + 三栏分类) / 龙头 4 维交叉打分 / 风格大小盘
  flip 检测 / 8 个风险信号。
- **报告契约 v1.0**：strict pydantic `ReviewReportSchema` 序列化为
  `summary.json`，driveOfficial 官网首屏 + 章节展示；附 `summary.md` 本地阅
  读 + 7 个 `*.md` per-section 文件 + `metrics.json` 本地审计 +
  `llm_calls.jsonl` LLM 调用审计。
- **`summary.json` 自动上传**：通过框架 `ReportUploader.upload(plugin_name=
  "市场复盘", trade_date=window.anchor)` 投递官网；失败 / 缺文件 / 用户关
  闭等场景全部 best-effort emit event 不阻断 run。

### Test coverage

`pytest tests` —— **227 passed**：dispatch 入口契约 / 数据层 sync / 7 个
metrics 模块 / 7 个 LLM section schema / pipeline 顺序 + 容错 / 报告 schema
往返 + extras 兜底 + 失败上传 / runner 全链路 / CLI 全 5 个子命令 e2e。

---

### Added (PR-7 release polish — 设计 §18.7)

- `cli.py`：`settings set <key> <value>` 持久化覆盖（写入 `mr_config`，DuckDB
  upsert via `ON CONFLICT (key) DO UPDATE SET ... updated_at = NOW()`；用
  `NOW()` 而非裸 `CURRENT_TIMESTAMP` 绕开 DuckDB 把后者解析成列引用的 bug）；
  `settings reset [<key>] [--yes]` 删除单字段或全部覆盖（不带 `--yes` 全删
  时打印确认提示退 2，避免误清空）。`MrConfig` 字段名 + 类型校验在 set 时
  完成（已知字段集 + 默认值同类型）。
- `pipeline.run_sections` 新增 `replay: LLMReplayPolicy | None` 参数；
  `runner._build_replay_policy()` 通过框架 `policy_from_env(
  policy_from_app_config(...))` 接线，让 `DEEPTRADE_FRESH_LLM` /
  `DEEPTRADE_NO_LLM_REPLAY` / `DEEPTRADE_REPLAY_ONLY` 环境变量 + 框架
  `llm.replay.enabled/.write/.ttl_days` 配置生效。`rt.config=None` 测试路径
  自动跳过（返回 `None` = 旧版无缓存语义）。
- `tests/test_cli_e2e.py`：6 个 settings set/reset 测试。
- `tests/test_runner.py`：2 个 replay policy 测试（None 容错 + 透传到 LLM
  call 的 `replay` kwarg）。

### Added (PR-6 CLI 实装 + runner 全链路 — 设计 §5.1 / §7 / §11)

新增 `runner.py` —— 全链路编排器：
- `RunParams` 冷冻 dataclass（trade_date / start / end / force_sync / llm_provider
  / no_llm / no_upload）。
- `RunOutcome`（run_id / status / report_dir / failed_sections / error）。
- `PreconditionError` —— 用户面错误（与系统错误区分，CLI 退 2 而非 1）。

### Added (PR-6 CLI 实装 + runner 全链路 — 设计 §5.1 / §7 / §11)

新增 `runner.py` —— 全链路编排器：
- `RunParams` 冷冻 dataclass（trade_date / start / end / force_sync / llm_provider
  / no_llm / no_upload）。
- `RunOutcome`（run_id / status / report_dir / failed_sections / error）。
- `PreconditionError` —— 用户面错误（与系统错误区分，CLI 退 2 而非 1）。
- `MrRunner.execute(params)` 完整流水线（Step 0..5）；
  `MrRunner.execute_sync_only(params)` 仅 Step 0..1。
- Step 0 窗口解析：先读 `mr_trade_cal`，空则 Tushare 拉一次 materialize 重读；
  无 `--trade-date / --start / --end` 时探测 `index_daily(000001.SH)` 锚 T。
- Step 1 数据：`data.sync_window` + `data.sync_sector_quotes`。
- Step 2 指标：`build_window_universes` + 7 个 PR-3 compute_*。
- Step 3 LLM：`_compute_input_fingerprint` 64-char sha256（NaN/Inf 安全 +
  dataclass→dict + sorted keys）+ `pipeline.run_sections(rt, bundle,
  input_fingerprint=)`。失败 section 不阻断后续，按设计 §11.3 隔离。
- Step 4 报告：`build_review_report` → `write_summary_json` +
  `write_summary_md` + `write_section_files` + `dump_metrics_json` +
  `dump_llm_calls_audit`。run_status = success / partial_failed。
- Step 5 上传：`maybe_upload_summary`（best-effort，事件流入 mr_events）。
- 持久化：`mr_runs` 单行 INSERT@running → finalize 时 UPDATE
  status/summary_json/error；`mr_events` 每条 emit 都写入（seq 自增）；
  `mr_stage_results` 7 节 LLM 响应 schema JSON。
- 致命错误（exception bubble up 到 execute）→ status=failed + mr_events.LOG
  事件 + return RunOutcome；PreconditionError 直接 raise 给 CLI。

`cli.py` 全面实装（替换 PR-1 全部 stub）：
- `_open_runtime()` —— Database + ConfigService + LLMManager +
  PluginContext + build_tushare_client。tushare.token 缺失时保留
  rt.tushare=None；Step 0 会用 PreconditionError 给出清晰提示。
- `_reports_root()` 共享辅助 —— runner 与 cmd_report 单一事实源，测试可
  monkeypatch 重定向到 tmp_path。
- `run` —— 完整流水线 + rich Markdown 终端摘要。`--trade-date` /
  `--start` / `--end` / `--force-sync` / `--llm` / `--no-upload`。
- `sync` —— 仅 Step 0..1。
- `history` —— mr_runs ORDER BY started_at DESC，rich Table 展示。
  `--mode day|range` 过滤。
- `report run_id [--full] [--section X]` —— UUID 前缀解析（至少 6 位 +
  CAST 解决 DuckDB UUID LIKE 限制）+ rich Markdown 渲染 summary.json /
  summary.md / 单 section md。
- `settings show` —— MrConfig defaults + mr_config 覆盖，按字段标 source。
- `main()` 已捕获 PreconditionError → 退 2；其他 PR-1 既有路径不变。

`render.py` 新增 `render_terminal_summary(report)` —— 短摘要供 cmd_run /
cmd_report 调用。

`report/builder.py` 新增 `write_summary_json(report_dir, report)` —— 单
IO 伴侣函数（纯 builder 仍纯，写盘单独走这里）。

### Added (PR-6 测试 — 219 passed，新增 22)

- `test_runner.py` (15)：sync-only 路径 / step 事件入库 / 不调 LLM；full
  路径 success 状态 / 报告文件齐 / summary.json 可 round-trip 回
  ReviewReportSchema / 7 LLM 调用 / fingerprint 透传 / mr_stage_results
  齐 7 行 / --no-upload 跳过 / partial_failed 路径 / WindowSpecError →
  PreconditionError / tushare=None + 隐式窗口 → PreconditionError /
  fingerprint 同输入相同输出（plugin_version 不同则不同）/ reports_dir
  每次 run 不同。
- `test_cli_e2e.py` (12)：monkeypatch `_open_runtime` + `_close_db` +
  `_reports_root`；smoke 每个子命令（run / sync / history / report /
  settings）+ --full / --section + run_id 前缀 + 不存在 run_id 退 2 +
  mutex flags PreconditionError → 退 2。
- `test_cli_skeleton.py` 精简到 2 个（PR-1 留下的 stub→real 后无意义部分
  删除，保留 --help / 无参 dispatch 契约）。

### Modified

- `cli.py` —— 全文重写（PR-1 stub 5 个全部替换成真实命令体）。
- `conftest.py` —— 新增 `FakeLLMClient` / `FakeLLMManager` /
  `_default_llm_responder` 测试 fixture（与 PR-4 test_pipeline 的 fake
  收敛到一处复用）。
- `render.py` —— 新增 `render_terminal_summary`。
- `report/builder.py` —— 新增 `write_summary_json`。
- `test_cli_skeleton.py` —— 删除已不适用的 stub-exit-2 测试，只留 dispatch
  入口契约。

### Bug fixes during PR-6

- DuckDB UUID 字段不能直接 `LIKE` —— `_resolve_run_id_prefix` CAST 到
  VARCHAR 才工作。
- PreconditionError 不能在 cmd 层被吃掉转成 status=failed（exit 1），
  应直传到 `main()` 让用户面错误正确退 2。
- `_reports_root()` 提取为模块级函数（不是常量）以便测试 monkeypatch
  重定向 cmd_report 的写盘路径。

### 延后 / 已知限制

- `settings set` / `settings reset` —— PR-6 仅 `show`；写入路径由 PR-7
  polish。
- IndexReturnJson.closeSeries / amountSeriesYi 仍为空（builder 纯，PR-6
  runner 也没补 mr_index_daily 读）；PR-7 可加。
- 同样 CapitalDailyRow.margin_balance_yi / margin_delta_yi 为 None。
- LLM replay policy 未连入 `LLMReplayPolicy`；目前所有调用 disable 缓存。

### Added (PR-5 报告 schema + 上传链路 — 设计 §15)

新增 `market_review/report/` 包，三个模块对齐 lub v0.16.1 契约：

- **`report/schema.py`** —— `ReviewReportSchema` 根模型（`extra="forbid"` +
  显式 `_extras: dict` 单一前向兼容入口）+ `ReportMeta` /
  `WindowMeta` / `ReportHeadline` / `HeadlineMetric` + `MetricsBlock` 子树：
  - `BreadthSnapshotJson` — 每日宽度（ladder 键 str 化以保 JSON 合法）
  - `IndexReturnJson` — 区间 cum + closeSeries / amountSeriesYi（v0.1
    builder 留空，PR-6 runner 可补 `mr_index_daily` 数据）
  - `SectorMatrixJson` — 板块 × 日期强度矩阵
  - `MetricsLeaderRow` — 不含 `rationale`（设计 §15.6 metrics 块无 LLM prose）
  - `StyleSeriesJson` —风格序列汇总
  - `MetricsRiskSignal` — 不含 `detail`（与 PR-4 RiskSignalJson 区分；只携
    `sample_count` + `samples_top_k`）
  - `MetricsBlock` —— 全部聚合
- **`report/builder.py`** —— `build_review_report(*, status, window, breadth,
  sentiment, capital, sectors, leaders, style, risk, sections,
  failed_sections, run_id, llm_provider, plugin_version, input_fingerprint,
  generated_at, error) -> ReviewReportSchema` 纯装配（无 IO / 无 DB / 无 LLM）。
  - `_build_meta()` —— title 单日 / 区间不同模板；ISO 8601 + +08:00 CN TZ
  - `_build_headline()` —— one_liner 取 overview 首段前 120 字，失败时回落
    `"{anchor} 市场复盘"`；core_metrics 从 OverviewSection 复制
  - `_build_metrics_block()` —— PR-3 dataclass → MetricsBlock pydantic：
    ladder 键 int→str；index_returns 几何链累计；sector_matrix 直接镜像；
    capital_daily 合并 north_series + mkt_series 按 trade_date；leader_table
    合并 primary + secondary；risk_signal 把 "positive" 严重度坍塌到 "info"
    （MetricsRiskSignal 不接 positive）
  - `_typed_section()` —— 防御性 isinstance 校验，shape 不对直接报错
- **`report/upload.py`** —— `maybe_upload_summary(ctx, *, run_id, report_dir,
  window) -> Iterator[StrategyEvent]` 镜像 lub `_maybe_upload_summary` 语义：
  - 找不到 ctx / 缺 summary.json → 单条 INFO 事件 + skip
  - 调框架 `ctx.make_report_uploader(run_id=...).upload(json_path,
    plugin_name="市场复盘", trade_date=window.anchor)`
  - status="ok" → INFO；status 开头 "skipped" → INFO；其他 → WARN
  - 框架理论上 never raise；defense-in-depth try/except 兜底捕获 → WARN
    event 不抛出
- `_camel` 函数 PR-4 schemas.py 模块级私有 → 报告子树通过相对 import 复用，
  保持 snake_case Python ↔ camelCase JSON wire 一致。

### Added (PR-5 测试 — 48 个新增，全套 197 passed)

- `test_report_schema.py` (20) —— 根模型 round-trip / `_extras` 兜底 /
  extra="forbid" 触发 / inputFingerprint 必须 64-char / 失败 section 保留
  error / failed-run 状态序列化 / camelCase wire / ladder str 键 / 各
  MetricsBlock 子模型默认值。
- `test_report_builder.py` (19) —— title 模板 / one_liner 回落 / theme_tags
  传递 / ladder int→str / index_returns 几何链 + name 查表 / sector_matrix
  镜像 / 空 matrix / capital_daily 合并 / leader_table 无 rationale /
  risk_signal severity "positive"→"info" + 丢 detail / failed section /
  缺 section 报 KeyError / 错类型报 TypeError。
- `test_upload_audit_payload.py` (9) —— `_FakeUploader` 记录调用；plugin_name
  + trade_date=window.anchor 精确匹配；range 模式 trade_date 取 anchor；
  ctx=None → skipped_no_ctx；无 summary.json → skipped_no_local_file；
  status=skipped_* 直传；failed_http → WARN；uploader 构造异常 → WARN +
  skipped_uploader_init_failed；upload() 抛异常 → WARN + raised；
  generator 单条事件。

### Modified

- 无（schemas.py 在 PR-4 已完成）。

### 延后到 PR-6

- runner.py 集成上述三个模块：driving sync_window → sync_sector_quotes →
  build_*_universes → compute_* metrics → run_sections → build_review_report →
  write summary.json + summary.md + section md → maybe_upload_summary。
- IndexReturnJson 的 closeSeries / amountSeriesYi 由 runner 直读
  `mr_index_daily` 补齐。
- `CapitalDailyRow.margin_balance_yi / margin_delta_yi` 同理由 runner 读
  `mr_margin` 补齐。

### Added (PR-4 LLM section — 设计 §5.4 + §15.5)

完整化 `market_review/schemas.py` —— 7 个 section pydantic 模型 + 共用基类
（`EvidenceItem` / `Finding` / `SectionBase` / `HeadlineMetric`）。所有模型
`model_config=ConfigDict(extra="forbid", populate_by_name=True,
alias_generator=_camel)`，Python 端用 snake_case，LLM JSON 线材用 camelCase
（`narrativeMd` / `headlineMetrics` 等，设计 §15.1 约定）。

新增 `market_review/prompts.py` —— 7 个 section system prompt + user prompt
构造器 + 共享 `HARD_DISCIPLINE` 段（设计 §5.4.2 六条硬性纪律）。`build_*`
函数接受 PR-3 metric dataclass，通过 deterministic compact JSON 序列化
（`sort_keys=True, separators=(",",":")`，prompt_hash 稳定）。`prev_context`
机制把 overview 的 marketTone + themeTags 注入 §2..§7（设计 §5.4.4 论调一致）。

新增 `market_review/pipeline.py` —— `MetricsBundle` 把 7 个 PR-3 review +
window 打包；`run_sections(rt, bundle)` 按 `SECTION_ORDER` 顺序串行调用
`LLMClient.complete_json`。每节捕 Exception 容错（设计 §11.3 per-section 隔
离），失败 section 用 placeholder 实例（market_tone="未知" / 假设题占位）+
error 字符串，不阻断后续 section。Profile 默认 `thinking=False,
reasoning_effort=medium, temperature=0.3, max_output_tokens=16384`。
`input_fingerprint` 透传到每个 complete_json 调用以驱动 LLM replay 缓存。

新增 `market_review/render.py` —— 7 个 per-section markdown 渲染器 +
summary.md 顶层渲染 + 本地审计写入：
- `render_section_md(section, result)` 按 section 类型选择模板：overview
  含 headline_metrics 表 + theme_tags；sectors 含三栏分类 + Top 表；
  sentiment 含每日温度表；capital 含北向 + 行业 + 个股 + LHB 亮点；
  leaders 含 4 维 score_breakdown 列；style 含 dominant_style + flip +
  range_summary；risk_outlook 含信号表 + 展望假设 + watch / fail 列。
- `render_summary_md(run_id, window, results)` 顶层一句话结论 + 章节链接 +
  PARTIAL 横幅（当任一 section 失败时）。
- `write_section_files()` / `write_summary_md()` 文件写入。
- `dump_metrics_json()` PR-3 dataclass → metrics.json 本地审计。
- `dump_llm_calls_audit()` per-section meta（latency_ms/prompt_hash/error）→
  llm_calls.jsonl。

### Added (PR-4 测试 — 57 个新增，全套 149 passed)

- `test_schemas.py` (16) — pydantic 校验 + camelCase alias 往返 +
  extra="forbid" 触发 + 各 section 必填字段 + Literal 约束。
- `test_prompts.py` (14) — 硬性纪律六规则全列 + system prompt 拼接 +
  prev_context 注入 / 省略 + JSON 解析合法 + deterministic 输出。
- `test_pipeline.py` (9) — FakeLLM 记录调用 + 7 节顺序 + schema 配对 +
  input_fingerprint 透传 + theme_tags 注入下游 + overview 失败下游容错 +
  单节失败隔离 + meta / SectionResult 完整。
- `test_render.py` (18) — 每节 md 包含核心字段 + 错误横幅 + summary 章节链
  + PARTIAL 横幅 + 文件写入 + JSON 序列化 + 审计 JSONL 行数。

### Modified

- `schemas.py` 从 PR-1 占位（SCHEMA_VERSION + Literal 别名）扩展到完整 7
  section + 共用基类（~280 行）；保留 SCHEMA_VERSION 兼容。

### Added (PR-3 指标层 — 设计 §5.3)

新增 `market_review/metrics/` 包，7 个纯只读聚合模块（无 Tushare、无 DB 写）：

- **breadth.py** — 市场宽度（§5.3.1）。`BreadthSnapshot`（n_total / n_up /
  n_down / n_flat / n_up5pct / n_down5pct / n_limit_up / n_limit_down /
  n_zhaban / up_ladder / n_lhb / total_amount_yi / index_returns）× 窗口序
  列；`BreadthReview` 加汇总（median_up_count / strongest&weakest day）。
- **sentiment.py** — 情绪温度计（§5.3.2）。0-100 thermometer 按 §5.3.2
  权重（pos_ratio 0.30 + top_ratio 0.20 + limit_up_intensity 0.15 +
  connection_health 0.10 + crash_ratio_inv 0.10 + lhb_buy_intensity 0.10 +
  north_inflow 0.05）线性归一化加权；v0.1 用窗内 plausible 参考带替代设计建
  议的 60 日 robust z-score，self-contained 无需 lookback。
- **capital.py** — 多口径资金流（§5.3.3）。6 类：NorthFlowRow（每日 + 累
  计）/ HsgtTopRow（anchor 日 Top10）/ MktFlowRow（大盘主力 vs 散户）/
  IndustryFlowRow ×2（行业 + 概念，Top/Bottom）/ StockFlowRow（universe
  内 Top 20 净流入/出）/ LhbFlowRow（每日龙虎榜家数 + 净买）。统一 /1e4
  万元→亿 转换。
- **sectors.py** — 板块轮动（§5.3.4）。SectorEntry × today_top / range_top
  / new_mainline / relay / fading 三栏分类；SectorMatrix（按窗口累计强度
  排序）。累计强度用 geometric chain（1+r1）(1+r2)... 计算更准。
- **leaders.py** — 龙头识别（§5.3.5）。4 维交叉打分（每维 0-25）：梯队
  log₂ 归一 + 涨幅百分位 + 资金百分位 + 题材命中。候选池 = 连板≥2 ∪
  Top-50 累涨。Primary（Top-K=5）+ Secondary（Top-K=10）+ 板块映射。
- **style.py** — 风格切换（§5.3.6）。沪深 300 vs 中证 1000 大小盘
  proxy；dominant_style 分类（large_cap / small_cap / balanced / rotating）
  按 ±2pp 阈值；flip_signal 用前/后半段 big_to_small_ratio 符号反转判定。
- **risk.py** — 风险信号（§5.3.7）。8 类：high_position_drop /
  stagnant_on_high_volume / index_volume_divergence / north_capital_outflow /
  limit_down_spread / block_trade_discount / margin_balance_swing /
  yaog_topping。每条 RiskSignal(triggered/severity/detail/affected_samples)。

新增 `data.sync_sector_quotes(rt, window)` — 数据层扩展，per-day 拉
`ths_daily`（同花顺所有板块）+ anchor 日拉 `dc_index`（东财备份），落
`mr_ths_daily` / `mr_dc_index`。`ths_daily` 不接受 `trade_date` 参数所以
用 `start_date=end_date=d` 单日 range 代替。

新增 `universe.build_window_universes(db, trade_dates)` 多日 universe 字典
辅助，metric 模块按日迭代时使用。

### Added (PR-3 测试 — 36 个新增，全套 92 passed)

- `test_metrics_breadth.py` (5) — 涨跌家数 / 涨停连板 / 指数收益 / 极值
  日 / amount 千元→亿换算。
- `test_metrics_sentiment.py` (6) — 0-100 边界 / 权重和=1 校验 / 缺键 /
  万元→亿 / 极值日 / 空 universe 零快照。
- `test_metrics_capital.py` (7) — 空入轴 / 万元→亿 / anchor-only Top10 /
  行业 Top&Bottom / universe 过滤 / LHB 每日 / 大盘主力 vs 散户。
- `test_metrics_sectors.py` (6) — 空入 / today_top 排序 / range_top
  geometric chain / persistence 计数 / 三栏分类 / matrix 排序。
- `test_metrics_leaders.py` (6) — ladder log₂ 归一 / 空 universe / 候选
  池并集 / score_breakdown 4 键 / min_score 过滤 / sector_map 填充。
- `test_metrics_style.py` (6) — 空入 / 大盘主导 / 小盘主导 / balanced /
  flip 信号 / 创业板增长指数序列携带。
- `test_metrics_risk.py` (5) — 8 个信号都生成 / 妖股见顶触发 / 跌停扩散
  critical 阈值 / 北向 outflow warning / 指数量价背离。

### Modified

- `data.py` — 新增 `sync_sector_quotes()` 函数（沿用 PR-2 内 helper 风格）。
- `universe.py` — 新增 `build_window_universes()` 辅助。

### Bug fixes during PR-3

- `risk._index_volume_divergence` 调用方误传 `db` 位置参数（函数只接收
  `breadth` 关键字），编写时漏过 type-check；通过 risk 测试发现并修复。

### Added (PR-2 数据层 — 设计 §3 / §5.2)

- `calendar.py`：`TradeCalendar` 复用 lub 同款实现（设计附录 B 约定，新插件独立一份保持隔离）。
- `windows.py`：`Window` immutable dataclass + `resolve_window()`。支持单日（隐式探测 / 显式 `--trade-date`）/ 区间（`--start ... --end`）三种入口；非交易日自动 snap 到最近开市日并通过 `snapped_from` 元组暴露原始输入。窗口跨度受 `MrConfig.max_window_days` 限制（默认 60，硬上限 252）。
- `runtime.py`：`MrRuntime` 轻量 dataclass（无 debate 隔离字段，设计 §1.3 「永不引入」）+ `build_tushare_client()`。
- `universe.py`：`build_universe()` + `UniverseSnapshot`。按设计 §3.1 三重防御过滤 ST / 退市 / 停牌，按 4 大板块（主板/创业板/科创板/北交所）筛选；`MrConfig.exclude_north_exchange` 可临时关闭北交所。
- `data.py`：核心数据层。
  - `fetch_latest_trade_date()` 探测 `index_daily(000001.SH, force_sync=True)` 锚定 T，杜绝本机时钟依赖（设计 §3.3）。
  - `sync_window(rt, window, *, force_sync=False) -> SyncResult` 按设计 §5.2 落表矩阵分发 27 个 API 到 `mr_*` 表：one-shot（`stock_basic` / `trade_cal`）/ per-trade-date 全市场循环（`daily` / `daily_basic` / `moneyflow` —— Tushare 6000 行响应上限强制循环）/ per-trade-date scalar（`stock_st` / `suspend_d` / `limit_step` / `limit_cpt_list` / `top_list` / `top_inst` / `ths_hot` / `dc_hot` / `margin` / `block_trade`）/ range（`limit_list_d` / `limit_list_ths` / `moneyflow_hsgt` / `hsgt_top10` / `moneyflow_mkt_dc` / `moneyflow_ind_ths` / `moneyflow_cnt_ths`）/ per-index range（`index_daily` / `index_dailybasic` × 8 个宽基）。
  - `_transform_hot()` 把 `ts_name`/`hot` 重命名为 `name`/`hot_value` 并注入 `source` 列，把 str 类型的 `rank` 强转 int 以匹配 mr_hot 的 (trade_date, source, rank) PK。
- 修正迁移 SQL 注释中误包含 `;` 导致 `sql.split(';')` 切分错位（DuckDB 报 "syntax error at field"）；checksum 已重算并同步进 yaml。
- 测试（pytest 51 个）：
  - `tests/conftest.py` — `mr_db` fixture（应用两个迁移到 tmp DuckDB）+ `FakeTushare` 假客户端（记录全部 call/materialize 调用，配 recipe-driven 响应）。
  - `tests/test_calendar.py`（5 个）—— TradeCalendar 类型归一 / is_open / pretrade / next_open / range。
  - `tests/test_windows.py`（14 个）—— 三种模式 + snap-backward + max_window_days + 互斥 + frozen。
  - `tests/test_universe.py`（10 个）—— 板块过滤 / 双重 ST / 退市名称-or-日期 / 停牌仅当日 / list_status='L' / 组合统计。
  - `tests/test_data.py`（11 个）—— fetch_latest_trade_date / one-shot / per-day / range / per-index / `force_sync` 透传 / empty_days / hot 转换。

### Deferred to later PRs

- ``ths_daily`` / ``dc_index`` —— 同花顺 / 东财板块行情。需要板块目录（`ConceptRepository`），PR-3 sectors.py 添加同步。
- ``cyq_perf`` —— 筹码每日指标。设计 §5.2 限定仅龙头候选集（成本控制），PR-3 leaders.py 实现 per-候选拉取。
- ``moneyflow_dc`` / ``kpl_concept`` / ``kpl_list`` / ``index_weight`` —— yaml 已 required，但不在 §5.2 落表矩阵里；后续 PR 各自的 metric 模块按需 `tushare.call` 直读 Tushare 缓存。

### Added (PR-1 骨架)

首次提交。本次仅落 v0.1.0 骨架（设计文档 §18 PR-1）：

### Added

- 仓库目录 `market_review/`（outer subdir）+ `market_review/market_review/`
  内层 Python 包，命名约定与 `limit_up_board` / `accumulation_probe_washout`
  对齐（kebab-case plugin id + snake_case package name）。
- `MarketReviewPlugin`（`market_review/plugin.py`）实现 Plugin Protocol 三件套
  `metadata` + `validate_static` + `dispatch`；`validate_static` 仅做轻量
  `import config` / `import schemas` 语法校验，禁止把 typer / rich / pandas /
  tushare 拉进 `sys.modules`（回归测试见 `tests/test_plugin_validate_static.py`，
  对齐 lub v0.12.3+ 契约）。
- `cli.py` typer 骨架：暴露 `run` / `sync` / `history` / `report` /
  `settings` 子命令，全部为 PR-1 stub（输出 "尚未实现" 并退出 2），仅保证
  `deeptrade market-review --help` 路径完整。
- `config.py`：`MrConfig` dataclass（设计 §8 字段），DB-backed
  `load_config` / `save_config` 留给 PR-6。
- `schemas.py`：占位模块，先暴露 `SCHEMA_VERSION="1.0"` + `SectionName` /
  `WindowMode` Literal。完整 7 个 section pydantic 模型留给 PR-4，
  `ReviewReportSchema` 根模型留给 PR-5。
- migrations：
  - `20260601_001_init.sql` —— 设计 §9.1 ~ §9.10 全部 29 张数据表
    （股票池 / 行情 / 指数 / 涨跌停 / 资金流 / 龙虎榜 / 板块 / 热榜 /
    融资融券 / 大宗 / 筹码 / runs / events / stage_results）。
  - `20260601_002_config.sql` —— `mr_config` 用户配置表（设计 §9）。
- `deeptrade_plugin.yaml`：版本 `0.1.0`、`min_framework_version` 与 lub 对齐到
  0.14.0、31 个 Tushare API 全部声明 required + 11 个 `cache_overrides`、
  30 张表全部声明 `purge_on_uninstall: true`、`table_prefix: mr_`、依赖列出
  pandas / pyarrow / numpy / tushare（不含 LightGBM）。
- `registry/index.json` 新增 `market-review` 条目，`latest_version` 占位为
  `market-review/v0.1.0`（待 PR-7 发布时由 tag + Release 实际生效）。

### Not yet implemented (deferred per design §18)

- PR-2 数据层：universe / windows / data.py + 全部 required API 落库 + 单测。
- PR-3 指标层：metrics/{breadth, sentiment, capital, sectors, leaders, style, risk}.py。
- PR-4 LLM section：schemas / prompts / pipeline / render。
- PR-5 报告 schema + 上传链路：report/{schema, builder}.py + runner 上传调用。
- PR-6 CLI 实现：runner / dashboard / settings / 终端摘要。
- PR-7 Release：CHANGELOG 收尾 + `market-review/v0.1.0` tag。
