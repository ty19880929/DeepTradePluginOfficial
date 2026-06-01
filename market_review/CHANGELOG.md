# market-review — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## v0.1.0 — Unreleased — 骨架 + 数据 + 指标 + LLM + 报告 + 全链路 CLI（PR-1 ~ PR-6）

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
