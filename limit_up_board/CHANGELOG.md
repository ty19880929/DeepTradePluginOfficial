# limit-up-board — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## v0.18.1 — 2026-05-27 — 兼容新框架严格 canonical_json（指纹 NaN/Inf/set 兜底）

新框架（deeptrade-quant ≥ 0.5，本地验证于 0.15.0）把 `deeptrade.core.fingerprint.canonical_json`
的契约改严：对 `NaN`/`Inf` 抛 `ValueError`、对 `set`/`frozenset` 抛 `TypeError`（不再
自动归一）。插件 `fingerprint.py` 在框架可用时直接复用框架原语，导致：

- **运行时隐患**：`build_input_fingerprint` 会哈希 Tushare 派生的候选浮点字段
  （新股常见 `volume_ratio`/`lgb_score` 为 NaN），在新框架下 `hash_json` 会**直接抛错**，
  使 run 指纹计算崩溃。
- CI `test_fingerprint.py` 两个断言旧"自动归一"行为的用例失败。

### Fixed

- **`fingerprint._normalize` 提为模块级、并在框架路径生效**：插件的 `canonical_json`
  现在「先归一（NaN/Inf→None、set→排序 list、dataclass→asdict、Mapping 键→str），
  再委托框架」，既保留本模块文档化的 NaN-safe 契约，又让框架的
  datetime/Decimal/numpy/BaseModel 富类型处理与版本化契约继续作为单一事实源。
  对新旧框架均兼容（0.4.2 与 0.15.0 全套测试通过）。
- 新增回归测试：`test_input_fingerprint_survives_nan_and_inf_in_candidates`
  （直接走 `build_input_fingerprint` 的 NaN/Inf 候选路径）。

## v0.18.0 — 2026-05-27 — 强势筛选→强势分析（全量进连板预测）+ 数据层确定性 + 双结论展示

应用户反馈："相同交易日、相同 LGB 模型、相同因子连续执行，连板预测结果每次都有
差异"。排查两次同日报告（`data_snapshot.json` 逐字段 diff）定位到不确定性主要在
**LLM 上游**，按影响排序：

1. **数据层特征漂移（主因）**：`data._fetch_history_window` 逐交易日抓 `daily`，
   对瞬时返回空的交易日 `continue` 静默丢弃，且空结果不写缓存（→ 每次重抓、结果
   可能不同）。这使 `daily_df` 近端窗口成员在两次运行间漂移 —— 同一交易日两次运行
   `ma10` 34/34、`ma20` 33/34、`ma5` 31/34 候选都不同，`lgb_score` 随之变化。
   结果是 `prompt_hash` 每次都变 → 框架 LLM replay 缓存永远 miss → 每次真打 LLM。
2. **漏斗放大**：强势分析阶段同时打分 + 筛选，仅 `selected==true` 进入连板预测；
   打分噪声（同一只 55 vs 70）使入选集 17 vs 28，连板预测看到完全不同的票池。
3. **LLM 采样噪声**：deepseek-v4-pro 即便 temp=0 仍非完全确定（次要）。

### Fixed — 数据层确定性

- **`data._fetch_history_window` 不再静默丢日**：新增 `retry_empty_days`，对 `daily`
  这类必需 API 的空交易日做一次 `force_sync=True` 重取（把该日写进 immutable 缓存，
  使后续运行命中冻结缓存），返回 `(frame, missing_days)`；`collect_round1` 把 `daily`
  窗口的残余缺口写入 `data_unavailable`（`daily_window_gap: …`），漂移从「静默+抖动」
  变为「确定+可见」。`lgb.dataset` 同步该行为，保持 train/serve 一致。

### Changed — 强势分析不再过滤（全量进连板预测）+ 确定性全局重排

- **连板预测对全部已分析候选运行**（`RoundResult.analyzed`）：`run_prediction` 入参
  `selected` 更名为 `candidates`；runner 单 LLM / 辩论两条路径都改为传 `analyzed`。
  候选集自此在重复运行间稳定，消除漏斗放大。
- **`selected` 降为建议性「强势推荐」标签**（advisory）：强势分析 system / user prompt
  相应改写（`PROMPT_TEMPLATE_VERSION` → `lub-prompts-v2`），不再用于淘汰候选。
- **Step 4.5 全局重排改为确定性**：`build_final_ranking_deterministic` 按
  `(-continuation_score, ts_code)` 排序合并多批结果，**不再调用 LLM**（移除第二个
  采样噪声源）；`run_final_ranking` 不再接收 `llm`。

### Changed — 报告双结论展示

- summary.md 强势分析段展示**全部已分析候选**（★ 标记强势推荐），连板预测段并列展示
  每只标的的**强势分析结论（分/级/理由）‖ 连板预测结论（分/Pred/Conf/理由）**。
- summary.json：`Counts.analyzed` 新增；`ScreeningItem.strongRecommended`、
  `PredictionCard.{strongScore,strongLevel,strongRationale,strongRecommended}` 新增
  （纯增量字段）。`round1_strong_targets.json` 改为导出全部强势分析裁决。

## v0.16.3 — 2026-05-26 — 候选筛选剔除明细从「前 3 只」改为全量

应用户反馈：summary.md / summary.json 此前只记录进入强势标的筛选前被剔除的
前 3 只股票，无法复盘"到底哪些标的、因何被排除"。现改为**完整记录全部被剔除
标的**及其剔除原因（仍按流通市值降序）。

### Changed

- **`data._apply_market_filter` 不再截断为 TOP 3**：`candidate_filter_summary`
  里的 `dropped_top3` 字段重命名为 `dropped`，遍历全部被剔除标的（去掉
  `ordered_idx[:3]` 切片），每项仍含 `ts_code / name / float_mv_yi /
  close_yuan / reasons`。
- **`render._render_candidate_filter_section`（summary.md）** 消费 `dropped`，
  小节标题由"剔除 TOP 3"改为"剔除明细（共 N 只，按流通市值降序）"，输出全部行。
- **`report.builder._build_filtering`（summary.json）** 的 `rejectedItems`
  现由 `dropped` 全量构建，不再只含前 3 项；schema 结构不变（纯增量）。

## v0.16.1 — 2026-05-25 — summary.json 静默失败链路修复

修复用户反馈"run 跑完终端正常但报告没上传到官网"的连锁缺陷。根因是
`ScreeningItem.evidence` / `PredictionCard.keyEvidence` schema 类型方向错配，
导致 `build_strategy_report` 静默抛 `ValidationError`，summary.json 不落盘，
上传链路又把"文件缺失"压成静默 skip——三层降级叠加后用户看不到任何信号。

### Fixed

- **`report/schema.py` 把 evidence 字段从 `EvidenceItem` 改成 `EvidenceItemStrict`**
  （根因）：``EvidenceItem(EvidenceItemStrict)`` 继承方向是「子类有更宽 value
  类型」，pydantic v2 拒绝把父类实例传给子类槽位；runner 产出的全是 strict
  对象，schema 槽位也得跟着 strict。symptoms 一旦出现，单 LLM 模式跑完
  summary.json 全部丢失。
- **`render.write_report` 把 `build_strategy_report` 失败回传给 runner**
  (Fix A)：返回值从 `Path` 改成 `tuple[Path, str | None]`，第二项是异常字符
  化结果；之前只 `logger.warning` 进每运行一份的本地日志，仪表盘 / 终端完全
  看不到。runner 新增 `_emit_json_build_failed` 把异常 emit 成 WARN 级
  `EventType.LOG`，前端 / 复盘脚本无需再翻 `~/.deeptrade/limit_up_board/logs/run-*.log`。
- **`_maybe_upload_summary` 增加 `json_path.is_file()` 兜底** (Fix B)：文件不存在
  时 emit 一条 `status="skipped_no_local_file"` 的 INFO 事件并 return，不再让
  框架 uploader 拿着不存在的路径走完整 HTTP 准备栈。两种触发条件：（a）
  Fix A 路径下 summary.json 没写出来（WARN 由 Fix A 先报，INFO 由 Fix B
  说明"所以跳过"）；（b）辩论模式（write_report 当前不为辩论模式生成
  summary.json，待 PR-X）。
- **`_execute_debate` 现在也调 `_maybe_upload_summary`** (Fix C)：补齐与
  `_execute_single` 的对称。当前辩论模式仍无 JSON 故会走 Fix B 的 skip
  分支并 emit INFO；未来辩论 schema 落地后自动启用上传，无需再改 runner。

### Tests

- `test_report_builder.py` 新增 `test_screening_accepts_evidence_item_strict` /
  `test_prediction_accepts_evidence_item_strict`，**直接用 `EvidenceItemStrict`
  构造 fixture**，正向覆盖 runner 真实喂给 builder 的数据形态——原有 fixtures
  全用 `EvidenceItem`（子类对象传父类槽位 OK，但反向才是生产路径），所以根因
  bug 长期没被测试发现。
- `test_upload_audit_payload.py::test_skipped_missing_file_emits_no_event` 改写
  为 `test_missing_local_summary_short_circuits`，断言新契约：文件缺失时插件
  emit 一条 `skipped_no_local_file` INFO 且**不**调框架 uploader。

### Notes

- 用户报告的具体异常：``3 validation errors for ScreeningItem.evidence.* — Input
  should be a valid dictionary or instance of EvidenceItem，input_type=EvidenceItemStrict``。
- 没改 `EvidenceItem` 类定义（保留给 `lub_stage_results` 历史 row 反序列化用，
  winrate replay 那条路径依旧依赖宽松 `value` 类型）。
- ``_maybe_upload_summary`` 的事件 payload 字段名沿用 v0.12.3+ 老约定，前端
  / 日志聚合无需变更。

## v0.15.1 — 2026-05-24 — LLM Replay 默认开启

将 ``LubConfig.llm_replay_enabled`` 默认值从 ``False`` 改为 ``True``。

### Changed

- **`LubConfig.llm_replay_enabled` 默认值 `False` → `True`**：v0.15.0 已落
  完整 Phase 3 基础设施，框架未就绪时 ``_complete_with_set_check`` 通过
  ``complete_json_supports_replay()`` 探测自动跳过 replay kwargs，所以打开
  默认值在 pre-Phase-2 框架上是 no-op；框架 Phase 2 合并后用户**不需要
  手动 settings set** 即可享受重放缓存。

### Notes

- 想保留旧行为（不读不写缓存）：`deeptrade limit-up-board settings set
  lub.llm_replay_enabled false`，或一次性 `--no-llm-replay`。
- 没有代码改动，只有一行 dataclass default 字段；无新增测试。
- ``llm_replay_write`` 默认仍 ``True``、``llm_replay_ttl_days`` 默认仍 ``None``。

## v0.15.0 — 2026-05-24 — Phase 3：接入框架 LLM Replay 缓存

针对连续执行不一致问题的 Phase 3 落地：插件层完整接入框架 `LLMReplayPolicy`
基础设施。本版本与 Phase 1 (v0.14.0) 配合后，"相同输入 → 相同输出"成为
默认行为；不依赖框架 Phase 2 的运行路径与 v0.14.0 等价（向后兼容）。

### Added

- **`limit_up_board/replay_policy.py`** (P3-PRE)：
  - `LLMReplayPolicy` dataclass —— 优先 import 框架版（`deeptrade.core.llm_client`），
    缺失时回退本地 stub，shape 与设计方案 §5.1.3 完全对齐；
  - `apply_replay_context(policy, stage_to_fingerprint=...)` —— ContextVar
    上下文管理器，借鉴 `apply_empty_array_policy` 模式，避免在多层 API
    间显式传 policy；
  - `build_replay_policy(cli=..., cfg_enabled=..., ...)` —— CLI 优先级 >
    LubConfig 默认值，决策表：`--replay-only` → 只读、`--no-llm-replay`
    → 全关、`--fresh-llm` → 不读但按 cfg 写、`cfg_enabled=True` → 灰度
    默认开；
  - `complete_json_supports_replay()` —— 运行时 inspect 框架 `LLMClient.complete_json`
    签名是否含 `replay` 形参；不支持时自动降级为 no-op。

- **`LubConfig.llm_replay_enabled` / `llm_replay_write` / `llm_replay_ttl_days`** (P3-C)：
  灰度期默认全部 `False / True / None`，即使框架就绪也保持 Phase 1 行为不变。
  `settings show / set` 自动覆盖三个新键；`validate_config` 校验 `ttl_days`
  为 `None` 或正整数。

- **CLI `--fresh-llm` / `--no-llm-replay` / `--replay-only`** (P3-B)：
  `cmd_run` 三个新 flag，三者互斥（典型用法见 README）。`--replay-only`
  在框架 Phase 2 未合并时 `PreconditionError` 提前退出（**不落 `lub_runs`
  行**，避免 audit 污染）；其余两个 flag 在 pre-Phase-2 框架下 silently
  no-op。

- **`_complete_with_set_check` attempt_meta** (P3-D)：
  meta 字典新增 `attempt_count` / `first_error_class` /
  `repair_hint_hash` / `final_prompt_hash` 四个字段，框架 Phase 2 写
  replay cache 时会一并持久化，便于复盘 set-mismatch / evidence-validation
  失败 → 自愈 → 成功的全过程。

### Changed

- **`_complete_with_set_check`** (P3-A) 新增可选 `stage=` kwarg。当传入且
  `complete_json_supports_replay()` 为 True 时，向 `complete_json` 透传
  `replay=` (从 ContextVar 取) / `stage=` / `schema_version=LLM_SCHEMA_VERSION`
  / `input_fingerprint=` (按 stage 查询)。框架不支持时**不**透传，
  即 pre-Phase-2 框架行为完全不变。
- 四个 LLM 调用点（screening / prediction / final_ranking / debate_revision）
  显式传入对应 `STAGE_*` 常量。
- `_worker_phase_a` / `_worker_phase_b` 接收 `replay_policy` +
  `input_fingerprint` 参数，在工作线程内重新进入 `apply_replay_context`
  （ContextVar 不跨 `ThreadPoolExecutor` 自动传播）。
- `RunParams` 新增 `fresh_llm / no_llm_replay / replay_only` 三个字段；
  落 `lub_runs.params_json` 便于复盘。
- `LubRunner.execute()` 在 `_record_run_start` **前**校验
  `--replay-only` 的框架支持，避免污染 run 历史。

### Notes

- 默认行为（`lub.llm_replay_enabled=False`）下，本版本运行路径与 v0.14.0
  byte-equivalent —— 灰度策略，验证稳定后通过 `settings set
  lub.llm_replay_enabled true` 启用，下个版本切换默认值。
- 框架 Phase 2 合并后，`fingerprint.py` 中 `try: from deeptrade.core.fingerprint`
  自动切换；`replay_policy.py` 同理。
- 完整新增测试 26 项：`test_replay_policy.py` (11)、
  `test_cli_replay_flags.py` (9)、`test_complete_with_set_check_meta.py` (6)，
  外加 `test_runner.py` 两个 execute() precondition 测试。

## v0.14.0 — 2026-05-24 — Phase 1：连续执行结果一致性（输入侧规范化）

针对「相同输入、连续两次执行结果不一致」问题的 Phase 1 修复批次，全部为
**插件侧低风险变更**，不依赖框架新能力。Phase 3（接入框架 LLM replay）
将在 v0.15.0 落地。

### Added

- **运行指纹 (P1-I / P1-K)**：新增 `limit_up_board/fingerprint.py`，提供
  `canonical_json` / `hash_json` / `hash_text` 与 `build_input_fingerprint()`。
  Step 1 完成后由 runner 计算 `input_fingerprint` 写入 `LubRuntime`，并在
  `summary.md` 顶部新增「运行指纹」段，列出 `input_fingerprint`、
  `LLM_SCHEMA_VERSION`、`PROMPT_TEMPLATE_VERSION`、`cache_summary` 占位。
  失败降级为 `None`，不阻塞 run。
- **版本号常量 (P1-J)**：`profiles.py` 新增 `LLM_SCHEMA_VERSION="lub-llm-schema-v1"`
  与 `PROMPT_TEMPLATE_VERSION="lub-prompts-v1"`，Schema / Prompt 模板变化时手动
  bump。Phase 3 框架 replay cache key 会消费这两个常量。
- **辩论 canonical-order 工具 (P1-H)**：`_run_providers_ordered()` helper —
  worker 实时事件按完成顺序，持久化/peer/报告按 providers 输入顺序，避免
  网络抖动改变报告顺序。

### Changed

- **温度默认值 (P1-L)**：所有 stage 的 `temperature` 由 `0.1/0.2` 降为 `0.0`
  （`fast/balanced/quality` 三档全部生效）。`STAGE_FINAL` 之前已是 0.0，保持。
- **候选股稳定排序 (P1-B)**：`collect_round1()` merge 后立即按 `(trade_date asc,
  first_time asc, limit_times desc, fd_amount desc, ts_code asc)` 排序，使用
  mergesort + `na_position="last"` 保证 NaN 行始终在末。新增私有 helper
  `_stable_sort_candidates_df()`。
- **历史序列稳定排序 + 去重 (P1-C)**：`_index_by_code()` 改为
  `sort_values(["ts_code","trade_date"], kind="mergesort")` + 按
  `(ts_code, trade_date)` 去重保留最后一行。daily / daily_basic / moneyflow /
  cyq_perf 同步生效。
- **龙虎榜 reason / 席位文本稳定 (P1-D / P1-E)**：`_aggregate_top_list_net()`
  reason 拼接增加 tie-breaker `(-net_amount, reason 升序)`；
  `_famous_seats_hits()` 输出按席位名排序。
- **finalist tie-breaker (P1-F)**：`select_finalists()` 排序键改为
  `(-continuation_score, rank, ts_code)`，`avoid` 池同步。
- **stage_results 入库顺序 (P1-G)**：`_write_stage_results()` 入库前防御性按
  `(rank or final_rank, ts_code)` 排序，与其他持久化层对齐。
- **辩论 Phase A 顺序 (P1-H)**：`_execute_debate()` Phase A 改用
  `result_by_provider[p] for p in providers` 收集；Phase B 通过共享 `survivor_map`
  自动继承 canonical 顺序。

### Fixed

- **单 LLM / sync 路径回填 `lub_runs.trade_date` (P1-A)**：v0.13.3 及之前只有辩论
  `_do_step_0_and_1` 在 Step 0 后调用 `_backfill_run_trade_date()`；`_iter_pipeline`
  和 `_iter_sync` 未调用，导致 `--trade-date` 未指定时 `lub_runs.trade_date=""`，
  `history` / `report` 按日聚合不到。现在三条路径行为一致。

### Notes

- 排序规范化**不改变**任何筛选或打分逻辑，仅消除非确定性。
- 与历史报告对比：候选/finalist 顺序与 v0.13.x 之前可能不同，分数与决定保持不变。
- Phase 3 (v0.15.0) 将接入框架 `LLMReplayPolicy`，启用 `--fresh-llm` /
  `--no-llm-replay` / `--replay-only` 三个 CLI 透传开关。

## v0.13.3 — 2026-05-24 — 报告上传链路下沉到框架

打板插件不再自带 `uploader.py`；上传 URL / 超时 / token / 全局开关全部走框架
`report.upload.*` 配置族（需要 `deeptrade-quant>=0.11.0`）。首次升级时插件
自动把旧 `lub.summary_upload_*` 配置搬到框架 + `secret_store`，搬完即清，
无需用户介入。

### Required

- `deeptrade-quant >= 0.11.0`（提供 `PluginContext.make_report_uploader` /
  `report.upload.*` 配置族 / `report_uploads` 审计表）。

### Removed

- `limit_up_board/uploader.py` 整文件；
- `LubConfig.summary_upload_enabled / summary_upload_url /
  summary_upload_timeout / summary_upload_token` 四个字段及对应校验。

### Changed

- `runner._maybe_upload_summary` 改为 `ctx.make_report_uploader().upload(...)`，
  事件 payload 字段名保持与 v0.12.3 兼容（`enabled / url / status / duration_ms /
  public_url / public_path / error_class`；同时新增 `public_index / public_date`）。
- 启动时跑一次 `migrate_legacy_upload_config`：把旧的 `lub.summary_upload_*` 行
  迁移到 `report.upload.*`（url / timeout 仅在框架仍是 `default` 时覆盖；token
  非空则一律入 `secret_store`；enabled=True 则一次性写到框架开关），完成后清掉
  旧行；幂等，重复调用为 no-op。

### Migration notes

- 用户升级后第一次跑任意 `deeptrade limit-up-board <cmd>`：旧的
  `summary_upload_enabled=True` 会被搬到 `report.upload.enabled=True`，后续请
  用 `deeptrade config set report.upload.enabled true/false` 调整。
- 想关掉上传：`deeptrade config set report.upload.enabled false`；想换
  endpoint：`deeptrade config set report.upload.url https://...`。
- `deeptrade config show` 会列出新的 `report.upload.*` 行；token 默认掩码。
