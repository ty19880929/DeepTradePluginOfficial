# limit-up-board — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

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
