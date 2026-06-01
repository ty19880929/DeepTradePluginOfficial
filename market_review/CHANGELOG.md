# market-review — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## v0.1.0 — Unreleased — 骨架 + 数据层（PR-1, PR-2）

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
