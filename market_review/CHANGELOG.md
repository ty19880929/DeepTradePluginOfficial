# market-review — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## v0.1.0 — Unreleased — 骨架（PR-1）

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
