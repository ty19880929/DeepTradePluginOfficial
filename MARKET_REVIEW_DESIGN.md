# market-review — A 股市场复盘插件设计文档

> 版本：草案 v0.1 · 2026-05-31  
> 状态：未实现，待评审  
> 框架契约：`deeptrade-quant >= 0.14.0`（沿用 lub v0.18.x / apw v0.7.x 的 PluginContext + LLMManager + 概念仓库三件套）  
> 文档定位：交付给后续 PR 系列实施的「完整功能 + 完整工程契约」蓝图

---

## 0. TL;DR

`market-review`（中文名：**市场复盘**）是 DeepTradePluginOfficial 仓库下的第 4 个 **strategy 型** 插件。它**不做选股**，而是：

1. 围绕「单个交易日」或「区间交易日」，把 A 股全市场（主板 + 创业板 + 科创板 + 北交所；剔除 ST / 退市整理 / 停牌）当日的**结构化数据**全量落库。
2. 在数据层之上聚合 6 大复盘维度的**确定性指标**（板块轮动 / 市场宽度与情绪 / 多口径资金流 / 龙头识别 / 风格切换 / 风险信号）。
3. 把指标喂给 LLM，分章节产出**叙述性 + 表格化的完整复盘报告**（markdown），并把每一节的结构化结果落到 `mr_stage_results` 用于审计与重渲。

用户调用形态：

```powershell
# 最近一个交易日的「日复盘」
deeptrade market-review run

# 指定单日
deeptrade market-review run --trade-date 20260530

# 区间复盘（周报 / 月报）
deeptrade market-review run --start 20260506 --end 20260530

# 仅落数据，不调 LLM
deeptrade market-review sync --start 20260101 --end 20260131

# 重渲历史 run（不重复跑 LLM）
deeptrade market-review report <run_id>
```

---

## 1. 目标与边界

### 1.1 解决什么问题

A 股交易者每个交易日（或每周/每月）都要做一次复盘，回答这些问题：

- 今天/这段时间是**什么板块**在涨？是**新主线**还是**旧主线接力**？
- 是**普涨**（赚钱效应好）还是**结构性**（指数涨但中位个股跌）？
- 钱从**哪儿**进、流向**哪儿**？北向、主力大单、行业、概念分别什么状态？
- **龙头**是谁？以**梯队 / 板块涨幅 / 资金净流入 / 龙虎榜 / 区间累计涨幅** 多维交叉看，能否锁定 1–3 只**最强龙头** + 一批**二线接力候选**？
- **风格**有没有切换？大盘 vs 小盘、价值 vs 成长、大金融/地产 vs 科技/题材？
- 有没有**风险信号**？高位股回撤、放量滞涨、大盘量价背离、北向连续净卖？
- **次日 / 下周**重点关注什么？

这些回答原本散落在十几个 Tushare API + 多张图表 + 主观经验里。本插件把它们**结构化、自动化、可追溯地**汇成一份报告。

### 1.2 与已有插件的区别（重要）

| 维度 | limit-up-board | accumulation-probe-washout | checkmate | **market-review** |
| --- | --- | --- | --- | --- |
| 输出 | T+1 候选股 ranking | 启动概率候选 | 趋势跟踪持仓建议 | **市场状态报告** |
| 时间窗口 | 单日 T | 单日 T | 单日 T | **单日 T 或区间 [T0, T1]** |
| LLM 任务 | 候选筛选 / ranking | 候选评分 | 风控/退出建议 | **多 section 叙事 + 结构化指标摘要** |
| LightGBM | 是（次日溢价概率） | 是（主升浪启动概率） | 否 | **否**（v0.1 不引入，v0.3+ 视需要） |
| 是否影响交易决策 | 直接（推荐买点） | 直接（候选） | 直接（持仓） | **间接**（复盘 → 人工决策） |
| 报告体量 | 单文件 summary.md | 单文件 summary.md | 单文件 summary.md | **多文件**：overview / sectors / leaders / capital / risk / outlook |

market-review 的位置：**为前 3 个选股/择时插件提供「市场环境」上下文**。后续 v0.X 可考虑把 market-review 的 `mr_daily_overview` 表暴露成框架级共享视图，供 lub/apw/checkmate 在 R1 prompt 中作为「市场背景」字段引用。

### 1.3 v0.1 范围 vs 后续

**v0.1 必做（MVP）**：

- 单日 / 区间复盘（区间长度 ≤ 60 个交易日）
- 6 大 LLM section（单 LLM 模式，**本插件后续也不会引入 Debate**——市场复盘是叙事报告而非选股，多 LLM 互检无收益且会破坏 narrative 一致性）
- 持久化全部原始数据（便于离线重跑）
- 命令行 markdown 输出 + 重渲
- `summary.json` 自动上传官网

**v0.2+ 暂不做**：

- LightGBM 板块涨幅预测（积分允许，但需先稳定 v0.1 指标）
- HTML / PDF / 飞书 / 邮件等第三方通道推送（v0.1 已通过框架 `ReportUploader` 上传到官网 —— 见 §15；其他通道在框架支持后再接）
- 跨区间对比（本周 vs 上周）
- 实盘盘中回声（依赖日内 API，超出 8000 积分定位）

**永不引入**：

- Debate 多 LLM 互检（理由同上）
- LLM 输入裁剪 / Top-N 截断（信息完整性优先，见 §5.4.5）

---

## 2. 顶层架构

### 2.1 仓库布局

完全沿用 monorepo 已有约定：

```
DeepTradePluginOfficial/
├── market_review/                    # outer dir = registry "subdir"
│   ├── deeptrade_plugin.yaml         # 插件清单（migrations + tables + tushare_apis）
│   ├── pytest.ini                    # pythonpath = .
│   ├── requirements-dev.txt
│   ├── CHANGELOG.md
│   ├── migrations/
│   │   ├── 20260601_001_init.sql
│   │   └── 20260601_002_config.sql
│   ├── tests/
│   │   ├── test_calendar.py
│   │   ├── test_data_universe.py
│   │   ├── test_metrics_sectors.py
│   │   ├── test_metrics_sentiment.py
│   │   ├── test_pipeline_plan.py
│   │   ├── test_prompts.py
│   │   ├── test_schemas.py
│   │   └── ...
│   ├── docs/                         # 与 limit_up_board/docs 同位
│   │   └── design.md                 # 本文档的副本（发布后同步）
│   └── market_review/                # inner Python package
│       ├── __init__.py
│       ├── plugin.py                 # MarketReviewPlugin（PluginProtocol 三件套）
│       ├── cli.py                    # typer CLI
│       ├── runtime.py                # MrRuntime dataclass
│       ├── runner.py                 # MrRunner（驱动 pipeline，写 mr_runs/mr_events）
│       ├── pipeline.py               # LLM 多 section 调度（每节一次 complete_json）
│       ├── data.py                   # 全量数据采集（Tushare → DB）
│       ├── universe.py               # 股票池构造（剔除 ST/退市/停牌）
│       ├── calendar.py               # TradeCalendar（同 lub 风格）
│       ├── windows.py                # 窗口解析（单日/区间）
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── breadth.py            # 涨跌家数 / 涨停 / 连板 / 炸板 / 晋级率
│       │   ├── sentiment.py          # 赚钱效应 / 亏钱效应 / 情绪温度计
│       │   ├── capital.py            # 资金流（北向 / 大单 / 行业 / 概念）
│       │   ├── sectors.py            # 板块涨幅榜 + 持续性 + 轮动矩阵
│       │   ├── leaders.py            # 龙头识别（多口径交叉）
│       │   ├── style.py              # 大小盘风格 / 成长价值
│       │   └── risk.py               # 高位回撤 / 量价背离 / 净卖压
│       ├── schemas.py                # Pydantic：每个 section 的输出 schema
│       ├── prompts.py                # 每个 section 的 system+user 模板
│       ├── render.py                 # JSON → markdown 多文件渲染
│       ├── report.py                 # 终端摘要 + report dir 落盘
│       ├── config.py                 # MrConfig（mr_config 表）
│       ├── cancellation.py           # 沿用 lub 的 SIGINT marker
│       ├── observability.py          # RunMetrics 计数
│       └── ui/                       # 可选：复用 lub LegacyStreamRenderer
│           ├── __init__.py
│           └── protocol.py
└── registry/index.json               # 新增 market-review 条目
```

### 2.2 Plugin Protocol 三件套

`market_review/plugin.py`：

```python
class MarketReviewPlugin:
    metadata = None  # 框架注入

    def validate_static(self, ctx):
        from . import config as _cfg   # noqa: F401  触发 dataclass 语法校验
        from . import schemas as _s    # noqa: F401  触发 pydantic 校验

    def dispatch(self, argv):
        from . import cli
        return cli.main(argv)
```

`deeptrade_plugin.yaml::entrypoint`：`market_review.plugin:MarketReviewPlugin`。

### 2.3 Runtime（MrRuntime）

```python
@dataclass
class MrRuntime:
    db: Database
    config: ConfigService
    llms: LLMManager
    plugin_id: str = "market-review"
    run_id: str | None = None
    tushare: TushareClient | None = None
    concept_repo: ConceptRepository | None = None  # 复用框架级 ths_concept_* 仓库
    input_fingerprint: str | None = None           # 与 lub 同款，便于 LLM replay
```

- 不带 `lgb_scorer` 字段（v0.1 无 LightGBM）。
- 不带 `_FrozenConfigService` / `ProviderConfigSnapshot`（本插件不会引入 debate 模式，无需 worker 隔离）。

### 2.4 框架契约

- `min_framework_version: 0.14.0`（与 lub 对齐，要求 `PluginContext.make_concept_repository` 可用）。
- `permissions.llm: true` / `permissions.llm_tools: false`。
- `permissions.tushare_apis` 详细列表见 §4。
- `table_prefix: mr_`（必须显式声明，框架默认会从 plugin_id 推导成 `market_review_`，与实际表名不符）。

---

## 3. 股票池与窗口

### 3.1 股票池规则（每个交易日构造一次）

规则原文：「主板 + 创业板 + 科创板 + 北交所，剔除 ST / 退市整理 / 停牌」。在数据层落实为：

| 维度 | 实现 |
| --- | --- |
| 来源表 | `mr_stock_basic`（=`stock_basic` 全量落库；按 `ts_code` 主键） |
| 板块过滤 | `market` ∈ {主板, 创业板, 科创板, 北交所} —— 不再排除任何板块 |
| ST 剔除 | 双重防御：(a) `name` 字段不以 `*ST` / `ST` 开头；(b) 在 `stock_st` 表里 `trade_date ∈ window` 且 `st_status` 非空的 `ts_code` 全部剔除 |
| 退市整理剔除 | (a) `name` 含「退」 → 剔除；(b) `delist_date` 不为空且 `delist_date <= window 末日` → 剔除 |
| 停牌剔除 | `suspend_d` 在 `trade_date` 上 `suspend_type='S'` → 当日剔除；区间复盘按「每一天单独构造池子」处理，停牌只剔除停牌当日 |

`market_review/universe.py::build_universe(trade_date) -> set[str]` 是单一入口，所有指标都基于它返回的池子统计。**严禁**指标模块直接 `SELECT * FROM mr_daily` 不带过滤。

### 3.2 窗口语义

复盘有 3 种用法：

| 模式 | CLI | 行为 |
| --- | --- | --- |
| **日复盘** | `run`（不带 `--trade-date`） | 自动探测最近已收盘交易日，`window = [T, T]` |
| **指定日** | `run --trade-date 20260530` | `window = [T, T]` |
| **区间复盘** | `run --start 20260506 --end 20260530` | `window = [T0, T1]`，必须都是交易日；非交易日自动向前 snap 到最近交易日，并在事件流提示 |

边界约束：

- `T1 - T0 + 1` 个交易日 ≤ `MrConfig.max_window_days`（默认 **60**，硬上限 252）。超出报错而非截断。
- 单日模式与区间模式共享 pipeline，但部分指标计算公式不同（见 §6.1）。

窗口解析在 `market_review/windows.py::resolve_window(...)`，返回 `Window(trade_dates: list[str], mode: Literal["day","range"], anchor: str)`，其中 `anchor` 是用于报告标题与文件命名的代表日期：单日=T，区间=T1。

### 3.3 最近交易日探测

完全沿用 lub 的做法（`data.py::fetch_latest_trade_date`）：用 `index_daily(ts_code=000001.SH, force_sync=True)` 探测最大 `trade_date`。不依赖本机时钟。

---

## 4. Tushare 接口清单

> **用户已开通 8000 积分覆盖范围内全部接口**，故本插件 v0.1 把所有用到的 API **一律声明为 `required`**，不设 `optional` 降级路径。运行期任一接口返回 `TushareUnauthorizedError` 视为致命错误，直接 `failed` 退出并写入 `mr_runs.error`，不再尝试半成品报告。
>
> 「接口可调但当日返回空」（如节假日 / 北交所 ETF 稀疏 / 热榜空 / 大宗无成交）属于**数据维度缺失**，按 §11.2 走「字段置 None + section 不引用」的轻度降级，仍记 `success`。

### 4.1 完整接口清单（v0.1 全部 required）

| API | 积分 | 用途 | cache class |
| --- | ---: | --- | --- |
| `stock_basic` | 120 | 股票池基础信息 | `static`（框架内置） |
| `trade_cal` | 120 | 交易日历 | `static`（框架内置） |
| `daily` | 120 | 全市场日线行情（区间复盘量最大） | `trade_day_immutable`（框架内置） |
| `daily_basic` | 2000 | 流通市值/换手率/PE/PB 等指标 | `trade_day_immutable`（框架内置） |
| `index_daily` | 200 | 宽基指数日线（上证/深成/创业板指/科创50/北证50/沪深300/中证500/中证1000/万得全A）+ 探测最新交易日 | `trade_day_immutable`（显式 cache_overrides） |
| `index_dailybasic` | 5000 | 指数当日基本面（总成交额 / 平均 PE） | `trade_day_immutable`（显式 cache_overrides） |
| `index_weight` | 2000 | 指数成分权重（沪深 300 / 中证 1000 等，用于风格分析） | `static`（框架内置） |
| `stock_st` | 2000 | ST 名单（剔除） | `trade_day_immutable`（框架内置） |
| `suspend_d` | 2000 | 停牌明细（剔除） | `trade_day_immutable`（框架内置） |
| `limit_list_d` | 2000 | 涨跌停明细 | `trade_day_immutable`（框架内置） |
| `limit_step` | 2000 | 连板天梯 | `trade_day_immutable`（框架内置） |
| `limit_cpt_list` | 5000 | 题材连板池 | `trade_day_immutable`（框架内置） |
| `limit_list_ths` | 5000 | 同花顺涨停榜 | `trade_day_immutable`（框架内置） |
| `moneyflow` | 2000 | 个股大单资金流（沪深 + 同花顺口径） | `trade_day_immutable`（框架内置） |
| `moneyflow_dc` | 5000 | 东财个股资金流（与 `moneyflow` 互验） | `trade_day_immutable`（显式 cache_overrides） |
| `moneyflow_hsgt` | 2000 | 北向资金日度汇总 | `trade_day_immutable`（显式 cache_overrides） |
| `hsgt_top10` | 2000 | 北向资金 Top10 | `trade_day_immutable`（框架内置） |
| `moneyflow_mkt_dc` | 5000 | 大盘东财口径资金流 | `trade_day_immutable`（显式 cache_overrides） |
| `moneyflow_ind_ths` | 5000 | 同花顺行业资金流 | `trade_day_immutable`（显式 cache_overrides） |
| `moneyflow_cnt_ths` | 5000 | 同花顺概念资金流 | `trade_day_immutable`（显式 cache_overrides） |
| `top_list` | 2000 | 龙虎榜个股 | `trade_day_immutable`（框架内置） |
| `top_inst` | 2000 | 龙虎榜机构席位 | `trade_day_immutable`（框架内置） |
| `ths_daily` | 5000 | 同花顺板块/概念指数日线 | `trade_day_immutable`（显式 cache_overrides） |
| `dc_index` | 5000 | 东财板块指数日线（与 `ths_daily` 互验） | `trade_day_immutable`（显式 cache_overrides） |
| `kpl_concept` | 5000 | 开盘啦概念列表 | `trade_day_immutable`（显式 cache_overrides） |
| `kpl_list` | 5000 | 开盘啦榜单（涨停/炸板/跌停/强势/自然涨停） | `trade_day_immutable`（显式 cache_overrides） |
| `ths_hot` | 5000 | 同花顺热榜 | `hot_or_anns`（框架内置） |
| `dc_hot` | 5000 | 东财热榜 | `hot_or_anns`（框架内置） |
| `cyq_perf` | 5000 | 筹码每日指标（龙头股筹码分布） | `trade_day_immutable`（框架内置） |
| `margin` | 2000 | 融资融券两市汇总 | `trade_day_immutable`（框架内置） |
| `block_trade` | 2000 | 大宗交易 | `trade_day_immutable`（框架内置） |

> 兼用 `ths_daily` 与 `dc_index` 的设计意图：板块边界两家厂商口径差异较大（同花顺更细，东财更粗），本插件 v0.1 默认 `MrConfig.sector_provider="ths"`，但**两套行情仍全量落库**，以便后续 v0.2 加「双口径校验」时直接读 DB，不再补拉历史。

### 4.2 yaml 片段

```yaml
permissions:
  tushare_apis:
    required:
      - stock_basic
      - trade_cal
      - daily
      - daily_basic
      - index_daily
      - index_dailybasic
      - index_weight
      - stock_st
      - suspend_d
      - limit_list_d
      - limit_step
      - limit_cpt_list
      - limit_list_ths
      - moneyflow
      - moneyflow_dc
      - moneyflow_hsgt
      - hsgt_top10
      - moneyflow_mkt_dc
      - moneyflow_ind_ths
      - moneyflow_cnt_ths
      - top_list
      - top_inst
      - ths_daily
      - dc_index
      - kpl_concept
      - kpl_list
      - ths_hot
      - dc_hot
      - cyq_perf
      - margin
      - block_trade
    # v0.1 全 required，不声明 optional 列表。
    # v0.13.2+：框架 API_CACHE_CLASS 表外的 API 必须显式声明 cache 策略，
    # 否则 TushareClient fallback 到 trade_day_immutable 并发 INFO 警告。
    cache_overrides:
      index_daily: trade_day_immutable
      index_dailybasic: trade_day_immutable
      moneyflow_dc: trade_day_immutable
      moneyflow_hsgt: trade_day_immutable
      moneyflow_mkt_dc: trade_day_immutable
      moneyflow_ind_ths: trade_day_immutable
      moneyflow_cnt_ths: trade_day_immutable
      ths_daily: trade_day_immutable
      dc_index: trade_day_immutable
      kpl_concept: trade_day_immutable
      kpl_list: trade_day_immutable
  llm: true
  llm_tools: false
```

### 4.3 与 limit-up-board 的接口重叠

下列 API 与 lub 同名同语义，但**插件之间 DB 表隔离**（`mr_*` vs `lub_*`），各自独立落库：`stock_basic`、`trade_cal`、`daily`、`daily_basic`、`stock_st`、`suspend_d`、`limit_list_d`、`limit_step`、`limit_cpt_list`、`limit_list_ths`、`moneyflow`、`top_list`、`top_inst`、`cyq_perf`、`index_daily`。框架的 Tushare HTTP cache 层会让相同窗口的真实请求合并，但**两个插件不能跨表互读**——这是 monorepo 既定的 Plan A 纯隔离约束。

---

## 5. Pipeline 设计

### 5.1 阶段流（`MrRunner.execute`）

```
Step 0  解析窗口（trade_date / start..end → Window）
Step 1  数据落库（universe + 全部 Tushare API；v0.1 全 required，无 optional 路径）
Step 2  指标聚合（metrics/*：breadth / sentiment / capital / sectors / leaders / style / risk）
Step 3  LLM 综合复盘（pipeline.run_sections，按 7 个 section 顺序串行调用）
Step 4  渲染与落库（render.write_report → reports_dir/<run_id>/...md + JSON）
```

每个 Step 对外发射 `StrategyEvent`（沿用 lub 的 EventType 枚举），便于 dashboard / 测试断言。

### 5.2 Step 1 — 数据落库 (`data.py`)

落库策略：**所有数据 v0.1 全量落本地表**，复盘是「事后分析」，区间内每日数据都需要可重复访问；让 Tushare cache + 本地 DB 双保险。表设计见 §9。

调用矩阵（窗口内每个交易日逐日拉，跨日数据走 start/end 整体拉）：

| 数据类别 | API | 调用粒度 | 落表 |
| --- | --- | --- | --- |
| 股票池底表 | `stock_basic` | 全市场一次 | `mr_stock_basic`（首跑全量；后续走框架内 stock_basic ttl） |
| ST 名单 | `stock_st` | start/end | `mr_stock_st` |
| 停牌 | `suspend_d`（可选） | start/end | `mr_suspend_d` |
| 行情 | `daily` | 按 trade_date 全市场（更快） | `mr_daily` |
| 基础指标 | `daily_basic` | 按 trade_date 全市场 | `mr_daily_basic` |
| 指数日线 | `index_daily` | 9 个宽基 × start/end | `mr_index_daily` |
| 指数指标 | `index_dailybasic` | 同上 | `mr_index_dailybasic` |
| 涨跌停 | `limit_list_d` | start/end | `mr_limit_list_d` |
| 连板天梯 | `limit_step` | start/end | `mr_limit_step` |
| 题材连板池 | `limit_cpt_list` | start/end | `mr_limit_cpt_list` |
| 同花顺涨停榜 | `limit_list_ths` | start/end | `mr_limit_ths` |
| 北向汇总 | `moneyflow_hsgt` | start/end | `mr_moneyflow_hsgt` |
| 北向 Top10 | `hsgt_top10` | start/end | `mr_hsgt_top10` |
| 大盘资金 | `moneyflow_mkt_dc` | start/end | `mr_moneyflow_mkt` |
| 行业资金 | `moneyflow_ind_ths` | start/end | `mr_moneyflow_ind_ths` |
| 概念资金 | `moneyflow_cnt_ths` | start/end | `mr_moneyflow_cnt_ths` |
| 个股大单 | `moneyflow` | 按 trade_date 全市场 | `mr_moneyflow` |
| 龙虎榜 | `top_list` | start/end | `mr_top_list` |
| 龙虎榜席位 | `top_inst` | start/end | `mr_top_inst` |
| 板块行情 | `ths_daily` | 同花顺所有板块 × start/end | `mr_ths_daily` |
| 板块行情备份 | `dc_index` | 东财所有板块 × start/end（仅末日） | `mr_dc_index` |
| 热榜 | `ths_hot` / `dc_hot` | start/end | `mr_hot` |
| 筹码 | `cyq_perf` | 按 trade_date 仅龙头候选集（成本控制） | `mr_cyq_perf` |
| 融资融券 | `margin` | start/end | `mr_margin` |
| 大宗交易 | `block_trade` | start/end | `mr_block_trade` |

`data.py::sync_window(rt, window, *, force_sync=False)` 是入口，内部把上述调用拆成串行任务并发射进度事件。

**关键约束**：

1. **股票池过滤一次性完成**：所有 trade_date 内的 universe 用 `mr_stock_basic` ∩ `mr_stock_st` ∩ `mr_suspend_d` 当日得到，缓存到 `Window.universes: dict[str, set[str]]`。
2. **资金/龙虎榜按 ts_code 落库后必须过滤 universe**：否则会混入 ST 数据污染指标。
3. **板块体系优先用同花顺**：`ths_daily` + 框架级 `concept_repo`（同花顺）；东财仅作 fallback 校验，**不混用避免概念冲突**。
4. **`force_sync` 透传**：`MrRunner.execute` 接 `--force-sync`，传给 Tushare client，绕过 cache class 的 fresh-once 语义。

### 5.3 Step 2 — 指标聚合 (`metrics/`)

所有指标模块的输入是「Window + 已落库的 DB」，输出是**纯 dataclass / dict 结构**，不做 IO，便于单元测试。

#### 5.3.1 breadth.py — 市场宽度

每个 trade_date 在 universe 内统计：

| 字段 | 含义 |
| --- | --- |
| `n_total` | 池中股票数 |
| `n_up`, `n_flat`, `n_down` | 涨/平/跌家数（`pct_chg` > 0.01 / |·| ≤ 0.01 / < -0.01） |
| `n_up5pct`, `n_down5pct` | 涨/跌 5% 以上家数 |
| `n_limit_up`, `n_limit_down`, `n_zhaban`, `n_dieting_open` | 涨停 / 跌停 / 炸板 / 跌停被打开 |
| `up2_count`..`up_max_count` | 2 连板..最高连板数（取 `limit_step`） |
| `first_board_total`, `first_board_success` | 首板尝试 vs 成功，**晋级率** |
| `n_lhb` | 进入龙虎榜家数 |
| `total_amount_yi` | 池子总成交额（亿） |
| `index_returns` | 9 大宽基的 `pct_chg`（dict） |

区间复盘：把每个 trade_date 的字段聚合成日期序列 `BreadthSeries`，并附加 `BreadthSummary`（区间内平均涨家数 / 涨停总和 / 最强情绪日 / 最弱情绪日）。

#### 5.3.2 sentiment.py — 市场情绪与赚钱效应

| 字段 | 计算 |
| --- | --- |
| `median_pct_chg` | 池内中位涨幅（区分赚钱效应） |
| `mean_pct_chg` | 池内均值涨幅（与中位差用于「指数 vs 个股背离」） |
| `pos_ratio` | 上涨家数占比 |
| `top_ratio` | 涨幅 > 5% 家数占比 |
| `crash_ratio` | 跌幅 > 5% 家数占比 |
| `limit_up_success_ratio` | T-1 首板在 T 是否封死涨停 |
| `connection_health` | 连板梯队完整度（有没有缺梯队的「断板」） |
| `hot_topic_count_ths` / `_dc` | 同花顺/东财热榜前 50 的成交额合计占大盘比 |
| `sentiment_score_0_100` | 综合温度计（按公式见下） |

`sentiment_score_0_100` 公式（v0.1，可在 mr_config 调权重）：

```
sentiment = 0.30 * pos_ratio_norm
          + 0.20 * top_ratio_norm
          + 0.15 * limit_up_intensity_norm        # n_limit_up / n_total
          + 0.10 * connection_health_norm         # 梯队完整度 0~1
          + 0.10 * (1 - crash_ratio_norm)
          + 0.10 * lhb_buy_intensity_norm
          + 0.05 * north_inflow_norm
```

各 `_norm` 用过去 60 个交易日 robust z-score（median + MAD）映射到 0~100。窗口不足 60 日时用全部可得历史。

#### 5.3.3 capital.py — 多口径资金流

按口径分类：

1. **北向**：`moneyflow_hsgt` → 日度净流入 / 累计 / Top10 个股
2. **大盘**：`moneyflow_mkt_dc` → 主力 / 大单 / 中单 / 小单 净额
3. **行业**：`moneyflow_ind_ths` → 区间 / 当日 行业 ranking
4. **概念**：`moneyflow_cnt_ths` → 区间 / 当日 概念 ranking
5. **个股**：`moneyflow` → 池内 Top 20 净流入 / 净流出
6. **龙虎榜**：`top_list` + `top_inst` → 一线游资席位（与框架的「famous seats」表对齐）

输出 `CapitalSummary`，包含每个口径的「净流入合计」、「Top-K 表」、「主力 vs 散户分歧度」。

#### 5.3.4 sectors.py — 板块轮动与持续性

核心问题：「**这段时间什么板块在涨，是新主线还是接力，会不会扩散**？」

| 维度 | 计算 |
| --- | --- |
| 当日涨幅榜 | 同花顺行业 + 概念 各取 Top 10（按 `pct_chg`） |
| 区间涨幅榜 | 区间累计涨幅 Top 10（首日开盘价 / 末日收盘价） |
| 板块持续性 | 区间内每日「该板块当日涨幅榜 Top 10 中出现的天数 / 总天数」 |
| 资金接力 | 当日资金净流入 Top 与前 N 日累计资金净流入 Top 的交集 |
| 板块强度矩阵 | 行：板块；列：trade_date；值：当日涨幅；按区间累计涨幅排序 |
| 涨停密度 | 当日板块内涨停个股数（用 `limit_cpt_list` 或 `concept_repo` 反查） |
| 龙头度 | 板块内 Top 1 个股涨幅 / 板块平均涨幅（接近 1 = 普涨；远大于 1 = 龙头独秀） |
| 高低切换 | 区间前半段 Top 板块 vs 后半段 Top 板块 的位次变化 |

输出 `SectorReview`：当日表（涨幅 + 资金 + 涨停密度）+ 区间矩阵 + 「新主线候选」/「接力候选」/「退潮板块」三栏分类。

#### 5.3.5 leaders.py — 龙头识别（多维交叉）

每只候选股按 4 个口径打分（每个口径 0~25）：

1. **梯队**：连板高度（n 连板 → 25\*log₂(n+1)/log₂(7) 截断）
2. **涨幅**：区间累计涨幅排名（百分位 × 25）
3. **资金**：区间累计主力净流入 + 北向净流入排名（百分位 × 25）
4. **题材**：所在题材的板块强度（板块 Top 10 命中 → 25；Top 20 → 15；否则 0）

汇总分 = 4 项之和 ∈ [0, 100]。给出 Top 5 龙头 + Top 10 二线接力 + 板块对照。

输出 `LeaderReview`，每条候选附 `industries / concepts` 列表（来自框架级 `concept_repo`），便于 LLM 在叙事中直接引用「该股属于光模块/AI 服务器/算力租赁三个概念」。

#### 5.3.6 style.py — 风格切换

| 风格轴 | 度量 |
| --- | --- |
| 大小盘 | 沪深 300 涨幅 vs 中证 2000 涨幅 |
| 价值成长 | 中证 800 价值 vs 中证 800 成长（暂用沪深 300 vs 创业板指代理） |
| 中字头 vs 题材 | 中字头股票池涨幅均值 vs 题材连板池涨幅均值 |
| 北向偏好 | 北向 Top10 集中度（大盘/小盘配比） |
| 量能结构 | 大单/小单成交占比变化 |

输出 `StyleReview.dominant_style` + `StyleReview.flip_signal`（区间前后半段是否切换）。

#### 5.3.7 risk.py — 风险信号

| 信号 | 触发条件 |
| --- | --- |
| 高位回撤 | 当日跌幅 > 7% 且 60 日累计涨幅 > 80% 的个股数 |
| 放量滞涨 | 量比 > 2 且涨幅 < 1% 的个股数 |
| 大盘量价背离 | 上证涨且总成交额环比缩量 > 15%（或反之） |
| 北向净卖 | 当日北向净流出 + 区间累计净流出 |
| 跌停扩散 | 跌停家数 + 跌停板块 |
| 大宗折价 | 大宗交易折价 > 5% 的成交额合计 |
| 融资骤变 | 融资余额环比 > ±3% 或绝对额变化 > 100 亿 |
| 妖股见顶 | 题材连板池中「连板数 ≥ 5 且当日炸板」的个股 |

输出 `RiskReview`，每条信号附触发明细 + 命中股票列表（截取 Top 5）。

### 5.4 Step 3 — LLM 综合复盘 (`pipeline.py`)

LLM 不做指标计算，**只做叙事 + 解读 + 结构化判断**。所有数字事实由 §5.3 提供。

#### 5.4.1 7 个 section（顺序执行）

| # | section | system 主题 | 输出 schema | 输出落点 |
| --- | --- | --- | --- | --- |
| 1 | `overview` | 大盘整体描述：今天/本周市场是什么状态？ | `OverviewSection` | `overview.md` |
| 2 | `sectors` | 板块轮动：主线 / 接力 / 退潮 | `SectorsSection` | `sectors.md` |
| 3 | `sentiment` | 情绪与赚钱效应 | `SentimentSection` | `sentiment.md` |
| 4 | `capital` | 资金面：北向 / 行业 / 概念 / 龙虎榜 | `CapitalSection` | `capital.md` |
| 5 | `leaders` | 龙头梳理 | `LeadersSection` | `leaders.md` |
| 6 | `style` | 风格切换 | `StyleSection` | `style.md` |
| 7 | `risk_outlook` | 风险 + 次日/下周展望 | `RiskOutlookSection` | `risk_outlook.md` |

每个 section：

1. 用 Step 2 的对应 dataclass 渲染 `user_prompt`（结构化 JSON，紧凑表示）。
2. 调用 `LLMClient.complete_json(system=..., user=..., schema=...)`，自动 schema 校验 + 1 次 repair retry。
3. 校验通过的 pydantic 对象写入 `mr_stage_results`（stage = section name），同时把 narrative 拼成 markdown 落到 reports_dir。

#### 5.4.2 Prompt 硬性纪律（统一段，每个 system 都拼一遍）

复用 lub 的 anti-hallucination 模板：

```
【硬性纪律】
1. 严禁使用外部搜索、新闻网站、公告网站、实时行情、社交媒体、机构观点或任何未提供的数据。
2. 严禁编造数据；不在输入中的字段一律不引用。
3. evidence 必须以 {field, value, unit, interpretation} 四元组形式给出，
   field 必须是本次 user prompt 中出现过的字段名。
4. 仅输出 JSON，不要 Markdown 代码块包裹，不要解释性前后缀。
5. 单 section 的 narrative 总长度不超过 1200 中文字，分 3~6 段；
   每段第一句必须是结论性句子，后续句给数据支撑。
```

#### 5.4.3 单 section 输出 schema（以 `OverviewSection` 举例）

```python
class OverviewMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=1, max_length=64)
    value: str | int | float | None
    unit: str = Field(min_length=1, max_length=16)
    interpretation: str = Field(min_length=1, max_length=120)

class OverviewSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_summary: str = Field(min_length=20, max_length=240)
    market_tone: Literal[
        "强势普涨", "结构性上涨", "震荡分化",
        "结构性下跌", "弱势普跌", "止跌反弹", "高位见顶"
    ]
    headline_metrics: list[OverviewMetric] = Field(min_length=3, max_length=6)
    narrative_md: str = Field(min_length=200, max_length=1200)
    # 给后续 section 的「连贯主题词」，由 LLM 自由生成，最长 4 个；
    # 后续 section 的 user prompt 会把它带进去保证用词一致。
    theme_tags: list[str] = Field(min_length=1, max_length=4)
```

其他 section 类似，每个都包含 `narrative_md`（叙事正文）+ 一段 `findings: list[Finding]`（结构化要点）。

#### 5.4.4 section 之间的上下文传递

第 N 个 section 的 user prompt 会注入：

- 第 1 个 section 输出的 `theme_tags` + `market_tone`（保持论调一致）
- 当前 section 的指标 dict
- 前 1 个 section 的 narrative 末段（避免重复表达，可省略）

#### 5.4.5 token 预算与信息完整性原则

**核心约束：信息完整性优先于 token 成本。** LLM 复盘报告的准确性直接挂钩输入信息的完整性，因此本插件**不做任何形式的裁剪、采样或 Top-N 截断**——所有聚合指标整体喂给 LLM。

- **不裁剪**：sectors section 的区间矩阵无论是 60×30 还是 60×400 都完整传入；leaders section 的候选名单完整传入；breadthSeries / capitalDaily / styleSeries 全序列传入。
- **不分批**：每个 section 单次 prompt 完成；不为节约 token 切片重组。
- **不重写 metric**：指标层只做「定义层」的聚合（每板块涨幅、每日宽度等），不做「为 LLM 减负」式的二次精简。
- **个股全量字段**：原始数据中能进入 metrics dataclass 的字段一律不删，包括稀疏字段（如某日北交所北向通道为空）—— null 也是信息。

**运行前提（CLI 层校验，启动即检）**：

| 模式 | 推荐 input context 上限 | 备注 |
| --- | --- | --- |
| 单日复盘 | ≥ 64k tokens | 7 个 section 总输入约 30~50k |
| 区间复盘（≤ 30 日） | ≥ 128k tokens | 区间矩阵 + 时序数据增长明显 |
| 区间复盘（30~60 日） | ≥ 200k tokens（推荐 1M context 模型） | sectors / capital 时序数据为主体 |

`MrConfig.llm_provider_default` 应当指向**已支持长 context** 的 provider（如 DeepSeek V3、Claude Sonnet/Opus 1M、GPT-4.1）。运行时 `pipeline.py::estimate_section_tokens(section, metrics)` 估算每节输入 token 数，若任一 section 超过 provider 声明的 `model.max_input_tokens`，**立即报错而非裁剪**，由用户切换到更大 context 的 provider 或缩小窗口。

> 设计立场（v0.1 显式约定，便于后续 review）：宁可让用户改 provider，也不让 LLM 拿到不完整的数据。复盘报告若漏掉了主线板块、龙头股或风险信号，整份报告的可信度即崩塌；裁剪带来的成本节约不值这个代价。

### 5.5 Step 4 — 渲染与落库 (`render.py` + `report.py`)

#### 5.5.1 reports_dir 布局

```
~/.deeptrade/reports/<run_id>/
├── summary.md                # 顶层汇总：BannerBar + 7 个 section 的标题 + 跳转链接 + 关键结论
├── summary.json              # ★ 官网契约文件，由 ReportUploader 上传；schema 见 §15
├── overview.md               # 1
├── sectors.md                # 2
├── sentiment.md              # 3
├── capital.md                # 4
├── leaders.md                # 5
├── style.md                  # 6
├── risk_outlook.md           # 7
├── metrics.json              # Step 2 的全部 dataclass 序列化结果（仅本地审计，不上传）
├── llm_calls.jsonl           # 每个 section 的 prompt + response 审计
└── input_fingerprint.txt     # 64-char sha256(规范化指标 JSON + window + plugin 版本)
```

`summary.json` 由 `report/builder.py::build_review_report(...) -> ReviewReportSchema` 装配后 `.model_dump_json(by_alias=True, indent=2)` 落盘；同次 run 结束时 `runner._maybe_upload_summary` 自动调框架 `ReportUploader.upload(...)` 推送到官网。完整 schema 与上传链路见 §15。

#### 5.5.2 summary.md 模板（节选）

```markdown
# 市场复盘 — 2026-05-30
（区间复盘：2026-05-06 → 2026-05-30，共 19 个交易日）

> 状态：success · run_id: a1b2c3...

## 一句话结论
震荡分化，新主线由「光模块/AI 算力」接棒，老主线「机器人」资金退潮。

## 核心指标
| 指标 | 数值 |
| --- | --- |
| 总涨家数中位值 | 2480 |
| 中位涨幅 | -0.2% |
| 涨停（总和） | 658 |
| 北向净流入 | -42 亿 |
| 情绪温度 | 38 / 100（偏冷） |
| 大盘成交额（区间均值） | 9120 亿 |

## 章节
- [板块轮动](sectors.md)
- [情绪与赚钱效应](sentiment.md)
- [资金面](capital.md)
- [龙头梳理](leaders.md)
- [风格切换](style.md)
- [风险与展望](risk_outlook.md)
```

#### 5.5.3 终端摘要

`market-review run` 结束后，console 直接打印「一句话结论 + 核心指标表 + 章节链接」，不全文打 markdown（避免刷屏）。`market-review report <run_id> --full` 才打全文。

---

## 6. 指标计算细节（单日 vs 区间差异）

| 指标族 | 单日复盘 | 区间复盘 |
| --- | --- | --- |
| 市场宽度 | 当日一行 | 序列 + 区间汇总（均值 / 最值 / 趋势） |
| 板块涨幅 | 当日 Top 10 | 当日 Top 10（末日） + 区间累计 Top 10 |
| 板块持续性 | 不计算 | 每个板块「Top 10 出现天数 / 总天数」 |
| 资金流 | 当日单日 | 当日单日 + 区间累计 |
| 龙头识别 | 仅看当日强度 | 区间累计涨幅 + 区间累计资金 + 当日强度 |
| 风格切换 | 仅给当日比值 | 前半段 vs 后半段 比值变化 |
| 情绪温度 | 当日单值 | 序列 + 区间均值 + 最强/最弱日 |

`Window.mode` 控制 metrics 模块是否计算「区间专属」字段。单日模式时区间字段为 `None` 且 prompt 不引用。

---

## 7. CLI 设计

完整子命令清单（按使用频率排序）：

### 7.1 `run`

```text
deeptrade market-review run \
    [--trade-date YYYYMMDD] \
    [--start YYYYMMDD --end YYYYMMDD] \
    [--force-sync] \
    [--llm <provider>] \
    [--sections <comma-list>] \
    [--no-dashboard] \
    [--fresh-llm | --no-llm-replay | --replay-only]
```

- `--trade-date` 与 `--start/--end` 互斥；都不指定 → 探测最新交易日，单日模式。
- `--sections` 默认全跑；可指定 `overview,sectors` 等子集做快查。
- 其余 flag 语义与 lub v0.15+ 对齐。

### 7.2 `sync`

只跑 Step 0 + Step 1，不调 LLM。用法：

```text
deeptrade market-review sync --start 20260101 --end 20260131
deeptrade market-review sync --trade-date 20260530 --force-sync
```

落库后再用 `run --replay-only` 走 LLM section 即可分离「数据准备」与「LLM 复盘」。

### 7.3 `history`

```text
deeptrade market-review history [--limit 20] [--mode day|range]
```

输出 `run_id / window / status / started_at`。

### 7.4 `report`

```text
deeptrade market-review report <run_id> [--full] [--section overview|sectors|...]
```

- 默认仅打印终端摘要。
- `--full` 走 rich.Markdown 打印 `summary.md`。
- `--section X` 仅打印某个 section 的 markdown。

### 7.5 `settings`

```text
deeptrade market-review settings              # interactive
deeptrade market-review settings show
deeptrade market-review settings set <key> <value>
deeptrade market-review settings reset
```

可配置项见 §8。

### 7.6 `prune`（v0.2 候选）

清理 `reports_dir` 老 run。v0.1 不实现，先用框架级 `deeptrade clean`。

---

## 8. 配置（`MrConfig` → `mr_config` 表）

```python
@dataclass
class MrConfig:
    # ---- 窗口 ----
    max_window_days: int = 60          # 区间复盘的最大交易日数
    # ---- 池子 ----
    exclude_north_exchange: bool = False  # 用户可临时关掉北交所（默认开启）
    # ---- 情绪温度计权重（必须和为 1） ----
    sentiment_weights: dict[str, float] = field(default_factory=lambda: {
        "pos_ratio": 0.30,
        "top_ratio": 0.20,
        "limit_up_intensity": 0.15,
        "connection_health": 0.10,
        "crash_ratio_inv": 0.10,
        "lhb_buy_intensity": 0.10,
        "north_inflow": 0.05,
    })
    # ---- 龙头识别 ----
    leaders_top_k: int = 5
    leaders_secondary_k: int = 10
    leaders_min_score: float = 50.0
    # ---- 板块 ----
    sectors_top_k: int = 10
    sector_provider: Literal["ths", "dc"] = "ths"   # v0.1 默认同花顺
    # ---- 报告 ----
    section_max_chars: int = 1200
    sections_enabled: list[str] = field(default_factory=lambda: [
        "overview", "sectors", "sentiment",
        "capital", "leaders", "style", "risk_outlook",
    ])
    # ---- LLM ----
    llm_provider_default: str | None = None  # None=框架默认
    # ---- LLM replay（沿用 lub v0.15+ 三字段语义） ----
    llm_replay_enabled: bool = True
    llm_replay_write: bool = True
    llm_replay_ttl_days: int | None = None
```

配置 key 前缀 `mr.`，落 `mr_config(key, value_json, updated_at)`。

---

## 9. 数据库 schema

`migrations/20260601_001_init.sql`（节选关键表，完整体在实现 PR 中给出）：

```sql
-- 9.1 池子 / 日历 / ST / 停牌
CREATE TABLE IF NOT EXISTS mr_stock_basic (
    ts_code   VARCHAR PRIMARY KEY,
    symbol    VARCHAR, name VARCHAR, area VARCHAR, industry VARCHAR,
    market    VARCHAR, exchange VARCHAR,
    list_status VARCHAR, list_date VARCHAR, delist_date VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS mr_trade_cal (
    exchange VARCHAR, cal_date VARCHAR, is_open INTEGER, pretrade_date VARCHAR,
    PRIMARY KEY (exchange, cal_date)
);
CREATE TABLE IF NOT EXISTS mr_stock_st (
    ts_code VARCHAR, trade_date VARCHAR, st_status VARCHAR,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE TABLE IF NOT EXISTS mr_suspend_d (
    ts_code VARCHAR, trade_date VARCHAR, suspend_type VARCHAR,
    PRIMARY KEY (ts_code, trade_date, suspend_type)
);

-- 9.2 行情 / 指标
CREATE TABLE IF NOT EXISTS mr_daily (
    ts_code VARCHAR, trade_date VARCHAR,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,
    change DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE TABLE IF NOT EXISTS mr_daily_basic (
    ts_code VARCHAR, trade_date VARCHAR, close DOUBLE,
    turnover_rate DOUBLE, turnover_rate_f DOUBLE, volume_ratio DOUBLE,
    pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE, ps DOUBLE, ps_ttm DOUBLE,
    total_share DOUBLE, float_share DOUBLE, free_share DOUBLE,
    total_mv DOUBLE, circ_mv DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- 9.3 指数
CREATE TABLE IF NOT EXISTS mr_index_daily (
    ts_code VARCHAR, trade_date VARCHAR,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,
    change DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE TABLE IF NOT EXISTS mr_index_dailybasic (
    ts_code VARCHAR, trade_date VARCHAR,
    total_mv DOUBLE, float_mv DOUBLE, total_share DOUBLE, float_share DOUBLE,
    free_share DOUBLE, turnover_rate DOUBLE, turnover_rate_f DOUBLE,
    pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- 9.4 涨跌停 / 连板
CREATE TABLE IF NOT EXISTS mr_limit_list_d (
    trade_date VARCHAR, ts_code VARCHAR, name VARCHAR, industry VARCHAR,
    close DOUBLE, pct_chg DOUBLE, amount DOUBLE,
    fd_amount DOUBLE, limit_amount DOUBLE,
    float_mv DOUBLE, total_mv DOUBLE, turnover_ratio DOUBLE,
    first_time VARCHAR, last_time VARCHAR, open_times INTEGER,
    up_stat VARCHAR, limit_times INTEGER, "limit" VARCHAR,
    PRIMARY KEY (trade_date, ts_code, "limit")
);
CREATE TABLE IF NOT EXISTS mr_limit_step (
    trade_date VARCHAR, ts_code VARCHAR, name VARCHAR,
    nums INTEGER, statibod VARCHAR,
    PRIMARY KEY (trade_date, ts_code)
);
CREATE TABLE IF NOT EXISTS mr_limit_cpt_list (
    trade_date VARCHAR, ts_code VARCHAR, name VARCHAR,
    days INTEGER, up_stat VARCHAR, cons_nums INTEGER,
    up_nums INTEGER, pct_chg DOUBLE, rank INTEGER,
    PRIMARY KEY (trade_date, ts_code)
);
CREATE TABLE IF NOT EXISTS mr_limit_ths (
    trade_date VARCHAR, ts_code VARCHAR, name VARCHAR, price DOUBLE,
    pct_chg DOUBLE, open_num INTEGER, lu_desc VARCHAR,
    limit_type VARCHAR, tag VARCHAR, status VARCHAR,
    first_lu_time VARCHAR, last_lu_time VARCHAR,
    limit_order DOUBLE, limit_amount DOUBLE,
    turnover_rate DOUBLE, free_float DOUBLE,
    PRIMARY KEY (trade_date, ts_code, limit_type)
);

-- 9.5 资金流
CREATE TABLE IF NOT EXISTS mr_moneyflow_hsgt (
    trade_date VARCHAR PRIMARY KEY,
    ggt_ss DOUBLE, ggt_sz DOUBLE, hgt DOUBLE, sgt DOUBLE,
    north_money DOUBLE, south_money DOUBLE
);
CREATE TABLE IF NOT EXISTS mr_hsgt_top10 (
    trade_date VARCHAR, ts_code VARCHAR, name VARCHAR,
    market_type VARCHAR, amount DOUBLE, net_amount DOUBLE,
    buy DOUBLE, sell DOUBLE, rank INTEGER,
    PRIMARY KEY (trade_date, ts_code, market_type)
);
CREATE TABLE IF NOT EXISTS mr_moneyflow_mkt (
    trade_date VARCHAR PRIMARY KEY,
    buy_sm_amount DOUBLE, sell_sm_amount DOUBLE,
    buy_md_amount DOUBLE, sell_md_amount DOUBLE,
    buy_lg_amount DOUBLE, sell_lg_amount DOUBLE,
    buy_elg_amount DOUBLE, sell_elg_amount DOUBLE,
    net_mf_amount DOUBLE
);
CREATE TABLE IF NOT EXISTS mr_moneyflow_ind_ths (
    trade_date VARCHAR, name VARCHAR,
    lead_stock VARCHAR, close DOUBLE, pct_change DOUBLE,
    company_num INTEGER, pct_change_stock DOUBLE,
    net_buy_amount DOUBLE, net_sell_amount DOUBLE, net_amount DOUBLE,
    PRIMARY KEY (trade_date, name)
);
CREATE TABLE IF NOT EXISTS mr_moneyflow_cnt_ths (
    trade_date VARCHAR, ts_code VARCHAR, name VARCHAR,
    lead_stock VARCHAR, close_price DOUBLE, pct_change DOUBLE,
    index_close DOUBLE, company_num INTEGER, pct_change_stock DOUBLE,
    net_buy_amount DOUBLE, net_sell_amount DOUBLE, net_amount DOUBLE,
    PRIMARY KEY (trade_date, ts_code)
);
CREATE TABLE IF NOT EXISTS mr_moneyflow (
    ts_code VARCHAR, trade_date VARCHAR,
    buy_sm_vol DOUBLE, buy_sm_amount DOUBLE,
    sell_sm_vol DOUBLE, sell_sm_amount DOUBLE,
    buy_md_vol DOUBLE, buy_md_amount DOUBLE,
    sell_md_vol DOUBLE, sell_md_amount DOUBLE,
    buy_lg_vol DOUBLE, buy_lg_amount DOUBLE,
    sell_lg_vol DOUBLE, sell_lg_amount DOUBLE,
    buy_elg_vol DOUBLE, buy_elg_amount DOUBLE,
    sell_elg_vol DOUBLE, sell_elg_amount DOUBLE,
    net_mf_vol DOUBLE, net_mf_amount DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- 9.6 龙虎榜
CREATE TABLE IF NOT EXISTS mr_top_list (
    trade_date VARCHAR, ts_code VARCHAR, reason VARCHAR,
    name VARCHAR, close DOUBLE, pct_change DOUBLE,
    turnover_rate DOUBLE, amount DOUBLE,
    l_sell DOUBLE, l_buy DOUBLE, l_amount DOUBLE,
    net_amount DOUBLE, net_rate DOUBLE, amount_rate DOUBLE,
    float_values DOUBLE
);
CREATE TABLE IF NOT EXISTS mr_top_inst (
    trade_date VARCHAR, ts_code VARCHAR, exalter VARCHAR,
    side INTEGER, reason VARCHAR,
    buy DOUBLE, buy_rate DOUBLE, sell DOUBLE, sell_rate DOUBLE,
    net_buy DOUBLE
);

-- 9.7 板块
CREATE TABLE IF NOT EXISTS mr_ths_daily (
    ts_code VARCHAR, trade_date VARCHAR,
    close DOUBLE, open DOUBLE, high DOUBLE, low DOUBLE,
    pre_close DOUBLE, pct_change DOUBLE, vol DOUBLE,
    amount DOUBLE, turnover_rate DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE TABLE IF NOT EXISTS mr_dc_index (
    ts_code VARCHAR, trade_date VARCHAR,
    pct_change DOUBLE, close DOUBLE,
    vol DOUBLE, amount DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- 9.8 其他
CREATE TABLE IF NOT EXISTS mr_hot (
    trade_date VARCHAR, source VARCHAR,
    rank INTEGER, ts_code VARCHAR, name VARCHAR,
    hot_value DOUBLE,
    PRIMARY KEY (trade_date, source, rank)
);
CREATE TABLE IF NOT EXISTS mr_margin (
    trade_date VARCHAR, exchange_id VARCHAR,
    rzye DOUBLE, rzmre DOUBLE, rzche DOUBLE,
    rqye DOUBLE, rqmcl DOUBLE, rzrqye DOUBLE,
    PRIMARY KEY (trade_date, exchange_id)
);
CREATE TABLE IF NOT EXISTS mr_block_trade (
    trade_date VARCHAR, ts_code VARCHAR,
    price DOUBLE, vol DOUBLE, amount DOUBLE,
    buyer VARCHAR, seller VARCHAR,
    PRIMARY KEY (trade_date, ts_code, buyer, seller)
);
CREATE TABLE IF NOT EXISTS mr_cyq_perf (
    trade_date VARCHAR, ts_code VARCHAR,
    his_low DOUBLE, his_high DOUBLE,
    cost_5pct DOUBLE, cost_15pct DOUBLE, cost_50pct DOUBLE,
    cost_85pct DOUBLE, cost_95pct DOUBLE,
    weight_avg DOUBLE, winner_rate DOUBLE,
    PRIMARY KEY (trade_date, ts_code)
);

-- 9.9 run 历史与 events（与 lub_runs / lub_events 对齐）
CREATE TABLE IF NOT EXISTS mr_runs (
    run_id UUID PRIMARY KEY,
    mode VARCHAR NOT NULL,          -- "day" | "range"
    start_date VARCHAR NOT NULL,
    end_date VARCHAR NOT NULL,
    anchor VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    params_json VARCHAR,
    summary_json VARCHAR,
    input_fingerprint VARCHAR,
    error VARCHAR
);
CREATE TABLE IF NOT EXISTS mr_events (
    run_id UUID NOT NULL,
    seq BIGINT NOT NULL,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    payload_json VARCHAR,
    PRIMARY KEY (run_id, seq)
);

-- 9.10 LLM section 结构化结果
CREATE TABLE IF NOT EXISTS mr_stage_results (
    run_id UUID NOT NULL,
    section VARCHAR NOT NULL,        -- overview / sectors / sentiment / capital / leaders / style / risk_outlook
    llm_provider VARCHAR,
    response_json VARCHAR NOT NULL,
    raw_response_json VARCHAR,
    PRIMARY KEY (run_id, section)
);
```

`migrations/20260601_002_config.sql`：

```sql
CREATE TABLE IF NOT EXISTS mr_config (
    key VARCHAR PRIMARY KEY,
    value_json VARCHAR NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

yaml `tables:` 全部列出，`purge_on_uninstall: true`。

---

## 10. 文件目录与 paths

| 路径 | 内容 |
| --- | --- |
| `~/.deeptrade/db.duckdb` | 框架共享 DB；`mr_*` 表落这里 |
| `~/.deeptrade/reports/<run_id>/` | 每次 run 的 markdown + JSON |
| `~/.deeptrade/market_review/cache/` | 预留：未来分时数据/重指标快照 |

v0.1 不需要 `~/.deeptrade/market_review/models/`（无 LightGBM）。

---

## 11. 错误处理与降级

### 11.1 致命错误（直接 `failed` 退出 1）

- `validate_static` 失败：池子 schema 错误 / 配置反序列化失败。
- `tushare.token` 未配置。
- **任一** Tushare API 抛 `TushareUnauthorizedError`、`TushareRateLimitError`、`TushareTransportError`（重试用尽）—— v0.1 所有 API 全 required，**不进入降级**。
- 用户的窗口超出 `max_window_days`。
- 单日窗口落在非交易日且 `--no-snap`（v0.1 暂不实现 no-snap，所有非交易日自动 snap）。
- `mr_runs.status='failed'`、`mr_runs.error=<异常类名 + message>`；`summary.md` 顶部红条 + `summary.json.meta.status='failed'`，仍调一次 `make_report_uploader.upload(...)` 以便官网记录失败 run。

### 11.2 数据维度缺失（仍记 `success`）

接口本身**调用成功但当日返回空**的合法场景，按维度置 None 处理，**不**触发 `partial_failed`：

- `hsgt_top10` 节假日为空 → `CapitalReview.northbound_top10=[]`，capital section 在该子块写「当日北向通道无数据」。
- `block_trade` 当日无成交 → `RiskReview.block_trade_discount_amount_yi=None`。
- `kpl_list` / `kpl_concept` 仅当日数据 → 跨多日区间复盘时按日逐条；任一日缺失对应日字段置 None。
- `ths_hot` / `dc_hot` 当日为空 → `SentimentReview.hot_sources` 缺该项。
- 北交所子集行情稀疏（自然现象）→ universe 内统计原样进入，不补 0。

### 11.3 `partial_failed`（运行继续，但报告标 PARTIAL）

仅以下两类触发：

- **板块体系 `ths_daily` 完整窗口都为空**（极罕见的全市场断数据） → sectors section 跳过，summary banner 标 `PARTIAL`，`meta.failedSections=['sectors']`。
- **某个 LLM section schema 校验 + repair retry 仍失败** → 该 section markdown 写「LLM 调用失败，原因 ...」，对应 `mr_stage_results` 行写空 response + `error_class`；`meta.failedSections` 追加该 section。其余 section 继续。

### 11.4 LLM section 互依失败

第 1 个 section `overview` 必须成功（提供 `theme_tags` + `market_tone` 给后续 section）。如果 `overview` 失败：

- 后续 section 用 `theme_tags=[]` + `market_tone="未知"` 兜底继续跑。
- `summary.md` 顶端打红条警告 `OVERVIEW SECTION FAILED — narratives may drift in tone`。
- `meta.status` 升级为 `partial_failed`，`meta.failedSections=['overview', ...其它失败]`。

---

## 12. 性能预算

| 维度 | 单日复盘 | 60 日区间复盘 |
| --- | --- | --- |
| Tushare 调用数 | ~25 | ~80（多数 API 按 start/end 整体拉一次） |
| Tushare 单次首跑耗时 | 15~30s | 1~3 分钟 |
| DB 写入量 | <30 MB | <300 MB |
| Step 2 指标聚合 | <2s | <8s |
| LLM 调用数（7 section）| 7 | 7（输入大小不同，区间略多） |
| 总 LLM 耗时 | 1~3 分钟 | 2~5 分钟 |
| 总耗时（首跑） | 2~5 分钟 | 4~10 分钟 |
| 重跑（cache 命中 + replay） | <30s | <60s |

> 优化要点：所有 Tushare 调用走 `force_sync=False`（默认）；区间复盘重跑命中 `trade_day_immutable` cache，几乎只跑 LLM。

---

## 13. 测试策略

### 13.1 单元测试（`tests/`，pytest）

每个 metrics 模块单测 + 每个 schema 单测：

- `test_universe.py` — ST/退市/停牌剔除组合
- `test_windows.py` — 单日/区间/snap 边界
- `test_breadth.py` — 涨跌家数 + 连板梯队
- `test_sentiment.py` — 温度计权重和必须为 1；缺失字段 fallback
- `test_capital.py` — 北向汇总 + 行业 Top + 个股 Top
- `test_sectors.py` — 区间矩阵 + 持续性
- `test_leaders.py` — 4 维交叉打分 + Top-K 排序
- `test_style.py` — 大小盘比值
- `test_risk.py` — 各信号触发阈值
- `test_schemas_overview.py` — 7 个 section schema
- `test_prompts.py` — 硬性纪律段必现 + section 之间 theme_tags 传递
- `test_render.py` — markdown 渲染 + reports_dir 布局
- `test_pipeline_section_fallback.py` — 单 section 失败时其余继续
- `test_cli.py` — typer CliRunner 验证子命令路由

### 13.2 集成测试

`tests/test_run_smoke.py`：

- mock Tushare client 返回固定 fixture（截取 2025-12 几个真实交易日）
- mock LLMClient 返回 schema-compliant 响应
- 完整跑 `run`（单日 + 区间），断言 reports_dir 文件齐 + DB 行数对

### 13.3 校验工具

- `python tools/check_registry.py`（仓库根的现有工具）必须通过
- `python tools/check_release.py market-review 0.1.0` 在发布前手动跑一次

---

## 14. registry / 发布流程

### 14.1 registry/index.json 新增

```json
"market-review": {
  "name": "市场复盘",
  "type": "strategy",
  "description": "A 股市场单日 / 区间复盘：板块轮动 / 情绪 / 资金 / 龙头 / 风格 / 风险 / 展望 七章节 LLM 综合报告",
  "repo": "ty19880929/DeepTradePluginOfficial",
  "subdir": "market_review",
  "tag_prefix": "market-review/",
  "min_framework_version": "0.14.0",
  "latest_version": "market-review/v0.1.0"
}
```

### 14.2 发布步骤（沿用现有 monorepo 流程）

1. PR 合并到 `main` —— `registry-check.yml` 校验 yaml + migration checksum。
2. 推 tag `market-review/v0.1.0` —— `plugin-release.yml` 跑 `check_release.py`，过则建 GitHub Release。
3. 用户侧：`deeptrade plugin install market-review`。

### 14.3 yaml 关键字段

```yaml
plugin_id: market-review
name: 市场复盘
version: 0.1.0
type: strategy
api_version: "1"
entrypoint: market_review.plugin:MarketReviewPlugin
description: A 股市场单日 / 区间复盘
author: DeepTrade

dependencies:
  - "tushare>=1.4"
  - "pandas>=2.2,<3"
  - "pyarrow>=15"
  - "numpy>=1.26,<3"

permissions:
  # 见 §4.3
  ...

table_prefix: mr_

tables:
  - { name: mr_stock_basic,        description: 股票池底表,                    purge_on_uninstall: true }
  - { name: mr_trade_cal,          description: 交易日历,                      purge_on_uninstall: true }
  - { name: mr_stock_st,           description: ST 名单,                       purge_on_uninstall: true }
  - { name: mr_suspend_d,          description: 停牌明细,                      purge_on_uninstall: true }
  - { name: mr_daily,              description: 全市场日线,                    purge_on_uninstall: true }
  - { name: mr_daily_basic,        description: 全市场基础指标,                purge_on_uninstall: true }
  - { name: mr_index_daily,        description: 宽基指数日线,                  purge_on_uninstall: true }
  - { name: mr_index_dailybasic,   description: 指数当日基础指标,              purge_on_uninstall: true }
  - { name: mr_limit_list_d,       description: 涨跌停明细,                    purge_on_uninstall: true }
  - { name: mr_limit_step,         description: 连板天梯,                      purge_on_uninstall: true }
  - { name: mr_limit_cpt_list,     description: 题材连板池,                    purge_on_uninstall: true }
  - { name: mr_limit_ths,          description: 同花顺涨停榜,                  purge_on_uninstall: true }
  - { name: mr_moneyflow_hsgt,     description: 北向资金汇总,                  purge_on_uninstall: true }
  - { name: mr_hsgt_top10,         description: 北向 Top10,                    purge_on_uninstall: true }
  - { name: mr_moneyflow_mkt,      description: 东财大盘资金流,                purge_on_uninstall: true }
  - { name: mr_moneyflow_ind_ths,  description: 同花顺行业资金流,              purge_on_uninstall: true }
  - { name: mr_moneyflow_cnt_ths,  description: 同花顺概念资金流,              purge_on_uninstall: true }
  - { name: mr_moneyflow,          description: 个股大单资金流,                purge_on_uninstall: true }
  - { name: mr_top_list,           description: 龙虎榜个股,                    purge_on_uninstall: true }
  - { name: mr_top_inst,           description: 龙虎榜席位,                    purge_on_uninstall: true }
  - { name: mr_ths_daily,          description: 同花顺板块/概念指数日线,       purge_on_uninstall: true }
  - { name: mr_dc_index,           description: 东财板块指数日线,              purge_on_uninstall: true }
  - { name: mr_hot,                description: 同花顺/东财热榜,               purge_on_uninstall: true }
  - { name: mr_margin,             description: 融资融券,                      purge_on_uninstall: true }
  - { name: mr_block_trade,        description: 大宗交易,                      purge_on_uninstall: true }
  - { name: mr_cyq_perf,           description: 筹码每日指标,                  purge_on_uninstall: true }
  - { name: mr_runs,                description: 本插件 run 历史,               purge_on_uninstall: true }
  - { name: mr_events,              description: 本插件 run 事件流,             purge_on_uninstall: true }
  - { name: mr_stage_results,      description: LLM section 结构化输出,        purge_on_uninstall: true }
  - { name: mr_config,             description: 本插件用户可调设置,            purge_on_uninstall: true }

migrations:
  - { version: "20260601_001", file: migrations/20260601_001_init.sql,    checksum: "sha256:..." }
  - { version: "20260601_002", file: migrations/20260601_002_config.sql,  checksum: "sha256:..." }
```

---

## 15. JSON 执行报告 schema 与上传链路

> 参考实现：`limit_up_board/limit_up_board/report/schema.py`（`StrategyReportSchema`）+ `report/builder.py`（`build_strategy_report`）+ `runner.py::_maybe_upload_summary`。本插件**完全沿用 lub v0.18.x 的契约语义**：严格 pydantic、`extra="forbid"`、驼峰命名前端字段、根级 `_extras` 兜底未来上游新增字段、`ReportUploader` 由框架统一管理上传链路（URL/token/重试）。

### 15.1 设计原则

1. **`summary.json` 是与官网前端的唯一契约**。`summary.md` 是给本地 CLI / rich 看的人类可读版本，schema 漂移以 `summary.json` 为准。
2. **严格 schema**：所有非根模型 `model_config = ConfigDict(extra="forbid")`；上游悄悄加字段必须显式合入 schema。
3. **根模型保留 `extras`**：根级 `_extras: dict` 兜底未来 `MetricsBlock` 等新增字段，避免静默丢数据。前端可整体忽略 `_extras` 字段，但服务端可以扫描日志中 `_extras` 非空的 run 来发现「上游加了我没收的字段」。
4. **驼峰字段名**：所有前端会消费的字段一律小驼峰 `headlineMetrics`、`failedSections`；snake_case 仅用于内部 dataclass。
5. **空值约定**：
   - 维度未计算（如单日模式下区间字段）→ 字段 `None`；
   - 维度数据为空（如当日北向 Top10）→ `list[] = []`；
   - LLM section 失败 → 该 section 字段保留，但 `narrativeMd=""` + `error` 子字段非空。
6. **`schemaVersion` 写死在 meta**：与插件版本解耦，前端可基于 `schemaVersion` 单独做兼容；schema 不变的 patch 不动 schema version。
7. **`generatedAt` 用 ISO 8601 含时区**（与 lub 一致）。
8. **失败也上传**：`status='failed'` / `'cancelled'` / `'partial_failed'` 都写出 summary.json 并上传，让官网拿到失败原因。

### 15.2 schema 概览（树状）

```
ReviewReportSchema
├── meta: ReportMeta
│   ├── title: str                          # "市场复盘 — 2026-05-30"（区间则 "2026-05-06 → 2026-05-30"）
│   ├── runId: str
│   ├── window: WindowMeta {mode, start, end, anchor, nDays, tradeDates[]}
│   ├── status: "success" | "partial_failed" | "failed" | "cancelled"
│   ├── failedSections: list[str]           # ["sectors","style"] 时配 PARTIAL banner
│   ├── sectionsEnabled: list[str]
│   ├── llmProvider: str                    # 实际使用的 provider 名
│   ├── schemaVersion: str                  # "1.0"
│   ├── pluginVersion: str                  # 来自 deeptrade_plugin.yaml
│   ├── inputFingerprint: str               # 64-char sha256，与本地 input_fingerprint.txt 一致
│   ├── generatedAt: str                    # ISO 8601 +08:00
│   └── error: str | None                   # status != success 时填充
├── headline: ReportHeadline                # 顶层一句话结论 + 核心指标表（前端首屏）
├── overview: OverviewSection
├── sectors: SectorsSection
├── sentiment: SentimentSection
├── capital: CapitalSection
├── leaders: LeadersSection
├── style: StyleSection
├── riskOutlook: RiskOutlookSection
├── metrics: MetricsBlock                   # 结构化数字（前端图表用），不经 LLM
│   ├── breadthSeries: list[BreadthSnapshotJson]
│   ├── indexReturns: dict[str, IndexReturnJson]
│   ├── sectorMatrix: SectorMatrixJson
│   ├── capitalDaily: list[CapitalDailyRow]
│   ├── leaderTable: list[LeaderCandidateJson]
│   ├── styleSeries: StyleSeriesJson
│   └── riskSignals: list[RiskSignalJson]
└── _extras: dict[str, Any]                 # 兜底未来上游新增
```

### 15.3 根模型 + meta

```python
# market_review/report/schema.py

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

ReportStatus = Literal["success", "partial_failed", "failed", "cancelled"]
SectionName = Literal[
    "overview", "sectors", "sentiment", "capital",
    "leaders", "style", "risk_outlook",
]
WindowMode = Literal["day", "range"]

SCHEMA_VERSION = "1.0"


class WindowMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: WindowMode
    start: str                       # YYYYMMDD
    end: str                         # YYYYMMDD
    anchor: str                      # YYYYMMDD（单日=T, 区间=T1）
    nDays: int = Field(ge=1)
    tradeDates: list[str]            # 全部交易日（用于前端做时间轴）


class ReportMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=120)
    runId: str
    window: WindowMeta
    status: ReportStatus
    failedSections: list[SectionName] = Field(default_factory=list)
    sectionsEnabled: list[SectionName]
    llmProvider: str = Field(min_length=1)
    schemaVersion: str = Field(default=SCHEMA_VERSION)
    pluginVersion: str
    inputFingerprint: str = Field(min_length=64, max_length=64)
    generatedAt: str                # ISO 8601 含 +08:00
    error: str | None = None


class HeadlineMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=24)
    value: str | int | float | None
    unit: str = Field(min_length=1, max_length=16)   # "亿" / "%" / "家" / "分" / "none"
    delta: float | None = None
    deltaUnit: str | None = None    # "%" / "pct_pt"
    interpretation: str | None = Field(default=None, max_length=80)


class ReportHeadline(BaseModel):
    """前端首屏卡片：一句话结论 + 6 个核心指标。"""
    model_config = ConfigDict(extra="forbid")
    oneLiner: str = Field(min_length=10, max_length=120)
    marketTone: Literal[
        "强势普涨", "结构性上涨", "震荡分化",
        "结构性下跌", "弱势普跌", "止跌反弹", "高位见顶", "未知"
    ]
    coreMetrics: list[HeadlineMetric] = Field(min_length=4, max_length=8)
    themeTags: list[str] = Field(min_length=0, max_length=4)
```

### 15.4 通用子模型（每个 LLM section 复用）

```python
class EvidenceItem(BaseModel):
    """与 lub EvidenceItemStrict 同款 4 元组：禁止数组/对象 value。"""
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=1, max_length=64)
    value: str | int | float | None
    unit: str = Field(min_length=1, max_length=16)
    interpretation: str = Field(min_length=1, max_length=120)


class Finding(BaseModel):
    """结构化要点（叙事正文之外的关键观察点）。"""
    model_config = ConfigDict(extra="forbid")
    headline: str = Field(min_length=1, max_length=80)
    detail: str = Field(min_length=1, max_length=240)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=5)
    severity: Literal["info", "positive", "warning", "critical"] = "info"


class SectionBase(BaseModel):
    """每个 LLM section 输出的统一基类。"""
    model_config = ConfigDict(extra="forbid")
    narrativeMd: str = Field(default="", max_length=2400)   # 失败时 ""
    findings: list[Finding] = Field(default_factory=list)
    error: str | None = None                                 # 该 section 失败原因
```

### 15.5 各 section 子模型

```python
# --- overview ---
class OverviewSection(SectionBase):
    marketTone: Literal[
        "强势普涨", "结构性上涨", "震荡分化",
        "结构性下跌", "弱势普跌", "止跌反弹", "高位见顶", "未知"
    ]
    headlineMetrics: list[HeadlineMetric] = Field(min_length=3, max_length=6)
    themeTags: list[str] = Field(min_length=0, max_length=4)


# --- sectors ---
class SectorEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    pctChg: float                 # 当日 / 区间累计涨幅 %
    netInflowYi: float | None     # 资金净流入（亿）
    limitUpCount: int | None      # 当日涨停个股数
    leaderTsCode: str | None
    leaderName: str | None
    persistenceDays: int | None   # 区间内 Top10 出现天数

class SectorClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    new_mainline: list[SectorEntry] = Field(default_factory=list)
    relay: list[SectorEntry] = Field(default_factory=list)
    fading: list[SectorEntry] = Field(default_factory=list)

class SectorsSection(SectionBase):
    provider: Literal["ths", "dc"]
    todayTop: list[SectorEntry] = Field(default_factory=list)
    rangeTop: list[SectorEntry] = Field(default_factory=list)
    classification: SectorClassification
    rotationCommentary: str = Field(default="", max_length=600)


# --- sentiment ---
class SentimentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tradeDate: str
    scoreOf100: float = Field(ge=0, le=100)
    nUp: int = Field(ge=0)
    nDown: int = Field(ge=0)
    medianPctChg: float
    nLimitUp: int = Field(ge=0)
    nLimitDown: int = Field(ge=0)
    nZhaban: int = Field(ge=0)

class SentimentSection(SectionBase):
    series: list[SentimentSnapshot]
    avgScore: float = Field(ge=0, le=100)
    strongestDay: str | None         # YYYYMMDD
    weakestDay: str | None
    moneyEffect: Literal["strong", "neutral", "weak"]
    losingEffect: Literal["light", "moderate", "heavy"]


# --- capital ---
class CapitalLeader(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tsCode: str
    name: str
    netInflowYi: float

class IndustryFlowRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    netInflowYi: float
    pctChg: float

class CapitalDailyRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tradeDate: str
    northMoneyYi: float | None      # 北向净流入（亿）
    mainNetInflowYi: float | None   # 大盘主力净额
    marginBalanceYi: float | None   # 两融余额
    marginDeltaYi: float | None     # 当日两融变化

class CapitalSection(SectionBase):
    northSummary: list[CapitalDailyRow]
    northTop10Today: list[CapitalLeader] = Field(default_factory=list)
    industryTop: list[IndustryFlowRow] = Field(default_factory=list)
    conceptTop: list[IndustryFlowRow] = Field(default_factory=list)
    stockTop: list[CapitalLeader] = Field(default_factory=list)
    stockBottom: list[CapitalLeader] = Field(default_factory=list)
    lhbHighlights: list[Finding] = Field(default_factory=list)


# --- leaders ---
class LeaderCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tsCode: str
    name: str
    score: float = Field(ge=0, le=100)
    scoreBreakdown: dict[str, float]      # ladder / return / capital / theme
    industries: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    sectorTopHit: list[str] = Field(default_factory=list)
    ladderHeight: int | None              # n 连板
    rangePctChg: float | None             # 区间累计涨幅 %
    cumMainInflowYi: float | None         # 区间累计主力净额（亿）
    rationale: str = Field(default="", max_length=240)

class LeadersSection(SectionBase):
    primary: list[LeaderCandidate]        # Top-K（默认 5）
    secondary: list[LeaderCandidate]      # 二线接力（默认 10）
    minScore: float                       # 进入名单的硬门槛
    sectorMap: dict[str, list[str]]       # 板块 → ts_code 列表


# --- style ---
class StyleSeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tradeDate: str
    largeCapRet: float                    # 沪深 300 当日 %
    smallCapRet: float                    # 中证 2000 当日 %
    valueRet: float | None
    growthRet: float | None
    bigToSmallRatio: float                # 大小盘相对强度

class StyleSection(SectionBase):
    dominantStyle: Literal[
        "large_cap", "small_cap", "growth", "value",
        "balanced", "rotating",
    ]
    flipSignal: bool                      # 区间内前后半段是否切换
    series: list[StyleSeriesPoint]
    rangeSummary: dict[str, float]        # avgBigToSmall / avgValueToGrowth ...


# --- risk_outlook ---
class RiskSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str                             # 与 metrics/risk.py 内信号 key 一致
    triggered: bool
    severity: Literal["info", "warning", "critical"]
    detail: str
    affectedSamples: list[str] = Field(default_factory=list, max_length=5)

class OutlookHypothesis(BaseModel):
    """LLM 给出的次日 / 下周展望假设（可证伪 + 观测点）。"""
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=60)
    rationale: str = Field(min_length=1, max_length=240)
    watchPoints: list[str] = Field(min_length=1, max_length=5)
    failTriggers: list[str] = Field(min_length=1, max_length=5)

class RiskOutlookSection(SectionBase):
    signals: list[RiskSignal]
    overallRisk: Literal["low", "moderate", "elevated", "high"]
    hypotheses: list[OutlookHypothesis] = Field(min_length=1, max_length=4)
```

### 15.6 `metrics` 块（结构化数字，前端图表用）

LLM 不产出该块，全部来自 §5.3 的 dataclass 序列化。前端用它画 K 线、热力图、柱状图，**不依赖 LLM 文本**。

```python
class BreadthSnapshotJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tradeDate: str
    nTotal: int
    nUp: int
    nDown: int
    nFlat: int
    nUp5pct: int
    nDown5pct: int
    nLimitUp: int
    nLimitDown: int
    nZhaban: int
    upLadder: dict[str, int]              # "2": 12, "3": 5 ...
    totalAmountYi: float
    indexReturns: dict[str, float]

class IndexReturnJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tsCode: str
    name: str
    pctChgWindow: float                   # 区间累计涨幅
    closeSeries: list[float]              # 区间每日 close
    amountSeriesYi: list[float]           # 区间每日成交额

class SectorMatrixJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sectors: list[str]                    # 行：sector 名（同花顺）
    tradeDates: list[str]                 # 列：日期
    pctChg: list[list[float]]             # 矩阵：每日每板块涨幅 %
    cumPctChg: list[float]                # 区间累计（与 sectors 等长）
    persistenceDays: list[int]

class LeaderCandidateJson(BaseModel):
    """与 LeaderCandidate 同字段但放在 metrics 块（前端可独立消费）。"""
    model_config = ConfigDict(extra="forbid")
    tsCode: str
    name: str
    score: float
    scoreBreakdown: dict[str, float]
    industries: list[str]
    concepts: list[str]
    sectorTopHit: list[str]
    ladderHeight: int | None
    rangePctChg: float | None
    cumMainInflowYi: float | None

class StyleSeriesJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series: list[StyleSeriesPoint]
    avgBigToSmall: float
    avgValueToGrowth: float | None
    halfPeriodFlip: bool

class RiskSignalJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    triggered: bool
    severity: Literal["info", "warning", "critical"]
    sampleCount: int
    samplesTopK: list[str] = Field(default_factory=list, max_length=10)


class MetricsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    breadthSeries: list[BreadthSnapshotJson]
    indexReturns: dict[str, IndexReturnJson]   # key = ts_code
    sectorMatrix: SectorMatrixJson
    capitalDaily: list[CapitalDailyRow]
    leaderTable: list[LeaderCandidateJson]
    styleSeries: StyleSeriesJson
    riskSignals: list[RiskSignalJson]
```

### 15.7 根模型

```python
class ReviewReportSchema(BaseModel):
    """根模型 — 前端 ``await fetch(url).json()`` 直接消费。"""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    meta: ReportMeta
    headline: ReportHeadline
    overview: OverviewSection
    sectors: SectorsSection
    sentiment: SentimentSection
    capital: CapitalSection
    leaders: LeadersSection
    style: StyleSection
    riskOutlook: RiskOutlookSection
    metrics: MetricsBlock
    extras: dict[str, Any] = Field(default_factory=dict, alias="_extras")
```

### 15.8 报告构建链路（`market_review/report/builder.py`）

完全沿用 lub `report/builder.py` 的纯装配模式：

```python
def build_review_report(
    *,
    status: RunStatus,
    window: Window,
    breadth: BreadthReview,
    sentiment: SentimentReview,
    capital: CapitalReview,
    sectors: SectorReview,
    leaders: LeaderReview,
    style: StyleReview,
    risk: RiskReview,
    sections: dict[str, SectionBase],          # 7 个 LLM section 实例
    failed_sections: list[str],
    run_id: str,
    llm_provider: str,
    plugin_version: str,
    input_fingerprint: str,
    generated_at: datetime | None = None,
) -> ReviewReportSchema:
    """纯数据装配 + 类型转换。不做 IO、不读 DB、不读文件。"""
    ...
```

调用方（`render.write_report`）：

```python
report_obj = build_review_report(...)
(root / "summary.json").write_text(
    report_obj.model_dump_json(by_alias=True, indent=2, exclude_none=False),
    encoding="utf-8",
)
```

### 15.9 上传链路（与 lub 完全同款）

`market_review/runner.py::_maybe_upload_summary` 复刻 lub v0.16.1 的实现：

```python
def _maybe_upload_summary(
    self, report_path: Path, window: Window
) -> Iterable[StrategyEvent]:
    """Best-effort POST summary.json via 框架 v0.11 ReportUploader."""
    if self._ctx is None:
        return
    json_path = report_path / "summary.json"
    if not json_path.is_file():
        yield self._rt.emit(
            EventType.LOG,
            f"summary.json 未生成，跳过上传：{json_path}",
            level=EventLevel.INFO,
            payload={
                "enabled": True,
                "status": "skipped_no_local_file",
                "json_path": str(json_path),
                "anchor": window.anchor,
            },
        )
        return
    uploader: ReportUploader = self._ctx.make_report_uploader(run_id=self._rt.run_id)
    result = uploader.upload(
        json_path,
        plugin_name="市场复盘",            # 与 lub "打板策略" 同位
        trade_date=window.anchor,         # 用 anchor 日期作为对外的 trade_date
    )
    # ...emit 与 lub 同款的 ok / failed 事件
```

要点：

1. **配置由框架统一管理**：`report.upload.enabled` / `report.upload.url` / `report.upload.token` / `report.upload.timeout` 是框架配置族，插件不读不写。
2. **`plugin_name` 走中文显示名**（与 lub "打板策略" 同位，本插件填 "市场复盘"），便于官网按插件聚合展示。
3. **`trade_date` 用 `window.anchor`**：单日模式 = T；区间模式 = T1（区间末日）。官网按 `(plugin_name, trade_date)` 索引时不会与 lub 的单日 run 冲突，因为 plugin_name 不同。
4. **失败也上传**：`status != 'success'` 时 `summary.json` 仍按 §15.7 schema 写出（`error` 字段填充），上传逻辑不变。
5. **静默上传开关**：框架返回 `status="skipped_*"`（用户未开 `report.upload.enabled`）时 runner 不打扰用户，与 lub 一致。

### 15.10 与 markdown 报告的关系

| 文件 | 用户消费方式 | schema 漂移风险 |
| --- | --- | --- |
| `summary.json` | 官网前端 + 自动化 | **零漂移**：每次新增字段必须改 schema + 加测试 |
| `summary.md` | 本地 CLI / `report --full` 渲染 | 允许微调（不影响契约） |
| 各 section `*.md` | 本地阅读 | 同上 |
| `metrics.json` | 临时调试（dataclass 序列化） | 不上传，仅本地审计 |
| `llm_calls.jsonl` | LLM 调用审计 | 不上传 |
| `input_fingerprint.txt` | 64-char sha256 | 与 `meta.inputFingerprint` 一致；不上传 |

**重要约束**：v0.1 起 `summary.json` 进入官网契约后，**任何 schema 变更必须升 `meta.schemaVersion`**（major 改字段语义、minor 加新可选字段、patch 改文档）。前端用 `meta.schemaVersion >= "1.0"` 做兼容判断。

### 15.11 测试要求（`tests/test_report_schema.py` + `tests/test_upload_audit_payload.py`）

参考 lub 的 `tests/test_report_builder.py` 和 `tests/test_upload_audit_payload.py`：

1. **schema 序列化往返**：`ReviewReportSchema.model_validate_json(obj.model_dump_json(by_alias=True))` 必须等价。
2. **`extra="forbid"` 触发**：在任一非根模型上多塞一个字段必须 `ValidationError`。
3. **`_extras` 兜底**：根模型多塞字段不报错，进入 `extras` 字典。
4. **空 section 兜底**：`failed_sections` 包含某 section 时，对应 section 字段仍存在且 `narrativeMd=""` + `error` 非空。
5. **fingerprint 长度**：`meta.inputFingerprint` 强制 64-char hex。
6. **上传 payload 校验**：mock `ReportUploader.upload` 验证传入 `plugin_name="市场复盘"` + `trade_date=window.anchor`。
7. **failed-status 上传**：`status='failed'` 时 `summary.json` 仍写出且能被 schema 解析。

---

## 16. 风险 / 待澄清问题

> 已确认的边界（不再列入风险表）：
>
> - **API 权限**：用户 8000 积分覆盖范围内全部 API 均已开通，且未来只升不降，因此 v0.1 不再考虑 tier 降级矩阵；任一接口失败直接 `failed`。
> - **Debate 模式**：本插件不会引入多 LLM 互检（市场复盘是叙事报告，非选股；多 LLM 反而破坏 narrative 一致性）。
> - **prompt 裁剪**：本插件不会做任何形式的输入裁剪，信息完整性优先，由用户选择长 context provider，见 §5.4.5。
> - **`summary.json` schema 稳定性**：v0.1 字段经 §15 推演已足够覆盖前端首屏 + 7 个 section + 结构化 metrics 图表，进入官网契约后即冻结；后续 schema 演进沿用 §15.10 的 `schemaVersion` 机制即可，v0.1 不预留更多防御。

| # | 问题 | 处置 |
| --- | --- | --- |
| 1 | 60 日区间 ×5000 只股票 ×多张表，DuckDB 写入压力？ | 用 `DataFrame.to_sql` 批量写 + 同事务多 insert；落库总量 <300 MB 不构成压力，但需要单独 benchmark |
| 2 | 同花顺板块体系按月小幅迭代，区间复盘跨月时成员变更如何处理？ | v0.1 用 `concept_repo.get_at(trade_date)`（框架已提供）取每日快照；v0.2+ 在 sectors section 显式标注「板块成员变更次数」 |
| 3 | section 间 narrative 风格漂移 | `theme_tags` + 文风约束段落写进每个 system；后续 v0.2 加 `tone_constraints` 配置 |
| 4 | 北交所成交稀疏 → 北向 / 大单字段大量为空 | universe 内单独标注「北交所占比」让 LLM 自行决定语义裁切（仅 narrative 表达层面） |
| 5 | 上传失败重试？ | v0.1 沿用 lub 的「best-effort + WARN」策略，不本地重试；用户可用 `market-review report <run_id> --upload-only` 手动重传（v0.2 实现） |
| 6 | 长 context provider 的可用性？ | CLI 启动时由 `pipeline.estimate_section_tokens` 估算各 section 输入 token，超过 provider 声明上限直接报错而非裁剪（见 §5.4.5）；建议默认 provider 选 Claude / DeepSeek / GPT-4.1 1M context |

---

## 17. 演进路线

| 版本 | 内容 |
| --- | --- |
| v0.1.0 | 本文档全部 MVP 范围（单 LLM、7 个 section、单日 + 区间、`summary.json` v1.0 + 自动上传） |
| v0.1.x | 修 bug + 调指标权重 + section 内容微调（不动 schema） |
| v0.2.0 | `market-review report <id> --upload-only` 手动重传 + `tone_constraints` 配置（控制 section 间叙事风格） + 板块成员变更次数注入 sectors section |
| v0.3.0 | LightGBM 板块涨幅预测（用 ths_daily 区间历史训练） → 「下一交易日板块强度概率」字段并喂给 sectors section；`meta.schemaVersion` 升 1.1（加可选字段，不破坏前端） |
| v0.4.0 | 跨区间对比（本周 vs 上周、本月 vs 上月）；schema 1.2 |
| v0.5.0 | HTML / PDF / 飞书 / 邮件等第三方通道推送（依赖框架 `ReportUploader` 扩展） |
| v0.6.0 | 与 limit-up-board / accumulation-probe-washout 集成：导出「市场背景」JSON（`summary.json.metrics` 的子集），作为下游 R1 prompt 的额外字段 |

> 显式声明**不会**进入演进路线：Debate / 多 LLM 互检、prompt 输入裁剪、低权限账号降级矩阵 —— 由 §1.3「永不引入」段落保障约束的可追溯性。

---

## 18. 实施 PR 拆分建议

1. **PR-1 / 骨架**：仓库结构 + plugin.py + cli skeleton（仅 `--help`）+ migrations + registry + check_registry 通过。
2. **PR-2 / 数据层**：`universe.py` + `windows.py` + `data.py::sync_window` + 全部 required API 落库 + 单测。
3. **PR-3 / 指标层**：`metrics/*` 7 个模块 + 单测。
4. **PR-4 / LLM section**：`schemas.py` + `prompts.py` + `pipeline.py` + `render.py` + 集成测试。
5. **PR-5 / 报告 schema + 上传链路**：`report/schema.py`（`ReviewReportSchema`）+ `report/builder.py`（`build_review_report`）+ `runner.py::_maybe_upload_summary` + `tests/test_report_schema.py` + `tests/test_upload_audit_payload.py`，对齐 lub v0.16.1 契约。
6. **PR-6 / 报告与 CLI 完整化**：`run` / `sync` / `history` / `report` / `settings` + 终端摘要 + 错误降级路径。
7. **PR-7 / Release v0.1.0**：CHANGELOG + tag `market-review/v0.1.0` + registry/index.json 写入 `latest_version`。

---

## 附 A：核心数据结构（Python）

```python
# windows.py
@dataclass(frozen=True)
class Window:
    mode: Literal["day", "range"]
    start: str
    end: str
    trade_dates: tuple[str, ...]
    anchor: str

    @property
    def n_days(self) -> int:
        return len(self.trade_dates)

# universe.py
@dataclass(frozen=True)
class UniverseSnapshot:
    trade_date: str
    ts_codes: frozenset[str]
    excluded_st: int
    excluded_delist: int
    excluded_suspend: int

# metrics/breadth.py
@dataclass
class BreadthSnapshot:
    trade_date: str
    n_total: int
    n_up: int
    n_down: int
    n_flat: int
    n_up5pct: int
    n_down5pct: int
    n_limit_up: int
    n_limit_down: int
    n_zhaban: int
    up_ladder: dict[int, int]   # {2: 12, 3: 5, 4: 1, ...}
    total_amount_yi: float
    index_returns: dict[str, float]

@dataclass
class BreadthReview:
    series: list[BreadthSnapshot]
    median_up_count: int
    sentiment_extreme_day: tuple[str, str]  # (strongest, weakest)

# metrics/leaders.py
@dataclass
class LeaderCandidate:
    ts_code: str
    name: str
    score: float                  # 0~100
    score_breakdown: dict[str, float]   # {"ladder": .., "return": .., "capital": .., "theme": ..}
    industries: list[str]
    concepts: list[str]
    sector_top_hit: list[str]

@dataclass
class LeaderReview:
    primary: list[LeaderCandidate]    # Top-K
    secondary: list[LeaderCandidate]  # 二线接力
```

---

## 附 B：和现有插件共用的代码点

| 复用对象 | 来源 | 用法 |
| --- | --- | --- |
| `TradeCalendar` | lub `calendar.py` 同款实现 | 几乎可直接抄；新插件再独立一份保持隔离 |
| `cancellation.install_sigint_marker` | lub `cancellation.py` | 抄一份 |
| `RunMetrics` / observability | lub `observability.py` | 简化版 |
| `_FrozenConfigService` | lub `runtime.py` | **不复用**（本插件不引入 debate） |
| `ConceptRepository` | 框架 v0.14.0 | 直接 inject |
| `LegacyStreamRenderer` 协议 | lub `ui/protocol.py` | v0.1 只需 legacy；rich dashboard 推迟到 v0.3 |
| `apply_empty_array_policy` 模式 | lub `schemas.py` | 本插件 section 的 `findings` 数组也用同款空数组策略 |

---

## 附 C：开发者快速上手

```powershell
# 1. 安装框架（无则）
pipx install deeptrade-quant

# 2. 在仓库根做 schema 校验
python tools/check_registry.py

# 3. 进入插件目录跑测试
cd market_review ; pytest

# 4. 单独跑某节
cd market_review ; pytest tests/test_metrics_sectors.py -k test_persistence

# 5. 本地 dry-run
deeptrade market-review sync --trade-date 20260530
deeptrade market-review run   --trade-date 20260530 --no-dashboard
deeptrade market-review report <run_id> --full
```

---

文档结束。后续 PR 系列将分阶段落地。
