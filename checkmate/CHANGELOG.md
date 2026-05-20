# Checkmate — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## v0.4.0 — 2026-05-19

**高级执行成本**。沿用 v0.3.0 全部业务逻辑，本版聚焦"执行端成本更接近真
盘"。PR-7.2（涨跌停队列分笔模拟）需 Tushare Pro 5min K 权限，本版未落地，
留待权限就绪后单独立项。

### 新增功能

- **动态滑点曲线**（PR-7.1，[checkmate.executor.dynamic_slippage_bps](checkmate/executor.py)）
  — 替换 v0.1 起的固定 5bps 模型。`ExecutionConfig.slippage_model={"fixed", "dynamic"}`，
  `"dynamic"` 模式按 ``amount_20d_avg`` 在 log10 空间分段线性插值，5 个
  breakpoint 默认配置：micro-cap (1e6 元/天) → 30bps；small (1e7) → 20；
  mid (1e8) → 10；large (1e9) → 5；mega (1e10) → 2。两端硬 clamp。
  `PendingOrder` 加 `amount_20d_avg` 字段；`backtest.execute_day` 从
  `feat_rows` 预构 amount_lookup 注入下一个 trading session 的订单；shard
  JSON round-trip 字段，resume 不丢精度。
- **市场冲击成本**（PR-7.3，[checkmate.executor.impact_bps](checkmate/executor.py)）
  — 经典 Almgren-Chriss 风格平方根模型：
  ``bps = impact_coefficient × sqrt(participation) × 100``。
  Calibration anchors：1% 参与度 → 10bps；25% → 50bps；100% → 100bps。
  `ExecutionConfig.impact_model={"none", "sqrt"}`，默认 ``"none"`` 完整
  back-compat；`impact_min_participation=0.005` 以下零 impact（由 bid-ask
  价差吸收）。`cost_breakdown` 从 4 键扩到 **5 键**（新增 ``impact``，
  ``"none"`` 模式恒 0），下游 report / dashboard 无需特殊判空。

### 默认行为

- `slippage_model="fixed"` 与 `impact_model="none"` 都是默认值；既有
  263 个 v0.3 测试零回归。要启用新模型须显式在 `BacktestParams.execution_cfg`
  里改字段，或在 v0.5+ CLI flag 直通。
- 既有 `cost_breakdown` 消费者（report.total_fees / by_industry / SVG 渲染
  等）正常累加 5 字段，零代码变更。

### 测试

- 35 个新测试：18 动态滑点（曲线 5 breakpoint + 中点插值 + 单调性 + 路由 +
  端到端 simulate_session 同日双 amount 价差）+ 17 impact（none / sqrt 模型
  + min participation 闸门 + sqrt 4 calibration anchor + slip+impact 叠加 +
  端到端大小单 cost 差异）。`pytest -q` 280 passed / ~2:44。

### 已知限制 / 后续路线

- **PR-7.2 涨跌停队列分笔模拟**：需 Tushare Pro 5min K 权限，**本版未落地**。
  当前 executor 跌停依然按"defer 直到 max_defer_days 后 cancel"启发式处理。
  权限就绪后可基于 5min 内开盘价 + 当日封单时刻的 impact 修正分笔出价
  概率，预计独立 release `v0.5.0`。
- impact 模型仅基于 `amount_20d_avg`（v0.1 已经有），未引入今日实际 amount
  作为分母 — 实盘做单时建议在 v0.5+ 切换到 today's actual amount，更贴近
  真实流动性窗口。

---

## v0.3.0 — 2026-05-19

**稳健性扫描**。沿用 v0.2.0 全部业务逻辑与渲染层，本版聚焦"避免过拟合 +
多参数对比 + 分市场状态评估"。

### 新增功能

- **参数网格回测**（[checkmate.grid](checkmate/grid.py)）— YAML 文件描述
  网格，CLI `--grid params.yaml` 触发笛卡尔展开 + 逐 cell 调
  `run_backtest`。每个 cell 自有独立 `config_hash` 与 checkpoint 目录，可
  独立断点续算。聚合 JSON 落到
  `~/.deeptrade/checkmate/reports/grid_<UTC-timestamp>.json`，含每 cell 的
  `(overrides, run_id, config_hash, metrics)` + 按 `--rank-by`（默认
  `final_equity`，可选 `cagr` / `max_drawdown` / `n_fills`）排序的 ranked 视图。
  CLI 末尾打印 top-10 排名表。
- **训练 / 验证 / OOS 切分保护器**（[checkmate.split](checkmate/split.py)）— CLI
  `--split "train=2014-2020 val=2021-2023 oos=2024-2026"`。年份简写自动
  补 1/1—12/31；可混用 `YYYY-MM-DD:YYYY-MM-DD` 精确范围。三段独立跑回测
  并在末尾打印 segment summary。当与 `--grid` 同用且网格窗口落在 OOS 区间
  内时，`forbid_rank_on_oos` 抛错并 Exit(2) — 防 cherry-pick 未来数据。
- **Regime breakdown 4×N 表**（PR-6.3）— `report.build_report` 在 by_regime
  桶里加上 `avg_hold_days`（每段平均持仓自然日），共 4 个指标
  （n_trades / total_pnl / win_rate / avg_hold_days）。Markdown / HTML 报告
  的 By regime 表都增加 avg_hold_days 列。Dashboard 新增
  `render_regime_breakdown_card`（在 backtest 模式中可显式调用，
  CLI 集成留给 v0.4）。

### 测试

- 35 个新测试：13 grid（expand_grid 笛卡尔 + yaml 校验 + 3×3 端到端 +
  排名顺序 + max_drawdown ascending）+ 17 split（年份/全显式 parse +
  错误用例 + OOS 守卫包含/部分重叠/无 oos noop）+ 5 regime breakdown
  （4 指标聚合 + markdown/html 列 + Console 快照 + 空表降级）。
  `pytest -q` 245 passed / ~3 min。

### 已知限制

- `--grid` 串行执行（受 Tushare 客户端限流约束）；并行 grid 是 v0.4 选项。
- `--rank-by` 仅支持 BacktestOutcome 自带的 4 个字段；接 ReportPayload
  完整指标（Sharpe / Calmar 等）等 v0.4。
- Regime breakdown 卡片层渲染原语就绪，但 `run_backtest` 的实时事件流
  尚未携带 by_regime 增量，需手动从 report 注入。

---

## v0.2.0 — 2026-05-19

**UI 升级 + 解释页 + 静态 HTML 报告**。沿用 v0.1.0 全部业务逻辑，本版聚焦
"可观察性 / 可解释性 / 可分享"。

### 新增功能

- **EventRenderer 协议层**（[checkmate.ui.protocol](checkmate/ui/protocol.py)）
  — `RenderEvent` dataclass + Protocol + `NullRenderer`；scan / signals /
  backtest 三个 orchestrators 全部改为 renderer 驱动（保留 `echo` 回调
  back-compat 让旧测试零改）。
- **`choose_renderer(no_dashboard, mode)` 工厂** — 5 路 fallback 矩阵：
  `no_dashboard` / 非 TTY / `CI` env / `DEEPTRADE_NO_DASHBOARD` env /
  `TERM=dumb`。`RichDashboardRenderer` 构造失败自动降级，UI 永不挂主流程。
- **`RichDashboardRenderer`** — Rich `Live` 区域；按 mode 渲染 7 个面板：
  Header（plugin / mode / run_id 短码 / 已耗时） / Config / StageStack /
  mode-specific（scan→Funnel；signals→Regime；backtest→Regime + Portfolio）/
  Log。三层容错：on_event try/except + Live 启动失败转 on-demand + factory
  兜底返回 Legacy。
- **`explain --json` schema** — 增 6 个顶层字段共享给 dashboard 卡片：
  `include_reasons`（中文化文案，由 features 启发式派生）/
  `exclude_reasons`（universe.reason_codes 直传）/ `score_breakdown`
  （features 5 组件）/ `entry_plan`（accepted 入场含 shares / stop_price /
  weight）/ `exit_plan`（hard_stop / trailing_stop 等含 details）/
  `risk_snapshot`（rejected 含 cancel_reason，accepted 含 sizing）。
- **`report --html` 静态片段**（[checkmate.report.to_html](checkmate/report.py)）
  — Jinja2 模板 + 内联 SVG：① equity 折线（+/-% delta badge + min/max
  标签 + 端点 marker） ② 月度收益热力图（年 × 月 12 列 HTML 表，cell
  颜色 brightness 由 \|return\| 归一） ③ exit_reason 饼图（industry →
  regime → 占位 fallback） + by_regime / by_industry 表。单文件双击浏览器
  直开，零 JS / 零 CDN / 零相对资源。

### 依赖

- 新增 `Jinja2>=3`（HTML 模板渲染）。

### 测试

- 49 个新测试覆盖：renderer factory 全 fallback 矩阵 / 17 dashboard 单测
  （含 stage_model + RichDashboardRenderer Console.capture() 结构性快照
  + 容错 + 全 lifecycle）/ 13 HTML 报告（3 SVG/heatmap 元素分立 + to_html
  整合 + write_to_disk + 自包含校验）。`pytest -q` 210 passed / ~2 min。

### 已知限制

- 黄金回测仍是 10 只 × ~20 日窗口（CI 时间预算）；完整 50 × 24 月仍由
  release-PR 手工触发并附 JSON。
- 月度热力图按"当月最后一日 equity"vs"上月最后一日 equity"算 month return；
  对于"窗口很短只有一个月"会出空表（这是预期，标 meta 提示）。
- exit_reason 饼图 v0.2 用 industry / regime 占位，专用的 closed-position
  exit_reason 桶要等 v0.3 Iter-6 把分桶维度补到 ReportPayload。

---

## v0.1.0 — 2026-05-19

**首版正式发布**。A 股 long-only 中期趋势跟踪闭环：sync → scan → signals →
backtest → report 五条主线全通，LLM-free / LightGBM-free（permissions.llm:
false），与既有 `limit-up-board` / `volume-anomaly` / `accumulation-probe-washout`
共存。

### 新增功能

- **股票池（universe）** — 历史 ST / 退市状态快照 (`stock_status_history`)
  + 6 类剔除原因码（`st` / `new_listing` / `low_amount` / `thin_trading`
  / `one_way_limit` / `price_band`）+ 流动性评分（亿元/天）。
- **特征（features）** — 五分组 17 个特征：均线（ma20/60/120 + ma_slope60）
  / 波动（Wilder atr20 + atr_pct）/ 强度（ret_60/120 + 横截面 rs_pctile）
  / 流动性（amount/turnover_20d_avg + limit_freq_60d）/ 回踩质量
  （drawdown_60d_high + quiet_score + above_ma20_days）+ 加权 score 与
  `score_breakdown` JSON。
- **市场环境（regime）** — 中证全指 / 沪深300 比 MA(120) + features-driven
  breadth ⇒ 4 状态 (`strong` / `neutral` / `weak` / `risk`) + 4 档
  `exposure_cap`。
- **信号判定（signals）** — 入场三类（突破 / 回踩 / 趋势延续，每类带板块
  pct_chg cap）；退出五类（hard_stop / risk_regime / defensive_profit /
  trailing_stop / time_exit）+ T+1 阻塞保护；同 ts_code 多类入场按优先级
  `breakout > continuation > pullback` 去重；`explain` JSON 含命中 / 失败
  条件列表 + 决策细节。
- **风控（risk）** — 纯函数 `size_position`（ATR 风险预算 + 100 整手
  floor）+ `apply_portfolio_constraints` 四阶段级联（regime/daily cap →
  sizing → single weight → industry cap），按 score 降序消耗 cap budget。
- **撮合（executor）** — 信号日 next-session 开盘 + 4 字段成本拆解
  （commission / stamp_tax / transfer_fee / slippage）；涨停取消买入、
  跌停 defer 卖出（最长 5 日）、跳空 > 5% 取消、T+1 风险事件、停牌缺数据
  取消；全部基于 RAW 价格 + `stk_limit` 交易所口径。
- **回测引擎（backtest）** — trade_cal 单日推进串通 universe→features→
  regime→signals→risk→executor；BLAKE2b-64 `config_hash` 寻址的 per-day
  checkpoint shards（`~/.deeptrade/checkmate/backtests/<hash>/days/*.json`）
  + `--resume` 默认开启 / `--fresh` 清空。
- **报告（report）** — CAGR / Sharpe / max Drawdown / Calmar / win_rate /
  avg_hold_days / limit_blocked_ratio / total_fees + by_regime / by_industry
  跨表 + equity_series；输出 stdout Markdown / JSON / disk-persisted。
- **CLI 七子命令** — `sync` / `scan` / `signals` / `explain` / `backtest` /
  `report` / `settings show|reset`。
- **数据库 10 张表** — 全部 `checkmate_` 前缀 + `purge_on_uninstall: true`：
  `stock_status_history` / `universe_daily` / `features_daily` /
  `regime_daily` / `signals` / `positions` / `trades` / `backtest_runs` /
  `runs` / `events`。

### 测试

- 161 个单测覆盖：calendar / data cache / status_history / universe / features
  / regime / signals (entry+exit) / risk / executor / scan / signals 编排 /
  backtest checkpoint / report；含 1 个黄金合成回测（10 只 × ~20 日窗口，
  metric 区间断言 + 同 config_hash byte-equal 决定性保护）。

### 已知限制 / v0.2+ 路线

- 仅 legacy stdout 渲染器；Rich Dashboard 留给 Iter-5 → v0.2.0。
- 无 LightGBM 评分 ; 设计文档 §1.2 明确首版 ML-free。
- executor 不支持部分成交（v0.4 / Iter-7 引入分钟级冲击成本模型时一并落地）。
- ex-div 日 stop_price 漂移由 signals 层在 Iter-7 + 1.0 校正层处理 ;
  v0.1 接受小幅误差。
- 全样本（2014–2026）回测算力 ≈ 30 min；CI 黄金测仅跑 10 只 × ~6 月窗口
  保证日级回归 ; release-PR 附完整回测产物 JSON 入描述。
