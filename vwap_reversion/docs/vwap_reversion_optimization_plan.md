# vwap_reversion 策略优化改造方案

> 版本：2026-07-01  
> 范围：基于当前 `vwap_reversion` 插件现状、已开放的 Tushare 8000 积分以内接口，以及额外可用的 ETF 实时日线 `rt_etf_k`，设计从模拟盘稳定运行到实盘券商接口替换的改造路线。

## 1. 结论摘要

当前 `vwap_reversion` 已具备可运行的单标的模拟盘闭环：`rt_etf_k` 实时累计快照 -> 快照差分 bar -> 当日 VWAP/成交量加权 sigma -> 固定 z-score 入场/出场 -> Paper 撮合 -> 报告与回放。这个基础是合理的，但仍偏向“最小闭环”，在真实日内交易里主要风险来自四点：

1. **信号过于单一**：固定 `k_entry/k_exit/k_stop` 只看当日 VWAP 偏离，缺少趋势日过滤、流动性约束、开收盘时段规则、盘口/成交活跃度确认，容易在单边行情中逆势接刀。
2. **数据能力没有充分利用**：现有实现主要使用 `rt_etf_k` 和 `trade_cal`，尚未使用 ETF 日线、复权、基金规模、净值、热榜、融资融券标的、涨跌停价、指数日线等数据做盘前过滤、参数自适应和盘后评估。
3. **模拟撮合偏理想化**：PaperBroker 以快照价加固定 bps 滑点成交，没有订单生命周期、部分成交、成交量参与率、价格保护、撤单和券商回报对账模型。
4. **回测样本受限**：目前只能回放已实时采集的 `vwr_bars`。在没有可靠 ETF 历史分钟/逐笔数据的前提下，日线数据不能重建真实日内 VWAP 路径，因此应把“采集-回放一致性”作为主要验证方式，把日线只用于盘前过滤和风险参数校准。

建议改造方向：保持策略核心为“日内 VWAP 偏离回归”，但升级为 **数据分层 + 动态阈值 + 两段式入场 + 交易日状态机 + 订单级模拟撮合 + 可替换 broker 适配器**。先在模拟盘长时间稳定运行，积累足够 `vwr_snapshots/vwr_bars` 后做 walk-forward 回放，再切换到实盘 broker adapter。

## 2. 当前功能诊断

### 2.1 已实现能力

- 数据源：单标的 `rt_etf_k` 实时轮询，字段映射为 `close -> last`、`vol -> cum_vol`、`amount -> cum_amount`、`num -> num_trades`，并保留 `pre_close/open/high/low/trade_time/bid_volume1/ask_volume1`。
- bar 构造：相邻累计快照差分得到区间成交量与成交额；首条快照代表开盘至当前累计；累计量回退时丢弃。
- 引擎：用 `cum_amount / cum_vol` 精确计算当日 VWAP；用区间成交均价累积 `Q` 计算成交量加权 sigma；输出 `z=(last-vwap)/sigma`。
- 信号：
  - `round_trip`：只允许低吸多腿，`z <= -k_entry` 开仓，回归到 `z >= -k_exit` 平仓。
  - `base_position_t`：有底仓时允许高抛低吸，`z >= k_entry` 可卖出底仓形成短腿，回归后买回。
  - 止损：硬止损、带止损、EOD 强平、日亏熔断。
- 风控：最大成交笔数、最短持仓、冷却时间、日亏熔断、EOD 停止交易。
- 回测：回放已落库 `vwr_bars`，复用实盘同一套 engine/session/paper broker。
- 报告：执行报告、交易汇总报告、回测报告。

### 2.2 主要不足

- **缺少标的筛选**：策略假定用户指定代码可交易、流动性足够、适合 T+0，没有盘前合法性、规模、成交额、停牌/摘牌、涨跌停、产品类型检查。
- **固定带宽不适应日内状态**：同一 `k_entry=2.0` 对不同 ETF、不同波动 regime、不同成交时段并不等价。
- **缺少趋势日保护**：VWAP 回归在均值回归日有效，在趋势日容易连续逆势亏损。当前只靠 `k_stop` 和日亏熔断兜底。
- **缺少入场确认**：触及 band 即成交，容易在快速下跌/拉升中被动接单。
- **盘口信息没有使用**：`rt_etf_k` 可提供一档买卖量，但当前未用于流动性/方向确认；同时该接口没有 ETF 一档买卖价，不能把盘口价格当作可得数据。
- **撮合模型过简化**：固定 bps 滑点不能反映成交量参与率、流动性变差、开收盘冲击、部分成交。
- **崩溃恢复只恢复 VWAP，不恢复 paper 持仓意图**：当前注释已说明 aborted run 的虚拟持仓不跨 run 延续。实盘前必须补齐 broker 持仓对账。
- **回测覆盖不足**：没有历史分钟/逐笔数据时，不应宣称能用日线严谨回测日内 VWAP 策略。

## 3. Tushare 可用数据边界

以下接口来自 Tushare MCP 元数据，按本策略可用性分层。

### 3.1 盘中实时核心数据

| 接口 | 用途 | 可用字段要点 | 策略设计含义 |
|---|---|---|---|
| `rt_etf_k` | ETF 实时日 K 快照 | `ts_code/name/pre_close/high/open/low/close/vol/amount/num`，可选 `trade_time/bid_volume1/ask_volume1` | 核心执行数据。可精确计算当日累计 VWAP，可差分成交量/成交额；只能得到快照最新价和一档量，不能得到 ETF 一档买卖价。 |

### 3.2 ETF 盘前/盘后数据

| 接口 | 用途 | 策略设计含义 |
|---|---|---|
| `fund_basic` | 场内基金列表、上市状态、基金类型、管理人、上市/摘牌日期、投资类型等 | 标的合法性、产品白名单、退市/非上市过滤。 |
| `fund_daily` | ETF 日线行情，含开高低收、成交量、成交额 | 计算日线波动率、成交额分位、跳空、趋势/震荡 regime、容量约束。 |
| `fund_adj` | ETF 复权因子 | 复权日线收益、历史波动与回撤计算。 |
| `fund_share` | 基金规模/份额 | 规模过滤、容量限制、异常份额变化监控。 |
| `fund_nav` | 基金净值 | 盘后折溢价粗略校验；不能替代盘中 IOPV。 |
| `fund_div` | 基金分红 | 复权校验、历史跳变解释。 |
| `etf_index` | ETF 基准指数列表 | 建立 ETF 与基准指数/主题的元数据关系。 |

### 3.3 市场状态与风险辅助数据

| 接口 | 用途 | 策略设计含义 |
|---|---|---|
| `trade_cal` | 交易日历 | 启停机、待机、跨日判断，已使用。 |
| `stk_limit` | 每日涨跌停价，包含基金 | 盘前加载 ETF 当日涨跌停；交易信号靠近涨跌停时禁止新开仓或缩小仓位。 |
| `margin_secs` | 融资融券标的，含 ETF | 作为流动性/可交易性标签，不作为 T+0 资格充分条件。 |
| `ths_hot` / `dc_hot` | ETF 热榜，盘中/盘后多次更新 | 拥挤度、热度事件过滤；热度极高时提高入场阈值或进入观察模式。 |
| `index_basic` / `index_daily` | 指数基础与日线 | 基准指数 regime、市场趋势过滤。 |
| `rt_idx_min` | 指数实时分钟 | 若 ETF 有明确基准指数且接口覆盖该指数，可作为盘中趋势过滤；不能替代 ETF 自身价格。 |
| `moneyflow_ind_ths` | 行业资金流，盘后 | 对行业/主题 ETF 做次日 regime 辅助，不用于盘中实时决策。 |

### 3.4 不应依赖或需谨慎使用的数据

- 日线数据无法重建真实日内 VWAP 路径，不能直接用于严谨的日内信号回测。
- `rt_min` / `rt_min_daily` 标注为 A 股实时分钟；除非实际验证支持 ETF 代码，否则不作为 ETF 策略核心依赖。
- `fund_nav` 是净值口径，不是盘中 IOPV；只能作为盘后折溢价风险参考。
- `rt_etf_k` 没有 ETF 一档买卖价字段，不能用它估算真实 spread，只能用 `bid_volume1/ask_volume1` 做弱流动性/方向信号。
- T+0 资格不能仅由 Tushare 基金元数据自动推断，建议使用用户维护的 T+0 ETF 白名单，并由策略做代码级校验。

## 4. 改造目标

### 4.1 策略目标

- 只交易可确认 T+0 且流动性足够的 ETF。
- 用 `rt_etf_k` 的累计量/额保持 VWAP 计算精确性。
- 在震荡/回归环境下做 VWAP 偏离回归；在趋势日、低流动性、数据异常时少交易或不交易。
- 所有实盘前逻辑先经过 PaperBroker 和回放验证。

### 4.2 工程目标

- 数据、信号、风控、执行、报告分层清晰。
- Paper broker 与真实 broker 共享同一订单接口，实盘替换只替换 adapter。
- 所有信号、抑制原因、订单状态、成交、持仓和对账结果可审计。
- 回测报告明确区分“已采集分钟级回放”和“日线辅助统计”，避免误导。

## 5. 功能设计改造

### 5.1 数据层：从单实时源升级为分层数据服务

新增 `vwap_reversion/data/` 或 `vwap_reversion/feed/` 下的数据服务：

- `EtfUniverseService`
  - 输入：用户白名单、`fund_basic`、`fund_share`、`fund_daily`、`margin_secs`。
  - 输出：可交易 ETF 列表与标签。
  - 过滤建议：
    - `fund_basic.status = L`，`market = E`。
    - 在用户维护的 `t0_whitelist` 内。
    - 最近 20 日平均成交额 >= 配置阈值，例如 2 亿。
    - 最近 20 日成交额分位不低于过去 1 年 30% 分位。
    - 基金份额/规模过小或剧烈变化时降权或禁用。

- `EtfDailyFeatureService`
  - 数据：`fund_daily`、`fund_adj`、`fund_share`、`fund_nav`、`stk_limit`。
  - 产出盘前特征：
    - `ret_1d/5d/20d`、`rv_20d`、`atr_pct_20d`、`gap_pct`。
    - `amount_ma20`、`amount_pctile_252`、`volatility_regime`。
    - `premium_discount`（盘后粗略值），异常时次日降低仓位。
    - 当日涨跌停价，作为价格保护边界。

- `RealtimeEtfSourceV2`
  - 继续使用 `rt_etf_k`，但加强质量控制：
    - `trade_time` 不前进且 `cum_vol` 不变超过 N 次，标记 stale。
    - `close` 超出当日 `high/low` 或超过涨跌停价，丢弃并告警。
    - `cum_amount/cum_vol` 隐含 VWAP 与 `last` 偏离异常时告警。
    - 上午/下午 session 切换时不把午休无成交当作异常。
  - 保存原始 row 的关键字段快照，便于排查上游数据。

建议新增表：

| 表 | 用途 |
|---|---|
| `vwr_etf_universe` | ETF 基础信息、白名单标签、是否启用、T+0 标签。 |
| `vwr_etf_daily` | ETF 日线、复权因子、规模、净值与涨跌停价缓存。 |
| `vwr_daily_features` | 盘前特征与 regime 标签。 |
| `vwr_data_quality` | 实时数据异常、stale、累计回退、字段缺失记录。 |

### 5.2 Bar 与 VWAP 引擎：保留精确 VWAP，新增稳健尺度

当前 VWAP 计算方式应保留：`VWAP = cum_amount / cum_vol`，这是 `rt_etf_k` 能提供的最可靠日内锚。

建议新增第二层尺度估计，避免单一全日成交量加权 sigma 在开盘和趋势日失真：

- `session_sigma`：现有成交量加权 sigma，继续用于报告和基础 z。
- `rolling_residual_sigma`：最近 N 根 bar 的 `last - vwap` 或 `log(last/vwap)` robust sigma，可用 MAD 或 winsorized std。
- `tod_sigma_prior`：按历史已采集 bar 统计的时间段先验波动，例如 09:35、10:00、14:30 各自的残差分布。
- 最终 z：

```text
residual = log(last / vwap)
scale = max(session_sigma_pct, rolling_sigma_pct, tod_sigma_prior_pct, min_sigma_pct)
z = residual / scale
```

这样可以避免早盘 sigma 过小导致误触发，也避免午后趋势行情中全日 sigma 被早盘成交稀释。

新增配置建议：

```text
scale_mode = hybrid
rolling_window_bars = 20
min_sigma_bps = 8
mad_winsor_pct = 0.02
tod_prior_min_days = 20
```

### 5.3 信号层：固定阈值升级为状态机

将当前“触带即开仓”改为两段式：

1. **Arm 阶段**：价格偏离 VWAP 到达阈值，只记录候选，不立即下单。
2. **Confirm 阶段**：出现回归迹象才开仓，例如：
   - `z` 从极值回收 `confirm_z_recover`。
   - 最近 1-2 根 bar 价格不再创新低/新高。
   - 区间成交额恢复但价格跌速/涨速放缓。
   - 一档量不明显反向失衡：低吸时 `bid_volume1/ask_volume1` 不低于阈值，高抛时相反。

入场规则示例：

```text
低吸 arm: z <= -entry_z_dynamic
低吸 confirm:
  z >= min(armed_z + confirm_z_recover, -entry_z_dynamic + confirm_z_recover)
  and price >= recent_low * (1 + min_rebound_bps)
  and not trend_day_down
  and liquidity_ok
```

出场规则建议：

- 主止盈：`z >= -exit_z_dynamic` 或 `last >= vwap - exit_band`。
- 时间止盈/止损：持仓超过 `max_holding_seconds` 仍未回归，按当前 z/盈亏分层处理。
- 分段平仓：可选 50% 在 `z=-0.5` 平，剩余在 `z=0` 或 VWAP 附近平。
- 趋势反证止损：入场后 VWAP 斜率继续恶化、价格继续远离、成交放大时提前止损。
- EOD：保持当前强平，但建议 `new_entry_cutoff_time` 早于 `eod_flat_time`，例如 14:40 后不再新开。

新增配置建议：

```text
signal_version = v2
entry_z_base = 2.0
exit_z_base = 0.3
stop_z_base = 3.5
confirm_z_recover = 0.35
min_rebound_bps = 3
max_holding_seconds = 1800
new_entry_cutoff_time = 14:40
allow_partial_exit = true
partial_exit_ratio = 0.5
```

### 5.4 Regime 过滤：避免在趋势日机械逆势

VWAP 回归策略最需要区分“震荡偏离”和“趋势偏离”。建议实现日内 regime classifier，不需要机器学习，先用规则足够：

盘前 regime：

- `fund_daily` 计算近 20/60 日波动率、近 5/20 日收益、成交额分位。
- 近期单边上涨/下跌且成交额放大时，提高 `entry_z` 或降低仓位。
- 前一日大幅跳空/大阴大阳后，早盘延长 warmup。

盘中 regime：

- `price_vs_open = last/open - 1`。
- `vwap_slope`：最近 N 根 VWAP 斜率。
- `range_position = (last - low) / (high - low)`。
- `trend_strength = abs(last - vwap) / intraday_range`。
- `volume_acceleration`：当前累计成交额相对历史同时间分位，若后续有足够采集样本。

过滤示例：

- 下跌趋势日：`last < open`、`vwap_slope < -threshold`、`last` 持续贴近日内低位、成交放大，则禁止低吸或把 `entry_z` 提高 30%-50%。
- 上涨趋势日：`base_position_t` 高抛腿同理过滤，避免过早卖飞底仓。
- 低成交日：降低交易频率或不交易。

### 5.5 动态阈值与仓位

把固定 `band_k_entry` 拆成基础阈值与动态乘数：

```text
entry_z_dynamic = entry_z_base
  * volatility_multiplier
  * liquidity_multiplier
  * regime_multiplier
  * time_of_day_multiplier
```

建议默认：

- 高波动/趋势日：阈值提高，仓位降低。
- 高流动性且历史回归胜率较好的时段：阈值略降低。
- 09:30-09:45 与 14:45 后：阈值提高或禁开。
- 午后第一根有效 bar：只恢复采集，不立即交易。

仓位 sizing：

```text
target_qty = min(
  configured_order_qty,
  cash_budget / last,
  participation_cap * recent_interval_vol,
  daily_amount_cap / last,
  volatility_target_cash / intraday_vol_pct
)
```

其中 `participation_cap` 初始建议 5%-10%，避免模拟盘产生无法实盘成交的量。

### 5.6 风控：从交易次数风控升级为账户/标的/数据三类风控

保留现有风控，新增：

- 数据风控：
  - stale 行情超过 N 秒停止新开仓。
  - 累计量回退、价格越界、涨跌停价附近禁止新开。
  - 连续拉取失败时若有持仓，按最后有效价进入风控状态，恢复行情后优先评估平仓。

- 账户风控：
  - 单标的最大名义敞口。
  - 单日最大换手金额。
  - 连续亏损腿数达到阈值后停止交易。
  - 最大订单参与率。

- 策略风控：
  - 每个 session 最多开仓次数。
  - 同向连续止损后进入冷静期。
  - `base_position_t` 模式下，保证 EOD 回到底仓数量。

新增配置建议：

```text
max_notional_pct = 0.25
max_turnover_pct = 2.0
max_consecutive_losses = 2
max_participation_rate = 0.10
stale_quote_seconds = 90
limit_price_guard_bps = 20
```

### 5.7 模拟撮合：升级为订单级 PaperBroker

实盘前 PaperBroker 应尽量模拟真实 broker 行为，而不是只在 signal 上立即 fill：

新增抽象：

```python
class Broker(Protocol):
    def submit_order(self, order: OrderIntent) -> OrderAccepted: ...
    def cancel_order(self, order_id: str) -> CancelResult: ...
    def poll(self) -> list[BrokerEvent]: ...
    def positions(self) -> list[Position]: ...
    def account(self) -> AccountSnapshot: ...
```

PaperBroker V2 行为：

- 支持订单状态：`submitted/accepted/partially_filled/filled/canceled/rejected/expired`。
- 支持 marketable limit：买入限价 = `last * (1 + price_protect_bps)`，卖出限价 = `last * (1 - price_protect_bps)`。
- 支持部分成交：单根 bar 可成交量不超过 `participation_rate * interval_vol`。
- 滑点模型：

```text
slippage_bps = base_slippage_bps
             + impact_coef * sqrt(order_qty / max(interval_vol, 1))
             + stale_penalty_bps
             + close_period_penalty_bps
```

- ETF 费用模型独立配置：佣金、最低佣金、平台费；不默认加入股票印花税。

建议新增表：

| 表 | 用途 |
|---|---|
| `vwr_orders` | 策略订单意图与状态。 |
| `vwr_order_events` | broker 回报、撤单、拒单、部分成交。 |
| `vwr_positions` | paper 持仓快照，支持崩溃恢复。 |
| `vwr_account_snapshots` | 资金、权益、可用现金、风险指标快照。 |

### 5.8 实盘切换路径

你已开通券商实盘交易接口，但建议按以下顺序切换：

1. **Paper only**：策略只连 Tushare 行情和 PaperBroker V2，至少运行 20-60 个交易日。
2. **Shadow live**：连接券商接口只读账户/持仓，不提交订单；比较 paper 订单与真实可交易约束。
3. **Live dry-run**：生成真实订单请求但在 adapter 层拦截，验证参数、风控、对账。
4. **Small capital live**：启用真实下单，单笔与单日额度极小，开启手动 kill switch。
5. **Scale up**：只有在订单拒绝率、滑点、对账异常、回撤均达标后扩大额度。

实盘 adapter 必须具备：

- 下单幂等键，避免重试重复下单。
- 持仓和订单回报对账，启动时以券商真实持仓为准。
- 交易前价格保护和数量合法性检查。
- 全局 kill switch：本地配置、命令行、异常自动触发三种路径。
- 所有真实委托和回报落库，不允许只记策略信号。

### 5.9 回测与评估体系

在当前数据权限下，应分三层评估：

1. **日线研究层**
   - 使用 `fund_daily/fund_adj/fund_share/fund_nav/stk_limit`。
   - 评估标的流动性、波动、跳空、容量、趋势 regime。
   - 不评估日内 VWAP 入场点收益。

2. **实时采集回放层**
   - 使用已采集的 `vwr_snapshots/vwr_bars`。
   - 这是策略信号与撮合的主要回测样本。
   - 要求“同参数同数据，paper run 与 backtest replay 信号/订单一致”。

3. **模拟盘前向验证层**
   - 每日 paper 运行，收盘生成报告。
   - 每周做 walk-forward：用过去 N 天调参，只评估未来 M 天。
   - 明确记录跳过日、异常日、接口失败日，不从样本中静默删除。

新增报告指标：

- 信号质量：触发次数、实际开仓次数、抑制原因分布、入场后最大不利/有利偏移 MAE/MFE。
- 交易质量：平均滑点、参与率、部分成交率、拒单率、订单存活时间。
- 收益风险：净盈亏、胜率、profit factor、最大回撤、日亏熔断次数、连续亏损。
- 稳定性：行情 stale 次数、累计回退次数、接口失败恢复时间。
- 分组统计：按标的、时间段、regime、热榜状态、波动率分位分组。

## 6. CLI 与配置建议

### 6.1 新增命令

```text
deeptrade vwap-reversion universe sync
deeptrade vwap-reversion universe show
deeptrade vwap-reversion daily sync --code 159518.SZ --start 20240101 --end 20260630
deeptrade vwap-reversion features build --date 20260701
deeptrade vwap-reversion run --code 159518.SZ --profile conservative
deeptrade vwap-reversion replay --code 159518.SZ --start 20260601 --end 20260630 --signal-version v2
deeptrade vwap-reversion broker shadow
deeptrade vwap-reversion broker reconcile
deeptrade vwap-reversion kill-switch on
```

### 6.2 配置分组

建议从当前扁平 `VwrConfig` 逐步演化为分组 dataclass，但数据库仍可保持 key-value：

- `MarketDataConfig`：轮询、stale、字段校验、缓存。
- `UniverseConfig`：白名单、成交额、规模、T+0 标签。
- `SignalConfig`：scale、动态阈值、确认入场、时间规则。
- `RiskConfig`：账户/订单/数据风控。
- `ExecutionConfig`：paper/live、滑点、参与率、价格保护。
- `BrokerConfig`：券商 adapter、只读/shadow/live 模式。

## 7. 实施路线

### P1：数据与可观测性补强

- 新增 ETF 日线/基础信息/规模/涨跌停价同步。
- 新增数据质量表与 stale 检测。
- 报告中加入数据质量与流动性章节。
- 不改变交易信号，降低引入风险。

验收：

- `python tools/check_registry.py` 通过。
- 新增数据服务单测覆盖字段缺失、空表、异常日期、涨跌停价过滤。
- 现有 `vwap_reversion` 测试零回归。

### P2：信号 V2 与动态阈值

- 引入 hybrid z-score、两段式入场、趋势日过滤、时间段规则。
- 保留 `signal_version = v1` 兼容当前策略。
- 报告输出 v1/v2 对比信号。

验收：

- synthetic 趋势日不逆势频繁开仓。
- synthetic 震荡日能在确认回归后入场。
- 所有抑制原因可落库并可报告。

### P3：订单级 PaperBroker V2

- 新增订单状态机、部分成交、参与率滑点、持仓快照。
- `TradingSession` 从“信号即成交”改为“信号 -> 订单 -> broker events -> position”。
- 保留旧 PaperBroker 用于 parity 测试，逐步迁移。

验收：

- 订单状态单测覆盖成交、部分成交、撤单、拒单、过期。
- 崩溃恢复后 paper 持仓和未完成订单可重建。
- 回放结果确定性一致。

### P4：前向模拟盘稳定期

- 使用真实 Tushare 行情连续 paper 运行。
- 每日自动生成执行、交易、数据质量、信号诊断报告。
- 每周生成 walk-forward 汇总。

进入下一阶段条件：

- 连续至少 20 个交易日无未解释崩溃。
- 行情异常均能降级处理。
- Paper 订单拒绝/过期/滑点逻辑可解释。
- 策略在扣费滑点后没有依赖单一异常日盈利。

### P5：券商接口 shadow 与实盘小额

- 实现 broker adapter，但默认只读/shadow。
- 启动时真实持仓对账。
- 小额真实下单前，必须通过 broker dry-run 和 kill switch 测试。

验收：

- 真实账户只读对账连续多日稳定。
- shadow 订单参数与券商规则一致，无非法数量/价格。
- 所有 broker 回报落库。

## 8. 需要避免的设计误区

- 不用 ETF 日线假装能回测分钟 VWAP 策略。
- 不把 `rt_etf_k.close` 当成可成交买卖价，只能作为最新价近似。
- 不在趋势日无条件均值回归。
- 不用热榜作为买卖方向信号，只作为拥挤度/风险标签。
- 不在没有真实持仓对账的情况下接入实盘卖出逻辑。
- 不让策略自动推断 T+0 资格，T+0 ETF 必须白名单化。

## 9. 推荐默认策略画像

保守版初始参数：

```text
poll_interval_seconds = 10-30
warmup_minutes = 20
new_entry_cutoff_time = 14:40
entry_z_base = 2.2
exit_z_base = 0.3
stop_z_base = 3.8
confirm_z_recover = 0.35
min_rebound_bps = 3
max_holding_seconds = 1800
max_trades_per_day = 6
max_consecutive_losses = 2
max_participation_rate = 0.05
daily_loss_limit_pct = 0.8-1.2
stale_quote_seconds = 90
limit_price_guard_bps = 20
```

适用场景：

- 高流动性 T+0 ETF。
- 日内非强趋势、成交活跃但不过热。
- 以小仓位多次验证，不追求高频频繁交易。

## 10. 优先级建议

最高优先级：

1. T+0 ETF 白名单与 ETF 日线/规模/涨跌停数据同步。
2. 数据质量风控和 stale 停止新开仓。
3. 两段式入场与趋势日过滤。
4. 订单级 PaperBroker 与持仓恢复。
5. broker adapter shadow/reconcile，再小额实盘。

中等优先级：

1. 热榜/资金流/指数 regime 辅助。
2. 分段止盈和动态仓位。
3. 多标的扫描与自动选择。

低优先级：

1. 复杂机器学习预测。
2. 过度依赖盘口量的方向判断。
3. 在样本不足时做大规模参数搜索。

## 11. 最小落地切片

如果希望尽快开始改造，建议第一批 PR 只做以下内容：

1. 新增 `docs/vwap_reversion_optimization_plan.md`。
2. 新增 ETF 日线与基础信息缓存表。
3. 新增 `universe sync/show` 和 `daily sync` 命令。
4. 新增 `stale_quote_seconds`、`new_entry_cutoff_time`、`max_consecutive_losses` 配置。
5. 在现有信号前加数据质量和时间窗闸门。
6. 报告增加“数据质量/被抑制信号原因/流动性”章节。

这批改动不会改变核心 VWAP 数学，也不会碰券商实盘接口，风险最低，但能显著提高模拟盘结果的可信度。
