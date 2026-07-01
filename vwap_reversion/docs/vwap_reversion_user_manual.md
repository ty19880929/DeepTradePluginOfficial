# vwap-reversion 用户手册

本文档基于 `vwap-reversion` 当前已实现功能编写，适用于插件版本 `0.2.0`。命令统一通过 DeepTrade 插件分发执行：

```bash
deeptrade vwap-reversion <subcommand> [options]
```

## 1. 策略定位与当前边界

`vwap-reversion` 是单标的 ETF 日内 VWAP 带回归策略插件。当前主流程是：

1. 轮询 Tushare `rt_etf_k` 实时 ETF 快照。
2. 用累计成交量和累计成交额差分生成区间 bar。
3. 在线计算当日 VWAP、成交量加权 sigma、上下轨和 z-score。
4. 根据 z-score 产生回归交易信号。
5. 使用插件内置 Paper 撮合器做模拟成交。
6. 收盘生成执行报告和交易汇总报告。

当前已实现的实盘形态是“实时采集 + 模拟交易”。插件内已有 broker adapter 边界和 shadow/order-paper 组件，但当前 CLI 的 `run` 命令仍使用同步 Paper 撮合，不会向真实券商下单。

当前回测是“回放已采集的 `vwr_bars`”。它不会用 ETF 日线数据重建日内 VWAP 路径。因此，必须先在实时 `run` 中采集过目标日期的数据，才能对这些日期做 `backtest`。

## 2. 数据与信号逻辑

### 2.1 实时行情

实时行情来源为 Tushare `rt_etf_k`。插件读取的关键字段包括：

- `close`：最新价，作为策略观察价和模拟成交参考价。
- `vol`：当日累计成交量。
- `amount`：当日累计成交额。
- `pre_close/open/high/low`：用于快照质量保护。
- `trade_time`、`bid_volume1`、`ask_volume1`：可落库留痕，当前不直接驱动信号。

上海 ETF 代码以 `.SH` 结尾时，实时源会自动附加 `topic=HQ_FND_TICK`。

### 2.2 bar 构建

插件保存每次实时快照到 `vwr_snapshots`，并用相邻快照做差：

- 首条有效快照的区间量为“开盘至当前”的累计量。
- 后续区间量为当前累计量减上一快照累计量。
- 没有新增成交量时不产生 bar，仅记录采样进度。
- 如果累计成交量或成交额回退，视为数据异常，本条快照不污染 VWAP 引擎。

崩溃或中断后同日重启时，插件会用已落库的 `vwr_bars` 重建 VWAP 引擎，并用最后一条快照对齐差分基线，避免重复计算开盘以来成交量。注意：Paper 虚拟持仓不会跨 run 延续，新 run 的交易状态从配置锚点重新开始。

### 2.3 VWAP、sigma 与 z-score

当前引擎使用成交量加权口径：

```text
VWAP = 当日累计成交额 / 当日累计成交量
sigma = sqrt(Q / V - VWAP^2)
z = (last - VWAP) / sigma
```

其中 `Q` 由每根区间 bar 的 `interval_amount^2 / interval_vol` 累加得到。`sigma <= 0` 时 z 为 `None`，不会产生交易信号。

### 2.4 入场与出场

当前支持两个信号版本：

- `v1`：单阶段 z-score 入场。低于 `-band_k_entry` 时做低吸多头腿；在底仓模式下，高于 `+band_k_entry` 时可做高抛空头腿。
- `v2`：两段式确认入场。先到达偏离阈值进入 armed 状态，再要求价格反弹或回落确认、z-score 回收确认，并通过趋势保护和高波动动态阈值。

持仓后的平仓优先级为：

1. `stop_hard`：价格相对入场价达到 `per_trade_stop_pct` 硬止损。
2. `stop_band`：z-score 继续恶化并穿过 `band_k_stop`。
3. `time_exit`：持仓时间超过 `max_holding_seconds`。
4. `revert_exit`：z-score 回归到退出阈值以内。
5. `circuit_break` 或 `eod_flat`：日亏熔断或收盘强平。

`min_holding_seconds` 只会抑制 `revert_exit`，不会阻止止损、熔断或 EOD 强平。

## 3. 配置管理

查看当前配置：

```bash
deeptrade vwap-reversion settings show
```

设置单个配置项：

```bash
deeptrade vwap-reversion settings set band_k_entry 2.5
deeptrade vwap-reversion settings set standby_across_days true
deeptrade vwap-reversion settings set eod_flat_time 14:50
```

配置值优先按 JSON 解析；无法解析时作为裸字符串保存。因此布尔值建议写 `true` 或 `false`，字符串时间可直接写 `14:50`。

重置为默认值：

```bash
deeptrade vwap-reversion settings reset --yes
```

`run` 和 `backtest` 的部分参数也支持命令行一次性覆盖。一次性覆盖只影响当次执行，不会写入 `vwr_config`。

## 4. 可配置参数详解

### 4.1 时区、待机与采集

| 参数 | 默认值 | 作用 | 约束与说明 |
| --- | --- | --- | --- |
| `market_timezone` | `Asia/Shanghai` | 市场时区。所有“今天”、开盘、午休、收盘、EOD 判断都使用该时区。 | 必须是可解析的 IANA 时区。Windows 环境依赖 `tzdata`。 |
| `standby_heartbeat_seconds` | `60` | 开盘前待机时的心跳间隔。 | 必须 `>= 5`。 |
| `standby_across_days` | `false` | 收盘后或非交易日启动时，是否等待到下一交易日 09:30。 | `false` 时直接退出并提示下一交易日；`true` 时跨日守候。 |
| `poll_interval_seconds` | `30` | 实时行情轮询间隔。 | 必须 `>= 5`。`run --poll-interval` 可一次性覆盖。 |

示例：

```bash
deeptrade vwap-reversion settings set poll_interval_seconds 15
deeptrade vwap-reversion settings set standby_across_days true
```

### 4.2 VWAP 带与信号

| 参数 | 默认值 | 作用 | 约束与说明 |
| --- | --- | --- | --- |
| `band_mode` | `vol_weighted` | 带宽模式配置项。 | 允许 `vol_weighted` 或 `time_std`。当前 VWAP 引擎实际使用成交量加权 sigma，建议保持默认值。 |
| `band_k_entry` | `2.0` | 入场 z-score 阈值。 | 必须 `> 0`。低吸多头腿使用 `z <= -band_k_entry`；底仓高抛腿使用 `z >= band_k_entry`。 |
| `band_k_exit` | `0.3` | 回归止盈阈值。 | 必须 `>= 0` 且 `< band_k_entry`。多头腿在 `z >= -band_k_exit` 时退出；底仓高抛腿在 `z <= band_k_exit` 时回补。 |
| `band_k_stop` | `3.5` | 带止损阈值。 | 必须 `> band_k_entry`。只有偏离相对入场继续恶化时才触发，避免“跳空入场后立刻带止损”。 |
| `warmup_minutes` | `15` | 开盘后预热时间。 | 必须 `>= 0`。预热期内采集和计算照常进行，但不交易。 |
| `signal_version` | `v1` | 信号版本。 | 允许 `v1` 或 `v2`。 |
| `confirm_z_recover` | `0.35` | `v2` 入场确认要求。 | 必须 `>= 0`。armed 后 z 需要相对 armed z 回收至少该值。 |
| `min_rebound_bps` | `3.0` | `v2` 价格确认要求。 | 必须 `>= 0`。低吸要求从最低价反弹至少该 bps；高抛要求从最高价回落至少该 bps。 |
| `max_holding_seconds` | `1800` | 单腿最长持有时间。 | 必须 `>= 0`。设为 `0` 表示关闭时间退出。 |
| `high_vol_sigma_bps` | `60.0` | `v2` 高波动判断阈值。 | 必须 `>= 0`。当 `sigma / vwap * 10000` 达到该值时，入场阈值会放大。 |
| `high_vol_entry_multiplier` | `1.2` | `v2` 高波动入场阈值乘数。 | 必须 `>= 1.0`。例如默认高波动下 `2.0` 会变为 `2.4`。 |
| `trend_guard_vwap_slope_bps` | `3.0` | `v2` VWAP 趋势保护。 | 必须 `>= 0`。设为 `0` 可关闭。低吸时若 VWAP 下行斜率过强会阻止入场；高抛腿对称处理。 |

较保守的示例：

```bash
deeptrade vwap-reversion settings set band_k_entry 2.5
deeptrade vwap-reversion settings set band_k_exit 0.2
deeptrade vwap-reversion settings set band_k_stop 4.0
deeptrade vwap-reversion settings set signal_version v2
deeptrade vwap-reversion settings set high_vol_entry_multiplier 1.5
```

较积极的示例：

```bash
deeptrade vwap-reversion settings set band_k_entry 1.8
deeptrade vwap-reversion settings set band_k_exit 0.4
deeptrade vwap-reversion settings set band_k_stop 3.2
deeptrade vwap-reversion settings set warmup_minutes 5
```

### 4.3 持仓模式

| 参数 | 默认值 | 作用 | 约束与说明 |
| --- | --- | --- | --- |
| `position_mode` | `round_trip` | 持仓锚点模式。 | 允许 `round_trip` 或 `base_position_t`。`run --position-mode` 和 `backtest --position-mode` 可一次性覆盖。 |
| `base_shares` | `0` | 底仓数量。 | 必须 `>= 0` 且为 100 的整数倍。`position_mode=base_position_t` 时必须 `> 0`。 |
| `order_qty` | `100` | 每次模拟成交数量。 | 必须为正的 100 整数倍。 |

`round_trip`：锚点为空仓，只允许低吸多头腿，入场买入、回归卖出。适合只验证日内回归低吸效果。

`base_position_t`：锚点为用户假设已有底仓。允许两类腿：

- 低吸多头腿：在底仓基础上加仓，回归后卖出加仓部分。
- 高抛空头腿：先卖出部分底仓，回落后买回。这里不是裸卖空，而是围绕已有底仓做 T。

底仓模式示例：

```bash
deeptrade vwap-reversion settings set position_mode base_position_t
deeptrade vwap-reversion settings set base_shares 1000
deeptrade vwap-reversion settings set order_qty 100
```

### 4.4 风控

| 参数 | 默认值 | 作用 | 约束与说明 |
| --- | --- | --- | --- |
| `max_trades_per_day` | `10` | 当日最大成交 fill 数。 | 必须 `>= 1`。开仓和平仓各算一笔。实际开新腿时会预留一笔平仓额度，因此建议至少设为 `2`。 |
| `min_holding_seconds` | `60` | 最短持有时间防抖。 | 必须 `>= 0`。只抑制 `revert_exit`，不抑制止损、熔断、EOD。 |
| `cooldown_seconds` | `60` | 平仓后再次开仓冷却时间。 | 必须 `>= 0`。 |
| `per_trade_stop_pct` | `0.8` | 单腿硬止损百分比。 | 必须 `> 0`。多头腿价格跌破入场价该比例触发；底仓高抛腿价格上涨该比例触发。 |
| `daily_loss_limit_pct` | `1.5` | 日亏熔断阈值。 | 必须 `> 0`。亏损达到 `initial_cash * daily_loss_limit_pct%` 时触发，平掉当前腿并停止新开仓。 |
| `kill_switch_enabled` | `false` | 全局停手开关。 | 开启后只抑制新开仓，不阻止已有持仓平仓。可用 `kill-switch on/off` 快速切换。 |
| `max_consecutive_losses` | `2` | 连续亏损腿上限。 | 必须 `>= 1`。达到后抑制当日新开仓。盈利平仓会清零连续亏损计数。 |
| `stale_quote_seconds` | `90` | 行情未推进保护。 | 必须 `>= poll_interval_seconds`。超过阈值无成交量、成交额或交易时间推进时，暂停新开仓；行情恢复后解除。 |
| `limit_price_guard_bps` | `20.0` | 快照价格质量保护。 | 必须 `>= 0`。若 `last` 明显高于 `high` 或低于 `low` 超过该 bps，本次采样跳过。 |
| `new_entry_cutoff_time` | `14:40` | 当日停止新开仓时间。 | 必须为 `HH:MM`，且在 `09:30` 之后、早于 `eod_flat_time`。只抑制新开仓。 |
| `eod_flat_time` | `14:55` | EOD 强平时间。 | 必须为 `HH:MM`，且在 `13:00` 到 `15:00` 之间。到点后平掉偏离腿并停止交易，只继续采集。 |

停手示例：

```bash
deeptrade vwap-reversion kill-switch on
deeptrade vwap-reversion kill-switch status
deeptrade vwap-reversion kill-switch off
```

### 4.5 模拟账户与成本

| 参数 | 默认值 | 作用 | 约束与说明 |
| --- | --- | --- | --- |
| `initial_cash` | `100000.0` | Paper 账户初始现金。 | 必须 `> 0`。日亏熔断也以该值为基准。 |
| `fee_bps` | `0.5` | 佣金，单位 bps。 | 必须 `>= 0`。ETF 无印花税模型。 |
| `min_fee_per_trade` | `0.0` | 每笔成交最低佣金，单位元。 | 必须 `>= 0`。佣金按 `max(成交额 * fee_bps / 10000, min_fee_per_trade)` 计算。 |
| `slippage_bps` | `1.0` | 滑点，单位 bps。 | 必须 `>= 0`。买入成交价按参考价上浮，卖出按参考价下浮。 |

成本模型示例：`fee_bps=0.5` 表示成交额的 0.005%，`slippage_bps=1.0` 表示每次成交按 0.01% 不利滑点调整。

```bash
deeptrade vwap-reversion settings set initial_cash 200000
deeptrade vwap-reversion settings set fee_bps 0.8
deeptrade vwap-reversion settings set min_fee_per_trade 0
deeptrade vwap-reversion settings set slippage_bps 2.0
```

按“每笔交易金额的万分之 2.5，最低 5 元”计算佣金：

```bash
deeptrade vwap-reversion settings set fee_bps 2.5
deeptrade vwap-reversion settings set min_fee_per_trade 5.0
```

## 5. 命令参考

### 5.1 实时模拟运行

```bash
deeptrade vwap-reversion run --code 159518.SZ
```

可选参数：

```bash
deeptrade vwap-reversion run --code 159518.SZ --poll-interval 15
deeptrade vwap-reversion run --code 159518.SZ --position-mode base_position_t
deeptrade vwap-reversion run --code 159518.SZ --no-dashboard
```

`--date` 仅允许填市场时区的今天。实时轮询历史日期没有意义，其他日期会拒绝启动。

启动时机：

- 交易日 09:30 前启动：创建 standby run，待机到 09:30 自动开始。
- 交易日 09:30 到 15:00 启动：立即运行；午休期间会运行但 sleep 到 13:00 恢复采集。
- 交易日 15:00 后或非交易日启动：默认退出；若 `standby_across_days=true`，等待到下一交易日 09:30。

运行中断：

- Ctrl-C 会把当前 run 标记为 `aborted`。
- 同日再次启动会重建 VWAP 计算状态，但不会继承上一 run 的 Paper 虚拟持仓。

### 5.2 回放回测

```bash
deeptrade vwap-reversion backtest --code 159518.SZ --start 20260601 --end 20260630
```

可选一次性覆盖：

```bash
deeptrade vwap-reversion backtest \
  --code 159518.SZ \
  --start 20260601 \
  --end 20260630 \
  --k-entry 2.5 \
  --k-exit 0.2 \
  --k-stop 4.0 \
  --warmup-minutes 10 \
  --position-mode round_trip
```

注意：

- 回测只读取已落库的 `vwr_bars`。
- 若日期窗口内没有已采集 bar，会报错提示先使用 `run` 采集。
- 回测复用实时 run 的 `VwapEngine + TradingSession + PaperBroker`，尽量保证“回放=实盘”。
- 每个交易日独立结算，`initial_cash` 每天重置，聚合指标写入 backtest run 的 `result_json`。

### 5.3 报告

查看或重新生成最近一次 run 的报告：

```bash
deeptrade vwap-reversion report
```

查看指定 run：

```bash
deeptrade vwap-reversion report --run-id <run_id>
```

只看执行报告或交易报告：

```bash
deeptrade vwap-reversion report --kind exec
deeptrade vwap-reversion report --kind trades
deeptrade vwap-reversion report --kind both
```

paper run 会生成两份 markdown：

```text
~/.deeptrade/vwap_reversion/reports/<code>/<trade_date>/execution_report.md
~/.deeptrade/vwap_reversion/reports/<code>/<trade_date>/trades_report.md
```

backtest 会生成：

```text
~/.deeptrade/vwap_reversion/reports/<code>/backtest_<start>_<end>/backtest_report.md
```

### 5.4 历史 run

```bash
deeptrade vwap-reversion history
deeptrade vwap-reversion history --code 159518.SZ
deeptrade vwap-reversion history --mode paper --limit 50
deeptrade vwap-reversion history --mode backtest
```

### 5.5 ETF 交易池缓存

同步 ETF 基础池：

```bash
deeptrade vwap-reversion universe sync
```

同步并标记用户维护的 T+0 白名单：

```bash
deeptrade vwap-reversion universe sync --t0-whitelist 159518.SZ,510300.SH
```

同步指定日期融资融券标记：

```bash
deeptrade vwap-reversion universe sync --margin-date 20260630
```

查看缓存：

```bash
deeptrade vwap-reversion universe show --limit 100
```

当前 `run --code` 不会自动依据 `vwr_etf_universe` 拦截代码，ETF 池缓存主要用于盘前检查和后续过滤扩展。

### 5.6 ETF 日线与盘前特征缓存

同步单只 ETF 日线及辅助数据：

```bash
deeptrade vwap-reversion daily sync --code 159518.SZ --start 20240101 --end 20260630
```

该命令会尝试缓存：

- `fund_daily`：开高低收、成交量、成交额等。
- `fund_adj`：复权因子。
- `fund_share`：份额。
- `fund_nav`：净值。
- `stk_limit`：涨跌停价。

构建盘前特征：

```bash
deeptrade vwap-reversion features build \
  --code 159518.SZ \
  --start 20260601 \
  --end 20260630 \
  --min-amount-ma20 200000000
```

当前特征包括：

- `ret_1d`、`ret_5d`
- `rv_20d`
- `atr_pct_20d`
- `amount_ma20`
- `amount_pctile_252`
- `gap_pct`
- `liquidity_ok`
- `volatility_regime`：`high`、`normal`、`low`、`unknown`
- `trend_regime`：`up`、`range`、`down`、`unknown`

当前实时策略不会自动消费这些日线特征。它们是盘前筛选、观察和后续策略过滤接入的缓存数据。

## 6. 常见执行场景

### 6.1 首次运行前检查配置

```bash
deeptrade vwap-reversion settings show
deeptrade vwap-reversion universe sync --t0-whitelist 159518.SZ
deeptrade vwap-reversion universe show
```

确认目标 ETF 能通过 Tushare 返回实时行情后再运行。

### 6.2 开盘前启动并自动待机

```bash
deeptrade vwap-reversion run --code 159518.SZ
```

若当前是交易日 09:30 前，插件会进入 standby。到 09:30 后自动进入采集和交易循环。若希望非交易日或收盘后启动也能守候到下一交易日：

```bash
deeptrade vwap-reversion settings set standby_across_days true
deeptrade vwap-reversion run --code 159518.SZ
```

### 6.3 盘中启动

```bash
deeptrade vwap-reversion run --code 159518.SZ --poll-interval 15
```

盘中启动会立即采样。首条 bar 会包含从开盘到当前的累计成交量，后续按快照差分推进。若处于午休，会等待到 13:00 继续。

### 6.4 使用无面板输出

```bash
deeptrade vwap-reversion run --code 159518.SZ --no-dashboard
```

适合日志环境、CI、远程终端或 Rich Live 面板显示异常时使用。即使 dashboard 渲染失败，策略也会降级为行式输出，不会因此中断 run。

### 6.5 启用底仓 T+0 模式

```bash
deeptrade vwap-reversion settings set position_mode base_position_t
deeptrade vwap-reversion settings set base_shares 1000
deeptrade vwap-reversion settings set order_qty 100
deeptrade vwap-reversion run --code 159518.SZ
```

该模式假设已有 `base_shares` 底仓。高于 VWAP 带时可先卖出 `order_qty`，回归后买回；低于 VWAP 带时也可加仓，回归后卖出加仓部分。Paper 账户只模拟策略腿，不会查询真实券商持仓。

### 6.6 临时停手

```bash
deeptrade vwap-reversion kill-switch on
```

开启后，新的入场信号会落库为被 `kill_switch` 抑制；已有腿仍允许按止盈、止损、EOD 等规则平仓。

恢复：

```bash
deeptrade vwap-reversion kill-switch off
```

### 6.7 收盘复盘

收盘后，paper run 会自动写入当日汇总并生成两份报告。也可以重新生成并在终端查看：

```bash
deeptrade vwap-reversion report --kind both
deeptrade vwap-reversion history --code 159518.SZ --mode paper
```

执行报告重点看：

- 采样次数、有效 bar 数、无新成交采样次数。
- 拉取失败、累计量回退、快照质量跳过、stale 保护。
- VWAP/sigma 收敛快照。
- 执行信号和被风控抑制的信号分布。

交易汇总重点看：

- 成交笔数、胜率、profit factor。
- 毛盈亏、净盈亏、费用、滑点、换手。
- 日内最大回撤、平均持仓时间。
- buy-and-hold 基准与是否触发日亏熔断。

### 6.8 用已采集数据做参数对比

先跑默认参数：

```bash
deeptrade vwap-reversion backtest --code 159518.SZ --start 20260601 --end 20260630
```

再用一次性覆盖对比：

```bash
deeptrade vwap-reversion backtest \
  --code 159518.SZ \
  --start 20260601 \
  --end 20260630 \
  --k-entry 2.5 \
  --k-exit 0.2 \
  --k-stop 4.0
```

这不会改变持久化 settings。适合在不污染默认配置的前提下比较参数。

## 7. 产物与数据表

插件使用 `vwr_` 表前缀。主要表包括：

| 表 | 内容 |
| --- | --- |
| `vwr_runs` | paper/backtest run 历史、参数快照、状态、结果。 |
| `vwr_events` | 运行事件流，执行报告的数据源。 |
| `vwr_snapshots` | 原始实时快照。 |
| `vwr_bars` | 快照差分后的区间 bar 及 VWAP/sigma/z 快照。 |
| `vwr_signals` | 策略信号，包括被风控抑制的信号。 |
| `vwr_trades` | Paper 或 backtest 模拟成交。 |
| `vwr_daily_summary` | paper run 的日终交易汇总。 |
| `vwr_trade_cal` | 本插件使用的交易日历缓存。 |
| `vwr_config` | 用户持久化配置。 |
| `vwr_etf_universe` | ETF 基础池、T+0 白名单、融资融券与启用标记。 |
| `vwr_etf_daily` | ETF 日线、复权、份额、净值、涨跌停缓存。 |
| `vwr_daily_features` | 盘前流动性、波动率、趋势 regime 特征。 |

## 8. 当前限制与使用注意

- 当前 `run` 是单标的实时模拟交易，不支持一次运行多个 ETF。
- 当前不会向真实券商下单，成交来自插件内 Paper 撮合器。
- Paper 撮合使用最新价加减滑点立即成交，不模拟盘口深度、排队、撤单和部分成交。
- 回测只能回放已采集的 `vwr_bars`，不能用日线还原日内路径。
- `universe`、`daily`、`features` 当前是辅助缓存命令，实时策略不会自动消费这些过滤结果。
- `band_mode=time_std` 当前可配置并通过校验，但实时 VWAP 引擎实际仍按成交量加权 sigma 工作。
- 同日 aborted run 重启会重建 VWAP 引擎，但不会继承上一 run 的 Paper 虚拟持仓。
- `max_trades_per_day` 按成交 fill 计数，开仓和平仓各算一笔；设置过低会导致新开仓被提前抑制，以保证已有腿仍有平仓额度。
