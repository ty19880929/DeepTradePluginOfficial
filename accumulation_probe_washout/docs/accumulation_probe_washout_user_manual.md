# accumulation-probe-washout 用户手册

`accumulation-probe-washout` 是面向 A 股沪深主板的“吸筹建仓 → 天量试盘 → 洗盘震仓 → 主升浪启动”识别策略插件。插件先用本地量化规则筛出结构化候选，再用 LightGBM 概率评分和 LLM 结构化判断辅助排序。

> 风险提示：插件输出是量化筛选和辅助判断结果，不构成投资建议。“主力建仓”不可被直接观测，本策略使用量价、换手、资金流、涨停历史、相对强度等代理特征进行近似识别。

## 1. 安装与依赖

```bash
deeptrade plugin install accumulation-probe-washout
```

插件声明的运行依赖：

| 依赖 | 作用 |
| --- | --- |
| `tushare>=1.4` | 拉取 A 股行情、资金流、交易日历、涨停历史等数据 |
| `pandas>=2.2,<3` | 数据清洗与特征计算 |
| `pyarrow>=15` | LightGBM 数据集和中间结果存储 |
| `lightgbm>=4.3,<5` | 主升浪概率模型训练与推理 |
| `scikit-learn>=1.4,<2` | 训练评估、交叉验证等模型工具 |

Tushare API 权限：

| 类型 | API | 用途 |
| --- | --- | --- |
| 必需 | `stock_basic` | 股票基础信息、上市日期、市场板块 |
| 必需 | `trade_cal` | 交易日历、回测和评估日期对齐 |
| 必需 | `daily` | 日线 OHLCV、成交额、涨跌幅 |
| 必需 | `daily_basic` | 换手率、流通市值 |
| 必需 | `stock_st` | ST 股票过滤 |
| 必需 | `moneyflow` | 建仓、试盘、洗盘、启动阶段资金流代理 |
| 可选 | `suspend_d` | 停牌过滤 |
| 可选 | `index_daily` | 相对沪深300等指数的强弱对比 |
| 可选 | `adj_factor` | 复权量能修正，降低除权/拆股对量比的干扰 |
| 可选 | `limit_list_d` | 最近 60 个交易日涨停历史特征 |
| 可选 | `limit_cpt_list` / `top_list` | 预留扩展数据 |

可选 API 失败时，插件会把对应字段写入 `missing_data`，并尽量降级运行；必需 API 失败通常会导致当次任务失败。

## 2. 策略流程

状态机：

```text
no_setup -> accumulating -> probe_seen -> washing_after_probe -> launch_ready
```

各状态含义：

| 状态 | 含义 | 是否进入观察池 |
| --- | --- | --- |
| `no_setup` | 不满足建仓结构，忽略 | 否 |
| `accumulating` | 低位建仓迹象成立，但尚未出现合格试盘 | 否，仅写入历史 |
| `probe_seen` | 出现天量试盘，但洗盘结构未完成 | 否，仅写入历史 |
| `washing_after_probe` | 试盘后缩量洗盘结构成立 | 是 |
| `launch_ready` | 洗盘后重新放量、均线和相对强度接近启动临界 | 是 |

执行链路：

1. 过滤股票池：沪深主板、非 ST、非停牌、上市天数、成交额、流通市值。
2. 计算建仓分：低位程度、横盘/缓慢抬升、资金净流入、上涨日放量、异常暴露惩罚。
3. 识别试盘日：最近窗口内寻找放量、放额、换手、振幅、K 线质量都合格的试盘日。
4. 计算洗盘分：试盘后回撤、缩量、支撑位、MA20/MA60、资金留存、时间窗口。
5. 计算启动分：当前放量、均线多头、接近/突破试盘高点、资金回流、相对指数强度。
6. 补充衍生特征：VCP 波动收敛、120/250 日阻力位、alpha、MA 距离、涨停历史、复权量能。
7. 写入 `apw_signal_history`；`washing_after_probe` 与 `launch_ready` 同步写入 `apw_watchlist`。
8. `analyze` 阶段读取观察池，先做 LGB 评分，再交给 LLM 输出结构化判断。

## 3. 常用命令

### 3.1 当日规则筛选，不调用 LLM

```bash
deeptrade accumulation-probe-washout screen
```

指定交易日：

```bash
deeptrade accumulation-probe-washout screen --date 20260630
```

限制后续 LLM 候选数量：

```bash
deeptrade accumulation-probe-washout screen --date 20260630 --max-candidates 50
```

强制同步数据：

```bash
deeptrade accumulation-probe-washout screen --date 20260630 --force-sync
```

适用场景：

| 场景 | 推荐命令 |
| --- | --- |
| 盘后只想看规则候选 | `screen --date YYYYMMDD` |
| 数据缓存可能过期 | `screen --date YYYYMMDD --force-sync` |
| 只构建观察池，暂不消耗 LLM | `screen` |

### 3.2 对观察池做 LLM 分析

```bash
deeptrade accumulation-probe-washout analyze --date 20260630
```

指定 LLM provider：

```bash
deeptrade accumulation-probe-washout analyze --date 20260630 --llm deepseek
```

只分析 `launch_ready`：

```bash
deeptrade accumulation-probe-washout analyze --date 20260630 --prediction launch_ready
```

本次跳过 LGB 评分：

```bash
deeptrade accumulation-probe-washout analyze --date 20260630 --no-lgb
```

说明：

- `analyze` 读取 `apw_watchlist` 中当日 `washing_after_probe` 和 `launch_ready` 候选。
- 默认会尝试使用已激活的 LightGBM 模型评分；未训练或不可用时降级为仅规则 + LLM。
- `--prediction` 实际按候选 `phase` 过滤，常用值是 `launch_ready` 或 `washing_after_probe`。

### 3.3 一键执行筛选 + 分析

```bash
deeptrade accumulation-probe-washout run --date 20260630
```

带 LLM provider 和强制同步：

```bash
deeptrade accumulation-probe-washout run --date 20260630 --force-sync --llm deepseek
```

适用场景：

| 场景 | 推荐命令 |
| --- | --- |
| 每日盘后完整扫描 | `run --date YYYYMMDD` |
| 第一次跑某日数据 | `run --date YYYYMMDD --force-sync` |
| 没有训练 LGB 或临时停用模型 | `run --date YYYYMMDD --no-lgb` |

### 3.4 回填历史规则命中

```bash
deeptrade accumulation-probe-washout screen --backfill-history --start 20260101 --end 20260630
```

覆盖已有历史：

```bash
deeptrade accumulation-probe-washout screen --backfill-history --start 20260101 --end 20260630 --overwrite
```

说明：

- 回填模式只写 `apw_signal_history`，不会改写 `apw_watchlist`。
- 默认会跳过已有 `apw_signal_history` 的交易日，适合中断后续跑。
- `--overwrite` 会删除对应日期旧结果后重新计算。

### 3.5 事后收益评估

```bash
deeptrade accumulation-probe-washout evaluate --from-date 20260101 --to-date 20260630
```

指定收益周期：

```bash
deeptrade accumulation-probe-washout evaluate --from-date 20260101 --to-date 20260630 --horizons 1,3,5,10
```

纳入早期状态：

```bash
deeptrade accumulation-probe-washout evaluate --from-date 20260101 --to-date 20260630 --include-early-phases
```

重新计算已完成结果：

```bash
deeptrade accumulation-probe-washout evaluate --from-date 20260101 --to-date 20260630 --force-recompute
```

说明：

- 默认只评估 `washing_after_probe` 和 `launch_ready`。
- `--include-early-phases` 会纳入 `accumulating` 和 `probe_seen`，适合研究早期信号质量。
- 评估会写入 `apw_realized_returns`，包含 T+1/T+3/T+5/T+10 收益、最大涨幅、最大回撤和标签。

### 3.6 统计分析

```bash
deeptrade accumulation-probe-washout stats --from 20260101 --to 20260630 --by phase
```

可用 `--by`：

| 维度 | 含义 |
| --- | --- |
| `phase` | 按本地规则状态统计 |
| `prediction` | 按 LLM 判断统计 |
| `main_pattern` | 按 LLM 主模式统计 |
| `launch_score_bin` | 按 LLM 启动分分箱 |
| `accumulation_score_bin` | 按建仓分分箱 |
| `probe_quality_score_bin` | 按试盘质量分分箱 |
| `washout_score_bin` | 按洗盘分分箱 |
| `launch_setup_score_bin` | 按启动准备分分箱 |
| `dimension_scores` | 按 LLM 六维评分相关性统计 |
| `lgb_score_bin` | 按 LGB 概率分分箱 |

### 3.7 清理观察池

预览将删除哪些观察池标的：

```bash
deeptrade accumulation-probe-washout prune --dry-run --date 20260630
```

实际清理：

```bash
deeptrade accumulation-probe-washout prune --date 20260630
```

删除规则：

| 规则 | 配置项 |
| --- | --- |
| `launch_ready` 连续空闲达到指定交易日数 | `prune_idle_days_launch_ready` |
| `washing_after_probe` 距试盘日超过最大洗盘窗口仍未启动 | `washout_max_trade_days` |
| 当前收盘价跌破试盘日低点 | `prune_drop_on_probe_low_break` |
| 当前收盘价跌破 MA60 | `prune_drop_on_ma60_break` |

### 3.8 查看历史和报告

```bash
deeptrade accumulation-probe-washout history --limit 20
```

```bash
deeptrade accumulation-probe-washout report --run-id <uuid>
```

`history` 查看近期任务；`report` 重新渲染某次 `analyze` 或 `run` 的 LLM 结果。

## 4. 配置管理

查看当前配置：

```bash
deeptrade accumulation-probe-washout settings show
```

设置配置：

```bash
deeptrade accumulation-probe-washout settings set <key> <json_value>
```

重置单个配置：

```bash
deeptrade accumulation-probe-washout settings reset --key <key>
```

清空所有覆盖值，恢复默认：

```bash
deeptrade accumulation-probe-washout settings reset
```

配置值按 JSON 解析。字符串建议加双引号，并在 shell 中正确转义：

```bash
deeptrade accumulation-probe-washout settings set baseline_index_code "\"000905.SH\""
deeptrade accumulation-probe-washout settings set lgb_enabled false
deeptrade accumulation-probe-washout settings set max_llm_candidates 50
```

## 5. 全量配置项说明

### 5.1 股票池与流动性

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `listed_days_min` | `120` | int | 剔除上市时间不足的次新股，降低无历史结构样本的噪声。 | `settings set listed_days_min 180` |
| `min_amount_yi` | `1.0` | float | T 日最低成交额，单位亿元。低于该值的股票不进入筛选。 | `settings set min_amount_yi 2.0` |
| `min_circ_mv_yi` | `20.0` | float | 最低流通市值，单位亿元。 | `settings set min_circ_mv_yi 30.0` |
| `max_circ_mv_yi` | `1500.0` | float | 最高流通市值，单位亿元；设为较大值可放宽大盘股过滤。 | `settings set max_circ_mv_yi 3000.0` |

调参建议：

- 想减少小票流动性风险：提高 `min_amount_yi` 和 `min_circ_mv_yi`。
- 想聚焦中小盘弹性：降低 `max_circ_mv_yi`，例如 800。
- 不建议把 `listed_days_min` 降得太低，次新股容易出现无建仓窗口的假信号。

### 5.2 回看窗口

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `base_lookback_trade_days` | `120` | int | 基础行情窗口，用于低位程度、长周期结构和支撑判断。 | `settings set base_lookback_trade_days 160` |
| `probe_lookback_trade_days` | `40` | int | 最近多少个交易日内寻找试盘日。 | `settings set probe_lookback_trade_days 50` |
| `accumulation_lookback_trade_days` | `60` | int | 建仓窗口，约 2-3 个月。 | `settings set accumulation_lookback_trade_days 70` |
| `accumulation_moneyflow_days` | `20` | int | 建仓分中统计资金净流入的最近交易日数。 | `settings set accumulation_moneyflow_days 30` |
| `washout_min_trade_days` | `3` | int | 试盘后最短洗盘天数，太短通常不算充分震仓。 | `settings set washout_min_trade_days 5` |
| `washout_max_trade_days` | `25` | int | 试盘后最长洗盘天数，超过后仍未启动会降低有效性。 | `settings set washout_max_trade_days 30` |

调参建议：

- 你的原始目标是“过去 2-3 个月”，默认 `accumulation_lookback_trade_days=60` 正好匹配。
- 若市场轮动较快，可把 `probe_lookback_trade_days` 从 40 降到 30。
- 若偏中线潜伏，可把 `base_lookback_trade_days` 和 `washout_max_trade_days` 适度放宽。

### 5.3 建仓阶段

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `accumulation_score_min` | `55.0` | float | 进入 `accumulating` 的最低建仓分。 | `settings set accumulation_score_min 60.0` |

建仓分主要看：

- 当前价格在 120 日区间中的位置是否偏低。
- 60 日内价格是否横盘或缓慢抬升。
- 最近资金流是否净流入。
- 上涨日成交量是否强于下跌日。
- 建仓窗口内是否已经出现过度暴露的巨量尖峰。

提高该阈值会减少候选数量，但可能漏掉早期建仓票；降低该阈值会扩大候选池，但噪声会增加。

### 5.4 试盘阶段

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `probe_volume_ratio_5d_min` | `2.5` | float | 试盘日成交量相对前 5 日均量的最低倍数。 | `settings set probe_volume_ratio_5d_min 3.0` |
| `probe_volume_ratio_20d_min` | `2.0` | float | 试盘日成交量相对前 20 日均量的最低倍数。 | `settings set probe_volume_ratio_20d_min 2.5` |
| `probe_volume_rank_pct_60d_min` | `90.0` | float | 试盘日成交量在最近 60 日内的分位要求。 | `settings set probe_volume_rank_pct_60d_min 95.0` |
| `probe_turnover_rate_min` | `2.0` | float | 试盘日最低换手率。 | `settings set probe_turnover_rate_min 3.0` |
| `probe_amplitude_pct_min` | `5.0` | float | 试盘日最低振幅百分比。 | `settings set probe_amplitude_pct_min 6.0` |
| `probe_quality_score_min` | `60.0` | float | 进入 `probe_seen` 的最低试盘质量分。 | `settings set probe_quality_score_min 65.0` |

当前版本已支持 `volume_adjust_enabled`，试盘量比会优先使用由 `adj_factor` 修正后的 `vol_adj`，避免除权导致的量比误判。

### 5.5 洗盘阶段

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `max_post_probe_drawdown_pct` | `15.0` | float | 试盘后允许的最大回撤参考值。 | `settings set max_post_probe_drawdown_pct 12.0` |
| `post_probe_volume_shrink_ratio_max` | `0.8` | float | 试盘后平均量能相对试盘前 20 日均量的最大比例；越低越要求缩量。 | `settings set post_probe_volume_shrink_ratio_max 0.7` |
| `washout_score_min` | `55.0` | float | 进入 `washing_after_probe` 的最低洗盘分。 | `settings set washout_score_min 60.0` |

洗盘分会惩罚：

- 回撤过深。
- 试盘低点被跌破。
- MA20 或 MA60 被跌破。
- 试盘后不缩量。
- 洗盘时间过短或过长。

### 5.6 启动阶段

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `launch_setup_score_min` | `55.0` | float | 进入 `launch_ready` 的最低启动准备分。 | `settings set launch_setup_score_min 60.0` |
| `launch_current_volume_ratio_5d_min` | `1.2` | float | T 日量能相对前 5 日均量的最低倍数。 | `settings set launch_current_volume_ratio_5d_min 1.5` |
| `launch_moneyflow_days` | `3` | int | 启动阶段统计最近 N 日资金净流入。 | `settings set launch_moneyflow_days 5` |
| `baseline_index_code` | `"000300.SH"` | string | 计算 20 日相对强度使用的基准指数。 | `settings set baseline_index_code "\"000905.SH\""` |

启动分由五类信息构成：

- 当前量能是否重新放大。
- 是否站上 MA5/MA10/MA20。
- 是否接近或突破试盘高点。
- 最近资金流是否回流。
- 20 日收益是否强于基准指数。

常用基准：

| 指数代码 | 说明 |
| --- | --- |
| `000300.SH` | 沪深300，默认，适合偏主板蓝筹/大中盘对比 |
| `000905.SH` | 中证500，适合中盘成长风格对比 |
| `000852.SH` | 中证1000，适合小盘风格对比 |
| `""` | 禁用指数相对强度，相关字段降级 |

### 5.7 LLM 批处理

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `max_llm_candidates` | `80` | int | `screen` 进入后续 LLM 候选的默认上限。 | `settings set max_llm_candidates 50` |
| `llm_batch_size` | `10` | int | 每个 LLM batch 的候选数量。 | `settings set llm_batch_size 8` |
| `llm_max_repair_retries` | `2` | int | LLM JSON 校验失败后的修复重试次数。 | `settings set llm_max_repair_retries 3` |

调参建议：

- LLM 上下文或输出经常超限：降低 `llm_batch_size`。
- 希望节省 LLM 成本：降低 `max_llm_candidates` 或只 `analyze --prediction launch_ready`。
- LLM 输出结构不稳定：提高 `llm_max_repair_retries`，但会增加耗时和成本。

### 5.8 评估标签

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `evaluate_default_horizons` | `"1,3,5,10"` | string | 默认评估周期配置；CLI `--horizons` 可覆盖。 | `settings set evaluate_default_horizons "\"1,3,5,10\""` |
| `label_t5_high_return_pct` | `8.0` | float | T+5 标签要求的最大高点涨幅。 | `settings set label_t5_high_return_pct 10.0` |
| `label_t5_max_drawdown_pct` | `8.0` | float | T+5 标签允许的最大回撤。 | `settings set label_t5_max_drawdown_pct 6.0` |
| `label_t10_high_return_pct` | `12.0` | float | T+10 标签要求的最大高点涨幅。 | `settings set label_t10_high_return_pct 15.0` |
| `label_t10_max_drawdown_pct` | `10.0` | float | T+10 标签允许的最大回撤。 | `settings set label_t10_max_drawdown_pct 8.0` |

标签定义：

- `label_launch_t5 = 1`：T+1 到 T+5 内最大高点涨幅达到阈值，且最大回撤不超过阈值。
- `label_launch_t10 = 1`：T+1 到 T+10 内最大高点涨幅达到阈值，且最大回撤不超过阈值。

### 5.9 观察池清理

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `prune_idle_days_launch_ready` | `5` | int | `launch_ready` 连续多少个交易日未刷新后清理。 | `settings set prune_idle_days_launch_ready 3` |
| `prune_drop_on_probe_low_break` | `true` | bool | 跌破试盘日低点时清理。 | `settings set prune_drop_on_probe_low_break false` |
| `prune_drop_on_ma60_break` | `true` | bool | 跌破 MA60 时清理。 | `settings set prune_drop_on_ma60_break false` |
| `prune_dry_run_default` | `false` | bool | 预留默认 dry-run 行为配置。当前 CLI 以 `--dry-run` 为准。 | `settings set prune_dry_run_default true` |

### 5.10 衍生特征

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `volume_adjust_enabled` | `true` | bool | 是否拉取 `adj_factor` 并用复权量能计算量比、缩量、量能事件。 | `settings set volume_adjust_enabled false` |
| `alpha_baseline_index_code` | `"000300.SH"` | string | alpha 特征的基准指数。 | `settings set alpha_baseline_index_code "\"000905.SH\""` |

当前已实现的衍生特征：

| 特征 | 来源 | 用途 |
| --- | --- | --- |
| `vol_adj` | `adj_factor` + `daily.vol` | 复权量比、缩量、启动放量判断 |
| `prior_limit_up_count_60d` | `limit_list_d` | 最近 60 个交易日涨停次数 |
| `days_since_last_limit_up` | `limit_list_d` | 距离最近一次涨停的交易日数 |
| `atr_10d_pct` / `bbw_20d` | 日线 | VCP 波动压缩 |
| `dist_to_120d_high_pct` / `dist_to_250d_high_pct` | 日线 | 长周期阻力位距离 |
| `alpha_5d_pct` / `alpha_20d_pct` / `alpha_60d_pct` | 个股日线 + 指数日线 | 相对强度 |
| `close_to_ma5_pct` / `close_to_ma20_pct` / `close_to_ma60_pct` | 日线 | 均线距离 |

### 5.11 LightGBM

| key | 默认值 | 类型 | 作用 | 示例 |
| --- | ---: | --- | --- | --- |
| `lgb_enabled` | `true` | bool | `analyze` 是否默认启用 LGB 推理。 | `settings set lgb_enabled false` |
| `lgb_label_source` | `"label_launch_t5"` | string | 默认训练标签，可选 `label_launch_t5`、`label_launch_t10`、`custom_t5`。 | `settings set lgb_label_source "\"label_launch_t10\""` |
| `lgb_label_threshold_pct` | `8.0` | float | `custom_t5` 的涨幅阈值。 | `settings set lgb_label_threshold_pct 10.0` |
| `lgb_label_drawdown_threshold_pct` | `8.0` | float | `custom_t5` 的最大回撤阈值。 | `settings set lgb_label_drawdown_threshold_pct 6.0` |
| `lgb_train_folds` | `5` | int | GroupKFold 折数，按 signal_date 分组防止同日泄漏。 | `settings set lgb_train_folds 3` |
| `lgb_train_min_samples` | `500` | int | 训练所需最少样本数。 | `settings set lgb_train_min_samples 1000` |
| `lgb_train_lookback_days` | `365` | int | 训练默认回看天数，供生命周期逻辑使用。 | `settings set lgb_train_lookback_days 540` |
| `lgb_max_models_to_keep` | `5` | int | LGB 模型保留数量策略。 | `settings set lgb_max_models_to_keep 8` |
| `lgb_max_datasets_to_keep` | `3` | int | LGB 数据集保留数量策略。 | `settings set lgb_max_datasets_to_keep 5` |
| `lgb_min_score_floor` | `25.0` | float/null | LGB 分数低于该值时给候选添加 `low_lgb_score` 风险标记；设为 `null` 关闭。 | `settings set lgb_min_score_floor null` |
| `lgb_decile_in_prompt` | `true` | bool | 是否把 LGB 十分位传给 LLM prompt。 | `settings set lgb_decile_in_prompt false` |

## 6. LightGBM 生命周期

### 6.1 训练前准备

训练需要先有历史信号和事后收益：

```bash
deeptrade accumulation-probe-washout screen --backfill-history --start 20250101 --end 20260601
deeptrade accumulation-probe-washout evaluate --from-date 20250101 --to-date 20260601
```

### 6.2 训练模型

```bash
deeptrade accumulation-probe-washout lgb train --start 20250101 --end 20260601
```

使用 T+10 标签：

```bash
deeptrade accumulation-probe-washout lgb train --start 20250101 --end 20260601 --label-source label_launch_t10
```

使用自定义 T+5 标签：

```bash
deeptrade accumulation-probe-washout lgb train --start 20250101 --end 20260601 --label-source custom_t5 --label-threshold 10 --label-drawdown-threshold 6
```

训练但不激活：

```bash
deeptrade accumulation-probe-washout lgb train --start 20250101 --end 20260601 --no-activate
```

### 6.3 查看和切换模型

```bash
deeptrade accumulation-probe-washout lgb list
deeptrade accumulation-probe-washout lgb info
deeptrade accumulation-probe-washout lgb info --model-id <model-id>
deeptrade accumulation-probe-washout lgb activate <model-id>
```

### 6.4 评估模型

```bash
deeptrade accumulation-probe-washout lgb evaluate --start 20260602 --end 20260630 --k 10
```

漂移分析：

```bash
deeptrade accumulation-probe-washout lgb evaluate --start 20260602 --end 20260630 --drift --model-id <candidate-model-id> --baseline <baseline-model-id>
```

### 6.5 清理模型和产物

保留最近 N 个非 active 模型：

```bash
deeptrade accumulation-probe-washout lgb prune --keep 5
```

清理 LGB 产物：

```bash
deeptrade accumulation-probe-washout lgb purge --datasets --yes
deeptrade accumulation-probe-washout lgb purge --models --yes
deeptrade accumulation-probe-washout lgb purge --predictions --yes
deeptrade accumulation-probe-washout lgb purge --checkpoints --yes
deeptrade accumulation-probe-washout lgb purge --all --yes
```

`purge` 是破坏性操作；不带 `--yes` 只会提示，不会执行。

## 7. 场景化执行方案

### 7.1 每日盘后找次日观察标的

```bash
deeptrade accumulation-probe-washout run --date 20260630 --force-sync
deeptrade accumulation-probe-washout report --run-id <run-id>
```

关注输出：

- `prediction=launch_ready` 且 `confidence=high/medium`。
- `launch_score` 靠前。
- `risk_flags` 为空或风险可解释。
- `next_session_watch` 中的突破、量能、支撑条件是否次日兑现。

### 7.2 只想做低成本量化初筛

```bash
deeptrade accumulation-probe-washout screen --date 20260630 --max-candidates 100
```

然后使用数据库中的 `apw_signal_history` 和 `apw_watchlist` 做自定义查看。此模式不调用 LLM，成本最低。

### 7.3 只分析最接近启动的票

```bash
deeptrade accumulation-probe-washout screen --date 20260630
deeptrade accumulation-probe-washout analyze --date 20260630 --prediction launch_ready --max-candidates 20
```

适用于市场热点多、观察池过大时，先聚焦最高时效的 `launch_ready`。

### 7.4 建立历史训练集

```bash
deeptrade accumulation-probe-washout screen --backfill-history --start 20240101 --end 20260630
deeptrade accumulation-probe-washout evaluate --from-date 20240101 --to-date 20260630
deeptrade accumulation-probe-washout stats --from 20240101 --to 20260630 --by phase
deeptrade accumulation-probe-washout stats --from 20240101 --to 20260630 --by launch_setup_score_bin
```

确认样本量和命中率稳定后再训练 LGB：

```bash
deeptrade accumulation-probe-washout lgb train --start 20240101 --end 20260630
```

### 7.5 市场很弱时减少假启动

建议提高启动和洗盘要求：

```bash
deeptrade accumulation-probe-washout settings set launch_setup_score_min 65
deeptrade accumulation-probe-washout settings set launch_current_volume_ratio_5d_min 1.5
deeptrade accumulation-probe-washout settings set washout_score_min 60
```

也可以降低候选数量：

```bash
deeptrade accumulation-probe-washout settings set max_llm_candidates 40
```

### 7.6 强势行情中扩大候选池

```bash
deeptrade accumulation-probe-washout settings set accumulation_score_min 50
deeptrade accumulation-probe-washout settings set probe_quality_score_min 55
deeptrade accumulation-probe-washout settings set max_llm_candidates 120
```

注意：放宽阈值会显著增加噪声，应配合 `evaluate` 和 `stats` 检查实际表现。

### 7.7 复盘某次运行

```bash
deeptrade accumulation-probe-washout history --limit 30
deeptrade accumulation-probe-washout report --run-id <run-id>
```

如果需要看分组表现：

```bash
deeptrade accumulation-probe-washout stats --from 20260601 --to 20260630 --by prediction
deeptrade accumulation-probe-washout stats --from 20260601 --to 20260630 --by lgb_score_bin
```

## 8. 输出字段解读

LLM 候选结果主要字段：

| 字段 | 含义 |
| --- | --- |
| `rank` | LLM 在当前 batch 内给出的排序 |
| `launch_score` | LLM 综合启动分，0-100 |
| `confidence` | `high` / `medium` / `low` |
| `prediction` | `launch_ready`、`watch_breakout`、`still_washing`、`probe_failed`、`avoid` |
| `main_pattern` | 主模式，如 `probe_washout_breakout`、`low_base_accumulation` 等 |
| `phase` | LLM 复核后的阶段 |
| `dimension_scores` | 建仓、试盘、洗盘、启动时机、资金确认、风险六维评分 |
| `rationale` | 简短判断理由 |
| `key_evidence` | 引用输入字段的关键证据 |
| `next_session_watch` | 次日需要观察的触发条件 |
| `invalidation_triggers` | 失效条件 |
| `risk_flags` | 风险标签 |
| `missing_data` | 缺失或降级的数据源 |

本地候选关键字段：

| 字段 | 含义 |
| --- | --- |
| `accumulation_score` | 建仓分 |
| `probe_quality_score` | 试盘质量分 |
| `washout_score` | 洗盘分 |
| `launch_setup_score` | 启动准备分 |
| `probe_volume_ratio_5d` / `probe_volume_ratio_20d` | 试盘日量比 |
| `post_probe_volume_shrink_ratio` | 试盘后缩量比例 |
| `current_volume_ratio_5d` / `current_volume_ratio_20d` | 当前启动量比 |
| `relative_strength_20d` | 20 日相对基准指数强弱 |
| `prior_limit_up_count_60d` | 最近 60 个交易日涨停次数 |
| `days_since_last_limit_up` | 距离最近一次涨停的交易日数 |
| `lgb_score` / `lgb_decile` | LGB 主升概率分和十分位 |

## 9. 数据表说明

| 表 | 作用 |
| --- | --- |
| `apw_signal_history` | 每日规则筛选命中明细，主键为 `trade_date + ts_code` |
| `apw_watchlist` | 当前观察池，一只股票一行 |
| `apw_stage_results` | LLM 分析结果 |
| `apw_runs` | 每次任务的运行记录 |
| `apw_events` | 每次任务的事件流 |
| `apw_realized_returns` | T+N 事后收益与标签 |
| `apw_config` | 用户持久化配置覆盖值 |
| `apw_lgb_models` | LGB 模型注册表 |
| `apw_lgb_predictions` | analyze 阶段 LGB 推理审计记录 |

卸载插件时，上述表都声明为 `purge_on_uninstall: true`，会随插件卸载清理。

## 10. 参数组合模板

### 保守模板

```bash
deeptrade accumulation-probe-washout settings set min_amount_yi 2.0
deeptrade accumulation-probe-washout settings set accumulation_score_min 60
deeptrade accumulation-probe-washout settings set probe_quality_score_min 65
deeptrade accumulation-probe-washout settings set washout_score_min 60
deeptrade accumulation-probe-washout settings set launch_setup_score_min 65
deeptrade accumulation-probe-washout settings set launch_current_volume_ratio_5d_min 1.5
```

适合弱市、震荡市、只想要少量高质量候选的场景。

### 均衡模板

使用默认配置即可：

```bash
deeptrade accumulation-probe-washout settings reset
```

适合日常盘后扫描。

### 进攻模板

```bash
deeptrade accumulation-probe-washout settings set min_amount_yi 1.0
deeptrade accumulation-probe-washout settings set accumulation_score_min 50
deeptrade accumulation-probe-washout settings set probe_quality_score_min 55
deeptrade accumulation-probe-washout settings set washout_score_min 50
deeptrade accumulation-probe-washout settings set max_llm_candidates 120
```

适合强趋势行情中扩大候选池。需要更严格执行失效条件。

## 11. 常见问题

### 为什么 screen 有候选，但 analyze 没有候选？

只有 `washing_after_probe` 和 `launch_ready` 会进入 `apw_watchlist`。`accumulating` 和 `probe_seen` 只写入 `apw_signal_history`，用于后续评估和训练。

### 为什么 LGB 没有评分？

常见原因：

- 尚未训练并激活模型。
- 本地缺少 `lightgbm` 依赖。
- 模型文件被删除。
- 模型的特征 schema 与当前插件不一致。
- 本次命令使用了 `--no-lgb` 或配置了 `lgb_enabled=false`。

### 为什么出现 `missing_data`？

说明某些可选或局部数据源缺失，例如 `moneyflow`、`index_daily`、`adj_factor`、`limit_list_d`。插件会尽量降级运行，但对应评分可信度需要下调。

### 如何处理候选过多？

可以：

```bash
deeptrade accumulation-probe-washout settings set max_llm_candidates 40
deeptrade accumulation-probe-washout settings set accumulation_score_min 60
deeptrade accumulation-probe-washout settings set probe_quality_score_min 65
deeptrade accumulation-probe-washout analyze --prediction launch_ready
```

### 如何恢复所有默认参数？

```bash
deeptrade accumulation-probe-washout settings reset
```

## 12. 建议的日常操作顺序

盘后：

```bash
deeptrade accumulation-probe-washout run --date YYYYMMDD --force-sync
```

查看结果：

```bash
deeptrade accumulation-probe-washout history --limit 5
deeptrade accumulation-probe-washout report --run-id <run-id>
```

清理失效观察池：

```bash
deeptrade accumulation-probe-washout prune --dry-run --date YYYYMMDD
deeptrade accumulation-probe-washout prune --date YYYYMMDD
```

每周或每月复盘：

```bash
deeptrade accumulation-probe-washout evaluate --from-date YYYYMMDD --to-date YYYYMMDD
deeptrade accumulation-probe-washout stats --from YYYYMMDD --to YYYYMMDD --by phase
deeptrade accumulation-probe-washout stats --from YYYYMMDD --to YYYYMMDD --by lgb_score_bin
```

模型维护：

```bash
deeptrade accumulation-probe-washout lgb train --start YYYYMMDD --end YYYYMMDD
deeptrade accumulation-probe-washout lgb evaluate --start YYYYMMDD --end YYYYMMDD
deeptrade accumulation-probe-washout lgb prune --keep 5
```
