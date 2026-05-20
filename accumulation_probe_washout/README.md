# accumulation-probe-washout (吸筹试盘洗盘主升浪策略)

A 股沪深主板上识别 **吸筹建仓 → 天量试盘 → 洗盘震仓 → 主升浪启动** 行为链路的官方 DeepTrade 插件。本地规则负责识别四阶段结构化证据，LLM 负责判断链路是否成立、是否接近启动临界点，并输出严格 JSON。

## 安装

```bash
deeptrade plugin install accumulation-probe-washout
```

依赖：`tushare>=1.4`, `pandas>=2.2,<3`, `pyarrow>=15`。框架在 install 阶段会自动安装。

## CLI

```bash
deeptrade accumulation-probe-washout screen [--date YYYYMMDD] [--max-candidates N] [--no-dashboard]
deeptrade accumulation-probe-washout screen --backfill-history --start YYYYMMDD --end YYYYMMDD [--overwrite]
deeptrade accumulation-probe-washout analyze [--date YYYYMMDD] [--llm <provider>] [--prediction launch_ready] [--no-lgb]
deeptrade accumulation-probe-washout run [--date YYYYMMDD] [--llm <provider>] [--no-lgb]
deeptrade accumulation-probe-washout evaluate --from-date YYYYMMDD --to-date YYYYMMDD \
    [--horizons 1,3,5,10] [--include-early-phases]
deeptrade accumulation-probe-washout stats --from YYYYMMDD --to YYYYMMDD --by <axis>
deeptrade accumulation-probe-washout prune [--dry-run]
deeptrade accumulation-probe-washout history [--limit 20]
deeptrade accumulation-probe-washout report --run-id <uuid>
deeptrade accumulation-probe-washout settings show
deeptrade accumulation-probe-washout settings set <key> <value>
deeptrade accumulation-probe-washout settings reset [--key <key>]

# LightGBM lifecycle (v0.5.0+)
deeptrade accumulation-probe-washout lgb train --start YYYYMMDD --end YYYYMMDD \
    [--label-source label_launch_t5|label_launch_t10|custom_t5] \
    [--label-threshold 8.0 --label-drawdown-threshold 8.0] \
    [--folds 5] [--no-activate] [--fresh] [--keep-checkpoint]
deeptrade accumulation-probe-washout lgb list
deeptrade accumulation-probe-washout lgb info [--model-id <id>]
deeptrade accumulation-probe-washout lgb activate <model-id>
deeptrade accumulation-probe-washout lgb evaluate --start YYYYMMDD --end YYYYMMDD \
    [--model-id <id>] [--k 10] [--drift --baseline <id>]
deeptrade accumulation-probe-washout lgb prune --keep N
deeptrade accumulation-probe-washout lgb purge --datasets|--models|--predictions|--checkpoints|--all [--yes]
```

通用 flag：`--date YYYYMMDD`（指定 T 日）、`--no-dashboard`（强制 legacy 流式输出）、`--llm <provider>`（覆盖框架默认 LLM provider）、`--no-lgb`（一次性跳过 LGB 评分）。

仪表盘自动 fallback：`--no-dashboard` / 非 TTY / `CI` 环境变量 / `DEEPTRADE_NO_DASHBOARD` / `TERM=dumb` 任一命中即降级到 line-per-event 输出。

## 状态机

```
no_setup → accumulating → probe_seen → washing_after_probe → launch_ready
```

只有 `washing_after_probe` 和 `launch_ready` 进入 `apw_watchlist` 与 LLM batch；`accumulating` / `probe_seen` 仍写 `apw_signal_history` 供 `evaluate` 与未来 lgb 训练（默认 `evaluate` 只统计 ≥`washing_after_probe`，`--include-early-phases` 关掉过滤）。

## 主要参数（apw_config 默认值）

| 类别 | key | 默认 | 说明 |
|-|-|-|-|
| 股票池 | `listed_days_min` | 120 | 新股过滤天数 |
| 股票池 | `min_amount_yi` | 1.0 | 当日最低成交额（亿元）|
| 吸筹 | `accumulation_lookback_trade_days` | 60 | 吸筹窗口 |
| 吸筹 | `accumulation_score_min` | 55 | 进入 `accumulating` 的分数下限 |
| 试盘 | `probe_volume_ratio_5d_min` | 2.5 | 试盘日量比下限（vs 5d 均量）|
| 试盘 | `probe_quality_score_min` | 60 | 进入 `probe_seen` 的分数下限 |
| 洗盘 | `washout_min_trade_days` / `washout_max_trade_days` | 3 / 25 | 洗盘日数窗口 |
| 洗盘 | `washout_score_min` | 55 | 进入 `washing_after_probe` 的分数下限 |
| 启动 | `launch_setup_score_min` | 55 | 进入 `launch_ready` 的分数下限 |
| 启动 | `launch_current_volume_ratio_5d_min` | 1.2 | 启动日量能下限 |
| LLM | `llm_batch_size` | 20 | 单批最大候选数 |
| LLM | `llm_max_repair_retries` | 2 | 校验失败 repair 次数 |
| 标签 | `label_t5_high_return_pct` / `label_t5_max_drawdown_pct` | 8 / 8 | T+5 启动标签阈值 |
| 标签 | `label_t10_high_return_pct` / `label_t10_max_drawdown_pct` | 12 / 10 | T+10 启动标签阈值 |

完整字段见 `accumulation_probe_washout/config.py`。通过 `settings set <key> <json_value>` 持久化覆盖。

## 数据表

每张表都加了 `purge_on_uninstall: true`，卸载即清理。

- `apw_watchlist` — 当前待观察标的（一只一行，PK=ts_code）
- `apw_signal_history` — 每日筛选命中明细（PK=trade_date, ts_code）
- `apw_stage_results` — LLM 分析结果（PK=run_id, ts_code）
- `apw_runs` / `apw_events` — run 历史 + 事件流
- `apw_realized_returns` — T+N 事后实际收益（PK=signal_date, ts_code）
- `apw_config` — 可调参数持久化覆盖

## 风险声明

- "主力吸筹"无法被直接观测；本插件用量价 + 资金流代理特征近似估计；
- 天量试盘与高位放量出货在形态上可能相似，最终判断高度依赖洗盘质量；
- 弱市中洗盘结构容易演变为破位；
- 所有 LLM 输出仅为辅助判断，**不构成任何投资建议**；
- Tushare 可选接口（`moneyflow` / `index_daily`）可能缺失或延迟，缺失字段会进入 `missing_data` 并降级运行；
- 表结构和 schema 一旦发布，后续升级需要 migration。

## 后续路线

- **辩论模式**：`analyze --debate --debate-llms <p1,p2,p3>`，复用 LUB 的 worker 编排范式（规划中）。
- **`adj_factor` 复权量比**：`volume_adjust_enabled` 配置位已在 v0.4.0 引入，实际的 `fetch_daily` 接口扩展计划在后续 PR 落地。
- **涨停历史特征**：`prior_limit_up_count_60d` / `days_since_last_limit_up` 已在 `FEATURE_NAMES` 中占位，等待接入 Tushare `limit_list_d`。

完整设计见飞书知识库《吸筹试盘洗盘主升浪插件 详细功能设计方案》。
