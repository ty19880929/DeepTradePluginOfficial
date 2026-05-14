# 成交量异动 Dashboard 兼容性 QA 矩阵

> 适用版本：`volume-anomaly v0.8.0`
> 更新时间：2026-05-14
> 设计 Plan 文档保留在仓外的 `docs/DeepTradePlugin/volume_anomaly/CLI_UI_Redesign_Plan.md`（按 LUB 同类设计文档惯例不入库）。

本文档记录方案 §8 提到的 15 组兼容性测试 / 自检结果。**自动化部分**由 `tests/test_ui_*.py` 覆盖（PR-1～PR-3 总计 48 个测试），**人工部分**在 v0.8.0 release tag 前于实机完成；本表既是验收清单也是日后回归 QA 的参考。

## 自动化覆盖

| 项 | 测试用例 | 测试文件 | 状态 |
|----|----------|----------|------|
| `--no-dashboard` 自救路径 | `TestChooseRenderer.test_no_dashboard_true_returns_legacy` | `tests/test_ui_protocol.py` | ✅ |
| non-TTY 自动 legacy | `TestChooseRenderer.test_non_tty_returns_legacy` | `tests/test_ui_protocol.py` | ✅ |
| `CI=true` 自动 legacy | `TestChooseRenderer.test_ci_env_returns_legacy` | `tests/test_ui_protocol.py` | ✅ |
| `DEEPTRADE_NO_DASHBOARD=1` | `TestChooseRenderer.test_deeptrade_no_dashboard_env_returns_legacy` | `tests/test_ui_protocol.py` | ✅ |
| `TERM=dumb` | `TestChooseRenderer.test_term_dumb_returns_legacy` | `tests/test_ui_protocol.py` | ✅ |
| TTY 默认拿 dashboard | `TestChooseRenderer.test_tty_no_fallbacks_returns_dashboard` | `tests/test_ui_protocol.py` | ✅ |
| `NO_COLOR=1` → dashboard + 无色 | `TestNoColor.test_no_color_flag_propagates` | `tests/test_ui_dashboard.py` | ✅ |
| Legacy v0.7.x 字节格式 | `TestLegacyStreamRenderer.*`（4 用例） | `tests/test_ui_protocol.py` | ✅ |
| Step 2 前缀文案锁定 | `TestLegacyStreamRenderer.test_step2_prefix_pipeline_alignment` | `tests/test_ui_protocol.py` | ✅ |
| renderer raise → 自动降级 legacy（U-8）| `TestDispatchToRenderer.test_raise_swaps_to_legacy_and_redispatches` | `tests/test_ui_protocol.py` | ✅ |
| close() 也 raise 时不崩 | `TestDispatchToRenderer.test_renderer_close_failure_is_swallowed` | `tests/test_ui_protocol.py` | ✅ |
| analyze 多批全成功（U-1）| `TestAnalyzeMultiBatchSuccess` | `tests/test_ui_dashboard.py` | ✅ |
| analyze 第二批 VALIDATION_FAILED → PARTIAL（U-2）| `TestAnalyzeValidationFailed` | `tests/test_ui_dashboard.py` | ✅ |
| analyze 空 watchlist（U-3）| `TestAnalyzeEmptyWatchlist` | `tests/test_ui_dashboard.py` | ✅ |
| KeyboardInterrupt → CANCELLED banner（U-4）| `TestCancelledOutcome` | `tests/test_ui_dashboard.py` | ✅ |
| screen 漏斗 5 字段（U-5）| `TestScreenFunnel.test_funnel_populated_from_payload` | `tests/test_ui_dashboard.py` | ✅ |
| TUSHARE_FALLBACK ⚠ 徽标 + 计数（U-9）| `TestTushareFallbackBadge` | `tests/test_ui_dashboard.py` | ✅ |
| 终端 < 80 列紧凑模式（U-10）| `TestCompactWidth.test_no_panel_borders_at_70_cols` | `tests/test_ui_dashboard.py` | ✅ |
| 紧凑模式漏斗单行 | `TestCompactWidth.test_funnel_compact_at_70_cols` | `tests/test_ui_dashboard.py` | ✅ |
| prune/evaluate 始终 legacy 不 emit 配置 LOG（U-12）| `TestPruneEvaluateForcedLegacy.*`（4 用例） | `tests/test_ui_dashboard.py` | ✅ |
| 漏斗 全/紧凑 渲染 | `TestRenderFunnelFull` + `TestRenderFunnelCompact`（7 用例） | `tests/test_ui_funnel.py` | ✅ |
| 快照 token 锁定 analyze + screen | `TestSnapshotU1Analyze` + `TestSnapshotU5Screen`（8 用例） | `tests/test_ui_snapshots.py` | ✅ |

`pytest` 当前总计 253 用例（v0.7.x 基线 222 + UI 新增 31），全部通过：

```text
253 passed in 11.91s
```

## 人工 QA（实机回退矩阵）

| # | 环境 | OS | 终端 | 命令 | 预期 | 状态 |
|---|------|-----|------|------|------|------|
| 1 | Windows 11 | Windows | Windows Terminal | `deeptrade volume-anomaly analyze` | dashboard 完整，进度条流畅 | ☐ |
| 2 | Windows 11 | Windows | conhost.exe (legacy console) | `deeptrade volume-anomaly screen` | dashboard truecolor 降级到 8 色，布局保持 | ☐ |
| 3 | Windows 11 | Windows | VS Code 集成终端 | `deeptrade volume-anomaly screen` | dashboard + 漏斗完整 | ☐ |
| 4 | Windows 11 | Windows | PowerShell `\| more` 重定向 | `deeptrade volume-anomaly analyze \| more` | legacy（无 ANSI） | ☐ |
| 5 | Windows 11 | Windows | `> out.txt` 重定向 | `deeptrade volume-anomaly analyze > out.txt` | legacy（out.txt 无 ANSI 字节） | ☐ |
| 6 | macOS | macOS | iTerm2 | `deeptrade volume-anomaly analyze` | dashboard 完整 | ☐ |
| 7 | Linux | Linux | tmux 内 | `deeptrade volume-anomaly screen` | dashboard 完整 / spinner 流畅 | ☐ |
| 8 | Linux | Linux | SSH 普通 xterm | `deeptrade volume-anomaly analyze` | dashboard 完整 | ☐ |
| 9 | CI | Linux | GitHub Actions runner | `CI=true pytest` 或 `deeptrade ... analyze` | legacy（`CI` 自动触发） | ☐ |
| 10 | 任意 | 任意 | `pytest` 捕获模式 | `pytest tests/` | legacy（非 TTY 自动触发） | ☐ |
| 11 | 任意 | 任意 | `--no-dashboard` flag | `deeptrade volume-anomaly analyze --no-dashboard` | legacy | ☐ |
| 12 | 任意 | 任意 | `NO_COLOR=1` | `NO_COLOR=1 deeptrade volume-anomaly analyze` | dashboard 启用但无色 | ☐ |
| 13 | 任意 | 任意 | `DEEPTRADE_NO_DASHBOARD=1` | `DEEPTRADE_NO_DASHBOARD=1 deeptrade volume-anomaly analyze` | legacy | ☐ |
| 14 | 任意 | 任意 | `TERM=dumb` | `TERM=dumb deeptrade volume-anomaly screen` | legacy | ☐ |
| 15 | 任意 | 任意 | prune / evaluate 强制 legacy | `deeptrade volume-anomaly prune --days 30` | legacy（即使 TTY） | ☐ |

> 状态标记：`✅` 通过 / `❌` 失败 / `☐` 未跑。在合并 release tag 前请把所有空 cell 填好。

## 已知小项 / 风险

* **`--no-dashboard` 与 v0.7.x 字节流一致** —— **加上 PR-1 的 `Step 2:` 前缀变化是预期的差异**（参见 plan §13 #3）。运行配置 LOG 在 legacy 路径下不会 emit（plan §6.3 由 runner 通过 `isinstance(renderer, LegacyStreamRenderer)` 抑制），保留 v0.7.x 字节流。
* **dashboard 在 80 列以下** —— 自动切换紧凑模式（无边框、漏斗合并为单行 `主板→…→量能` 形式）。该路径已由 `TestCompactWidth` 覆盖。
* **`pipeline.py` message 文本变更**（PR-1）—— 用户脚本若全文 grep `走势分析（主升浪启动预测）` 仍能匹配，因为新前缀 `Step 2: ` 是叠加而非替换；若 grep `^走势分析` 会失效，已在 release notes 提示。
* **Settings LOG 写入 `va_events`** —— 仅在 dashboard renderer 下持久化；legacy 路径完全跳过该 row，避免 `history` / `report` 对 legacy 用户行为的回归。
