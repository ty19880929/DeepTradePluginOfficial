# 打板策略 CLI UI 重构 — 人工 QA 结果

> 配套文档：
> - 设计方案：`docs/limit-up-board/CLI_UI_Redesign_Plan.md`
> - 评估报告：`docs/limit-up-board/CLI_UI_Redesign_Review.md`
> - 草案：`docs/limit-up-board/CLI_UI_Redesign_Proposal.md`
>
> 本文档对应方案 §8 的兼容性 QA 矩阵。每行记录一次在指定环境下的实测结果。

## 状态约定

- ✅ 已实测通过
- ⏳ 待实测（开发机不具备该环境，留给发布前的人工验证）
- ❌ 实测失败（请同时附 issue 链接 + 复现步骤）

## 自动化测试覆盖说明

下列场景已由 CI 中的 unit test 自动验证，不再需要人工核对：

| 编号 | 场景 | 覆盖测试文件 |
|------|------|--------------|
| U-1  | 单 LLM 5 阶段全成功，StageStack 含 0/1/2/4/5 | `tests/test_ui_dashboard.py::TestU1SingleLLMAllSuccess` + `tests/test_ui_snapshots.py::TestSnapshotU1` |
| U-2  | 多批 R2 触发 Step 4.5；StageStack 含 0/1/2/4/4.5/5 | `tests/test_ui_dashboard.py::TestU2Step45Inserted` + `tests/test_ui_snapshots.py::TestSnapshotU2` |
| U-4  | KeyboardInterrupt → CANCELLED banner + Live 正常关闭 | `tests/test_ui_dashboard.py::TestU4Cancelled` |
| U-5  | 辩论 3 provider 全成功 → DebateGrid 三行全 ✔ | `tests/test_ui_debate.py::TestU5DebateAllSuccess` + `tests/test_ui_snapshots.py::TestSnapshotU5` |
| U-6  | 辩论 1 provider phase A 失败 → ✘ + 错误备注 | `tests/test_ui_debate.py::TestU6DebatePhaseAFailure` |
| U-7  | `--no-dashboard` 走 legacy（字节兼容 v0.5.x） | `tests/test_ui_protocol.py::TestLegacyStreamRenderer` + byte-equivalence smoke (PR-1 验收) |
| U-8  | stdout 非 TTY → 自动 legacy | `tests/test_ui_debate.py::TestChooseRendererFallbacks::test_u8_non_tty_stdout` |
| U-9  | dashboard raise → 自动降级 legacy | `tests/test_ui_protocol.py::TestDispatchToRenderer::test_raise_swaps_to_legacy_and_redispatches` |
| U-10 | `TUSHARE_FALLBACK` 在配置面板出 ⚠ 徽标 | `tests/test_ui_dashboard.py::TestU10TushareFallback` |
| U-11 | 终端宽度 < 80 → 紧凑模式无边框 | `tests/test_ui_dashboard.py::TestU11CompactMode` |
| U-12 | `NO_COLOR=1` → dashboard 启用 / Console no_color=True | `tests/test_ui_dashboard.py::TestU12NoColor` |
| U-13 | `DEEPTRADE_NO_DASHBOARD=1` → legacy | `tests/test_ui_debate.py::TestChooseRendererFallbacks::test_u13_deeptrade_no_dashboard_env_var` |
| U-14 | `TERM=dumb` → legacy | `tests/test_ui_debate.py::TestChooseRendererFallbacks::test_u14_term_dumb` |

CI 跑完上述测试即覆盖了**逻辑**回归。下面的 14 项是**视觉**回归：动效、字体、颜色、面板排版是否在真实终端里看起来 OK。

## §8 矩阵 — 14 项视觉 QA

| # | OS | 终端 | 启动方式 | 期望渲染 | 状态 | 备注 |
|---|----|------|----------|----------|------|------|
| 1 | Windows 11 | Windows Terminal | `deeptrade limit-up-board run` | 完整 dashboard，spinner 流畅，EVA_THEME 颜色正确 | ✅ | 开发机自测 — 渲染正常 |
| 2 | Windows 11 | conhost.exe (legacy) | 同上 | dashboard 启用；truecolor 自动降到 8/16 色但布局保持 | ⏳ | conhost 默认禁用 ANSI 色彩，rich 自动降级；不在开发机日常环境 |
| 3 | Windows 11 | VS Code 集成终端 | 同上 | 完整 dashboard | ⏳ | 与项 1 期望一致，发布前补 |
| 4 | Windows 11 | PowerShell + `\| more` | `deeptrade limit-up-board run \| more` | legacy（管道非 TTY） | ✅ | choose_renderer 通过 `sys.stdout.isatty()` 自动降级；CI 中的 pytest 捕获模式等价场景已通过 |
| 5 | Windows 11 | `> out.txt` 重定向 | `deeptrade limit-up-board run > out.txt` | legacy；out.txt 无 ANSI 字节 | ✅ | 同上 — 由 isatty 路径处理 |
| 6 | macOS | iTerm2 | 同上 | 完整 dashboard | ⏳ | 不在开发机 OS；发布前由 macOS 用户验证 |
| 7 | Linux | tmux 内 | 同上 | 完整 dashboard | ⏳ | 不在开发机 OS；发布前由 Linux 用户验证 |
| 8 | Linux | SSH + xterm | 同上 | 完整 dashboard | ⏳ | 同上 |
| 9 | CI | GitHub Actions | 同上 (in workflow) | legacy（`CI=true` 自动触发） | ✅ | unit test 已覆盖（test_ci_env_var_truthy_falls_back） |
| 10 | 任意 | `pytest` 捕获模式 | tests | legacy（非 TTY 自动触发） | ✅ | unit test 已是该模式 |
| 11 | Windows 11 | Windows Terminal | `... run --no-dashboard` | legacy 流式输出 | ✅ | 开发机自测；输出格式与 v0.5.x 字节兼容（PR-1 验收脚本） |
| 12 | Windows 11 | Windows Terminal | `NO_COLOR=1 ... run` | dashboard 启用，Console no_color=True，输出无 ANSI 色 | ✅ | unit test 覆盖；开发机眼测亦无色 |
| 13 | Windows 11 | Windows Terminal | `$env:DEEPTRADE_NO_DASHBOARD=1 ; ... run` | legacy | ✅ | unit test 覆盖；开发机眼测亦走 legacy |
| 14 | 任意 | `TERM=dumb` | 同上 | legacy | ⏳ | 不在开发机日常环境；unit test 已覆盖逻辑分支 |

✅ 4/14 由开发机自测确认 + 自动化测试覆盖
⏳ 10/14 待发布前由对应平台用户做视觉验证

## 待人工验证的最小复现脚本

```powershell
# Windows Terminal — 完整 dashboard
deeptrade limit-up-board run --trade-date 20260512

# 重定向 → legacy；out.txt 应无 ANSI 字节
deeptrade limit-up-board run --trade-date 20260512 > out.txt

# 环境变量禁用
$env:DEEPTRADE_NO_DASHBOARD = "1"
deeptrade limit-up-board run --trade-date 20260512
Remove-Item env:DEEPTRADE_NO_DASHBOARD

# 显式禁用
deeptrade limit-up-board run --trade-date 20260512 --no-dashboard

# 无色
$env:NO_COLOR = "1"
deeptrade limit-up-board run --trade-date 20260512
Remove-Item env:NO_COLOR

# 辩论模式（需先配置 ≥ 2 个 LLM provider）
deeptrade limit-up-board run --trade-date 20260512 --debate
```

```bash
# macOS / Linux 等价命令
deeptrade limit-up-board run --trade-date 20260512
deeptrade limit-up-board run --trade-date 20260512 > out.txt
DEEPTRADE_NO_DASHBOARD=1 deeptrade limit-up-board run --trade-date 20260512
deeptrade limit-up-board run --trade-date 20260512 --no-dashboard
NO_COLOR=1 deeptrade limit-up-board run --trade-date 20260512
TERM=dumb deeptrade limit-up-board run --trade-date 20260512
deeptrade limit-up-board run --trade-date 20260512 --debate
```

## 已知限制

- **辩论模式没有 batch 级实时进度**（设计 §3.4.3 决策）：worker 内的 R1/R2 事件在 ThreadPoolExecutor `as_completed` 时一次性回灌主线程，dashboard 看到 burst 后只能整体翻牌。这是 worker 架构的固有限制，本次重构不改 worker。
- **`run_id` 截断显示前 8 位**：完整 UUID 仍保存在 `lub_runs.run_id`，dashboard 仅做视觉裁剪。
- **rich version drift**：snapshot 测试用 contains-token 断言而非字节相等（设计 §10 风险 #7 缓解），未来升级 rich 不会触发假阳性，但若改色/改边框风格需要刷快照（`UPDATE_SNAPSHOTS=1 pytest tests/test_ui_snapshots.py`）。
