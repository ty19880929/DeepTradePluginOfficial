# DeepTradePluginOfficial

[DeepTrade](https://github.com/ty19880929/DeepTrade) 框架的官方插件 monorepo。DeepTrade 是一款基于插件、由 LLM 驱动的 A 股选股 CLI；本仓库统一收纳官方维护的策略插件，并通过 `registry/index.json` 暴露给框架的 `deeptrade plugin install <短名>` 命令。

> 📖 **在线插件目录**：[deeptrade.tiey.ai/plugins](https://deeptrade.tiey.ai/plugins) — 自动同步本注册表

## 插件清单

| 插件 ID | 类型 | 当前版本 | 简介 | 子目录 |
|---------|------|---------|------|--------|
| `limit-up-board` | strategy | 0.5.1 | 打板策略：双轮 LLM 漏斗 + LightGBM 连板概率评分（量化锚点 ⊕ LLM 决策） | [limit_up_board/](./limit_up_board) |
| `volume-anomaly` | strategy | 0.7.0 | 成交量异动策略：主板放量筛选 + LLM 主升浪启动预测 + LightGBM 启动概率评分 | [volume_anomaly/](./volume_anomaly) |

各插件版本号以其 `deeptrade_plugin.yaml` 中的 `version` 字段为准。

## 安装

```bash
# 先安装框架
pipx install deeptrade-quant

# 通过短名安装插件（框架会自动查询本仓库的 registry/index.json）
deeptrade plugin install limit-up-board
deeptrade plugin install volume-anomaly
```

框架会按 `registry/index.json` 中的 `repo` 与 `tag_prefix`，调用 GitHub Releases API 解析最新匹配的 tag，再从对应 `subdir` 拉取插件源码与迁移脚本。

### 插件可选依赖

| 插件 | 依赖 | 用途 |
|------|------|------|
| `limit-up-board` ≥ 0.5.0 | `lightgbm>=4.3`、`scikit-learn>=1.4` | LightGBM 连板概率评分（训练 + 推理）。缺包时 `validate_static` 会在 install 阶段给出友好提示。 |
| `volume-anomaly` ≥ 0.7.0 | `lightgbm>=4.3`、`scikit-learn>=1.4`、`pyarrow>=14` | LightGBM 主升浪启动概率评分（训练 + 推理 + checkpoint snapshot）。框架在 install / upgrade 阶段自动 `uv pip install`。 |

手动安装：

```bash
pipx inject deeptrade-quant lightgbm scikit-learn
```

或者使用 venv（详见下文「插件本地测试指南」）。

## 插件本地测试指南

每个插件的测试都通过 `pytest` 运行（`pytest.ini::pythonpath = .` 要求**从插件子目录内执行**）。
本仓库不直接 import `deeptrade.*` —— 测试运行时框架必须可被 import，所以先装好 `deeptrade-quant`。

### 1. 安装框架与插件依赖

```powershell
# 安装框架（让 deeptrade.core.* 可被 import）
pipx install deeptrade-quant

# 装齐插件的运行 + 测试依赖（与各插件 deeptrade_plugin.yaml::dependencies 对齐）
pip install -r limit_up_board/requirements-dev.txt
pip install -r volume_anomaly/requirements-dev.txt
```

> Windows 用户若用 `pipx`，可改用 `pipx inject deeptrade-quant -r limit_up_board/requirements-dev.txt`，
> 让依赖注入到与 framework 同一个 venv，避免 import 路径混乱。

### 2. 跑测试

```powershell
# limit-up-board 全套（不含 LightGBM smoke）
cd limit_up_board ; pytest -m "not slow"

# volume-anomaly 同形
cd volume_anomaly ; pytest -m "not slow"

# 跑单文件 / 单用例
cd limit_up_board ; pytest tests/test_v04_settings.py
cd limit_up_board ; pytest tests/test_lgb_scorer.py::TestName::test_case
```

CI（`.github/workflows/plugin-tests.yml`）以同样的命令在每个 PR / push 上执行。

### 3. 常见报错

| 现象 | 根因 | 处置 |
|------|------|------|
| `ModuleNotFoundError: deeptrade` | 框架没在当前解释器里 | `pipx install deeptrade-quant` 或在 venv 内 `pip install deeptrade-quant` |
| `ModuleNotFoundError: pyarrow` / `lightgbm` / `sklearn` | 没装 requirements-dev.txt | 见上方步骤 1 |
| LightGBM smoke 测试超时 | 默认 `-m "not slow"` 已跳过 | 真要跑 LightGBM 训练 smoke：`pytest -m "slow"`（需真 Tushare 凭据） |

## 仓库结构

```
DeepTradePluginOfficial/
├── registry/
│   └── index.json              # 插件注册表，由 `deeptrade plugin install <短名>` 消费
├── tools/
│   ├── check_registry.py       # 校验 index.json 与各插件 yaml / 迁移 checksum 一致
│   └── check_release.py        # tag 推送时校验 yaml.version 与 tag 版本一致
├── .github/workflows/
│   ├── registry-check.yml      # PR / push 到 main 触发 registry 校验
│   └── plugin-release.yml      # 推送 `*/v*` tag 触发版本校验并自动创建 GitHub Release
├── limit_up_board/
│   ├── deeptrade_plugin.yaml   # 插件清单（permissions / migrations / tables）
│   ├── migrations/             # SQL 迁移脚本（带 sha256 checksum）
│   ├── limit_up_board/         # 内层 Python 包（plugin / pipeline / runner / data ...）
│   ├── tests/
│   └── pytest.ini
└── volume_anomaly/             # 结构同上
```

## 版本管理

每个插件维护独立的 SemVer 发布线，tag 形式为 `<plugin-id>/v<X.Y.Z>`，例如：

- `limit-up-board/v0.4.0`
- `volume-anomaly/v0.6.0`

发布流程：

1. 在对应插件目录修改代码并更新 `deeptrade_plugin.yaml` 中的 `version`
2. 合入 main 后，推送匹配的 tag（如 `git tag limit-up-board/v0.4.1 && git push origin limit-up-board/v0.4.1`）
3. `plugin-release.yml` 工作流会调用 `tools/check_release.py` 验证 tag 版本与 yaml 一致，校验通过后自动创建 GitHub Release

框架 CLI 在安装时会通过 GitHub Releases API 解析 `tag_prefix` 下最新版本，因此**未发布 Release 的 tag 不会被框架使用**。

## CI 校验

每次 PR 与 push 到 main 都会运行 `registry-check.yml`，由 `tools/check_registry.py` 执行以下校验：

- `registry/index.json` schema 合法（`schema_version=1`）
- 每个条目的必填字段齐全，`type` 仅允许 `strategy` / `channel`
- `tag_prefix` 必须以 `/` 结尾，`repo` 必须为 `owner/repo` 形式
- 每个 `subdir/deeptrade_plugin.yaml` 与注册表项的 `plugin_id` / `name` / `type` 一致
- 每条迁移文件的 `sha256` checksum 与 yaml 中声明的一致（与框架 `_verify_migration_checksum` 同款逻辑）

## 新增插件

1. 在仓库根目录新建 `<plugin-id>/` 子目录，包含：
   - `deeptrade_plugin.yaml`（plugin_id / name / version / type / api_version / entrypoint / permissions / migrations / tables）
   - `migrations/<version>_<name>.sql` 及对应 sha256 checksum
   - 内层 Python 包（与外层目录同名）
   - 推荐补齐 `tests/` 与 `pytest.ini`
2. 在 `registry/index.json` 中新增条目，确保 `subdir`、`tag_prefix`、`repo`、`min_framework_version` 与插件实际情况匹配
3. 本地运行 `python tools/check_registry.py` 通过后提 PR
4. 合入 main 后推送 `<plugin-id>/v0.1.0` tag 触发首次发布

## 起源

代码迁移自 [deeptrade](https://github.com/ty19880929/deeptrade) 主仓库 commit `8c82e1f`（tag `archive/with-builtin-plugins-v0.1.0-preview`）。迁移背景与设计可参考框架仓库下的 `docs/distribution-and-plugin-install-design.md`。
