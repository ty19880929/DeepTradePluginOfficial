# DeepTradePluginOfficial

Official plugin monorepo for [DeepTrade](https://github.com/ty19880929/deeptrade) — a plugin-based A-share screening CLI driven by LLMs.

## Plugins

| Plugin ID | Type | Description | Subdir |
|-----------|------|-------------|--------|
| `limit-up-board` | strategy | 打板策略：双轮 LLM 漏斗（强势标的分析 → 连板预测） | [limit_up_board/](./limit_up_board) |
| `volume-anomaly` | strategy | 成交量异动策略：主板放量筛选 + LLM 主升浪启动预测 | [volume_anomaly/](./volume_anomaly) |
| `stdout-channel` | channel | 参考通知通道：把 payload 打印到标准输出 | [stdout/](./stdout) |

## Install

```bash
# Install the framework first
pipx install deeptrade-quant

# Install a plugin by short name (queries this repo's registry/index.json)
deeptrade plugin install limit-up-board
deeptrade plugin install volume-anomaly
deeptrade plugin install stdout-channel
```

## Repo layout

```
DeepTradePluginOfficial/
├── registry/
│   └── index.json          # Plugin registry consumed by `deeptrade plugin install <short-name>`
├── limit_up_board/
│   ├── deeptrade_plugin.yaml
│   ├── migrations/
│   ├── limit_up_board/     # Inner Python package
│   ├── tests/
│   └── pytest.ini
├── volume_anomaly/         # Same shape
└── stdout/                 # Same shape (channel plugin)
```

## Versioning

Each plugin has its own SemVer release line. Tags follow the pattern `<plugin-id>/v<X.Y.Z>` (e.g. `limit-up-board/v0.4.0`). The framework CLI resolves the latest matching tag via the GitHub releases API.

## Origin

Code migrated from the [deeptrade](https://github.com/ty19880929/deeptrade) main repo at commit `8c82e1f` (tag `archive/with-builtin-plugins-v0.1.0-preview`). See `docs/distribution-and-plugin-install-design.md` in the framework repo for the migration plan.

## Contributing a new plugin

1. Add a subdirectory `<plugin-id>/` with `deeptrade_plugin.yaml`, `migrations/`, and the inner Python package
2. Add an entry to `registry/index.json`
3. Open a PR; once merged, push a tag `<plugin-id>/v0.1.0` to trigger release
