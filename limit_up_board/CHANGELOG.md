# limit-up-board — Changelog

All notable changes to this plugin land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## v0.13.3 — 2026-05-24 — 报告上传链路下沉到框架

打板插件不再自带 `uploader.py`；上传 URL / 超时 / token / 全局开关全部走框架
`report.upload.*` 配置族（需要 `deeptrade-quant>=0.11.0`）。首次升级时插件
自动把旧 `lub.summary_upload_*` 配置搬到框架 + `secret_store`，搬完即清，
无需用户介入。

### Required

- `deeptrade-quant >= 0.11.0`（提供 `PluginContext.make_report_uploader` /
  `report.upload.*` 配置族 / `report_uploads` 审计表）。

### Removed

- `limit_up_board/uploader.py` 整文件；
- `LubConfig.summary_upload_enabled / summary_upload_url /
  summary_upload_timeout / summary_upload_token` 四个字段及对应校验。

### Changed

- `runner._maybe_upload_summary` 改为 `ctx.make_report_uploader().upload(...)`，
  事件 payload 字段名保持与 v0.12.3 兼容（`enabled / url / status / duration_ms /
  public_url / public_path / error_class`；同时新增 `public_index / public_date`）。
- 启动时跑一次 `migrate_legacy_upload_config`：把旧的 `lub.summary_upload_*` 行
  迁移到 `report.upload.*`（url / timeout 仅在框架仍是 `default` 时覆盖；token
  非空则一律入 `secret_store`；enabled=True 则一次性写到框架开关），完成后清掉
  旧行；幂等，重复调用为 no-op。

### Migration notes

- 用户升级后第一次跑任意 `deeptrade limit-up-board <cmd>`：旧的
  `summary_upload_enabled=True` 会被搬到 `report.upload.enabled=True`，后续请
  用 `deeptrade config set report.upload.enabled true/false` 调整。
- 想关掉上传：`deeptrade config set report.upload.enabled false`；想换
  endpoint：`deeptrade config set report.upload.url https://...`。
- `deeptrade config show` 会列出新的 `report.upload.*` 行；token 默认掩码。
