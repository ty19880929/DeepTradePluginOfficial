# Migration 命名规范（v0.13.0 起）

## 规则

新增迁移文件必须使用**创建当日**的日期前缀，**不**使用预期发布日期或未来日期：

```
migrations/<YYYYMMDD>_<NNN>_<short-name>.sql
```

* `YYYYMMDD` = 创建该 PR 的当日（本机时区）。
* `<NNN>` = 当日递增的三位序号（`001`、`002`、...），同日多次迁移时按落地顺序递增。
* `<short-name>` = 用 `snake_case` 简短描述本次结构变更。
* `deeptrade_plugin.yaml::migrations.version` 字段等于 `<YYYYMMDD>_<NNN>`（不含 `.sql`）。
* `deeptrade_plugin.yaml::migrations.checksum` 必须填实际 SHA256（`tools/check_registry.py` 会校验）。

为什么？

* 日期前缀负责 **全局顺序**，框架按 `version` 字典序应用迁移。
* 使用未来日期会让 `git log` 与迁移历史在视觉上错位（"为什么 PR 在 2026-05-23 合并，迁移文件叫
  20260601\_001？"），且新人 onboard 时不利于推断改动顺序。
* 多人同日发起多次 PR 时，`<NNN>` 递增解决并发命名碰撞；rebase 时该字段是
  唯一需要重新分配的部分。

## 历史例外（不要改名）

以下迁移在 v0.5.0 / v0.5.5 / v0.6 / v0.7 期间已经发布，**禁止重命名**——
改名会导致旧库的 `applied_migrations` 表对不上、首次升级即失败：

| version            | file                                                | 创建上下文 |
|--------------------|-----------------------------------------------------|------------|
| `20260601_001`     | `migrations/20260601_001_lgb_tables.sql`            | v0.5.0 历史日期 |
| `20260601_002`     | `migrations/20260601_002_prediction_records.sql`    | v0.7 历史日期 |
| `20260601_003`     | `migrations/20260601_003_winrate_reviews.sql`       | v0.11 历史日期 |

新迁移（v0.13.0 之后）已按本规范执行：

| version            | file                                                | 创建日期 |
|--------------------|-----------------------------------------------------|----------|
| `20260524_001`     | `migrations/20260524_001_lgb_calibration.sql`       | 2026-05-24 |

## PR Checklist

每次新增迁移的 PR 应包含：

- [ ] 文件名前缀是**今天**的日期？
- [ ] `deeptrade_plugin.yaml::migrations` 已新增条目且 `version` 与文件名一致？
- [ ] `checksum:` 已用 `python -c "import hashlib,sys; print('sha256:'+hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>` 重新计算？
- [ ] `python tools/check_registry.py` 绿？
- [ ] 与上一行 version 相比是字典序递增？

## 校验工具

`python tools/check_registry.py` 校验：

1. `version` 字段不重复、`file` 路径存在；
2. 迁移文件实际 SHA256 与 `checksum` 一致。

它**不**校验日期是否为"今天"——该约束依赖 PR review。
