-- volume-anomaly v0.7.0: plugin-local key/value config store.
--
-- 设计文档：docs/DeepTradePlugin/volume_anomaly/lightgbm_design.md §10
--
-- 与 limit-up-board 的 lub_config 表同构：key 全名带 'va.' 前缀（如
-- 'va.lgb_enabled' / 'va.lgb_min_score_floor'），value_json 用 JSON 编码。
-- 不走框架 ConfigService 的 app_config 表 —— ConfigService 只允许已知 key
-- (AppConfig 字段)，无法承载插件私有配置；va_config 是 VA 自己的命名空间，
-- 不污染框架配置。

CREATE TABLE IF NOT EXISTS va_config (
    key        VARCHAR PRIMARY KEY,
    value_json VARCHAR NOT NULL
);
