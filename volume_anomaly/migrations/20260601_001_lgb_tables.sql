-- volume-anomaly v0.7.0: LightGBM 主升浪启动概率评分相关表。
-- 设计文档：docs/DeepTradePlugin/volume_anomaly/lightgbm_design.md §3.5
--
-- 与既有 va_* 表保持同前缀；purge_on_uninstall 由 deeptrade_plugin.yaml.tables
-- 集中声明，本文件只负责 schema 创建。

-- 模型版本登记：一行 = 一个落盘模型文件的元数据 + 训练指标。
-- 任意时刻最多一行 is_active=TRUE，由 train / activate CLI 在事务内切换。
-- 与 limit-up-board 的同名表相比，多了 label_threshold_pct / label_source
-- 两列，用于显式记录每个模型训练时所用的标签语义（VA 支持
-- max_ret_5d / ret_t3 / max_ret_10d 三种标签源，且阈值可调）。
CREATE TABLE IF NOT EXISTS va_lgb_models (
    model_id            VARCHAR PRIMARY KEY,
    schema_version      INTEGER NOT NULL,
    train_start_date    VARCHAR NOT NULL,
    train_end_date      VARCHAR NOT NULL,
    n_samples           INTEGER NOT NULL,
    n_positive          INTEGER NOT NULL,
    cv_auc_mean         DOUBLE,
    cv_auc_std          DOUBLE,
    cv_logloss_mean     DOUBLE,
    feature_count       INTEGER NOT NULL,
    feature_list_json   VARCHAR NOT NULL,
    hyperparams_json    VARCHAR NOT NULL,
    label_threshold_pct DOUBLE NOT NULL,
    label_source        VARCHAR NOT NULL,
    framework_version   VARCHAR,
    plugin_version      VARCHAR NOT NULL,
    git_commit          VARCHAR,
    file_path           VARCHAR NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_va_lgb_models_active
    ON va_lgb_models(is_active, created_at DESC);

-- 每次 analyze run 的 LGB 推理结果审计：一行 = 一只候选股 × 一次 run。
-- (a) 故障复盘时可回看每只股的特征向量摘要；
-- (b) 离线 backtest 工具可与 va_realized_returns join；
-- (c) lgb evaluate / lgb info --recent-N 复用同一张表。
CREATE TABLE IF NOT EXISTS va_lgb_predictions (
    run_id                UUID NOT NULL,
    trade_date            VARCHAR NOT NULL,
    ts_code               VARCHAR NOT NULL,
    model_id              VARCHAR NOT NULL,
    lgb_score             DOUBLE NOT NULL,
    lgb_decile            INTEGER,
    feature_hash          VARCHAR NOT NULL,
    feature_missing_json  VARCHAR,
    created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, ts_code)
);

CREATE INDEX IF NOT EXISTS ix_va_lgb_predictions_trade_date
    ON va_lgb_predictions(trade_date, model_id);
