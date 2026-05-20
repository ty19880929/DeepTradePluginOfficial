-- accumulation-probe-washout v0.5.0: LightGBM 主升浪启动概率评分模型注册表。
--
-- 设计：每行 = 一个落盘 booster 文件的元数据快照。任意时刻最多一行
-- is_active=TRUE，由 train/activate CLI 在事务内切换。与 VA 同名表相比，
-- APW 的 label_source 取 'label_launch_t5' / 'label_launch_t10' / 'custom_t5'
-- 之一；label_threshold_pct 仅在 custom_t5 时使用（其他场景 NULL），因为
-- label_launch_t5 / t10 列本身已编码"收益 + 回撤"约束 (config.label_t5_*
-- / label_t10_*).

CREATE TABLE IF NOT EXISTS apw_lgb_models (
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
    label_source        VARCHAR NOT NULL,
    label_threshold_pct DOUBLE,
    framework_version   VARCHAR,
    plugin_version      VARCHAR NOT NULL,
    git_commit          VARCHAR,
    file_path           VARCHAR NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_apw_lgb_models_active
    ON apw_lgb_models(is_active, created_at DESC);
