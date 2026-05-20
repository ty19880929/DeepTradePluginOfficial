-- accumulation-probe-washout v0.6.0: 每次 analyze run 的 LGB 推理结果审计。
--
-- 一行 = 一只候选股 × 一次 run。三类下游消费者：
--   (a) 故障复盘 — 回看每只股的特征向量摘要 (feature_hash + missing list);
--   (b) 离线 backtest — JOIN apw_realized_returns 复盘分层效果;
--   (c) lgb evaluate / lgb info --recent-N — 复用同一张表。
--
-- PK (run_id, ts_code) 与 apw_stage_results 对齐，便于一对一 JOIN。

CREATE TABLE IF NOT EXISTS apw_lgb_predictions (
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

CREATE INDEX IF NOT EXISTS ix_apw_lgb_predictions_trade_date
    ON apw_lgb_predictions(trade_date, model_id);
