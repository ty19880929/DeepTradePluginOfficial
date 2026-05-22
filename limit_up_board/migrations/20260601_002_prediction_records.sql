-- limit-up-board v0.11.0: 单 LLM 模式 T 日连板预测留痕。
-- 设计：飞书知识库《limit-up-board 胜率分析 — PR 实施计划》PR #1。
--
-- 主键 (trade_date, ts_code) — 后一次单 LLM run 覆盖同日同股旧记录。
-- 辩论模式不写入本表，避免多 provider 聚合口径未定义前污染胜率样本。

CREATE TABLE IF NOT EXISTS lub_prediction_records (
    trade_date          VARCHAR NOT NULL,
    next_trade_date     VARCHAR NOT NULL,
    ts_code             VARCHAR NOT NULL,
    name                VARCHAR NOT NULL,
    run_id              VARCHAR NOT NULL,
    prediction          VARCHAR NOT NULL,
    rank                INTEGER NOT NULL,
    continuation_score  DOUBLE,
    confidence          VARCHAR,
    t_close_price       DOUBLE,
    lgb_score           DOUBLE,
    lgb_decile          INTEGER,
    raw_prediction_json VARCHAR,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, ts_code)
);

CREATE INDEX IF NOT EXISTS ix_lub_prediction_records_date
    ON lub_prediction_records(trade_date);

CREATE INDEX IF NOT EXISTS ix_lub_prediction_records_run
    ON lub_prediction_records(run_id);

CREATE INDEX IF NOT EXISTS ix_lub_prediction_records_prediction
    ON lub_prediction_records(prediction, trade_date);
