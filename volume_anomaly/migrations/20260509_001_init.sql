-- volume-anomaly strategy: full plugin schema (Plan A pure isolation).

CREATE TABLE IF NOT EXISTS va_watchlist (
    ts_code             VARCHAR PRIMARY KEY,
    name                VARCHAR,
    industry            VARCHAR,
    tracked_since       VARCHAR NOT NULL,
    last_screened       VARCHAR NOT NULL,
    last_pct_chg        DOUBLE,
    last_close          DOUBLE,
    last_vol            DOUBLE,
    last_amount         DOUBLE,
    last_body_ratio     DOUBLE,
    last_turnover_rate  DOUBLE,
    last_vol_ratio_5d   DOUBLE,
    last_max_vol_60d    DOUBLE
);

CREATE TABLE IF NOT EXISTS va_anomaly_history (
    trade_date          VARCHAR NOT NULL,
    ts_code             VARCHAR NOT NULL,
    name                VARCHAR,
    industry            VARCHAR,
    pct_chg             DOUBLE,
    close               DOUBLE,
    open                DOUBLE,
    high                DOUBLE,
    low                 DOUBLE,
    vol                 DOUBLE,
    amount              DOUBLE,
    body_ratio          DOUBLE,
    turnover_rate       DOUBLE,
    vol_ratio_5d        DOUBLE,
    max_vol_60d         DOUBLE,
    raw_metrics_json    VARCHAR,
    PRIMARY KEY (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS va_stage_results (
    run_id              UUID NOT NULL,
    stage               VARCHAR NOT NULL,
    batch_no            INTEGER,
    trade_date          VARCHAR NOT NULL,
    ts_code             VARCHAR NOT NULL,
    name                VARCHAR,
    rank                INTEGER,
    launch_score        DOUBLE,
    confidence          VARCHAR,
    prediction          VARCHAR,
    pattern             VARCHAR,
    rationale           VARCHAR,
    tracked_days        INTEGER,
    evidence_json       VARCHAR,
    risk_flags_json     VARCHAR,
    raw_response_json   VARCHAR,
    dim_washout         DOUBLE,
    dim_pattern         DOUBLE,
    dim_capital         DOUBLE,
    dim_sector          DOUBLE,
    dim_historical      DOUBLE,
    dim_risk            DOUBLE,
    PRIMARY KEY (run_id, stage, ts_code)
);

CREATE TABLE IF NOT EXISTS va_runs (
    run_id        UUID PRIMARY KEY,
    mode          VARCHAR NOT NULL,
    trade_date    VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    is_intraday   BOOLEAN NOT NULL DEFAULT FALSE,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP,
    params_json   VARCHAR,
    summary_json  VARCHAR,
    error         VARCHAR
);

CREATE TABLE IF NOT EXISTS va_events (
    run_id       UUID NOT NULL,
    seq          BIGINT NOT NULL,
    event_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level        VARCHAR NOT NULL,
    event_type   VARCHAR NOT NULL,
    message      VARCHAR NOT NULL,
    payload_json VARCHAR,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS va_realized_returns (
    anomaly_date    VARCHAR NOT NULL,
    ts_code         VARCHAR NOT NULL,
    t_close         DOUBLE,
    t1_close        DOUBLE,
    t3_close        DOUBLE,
    t5_close        DOUBLE,
    t10_close       DOUBLE,
    ret_t1          DOUBLE,
    ret_t3          DOUBLE,
    ret_t5          DOUBLE,
    ret_t10         DOUBLE,
    max_close_5d    DOUBLE,
    max_close_10d   DOUBLE,
    max_ret_5d      DOUBLE,
    max_ret_10d     DOUBLE,
    max_dd_5d       DOUBLE,
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_status     VARCHAR NOT NULL,
    PRIMARY KEY (anomaly_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_va_realized_returns_date
    ON va_realized_returns(anomaly_date, data_status);
