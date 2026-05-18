-- accumulation-probe-washout strategy: full plugin schema (Plan A pure isolation).
--
-- Tables:
--   apw_watchlist          — current candidates being watched (PK=ts_code)
--   apw_signal_history     — every daily screen hit (PK=trade_date,ts_code)
--   apw_stage_results      — LLM analyze output rows (PK=run_id,ts_code)
--   apw_runs               — per-plugin run history (replaces shared strategy_runs)
--   apw_events             — per-plugin run event stream (replaces shared strategy_events)
--   apw_realized_returns   — T+N post-hoc returns (PK=signal_date,ts_code)
--   apw_config             — user-tunable settings

CREATE TABLE IF NOT EXISTS apw_watchlist (
    ts_code               VARCHAR PRIMARY KEY,
    name                  VARCHAR,
    first_seen_date       VARCHAR NOT NULL,
    last_seen_date        VARCHAR NOT NULL,
    phase                 VARCHAR NOT NULL,
    probe_date            VARCHAR,
    accumulation_score    DOUBLE,
    probe_quality_score   DOUBLE,
    washout_score         DOUBLE,
    launch_setup_score    DOUBLE,
    latest_launch_score   DOUBLE,
    latest_prediction     VARCHAR,
    latest_confidence     VARCHAR,
    raw_candidate_json    VARCHAR NOT NULL,
    updated_at            TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS apw_signal_history (
    trade_date            VARCHAR NOT NULL,
    ts_code               VARCHAR NOT NULL,
    name                  VARCHAR,
    phase                 VARCHAR NOT NULL,
    probe_date            VARCHAR,
    accumulation_score    DOUBLE,
    probe_quality_score   DOUBLE,
    washout_score         DOUBLE,
    launch_setup_score    DOUBLE,
    raw_candidate_json    VARCHAR NOT NULL,
    created_at            TIMESTAMP NOT NULL,
    PRIMARY KEY (trade_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_apw_signal_history_date
    ON apw_signal_history(trade_date);
CREATE INDEX IF NOT EXISTS idx_apw_signal_history_code_date
    ON apw_signal_history(ts_code, trade_date);

CREATE TABLE IF NOT EXISTS apw_stage_results (
    run_id                      UUID NOT NULL,
    trade_date                  VARCHAR NOT NULL,
    ts_code                     VARCHAR NOT NULL,
    candidate_id                VARCHAR NOT NULL,
    rank                        INTEGER NOT NULL,
    launch_score                DOUBLE NOT NULL,
    confidence                  VARCHAR NOT NULL,
    prediction                  VARCHAR NOT NULL,
    main_pattern                VARCHAR NOT NULL,
    phase                       VARCHAR NOT NULL,
    dimension_scores_json       VARCHAR NOT NULL,
    key_evidence_json           VARCHAR NOT NULL,
    rationale                   VARCHAR NOT NULL,
    next_session_watch_json     VARCHAR NOT NULL,
    invalidation_triggers_json  VARCHAR NOT NULL,
    risk_flags_json             VARCHAR NOT NULL,
    missing_data_json           VARCHAR NOT NULL,
    raw_response_json           VARCHAR NOT NULL,
    created_at                  TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_apw_stage_results_trade_date
    ON apw_stage_results(trade_date);
CREATE INDEX IF NOT EXISTS idx_apw_stage_results_run_id
    ON apw_stage_results(run_id);
CREATE INDEX IF NOT EXISTS idx_apw_stage_results_code_date
    ON apw_stage_results(ts_code, trade_date);

CREATE TABLE IF NOT EXISTS apw_runs (
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

CREATE TABLE IF NOT EXISTS apw_events (
    run_id       UUID NOT NULL,
    seq          BIGINT NOT NULL,
    event_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level        VARCHAR NOT NULL,
    event_type   VARCHAR NOT NULL,
    message      VARCHAR NOT NULL,
    payload_json VARCHAR,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS apw_realized_returns (
    signal_date           VARCHAR NOT NULL,
    ts_code               VARCHAR NOT NULL,
    probe_date            VARCHAR,
    prediction            VARCHAR,
    launch_score          DOUBLE,
    phase                 VARCHAR,
    close_t               DOUBLE,
    close_t1              DOUBLE,
    close_t3              DOUBLE,
    close_t5              DOUBLE,
    close_t10             DOUBLE,
    ret_t1_pct            DOUBLE,
    ret_t3_pct            DOUBLE,
    ret_t5_pct            DOUBLE,
    ret_t10_pct           DOUBLE,
    max_high_t5_pct       DOUBLE,
    max_high_t10_pct      DOUBLE,
    max_drawdown_t5_pct   DOUBLE,
    max_drawdown_t10_pct  DOUBLE,
    label_launch_t5       INTEGER,
    label_launch_t10      INTEGER,
    data_status           VARCHAR NOT NULL,
    computed_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_apw_realized_returns_signal_date
    ON apw_realized_returns(signal_date);
CREATE INDEX IF NOT EXISTS idx_apw_realized_returns_ts_code
    ON apw_realized_returns(ts_code);
CREATE INDEX IF NOT EXISTS idx_apw_realized_returns_label_t5
    ON apw_realized_returns(label_launch_t5);
CREATE INDEX IF NOT EXISTS idx_apw_realized_returns_label_t10
    ON apw_realized_returns(label_launch_t10);

CREATE TABLE IF NOT EXISTS apw_config (
    key         VARCHAR PRIMARY KEY,
    value_json  VARCHAR NOT NULL,
    updated_at  TIMESTAMP NOT NULL
);
