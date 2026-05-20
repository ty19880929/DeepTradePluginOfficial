-- checkmate trend-following strategy: full plugin schema (Iter-0 PR-0.2 init).
--
-- Tables (all under prefix `checkmate_`, all `purge_on_uninstall: true`):
--   checkmate_stock_status_history  — survivorship snapshots (ts_code × as_of_date)
--   checkmate_universe_daily        — daily eligible universe + reason_codes
--   checkmate_features_daily        — qfq-based features + score breakdown
--   checkmate_regime_daily          — market regime classification per trade_date
--   checkmate_signals               — entry / exit / defensive signals
--   checkmate_positions             — position state machine (pending/holding/defensive/closed)
--   checkmate_trades                — fills (backtest + future paper/live)
--   checkmate_backtest_runs         — backtest-run registry (config_hash, metrics)
--   checkmate_runs                  — per-plugin run history (replaces strategy_runs)
--   checkmate_events                — per-plugin run event stream (replaces strategy_events)


-- 1) Survivorship snapshots
CREATE TABLE IF NOT EXISTS checkmate_stock_status_history (
    ts_code        VARCHAR NOT NULL,
    as_of_date     VARCHAR NOT NULL,
    list_status    VARCHAR NOT NULL,
    is_st          BOOLEAN NOT NULL DEFAULT FALSE,
    name           VARCHAR,
    industry       VARCHAR,
    list_date      VARCHAR,
    delist_date    VARCHAR,
    raw_event_json VARCHAR,
    updated_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_checkmate_status_asof
    ON checkmate_stock_status_history(as_of_date);


-- 2) Daily eligible universe
CREATE TABLE IF NOT EXISTS checkmate_universe_daily (
    trade_date       VARCHAR NOT NULL,
    ts_code          VARCHAR NOT NULL,
    eligible         BOOLEAN NOT NULL,
    reason_codes     VARCHAR NOT NULL,  -- JSON array of reason strings
    liquidity_score  DOUBLE,
    amount_20d_avg   DOUBLE,
    turnover_20d_avg DOUBLE,
    list_status      VARCHAR,
    is_st            BOOLEAN,
    name             VARCHAR,
    industry         VARCHAR,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_checkmate_universe_eligible
    ON checkmate_universe_daily(trade_date, eligible);


-- 3) Daily features (qfq-based)
CREATE TABLE IF NOT EXISTS checkmate_features_daily (
    trade_date        VARCHAR NOT NULL,
    ts_code           VARCHAR NOT NULL,
    close_qfq         DOUBLE,
    ma20              DOUBLE,
    ma60              DOUBLE,
    ma120             DOUBLE,
    ma_slope60        DOUBLE,
    atr20             DOUBLE,
    atr_pct           DOUBLE,
    ret_60            DOUBLE,
    ret_120           DOUBLE,
    rs60_pctile       DOUBLE,
    rs120_pctile      DOUBLE,
    amount_20d_avg    DOUBLE,
    turnover_20d_avg  DOUBLE,
    limit_freq_60d    DOUBLE,
    drawdown_60d_high DOUBLE,
    quiet_score       DOUBLE,
    above_ma20_days   INTEGER,
    score             DOUBLE,
    score_breakdown   VARCHAR,  -- JSON object
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_checkmate_features_score
    ON checkmate_features_daily(trade_date, score);


-- 4) Daily market regime
CREATE TABLE IF NOT EXISTS checkmate_regime_daily (
    trade_date              VARCHAR PRIMARY KEY,
    regime                  VARCHAR NOT NULL,   -- strong / neutral / weak / risk
    exposure_cap            DOUBLE NOT NULL,    -- [0, 1]
    breadth_ma120           DOUBLE,
    breadth_limit_down_5d   DOUBLE,
    index_csi_above_ma120   BOOLEAN,
    index_hs300_above_ma120 BOOLEAN,
    payload_json            VARCHAR,
    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- 5) Daily signals
CREATE TABLE IF NOT EXISTS checkmate_signals (
    signal_date  VARCHAR NOT NULL,
    ts_code      VARCHAR NOT NULL,
    action       VARCHAR NOT NULL,   -- enter / hold / defensive / exit
    signal_type  VARCHAR,            -- breakout / pullback / continuation / stop_loss / ...
    score        DOUBLE,
    explain      VARCHAR,            -- JSON with include_reasons/exclude_reasons/...
    run_id       UUID,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, ts_code, action)
);
CREATE INDEX IF NOT EXISTS idx_checkmate_signals_run_id
    ON checkmate_signals(run_id);
CREATE INDEX IF NOT EXISTS idx_checkmate_signals_code
    ON checkmate_signals(ts_code, signal_date);


-- 6) Position state machine
CREATE TABLE IF NOT EXISTS checkmate_positions (
    ts_code           VARCHAR NOT NULL,
    entry_date        VARCHAR NOT NULL,
    entry_price_raw   DOUBLE,
    entry_price_qfq   DOUBLE,
    shares            BIGINT NOT NULL DEFAULT 0,
    stop_price        DOUBLE,
    state             VARCHAR NOT NULL,   -- pending / holding / defensive / closed
    risk_R            DOUBLE,             -- 1R risk per share at entry
    peak_pnl_R        DOUBLE,
    exit_date         VARCHAR,
    exit_price_raw    DOUBLE,
    exit_reason       VARCHAR,
    run_id            UUID,
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, entry_date)
);
CREATE INDEX IF NOT EXISTS idx_checkmate_positions_state
    ON checkmate_positions(state);
CREATE INDEX IF NOT EXISTS idx_checkmate_positions_run_id
    ON checkmate_positions(run_id);


-- 7) Trade ledger
CREATE TABLE IF NOT EXISTS checkmate_trades (
    trade_id        UUID PRIMARY KEY,
    run_id          UUID NOT NULL,
    ts_code         VARCHAR NOT NULL,
    side            VARCHAR NOT NULL,   -- buy / sell
    order_date      VARCHAR NOT NULL,
    fill_date       VARCHAR,
    fill_price_raw  DOUBLE,
    fill_price_qfq  DOUBLE,
    shares          BIGINT NOT NULL,
    cost_breakdown  VARCHAR,            -- JSON {commission, stamp_tax, transfer_fee, slippage, impact}
    exit_reason     VARCHAR,
    cancel_reason   VARCHAR,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_checkmate_trades_run_id
    ON checkmate_trades(run_id);
CREATE INDEX IF NOT EXISTS idx_checkmate_trades_fill_date
    ON checkmate_trades(fill_date);
CREATE INDEX IF NOT EXISTS idx_checkmate_trades_code
    ON checkmate_trades(ts_code, order_date);


-- 8) Backtest run registry
CREATE TABLE IF NOT EXISTS checkmate_backtest_runs (
    run_id        UUID PRIMARY KEY,
    config_hash   VARCHAR NOT NULL,
    code_version  VARCHAR NOT NULL,   -- plugin version + git sha short
    start_date    VARCHAR NOT NULL,
    end_date      VARCHAR NOT NULL,
    started_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at   TIMESTAMP,
    status        VARCHAR NOT NULL,
    metrics_json  VARCHAR,
    config_json   VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_checkmate_backtest_config_hash
    ON checkmate_backtest_runs(config_hash);


-- 9) Plugin-owned run history (replaces shared strategy_runs)
CREATE TABLE IF NOT EXISTS checkmate_runs (
    run_id        UUID PRIMARY KEY,
    mode          VARCHAR NOT NULL,   -- scan / signals / backtest / sync / explain / report
    trade_date    VARCHAR,
    status        VARCHAR NOT NULL,
    exit_code     INTEGER,
    started_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at   TIMESTAMP,
    params_json   VARCHAR,
    summary_json  VARCHAR,
    error         VARCHAR
);


-- 10) Plugin-owned event stream (replaces shared strategy_events)
CREATE TABLE IF NOT EXISTS checkmate_events (
    run_id       UUID NOT NULL,
    seq          BIGINT NOT NULL,
    event_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level        VARCHAR NOT NULL,
    event_type   VARCHAR NOT NULL,
    message      VARCHAR NOT NULL,
    payload_json VARCHAR,
    PRIMARY KEY (run_id, seq)
);
