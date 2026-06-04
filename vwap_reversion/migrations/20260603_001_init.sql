-- vwap-reversion strategy: full plugin schema (P0 scaffold).
-- DuckDB dialect. All ts columns are epoch seconds (UTC-based, unambiguous).
-- trade_date is the Asia/Shanghai calendar day (YYYYMMDD). See design §4 / §9.

-- One row per run. status ∈ {standby, running, done, aborted} / mode ∈ {paper, backtest}.
-- trade_date: paper = single YYYYMMDD / backtest = "YYYYMMDD-YYYYMMDD" window.
-- report_dir: 收盘双报告 / backtest 报告落盘目录（P3, daemon/cli 回写）.
-- result_json: backtest 多日聚合指标（P3）.
CREATE TABLE IF NOT EXISTS vwr_runs (
    run_id        VARCHAR PRIMARY KEY,
    mode          VARCHAR,
    code          VARCHAR,
    trade_date    VARCHAR,
    status        VARCHAR,
    params_json   VARCHAR,
    initial_cash  DOUBLE,
    final_cash    DOUBLE,
    report_dir    VARCHAR,
    result_json   VARCHAR,
    started_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at   TIMESTAMP
);

-- Plugin-owned event stream (replaces framework strategy_events). Source of the
-- 执行报告 + the live 策略执行记录 panel.
CREATE TABLE IF NOT EXISTS vwr_events (
    run_id        VARCHAR,
    seq           BIGINT,
    ts            BIGINT,
    event_type    VARCHAR,
    level         VARCHAR,
    message       VARCHAR,
    payload_json  VARCHAR,
    PRIMARY KEY (run_id, seq)
);

-- Raw cumulative snapshots polled from rt_etf_k. Not bound to a run so multiple
-- backtests can replay the same sampled corpus. VWAP = cum_amount / cum_vol.
-- Column ←→ rt_etf_k mapping: last←close / cum_vol←vol / cum_amount←amount /
-- num_trades←num. ts is the local poll epoch (sampling axis) and trade_time is
-- the exchange-side time string kept verbatim for audit only.
CREATE TABLE IF NOT EXISTS vwr_snapshots (
    code          VARCHAR,
    trade_date    VARCHAR,
    ts            BIGINT,
    last          DOUBLE,
    cum_vol       DOUBLE,
    cum_amount    DOUBLE,
    num_trades    DOUBLE,
    pre_close     DOUBLE,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    bid_volume1   DOUBLE,
    ask_volume1   DOUBLE,
    trade_time    VARCHAR,
    source        VARCHAR DEFAULT 'realtime',
    PRIMARY KEY (code, trade_date, ts)
);

-- Derived per-interval bars (diff of consecutive snapshots, or a backfilled
-- minute bar). Carries the engine's online VWAP/σ/band/z snapshot. Not bound
-- to a run. source ∈ {realtime, backfill}.
CREATE TABLE IF NOT EXISTS vwr_bars (
    code            VARCHAR,
    trade_date      VARCHAR,
    ts              BIGINT,
    interval_vol    DOUBLE,
    interval_amount DOUBLE,
    last            DOUBLE,
    cum_vol         DOUBLE,
    cum_amount      DOUBLE,
    vwap            DOUBLE,
    sigma           DOUBLE,
    band_upper      DOUBLE,
    band_lower      DOUBLE,
    z               DOUBLE,
    source          VARCHAR DEFAULT 'realtime',
    PRIMARY KEY (code, trade_date, ts)
);

-- Engine signals (executed or risk-suppressed). suppressed_by NULL ⇒ executed.
CREATE TABLE IF NOT EXISTS vwr_signals (
    run_id        VARCHAR,
    ts            BIGINT,
    side          VARCHAR,
    z             DOUBLE,
    vwap          DOUBLE,
    sigma         DOUBLE,
    price         DOUBLE,
    reason        VARCHAR,
    suppressed_by VARCHAR,
    PRIMARY KEY (run_id, ts)
);

-- Simulated (paper / backtest) fills.
CREATE TABLE IF NOT EXISTS vwr_trades (
    run_id          VARCHAR,
    seq             BIGINT,
    ts              BIGINT,
    side            VARCHAR,
    qty             DOUBLE,
    price           DOUBLE,
    fee             DOUBLE,
    slippage        DOUBLE,
    realized_pnl    DOUBLE,
    cash_after      DOUBLE,
    position_after  DOUBLE,
    PRIMARY KEY (run_id, seq)
);

-- Per-day trade summary. Source of the 交易汇总报告.
CREATE TABLE IF NOT EXISTS vwr_daily_summary (
    code                VARCHAR,
    trade_date          VARCHAR,
    run_id              VARCHAR,
    n_trades            INTEGER,
    n_wins              INTEGER,
    win_rate            DOUBLE,
    profit_factor       DOUBLE,
    gross_pnl           DOUBLE,
    net_pnl             DOUBLE,
    total_fee           DOUBLE,
    total_slippage      DOUBLE,
    turnover            DOUBLE,
    max_drawdown        DOUBLE,
    avg_holding_seconds DOUBLE,
    final_cash          DOUBLE,
    buy_hold_pnl        DOUBLE,
    circuit_broken      INTEGER,
    PRIMARY KEY (code, trade_date)
);

-- tushare trade_cal cache (exchange=SSE). is_open: 1 trading day / 0 closed.
CREATE TABLE IF NOT EXISTS vwr_trade_cal (
    exchange      VARCHAR,
    cal_date      VARCHAR,
    is_open       INTEGER,
    pretrade_date VARCHAR,
    PRIMARY KEY (exchange, cal_date)
);

-- User-tunable settings. Mirrors the lub_config (key, value_json) shape.
CREATE TABLE IF NOT EXISTS vwr_config (
    key        VARCHAR PRIMARY KEY,
    value_json VARCHAR
);
