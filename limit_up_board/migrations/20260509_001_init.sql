-- limit-up-board strategy: full plugin schema (Plan A pure isolation).

CREATE TABLE IF NOT EXISTS lub_stock_basic (
    ts_code VARCHAR PRIMARY KEY,
    symbol VARCHAR, name VARCHAR, area VARCHAR, industry VARCHAR,
    market VARCHAR, exchange VARCHAR,
    list_status VARCHAR, list_date VARCHAR, delist_date VARCHAR,
    is_hs VARCHAR, act_name VARCHAR, act_ent_type VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lub_trade_cal (
    exchange VARCHAR, cal_date VARCHAR, is_open INTEGER, pretrade_date VARCHAR,
    PRIMARY KEY (exchange, cal_date)
);

CREATE TABLE IF NOT EXISTS lub_daily (
    ts_code VARCHAR, trade_date VARCHAR,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,
    change DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS lub_daily_basic (
    ts_code VARCHAR, trade_date VARCHAR, close DOUBLE,
    turnover_rate DOUBLE, turnover_rate_f DOUBLE, volume_ratio DOUBLE,
    pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE, ps DOUBLE, ps_ttm DOUBLE,
    total_share DOUBLE, float_share DOUBLE, free_share DOUBLE,
    total_mv DOUBLE, circ_mv DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS lub_moneyflow (
    ts_code         VARCHAR,
    trade_date      VARCHAR,
    buy_sm_vol      DOUBLE, buy_sm_amount   DOUBLE,
    sell_sm_vol     DOUBLE, sell_sm_amount  DOUBLE,
    buy_md_vol      DOUBLE, buy_md_amount   DOUBLE,
    sell_md_vol     DOUBLE, sell_md_amount  DOUBLE,
    buy_lg_vol      DOUBLE, buy_lg_amount   DOUBLE,
    sell_lg_vol     DOUBLE, sell_lg_amount  DOUBLE,
    buy_elg_vol     DOUBLE, buy_elg_amount  DOUBLE,
    sell_elg_vol    DOUBLE, sell_elg_amount DOUBLE,
    net_mf_vol      DOUBLE, net_mf_amount   DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS lub_limit_list_d (
    trade_date     VARCHAR NOT NULL,
    ts_code        VARCHAR NOT NULL,
    name           VARCHAR,
    industry       VARCHAR,
    close          DOUBLE,
    pct_chg        DOUBLE,
    amount         DOUBLE,
    fd_amount      DOUBLE,
    limit_amount   DOUBLE,
    float_mv       DOUBLE,
    total_mv       DOUBLE,
    turnover_ratio DOUBLE,
    first_time     VARCHAR,
    last_time      VARCHAR,
    open_times     INTEGER,
    up_stat        VARCHAR,
    limit_times    INTEGER,
    "limit"        VARCHAR NOT NULL,
    PRIMARY KEY (trade_date, ts_code, "limit")
);

CREATE TABLE IF NOT EXISTS lub_limit_ths (
    trade_date         VARCHAR NOT NULL,
    ts_code            VARCHAR NOT NULL,
    name               VARCHAR,
    price              DOUBLE,
    pct_chg            DOUBLE,
    open_num           INTEGER,
    lu_desc            VARCHAR,
    limit_type         VARCHAR NOT NULL,
    tag                VARCHAR,
    status             VARCHAR,
    first_lu_time      VARCHAR,
    last_lu_time       VARCHAR,
    limit_order        DOUBLE,
    limit_amount       DOUBLE,
    turnover_rate      DOUBLE,
    free_float         DOUBLE,
    lu_limit_order     DOUBLE,
    limit_up_suc_rate  DOUBLE,
    turnover           DOUBLE,
    sum_float          DOUBLE,
    market_type        VARCHAR,
    PRIMARY KEY (trade_date, ts_code, limit_type)
);

CREATE TABLE IF NOT EXISTS lub_stage_results (
    run_id              UUID NOT NULL,
    stage               VARCHAR NOT NULL,
    llm_provider        VARCHAR,
    batch_no            INTEGER,
    trade_date          VARCHAR NOT NULL,
    ts_code             VARCHAR NOT NULL,
    name                VARCHAR,
    score               DOUBLE,
    rank                INTEGER,
    decision            VARCHAR,
    rationale           VARCHAR,
    evidence_json       VARCHAR,
    risk_flags_json     VARCHAR,
    raw_response_json   VARCHAR,
    PRIMARY KEY (run_id, stage, ts_code)
);
CREATE INDEX IF NOT EXISTS ix_lub_stage_results_run_provider
    ON lub_stage_results(run_id, llm_provider, stage);

CREATE TABLE IF NOT EXISTS lub_runs (
    run_id        UUID PRIMARY KEY,
    trade_date    VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    is_intraday   BOOLEAN NOT NULL DEFAULT FALSE,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP,
    params_json   VARCHAR,
    summary_json  VARCHAR,
    error         VARCHAR
);

CREATE TABLE IF NOT EXISTS lub_events (
    run_id       UUID NOT NULL,
    seq          BIGINT NOT NULL,
    event_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level        VARCHAR NOT NULL,
    event_type   VARCHAR NOT NULL,
    message      VARCHAR NOT NULL,
    payload_json VARCHAR,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS lub_top_list (
    trade_date     VARCHAR NOT NULL,
    ts_code        VARCHAR NOT NULL,
    reason         VARCHAR NOT NULL,
    name           VARCHAR,
    close          DOUBLE,
    pct_change     DOUBLE,
    turnover_rate  DOUBLE,
    amount         DOUBLE,
    l_sell         DOUBLE,
    l_buy          DOUBLE,
    l_amount       DOUBLE,
    net_amount     DOUBLE,
    net_rate       DOUBLE,
    amount_rate    DOUBLE,
    float_values   DOUBLE
);

CREATE TABLE IF NOT EXISTS lub_top_inst (
    trade_date  VARCHAR NOT NULL,
    ts_code     VARCHAR NOT NULL,
    exalter     VARCHAR NOT NULL,
    side        INTEGER NOT NULL,
    reason      VARCHAR NOT NULL,
    buy         DOUBLE,
    buy_rate    DOUBLE,
    sell        DOUBLE,
    sell_rate   DOUBLE,
    net_buy     DOUBLE
);

CREATE TABLE IF NOT EXISTS lub_cyq_perf (
    trade_date  VARCHAR NOT NULL,
    ts_code     VARCHAR NOT NULL,
    his_low     DOUBLE,
    his_high    DOUBLE,
    cost_5pct   DOUBLE,
    cost_15pct  DOUBLE,
    cost_50pct  DOUBLE,
    cost_85pct  DOUBLE,
    cost_95pct  DOUBLE,
    weight_avg  DOUBLE,
    winner_rate DOUBLE,
    PRIMARY KEY (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS lub_config (
    key         VARCHAR PRIMARY KEY,
    value_json  VARCHAR NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
