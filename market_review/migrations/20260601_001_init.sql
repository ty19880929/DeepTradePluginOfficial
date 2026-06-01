-- market-review strategy: full plugin schema (Plan A pure isolation).
-- Tables defined here back the v0.1.0 data layer; field shapes follow
-- design §9.1 ~ §9.10 of MARKET_REVIEW_DESIGN.md. Every table is prefixed
-- ``mr_`` per design §2.4 / §14.3.

-- ---------------------------------------------------------------
-- §9.1  Universe / calendar / exclusions
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_stock_basic (
    ts_code     VARCHAR PRIMARY KEY,
    symbol      VARCHAR,
    name        VARCHAR,
    area        VARCHAR,
    industry    VARCHAR,
    market      VARCHAR,
    exchange    VARCHAR,
    list_status VARCHAR,
    list_date   VARCHAR,
    delist_date VARCHAR,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mr_trade_cal (
    exchange      VARCHAR,
    cal_date      VARCHAR,
    is_open       INTEGER,
    pretrade_date VARCHAR,
    PRIMARY KEY (exchange, cal_date)
);

CREATE TABLE IF NOT EXISTS mr_stock_st (
    ts_code    VARCHAR,
    trade_date VARCHAR,
    st_status  VARCHAR,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS mr_suspend_d (
    ts_code      VARCHAR,
    trade_date   VARCHAR,
    suspend_type VARCHAR,
    PRIMARY KEY (ts_code, trade_date, suspend_type)
);

-- ---------------------------------------------------------------
-- §9.2  Daily quotes / basic indicators
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_daily (
    ts_code    VARCHAR,
    trade_date VARCHAR,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    pre_close  DOUBLE,
    change     DOUBLE,
    pct_chg    DOUBLE,
    vol        DOUBLE,
    amount     DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS mr_daily_basic (
    ts_code          VARCHAR,
    trade_date       VARCHAR,
    close            DOUBLE,
    turnover_rate    DOUBLE,
    turnover_rate_f  DOUBLE,
    volume_ratio     DOUBLE,
    pe               DOUBLE,
    pe_ttm           DOUBLE,
    pb               DOUBLE,
    ps               DOUBLE,
    ps_ttm           DOUBLE,
    total_share      DOUBLE,
    float_share      DOUBLE,
    free_share       DOUBLE,
    total_mv         DOUBLE,
    circ_mv          DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- ---------------------------------------------------------------
-- §9.3  Index daily + dailybasic
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_index_daily (
    ts_code    VARCHAR,
    trade_date VARCHAR,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    pre_close  DOUBLE,
    change     DOUBLE,
    pct_chg    DOUBLE,
    vol        DOUBLE,
    amount     DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS mr_index_dailybasic (
    ts_code          VARCHAR,
    trade_date       VARCHAR,
    total_mv         DOUBLE,
    float_mv         DOUBLE,
    total_share      DOUBLE,
    float_share      DOUBLE,
    free_share       DOUBLE,
    turnover_rate    DOUBLE,
    turnover_rate_f  DOUBLE,
    pe               DOUBLE,
    pe_ttm           DOUBLE,
    pb               DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- ---------------------------------------------------------------
-- §9.4  Limit-up / ladder / theme connections
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_limit_list_d (
    trade_date     VARCHAR,
    ts_code        VARCHAR,
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
    "limit"        VARCHAR,
    PRIMARY KEY (trade_date, ts_code, "limit")
);

CREATE TABLE IF NOT EXISTS mr_limit_step (
    trade_date VARCHAR,
    ts_code    VARCHAR,
    name       VARCHAR,
    nums       INTEGER,
    statibod   VARCHAR,
    PRIMARY KEY (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS mr_limit_cpt_list (
    trade_date VARCHAR,
    ts_code    VARCHAR,
    name       VARCHAR,
    days       INTEGER,
    up_stat    VARCHAR,
    cons_nums  INTEGER,
    up_nums    INTEGER,
    pct_chg    DOUBLE,
    rank       INTEGER,
    PRIMARY KEY (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS mr_limit_ths (
    trade_date    VARCHAR,
    ts_code       VARCHAR,
    name          VARCHAR,
    price         DOUBLE,
    pct_chg       DOUBLE,
    open_num      INTEGER,
    lu_desc       VARCHAR,
    limit_type    VARCHAR,
    tag           VARCHAR,
    status        VARCHAR,
    first_lu_time VARCHAR,
    last_lu_time  VARCHAR,
    limit_order   DOUBLE,
    limit_amount  DOUBLE,
    turnover_rate DOUBLE,
    free_float    DOUBLE,
    PRIMARY KEY (trade_date, ts_code, limit_type)
);

-- ---------------------------------------------------------------
-- §9.5  Money flow (multi-source)
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_moneyflow_hsgt (
    trade_date   VARCHAR PRIMARY KEY,
    ggt_ss       DOUBLE,
    ggt_sz       DOUBLE,
    hgt          DOUBLE,
    sgt          DOUBLE,
    north_money  DOUBLE,
    south_money  DOUBLE
);

CREATE TABLE IF NOT EXISTS mr_hsgt_top10 (
    trade_date  VARCHAR,
    ts_code     VARCHAR,
    name        VARCHAR,
    market_type VARCHAR,
    amount      DOUBLE,
    net_amount  DOUBLE,
    buy         DOUBLE,
    sell        DOUBLE,
    rank        INTEGER,
    PRIMARY KEY (trade_date, ts_code, market_type)
);

CREATE TABLE IF NOT EXISTS mr_moneyflow_mkt (
    trade_date       VARCHAR PRIMARY KEY,
    buy_sm_amount    DOUBLE,
    sell_sm_amount   DOUBLE,
    buy_md_amount    DOUBLE,
    sell_md_amount   DOUBLE,
    buy_lg_amount    DOUBLE,
    sell_lg_amount   DOUBLE,
    buy_elg_amount   DOUBLE,
    sell_elg_amount  DOUBLE,
    net_mf_amount    DOUBLE
);

CREATE TABLE IF NOT EXISTS mr_moneyflow_ind_ths (
    trade_date         VARCHAR,
    name               VARCHAR,
    lead_stock         VARCHAR,
    close              DOUBLE,
    pct_change         DOUBLE,
    company_num        INTEGER,
    pct_change_stock   DOUBLE,
    net_buy_amount     DOUBLE,
    net_sell_amount    DOUBLE,
    net_amount         DOUBLE,
    PRIMARY KEY (trade_date, name)
);

CREATE TABLE IF NOT EXISTS mr_moneyflow_cnt_ths (
    trade_date         VARCHAR,
    ts_code            VARCHAR,
    name               VARCHAR,
    lead_stock         VARCHAR,
    close_price        DOUBLE,
    pct_change         DOUBLE,
    index_close        DOUBLE,
    company_num        INTEGER,
    pct_change_stock   DOUBLE,
    net_buy_amount     DOUBLE,
    net_sell_amount    DOUBLE,
    net_amount         DOUBLE,
    PRIMARY KEY (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS mr_moneyflow (
    ts_code           VARCHAR,
    trade_date        VARCHAR,
    buy_sm_vol        DOUBLE,
    buy_sm_amount     DOUBLE,
    sell_sm_vol       DOUBLE,
    sell_sm_amount    DOUBLE,
    buy_md_vol        DOUBLE,
    buy_md_amount     DOUBLE,
    sell_md_vol       DOUBLE,
    sell_md_amount    DOUBLE,
    buy_lg_vol        DOUBLE,
    buy_lg_amount     DOUBLE,
    sell_lg_vol       DOUBLE,
    sell_lg_amount    DOUBLE,
    buy_elg_vol       DOUBLE,
    buy_elg_amount    DOUBLE,
    sell_elg_vol      DOUBLE,
    sell_elg_amount   DOUBLE,
    net_mf_vol        DOUBLE,
    net_mf_amount     DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- ---------------------------------------------------------------
-- §9.6  Dragon-tiger list
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_top_list (
    trade_date    VARCHAR,
    ts_code       VARCHAR,
    reason        VARCHAR,
    name          VARCHAR,
    close         DOUBLE,
    pct_change    DOUBLE,
    turnover_rate DOUBLE,
    amount        DOUBLE,
    l_sell        DOUBLE,
    l_buy         DOUBLE,
    l_amount      DOUBLE,
    net_amount    DOUBLE,
    net_rate      DOUBLE,
    amount_rate   DOUBLE,
    float_values  DOUBLE
);

CREATE TABLE IF NOT EXISTS mr_top_inst (
    trade_date VARCHAR,
    ts_code    VARCHAR,
    exalter    VARCHAR,
    side       INTEGER,
    reason     VARCHAR,
    buy        DOUBLE,
    buy_rate   DOUBLE,
    sell       DOUBLE,
    sell_rate  DOUBLE,
    net_buy    DOUBLE
);

-- ---------------------------------------------------------------
-- §9.7  Sector / concept daily lines
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_ths_daily (
    ts_code       VARCHAR,
    trade_date    VARCHAR,
    close         DOUBLE,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    pre_close     DOUBLE,
    pct_change    DOUBLE,
    vol           DOUBLE,
    amount        DOUBLE,
    turnover_rate DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS mr_dc_index (
    ts_code    VARCHAR,
    trade_date VARCHAR,
    pct_change DOUBLE,
    close      DOUBLE,
    vol        DOUBLE,
    amount     DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- ---------------------------------------------------------------
-- §9.8  Hot lists / margin / block / chips
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_hot (
    trade_date VARCHAR,
    source     VARCHAR,
    rank       INTEGER,
    ts_code    VARCHAR,
    name       VARCHAR,
    hot_value  DOUBLE,
    PRIMARY KEY (trade_date, source, rank)
);

CREATE TABLE IF NOT EXISTS mr_margin (
    trade_date  VARCHAR,
    exchange_id VARCHAR,
    rzye        DOUBLE,
    rzmre       DOUBLE,
    rzche       DOUBLE,
    rqye        DOUBLE,
    rqmcl       DOUBLE,
    rzrqye      DOUBLE,
    PRIMARY KEY (trade_date, exchange_id)
);

CREATE TABLE IF NOT EXISTS mr_block_trade (
    trade_date VARCHAR,
    ts_code    VARCHAR,
    price      DOUBLE,
    vol        DOUBLE,
    amount     DOUBLE,
    buyer      VARCHAR,
    seller     VARCHAR,
    PRIMARY KEY (trade_date, ts_code, buyer, seller)
);

CREATE TABLE IF NOT EXISTS mr_cyq_perf (
    trade_date  VARCHAR,
    ts_code     VARCHAR,
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

-- ---------------------------------------------------------------
-- §9.9  Per-plugin run history (replaces framework strategy_runs / events)
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_runs (
    run_id            UUID PRIMARY KEY,
    mode              VARCHAR NOT NULL,
    start_date        VARCHAR NOT NULL,
    end_date          VARCHAR NOT NULL,
    anchor            VARCHAR NOT NULL,
    status            VARCHAR NOT NULL,
    started_at        TIMESTAMP NOT NULL,
    finished_at       TIMESTAMP,
    params_json       VARCHAR,
    summary_json      VARCHAR,
    input_fingerprint VARCHAR,
    error             VARCHAR
);

CREATE TABLE IF NOT EXISTS mr_events (
    run_id       UUID NOT NULL,
    seq          BIGINT NOT NULL,
    event_time   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level        VARCHAR NOT NULL,
    event_type   VARCHAR NOT NULL,
    message      VARCHAR NOT NULL,
    payload_json VARCHAR,
    PRIMARY KEY (run_id, seq)
);

-- ---------------------------------------------------------------
-- §9.10  LLM section structured outputs
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS mr_stage_results (
    run_id            UUID NOT NULL,
    section           VARCHAR NOT NULL,
    llm_provider      VARCHAR,
    response_json     VARCHAR NOT NULL,
    raw_response_json VARCHAR,
    PRIMARY KEY (run_id, section)
);
