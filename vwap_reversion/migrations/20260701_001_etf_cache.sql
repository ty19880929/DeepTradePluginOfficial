-- vwap-reversion P1 optimization: ETF universe and daily feature cache.

CREATE TABLE IF NOT EXISTS vwr_etf_universe (
    ts_code        VARCHAR PRIMARY KEY,
    name           VARCHAR,
    fund_type      VARCHAR,
    invest_type    VARCHAR,
    market         VARCHAR,
    status         VARCHAR,
    list_date      VARCHAR,
    delist_date    VARCHAR,
    management     VARCHAR,
    benchmark      VARCHAR,
    margin_eligible INTEGER DEFAULT 0,
    t0_eligible     INTEGER DEFAULT 0,
    enabled         INTEGER DEFAULT 1,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vwr_etf_daily (
    ts_code         VARCHAR,
    trade_date      VARCHAR,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    pre_close       DOUBLE,
    pct_chg         DOUBLE,
    vol             DOUBLE,
    amount          DOUBLE,
    adj_factor      DOUBLE,
    fd_share        DOUBLE,
    unit_nav        DOUBLE,
    adj_nav         DOUBLE,
    up_limit        DOUBLE,
    down_limit      DOUBLE,
    source_json     VARCHAR,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE TABLE IF NOT EXISTS vwr_daily_features (
    ts_code            VARCHAR,
    trade_date         VARCHAR,
    ret_1d             DOUBLE,
    ret_5d             DOUBLE,
    rv_20d             DOUBLE,
    atr_pct_20d        DOUBLE,
    amount_ma20        DOUBLE,
    amount_pctile_252  DOUBLE,
    gap_pct            DOUBLE,
    liquidity_ok       INTEGER,
    volatility_regime  VARCHAR,
    trend_regime       VARCHAR,
    source_json        VARCHAR,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
