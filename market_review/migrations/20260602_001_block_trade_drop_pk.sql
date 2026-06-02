-- market-review v0.1.1 — drop PK on mr_block_trade.
--
-- Why: the original PK (trade_date, ts_code, buyer, seller) defined in
-- 20260601_001_init.sql is too restrictive for Tushare ``block_trade``
-- semantics. Tushare returns multiple rows sharing the same
-- (trade_date, ts_code, buyer, seller) tuple — most commonly because
-- ``buyer="机构专用"`` is a generic seat label used by any institutional
-- account, and a single (ts_code, day) can carry several distinct block
-- trades between the same nominal buyer/seller at different prices/volumes.
-- There is no row-id field in Tushare's payload, so no deterministic key
-- can identify a unique row. A 0.1.0 install therefore crashed sync_window
-- with a ConstraintException on busy days.
--
-- Fix: rebuild the table without a PRIMARY KEY constraint and add a plain
-- non-unique index on (trade_date, ts_code) to support the lookup pattern
-- used by mr.metrics.risk._block_trade_discount. Re-sync idempotency moves
-- from materialize()'s per-row DELETE into data.py (DELETE-by-trade_date
-- before INSERT — see _per_day_block_trade).
--
-- DuckDB has no ALTER TABLE DROP PRIMARY KEY, so this is done as
-- CREATE/INSERT/DROP/RENAME. The intermediate table is dropped on commit
-- and any prior data (if a 0.1.0 install crashed mid-sync) is preserved.

CREATE TABLE mr_block_trade__new (
    trade_date VARCHAR,
    ts_code    VARCHAR,
    price      DOUBLE,
    vol        DOUBLE,
    amount     DOUBLE,
    buyer      VARCHAR,
    seller     VARCHAR
);

INSERT INTO mr_block_trade__new (trade_date, ts_code, price, vol, amount, buyer, seller)
SELECT trade_date, ts_code, price, vol, amount, buyer, seller FROM mr_block_trade;

DROP TABLE mr_block_trade;

ALTER TABLE mr_block_trade__new RENAME TO mr_block_trade;

CREATE INDEX IF NOT EXISTS idx_mr_block_trade_date_code
    ON mr_block_trade (trade_date, ts_code);
