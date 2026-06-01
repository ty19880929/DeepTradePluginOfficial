-- market-review: user-tunable config table (design §8 / §9).
-- Separated from 001_init.sql so that future config-schema bumps don't
-- force a re-checksum of the big data-table migration.

CREATE TABLE IF NOT EXISTS mr_config (
    key        VARCHAR PRIMARY KEY,
    value_json VARCHAR NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
