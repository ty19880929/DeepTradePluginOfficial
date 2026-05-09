-- stdout channel: delivery log so we can verify the payload was actually
-- consumed (not just acknowledged with a fake "✔ push success").
CREATE TABLE IF NOT EXISTS stdout_channel_log (
    delivered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    plugin_id    VARCHAR NOT NULL,
    run_id       UUID NOT NULL,
    status       VARCHAR NOT NULL,
    title        VARCHAR NOT NULL,
    section_count INTEGER NOT NULL,
    item_count    INTEGER NOT NULL,
    metric_count  INTEGER NOT NULL,
    PRIMARY KEY (run_id, plugin_id, delivered_at)
);
