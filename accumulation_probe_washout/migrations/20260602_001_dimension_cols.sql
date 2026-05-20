-- accumulation-probe-washout v0.3.0: physicalise the per-dimension scores.
--
-- Background: ``apw_stage_results.dimension_scores_json`` already carries the
-- six LLM-produced dimension scores (accumulation / probe / washout /
-- launch_timing / capital_confirmation / risk). Stats / lgb-side queries want
-- to GROUP BY / aggregate / Pearson-correlate on these without paying a
-- json_extract per row. PR-1 adds the six DOUBLE columns + a launch_timing
-- index, with the runner dual-writing both the json column and the columns
-- so a roll-back can drop the columns without losing data.
--
-- Column nullability -- NULL on rows written before this migration. The
-- back-fill is intentionally NOT done here (the json blob is still the
-- canonical source of truth for older rows, dual-write only kicks in for
-- new rows). A later migration can ``UPDATE ... SET dim_x =
-- json_extract(...)`` if a back-fill becomes useful.

ALTER TABLE apw_stage_results ADD COLUMN dim_accumulation         DOUBLE;
ALTER TABLE apw_stage_results ADD COLUMN dim_probe                DOUBLE;
ALTER TABLE apw_stage_results ADD COLUMN dim_washout              DOUBLE;
ALTER TABLE apw_stage_results ADD COLUMN dim_launch_timing        DOUBLE;
ALTER TABLE apw_stage_results ADD COLUMN dim_capital_confirmation DOUBLE;
ALTER TABLE apw_stage_results ADD COLUMN dim_risk                 DOUBLE;

CREATE INDEX IF NOT EXISTS idx_apw_stage_results_dim_launch_timing
    ON apw_stage_results(dim_launch_timing);
