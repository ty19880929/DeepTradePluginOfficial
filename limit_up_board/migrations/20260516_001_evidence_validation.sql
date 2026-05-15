-- limit-up-board v0.6.4 / P0-2 — evidence 字段白名单校验。
--
-- 新列 `evidence_validation_errors_json` 记录写入该 stage 结果时累积的
-- evidence-field whitelist 违例（pipeline._make_evidence_field_validator）。
-- v0.6.4 阶段保持 NULL（pipeline 通过 VALIDATION_FAILED 事件 surface），
-- 列名 / 类型先固化下来，v0.7 引入 set-check 同款"恢复后留痕"时回填。
--
-- 默认 NULL —— 历史行无需迁移数据。
ALTER TABLE lub_stage_results
    ADD COLUMN evidence_validation_errors_json TEXT;
