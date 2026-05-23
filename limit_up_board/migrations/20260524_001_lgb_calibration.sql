-- limit-up-board v0.13.0 (P2-2)：为 lub_lgb_models 预留校准元数据列。
--
-- 当前 lgb_score 是 LightGBM 的未校准 sigmoid 输出，绝对水平不可解读为
-- P(Y=1|X)；v0.13.0 起 prompt / report / CLI 文案统一改称「未校准排序分」。
-- 校准器训练 / 加载 / Brier evaluate 流程将在 v0.13.1 落地，本迁移先把
-- schema 准备好（旧行 calibration_* 列保持 NULL，scorer 自然走 raw 分支）。
--
-- DuckDB ALTER TABLE ADD COLUMN ... IF NOT EXISTS 兼容性：
--   * 该语法在 v0.10+ 可用；早于 v0.10 会回退到普通 ADD COLUMN，重复执行报
--     "Column already exists"。迁移由框架 _apply_migrations 顺序执行、且
--     applied_migrations 表会跳过已应用的 version，因此这里使用普通 ADD
--     COLUMN 即可，幂等性由 applied_migrations 保证。

ALTER TABLE lub_lgb_models ADD COLUMN calibration_method VARCHAR;
ALTER TABLE lub_lgb_models ADD COLUMN calibration_brier  DOUBLE;
ALTER TABLE lub_lgb_models ADD COLUMN calibration_samples INTEGER;
