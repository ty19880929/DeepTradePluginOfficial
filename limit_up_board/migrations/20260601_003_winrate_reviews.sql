-- limit-up-board v0.11.0: 胜率分析 LLM Review 持久化。
-- 设计：飞书知识库《limit-up-board 胜率分析 — PR 实施计划》PR #4。
--
-- 每次执行 `winrate llm-review` 写一行：
--   - payload_json  — 提交给 LLM 的完整 payload（strategy_context + performance_evidence + review_task）
--   - response_json — LLM 原始结构化返回（Pydantic 模型 dump）
--   - report_path   — 可选写入的 markdown 报告路径，省略 --output 时为 NULL

CREATE TABLE IF NOT EXISTS lub_winrate_reviews (
    review_id        VARCHAR PRIMARY KEY,
    window_start     VARCHAR NOT NULL,
    window_end       VARCHAR NOT NULL,
    llm_provider     VARCHAR NOT NULL,
    llm_model        VARCHAR,
    sample_total     INTEGER NOT NULL,
    sample_resolved  INTEGER NOT NULL,
    strict_win_rate  DOUBLE,
    non_loss_rate    DOUBLE,
    payload_json     VARCHAR NOT NULL,
    response_json    VARCHAR,
    report_path      VARCHAR,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_lub_winrate_reviews_window
    ON lub_winrate_reviews(window_start, window_end);

CREATE INDEX IF NOT EXISTS ix_lub_winrate_reviews_created
    ON lub_winrate_reviews(created_at);
