-- 同花顺指数目录 — ts_code → 中文名映射，覆盖 .TI (行业) / .CI (概念) / 其它
-- 来源 Tushare ``ths_index`` API（catalog 接口，``static`` cache 类别，盘后不回溯）。
--
-- ``mr_moneyflow_cnt_ths`` 只覆盖概念板块，``moneyflow_ind_ths`` 历史落表丢弃了
-- ``ts_code`` 列。``mr_ths_daily`` 又只存 ts_code 不存 name —— 导致 sectors 章节
-- 在行业指数（如 ``883422.TI``）上 fallback 到代码字面渲染（v0.1.5 bug）。
-- v0.1.6 起新增本表作为 ts_code → name 的主查询入口，覆盖所有 THS 指数类型。

CREATE TABLE IF NOT EXISTS mr_ths_index (
    ts_code   VARCHAR PRIMARY KEY,
    name      VARCHAR,
    count     INTEGER,
    exchange  VARCHAR,
    list_date VARCHAR,
    type      VARCHAR
);
