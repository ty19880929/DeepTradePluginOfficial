"""LightGBM 评分子包（v0.5+；v0.13.0 起 lgb_score 明确为未校准模型排序分）。

v0.13.0 (P2-2)：原措辞「次日最大溢价**概率**」改为「未校准模型排序分」。
底层标签语义不变——T+1 最高价 ≥ T 收盘价 × (1 + 阈值%) 为正例（见
``config.py::LubConfig.lgb_label_threshold_pct``）——但模型输出未做 Platt /
Isotonic 校准，绝对水平不可解读为 P(Y=1|X)；仅作为相对排序信号供 LLM 参考。
Brier / reliability 校准评估 + isotonic calibrator 训练计划在 v0.13.1 落地，
``lub_lgb_models`` 表已经预留 ``calibration_method / calibration_brier /
calibration_samples`` 三列（迁移 20260524_001_lgb_calibration.sql）。

该信号是强势初筛 / 连板预测 prompt 的量化锚点之一，**不是「次日连板」也不是
「次日真实可实现收益」**。

子模块的职责划分见 ``lightgbm_design.md §2.1``：

* :mod:`features` — 训练 + 推理共用的特征工程（单一来源，避免 train/infer skew）
* :mod:`labels`   — T+1 标签构造（仅训练用）
* :mod:`dataset`  — 训练矩阵构建（后续 PR）
* :mod:`trainer`  — LightGBM 拟合 + 交叉验证 + 落盘（后续 PR）
* :mod:`scorer`   — 推理：加载模型、批量打分、错误降级（后续 PR）
* :mod:`registry` — 模型版本登记（``lub_lgb_models`` 表，后续 PR）
* :mod:`audit`    — 推理结果落 ``lub_lgb_predictions``（后续 PR）
* :mod:`paths`    — 模型 / 训练快照本地存储路径解析（后续 PR）

PR-1.1 仅落地 features + labels；其余模块在后续 PR 中按需新增。
"""

from __future__ import annotations

__all__: list[str] = []
