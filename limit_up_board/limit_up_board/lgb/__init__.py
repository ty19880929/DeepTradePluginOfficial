"""LightGBM 次日最大溢价概率评分子包（v0.5+）。

「次日最大溢价概率」语义见 ``config.py::LubConfig.lgb_label_threshold_pct``：
T+1 最高价 ≥ T 收盘价 × (1 + 阈值%) 为正例。该信号是 R1 / R2 prompt 的
量化锚点之一，**不是「次日连板」也不是「次日真实可实现收益」**。

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
