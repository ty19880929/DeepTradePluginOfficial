"""LightGBM 连板概率评分子包（v0.5+）。

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
