"""PR-1.3 — LightGBM trainer 单元测试（不跑真训练）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from limit_up_board.lgb.dataset import LgbDataset
from limit_up_board.lgb.features import FEATURE_NAMES
from limit_up_board.lgb.trainer import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_NUM_BOOST_ROUND,
    LGB_PARAMS,
    train_lightgbm,
)


def _empty_dataset() -> LgbDataset:
    return LgbDataset(
        feature_matrix=pd.DataFrame(columns=FEATURE_NAMES),
        labels=pd.Series([], dtype="Int64", name="label"),
        sample_index=pd.DataFrame(
            columns=["ts_code", "trade_date", "next_trade_date", "pct_chg_t1"]
        ),
        split_groups=pd.Series([], dtype="Int64", name="split_group"),
    )


def _toy_dataset(n_per_day: int = 60, n_days: int = 10, *, with_signal: bool = True) -> LgbDataset:
    """合成训练集：50 个特征 + 二分类标签 + 跨日 split_groups。

    若 ``with_signal=True``，特征 0 与 label 强相关（AUC 远 > 0.5）；
    若 False，所有特征独立于 label（AUC ≈ 0.5）。
    """
    rng = np.random.default_rng(123)
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    sample_meta: list[dict[str, object]] = []
    groups: list[int] = []
    base_trade_date = 20260501
    for day_idx in range(n_days):
        trade_date = str(base_trade_date + day_idx)
        for i in range(n_per_day):
            feat = rng.standard_normal(len(FEATURE_NAMES)).astype(float)
            label_logit = feat[0] if with_signal else rng.standard_normal()
            label = 1 if label_logit > 0 else 0
            rows.append({name: float(v) for name, v in zip(FEATURE_NAMES, feat, strict=False)})
            labels.append(label)
            sample_meta.append(
                {
                    "ts_code": f"T{day_idx:02d}{i:03d}.SZ",
                    "trade_date": trade_date,
                    "next_trade_date": str(base_trade_date + day_idx + 1),
                    "pct_chg_t1": float(label_logit * 2),
                }
            )
            groups.append(int(trade_date))
    feature_matrix = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return LgbDataset(
        feature_matrix=feature_matrix.reset_index(drop=True),
        labels=pd.Series(labels, dtype="Int64", name="label").reset_index(drop=True),
        sample_index=pd.DataFrame(sample_meta).reset_index(drop=True),
        split_groups=pd.Series(groups, dtype="Int64", name="split_group").reset_index(drop=True),
    )


class TestHyperparamShape:
    def test_default_params_complete(self) -> None:
        # design §6.3 — 关键字段在
        for key in (
            "objective",
            "learning_rate",
            "num_leaves",
            "min_data_in_leaf",
            "feature_fraction",
            "lambda_l2",
            "is_unbalance",
            "deterministic",
            "seed",
        ):
            assert key in LGB_PARAMS, key
        assert LGB_PARAMS["objective"] == "binary"
        assert LGB_PARAMS["seed"] == 42
        assert LGB_PARAMS["deterministic"] is True

    def test_default_round_constants(self) -> None:
        assert DEFAULT_NUM_BOOST_ROUND == 1500
        assert DEFAULT_EARLY_STOPPING_ROUNDS == 100


class TestTrainGuards:
    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            train_lightgbm(_empty_dataset())

    def test_unlabeled_rows_raise(self) -> None:
        ds = _toy_dataset(n_per_day=4, n_days=2)
        # 把第 0 行 label 改成 NA → 应该报错
        ds.labels.iloc[0] = pd.NA
        with pytest.raises(ValueError, match="filter_labeled"):
            train_lightgbm(ds)


class TestTrainSmoke:
    """端到端 smoke：用合成数据验证训练流程通畅 + AUC > 0.5。

    标记 slow 但本身耗时极短（合成数据 + 600 round），CI 默认运行。
    """

    @pytest.mark.slow
    def test_train_with_signal_yields_positive_auc(self) -> None:
        ds = _toy_dataset(n_per_day=40, n_days=6, with_signal=True)
        # 默认 min_data_in_leaf=80 是为生产 ~1500 样本设计；合成 240 样本场景下需放宽
        # 让 booster 能产生 split，否则 best_iteration=0、AUC=0.5（无学习信号）。
        result = train_lightgbm(
            ds,
            folds=3,
            num_boost_round=200,
            early_stopping_rounds=50,
            hyperparams={
                "min_data_in_leaf": 5,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
            },
        )
        # 有信号 → AUC 显著 > 0.5
        assert result.cv_auc_mean is not None
        assert result.cv_auc_mean > 0.55, f"AUC={result.cv_auc_mean}"
        # 特征重要性返回 top-20
        assert 1 <= len(result.feature_importance) <= 20
        # 至少 1 个非零重要性
        assert max(score for _, score in result.feature_importance) > 0
        # hyperparams 快照含默认 + 调用方覆盖项；这里我们覆盖了 3 个键，其余应保留默认。
        overridden = {"min_data_in_leaf", "feature_fraction", "bagging_fraction"}
        for k, v in LGB_PARAMS.items():
            if k in overridden:
                continue
            assert result.hyperparams[k] == v

    @pytest.mark.slow
    def test_train_skip_cv_when_few_groups(self) -> None:
        # 只有 2 个交易日，folds=5 时 GroupKFold 会失败 → 应跳过 CV 但仍返回结果
        ds = _toy_dataset(n_per_day=20, n_days=2)
        result = train_lightgbm(
            ds,
            folds=5,
            num_boost_round=50,
            early_stopping_rounds=20,
            hyperparams={"min_data_in_leaf": 5},
        )
        assert result.cv_auc_mean is None
        assert result.cv_auc_std is None
        # 模型仍然训出来了
        assert result.model is not None
