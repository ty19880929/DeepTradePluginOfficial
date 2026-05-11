"""PR-1.3 — train + activate + scoring artifact lifecycle smoke test.

This test exercises the full PR-1.3 stack **without** going through the typer
CLI: trainer → save_model() → meta.json → ``insert_model`` → ``set_active`` →
``get_active``. CLI itself is plumbing; the artifact-on-disk + DB-row contract
is the load-bearing thing we want to lock in.

Mark: ``slow`` (because LightGBM training, even on toy data, is non-trivial).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deeptrade.core.db import Database

from limit_up_board.lgb import paths as lgb_paths
from limit_up_board.lgb.dataset import LgbDataset
from limit_up_board.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from limit_up_board.lgb.registry import (
    ModelRecord,
    ensure_unique_model_id,
    get_active,
    get_model,
    insert_model,
    mint_model_id,
    set_active,
)
from limit_up_board.lgb.trainer import train_lightgbm

pytestmark = pytest.mark.slow

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_001_lgb_tables.sql"
)


def _seeded_dataset(*, n_per_day: int = 40, n_days: int = 6) -> LgbDataset:
    """Synthetic LgbDataset with a learnable signal (feature 0 ↔ label)."""
    rng = np.random.default_rng(99)
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    sample_meta: list[dict[str, object]] = []
    groups: list[int] = []
    base = 20260501
    for d in range(n_days):
        td = str(base + d)
        for i in range(n_per_day):
            feat = rng.standard_normal(len(FEATURE_NAMES)).astype(float)
            label = 1 if feat[0] > 0 else 0
            rows.append({name: float(v) for name, v in zip(FEATURE_NAMES, feat, strict=False)})
            labels.append(label)
            sample_meta.append(
                {
                    "ts_code": f"S{d:02d}{i:03d}",
                    "trade_date": td,
                    "next_trade_date": str(base + d + 1),
                    "pct_chg_t1": float(feat[0] * 2),
                }
            )
            groups.append(int(td))
    return LgbDataset(
        feature_matrix=pd.DataFrame(rows, columns=FEATURE_NAMES),
        labels=pd.Series(labels, dtype="Int64", name="label"),
        sample_index=pd.DataFrame(sample_meta),
        split_groups=pd.Series(groups, dtype="Int64", name="split_group"),
    )


@pytest.fixture
def isolated_home(tmp_path, monkeypatch) -> Path:  # type: ignore[no-untyped-def]
    """Redirect ~/.deeptrade to tmp_path; model files stay sandboxed."""
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))
    return home


@pytest.fixture
def lgb_db(isolated_home: Path) -> Database:  # noqa: ARG001 — fixture chained for ordering
    from deeptrade.core import paths

    db = Database(paths.db_path())
    sql_text = MIGRATION_FILE.read_text(encoding="utf-8")
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    return db


def test_train_save_activate_lookup_roundtrip(
    isolated_home: Path,  # noqa: ARG001
    lgb_db: Database,
) -> None:
    # 1. Train
    ds = _seeded_dataset()
    result = train_lightgbm(
        ds,
        folds=3,
        num_boost_round=150,
        early_stopping_rounds=40,
        # 见 trainer 测试同款解释：min_data_in_leaf=80 太严，合成 240 样本下无法分裂。
        hyperparams={
            "min_data_in_leaf": 5,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
        },
    )

    # CV passes minimum sanity threshold
    assert result.cv_auc_mean is not None and result.cv_auc_mean > 0.55

    # 2. Save model + meta + dataset under the isolated home
    lgb_paths.ensure_layout()
    model_id = ensure_unique_model_id(
        lgb_db,
        mint_model_id(
            train_end_date="20260530", schema_version=SCHEMA_VERSION, git_commit="testcommit"
        ),
    )
    model_path = lgb_paths.models_dir() / lgb_paths.model_file_name(model_id)
    meta_path = lgb_paths.models_dir() / lgb_paths.meta_file_name(model_id)
    result.model.save_model(str(model_path))
    meta_path.write_text(
        json.dumps({"model_id": model_id, "feature_names": FEATURE_NAMES}),
        encoding="utf-8",
    )

    assert model_path.is_file()
    assert meta_path.is_file()

    # 3. Registry row + activate in one shot
    rel_path = model_path.relative_to(lgb_paths.plugin_data_dir())
    insert_model(
        lgb_db,
        ModelRecord(
            model_id=model_id,
            schema_version=SCHEMA_VERSION,
            train_start_date="20260101",
            train_end_date="20260530",
            n_samples=ds.n_samples,
            n_positive=ds.n_positive,
            cv_auc_mean=result.cv_auc_mean,
            cv_auc_std=result.cv_auc_std,
            cv_logloss_mean=result.cv_logloss_mean,
            feature_count=len(FEATURE_NAMES),
            feature_list_json=json.dumps(FEATURE_NAMES),
            hyperparams_json=json.dumps(result.hyperparams),
            framework_version="0.2.0",
            plugin_version="0.5.0-alpha.2",
            git_commit="testcommit",
            file_path=str(rel_path).replace("\\", "/"),
        ),
        activate=True,
    )

    active = get_active(lgb_db)
    assert active is not None and active.model_id == model_id
    assert active.cv_auc_mean is not None and active.cv_auc_mean > 0.55

    # 4. Booster file is reloadable + feature_name matches
    import lightgbm as lgb_mod

    booster = lgb_mod.Booster(model_file=str(model_path))
    assert list(booster.feature_name()) == FEATURE_NAMES

    # 5. set_active to a different (fake) row, original goes inactive
    insert_model(
        lgb_db,
        ModelRecord(
            model_id="placeholder",
            schema_version=SCHEMA_VERSION,
            train_start_date="20260101",
            train_end_date="20260530",
            n_samples=10,
            n_positive=5,
            feature_count=len(FEATURE_NAMES),
            feature_list_json="[]",
            hyperparams_json="{}",
            plugin_version="0.5.0-alpha.2",
            file_path="models/lgb_model_placeholder.txt",
        ),
        activate=True,
    )
    refreshed = get_model(lgb_db, model_id)
    assert refreshed is not None and refreshed.is_active is False
    new_active = get_active(lgb_db)
    assert new_active is not None and new_active.model_id == "placeholder"
    assert set_active(lgb_db, model_id) is True  # 切回去
    assert get_active(lgb_db).model_id == model_id  # type: ignore[union-attr]
