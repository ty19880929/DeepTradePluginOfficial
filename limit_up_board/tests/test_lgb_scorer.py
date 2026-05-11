"""PR-2.1 — :class:`LgbScorer` 单元测试。

覆盖设计文档 §7.3 的全部 5 个故障降级分支 + 线程安全：

1. ``lub_lgb_models`` 无 active 行           → loaded=False, all scores NaN
2. 模型文件缺失（DB 有行但磁盘无）           → loaded=False, all scores NaN
3. 模型 schema mismatch (feature 名/数不对)   → loaded=False, all scores NaN
4. 推理时 LightGBM 抛异常                    → loaded=False / all scores NaN
5. lightgbm 包未安装（模拟 ImportError）     → loaded=False, all scores NaN
+ 线程安全：``ThreadPoolExecutor`` 8 并发调用结果一致 + 无异常。

测试本身只需要 lightgbm 来构造 toy booster；CI 安装 ``requirements.txt`` 后
全部可跑。
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptrade.core.db import Database

from limit_up_board.lgb import paths as lgb_paths
from limit_up_board.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from limit_up_board.lgb.registry import ModelRecord, insert_model
from limit_up_board.lgb.scorer import (
    LgbScorer,
    attach_deciles,
)

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_001_lgb_tables.sql"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``~/.deeptrade`` to a tmp dir so model files are sandboxed."""
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))
    return home


@pytest.fixture
def lgb_db(isolated_home: Path) -> Database:  # noqa: ARG001 — chaining
    from deeptrade.core import paths

    db = Database(paths.db_path())
    sql_text = MIGRATION_FILE.read_text(encoding="utf-8")
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    return db


def _toy_feature_frame(n: int = 4, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, len(FEATURE_NAMES))).astype("float64")
    return pd.DataFrame(arr, columns=FEATURE_NAMES, index=[f"S{i:03d}.SH" for i in range(n)])


def _train_toy_booster(n_per_day: int = 40, n_days: int = 5) -> Any:
    """Quickly fit a tiny booster on synthetic data so a real file exists."""
    import lightgbm as lgb_mod

    rng = np.random.default_rng(13)
    X = rng.standard_normal((n_per_day * n_days, len(FEATURE_NAMES))).astype("float64")
    y = (X[:, 0] > 0).astype(int)
    ds = lgb_mod.Dataset(X, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=False)
    return lgb_mod.train(
        {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": 7,
            "min_data_in_leaf": 5,
        },
        ds,
        num_boost_round=15,
    )


def _persist_active_model(
    db: Database, *, model_id: str = "20260530_1_test", feature_count_override: int | None = None
) -> Path:
    """Train a tiny booster, save to disk, register & activate. Returns model file path."""
    lgb_paths.ensure_layout()
    booster = _train_toy_booster()
    model_path = lgb_paths.models_dir() / lgb_paths.model_file_name(model_id)
    booster.save_model(str(model_path))

    rel = model_path.relative_to(lgb_paths.plugin_data_dir())
    insert_model(
        db,
        ModelRecord(
            model_id=model_id,
            schema_version=SCHEMA_VERSION,
            train_start_date="20260101",
            train_end_date="20260530",
            n_samples=200,
            n_positive=80,
            cv_auc_mean=0.66,
            cv_auc_std=0.01,
            cv_logloss_mean=0.55,
            feature_count=(
                feature_count_override
                if feature_count_override is not None
                else len(FEATURE_NAMES)
            ),
            feature_list_json=json.dumps(list(FEATURE_NAMES)),
            hyperparams_json="{}",
            framework_version="0.2.0",
            plugin_version="0.5.0-beta.1",
            git_commit="testcommit",
            file_path=str(rel).replace("\\", "/"),
        ),
        activate=True,
    )
    return model_path


# ---------------------------------------------------------------------------
# 1. No active model
# ---------------------------------------------------------------------------


def test_no_active_model_loads_false(lgb_db: Database) -> None:
    scorer = LgbScorer(lgb_db)
    df = _toy_feature_frame()
    out = scorer.score_batch(df)
    assert scorer.loaded is False
    assert scorer.model_id is None
    assert scorer.load_error == "no_active_model"
    assert out["lgb_score"].isna().all()
    # diagnostics 仍写入（hash 与 missing payload 不依赖模型）
    assert list(out.columns) == ["lgb_score", "feature_hash", "feature_missing_json"]
    assert all(h == "" for h in out["feature_hash"])


# ---------------------------------------------------------------------------
# 2. Model file missing
# ---------------------------------------------------------------------------


def test_model_file_missing(lgb_db: Database) -> None:
    """DB row exists but the booster file is not on disk."""
    insert_model(
        lgb_db,
        ModelRecord(
            model_id="ghost",
            schema_version=SCHEMA_VERSION,
            train_start_date="20260101",
            train_end_date="20260530",
            n_samples=100,
            n_positive=20,
            feature_count=len(FEATURE_NAMES),
            feature_list_json="[]",
            hyperparams_json="{}",
            plugin_version="0.5.0-beta.1",
            file_path="models/lgb_model_ghost.txt",
        ),
        activate=True,
    )

    scorer = LgbScorer(lgb_db)
    df = _toy_feature_frame()
    out = scorer.score_batch(df)
    assert scorer.loaded is False
    assert (scorer.load_error or "").startswith("model_file_missing:")
    assert out["lgb_score"].isna().all()


# ---------------------------------------------------------------------------
# 3. Schema mismatch
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_schema_mismatch_refuses_load(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    """Booster trained against a different feature list → load refused."""
    import lightgbm as lgb_mod

    lgb_paths.ensure_layout()
    bad_feature_names = ["bad_a", "bad_b", "bad_c"]
    X = np.zeros((10, 3), dtype="float64")
    y = np.array([0, 1] * 5)
    ds = lgb_mod.Dataset(X, label=y, feature_name=bad_feature_names, free_raw_data=False)
    booster = lgb_mod.train(
        {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 1},
        ds,
        num_boost_round=3,
    )
    model_file = lgb_paths.models_dir() / "lgb_model_bad.txt"
    booster.save_model(str(model_file))

    rel = model_file.relative_to(lgb_paths.plugin_data_dir())
    insert_model(
        lgb_db,
        ModelRecord(
            model_id="bad",
            schema_version=SCHEMA_VERSION,
            train_start_date="20260101",
            train_end_date="20260530",
            n_samples=10,
            n_positive=5,
            feature_count=3,
            feature_list_json=json.dumps(bad_feature_names),
            hyperparams_json="{}",
            plugin_version="0.5.0-beta.1",
            file_path=str(rel).replace("\\", "/"),
        ),
        activate=True,
    )

    scorer = LgbScorer(lgb_db)
    df = _toy_feature_frame()
    out = scorer.score_batch(df)
    assert scorer.loaded is False
    assert (scorer.load_error or "").startswith("schema_mismatch:")
    assert out["lgb_score"].isna().all()


# ---------------------------------------------------------------------------
# 4. Predict raises
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_failure_degrades_to_nan(
    isolated_home: Path, lgb_db: Database, monkeypatch: pytest.MonkeyPatch  # noqa: ARG001
) -> None:
    """Patch the loaded booster to raise inside predict()."""
    _persist_active_model(lgb_db)
    scorer = LgbScorer(lgb_db)
    df = _toy_feature_frame()
    scorer.warmup()  # ensure loaded
    assert scorer.loaded is True

    class _Boom:
        def predict(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("synthetic predict failure")

    # Swap booster reference; lock is unused outside score_batch so direct
    # attribute mutation is safe for this test.
    loaded = scorer._loaded  # noqa: SLF001
    assert loaded is not None
    object.__setattr__(loaded, "booster", _Boom())

    out = scorer.score_batch(df)
    assert out["lgb_score"].isna().all()
    # diagnostics still populated
    assert all(len(h) > 0 for h in out["feature_hash"])


# ---------------------------------------------------------------------------
# 5. lightgbm import error during load
# ---------------------------------------------------------------------------


def test_lightgbm_import_error_during_load(
    lgb_db: Database, monkeypatch: pytest.MonkeyPatch, isolated_home: Path  # noqa: ARG001
) -> None:
    """Simulate the booster-load path failing because lightgbm is not importable."""
    # Set up a valid registry row + a real file on disk so we know the failure
    # comes from the import shim, not from row/file resolution.
    _persist_active_model(lgb_db)

    from limit_up_board.lgb import scorer as scorer_mod

    def _raise(_model_file: str) -> Any:
        raise RuntimeError("lightgbm 未安装：pip install 'lightgbm>=4.3' …")

    monkeypatch.setattr(scorer_mod, "_load_booster", _raise)

    scorer = LgbScorer(lgb_db)
    df = _toy_feature_frame()
    out = scorer.score_batch(df)
    assert scorer.loaded is False
    assert (scorer.load_error or "").startswith("booster_load_failed:")
    assert out["lgb_score"].isna().all()


# ---------------------------------------------------------------------------
# Happy path + thread safety
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_score_batch_happy_path(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    _persist_active_model(lgb_db, model_id="20260530_1_happy")
    scorer = LgbScorer(lgb_db)
    df = _toy_feature_frame(n=6)
    out = scorer.score_batch(df)
    assert scorer.loaded is True
    assert scorer.model_id == "20260530_1_happy"
    assert out["lgb_score"].notna().all()
    assert (out["lgb_score"].between(0.0, 1.0)).all()
    assert all(len(h) > 0 for h in out["feature_hash"])
    # All-not-NaN inputs → empty missing list per row
    assert all(json.loads(s) == [] for s in out["feature_missing_json"])


@pytest.mark.slow
def test_score_batch_concurrent(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    """8 worker threads share the same scorer instance and produce identical
    output (booster.predict is thread-safe + LgbScorer holds no mutable state)."""
    _persist_active_model(lgb_db, model_id="20260530_1_concurrent")
    scorer = LgbScorer(lgb_db)
    df = _toy_feature_frame(n=12, seed=42)
    # Pre-warm so the lazy load happens once on the main thread (still tests
    # concurrent score_batch — which is what the design promises).
    scorer.warmup()

    def _do() -> pd.DataFrame:
        return scorer.score_batch(df)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _do(), range(8)))

    base = results[0]["lgb_score"].to_numpy()
    for r in results[1:]:
        np.testing.assert_allclose(r["lgb_score"].to_numpy(), base, rtol=0, atol=0)


def test_empty_dataframe_returns_empty(lgb_db: Database) -> None:
    scorer = LgbScorer(lgb_db)
    empty = pd.DataFrame(columns=list(FEATURE_NAMES))
    out = scorer.score_batch(empty)
    assert list(out.columns) == ["lgb_score", "feature_hash", "feature_missing_json"]
    assert len(out) == 0


def test_missing_columns_input_degrades(lgb_db: Database) -> None:
    """A caller-side bug (wrong columns) must not crash the run; degrade to NaN."""
    scorer = LgbScorer(lgb_db)
    df = pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]})
    out = scorer.score_batch(df)
    # reindex makes all FEATURE_NAMES columns appear as NaN; assert_columns passes,
    # but model isn't loaded (no active row in this test) → still NaN scores.
    assert out["lgb_score"].isna().all()


def test_warmup_idempotent(lgb_db: Database) -> None:
    """Calling warmup twice (or before any score_batch) must not double-load."""
    scorer = LgbScorer(lgb_db)
    scorer.warmup()
    first = scorer.load_error
    scorer.warmup()
    assert scorer.load_error == first


# ---------------------------------------------------------------------------
# Decile attachment
# ---------------------------------------------------------------------------


def test_attach_deciles_assigns_1_to_10() -> None:
    df = pd.DataFrame(
        {"lgb_score": [0.05 * i for i in range(20)]},  # 0.0, 0.05, …, 0.95
        index=[f"S{i:03d}" for i in range(20)],
    )
    deciles = attach_deciles(df, n_buckets=10)
    assert int(deciles.min()) == 1
    assert int(deciles.max()) == 10
    # 20 evenly distributed scores → each bucket gets 2.
    counts = deciles.value_counts().to_dict()
    assert all(v == 2 for v in counts.values())


def test_attach_deciles_too_few_returns_na() -> None:
    df = pd.DataFrame({"lgb_score": [0.1, 0.5, 0.9]}, index=["a", "b", "c"])
    deciles = attach_deciles(df, n_buckets=10)
    assert deciles.isna().all()


def test_attach_deciles_nan_preserved() -> None:
    scores = [float("nan")] * 5 + [0.1 * i for i in range(15)]
    df = pd.DataFrame({"lgb_score": scores}, index=[f"S{i}" for i in range(20)])
    deciles = attach_deciles(df, n_buckets=10)
    # first 5 NaN inputs → NaN deciles; remaining 15 get buckets
    assert deciles.iloc[:5].isna().all()
    assert deciles.iloc[5:].notna().all()


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


def test_audit_record_predictions_skips_invalid(lgb_db: Database) -> None:
    from limit_up_board.lgb.audit import record_predictions

    inserted = record_predictions(
        lgb_db,
        run_id="00000000-0000-0000-0000-000000000001",
        trade_date="20260530",
        model_id="m1",
        rows=[
            {"ts_code": "600519.SH", "lgb_score": 0.73, "lgb_decile": 8, "feature_hash": "abc"},
            {"ts_code": "000001.SZ", "lgb_score": None},  # skipped
            {"lgb_score": 0.5},  # missing ts_code → skipped
            {"ts_code": "300033.SZ", "lgb_score": "0.42", "lgb_decile": "5"},
        ],
    )
    assert inserted == 2
    rows = lgb_db.fetchall(
        "SELECT ts_code, lgb_score, lgb_decile FROM lub_lgb_predictions "
        "ORDER BY ts_code"
    )
    codes = {str(r[0]) for r in rows}
    assert codes == {"300033.SZ", "600519.SH"}
