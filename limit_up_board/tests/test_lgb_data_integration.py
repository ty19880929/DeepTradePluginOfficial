"""PR-2.2 — :func:`data._attach_lgb_scores` 与 ``Round1Bundle`` 的集成测试。

测试目标（lightgbm_iteration_plan.md §3.1 PR-2.2 验收）：
* 模型存在  → 候选股 ``lgb_score`` ∈ (0, 100]、``lgb_decile``∈ 1..10、
  ``lgb_feature_missing`` 为 list；``bundle.lgb_model_id`` 填充；
  ``bundle.lgb_predictions`` 含 audit payload。
* 模型不存在 / scorer=None → 候选 ``lgb_score=None``；``bundle.lgb_model_id=None``；
  ``bundle.data_unavailable`` 提示原因（除 scorer=None 的"用户显式禁用"分支外）。
* 模型异常（mock 推理抛错）→ 候选 ``lgb_score=None``；run 不中断。

不依赖任何 tushare/LLM；用 hand-crafted ``Round1Bundle`` + ``candidates_df``
+ 配套 lookup 直接驱动 ``_attach_lgb_scores``——这是 lightgbm_design.md §7.2
里抽象描述的 collect_round1 末段的等价单元。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from deeptrade.core.db import Database

from limit_up_board.data import (
    Round1Bundle,
    SectorStrength,
    _attach_lgb_scores,
)
from limit_up_board.lgb import paths as lgb_paths
from limit_up_board.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from limit_up_board.lgb.registry import ModelRecord, insert_model
from limit_up_board.lgb.scorer import LgbScorer

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_001_lgb_tables.sql"
)


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_lgb_scorer.py style)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


def _train_toy_booster(seed: int = 11) -> Any:
    import lightgbm as lgb_mod

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((200, len(FEATURE_NAMES))).astype("float64")
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
        num_boost_round=20,
    )


def _persist_active_model(db: Database, *, model_id: str = "20260530_1_int") -> None:
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
            n_positive=100,
            cv_auc_mean=0.75,
            cv_auc_std=0.02,
            cv_logloss_mean=0.55,
            feature_count=len(FEATURE_NAMES),
            feature_list_json=json.dumps(list(FEATURE_NAMES)),
            hyperparams_json="{}",
            framework_version="0.2.0",
            plugin_version="0.5.0-beta.1",
            git_commit="testcommit",
            file_path=str(rel).replace("\\", "/"),
        ),
        activate=True,
    )


# ---------------------------------------------------------------------------
# Synthetic Round1Bundle inputs
# ---------------------------------------------------------------------------


def _make_inputs(n: int = 12) -> tuple[Round1Bundle, pd.DataFrame, dict, dict, dict, pd.DataFrame]:
    """Hand-crafted bundle + candidates_df + auxiliary frames.

    The shape mirrors what ``collect_round1`` would produce just before
    ``_attach_lgb_scores`` runs: candidate dicts (``bundle.candidates``)
    sourced from ``_build_candidate_rows`` + raw frames for feature derivation.
    """
    rows = []
    candidate_dicts: list[dict[str, Any]] = []
    rng_amt = 1e8
    for i in range(n):
        ts = f"60{i:04d}.SH"
        rows.append(
            {
                "ts_code": ts,
                "name": f"S{i:02d}",
                "trade_date": "20260530",
                "industry": "电子",
                "industry_basic": "电子",
                "market": "主板",
                "exchange": "SSE",
                "list_date": "20180101",
                "close": 8.0 + i * 0.2,
                "pct_chg": 9.97,
                "amount": rng_amt + i * 1e6,
                "fd_amount": 2e7 + i * 5e5,
                "limit_amount": 8e7 + i * 1e6,
                "float_mv": 3e9 + i * 1e8,
                "total_mv": 4e9 + i * 1e8,
                "open_times": i % 3,
                "limit_times": 1,
                "first_time": "09:30:00",
                "last_time": "10:00:00",
                "up_stat": "1/1",
                "turnover_ratio": 4.5 + i * 0.1,
            }
        )
        candidate_dicts.append({"candidate_id": ts, "ts_code": ts, "name": f"S{i:02d}"})
    candidates_df = pd.DataFrame(rows)

    daily_rows: list[dict[str, Any]] = []
    daily_basic_rows: list[dict[str, Any]] = []
    moneyflow_rows: list[dict[str, Any]] = []
    for r in rows:
        ts = r["ts_code"]
        for d in range(30):
            daily_rows.append(
                {
                    "ts_code": ts,
                    "trade_date": f"2026050{d % 10}",
                    "open": 8.0 + d * 0.05,
                    "high": 8.2 + d * 0.05,
                    "low": 7.8 + d * 0.05,
                    "close": 8.0 + d * 0.05,
                    "pre_close": 7.95 + d * 0.05,
                    "pct_chg": 0.6,
                    "amount": 5e5,
                    "vol": 1000,
                }
            )
            daily_basic_rows.append(
                {
                    "ts_code": ts,
                    "trade_date": f"2026050{d % 10}",
                    "turnover_rate": 3.0,
                    "volume_ratio": 1.5,
                    "circ_mv": 3e7,
                }
            )
        for d in range(5):
            moneyflow_rows.append(
                {
                    "ts_code": ts,
                    "trade_date": f"2026052{d}",
                    "net_mf_amount": 1e4 * (1 if d % 2 else -1),
                    "buy_lg_amount": 5e4,
                    "buy_elg_amount": 3e4,
                }
            )
    daily_df = pd.DataFrame(daily_rows)
    daily_basic_df = pd.DataFrame(daily_basic_rows)
    moneyflow_df = pd.DataFrame(moneyflow_rows)

    bundle = Round1Bundle(
        trade_date="20260530",
        next_trade_date="20260531",
        candidates=candidate_dicts,
        market_summary={
            "limit_up_count": n,
            "limit_step_distribution": {"1": n - 1, "2": 1},
            "yesterday_failure_rate": {"rate_pct": 12.0},
            "yesterday_winners_today": {"continuation_rate_pct": 33.3},
            "limit_step_trend": {"high_board_delta": 1},
        },
        sector_strength=SectorStrength(source="limit_cpt_list", data={"top_sectors": []}),
        data_unavailable=[],
    )

    return bundle, candidates_df, {}, {}, {}, daily_df  # placeholder; real frames built below


def _build_kwargs(
    candidates_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    moneyflow_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "candidates_df": candidates_df,
        "daily_df": daily_df,
        "daily_basic_df": daily_basic_df,
        "moneyflow_df": moneyflow_df,
        "top_list_df": None,
        "top_inst_df": None,
        "cyq_perf_df": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scorer_none_fills_disabled(lgb_db: Database) -> None:
    """User passed ``--no-lgb``: scorer is None, no data_unavailable annotation."""
    bundle, candidates_df, _, _, _, daily_df = _make_inputs()
    daily_basic_df = pd.DataFrame()
    moneyflow_df = pd.DataFrame()
    kwargs = _build_kwargs(candidates_df, daily_df, daily_basic_df, moneyflow_df)
    _attach_lgb_scores(bundle, scorer=None, **kwargs)

    assert bundle.lgb_model_id is None
    assert all(c["lgb_score"] is None for c in bundle.candidates)
    assert all(c["lgb_decile"] is None for c in bundle.candidates)
    # explicit --no-lgb path → don't pollute data_unavailable
    assert not any("lgb_model" in u for u in bundle.data_unavailable)


def test_scorer_no_active_model_marks_unavailable(lgb_db: Database) -> None:
    """No active model row → all candidates get None and data_unavailable mentions LGB."""
    scorer = LgbScorer(lgb_db)
    bundle, candidates_df, _, _, _, daily_df = _make_inputs()
    daily_basic_df = pd.DataFrame()
    moneyflow_df = pd.DataFrame()
    kwargs = _build_kwargs(candidates_df, daily_df, daily_basic_df, moneyflow_df)
    _attach_lgb_scores(bundle, scorer=scorer, **kwargs)

    assert bundle.lgb_model_id is None
    assert all(c["lgb_score"] is None for c in bundle.candidates)
    assert any("lgb_model" in u for u in bundle.data_unavailable)


@pytest.mark.slow
def test_scorer_loaded_attaches_scores_and_audit(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    """Happy path: candidates get lgb_score ∈ (0, 100], deciles, audit rows."""
    _persist_active_model(lgb_db, model_id="20260530_1_happy")
    scorer = LgbScorer(lgb_db)

    bundle, candidates_df, _, _, _, daily_df = _make_inputs(n=12)
    daily_basic_df = pd.DataFrame()
    moneyflow_df = pd.DataFrame()
    kwargs = _build_kwargs(candidates_df, daily_df, daily_basic_df, moneyflow_df)
    _attach_lgb_scores(bundle, scorer=scorer, **kwargs)

    assert bundle.lgb_model_id == "20260530_1_happy"
    scores = [c["lgb_score"] for c in bundle.candidates]
    assert all(s is not None for s in scores)
    assert all(0.0 <= s <= 100.0 for s in scores)
    # decile is in 1..10 once we have ≥ 10 candidates
    deciles = [c["lgb_decile"] for c in bundle.candidates if c["lgb_decile"] is not None]
    assert all(1 <= d <= 10 for d in deciles)
    # feature_missing is a list (may be empty)
    assert all(isinstance(c["lgb_feature_missing"], list) for c in bundle.candidates)
    # audit payload built
    assert len(bundle.lgb_predictions) == 12
    assert all("feature_hash" in row for row in bundle.lgb_predictions)


@pytest.mark.slow
def test_scorer_predict_failure_degrades_gracefully(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    """score_batch failing inside scorer must not crash _attach_lgb_scores."""
    _persist_active_model(lgb_db, model_id="20260530_1_fail")
    scorer = LgbScorer(lgb_db)
    scorer.warmup()
    assert scorer.loaded

    class _Boom:
        def predict(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("forced predict error")

    object.__setattr__(scorer._loaded, "booster", _Boom())  # noqa: SLF001

    bundle, candidates_df, _, _, _, daily_df = _make_inputs(n=12)
    daily_basic_df = pd.DataFrame()
    moneyflow_df = pd.DataFrame()
    kwargs = _build_kwargs(candidates_df, daily_df, daily_basic_df, moneyflow_df)
    _attach_lgb_scores(bundle, scorer=scorer, **kwargs)

    # model_id is still set (scorer reported "loaded") but every score is None
    # because predict raised → audit payload is empty.
    assert all(c["lgb_score"] is None for c in bundle.candidates)
    assert bundle.lgb_predictions == []


def test_runparams_lgb_enabled_default_true() -> None:
    """Sanity: omitting --no-lgb keeps the legacy default behavior (lgb_enabled=True)."""
    from limit_up_board.runner import RunParams

    rp = RunParams()
    assert rp.lgb_enabled is True


@pytest.mark.slow
def test_attach_scores_fewer_than_ten_candidates_decile_none(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    """When batch is < 10, decile is None per design §3.1."""
    _persist_active_model(lgb_db, model_id="20260530_1_small")
    scorer = LgbScorer(lgb_db)
    bundle, candidates_df, _, _, _, daily_df = _make_inputs(n=5)
    daily_basic_df = pd.DataFrame()
    moneyflow_df = pd.DataFrame()
    kwargs = _build_kwargs(candidates_df, daily_df, daily_basic_df, moneyflow_df)
    _attach_lgb_scores(bundle, scorer=scorer, **kwargs)

    assert all(c["lgb_score"] is not None for c in bundle.candidates)
    assert all(c["lgb_decile"] is None for c in bundle.candidates)


@pytest.mark.slow
def test_record_predictions_writes_table(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    """End-to-end: attach scores → record predictions → row count > 0."""
    from limit_up_board.lgb.audit import record_predictions

    _persist_active_model(lgb_db, model_id="20260530_1_audit")
    scorer = LgbScorer(lgb_db)
    bundle, candidates_df, _, _, _, daily_df = _make_inputs(n=12)
    daily_basic_df = pd.DataFrame()
    moneyflow_df = pd.DataFrame()
    kwargs = _build_kwargs(candidates_df, daily_df, daily_basic_df, moneyflow_df)
    _attach_lgb_scores(bundle, scorer=scorer, **kwargs)

    n = record_predictions(
        lgb_db,
        run_id="00000000-0000-0000-0000-000000000aaa",
        trade_date=bundle.trade_date,
        model_id=bundle.lgb_model_id or "missing",
        rows=bundle.lgb_predictions,
    )
    assert n == 12
    rows = lgb_db.fetchall(
        "SELECT COUNT(*) FROM lub_lgb_predictions WHERE run_id = ?",
        ("00000000-0000-0000-0000-000000000aaa",),
    )
    assert rows[0][0] == 12
