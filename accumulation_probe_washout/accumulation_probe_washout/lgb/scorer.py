"""Online LGB scoring with strict 5-branch fallback.

Contract (mirrors VA / limit-up-board's scorer.py):
    branch 1 — no active model in apw_lgb_models       → lgb_score=None batch-wide
    branch 2 — booster file missing on disk            → lgb_score=None batch-wide
    branch 3 — feature schema mismatch                 → lgb_score=None batch-wide
    branch 4 — Booster.predict raises (per-row)        → lgb_score=None for that row
    branch 5 — lightgbm ImportError                    → lgb_score=None batch-wide

Every branch emits a structured ``LGB_DEGRADE_*`` event but **never**
aborts the run — LLM analysis proceeds with NaN lgb_score / lgb_decile.

The scorer is lazily constructed in :func:`build_lgb_scorer` so the
``lightgbm`` import (~150ms cold) and booster load only happen for runs
that actually need scoring.
"""

from __future__ import annotations

import json
import logging
import math
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .features import FEATURE_NAMES, SCHEMA_VERSION, build_feature_frame, feature_hash
from . import registry as _registry

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-row + per-batch result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CandidateScore:
    """Per-candidate LGB output appended to the candidate dict."""

    ts_code: str
    lgb_score: float | None  # 0-100; None on any degrade branch
    lgb_decile: int | None   # 1..10 across the *current batch*; None when score is None
    feature_hash: str
    feature_missing: list[str] = field(default_factory=list)


@dataclass
class ScoreOutcome:
    """Returned by :meth:`LgbScorer.score_batch`."""

    scores: list[CandidateScore]
    degrade_reason: str | None = None   # human-readable; None on success
    model_id: str | None = None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class LgbScorer:
    """Lazy booster holder + batch scoring entrypoint.

    Construct via :func:`build_lgb_scorer`; the constructor itself never
    raises so the runner can always set ``ApwRuntime.lgb_scorer`` to a
    non-None value when LGB is configured-on. Degrade branches surface at
    ``score_batch`` time.
    """

    def __init__(
        self,
        *,
        booster: Any | None,
        model_record: _registry.ModelRecord | None,
        degrade_reason: str | None,
    ) -> None:
        self._booster = booster
        self._model_record = model_record
        self._initial_degrade = degrade_reason

    @property
    def model_record(self) -> _registry.ModelRecord | None:
        return self._model_record

    @property
    def model_id(self) -> str | None:
        return None if self._model_record is None else self._model_record.model_id

    @property
    def initial_degrade(self) -> str | None:
        return self._initial_degrade

    def score_batch(
        self, candidates: list[dict[str, Any]]
    ) -> ScoreOutcome:
        """Score a batch of candidate dicts.

        ``candidates`` are the same dicts produced by ``pack_candidate`` (so
        they carry the v0.4.0 feature keys). Returns one
        :class:`CandidateScore` per input row, **in input order**.
        """
        # Empty batch is degenerate, not a degrade — no scoring to do.
        if not candidates:
            return ScoreOutcome(
                scores=[], degrade_reason=None, model_id=self.model_id
            )
        empty_scores = [
            CandidateScore(
                ts_code=str(c.get("ts_code", "")),
                lgb_score=None,
                lgb_decile=None,
                feature_hash=feature_hash(c),
            )
            for c in candidates
        ]
        if self._initial_degrade is not None:
            return ScoreOutcome(
                scores=empty_scores,
                degrade_reason=self._initial_degrade,
                model_id=self.model_id,
            )
        if self._booster is None:
            return ScoreOutcome(
                scores=empty_scores,
                degrade_reason="LGB_DEGRADE_NO_BOOSTER",
                model_id=self.model_id,
            )

        try:
            features = build_feature_frame(candidate_rows=candidates)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LGB feature build failed; degrading")
            return ScoreOutcome(
                scores=empty_scores,
                degrade_reason=f"LGB_DEGRADE_FEATURE_BUILD_FAIL: {exc}",
                model_id=self.model_id,
            )

        # Branch 3 — schema mismatch (catches drift introduced after model
        # training).
        if list(features.columns) != list(FEATURE_NAMES):
            return ScoreOutcome(
                scores=empty_scores,
                degrade_reason="LGB_DEGRADE_SCHEMA_MISMATCH",
                model_id=self.model_id,
            )

        # ---- Per-row predict with branch-4 isolation
        scores: list[CandidateScore] = []
        try:
            raw_probs = self._booster.predict(features.values)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.warning("LGB predict raised batch-wide: %s\n%s", exc, tb)
            return ScoreOutcome(
                scores=empty_scores,
                degrade_reason=f"LGB_DEGRADE_PREDICT_FAIL: {exc}",
                model_id=self.model_id,
            )

        probs = pd.Series(raw_probs, index=features.index)
        valid = probs.dropna()
        if len(valid) >= 10:
            quantiles = valid.quantile([i / 10.0 for i in range(1, 10)]).tolist()
        else:
            quantiles = []

        for cand in candidates:
            ts = str(cand.get("ts_code", ""))
            fh = feature_hash(cand)
            if ts not in probs.index or pd.isna(probs[ts]):
                scores.append(
                    CandidateScore(
                        ts_code=ts, lgb_score=None, lgb_decile=None,
                        feature_hash=fh,
                    )
                )
                continue
            raw = float(probs[ts])
            score_100 = round(raw * 100.0, 2)
            decile = _decile_of(raw, quantiles) if quantiles else None
            # Build missing-feature list for audit (cheap; per-row).
            feat_row = features.loc[ts]
            missing = [
                name
                for name in FEATURE_NAMES
                if pd.isna(feat_row.get(name, math.nan))
            ]
            scores.append(
                CandidateScore(
                    ts_code=ts,
                    lgb_score=score_100,
                    lgb_decile=decile,
                    feature_hash=fh,
                    feature_missing=missing,
                )
            )

        return ScoreOutcome(
            scores=scores, degrade_reason=None, model_id=self.model_id
        )

    def persist_predictions(
        self,
        db: Database,
        outcome: ScoreOutcome,
        *,
        run_id: str,
        trade_date: str,
    ) -> int:
        """Write the successful per-row scores to ``apw_lgb_predictions``.

        Degrade branches don't write predictions (apw_lgb_predictions.
        lgb_score is NOT NULL by schema). Returns the number of rows written.
        """
        if outcome.degrade_reason is not None or outcome.model_id is None:
            return 0
        n = 0
        for s in outcome.scores:
            if s.lgb_score is None:
                continue
            db.execute(
                "DELETE FROM apw_lgb_predictions WHERE run_id = ? AND ts_code = ?",
                (run_id, s.ts_code),
            )
            db.execute(
                """
                INSERT INTO apw_lgb_predictions
                (run_id, trade_date, ts_code, model_id, lgb_score,
                 lgb_decile, feature_hash, feature_missing_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, trade_date, s.ts_code, outcome.model_id,
                    float(s.lgb_score),
                    None if s.lgb_decile is None else int(s.lgb_decile),
                    s.feature_hash,
                    json.dumps(s.feature_missing, ensure_ascii=False),
                ),
            )
            n += 1
        return n


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_lgb_scorer(db: Database) -> LgbScorer:
    """Return an :class:`LgbScorer` parameterised by the active model.

    Branches 1 / 2 / 3 / 5 are resolved here so the runner only deals with
    a flat ``LgbScorer`` instance regardless of state. Branch 4 is per-row
    and surfaces at :meth:`LgbScorer.score_batch` time.
    """
    # ---- Branch 5 — lightgbm install missing
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError:
        logger.warning("lightgbm not installed; LGB scoring disabled run-wide")
        return LgbScorer(
            booster=None,
            model_record=None,
            degrade_reason="LGB_DEGRADE_NO_LIGHTGBM",
        )

    # ---- Branch 1 — no active model
    active = _registry.get_active(db)
    if active is None:
        return LgbScorer(
            booster=None,
            model_record=None,
            degrade_reason="LGB_DEGRADE_NO_MODEL",
        )

    # ---- Branch 2 — file missing
    booster_path = Path(active.file_path)
    if not booster_path.exists():
        return LgbScorer(
            booster=None,
            model_record=active,
            degrade_reason=f"LGB_DEGRADE_FILE_MISSING: {booster_path}",
        )

    # ---- Branch 3a — schema_version drift
    if active.schema_version != SCHEMA_VERSION:
        return LgbScorer(
            booster=None,
            model_record=active,
            degrade_reason=(
                f"LGB_DEGRADE_SCHEMA_MISMATCH: "
                f"model.schema_version={active.schema_version} "
                f"runtime={SCHEMA_VERSION}"
            ),
        )

    # ---- Load booster
    try:
        booster = lgb.Booster(model_file=str(booster_path))
    except Exception as exc:  # noqa: BLE001
        return LgbScorer(
            booster=None,
            model_record=active,
            degrade_reason=f"LGB_DEGRADE_LOAD_FAIL: {exc}",
        )

    # ---- Branch 3b — feature list drift between FEATURE_NAMES + model file
    try:
        model_feats = json.loads(active.feature_list_json)
    except (json.JSONDecodeError, TypeError):
        model_feats = []
    if list(model_feats) != list(FEATURE_NAMES):
        return LgbScorer(
            booster=None,
            model_record=active,
            degrade_reason="LGB_DEGRADE_SCHEMA_MISMATCH: feature_list",
        )

    return LgbScorer(
        booster=booster,
        model_record=active,
        degrade_reason=None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decile_of(value: float, quantile_cutoffs: list[float]) -> int:
    """Return ``1..10`` based on cutoff list (length 9, e.g. q10, q20, …, q90)."""
    for i, q in enumerate(quantile_cutoffs, start=1):
        if value <= q:
            return i
    return 10
