"""P1-F: ``select_finalists`` explicit tie-breaker tests."""

from __future__ import annotations

from limit_up_board.pipeline import select_finalists
from limit_up_board.schemas import ContinuationCandidate, EvidenceItemStrict


def _mk(
    *,
    ts_code: str,
    name: str = "Stock",
    rank: int = 1,
    score: float = 50.0,
    prediction: str = "top_candidate",
) -> ContinuationCandidate:
    return ContinuationCandidate(
        candidate_id=ts_code,
        ts_code=ts_code,
        name=name,
        rank=rank,
        continuation_score=score,
        confidence="medium",
        prediction=prediction,  # type: ignore[arg-type]
        rationale="r",
        key_evidence=[EvidenceItemStrict(field="x", value="1", unit="none", interpretation="ok")],
        next_day_watch_points=["w"],
        failure_triggers=["f"],
        missing_data=[],
    )


def test_select_finalists_tie_breaker_by_rank() -> None:
    """Same continuation_score → lower rank wins."""
    preds = [
        _mk(ts_code="300001.SZ", rank=3, score=70.0),
        _mk(ts_code="000001.SZ", rank=1, score=70.0),
        _mk(ts_code="600000.SH", rank=2, score=70.0),
    ]
    out = select_finalists(preds)
    assert [c.ts_code for c in out] == ["000001.SZ", "600000.SH", "300001.SZ"]


def test_select_finalists_tie_breaker_by_ts_code_when_rank_also_tied() -> None:
    """Same score AND same rank → ts_code asc settles it."""
    preds = [
        _mk(ts_code="600000.SH", rank=1, score=70.0),
        _mk(ts_code="000001.SZ", rank=1, score=70.0),
        _mk(ts_code="300001.SZ", rank=1, score=70.0),
    ]
    out = select_finalists(preds)
    assert [c.ts_code for c in out] == ["000001.SZ", "300001.SZ", "600000.SH"]


def test_select_finalists_invariant_to_input_shuffle() -> None:
    base = [
        _mk(ts_code="000001.SZ", rank=1, score=80.0),
        _mk(ts_code="600000.SH", rank=2, score=80.0),
        _mk(ts_code="300001.SZ", rank=3, score=70.0),
        _mk(ts_code="002001.SZ", rank=4, score=60.0, prediction="avoid"),
    ]
    a = select_finalists(base)
    b = select_finalists(list(reversed(base)))
    assert [c.ts_code for c in a] == [c.ts_code for c in b]


def test_select_finalists_primary_score_desc_preserved() -> None:
    """Tie-breaker must not break score-desc primary ordering."""
    preds = [
        _mk(ts_code="000001.SZ", rank=99, score=50.0),  # low score, high rank
        _mk(ts_code="600000.SH", rank=1, score=90.0),   # high score wins
    ]
    out = select_finalists(preds)
    assert out[0].ts_code == "600000.SH"
    assert out[1].ts_code == "000001.SZ"


def test_select_finalists_avoid_pool_also_tie_broken() -> None:
    preds = [
        _mk(ts_code="300001.SZ", rank=2, score=30.0, prediction="avoid"),
        _mk(ts_code="000001.SZ", rank=1, score=30.0, prediction="avoid"),
        # Need at least 5 predictions so avoid sampling cap (len//5) returns ≥1
        _mk(ts_code="600000.SH", rank=1, score=80.0),
        _mk(ts_code="600001.SH", rank=2, score=80.0),
        _mk(ts_code="600002.SH", rank=3, score=80.0),
    ]
    out = select_finalists(preds)
    # First three are top_candidates; the avoid sample comes next, sorted by rank
    avoid_samples = [c for c in out if c.prediction == "avoid"]
    assert avoid_samples and avoid_samples[0].ts_code == "000001.SZ"
