"""``scan`` orchestrator — single-day end-to-end pipeline.

Pipeline (development_plan §4 + iteration_tasks.md §3 PR-2.3):

    universe → features → regime → score

Side effects per scan run:
  * INSERT OR REPLACE into ``checkmate_universe_daily`` (one row per ts_code)
  * INSERT OR REPLACE into ``checkmate_features_daily`` (one row per eligible ts_code)
  * INSERT OR REPLACE into ``checkmate_regime_daily``   (one row for trade_date)
  * INSERT into ``checkmate_runs`` (one row, mode='scan')
  * INSERT into ``checkmate_events`` (N rows, one per pipeline step)

A repeat run for the same ``(trade_date, ts_code)`` is safe: the three daily
tables use ``INSERT OR REPLACE`` keyed on their PK; ``checkmate_runs`` /
``checkmate_events`` get a fresh ``run_id`` every invocation so history is
never lost.

The CLI invocation in ``cli.cmd_scan`` is a thin shell over :func:`run_scan`.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .calendar import load_trade_calendar
from .config import FeaturesConfig, RegimeConfig, UniverseConfig
from .features import compute_features_frame, upsert_features_daily
from .regime import classify_regime, compute_breadth_limit_down_5d, upsert_regime_daily
from .runtime import CheckmateRuntime
from .ui import EventRenderer, LegacyStreamRenderer, RenderEvent
from .universe import build_universe, upsert_universe_daily

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class ScanParams:
    trade_date: str | None = None  # YYYYMMDD or YYYY-MM-DD; default = prev_session(today)
    quiet: bool = False
    universe_cfg: UniverseConfig | None = None
    features_cfg: FeaturesConfig | None = None
    regime_cfg: RegimeConfig | None = None


@dataclass
class ScanOutcome:
    run_id: str
    trade_date: str
    n_universe: int
    n_eligible: int
    n_features: int
    regime: str
    exposure_cap: float
    started_at: datetime
    finished_at: datetime
    status: str = "success"
    error: str | None = None
    # In-memory artefacts — useful for the CLI's stdout summary; tests inspect
    # these without re-reading the DB.
    top_scored: list[dict[str, Any]] = field(default_factory=list)
    reason_breakdown: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event helpers — minimal, no dependency on the framework's StrategyEvent
# (Iter-5 will swap this for the EventRenderer protocol).
# ---------------------------------------------------------------------------


class _Events:
    """Per-run event sink: appends to ``checkmate_events`` + dispatches to renderer.

    The renderer is the v0.2.0 abstraction (see :mod:`checkmate.ui.protocol`);
    pass any :class:`EventRenderer` (legacy stream, dashboard, null, or a
    test capture). DB persistence failures only log; renderer failures are
    swallowed so a broken UI never crashes a run.
    """

    def __init__(self, db: Any, run_id: str, renderer: EventRenderer) -> None:
        self.db = db
        self.run_id = run_id
        self.renderer = renderer
        self._seq = 0

    def emit(
        self,
        event_type: str,
        message: str,
        *,
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._seq += 1
        try:
            self.db.execute(
                """
                INSERT INTO checkmate_events
                    (run_id, seq, event_time, level, event_type, message, payload_json)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
                """,
                [
                    self.run_id, self._seq, level, event_type, message,
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                ],
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed to persist event %s", event_type)
        try:
            self.renderer.on_event(RenderEvent(
                type=event_type, message=message, level=level, payload=payload or {},
            ))
        except Exception:  # noqa: BLE001
            logger.exception("renderer on_event raised for %s", event_type)


# ---------------------------------------------------------------------------
# Trade-date resolution
# ---------------------------------------------------------------------------


def _resolve_trade_date(rt: CheckmateRuntime, user_specified: str | None) -> str:
    if user_specified:
        return user_specified.replace("-", "")
    cal = load_trade_calendar(rt.tushare)
    today = datetime.now().strftime("%Y%m%d")
    if cal.is_trade_day(today):
        return today
    return cal.prev_session(today)


# ---------------------------------------------------------------------------
# Run row helpers (one-shot, no events column overhead)
# ---------------------------------------------------------------------------


def _write_run_row_start(
    db: Any, run_id: str, trade_date: str, params: ScanParams,
) -> None:
    db.execute(
        """
        INSERT INTO checkmate_runs
            (run_id, mode, trade_date, status, started_at, params_json)
        VALUES (?, 'scan', ?, 'running', CURRENT_TIMESTAMP, ?)
        """,
        [
            run_id, trade_date,
            json.dumps({"quiet": params.quiet}, ensure_ascii=False),
        ],
    )


def _write_run_row_finish(
    db: Any, run_id: str, status: str, exit_code: int,
    summary: dict[str, Any], error: str | None,
) -> None:
    db.execute(
        """
        UPDATE checkmate_runs
        SET status=?, exit_code=?, finished_at=CURRENT_TIMESTAMP,
            summary_json=?, error=?
        WHERE run_id=?
        """,
        [
            status, exit_code,
            json.dumps(summary, ensure_ascii=False, default=str),
            error, run_id,
        ],
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_scan(
    rt: CheckmateRuntime,
    params: ScanParams,
    *,
    renderer: EventRenderer | None = None,
    echo: Callable[[str], None] | None = None,
) -> ScanOutcome:
    """Execute the scan pipeline for one trade_date. See module docstring.

    ``renderer`` is the preferred sink (v0.2.0+). ``echo`` is kept as a
    back-compat shim — if provided without a renderer, the orchestrator
    wraps it in a :class:`LegacyStreamRenderer` whose sink is ``echo``.
    Tests that pass ``echo=log.append`` continue to work; new code should
    pass ``renderer`` directly.
    """
    if renderer is None:
        renderer = LegacyStreamRenderer(sink=echo) if echo is not None else LegacyStreamRenderer()
    started_at = datetime.now()
    run_id = str(uuid.uuid4())
    rt.run_id = run_id  # propagate so future emit/audit calls can pick it up

    trade_date = _resolve_trade_date(rt, params.trade_date)
    _write_run_row_start(rt.db, run_id, trade_date, params)

    renderer.on_run_start(run_id=run_id, mode="scan", params=params)
    events = _Events(rt.db, run_id, renderer)
    events.emit("RUN_STARTED", f"scan run started for {trade_date}",
                payload={"run_id": run_id, "trade_date": trade_date})

    try:
        # ---- Step 0: universe
        events.emit("STEP_STARTED", "Step 0: building universe")
        universe_cfg = params.universe_cfg or UniverseConfig()
        snap = build_universe(rt, trade_date, universe_cfg)
        upsert_universe_daily(rt.db, snap)
        reason_breakdown = snap.reason_breakdown()
        events.emit(
            "STEP_FINISHED",
            f"Step 0 done — total={len(snap.rows)} eligible={len(snap.eligible)} "
            f"excluded={len(snap.excluded)}",
            payload={
                "total": len(snap.rows),
                "eligible": len(snap.eligible),
                "excluded": len(snap.excluded),
                "reason_breakdown": reason_breakdown,
            },
        )

        # ---- Step 1: features (eligible only — non-eligible names don't
        # benefit from feature computation, and skipping them avoids
        # ~30% of the per-day Tushare / parquet round-trips).
        events.emit("STEP_STARTED", "Step 1: computing features")
        features_cfg = params.features_cfg or FeaturesConfig()
        eligible_codes = [r.ts_code for r in snap.eligible]
        if eligible_codes:
            _, feat_rows = compute_features_frame(rt, trade_date, eligible_codes, features_cfg)
            n_features = upsert_features_daily(rt.db, feat_rows)
        else:
            feat_rows = []
            n_features = 0
        events.emit(
            "STEP_FINISHED",
            f"Step 1 done — features_rows={n_features}",
            payload={"features_rows": n_features},
        )

        # ---- Step 2: regime
        events.emit("STEP_STARTED", "Step 2: classifying regime")
        regime_cfg = params.regime_cfg or RegimeConfig()
        breadth_ld5 = compute_breadth_limit_down_5d(
            rt, trade_date, [r.ts_code for r in snap.rows],
        )
        regime_row = classify_regime(
            rt, trade_date, regime_cfg, breadth_limit_down_5d=breadth_ld5,
        )
        upsert_regime_daily(rt.db, regime_row)
        events.emit(
            "STEP_FINISHED",
            f"Step 2 done — regime={regime_row.regime} exposure_cap={regime_row.exposure_cap}",
            payload={
                "regime": regime_row.regime,
                "exposure_cap": regime_row.exposure_cap,
                "breadth_ma120": regime_row.breadth_ma120,
                "breadth_limit_down_5d": regime_row.breadth_limit_down_5d,
            },
        )

        # ---- Top-scored snapshot for the CLI's stdout summary
        top_scored = sorted(
            (r for r in feat_rows if r.score is not None),
            key=lambda r: r.score or 0.0,
            reverse=True,
        )[:30]
        top_payload = [
            {
                "ts_code": r.ts_code, "score": r.score,
                "close_qfq": r.close_qfq, "ret_60": r.ret_60,
                "rs60_pctile": r.rs60_pctile,
            }
            for r in top_scored
        ]

        outcome = ScanOutcome(
            run_id=run_id, trade_date=trade_date,
            n_universe=len(snap.rows), n_eligible=len(snap.eligible),
            n_features=n_features,
            regime=regime_row.regime, exposure_cap=regime_row.exposure_cap,
            started_at=started_at, finished_at=datetime.now(),
            status="success", error=None,
            top_scored=top_payload,
            reason_breakdown=reason_breakdown,
        )

        events.emit("RUN_FINISHED", f"scan run finished — {len(snap.eligible)} eligible",
                    payload={"n_eligible": len(snap.eligible), "regime": regime_row.regime})
        _write_run_row_finish(
            rt.db, run_id, status="success", exit_code=0,
            summary={
                "n_universe": outcome.n_universe,
                "n_eligible": outcome.n_eligible,
                "n_features": outcome.n_features,
                "regime": outcome.regime,
                "exposure_cap": outcome.exposure_cap,
                "reason_breakdown": reason_breakdown,
            },
            error=None,
        )
        try:
            renderer.on_run_finish(outcome)
        except Exception:  # noqa: BLE001
            logger.exception("renderer on_run_finish raised")
        return outcome
    except Exception as exc:
        events.emit("RUN_FAILED", f"scan crashed: {exc}", level="error",
                    payload={"error": str(exc)})
        _write_run_row_finish(
            rt.db, run_id, status="failed", exit_code=1,
            summary={}, error=str(exc),
        )
        raise
    finally:
        try:
            renderer.close()
        except Exception:  # noqa: BLE001
            logger.exception("renderer close raised")
