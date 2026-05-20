"""Daily market regime classification for Checkmate.

Three signals drive the 4-state regime tag (development_plan §7 + iteration_tasks
§3 PR-2.2):

* ``index_csi_above_ma120`` — 中证全指 close > MA(120)
* ``index_hs300_above_ma120`` — 沪深300 close > MA(120)
* ``breadth_ma120`` — fraction of names in ``checkmate_features_daily`` for
  ``trade_date`` with ``close_qfq > ma120`` (the "breadth" signal)

Plus an optional ``breadth_limit_down_5d`` proxy (None when not supplied — PR-2.2
leaves the limit-down breadth to PR-2.3 ``scan`` orchestration, where the
universe rows are already in memory and a count is essentially free).

Decision rules (see :class:`RegimeConfig` docstring for the exact thresholds):

* ``risk``    — ``breadth_ma120 < cfg.breadth_risk`` (broad capitulation)
* ``strong``  — ``breadth_ma120 >= cfg.breadth_strong`` AND both indices > MA120
* ``neutral`` — ``breadth_ma120 >= cfg.breadth_weak``   AND ≥ 1 index > MA120
* ``weak``    — anything else

The 4 regimes map to a portfolio-level ``exposure_cap`` ∈ [0, 1] consumed by
the risk module (Iter-3).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from . import data
from .config import RegimeConfig
from .runtime import CheckmateRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class RegimeRow:
    trade_date: str
    regime: str
    exposure_cap: float
    breadth_ma120: float | None = None
    breadth_limit_down_5d: float | None = None
    index_csi_above_ma120: bool | None = None
    index_hs300_above_ma120: bool | None = None
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _index_above_ma(
    tushare: Any, index_code: str, trade_date: str, n: int,
) -> tuple[bool | None, float | None, float | None]:
    """Pull enough ``index_daily`` history to evaluate ``close > MA(n)`` at trade_date.

    Returns ``(above_flag, close, ma_n)``. Any of the three may be ``None``
    when the data is missing — callers degrade gracefully.
    """
    if tushare is None:
        return None, None, None
    # Pull ~2x n calendar-day buffer so trading-day gaps don't truncate.
    window_start = (
        datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=int(n * 1.7) + 30)
    ).strftime("%Y%m%d")
    try:
        df = data.fetch_index_daily(tushare, index_code, window_start, trade_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("index_daily fetch failed for %s: %s", index_code, exc)
        return None, None, None
    if df is None or df.empty or "close" not in df.columns:
        return None, None, None
    df = df.sort_values("trade_date")
    df = df[df["trade_date"].astype(str) <= trade_date]
    if len(df) < n:
        return None, None, None
    closes = df["close"].astype(float)
    ma = float(closes.tail(n).mean())
    last_close = float(closes.iloc[-1])
    if not (math.isfinite(ma) and math.isfinite(last_close)):
        return None, last_close, ma
    return last_close > ma, last_close, ma


# ---------------------------------------------------------------------------
# Breadth helpers
# ---------------------------------------------------------------------------


def _breadth_from_features(db: Any, trade_date: str) -> float | None:
    """Fraction of ``checkmate_features_daily`` rows with ``close_qfq > ma120``."""
    row = db.execute(
        """
        SELECT
            SUM(CASE WHEN close_qfq IS NOT NULL AND ma120 IS NOT NULL
                      AND close_qfq > ma120 THEN 1 ELSE 0 END) AS above_n,
            SUM(CASE WHEN ma120 IS NOT NULL THEN 1 ELSE 0 END) AS total_n
        FROM checkmate_features_daily
        WHERE trade_date = ?
        """,
        [trade_date],
    ).fetchone()
    if row is None:
        return None
    above_n, total_n = row
    if not total_n:
        return None
    return float(above_n) / float(total_n)


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def _decide(
    breadth_ma120: float | None,
    idx_csi: bool | None,
    idx_hs300: bool | None,
    cfg: RegimeConfig,
) -> tuple[str, float]:
    """Map the three signals to a (regime, exposure_cap) tuple.

    Missing breadth → defaults to 0.5 ("neutral breadth") so a thin features
    table (e.g. very small universe) doesn't trip the risk gate. Missing index
    flags fall through to ``weak`` (both bools default to False).
    """
    b = breadth_ma120 if breadth_ma120 is not None else 0.5
    csi = bool(idx_csi) if idx_csi is not None else False
    hs = bool(idx_hs300) if idx_hs300 is not None else False
    n_above = int(csi) + int(hs)

    if b < cfg.breadth_risk:
        return "risk", cfg.exposure_risk
    if b >= cfg.breadth_strong and n_above == 2:
        return "strong", cfg.exposure_strong
    if b >= cfg.breadth_weak and n_above >= 1:
        return "neutral", cfg.exposure_neutral
    return "weak", cfg.exposure_weak


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def classify_regime(
    rt: CheckmateRuntime,
    trade_date: str,
    cfg: RegimeConfig | None = None,
    *,
    breadth_limit_down_5d: float | None = None,
) -> RegimeRow:
    """Build a :class:`RegimeRow` for ``trade_date``.

    The optional ``breadth_limit_down_5d`` is a pass-through metric — the
    scan orchestrator (PR-2.3) populates it from in-memory universe rows.
    Iter-3+ may extend the decision logic to fold this signal in; for now it
    is recorded for explain/dashboard use only.
    """
    cfg = cfg or RegimeConfig()

    csi_above, csi_close, csi_ma = _index_above_ma(
        rt.tushare, cfg.index_csi_code, trade_date, cfg.ma_window,
    )
    hs_above, hs_close, hs_ma = _index_above_ma(
        rt.tushare, cfg.index_hs300_code, trade_date, cfg.ma_window,
    )
    breadth = _breadth_from_features(rt.db, trade_date)

    regime, exposure_cap = _decide(breadth, csi_above, hs_above, cfg)

    payload = {
        "config": {
            "ma_window": cfg.ma_window,
            "breadth_strong": cfg.breadth_strong,
            "breadth_weak": cfg.breadth_weak,
            "breadth_risk": cfg.breadth_risk,
        },
        "index_csi": {
            "code": cfg.index_csi_code,
            "close": csi_close,
            "ma120": csi_ma,
        },
        "index_hs300": {
            "code": cfg.index_hs300_code,
            "close": hs_close,
            "ma120": hs_ma,
        },
    }

    return RegimeRow(
        trade_date=trade_date,
        regime=regime,
        exposure_cap=exposure_cap,
        breadth_ma120=breadth,
        breadth_limit_down_5d=breadth_limit_down_5d,
        index_csi_above_ma120=csi_above,
        index_hs300_above_ma120=hs_above,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def upsert_regime_daily(db: Any, row: RegimeRow) -> int:
    """INSERT OR REPLACE one row into ``checkmate_regime_daily``."""
    db.execute(
        """
        INSERT OR REPLACE INTO checkmate_regime_daily
            (trade_date, regime, exposure_cap,
             breadth_ma120, breadth_limit_down_5d,
             index_csi_above_ma120, index_hs300_above_ma120,
             payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            row.trade_date, row.regime, row.exposure_cap,
            row.breadth_ma120, row.breadth_limit_down_5d,
            row.index_csi_above_ma120, row.index_hs300_above_ma120,
            json.dumps(row.payload, ensure_ascii=False, default=str),
        ],
    )
    return 1


# ---------------------------------------------------------------------------
# Optional helper for PR-2.3 — count down-limit days in the recent window
# from the universe's cached daily frames. Lives here so the scan orchestrator
# can compute it cheaply once we already have the eligible ts_code list.
# ---------------------------------------------------------------------------


def compute_breadth_limit_down_5d(
    rt: CheckmateRuntime,
    trade_date: str,
    ts_codes: list[str],
    *,
    window: int = 5,
    threshold: float = -0.097,
) -> float | None:
    """Average per-symbol count of down-limit days over the recent ``window``
    sessions, normalised to ``[0, 1]`` (1 = every symbol limit-down every day).

    Returns ``None`` if no symbol has cached daily data. The threshold default
    matches ``-9.7%`` (main-board down limit).
    """
    if not ts_codes:
        return None
    window_start = (
        datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=window * 3)
    ).strftime("%Y%m%d")
    total_days = 0
    limit_down_days = 0
    for ts_code in ts_codes:
        try:
            df = data.fetch_daily_raw(rt.tushare, ts_code, window_start, trade_date)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty or not {"close", "pre_close"}.issubset(df.columns):
            continue
        tail = df.tail(window)
        n = len(tail)
        if n == 0:
            continue
        pre = tail["pre_close"].astype(float).values
        cls = tail["close"].astype(float).values
        # Avoid division by zero.
        ok = pre > 0
        pct = (cls[ok] / pre[ok]) - 1.0
        limit_down_days += int((pct < threshold).sum())
        total_days += int(ok.sum())
    if total_days == 0:
        return None
    return round(limit_down_days / total_days, 6)
