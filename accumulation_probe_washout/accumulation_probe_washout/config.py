"""ApwConfig — strategy parameters with defaults + apw_config table bridge.

All parameters live in this dataclass so unit tests can construct them without
touching the database. ``ApwConfigStore`` is the thin DB wrapper used at runtime
(M6 wires settings show/set on top of it; M2 only needs in-memory defaults).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

# Default values mirror the source spec §6.2 + §8 of the design doc. Tweaking
# them without bumping the migration is fine; persistent override goes through
# apw_config.


@dataclass
class ApwConfig:
    # ---- universe / liquidity ----
    listed_days_min: int = 120
    min_amount_yi: float = 1.0
    min_circ_mv_yi: float = 20.0
    max_circ_mv_yi: float = 1500.0

    # ---- lookback windows (trade days) ----
    base_lookback_trade_days: int = 120
    probe_lookback_trade_days: int = 40
    accumulation_lookback_trade_days: int = 60
    accumulation_moneyflow_days: int = 20

    washout_min_trade_days: int = 3
    washout_max_trade_days: int = 25

    # ---- accumulation ----
    accumulation_score_min: float = 55.0

    # ---- probe day ----
    probe_volume_ratio_5d_min: float = 2.5
    probe_volume_ratio_20d_min: float = 2.0
    probe_volume_rank_pct_60d_min: float = 90.0
    probe_turnover_rate_min: float = 2.0
    probe_amplitude_pct_min: float = 5.0
    probe_quality_score_min: float = 60.0

    # ---- washout ----
    max_post_probe_drawdown_pct: float = 15.0
    post_probe_volume_shrink_ratio_max: float = 0.8
    washout_score_min: float = 55.0

    # ---- launch ----
    launch_setup_score_min: float = 55.0
    launch_current_volume_ratio_5d_min: float = 1.2
    # Last N trade days used to compute current_moneyflow_net_yi (亿元) — fed
    # into capital_score in compute_launch_setup. 3 ≈ current + last 2 sessions,
    # which damps single-day spike noise while staying close to "now".
    launch_moneyflow_days: int = 3
    # Baseline index for relative_strength_20d (HS300 by default; CSI500 =
    # 000905.SH is the other common choice). Empty string disables the index
    # adjustment and falls back to raw 20-day stock return.
    baseline_index_code: str = "000300.SH"

    # ---- batching / LLM ----
    max_llm_candidates: int = 80
    llm_batch_size: int = 20
    llm_max_repair_retries: int = 2

    # ---- evaluate ----
    evaluate_default_horizons: str = "1,3,5,10"
    label_t5_high_return_pct: float = 8.0
    label_t5_max_drawdown_pct: float = 8.0
    label_t10_high_return_pct: float = 12.0
    label_t10_max_drawdown_pct: float = 10.0

    # ---- prune (v0.3.0 — phase-aware watchlist cleanup) ----
    # Defaults reflect the design table in §3.1.5 of the migration plan.
    prune_idle_days_launch_ready: int = 5
    prune_drop_on_probe_low_break: bool = True
    prune_drop_on_ma60_break: bool = True
    prune_dry_run_default: bool = False

    # ---- v0.4.0 — extended feature engineering ----
    # adj_factor handling for probe / current volume ratios. When True the
    # screen pipeline will (in a follow-up PR) divide vol by adj_factor so
    # corporate-action splits / dividends don't break long-window ratios.
    # PR-2 ships the config knob; the actual ratio rewrite ships with PR-3.
    volume_adjust_enabled: bool = True
    # Index code used for alpha_*_pct features. Defaults to the same baseline
    # that ``relative_strength_20d`` uses so the two stay consistent.
    alpha_baseline_index_code: str = "000300.SH"


# Keys ApwConfigStore exposes via settings show / set. Locked here so we can
# reject unknown keys at write time (PreconditionError surface in M6).
ALLOWED_KEYS: tuple[str, ...] = tuple(f.name for f in fields(ApwConfig))


def to_dict(cfg: ApwConfig) -> dict[str, Any]:
    return asdict(cfg)


def from_dict(values: dict[str, Any]) -> ApwConfig:
    """Build a config, ignoring unknown keys (forward compat)."""
    known = {f.name for f in fields(ApwConfig)}
    safe = {k: v for k, v in values.items() if k in known}
    return ApwConfig(**safe)


class ApwConfigStore:
    """Thin wrapper around apw_config table.

    Each row is (key, value_json, updated_at). Unknown keys are rejected so we
    don't silently lose user input.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    def load(self) -> ApwConfig:
        rows = self.db.fetchall("SELECT key, value_json FROM apw_config")
        values: dict[str, Any] = {}
        for row in rows or []:
            key = row[0]
            try:
                values[key] = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                continue
        return from_dict(values)

    def set(self, key: str, value: Any) -> None:
        if key not in ALLOWED_KEYS:
            raise ValueError(f"unknown apw_config key: {key!r}")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        payload = json.dumps(value, ensure_ascii=False)
        # DuckDB upsert via DELETE + INSERT (avoid driver-specific ON CONFLICT).
        self.db.execute("DELETE FROM apw_config WHERE key = ?", [key])
        self.db.execute(
            "INSERT INTO apw_config (key, value_json, updated_at) VALUES (?, ?, ?)",
            [key, payload, now],
        )

    def items(self) -> list[tuple[str, Any]]:
        rows = self.db.fetchall("SELECT key, value_json FROM apw_config ORDER BY key")
        out: list[tuple[str, Any]] = []
        for row in rows or []:
            try:
                out.append((row[0], json.loads(row[1])))
            except (json.JSONDecodeError, TypeError):
                continue
        return out
