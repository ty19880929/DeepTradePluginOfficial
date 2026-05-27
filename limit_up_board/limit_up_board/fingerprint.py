"""Plugin-local fingerprint helpers (P1-I).

This module gives limit-up-board a deterministic ``input_fingerprint`` for
every run, *before* the framework's canonical fingerprint helpers land in
``deeptrade.core.fingerprint``. The contract:

* ``canonical_json(obj)`` — JSON with sorted keys, no NaN, deterministic
  encoding for datetime/Decimal/dataclass-like objects.
* ``hash_json(obj)``      — sha256 hex of ``canonical_json(obj)``.
* ``hash_text(s)``        — sha256 hex of utf-8 encoded text.
* ``build_input_fingerprint(...)`` — composes the run-level payload used to
  detect "same input, same expected output" across consecutive runs.

When the framework eventually ships ``deeptrade.core.fingerprint``, the
top-of-file ``try / except ImportError`` will silently rebind the three
primitives so all callers stay on the framework version with no other
plugin edits required. ``build_input_fingerprint`` itself is always
plugin-owned because the payload is business-specific.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Primitives — try framework first, fall back to local implementation.
# ---------------------------------------------------------------------------


def _normalize(obj: Any) -> Any:
    """Recursively normalize values into the subset the framework's strict
    ``canonical_json`` accepts, preserving this module's documented contract
    (``no NaN`` + deterministic set ordering).

    The framework's ``canonical_json`` (deeptrade-quant ≥ 0.5) deliberately
    **raises** on ``NaN``/``Inf`` (``ValueError``) and on ``set``/``frozenset``
    (``TypeError``) rather than coercing them — see its module docstring. But
    ``build_input_fingerprint`` feeds it Tushare-derived floats (which can be
    NaN) and occasionally set-valued fields, so without pre-normalization the
    run-level fingerprint would crash at runtime. We therefore:

    * NaN/Inf → ``None`` (incl. numpy NaN — ``np.float64`` is a ``float`` subclass);
    * ``set``/``frozenset`` → list sorted by canonical JSON of each element;
    * dataclasses → ``asdict``; tuples → lists; Mapping keys → ``str``.

    Rich scalars the framework handles natively (``datetime`` / ``Decimal`` /
    numpy scalars / ``BaseModel``) are passed through unchanged so the framework
    can encode them — we only sand off the values it refuses to encode.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):  # numpy float64 is a float subclass → covered
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        # Sets have no intrinsic order; sort for determinism.
        return sorted(
            (_normalize(v) for v in obj),
            key=lambda x: json.dumps(x, sort_keys=True, default=str),
        )
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _normalize(dataclasses.asdict(obj))
    return obj


try:
    from deeptrade.core.fingerprint import (  # type: ignore[import-not-found]
        canonical_json as _framework_canonical_json,
        hash_text,
    )

    def canonical_json(obj: Any) -> str:
        # Normalize first (NaN/Inf/sets), then delegate to the framework so its
        # rich datetime/Decimal/numpy/BaseModel handling + frozen versioned
        # contract remain the source of truth for everything else.
        return _framework_canonical_json(_normalize(obj))

    def hash_json(obj: Any) -> str:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()

except ImportError:  # pragma: no cover branch — local fallback when framework absent

    def canonical_json(obj: Any) -> str:
        return json.dumps(
            _normalize(obj),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )

    def hash_json(obj: Any) -> str:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()

    def hash_text(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Plugin-specific payload composition
# ---------------------------------------------------------------------------

# Candidate fields that enter the fingerprint. Allowlist (not blocklist) so
# new debug/display fields don't accidentally invalidate caches. Bump
# LLM_SCHEMA_VERSION when this list changes.
_CANDIDATE_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "ts_code",
    "name",
    "industry",
    "limit_times",
    "first_time",
    "last_time",
    "open_times",
    "fd_amount",
    "close",
    "pre_close",
    "pct_chg",
    "volume_ratio",
    "turnover_rate_f",
    "amount",
    "circ_mv",
    "float_mv_yi",
    "lgb_score",
    "lgb_decile",
    "lgb_feature_missing",
)


def _candidate_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Project a single candidate row to the fingerprint allowlist."""
    return {k: candidate.get(k) for k in _CANDIDATE_FINGERPRINT_FIELDS}


def build_input_fingerprint(
    *,
    trade_date: str,
    next_trade_date: str,
    daily_lookback: int,
    moneyflow_lookback: int,
    lub_config: Any,
    bundle: Any,
    stage_profiles: Mapping[str, Any],
    llm_schema_version: str,
    prompt_template_version: str,
) -> tuple[str, dict[str, Any]]:
    """Compute the run-level ``(sha256_hex, payload_dict)``.

    Parameters mirror what the runner has on hand at the end of Step 1:

    * ``trade_date`` / ``next_trade_date``  — resolved T / T+1.
    * ``daily_lookback`` / ``moneyflow_lookback`` — RunParams windowing.
    * ``lub_config`` — :class:`LubConfig` dataclass (full field set).
    * ``bundle`` — :class:`Round1Bundle` (candidates, market_summary,
      sector_strength, data_unavailable, lgb_model_id).
    * ``stage_profiles`` — ``{stage: StageProfile}`` for SCREENING /
      PREDICTION / FINAL / REVISION; profile changes invalidate cache.
    * ``llm_schema_version`` / ``prompt_template_version`` — version
      sentinels from ``profiles.py``.

    The payload is canonical-JSONed and hashed; the dict is returned too so
    the renderer / audit layer can surface what went into the hash.
    """
    cfg_payload = (
        dataclasses.asdict(lub_config)
        if dataclasses.is_dataclass(lub_config) and not isinstance(lub_config, type)
        else dict(getattr(lub_config, "__dict__", {}))
    )

    candidates_sorted = sorted(
        (getattr(bundle, "candidates", []) or []),
        key=lambda c: str(c.get("ts_code", "")),
    )

    sector_strength_obj = getattr(bundle, "sector_strength", None)
    if dataclasses.is_dataclass(sector_strength_obj) and not isinstance(sector_strength_obj, type):
        sector_strength_payload: Any = dataclasses.asdict(sector_strength_obj)
    elif sector_strength_obj is None:
        sector_strength_payload = None
    else:
        sector_strength_payload = sector_strength_obj

    payload: dict[str, Any] = {
        "schema": "lub-input-fingerprint-v1",
        "llm_schema_version": llm_schema_version,
        "prompt_template_version": prompt_template_version,
        "trade_date": trade_date,
        "next_trade_date": next_trade_date,
        "daily_lookback": int(daily_lookback),
        "moneyflow_lookback": int(moneyflow_lookback),
        "config": cfg_payload,
        "candidates": [_candidate_view(c) for c in candidates_sorted],
        "market_summary": dict(getattr(bundle, "market_summary", {}) or {}),
        "sector_strength": sector_strength_payload,
        "data_unavailable": sorted(getattr(bundle, "data_unavailable", []) or []),
        "lgb_model_id": getattr(bundle, "lgb_model_id", None),
        "stage_profiles": {
            stage: (
                dataclasses.asdict(prof)
                if dataclasses.is_dataclass(prof) and not isinstance(prof, type)
                else dict(getattr(prof, "__dict__", {}))
            )
            for stage, prof in sorted(stage_profiles.items())
        },
    }
    return hash_json(payload), payload


__all__ = [
    "canonical_json",
    "hash_json",
    "hash_text",
    "build_input_fingerprint",
]
