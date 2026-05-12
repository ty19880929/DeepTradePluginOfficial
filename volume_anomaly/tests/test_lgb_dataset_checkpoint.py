"""PR-1.2 — checkpoint fingerprint + resume semantics.

Lower-level than test_lgb_dataset.py: focuses on the on-disk artifacts that
back Phase-1 resume.

Coverage:
    * fingerprint digest is stable for identical inputs
    * fingerprint digest differs when any field changes
    * save_day_shard ↔ load_day_shard round-trip preserves SHARD_COLUMNS
    * completed_dates reads disk truth (independent of state.json)
    * open_or_create rejects a mismatched fingerprint under the same digest dir
      (CheckpointMismatch path)
    * collect_training_window with checkpoint_resume=True picks up where it
      crashed (simulated mid-window)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from deeptrade.core.db import Database
from volume_anomaly.calendar import TradeCalendar
from volume_anomaly.lgb import paths as lgb_paths
from volume_anomaly.lgb.checkpoint import (
    META_COLUMNS,
    SHARD_COLUMNS,
    CheckpointFingerprint,
    CheckpointMismatch,
    CheckpointState,
    checkpoint_dir,
    completed_dates,
    delete_checkpoint,
    load_day_shard,
    load_state,
    open_or_create,
    record_day_done,
    save_day_shard,
    save_state,
    shard_path,
    state_path,
)
from volume_anomaly.lgb.dataset import collect_training_window
from volume_anomaly.lgb.features import FEATURE_NAMES, SCHEMA_VERSION

# Shared fixtures live in conftest.py at the top of tests/.
from conftest import FakeTushareClient, _build_fixtures, trade_cal_df  # type: ignore[import-not-found]


def _fp(**overrides: Any) -> CheckpointFingerprint:
    base = {
        "start_date": "20260601",
        "end_date": "20260612",
        "schema_version": SCHEMA_VERSION,
        "label_threshold_pct": 5.0,
        "label_source": "max_ret_5d",
        "daily_lookback": 5,
        "moneyflow_lookback": 5,
        "main_board_only": True,
        "baseline_index_code": "000300.SH",
    }
    base.update(overrides)
    return CheckpointFingerprint(**base)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_digest_stable_for_same_inputs() -> None:
    a = _fp().digest()
    b = _fp().digest()
    assert a == b
    assert len(a) == 12


@pytest.mark.parametrize(
    "field, new_value",
    [
        ("start_date", "20260101"),
        ("end_date", "20260701"),
        ("schema_version", SCHEMA_VERSION + 1),
        ("label_threshold_pct", 8.0),
        ("label_source", "ret_t3"),
        ("daily_lookback", 60),
        ("moneyflow_lookback", 10),
        ("main_board_only", False),
        ("baseline_index_code", "000905.SH"),
    ],
)
def test_fingerprint_digest_changes_per_field(field: str, new_value: Any) -> None:
    base = _fp().digest()
    modified = _fp(**{field: new_value}).digest()
    assert base != modified, f"{field} change should bump digest"


# ---------------------------------------------------------------------------
# Shard round-trip
# ---------------------------------------------------------------------------


def _sample_shard_df(n_rows: int = 2) -> pd.DataFrame:
    """Build a valid shard frame with the right columns + dummy values."""
    data: dict[str, list[Any]] = {col: [0.0] * n_rows for col in FEATURE_NAMES}
    data["label"] = pd.array([1, 0][:n_rows], dtype="Int64")
    data["ts_code"] = [f"00000{i}.SZ" for i in range(n_rows)]
    data["anomaly_date"] = ["20260608"] * n_rows
    data["max_ret_5d"] = [6.5, 2.0][:n_rows]
    data["data_status"] = ["complete"] * n_rows
    return pd.DataFrame(data)[SHARD_COLUMNS]


def test_save_and_load_day_shard_roundtrip(isolated_plugin_data_dir: Path) -> None:
    fp = _fp()
    digest = fp.digest()
    save_day_shard(digest, "20260608", _sample_shard_df(n_rows=2))
    loaded = load_day_shard(digest, "20260608")
    assert loaded is not None
    assert list(loaded.columns) == SHARD_COLUMNS
    assert len(loaded) == 2


def test_save_day_shard_rejects_missing_columns(
    isolated_plugin_data_dir: Path,
) -> None:
    fp = _fp()
    bad = _sample_shard_df(n_rows=1).drop(columns=["label"])
    with pytest.raises(ValueError, match="missing required columns"):
        save_day_shard(fp.digest(), "20260608", bad)


def test_completed_dates_reads_disk_not_state(
    isolated_plugin_data_dir: Path,
) -> None:
    fp = _fp()
    digest = fp.digest()
    # Pretend state.json exists but lists nothing; save a shard directly.
    open_or_create(fp)
    save_day_shard(digest, "20260608", _sample_shard_df())
    save_day_shard(digest, "20260610", _sample_shard_df())
    assert completed_dates(digest) == {"20260608", "20260610"}


def test_load_day_shard_missing_returns_none(
    isolated_plugin_data_dir: Path,
) -> None:
    fp = _fp()
    assert load_day_shard(fp.digest(), "29990101") is None


# ---------------------------------------------------------------------------
# State open / mismatch / record_day_done
# ---------------------------------------------------------------------------


def test_open_or_create_then_idempotent(isolated_plugin_data_dir: Path) -> None:
    fp = _fp()
    state1 = open_or_create(fp)
    state2 = open_or_create(fp)
    assert state1.fingerprint == state2.fingerprint
    assert state_path(fp.digest()).is_file()


def test_open_or_create_rejects_mismatched_fingerprint(
    isolated_plugin_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp = _fp()
    digest = fp.digest()
    # Write a doctored state.json whose internal digest disagrees with the
    # directory name.
    target_dir = checkpoint_dir(digest)
    target_dir.mkdir(parents=True, exist_ok=True)
    bad_fp = _fp(end_date="20260101")
    poisoned = CheckpointState(fingerprint=bad_fp)
    # Force the disk record into the wrong dir.
    poisoned_text = poisoned.to_dict()
    poisoned_text["digest"] = digest  # lie about digest so load gets past first gate
    (target_dir / "state.json").write_text(
        __import__("json").dumps(poisoned_text), encoding="utf-8"
    )
    with pytest.raises(CheckpointMismatch):
        load_state(digest)


def test_record_day_done_writes_state(isolated_plugin_data_dir: Path) -> None:
    fp = _fp()
    digest = fp.digest()
    open_or_create(fp)
    record_day_done(digest, "20260608")
    state = load_state(digest)
    assert state is not None
    assert "20260608" in state.completed_dates


def test_delete_checkpoint_cleans_dir(isolated_plugin_data_dir: Path) -> None:
    fp = _fp()
    digest = fp.digest()
    open_or_create(fp)
    save_day_shard(digest, "20260608", _sample_shard_df())
    assert checkpoint_dir(digest).is_dir()
    delete_checkpoint(digest)
    assert not checkpoint_dir(digest).is_dir()


# ---------------------------------------------------------------------------
# Resume semantics — mid-window crash recovery
# ---------------------------------------------------------------------------


def test_resume_skips_already_completed_dates(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
) -> None:
    """Two runs: first one only processes 3 of 5 dates (simulated by
    pre-seeding shards for the last 2). Second pass must NOT recompute them."""
    cal = TradeCalendar(trade_cal_df())
    fixtures = _build_fixtures()

    # Step 1: write shards for the last two anomaly_dates manually.
    fp = _fp()
    open_or_create(fp)
    digest = fp.digest()
    for d in ("20260611", "20260612"):
        save_day_shard(digest, d, _sample_shard_df())
        record_day_done(digest, d)
    assert completed_dates(digest) == {"20260611", "20260612"}

    # Step 2: full window run. The progress callback should report "resumed"
    # for the two pre-seeded dates (n=-1 in the callback contract).
    seen: list[tuple[str, int, int]] = []
    collect_training_window(
        tushare=FakeTushareClient(fixtures),
        db=db_with_anomalies,
        calendar=cal,
        start_date="20260601",
        end_date="20260612",
        daily_lookback=5,
        on_day=lambda T, n, cum: seen.append((T, n, cum)),
    )
    resumed_dates = {T for T, n, _ in seen if n == -1}
    fresh_dates = {T for T, n, _ in seen if n >= 0}
    assert resumed_dates == {"20260611", "20260612"}
    # The first three dates should still be processed fresh.
    assert fresh_dates == {"20260608", "20260609", "20260610"}


def test_resume_disabled_purges_existing_checkpoint(
    db_with_anomalies: Database,
    isolated_plugin_data_dir: Path,
) -> None:
    cal = TradeCalendar(trade_cal_df())
    fp = _fp()
    digest = fp.digest()
    open_or_create(fp)
    save_day_shard(digest, "20260611", _sample_shard_df())
    record_day_done(digest, "20260611")
    assert completed_dates(digest) == {"20260611"}

    collect_training_window(
        tushare=FakeTushareClient(_build_fixtures()),
        db=db_with_anomalies,
        calendar=cal,
        start_date="20260601",
        end_date="20260612",
        daily_lookback=5,
        checkpoint_resume=False,
    )
    # checkpoint_resume=False wipes the dir at start, so the old shard is gone
    # and 20260611 was reprocessed from scratch.
    fresh = completed_dates(digest)
    assert "20260611" in fresh  # repopulated
    # But the pre-seeded shard was not the "real" one — the fresh shard will
    # have feature columns derived from the stub Tushare data instead.
    df = load_day_shard(digest, "20260611")
    assert df is not None
    assert "f_st_last_close_yuan" in df.columns
