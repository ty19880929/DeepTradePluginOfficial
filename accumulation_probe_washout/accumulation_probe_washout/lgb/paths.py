"""On-disk locations for APW LightGBM artefacts.

Layout::

    <paths.db_path().parent>/accumulation_probe_washout/
    ├── models/        # booster files registered in apw_lgb_models
    ├── datasets/      # parquet snapshots of each training matrix
    ├── checkpoints/   # Phase-1 collection shards (resumable trains)
    └── reports/       # lgb evaluate / drift JSON dumps

We anchor to ``deeptrade.core.paths.db_path().parent`` so the plugin's
on-disk state lives next to the DB file. The future framework API
``paths.plugin_data_dir(plugin_id)`` will be a drop-in upgrade.
"""

from __future__ import annotations

from pathlib import Path


PLUGIN_ID = "accumulation-probe-washout"
_PLUGIN_DIRNAME = "accumulation_probe_washout"


def _plugin_root() -> Path:
    # Import inside the function so importing this module does not require
    # the framework to be importable at unit-test setup time.
    from deeptrade.core import paths as _fw_paths  # noqa: PLC0415
    return _fw_paths.db_path().parent / _PLUGIN_DIRNAME


def models_dir() -> Path:
    p = _plugin_root() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def datasets_dir() -> Path:
    p = _plugin_root() / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoints_dir() -> Path:
    p = _plugin_root() / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir() -> Path:
    """JSON dump target for ``lgb evaluate`` and ``lgb evaluate --drift``."""
    p = _plugin_root() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p
