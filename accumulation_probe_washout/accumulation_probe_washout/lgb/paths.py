"""On-disk locations for APW LightGBM artefacts.

Layout::

    ~/.deeptrade/accumulation_probe_washout/
    ├── models/        # booster files registered in apw_lgb_models
    ├── datasets/      # parquet snapshots of each training matrix
    └── checkpoints/   # Phase-1 collection shards (resumable trains)

The framework's ``deeptrade.core.paths`` exposes the per-plugin root via
``plugin_data_dir("accumulation-probe-washout")``; we just build the three
sub-directories on top.
"""

from __future__ import annotations

from pathlib import Path


PLUGIN_ID = "accumulation-probe-washout"


def _plugin_root() -> Path:
    # Import inside the function so importing this module does not require the
    # framework to be importable (keeps unit tests that exercise features.py
    # without ``deeptrade.core.paths`` working).
    from deeptrade.core import paths as _fw_paths  # noqa: PLC0415
    return Path(_fw_paths.plugin_data_dir(PLUGIN_ID))


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
