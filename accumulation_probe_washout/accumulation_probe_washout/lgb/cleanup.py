"""Retention helpers for the APW LightGBM lifecycle.

* :func:`prune_models` — keep the active model + N most-recent; delete the
  rest (registry row + booster + dataset snapshot).
* :func:`purge` — wholesale clear of a chosen artefact scope.

Both functions are exposed by the ``lgb prune`` / ``lgb purge`` CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import registry as _registry
from .paths import checkpoints_dir, datasets_dir, models_dir, reports_dir

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)


@dataclass
class PruneReport:
    kept: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)


def prune_models(db: Database, *, keep: int) -> PruneReport:
    """Keep the active model + ``keep`` most-recent non-active rows.

    The active model is ALWAYS preserved (deleting it would silently flip
    the runtime into a no-model fallback path); callers wanting to remove
    it must ``lgb activate`` a different row first.
    """
    if keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")
    all_models = _registry.list_models(db)  # sorted DESC by created_at
    active_id = next((m.model_id for m in all_models if m.is_active), None)

    rep = PruneReport()
    kept_non_active = 0
    for m in all_models:
        if m.is_active:
            rep.kept.append(m.model_id)
            continue
        if kept_non_active < keep:
            rep.kept.append(m.model_id)
            kept_non_active += 1
            continue
        _delete_model_artefacts(db, m.model_id, m.file_path, rep)
    return rep


def _delete_model_artefacts(
    db: Database, model_id: str, file_path: str, rep: PruneReport
) -> None:
    p = Path(file_path)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            rep.missing_files.append(str(p))
    else:
        rep.missing_files.append(str(p))
    # Best-effort dataset snapshot removal — naming convention: <model_id>.parquet
    ds = datasets_dir() / f"{model_id}.parquet"
    if ds.exists():
        try:
            ds.unlink()
        except OSError:
            pass
    _registry.delete_model(db, model_id)
    rep.deleted.append(model_id)


@dataclass
class PurgeReport:
    scope: str
    files_removed: int = 0
    rows_removed: int = 0


def purge(
    db: Database,
    *,
    datasets: bool = False,
    models: bool = False,
    predictions: bool = False,
    checkpoints: bool = False,
) -> list[PurgeReport]:
    """Wholesale clear by scope. ``predictions`` is reserved for PR-4 — the
    table does not exist yet, so the function silently no-ops in that branch
    when the table is missing.
    """
    out: list[PurgeReport] = []
    if datasets:
        out.append(_purge_dir("datasets", datasets_dir()))
    if checkpoints:
        out.append(_purge_dir("checkpoints", checkpoints_dir()))
    if models:
        rep = _purge_dir("models", models_dir())
        # Also wipe the registry table — booster files are useless without it.
        n = db.fetchone("SELECT COUNT(*) FROM apw_lgb_models")[0]
        db.execute("DELETE FROM apw_lgb_models")
        rep.rows_removed = int(n or 0)
        out.append(rep)
    if predictions:
        # apw_lgb_predictions is added in PR-4; gracefully skip until then.
        try:
            n = db.fetchone("SELECT COUNT(*) FROM apw_lgb_predictions")[0]
            db.execute("DELETE FROM apw_lgb_predictions")
            out.append(PurgeReport(scope="predictions", rows_removed=int(n or 0)))
        except Exception:  # noqa: BLE001 — table missing (pre-PR-4)
            out.append(PurgeReport(scope="predictions", rows_removed=0))
    # reports_dir is not a default scope — silent.
    _ = reports_dir
    return out


def _purge_dir(scope: str, root: Path) -> PurgeReport:
    rep = PurgeReport(scope=scope)
    if not root.exists():
        return rep
    for p in sorted(root.rglob("*"), reverse=True):
        try:
            if p.is_file():
                p.unlink()
                rep.files_removed += 1
            elif p.is_dir() and p != root:
                p.rmdir()
        except OSError:
            continue
    return rep
