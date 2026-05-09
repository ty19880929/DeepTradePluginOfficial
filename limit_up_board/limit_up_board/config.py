"""Plugin-local settings (v0.4).

Persisted in the ``lub_config`` table. Defaults live on :class:`LubConfig` and
are re-applied automatically when a row is missing — DB rows only override.

Currently exposed via ``deeptrade limit-up-board settings``:
    * ``max_float_mv_yi``  — 流通市值上限（亿）
    * ``max_close_yuan``   — 当前股价上限（元）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


@dataclass
class LubConfig:
    """User-tunable run filters. Defaults reflect a typical 打板 watchlist."""

    max_float_mv_yi: float = 100.0
    max_close_yuan: float = 15.0


_KEY_PREFIX = "lub."


def _full_key(field_name: str) -> str:
    return f"{_KEY_PREFIX}{field_name}"


def load_config(db: Database) -> LubConfig:
    """Materialize a :class:`LubConfig` from ``lub_config``; missing rows fall
    through to the dataclass default."""
    overrides: dict[str, Any] = {}
    for f in fields(LubConfig):
        row = db.fetchone("SELECT value_json FROM lub_config WHERE key = ?", (_full_key(f.name),))
        if row is not None:
            overrides[f.name] = json.loads(row[0])
    return LubConfig(**overrides)


def save_config(db: Database, cfg: LubConfig) -> None:
    """Upsert every field of *cfg* into ``lub_config``."""
    with db.transaction():
        for f in fields(LubConfig):
            key = _full_key(f.name)
            value = getattr(cfg, f.name)
            payload = json.dumps(value)
            db.execute("DELETE FROM lub_config WHERE key = ?", (key,))
            db.execute(
                "INSERT INTO lub_config(key, value_json) VALUES (?, ?)",
                (key, payload),
            )


def list_for_show(db: Database) -> list[tuple[str, Any, str]]:
    """``[(key, value, source)]`` for the ``settings show`` table.

    ``source`` is ``"persisted"`` if the field has a row in ``lub_config``,
    otherwise ``"default"``.
    """
    out: list[tuple[str, Any, str]] = []
    defaults = LubConfig()
    for f in fields(LubConfig):
        key = _full_key(f.name)
        row = db.fetchone("SELECT value_json FROM lub_config WHERE key = ?", (key,))
        if row is not None:
            out.append((key, json.loads(row[0]), "persisted"))
        else:
            out.append((key, getattr(defaults, f.name), "default"))
    return out
