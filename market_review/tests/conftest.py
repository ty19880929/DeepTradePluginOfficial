"""Shared fixtures for market-review tests.

The DB fixture applies the plugin's two migration files into a tmp
DuckDB and hands back a :class:`Database` ready for ``mr_*`` writes. We
don't go through the framework's auto-migrate flow because that would
require a full ``PluginContext`` install — overkill for unit tests.

The :class:`FakeTushare` fake exposes the subset of ``TushareClient`` that
:mod:`market_review.data` actually uses (``call`` + ``materialize``). It
records every invocation so tests can assert on the exact call pattern
(per-day loop vs range call vs per-index call). ``materialize`` is a
no-op that just returns ``len(df)``; tests focus on orchestration, not on
the framework's persistence layer (which lub's own test suite covers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from deeptrade.core.db import Database

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
MIGRATION_FILES: tuple[Path, ...] = (
    MIGRATIONS_DIR / "20260601_001_init.sql",
    MIGRATIONS_DIR / "20260601_002_config.sql",
)


@pytest.fixture
def mr_db(tmp_path: Path) -> Database:
    """A fresh DuckDB with all mr_* tables created.

    Returned :class:`Database` is *not* automatically closed by the fixture
    because pytest's tmp_path cleanup nukes the file when the test exits;
    closing explicitly inside the test is optional.
    """
    db = Database(tmp_path / "mr_test.duckdb", auto_migrate=False)
    for migration in MIGRATION_FILES:
        sql_text = migration.read_text(encoding="utf-8")
        for stmt in sql_text.split(";"):
            stmt = stmt.strip()
            if stmt:
                db.execute(stmt)
    return db


# ---------------------------------------------------------------------------
# FakeTushare — minimal stand-in for ``deeptrade.core.tushare_client.TushareClient``
# ---------------------------------------------------------------------------


@dataclass
class _Call:
    """One recorded ``tushare.call(api, ...)`` invocation."""

    api: str
    trade_date: str | None
    params: dict[str, Any] | None
    force_sync: bool


@dataclass
class _Materialize:
    """One recorded ``tushare.materialize(table, df, key_cols=...)`` invocation."""

    table: str
    rows: int
    key_cols: list[str] | None


class FakeTushare:
    """A test fake exposing the ``call`` + ``materialize`` subset.

    Configure responses with :meth:`set_response` (callable taking the call
    kwargs and returning a DataFrame) or :meth:`set_static` (a fixed
    DataFrame returned every time). APIs without a configured response
    return an empty DataFrame.

    Every call lands in :attr:`calls` and every materialize lands in
    :attr:`materializes`, in invocation order.
    """

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self.materializes: list[_Materialize] = []
        self._responses: dict[str, Callable[..., pd.DataFrame]] = {}

    # --- configuration ----------------------------------------------------
    def set_response(self, api: str, responder: Callable[..., pd.DataFrame]) -> None:
        self._responses[api] = responder

    def set_static(self, api: str, df: pd.DataFrame) -> None:
        self._responses[api] = lambda **_: df.copy()

    # --- TushareClient surface --------------------------------------------
    def call(
        self,
        api_name: str,
        *,
        trade_date: str | None = None,
        params: dict[str, Any] | None = None,
        fields: str | None = None,  # noqa: ARG002 — accepted for signature parity
        force_sync: bool = False,
    ) -> pd.DataFrame:
        self.calls.append(_Call(
            api=api_name, trade_date=trade_date,
            params=dict(params) if params else None, force_sync=force_sync,
        ))
        responder = self._responses.get(api_name)
        if responder is None:
            return pd.DataFrame()
        df = responder(trade_date=trade_date, params=params, force_sync=force_sync)
        return df if df is not None else pd.DataFrame()

    def materialize(
        self,
        table_name: str,
        df: pd.DataFrame,
        *,
        key_cols: list[str] | None = None,
    ) -> int:
        n = 0 if df is None else int(len(df))
        self.materializes.append(_Materialize(table=table_name, rows=n, key_cols=key_cols))
        return n

    # --- assertion helpers ------------------------------------------------
    def calls_to(self, api: str) -> list[_Call]:
        return [c for c in self.calls if c.api == api]

    def materializes_to(self, table: str) -> list[_Materialize]:
        return [m for m in self.materializes if m.table == table]


@pytest.fixture
def fake_tushare() -> FakeTushare:
    return FakeTushare()
