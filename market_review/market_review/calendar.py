"""TradeCalendar — pure helper around ``tushare.trade_cal`` frames.

Adapted from ``limit_up_board.calendar`` per design Appendix B ("几乎可直接抄；
新插件再独立一份保持隔离"). Same column contract, same row-norm logic.

Required columns on the input frame: ``cal_date`` (YYYYMMDD str), ``is_open``
(0/1), optionally ``pretrade_date`` (YYYYMMDD str or null on non-trading days).
"""

from __future__ import annotations

import pandas as pd


class TradeCalendar:
    """Read-only view over a normalized ``trade_cal`` DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        if not {"cal_date", "is_open"}.issubset(df.columns):
            raise ValueError("trade_cal frame missing required columns")

        df = df.copy()
        # tushare's trade_cal occasionally returns cal_date as int64; JSON
        # cache round-trip can also widen "20260428" → 20260428. Normalize
        # to string up-front so all downstream comparisons are str↔str.
        df["cal_date"] = df["cal_date"].astype(str)
        if "pretrade_date" in df.columns:
            df["pretrade_date"] = df["pretrade_date"].apply(
                lambda v: str(v) if pd.notna(v) else None
            )
        # is_open occasionally arrives as "1"/"0" strings — coerce to int
        # for the ``== 1`` comparison.
        df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(int)

        sorted_df = df.sort_values("cal_date").drop_duplicates("cal_date", keep="last")
        self._df = sorted_df.reset_index(drop=True)
        self._idx: dict[str, int] = {
            str(row.cal_date): i for i, row in enumerate(self._df.itertuples(index=False))
        }

    def is_open(self, date: str) -> bool:
        idx = self._idx.get(date)
        if idx is None:
            return False
        return int(self._df.at[idx, "is_open"]) == 1

    def pretrade_date(self, date: str) -> str:
        """Return the most recent open day strictly BEFORE ``date``."""
        candidates = self._df[self._df["cal_date"] < date]
        opens = candidates[candidates["is_open"] == 1]
        if opens.empty:
            raise ValueError(f"no prior open trading day before {date}")
        return str(opens.iloc[-1]["cal_date"])

    def next_open(self, date: str) -> str:
        """Return the first open day strictly AFTER ``date``."""
        candidates = self._df[self._df["cal_date"] > date]
        opens = candidates[candidates["is_open"] == 1]
        if opens.empty:
            raise ValueError(f"no future open trading day after {date}")
        return str(opens.iloc[0]["cal_date"])

    def latest_closed_on_or_before(self, date: str) -> str:
        """Return the latest open day ≤ ``date`` (i.e. ``date`` itself if open, else preceding)."""
        if self.is_open(date):
            return date
        return self.pretrade_date(date)

    def range(self, start: str, end: str) -> list[str]:
        """Open trading days T with ``start <= T <= end``, ascending.

        Returns ``[]`` when ``start > end`` (caller convenience for invalid windows).
        """
        if start > end:
            return []
        mask = (
            (self._df["cal_date"] >= start)
            & (self._df["cal_date"] <= end)
            & (self._df["is_open"] == 1)
        )
        return [str(d) for d in self._df.loc[mask, "cal_date"].tolist()]
