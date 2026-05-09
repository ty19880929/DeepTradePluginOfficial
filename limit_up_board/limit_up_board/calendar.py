"""TradeCalendar — pure helper around tushare trade_cal frames."""

from __future__ import annotations

import pandas as pd


class TradeCalendar:
    """Read-only view over the trade_cal DataFrame.

    Required columns: cal_date (str YYYYMMDD), is_open (0/1), pretrade_date (str).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not {"cal_date", "is_open"}.issubset(df.columns):
            raise ValueError("trade_cal frame missing required columns")

        # ⚠ Bug fix: tushare trade_cal returns cal_date as int64 in some
        # environments; JSON cache round-trip can also widen "20260428" → 20260428.
        # Normalize to string up-front so all downstream comparisons are str↔str.
        df = df.copy()
        df["cal_date"] = df["cal_date"].astype(str)
        if "pretrade_date" in df.columns:
            # Some non-trading days have NaN pretrade_date; preserve NaN, stringify the rest
            df["pretrade_date"] = df["pretrade_date"].apply(
                lambda v: str(v) if pd.notna(v) else None
            )
        # is_open is sometimes "1"/"0" strings — coerce to int for ==1 comparison
        df["is_open"] = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(int)

        # Normalize: deduplicate on cal_date, keeping the SSE/last row
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
        """Return the most recent open day strictly BEFORE `date`."""
        # Walk backwards through cal_date sorted index
        candidates = self._df[self._df["cal_date"] < date]
        opens = candidates[candidates["is_open"] == 1]
        if opens.empty:
            raise ValueError(f"no prior open trading day before {date}")
        return str(opens.iloc[-1]["cal_date"])

    def next_open(self, date: str) -> str:
        """Return the first open day strictly AFTER `date`."""
        candidates = self._df[self._df["cal_date"] > date]
        opens = candidates[candidates["is_open"] == 1]
        if opens.empty:
            raise ValueError(f"no future open trading day after {date}")
        return str(opens.iloc[0]["cal_date"])

    def latest_closed_on_or_before(self, date: str) -> str:
        """Return the latest open day ≤ `date` (i.e. `date` itself if open, else preceding)."""
        if self.is_open(date):
            return date
        return self.pretrade_date(date)
