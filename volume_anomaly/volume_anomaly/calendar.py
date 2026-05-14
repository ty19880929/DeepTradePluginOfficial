"""TradeCalendar — pure helper around tushare trade_cal frames.

Self-contained: each plugin owns its own copy rather than importing from
sibling plugins (per plugin contract — plugins are autonomous units).
"""

from __future__ import annotations

import pandas as pd


class TradeCalendar:
    """Read-only view over the trade_cal DataFrame.

    Required columns: cal_date (str YYYYMMDD), is_open (0/1), pretrade_date (str).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not {"cal_date", "is_open"}.issubset(df.columns):
            raise ValueError("trade_cal frame missing required columns")
        df = df.copy()
        df["cal_date"] = df["cal_date"].astype(str)
        if "pretrade_date" in df.columns:
            df["pretrade_date"] = df["pretrade_date"].apply(
                lambda v: str(v) if pd.notna(v) else None
            )
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
        candidates = self._df[self._df["cal_date"] < date]
        opens = candidates[candidates["is_open"] == 1]
        if opens.empty:
            raise ValueError(f"no prior open trading day before {date}")
        return str(opens.iloc[-1]["cal_date"])

    def next_open(self, date: str) -> str:
        candidates = self._df[self._df["cal_date"] > date]
        opens = candidates[candidates["is_open"] == 1]
        if opens.empty:
            raise ValueError(f"no future open trading day after {date}")
        return str(opens.iloc[0]["cal_date"])

    def open_dates_in_range(self, start: str, end: str) -> list[str]:
        """Return sorted YYYYMMDD trade dates with is_open==1 in [start, end].

        Both endpoints inclusive. Non-open dates (weekends, holidays) and dates
        outside the loaded calendar window are silently dropped.
        """
        if start > end:
            return []
        df = self._df
        mask = (df["cal_date"] >= start) & (df["cal_date"] <= end) & (df["is_open"] == 1)
        return [str(v) for v in df.loc[mask, "cal_date"].tolist()]
