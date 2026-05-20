"""Train/val/OOS split parser + cherry-pick guard tests (PR-6.2)."""

from __future__ import annotations

import pytest

from checkmate.split import (
    Segment,
    Split,
    forbid_rank_on_oos,
    parse_split,
)


# ---------------------------------------------------------------------------
# parse_split — year shorthand
# ---------------------------------------------------------------------------


class TestParseYearShorthand:
    def test_three_segments(self) -> None:
        s = parse_split("train=2014-2020 val=2021-2023 oos=2024-2026")
        assert s.train == Segment("train", "20140101", "20201231")
        assert s.val == Segment("val", "20210101", "20231231")
        assert s.oos == Segment("oos", "20240101", "20261231")

    def test_train_only(self) -> None:
        s = parse_split("train=2020-2023")
        assert s.train.start == "20200101" and s.train.end == "20231231"
        assert s.val is None
        assert s.oos is None


# ---------------------------------------------------------------------------
# parse_split — explicit YYYY-MM-DD bounds
# ---------------------------------------------------------------------------


class TestParseExplicit:
    def test_mixed_with_explicit_dates(self) -> None:
        s = parse_split(
            "train=2014-01-01:2020-06-30 "
            "val=2020-07-01:2022-12-31 "
            "oos=2023-01-01:2024-12-31"
        )
        assert s.train.start == "20140101" and s.train.end == "20200630"
        assert s.val.start == "20200701" and s.val.end == "20221231"
        assert s.oos.start == "20230101" and s.oos.end == "20241231"


# ---------------------------------------------------------------------------
# parse_split — error cases
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_empty_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_split("")

    def test_missing_train_raises(self) -> None:
        with pytest.raises(ValueError, match="train"):
            parse_split("val=2020-2021")

    def test_overlap_train_val(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            parse_split("train=2014-2020 val=2020-2022")

    def test_overlap_val_oos(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            parse_split("train=2014-2018 val=2019-2022 oos=2022-2024")

    def test_unknown_segment_name(self) -> None:
        with pytest.raises(ValueError, match="unknown segment"):
            parse_split("train=2014-2020 wat=2021-2023")

    def test_duplicate_segment(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            parse_split("train=2014-2018 train=2019-2022")

    def test_bad_range_format(self) -> None:
        with pytest.raises(ValueError, match="unrecognised"):
            parse_split("train=14-20")

    def test_start_after_end(self) -> None:
        with pytest.raises(ValueError, match="start"):
            parse_split("train=2020-2014")


# ---------------------------------------------------------------------------
# Cherry-pick guard
# ---------------------------------------------------------------------------


def _full_split() -> Split:
    return parse_split("train=2014-2020 val=2021-2023 oos=2024-2026")


class TestForbidRankOnOos:
    def test_grid_inside_train_passes(self) -> None:
        s = _full_split()
        forbid_rank_on_oos(s, grid_start="2015-01-01", grid_end="2018-12-31")

    def test_grid_inside_val_passes(self) -> None:
        s = _full_split()
        forbid_rank_on_oos(s, grid_start="2022-01-01", grid_end="2022-12-31")

    def test_grid_inside_oos_raises(self) -> None:
        s = _full_split()
        with pytest.raises(ValueError, match="OOS"):
            forbid_rank_on_oos(s, grid_start="2024-06-01", grid_end="2025-06-30")

    def test_grid_straddles_val_oos_raises(self) -> None:
        """If the grid window dips into OOS at all, refuse."""
        s = _full_split()
        with pytest.raises(ValueError, match="OOS"):
            forbid_rank_on_oos(s, grid_start="2023-06-01", grid_end="2024-06-30")

    def test_no_oos_segment_makes_guard_noop(self) -> None:
        """When --split has no oos, the guard does nothing — caller is OK."""
        s = parse_split("train=2014-2018 val=2019-2022")
        forbid_rank_on_oos(s, grid_start="2014-01-01", grid_end="2022-12-31")

    def test_yyyymmdd_input_accepted(self) -> None:
        """forbid_rank_on_oos must accept dashes and YYYYMMDD interchangeably."""
        s = _full_split()
        with pytest.raises(ValueError, match="OOS"):
            forbid_rank_on_oos(s, grid_start="20240601", grid_end="20250630")
