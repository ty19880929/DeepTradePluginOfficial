"""v0.16.0 — _build_candidate_rows attaches industries / concepts / regions.

The 框架级 ``ConceptRepository`` is stubbed (it's an external dependency we
don't want to drag in); we just need ``boards_by_stock(ts_code)`` to return
a list of objects with ``ts_code`` / ``name`` / ``type`` fields, mirroring
the real ``ConceptBoard`` dataclass shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from limit_up_board.data import _build_candidate_rows


@dataclass(frozen=True)
class _Board:
    ts_code: str
    name: str
    type: str


class _StubRepo:
    """Minimal stand-in for ConceptRepository.boards_by_stock."""

    def __init__(self, table: dict[str, list[_Board]]) -> None:
        self._table = table

    def boards_by_stock(self, ts_code: str, type: str | None = None) -> list[_Board]:  # noqa: ARG002, A002
        # Real signature accepts a ``type=`` filter; we ignore it because
        # _build_candidate_rows always queries the unfiltered list and buckets
        # in Python.
        return list(self._table.get(ts_code, []))


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600519.SH"],
            "name": ["平安银行", "贵州茅台"],
            "industry_basic": ["银行", "白酒"],
            "first_time": ["09:30:00", "09:30:00"],
            "last_time": ["09:30:01", "09:30:01"],
            "open_times": [0, 0],
            "limit_times": [1, 1],
            "up_stat": ["1/1", "1/1"],
            "pct_chg": [10.0, 10.0],
            "close": [12.0, 1800.0],
            "turnover_ratio": [3.5, 1.2],
            "fd_amount": [1.5e9, 2.0e10],
            "limit_amount": [1.5e9, 2.0e10],
            "amount": [3e9, 4e10],
            "total_mv": [5e10, 2e12],
            "float_mv": [5e10, 2e12],
        }
    )


class TestConceptAttach:
    def test_buckets_by_type(self) -> None:
        repo = _StubRepo(
            {
                "000001.SZ": [
                    _Board("885338.TI", "银行", "I"),
                    _Board("885712.TI", "沪深300_概念", "N"),
                    _Board("886000.TI", "深圳", "R"),
                ],
                "600519.SH": [
                    _Board("885001.TI", "白酒", "I"),
                    _Board("885900.TI", "消费升级", "N"),
                    _Board("885901.TI", "品牌消费", "N"),
                    _Board("886100.TI", "贵州", "R"),
                ],
            }
        )
        rows = _build_candidate_rows(_toy_frame(), None, concept_repo=repo)
        by_code = {r["ts_code"]: r for r in rows}

        assert by_code["000001.SZ"]["industries"] == [{"ts_code": "885338.TI", "name": "银行"}]
        assert by_code["000001.SZ"]["concepts"] == [
            {"ts_code": "885712.TI", "name": "沪深300_概念"}
        ]
        assert by_code["000001.SZ"]["regions"] == [{"ts_code": "886000.TI", "name": "深圳"}]

        # 600519.SH: 2 个概念全量保留，证明不截断
        assert len(by_code["600519.SH"]["concepts"]) == 2
        assert by_code["600519.SH"]["concepts"] == [
            {"ts_code": "885900.TI", "name": "消费升级"},
            {"ts_code": "885901.TI", "name": "品牌消费"},
        ]

    def test_no_repo_emits_empty_lists(self) -> None:
        rows = _build_candidate_rows(_toy_frame(), None, concept_repo=None)
        for r in rows:
            assert r["industries"] == []
            assert r["concepts"] == []
            assert r["regions"] == []

    def test_repo_returns_empty_emits_empty_lists(self) -> None:
        """Snapshot未同步时，boards_by_stock 全部返回 []，candidate 应仍带三个键。"""
        rows = _build_candidate_rows(_toy_frame(), None, concept_repo=_StubRepo({}))
        for r in rows:
            assert r["industries"] == []
            assert r["concepts"] == []
            assert r["regions"] == []
            # 老的 industry 字段（来自 industry_basic）仍保留，未被替换
            assert r["industry"] in {"银行", "白酒"}


class TestSectorStrengthSimplified:
    def test_resolve_returns_unavailable_when_no_data(self) -> None:
        from limit_up_board.data import resolve_sector_strength

        out = resolve_sector_strength(limit_cpt_list=None)
        assert out.source == "unavailable"
        assert out.data == {"top_sectors": []}

    def test_resolve_uses_limit_cpt_list_when_available(self) -> None:
        from limit_up_board.data import resolve_sector_strength

        df = pd.DataFrame(
            {
                "name": ["人工智能", "新能源"],
                "rank": [1, 2],
                "up_nums": [12, 8],
            }
        )
        out = resolve_sector_strength(limit_cpt_list=df)
        assert out.source == "limit_cpt_list"
        assert out.data["top_sectors"][0]["name"] == "人工智能"
