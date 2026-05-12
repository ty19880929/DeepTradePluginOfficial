"""Tests for :mod:`limit_up_board.lgb.checkpoint` —— Phase-1 续训 checkpoint。

覆盖：
* fingerprint 稳定性（同输入 → 同 digest；任一字段改动 → 不同 digest）
* state.json 原子写 + 回读
* shard 写盘 → completed_dates / load_day_shard / assemble_full_dataset 自洽
* open_or_create 的 3 个分支：新建 / 复用 / 损坏报错
* count_checkpoints / purge_all_checkpoints / delete_checkpoint
* 与 collect_training_window 的集成：skip_dates 跳过抓数，day_sink 收到 shard
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from limit_up_board.calendar import TradeCalendar
from limit_up_board.lgb import checkpoint as lgb_checkpoint
from limit_up_board.lgb.checkpoint import (
    SHARD_COLUMNS,
    CheckpointFingerprint,
    CheckpointMismatch,
    CheckpointState,
    assemble_full_dataset,
    completed_dates,
    count_checkpoints,
    day_bundle_to_shard,
    delete_checkpoint,
    load_day_shard,
    load_state,
    open_or_create,
    purge_all_checkpoints,
    record_day_done,
    save_day_shard,
    save_state,
)
from limit_up_board.lgb.features import FEATURE_NAMES, SCHEMA_VERSION

# 复用 dataset 的 fixture builder
from test_lgb_dataset import (  # type: ignore[import-not-found]
    _ALL_DATES_INCL_T1,
    _TRADE_DATES,
    FakeTushareClient,
    _build_fixtures,
    _trade_cal_df,
)


# ---------------------------------------------------------------------------
# Common fixture: isolated DEEPTRADE_HOME so checkpoint paths live in tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))
    return home


def _make_fp(**overrides: Any) -> CheckpointFingerprint:
    base: dict[str, Any] = {
        "start_date": "20260101",
        "end_date": "20260301",
        "schema_version": SCHEMA_VERSION,
        "label_threshold_pct": 7.0,
        "daily_lookback": 30,
        "moneyflow_lookback": 5,
        "min_float_mv_yi": 30.0,
        "max_float_mv_yi": 100.0,
        "max_close_yuan": 15.0,
    }
    base.update(overrides)
    return CheckpointFingerprint(**base)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_digest_is_12_hex_chars(self) -> None:
        fp = _make_fp()
        d = fp.digest()
        assert len(d) == 12
        assert all(c in "0123456789abcdef" for c in d)

    def test_same_inputs_same_digest(self) -> None:
        assert _make_fp().digest() == _make_fp().digest()

    @pytest.mark.parametrize(
        "field,new_value",
        [
            ("start_date", "20260102"),
            ("end_date", "20260302"),
            ("schema_version", SCHEMA_VERSION + 1),
            ("label_threshold_pct", 6.9),
            ("daily_lookback", 31),
            ("moneyflow_lookback", 6),
            ("min_float_mv_yi", 29.9),
            ("max_float_mv_yi", 100.1),
            ("max_close_yuan", 14.99),
        ],
    )
    def test_any_field_change_changes_digest(self, field: str, new_value: Any) -> None:
        baseline = _make_fp().digest()
        modified = _make_fp(**{field: new_value}).digest()
        assert baseline != modified, f"digest should change when {field} changes"


# ---------------------------------------------------------------------------
# State I/O round-trip
# ---------------------------------------------------------------------------


class TestStateIO:
    def test_save_then_load(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp = _make_fp()
        state = CheckpointState(
            fingerprint=fp,
            completed_dates=["20260105", "20260106"],
            plugin_version="0.5.5",
        )
        save_state(state)

        loaded = load_state(fp.digest())
        assert loaded is not None
        assert loaded.fingerprint == fp
        assert loaded.completed_dates == ["20260105", "20260106"]
        assert loaded.plugin_version == "0.5.5"
        assert loaded.created_at  # auto-stamped
        assert loaded.updated_at

    def test_load_missing_returns_none(self, isolated_home: Path) -> None:  # noqa: ARG002
        assert load_state("deadbeefcafe") is None

    def test_corrupted_json_raises(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp = _make_fp()
        digest = fp.digest()
        sp = lgb_checkpoint.state_path(digest)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(CheckpointMismatch):
            load_state(digest)

    def test_digest_dirname_mismatch_raises(self, isolated_home: Path) -> None:  # noqa: ARG002
        """state.json fingerprint 与目录名 digest 不一致（拷贝/损坏）→ raise。"""
        fp = _make_fp()
        # 写到一个错误的目录名下，但 state.json 内部 fingerprint 仍是 fp
        wrong_digest = "0" * 12
        sp = lgb_checkpoint.state_path(wrong_digest)
        sp.parent.mkdir(parents=True, exist_ok=True)
        state = CheckpointState(fingerprint=fp, plugin_version="x")
        save_state(state)  # 这会写到正确的 digest 目录
        # 现在把内容拷到 wrong_digest 目录
        real_sp = lgb_checkpoint.state_path(fp.digest())
        sp.write_text(real_sp.read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(CheckpointMismatch):
            load_state(wrong_digest)


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------


def _toy_shard(trade_date: str, n_rows: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        row: dict[str, Any] = {col: float(i) for col in FEATURE_NAMES}
        row["label"] = pd.NA if i % 3 == 2 else (1 if i % 2 == 0 else 0)
        row["ts_code"] = f"00000{i}.SZ"
        row["trade_date"] = trade_date
        row["next_trade_date"] = "20260102"
        row["pct_chg_t1"] = 5.5 + i
        rows.append(row)
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("Int64")
    return df[SHARD_COLUMNS]


class TestShardIO:
    def test_save_and_load_round_trip(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp = _make_fp()
        digest = fp.digest()
        shard = _toy_shard("20260105")
        save_day_shard(digest, "20260105", shard)

        got = load_day_shard(digest, "20260105")
        assert got is not None
        assert list(got.columns) == SHARD_COLUMNS
        assert len(got) == 2
        assert got["trade_date"].iloc[0] == "20260105"

    def test_missing_shard_returns_none(self, isolated_home: Path) -> None:  # noqa: ARG002
        assert load_day_shard("missingdigest", "20260105") is None

    def test_save_rejects_extra_required_cols_missing(
        self, isolated_home: Path,  # noqa: ARG002
    ) -> None:
        fp = _make_fp()
        bad = pd.DataFrame({"ts_code": ["x"]})  # missing nearly everything
        with pytest.raises(ValueError, match="missing required columns"):
            save_day_shard(fp.digest(), "20260105", bad)

    def test_empty_shard_allowed(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp = _make_fp()
        digest = fp.digest()
        empty = pd.DataFrame(columns=SHARD_COLUMNS)
        save_day_shard(digest, "20260105", empty)
        assert "20260105" in completed_dates(digest)


class TestCompletedDates:
    def test_returns_disk_truth(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp = _make_fp()
        digest = fp.digest()
        save_day_shard(digest, "20260105", _toy_shard("20260105"))
        save_day_shard(digest, "20260106", _toy_shard("20260106"))
        assert completed_dates(digest) == {"20260105", "20260106"}

    def test_empty_when_dir_missing(self, isolated_home: Path) -> None:  # noqa: ARG002
        assert completed_dates("nosuchdigest") == set()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


class TestAssemble:
    def test_concat_three_days(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp = _make_fp()
        digest = fp.digest()
        for d in ("20260105", "20260106", "20260107"):
            save_day_shard(digest, d, _toy_shard(d, n_rows=2))

        ds = assemble_full_dataset(
            digest,
            label_threshold_pct=fp.label_threshold_pct,
            daily_lookback=fp.daily_lookback,
            moneyflow_lookback=fp.moneyflow_lookback,
            trade_dates=["20260105", "20260106", "20260107"],
        )
        assert ds.n_samples == 6
        assert list(ds.feature_matrix.columns) == FEATURE_NAMES
        assert set(int(g) for g in ds.split_groups.dropna()) == {
            20260105,
            20260106,
            20260107,
        }
        assert ds.trade_dates == ["20260105", "20260106", "20260107"]

    def test_assemble_empty_dir(self, isolated_home: Path) -> None:  # noqa: ARG002
        ds = assemble_full_dataset(
            "nosuch",
            label_threshold_pct=7.0,
            daily_lookback=30,
            moneyflow_lookback=5,
        )
        assert ds.n_samples == 0
        assert list(ds.feature_matrix.columns) == FEATURE_NAMES


# ---------------------------------------------------------------------------
# Lifecycle (open_or_create / record_day_done / delete / purge)
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_open_or_create_new(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp = _make_fp()
        state = open_or_create(fp, plugin_version="0.5.5")
        assert state.fingerprint == fp
        assert state.completed_dates == []
        assert lgb_checkpoint.state_path(fp.digest()).is_file()

    def test_open_or_create_reuses_existing(
        self, isolated_home: Path,  # noqa: ARG002
    ) -> None:
        fp = _make_fp()
        first = open_or_create(fp, plugin_version="0.5.5")
        first.completed_dates = ["20260101"]
        save_state(first)
        second = open_or_create(fp, plugin_version="0.5.6")
        assert second.completed_dates == ["20260101"]
        # plugin_version 不被覆盖（保留原始）
        assert second.plugin_version == "0.5.5"

    def test_record_day_done_is_idempotent(
        self, isolated_home: Path,  # noqa: ARG002
    ) -> None:
        fp = _make_fp()
        open_or_create(fp)
        record_day_done(fp.digest(), "20260105")
        record_day_done(fp.digest(), "20260105")
        state = load_state(fp.digest())
        assert state is not None
        assert state.completed_dates == ["20260105"]

    def test_delete_checkpoint_wipes_dir(
        self, isolated_home: Path,  # noqa: ARG002
    ) -> None:
        fp = _make_fp()
        digest = fp.digest()
        open_or_create(fp)
        save_day_shard(digest, "20260105", _toy_shard("20260105"))
        assert lgb_checkpoint.checkpoint_dir(digest).is_dir()
        delete_checkpoint(digest)
        assert not lgb_checkpoint.checkpoint_dir(digest).is_dir()

    def test_count_and_purge_all(self, isolated_home: Path) -> None:  # noqa: ARG002
        fp1 = _make_fp()
        fp2 = _make_fp(start_date="20260201")
        open_or_create(fp1)
        open_or_create(fp2)
        save_day_shard(fp1.digest(), "20260105", _toy_shard("20260105"))
        save_day_shard(fp2.digest(), "20260205", _toy_shard("20260205"))

        n_ck, n_shards = count_checkpoints()
        assert n_ck == 2
        assert n_shards == 2

        n_removed, errs = purge_all_checkpoints()
        assert n_removed == 2
        assert errs == []
        assert count_checkpoints() == (0, 0)


# ---------------------------------------------------------------------------
# day_bundle_to_shard adapter
# ---------------------------------------------------------------------------


class TestDayBundleAdapter:
    def test_empty_bundle_returns_empty_shard(self) -> None:
        empty_fm = pd.DataFrame(columns=FEATURE_NAMES)
        empty_lb = pd.Series([], dtype="Int64", name="label")
        empty_meta = pd.DataFrame(
            columns=["ts_code", "trade_date", "next_trade_date", "pct_chg_t1"]
        )
        shard = day_bundle_to_shard(
            feature_matrix=empty_fm, labels=empty_lb, sample_meta=empty_meta
        )
        assert list(shard.columns) == SHARD_COLUMNS
        assert len(shard) == 0

    def test_round_trip_with_data(self) -> None:
        fm = pd.DataFrame(
            [[1.0] * len(FEATURE_NAMES), [2.0] * len(FEATURE_NAMES)],
            columns=FEATURE_NAMES,
            index=["AAA.SZ", "BBB.SH"],
        )
        lb = pd.Series([1, 0], index=["AAA.SZ", "BBB.SH"], dtype="Int64", name="label")
        meta = pd.DataFrame(
            {
                "ts_code": ["AAA.SZ", "BBB.SH"],
                "trade_date": ["20260105", "20260105"],
                "next_trade_date": ["20260106", "20260106"],
                "pct_chg_t1": [10.0, 5.0],
            }
        )
        shard = day_bundle_to_shard(feature_matrix=fm, labels=lb, sample_meta=meta)
        assert list(shard.columns) == SHARD_COLUMNS
        assert shard["ts_code"].tolist() == ["AAA.SZ", "BBB.SH"]
        assert shard["label"].tolist() == [1, 0]
        assert shard["f_lim_open_times"].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Integration with collect_training_window
# ---------------------------------------------------------------------------


class TrackedTushare:
    """Wrap FakeTushareClient and record (api_name, trade_date) tuples called."""

    def __init__(self, inner: FakeTushareClient) -> None:
        self.inner = inner
        self.calls: list[tuple[str, str]] = []

    def call(self, api_name: str, **kw: Any) -> pd.DataFrame:
        td = kw.get("trade_date")
        if td is None:
            params = kw.get("params") or {}
            td = params.get("trade_date") or f"{params.get('start_date')}:{params.get('end_date')}"
        self.calls.append((api_name, str(td)))
        return self.inner.call(api_name, **kw)


@pytest.fixture
def calendar() -> TradeCalendar:
    return TradeCalendar(_trade_cal_df(_ALL_DATES_INCL_T1))


@pytest.fixture
def tushare() -> FakeTushareClient:
    return FakeTushareClient(_build_fixtures())


class TestCollectTrainingWindowResume:
    def test_skip_dates_means_no_tushare_calls_for_those_days(
        self,
        isolated_home: Path,  # noqa: ARG002
        tushare: FakeTushareClient,
        calendar: TradeCalendar,
    ) -> None:
        from limit_up_board.lgb.dataset import collect_training_window

        tracked = TrackedTushare(tushare)
        skip = {_TRADE_DATES[0], _TRADE_DATES[1]}  # 跳过前两天

        events: list[tuple[str, int]] = []
        collect_training_window(
            tushare=tracked,
            calendar=calendar,
            start_date=_TRADE_DATES[0],
            end_date=_TRADE_DATES[-1],
            on_day=lambda T, n, cum: events.append((T, n)),
            skip_dates=skip,
        )

        # 被 skip 的日子 on_day 收到 n=-1 信号
        resumed = [T for T, n in events if n == -1]
        assert set(resumed) == skip

        # 跳过的日子绝不能触发 stock_st(trade_date=<跳过日>) ——
        # 该 API 仅在 collect_day_samples 中以当前 trade_date 调用，是该日整段
        # 处理是否被绕开的可靠信号。（limit_list_d 不行：次日的 yesterday
        # context 会以 prev_trade_date 形式查询昨日涨停，这是预期的。）
        stock_st_calls = {td for api, td in tracked.calls if api == "stock_st"}
        for d in skip:
            assert d not in stock_st_calls, (
                f"skipped {d} should not trigger stock_st (per-day work)"
            )

    def test_day_sink_receives_shard_per_processed_day(
        self,
        isolated_home: Path,  # noqa: ARG002
        tushare: FakeTushareClient,
        calendar: TradeCalendar,
    ) -> None:
        from limit_up_board.lgb.dataset import collect_training_window

        sink_calls: list[tuple[str, pd.DataFrame]] = []

        def _sink(T: str, df: pd.DataFrame) -> None:
            sink_calls.append((T, df))

        collect_training_window(
            tushare=tushare,
            calendar=calendar,
            start_date=_TRADE_DATES[0],
            end_date=_TRADE_DATES[-1],
            day_sink=_sink,
        )

        # 每个处理过的交易日都触发一次 sink，列契约固定
        assert [T for T, _ in sink_calls] == _TRADE_DATES
        for _, shard in sink_calls:
            assert list(shard.columns) == SHARD_COLUMNS

    def test_resume_after_partial_completion_only_fetches_remaining(
        self,
        isolated_home: Path,  # noqa: ARG002
        tushare: FakeTushareClient,
        calendar: TradeCalendar,
    ) -> None:
        """模拟首次跑前两天崩溃 → 第二次以 skip_dates 续训只跑后三天。"""
        from limit_up_board.lgb.dataset import collect_training_window

        # 首次运行：处理全部 5 天 → 落 shard 到 checkpoint
        fp = CheckpointFingerprint(
            start_date=_TRADE_DATES[0],
            end_date=_TRADE_DATES[-1],
            schema_version=SCHEMA_VERSION,
            label_threshold_pct=7.0,
            daily_lookback=30,
            moneyflow_lookback=5,
            min_float_mv_yi=0.0,
            max_float_mv_yi=100.0,
            max_close_yuan=15.0,
        )
        digest = fp.digest()
        open_or_create(fp)

        # 模拟"崩在第 3 天"：只处理前两天，落盘
        def _sink(T: str, df: pd.DataFrame) -> None:
            save_day_shard(digest, T, df)
            record_day_done(digest, T)

        collect_training_window(
            tushare=tushare,
            calendar=calendar,
            start_date=_TRADE_DATES[0],
            end_date=_TRADE_DATES[1],
            day_sink=_sink,
        )
        assert completed_dates(digest) == {_TRADE_DATES[0], _TRADE_DATES[1]}

        # 第二次运行：完整窗口 + skip_dates → 只处理后 3 天
        tracked = TrackedTushare(tushare)
        collect_training_window(
            tushare=tracked,
            calendar=calendar,
            start_date=_TRADE_DATES[0],
            end_date=_TRADE_DATES[-1],
            skip_dates=completed_dates(digest),
            day_sink=_sink,
        )
        stock_st_calls = {td for api, td in tracked.calls if api == "stock_st"}
        for d in (_TRADE_DATES[0], _TRADE_DATES[1]):
            assert d not in stock_st_calls
        # 后 3 天必然各触发一次 stock_st
        for d in _TRADE_DATES[2:]:
            assert d in stock_st_calls

        # 5 天全部落盘
        assert completed_dates(digest) == set(_TRADE_DATES)

        ds = assemble_full_dataset(
            digest,
            label_threshold_pct=fp.label_threshold_pct,
            daily_lookback=fp.daily_lookback,
            moneyflow_lookback=fp.moneyflow_lookback,
            trade_dates=list(_TRADE_DATES),
        )
        # 同 test_lgb_dataset 一致：5 天 × 2 candidates = 10 samples
        assert ds.n_samples == 10
