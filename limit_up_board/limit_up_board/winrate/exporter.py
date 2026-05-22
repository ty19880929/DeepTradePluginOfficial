"""逐股明细导出 — JSON / CSV。

PR #3 — 与 ``summary`` 共用 resolve/aggregate 链路；只是把 ``ResolvedRecord``
列表展开为可写入文件的扁平结构。CSV 列顺序固定，方便人工导入 Excel。
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from .resolver import ResolvedRecord
    from .stats import GroupStat, WinrateSummary


Format = Literal["json", "csv"]


# 列顺序与 PR 实施计划 §5.3 一致；下游 Excel 复盘模板可对齐这套列名。
CSV_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "next_trade_date",
    "ts_code",
    "name",
    "prediction",
    "rank",
    "continuation_score",
    "confidence",
    "t_close_price",
    "t1_open_price",
    "open_vs_limit_pct",
    "outcome",
    "run_id",
    "lgb_score",
    "lgb_decile",
)


@dataclass(frozen=True)
class ExportPayload:
    """Fully composed export payload — held in memory just long enough to
    serialize to disk. Tests can construct one directly without writing a
    file."""

    generated_at: str
    window: dict[str, str]
    summary: dict[str, object]
    by_prediction: list[dict[str, object]]
    records: list[dict[str, object]]


# ---------------------------------------------------------------------------
# Build payload
# ---------------------------------------------------------------------------


def _summary_to_dict(s: WinrateSummary) -> dict[str, object]:
    return {
        "total": s.total,
        "resolved": s.resolved,
        "unresolved": s.unresolved,
        "win": s.win,
        "flat": s.flat,
        "loss": s.loss,
        "strict_win_rate": s.strict_win_rate,
        "non_loss_rate": s.non_loss_rate,
        "avg_open_vs_limit_pct": s.avg_open_vs_limit_pct,
    }


def _group_to_dict(g: GroupStat) -> dict[str, object]:
    return {
        "key": g.key,
        "total": g.total,
        "resolved": g.resolved,
        "win": g.win,
        "flat": g.flat,
        "loss": g.loss,
        "strict_win_rate": g.strict_win_rate,
        "avg_open_vs_limit_pct": g.avg_open_vs_limit_pct,
    }


def _record_to_dict(r: ResolvedRecord) -> dict[str, object]:
    rec = r.record
    return {
        "trade_date": rec.trade_date,
        "next_trade_date": rec.next_trade_date,
        "ts_code": rec.ts_code,
        "name": rec.name,
        "prediction": rec.prediction,
        "rank": rec.rank,
        "continuation_score": rec.continuation_score,
        "confidence": rec.confidence,
        "t_close_price": rec.t_close_price,
        "t1_open_price": r.t1_open_price,
        "open_vs_limit_pct": r.open_vs_limit_pct,
        "outcome": r.outcome,
        "run_id": rec.run_id,
        "lgb_score": rec.lgb_score,
        "lgb_decile": rec.lgb_decile,
    }


def build_payload(
    *,
    window_start: str,
    window_end: str,
    summary: WinrateSummary,
    by_prediction: list[GroupStat],
    resolved: list[ResolvedRecord],
    generated_at: datetime | None = None,
) -> ExportPayload:
    return ExportPayload(
        generated_at=(generated_at or datetime.now()).strftime("%Y-%m-%dT%H:%M:%S"),
        window={"start": window_start, "end": window_end},
        summary=_summary_to_dict(summary),
        by_prediction=[_group_to_dict(g) for g in by_prediction],
        records=[_record_to_dict(r) for r in resolved],
    )


# ---------------------------------------------------------------------------
# Format selection + writers
# ---------------------------------------------------------------------------


def infer_format(output_path: str, explicit: str | None = None) -> Format:
    """Resolve output format.

    Priority: explicit ``--format`` flag > file extension > default ``json``.
    """
    if explicit:
        ex = explicit.lower()
        if ex in ("json", "csv"):
            return ex  # type: ignore[return-value]
        raise ValueError(f"unsupported --format: {explicit}; must be json or csv")
    low = output_path.lower()
    if low.endswith(".csv"):
        return "csv"
    if low.endswith(".json"):
        return "json"
    return "json"


def serialize_json(payload: ExportPayload) -> str:
    return json.dumps(
        {
            "generated_at": payload.generated_at,
            "window": payload.window,
            "summary": payload.summary,
            "by_prediction": payload.by_prediction,
            "records": payload.records,
        },
        ensure_ascii=False,
        indent=2,
    )


def serialize_csv(payload: ExportPayload) -> str:
    """Render逐股 records to CSV. Summary / by_prediction not included in CSV
    by design — that's what JSON is for. CSV is the per-row复盘 friendly form."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for r in payload.records:
        writer.writerow(r)
    return buf.getvalue()


def write_to_disk(payload: ExportPayload, output_path: str, fmt: Format) -> None:
    """Write payload to disk in the requested format.

    File is opened with utf-8 + newline='' so DictWriter doesn't emit
    spurious blank rows on Windows.
    """
    if fmt == "json":
        text = serialize_json(payload)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return
    # csv
    text = serialize_csv(payload)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
