"""Renderers: section schema → markdown + reports_dir write helpers (§5.5).

Per design §5.5.1 the reports directory layout is::

    ~/.deeptrade/reports/<run_id>/
        summary.md
        overview.md
        sectors.md
        sentiment.md
        capital.md
        leaders.md
        style.md
        risk_outlook.md
        metrics.json
        llm_calls.jsonl
        input_fingerprint.txt
        summary.json          (← PR-5 contract file, not generated here)

PR-4 owns the per-section ``*.md`` rendering + ``summary.md`` top-level
index + the local-audit ``metrics.json`` / ``llm_calls.jsonl`` dumps.
PR-5 builds ``summary.json`` (the upload-to-website contract) on top.

The renderers are pure (string in / string out) — file IO is split into
explicit ``write_*`` helpers so tests can assert markdown shape without
touching the filesystem.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .pipeline import SectionResult
from .schemas import (
    SECTION_ORDER,
    CapitalSection,
    EvidenceItem,
    Finding,
    LeadersSection,
    OverviewSection,
    RiskOutlookSection,
    SectionBase,
    SectionName,
    SectorsSection,
    SentimentSection,
    StyleSection,
)


SECTION_TITLES: dict[SectionName, str] = {
    "overview": "大盘整体",
    "sectors": "板块轮动",
    "sentiment": "情绪与赚钱效应",
    "capital": "资金面",
    "leaders": "龙头梳理",
    "style": "风格切换",
    "risk_outlook": "风险与展望",
}

# Source-of-truth filename per section (used by summary.md links + by
# :func:`write_section_files`). Single-source-of-truth means tests can pin
# the contract.
SECTION_FILES: dict[SectionName, str] = {
    "overview": "overview.md",
    "sectors": "sectors.md",
    "sentiment": "sentiment.md",
    "capital": "capital.md",
    "leaders": "leaders.md",
    "style": "style.md",
    "risk_outlook": "risk_outlook.md",
}


# ---------------------------------------------------------------------------
# Per-section markdown renderers
# ---------------------------------------------------------------------------


def render_section_md(section: SectionName, result: SectionResult) -> str:
    """Build the per-section markdown body."""
    schema = result.schema
    title = SECTION_TITLES[section]
    header = f"# {title}\n"
    if result.error:
        header += f"\n> ⚠ LLM 调用失败：{result.error}\n"

    if section == "overview":
        body = _render_overview(schema)  # type: ignore[arg-type]
    elif section == "sectors":
        body = _render_sectors(schema)  # type: ignore[arg-type]
    elif section == "sentiment":
        body = _render_sentiment(schema)  # type: ignore[arg-type]
    elif section == "capital":
        body = _render_capital(schema)  # type: ignore[arg-type]
    elif section == "leaders":
        body = _render_leaders(schema)  # type: ignore[arg-type]
    elif section == "style":
        body = _render_style(schema)  # type: ignore[arg-type]
    elif section == "risk_outlook":
        body = _render_risk_outlook(schema)  # type: ignore[arg-type]
    else:
        body = _render_generic(schema)

    return header + "\n" + body


def _render_narrative_and_findings(schema: SectionBase) -> str:
    parts: list[str] = []
    if schema.narrative_md:
        parts.append(schema.narrative_md.strip())
    if schema.findings:
        parts.append("\n## 关键观察\n" + _render_findings(schema.findings))
    return "\n\n".join(parts) + "\n"


def _render_findings(findings: list[Finding]) -> str:
    lines: list[str] = []
    for f in findings:
        lines.append(f"- **{f.headline}** _(severity: {f.severity})_  ")
        lines.append(f"  {f.detail}")
        if f.evidence:
            lines.append("  - evidence:")
            for ev in f.evidence:
                lines.append(f"    - {_render_evidence(ev)}")
    return "\n".join(lines)


def _render_evidence(ev: EvidenceItem) -> str:
    val_display = "null" if ev.value is None else f"{ev.value}"
    unit_display = "" if ev.unit == "none" else ev.unit
    return f"`{ev.field}` = {val_display} {unit_display} — {ev.interpretation}"


def _render_overview(schema: OverviewSection) -> str:
    parts = [f"> **市场态势**：{schema.market_tone}"]
    if schema.theme_tags:
        parts.append("**主题词**: " + " · ".join(f"`{t}`" for t in schema.theme_tags))
    parts.append(_render_narrative_and_findings(schema))
    if schema.headline_metrics:
        parts.append("## 核心指标\n")
        parts.append("| 指标 | 值 | 单位 | 说明 |")
        parts.append("| --- | --- | --- | --- |")
        for m in schema.headline_metrics:
            val = "—" if m.value is None else m.value
            interp = m.interpretation or ""
            unit = "" if m.unit == "none" else m.unit
            parts.append(f"| {m.label} | {val} | {unit} | {interp} |")
    return "\n".join(parts) + "\n"


def _render_sectors(schema: SectorsSection) -> str:
    parts: list[str] = [_render_narrative_and_findings(schema)]
    parts.append(f"_板块体系：`{schema.provider}`_\n")
    if schema.today_top:
        parts.append("## 今日 Top\n")
        parts.append(_sector_table(schema.today_top))
    if schema.range_top:
        parts.append("## 区间 Top\n")
        parts.append(_sector_table(schema.range_top))
    cls = schema.classification
    parts.append("## 分类\n")
    parts.append("### 新主线候选\n" + _sector_bullets(cls.new_mainline))
    parts.append("### 接力\n" + _sector_bullets(cls.relay))
    parts.append("### 退潮\n" + _sector_bullets(cls.fading))
    if schema.rotation_commentary:
        parts.append("## 轮动评注\n\n" + schema.rotation_commentary.strip())
    return "\n".join(parts) + "\n"


def _sector_table(entries) -> str:
    lines = ["| 板块 | 涨幅 (%) | 资金净流入 (亿) | 涨停数 | 持续性天数 |",
             "| --- | --- | --- | --- | --- |"]
    for e in entries:
        net = "—" if e.net_inflow_yi is None else f"{e.net_inflow_yi:.2f}"
        lu = "—" if e.limit_up_count is None else str(e.limit_up_count)
        days = "—" if e.persistence_days is None else str(e.persistence_days)
        lines.append(f"| {e.name} | {e.pct_chg:.2f} | {net} | {lu} | {days} |")
    return "\n".join(lines) + "\n"


def _sector_bullets(entries) -> str:
    if not entries:
        return "_(无)_\n"
    return "\n".join(f"- {e.name} — {e.pct_chg:.2f}%" for e in entries) + "\n"


def _render_sentiment(schema: SentimentSection) -> str:
    parts: list[str] = [_render_narrative_and_findings(schema)]
    parts.append(f"_平均情绪得分：{schema.avg_score:.2f} / 100_  ")
    parts.append(f"_最强日：{schema.strongest_day or '—'} · 最弱日：{schema.weakest_day or '—'}_  ")
    parts.append(f"_赚钱效应：{schema.money_effect} · 亏钱效应：{schema.losing_effect}_\n")
    if schema.series:
        parts.append("## 每日情绪温度\n")
        parts.append("| 日期 | 分数 | 涨/跌 | 中位涨幅 (%) | 涨停 / 跌停 / 炸板 |")
        parts.append("| --- | --- | --- | --- | --- |")
        for s in schema.series:
            parts.append(
                f"| {s.trade_date} | {s.score_of_100:.1f} | {s.n_up}/{s.n_down} | "
                f"{s.median_pct_chg:.2f} | {s.n_limit_up}/{s.n_limit_down}/{s.n_zhaban} |"
            )
    return "\n".join(parts) + "\n"


def _render_capital(schema: CapitalSection) -> str:
    parts: list[str] = [_render_narrative_and_findings(schema)]
    if schema.north_summary:
        parts.append("## 北向资金 每日\n")
        parts.append("| 日期 | 北向 (亿) | 主力净额 (亿) |")
        parts.append("| --- | --- | --- |")
        for r in schema.north_summary:
            north = "—" if r.north_money_yi is None else f"{r.north_money_yi:.2f}"
            main = "—" if r.main_net_inflow_yi is None else f"{r.main_net_inflow_yi:.2f}"
            parts.append(f"| {r.trade_date} | {north} | {main} |")
    if schema.industry_top:
        parts.append("## 行业资金 Top\n")
        parts.append(_flow_table(schema.industry_top))
    if schema.concept_top:
        parts.append("## 概念资金 Top\n")
        parts.append(_flow_table(schema.concept_top))
    if schema.stock_top:
        parts.append("## 个股资金净流入 Top\n")
        parts.append(_leader_flow_table(schema.stock_top))
    if schema.stock_bottom:
        parts.append("## 个股资金净流出 Top\n")
        parts.append(_leader_flow_table(schema.stock_bottom))
    if schema.lhb_highlights:
        parts.append("## 龙虎榜亮点\n")
        parts.append(_render_findings(schema.lhb_highlights))
    return "\n".join(parts) + "\n"


def _flow_table(rows) -> str:
    lines = ["| 名称 | 净流入 (亿) | 涨幅 (%) |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r.name} | {r.net_inflow_yi:.2f} | {r.pct_chg:.2f} |")
    return "\n".join(lines) + "\n"


def _leader_flow_table(rows) -> str:
    lines = ["| ts_code | 名称 | 净流入 (亿) |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r.ts_code} | {r.name} | {r.net_inflow_yi:.2f} |")
    return "\n".join(lines) + "\n"


def _render_leaders(schema: LeadersSection) -> str:
    parts: list[str] = [_render_narrative_and_findings(schema)]
    parts.append(f"_最低入选分：{schema.min_score:.1f}_  \n")
    if schema.primary:
        parts.append("## 一线龙头 (primary)\n")
        parts.append(_leader_table(schema.primary))
    if schema.secondary:
        parts.append("## 二线接力 (secondary)\n")
        parts.append(_leader_table(schema.secondary))
    if schema.sector_map:
        parts.append("## 板块 → 龙头映射\n")
        for sector, codes in schema.sector_map.items():
            parts.append(f"- **{sector}**：{', '.join(codes)}")
        parts.append("")
    return "\n".join(parts) + "\n"


def _leader_table(candidates) -> str:
    lines = [
        "| ts_code | 名称 | 总分 | 梯队 | 涨幅 | 资金 | 题材 | 连板高度 | 区间涨幅 (%) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in candidates:
        b = c.score_breakdown or {}
        ladder_n = "—" if c.ladder_height is None else str(c.ladder_height)
        rng = "—" if c.range_pct_chg is None else f"{c.range_pct_chg:.2f}"
        lines.append(
            f"| {c.ts_code} | {c.name} | {c.score:.1f} | "
            f"{b.get('ladder', 0):.1f} | {b.get('return', 0):.1f} | "
            f"{b.get('capital', 0):.1f} | {b.get('theme', 0):.1f} | "
            f"{ladder_n} | {rng} |"
        )
    return "\n".join(lines) + "\n"


def _render_style(schema: StyleSection) -> str:
    parts: list[str] = [_render_narrative_and_findings(schema)]
    parts.append(f"_主导风格：`{schema.dominant_style}` · 是否切换：`{schema.flip_signal}`_\n")
    if schema.range_summary:
        parts.append("## 区间汇总\n")
        for k, v in schema.range_summary.items():
            parts.append(f"- **{k}**: {v}")
        parts.append("")
    if schema.series:
        parts.append("## 每日风格\n")
        parts.append("| 日期 | 大盘 (%) | 小盘 (%) | 成长 (%) | 大/小 累计 |")
        parts.append("| --- | --- | --- | --- | --- |")
        for s in schema.series:
            growth = "—" if s.growth_ret is None else f"{s.growth_ret:.2f}"
            parts.append(
                f"| {s.trade_date} | {s.large_cap_ret:.2f} | {s.small_cap_ret:.2f} | "
                f"{growth} | {s.big_to_small_ratio:.2f} |"
            )
    return "\n".join(parts) + "\n"


def _render_risk_outlook(schema: RiskOutlookSection) -> str:
    parts: list[str] = [_render_narrative_and_findings(schema)]
    parts.append(f"_总体风险等级：**{schema.overall_risk}**_\n")
    if schema.signals:
        parts.append("## 风险信号\n")
        parts.append("| 名称 | 触发 | 等级 | 说明 |")
        parts.append("| --- | --- | --- | --- |")
        for s in schema.signals:
            parts.append(f"| {s.name} | {'✓' if s.triggered else ' '} | {s.severity} | {s.detail} |")
    if schema.hypotheses:
        parts.append("\n## 展望与假设\n")
        for h in schema.hypotheses:
            parts.append(f"### {h.title}")
            parts.append(h.rationale)
            parts.append("**观察点**:")
            for wp in h.watch_points:
                parts.append(f"- {wp}")
            parts.append("**可证伪条件**:")
            for ft in h.fail_triggers:
                parts.append(f"- {ft}")
            parts.append("")
    return "\n".join(parts) + "\n"


def _render_generic(schema: SectionBase) -> str:
    """Fallback for any SectionBase subclass without a dedicated renderer."""
    return _render_narrative_and_findings(schema)


# ---------------------------------------------------------------------------
# Summary + audit writers
# ---------------------------------------------------------------------------


def render_summary_md(
    *,
    run_id: str,
    window: Any,
    results: dict[SectionName, SectionResult],
) -> str:
    """Top-level summary.md (design §5.5.2). Brief — one-liner + section links.

    The richer summary.json (PR-5) carries the full structured contract;
    this markdown is for local terminal rendering / human readers.
    """
    overview = results.get("overview")
    one_liner = ""
    market_tone = "未知"
    if overview is not None and isinstance(overview.schema, OverviewSection):
        market_tone = overview.schema.market_tone
        # Take the first 100 chars of the narrative as a one-liner.
        nm = overview.schema.narrative_md or ""
        one_liner = nm.split("\n", 1)[0].strip()[:100]

    span = (
        f"区间：{window.start} → {window.end}（{window.n_days} 个交易日）"
        if window.mode == "range"
        else f"单日：{window.anchor}"
    )

    failed = [s for s, r in results.items() if r.error]
    failed_banner = ""
    if failed:
        failed_banner = (
            f"\n> ⚠ 部分 section LLM 调用失败：{', '.join(failed)}（PARTIAL）\n"
        )

    lines: list[str] = [
        f"# 市场复盘 — {window.anchor}",
        f"_{span}_",
        f"\n> run_id: `{run_id}` · market_tone: **{market_tone}**",
        failed_banner,
        f"\n## 一句话结论\n\n{one_liner or '（暂无概述）'}\n",
        "\n## 章节",
    ]
    for section in SECTION_ORDER:
        title = SECTION_TITLES[section]
        filename = SECTION_FILES[section]
        marker = " ⚠" if section in failed else ""
        lines.append(f"- [{title}]({filename}){marker}")
    return "\n".join(lines) + "\n"


def write_section_files(
    reports_dir: Path,
    results: dict[SectionName, SectionResult],
) -> dict[SectionName, Path]:
    """Write each section's markdown + return the filename map."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out: dict[SectionName, Path] = {}
    for section in SECTION_ORDER:
        if section not in results:
            continue
        path = reports_dir / SECTION_FILES[section]
        path.write_text(render_section_md(section, results[section]), encoding="utf-8")
        out[section] = path
    return out


def write_summary_md(
    reports_dir: Path,
    *,
    run_id: str,
    window: Any,
    results: dict[SectionName, SectionResult],
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "summary.md"
    path.write_text(
        render_summary_md(run_id=run_id, window=window, results=results),
        encoding="utf-8",
    )
    return path


def dump_metrics_json(
    reports_dir: Path,
    bundle: Any,  # MetricsBundle — typed as Any to avoid circular import
) -> Path:
    """Serialize PR-3 metric reviews into ``metrics.json`` for local audit.

    Each field of ``bundle`` is converted via ``dataclasses.asdict`` so the
    output is JSON-serializable. ``frozenset`` / non-JSON types raise here;
    PR-3 dataclasses are all JSON-safe by design.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "metrics.json"
    payload = {
        "window": _safe_asdict(bundle.window),
        "breadth": _safe_asdict(bundle.breadth),
        "sentiment": _safe_asdict(bundle.sentiment),
        "capital": _safe_asdict(bundle.capital),
        "sectors": _safe_asdict(bundle.sectors),
        "leaders": _safe_asdict(bundle.leaders),
        "style": _safe_asdict(bundle.style),
        "risk": _safe_asdict(bundle.risk),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def dump_llm_calls_audit(
    reports_dir: Path,
    results: dict[SectionName, SectionResult],
) -> Path:
    """One JSONL row per section: section / meta (input_tokens / latency_ms /
    prompt_hash / ...) / error. Prompt content itself is NOT dumped here —
    framework's replay cache holds the canonical record."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "llm_calls.jsonl"
    rows: list[str] = []
    for section in SECTION_ORDER:
        if section not in results:
            continue
        r = results[section]
        row = {
            "section": section,
            "meta": _safe_jsonable(r.meta),
            "error": r.error,
        }
        rows.append(json.dumps(row, ensure_ascii=False, default=str))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _safe_asdict(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return _safe_jsonable(asdict(obj))
    return _safe_jsonable(obj)


def _safe_jsonable(obj: Any) -> Any:
    """Recursively coerce frozensets / tuples to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v) for v in obj]
    if isinstance(obj, frozenset):
        return sorted(_safe_jsonable(v) for v in obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Terminal summary (design §5.5.3)
# ---------------------------------------------------------------------------


def render_terminal_summary(report: Any) -> str:
    """Short post-run console summary — design §5.5.3.

    Just the headline + section roster + DB / report paths. ``report`` is a
    :class:`ReviewReportSchema` (PR-5); typed as ``Any`` here to avoid a
    cyclic ``report → render → report`` import.
    """
    meta = report.meta
    headline = report.headline
    lines: list[str] = [
        f"# {meta.title}",
        f"_run_id: `{meta.run_id}` · status: **{meta.status}** · "
        f"market_tone: **{headline.market_tone}**_",
    ]
    if meta.failed_sections:
        lines.append(f"⚠ failed sections: {', '.join(meta.failed_sections)}")
    lines.append(f"\n## 一句话结论\n\n{headline.one_liner}\n")
    if headline.core_metrics:
        lines.append("## 核心指标\n")
        lines.append("| 指标 | 值 |")
        lines.append("| --- | --- |")
        for m in headline.core_metrics:
            val = "—" if m.value is None else m.value
            unit = "" if m.unit == "none" else m.unit
            lines.append(f"| {m.label} | {val} {unit} |")
    lines.append("\n## 章节")
    for section in SECTION_ORDER:
        title = SECTION_TITLES[section]
        marker = " ⚠" if section in meta.failed_sections else ""
        lines.append(f"- {title}{marker}")
    return "\n".join(lines) + "\n"
