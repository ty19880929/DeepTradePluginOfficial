"""Render & export the limit-up-board final report.

DESIGN §12.8.3 + the v0.3.1 banner / S5 rules:
    * partial_failed / failed / cancelled  → red banner at top of summary.md
    * is_intraday=True                     → yellow `INTRADAY MODE` banner
    * Both stack
    * round2_predictions.json contains ALL R2 predictions (with batch_local_rank)
    * round2_final_ranking.json only emitted when R2 was multi-batch
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deeptrade.core import paths
from deeptrade.core.run_status import RunStatus

from .data import Round1Bundle
from .schemas import (
    ContinuationCandidate,
    FinalRankingResponse,
    StrongCandidate,
)

if TYPE_CHECKING:  # pragma: no cover
    from .runner import ProviderDebateResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def render_banners(
    *,
    status: RunStatus,
    is_intraday: bool,
    failed_batch_ids: list[str] | None = None,
) -> str:
    """Top-of-report banner stack — markdown blockquote style.

    F-L3: when ``status == PARTIAL_FAILED`` and ``failed_batch_ids`` is
    non-empty, the banner enumerates which batches failed so users don't have
    to grep ``llm_calls.jsonl`` to find them.
    """
    parts: list[str] = []
    if status in {RunStatus.PARTIAL_FAILED, RunStatus.FAILED, RunStatus.CANCELLED}:
        marker = {
            RunStatus.PARTIAL_FAILED: "🚨 **PARTIAL — 本次结果不完整，不可作为有效筛选结果**",
            RunStatus.FAILED: "🚨 **FAILED — 运行失败**",
            RunStatus.CANCELLED: "⏹ **CANCELLED — 用户中断**",
        }[status]
        parts.append(f"> {marker}")
        if status == RunStatus.PARTIAL_FAILED and failed_batch_ids:
            parts.append(f"> 失败批次：`{', '.join(failed_batch_ids)}`（详见 `llm_calls.jsonl`）")
    if is_intraday:
        parts.append("> ⚠ **INTRADAY MODE** — 数据可能不完整，仅供盘中观察，不可与日终结果混用")
    return "\n".join(parts) + ("\n\n" if parts else "")


# ---------------------------------------------------------------------------
# Markdown body
# ---------------------------------------------------------------------------


def render_summary_md(
    *,
    status: RunStatus,
    is_intraday: bool,
    bundle: Round1Bundle,
    selected: list[StrongCandidate],
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None,
    failed_batch_ids: list[str] | None = None,
) -> str:
    """Build the full summary.md content."""
    out = [
        render_banners(status=status, is_intraday=is_intraday, failed_batch_ids=failed_batch_ids)
    ]
    out.append("# 打板策略报告\n")
    out.append(
        f"- trade_date: **{bundle.trade_date}**\n"
        f"- next_trade_date: **{bundle.next_trade_date}**\n"
        f"- status: `{status.value}`\n"
        f"- intraday: `{is_intraday}`\n"
        f"- lgb_model_id: {_lgb_model_id_repr(bundle)}\n"
    )

    # Sector strength source label is meaningful — surface it.
    out.append(
        f"\n*sector_strength_source*: `{bundle.sector_strength.source}`  "
        f"_(可信度：limit_cpt_list > lu_desc_aggregation > industry_fallback)_\n"
    )

    if bundle.data_unavailable:
        out.append(f"\n*data_unavailable*: `{bundle.data_unavailable}`\n")

    # ----- R1 -----
    out.append(f"\n## R1 强势标的（{len(selected)}/{len(bundle.candidates)} selected）\n")
    if selected:
        out.append("| Rank | Code | Name | T收盘 (元) | Score | LGB | Level | Theme/Industry | Rationale |\n")
        out.append("|------|------|------|-----------|-------|----:|-------|----------------|-----------|\n")
        for i, c in enumerate(selected, 1):
            theme = _industry_for(c.candidate_id, bundle.candidates)
            out.append(
                f"| {i} | `{c.ts_code}` | {c.name} | {_close_for(c.candidate_id, bundle.candidates)} | "
                f"{c.score:.1f} | {_lgb_cell(c.candidate_id, bundle.candidates)} | {c.strength_level} | {theme} | {c.rationale} |\n"
            )
    else:
        out.append("_(本轮无强势标的)_\n")

    # ----- R2 / Final -----
    if predictions:
        if final_ranking is not None:
            out.append("\n## 次日连板预测（按 final_rank 排序）\n")
            out.append("| # | Code | Name | T收盘 (元) | LGB | Final Pred | Conf. | Δ vs batch | Reason |\n")
            out.append("|---|------|------|-----------|----:|-----------|-------|-----------|--------|\n")
            for fi in sorted(final_ranking.finalists, key=lambda f: f.final_rank):
                out.append(
                    f"| {fi.final_rank} | `{fi.ts_code}` | "
                    f"{_name_for(fi.candidate_id, predictions)} | "
                    f"{_close_for(fi.candidate_id, bundle.candidates)} | "
                    f"{_lgb_cell(fi.candidate_id, bundle.candidates)} | "
                    f"{fi.final_prediction} | {fi.final_confidence} | "
                    f"{fi.delta_vs_batch} | {fi.reason_vs_peers} |\n"
                )
        else:
            out.append("\n## 次日连板预测（单批）\n")
            out.append("| Rank | Code | Name | T收盘 (元) | Score | LGB | Conf. | Pred | Rationale |\n")
            out.append("|------|------|------|-----------|-------|----:|-------|------|-----------|\n")
            for p in sorted(predictions, key=lambda x: x.rank):
                out.append(
                    f"| {p.rank} | `{p.ts_code}` | {p.name} | "
                    f"{_close_for(p.candidate_id, bundle.candidates)} | "
                    f"{p.continuation_score:.1f} | "
                    f"{_lgb_cell(p.candidate_id, bundle.candidates)} | "
                    f"{p.confidence} | "
                    f"{p.prediction} | {p.rationale} |\n"
                )
    else:
        out.append("\n## 次日连板预测\n_(本轮无候选标的)_\n")

    out.append("\n---\n*免责声明：本报告仅用于策略研究，不构成投资建议。*\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Report directory writer
# ---------------------------------------------------------------------------


def write_report(
    run_id: str,
    *,
    status: RunStatus,
    is_intraday: bool,
    bundle: Round1Bundle,
    selected: list[StrongCandidate],
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None,
    extra_files: dict[str, str] | None = None,
    reports_root: Path | None = None,
    failed_batch_ids: list[str] | None = None,
    debate_results: list[ProviderDebateResult] | None = None,
) -> Path:
    """Write the report directory and return its path.

    In single-LLM mode the existing 6-file layout is preserved. In debate
    mode, ``selected`` / ``predictions`` / ``final_ranking`` are typically
    empty and the per-provider results are persisted under
    ``debate/<provider>/`` plus a 《多 LLM 辩论结果》 section in summary.md.
    """
    root = (reports_root or paths.reports_dir()) / str(run_id)
    root.mkdir(parents=True, exist_ok=True)

    # 1. summary.md
    if debate_results:
        md = render_debate_summary_md(
            status=status,
            is_intraday=is_intraday,
            bundle=bundle,
            results=debate_results,
            failed_batch_ids=failed_batch_ids,
        )
    else:
        md = render_summary_md(
            status=status,
            is_intraday=is_intraday,
            bundle=bundle,
            selected=selected,
            predictions=predictions,
            final_ranking=final_ranking,
            failed_batch_ids=failed_batch_ids,
        )
    (root / "summary.md").write_text(md, encoding="utf-8")

    # 2. round1_strong_targets.json
    (root / "round1_strong_targets.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3. round2_predictions.json (ALL predictions, with batch_local_rank)
    cand_by_id = {c.get("candidate_id"): c for c in bundle.candidates}
    r2_out = []
    for p in predictions:
        rec = p.model_dump(mode="json")
        rec["batch_local_rank"] = p.rank  # explicit alias for downstream tools
        if final_ranking is not None:
            match = next(
                (f for f in final_ranking.finalists if f.candidate_id == p.candidate_id),
                None,
            )
            if match is not None:
                rec["final_rank"] = match.final_rank
                rec["delta_vs_batch"] = match.delta_vs_batch
        # v0.5 — surface the candidate's LGB attribution in the JSON export
        # (see lightgbm_design.md §11.3). Always present (None when LGB
        # disabled / not loaded for this run) so downstream tools have a
        # stable schema.
        src = cand_by_id.get(p.candidate_id, {})
        rec["lgb_score"] = src.get("lgb_score")
        rec["lgb_decile"] = src.get("lgb_decile")
        rec["lgb_model_id"] = bundle.lgb_model_id
        r2_out.append(rec)
    (root / "round2_predictions.json").write_text(
        json.dumps(r2_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. round2_final_ranking.json (only when multi-batch / final_ranking ran)
    if final_ranking is not None:
        (root / "round2_final_ranking.json").write_text(
            json.dumps(final_ranking.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 5. data_snapshot.json
    snapshot: dict[str, Any] = {
        "trade_date": bundle.trade_date,
        "next_trade_date": bundle.next_trade_date,
        "status": status.value,
        "is_intraday": is_intraday,
        "candidates": bundle.candidates,
        "market_summary": bundle.market_summary,
        "sector_strength": asdict(bundle.sector_strength),
        "data_unavailable": bundle.data_unavailable,
        "debate_mode": bool(debate_results),
        "debate_providers": (
            [r.provider for r in debate_results] if debate_results else []
        ),
    }
    (root / "data_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 6. llm_calls.jsonl is written incrementally by the runner; we touch it
    # here so the file always exists in the report dir.
    (root / "llm_calls.jsonl").touch(exist_ok=True)

    # 7. debate/ — per-provider details
    if debate_results:
        debate_dir = root / "debate"
        debate_dir.mkdir(parents=True, exist_ok=True)
        for r in debate_results:
            pdir = debate_dir / r.provider
            pdir.mkdir(parents=True, exist_ok=True)
            if r.r1_result and r.r1_result.selected:
                (pdir / "round1_strong_targets.json").write_text(
                    json.dumps(
                        [s.model_dump(mode="json") for s in r.r1_result.selected],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if r.r2_result and r.r2_result.predictions:
                (pdir / "round2_initial.json").write_text(
                    json.dumps(
                        [p.model_dump(mode="json") for p in r.r2_result.predictions],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if r.final_initial is not None:
                (pdir / "round2_final_ranking.json").write_text(
                    json.dumps(
                        r.final_initial.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if r.revision and r.revision.revised:
                (pdir / "round2_revised.json").write_text(
                    json.dumps(
                        {
                            "revision_summary": r.revision.revision_summary,
                            "candidates": [
                                c.model_dump(mode="json") for c in r.revision.revised
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if r.error:
                (pdir / "error.txt").write_text(r.error, encoding="utf-8")

    # Caller-supplied extras (e.g. R1/R2 raw responses captured during run)
    if extra_files:
        for name, content in extra_files.items():
            (root / name).write_text(content, encoding="utf-8")

    return root


# ---------------------------------------------------------------------------
# Debate-mode markdown
# ---------------------------------------------------------------------------


_PRED_GLYPH = {"top_candidate": "▲", "watchlist": "○", "avoid": "▽"}


def _glyph(pred: str | None) -> str:
    return _PRED_GLYPH.get(pred or "", "—")


def render_debate_summary_md(
    *,
    status: RunStatus,
    is_intraday: bool,
    bundle: Round1Bundle,
    results: list[ProviderDebateResult],
    failed_batch_ids: list[str] | None = None,
) -> str:
    """Build summary.md content for debate-mode runs.

    Layout:
      Banners
      Header
      §1 参与 LLM 与状态
      §2 每个 LLM 的 initial → revised 对照表
      §3 跨 LLM 分歧矩阵 — Initial
      §4 跨 LLM 分歧矩阵 — Revised
      §5 各 LLM 的 revision_summary
    """
    out: list[str] = [
        render_banners(status=status, is_intraday=is_intraday, failed_batch_ids=failed_batch_ids)
    ]
    out.append("# 打板策略报告（多 LLM 辩论）\n")
    out.append(
        f"- trade_date: **{bundle.trade_date}**\n"
        f"- next_trade_date: **{bundle.next_trade_date}**\n"
        f"- status: `{status.value}`\n"
        f"- intraday: `{is_intraday}`\n"
        f"- providers: {', '.join(f'`{r.provider}`' for r in results)}\n"
        f"- lgb_model_id: {_lgb_model_id_repr(bundle)}\n"
    )
    out.append(
        f"\n*sector_strength_source*: `{bundle.sector_strength.source}`  "
        f"_(可信度：limit_cpt_list > lu_desc_aggregation > industry_fallback)_\n"
    )
    if bundle.data_unavailable:
        out.append(f"\n*data_unavailable*: `{bundle.data_unavailable}`\n")

    # ----- §1 参与 LLM 状态 ------------------------------------------------
    out.append("\n## 1. 参与 LLM 状态\n\n")
    out.append("| Provider | R1 入选 | R2 预测 | R2 失败批 | final_ranking | R3 修订 | Error |\n")
    out.append("|----------|--------:|--------:|---------:|:-------------:|:------:|-------|\n")
    for r in results:
        n_r1 = len(r.r1_result.selected) if r.r1_result else 0
        n_r2 = len(r.r2_result.predictions) if r.r2_result else 0
        r2_fails = (
            r.r2_result.failed_batches if r.r2_result and r.r2_result.failed_batches else 0
        )
        if not r.final_attempted:
            fr = "—"
        elif r.final_initial is not None:
            fr = "✔"
        else:
            fr = "✘"
        if r.revision is None:
            rv = "—"
        elif r.revision.success:
            rv = "✔"
        else:
            rv = "✘"
        err = (r.error or "").replace("|", "/")[:60]
        out.append(
            f"| `{r.provider}` | {n_r1} | {n_r2} | {r2_fails} | {fr} | {rv} | {err} |\n"
        )

    # ----- §2 每个 LLM 的 initial → revised 对照 ----------------------------
    for r in results:
        out.append(f"\n## 2.{results.index(r) + 1} `{r.provider}` 修订对照\n")
        if r.error:
            out.append(f"\n_(provider {r.provider} 整体失败: {r.error})_\n")
            continue
        initial = r.initial_predictions
        revised = {c.candidate_id: c for c in r.revised_predictions}
        if not initial:
            out.append("\n_(无连板预测候选)_\n")
            continue
        if r.revision and r.revision.revision_summary:
            out.append(f"\n**revision_summary**: {r.revision.revision_summary}\n")
        elif r.revision and not r.revision.success:
            out.append(
                f"\n_(R3 修订失败: {r.revision.error or '未知'}；下表 Revised 列回退为初始判断)_\n"
            )
        out.append(
            "\n| # | Code | Name | T收盘 (元) | Initial Pred | Init Score | Revised Pred | "
            "Rev Score | Δ | Revision Note |\n"
        )
        out.append(
            "|---|------|------|-----------|-------------|-----------:|-------------|"
            "----------:|---|---------------|\n"
        )
        # Sort by revised rank if available; else by initial rank.
        def _sort_by_revised_rank(
            p: ContinuationCandidate, _rev: dict = revised
        ) -> int:
            rev = _rev.get(p.candidate_id)
            return rev.rank if rev else p.rank
        for p in sorted(initial, key=_sort_by_revised_rank):
            rev = revised.get(p.candidate_id)
            init_pred = p.prediction
            init_score = p.continuation_score
            rev_pred = rev.prediction if rev else init_pred
            rev_score = rev.continuation_score if rev else init_score
            delta = _delta_label(init_pred, rev_pred, init_score, rev_score) if rev else "—"
            note = (rev.revision_note if rev else "—").replace("|", "/")
            out.append(
                f"| {rev.rank if rev else p.rank} | `{p.ts_code}` | {p.name} | "
                f"{_close_for(p.candidate_id, bundle.candidates)} | "
                f"{init_pred} | {init_score:.0f} | {rev_pred} | {rev_score:.0f} | "
                f"{delta} | {note} |\n"
            )

    # ----- §3 / §4 分歧矩阵 -------------------------------------------------
    out.append("\n## 3. 跨 LLM 分歧矩阵 — Initial\n\n")
    out.append(_render_disagreement_matrix(results, bundle, mode="initial"))
    out.append("\n## 4. 跨 LLM 分歧矩阵 — Revised\n\n")
    out.append(_render_disagreement_matrix(results, bundle, mode="revised"))
    out.append(
        "\n_说明: ▲ = top_candidate, ○ = watchlist, ▽ = avoid, — = 该 LLM 未将此股纳入预测_\n"
    )

    out.append("\n---\n*免责声明：本报告仅用于策略研究，不构成投资建议。*\n")
    return "".join(out)


def _render_disagreement_matrix(
    results: list[ProviderDebateResult],
    bundle: Round1Bundle,
    *,
    mode: str,
) -> str:
    """Build the cross-LLM disagreement matrix in markdown."""
    union_ids: dict[str, tuple[str, str]] = {}
    pred_by_provider: dict[str, dict[str, Any]] = {}
    for r in results:
        preds: list[Any]
        if mode == "initial":
            preds = list(r.initial_predictions)
        else:
            preds = list(r.revised_predictions) or list(r.initial_predictions)
        prov_map: dict[str, Any] = {}
        for p in preds:
            prov_map[p.candidate_id] = p
            union_ids.setdefault(p.candidate_id, (p.ts_code, p.name))
        pred_by_provider[r.provider] = prov_map

    if not union_ids:
        return "_(无候选)_\n"

    # Sort candidates by appearance count (most provider coverage first), then ts_code.
    def coverage(cid: str) -> int:
        return sum(1 for m in pred_by_provider.values() if cid in m)

    sorted_ids = sorted(union_ids.keys(), key=lambda c: (-coverage(c), c))

    headers = ["Code", "Name", "T收盘 (元)"] + [f"`{r.provider}`" for r in results]
    sep = ["------"] * len(headers)
    md = ["| " + " | ".join(headers) + " |\n", "| " + " | ".join(sep) + " |\n"]
    for cid in sorted_ids:
        ts_code, name = union_ids[cid]
        cells = [
            f"`{ts_code}`",
            name,
            _close_for(cid, bundle.candidates),
        ]
        for r in results:
            p = pred_by_provider.get(r.provider, {}).get(cid)
            if p is None:
                cells.append("—")
            else:
                cells.append(f"{_glyph(p.prediction)} {p.continuation_score:.0f}")
        md.append("| " + " | ".join(cells) + " |\n")
    return "".join(md)


def _delta_label(
    init_pred: str, rev_pred: str, init_score: float, rev_score: float
) -> str:
    """Encode the change from initial to revised prediction in one cell."""
    rank = {"avoid": 0, "watchlist": 1, "top_candidate": 2}
    init_r = rank.get(init_pred, 1)
    rev_r = rank.get(rev_pred, 1)
    if rev_r > init_r:
        return f"⬆ +{rev_score - init_score:.0f}"
    if rev_r < init_r:
        return f"⬇ {rev_score - init_score:.0f}"
    diff = rev_score - init_score
    if abs(diff) < 0.5:
        return "= 0"
    return f"= {diff:+.0f}"


def render_terminal_summary(
    run_id: str,
    *,
    reports_root: Path | None = None,
    console: Any = None,
) -> None:
    """Print a concise, friendly summary of a finished run to the terminal.

    Reads from ``reports/<run_id>/`` so it works for both:
      - just-finished runs (called from ``cmd_run`` after the dashboard exits)
      - historical runs (called from ``cmd_report <run_id>``)

    Output sections (only the ones with data are shown):
      - Header line: trade_date / next_trade_date / status / counts
      - "次日重点关注" — R2 top_candidate picks (full table with rationale)
      - "观察仓"        — R2 watchlist (compact, no rationale)
      - "回避"          — R2 avoid (compact)
      - Footer: report directory + how to re-display
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from deeptrade.theme import EVA_THEME

    root = (reports_root or paths.reports_dir()) / str(run_id)
    if not root.is_dir():
        return

    if console is None:
        console = Console(theme=EVA_THEME)

    snap = _safe_load_json(root / "data_snapshot.json", default={})
    r1 = _safe_load_json(root / "round1_strong_targets.json", default=[])
    r2 = _safe_load_json(root / "round2_predictions.json", default=[])
    has_final = (root / "round2_final_ranking.json").is_file()
    debate_mode = bool(snap.get("debate_mode", False))

    trade_date = snap.get("trade_date", "?")
    next_trade_date = snap.get("next_trade_date", "?")
    status = snap.get("status", "unknown")
    is_intraday = bool(snap.get("is_intraday", False))
    candidates = snap.get("candidates", [])
    close_lookup = {c.get("ts_code"): c.get("close_yuan") for c in candidates}
    n_total = len(candidates)
    n_selected = sum(1 for c in r1 if c.get("selected"))

    # ----- Banner / status -------------------------------------------------
    if status in ("partial_failed", "failed", "cancelled"):
        banner_style = "headline.alert" if status == "partial_failed" else "headline.fatal"
        banner_label = {
            "partial_failed": "PARTIAL — 结果不完整",
            "failed": "FAILED — 运行失败",
            "cancelled": "CANCELLED — 用户中断",
        }.get(status, status)
        console.print(Panel(banner_label, style=banner_style, border_style="panel.border.error"))
    if is_intraday:
        console.print(
            Panel(
                "INTRADAY MODE — 数据可能不完整，仅供盘中观察",
                style="headline.alert",
                border_style="panel.border.warn",
            )
        )

    # ----- Header line -----------------------------------------------------
    mode_tag = "辩论" if debate_mode else ""
    console.print(
        f"[title]打板策略{mode_tag}[/title]  "
        f"[k.label]T=[/k.label][k.value]{trade_date}[/k.value]  "
        f"[k.label]T+1=[/k.label][k.value]{next_trade_date}[/k.value]  "
        f"[k.label]入选/候选=[/k.label][k.value]{n_selected}/{n_total}[/k.value]  "
        f"[k.label]状态=[/k.label][status.{'success' if status == 'success' else 'error'}]{status}[/]"
    )

    if debate_mode:
        _render_debate_terminal(console, root, snap, close_lookup)
    elif not r2:
        console.print("[subtitle](本轮无连板预测候选)[/subtitle]")
    else:
        # When final_ranking ran, sort by final_rank; else by rank
        sort_key = "final_rank" if has_final else "rank"
        # Group by prediction
        groups: dict[str, list[dict]] = {"top_candidate": [], "watchlist": [], "avoid": []}
        for p in r2:
            groups.setdefault(p.get("prediction", "watchlist"), []).append(p)
        for g in groups.values():
            g.sort(key=lambda x: x.get(sort_key, x.get("rank", 0)))

        # All three groups share the same table layout; visual hierarchy comes
        # from title/border styles only.
        section_styles = [
            ("top_candidate", "次日重点关注", "title", "panel.border.ok"),
            ("watchlist", "观察仓", "subtitle", "panel.border.primary"),
            ("avoid", "回避", "subtitle", "panel.border.warn"),
        ]
        for key, label, title_style, border_style in section_styles:
            group = groups[key]
            if not group:
                continue
            _render_prediction_table(
                console,
                Table,
                group,
                title=f"{label} · {len(group)} 只",
                title_style=title_style,
                border_style=border_style,
                sort_key=sort_key,
                close_lookup=close_lookup,
            )

    # ----- Footer ---------------------------------------------------------
    console.print(f"\n[k.label]报告目录:[/k.label] [k.value]{root}[/k.value]")
    console.print(
        f"[k.label]完整报告:[/k.label] [k.value]deeptrade strategy report {run_id}[/k.value]  "
        "[subtitle](查看 markdown 全文 + R1 全表 + 数据快照)[/subtitle]"
    )
    console.print("[subtitle]免责声明: 本报告仅用于策略研究，不构成投资建议。[/subtitle]")


def _safe_load_json(path: Path, *, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _conf_short(c: str) -> str:
    """Map LLM 'high'/'medium'/'low' to a 1-char display."""
    return {"high": "高", "medium": "中", "low": "低"}.get(c, c[:1].upper() if c else "?")


def _render_prediction_table(
    console: Any,
    table_cls: Any,
    group: list[dict],
    *,
    title: str,
    title_style: str,
    border_style: str,
    sort_key: str,
    close_lookup: dict,
) -> None:
    """Render one prediction-class section as a uniform table.

    All three R2 prediction classes (top_candidate / watchlist / avoid) share the
    same column layout — visual hierarchy is conveyed via title_style and
    border_style only. v0.5: an ``LGB`` column was added between ``分`` and
    ``信`` (lightgbm_design.md §11.2).
    """
    t = table_cls(
        title=title,
        title_style=title_style,
        border_style=border_style,
        header_style="k.label",
        expand=True,
    )
    t.add_column("#", justify="right", width=3)
    t.add_column("代码", style="k.value", no_wrap=True, width=11)
    t.add_column("名称", no_wrap=True, max_width=10)
    t.add_column("T收盘", justify="right", width=7)
    t.add_column("分", justify="right", width=4)
    t.add_column("LGB", justify="right", width=4)
    t.add_column("信", width=4)
    t.add_column("理由", overflow="fold")
    for p in group:
        t.add_row(
            str(p.get(sort_key, p.get("rank", "?"))),
            p.get("ts_code", "?"),
            p.get("name", "?"),
            _close_str(close_lookup.get(p.get("ts_code"))),
            f"{p.get('continuation_score', 0):.0f}",
            _lgb_compact(p),
            _conf_short(p.get("confidence", "")),
            p.get("rationale", ""),
        )
    console.print(t)


def export_llm_calls(run_id: str, db, *, reports_root: Path | None = None) -> int:  # noqa: ANN001
    """Pull this run's llm_calls rows into reports/<run_id>/llm_calls.jsonl."""
    root = (reports_root or paths.reports_dir()) / str(run_id)
    root.mkdir(parents=True, exist_ok=True)
    rows = db.fetchall(
        "SELECT call_id, model, prompt_hash, input_tokens, output_tokens, "
        "latency_ms, validation_status, error, created_at "
        "FROM llm_calls WHERE run_id = ? ORDER BY created_at",
        (run_id,),
    )
    out_path = root / "llm_calls.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "call_id": str(row[0]),
                        "model": row[1],
                        "prompt_hash": row[2],
                        "input_tokens": row[3],
                        "output_tokens": row[4],
                        "latency_ms": row[5],
                        "validation_status": row[6],
                        "error": row[7],
                        "created_at": str(row[8]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _industry_for(cid: str, candidates: list[dict[str, Any]]) -> str:
    for c in candidates:
        if c["candidate_id"] == cid:
            return str(c.get("industry") or c.get("lu_desc") or "—")
    return "—"


def _name_for(cid: str, predictions: list[ContinuationCandidate]) -> str:
    for p in predictions:
        if p.candidate_id == cid:
            return p.name
    return cid


def _close_for(cid: str, candidates: list[dict[str, Any]]) -> str:
    """Format the T-close price (元) for a candidate, '—' when missing."""
    for c in candidates:
        if c.get("candidate_id") == cid:
            return _close_str(c.get("close_yuan"))
    return "—"


def _close_str(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# v0.5 — LGB rendering helpers
# ---------------------------------------------------------------------------


def _lgb_model_id_repr(bundle: Round1Bundle) -> str:
    """Header line label: ``\\`<model_id>\\``` or ``\\`disabled\\``` for the
    summary.md metadata block."""
    if bundle.lgb_model_id:
        return f"`{bundle.lgb_model_id}`"
    return "`disabled`"


def _lgb_cell(cid: str, candidates: list[dict[str, Any]]) -> str:
    """One table-cell representation of a candidate's lgb_score / lgb_decile.

    Format: ``73 (d8)`` for score=73, decile=8; ``—`` when missing.
    """
    for c in candidates:
        if c.get("candidate_id") != cid:
            continue
        score = c.get("lgb_score")
        decile = c.get("lgb_decile")
        if score is None:
            return "—"
        try:
            s_str = f"{float(score):.0f}"
        except (TypeError, ValueError):
            return "—"
        if decile is None:
            return s_str
        return f"{s_str} (d{int(decile)})"
    return "—"


def _lgb_compact(c: dict[str, Any]) -> str:
    """Compact terminal-table representation. Width-3 friendly."""
    score = c.get("lgb_score")
    if score is None:
        return "—"
    try:
        return f"{float(score):.0f}"
    except (TypeError, ValueError):
        return "—"


def _render_debate_terminal(
    console: Any,
    root: Path,
    snap: dict[str, Any],
    close_lookup: dict[str, Any],
) -> None:
    """Render debate-mode tables: per-provider summary + cross-LLM matrix."""
    from rich.table import Table

    debate_dir = root / "debate"
    if not debate_dir.is_dir():
        console.print("[subtitle](无 debate/ 目录，可能 Phase A 全部失败)[/subtitle]")
        return

    providers: list[str] = list(snap.get("debate_providers") or [])
    if not providers:
        providers = sorted(p.name for p in debate_dir.iterdir() if p.is_dir())

    # Load each provider's initial + revised
    initials: dict[str, list[dict]] = {}
    reviseds: dict[str, list[dict]] = {}
    revision_summaries: dict[str, str] = {}
    for p in providers:
        pdir = debate_dir / p
        initials[p] = _safe_load_json(pdir / "round2_initial.json", default=[]) or []
        rev_obj = _safe_load_json(pdir / "round2_revised.json", default={}) or {}
        reviseds[p] = rev_obj.get("candidates", []) if isinstance(rev_obj, dict) else []
        if isinstance(rev_obj, dict):
            revision_summaries[p] = rev_obj.get("revision_summary", "") or ""

    # ----- Provider status panel ------------------------------------------
    status_t = Table(
        title=f"参与 LLM · {len(providers)} 位",
        title_style="title",
        border_style="panel.border.primary",
        header_style="k.label",
    )
    status_t.add_column("Provider", style="k.value", no_wrap=True)
    status_t.add_column("R2 初始", justify="right", width=8)
    status_t.add_column("R3 修订", justify="right", width=8)
    status_t.add_column("升级", justify="right", width=4)
    status_t.add_column("保持", justify="right", width=4)
    status_t.add_column("降级", justify="right", width=4)
    for p in providers:
        init_map = {c.get("candidate_id"): c for c in initials.get(p, [])}
        ups = downs = keeps = 0
        for r in reviseds.get(p, []):
            cid = r.get("candidate_id")
            init = init_map.get(cid)
            if init is None:
                continue
            ranking = {"avoid": 0, "watchlist": 1, "top_candidate": 2}
            i_r = ranking.get(str(init.get("prediction") or ""), 1)
            r_r = ranking.get(str(r.get("prediction") or ""), 1)
            if r_r > i_r:
                ups += 1
            elif r_r < i_r:
                downs += 1
            else:
                keeps += 1
        status_t.add_row(
            p,
            str(len(initials.get(p, []))),
            str(len(reviseds.get(p, []))),
            str(ups),
            str(keeps),
            str(downs),
        )
    console.print(status_t)

    # ----- Cross-LLM disagreement matrix (Revised) ------------------------
    union_ids: dict[str, tuple[str, str]] = {}
    pred_by_provider: dict[str, dict[str, dict]] = {}
    for p in providers:
        prov_map: dict[str, dict] = {}
        # Prefer revised; fall back to initial when R3 failed for this provider.
        source = reviseds.get(p) or initials.get(p, [])
        for c in source:
            cid = c.get("candidate_id")
            if cid is None:
                continue
            prov_map[cid] = c
            union_ids.setdefault(
                cid, (c.get("ts_code", "?"), c.get("name", "?"))
            )
        pred_by_provider[p] = prov_map

    if not union_ids:
        console.print("[subtitle](所有 LLM 均无连板预测候选)[/subtitle]")
        return

    def coverage(cid: str) -> int:
        return sum(1 for m in pred_by_provider.values() if cid in m)

    sorted_ids = sorted(union_ids.keys(), key=lambda c: (-coverage(c), c))

    matrix = Table(
        title=f"分歧矩阵 (Revised) · {len(sorted_ids)} 只 × {len(providers)} 个 LLM",
        title_style="title",
        border_style="panel.border.ok",
        header_style="k.label",
    )
    matrix.add_column("代码", style="k.value", no_wrap=True, width=11)
    matrix.add_column("名称", no_wrap=True, max_width=10)
    matrix.add_column("T收盘", justify="right", width=7)
    for p in providers:
        matrix.add_column(p, justify="center", width=8)
    for cid in sorted_ids:
        ts_code, name = union_ids[cid]
        row = [ts_code, name, _close_str(close_lookup.get(ts_code))]
        for p in providers:
            entry: dict | None = pred_by_provider[p].get(cid)
            if entry is None:
                row.append("—")
            else:
                row.append(
                    f"{_glyph(entry.get('prediction'))} "
                    f"{float(entry.get('continuation_score', 0)):.0f}"
                )
        matrix.add_row(*row)
    console.print(matrix)
    console.print(
        "[subtitle]▲ top_candidate / ○ watchlist / ▽ avoid / — 该 LLM 未将此股纳入预测[/subtitle]"
    )

    # ----- revision_summary per provider ----------------------------------
    for p in providers:
        s = revision_summaries.get(p)
        if s:
            console.print(f"[k.label]{p} · revision_summary:[/k.label] {s}")
