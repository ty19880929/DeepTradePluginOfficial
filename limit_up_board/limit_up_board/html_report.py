"""Self-contained HTML report renderer for 打板策略 runs.

输出单文件 ``summary.html``：Tailwind CSS 通过 play CDN 加载，所有数据内联。
设计目标：
    * 单文件可分享：微信 / 邮件 / U 盘拷贝即可在 PC 或手机端浏览器打开
    * 自适应布局：≤640px 卡片化，641-960 紧凑，>960 全宽
    * 零模板引擎依赖：纯 f-string 拼接，所有用户/LLM 字符串经 ``html.escape``
    * 失败不影响运行：HTML 生成抛出由 ``write_report`` 捕获并降级（仍写 md / JSON）

v0.9 仅实现单 LLM 模式；辩论模式的 ``render_debate_summary_html`` 在下个迭代补齐。
"""

from __future__ import annotations

import html
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from deeptrade.core.run_status import RunStatus

from .data import Round1Bundle
from .schemas import (
    ContinuationCandidate,
    FinalRankingResponse,
    StrongCandidate,
)

# ---------------------------------------------------------------------------
# 视觉常量
# ---------------------------------------------------------------------------

# 三类预测的色彩 token —— Tailwind 颜色（明亮配色：绿/橙/红）
_PRED_BADGE = {
    "top_candidate": ("bg-emerald-500", "text-white", "🔥 重点关注", "border-emerald-300"),
    "watchlist": ("bg-amber-400", "text-white", "👀 观察", "border-amber-300"),
    "avoid": ("bg-rose-500", "text-white", "⛔ 回避", "border-rose-300"),
}

_STRENGTH_BADGE = {
    "high": ("bg-emerald-100", "text-emerald-800", "强"),
    "medium": ("bg-amber-100", "text-amber-800", "中"),
    "low": ("bg-slate-100", "text-slate-700", "弱"),
}

_CONF_LABEL = {"high": "高", "medium": "中", "low": "低"}
_DELTA_BADGE = {
    "upgraded": ("bg-emerald-100", "text-emerald-700", "⬆ 升"),
    "kept": ("bg-slate-100", "text-slate-600", "= 保"),
    "downgraded": ("bg-rose-100", "text-rose-700", "⬇ 降"),
}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def render_summary_html(
    *,
    status: RunStatus,
    bundle: Round1Bundle,
    selected: list[StrongCandidate],
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None,
    failed_batch_ids: list[str] | None = None,
    run_id: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """构建单 LLM 模式 summary.html 完整字符串。"""
    generated_at = generated_at or datetime.now(timezone.utc).astimezone()
    cand_by_id = {c.get("candidate_id"): c for c in bundle.candidates}

    title = f"打板策略报告 · T={e(bundle.trade_date)}"

    body_parts: list[str] = [
        _render_banner(status, failed_batch_ids),
        _render_header(bundle, status),
        _render_meta_cards(bundle, status, len(selected), len(predictions)),
        _render_sector_strength_note(bundle),
        _render_candidate_filter_section(bundle),
        _render_lgb_distribution_section(bundle),
        _render_screening_section(selected, cand_by_id),
        _render_prediction_section(predictions, final_ranking, cand_by_id),
        _render_data_snapshot_section(bundle),
        _render_failed_batches_section(failed_batch_ids),
        _render_footer(run_id, generated_at),
    ]
    body = "\n".join(p for p in body_parts if p)

    return _PAGE_TEMPLATE.format(
        title=title,
        body=body,
        custom_css=_CUSTOM_CSS,
    )


# ---------------------------------------------------------------------------
# 各 section 渲染
# ---------------------------------------------------------------------------


def _render_banner(
    status: RunStatus, failed_batch_ids: list[str] | None
) -> str:
    """partial / failed / cancelled 顶部告警条；success 时返回空串。"""
    if status not in {RunStatus.PARTIAL_FAILED, RunStatus.FAILED, RunStatus.CANCELLED}:
        return ""
    palette = {
        RunStatus.PARTIAL_FAILED: ("bg-amber-50", "border-amber-400", "text-amber-900", "🚨", "PARTIAL — 本次结果不完整，不可作为有效筛选结果"),
        RunStatus.FAILED: ("bg-rose-50", "border-rose-400", "text-rose-900", "🚨", "FAILED — 运行失败"),
        RunStatus.CANCELLED: ("bg-slate-50", "border-slate-400", "text-slate-900", "⏹", "CANCELLED — 用户中断"),
    }[status]
    bg, border, txt, icon, label = palette
    extra = ""
    if status == RunStatus.PARTIAL_FAILED and failed_batch_ids:
        chips = "".join(
            f'<span class="inline-block px-2 py-0.5 m-0.5 text-xs rounded bg-white border border-amber-300 text-amber-800">{e(b)}</span>'
            for b in failed_batch_ids
        )
        extra = f'<div class="mt-2 text-sm"><span class="font-medium">失败批次：</span>{chips}</div>'
    return (
        f'<section class="mb-4 p-4 rounded-lg border-l-4 {bg} {border} {txt}">'
        f'<div class="font-semibold text-base flex items-center gap-2">'
        f'<span class="text-xl">{icon}</span><span>{e(label)}</span>'
        f"</div>{extra}"
        f"</section>"
    )


def _render_header(bundle: Round1Bundle, status: RunStatus) -> str:
    status_color = {
        RunStatus.SUCCESS: "text-emerald-600",
        RunStatus.PARTIAL_FAILED: "text-amber-600",
        RunStatus.FAILED: "text-rose-600",
        RunStatus.CANCELLED: "text-slate-500",
    }.get(status, "text-slate-600")
    return (
        '<header class="mb-6">'
        '<h1 class="text-2xl sm:text-3xl font-bold text-slate-900">'
        "📈 打板策略报告"
        "</h1>"
        '<p class="mt-1 text-sm text-slate-500">'
        f'盘后基于 T 日涨停池预测 T+1 次日连板候选 · 状态 <span class="{status_color} font-medium">{e(status.value)}</span>'
        "</p>"
        "</header>"
    )


def _render_meta_cards(
    bundle: Round1Bundle,
    status: RunStatus,
    n_selected: int,
    n_predictions: int,
) -> str:
    """4 张元信息卡片，auto-fit grid 在窄屏自动落 2 列。"""
    lgb_label = e(bundle.lgb_model_id) if bundle.lgb_model_id else "disabled"
    cards = [
        ("T 交易日", e(bundle.trade_date), "text-slate-900"),
        ("T+1 预测日", e(bundle.next_trade_date), "text-slate-900"),
        ("强势/候选/预测", f"{n_selected} / {len(bundle.candidates)} / {n_predictions}", "text-indigo-700"),
        ("LGB 模型", lgb_label, "text-cyan-700"),
    ]
    html_cards = "".join(
        '<div class="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">'
        f'<div class="text-xs text-slate-500 uppercase tracking-wider">{e(label)}</div>'
        f'<div class="mt-1 text-lg font-semibold {color} break-all">{value}</div>'
        "</div>"
        for label, value, color in cards
    )
    return (
        '<section class="mb-6 grid gap-3 grid-cols-2 lg:grid-cols-4">'
        f"{html_cards}"
        "</section>"
    )


def _render_sector_strength_note(bundle: Round1Bundle) -> str:
    """板块强度来源 + data_unavailable 提示行。"""
    parts: list[str] = []
    parts.append(
        '<div class="text-xs text-slate-500">'
        '板块强度来源：<code class="px-1 py-0.5 bg-slate-100 rounded">'
        f"{e(bundle.sector_strength.source)}"
        "</code> "
        '<span class="text-slate-400">'
        "（可信度：limit_cpt_list &gt; lu_desc_aggregation &gt; industry_fallback）"
        "</span>"
        "</div>"
    )
    if bundle.data_unavailable:
        chips = "".join(
            f'<span class="inline-block px-2 py-0.5 m-0.5 text-xs rounded bg-rose-50 border border-rose-200 text-rose-700">{e(s)}</span>'
            for s in bundle.data_unavailable
        )
        parts.append(
            '<div class="mt-1 text-xs text-slate-500">'
            "data_unavailable："
            f"{chips}"
            "</div>"
        )
    return f'<section class="mb-6">{"".join(parts)}</section>'


def _render_candidate_filter_section(bundle: Round1Bundle) -> str:
    """候选筛选 summary（流通市值 / 股价剔除）。"""
    fs = (
        bundle.market_summary.get("candidate_filter_summary")
        if isinstance(bundle.market_summary, dict)
        else None
    )
    if not isinstance(fs, dict):
        return ""
    before = fs.get("before")
    after = fs.get("after")
    if before is None or after is None or before == after:
        return ""
    dropped = fs.get("dropped_top3") or []
    drop_rows = "".join(
        _filter_drop_row(d) for d in dropped
    )
    drop_table = ""
    if drop_rows:
        drop_table = (
            '<div class="mt-3 overflow-x-auto">'
            '<table class="min-w-full text-xs">'
            "<thead><tr class=\"bg-slate-50 text-slate-600\">"
            '<th class="px-3 py-2 text-left">Code</th>'
            '<th class="px-3 py-2 text-left">Name</th>'
            '<th class="px-3 py-2 text-right">流通市值(亿)</th>'
            '<th class="px-3 py-2 text-right">收盘(元)</th>'
            '<th class="px-3 py-2 text-left">剔除原因</th>'
            "</tr></thead>"
            f'<tbody class="divide-y divide-slate-100">{drop_rows}</tbody>'
            "</table>"
            "</div>"
        )
    return (
        '<section class="mb-6 p-4 bg-white rounded-lg border border-slate-200">'
        '<h2 class="text-base font-semibold text-slate-800 mb-2">🧹 候选筛选</h2>'
        '<div class="text-sm text-slate-600">'
        f"进入筛选：<b>{int(before)}</b> 只 · 通过：<b class=\"text-emerald-600\">{int(after)}</b> 只 · "
        f"剔除：<b class=\"text-rose-600\">{int(before) - int(after)}</b> 只"
        "</div>"
        '<div class="mt-1 text-xs text-slate-500">'
        f"阈值：min_float_mv_yi={e(fs.get('min_float_mv_yi'))}亿 · "
        f"max_float_mv_yi={e(fs.get('max_float_mv_yi'))}亿 · "
        f"max_close_yuan={e(fs.get('max_close_yuan'))}元（闭区间）"
        "</div>"
        f"{drop_table}"
        "</section>"
    )


def _filter_drop_row(d: dict[str, Any]) -> str:
    mv = d.get("float_mv_yi")
    cl = d.get("close_yuan")
    reasons = ", ".join(d.get("reasons") or [])
    return (
        "<tr>"
        f'<td class="px-3 py-2 font-mono text-slate-700">{e(d.get("ts_code") or "")}</td>'
        f'<td class="px-3 py-2">{e(d.get("name") or "—")}</td>'
        f'<td class="px-3 py-2 text-right">{_num(mv, 2)}</td>'
        f'<td class="px-3 py-2 text-right">{_num(cl, 2)}</td>'
        f'<td class="px-3 py-2 text-slate-600">{e(reasons)}</td>'
        "</tr>"
    )


def _render_lgb_distribution_section(bundle: Round1Bundle) -> str:
    """LGB 评分分布 + 10 桶 SVG 直方图。"""
    if not bundle.lgb_model_id or not bundle.candidates:
        return ""
    scores = [
        c["lgb_score"] for c in bundle.candidates if c.get("lgb_score") is not None
    ]
    if not scores:
        return ""
    arr = sorted(float(s) for s in scores)
    n = len(arr)
    lo, hi = arr[0], arr[-1]
    p25 = _quantile(arr, 0.25)
    med = _quantile(arr, 0.5)
    p75 = _quantile(arr, 0.75)

    buckets = [0] * 10
    for s in arr:
        idx = min(9, max(0, int(s // 10)))
        buckets[idx] += 1
    max_bucket = max(buckets) or 1
    svg = _render_lgb_histogram_svg(buckets, max_bucket)

    return (
        '<section class="mb-6 p-4 bg-white rounded-lg border border-slate-200">'
        '<h2 class="text-base font-semibold text-slate-800 mb-2">📊 LGB 评分分布</h2>'
        '<div class="text-xs text-slate-500 mb-2">次日最大溢价概率（0–100）</div>'
        '<div class="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-3 text-center">'
        f"{_stat_card('n', n)}{_stat_card('min', f'{lo:.1f}')}{_stat_card('p25', f'{p25:.1f}')}"
        f"{_stat_card('median', f'{med:.1f}')}{_stat_card('p75', f'{p75:.1f}')}{_stat_card('max', f'{hi:.1f}')}"
        "</div>"
        f"{svg}"
        "</section>"
    )


def _stat_card(label: str, value: Any) -> str:
    return (
        '<div class="bg-slate-50 rounded p-2">'
        f'<div class="text-[10px] uppercase text-slate-400 tracking-wide">{e(label)}</div>'
        f'<div class="text-sm font-semibold text-slate-800">{e(value)}</div>'
        "</div>"
    )


def _render_lgb_histogram_svg(buckets: list[int], max_bucket: int) -> str:
    """纯 SVG 柱状图：10 个桶，宽度自适应。"""
    bar_width = 36
    gap = 4
    chart_h = 120
    label_h = 32
    width = 10 * bar_width + 9 * gap
    bars: list[str] = []
    for i, count in enumerate(buckets):
        x = i * (bar_width + gap)
        h = int((count / max_bucket) * chart_h) if count else 0
        y = chart_h - h
        bars.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" '
            f'rx="2" fill="#6366f1" opacity="{0.4 + 0.6 * (count / max_bucket):.2f}">'
            f"<title>{i * 10}-{i * 10 + 9}: {count}</title>"
            "</rect>"
        )
        bars.append(
            f'<text x="{x + bar_width / 2}" y="{y - 4}" text-anchor="middle" '
            f'font-size="10" fill="#475569">{count if count else ""}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2}" y="{chart_h + 14}" text-anchor="middle" '
            f'font-size="10" fill="#94a3b8">{i * 10}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2}" y="{chart_h + 26}" text-anchor="middle" '
            f'font-size="9" fill="#cbd5e1">-{i * 10 + 9}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {chart_h + label_h}" '
        f'class="w-full h-auto max-w-2xl" preserveAspectRatio="xMinYMin meet">'
        + "".join(bars)
        + "</svg>"
    )


def _render_screening_section(
    selected: list[StrongCandidate],
    cand_by_id: dict[str, dict[str, Any]],
) -> str:
    """Step 2 强势初筛入选表（PC 用 table，手机用卡片）。"""
    header = (
        '<section class="mb-6">'
        '<h2 class="text-lg font-semibold text-slate-800 mb-3">'
        f"🎯 Step 2 · 强势初筛入选（{len(selected)}）"
        "</h2>"
    )
    if not selected:
        return header + '<div class="text-sm text-slate-500 italic">本轮无强势标的</div></section>'

    # PC 表格
    rows = []
    for i, c in enumerate(selected, 1):
        src = cand_by_id.get(c.candidate_id, {})
        theme = src.get("industry") or src.get("lu_desc") or "—"
        rows.append(_screening_row(i, c, src, theme))

    table = (
        '<div class="hidden sm:block bg-white rounded-lg border border-slate-200 overflow-hidden">'
        '<div class="overflow-x-auto">'
        '<table class="min-w-full text-sm">'
        '<thead class="bg-slate-50 text-slate-600 text-xs uppercase tracking-wider">'
        "<tr>"
        '<th class="px-3 py-2 text-right">#</th>'
        '<th class="px-3 py-2 text-left">Code</th>'
        '<th class="px-3 py-2 text-left">Name</th>'
        '<th class="px-3 py-2 text-right">T收盘</th>'
        '<th class="px-3 py-2 text-right">Score</th>'
        '<th class="px-3 py-2 text-right">LGB</th>'
        '<th class="px-3 py-2 text-center">Level</th>'
        '<th class="px-3 py-2 text-left">Theme</th>'
        '<th class="px-3 py-2 text-left">Rationale</th>'
        "</tr></thead>"
        f'<tbody class="divide-y divide-slate-100">{"".join(rows)}</tbody>'
        "</table>"
        "</div></div>"
    )

    # 移动端卡片
    cards = "".join(_screening_card(i, c, cand_by_id) for i, c in enumerate(selected, 1))
    mobile = f'<div class="sm:hidden space-y-2">{cards}</div>'

    return header + table + mobile + "</section>"


def _screening_row(
    i: int, c: StrongCandidate, src: dict[str, Any], theme: Any
) -> str:
    bg, txt, label = _STRENGTH_BADGE.get(c.strength_level, ("bg-slate-100", "text-slate-700", c.strength_level))
    evidence_html = _evidence_panel(c.evidence, "evidence")
    risk_html = _risk_chips(c.risk_flags)
    return (
        "<tr>"
        f'<td class="px-3 py-2 text-right text-slate-500">{i}</td>'
        f'<td class="px-3 py-2 font-mono text-slate-800">{e(c.ts_code)}</td>'
        f'<td class="px-3 py-2 font-medium">{e(c.name)}</td>'
        f'<td class="px-3 py-2 text-right">{_close_str(src.get("close_yuan"))}</td>'
        f'<td class="px-3 py-2 text-right">{c.score:.1f}</td>'
        f'<td class="px-3 py-2 text-right text-cyan-700">{_lgb_cell(src)}</td>'
        f'<td class="px-3 py-2 text-center"><span class="inline-block px-2 py-0.5 text-xs rounded {bg} {txt}">{e(label)}</span></td>'
        f'<td class="px-3 py-2 text-slate-600">{e(theme)}</td>'
        '<td class="px-3 py-2">'
        f'<div class="text-slate-700">{e(c.rationale)}</div>'
        f"{risk_html}{evidence_html}"
        "</td>"
        "</tr>"
    )


def _screening_card(
    i: int, c: StrongCandidate, cand_by_id: dict[str, dict[str, Any]]
) -> str:
    src = cand_by_id.get(c.candidate_id, {})
    theme = src.get("industry") or src.get("lu_desc") or "—"
    bg, txt, label = _STRENGTH_BADGE.get(c.strength_level, ("bg-slate-100", "text-slate-700", c.strength_level))
    return (
        '<div class="bg-white rounded-lg border border-slate-200 p-3">'
        '<div class="flex items-center justify-between">'
        '<div class="flex items-center gap-2">'
        f'<span class="text-xs text-slate-400">#{i}</span>'
        f'<span class="font-mono text-sm text-slate-700">{e(c.ts_code)}</span>'
        f'<span class="font-semibold">{e(c.name)}</span>'
        "</div>"
        f'<span class="inline-block px-2 py-0.5 text-xs rounded {bg} {txt}">{e(label)}</span>'
        "</div>"
        '<div class="mt-2 grid grid-cols-3 gap-2 text-xs">'
        f'<div><span class="text-slate-400">T收盘</span><div class="font-medium">{_close_str(src.get("close_yuan"))}</div></div>'
        f'<div><span class="text-slate-400">Score</span><div class="font-medium">{c.score:.1f}</div></div>'
        f'<div><span class="text-slate-400">LGB</span><div class="font-medium text-cyan-700">{_lgb_cell(src)}</div></div>'
        "</div>"
        f'<div class="mt-2 text-xs text-slate-500">题材：{e(theme)}</div>'
        f'<div class="mt-2 text-sm text-slate-700">{e(c.rationale)}</div>'
        f"{_risk_chips(c.risk_flags)}"
        f"{_evidence_panel(c.evidence, 'evidence')}"
        "</div>"
    )


def _render_prediction_section(
    predictions: list[ContinuationCandidate],
    final_ranking: FinalRankingResponse | None,
    cand_by_id: dict[str, dict[str, Any]],
) -> str:
    """Step 4/4.5 次日连板预测。"""
    if not predictions:
        return (
            '<section class="mb-6">'
            '<h2 class="text-lg font-semibold text-slate-800 mb-3">🔮 Step 4 · 次日连板预测</h2>'
            '<div class="text-sm text-slate-500 italic">本轮无候选标的</div>'
            "</section>"
        )

    # 计算 final_rank 映射
    final_map: dict[str, dict[str, Any]] = {}
    if final_ranking is not None:
        for fi in final_ranking.finalists:
            final_map[fi.candidate_id] = {
                "final_rank": fi.final_rank,
                "final_prediction": fi.final_prediction,
                "final_confidence": fi.final_confidence,
                "delta_vs_batch": fi.delta_vs_batch,
                "reason_vs_peers": fi.reason_vs_peers,
            }
    multi_batch = final_ranking is not None

    # 按 prediction 分组（multi-batch 时用 final_prediction，否则用 prediction）
    groups: dict[str, list[tuple[ContinuationCandidate, dict[str, Any]]]] = {
        "top_candidate": [],
        "watchlist": [],
        "avoid": [],
    }
    for p in predictions:
        fr = final_map.get(p.candidate_id)
        pred_key = (fr["final_prediction"] if fr else p.prediction)
        groups.setdefault(pred_key, []).append((p, fr or {}))

    # 组内排序：multi-batch 用 final_rank，否则用 rank
    sort_key = "final_rank" if multi_batch else "rank"
    for g in groups.values():
        g.sort(key=lambda pair: (pair[1].get(sort_key) if pair[1] else None) or pair[0].rank)

    title_suffix = "（按 final_rank 全局重排）" if multi_batch else "（单批，按 rank）"
    parts: list[str] = [
        '<section class="mb-6">'
        f'<h2 class="text-lg font-semibold text-slate-800 mb-3">🔮 Step 4 · 次日连板预测 <span class="text-sm font-normal text-slate-500">{e(title_suffix)}</span></h2>'
    ]
    section_order = [
        ("top_candidate", "border-emerald-300"),
        ("watchlist", "border-amber-300"),
        ("avoid", "border-rose-300"),
    ]
    for key, _border in section_order:
        group = groups.get(key) or []
        if not group:
            continue
        parts.append(_render_prediction_group(key, group, cand_by_id, multi_batch))
    parts.append("</section>")
    return "".join(parts)


def _render_prediction_group(
    pred_key: str,
    group: list[tuple[ContinuationCandidate, dict[str, Any]]],
    cand_by_id: dict[str, dict[str, Any]],
    multi_batch: bool,
) -> str:
    bg, txt, label, border = _PRED_BADGE.get(
        pred_key, ("bg-slate-100", "text-slate-700", pred_key, "border-slate-300")
    )
    cards = "".join(
        _render_prediction_card(p, fr, cand_by_id, multi_batch) for p, fr in group
    )
    return (
        f'<div class="mb-5">'
        f'<div class="flex items-center gap-2 mb-3">'
        f'<span class="inline-block px-3 py-1 rounded-full text-sm {bg} {txt} font-medium">{e(label)}</span>'
        f'<span class="text-sm text-slate-500">· {len(group)} 只</span>'
        f"</div>"
        f'<div class="grid grid-cols-1 lg:grid-cols-2 gap-3">{cards}</div>'
        f"</div>"
    )


def _render_prediction_card(
    p: ContinuationCandidate,
    fr: dict[str, Any],
    cand_by_id: dict[str, dict[str, Any]],
    multi_batch: bool,
) -> str:
    src = cand_by_id.get(p.candidate_id, {})
    rank_label = (
        f"#{fr['final_rank']}（批内 #{p.rank}）"
        if multi_batch and fr
        else f"#{p.rank}"
    )
    pred = (fr.get("final_prediction") if fr else p.prediction) or p.prediction
    conf = (fr.get("final_confidence") if fr else p.confidence) or p.confidence
    bg, txt, label, border = _PRED_BADGE.get(
        pred, ("bg-slate-100", "text-slate-700", pred, "border-slate-300")
    )

    # delta_vs_batch chip (multi-batch only)
    delta_html = ""
    if multi_batch and fr.get("delta_vs_batch"):
        dbg, dtxt, dlabel = _DELTA_BADGE.get(
            fr["delta_vs_batch"], ("bg-slate-100", "text-slate-600", fr["delta_vs_batch"])
        )
        delta_html = (
            f'<span class="inline-block px-2 py-0.5 text-xs rounded {dbg} {dtxt} ml-1">{e(dlabel)}</span>'
        )

    reason_block = ""
    if multi_batch and fr.get("reason_vs_peers"):
        reason_block = (
            '<div class="mt-2 p-2 bg-indigo-50 border-l-2 border-indigo-300 text-xs text-indigo-900 rounded-r">'
            '<span class="font-medium">vs peers：</span>'
            f"{e(fr['reason_vs_peers'])}"
            "</div>"
        )

    watch_pts = _bullet_list(p.next_day_watch_points, "次日观察点", "text-emerald-700", "bg-emerald-50")
    fail_triggers = _bullet_list(p.failure_triggers, "失败触发条件", "text-rose-700", "bg-rose-50")

    return (
        f'<article class="bg-white rounded-lg border-l-4 {border} border border-slate-200 p-4 shadow-sm">'
        # header
        '<div class="flex items-start justify-between gap-2">'
        '<div>'
        '<div class="flex items-center gap-2 flex-wrap">'
        f'<span class="text-xs text-slate-400">{e(rank_label)}</span>'
        f'<span class="font-mono text-sm text-slate-700">{e(p.ts_code)}</span>'
        f'<span class="font-semibold text-slate-900">{e(p.name)}</span>'
        f'<span class="inline-block px-2 py-0.5 text-xs rounded {bg} {txt}">{e(label)}</span>'
        f"{delta_html}"
        "</div>"
        '<div class="mt-1 text-xs text-slate-500">'
        f'信心：{e(_CONF_LABEL.get(conf, conf))} · 评分：{p.continuation_score:.0f}'
        "</div>"
        "</div>"
        '<div class="text-right text-xs text-slate-400 shrink-0">'
        f'<div>T 收盘<div class="text-sm font-semibold text-slate-700">{_close_str(src.get("close_yuan"))}</div></div>'
        f'<div class="mt-1">LGB<div class="text-sm font-semibold text-cyan-700">{_lgb_cell(src)}</div></div>'
        "</div>"
        "</div>"
        # rationale
        f'<p class="mt-3 text-sm text-slate-700 leading-relaxed">{e(p.rationale)}</p>'
        f"{reason_block}"
        # watch / fail
        f'<div class="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">{watch_pts}{fail_triggers}</div>'
        # evidence
        f"{_evidence_panel(p.key_evidence, 'key_evidence')}"
        f"{_missing_data_chips(p.missing_data)}"
        "</article>"
    )


def _bullet_list(items: list[str], title: str, txt_color: str, bg_color: str) -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{e(s)}</li>" for s in items)
    return (
        f'<div class="p-2 {bg_color} rounded text-xs {txt_color}">'
        f'<div class="font-medium mb-1">{e(title)}</div>'
        f'<ul class="list-disc list-inside space-y-0.5">{lis}</ul>'
        "</div>"
    )


def _evidence_panel(evidence_items: Any, title: str) -> str:
    """<details> 折叠面板，PC + 手机都可点开。"""
    if not evidence_items:
        return ""
    rows = []
    for ev in evidence_items:
        # ev 是 EvidenceItem pydantic 实例
        field = e(getattr(ev, "field", ""))
        value = e(getattr(ev, "value", ""))
        unit = e(getattr(ev, "unit", ""))
        interp = e(getattr(ev, "interpretation", ""))
        rows.append(
            "<tr>"
            f'<td class="px-2 py-1 font-mono text-slate-600 align-top">{field}</td>'
            f'<td class="px-2 py-1 text-slate-800 align-top">{value} <span class="text-slate-400">{unit}</span></td>'
            f'<td class="px-2 py-1 text-slate-600 align-top">{interp}</td>'
            "</tr>"
        )
    return (
        '<details class="mt-2 text-xs">'
        f'<summary class="cursor-pointer text-indigo-600 hover:text-indigo-800 select-none">▸ 查看 {len(evidence_items)} 条 {e(title)}</summary>'
        '<div class="mt-1 bg-slate-50 rounded p-2 overflow-x-auto">'
        '<table class="min-w-full text-xs">'
        '<thead><tr class="text-slate-500">'
        '<th class="px-2 py-1 text-left">字段</th>'
        '<th class="px-2 py-1 text-left">值</th>'
        '<th class="px-2 py-1 text-left">解读</th>'
        f'</tr></thead><tbody class="divide-y divide-slate-200">{"".join(rows)}</tbody>'
        "</table>"
        "</div>"
        "</details>"
    )


def _risk_chips(flags: list[str]) -> str:
    if not flags:
        return ""
    chips = "".join(
        f'<span class="inline-block px-1.5 py-0.5 m-0.5 text-[10px] rounded bg-rose-50 border border-rose-200 text-rose-700">{e(s)}</span>'
        for s in flags
    )
    return f'<div class="mt-1">{chips}</div>'


def _missing_data_chips(missing: list[str]) -> str:
    if not missing:
        return ""
    chips = "".join(
        f'<span class="inline-block px-1.5 py-0.5 m-0.5 text-[10px] rounded bg-slate-50 border border-slate-200 text-slate-500">{e(s)}</span>'
        for s in missing
    )
    return (
        f'<div class="mt-2 text-[10px] text-slate-400">missing data：{chips}</div>'
    )


def _render_data_snapshot_section(bundle: Round1Bundle) -> str:
    """折叠面板：market_summary 关键字段 + 全部 candidates 简表。"""
    if not bundle.candidates:
        return ""
    # market_summary 中常见字段（除 candidate_filter_summary 已单独展示）
    ms = bundle.market_summary or {}
    ms_items = []
    for k, v in ms.items():
        if k == "candidate_filter_summary":
            continue
        ms_items.append(
            f'<dt class="text-xs text-slate-400">{e(k)}</dt>'
            f'<dd class="text-sm text-slate-700 mb-2 break-all">{e(_compact_value(v))}</dd>'
        )
    ms_block = ""
    if ms_items:
        ms_block = (
            '<div class="mb-3">'
            '<h3 class="text-sm font-medium text-slate-700 mb-2">market_summary</h3>'
            f'<dl class="grid grid-cols-1 sm:grid-cols-2 gap-x-4">{"".join(ms_items)}</dl>'
            "</div>"
        )

    # 全部 candidates 简表
    rows = "".join(_candidate_row(c) for c in bundle.candidates)
    cand_block = (
        '<div>'
        '<h3 class="text-sm font-medium text-slate-700 mb-2">全部候选股</h3>'
        '<div class="overflow-x-auto">'
        '<table class="min-w-full text-xs">'
        '<thead class="text-slate-500 bg-slate-50">'
        "<tr>"
        '<th class="px-2 py-1 text-left">Code</th>'
        '<th class="px-2 py-1 text-left">Name</th>'
        '<th class="px-2 py-1 text-right">收盘(元)</th>'
        '<th class="px-2 py-1 text-right">流通市值(亿)</th>'
        '<th class="px-2 py-1 text-left">题材/行业</th>'
        '<th class="px-2 py-1 text-right">LGB</th>'
        "</tr></thead>"
        f'<tbody class="divide-y divide-slate-100">{rows}</tbody>'
        "</table>"
        "</div></div>"
    )

    return (
        '<section class="mb-6">'
        '<details class="bg-white rounded-lg border border-slate-200 overflow-hidden">'
        '<summary class="px-4 py-3 cursor-pointer select-none text-slate-700 font-medium hover:bg-slate-50">'
        f"📦 数据快照（{len(bundle.candidates)} 只候选股 + market_summary）"
        "</summary>"
        '<div class="p-4 border-t border-slate-200">'
        f"{ms_block}{cand_block}"
        "</div>"
        "</details>"
        "</section>"
    )


def _candidate_row(c: dict[str, Any]) -> str:
    theme = c.get("industry") or c.get("lu_desc") or "—"
    return (
        "<tr>"
        f'<td class="px-2 py-1 font-mono text-slate-700">{e(c.get("ts_code") or "")}</td>'
        f'<td class="px-2 py-1">{e(c.get("name") or "—")}</td>'
        f'<td class="px-2 py-1 text-right">{_close_str(c.get("close_yuan"))}</td>'
        f'<td class="px-2 py-1 text-right">{_num(c.get("float_mv_yi"), 2)}</td>'
        f'<td class="px-2 py-1 text-slate-600">{e(theme)}</td>'
        f'<td class="px-2 py-1 text-right text-cyan-700">{_lgb_cell(c)}</td>'
        "</tr>"
    )


def _render_failed_batches_section(failed_batch_ids: list[str] | None) -> str:
    if not failed_batch_ids:
        return ""
    chips = "".join(
        f'<span class="inline-block px-2 py-0.5 m-0.5 text-xs rounded bg-amber-50 border border-amber-300 text-amber-800 font-mono">{e(b)}</span>'
        for b in failed_batch_ids
    )
    return (
        '<section class="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">'
        '<h2 class="text-base font-semibold text-amber-900 mb-2">⚠ 失败批次清单</h2>'
        f'<div>{chips}</div>'
        '<div class="mt-2 text-xs text-amber-700">详细错误请查阅 <code>llm_calls.jsonl</code></div>'
        "</section>"
    )


def _render_footer(run_id: str | None, generated_at: datetime) -> str:
    gen = generated_at.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    rid = f"run_id: <code>{e(run_id)}</code> · " if run_id else ""
    return (
        '<footer class="mt-8 pt-4 border-t border-slate-200 text-xs text-slate-400 text-center">'
        f"{rid}生成时间：{e(gen)}<br>"
        "免责声明：本报告仅用于策略研究，不构成投资建议。"
        "</footer>"
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def e(v: Any) -> str:  # noqa: D401 — short name on purpose, used in dozens of places
    """HTML-escape ``v`` after stringification. None / nan 显示为 '—'。"""
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and v != v:  # NaN
            return "—"
    except Exception:  # noqa: BLE001
        pass
    return html.escape(str(v), quote=True)


def _num(v: Any, ndigits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return "—"


def _close_str(v: Any) -> str:
    return _num(v, 2)


def _lgb_cell(src: dict[str, Any]) -> str:
    """``73 (d8)`` 或 ``73`` 或 ``—``。"""
    if not isinstance(src, dict):
        return "—"
    score = src.get("lgb_score")
    decile = src.get("lgb_decile")
    if score is None:
        return "—"
    try:
        s_str = f"{float(score):.0f}"
    except (TypeError, ValueError):
        return "—"
    if decile is None:
        return s_str
    try:
        return f"{s_str} (d{int(decile)})"
    except (TypeError, ValueError):
        return s_str


def _quantile(sorted_arr: list[float], q: float) -> float:
    if not sorted_arr:
        return float("nan")
    if len(sorted_arr) == 1:
        return sorted_arr[0]
    pos = q * (len(sorted_arr) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_arr) - 1)
    frac = pos - lo
    return sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac


def _compact_value(v: Any) -> str:
    """market_summary 字段紧凑显示：dict/list 截断到前 200 字符。"""
    if v is None:
        return "—"
    if isinstance(v, (dict, list)):
        import json as _json

        s = _json.dumps(v, ensure_ascii=False, default=str)
        if len(s) > 200:
            return s[:200] + "…"
        return s
    return str(v)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


_CUSTOM_CSS = """
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  details > summary { list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details[open] > summary > .marker::before { content: "▾"; }
  /* Tailwind play CDN 不支持自定义 JIT 配置，这里手补一个 break-all on td */
  td.break-all { word-break: break-all; }
"""


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>{custom_css}</style>
</head>
<body class="bg-slate-50 text-slate-900">
<main class="max-w-5xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
{body}
</main>
</body>
</html>
"""
