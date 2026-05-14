"""Event → user-visible-text mappings.

Plan §3.2.4 / §3.3 — keeps the de-technicalisation rules in one file so
adding a new pipeline stage requires only an entry in ``STAGE_TITLES_*``,
not a change to the dashboard widget.

The runner / pipeline emit messages prefixed ``Step <id>: <detail>``
(including PR-1's "Step 2:" alignment for 走势分析); we extract ``<id>``
with :func:`parse_stage_id` and look up the user-facing title in the
mode-specific titles dict. Unknown stage ids degrade gracefully to
``Step <id>``.
"""

from __future__ import annotations

import re

# Maps the numeric stage id (matching the ``Step X:`` prefix the pipeline /
# runner emits) to a Chinese title the user actually wants to read. The two
# modes own separate tables because their semantics differ — analyze's "Step
# 1" is data assembly, screen's "Step 1" is the anomaly funnel.

STAGE_TITLES_ANALYZE: dict[str, str] = {
    "0": "核对交易日期",
    "1": "组装候选包（数据装配）",
    "2": "走势分析（主升浪启动预测）",
    "5": "生成走势分析报告",
}

STAGE_TITLES_SCREEN: dict[str, str] = {
    "0": "核对交易日期",
    "1": "异动筛选（主板量能漏斗）",
    "5": "生成筛选报告",
}


_STAGE_ID_RE = re.compile(r"^Step\s+(\d+(?:\.\d+)?)\s*:")


def parse_stage_id(message: str) -> str | None:
    """Extract ``"0"`` / ``"1"`` / ``"2"`` / ``"5"`` from messages like
    ``"Step 2: 走势分析（主升浪启动预测）"``. Returns ``None`` if the
    message has no recognisable prefix — caller treats that as a non-stage
    event.
    """
    m = _STAGE_ID_RE.match(message)
    return m.group(1) if m else None


def title_for(stage_id: str, mode: str) -> str:
    """Look up the user-facing title for ``stage_id`` under ``mode``.

    Falls back to ``Step <stage_id>`` so future pipeline additions display
    sanely without a code change here (Plan §3.2.4 — "graceful degradation").
    Unknown modes default to the analyze table since that's the LLM-heavy
    path users are most likely to hit.
    """
    table = (
        STAGE_TITLES_SCREEN if mode == "screen" else STAGE_TITLES_ANALYZE
    )
    return table.get(stage_id, f"Step {stage_id}")


__all__ = [
    "STAGE_TITLES_ANALYZE",
    "STAGE_TITLES_SCREEN",
    "parse_stage_id",
    "title_for",
]
