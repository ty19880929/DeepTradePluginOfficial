"""Event → user-visible-text mappings.

Plan §3.2.4 / §3.3 — keeps the de-technicalisation rules in one file so
adding a new pipeline stage requires only an entry in ``STAGE_TITLES``,
not a change to the dashboard widget.

The runner / pipeline emit messages prefixed ``Step <id>: <detail>``; we
extract ``<id>`` with :func:`parse_stage_id` and look up the user-facing
title in :data:`STAGE_TITLES`. Unknown stage ids degrade gracefully to
``Step <id>``.
"""

from __future__ import annotations

import re

# Maps the numeric stage id (matching the ``Step X:`` prefix the pipeline
# emits) to a Chinese title the user actually wants to read.
STAGE_TITLES: dict[str, str] = {
    "0": "核对交易日期",
    "1": "捕获基础标的",
    "2": "R1 强势标的初筛",
    "4": "R2 连板潜力预测",
    "4.5": "全局重排（多批合并）",
    "4.7": "R3 辩论修订",
    "5": "生成策略报告",
}


_STAGE_ID_RE = re.compile(r"^Step\s+(\d+(?:\.\d+)?)\s*:")


def parse_stage_id(message: str) -> str | None:
    """Extract ``"0"`` / ``"4.5"`` / ``"4.7"`` from messages like
    ``"Step 4.5: final_ranking ..."``. Returns ``None`` if the message has
    no recognisable prefix — caller treats that as a non-stage event.
    """
    m = _STAGE_ID_RE.match(message)
    return m.group(1) if m else None


def title_for(stage_id: str) -> str:
    """Look up the user-facing title for ``stage_id``.

    Falls back to ``Step <stage_id>`` so future pipeline additions display
    sanely without a code change here (Plan §3.2.4 — "graceful degradation").
    """
    return STAGE_TITLES.get(stage_id, f"Step {stage_id}")


__all__ = ["STAGE_TITLES", "parse_stage_id", "title_for"]
