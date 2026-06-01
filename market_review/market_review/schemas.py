"""Pydantic schemas for market-review LLM section outputs (PR-1 placeholder).

PR-1 ships only the schema-version constant + ``SectionName`` Literal so that
:meth:`MarketReviewPlugin.validate_static` has a light, importable target.
The full :class:`OverviewSection` / :class:`SectorsSection` / ... models land
in PR-4; the full :class:`ReviewReportSchema` root + nested models (§15) land
in PR-5.
"""

from __future__ import annotations

from typing import Literal

SCHEMA_VERSION = "1.0"

SectionName = Literal[
    "overview",
    "sectors",
    "sentiment",
    "capital",
    "leaders",
    "style",
    "risk_outlook",
]

WindowMode = Literal["day", "range"]
