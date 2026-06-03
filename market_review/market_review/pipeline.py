"""Pipeline — runs the 7 LLM sections sequentially (design §5.4.1).

Public entry:
    :func:`run_sections(rt, bundle, *, llm_provider=..., ...) -> dict[SectionName, SectionResult]`

Order is :data:`market_review.schemas.SECTION_ORDER`. ``overview`` runs
first; its ``market_tone`` + ``theme_tags`` are piped into the remaining
6 sections as ``prev_context`` so the narrative tone stays consistent
(design §5.4.4).

Per-section failures don't kill the pipeline (design §11.3): each call is
wrapped in try/except; a failure produces a default schema instance with
``narrative_md=""``, ``error=<class>: <message>``, and the loop continues.
The caller (PR-5 runner) reads :class:`SectionResult.error` to populate
:class:`ReportMeta.failedSections` and the ``PARTIAL`` banner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .prompts import (
    build_capital_user_prompt,
    build_leaders_user_prompt,
    build_overview_user_prompt,
    build_prev_context,
    build_risk_outlook_user_prompt,
    build_sectors_user_prompt,
    build_sentiment_user_prompt,
    build_style_user_prompt,
    system_prompt,
)
from .schemas import (
    SCHEMA_VERSION,
    SECTION_ORDER,
    SECTION_SCHEMAS,
    OverviewSection,
    SectionBase,
    SectionName,
)

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.llm import StageProfile as _StageProfile

    from .metrics.breadth import BreadthReview
    from .metrics.capital import CapitalReview
    from .metrics.leaders import LeaderReview
    from .metrics.risk import RiskReview
    from .metrics.sectors import SectorReview
    from .metrics.sentiment import SentimentReview
    from .metrics.style import StyleReview
    from .runtime import MrRuntime
    from .windows import Window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle of PR-3 metric reviews — the LLM's input
# ---------------------------------------------------------------------------


@dataclass
class MetricsBundle:
    """All 7 PR-3 metric reviews + the window, packaged for the pipeline.

    Build with :func:`build_metrics_bundle` (PR-6 runner) and pass to
    :func:`run_sections`. Keeping this in one place lets test fixtures
    construct minimal bundles without re-listing every metric type.
    """

    window: Window
    breadth: BreadthReview
    sentiment: SentimentReview
    capital: CapitalReview
    sectors: SectorReview
    leaders: LeaderReview
    style: StyleReview
    risk: RiskReview


@dataclass
class SectionResult:
    """One section's LLM result + audit metadata.

    ``schema`` is the validated pydantic model. On failure, it is a default
    instance of the section's schema (e.g. :class:`OverviewSection`) with
    ``narrative_md=""`` + ``error`` set; downstream code (PR-5 builder)
    detects the failure by checking ``error``.
    """

    section: SectionName
    schema: SectionBase
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Default LLM tuning — PR-4 ships one profile; PR-6 may tune per section.
# ---------------------------------------------------------------------------


def _default_profile():
    """Conservative default — moderate temperature, JSON-friendly token cap.

    ``thinking=False`` keeps single-section latency under 30s on most
    providers; ``temperature=0.3`` is low enough that the LLM doesn't
    drift away from the schema yet high enough that narratives don't read
    as boilerplate. Imported lazily to keep the framework dependency out of
    schemas.py / module import time.
    """
    from deeptrade.plugins_api.llm import StageProfile  # noqa: PLC0415

    return StageProfile(
        thinking=False,
        reasoning_effort="medium",
        temperature=0.3,
        max_output_tokens=16_384,
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_sections(
    rt: MrRuntime,
    bundle: MetricsBundle,
    *,
    llm_provider: str | None = None,
    profile: _StageProfile | None = None,
    input_fingerprint: str | None = None,
    replay=None,
) -> dict[SectionName, SectionResult]:
    """Execute :data:`SECTION_ORDER` against the LLM, return results by name.

    ``llm_provider`` — explicit provider name to pin (CLI ``--llm``).
    ``None`` defers to the framework default (per
    :meth:`LLMManager.get_client`).

    ``profile`` overrides the default :class:`StageProfile`; pass ``None`` to
    use :func:`_default_profile`. The same profile is reused for all 7
    sections in v0.1 (design §5.4 doesn't suggest per-section variants).

    ``input_fingerprint`` is the 64-char sha256 the PR-5 runner computes
    over the input bundle; surfaced into every LLM cache key so retries +
    replay key the same way.

    ``replay`` (v0.1.0 PR-7) — optional :class:`LLMReplayPolicy` controlling
    framework-level LLM response caching. When ``None``, the framework
    treats the call as cache-disabled (pre-cache behavior). Runner builds
    this via :func:`policy_from_env(policy_from_app_config(...))` so user
    env vars (``DEEPTRADE_FRESH_LLM`` etc.) take effect.
    """
    client = rt.llms.get_client(llm_provider, plugin_id=rt.plugin_id, run_id=rt.run_id)
    effective_profile = profile or _default_profile()

    results: dict[SectionName, SectionResult] = {}
    overview_schema: OverviewSection | None = None

    for section in SECTION_ORDER:
        try:
            sys_prompt = system_prompt(section)
            user_prompt = _build_user_prompt(
                section, bundle=bundle, overview=overview_schema,
            )
            validated, meta = client.complete_json(
                system=sys_prompt,
                user=user_prompt,
                schema=SECTION_SCHEMAS[section],
                profile=effective_profile,
                stage=section,
                schema_version=SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                replay=replay,
            )
            results[section] = SectionResult(
                section=section,
                schema=validated,  # type: ignore[arg-type] — Pydantic v2 BaseModel
                meta=meta,
            )
            if section == "overview" and isinstance(validated, OverviewSection):
                overview_schema = validated
        except Exception as exc:  # noqa: BLE001 — design §11.3 per-section isolation
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("section %s failed: %s", section, err_msg)
            schema_cls = SECTION_SCHEMAS[section]
            placeholder = _placeholder(schema_cls, err_msg)
            results[section] = SectionResult(
                section=section,
                schema=placeholder,
                error=err_msg,
            )
            # When overview itself fails, downstream sections still get
            # default theme_tags / market_tone (design §11.4) — placeholder
            # has market_tone="未知" + theme_tags=[].
            if section == "overview" and isinstance(placeholder, OverviewSection):
                overview_schema = placeholder

    return results


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_user_prompt(
    section: SectionName,
    *,
    bundle: MetricsBundle,
    overview: OverviewSection | None,
) -> str:
    """Route to the per-section user-prompt builder."""
    prev_context = build_prev_context(overview)
    if section == "overview":
        return build_overview_user_prompt(
            window=bundle.window,
            breadth=bundle.breadth,
            sentiment=bundle.sentiment,
            capital=bundle.capital,
            sectors=bundle.sectors,
            leaders=bundle.leaders,
            style=bundle.style,
            risk=bundle.risk,
        )
    if section == "sectors":
        return build_sectors_user_prompt(
            window=bundle.window, sectors=bundle.sectors, prev_context=prev_context,
        )
    if section == "sentiment":
        return build_sentiment_user_prompt(
            window=bundle.window,
            sentiment=bundle.sentiment,
            breadth=bundle.breadth,
            prev_context=prev_context,
        )
    if section == "capital":
        return build_capital_user_prompt(
            window=bundle.window, capital=bundle.capital, prev_context=prev_context,
        )
    if section == "leaders":
        return build_leaders_user_prompt(
            window=bundle.window, leaders=bundle.leaders, sectors=bundle.sectors,
            prev_context=prev_context,
        )
    if section == "style":
        return build_style_user_prompt(
            window=bundle.window, style=bundle.style, prev_context=prev_context,
        )
    if section == "risk_outlook":
        return build_risk_outlook_user_prompt(
            window=bundle.window, risk=bundle.risk,
            breadth=bundle.breadth, capital=bundle.capital,
            prev_context=prev_context,
        )
    raise ValueError(f"unknown section: {section!r}")


def _placeholder(schema_cls: type[SectionBase], error: str) -> SectionBase:
    """Build a default-valued section schema carrying the failure reason.

    For sections with required fields (e.g. :class:`OverviewSection` needs
    ``market_tone`` + ``headline_metrics``), supply minimal-valid defaults
    so the placeholder still validates. The downstream report builder
    treats any non-None ``error`` as the failure signal — the rest of the
    payload is best-effort.
    """
    from .schemas import (  # noqa: PLC0415 — avoid circular import at module load
        HeadlineMetric, OutlookHypothesis, OverviewSection, RiskOutlookSection,
    )

    if schema_cls is OverviewSection:
        return OverviewSection(
            market_tone="未知",
            headline_metrics=[
                HeadlineMetric(label="LLM 失败", value=None, unit="none"),
                HeadlineMetric(label="LLM 失败", value=None, unit="none"),
                HeadlineMetric(label="LLM 失败", value=None, unit="none"),
            ],
            theme_tags=[],
            error=error,
        )
    if schema_cls is RiskOutlookSection:
        return RiskOutlookSection(
            hypotheses=[OutlookHypothesis(
                title="LLM 失败",
                rationale="本节 LLM 调用失败，无法生成展望，详见 error 字段。",
                watch_points=["手动复盘本节数据"],
                fail_triggers=["LLM 二次调用仍失败 → 切换 provider"],
            )],
            error=error,
        )
    # Generic sections (sectors / sentiment / capital / leaders / style)
    # all have only optional fields beyond SectionBase; default ctor works.
    return schema_cls(error=error)  # type: ignore[call-arg]
