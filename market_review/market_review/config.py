"""MrConfig — user-tunable plugin config dataclass (PR-1 skeleton).

Mirrors the design §8 spec. PR-1 ships only the dataclass declaration so
that :meth:`MarketReviewPlugin.validate_static` can import it as a syntax
check. The DB-backed load / save / set-from-string helpers (the analogue of
``limit_up_board.config.load_config`` etc.) land in PR-6 together with the
``settings`` subcommand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MrConfig:
    """User-tunable settings persisted in ``mr_config(key, value_json, ...)``.

    Design §8 — fields here drive: (1) window guardrails, (2) universe
    inclusion, (3) sentiment temperature weights, (4) leader scoring cutoffs,
    (5) sector provider, (6) section enable list, (7) LLM provider + replay.

    Anything that would belong to *per-run* CLI flags (e.g. ``--no-llm``,
    ``--force-sync``) stays out of this dataclass and lives on ``RunParams``
    inside ``runner.py`` (added in PR-6).
    """

    # ---- 窗口 ----
    max_window_days: int = 60
    """区间复盘允许的最大交易日数；硬上限 252，超出 CLI 直接报错（设计 §3.2）。"""

    # ---- 池子 ----
    exclude_north_exchange: bool = False
    """北交所默认开启；用户可临时关掉（设计 §3.1）。"""

    # ---- 情绪温度计权重（和必须为 1，校验在 PR-3 metrics/sentiment.py）----
    sentiment_weights: dict[str, float] = field(
        default_factory=lambda: {
            "pos_ratio": 0.30,
            "top_ratio": 0.20,
            "limit_up_intensity": 0.15,
            "connection_health": 0.10,
            "crash_ratio_inv": 0.10,
            "lhb_buy_intensity": 0.10,
            "north_inflow": 0.05,
        }
    )

    # ---- 龙头识别 ----
    leaders_top_k: int = 5
    leaders_secondary_k: int = 10
    leaders_min_score: float = 50.0

    # ---- 板块 ----
    sectors_top_k: int = 10
    sector_provider: Literal["ths", "dc"] = "ths"

    # ---- 报告 ----
    section_max_chars: int = 1200
    sections_enabled: list[str] = field(
        default_factory=lambda: [
            "overview",
            "sectors",
            "sentiment",
            "capital",
            "leaders",
            "style",
            "risk_outlook",
        ]
    )

    # ---- LLM ----
    llm_provider_default: str | None = None
    """``None`` → 走框架默认 provider。CLI ``--llm`` 覆盖此值（设计 §7.1）。"""

    # ---- LLM replay（沿用 lub v0.15+ 三字段语义）----
    llm_replay_enabled: bool = True
    llm_replay_write: bool = True
    llm_replay_ttl_days: int | None = None
