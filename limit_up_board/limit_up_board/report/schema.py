"""``StrategyReportSchema`` pydantic 模型 —— 与前端 TS interface 一一对应。

字段语义、空值约定、与 ``schema_and_ui_proposal.md`` 的偏离都记录在
``E:/personal/docs/DeepTradeWebsite/1/frontend_migration_notes.md`` 中，
本文件是该说明的「执行版」：

* ``extra="forbid"`` 在所有非根模型上；上游悄悄加字段必须显式合入 schema
* 根模型保留 ``_extras: dict`` 兜底未来 ``market_summary`` 新增字段，
  防止 JSON 序列化阶段静默丢数据
* ``model_dump(mode="json")`` 输出能直接 ``json.dumps`` 落盘
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import EvidenceItemStrict

# ---------------------------------------------------------------------------
# meta / counts / dataSource
# ---------------------------------------------------------------------------

ReportStatus = Literal["success", "partial_failed", "failed", "cancelled"]
ThemeStrengthSource = Literal[
    "limit_cpt_list", "unavailable"
]


class Counts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial: int = Field(ge=0)
    selected: int = Field(ge=0)
    predicted: int = Field(ge=0)


class DataSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themeStrength: ThemeStrengthSource
    unavailable: list[str] = Field(default_factory=list)


class ReportMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    run_id: str
    trade_date_t: str
    trade_date_t1: str
    status: ReportStatus
    # "disabled" 表示 LGB 未启用；非空字符串
    model_version: str = Field(min_length=1)
    counts: Counts
    dataSource: DataSource
    # partial_failed 时枚举失败批次（"初筛#3"、"预测#1"、"全局重排"）；其它状态 []
    failedBatches: list[str] = Field(default_factory=list)
    # ISO 8601，含时区
    generatedAt: str


# ---------------------------------------------------------------------------
# scoreDistribution
# ---------------------------------------------------------------------------


class ScoreStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(ge=0)
    min: float
    p25: float
    median: float
    p75: float
    max: float


class HistogramBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    range: str  # "0-9" .. "90-99"
    count: int = Field(ge=0)


class ScoreDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stats: ScoreStats
    histogram: list[HistogramBucket]


# ---------------------------------------------------------------------------
# LGB 单元（screening / prediction / market candidates 复用）
# ---------------------------------------------------------------------------


class LgbCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float | None
    rank: str | None  # "d1" .. "d10"


# ---------------------------------------------------------------------------
# step2_screening
# ---------------------------------------------------------------------------


class ScreeningItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(ge=1)
    code: str
    name: str
    close: float
    score: float
    lgb: LgbCell
    level: Literal["强", "中", "弱"]
    theme: str
    rationale: str
    tags: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItemStrict]
    missingData: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# step4_prediction
# ---------------------------------------------------------------------------

PredictionGroup = Literal["top_candidate", "watchlist", "avoid"]
DeltaVsBatch = Literal["upgraded", "kept", "downgraded"]


class PredictionCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(ge=1)  # 多批用 final_rank，单批用 rank
    code: str
    name: str
    prediction: PredictionGroup
    confidence: Literal["高", "中", "低"]
    score: float
    close: float
    lgb: LgbCell
    rationale: str
    observationPoints: list[str]
    failureConditions: list[str]
    keyEvidence: list[EvidenceItemStrict]
    missingData: list[str] = Field(default_factory=list)
    # 多批模式专用，单批为 None
    batchLocalRank: int | None
    deltaVsBatch: DeltaVsBatch | None
    reasonVsPeers: str | None


class Step4Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_candidate: list[PredictionCard] = Field(default_factory=list)
    watchlist: list[PredictionCard] = Field(default_factory=list)
    avoid: list[PredictionCard] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# marketSnapshot
# ---------------------------------------------------------------------------


class LimitStepTrend(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_height: int = Field(ge=0)
    max_height_prev: int = Field(ge=0)
    high_board_delta: int
    total_limit_up_delta: int
    interpretation: str  # "spectrum_lifting" / "spectrum_collapsing" / "stable"


class YesterdayFailureRate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trade_date_prev: str
    u_count: int = Field(ge=0)
    z_count: int = Field(ge=0)
    rate_pct: float | None
    interpretation: str | None  # "high" / "moderate" / "low" / None


class YesterdayWinnersToday(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trade_date_prev: str
    n_winners: int = Field(ge=0)
    n_continued_today: int = Field(ge=0)
    continuation_rate_pct: float | None
    n_negative_today: int = Field(ge=0)
    avg_pct_chg_today: float | None
    interpretation: str | None  # "strong_money_effect" / "neutral" / "weak_money_effect" / None


class MarketCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    name: str
    close: float | None
    float_mv: float | None  # 亿元
    theme: str
    lgb: LgbCell


class MarketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit_up_count: int = Field(ge=0)
    limit_step_distribution: dict[str, int] = Field(default_factory=dict)
    limit_step_distribution_prev: dict[str, int] = Field(default_factory=dict)
    limit_step_trend: LimitStepTrend | None
    yesterday_failure_rate: YesterdayFailureRate | None
    yesterday_winners_today: YesterdayWinnersToday | None
    candidates: list[MarketCandidate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# filteringDetails
# ---------------------------------------------------------------------------


class FilteringThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_float_mv_yi: float
    max_float_mv_yi: float
    max_close_yuan: float


class RejectedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    name: str
    float_mv: float | None
    close: float | None
    reason: str  # 已 join 成 "float_mv>200.0, close>30.0"


class FilteringLog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entered: int = Field(ge=0)
    passed: int = Field(ge=0)
    rejected: int = Field(ge=0)
    thresholds: FilteringThresholds
    rejectedItems: list[RejectedItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 根模型
# ---------------------------------------------------------------------------


class StrategyReportSchema(BaseModel):
    """根模型 —— 前端直接 ``await fetch(url).json()`` 后按此结构消费。

    与原 schema 提案的偏离详见 ``frontend_migration_notes.md``。
    """

    # 根模型 forbid extras：未来新增字段必须显式合入 schema；
    # 但 ``_extras`` 通过 alias 允许构造期既能传 ``extras=`` 也能传 ``_extras=``。
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    meta: ReportMeta
    marketSnapshot: MarketSnapshot
    # LGB 未启用 / 无候选时为 None，前端整块不渲染
    scoreDistribution: ScoreDistribution | None
    step2_screening: list[ScreeningItem] = Field(default_factory=list)
    step4_prediction: Step4Prediction
    filteringDetails: FilteringLog
    # market_summary 里未显式映射的 key 全部兜底进来；前端可忽略
    extras: dict[str, Any] = Field(default_factory=dict, alias="_extras")
