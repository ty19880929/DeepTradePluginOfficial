"""Pure metric aggregations on the synced ``mr_*`` tables (design §5.3).

Seven modules, one per design subsection:

- :mod:`market_review.metrics.breadth`   — §5.3.1 市场宽度
- :mod:`market_review.metrics.sentiment` — §5.3.2 情绪温度计
- :mod:`market_review.metrics.capital`   — §5.3.3 多口径资金流
- :mod:`market_review.metrics.sectors`   — §5.3.4 板块轮动
- :mod:`market_review.metrics.leaders`   — §5.3.5 龙头识别
- :mod:`market_review.metrics.style`     — §5.3.6 风格切换
- :mod:`market_review.metrics.risk`      — §5.3.7 风险信号

Every module follows the same shape:

1. Public entry: ``compute_<x>(db, window, universes, *, ...) -> XReview``
2. Read-only against the DB (no Tushare, no mutate).
3. Result is a pure dataclass / dict tree — JSON-serializable by PR-5
   ``report/schema.py`` without further transformation.

Tests live in ``tests/test_metrics_<x>.py``; conftest's ``mr_db`` fixture
gives each test a fresh DuckDB with the migrations applied.
"""
