"""Report assembly + upload — design §15.

Three modules:

- :mod:`market_review.report.schema`  — :class:`ReviewReportSchema` root +
  :class:`MetricsBlock` (structured numerics, design §15.6) + window /
  headline / meta sub-models. Strict (``extra="forbid"``) with a single
  ``_extras`` escape hatch on the root for forward-compat.
- :mod:`market_review.report.builder` — :func:`build_review_report`,
  the *pure* assembly function that takes PR-3 metric reviews + PR-4
  section schemas and returns a :class:`ReviewReportSchema`. No IO,
  no DB, no LLM — designed for testable, deterministic assembly.
- :mod:`market_review.report.upload`  — :func:`maybe_upload_summary`,
  the framework :class:`ReportUploader` adapter. Best-effort POST of
  ``summary.json`` to the user's configured endpoint; emits
  :class:`StrategyEvent` records for ok / failed / skipped paths.
"""
