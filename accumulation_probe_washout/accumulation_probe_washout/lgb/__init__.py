"""LightGBM subsystem for accumulation-probe-washout (lands in v0.4.0+).

Layout:
    paths       — on-disk locations for models / datasets / checkpoints
    features    — FEATURE_NAMES / SCHEMA_VERSION / build_feature_frame

Heavier modules (labels / dataset / trainer / scorer / evaluate / registry /
checkpoint / cleanup) ship in PR-3 / PR-4. ``__init__`` intentionally stays
empty so the new feature module can be imported without dragging in
lightgbm / sklearn (PR-3 dependencies).
"""
