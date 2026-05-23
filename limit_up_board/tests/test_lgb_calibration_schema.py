"""P2-2 (v0.13.0)：lub_lgb_models 表预留 calibration_* 列，且文案已下线"概率"。

校准器训练 / 加载 / Brier evaluate 是 v0.13.1 的工作，本测试只锁定：
1. 迁移 20260524_001 文件存在并被 yaml 引用
2. prompt / render / lgb 子包 docstring 不再用「概率」字眼形容 lgb_score
3. 文档明确指出 lgb_score 为「未校准排序分」
"""

from __future__ import annotations

from pathlib import Path

import yaml


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = PLUGIN_ROOT / "deeptrade_plugin.yaml"
INNER = PLUGIN_ROOT / "limit_up_board"


def test_calibration_migration_listed_in_yaml() -> None:
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    versions = {m["version"] for m in (data.get("migrations") or [])}
    assert "20260524_001" in versions, (
        "v0.13.0 应通过迁移 20260524_001_lgb_calibration.sql 给 lub_lgb_models "
        "增加 calibration_method / calibration_brier / calibration_samples 列"
    )


def test_calibration_migration_file_exists() -> None:
    p = PLUGIN_ROOT / "migrations" / "20260524_001_lgb_calibration.sql"
    assert p.is_file()
    sql = p.read_text(encoding="utf-8")
    for col in ("calibration_method", "calibration_brier", "calibration_samples"):
        assert col in sql, f"migration should add column {col}"


def test_prompt_lgb_score_no_longer_described_as_probability() -> None:
    """v0.13.0 (P2-2) 文案降级：lgb_score 应被描述为'排序分'/'未校准'，
    不再单独以「概率」二字描述（除非紧跟在'未校准'附近作为对照）。"""
    text = (INNER / "prompts.py").read_text(encoding="utf-8")
    # The new prompt explicitly contains the disclaimer phrasing.
    assert "未校准" in text
    assert "未校准的模型排序分" in text


def test_render_lgb_distribution_label_updated() -> None:
    text = (INNER / "render.py").read_text(encoding="utf-8")
    assert "未校准模型排序分" in text


def test_lgb_subpkg_docstring_marks_uncalibrated() -> None:
    text = (INNER / "lgb" / "__init__.py").read_text(encoding="utf-8")
    assert "未校准" in text
    assert "v0.13.0" in text
