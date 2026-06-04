"""测试全局隔离：plugin_data_dir（报告落盘等）一律指向 tmp，绝不碰真实 ~/.deeptrade。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_plugin_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import deeptrade.core.paths as fw_paths

    monkeypatch.setattr(fw_paths, "db_path", lambda: tmp_path / "deeptrade.duckdb")
    yield
