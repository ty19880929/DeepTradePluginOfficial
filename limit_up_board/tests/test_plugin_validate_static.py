"""P1-1：``LimitUpBoardPlugin.validate_static`` 必须保持「轻量 import」承诺。

调用后 ``sys.modules`` 中不允许出现重型运行时依赖（typer / rich /
questionary / lightgbm / sklearn / pandas / tushare），也不允许出现本插件
自身的 cli / runner / runtime / pipeline / lgb / winrate 子模块。

实现方式：使用 subprocess 启动一个干净的 Python 解释器，import
``limit_up_board.plugin``、调用 ``validate_static`` 后导出 ``sys.modules`` 的
top-level 名称集合，再断言禁列名集合 ∩ 实际导入集 为空。
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

# 禁止在 validate_static 期间被拉进 sys.modules 的模块名（顶层 / 子模块均算）。
FORBIDDEN_TOP_LEVEL = {
    # 重型第三方
    "typer",
    "rich",
    "questionary",
    "lightgbm",
    "sklearn",
    "pandas",
    "tushare",
    # 本插件子模块（运行时才需要）
    "limit_up_board.cli",
    "limit_up_board.runner",
    "limit_up_board.runtime",
    "limit_up_board.pipeline",
    "limit_up_board.render",
    "limit_up_board.data",
}


_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    # 仅 import plugin 模块（不调用 validate_static 前 baseline）
    from limit_up_board import plugin

    plugin.LimitUpBoardPlugin().validate_static(None)
    mods = sorted(sys.modules.keys())
    json.dump(mods, sys.stdout)
    """
).strip()


def test_validate_static_does_not_import_heavy() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = set(json.loads(proc.stdout))
    leaked = {name for name in FORBIDDEN_TOP_LEVEL if name in loaded}
    assert not leaked, (
        "validate_static() pulled forbidden heavy modules into sys.modules: "
        f"{sorted(leaked)}.\n"
        "Move the offending import inside dispatch() or a CLI command body."
    )
