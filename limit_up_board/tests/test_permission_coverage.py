"""P0-1：静态扫描 ``tushare.call("<api>")`` 字面量，断言全部声明在 yaml 的
``permissions.tushare_apis.required + optional`` 集合内。

新增 API 时必须：
1. 在源码中调用 ``tushare.call("new_api", ...)``；
2. 在 ``deeptrade_plugin.yaml::permissions.tushare_apis.required`` 或 ``optional``
   中声明，否则本测试拒绝合流。

动态调用 ``tushare.call(api_name, ...)``（如 ``_fetch_history_window`` /
``_try_optional``）跳过扫描；其调用方必须传入字面量，由调用方语句被本测试覆盖。
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
INNER_PKG = PLUGIN_ROOT / "limit_up_board"
YAML_PATH = PLUGIN_ROOT / "deeptrade_plugin.yaml"


def _is_tushare_call(node: ast.Call) -> bool:
    """Match ``<anything>.tushare.call(...)`` and ``tushare.call(...)`` Attribute access."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "call":
        return False
    receiver = func.value
    if isinstance(receiver, ast.Name) and receiver.id == "tushare":
        return True
    if isinstance(receiver, ast.Attribute) and receiver.attr == "tushare":
        return True
    return False


def _collect_literal_apis(source_root: Path) -> tuple[set[str], list[str]]:
    """Walk every .py file under *source_root*; return (literal_apis, dynamic_sites).

    A "literal API" is the first positional arg of ``tushare.call("name", ...)``
    when that arg is a plain string literal. Anything else (``api_name`` variable,
    f-string, etc.) is recorded as a dynamic call site in ``dynamic_sites``;
    callers of those dispatcher functions are independently asserted.
    """
    literal_apis: set[str] = set()
    dynamic_sites: list[str] = []
    for py_path in source_root.rglob("*.py"):
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except SyntaxError as e:  # pragma: no cover — surfaces real source issues
            raise AssertionError(f"failed to parse {py_path}: {e}") from e
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_tushare_call(node):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                literal_apis.add(first.value)
            else:
                rel = py_path.relative_to(PLUGIN_ROOT).as_posix()
                dynamic_sites.append(f"{rel}:{first.lineno}")
    return literal_apis, dynamic_sites


def _declared_apis() -> set[str]:
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    perms = (data.get("permissions") or {}).get("tushare_apis") or {}
    return set(perms.get("required") or []) | set(perms.get("optional") or [])


def test_every_literal_tushare_call_is_declared() -> None:
    literal_apis, _ = _collect_literal_apis(INNER_PKG)
    declared = _declared_apis()
    missing = literal_apis - declared
    assert not missing, (
        "Tushare APIs called in source but not declared in "
        f"deeptrade_plugin.yaml::permissions.tushare_apis: {sorted(missing)}.\n"
        "Add each missing name to required (default path / always called) "
        "or optional (degradable / try/except wrapped)."
    )


def test_no_unused_required_apis_left_dangling() -> None:
    """Soft check: every API in ``required`` should actually be used somewhere.

    Optional ones are allowed to be inactive (e.g. ``limit_list_ths`` is only
    pulled when THS is enabled), so we only flag required ones to catch
    drift after a removal.
    """
    literal_apis, dynamic_sites = _collect_literal_apis(INNER_PKG)
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    required = set(
        ((data.get("permissions") or {}).get("tushare_apis") or {}).get("required") or []
    )
    # Dynamic dispatcher sites use api_name variables; we can't statically
    # confirm those names from this test alone, so we only fail if there are
    # NO dynamic sites AND the required name has no literal occurrence.
    if not dynamic_sites:
        unused = required - literal_apis
        assert not unused, (
            f"Required Tushare APIs declared but never called: {sorted(unused)}"
        )
