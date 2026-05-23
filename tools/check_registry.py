"""Validate registry/index.json against actual plugin sources.

Checks:
  - index.json is well-formed and matches the schema (schema_version=1)
  - Each entry has all required fields
  - Each entry's `subdir` exists and contains deeptrade_plugin.yaml
  - yaml.plugin_id == index key
  - yaml.name == index name; yaml.type == index type
  - tag_prefix ends with "/"
  - repo is owner/repo form
  - latest_version is a non-empty string that starts with tag_prefix
    (consumed by framework >= v0.8 to skip the GitHub Releases API call)
  - Each migration's sha256 checksum matches the file content
    (matches the framework's _verify_migration_checksum logic)
  - Every literal ``tushare.call("<api>")`` in the plugin's Python sources
    is declared in ``permissions.tushare_apis.required`` or ``optional``
    (P0-1: prevents the install-time silent-403 first-run failure mode).
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FIELDS = {
    "name", "type", "description", "repo",
    "subdir", "tag_prefix", "min_framework_version",
    "latest_version",
}
ALLOWED_TYPES = {"strategy", "channel"}


def _is_tushare_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "call":
        return False
    receiver = func.value
    if isinstance(receiver, ast.Name) and receiver.id == "tushare":
        return True
    if isinstance(receiver, ast.Attribute) and receiver.attr == "tushare":
        return True
    return False


def _missing_tushare_apis(plugin_subdir: Path, yaml_data: dict) -> set[str]:
    """Return the set of literal Tushare APIs called by source under
    *plugin_subdir* that are NOT declared in
    ``permissions.tushare_apis.required`` ∪ ``optional``.

    Dynamic dispatch sites (``tushare.call(api_name, ...)``) are skipped;
    their literal call sites in callers are covered separately.
    """
    perms = (yaml_data.get("permissions") or {}).get("tushare_apis") or {}
    declared: set[str] = set(perms.get("required") or []) | set(perms.get("optional") or [])

    literal_apis: set[str] = set()
    for py_path in plugin_subdir.rglob("*.py"):
        # Skip test files — they may stub or assert against API names not
        # actually called by the plugin runtime.
        if "tests" in py_path.relative_to(plugin_subdir).parts:
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_tushare_call(node):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                literal_apis.add(first.value)
    return literal_apis - declared


def main() -> int:
    errors: list[str] = []

    registry_path = ROOT / "registry" / "index.json"
    if not registry_path.is_file():
        print(f"FAIL: missing {registry_path}", file=sys.stderr)
        return 1

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"FAIL: invalid JSON: {e}", file=sys.stderr)
        return 1

    if registry.get("schema_version") != 1:
        errors.append(f"schema_version must be 1, got {registry.get('schema_version')!r}")

    plugins = registry.get("plugins")
    if not isinstance(plugins, dict) or not plugins:
        errors.append("plugins must be a non-empty object")
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    for plugin_id, entry in plugins.items():
        prefix = f"[{plugin_id}]"

        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
            continue

        if entry["type"] not in ALLOWED_TYPES:
            errors.append(f"{prefix} type must be one of {ALLOWED_TYPES}, got {entry['type']!r}")

        if not entry["tag_prefix"].endswith("/"):
            errors.append(f"{prefix} tag_prefix must end with '/'")

        if "/" not in entry["repo"]:
            errors.append(f"{prefix} repo must be in 'owner/repo' form, got {entry['repo']!r}")

        latest_version = entry["latest_version"]
        if not isinstance(latest_version, str) or not latest_version:
            errors.append(f"{prefix} latest_version must be a non-empty string")
        elif not latest_version.startswith(entry["tag_prefix"]):
            errors.append(
                f"{prefix} latest_version {latest_version!r} must start with "
                f"tag_prefix {entry['tag_prefix']!r} (use the full tag, e.g. "
                f"{entry['tag_prefix']}v0.1.0)"
            )

        subdir = ROOT / entry["subdir"]
        if not subdir.is_dir():
            errors.append(f"{prefix} subdir {entry['subdir']!r} does not exist")
            continue

        yaml_path = subdir / "deeptrade_plugin.yaml"
        if not yaml_path.is_file():
            errors.append(f"{prefix} missing {yaml_path.relative_to(ROOT)}")
            continue

        try:
            yaml_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            errors.append(f"{prefix} cannot parse yaml: {e}")
            continue

        if yaml_data.get("plugin_id") != plugin_id:
            errors.append(
                f"{prefix} yaml.plugin_id={yaml_data.get('plugin_id')!r} "
                f"!= index key {plugin_id!r}"
            )

        if yaml_data.get("name") != entry["name"]:
            errors.append(
                f"{prefix} yaml.name={yaml_data.get('name')!r} "
                f"!= index.name={entry['name']!r}"
            )

        if yaml_data.get("type") != entry["type"]:
            errors.append(
                f"{prefix} yaml.type={yaml_data.get('type')!r} "
                f"!= index.type={entry['type']!r}"
            )

        missing_apis = _missing_tushare_apis(subdir, yaml_data)
        if missing_apis:
            errors.append(
                f"{prefix} tushare.call() literals not declared in "
                f"permissions.tushare_apis: {sorted(missing_apis)}"
            )

        for mig in yaml_data.get("migrations", []) or []:
            file_rel = mig.get("file")
            expected = mig.get("checksum")
            version = mig.get("version", "?")
            if not file_rel or not expected:
                errors.append(f"{prefix} migration {version!r} missing file or checksum")
                continue
            mig_path = subdir / file_rel
            if not mig_path.is_file():
                errors.append(f"{prefix} migration file missing: {file_rel}")
                continue
            sql_text = mig_path.read_text(encoding="utf-8")
            actual = "sha256:" + hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
            if actual != expected:
                errors.append(
                    f"{prefix} checksum mismatch for {file_rel}\n"
                    f"  expected: {expected}\n"
                    f"  actual:   {actual}"
                )

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"OK: registry has {len(plugins)} plugins, all consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
