"""Verify a release tag against the plugin's deeptrade_plugin.yaml.

Called from plugin-release.yml when a tag of form `<plugin-id>/v<X.Y.Z>` is pushed.

Usage: python tools/check_release.py <plugin_id> <version>

Checks:
  - plugin_id is in registry/index.json
  - subdir/deeptrade_plugin.yaml exists
  - yaml.plugin_id == plugin_id (from tag)
  - yaml.version == version (from tag)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main(plugin_id: str, version: str) -> int:
    errors: list[str] = []

    registry_path = ROOT / "registry" / "index.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    plugins = registry["plugins"]

    if plugin_id not in plugins:
        print(
            f"FAIL: plugin_id {plugin_id!r} not in registry/index.json. "
            f"Known: {sorted(plugins)}",
            file=sys.stderr,
        )
        return 1

    entry = plugins[plugin_id]
    subdir = ROOT / entry["subdir"]
    yaml_path = subdir / "deeptrade_plugin.yaml"
    if not yaml_path.is_file():
        print(f"FAIL: missing {yaml_path.relative_to(ROOT)}", file=sys.stderr)
        return 1

    yaml_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if yaml_data.get("plugin_id") != plugin_id:
        errors.append(
            f"yaml.plugin_id={yaml_data.get('plugin_id')!r} != tag {plugin_id!r}"
        )

    if yaml_data.get("version") != version:
        errors.append(
            f"yaml.version={yaml_data.get('version')!r} != tag {version!r}; "
            f"bump the version in {yaml_path.relative_to(ROOT)} before tagging"
        )

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"OK: {plugin_id} v{version} matches yaml")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python tools/check_release.py <plugin_id> <version>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
