# Repository Guidelines

## Project Structure & Module Organization

This repository is a DeepTrade official plugin monorepo. `registry/index.json` is the public plugin registry consumed by `deeptrade plugin install <short-name>`. Shared maintenance scripts live in `tools/`.

Each plugin is self-contained:

- `limit_up_board/` and `volume_anomaly/`: plugin roots.
- `<plugin>/deeptrade_plugin.yaml`: plugin metadata, dependencies, migrations, and entrypoint.
- `<plugin>/migrations/`: SQL migrations referenced by checksum from the YAML file.
- `<plugin>/<python_package>/`: source code, including `plugin.py`, `cli.py`, `runner.py`, `pipeline.py`, `schemas.py`, `ui/`, and `lgb/` modules.
- `<plugin>/tests/`: pytest suites and snapshot fixtures.

## Build, Test, and Development Commands

- `python tools/check_registry.py`: validates registry schema, plugin metadata consistency, and migration checksums.
- `python tools/check_release.py limit-up-board 0.5.1`: verifies a release tag version against the plugin YAML.
- `python -m pytest limit_up_board`: runs the `limit_up_board` tests using its `pytest.ini`.
- `python -m pytest volume_anomaly`: runs the `volume_anomaly` tests.
- `python -m pytest limit_up_board -m "not slow"`: skips long-running LightGBM tests where marked.

Run commands from the repository root unless a test requires plugin-local paths.

## Coding Style & Naming Conventions

Use Python 3 style with `from __future__ import annotations`, explicit type hints where useful, and 4-space indentation. Keep module and package names snake_case. Plugin IDs and release tags use kebab-case, for example `volume-anomaly/v0.8.3`.

Keep framework entry classes in `plugin.py` small: static validation should avoid network access and defer runtime work to `cli.py`, `runner.py`, or `pipeline.py`. Do not add generated caches, model artifacts, or virtual environments to the repository.

## Testing Guidelines

Tests use pytest. Place tests under the matching plugin’s `tests/` directory and name files `test_*.py`. Snapshot outputs live under `tests/snapshots/`; update them only when UI/rendering output changes intentionally.

For model or LightGBM work, include focused unit coverage for dataset creation, labels, features, scoring, registry behavior, and prompt-injection safeguards when relevant.

## Commit & Pull Request Guidelines

Prefer scoped commit messages matching recent history, such as `feat(volume-anomaly): ...`, `fix(volume-anomaly): ...`, or `volume-anomaly v0.8.0 (PR-1/3): ...`. Avoid vague messages like `update` for reviewable work.

Pull requests should describe the changed plugin, user-visible behavior, tests run, and any migration or registry impact. Link issues when available. Include screenshots or snapshot diffs for UI/dashboard changes.

## Release & Configuration Notes

Each plugin version is independent and must match `deeptrade_plugin.yaml`. Release tags use `<plugin-id>/v<X.Y.Z>`, for example `limit-up-board/v0.5.1`. When changing migrations, update the YAML checksum and confirm `python tools/check_registry.py` passes.
