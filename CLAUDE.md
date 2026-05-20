# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A monorepo of **official plugins** for the [DeepTrade](https://github.com/ty19880929/deeptrade) framework — an LLM-driven A-share stock-screening CLI. Plugins are not imported here directly; the framework's `deeptrade plugin install <short-name>` resolves them by reading `registry/index.json`, calling the GitHub Releases API for the matching `tag_prefix`, and pulling the plugin source from the resolved tag's `subdir`.

Three plugins live here today:

- `limit_up_board/` — strategy `limit-up-board` (打板策略, dual-round LLM funnel)
- `accumulation_probe_washout/` — strategy `accumulation-probe-washout` (吸筹→试盘→洗盘→主升浪启动链路 + LightGBM 评分)
- `checkmate/` — strategy `checkmate` (A 股 long-only 中期趋势跟踪)

The framework itself lives in another repo and is **not vendored**. Imports like `from deeptrade.core.db import Database` resolve at install time inside the user's `pipx install deeptrade-quant` environment; running anything in this repo locally requires `deeptrade-quant` to be importable.

## Commands

Run from the repo root unless noted.

```powershell
# Validate registry/index.json against every plugin's yaml + migration checksums.
# This is the same check CI runs on every PR / push to main.
python tools/check_registry.py

# Verify a release tag matches the plugin yaml. Only used by the plugin-release workflow,
# but you can run it locally to dry-run a release before pushing the tag.
python tools/check_release.py <plugin-id> <X.Y.Z>
```

Per-plugin tests (run from the plugin's own directory — each has its own `pytest.ini` with `pythonpath = .`):

```powershell
# limit-up-board
cd limit_up_board ; pytest                               # full suite
cd limit_up_board ; pytest tests/test_v04_settings.py    # single file
cd limit_up_board ; pytest tests/test_phase_a_factors.py::TestName::test_case  # single test

# accumulation-probe-washout (same shape)
cd accumulation_probe_washout ; pytest
```

Tests import the plugin under its own package name (e.g. `from limit_up_board.config import ...`), and they import `deeptrade.*` from the installed framework. If `pytest` errors with `ModuleNotFoundError: deeptrade`, the framework is missing from the active interpreter — install it (`pipx install deeptrade-quant` or pip in a venv).

## Repo architecture

### Two layers per plugin: outer dir vs. inner package

```
limit_up_board/                 # outer dir = registry "subdir"
├── deeptrade_plugin.yaml       # manifest the framework reads at install
├── migrations/                 # SQL migrations referenced by the yaml
├── pytest.ini
├── tests/
└── limit_up_board/             # inner Python package — same name on purpose
    ├── plugin.py               # entrypoint class (yaml: entrypoint: limit_up_board.plugin:LimitUpBoardPlugin)
    ├── cli.py                  # typer CLI; `dispatch()` forwards argv here
    ├── runtime.py              # LubRuntime dataclass — bundles db / config / llms / tushare
    ├── runner.py, pipeline.py, data.py, schemas.py, prompts.py, render.py, ...
```

The outer directory name is what `registry/index.json` references via `subdir`. The inner package name is what the yaml's `entrypoint` references and what `tests/` imports. They match by convention; do not rename one without the other.

### The Plugin Protocol

The framework expects three things on the entrypoint class (`plugin.py`):

- `metadata` — class attribute set to `None`; the framework injects the real metadata after install.
- `validate_static(ctx)` — install-time, no-network sanity check. The current plugins just import `schemas` to surface import errors.
- `dispatch(argv)` — the framework forwards `deeptrade <plugin-id> <argv...>` here verbatim. Strategy plugins delegate to `cli.main(argv)`; channel plugins also implement `push(ctx, payload)`.

`runtime.py` defines a `*Runtime` dataclass that bundles `Database`, `ConfigService`, `LLMManager`, plus optional `TushareClient` and `run_id`. The CLI builds one via `_open_runtime()`, hands it to the runner, and closes the DB in a `finally`. The runner pulls per-provider LLM clients on demand via `rt.llms.get_client(name, plugin_id=rt.plugin_id, run_id=rt.run_id)` rather than holding a single LLM client field — this is the v0.6 contract and applies to both strategy plugins.

`limit_up_board.runtime.open_worker_runtime` documents a critical invariant for debate-mode (multi-LLM concurrent) workers: each worker gets its own `Database` connection and `LLMManager`, but the `ConfigService` is **shared with the main thread on purpose** (its `SecretStore` keyring probe is racy under per-worker construction). That sharing implies `Database.fetchone/fetchall` must hold the write lock across the full execute+fetch round-trip — see the docstring before touching `deeptrade.core.db` interactions.

### CLI surface (current subcommands)

- `limit-up-board`: `run`, `sync`, `history`, `report`, `settings show`, plus the **`lgb`** subcommand group: `train` / `evaluate` / `info` / `list` / `activate` / `prune` / `purge` / `refresh-features`. `run --no-lgb` is a one-shot opt-out. `run --no-dashboard` (v0.6+) disables the rich animated dashboard and falls back to the v0.5.x-compatible line-per-event stream. `run --llm <provider>` (v0.6.8+) pins the non-debate-mode LLM provider for one run (overrides the framework default; mutex with `--debate` / `--debate-llms`; unknown / unconfigured provider fails fast via `PreconditionError` so no `lub_runs` row is written).
- `accumulation-probe-washout`: `screen`, `analyze`, `run`, `prune`, `evaluate`, `stats`, `history`, `report`, `settings show|set|reset`, plus the **`lgb`** subcommand group: `train` / `evaluate` / `info` / `list` / `activate` / `prune` / `purge`. `analyze --no-lgb` / `run --no-lgb` are one-shot opt-outs; `stats --by lgb_score_bin` aggregates `apw_lgb_predictions` ⋈ `apw_realized_returns`. **`screen --backfill-history --start --end [--overwrite]` (v0.3.0+)** batch-replays the LLM-free screen rules over a historical range to bootstrap `apw_signal_history` (so `lgb train` has a training corpus without weeks of daily `screen` runs). Backfill writes only to `apw_signal_history`, never `apw_watchlist`; default behaviour skips dates that already have rows (DB-resident resume), `--overwrite` does a wholesale `DELETE … WHERE trade_date=?` before re-inserting. Forced legacy renderer (the dashboard's single-day StageStack would mislead). `prune` (v0.3.0+) is phase-aware: launch_ready idle ≥ `prune_idle_days_launch_ready` / washout past `washout_max_trade_days` / close < probe_low / close < MA60 → delete; `--dry-run` reports candidates without writing.

These are exposed to users as `deeptrade <plugin-id> <subcommand>` once the framework dispatches into `cli.main(argv)`.

#### `limit-up-board lgb` (v0.5+)

The `lgb` group manages the LightGBM 连板概率 booster lifecycle (offline training → registered model file → R1/R2 prompt-side scoring). Highlights:

- `lgb train --start --end` fits a new booster (GroupKFold by trade_date) and registers it in `lub_lgb_models`. Each model file lands under `~/.deeptrade/limit_up_board/models/`; the training matrix snapshot under `datasets/` (parquet) so `lgb evaluate --drift` can compare distributions later. **Phase-1 (data collection) is checkpointed (v0.5.5+)**: per-day shards land under `~/.deeptrade/limit_up_board/checkpoints/<digest>/days/<YYYYMMDD>.parquet`, keyed by a fingerprint of (training window + filter params + `SCHEMA_VERSION` + lookbacks). If the run crashes or is Ctrl-C'd, re-running the same command auto-resumes from the last completed day; `--fresh` discards the checkpoint and re-fetches. On training success the checkpoint dir is deleted; on `train_lightgbm` failure the shards remain so a retry doesn't redo Tushare pulls. LightGBM fit itself is not checkpointed (deterministic, seed=42).
- `lgb evaluate --start --end [--model-id]` reports AUC / logloss / Top-K hit-rate vs per-day baseline; JSON dump under `reports/`.
- `lgb evaluate --drift --baseline <id>` adds per-feature PSI (10-bin) against the baseline model's training snapshot. Output is sorted by PSI desc with status labels (`stable` < 0.10, `moderate` < 0.25, `shift` ≥ 0.25).
- `lgb info [--model-id] [--recent-N]` shows model metadata + how many runs / trade dates have used it, with an optional per-day score-distribution snapshot.
- `lgb prune --keep N` is the maintenance broom (keeps active + N most-recent; deletes the rest). `lgb purge --datasets / --models / --predictions / --checkpoints / --all` is the scorched-earth alternative for "I want to reset / reclaim disk"; both require explicit scope flags and the latter prompts for confirmation unless `--yes`. `--checkpoints` wipes all in-flight Phase-1 training shards.
- Inference (`run`) is wired through `LubRuntime.lgb_scorer`; failure paths (no active model / file missing / schema mismatch / predict raise / `lightgbm` not installed) all degrade to `lgb_score=None` without blocking the LLM stages. See `lgb/scorer.py` for the 5-branch contract and `lightgbm_design.md §7.3`.

#### `limit-up-board run` dashboard (v0.6+)

`run` (and `sync`) drive output through an `EventRenderer` protocol (`limit_up_board/ui/protocol.py`). The factory `limit_up_board.ui.choose_renderer(no_dashboard=False)` picks between two implementations:

- **`RichDashboardRenderer`** — rich `Live` region with header / config / stages / (debate grid) / log panels (Plan §4.1, §4.2). Single-LLM mode uses an event-driven `StageStack` (Step 0/1/2/4/4.5/5); debate mode adds a Provider × {Phase A, R3 修订, 备注} summary grid on top of Step 0/1.
- **`LegacyStreamRenderer`** — one stdout line per event (`  {glyph} [{event_type}] {message}`), byte-identical to v0.5.x.

Fallback to legacy whenever any of: caller passed `--no-dashboard`; `sys.stdout` not a TTY (pipes, redirects, pytest capture); `CI` truthy; `DEEPTRADE_NO_DASHBOARD` truthy; `TERM=dumb`. `NO_COLOR=1` keeps the dashboard but disables Console colour. UI failures never crash a run — `LubRunner._dispatch_to_renderer` catches and degrades to legacy mid-stream (Plan §3.6.1). Pipeline / data / lgb / report subsystems are renderer-unaware: the dashboard only consumes `StrategyEvent` instances, never mutates them, never writes to DB. See `docs/limit-up-board/CLI_UI_Redesign_Plan.md` for the full design.

#### `accumulation-probe-washout lgb` (v0.5+)

Mirrors the `limit-up-board lgb` design with APW-specific twists:

- **Labels come from `apw_realized_returns`** — `apw_lgb_models` records `label_source` (`label_launch_t5` (default) / `label_launch_t10` / `custom_t5`) and `label_threshold_pct` (only used for `custom_t5`). The `label_launch_t5` / `label_launch_t10` columns are pre-computed by `evaluate` and already encode the configured 收益 + 回撤 envelope, so training does **zero** extra Tushare calls.
- `lgb train --start --end [--label-source] [--label-threshold] [--label-drawdown-threshold] [--folds] [--no-activate] [--fresh] [--keep-checkpoint]` fits a new booster (GroupKFold by signal_date, no same-day leakage). Phase-1 collection is checkpointed by BLAKE2b-64 fingerprint of (window + label config + `SCHEMA_VERSION` + 3 lookbacks + `baseline_index_code` + `volume_adjust_enabled`); shards land under `<plugin_data_dir>/checkpoints/<digest>/days/<YYYYMMDD>.parquet`. Train success deletes the dir; failures keep shards for resume.
- `lgb evaluate --start --end [--model-id] [--k] [--drift --baseline <id>]` runs AUC / logloss / Top-K hit-rate vs per-day baseline; label config auto-read from `apw_lgb_models`. `--drift` adds 10-bin PSI vs the baseline model's `dataset.parquet` snapshot, sorted by PSI desc with `stable` / `moderate` / `shift` status. JSON dumps under `<plugin_data_dir>/reports/lgb_evaluate_*.json` and `reports/lgb_drift_*.json`.
- `lgb info [--model-id]` prints registry metadata (CV AUC, label config, on-disk paths).
- `lgb list` (★ = active), `lgb activate <id>`, `lgb prune --keep N`, `lgb purge --datasets / --models / --predictions / --checkpoints / --all [--yes]`.
- Inference (`analyze` / `run`) is wired through `ApwRuntime.lgb_scorer` (Step 1.5 before LLM); failure paths (no active / file missing / schema mismatch / predict raise / `lightgbm` not installed) all degrade to `lgb_score=None` without blocking LLM. `analyze --no-lgb` / `run --no-lgb` is one-shot; `ApwConfig.lgb_enabled=false` is persistent (settable via `settings set`). Rows with `lgb_score < lgb_min_score_floor` get a `low_lgb_score` tag in `risk_flags_local` — visible to LLM, doesn't filter the candidate.

Third-party runtime deps are declared in `deeptrade_plugin.yaml::dependencies` (PEP 508). The framework `uv pip install`s them before `validate_static` runs (v0.4.0+; see `plugin_required_dependencies.md`). For `limit-up-board` this includes `tushare`, `pandas`, `numpy`, `lightgbm`, and `scikit-learn`; for `accumulation-probe-washout` (v0.5.0+), `tushare`, `pandas`, `pyarrow`, `lightgbm`, and `scikit-learn`.

### Per-plugin DB tables

Every plugin owns its tables and prefixes them (e.g. `lub_*`, `apw_*`, `checkmate_*`). Each table is declared in the yaml's `tables:` block with `purge_on_uninstall: true` so the framework can clean up. Plugins replace the framework's shared `strategy_runs` / `strategy_events` with their own `*_runs` / `*_events` tables (this is the post-v0.4 plugin-owned-history pattern).

## Release flow (matters for any change touching a plugin)

1. Edit code in the plugin subdir.
2. Bump `version:` in that plugin's `deeptrade_plugin.yaml`.
3. If you changed a SQL migration, **regenerate its sha256 checksum** in the yaml. `tools/check_registry.py` will fail CI if `sha256:<hex>` of the file doesn't match. Use the framework's `_verify_migration_checksum` logic, or just `python -c "import hashlib,sys; print('sha256:'+hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" path/to/file.sql`.
4. Open PR → `registry-check.yml` runs `tools/check_registry.py`.
5. After merge, push tag `<plugin-id>/v<X.Y.Z>` (e.g. `limit-up-board/v0.4.1`). `plugin-release.yml` runs `tools/check_release.py` (yaml.version must equal tag version) and creates a GitHub Release.

The framework's installer only resolves **published Releases**, so a tag without a Release is invisible to `deeptrade plugin install`.

## Adding a new plugin

1. Create `<plugin-id>/` at repo root with `deeptrade_plugin.yaml`, `migrations/<version>_<name>.sql`, an inner Python package of the same name, and (recommended) `tests/` + `pytest.ini`.
2. Add an entry to `registry/index.json` (`schema_version: 1`, `type` ∈ {`strategy`, `channel`}, `tag_prefix` must end with `/`, `repo` in `owner/repo` form).
3. Run `python tools/check_registry.py` locally before PR.

## Naming and conventions worth knowing

- **Plugin id** uses kebab-case (`limit-up-board`); **inner Python package** uses snake_case (`limit_up_board`). Both must match the values in `index.json` and `deeptrade_plugin.yaml` — `check_registry.py` enforces it.
- **Migration filename**: `<YYYYMMDD>_<NNN>_<name>.sql`, with the version string in the yaml as `<YYYYMMDD>_<NNN>` (no `.sql`). The yaml's `file:` field is the path relative to the plugin subdir.
- **Table naming**: every table is prefixed by a short plugin tag (`lub_`, `apw_`, `checkmate_`). When adding a table, both create it in the migration AND list it under `tables:` in the yaml — the framework relies on the yaml list for `purge_on_uninstall`.
