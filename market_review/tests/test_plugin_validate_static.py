"""``MarketReviewPlugin.validate_static`` must stay light-weight.

Mirrors ``limit_up_board.tests.test_plugin_validate_static``: after import +
``validate_static`` runs, ``sys.modules`` MUST NOT contain heavy third-party
deps (typer / rich / pandas / tushare / lightgbm / sklearn) nor any of this
plugin's own runtime-only submodules (cli / runner / runtime / pipeline /
render / data). Those belong inside :meth:`dispatch`, loaded lazily on
``deeptrade market-review <subcommand>``.

We use a subprocess so the test does not contaminate its own ``sys.modules``
with anything previously imported by the pytest session itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

# Names that MUST NOT appear in ``sys.modules`` after a clean
# ``validate_static`` call. The lub list is the canonical reference; we
# extend with anything market-review will likely lazy-import in its own
# cli.py / runner.py once they exist (currently cli.py imports typer / rich
# unconditionally — that's fine, cli.py itself is what must stay out).
FORBIDDEN_TOP_LEVEL = {
    # Heavy third-party deps
    "typer",
    "rich",
    "questionary",
    "lightgbm",
    "sklearn",
    "pandas",
    "tushare",
    # Plugin's own runtime-only modules (PR-2..6 will populate these).
    "market_review.cli",
    "market_review.runner",
    "market_review.runtime",
    "market_review.pipeline",
    "market_review.render",
    "market_review.data",
}


_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    from market_review import plugin

    plugin.MarketReviewPlugin().validate_static(None)
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


def test_validate_static_succeeds_with_none_context() -> None:
    """Framework passes a ``PluginContext`` at install time, but PR-1 stub
    keeps the parameter unused. Make sure ``None`` doesn't crash anything."""

    from market_review.plugin import MarketReviewPlugin  # noqa: PLC0415

    MarketReviewPlugin().validate_static(None)  # type: ignore[arg-type]
