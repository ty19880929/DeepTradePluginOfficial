"""Process-wide SIGINT marker so runner can normalise cancellation-derived
exceptions (DuckDB InterruptException, requests ConnectionError after SSL
interrupt, concurrent.futures.CancelledError, etc.) into ``RunStatus.CANCELLED``
without resorting to fragile substring matching on exception messages.

Contract
--------
* :func:`install_sigint_marker` — call once at ``cli.main`` entry; idempotent.
  Installs a SIGINT handler that sets a process-wide flag *and* then defers to
  :func:`signal.default_int_handler` so the main thread still raises
  :class:`KeyboardInterrupt` exactly as before.
* :func:`cancel_requested` — true iff a SIGINT was delivered to the process
  since the marker was installed.
* :func:`reset_marker` — for tests / future embedded callers.

Per-plugin copy of the same module shipped in ``limit_up_board`` and
``checkmate`` (CLAUDE.md "Two layers per plugin" — official plugins
intentionally don't share code).

Safe-degrade: when ``signal.signal`` fails (off main thread, pytest capture,
embedded interpreter), the marker stays false and the runner falls back to
the ``KeyboardInterrupt``-only path, i.e. the pre-change behaviour.
"""

from __future__ import annotations

import signal
import threading

_marker = threading.Event()
_installed = False
_prev_handler: object | None = None


def install_sigint_marker() -> None:
    """Install the SIGINT handler once. Subsequent calls are no-ops."""
    global _installed, _prev_handler
    if _installed:
        return
    try:
        _prev_handler = signal.getsignal(signal.SIGINT)
    except (ValueError, OSError):
        _prev_handler = None

    def _handler(signum: int, frame: object) -> None:
        _marker.set()
        signal.default_int_handler(signum, frame)  # type: ignore[arg-type]

    try:
        signal.signal(signal.SIGINT, _handler)
        _installed = True
    except (ValueError, OSError):
        _installed = False


def cancel_requested() -> bool:
    return _marker.is_set()


def reset_marker() -> None:
    _marker.clear()


__all__ = ["install_sigint_marker", "cancel_requested", "reset_marker"]
