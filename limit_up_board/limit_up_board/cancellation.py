"""Process-wide SIGINT marker so runner can normalise cancellation-derived
exceptions (DuckDB InterruptException, requests ConnectionError after SSL
interrupt, concurrent.futures.CancelledError, etc.) into ``RunStatus.CANCELLED``
without resorting to fragile substring matching on exception messages.

Contract
--------
* :func:`install_sigint_marker` — call once at ``cli.main`` entry; idempotent.
  Installs a SIGINT handler that sets a process-wide flag *and* then defers to
  :func:`signal.default_int_handler` so the main thread still raises
  :class:`KeyboardInterrupt` exactly as before. Worker threads in C-level
  blocking calls that translate the interrupt into a derived exception
  (DuckDB ``InterruptException``, requests SSL break, …) are then detectable
  by reading the flag.
* :func:`cancel_requested` — true iff a SIGINT was delivered to the process
  since the marker was installed. The current CLI is single-run-per-process,
  so we never reset; if a future caller embeds the runner in a long-lived
  process it must call :func:`reset_marker` between runs.
* :func:`reset_marker` — clears the flag. Exposed for tests and future
  embedded callers; not used by the CLI.

Why this exists
---------------
Ctrl+C reaches the runner in three flavours:

1. ``KeyboardInterrupt`` raised at a Python frame in the main thread — the
   existing ``except KeyboardInterrupt`` already handles this.
2. A C-level blocking call (DuckDB query, tushare HTTPS read, …) that
   surfaces the interrupt as a derived exception type — currently misclassified
   as a real failure because the type is ``RuntimeError`` /
   ``InterruptException`` / ``ConnectionError``, not ``KeyboardInterrupt``.
3. A worker thread that never sees ``KeyboardInterrupt`` at all (only the
   main thread does) but raises a derived exception when its blocking call
   aborts.

The marker lets the runner's bottom-of-pipeline ``except Exception`` branch
ask "did the user already press Ctrl+C?" and short-circuit to the cancel
path when the answer is yes. Exception type / message text are never
inspected — those are unstable across versions and DB backends.

Windows note
------------
SIGINT is delivered by the CPython runtime for ``CTRL_C_EVENT`` on
Windows, so ``signal.signal(SIGINT, ...)`` works in PowerShell / cmd.
``CTRL_BREAK_EVENT`` is intentionally not covered; users overwhelmingly hit
Ctrl+C, not Ctrl+Break.

Safe-degrade
------------
``signal.signal`` can fail with ``ValueError`` (called off main thread) or
``OSError`` (pytest capture, embedded interpreters). When it does we leave
the marker permanently false and the runner falls back to recognising only
true ``KeyboardInterrupt``, i.e. the pre-change behaviour. The new code
path never makes things worse than before.
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
        # Preserve KeyboardInterrupt semantics — the default handler raises
        # KeyboardInterrupt at the next Python bytecode boundary, exactly
        # like the pre-install behaviour.
        signal.default_int_handler(signum, frame)  # type: ignore[arg-type]

    try:
        signal.signal(signal.SIGINT, _handler)
        _installed = True
    except (ValueError, OSError):
        # Off-main-thread, pytest capture, embedded interpreter — leave the
        # marker permanently unset. Runner degrades to "KeyboardInterrupt
        # only" detection, identical to the pre-change behaviour.
        _installed = False


def cancel_requested() -> bool:
    """True iff SIGINT has been received since :func:`install_sigint_marker`."""
    return _marker.is_set()


def reset_marker() -> None:
    """Clear the marker. For tests and future embedded callers only."""
    _marker.clear()


__all__ = ["install_sigint_marker", "cancel_requested", "reset_marker"]
