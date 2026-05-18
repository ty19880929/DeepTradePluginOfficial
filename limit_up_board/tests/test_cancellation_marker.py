"""Unit tests for limit_up_board.cancellation.

Verifies the SIGINT marker contract:
* install is idempotent
* simulating the handler sets the marker (without actually killing pytest)
* reset clears it
* install on a non-main thread / missing signal support degrades silently
"""

from __future__ import annotations

import signal
import threading

import pytest

from limit_up_board import cancellation


@pytest.fixture(autouse=True)
def _reset() -> None:
    cancellation.reset_marker()
    # Reset the installed-once flag so each test sees a fresh install path.
    cancellation._installed = False  # type: ignore[attr-defined]
    cancellation._prev_handler = None  # type: ignore[attr-defined]
    yield
    cancellation.reset_marker()


def test_install_is_idempotent_and_marker_starts_false() -> None:
    cancellation.install_sigint_marker()
    cancellation.install_sigint_marker()  # second call is no-op
    assert cancellation.cancel_requested() is False


def test_handler_sets_marker_and_preserves_keyboardinterrupt() -> None:
    cancellation.install_sigint_marker()
    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler)
    # Simulate signal delivery in-process — invoking the handler should set
    # the marker and then defer to default_int_handler which raises KI.
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)  # type: ignore[misc]
    assert cancellation.cancel_requested() is True


def test_reset_marker_clears_state() -> None:
    cancellation.install_sigint_marker()
    handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)  # type: ignore[misc]
    assert cancellation.cancel_requested() is True
    cancellation.reset_marker()
    assert cancellation.cancel_requested() is False


def test_install_off_main_thread_degrades_silently() -> None:
    # signal.signal raises ValueError off the main thread; install_sigint_marker
    # must swallow it and leave the marker unset. Runner falls back to the
    # KeyboardInterrupt-only path, i.e. pre-change behaviour.
    errors: list[BaseException] = []

    def _target() -> None:
        try:
            cancellation.install_sigint_marker()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    assert errors == []
    assert cancellation.cancel_requested() is False
