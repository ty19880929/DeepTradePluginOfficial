"""P1-H: ``_run_providers_ordered`` canonical-order guarantee."""

from __future__ import annotations

import threading
import time

from limit_up_board.runner import _run_providers_ordered


def test_run_providers_ordered_completion_order_irrelevant() -> None:
    """Worker that returns its provider name in completion order ≠ input order;
    the returned list must still match ``providers`` input order."""
    providers = ["alpha", "beta", "gamma"]
    sleep_map = {"alpha": 0.05, "beta": 0.001, "gamma": 0.025}
    completion: list[str] = []

    def worker(p: str) -> str:
        time.sleep(sleep_map[p])
        return f"R-{p}"

    def on_complete(p: str, r: str) -> None:
        completion.append(p)

    out = _run_providers_ordered(providers, worker, on_complete=on_complete)

    # Beta finishes first (fastest sleep), alpha last
    assert completion[0] == "beta"
    assert completion[-1] == "alpha"
    # Canonical order is providers input order, NOT completion order
    assert out == ["R-alpha", "R-beta", "R-gamma"]


def test_run_providers_ordered_real_time_events_in_completion_order() -> None:
    """``on_complete`` callback fires in completion order so dashboards stay
    responsive while persistence consumes the returned canonical list."""
    providers = ["p1", "p2", "p3"]
    sleep_map = {"p1": 0.05, "p2": 0.001, "p3": 0.025}
    completion: list[str] = []

    def worker(p: str) -> str:
        time.sleep(sleep_map[p])
        return p

    def on_complete(p: str, r: str) -> None:
        completion.append(p)

    out = _run_providers_ordered(providers, worker, on_complete=on_complete)

    # Sorted by sleep duration → p2, p3, p1
    assert completion == ["p2", "p3", "p1"]
    # But results are canonical
    assert out == ["p1", "p2", "p3"]


def test_run_providers_ordered_empty_input() -> None:
    """Empty providers list returns an empty list without spinning the pool."""
    called = threading.Event()

    def worker(_: str) -> str:
        called.set()
        return "x"

    out = _run_providers_ordered([], worker)
    assert out == []
    assert not called.is_set()


def test_run_providers_ordered_runs_in_parallel() -> None:
    """All workers should run concurrently — total time ≈ longest sleep,
    not the sum (regression guard for accidental serialisation)."""
    providers = ["a", "b", "c"]
    barrier = threading.Barrier(len(providers))

    def worker(p: str) -> str:
        # If workers run serially, this barrier deadlocks (timeout below trips).
        barrier.wait(timeout=2.0)
        return p

    out = _run_providers_ordered(providers, worker)
    assert out == ["a", "b", "c"]
