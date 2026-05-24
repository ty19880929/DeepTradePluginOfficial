"""Plugin-local LLM replay policy adapter (P3).

Provides three things so the rest of the plugin code doesn't have to know
whether the framework has finished implementing the LLM replay cache:

1. ``LLMReplayPolicy`` dataclass — re-exported from the framework when
   available, otherwise a local stub that carries the same fields. The
   plugin never asserts the type comes from one specific module.
2. ``apply_replay_context(policy, *, stage_to_fingerprint)`` — context
   manager that stashes the active policy + per-stage fingerprints in
   ``ContextVar``s. Mirrors the pattern used by
   :func:`limit_up_board.schemas.apply_empty_array_policy` so worker
   threads inside debate mode inherit replay state without new function
   parameters across every layer.
3. ``build_replay_policy(...)`` — turns CLI flags (``--fresh-llm`` /
   ``--no-llm-replay`` / ``--replay-only``) plus :class:`LubConfig`
   defaults into a single policy object. CLI flags are mutually exclusive
   and pre-validated by typer.
4. ``complete_json_supports_replay()`` — runtime feature detection on
   ``LLMClient.complete_json``. ``False`` while the framework is on
   pre-Phase-2; pipeline silently skips replay knobs in that case.

The adapter is intentionally permissive: when the framework hasn't merged
replay support yet, passing ``--fresh-llm`` / ``--no-llm-replay`` is a
no-op (logged) and ``--replay-only`` is a hard error (user explicitly
asked for replay).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from inspect import signature
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. LLMReplayPolicy dataclass (framework-or-local)
# ---------------------------------------------------------------------------
try:  # pragma: no cover — branch resolved at import time
    from deeptrade.core.llm_client import LLMReplayPolicy  # type: ignore[import-not-found]
    _FRAMEWORK_REPLAY_TYPE = True
except ImportError:  # pragma: no cover branch — local stub

    @dataclass(frozen=True)
    class LLMReplayPolicy:  # type: ignore[no-redef]
        """Local stub matching the framework dataclass shape (design doc §5.1.3).

        ``read_enabled`` / ``write_enabled`` independently gate cache I/O.
        ``replay_only`` short-circuits a miss into ``PreconditionError``
        instead of falling back to a real LLM call.
        ``cache_namespace`` defaults to ``plugin_id`` at the framework layer;
        the plugin leaves it ``None`` and lets the framework fill it in.
        ``ttl_days`` ``None`` = no expiry (framework default).
        """

        read_enabled: bool = True
        write_enabled: bool = True
        replay_only: bool = False
        cache_namespace: str | None = None
        ttl_days: int | None = None

    _FRAMEWORK_REPLAY_TYPE = False


# ---------------------------------------------------------------------------
# 2. Active-policy ContextVar (mirrors apply_empty_array_policy)
# ---------------------------------------------------------------------------

_DEFAULT_POLICY = LLMReplayPolicy(read_enabled=False, write_enabled=False)
_replay_policy_ctx: ContextVar[LLMReplayPolicy] = ContextVar(
    "lub_llm_replay_policy", default=_DEFAULT_POLICY
)

# Per-stage input_fingerprint snapshot. The runner's input_fingerprint covers
# the *run* (computed once after Step 1); each stage call site can register
# stage-specific extra under its own key if it ever needs to. For now all
# four stages share the same run-level fingerprint, so this is a thin map
# keyed by stage name pointing to the same string.
_stage_fingerprint_ctx: ContextVar[dict[str, str | None]] = ContextVar(
    "lub_llm_stage_fingerprint", default={}
)


@contextmanager
def apply_replay_context(
    policy: LLMReplayPolicy,
    *,
    stage_to_fingerprint: dict[str, str | None] | None = None,
) -> Iterator[None]:
    """Activate ``policy`` (and per-stage fingerprints) for this with-block.

    Mirrors :func:`limit_up_board.schemas.apply_empty_array_policy`:
    ``_complete_with_set_check`` reads the ContextVars instead of taking new
    function parameters, so worker threads in debate mode inherit replay
    semantics automatically once the runner enters the context before
    fan-out.

    ContextVar values do NOT auto-propagate to ``ThreadPoolExecutor``
    workers — callers that fan out must enter this context inside each
    worker (see ``_worker_phase_a`` / ``_worker_phase_b``).
    """
    pol_token = _replay_policy_ctx.set(policy)
    fp_token = _stage_fingerprint_ctx.set(dict(stage_to_fingerprint or {}))
    try:
        yield
    finally:
        _stage_fingerprint_ctx.reset(fp_token)
        _replay_policy_ctx.reset(pol_token)


def get_active_policy() -> LLMReplayPolicy:
    """Return the active LLMReplayPolicy from the ContextVar."""
    return _replay_policy_ctx.get()


def get_stage_fingerprint(stage: str) -> str | None:
    """Return the fingerprint registered for ``stage`` (or None)."""
    return _stage_fingerprint_ctx.get().get(stage)


# ---------------------------------------------------------------------------
# 3. CLI / config → LLMReplayPolicy builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReplayCLIFlags:
    """CLI flag triple — exactly one (or none) may be True. Mutex enforced
    by the CLI layer before constructing this."""

    fresh_llm: bool = False
    no_llm_replay: bool = False
    replay_only: bool = False


def build_replay_policy(
    *,
    cli: _ReplayCLIFlags,
    cfg_enabled: bool,
    cfg_write: bool,
    cfg_ttl_days: int | None,
) -> LLMReplayPolicy:
    """Resolve final policy from CLI flags + LubConfig.

    Decision table (CLI wins over config; mutex enforced upstream):

      --replay-only        → read=T, write=F, replay_only=T   (offline replay)
      --no-llm-replay      → read=F, write=F                  (bypass cache)
      --fresh-llm          → read=F, write=cfg_write          (refresh + maybe cache)
      cfg_enabled=True     → read=T, write=cfg_write          (default-on grey state)
      otherwise            → read=F, write=F                  (Phase 1 fallback)

    Independent of whether the framework actually consumes the policy — the
    pipeline checks :func:`complete_json_supports_replay` separately.
    """
    if cli.replay_only:
        return LLMReplayPolicy(
            read_enabled=True,
            write_enabled=False,
            replay_only=True,
            ttl_days=cfg_ttl_days,
        )
    if cli.no_llm_replay:
        return LLMReplayPolicy(read_enabled=False, write_enabled=False)
    if cli.fresh_llm:
        return LLMReplayPolicy(
            read_enabled=False,
            write_enabled=cfg_write,
            ttl_days=cfg_ttl_days,
        )
    if cfg_enabled:
        return LLMReplayPolicy(
            read_enabled=True,
            write_enabled=cfg_write,
            ttl_days=cfg_ttl_days,
        )
    return LLMReplayPolicy(read_enabled=False, write_enabled=False)


# ---------------------------------------------------------------------------
# 4. Framework feature detection
# ---------------------------------------------------------------------------


def complete_json_supports_replay() -> bool:
    """Return True iff ``LLMClient.complete_json`` accepts ``replay=`` kwarg.

    Cached at module import time via :func:`functools.cache` would be
    elegant, but the plugin can run before the framework finishes loading
    in some test paths — keep this as a plain function (cheap) so tests
    can monkeypatch the LLMClient symbol freely.
    """
    try:
        from deeptrade.core.llm_client import LLMClient  # noqa: PLC0415
    except ImportError:
        return False
    try:
        params = signature(LLMClient.complete_json).parameters
    except (TypeError, ValueError):
        return False
    # Framework Phase 2 design: ``replay``, ``stage``, ``schema_version``,
    # ``input_fingerprint`` are added together. Treat ``replay`` as the
    # canary — if the framework lands the others without replay we have
    # bigger problems.
    return "replay" in params


__all__ = [
    "LLMReplayPolicy",
    "_ReplayCLIFlags",
    "apply_replay_context",
    "build_replay_policy",
    "complete_json_supports_replay",
    "get_active_policy",
    "get_stage_fingerprint",
]
