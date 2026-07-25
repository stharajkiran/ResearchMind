"""Tracing abstraction for the agent layer.

The agent nodes decorate themselves with ``@trace`` (see ``tools.py`` /
``router.py``). That decorator is sourced from a :class:`Tracer` implementation
selected at import time by :func:`get_tracer`:

- :class:`LangSmithTracer` — wraps ``langsmith.traceable`` and configures the
  LangChain/LangSmith env vars. Used when tracing is enabled *and* langsmith is
  importable.
- :class:`NoOpTracer` — an identity decorator. Used when tracing is disabled or
  langsmith is not installed, so the agent runs without any LangSmith
  credentials (matters for the demo and CI).

This mirrors the interface/implementation split used by the retrieval, session,
feedback and ingestion layers: swapping the backend is a one-line change here.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

logger = logging.getLogger("agent_tracing")

F = TypeVar("F", bound=Callable)


def _tracing_enabled() -> bool:
    """Tracing is on when explicitly requested via env, or an API key is present."""
    flag = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
    if flag in {"true", "1", "yes"}:
        return True
    if flag in {"false", "0", "no"}:
        return False
    # No explicit flag — enable only if a key is available.
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


class Tracer(ABC):
    """Interface for run tracing. Implementations provide a decorator + setup."""

    @abstractmethod
    def configure(self) -> None:
        """Set up any global state (env vars, clients) the tracer needs."""

    @abstractmethod
    def trace(self, fn: F) -> F:
        """Decorate ``fn`` so its execution is traced. May be a no-op."""


class LangSmithTracer(Tracer):
    """Traces agent nodes to LangSmith via ``langsmith.traceable``."""

    def __init__(self) -> None:
        # Imported lazily so the module never hard-depends on langsmith.
        from langsmith import traceable

        self._traceable = traceable

    def configure(self) -> None:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault(
            "LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "researchmind")
        )

    def trace(self, fn: F) -> F:
        return self._traceable(fn)


class NoOpTracer(Tracer):
    """No tracing — ``trace`` returns the function unchanged."""

    def configure(self) -> None:  # nothing to set up
        pass

    def trace(self, fn: F) -> F:
        return fn


def get_tracer() -> Tracer:
    """Pick a tracer based on env config and langsmith availability."""
    if not _tracing_enabled():
        return NoOpTracer()
    try:
        return LangSmithTracer()
    except ImportError:
        logger.warning(
            "LANGCHAIN_TRACING_V2 is enabled but langsmith is not installed; "
            "falling back to NoOpTracer."
        )
        return NoOpTracer()


# Module-level singleton selected at import time. Agent nodes decorate with
# ``@trace``; scripts call ``configure_tracing()`` once at startup.
_tracer: Tracer = get_tracer()

#: Decorator applied to agent nodes (``@trace``).
trace = _tracer.trace


def configure_tracing() -> None:
    """Backward-compatible entry point — configures the active tracer."""
    _tracer.configure()
