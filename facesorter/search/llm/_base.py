"""
facesorter/search/llm/_base.py
Shared lifecycle for the lazily loaded local chat backends.

``LazyChatBackend`` owns the parts that are identical for every runtime: load the weights on
the first ``chat()`` (once, even under concurrent callers), serialise generation, release the
weights on ``close()``, and map every runtime failure to ``ModelUnavailableError``.

Lifecycle decision: ``close()`` frees the weights but does not poison the object. A later
``chat()`` transparently reloads the model. This keeps ``with`` blocks and long-lived UI
processes simple (an idle app can close the model to give the memory back, then keep using the
same object).
"""

from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

from ..errors import ModelUnavailableError

logger = logging.getLogger(__name__)

# Reasoning models (Qwen3 and friends) may emit an empty or filled <think> block even when
# asked not to; the caller only wants the answer.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a leading ``<think>...</think>`` block and surrounding whitespace."""
    return _THINK_BLOCK.sub("", text, count=1).strip()


def normalise_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Copy ``messages`` into plain ``{"role", "content"}`` dicts, validating the shape."""
    if not messages:
        raise ValueError("chat() needs at least one message")
    out: list[dict[str, str]] = []
    for message in messages:
        try:
            out.append({"role": str(message["role"]), "content": str(message["content"])})
        except KeyError as exc:
            raise ValueError(f"chat message is missing the {exc} key: {message!r}") from exc
    return out


class LazyChatBackend(ABC):
    """Base class: lazy, thread-safe loading plus serialised generation.

    Subclasses implement ``_load`` (return an opaque handle) and ``_generate``; ``_unload`` and
    ``_release_memory`` are optional hooks.
    """

    name: str

    def __init__(self) -> None:
        # One re-entrant lock guards load, generate and unload: these runtimes are not safe
        # for concurrent generation, and a chat must not race a close().
        self._lock = threading.RLock()
        self._handle: Any = None

    # -- subclass hooks ---------------------------------------------------------------

    @abstractmethod
    def _load(self) -> Any:
        """Load the weights and return a handle. Raise ``ModelUnavailableError`` on failure."""

    @abstractmethod
    def _generate(self, handle: Any, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Return the raw reply text."""

    def _unload(self, handle: Any) -> None:  # noqa: B027 - optional hook
        """Close ``handle`` explicitly, if the runtime needs it. Default: nothing."""

    def _release_memory(self) -> None:  # noqa: B027 - optional hook
        """Called after the handle is dropped; return cached memory to the OS."""

    # -- public interface -------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        """True while the weights are in memory."""
        return self._handle is not None

    def chat(self, messages: Sequence[Mapping[str, str]], max_tokens: int = 160) -> str:
        """Greedy-decode a reply for OpenAI-style ``{"role", "content"}`` messages.

        Loads the model on first use (and again after ``close()``). Raises
        ``ModelUnavailableError`` if the runtime or weights cannot be used, ``ValueError``
        for malformed messages or ``max_tokens < 1``.
        """
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        prepared = normalise_messages(messages)
        with self._lock:
            if self._handle is None:
                logger.info("Loading local LLM %s", self.name)
                self._handle = self._load()
            try:
                return strip_thinking(self._generate(self._handle, prepared, max_tokens))
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(
                    f"The local LLM {self.name} failed while generating: {exc}"
                ) from exc

    def close(self) -> None:
        """Release the weights. Safe to call repeatedly; a later ``chat()`` reloads."""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle is not None:
                logger.info("Unloading local LLM %s", self.name)
                try:
                    self._unload(handle)
                except Exception:  # best effort: closing must never raise
                    logger.debug("Error while unloading %s", self.name, exc_info=True)
                del handle  # drop the last reference before collecting
                try:
                    self._release_memory()
                except Exception:
                    logger.debug("Error while releasing memory", exc_info=True)

    def __enter__(self) -> LazyChatBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "loaded" if self.loaded else "not loaded"
        return f"{type(self).__name__}({self.name!r}, {state})"
