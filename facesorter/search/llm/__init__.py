"""
facesorter/search/llm
Local LLM backends for query parsing, behind the ``LLMBackend`` protocol.

* ``MLXBackend``: Apple silicon via ``mlx_lm`` (fastest on a Mac).
* ``LlamaCppBackend``: GGUF models via ``llama-cpp-python`` (any OS).
* ``load_llm``: pick one. ``"auto"`` prefers MLX on an Apple-silicon Mac, then llama.cpp, and
  returns ``None`` (rules-only mode, not an error) when neither runtime is installed.

Loading is lazy twice over: ``load_llm`` imports no runtime and loads no weights; the first
``chat()`` does. ``--help`` and rules-only code paths therefore never pay for a model.

Environment overrides (explicit arguments win over them):

* ``FACESORTER_LLM_BACKEND``: ``auto`` | ``mlx`` | ``llama-cpp`` | ``none``
* ``FACESORTER_LLM_MODEL``: MLX repo id or directory; for llama.cpp a ``.gguf`` path,
  ``repo_id:filename`` or ``repo_id``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys

from ..errors import ModelUnavailableError
from ..types import LLMBackend
from .llamacpp_backend import DEFAULT_LLAMA_MODEL, LlamaCppBackend
from .mlx_backend import DEFAULT_MLX_MODEL, MLXBackend

logger = logging.getLogger(__name__)

__all__ = [
    "BACKEND_ENV",
    "DEFAULT_MODELS",
    "MODEL_ENV",
    "LlamaCppBackend",
    "MLXBackend",
    "available_backends",
    "load_llm",
]

BACKEND_ENV = "FACESORTER_LLM_BACKEND"
MODEL_ENV = "FACESORTER_LLM_MODEL"

DEFAULT_MODELS: dict[str, str] = {
    "mlx": DEFAULT_MLX_MODEL,
    "llama-cpp": DEFAULT_LLAMA_MODEL,
}

_ALIASES = {
    "llama-cpp": "llama-cpp",
    "llama_cpp": "llama-cpp",
    "llamacpp": "llama-cpp",
    "llama.cpp": "llama-cpp",
    "gguf": "llama-cpp",
    "mlx": "mlx",
    "auto": "auto",
    "none": "none",
    "off": "none",
}
_CHOICES = "auto, mlx, llama-cpp, none"


def _importable(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""
    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def available_backends() -> dict[str, bool]:
    """Which backends can run here: ``{"mlx": ..., "llama-cpp": ...}``.

    MLX needs Apple silicon and ``mlx_lm``; llama.cpp needs ``llama_cpp``. Nothing is imported.
    """
    return {
        "mlx": _is_apple_silicon() and _importable("mlx_lm"),
        "llama-cpp": _importable("llama_cpp"),
    }


def _canonical(name: str, source: str) -> str:
    key = _ALIASES.get(name.strip().lower())
    if key is None:
        raise ModelUnavailableError(
            f"Unknown LLM backend {name!r} (from {source}); choose one of: {_CHOICES}."
        )
    return key


def load_llm(backend: str = "auto", model: str | None = None) -> LLMBackend | None:
    """Return a local LLM backend, or ``None`` for rules-only mode.

    ``backend``: ``"auto"`` (MLX on Apple silicon when ``mlx_lm`` is installed, else
    llama.cpp when ``llama_cpp`` is installed, else ``None``), ``"mlx"``, ``"llama-cpp"`` or
    ``"none"``. The environment variable ``FACESORTER_LLM_BACKEND`` replaces ``"auto"``;
    ``FACESORTER_LLM_MODEL`` supplies ``model`` when it is ``None``.

    The returned backend has loaded nothing yet; weights load on its first ``chat()``.

    Raises ``ModelUnavailableError`` for an unknown backend name, or when ``"mlx"`` or
    ``"llama-cpp"`` is requested explicitly but cannot run here (the message says what to
    install). ``"auto"`` never raises for a missing runtime.
    """
    choice = _canonical(backend, "the backend argument")
    if choice == "auto" and os.environ.get(BACKEND_ENV):
        choice = _canonical(os.environ[BACKEND_ENV], BACKEND_ENV)
    if model is None:
        model = os.environ.get(MODEL_ENV) or None

    if choice == "none":
        return None

    available = available_backends()
    if choice == "auto":
        choice = next((name for name in ("mlx", "llama-cpp") if available[name]), "none")
        if choice == "none":
            logger.info("No local LLM runtime found; using rules-only query parsing.")
            return None
    elif not available[choice]:
        raise ModelUnavailableError(_unavailable_message(choice))

    chosen = model or DEFAULT_MODELS[choice]
    logger.debug("Selected LLM backend %s with model %s", choice, chosen)
    if choice == "mlx":
        return MLXBackend(chosen)
    return LlamaCppBackend.from_spec(chosen)


def _unavailable_message(choice: str) -> str:
    if choice == "mlx":
        if not _is_apple_silicon():
            return (
                "The MLX backend only runs on macOS with Apple silicon. Use the llama.cpp "
                'backend instead: pip install "facesorter[search-llama]" '
                "(the search feature needs Python >= 3.10)"
            )
        return (
            "mlx_lm is not installed. Install it with: "
            'pip install "facesorter[search-mlx]" (the search feature needs Python >= 3.10; '
            "the MLX extra needs >= 3.11)"
        )
    return (
        "llama-cpp-python is not installed. Install it with: "
        'pip install "facesorter[search-llama]" (the search feature needs Python >= 3.10)'
    )
