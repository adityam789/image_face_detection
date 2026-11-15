"""
facesorter/search/llm/llamacpp_backend.py
Local chat model through ``llama-cpp-python`` (GGUF files; Linux, Windows and macOS).

The model is either a local ``.gguf`` file or a Hugging Face repo plus a filename pattern that
``Llama.from_pretrained`` downloads and caches. ``llama_cpp`` is imported only when the first
``chat()`` loads the weights.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from ..errors import ModelUnavailableError
from ._base import LazyChatBackend
from ._hub import announce_download

logger = logging.getLogger(__name__)

# Official Qwen GGUF repo (public, Apache-2.0); Q4_K_M is 1.12 GB.
DEFAULT_LLAMA_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_LLAMA_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_LLAMA_MODEL = f"{DEFAULT_LLAMA_REPO}:{DEFAULT_LLAMA_FILENAME}"
# Used when a repo id is given without a filename; matches Q4_K_M in any letter case.
DEFAULT_QUANT_GLOB = "*[qQ]4_[kK]_[mM].gguf"
INSTALL_HINT = 'pip install "facesorter[search-llama]" (the search feature needs Python >= 3.10)'

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_APPROX_SIZE = {(DEFAULT_LLAMA_REPO, DEFAULT_LLAMA_FILENAME): "1.1 GB"}


class LlamaCppBackend(LazyChatBackend):
    """``LLMBackend`` backed by ``llama_cpp.Llama``.

    Give either ``model_path`` (a local GGUF file) or ``repo_id`` plus ``filename`` (an exact
    name or glob such as ``"*q4_k_m.gguf"``). With no arguments the default Qwen2.5-1.5B
    Q4_K_M model is downloaded on first use. Decoding is greedy (temperature 0).
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = 2048,
        n_gpu_layers: int | None = None,
    ) -> None:
        super().__init__()
        if model_path is not None and (repo_id is not None or filename is not None):
            raise ValueError("pass either model_path or repo_id/filename, not both")
        if model_path is None and repo_id is None:
            if filename is not None:
                raise ValueError("filename needs a repo_id")
            repo_id, filename = DEFAULT_LLAMA_REPO, DEFAULT_LLAMA_FILENAME
        if repo_id is not None and filename is None:
            filename = DEFAULT_QUANT_GLOB
        self.model_path = model_path
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers  # None = offload everything the build supports
        self.name = f"llama-cpp:{model_path}" if model_path else f"llama-cpp:{repo_id}/{filename}"

    @classmethod
    def from_spec(cls, spec: str) -> LlamaCppBackend:
        """Build from one string: a ``.gguf`` path, ``repo_id:filename`` or a bare ``repo_id``.

        This is the form used by ``FACESORTER_LLM_MODEL`` and ``load_llm(model=...)``.
        """
        if os.path.exists(spec) or _WINDOWS_DRIVE.match(spec):
            return cls(model_path=spec)
        repo_id, sep, filename = spec.partition(":")
        if sep:
            return cls(repo_id=repo_id, filename=filename or None)
        if spec.lower().endswith(".gguf"):  # a path that does not exist (yet): fail at load
            return cls(model_path=spec)
        return cls(repo_id=spec)

    def _load(self) -> Any:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelUnavailableError(
                f"llama-cpp-python is not installed, so the llama.cpp LLM backend cannot run. "
                f"Install it with: {INSTALL_HINT}"
            ) from exc
        kwargs: dict[str, Any] = {
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self._gpu_layers(),
            "verbose": False,
        }
        logger.debug("Loading GGUF model %s with %s", self.name, kwargs)
        try:
            if self.model_path is not None:
                if not os.path.isfile(self.model_path):
                    raise FileNotFoundError(f"no such GGUF file: {self.model_path}")
                return Llama(model_path=self.model_path, **kwargs)
            assert self.repo_id is not None and self.filename is not None
            size = _APPROX_SIZE.get((self.repo_id, self.filename), "size unknown")
            announce_download(self.repo_id, size, self.filename)
            return Llama.from_pretrained(repo_id=self.repo_id, filename=self.filename, **kwargs)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load the GGUF model {self.name!r}: {exc}. Check the path or "
                f"repository name and your network connection (the first run downloads it)."
            ) from exc

    def _gpu_layers(self) -> int:
        if self.n_gpu_layers is not None:
            return self.n_gpu_layers
        try:
            import llama_cpp

            supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
            return -1 if supports is not None and supports() else 0
        except Exception:
            return 0

    def _generate(self, handle: Any, messages: list[dict[str, str]], max_tokens: int) -> str:
        reply = handle.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.0
        )
        return reply["choices"][0]["message"]["content"] or ""

    def _unload(self, handle: Any) -> None:
        handle.close()
