"""
facesorter/search/llm/mlx_backend.py
Local chat model on Apple silicon through ``mlx_lm``.

Measured with the default model on an M-series Mac: ~1.3 s per query-parsing call, ~1.4 GB
resident. ``mlx_lm`` is imported only when the first ``chat()`` loads the weights.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from ..errors import ModelUnavailableError
from ._base import LazyChatBackend
from ._hub import announce_download

logger = logging.getLogger(__name__)

DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
INSTALL_HINT = (
    'pip install "facesorter[search-mlx]" '
    "(the search feature needs Python >= 3.10; the MLX extra needs >= 3.11)"
)

# Approximate download sizes for the notice (the default model is 0.88 GB of safetensors).
_APPROX_SIZE = {DEFAULT_MLX_MODEL: "0.9 GB"}


class MLXBackend(LazyChatBackend):
    """``LLMBackend`` backed by ``mlx_lm`` (macOS, Apple silicon).

    ``model`` is a Hugging Face repo id of an MLX-converted model or a local directory.
    Weights load on the first ``chat()``; ``close()`` releases them (a later ``chat()``
    reloads).
    """

    def __init__(self, model: str = DEFAULT_MLX_MODEL) -> None:
        super().__init__()
        self.model = model
        self.name = f"mlx:{model}"

    def _load(self) -> Any:
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ModelUnavailableError(
                f"mlx_lm is not installed, so the MLX LLM backend cannot run. "
                f"Install it with: {INSTALL_HINT}"
            ) from exc
        announce_download(self.model, _APPROX_SIZE.get(self.model, "size unknown"))
        logger.debug("Calling mlx_lm.load(%r)", self.model)
        try:
            loaded = load(self.model)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load the MLX model {self.model!r}: {exc}. Check the model name "
                f"and your network connection (the first run downloads the weights)."
            ) from exc
        model, tokenizer = loaded[0], loaded[1]
        logger.debug("MLX model %s ready", self.model)
        return model, tokenizer

    def _generate(self, handle: Any, messages: list[dict[str, str]], max_tokens: int) -> str:
        from mlx_lm import generate

        model, tokenizer = handle
        kwargs: dict[str, Any] = {}
        if _accepts_enable_thinking(tokenizer):
            kwargs["enable_thinking"] = False  # Qwen3: skip the reasoning preamble
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
        # mlx_lm.generate decodes greedily (argmax) unless a sampler is passed.
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    def _release_memory(self) -> None:
        gc.collect()
        import mlx.core as mx

        clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
        if clear is not None:
            clear()


def _accepts_enable_thinking(tokenizer: Any) -> bool:
    """True if the tokenizer's chat template mentions ``enable_thinking`` (Qwen3 family)."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):  # several named templates
        template = " ".join(str(v) for v in template.values())
    return isinstance(template, str) and "enable_thinking" in template
