"""
facesorter/search/encoders/onnx_text.py
SigLIP 2 text encoder that runs without PyTorch (onnxruntime + ``tokenizers``).

Used at query time, where loading PyTorch would cost seconds and gigabytes. The ONNX export
is ``onnx-community/siglip2-base-patch16-224-ONNX``. Three details are essential to match the
PyTorch/open_clip embeddings the image index was built with:

* Text is canonicalised exactly like open_clip does before tokenising (underscores to
  spaces, all ASCII punctuation removed, lowercase, whitespace collapsed). Without it only
  ~22% of captions tokenise identically.
* Token ids are padded with 0 (or truncated) to exactly 64 positions.
* The embedding is the model's ``pooler_output`` (second output), L2-normalised.

onnxruntime runs on ``CPUExecutionProvider`` only: the text tower is small (~60-110 ms per
query on CPU), while GPU/CoreML providers add seconds of graph compilation at start-up and
do not support every fp16 op, so they would make a one-shot CLI query slower, not faster.
"""

from __future__ import annotations

import logging
import string
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ModelUnavailableError
from .registry import EMBEDDING_DIM, INSTALL_HINT, MODEL_NAME

logger = logging.getLogger(__name__)

REPO_ID = "onnx-community/siglip2-base-patch16-224-ONNX"
CONTEXT_LENGTH = 64

_FILES = {
    "fp16": "onnx/text_model_fp16.onnx",
    "int8": "onnx/text_model_int8.onnx",
}
QUANTIZATIONS = tuple(_FILES)

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def canonicalize(text: str) -> str:
    """Normalise ``text`` the way open_clip's SigLIP tokenizer does before tokenising."""
    return " ".join(text.replace("_", " ").translate(_PUNCTUATION).lower().split())


class OnnxTextEncoder:
    """Text half of the SigLIP 2 B/16 (webli) model, via onnxruntime. Thread-safe.

    The session is created lazily on first use (or by :meth:`load`). ``encode_text`` is
    safe to call from several threads. ``cache_dir`` overrides the Hugging Face cache.
    """

    name = MODEL_NAME
    dim = EMBEDDING_DIM

    def __init__(
        self,
        quantization: str = "fp16",
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        if quantization not in _FILES:
            raise ValueError(
                f"Unknown quantization {quantization!r}; choose one of {', '.join(QUANTIZATIONS)}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.quantization = quantization
        self.batch_size = batch_size
        self._cache_dir = None if cache_dir is None else str(cache_dir)
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None
        self._output_index = 1
        self._eos_id = 1

    # ------------------------------------------------------------------ loading

    def _fetch(self, filename: str) -> str:
        """Return a local path for a repo file, preferring the cache over the network."""
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # ImportError, or a broken install failing at import time
            raise ModelUnavailableError(
                f"huggingface_hub is unavailable ({exc}). {INSTALL_HINT}"
            ) from exc
        try:
            # Cache first: no network round trip once the model has been downloaded.
            return str(
                hf_hub_download(REPO_ID, filename, cache_dir=self._cache_dir, local_files_only=True)
            )
        except Exception:
            logger.info("%s not in the local cache; downloading from %s", filename, REPO_ID)
        try:
            return str(hf_hub_download(REPO_ID, filename, cache_dir=self._cache_dir))
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not fetch {filename} from {REPO_ID} ({exc}). "
                "Connect to the internet once to download the text model, or pre-populate "
                "the Hugging Face cache."
            ) from exc

    def load(self) -> None:
        """Create the tokenizer and ONNX session now (idempotent)."""
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except Exception as exc:  # ImportError, or OSError/RuntimeError from a broken binary
            raise ModelUnavailableError(
                f"onnxruntime and tokenizers are required for text search ({exc}). {INSTALL_HINT}"
            ) from exc
        model_path = self._fetch(_FILES[self.quantization])
        tokenizer_path = self._fetch("tokenizer.json")
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            options = ort.SessionOptions()
            options.log_severity_level = 3  # errors only
            session = ort.InferenceSession(
                model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load the text model from {model_path}: {exc}"
            ) from exc
        names = [o.name for o in session.get_outputs()]
        self._output_index = names.index("pooler_output") if "pooler_output" in names else 1
        # open_clip keeps the end-of-sequence token when it truncates; 1 is SigLIP's <eos>.
        eos = tokenizer.token_to_id("<eos>")
        self._eos_id = 1 if eos is None else int(eos)
        self._tokenizer = tokenizer
        self._session = session
        logger.info("Loaded %s text encoder (%s)", MODEL_NAME, self.quantization)

    def close(self) -> None:
        """Drop the session and tokenizer (they are recreated on next use)."""
        with self._lock:
            self._session = None
            self._tokenizer = None

    # ------------------------------------------------------------------ encoding

    def _token_ids(self, tokenizer: Any, texts: Sequence[str]) -> np.ndarray:
        ids = np.zeros((len(texts), CONTEXT_LENGTH), dtype=np.int64)
        encodings = tokenizer.encode_batch([canonicalize(t) for t in texts])
        for row, enc in enumerate(encodings):
            tokens = enc.ids
            if len(tokens) > CONTEXT_LENGTH:
                # Truncate like open_clip: cut the text but keep <eos> in the last slot.
                tokens = [*tokens[: CONTEXT_LENGTH - 1], self._eos_id]
            ids[row, : len(tokens)] = tokens
        return ids

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``float32`` ``(len(texts), 768)`` unit-length embeddings."""
        if isinstance(texts, str):
            raise TypeError("encode_text expects a sequence of strings, not a single string")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            self._load_locked()
            session, tokenizer, index = self._session, self._tokenizer, self._output_index
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            ids = self._token_ids(tokenizer, texts[start : start + self.batch_size])
            out = session.run(None, {"input_ids": ids})[index]
            chunks.append(np.asarray(out, dtype=np.float32))
        emb = np.concatenate(chunks, axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)
