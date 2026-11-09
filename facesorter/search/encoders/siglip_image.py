"""
facesorter/search/encoders/siglip_image.py
SigLIP 2 image encoder (open_clip + PyTorch) used at indexing time.

Model: open_clip ``ViT-B-16-SigLIP2`` with the ``webli`` weights (768-d embeddings). Device
``"auto"`` picks Apple GPU (mps), then CUDA, then CPU. Half precision is used on mps/cuda
(measured: identical COCO retrieval quality to fp32, and faster); the CPU uses fp32.

PyTorch and open_clip are imported lazily, only when the model is first needed.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ModelUnavailableError
from .registry import EMBEDDING_DIM, INSTALL_HINT, MODEL_NAME

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

logger = logging.getLogger(__name__)

ARCHITECTURE = "ViT-B-16-SigLIP2"
PRETRAINED = "webli"


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``"mps"``, ``"cuda"`` or ``"cpu"``; pass explicit names through."""
    if device != "auto":
        return device
    try:
        import torch
    except Exception as exc:  # ImportError, or OSError/RuntimeError from a broken torch
        raise ModelUnavailableError(f"PyTorch is unavailable ({exc}). {INSTALL_HINT}") from exc
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SigLIPImageEncoder:
    """Image half of SigLIP 2 B/16 (webli). Thread-safe; the model loads on first use.

    ``encode_images`` accepts PIL images of any size or mode (the open_clip transform
    resizes them) and processes them in batches of ``batch_size``.
    """

    name = MODEL_NAME
    dim = EMBEDDING_DIM

    def __init__(self, device: str = "auto", batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.requested_device = device
        self.batch_size = batch_size
        self.device: str | None = None
        self._lock = threading.Lock()
        self._model: Any = None
        self._preprocess: Any = None
        self._torch: Any = None
        self._dtype: Any = None

    def load(self) -> None:
        """Load the model now (idempotent). Downloads the weights on first ever use."""
        with self._lock:
            self._load_locked()

    def _build(self, device: str) -> tuple[Any, Any, Any]:
        """Create the model on ``device``; half precision on mps/cuda, fp32 on cpu."""
        torch = self._torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            ARCHITECTURE, pretrained=PRETRAINED
        )
        dtype = torch.float16 if device.split(":")[0] in ("mps", "cuda") else torch.float32
        return model.to(device=device, dtype=dtype).eval(), preprocess, dtype

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip  # noqa: F401
            import torch
        except Exception as exc:  # ImportError, or OSError/RuntimeError from a broken torch
            raise ModelUnavailableError(
                f"open_clip and PyTorch are required to index photos ({exc}). {INSTALL_HINT}"
            ) from exc
        self._torch = torch
        device = resolve_device(self.requested_device)
        try:
            try:
                model, preprocess, dtype = self._build(device)
            except Exception as exc:
                if device == "cpu":
                    raise
                logger.warning(
                    "Could not load %s on %s (%s); falling back to the CPU", MODEL_NAME, device, exc
                )
                device = "cpu"
                model, preprocess, dtype = self._build(device)
        except Exception as exc:
            self._torch = None
            raise ModelUnavailableError(
                f"Could not load {ARCHITECTURE} ({PRETRAINED}) on {device}: {exc}. "
                "The weights (~1.5 GB) download once from Hugging Face; check your "
                "connection, or choose another device."
            ) from exc
        self._model, self._preprocess = model, preprocess
        self._dtype, self.device = dtype, device
        logger.info("Loaded %s image encoder on %s (%s)", MODEL_NAME, device, dtype)

    def close(self) -> None:
        """Free the model and any GPU memory; it reloads on next use."""
        with self._lock:
            torch = self._torch
            self._model = self._preprocess = self._dtype = None
            self._torch = None
        gc.collect()
        if torch is not None:
            try:
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Could not release accelerator cache", exc_info=True)

    def _forward(self, tensors: list[Any]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack(tensors).to(device=self.device, dtype=self._dtype)
        with torch.inference_mode():
            feats = self._model.encode_image(batch, normalize=True)
        return feats.float().cpu().numpy()

    def _fallback_to_cpu(self, exc: Exception) -> None:
        logger.warning(
            "Image encoding failed on %s (%s); falling back to the CPU in fp32", self.device, exc
        )
        self._model, self._preprocess, self._dtype = self._build("cpu")
        self.device = "cpu"

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return ``float32`` ``(len(images), 768)`` unit-length embeddings.

        If the forward pass fails on mps/cuda, the model is rebuilt on the CPU (fp32), a
        warning is logged, and the batch is retried once.
        """
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            self._load_locked()
            chunks = []
            for start in range(0, len(images), self.batch_size):
                tensors = [
                    self._preprocess(im if im.mode == "RGB" else im.convert("RGB"))
                    for im in images[start : start + self.batch_size]
                ]
                try:
                    chunks.append(self._forward(tensors))
                except Exception as exc:
                    if self.device == "cpu":
                        raise
                    self._fallback_to_cpu(exc)
                    chunks.append(self._forward(tensors))
        emb = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)
