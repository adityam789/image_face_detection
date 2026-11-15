"""
facesorter/search/encoders
Embedding models and fast image decoding for photo search.

* ``SigLIPImageEncoder``: SigLIP 2 B/16 image tower (PyTorch / open_clip), for indexing.
* ``OnnxTextEncoder``: the matching text tower on onnxruntime, no PyTorch, for queries.
* ``load_for_model`` / ``iter_decoded``: cheap, parallel decoding of large photos.

Heavy dependencies are imported lazily, so importing this package is cheap. Use
``load_image_encoder`` / ``load_text_encoder`` to get a ready encoder by model name (the
name recorded in the index, e.g. ``store.model_name``), and ``model_dimension(name)`` to
learn the embedding size without loading weights.

Other models, including the test fakes, are plugged in with ``register_model``::

    from facesorter.search import testing
    from facesorter.search.encoders import register_model, unregister_model

    register_model(
        testing.FAKE_MODEL_NAME,                       # "fake-color-16"
        lambda device: testing.FakeImageEncoder(),
        lambda quantization: testing.FakeTextEncoder(),
        dim=testing.FAKE_DIM,
    )
    # ... load_image_encoder("fake-color-16") now works ...
    unregister_model(testing.FAKE_MODEL_NAME)
"""

from __future__ import annotations

from typing import Any

from ..types import ImageEncoder, TextEncoder
from .decode import DEFAULT_TARGET, iter_decoded, load_for_model
from .onnx_text import QUANTIZATIONS, OnnxTextEncoder
from .registry import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    ModelSpec,
    available_models,
    model_dimension,
    register_model,
    spec_for,
    unregister_model,
)
from .siglip_image import SigLIPImageEncoder

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TARGET",
    "EMBEDDING_DIM",
    "QUANTIZATIONS",
    "ModelSpec",
    "OnnxTextEncoder",
    "SigLIPImageEncoder",
    "available_models",
    "iter_decoded",
    "load_for_model",
    "load_image_encoder",
    "load_text_encoder",
    "model_dimension",
    "register_model",
    "unregister_model",
]

register_model(
    DEFAULT_MODEL,
    lambda device: SigLIPImageEncoder(device=device),
    lambda quantization: OnnxTextEncoder(quantization=quantization),
    dim=EMBEDDING_DIM,
)


def _loaded(encoder: Any) -> Any:
    load = getattr(encoder, "load", None)
    if callable(load):
        load()
    return encoder


def load_image_encoder(name: str = DEFAULT_MODEL, device: str = "auto") -> ImageEncoder:
    """Return an image encoder for model ``name`` with its weights loaded (if it has a
    ``load()``), or raise ``ModelUnavailableError`` (unknown name, missing runtime, ...)."""
    return _loaded(spec_for(name).image_factory(device))


def load_text_encoder(name: str = DEFAULT_MODEL, quantization: str = "fp16") -> TextEncoder:
    """Return a loaded text encoder for model ``name``, or raise ``ModelUnavailableError``.

    ``quantization`` is ``"fp16"`` (default, matches the PyTorch reference to cosine
    ~1.0) or ``"int8"`` (smaller and faster, slightly less exact); alternative models may
    ignore it.
    """
    return _loaded(spec_for(name).text_factory(quantization))
