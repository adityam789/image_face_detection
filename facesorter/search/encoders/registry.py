"""
facesorter/search/encoders/registry.py
Model names, dimensions and the public model registry.

``DEFAULT_MODEL`` / ``MODEL_NAME`` / ``EMBEDDING_DIM`` are defined here once and shared by the
SigLIP image encoder, the ONNX text encoder and the loaders. Other models (alternative
encoders, test fakes) are made available to ``load_image_encoder(name)`` /
``load_text_encoder(name)`` with :func:`register_model`.

This module imports nothing heavy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import ModelUnavailableError

MODEL_NAME = "siglip2-b16-webli"
EMBEDDING_DIM = 768
DEFAULT_MODEL = MODEL_NAME

# Shared by error messages: the pip extra is empty on Python 3.9.
INSTALL_HINT = (
    "Photo search needs Python 3.10 or newer; install its dependencies with: "
    'pip install "facesorter[search]"'
)

ImageFactory = Callable[[str], Any]  # (device) -> ImageEncoder
TextFactory = Callable[[str], Any]  # (quantization) -> TextEncoder


@dataclass(frozen=True)
class ModelSpec:
    """How to build the two halves of one embedding model."""

    name: str
    image_factory: ImageFactory
    text_factory: TextFactory
    dim: int | None = None


_MODELS: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    image_factory: ImageFactory,
    text_factory: TextFactory,
    dim: int | None = None,
    replace: bool = False,
) -> None:
    """Make ``name`` loadable by ``load_image_encoder`` / ``load_text_encoder``.

    ``image_factory(device)`` and ``text_factory(quantization)`` return encoders; they must be
    cheap (heavy work belongs in the encoder's lazy ``load()``, which the loaders call if the
    object has one). ``dim`` is the embedding size; if omitted, :func:`model_dimension` asks
    the text encoder. Registering an existing name raises ``ValueError`` unless ``replace``.

    Test fakes are registered like this (and should be unregistered afterwards)::

        from facesorter.search import testing

        register_model(
            testing.FAKE_MODEL_NAME,
            lambda device: testing.FakeImageEncoder(),
            lambda quantization: testing.FakeTextEncoder(),
            dim=testing.FAKE_DIM,
        )
        ...
        unregister_model(testing.FAKE_MODEL_NAME)
    """
    if not name:
        raise ValueError("model name must not be empty")
    if name in _MODELS and not replace:
        raise ValueError(f"model {name!r} is already registered")
    _MODELS[name] = ModelSpec(name, image_factory, text_factory, dim)


def unregister_model(name: str) -> None:
    """Remove a registered model; raises ``ModelUnavailableError`` if it is unknown."""
    spec_for(name)
    del _MODELS[name]


def available_models() -> tuple[str, ...]:
    """Names accepted by :func:`load_image_encoder` and :func:`load_text_encoder`."""
    return tuple(_MODELS)


def spec_for(name: str) -> ModelSpec:
    try:
        return _MODELS[name]
    except KeyError:
        raise ModelUnavailableError(
            f"Unknown model {name!r}; available: {', '.join(available_models()) or 'none'}"
        ) from None


def model_dimension(name: str) -> int:
    """Embedding size of model ``name`` without loading any weights (when registered with
    ``dim``; otherwise a lazily-constructed text encoder is asked)."""
    spec = spec_for(name)
    if spec.dim is not None:
        return spec.dim
    return int(spec.text_factory("fp16").dim)
