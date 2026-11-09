"""Tests for the facesorter.search.encoders package API, plus gated real-model checks.

The real-model tests need the SigLIP 2 weights (cached or downloadable) and are skipped
unless FACESORTER_MODEL_TESTS=1.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from facesorter.search.encoders import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    OnnxTextEncoder,
    SigLIPImageEncoder,
    available_models,
    load_for_model,
    load_image_encoder,
    load_text_encoder,
    model_dimension,
    register_model,
    registry,
    spec_for,
    unregister_model,
)
from facesorter.search.errors import ModelUnavailableError
from facesorter.search.types import ImageEncoder, TextEncoder


class _Fake:
    def __init__(self, kind: str, arg: str) -> None:
        self.kind, self.arg, self.loaded = kind, arg, 0
        self.dim = 5

    def load(self) -> None:
        self.loaded += 1


@pytest.fixture
def fake_models() -> Iterator[None]:
    register_model("fake-m", lambda device: _Fake("image", device), lambda q: _Fake("text", q), 5)
    yield
    unregister_model("fake-m")


def test_default_model_and_listing() -> None:
    assert DEFAULT_MODEL == "siglip2-b16-webli"
    assert available_models() == (DEFAULT_MODEL,)
    assert SigLIPImageEncoder.name == OnnxTextEncoder.name == DEFAULT_MODEL
    assert SigLIPImageEncoder.dim == OnnxTextEncoder.dim == 768


def test_encoders_satisfy_protocols() -> None:
    assert isinstance(SigLIPImageEncoder(), ImageEncoder)
    assert isinstance(OnnxTextEncoder(), TextEncoder)


def test_load_image_encoder_passes_device_and_loads(fake_models: None) -> None:
    enc = load_image_encoder("fake-m", device="cpu")
    assert (enc.kind, enc.arg, enc.loaded) == ("image", "cpu", 1)


def test_load_text_encoder_passes_quantization_and_loads(fake_models: None) -> None:
    enc = load_text_encoder("fake-m", quantization="int8")
    assert (enc.kind, enc.arg, enc.loaded) == ("text", "int8", 1)


def test_factories_without_load_are_accepted() -> None:
    class Plain:
        name, dim = "plain", 3

    register_model("plain", lambda device: Plain(), lambda q: Plain(), 3)
    try:
        assert isinstance(load_image_encoder("plain"), Plain)
        assert isinstance(load_text_encoder("plain"), Plain)
    finally:
        unregister_model("plain")


def test_default_loaders_build_the_siglip_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SigLIPImageEncoder, "load", lambda self: None)
    monkeypatch.setattr(OnnxTextEncoder, "load", lambda self: None)
    image = load_image_encoder(device="cpu")
    text = load_text_encoder(quantization="int8")
    assert isinstance(image, SigLIPImageEncoder) and image.requested_device == "cpu"
    assert isinstance(text, OnnxTextEncoder) and text.quantization == "int8"


def test_register_unregister_and_listing(fake_models: None) -> None:
    assert available_models() == (DEFAULT_MODEL, "fake-m")
    with pytest.raises(ValueError, match="already registered"):
        register_model("fake-m", lambda d: None, lambda q: None, 5)
    register_model("fake-m", lambda d: _Fake("image2", d), lambda q: None, 6, replace=True)
    assert load_image_encoder("fake-m").kind == "image2"
    assert model_dimension("fake-m") == 6
    unregister_model("fake-m")
    assert available_models() == (DEFAULT_MODEL,)
    with pytest.raises(ModelUnavailableError):
        unregister_model("fake-m")
    register_model("fake-m", lambda d: None, lambda q: None, 5)  # fixture teardown unregisters


def test_register_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        register_model("", lambda d: None, lambda q: None, 1)


def test_model_dimension() -> None:
    assert model_dimension(DEFAULT_MODEL) == 768 == EMBEDDING_DIM
    register_model("nodim", lambda d: None, lambda q: _Fake("text", q))
    try:
        assert model_dimension("nodim") == 5  # asked the lazily built text encoder
    finally:
        unregister_model("nodim")
    with pytest.raises(ModelUnavailableError):
        model_dimension("nope")


def test_testing_fakes_can_be_registered() -> None:
    from facesorter.search import testing

    register_model(
        testing.FAKE_MODEL_NAME,
        lambda device: testing.FakeImageEncoder(),
        lambda quantization: testing.FakeTextEncoder(),
        dim=testing.FAKE_DIM,
    )
    try:
        image = load_image_encoder(testing.FAKE_MODEL_NAME)
        text = load_text_encoder(testing.FAKE_MODEL_NAME)
        assert model_dimension("fake-color-16") == 16 == image.dim == text.dim
        scores = (
            text.encode_text(["red"]) @ image.encode_images([testing.make_color_image("red")]).T
        )
        assert scores[0, 0] > 0.5
    finally:
        unregister_model(testing.FAKE_MODEL_NAME)


def test_constants_defined_once() -> None:
    from facesorter.search.encoders import onnx_text, registry, siglip_image

    assert onnx_text.MODEL_NAME is siglip_image.MODEL_NAME is registry.MODEL_NAME
    assert onnx_text.EMBEDDING_DIM == siglip_image.EMBEDDING_DIM == registry.EMBEDDING_DIM


def test_unknown_model_names() -> None:
    with pytest.raises(ModelUnavailableError, match=r"nope.*siglip2-b16-webli"):
        load_image_encoder("nope")
    with pytest.raises(ModelUnavailableError, match="nope"):
        load_text_encoder("nope")
    with pytest.raises(ModelUnavailableError):
        spec_for("nope")


def test_empty_registry_message() -> None:
    saved = dict(registry._MODELS)
    registry._MODELS.clear()
    try:
        with pytest.raises(ModelUnavailableError, match="available: none"):
            spec_for("x")
    finally:
        registry._MODELS.update(saved)


def test_importing_the_package_is_cheap() -> None:
    code = (
        "import sys, facesorter.search.encoders as e;"
        "heavy = ['torch','open_clip','onnxruntime','tokenizers','huggingface_hub',"
        "'pillow_heif','transformers'];"
        "bad = [m for m in heavy if m in sys.modules];"
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- real models

real_models = pytest.mark.skipif(
    os.environ.get("FACESORTER_MODEL_TESTS") != "1",
    reason="set FACESORTER_MODEL_TESTS=1 to run tests that load the real SigLIP 2 weights",
)

CAPTIONS = [
    "a dog sleeping on a couch",
    "two people riding horses on the beach",
    "a red double decker bus in the city",
    "a slice of pizza on a plate",
    "A man playing tennis!",
    "a giraffe standing in a field",
    "a kitchen with white cabinets",
    "a train at the station_platform",
    "an airplane flying in the sky",
    "a bowl of fruit, with bananas & apples",
]


@pytest.fixture(scope="module")
def image_encoder() -> SigLIPImageEncoder:
    enc = load_image_encoder()
    yield enc  # type: ignore[misc]
    enc.close()


@pytest.fixture(scope="module")
def text_encoder() -> OnnxTextEncoder:
    return load_text_encoder()


@real_models
def test_real_onnx_text_matches_torch_text(text_encoder: OnnxTextEncoder) -> None:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2", pretrained="webli")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2")
    with torch.no_grad():
        reference = model.encode_text(tokenizer(CAPTIONS), normalize=True).numpy()
    ours = text_encoder.encode_text(CAPTIONS)
    assert ours.shape == (10, 768) and ours.dtype == np.float32
    assert (ours * reference).sum(axis=1).min() >= 0.999


LONG_CAPTION = " ".join(f"a photo of a dog number {i} chasing a ball" for i in range(30))


@real_models
def test_real_long_caption_token_ids_and_embedding_match_open_clip(
    text_encoder: OnnxTextEncoder,
) -> None:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2", pretrained="webli")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP2")
    texts = [*CAPTIONS, LONG_CAPTION]
    reference_ids = tokenizer(texts).numpy()
    text_encoder.load()
    ours = text_encoder._token_ids(text_encoder._tokenizer, texts)
    assert np.array_equal(ours, reference_ids)
    assert ours[-1, -1] == 1  # the long caption keeps its EOS
    with torch.no_grad():
        reference = model.encode_text(tokenizer(texts), normalize=True).numpy()
    assert (text_encoder.encode_text(texts) * reference).sum(axis=1).min() >= 0.999


@real_models
def test_real_text_batch_equals_single(text_encoder: OnnxTextEncoder) -> None:
    batch = text_encoder.encode_text(CAPTIONS)
    singles = np.concatenate([text_encoder.encode_text([c]) for c in CAPTIONS])
    assert (batch * singles).sum(axis=1).min() >= 0.9999


def _colour(rgb: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (300, 200), rgb)


@real_models
def test_real_image_encoder_unit_norm_and_ranks_colours(
    image_encoder: SigLIPImageEncoder, text_encoder: OnnxTextEncoder
) -> None:
    images = [_colour((220, 20, 20)), _colour((20, 200, 40)), _colour((20, 40, 220))]
    queries = ["a red image", "a green image", "a blue image"]
    emb = image_encoder.encode_images(images)
    assert emb.shape == (3, 768) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
    scores = text_encoder.encode_text(queries) @ emb.T
    assert scores.argmax(axis=1).tolist() == [0, 1, 2]
    assert scores.argmax(axis=0).tolist() == [0, 1, 2]


@real_models
def test_real_fast_decode_keeps_embedding(
    image_encoder: SigLIPImageEncoder, tmp_path: Path
) -> None:
    scene = Image.new("RGB", (3000, 2000), (135, 190, 235))
    draw = ImageDraw.Draw(scene)
    draw.rectangle((0, 1300, 3000, 2000), fill=(60, 140, 60))
    draw.ellipse((2200, 150, 2600, 550), fill=(250, 220, 40))
    draw.polygon([(800, 1300), (1400, 600), (2000, 1300)], fill=(150, 110, 80))
    path = tmp_path / "scene.jpg"
    scene.save(path, quality=92)
    naive = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = image_encoder.encode_images([naive])[0]
    b = image_encoder.encode_images([load_for_model(path)])[0]
    assert float(a @ b) >= 0.99
