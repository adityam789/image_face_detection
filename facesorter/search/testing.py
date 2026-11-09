"""
facesorter/search/testing.py
Deterministic stand-ins for the heavy components, for tests and offline demos.

* ``FakeImageEncoder`` / ``FakeTextEncoder`` share a tiny vocabulary of colour words: a
  solid-colour image embeds near the text of its colour name ("red", "blue", ...), so
  search results can be asserted exactly without any model weights.
* ``ScriptedLLM`` replays canned replies keyed by the last user message.
* ``make_color_image`` builds the matching synthetic photos.

Nothing here imports torch, onnxruntime or any model runtime.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image

FAKE_MODEL_NAME = "fake-color-16"
FAKE_DIM = 16

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 60, 220),
    "yellow": (230, 210, 40),
    "orange": (240, 140, 20),
    "purple": (130, 40, 170),
    "black": (15, 15, 15),
    "white": (240, 240, 240),
}


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _prototype(name: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return _unit(rng.standard_normal(FAKE_DIM).astype(np.float32))


_PROTOTYPES = {name: _prototype(name) for name in COLORS}


def make_color_image(color: str, size: tuple[int, int] = (32, 24)) -> Image.Image:
    """A solid-colour RGB image whose fake embedding points at ``color``."""
    return Image.new("RGB", size, COLORS[color])


class FakeImageEncoder:
    """Embeds an image by soft-assigning its mean colour to the colour vocabulary."""

    name = FAKE_MODEL_NAME
    dim = FAKE_DIM

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = np.zeros((len(images), FAKE_DIM), dtype=np.float32)
        names = list(COLORS)
        centres = np.array([COLORS[n] for n in names], dtype=np.float32)
        protos = np.stack([_PROTOTYPES[n] for n in names])
        for i, image in enumerate(images):
            mean = np.asarray(image.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)
            dist = np.linalg.norm(centres - mean, axis=1)
            weights = np.exp(-(dist - dist.min()) / 25.0)
            out[i] = _unit((weights[:, None] * protos).sum(axis=0))
        return out


class FakeTextEncoder:
    """Embeds text as the sum of its colour-word prototypes (hash vector if none)."""

    name = FAKE_MODEL_NAME
    dim = FAKE_DIM

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r"[a-z]+", text.lower())
            hits = [_PROTOTYPES[w] for w in words if w in _PROTOTYPES]
            if hits:
                out[i] = _unit(np.sum(hits, axis=0))
            else:
                out[i] = _prototype("text:" + text.lower().strip())
        return out


class ScriptedLLM:
    """Replays canned replies. ``replies`` maps the last user message to the reply text.

    Unknown messages return ``default`` (an empty JSON object unless overridden). Every
    call is recorded in ``calls`` so tests can assert on the prompt.
    """

    name = "scripted"

    def __init__(self, replies: Mapping[str, str] | None = None, default: str = "{}") -> None:
        self.replies = dict(replies or {})
        self.default = default
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: Sequence[Mapping[str, str]], max_tokens: int = 160) -> str:
        self.calls.append([dict(m) for m in messages])
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self.replies.get(last_user, self.default)
