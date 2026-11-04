"""Shared fakes and fixtures.

No test needs InsightFace, model weights, network access or real photos. Test images are
tiny solid-colour PNG/JPEG files whose red channel encodes a numeric "image id"; the fake
face model reads that id back from the pixels and returns the faces registered for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
from PIL import Image

EMBEDDING_DIM = 8
ID_SCALE = 10  # red channel = image id * ID_SCALE, so lossy codecs still decode correctly

# Image ids understood by the standard ``fake_model`` fixture.
ALICE, BOB, GROUP, NO_FACE, LOW_CONF, CRASH, ALICE_2, BOB_2 = 1, 2, 3, 4, 5, 6, 7, 8


def basis(index: int, *, tilt: float = 0.0) -> np.ndarray:
    """Unit axis vector ``e_index`` plus an optional small tilt towards the next axis."""
    vec = np.zeros(EMBEDDING_DIM)
    vec[index] = 1.0
    vec[(index + 1) % EMBEDDING_DIM] += tilt
    return vec


@dataclass
class FakeFace:
    """Mimics the attributes of an InsightFace ``Face`` that facesorter reads."""

    bbox: np.ndarray
    embedding: np.ndarray
    det_score: float


@dataclass
class FakeModel:
    """Stand-in for ``insightface.app.FaceAnalysis`` driven by the id encoded in the pixels."""

    faces_by_id: dict[int, list[FakeFace]] = field(default_factory=dict)
    failing_ids: set[int] = field(default_factory=set)
    seen_ids: list[int] = field(default_factory=list)

    def get(self, img_bgr: np.ndarray) -> list[FakeFace]:
        """Return the faces registered for the image id found at pixel (0, 0)."""
        image_id = int(round(float(img_bgr[0, 0, 2]) / ID_SCALE))  # BGR: red is channel 2
        self.seen_ids.append(image_id)
        if image_id in self.failing_ids:
            raise RuntimeError(f"model failed on image {image_id}")
        return list(self.faces_by_id.get(image_id, []))


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every test in an empty directory so the repo's ./config.ini is never picked up."""
    cwd = tmp_path_factory.mktemp("cwd")
    monkeypatch.chdir(cwd)
    return cwd


@pytest.fixture
def make_face() -> Callable[..., FakeFace]:
    """Factory ``make_face(embedding, score=0.99, size=50)`` for a fake detected face."""

    def _make(embedding: Any, score: float = 0.99, size: int = 50) -> FakeFace:
        box = np.array([10.0, 10.0, 10.0 + size, 10.0 + size])
        return FakeFace(bbox=box, embedding=np.asarray(embedding, dtype=float), det_score=score)

    return _make


@pytest.fixture
def make_model() -> Callable[..., FakeModel]:
    """Factory ``make_model(faces_by_id, failing_ids=())`` for custom fake models."""

    def _make(faces_by_id: dict[int, list[FakeFace]], failing_ids: Any = ()) -> FakeModel:
        return FakeModel(faces_by_id=dict(faces_by_id), failing_ids=set(failing_ids))

    return _make


@pytest.fixture
def fake_model(
    make_face: Callable[..., FakeFace], make_model: Callable[..., FakeModel]
) -> FakeModel:
    """Model that knows two people (Alice, Bob) plus no-face, low-confidence and crashing ids."""
    alice, bob = basis(0), basis(1)
    return make_model(
        {
            ALICE: [make_face(alice)],
            BOB: [make_face(bob)],
            GROUP: [make_face(alice, size=60), make_face(bob, size=30)],
            LOW_CONF: [make_face(alice, score=0.5)],
            ALICE_2: [make_face(basis(0, tilt=0.05))],  # same person, slightly different shot
            BOB_2: [make_face(basis(1, tilt=0.05))],
        },
        failing_ids={CRASH},
    )


@pytest.fixture
def make_image() -> Callable[..., str]:
    """Factory ``make_image(path, image_id, size=(32, 32), **save_kwargs)`` returning the path.

    The format follows the file extension; parent directories are created.
    """

    def _make(
        path: Any, image_id: int, size: tuple[int, int] = (32, 32), **save_kwargs: Any
    ) -> str:
        path = str(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new("RGB", size, (image_id * ID_SCALE, 128, 128)).save(path, **save_kwargs)
        return path

    return _make


@pytest.fixture
def photo_dir(
    tmp_path: Path, make_image: Callable[..., str], fake_model: FakeModel
) -> SimpleNamespace:
    """A small photo library with two people and every kind of awkward file.

    Attributes: ``source`` / ``output`` (str paths), ``model`` and ``files``
    (relative path -> absolute path). Layout (ids in brackets)::

        alice_1.png [Alice]   bob_1.PNG [Bob]        group.png [Alice + Bob]
        nofaces.png [none]    blurry.png [score 0.5] crash.png [model raises]
        broken.jpg [corrupt]  notes.txt [not an image]
        sub/alice_2.png [Alice]   sub/bob_2.png [Bob]
        sub/deeper/alice_1.png [Alice; same basename as the top-level alice_1.png]
    """
    source = tmp_path / "source"
    layout = {
        "alice_1.png": ALICE,
        "bob_1.PNG": BOB,
        "group.png": GROUP,
        "nofaces.png": NO_FACE,
        "blurry.png": LOW_CONF,
        "crash.png": CRASH,
        "sub/alice_2.png": ALICE_2,
        "sub/bob_2.png": BOB_2,
        "sub/deeper/alice_1.png": ALICE,
    }
    files = {rel: make_image(source / rel, image_id) for rel, image_id in layout.items()}
    for rel, content in {"broken.jpg": b"not really a jpeg", "notes.txt": b"hello"}.items():
        (source / rel).write_bytes(content)
        files[rel] = str(source / rel)
    return SimpleNamespace(
        source=str(source), output=str(tmp_path / "output"), model=fake_model, files=files
    )


@pytest.fixture
def reference_dir(tmp_path: Path, make_image: Callable[..., str]) -> str:
    """Directory with two reference photos of Alice."""
    make_image(tmp_path / "reference" / "ref_1.png", ALICE)
    make_image(tmp_path / "reference" / "ref_2.png", ALICE_2)
    return str(tmp_path / "reference")


@pytest.fixture
def write_config(isolated_cwd: Path) -> Callable[..., str]:
    """Factory ``write_config(body, name="config.ini")`` writing an INI file into the test cwd.

    With the default name the file is the implicit ``./config.ini`` the CLI auto-loads.
    """

    def _write(body: str, name: str = "config.ini") -> str:
        path = isolated_cwd / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def tree() -> Callable[[str], set[str]]:
    """Function listing all files below a directory as relative, forward-slash paths."""

    def _tree(root: str) -> set[str]:
        return {
            os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
            for d, _, names in os.walk(root)
            for f in names
        }

    return _tree
