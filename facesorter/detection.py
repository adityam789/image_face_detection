"""
facesorter/detection.py
Face detection and embedding extraction using InsightFace's ``buffalo_l`` model pack
(SCRFD detector + ResNet50 recognition model producing 512-d embeddings).

InsightFace is imported lazily so that importing this module is cheap and does not
require the model files to be present.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)


@dataclass(frozen=True, eq=False)
class Face:
    """A detected face: bounding box ``(x1, y1, x2, y2)``, embedding, and detection score."""

    box: tuple[int, int, int, int]
    embedding: np.ndarray
    score: float

    @property
    def area(self) -> int:
        """Area of the bounding box in pixels."""
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def get_face_model(providers: Sequence[str] | None = None) -> FaceAnalysis:
    """
    Initialize and return the InsightFace analysis model (buffalo_l).

    Args:
        providers: ONNX Runtime execution providers. Defaults to CPU only.
    """
    from insightface.app import FaceAnalysis  # lazy: heavy import

    model = FaceAnalysis(providers=list(providers or DEFAULT_PROVIDERS))
    model.prepare(ctx_id=0)
    return model


def detect_faces_and_embeddings(model: FaceAnalysis, image: Image.Image) -> list[Face]:
    """
    Detect faces in an RGB PIL image and extract their embeddings.

    ``model`` only needs a ``get(img_bgr)`` method returning objects with ``bbox``,
    ``embedding`` and ``det_score`` attributes, which makes it easy to fake in tests.
    """
    img_bgr = np.ascontiguousarray(np.array(image)[:, :, ::-1])  # InsightFace expects BGR
    faces = []
    for raw in model.get(img_bgr):
        x1, y1, x2, y2 = (int(v) for v in raw.bbox)
        faces.append(
            Face(box=(x1, y1, x2, y2), embedding=raw.embedding, score=float(raw.det_score))
        )
    return faces
