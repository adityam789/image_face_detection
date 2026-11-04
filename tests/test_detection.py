from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from conftest import CRASH
from PIL import Image

from facesorter import detection
from facesorter.detection import Face, detect_faces_and_embeddings, get_face_model


class RecordingModel:
    """Captures the array it was given and returns canned faces."""

    def __init__(self, faces):
        self.faces = faces
        self.received = None

    def get(self, img_bgr):
        self.received = img_bgr
        return self.faces


def raw_face(bbox, score=0.9, embedding=(1.0, 0.0)):
    return SimpleNamespace(bbox=np.array(bbox), embedding=np.array(embedding), det_score=score)


class TestFace:
    def test_area(self):
        assert Face((10, 20, 40, 60), np.zeros(2), 0.9).area == 30 * 40

    @pytest.mark.parametrize("box", [(10, 10, 10, 50), (50, 50, 10, 10), (0, 0, -5, 5)])
    def test_degenerate_boxes_have_zero_area(self, box):
        assert Face(box, np.zeros(2), 0.9).area == 0

    def test_equality_ignores_embedding_arrays(self):
        a = Face((0, 0, 1, 1), np.zeros(2), 0.5)
        assert a != Face((0, 0, 1, 1), np.zeros(2), 0.5)  # eq=False: identity comparison
        assert a == a


class TestDetectFacesAndEmbeddings:
    def test_converts_rgb_to_contiguous_bgr(self):
        image = Image.new("RGB", (4, 3), (10, 20, 30))
        model = RecordingModel([])
        detect_faces_and_embeddings(model, image)
        assert model.received.shape == (3, 4, 3)
        assert list(model.received[0, 0]) == [30, 20, 10]
        assert model.received.flags["C_CONTIGUOUS"]

    def test_maps_fields_and_coerces_types(self):
        model = RecordingModel([raw_face([1.9, 2.2, 30.7, 40.1], score=np.float32(0.75))])
        (face,) = detect_faces_and_embeddings(model, Image.new("RGB", (50, 50)))
        assert face.box == (1, 2, 30, 40)
        assert all(type(v) is int for v in face.box)
        assert type(face.score) is float
        assert face.score == pytest.approx(0.75)
        assert face.embedding.tolist() == [1.0, 0.0]

    def test_embedding_passes_through_unmodified(self):
        # No normalisation here: [3, 4] must not become [0.6, 0.8].
        model = RecordingModel([raw_face([0, 0, 5, 5], embedding=(3.0, 4.0))])
        (face,) = detect_faces_and_embeddings(model, Image.new("RGB", (10, 10)))
        assert face.embedding.tolist() == [3.0, 4.0]

    def test_no_faces_gives_empty_list(self):
        assert detect_faces_and_embeddings(RecordingModel([]), Image.new("RGB", (4, 4))) == []

    def test_keeps_model_order_and_low_scores(self):
        # Filtering by confidence is the caller's job; detection returns everything.
        model = RecordingModel([raw_face([0, 0, 5, 5], 0.2), raw_face([0, 0, 9, 9], 0.99)])
        faces = detect_faces_and_embeddings(model, Image.new("RGB", (10, 10)))
        assert [round(f.score, 2) for f in faces] == [0.2, 0.99]

    def test_model_errors_propagate(self, fake_model, make_image, tmp_path):
        from facesorter.utils import load_image

        image = load_image(make_image(tmp_path / "crash.png", CRASH))
        with pytest.raises(RuntimeError, match="model failed"):
            detect_faces_and_embeddings(fake_model, image)


class TestGetFaceModel:
    @pytest.fixture
    def fake_insightface(self, monkeypatch):
        """Install a fake ``insightface.app`` so no weights or onnxruntime are needed."""
        created = []

        class FakeFaceAnalysis:
            def __init__(self, providers):
                self.providers = providers
                self.prepared_with = None
                created.append(self)

            def prepare(self, ctx_id):
                self.prepared_with = ctx_id

        app = ModuleType("insightface.app")
        app.FaceAnalysis = FakeFaceAnalysis
        package = ModuleType("insightface")
        package.app = app
        monkeypatch.setitem(sys.modules, "insightface", package)
        monkeypatch.setitem(sys.modules, "insightface.app", app)
        return created

    def test_defaults_to_cpu_provider_and_prepares(self, fake_insightface):
        model = get_face_model()
        assert model.providers == list(detection.DEFAULT_PROVIDERS) == ["CPUExecutionProvider"]
        assert model.prepared_with == 0

    def test_custom_providers_are_passed_as_list(self, fake_insightface):
        model = get_face_model(("CoreMLExecutionProvider", "CPUExecutionProvider"))
        assert model.providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
