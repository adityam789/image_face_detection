"""Face scanning and clustering into named people for photo search.

A fake face model reads a numeric "image id" from the red channel of solid-colour test images
and returns the faces registered for it, so no weights, network or real photos are involved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pytest

from facesorter.search import people
from facesorter.search.errors import ModelUnavailableError, SearchError
from facesorter.search.people import (
    ClusterReport,
    ScanReport,
    cluster_faces,
    cluster_summary,
    scan_faces,
)
from facesorter.search.store import FaceRow, Store
from facesorter.search.testing import (
    FAKE_DIM,
    FAKE_MODEL_NAME,
    FakeImageEncoder,
    FakeTextEncoder,
    make_color_image,
)
from facesorter.search.types import Filters, PhotoMeta

ID_SCALE = 10  # red channel = image id * ID_SCALE (same convention as tests/conftest.py)
FACE_DIM = 8
DEFAULT_BOX = (0.2, 0.2, 0.6, 0.6)  # fraction of the image the model actually receives

ALICE, BOB, BOBBY, CAROL = 0, 1, 2, 3  # identity axes


def ident(axis: int, tilt: float = 0.0) -> np.ndarray:
    vec = np.zeros(FACE_DIM)
    vec[axis] = 1.0
    vec[(axis + 4) % FACE_DIM] += tilt
    return vec


@dataclass
class FakeFace:
    bbox: np.ndarray
    embedding: np.ndarray
    det_score: float


@dataclass
class FakeModel:
    """``faces_by_id``: image id -> list of ``(embedding, score)`` or ``(embedding, score, box)``
    where ``box`` is a fraction of the image the model receives."""

    faces_by_id: dict = field(default_factory=dict)
    failing: set = field(default_factory=set)
    interrupt_on: set = field(default_factory=set)
    seen: list = field(default_factory=list)
    shapes: list = field(default_factory=list)

    def get(self, img_bgr: np.ndarray) -> list[FakeFace]:
        image_id = int(round(float(img_bgr[0, 0, 2]) / ID_SCALE))
        self.seen.append(image_id)
        h, w = img_bgr.shape[:2]
        self.shapes.append((h, w))
        if image_id in self.interrupt_on:
            raise KeyboardInterrupt
        if image_id in self.failing:
            raise RuntimeError(f"model failed on image {image_id}")
        out = []
        for spec in self.faces_by_id.get(image_id, []):
            emb, score, *rest = spec
            fx1, fy1, fx2, fy2 = rest[0] if rest else DEFAULT_BOX
            bbox = np.array([fx1 * w, fy1 * h, fx2 * w, fy2 * h])
            out.append(FakeFace(bbox=bbox, embedding=np.asarray(emb, dtype=float), det_score=score))
        return out


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "index.db", model_name=FAKE_MODEL_NAME, dim=FAKE_DIM) as s:
        yield s


@pytest.fixture
def library(store, tmp_path, make_image):
    """``library(name, image_id, color="red", size=(32, 32))`` creates and indexes a photo."""
    encoder = FakeImageEncoder()

    def add(name, image_id, color="red", size=(32, 32)):
        path = make_image(tmp_path / "photos" / name, image_id, size=size)
        return index(store, path, encoder, color)

    return add


def index(store, path, encoder=None, color="red"):
    encoder = encoder or FakeImageEncoder()
    st = os.stat(path)
    meta = PhotoMeta(path=path, size=st.st_size, mtime_ns=st.st_mtime_ns)
    vec = encoder.encode_images([make_color_image(color)])[0]
    return store.upsert_photos([(meta, None, vec)])[0]


def person_photos(store, name):
    """Photo ids that contain a face of the named person (through the real search filter)."""
    qvec = FakeTextEncoder().encode_text(["red"])[0]
    hits, _ = store.top_k(qvec, Filters(people=(name,)), k=1000)
    return {pid for pid, _ in hits}


def cluster_of(store, photo_id, idx=0):
    return store.get_faces(photo_id)[idx].cluster_id


# --------------------------------------------------------------------------- scanning


def test_scan_stores_faces_with_boxes_embeddings_and_report(store, library):
    alice1 = library("a1.png", 1)
    group = library("g.png", 2)
    empty = library("n.png", 3)
    blurry = library("b.png", 4)
    model = FakeModel(
        {
            1: [(ident(ALICE), 0.99)],
            2: [(ident(ALICE), 0.95, (0.1, 0.1, 0.4, 0.4)), (ident(BOB), 0.9)],
            4: [(ident(CAROL), 0.5)],  # below the default confidence of 0.8
        }
    )
    ticks = []
    report = scan_faces(
        store, model=model, progress=lambda done, total: ticks.append((done, total))
    )

    assert isinstance(report, ScanReport)
    assert (report.photos_scanned, report.faces_found, report.photos_without_faces) == (4, 3, 2)
    assert report.failed == () and report.seconds >= 0
    assert ticks == [(1, 4), (2, 4), (3, 4), (4, 4)]
    faces = store.get_faces(group)
    assert len(faces) == 2 and all(f.cluster_id is None for f in faces)
    assert (faces[0].x1, faces[0].y1, faces[0].x2, faces[0].y2) == (3.0, 3.0, 12.0, 12.0)
    assert faces[0].embedding is not None and faces[0].embedding.shape == (FACE_DIM,)
    assert store.get_faces(alice1)[0].embedding[ALICE] == pytest.approx(1.0, abs=1e-3)
    assert store.get_faces(empty) == [] and store.get_faces(blurry) == []
    assert store.photos_needing_face_scan() == []  # all marked, including those without faces


def test_scan_confidence_threshold_and_validation(store, library):
    library("a.png", 1)
    model = FakeModel({1: [(ident(ALICE), 0.5)]})
    assert scan_faces(store, model=model, confidence=0.4).faces_found == 1
    with pytest.raises(ValueError, match="confidence"):
        scan_faces(store, model=model, confidence=1.5)


def test_scan_is_incremental_and_respects_limit(store, library):
    for i in range(1, 5):
        library(f"p{i}.png", i)
    model = FakeModel({i: [(ident(ALICE), 0.99)] for i in range(1, 5)})
    first = scan_faces(store, model=model, limit=3)
    assert first.photos_scanned == 3 and len(store.photos_needing_face_scan()) == 1
    second = scan_faces(store, model=model)
    assert second.photos_scanned == 1
    assert scan_faces(store, model=model).photos_scanned == 0
    assert sorted(model.seen) == [1, 2, 3, 4]  # nothing detected twice
    with pytest.raises(ValueError):
        scan_faces(store, model=model, limit=-1)


def test_scan_resumes_after_keyboard_interrupt(store, library):
    for i in range(1, 6):
        library(f"p{i}.png", i)
    model = FakeModel({i: [(ident(ALICE), 0.99)] for i in range(1, 6)}, interrupt_on={4})
    with pytest.raises(KeyboardInterrupt):
        scan_faces(store, model=model)
    assert len(store.photos_needing_face_scan()) == 2  # photos 4 and 5 remain; 1-3 saved
    assert [len(store.get_faces(i)) for i in (1, 2, 3)] == [1, 1, 1]

    model.interrupt_on.clear()
    report = scan_faces(store, model=model)
    assert report.photos_scanned == 2 and report.faces_found == 2
    assert store.photos_needing_face_scan() == []
    assert model.seen == [1, 2, 3, 4, 4, 5]  # only the interrupted photo was redone


def test_changed_file_is_rescanned(store, library, make_image):
    pid = library("a.png", 1)
    model = FakeModel({1: [(ident(ALICE), 0.99)], 2: [(ident(BOB), 0.99), (ident(CAROL), 0.99)]})
    scan_faces(store, model=model)
    assert len(store.get_faces(pid)) == 1

    path = store.get_photos([pid])[0].meta.path
    make_image(path, 2, size=(64, 48))  # different content and size
    assert index(store, path) == pid  # re-index: same id, new size/mtime
    assert store.get_faces(pid) == []  # stale faces dropped by the store
    assert [p.id for p in store.photos_needing_face_scan()] == [pid]

    report = scan_faces(store, model=model)
    assert report.photos_scanned == 1 and report.faces_found == 2
    assert len(store.get_faces(pid)) == 2


def test_failed_photos_are_recorded_not_raised(store, library, tmp_path):
    good = library("good.png", 1)
    crash = library("crash.png", 2)
    zero = library("zero.png", 3)
    broken = tmp_path / "photos" / "broken.jpg"
    broken.write_bytes(b"not really a jpeg")
    broken_id = index(store, str(broken))
    gone = library("gone.png", 5)
    os.remove(store.get_photos([gone])[0].meta.path)
    model = FakeModel({1: [(ident(ALICE), 0.99)], 3: [(np.zeros(FACE_DIM), 0.99)]}, failing={2})

    report = scan_faces(store, model=model)

    assert report.photos_scanned == 1 and report.faces_found == 1
    failed = dict(report.failed)
    paths = {pid: store.get_photos([pid])[0].meta.path for pid in (crash, zero, broken_id, gone)}
    assert set(failed) == set(paths.values())
    assert "RuntimeError" in failed[paths[crash]] and "model failed" in failed[paths[crash]]
    assert "embedding" in failed[paths[zero]]  # zero vector rejected by the store
    assert len(store.get_faces(good)) == 1
    assert store.get_faces(zero) == []  # nothing half-written
    # failed photos stay pending so a later run retries them
    assert {p.id for p in store.photos_needing_face_scan()} == set(paths)


def test_large_images_are_downscaled_and_boxes_scaled_back(store, library):
    big = library("big.png", 1, size=(3200, 2000))
    small = library("small.png", 1, size=(400, 200))
    model = FakeModel({1: [(ident(ALICE), 0.99, (0.25, 0.25, 0.5, 0.5))]})
    scan_faces(store, model=model)

    assert model.shapes == [(1000, 1600), (200, 400)]  # detector saw 1600 px max, small as is
    f = store.get_faces(big)[0]
    assert (f.x1, f.y1, f.x2, f.y2) == (800.0, 500.0, 1600.0, 1000.0)  # full-size coordinates
    g = store.get_faces(small)[0]
    assert (g.x1, g.y1, g.x2, g.y2) == (100.0, 50.0, 200.0, 100.0)


def test_extreme_aspect_scales_x_and_y_separately(store, library):
    # 4000x31 shrinks to 1600x12: x is scaled back by 2.5 but y by 31 / 12 (rounding).
    pid = library("strip.png", 1, size=(4000, 31))
    model = FakeModel({1: [(ident(ALICE), 0.99, (0.25, 0.25, 0.5, 0.5))]})
    scan_faces(store, model=model)
    assert model.shapes == [(12, 1600)]
    f = store.get_faces(pid)[0]
    assert (f.x1, f.x2) == (1000.0, 2000.0)
    assert (f.y1, f.y2) == (pytest.approx(3 * 31 / 12), pytest.approx(6 * 31 / 12))


def test_boxes_are_clipped_to_the_image(store, library):
    pid = library("a.png", 1, size=(100, 50))
    model = FakeModel({1: [(ident(ALICE), 0.99, (-0.2, -0.2, 1.5, 1.5))]})
    scan_faces(store, model=model)
    f = store.get_faces(pid)[0]
    assert (f.x1, f.y1, f.x2, f.y2) == (0.0, 0.0, 100.0, 50.0)


def test_tiny_aspect_ratio_never_resizes_to_zero(store, library):
    pid = library("strip.png", 1, size=(8000, 2))
    model = FakeModel({1: [(ident(ALICE), 0.99)]})
    assert scan_faces(store, model=model).photos_scanned == 1
    assert model.shapes == [(1, 1600)]
    assert len(store.get_faces(pid)) == 1


def test_default_model_is_loaded_lazily(store, library, monkeypatch):
    import facesorter.detection as detection

    loads = []
    fake = FakeModel({1: [(ident(ALICE), 0.99)]})
    monkeypatch.setattr(detection, "get_face_model", lambda: loads.append(1) or fake)

    assert scan_faces(store).photos_scanned == 0 and loads == []  # nothing to do: no load
    library("a.png", 1)
    assert scan_faces(store).faces_found == 1 and loads == [1]


def test_missing_insightface_gives_install_hint(store, library, monkeypatch):
    import facesorter.detection as detection

    def boom():
        raise ImportError("No module named 'insightface'")

    monkeypatch.setattr(detection, "get_face_model", boom)
    library("a.png", 1)
    with pytest.raises(ModelUnavailableError) as err:
        scan_faces(store)
    assert 'pip install "facesorter[search]"' in str(err.value)
    assert "insightface" in str(err.value)
    assert store.photos_needing_face_scan()  # nothing was marked


def test_model_load_failure_is_typed(store, library, monkeypatch):
    import facesorter.detection as detection

    def boom():
        raise OSError("offline")

    monkeypatch.setattr(detection, "get_face_model", boom)
    library("a.png", 1)
    with pytest.raises(ModelUnavailableError, match="offline"):
        scan_faces(store)


# --------------------------------------------------------------------------- clustering


def standard_model():
    return FakeModel(
        {
            1: [(ident(ALICE), 0.99)],
            2: [(ident(ALICE, 0.02), 0.99)],
            3: [(ident(BOB), 0.99)],
            4: [(ident(ALICE), 0.99), (ident(BOB, 0.02), 0.99)],
            5: [(ident(BOB, 0.03), 0.99)],
        }
    )


@pytest.fixture
def scanned(store, library):
    """Alice in photos 1, 2, 4; Bob in 3, 4, 5; all scanned but not clustered."""
    ids = {i: library(f"p{i}.png", i, "red" if i != 5 else "blue") for i in range(1, 6)}
    scan_faces(store, model=standard_model())
    return ids


def test_people_in_several_photos_form_one_cluster_each(store, scanned):
    report = cluster_faces(store)

    assert isinstance(report, ClusterReport)
    assert (report.faces, report.clusters, report.singletons) == (6, 2, 0)
    assert (report.named_kept, report.names_dropped) == (0, ())
    alice = {cluster_of(store, scanned[i]) for i in (1, 2)} | {cluster_of(store, scanned[4], 0)}
    bob = {cluster_of(store, scanned[i]) for i in (3, 5)} | {cluster_of(store, scanned[4], 1)}
    assert len(alice) == len(bob) == 1 and alice != bob
    assert None not in alice | bob
    summary = cluster_summary(store)
    assert [(c.face_count, c.photo_count) for c in summary] == [(3, 3), (3, 3)]


def test_cluster_summary_sorts_by_face_count_descending(store, library):
    for i in range(1, 5):
        library(f"p{i}.png", i)
    model = FakeModel(
        {
            1: [(ident(ALICE), 0.99)],
            2: [(ident(BOB), 0.99)],
            3: [(ident(BOB, 0.01), 0.99)],
            4: [(ident(BOB, 0.02), 0.99), (ident(CAROL), 0.99)],
        }
    )
    scan_faces(store, model=model)
    cluster_faces(store)
    counts = [c.face_count for c in cluster_summary(store)]
    assert counts == [3, 1, 1]


def test_clustering_is_deterministic_and_idempotent(store, scanned):
    cluster_faces(store)
    before = {i: [f.cluster_id for f in store.get_faces(p)] for i, p in scanned.items()}
    cluster_faces(store)
    after = {i: [f.cluster_id for f in store.get_faces(p)] for i, p in scanned.items()}
    assert before == after


def test_nothing_to_cluster(store):
    report = cluster_faces(store)
    assert (report.faces, report.clusters, report.named_kept) == (0, 0, 0)
    assert cluster_summary(store) == []


def test_argument_validation(store):
    with pytest.raises(ValueError, match="threshold"):
        cluster_faces(store, threshold=-0.1)
    with pytest.raises(ValueError, match="threshold"):
        cluster_faces(store, threshold=float("nan"))
    with pytest.raises(ValueError, match="min_cluster_faces"):
        cluster_faces(store, min_cluster_faces=0)


def test_names_carry_over_after_new_photos_and_recluster(store, library, scanned):
    cluster_faces(store)
    alice_id = cluster_of(store, scanned[1])
    bob_id = cluster_of(store, scanned[3])
    store.name_cluster(alice_id, "Alice")
    store.name_cluster(bob_id, "Bob")
    assert person_photos(store, "Bob") == {scanned[i] for i in (3, 4, 5)}

    new_bob = library("new_bob.png", 6)
    new_carol = library("new_carol.png", 7)
    model = FakeModel({6: [(ident(BOB, 0.01), 0.99)], 7: [(ident(CAROL), 0.99)]})
    assert scan_faces(store, model=model).photos_scanned == 2
    report = cluster_faces(store)

    assert (report.faces, report.clusters, report.named_kept, report.names_dropped) == (8, 3, 2, ())
    assert sorted(store.known_names()) == ["Alice", "Bob"]
    assert person_photos(store, "Bob") == {scanned[i] for i in (3, 4, 5)} | {new_bob}
    assert person_photos(store, "Alice") == {scanned[i] for i in (1, 2, 4)}
    # ids are stable for people who did not change identity, and Carol is unnamed
    assert cluster_of(store, scanned[1]) == alice_id and cluster_of(store, scanned[3]) == bob_id
    carol_id = cluster_of(store, new_carol)
    assert carol_id not in (alice_id, bob_id)
    assert {c.cluster_id: c.name for c in store.clusters()}[carol_id] is None


def two_bobs(store, library):
    """Bob (3 faces) and Bobby (2 faces) are separate clusters at threshold 0.05 and one
    cluster at 0.3 (cosine distance between them is about 0.1)."""
    for i in range(1, 6):
        library(f"p{i}.png", i)
    model = FakeModel(
        {
            1: [(ident(BOB), 0.99)],
            2: [(ident(BOB), 0.99)],
            3: [(ident(BOB), 0.99)],
            4: [(ident(BOB, 0.5), 0.99)],
            5: [(ident(BOB, 0.5), 0.99)],
        }
    )
    scan_faces(store, model=model)
    cluster_faces(store, threshold=0.05)
    big, small = cluster_of(store, 1), cluster_of(store, 4)
    assert big != small
    return big, small


def test_merging_two_named_clusters_keeps_the_larger_ones_name(store, library):
    big, small = two_bobs(store, library)
    store.name_cluster(big, "Bob")
    store.name_cluster(small, "Bobby")

    report = cluster_faces(store, threshold=0.3)

    assert report.clusters == 1 and report.faces == 5
    assert report.named_kept == 1 and report.names_dropped == ("Bobby",)
    assert store.known_names() == ["Bob"]
    assert {c.name for c in store.clusters()} == {"Bob"}
    assert cluster_of(store, 1) == cluster_of(store, 4) == big  # the larger id survives


def test_equal_sized_merge_is_resolved_by_lower_old_id(store, library):
    for i in range(1, 5):
        library(f"p{i}.png", i)
    model = FakeModel(
        {
            1: [(ident(BOB), 0.99)],
            2: [(ident(BOB), 0.99)],
            3: [(ident(BOB, 0.5), 0.99)],
            4: [(ident(BOB, 0.5), 0.99)],
        }
    )
    scan_faces(store, model=model)
    cluster_faces(store, threshold=0.05)
    a, b = cluster_of(store, 1), cluster_of(store, 3)
    store.name_cluster(b, "Zed")
    store.name_cluster(a, "Amy")
    report = cluster_faces(store, threshold=0.3)
    assert report.names_dropped == ("Zed",) and store.known_names() == ["Amy"]  # a < b
    assert (a, b) == (1, 2)
    assert {cluster_of(store, i) for i in (1, 2, 3, 4)} == {a}  # the lowest old id is reused


def test_name_falls_back_to_a_pure_cluster_when_its_best_one_is_taken(store, library):
    """Bob (11 faces) is split in 6 + 4 when his bridging face goes away, and Alice's 8 faces
    merge into the 6-face half. Alice's cluster (14 faces) holds 6 of Bob's faces, but the pure
    4-face Bob cluster must get his name rather than leaving Bob unnamed."""

    def at(deg):
        rad = np.radians(deg)
        vec = np.zeros(FACE_DIM)
        vec[0], vec[1] = np.cos(rad), np.sin(rad)
        return vec

    layout = {}  # image id -> angle
    bob6, bob4, alice = range(1, 7), range(7, 11), range(12, 20)
    layout.update({i: 0 for i in bob6})
    layout.update({i: 60 for i in bob4})
    layout.update({11: 30})  # the bridge between Bob's two halves
    layout.update({i: -35 for i in alice})
    ids = {i: library(f"p{i}.png", i) for i in layout}
    scan_faces(store, model=FakeModel({i: [(at(a), 0.99)] for i, a in layout.items()}))

    report = cluster_faces(store, threshold=0.15)  # Bob chains through the bridge, Alice apart
    assert report.clusters == 2
    bob_id, alice_id = cluster_of(store, ids[1]), cluster_of(store, ids[12])
    assert bob_id != alice_id and cluster_of(store, ids[7]) == bob_id
    store.name_cluster(bob_id, "Bob")
    store.name_cluster(alice_id, "Alice")
    store.set_faces(ids[11], [])  # the bridge disappears (for example the photo was re-scanned)

    report = cluster_faces(store, threshold=0.2)  # Alice (distance 0.18) now joins Bob's 6

    assert (report.clusters, report.named_kept, report.names_dropped) == (2, 2, ())
    assert person_photos(store, "Bob") == {ids[i] for i in bob4}
    assert person_photos(store, "Alice") == {ids[i] for i in (*bob6, *alice)}


@pytest.mark.parametrize(("carol_faces", "bob_keeps_name"), [(4, False), (2, True)])
def test_fallback_needs_at_least_half_of_the_cluster(store, library, carol_faces, bob_keeps_name):
    """Alice (8 faces) beats Bob (2 faces) for the big cluster; Bob then falls back to the
    small cluster M holding one of his faces, but only if that is at least half of M."""
    total = 10 + carol_faces
    ids = {i: library(f"p{i}.png", i) for i in range(1, total + 1)}
    model = FakeModel(
        {i: [(ident(ALICE if i <= 10 else CAROL), 0.99)] for i in range(1, total + 1)}
    )
    scan_faces(store, model=model)
    old = {(ids[i], 0): 1 for i in range(1, 9)}  # Alice
    old.update({(ids[9], 0): 2, (ids[10], 0): 2, (ids[11], 0): 2})  # Bob: 2 + 1 faces
    old.update({(ids[i], 0): 3 for i in range(12, total + 1)})  # unnamed rest of M
    store.recluster(old)
    store.name_cluster(1, "Alice")
    store.name_cluster(2, "Bob")

    report = cluster_faces(store)

    assert report.clusters == 2
    assert person_photos(store, "Alice") == {ids[i] for i in range(1, 11)}
    assert store.known_names() == (["Alice", "Bob"] if bob_keeps_name else ["Alice"])
    assert report.names_dropped == (() if bob_keeps_name else ("Bob",))
    if bob_keeps_name:
        assert person_photos(store, "Bob") == {ids[11], ids[12]}


def test_split_gives_the_name_to_the_larger_part(store, library):
    big, small = two_bobs(store, library)
    cluster_faces(store, threshold=0.3)  # merge, then name the merged cluster
    merged = cluster_of(store, 1)
    store.name_cluster(merged, "Bob")

    report = cluster_faces(store, threshold=0.05)  # split again

    assert report.clusters == 2 and report.named_kept == 1 and report.names_dropped == ()
    assert person_photos(store, "Bob") == {1, 2, 3}  # the 3-face part keeps the name
    assert [c.name for c in cluster_summary(store)] == ["Bob", None]


def test_name_is_dropped_when_its_cluster_dissolves(store, library):
    big, small = two_bobs(store, library)
    store.name_cluster(small, "Bobby")
    store.name_cluster(big, "Bob")

    report = cluster_faces(store, threshold=0.05, min_cluster_faces=3)

    assert report.clusters == 1 and report.named_kept == 1
    assert report.names_dropped == ("Bobby",)
    assert store.known_names() == ["Bob"]
    assert cluster_of(store, 4) is None and cluster_of(store, 5) is None


def test_min_cluster_faces_leaves_strays_unassigned(store, scanned, library):
    stray = library("stray.png", 8)
    scan_faces(store, model=FakeModel({8: [(ident(CAROL), 0.99)]}))

    default = cluster_faces(store)
    assert default.clusters == 3 and default.singletons == 1
    assert cluster_of(store, stray) is not None

    strict = cluster_faces(store, min_cluster_faces=2)
    assert (strict.faces, strict.clusters, strict.singletons) == (7, 2, 1)
    assert cluster_of(store, stray) is None
    assert cluster_of(store, scanned[1]) is not None
    assert len(cluster_summary(store)) == 2  # the stray is not a "person"


def test_too_many_faces_raises_a_clear_error(store, scanned, monkeypatch):
    assert people.MAX_FACES == 10_000
    monkeypatch.setattr(people, "MAX_FACES", 5)
    with pytest.raises(SearchError, match="too many") as err:
        cluster_faces(store)
    assert "--limit" in str(err.value) and "threshold" in str(err.value)
    assert all(f.cluster_id is None for f in store.get_faces(scanned[1]))  # nothing changed


def test_clustered_faces_without_embeddings_are_never_left_under_a_stale_id(store, scanned):
    cluster_faces(store)
    legacy = scanned[1]
    old = cluster_of(store, legacy)
    store.set_faces(legacy, [FaceRow(old, 0, 0, 5, 5)])  # clustered, but no embedding

    cluster_faces(store)

    assert cluster_of(store, legacy) is None
    assert len(cluster_summary(store)) == 2


def test_cluster_summary_matches_store_clusters_sorted(store, scanned):
    cluster_faces(store)
    assert cluster_summary(store) == store.clusters()


# --------------------------------------------------------------------------- end to end


def test_naming_a_cluster_filters_search_to_that_person(store, library):
    p_alice = library("alice.png", 1, "red")
    p_bob = library("bob.png", 2, "red")
    p_both = library("both.png", 3, "red")
    p_bob_blue = library("bob_blue.png", 4, "blue")
    p_none = library("none.png", 5, "red")
    model = FakeModel(
        {
            1: [(ident(ALICE), 0.99)],
            2: [(ident(BOB), 0.99)],
            3: [(ident(ALICE, 0.01), 0.99), (ident(BOB, 0.01), 0.99)],
            4: [(ident(BOB, 0.02), 0.99)],
        }
    )
    scan_faces(store, model=model)
    cluster_faces(store)
    query = FakeTextEncoder().encode_text(["red"])[0]

    everyone, _ = store.top_k(query, None, k=10)
    assert {pid for pid, _ in everyone} == {p_alice, p_bob, p_both, p_bob_blue, p_none}

    store.name_cluster(cluster_of(store, p_bob), "Bob")
    hits, candidates = store.top_k(query, Filters(people=("Bob",)), k=10)
    assert candidates == 3
    assert {pid for pid, _ in hits} == {p_bob, p_both, p_bob_blue}
    assert hits[-1][0] == p_bob_blue  # still ranked by the visual text: red beats blue
    assert store.names_for_photos([p_both, p_alice]) == {p_both: ["Bob"]}

    store.name_cluster(cluster_of(store, p_alice), "Alice")
    both, _ = store.top_k(query, Filters(people=("alice", "BOB")), k=10)
    assert [pid for pid, _ in both] == [p_both]
