"""Store: generic meta, face embeddings, scan marks and re-clustering without re-detection."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from facesorter.search.errors import SearchError
from facesorter.search.store import DuplicateNameError, FaceRow, Store
from facesorter.search.types import PhotoMeta

DIM = 8


def vec(seed: int, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def photo(path: str, size: int = 1, mtime_ns: int = 1, seed: int = 0):
    return (PhotoMeta(path=path, size=size, mtime_ns=mtime_ns), None, vec(seed))


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "i.db", model_name="m", dim=DIM) as s:
        yield s


@pytest.fixture
def three(store):
    return store, store.upsert_photos([photo(f"/p{i}.jpg", seed=i) for i in range(3)])


def face(cluster, seed=None, dim=5, box=10):
    return FaceRow(cluster, 0, 0, box, box, None if seed is None else vec(seed, dim))


# --------------------------------------------------------------------------- meta


def test_meta_round_trip_default_and_persistence(tmp_path):
    path = tmp_path / "i.db"
    with Store.open(path, model_name="m", dim=DIM) as s:
        assert s.get_meta("library_roots") is None
        assert s.get_meta("library_roots", "[]") == "[]"
        s.set_meta("library_roots", '["/a", "/b"]')
        s.set_meta("indexed_at", "2025-01-02T03:04:05")
        s.set_meta("indexed_at", "2025-02-02T03:04:05")  # overwrite
        assert s.get_meta("model_name") == "m" and s.get_meta("dim") == str(DIM)  # readable
    with Store.open(path, create=False) as s:
        assert s.get_meta("library_roots") == '["/a", "/b"]'
        assert s.get_meta("indexed_at") == "2025-02-02T03:04:05"
        assert (s.model_name, s.dim) == ("m", DIM)


@pytest.mark.parametrize("key", ["model_name", "dim", "created", "schema", "schema_version", ""])
def test_reserved_or_empty_meta_keys_rejected(store, key):
    with pytest.raises(ValueError, match="meta key"):
        store.set_meta(key, "x")
    assert store.model_name == "m" and store.dim == DIM


def test_non_string_meta_value_rejected(store):
    with pytest.raises(ValueError, match="string"):
        store.set_meta("k", 5)  # type: ignore[arg-type]
    assert store.get_meta("k") is None


# --------------------------------------------------------------------------- face embeddings


def test_face_embedding_round_trip_is_normalised_float32(three):
    store, (a, *_) = three
    raw = vec(1, 5) * 7  # not unit length, length 5 (not the photo dim)
    store.set_faces(a, [FaceRow(1, 0, 0, 4, 4, raw), FaceRow(None, 1, 1, 2, 2)])
    got = store.get_faces(a)
    assert got[0].embedding.dtype == np.float32 and got[0].embedding.shape == (5,)
    assert np.allclose(got[0].embedding, raw / np.linalg.norm(raw), atol=1e-3)
    assert float(np.linalg.norm(got[0].embedding)) == pytest.approx(1.0, abs=1e-3)
    assert got[1].embedding is None
    assert got[0] == FaceRow(1, 0, 0, 4, 4)  # equality ignores the embedding


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(4, dtype=np.int32),
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([1.0, np.inf], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    ],
    ids=["int", "nan", "inf", "zero", "2d", "empty"],
)
def test_bad_face_embeddings_rejected_and_old_faces_kept(three, bad):
    store, (a, *_) = three
    store.set_faces(a, [face(1, seed=1)])
    with pytest.raises(ValueError, match="face 1 embedding"):
        store.set_faces(a, [face(2, seed=2), FaceRow(2, 0, 0, 1, 1, bad)])
    assert [f.cluster_id for f in store.get_faces(a)] == [1]


def test_face_embeddings_arrays(three):
    store, (a, b, c) = three
    empty = store.face_embeddings()
    assert [x.shape for x in empty] == [(0,), (0,), (0, 0)]
    assert [x.dtype for x in empty] == [np.int64, np.int32, np.float32]
    store.set_faces(c, [face(1, seed=3), face(1)])  # second face has no embedding
    store.set_faces(a, [face(2), face(2, seed=1), face(1, seed=2)])
    store.set_faces(b, [])
    ids, idx, emb = store.face_embeddings()
    assert ids.tolist() == [a, a, c] and idx.tolist() == [1, 2, 0]
    assert ids.dtype == np.int64 and idx.dtype == np.int32 and emb.dtype == np.float32
    assert emb.shape == (3, 5)
    expected = vec(1, 5) / np.linalg.norm(vec(1, 5))
    assert np.allclose(emb[0], expected, atol=1e-3)


def test_mixed_embedding_sizes_raise(three):
    store, (a, b, _) = three
    store.set_faces(a, [face(1, seed=1, dim=5)])
    store.set_faces(b, [face(1, seed=1, dim=6)])
    with pytest.raises(SearchError, match="different sizes"):
        store.face_embeddings()


def test_face_embeddings_survive_reopen_and_cascade_on_delete(tmp_path):
    path = tmp_path / "i.db"
    with Store.open(path, model_name="m", dim=DIM) as s:
        a, b = s.upsert_photos([photo("/a"), photo("/b", seed=1)])
        s.set_faces(a, [face(1, seed=1)])
        s.set_faces(b, [face(1, seed=2)])
    with Store.open(path, create=False) as s:
        assert s.face_embeddings()[0].tolist() == [a, b]
        s.delete_paths(["/a"])
        assert s.face_embeddings()[0].tolist() == [b]


# --------------------------------------------------------------------------- scan marks


def test_photos_needing_face_scan_and_marks(three):
    store, ids = three
    assert [p.id for p in store.photos_needing_face_scan()] == ids
    assert [p.id for p in store.photos_needing_face_scan(2)] == ids[:2]
    assert store.photos_needing_face_scan(0) == []
    store.mark_faces_scanned(ids[1])
    store.mark_faces_scanned(ids[1])  # idempotent
    needing = store.photos_needing_face_scan()
    assert [p.id for p in needing] == [ids[0], ids[2]]
    assert needing[0].meta.path == "/p0.jpg"  # full IndexedPhoto
    assert [p.id for p in store.photos_needing_face_scan(1)] == [ids[0]]
    for i in ids:
        store.mark_faces_scanned(i)
    assert store.photos_needing_face_scan() == []


def test_scan_arguments_validated(store):
    with pytest.raises(ValueError, match="unknown photo"):
        store.mark_faces_scanned(99)
    with pytest.raises(ValueError, match="limit"):
        store.photos_needing_face_scan(-1)


@pytest.mark.parametrize("changed", [{"size": 2}, {"mtime_ns": 9}])
def test_changed_file_clears_scan_mark_and_faces(three, changed):
    store, (a, b, _) = three
    for i in (a, b):
        store.set_faces(i, [face(1, seed=i)])
        store.mark_faces_scanned(i)
    store.upsert_photos([photo("/p0.jpg", **changed)])
    assert store.get_faces(a) == [] and store.get_faces(b) != []
    assert [p.id for p in store.photos_needing_face_scan()][0] == a
    assert b not in [p.id for p in store.photos_needing_face_scan()]


def test_metadata_only_reindex_keeps_scan_mark(three):
    store, (a, *_) = three
    store.mark_faces_scanned(a)
    store.upsert_photos([photo("/p0.jpg", seed=5)])  # same size / mtime, new embedding
    assert a not in [p.id for p in store.photos_needing_face_scan()]


def test_stale_scan_detected_even_without_upsert(tmp_path):
    path = tmp_path / "i.db"
    with Store.open(path, model_name="m", dim=DIM) as s:
        (a,) = s.upsert_photos([photo("/a")])
        s.mark_faces_scanned(a)
    raw = sqlite3.connect(path)  # e.g. a file changed by some other tool
    raw.execute("UPDATE photos SET mtime_ns = 12345")
    raw.commit()
    raw.close()
    with Store.open(path, create=False) as s:
        assert [p.id for p in s.photos_needing_face_scan()] == [a]


def test_scan_marks_cascade_with_photo_delete(three):
    store, (a, *_) = three
    store.mark_faces_scanned(a)
    store.delete_paths(["/p0.jpg"])
    (new,) = store.upsert_photos([photo("/p0.jpg")])
    assert new != a and new in [p.id for p in store.photos_needing_face_scan()]


def test_clear_faces_clears_scans_unless_kept(three):
    store, ids = three
    for i in ids:
        store.set_faces(i, [face(1)])
        store.mark_faces_scanned(i)
    store.clear_faces(keep_scans=True)
    assert store.stats().faces == 0 and store.photos_needing_face_scan() == []
    store.clear_faces()
    assert [p.id for p in store.photos_needing_face_scan()] == ids


# --------------------------------------------------------------------------- recluster


@pytest.fixture
def clustered(three):
    store, (a, b, c) = three
    store.set_faces(a, [face(1, seed=1), face(2, seed=2)])
    store.set_faces(b, [face(1, seed=3)])
    store.set_faces(c, [face(3, seed=4)])
    store.name_cluster(1, "Bob")
    store.name_cluster(3, "Carol")
    return store, (a, b, c)


def cluster_map(store):
    return {c.cluster_id: (c.face_count, c.name) for c in store.clusters()}


def test_recluster_without_names_keeps_surviving_names(clustered):
    store, (a, b, c) = clustered
    # merge cluster 2 into 1, cluster 3 disappears (its face becomes unclustered)
    store.recluster({(a, 1): 1, (c, 0): None})
    assert cluster_map(store) == {1: (3, "Bob")}
    assert store.known_names() == ["Bob"]
    assert store.stats().named_people == 1  # Carol's row removed with her cluster
    assert store.get_faces(c)[0].cluster_id is None


def test_recluster_with_names_replaces_people_table(clustered):
    store, (a, b, c) = clustered
    store.recluster(
        {(a, 0): 10, (a, 1): 11, (b, 0): 10, (c, 0): 12},
        names={10: "  Robert  ", 12: "Carol"},
    )
    assert cluster_map(store) == {10: (2, "Robert"), 11: (1, None), 12: (1, "Carol")}
    assert store.stats().named_people == 2
    store.recluster({}, names={})  # an empty mapping clears every name
    assert store.known_names() == []


def test_recluster_blank_names_are_skipped_and_same_cluster_may_repeat_a_name(clustered):
    store, _ = clustered
    store.recluster({}, names={1: "Bob", 2: "   ", 3: "Carol"})
    assert store.known_names() == ["Bob", "Carol"]


def test_recluster_rolls_back_on_bad_input(clustered):
    store, (a, b, c) = clustered
    before = cluster_map(store)
    with pytest.raises(ValueError, match="unknown face"):
        store.recluster({(a, 0): 9, (a, 7): 9})
    with pytest.raises(DuplicateNameError):
        store.recluster({(a, 0): 9}, names={1: "Bob", 2: "BOB"})
    with pytest.raises(ValueError, match="unknown face cluster 42"):
        store.recluster({(a, 0): 9}, names={42: "Nobody"})
    assert cluster_map(store) == before
    assert store.known_names() == ["Bob", "Carol"]


def test_recluster_then_people_filter_follows_new_clusters(clustered):
    from facesorter.search.types import Filters

    store, (a, b, c) = clustered
    store.recluster({(c, 0): 1})  # Carol's face joins Bob's cluster
    got = store.candidate_ids(Filters(people=("bob",))).tolist()
    assert got == [a, b, c]
    assert store.candidate_ids(Filters(people=("carol",))).tolist() == []
