"""Tests for facesorter.search.store: persistence, filters, exact search, faces, feedback."""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date, datetime

import numpy as np
import pytest

from facesorter.search.errors import IndexMismatchError, IndexNotFoundError, SearchError
from facesorter.search.store import (
    DuplicateNameError,
    FaceRow,
    Store,
    fold,
    haversine_km,
)
from facesorter.search.types import (
    DateRange,
    Filters,
    PeopleDirectory,
    PhotoMeta,
    PlaceFilter,
    PlaceLabel,
)

DIM = 8
MODEL = "fake-model"


# --------------------------------------------------------------------------- helpers


def unit(rng: np.random.Generator, dim: int = DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def meta(path: str, taken: str | None = None, source: str | None = None, **kw) -> PhotoMeta:
    when = datetime.fromisoformat(taken) if taken else None
    return PhotoMeta(
        path=path,
        size=kw.pop("size", 100),
        mtime_ns=kw.pop("mtime_ns", 5),
        taken=when,
        date_source=source or ("exif" if when else "none"),  # type: ignore[arg-type]
        **kw,
    )


def open_store(tmp_path, **kw) -> Store:
    return Store.open(tmp_path / "index.db", model_name=MODEL, dim=DIM, **kw)


def stored(vec: np.ndarray) -> np.ndarray:
    """What the store actually keeps: normalised, then rounded through float16."""
    v = (vec / np.linalg.norm(vec)).astype(np.float16).astype(np.float32)
    return v


def brute_force(vecs: dict[int, np.ndarray], query: np.ndarray, allowed, k: int):
    q = query / np.linalg.norm(query)
    rows = [(i, round(float(stored(v) @ q), 6)) for i, v in vecs.items() if i in allowed]
    rows.sort(key=lambda t: (-t[1], t[0]))
    return rows[:k]


@pytest.fixture
def library(tmp_path):
    """A small library: 6 photos across places/dates with known faces."""
    rng = np.random.default_rng(1)
    store = open_store(tmp_path)
    keys = PlaceLabel("Key West", "Florida", "United States", "US")
    miami = PlaceLabel("Miami", "Florida", "United States", "US")
    paris = PlaceLabel("Paris", "Île-de-France", "France", "FR")
    sp = PlaceLabel("São Paulo", "São Paulo", "Brazil", "BR")
    specs = [
        ("/p/a.jpg", "2023-06-10T10:00:00", "exif", 24.5551, -81.7800, keys),
        ("/p/b.jpg", "2023-06-20T23:59:59", "exif", 25.7617, -80.1918, miami),
        ("/p/c.jpg", "2023-07-01T00:00:00", "folder", 48.8566, 2.3522, paris),
        ("/p/d.jpg", "2023-06-15T12:00:00", "mtime", None, None, None),
        ("/p/e.jpg", None, "none", -23.5505, -46.6333, sp),
        ("/p/f.jpg", "2020-01-01T08:00:00", "exif", None, None, None),
    ]
    vecs = {}
    records = []
    for path, taken, source, lat, lon, place in specs:
        v = unit(rng)
        records.append((meta(path, taken, source, lat=lat, lon=lon), place, v))
    ids = store.upsert_photos(records)
    by_name = {s[0][3]: i for s, i in zip(specs, ids)}  # "a" -> id
    for i, (_, _, v) in zip(ids, records):
        vecs[i] = v
    yield store, by_name, vecs
    store.close()


def make_cluster(store, ids_by_cluster: dict[int, list[int]], name_map: dict[int, str]):
    for cluster, photo_ids in ids_by_cluster.items():
        for pid in photo_ids:
            existing = store.get_faces(pid)
            store.set_faces(pid, [*existing, FaceRow(cluster, 0, 0, 10, 10)])
    for cluster, name in name_map.items():
        store.name_cluster(cluster, name)


# --------------------------------------------------------------------------- round trips


def test_round_trip_and_self_search(tmp_path):
    rng = np.random.default_rng(0)
    v = unit(rng)
    m = PhotoMeta(
        path="/x/y.heic",
        size=123,
        mtime_ns=1_700_000_000_123_456_789,
        width=4000,
        height=3000,
        taken=datetime(2022, 3, 4, 5, 6, 7),
        date_source="exif",
        lat=10.5,
        lon=-20.25,
        camera="Pixel 9",
    )
    place = PlaceLabel("Key West", "Florida", "United States", "US")
    with open_store(tmp_path) as store:
        (pid,) = store.upsert_photos([(m, place, v)])
        (photo,) = store.get_photos([pid])
        assert photo.id == pid and photo.meta == m and photo.place == place
        hits, n = store.top_k(v, None, 3)
        assert n == 1 and hits[0][0] == pid and hits[0][1] == pytest.approx(1.0, abs=1e-3)
        assert (store.model_name, store.dim) == (MODEL, DIM)


def test_photo_without_place_or_date(tmp_path):
    with open_store(tmp_path) as store:
        (pid,) = store.upsert_photos([(meta("/a.jpg"), None, unit(np.random.default_rng(0)))])
        (photo,) = store.get_photos([pid])
        assert photo.place is None and photo.meta.taken is None and photo.meta.lat is None


def test_nonfinite_gps_is_stored_as_missing(tmp_path):
    with open_store(tmp_path) as store:
        m = meta("/a.jpg", lat=float("nan"), lon=float("inf"))
        (pid,) = store.upsert_photos([(m, None, unit(np.random.default_rng(0)))])
        assert store.get_photos([pid])[0].meta.lat is None
        assert store.get_photos([pid])[0].meta.lon is None


def test_get_photos_order_missing_and_repeats(library):
    store, ids, _ = library
    got = store.get_photos([ids["c"], 9999, ids["a"], ids["c"]])
    assert [p.id for p in got] == [ids["c"], ids["a"], ids["c"]]
    assert store.get_photos([]) == []


def test_empty_upsert_and_empty_store_search(tmp_path):
    with open_store(tmp_path) as store:
        assert store.upsert_photos([]) == []
        assert store.top_k(unit(np.random.default_rng(0)), None, 5) == ([], 0)
        assert store.count() == 0
        assert store.candidate_ids(None).tolist() == []


# --------------------------------------------------------------------------- upsert semantics


def test_upsert_is_idempotent_and_keeps_ids(tmp_path):
    rng = np.random.default_rng(2)
    records = [(meta(f"/p/{i}.jpg"), None, unit(rng)) for i in range(5)]
    with open_store(tmp_path) as store:
        first = store.upsert_photos(records)
        before = store.top_k(records[0][2], None, 5)
        second = store.upsert_photos(records)
        assert first == second
        assert store.count() == 5
        assert store.top_k(records[0][2], None, 5) == before


def test_update_replaces_metadata_and_vector_in_place(tmp_path):
    rng = np.random.default_rng(3)
    old, new = unit(rng), unit(rng)
    with open_store(tmp_path) as store:
        other = store.upsert_photos([(meta("/o.jpg"), None, unit(rng))])[0]
        (pid,) = store.upsert_photos([(meta("/a.jpg", size=1), None, old)])
        store.top_k(old, None, 1)  # load the matrix so the update is applied incrementally
        again = store.upsert_photos(
            [(meta("/a.jpg", "2024-01-02T03:04:05", size=2), PlaceLabel("Rome"), new)]
        )
        assert again == [pid] and pid != other
        photo = store.get_photos([pid])[0]
        assert photo.meta.size == 2 and photo.place == PlaceLabel("Rome")
        assert store.top_k(new, None, 1)[0][0][0] == pid
        assert store.top_k(new, None, 1)[0][0][1] == pytest.approx(1.0, abs=1e-3)
        assert store.signatures()["/a.jpg"] == (2, 5)


def test_later_duplicate_in_batch_wins(tmp_path):
    rng = np.random.default_rng(4)
    with open_store(tmp_path) as store:
        ids = store.upsert_photos(
            [(meta("/a.jpg", size=1), None, unit(rng)), (meta("/a.jpg", size=9), None, unit(rng))]
        )
        assert ids[0] == ids[1] and store.count() == 1
        assert store.signatures() == {"/a.jpg": (9, 5)}


def test_faces_dropped_only_when_file_changed(tmp_path):
    rng = np.random.default_rng(5)
    with open_store(tmp_path) as store:
        (pid,) = store.upsert_photos([(meta("/a.jpg"), None, unit(rng))])
        store.set_faces(pid, [FaceRow(1, 0, 0, 5, 5)])
        store.upsert_photos([(meta("/a.jpg", camera="new"), PlaceLabel("Rome"), unit(rng))])
        assert len(store.get_faces(pid)) == 1  # same size/mtime: faces still describe it
        store.upsert_photos([(meta("/a.jpg", mtime_ns=77), None, unit(rng))])
        assert store.get_faces(pid) == []


def test_ids_are_never_reused(tmp_path):
    rng = np.random.default_rng(6)
    with open_store(tmp_path) as store:
        (a,) = store.upsert_photos([(meta("/a.jpg"), None, unit(rng))])
        store.delete_paths(["/a.jpg"])
        (b,) = store.upsert_photos([(meta("/b.jpg"), None, unit(rng))])
        assert b > a


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(DIM, dtype=np.float32),
        np.ones(DIM + 1, dtype=np.float32),
        np.ones((2, DIM), dtype=np.float32),
        np.ones(DIM, dtype=np.int32),
        np.array([1.0] * (DIM - 1) + [np.nan], dtype=np.float32),
        np.array([1.0] * (DIM - 1) + [np.inf], dtype=np.float32),
    ],
    ids=["zero", "too-long", "2d", "int-dtype", "nan", "inf"],
)
def test_bad_embeddings_rejected_without_partial_write(tmp_path, bad):
    good = unit(np.random.default_rng(0))
    with open_store(tmp_path) as store:
        with pytest.raises(ValueError, match="record 1"):
            store.upsert_photos([(meta("/ok.jpg"), None, good), (meta("/bad.jpg"), None, bad)])
        assert store.count() == 0


def test_bad_date_source_rejected(tmp_path):
    bad = PhotoMeta(path="/a", size=1, mtime_ns=1, date_source="bogus")  # type: ignore[arg-type]
    with open_store(tmp_path) as store:
        with pytest.raises(ValueError, match="date_source"):
            store.upsert_photos([(bad, None, unit(np.random.default_rng(0)))])


def test_unnormalised_and_row_vector_input_is_accepted(tmp_path):
    v = np.random.default_rng(0).standard_normal(DIM) * 50  # float64, not unit length
    with open_store(tmp_path) as store:
        (pid,) = store.upsert_photos([(meta("/a.jpg"), None, v)])
        hits, _ = store.top_k((v * 3).reshape(1, -1), None, 1)
        assert hits[0][0] == pid and hits[0][1] == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- delete / prune


def test_delete_and_prune_keep_search_exact(tmp_path):
    rng = np.random.default_rng(7)
    paths = [f"/p/{i}.jpg" for i in range(40)]
    recs = [(meta(p), None, unit(rng)) for p in paths]
    q = unit(rng)
    with open_store(tmp_path) as store:
        ids = store.upsert_photos(recs)
        vecs = {i: r[2] for i, r in zip(ids, recs)}
        store.top_k(q, None, 1)  # load the matrix so removals are applied incrementally
        assert store.delete_paths(["/p/0.jpg", "/p/5.jpg", "/p/39.jpg", "/nope.jpg"]) == 3
        keep = set(paths[:30]) - {"/p/0.jpg", "/p/5.jpg", "/p/7.jpg"}
        assert store.prune(keep) == 40 - 3 - len(keep)
        alive = {i for i, p in zip(ids, paths) if p in keep}
        assert store.count() == len(alive)
        hits, n = store.top_k(q, None, 100)
        assert n == len(alive) and hits == brute_force(vecs, q, alive, 100)
        # appending after removals still works and the new row is findable
        new_vec = unit(rng)
        (new_id,) = store.upsert_photos([(meta("/p/new.jpg"), None, new_vec)])
        hits, n = store.top_k(new_vec, None, 1)
        assert hits[0][0] == new_id and n == len(alive) + 1


def test_prune_returns_removed_count_and_cascades(library):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"], ids["b"]]}, {1: "Bob"})
    store.record_feedback("q", ids["a"], True)
    assert store.prune({"/p/b.jpg", "/p/c.jpg"}) == 4
    assert store.count() == 2
    assert store.feedback_rows() == []
    assert store.get_faces(ids["a"]) == []
    assert store.clusters()[0].face_count == 1


def test_delete_paths_empty_and_unloaded(tmp_path):
    with open_store(tmp_path) as store:
        assert store.delete_paths([]) == 0
        store.upsert_photos([(meta("/a.jpg"), None, unit(np.random.default_rng(0)))])
        assert store.delete_paths(["/a.jpg", "/a.jpg"]) == 1  # matrix never loaded
        assert store.top_k(unit(np.random.default_rng(1)), None, 3) == ([], 0)


def test_signatures(library):
    store, _, _ = library
    sigs = store.signatures()
    assert sigs["/p/a.jpg"] == (100, 5) and len(sigs) == 6


# --------------------------------------------------------------------------- open / mismatch


def test_open_missing_without_create(tmp_path):
    with pytest.raises(IndexNotFoundError, match="facesorter index"):
        Store.open(tmp_path / "nope.db", model_name=MODEL, dim=DIM, create=False)
    assert not (tmp_path / "nope.db").exists()


def test_empty_file_without_create_is_not_found(tmp_path):
    (tmp_path / "e.db").write_bytes(b"")
    with pytest.raises(IndexNotFoundError):
        Store.open(tmp_path / "e.db", create=False)


def test_create_makes_parent_dirs_and_requires_model(tmp_path):
    nested = tmp_path / "a" / "b" / "index.db"
    with pytest.raises(ValueError, match="required"):
        Store.open(nested)
    Store.open(nested, model_name=MODEL, dim=DIM).close()
    assert nested.exists()
    with pytest.raises(ValueError, match="positive"):
        Store.open(tmp_path / "z.db", model_name=MODEL, dim=0)


def test_adopts_model_and_dim_from_existing_index(tmp_path):
    open_store(tmp_path).close()
    with Store.open(tmp_path / "index.db", create=False) as store:
        assert (store.model_name, store.dim) == (MODEL, DIM)


def test_model_mismatch_says_reindex(tmp_path):
    open_store(tmp_path).close()
    with pytest.raises(IndexMismatchError, match="Re-index"):
        Store.open(tmp_path / "index.db", model_name="other", dim=DIM)
    with pytest.raises(IndexMismatchError, match="Re-index"):
        Store.open(tmp_path / "index.db", model_name=MODEL, dim=DIM + 1)


def test_schema_version_mismatch(tmp_path):
    open_store(tmp_path).close()
    raw = sqlite3.connect(tmp_path / "index.db")
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(IndexMismatchError, match="schema version 99"):
        open_store(tmp_path)


def test_foreign_sqlite_database_is_refused_and_untouched(tmp_path):
    path = tmp_path / "other.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE notes(x)")
    raw.commit()
    raw.close()
    with pytest.raises(IndexMismatchError, match="not a facesorter"):
        Store.open(path, model_name=MODEL, dim=DIM)
    raw = sqlite3.connect(path)
    tables = [r[0] for r in raw.execute("SELECT name FROM sqlite_master")]
    raw.close()
    assert tables == ["notes"]


def test_garbage_file_is_refused(tmp_path):
    path = tmp_path / "junk.db"
    path.write_bytes(b"this is not a database at all " * 200)
    with pytest.raises(IndexMismatchError, match="not a readable facesorter"):
        Store.open(path, model_name=MODEL, dim=DIM)
    assert path.read_bytes().startswith(b"this is not")


def test_damaged_meta_is_reported(tmp_path):
    open_store(tmp_path).close()
    raw = sqlite3.connect(tmp_path / "index.db")
    raw.execute("DELETE FROM meta WHERE key = 'dim'")
    raw.commit()
    raw.close()
    with pytest.raises(IndexMismatchError, match="damaged"):
        open_store(tmp_path)


def test_wrong_size_embedding_blob_is_reported(tmp_path):
    with open_store(tmp_path) as store:
        store.upsert_photos([(meta("/a.jpg"), None, unit(np.random.default_rng(0)))])
    raw = sqlite3.connect(tmp_path / "index.db")
    raw.execute("UPDATE embeddings SET vec = x'0102'")
    raw.commit()
    raw.close()
    with open_store(tmp_path) as store:
        with pytest.raises(IndexMismatchError, match="wrong size"):
            store.top_k(unit(np.random.default_rng(1)), None, 1)


def test_closed_store_raises_and_close_is_idempotent(tmp_path):
    store = open_store(tmp_path)
    store.close()
    store.close()
    with pytest.raises(SearchError, match="closed"):
        store.count()
    assert store.path == tmp_path / "index.db"


# --------------------------------------------------------------------------- date filter


def test_date_range_is_inclusive_on_both_ends(library):
    store, ids, _ = library
    june = DateRange(date(2023, 6, 10), date(2023, 6, 20))
    got = store.candidate_ids(Filters(date_range=june))
    assert set(got.tolist()) == {ids["a"], ids["b"]}  # 10th 10:00 and 20th 23:59:59 both in
    one_day = DateRange(date(2023, 7, 1), date(2023, 7, 1))
    assert store.candidate_ids(Filters(date_range=one_day)).tolist() == [ids["c"]]  # midnight
    just_before = DateRange(date(2023, 6, 1), date(2023, 6, 9))
    assert store.candidate_ids(Filters(date_range=just_before)).tolist() == []


def test_date_filter_excludes_mtime_and_undated(library):
    store, ids, _ = library
    everything = DateRange(date(1900, 1, 1), date(2100, 1, 1))
    got = set(store.candidate_ids(Filters(date_range=everything)).tolist())
    assert got == {ids["a"], ids["b"], ids["c"], ids["f"]}
    assert ids["d"] not in got and ids["e"] not in got  # mtime-derived and undated
    # ... but they are all searchable without a date filter
    assert len(store.candidate_ids(None)) == 6


# --------------------------------------------------------------------------- place filter


def test_place_levels_are_case_and_accent_insensitive(library):
    store, ids, _ = library
    f = lambda **kw: set(store.candidate_ids(Filters(place=PlaceFilter(**kw))).tolist())  # noqa: E731
    assert f(city="KEY west") == {ids["a"]}
    assert f(city="sao paulo") == {ids["e"]}
    assert f(city="  São   Paulo ") == {ids["e"]}
    assert f(region="ile-de-france") == {ids["c"]}
    assert f(region="florida") == {ids["a"], ids["b"]}
    assert f(country="united states") == {ids["a"], ids["b"]}
    assert f(country="brazil", city="Miami") == set()  # levels are ANDed
    assert f(city="Atlantis") == set()
    assert f(text="nothing structured") == set(range(1, 0)) | set(ids.values())  # no-op filter


def test_fold():
    assert fold("  Île-de-France ") == "ile-de-france"
    assert fold("STRASSE") == fold("straße")


def test_near_radius_boundary_is_inclusive(library):
    store, ids, _ = library
    d = float(haversine_km(24.5551, -81.7800, 25.7617, -80.1918))
    centre = (24.5551, -81.7800)

    def near(radius):
        flt = Filters(place=PlaceFilter(near=(*centre, radius)))
        return set(store.candidate_ids(flt).tolist())

    assert near(d) == {ids["a"], ids["b"]}  # exactly at the boundary
    assert near(d * (1 - 1e-6)) == {ids["a"]}  # just inside the boundary: b is out
    assert near(0.0) == {ids["a"]}  # zero radius still matches the exact point
    assert near(20_000) == {ids["a"], ids["b"], ids["c"], ids["e"]}  # whole planet; GPS-less out


def test_near_combined_with_named_level(library):
    store, ids, _ = library
    place = PlaceFilter(country="France", near=(25.0, -80.0, 500))
    assert store.candidate_ids(Filters(place=place)).tolist() == []
    place = PlaceFilter(country="United States", near=(25.0, -80.0, 120))
    assert set(store.candidate_ids(Filters(place=place)).tolist()) == {ids["b"]}


@pytest.mark.parametrize(
    ("centre", "radius", "expected"),
    [
        ((0.0, 179.9), 50, {"east", "west"}),  # box crosses +180
        ((0.0, -179.9), 50, {"east", "west"}),  # box crosses -180
        ((89.9, 0.0), 100, {"pole_a", "pole_b"}),  # reaches the pole: any longitude
        ((0.0, 540.0 - 0.1), 50, {"east", "west"}),  # longitude outside [-180, 180) wraps
    ],
)
def test_near_wraps_antimeridian_and_poles(tmp_path, centre, radius, expected):
    rng = np.random.default_rng(0)
    spots = {
        "east": (0.0, 179.95),
        "west": (0.0, -179.95),
        "far": (0.0, 100.0),
        "pole_a": (89.95, 10.0),
        "pole_b": (89.95, -170.0),
    }
    with open_store(tmp_path) as store:
        ids = store.upsert_photos(
            [(meta(f"/{n}", lat=la, lon=lo), None, unit(rng)) for n, (la, lo) in spots.items()]
        )
        by_id = dict(zip(ids, spots))
        got = store.candidate_ids(Filters(place=PlaceFilter(near=(*centre, radius))))
        assert {by_id[i] for i in got.tolist()} == expected


def test_near_validation(library):
    store, _, _ = library
    for bad in [(95.0, 0.0, 10.0), (0.0, 0.0, -1.0), (float("nan"), 0.0, 5.0)]:
        with pytest.raises(ValueError, match="near"):
            store.candidate_ids(Filters(place=PlaceFilter(near=bad)))


# --------------------------------------------------------------------------- people filter


def test_people_filter_requires_every_named_person(library):
    store, ids, _ = library
    make_cluster(
        store,
        {1: [ids["a"], ids["b"]], 2: [ids["b"], ids["c"]], 3: [ids["a"], ids["f"]]},
        {1: "Bob", 2: "Alice"},  # cluster 3 stays unnamed
    )
    cand = lambda *names: set(store.candidate_ids(Filters(people=names)).tolist())  # noqa: E731
    assert cand("bob") == {ids["a"], ids["b"]}
    assert cand("Bob", "ALICE") == {ids["b"]}
    assert cand("alice", "bob", "Bob") == {ids["b"]}  # repeated names are harmless
    assert cand("Carol") == set()
    assert cand("Bob", "Carol") == set()


def test_people_names_fold_accents(library):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"]]}, {1: "Zoë"})
    assert store.candidate_ids(Filters(people=("ZOE",))).tolist() == [ids["a"]]
    assert store.candidate_ids(Filters(people=("zoë",))).tolist() == [ids["a"]]


def test_all_filters_combined(library):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"], ids["b"], ids["c"], ids["f"]]}, {1: "Bob"})
    flt = Filters(
        date_range=DateRange(date(2023, 6, 1), date(2023, 6, 30)),
        place=PlaceFilter(region="Florida"),
        people=("Bob",),
    )
    assert set(store.candidate_ids(flt).tolist()) == {ids["a"], ids["b"]}
    flt2 = Filters(
        date_range=DateRange(date(2023, 6, 1), date(2023, 6, 15)),
        place=PlaceFilter(region="Florida"),
        people=("Bob",),
    )
    assert store.candidate_ids(flt2).tolist() == [ids["a"]]


# --------------------------------------------------------------------------- search


def test_top_k_matches_brute_force_with_and_without_filters(library):
    store, ids, vecs = library
    rng = np.random.default_rng(11)
    for _ in range(5):
        q = unit(rng)
        hits, n = store.top_k(q, None, 4)
        assert n == 6 and hits == brute_force(vecs, q, set(vecs), 4)
        flt = Filters(place=PlaceFilter(country="United States"))
        hits, n = store.top_k(q, flt, 4)
        assert n == 2 and hits == brute_force(vecs, q, {ids["a"], ids["b"]}, 4)
        hits, n = store.top_k(q, Filters(), 2)  # empty filters == no filters
        assert n == 6 and hits == brute_force(vecs, q, set(vecs), 2)


def test_k_larger_than_pool_and_no_candidates(library):
    store, ids, _ = library
    q = unit(np.random.default_rng(0))
    hits, n = store.top_k(q, Filters(place=PlaceFilter(country="Brazil")), 50)
    assert n == 1 and [h[0] for h in hits] == [ids["e"]]
    assert store.top_k(q, Filters(place=PlaceFilter(country="Narnia")), 5) == ([], 0)
    assert store.top_k(q, None, 100)[1] == 6


def test_query_validation(library):
    store, _, _ = library
    for bad in [np.zeros(DIM), np.ones(DIM + 2), np.full(DIM, np.nan), np.ones(DIM, dtype=int)]:
        with pytest.raises(ValueError, match="query vector"):
            store.top_k(bad, None, 3)
    with pytest.raises(ValueError, match="k must"):
        store.top_k(np.ones(DIM, dtype=np.float32), None, 0)


def test_tied_scores_break_by_ascending_id(tmp_path):
    v = unit(np.random.default_rng(0))
    other = unit(np.random.default_rng(1))
    with open_store(tmp_path) as store:
        ids = store.upsert_photos(
            [(meta(f"/p/{i}.jpg"), None, v if i % 2 else other) for i in range(12)]
        )
        tied = sorted(i for i, n in zip(ids, range(12)) if n % 2)
        for k in (1, 3, 6, 20):
            hits, _ = store.top_k(v, None, k)
            assert [h[0] for h in hits][: min(k, 6)] == tied[: min(k, 6)]
        # k cuts through the tie group: always the lowest ids, in order
        assert [h[0] for h in store.top_k(v, None, 4)[0]] == tied[:4]


@pytest.mark.parametrize("threshold", [0.0, 0.2, 1.0])
def test_gather_and_scan_plans_agree(tmp_path, threshold):
    rng = np.random.default_rng(21)
    n = 400
    vecs = {}
    with open_store(tmp_path, gather_threshold=threshold) as store:
        recs = []
        for i in range(n):
            taken = f"2023-{1 + i % 12:02d}-{1 + i % 28:02d}T10:00:00"
            recs.append(
                (meta(f"/p/{i}.jpg", taken, lat=float(i % 50), lon=float(i % 7)), None, unit(rng))
            )
        ids = store.upsert_photos(recs)
        vecs = {i: r[2] for i, r in zip(ids, recs)}
        q = unit(rng)
        for months in [(1, 1), (3, 5), (1, 12)]:
            flt = Filters(date_range=DateRange(date(2023, months[0], 1), date(2023, months[1], 28)))
            allowed = set(store.candidate_ids(flt).tolist())
            hits, count = store.top_k(q, flt, 15)
            assert count == len(allowed)
            assert hits == brute_force(vecs, q, allowed, 15)
        assert store._plan(10, 100) == ("gather" if threshold >= 0.1 else "scan")
        assert store._plan(100, 100) == ("gather" if threshold >= 1.0 else "scan")


def test_gather_blocks_match_single_block(tmp_path, monkeypatch):
    import facesorter.search.store as mod

    monkeypatch.setattr(mod, "_GATHER_BLOCK", 7)
    rng = np.random.default_rng(5)
    with open_store(tmp_path, gather_threshold=1.0) as store:
        recs = [(meta(f"/p/{i}.jpg", "2023-01-01T00:00:00"), None, unit(rng)) for i in range(50)]
        ids = store.upsert_photos(recs)
        q = unit(rng)
        flt = Filters(date_range=DateRange(date(2023, 1, 1), date(2023, 1, 1)))
        hits, n = store.top_k(q, flt, 10)
        assert n == 50
        assert hits == brute_force({i: r[2] for i, r in zip(ids, recs)}, q, set(ids), 10)


def test_search_after_incremental_growth_beyond_capacity(tmp_path):
    rng = np.random.default_rng(8)
    with open_store(tmp_path) as store:
        store.upsert_photos([(meta("/first.jpg"), None, unit(rng))])
        store.top_k(unit(rng), None, 1)  # loaded, capacity 1024
        recs = [(meta(f"/p/{i}.jpg"), None, unit(rng)) for i in range(1200)]
        ids = store.upsert_photos(recs)
        assert store.top_k(recs[1100][2], None, 1)[0][0][0] == ids[1100]
        assert store.top_k(unit(rng), None, 1)[1] == 1201


# --------------------------------------------------------------------------- persistence


def test_persistence_across_close_and_reopen(library, tmp_path):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"], ids["b"]]}, {1: "Bob"})
    store.record_feedback("beach", ids["a"], True)
    q = unit(np.random.default_rng(99))
    flt = Filters(date_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)), people=("bob",))
    before = (store.top_k(q, flt, 5), store.top_k(q, None, 6), store.stats())
    photos = store.get_photos(list(ids.values()))
    store.close()

    with Store.open(store.path, model_name=MODEL, dim=DIM, create=False) as again:
        assert again.top_k(q, flt, 5) == before[0]
        assert again.top_k(q, None, 6) == before[1]
        assert again.stats() == before[2]
        assert again.get_photos(list(ids.values())) == photos
        assert again.known_names() == ["Bob"]
        assert [r.query for r in again.feedback_rows()] == ["beach"]


def test_deleting_a_row_the_stale_matrix_never_saw(tmp_path):
    rng = np.random.default_rng(13)
    with open_store(tmp_path) as reader, open_store(tmp_path) as writer:
        writer.upsert_photos([(meta("/a.jpg"), None, unit(rng))])
        reader.top_k(unit(rng), None, 1)  # loaded: one row
        writer.upsert_photos([(meta("/b.jpg"), None, unit(rng))])
        assert reader.delete_paths(["/b.jpg"]) == 1  # not in reader's matrix yet
        assert reader.top_k(unit(rng), None, 5)[1] == 1


def test_second_connection_changes_are_picked_up(tmp_path):
    rng = np.random.default_rng(12)
    v = unit(rng)
    with open_store(tmp_path) as reader:
        with open_store(tmp_path) as writer:
            writer.upsert_photos([(meta("/a.jpg"), None, unit(rng))])
            assert reader.top_k(v, None, 5)[1] == 1  # matrix loaded now
            (new,) = writer.upsert_photos([(meta("/b.jpg"), None, v)])
            hits, n = reader.top_k(v, None, 5)  # sees the other process's write
            assert n == 2 and hits[0][0] == new
            writer.delete_paths(["/b.jpg"])
            assert reader.top_k(v, None, 5)[1] == 1


# --------------------------------------------------------------------------- stats


def test_stats(library):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"], ids["b"]]}, {1: "Bob"})
    store.record_feedback("q", ids["a"], False)
    s = store.stats()
    assert s.photos == 6 and s.with_gps == 4 and s.with_place == 4
    assert s.by_date_source == {"exif": 3, "folder": 1, "mtime": 1, "none": 1}
    assert (s.date_min, s.date_max) == (date(2020, 1, 1), date(2023, 7, 1))
    assert (s.faces, s.named_people, s.feedback) == (2, 1, 1)


def test_stats_empty(tmp_path):
    with open_store(tmp_path) as store:
        s = store.stats()
        assert s.photos == 0 and s.date_min is None and s.date_max is None
        assert s.by_date_source == {"exif": 0, "folder": 0, "mtime": 0, "none": 0}


# --------------------------------------------------------------------------- faces + people


def test_set_faces_replaces_and_get_faces(library):
    store, ids, _ = library
    store.set_faces(ids["a"], [FaceRow(1, 0, 0, 4, 4), FaceRow(None, 5, 5, 9, 9)])
    assert store.get_faces(ids["a"]) == [FaceRow(1, 0, 0, 4, 4), FaceRow(None, 5, 5, 9, 9)]
    store.set_faces(ids["a"], [FaceRow(2, 1, 1, 2, 2)])
    assert store.get_faces(ids["a"]) == [FaceRow(2, 1, 1, 2, 2)]
    store.set_faces(ids["a"], [])
    assert store.get_faces(ids["a"]) == []
    with pytest.raises(ValueError, match="unknown photo"):
        store.set_faces(9999, [])


def test_clusters_listing(library):
    store, ids, _ = library
    # cluster 1: 7 photos' worth is not available, so use 3 photos with 4 faces
    store.set_faces(ids["a"], [FaceRow(1, 0, 0, 10, 10), FaceRow(1, 0, 0, 50, 50)])
    store.set_faces(ids["b"], [FaceRow(1, 0, 0, 30, 30), FaceRow(2, 0, 0, 5, 5)])
    store.set_faces(ids["c"], [FaceRow(1, 0, 0, 20, 20), FaceRow(None, 0, 0, 1, 1)])
    store.name_cluster(2, "Alice")
    c1, c2 = store.clusters()
    assert (c1.cluster_id, c1.face_count, c1.photo_count, c1.name) == (1, 4, 3, None)
    assert c1.sample_photo_ids == (ids["a"], ids["b"], ids["c"])  # largest face first
    assert (c2.cluster_id, c2.face_count, c2.photo_count, c2.name) == (2, 1, 1, "Alice")


def test_cluster_samples_capped_at_six(tmp_path):
    rng = np.random.default_rng(0)
    with open_store(tmp_path) as store:
        ids = store.upsert_photos([(meta(f"/p/{i}"), None, unit(rng)) for i in range(10)])
        for i, pid in enumerate(ids):
            store.set_faces(pid, [FaceRow(7, 0, 0, 1 + i, 1 + i)])
        (cluster,) = store.clusters()
        assert cluster.photo_count == 10 and len(cluster.sample_photo_ids) == 6
        assert cluster.sample_photo_ids == tuple(reversed(ids))[:6]


def test_name_cluster_rules(library):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"]], 2: [ids["b"]]}, {})
    store.name_cluster(1, "  Bob   Smith ")
    assert store.known_names() == ["Bob Smith"]
    with pytest.raises(DuplicateNameError, match="already used"):
        store.name_cluster(2, "bob smith")
    assert isinstance(DuplicateNameError("x"), ValueError)
    store.name_cluster(1, "BOB SMITH")  # same cluster: re-case is fine
    assert store.known_names() == ["BOB SMITH"]
    store.name_cluster(1, "")  # empty name unnames
    assert store.known_names() == []
    store.name_cluster(2, "Bob Smith")  # name is free again
    store.name_cluster(2, None)
    assert store.clusters()[0].name is None
    with pytest.raises(ValueError, match="unknown face cluster"):
        store.name_cluster(42, "Nobody")


def test_orphaned_name_does_not_block_reuse_or_show_up(library):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"]], 2: [ids["b"]]}, {1: "Bob"})
    store.set_faces(ids["a"], [])  # cluster 1 disappears but its name row remains
    assert store.known_names() == []
    store.name_cluster(2, "Bob")  # reuse allowed because the old owner has no faces
    assert store.known_names() == ["Bob"]
    assert store.candidate_ids(Filters(people=("bob",))).tolist() == [ids["b"]]


def test_clear_faces_drops_names_unless_kept(library):
    store, ids, _ = library
    make_cluster(store, {1: [ids["a"]]}, {1: "Bob"})
    store.clear_faces(keep_names=True)
    assert store.stats().faces == 0 and store.stats().named_people == 1
    store.clear_faces()
    assert store.stats().named_people == 0 and store.clusters() == []


def test_known_names_satisfy_people_directory(library):
    store, ids, _ = library
    assert isinstance(store, PeopleDirectory)
    make_cluster(store, {1: [ids["a"]], 2: [ids["b"]]}, {1: "Émile", 2: "alice"})
    assert store.known_names() == ["alice", "Émile"]


def test_names_for_photos(library):
    store, ids, _ = library
    make_cluster(
        store, {1: [ids["a"], ids["b"]], 2: [ids["a"]], 3: [ids["c"]]}, {1: "Bob", 2: "al"}
    )
    got = store.names_for_photos([ids["a"], ids["b"], ids["c"], ids["a"], 9999])
    assert got == {ids["a"]: ["al", "Bob"], ids["b"]: ["Bob"]}


# --------------------------------------------------------------------------- feedback


def test_feedback_record_rows_and_summary(library):
    store, ids, _ = library
    assert store.feedback_summary().precision is None
    assert store.feedback_summary().mean_precision is None
    store.record_feedback(" beach ", ids["a"], True)
    store.record_feedback("beach", ids["b"], False)
    store.record_feedback("beach", ids["c"], True)
    store.record_feedback("beach", ids["b"], True)  # re-rating: the latest wins
    store.record_feedback("sunset", ids["a"], False)
    rows = store.feedback_rows()
    assert [r.relevant for r in rows] == [True, False, True, True, False]
    assert (
        rows[0].query == "beach" and rows[0].photo_id == ids["a"] and rows[0].ts.endswith("+00:00")
    )
    assert [r.id for r in rows] == sorted(r.id for r in rows)
    s = store.feedback_summary()
    by_query = {q.query: q for q in s.per_query}
    assert (by_query["beach"].rated, by_query["beach"].relevant) == (3, 3)
    assert by_query["beach"].precision == 1.0 and by_query["sunset"].precision == 0.0
    assert (s.rated, s.relevant) == (4, 3)
    assert s.precision == pytest.approx(0.75) and s.mean_precision == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unknown photo"):
        store.record_feedback("x", 9999, True)


# --------------------------------------------------------------------------- concurrency


def test_threaded_reads_while_writing(tmp_path):
    rng = np.random.default_rng(31)
    base = [(meta(f"/base/{i}.jpg", "2023-05-05T05:05:05"), None, unit(rng)) for i in range(200)]
    queries = [unit(rng) for _ in range(8)]
    errors: list[BaseException] = []
    with open_store(tmp_path) as store:
        store.upsert_photos(base)
        flt = Filters(date_range=DateRange(date(2023, 5, 1), date(2023, 5, 31)))
        stop = threading.Event()

        def reader(worker: int) -> None:
            try:
                while not stop.is_set():
                    for q in queries:
                        hits, n = store.top_k(q, flt if worker % 2 else None, 3)
                        assert n >= 200 and len(hits) == 3
                        store.get_photos([h[0] for h in hits])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for batch in range(10):  # a single writer keeps appending and re-upserting
            store.upsert_photos(
                [
                    (meta(f"/new/{batch}/{j}.jpg", "2023-05-06T00:00:00"), None, unit(rng))
                    for j in range(20)
                ]
            )
            store.upsert_photos(base[:5])
        stop.set()
        for t in threads:
            t.join(timeout=20)
        assert not errors, errors
        assert store.count() == 400


# --------------------------------------------------------------------------- latency


def test_latency_sanity_20k(tmp_path, capsys):
    n, dim = 20_000, 384
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((n, dim)).astype(np.float32)
    with Store.open(tmp_path / "big.db", model_name="synthetic", dim=dim) as store:
        t0 = time.perf_counter()
        for start in range(0, n, 5000):
            store.upsert_photos(
                [
                    (
                        meta(f"/p/{i}.jpg", f"20{10 + i % 14}-{1 + i % 12:02d}-10T00:00:00"),
                        None,
                        mat[i],
                    )
                    for i in range(start, start + 5000)
                ]
            )
        t_insert = time.perf_counter() - t0
        q = mat[123] + 0.2 * rng.standard_normal(dim).astype(np.float32)
        t0 = time.perf_counter()
        store.top_k(q, None, 24)  # includes the lazy load
        t_cold = time.perf_counter() - t0
        flt_wide = Filters(date_range=DateRange(date(2010, 1, 1), date(2019, 12, 31)))
        flt_narrow = Filters(date_range=DateRange(date(2012, 3, 1), date(2012, 3, 31)))
        timings = {}
        for label, flt in [("none", None), ("wide", flt_wide), ("narrow", flt_narrow)]:
            runs = []
            for _ in range(5):
                t0 = time.perf_counter()
                hits, cands = store.top_k(q, flt, 24)
                runs.append((time.perf_counter() - t0) * 1000)
            timings[label] = (float(np.median(runs)), cands)
        assert store.top_k(q, None, 1)[0][0][0] == 124  # ids start at 1: row 123 -> id 124
    with capsys.disabled():
        print(
            f"\n[latency 20k x {dim}] insert {t_insert:.1f}s, "
            f"cold first query {t_cold * 1000:.0f} ms, "
            + ", ".join(f"{k}: {v[0]:.1f} ms ({v[1]} cands)" for k, v in timings.items())
        )
