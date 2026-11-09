"""Store: path canonicalisation, date precision, browse, folding, robustness, regressions."""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from facesorter.search.errors import IndexNotFoundError, SearchError
from facesorter.search.store import FaceRow, Store, canonical_path, fold
from facesorter.search.types import DateRange, Filters, PhotoMeta, PlaceFilter, PlaceLabel

DIM = 8


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def rec(path, taken=None, source="exif", precision="day", place=None, seed=0, **kw):
    when = datetime.fromisoformat(taken) if taken else None
    m = PhotoMeta(
        path=path,
        size=1,
        mtime_ns=1,
        taken=when,
        date_source=source,
        date_precision=precision,
        **kw,
    )
    return (m, place, unit(seed))


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "i.db", model_name="m", dim=DIM) as s:
        yield s


def rng_filter(a: date, b: date | None = None) -> Filters:
    return Filters(date_range=DateRange(a, b or a))


# --------------------------------------------------------------------------- regressions


def test_duplicate_new_path_in_one_batch_does_not_corrupt_loaded_matrix(store):
    store.upsert_photos([rec("/first.jpg", seed=1)])
    store.top_k(unit(9), None, 5)  # matrix is now loaded
    ids = store.upsert_photos([rec("/dup.jpg", seed=2), rec("/dup.jpg", seed=3)])
    assert ids[0] == ids[1]
    hits, candidates = store.top_k(unit(3), None, 10)
    assert candidates == store.count() == 2
    assert len({h[0] for h in hits}) == len(hits) == 2
    assert hits[0][0] == ids[0] and hits[0][1] == pytest.approx(1.0, abs=1e-3)  # last record won


def test_tie_break_after_delete_uses_ascending_id(store):
    v = unit(4)
    ids = store.upsert_photos([rec(f"/p{i}", seed=4) for i in range(3)])
    store.top_k(v, None, 1)  # load the matrix, then delete the first row (swap-delete)
    assert store.delete_paths(["/p0"]) == 1
    assert [h[0] for h in store.top_k(v, None, 2)[0]] == [ids[1], ids[2]]


# --------------------------------------------------------------------------- paths


def test_paths_are_canonicalised_on_write(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    spellings = [
        str(tmp_path / "a.jpg"),
        "a.jpg",
        Path("a.jpg"),
        str(tmp_path / "sub" / ".." / "a.jpg"),
    ]
    ids = {store.upsert_photos([rec(p, seed=1)])[0] for p in spellings}  # type: ignore[arg-type]
    assert len(ids) == 1 and store.count() == 1
    store.upsert_photos([rec("~/h.jpg", seed=2)])
    assert set(store.signatures()) == {
        os.path.abspath(tmp_path / "a.jpg"),
        os.path.abspath(tmp_path / "home" / "h.jpg"),
    }
    (photo,) = store.get_photos(list(ids))
    assert photo.meta.path == os.path.abspath(tmp_path / "a.jpg")
    assert canonical_path(Path("x/../y")) == os.path.abspath("y")


def test_non_str_path_is_a_type_error():
    with pytest.raises(TypeError, match="str or PathLike"):
        canonical_path(b"/bytes/path")  # type: ignore[arg-type]


def test_delete_and_prune_accept_pathlike(store, tmp_path):
    paths = [tmp_path / f"{n}.jpg" for n in "abc"]
    store.upsert_photos([rec(str(p), seed=i) for i, p in enumerate(paths)])
    assert store.delete_paths([paths[0]]) == 1
    assert store.prune({paths[1]}) == 1  # a Path in the keep set must still match
    assert set(store.signatures()) == {str(paths[1])}
    assert store.prune([str(paths[1])]) == 0


def test_prune_refuses_empty_collection_unless_allowed(store):
    store.upsert_photos([rec("/a.jpg", seed=1), rec("/b.jpg", seed=2)])
    for empty in (set(), [], ()):
        with pytest.raises(ValueError, match="allow_empty"):
            store.prune(empty)
    assert store.count() == 2
    assert store.prune([], allow_empty=True) == 2
    assert store.count() == 0


# --------------------------------------------------------------------------- date precision


@pytest.fixture
def mixed(store):
    ids = store.upsert_photos(
        [
            rec("/day_jul1", "2023-07-01T10:00:00", seed=1),
            rec("/day_jul31", "2023-07-31T23:59:59", seed=2),
            rec("/month_jul", "2023-07-01T00:00:00", "folder", "month", seed=3),
            rec("/month_feb", "2024-02-01T00:00:00", "folder", "month", seed=4),
            rec("/year_2023", "2023-01-01T00:00:00", "folder", "year", seed=5),
            rec("/mtime_jul", "2023-07-15T00:00:00", "mtime", "day", seed=6),
            rec("/undated", None, "none", seed=7),
        ]
    )
    names = ["day_jul1", "day_jul31", "month_jul", "month_feb", "year_2023", "mtime_jul", "undated"]
    by = dict(zip(names, ids))

    def match(flt: Filters) -> set[str]:
        got = set(store.candidate_ids(flt).tolist())
        return {n for n, i in by.items() if i in got}

    return store, by, match


def test_date_precision_round_trips(mixed):
    store, by, _ = mixed
    got = {p.meta.path: p.meta.date_precision for p in store.get_photos(list(by.values()))}
    assert got["/month_jul"] == "month" and got["/year_2023"] == "year"
    assert got["/day_jul1"] == "day"


def test_mid_month_day_matches_month_precision_but_not_other_days(mixed):
    _, _, match = mixed
    got = match(rng_filter(date(2023, 7, 15)))
    assert got == {"month_jul", "year_2023"}  # not the July 1 / July 31 day photos


def test_day_boundaries_and_month_edges(mixed):
    _, _, match = mixed
    assert match(rng_filter(date(2023, 7, 1))) == {"day_jul1", "month_jul", "year_2023"}
    assert match(rng_filter(date(2023, 7, 31))) == {"day_jul31", "month_jul", "year_2023"}
    assert match(rng_filter(date(2023, 8, 1))) == {"year_2023"}  # month_jul ended on the 31st
    assert match(rng_filter(date(2023, 6, 30))) == {"year_2023"}
    assert match(rng_filter(date(2023, 6, 30), date(2023, 7, 1))) == {
        "day_jul1",
        "month_jul",
        "year_2023",
    }


def test_range_inside_another_month_does_not_match(mixed):
    _, _, match = mixed
    assert match(rng_filter(date(2024, 3, 1), date(2024, 3, 31))) == set()
    assert match(rng_filter(date(2024, 2, 10), date(2024, 2, 12))) == {"month_feb"}
    assert match(rng_filter(date(2024, 1, 1), date(2024, 1, 31))) == set()


def test_year_precision_matches_any_range_in_its_year(mixed):
    _, _, match = mixed
    assert "year_2023" in match(rng_filter(date(2023, 1, 1)))
    assert "year_2023" in match(rng_filter(date(2023, 12, 31)))
    assert "year_2023" in match(rng_filter(date(2022, 12, 1), date(2023, 1, 1)))
    assert "year_2023" not in match(rng_filter(date(2022, 1, 1), date(2022, 12, 31)))
    assert "year_2023" not in match(rng_filter(date(2024, 1, 1), date(2024, 1, 1)))


def test_leap_february_month_span(store):
    (pid,) = store.upsert_photos([rec("/feb", "2024-02-01T00:00:00", "folder", "month")])
    assert store.candidate_ids(rng_filter(date(2024, 2, 29))).tolist() == [pid]
    assert store.candidate_ids(rng_filter(date(2024, 3, 1))).tolist() == []


def test_untrusted_sources_never_match_even_with_precision(mixed):
    _, _, match = mixed
    everything = rng_filter(date(1900, 1, 1), date(2100, 1, 1))
    assert match(everything) == {"day_jul1", "day_jul31", "month_jul", "month_feb", "year_2023"}


def test_stats_ignore_precision(mixed):
    store, _, _ = mixed
    s = store.stats()
    assert s.photos == 7 and s.by_date_source == {"exif": 2, "folder": 3, "mtime": 1, "none": 1}
    assert (s.date_min, s.date_max) == (date(2023, 1, 1), date(2024, 2, 1))


def test_bad_date_precision_rejected(store):
    bad = PhotoMeta(path="/a", size=1, mtime_ns=1, date_precision="decade")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="date_precision"):
        store.upsert_photos([(bad, None, unit(0))])
    assert store.count() == 0


# --------------------------------------------------------------------------- browse


def test_browse_orders_and_counts(mixed):
    store, by, _ = mixed
    names = {v: k for k, v in by.items()}

    def run(order, k=20, flt=None):
        ids, n = store.browse(flt, k, order)
        return [names[i] for i in ids], n

    # taken: year 2023-01-01, jul1 10:00, month_jul 07-01 00:00, mtime_jul 07-15, jul31, feb 2024
    newest, n = run("newest")
    assert n == 7
    assert newest == [
        "month_feb",
        "day_jul31",
        "mtime_jul",
        "day_jul1",
        "month_jul",
        "year_2023",
        "undated",
    ]
    oldest, _ = run("oldest")
    assert oldest == [
        "year_2023",
        "month_jul",
        "day_jul1",
        "mtime_jul",
        "day_jul31",
        "month_feb",
        "undated",
    ]
    assert run("id")[0] == list(by)
    assert run("newest", k=2) == (["month_feb", "day_jul31"], 7)
    # filtered, with the candidates count independent of k
    assert run("newest", k=1, flt=rng_filter(date(2023, 7, 15))) == (["month_jul"], 2)


def test_browse_ties_break_by_id_and_no_match(store):
    ids = store.upsert_photos(
        [rec(f"/t{i}", "2023-05-05T05:05:05", seed=i) for i in range(4)] + [rec("/none", None)]
    )
    assert store.browse(None, 10, "newest")[0] == [*ids[:4], ids[4]]
    assert store.browse(None, 10, "oldest")[0] == [*ids[:4], ids[4]]
    assert store.browse(rng_filter(date(1999, 1, 1)), 5) == ([], 0)


def test_browse_with_near_filter_orders_in_python(store):
    ids = store.upsert_photos(
        [
            rec("/a", "2021-01-01T00:00:00", lat=10.0, lon=10.0, seed=1),
            rec("/b", "2023-01-01T00:00:00", lat=10.01, lon=10.0, seed=2),
            rec("/c", None, "none", lat=10.0, lon=10.01, seed=3),
            rec("/far", "2024-01-01T00:00:00", lat=50.0, lon=50.0, seed=4),
            rec("/nogps", "2024-06-01T00:00:00", seed=5),
        ]
    )
    flt = Filters(place=PlaceFilter(near=(10.0, 10.0, 5.0)))
    assert store.browse(flt, 10, "newest") == ([ids[1], ids[0], ids[2]], 3)
    assert store.browse(flt, 10, "oldest") == ([ids[0], ids[1], ids[2]], 3)
    assert store.browse(flt, 2, "id") == ([ids[0], ids[1]], 3)
    nothing = Filters(place=PlaceFilter(near=(-60.0, 0.0, 1.0)))
    assert store.browse(nothing, 5) == ([], 0)


def test_browse_validation_and_empty_store(store):
    assert store.browse(None, 5) == ([], 0)
    with pytest.raises(ValueError, match="k must"):
        store.browse(None, 0)
    with pytest.raises(ValueError, match="order"):
        store.browse(None, 5, "random")


# --------------------------------------------------------------------------- folding


@pytest.mark.parametrize(
    ("typed", "stored"),
    [
        ("Tromso", "Tromsø"),
        ("ALESUND", "Ålesund"),
        ("aeroskobing", "Ærøskøbing"),
        ("Strasse", "Straße"),
        ("Lodz", "Łódź"),
        ("Dakar", "Đakar"),
        ("istanbul", "İstanbul"),
        ("Reykjavik thorshofn", "Reykjavik Þórshöfn"),
        ("coeur", "Cœur"),
        ("  sao   paulo ", "São Paulo"),
    ],
)
def test_fold_maps_letters_without_decomposition(typed, stored):
    assert fold(typed) == fold(stored)


def test_place_and_people_use_the_stronger_fold(store):
    (pid,) = store.upsert_photos([rec("/n", place=PlaceLabel("Tromsø", "Troms", "Norway"))])
    for typed in ("Tromso", "TROMSØ", "tromsø"):
        assert store.candidate_ids(Filters(place=PlaceFilter(city=typed))).tolist() == [pid]
    store.set_faces(pid, [FaceRow(1, 0, 0, 5, 5)])
    store.name_cluster(1, "Bjørn")
    assert store.candidate_ids(Filters(people=("bjorn",))).tolist() == [pid]
    assert store.known_names() == ["Bjørn"]


# --------------------------------------------------------------------------- robustness


def test_open_directory_is_a_clean_error(tmp_path):
    with pytest.raises(SearchError, match="directory"):
        Store.open(tmp_path, model_name="m", dim=DIM)
    with pytest.raises(SearchError, match="directory"):
        Store.open(tmp_path, create=False)


def test_failed_create_leaves_no_file(tmp_path):
    path = tmp_path / "new.db"
    with pytest.raises(ValueError, match="required"):
        Store.open(path)
    with pytest.raises(ValueError, match="positive"):
        Store.open(path, model_name="m", dim=0)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_failed_open_keeps_preexisting_empty_file(tmp_path):
    path = tmp_path / "empty.db"
    path.write_bytes(b"")
    with pytest.raises(IndexNotFoundError):
        Store.open(path, create=False)
    assert path.exists()


def test_cannot_open_unreachable_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises((SearchError, OSError)):
        Store.open(blocker / "child" / "i.db", model_name="m", dim=DIM)


class _FailingCommit:
    """Connection proxy whose first COMMIT fails (sqlite3.Connection cannot be patched)."""

    def __init__(self, db):
        self._db = db
        self.failed = False

    def execute(self, sql, *args):
        if sql == "COMMIT" and not self.failed:
            self.failed = True
            raise sqlite3.OperationalError("disk I/O error")
        return self._db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._db, name)


def test_failed_commit_invalidates_the_matrix(store):
    store.upsert_photos([rec("/a.jpg", seed=1)])
    store.top_k(unit(1), None, 1)  # loaded
    store._db = proxy = _FailingCommit(store._db)
    with pytest.raises(sqlite3.OperationalError):
        store.upsert_photos([rec("/b.jpg", seed=2)])
    assert proxy.failed and store._loaded is False
    hits, n = store.top_k(unit(2), None, 5)  # reloaded from the (rolled back) database
    assert n == 1 and store.count() == 1
    assert store.upsert_photos([rec("/b.jpg", seed=2)])  # and the store still works
    assert store.top_k(unit(2), None, 5)[1] == 2


def test_empty_place_filter_is_a_no_op(store):
    store.upsert_photos([rec("/a", place=PlaceLabel("X")), rec("/b", seed=1)])
    assert len(store.candidate_ids(Filters(place=PlaceFilter()))) == 2
    assert len(store.candidate_ids(Filters(place=PlaceFilter(city="  ")))) == 2


def test_path_uniqueness_is_case_sensitive(store):
    a, b = store.upsert_photos([rec("/Case.jpg", seed=1), rec("/case.jpg", seed=2)])
    assert a != b and store.count() == 2
