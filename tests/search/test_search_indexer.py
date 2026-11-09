"""Tests for facesorter.search.indexer: the incremental indexing pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facesorter.errors import InvalidPathError
from facesorter.search import indexer as indexer_module
from facesorter.search import testing
from facesorter.search.encoders import register_model, unregister_model
from facesorter.search.errors import (
    IndexMismatchError,
    ModelUnavailableError,
    SearchError,
)
from facesorter.search.indexer import Indexer, IndexReport, open_indexer
from facesorter.search.store import Store
from facesorter.search.testing import FakeImageEncoder, FakeTextEncoder, make_color_image
from facesorter.search.types import DateRange, Filters, PlaceFilter, PlaceLabel

PARIS = (48.8566, 2.3522)
KEY_WEST = (24.5551, -81.7800)
PARIS_LABEL = PlaceLabel("Paris", "Île-de-France", "France", "FR")
KEY_WEST_LABEL = PlaceLabel("Key West", "Florida", "United States", "US")


# ----------------------------------------------------------------------------- helpers


def _dms(value: float) -> tuple[float, float, float]:
    value = abs(value)
    deg = int(value)
    minutes = int((value - deg) * 60)
    return float(deg), float(minutes), round((value - deg - minutes / 60) * 3600, 4)


def exif_for(taken: str | None = None, gps: tuple[float, float] | None = None) -> Image.Exif:
    ex = Image.Exif()
    if taken:
        ex.get_ifd(0x8769)[0x9003] = taken  # DateTimeOriginal
    if gps:
        lat, lon = gps
        ex.get_ifd(0x8825).update(
            {
                1: "N" if lat >= 0 else "S",
                2: _dms(lat),
                3: "E" if lon >= 0 else "W",
                4: _dms(lon),
            }
        )
    return ex


def make_photo(
    path: Path,
    color: str = "red",
    taken: str | None = None,
    gps: tuple[float, float] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if taken or gps:
        kwargs["exif"] = exif_for(taken, gps)
    make_color_image(color).save(path, **kwargs)
    return path


class FakeGazetteer:
    """Labels coordinates near Paris / Key West; counts calls."""

    def __init__(self) -> None:
        self.many_calls: list[int] = []

    def label_for(self, lat: float, lon: float) -> PlaceLabel | None:
        if abs(lat - PARIS[0]) < 1 and abs(lon - PARIS[1]) < 1:
            return PARIS_LABEL
        if abs(lat - KEY_WEST[0]) < 1 and abs(lon - KEY_WEST[1]) < 1:
            return KEY_WEST_LABEL
        return None

    def label_many(self, lats, lons):
        self.many_calls.append(len(lats))
        return [self.label_for(a, b) for a, b in zip(lats, lons)]

    def resolve(self, name: str) -> PlaceFilter | None:  # pragma: no cover - protocol filler
        return None


class LabelForOnlyGazetteer:
    """A gazetteer without the vectorised ``label_many`` (only the protocol methods)."""

    def label_for(self, lat, lon):
        return PARIS_LABEL

    def resolve(self, name):  # pragma: no cover - protocol filler
        return None


class CountingEncoder(FakeImageEncoder):
    """Counts the images it embeds; optionally raises on a chosen call."""

    def __init__(self, fail_on_call: int | None = None, error: BaseException | None = None):
        self.calls = 0
        self.images = 0
        self.fail_on_call = fail_on_call
        self.error = error or KeyboardInterrupt()

    def encode_images(self, images):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise self.error
        self.images += len(images)
        return super().encode_images(images)


@pytest.fixture
def fake_model():
    register_model(
        testing.FAKE_MODEL_NAME,
        lambda device: FakeImageEncoder(),
        lambda quantization: FakeTextEncoder(),
        dim=testing.FAKE_DIM,
    )
    yield testing.FAKE_MODEL_NAME
    unregister_model(testing.FAKE_MODEL_NAME)


@pytest.fixture
def lib(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    root.mkdir()
    return root


@pytest.fixture
def store(tmp_path: Path):
    s = Store.open(tmp_path / "idx" / "index.sqlite", model_name="fake-color-16", dim=16)
    yield s
    s.close()


def make_indexer(store: Store, gazetteer="default", encoder=None, **kw) -> Indexer:
    gaz = FakeGazetteer() if gazetteer == "default" else gazetteer
    return Indexer(store, encoder or FakeImageEncoder(), gaz, **kw)


def populate(lib: Path) -> dict[str, Path]:
    """A small library in year/month/place folders with EXIF dates and GPS."""
    return {
        "paris": make_photo(
            lib / "2023" / "07 - July" / "Paris" / "eiffel.jpg",
            "red",
            taken="2023:07:14 10:00:00",
            gps=PARIS,
        ),
        "keywest": make_photo(
            lib / "2023" / "08 - August" / "Key West" / "beach.jpg",
            "blue",
            taken="2023:08:02 15:30:00",
            gps=KEY_WEST,
        ),
        "folder": make_photo(lib / "2022" / "03 - March" / "tree.jpg", "green"),
        "png": make_photo(lib / "2021" / "snow.png", "white"),
    }


def paths_in(store: Store) -> set[str]:
    return set(store.signatures())


def top_color(store: Store, color: str) -> str:
    vec = FakeTextEncoder().encode_text([color])[0]
    ids, _ = store.top_k(vec, None, 1)
    return store.get_photos([ids[0][0]])[0].meta.path


def tree_digest(root: Path) -> dict[str, tuple[str, int]]:
    return {
        str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ----------------------------------------------------------------------------- basics


class TestFirstRun:
    def test_indexes_everything_with_metadata_and_places(self, store, lib):
        photos = populate(lib)
        indexer = make_indexer(store)
        report = indexer.run(lib)

        assert isinstance(report, IndexReport)
        assert (report.scanned, report.new, report.updated) == (4, 4, 0)
        assert (report.unchanged, report.removed, report.failed) == (0, 0, ())
        assert report.model_name == "fake-color-16"
        assert report.seconds >= 0
        assert paths_in(store) == {str(p) for p in photos.values()}

        by_path = {p.meta.path: p for p in store.get_photos(range(1, 5))}
        paris = by_path[str(photos["paris"])]
        assert paris.meta.taken == datetime(2023, 7, 14, 10)
        assert paris.meta.date_source == "exif"
        assert paris.place == PARIS_LABEL
        assert by_path[str(photos["keywest"])].place == KEY_WEST_LABEL
        folder = by_path[str(photos["folder"])]
        assert (folder.meta.date_source, folder.meta.date_precision) == ("folder", "month")
        assert folder.meta.taken == datetime(2022, 3, 1)
        assert folder.place is None  # no GPS, no label

    def test_embeddings_are_the_encoders(self, store, lib):
        photos = populate(lib)
        make_indexer(store).run(lib)
        assert top_color(store, "red") == str(photos["paris"])
        assert top_color(store, "blue") == str(photos["keywest"])
        assert top_color(store, "green") == str(photos["folder"])

    def test_filters_use_the_stored_labels_and_dates(self, store, lib):
        photos = populate(lib)
        make_indexer(store).run(lib)
        in_paris = store.candidate_ids(Filters(place=PlaceFilter(city="Paris")))
        assert [p.meta.path for p in store.get_photos(in_paris)] == [str(photos["paris"])]
        summer = Filters(date_range=DateRange(date(2023, 6, 1), date(2023, 8, 31)))
        assert len(store.candidate_ids(summer)) == 2

    def test_records_roots_and_time(self, store, lib):
        populate(lib)
        make_indexer(store).run(lib)
        assert json.loads(store.get_meta("library_roots")) == [str(lib)]
        stamp = datetime.fromisoformat(store.get_meta("indexed_at"))
        assert stamp.tzinfo is not None
        assert abs((datetime.now().astimezone() - stamp).total_seconds()) < 60

    def test_roots_are_merged_and_deduplicated(self, store, tmp_path, lib):
        other = tmp_path / "other"
        make_photo(other / "a.jpg")
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        indexer.run(other)
        indexer.run(lib)
        assert json.loads(store.get_meta("library_roots")) == [str(lib), str(other)]

    def test_bad_stored_roots_are_replaced(self, store, lib):
        populate(lib)
        store.set_meta("library_roots", "{not json")
        make_indexer(store).run(lib)
        assert json.loads(store.get_meta("library_roots")) == [str(lib)]
        store.set_meta("library_roots", json.dumps({"a": 1}))
        make_indexer(store).run(lib)
        assert json.loads(store.get_meta("library_roots")) == [str(lib)]
        store.set_meta("library_roots", json.dumps(["/x", 5, None]))
        make_indexer(store).run(lib)
        assert json.loads(store.get_meta("library_roots")) == ["/x", str(lib)]

    def test_library_files_are_never_modified(self, store, lib):
        populate(lib)
        before = tree_digest(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        indexer.run(lib, force=True)
        (lib / "gone.jpg").write_bytes(b"x")
        (lib / "2021" / "snow.png").unlink()
        indexer.run(lib)
        after = tree_digest(lib)
        assert {k: v for k, v in after.items() if not k.endswith("gone.jpg")} == {
            k: v for k, v in before.items() if not k.endswith("snow.png")
        }

    def test_missing_root_raises(self, store, tmp_path):
        with pytest.raises(InvalidPathError):
            make_indexer(store).run(tmp_path / "nope")

    def test_empty_library_on_empty_index_is_fine(self, store, lib):
        report = make_indexer(store).run(lib)
        assert (report.scanned, report.new, report.removed) == (0, 0, 0)

    def test_accepts_str_root_and_pathlike(self, store, lib):
        populate(lib)
        assert make_indexer(store).run(str(lib)).new == 4


# ----------------------------------------------------------------------------- incremental


class TestIncremental:
    def test_rerun_skips_unchanged_without_touching_the_files(self, store, lib, monkeypatch):
        populate(lib)
        encoder = CountingEncoder()
        indexer = make_indexer(store, encoder=encoder)
        indexer.run(lib)
        assert encoder.images == 4

        def boom(*a, **k):
            raise AssertionError("an unchanged library must not be opened")

        monkeypatch.setattr(indexer_module, "read_metadata", boom)
        monkeypatch.setattr(indexer_module, "iter_decoded", boom)
        report = indexer.run(lib)
        assert (report.scanned, report.unchanged, report.new, report.updated) == (4, 4, 0, 0)
        assert encoder.images == 4

    def test_modified_file_is_re_embedded(self, store, lib):
        photos = populate(lib)
        encoder = CountingEncoder()
        indexer = make_indexer(store, encoder=encoder)
        indexer.run(lib)
        assert top_color(store, "green") == str(photos["folder"])
        photo_id = store.get_photos([3])[0].id

        target = photos["paris"]
        old_ns = target.stat().st_mtime_ns
        make_photo(target, "green", taken="2023:07:14 10:00:00", gps=PARIS)
        os.utime(target, ns=(old_ns + 5_000_000_000, old_ns + 5_000_000_000))
        report = indexer.run(lib)

        assert (report.updated, report.new, report.unchanged) == (1, 0, 3)
        assert encoder.images == 5
        assert store.count() == 4
        assert store.get_photos([photo_id])[0].id == photo_id  # ids are kept
        vec = FakeTextEncoder().encode_text(["green"])[0]
        top2 = {store.get_photos([i])[0].meta.path for i, _ in store.top_k(vec, None, 2)[0]}
        assert top2 == {str(target), str(photos["folder"])}

    def test_deleted_file_is_pruned_and_new_file_added(self, store, lib):
        photos = populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        photos["png"].unlink()
        added = make_photo(lib / "2024" / "new.jpg", "orange", taken="2024:01:02 03:04:05")
        report = indexer.run(lib)
        assert (report.scanned, report.new, report.removed, report.unchanged) == (4, 1, 1, 3)
        assert str(added) in paths_in(store)
        assert str(photos["png"]) not in paths_in(store)
        assert store.count() == 4

    def test_prune_leaves_other_roots_alone(self, store, tmp_path, lib):
        other = tmp_path / "other"
        kept = make_photo(other / "keep.jpg")
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        indexer.run(other)
        (lib / "2021" / "snow.png").unlink()
        report = indexer.run(lib)
        assert report.removed == 1
        assert str(kept) in paths_in(store)

    def test_prune_ignores_a_sibling_folder_with_a_common_prefix(self, store, tmp_path, lib):
        sibling = tmp_path / "library-backup"
        kept = make_photo(sibling / "keep.jpg")
        make_photo(lib / "a.jpg")
        indexer = make_indexer(store)
        indexer.run(sibling)
        indexer.run(lib)
        assert str(kept) in paths_in(store)

    def test_force_redoes_everything(self, store, lib):
        populate(lib)
        encoder = CountingEncoder()
        indexer = make_indexer(store, encoder=encoder)
        indexer.run(lib)
        report = indexer.run(lib, force=True)
        assert (report.updated, report.new, report.unchanged) == (4, 0, 0)
        assert encoder.images == 8
        assert store.count() == 4

    def test_limit_processes_a_prefix_and_never_prunes(self, store, lib):
        photos = populate(lib)
        indexer = make_indexer(store)
        report = indexer.run(lib, limit=2)
        assert (report.scanned, report.new) == (4, 2)
        assert store.count() == 2
        assert report.removed == 0

        # a later full run finishes the rest
        report = indexer.run(lib)
        assert (report.new, report.unchanged) == (2, 2)

        # limit runs do not prune either
        photos["png"].unlink()
        report = indexer.run(lib, limit=1)
        assert report.removed == 0 and str(photos["png"]) in paths_in(store)
        assert indexer.run(lib).removed == 1

    @pytest.mark.parametrize("limit", [0, -3])
    def test_bad_limit(self, store, lib, limit):
        with pytest.raises(ValueError, match="limit"):
            make_indexer(store).run(lib, limit=limit)


class TestEmptyScanProtection:
    def test_refuses_to_wipe_the_index_when_nothing_is_found(self, store, lib):
        photos = populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        for p in photos.values():
            p.unlink()
        with pytest.raises(SearchError, match="allow_empty_prune") as info:
            indexer.run(lib)
        assert "--allow-empty-prune" in str(info.value)
        assert store.count() == 4

    def test_unmounted_drive_is_refused_not_wiped(self, store, tmp_path, lib):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        # the folder is still there but empty, like a mount point
        for p in sorted(lib.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        with pytest.raises(SearchError):
            indexer.run(lib, limit=1)
        assert store.count() == 4

    def test_opt_in_removes_everything_from_that_root(self, store, lib):
        photos = populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        for p in photos.values():
            p.unlink()
        report = indexer.run(lib, allow_empty_prune=True)
        assert (report.scanned, report.removed) == (0, 4)
        assert store.count() == 0

    def test_an_empty_new_root_does_not_touch_other_roots(self, store, tmp_path, lib):
        populate(lib)
        empty = tmp_path / "empty"
        empty.mkdir()
        indexer = make_indexer(store)
        indexer.run(lib)
        report = indexer.run(empty)
        assert (report.scanned, report.removed) == (0, 0)
        assert store.count() == 4


# ----------------------------------------------------------------------------- failures


class TestFailures:
    def test_unreadable_file_is_reported_and_others_are_indexed(self, store, lib):
        populate(lib)
        bad = lib / "2023" / "broken.jpg"
        bad.write_bytes(b"this is not a jpeg")
        report = make_indexer(store).run(lib)
        assert (report.scanned, report.new) == (5, 4)
        assert [p for p, _ in report.failed] == [str(bad)]
        assert str(bad) in report.failed[0][1]
        assert str(bad) not in paths_in(store)

    def test_decode_failures_are_reported_and_retried(self, store, lib):
        populate(lib)
        rng = np.random.default_rng(1)
        noisy = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))
        good = lib / "noisy.jpg"
        noisy.save(good, quality=95)
        truncated = lib / "truncated.jpg"
        truncated.write_bytes(good.read_bytes()[:3000])
        good.unlink()

        indexer = make_indexer(store)
        report = indexer.run(lib)
        assert report.new == 4
        assert [p for p, _ in report.failed] == [str(truncated)]
        # not remembered as indexed, so the next run tries again (and fails again)
        again = indexer.run(lib)
        assert (again.unchanged, again.new) == (4, 0)
        assert [p for p, _ in again.failed] == [str(truncated)]

    def test_a_fixed_file_is_picked_up_next_run(self, store, lib):
        bad = lib / "x.jpg"
        bad.write_bytes(b"junk")
        indexer = make_indexer(store)
        assert len(indexer.run(lib).failed) == 1
        make_photo(bad, "yellow")
        report = indexer.run(lib)
        assert (report.new, report.failed) == (1, ())

    def test_file_that_vanishes_after_listing_is_reported_not_pruned(self, store, lib, monkeypatch):
        photos = populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        ghost = str(lib / "ghost.jpg")
        real = indexer_module.iter_library

        def listing(root, exclude_dirs=(), **kwargs):
            yield from real(root, exclude_dirs, **kwargs)
            yield Path(ghost)

        monkeypatch.setattr(indexer_module, "iter_library", listing)
        report = indexer.run(lib)
        assert report.unchanged == 4
        assert [p for p, _ in report.failed] == [ghost]
        assert "ghost.jpg" in report.failed[0][1]
        assert store.count() == 4 and str(photos["png"]) in paths_in(store)

    def test_encoder_failure_is_fatal_and_typed(self, store, lib):
        populate(lib)
        encoder = CountingEncoder(fail_on_call=2, error=RuntimeError("out of memory"))
        indexer = make_indexer(store, encoder=encoder, batch_size=1, chunk_size=1)
        with pytest.raises(SearchError, match="out of memory"):
            indexer.run(lib)
        assert store.count() == 1  # the finished chunk survived

    def test_encoder_returning_wrong_shape(self, store, lib):
        populate(lib)

        class Wrong(FakeImageEncoder):
            def encode_images(self, images):
                return np.zeros((len(images) + 1, 16), dtype=np.float32)

        with pytest.raises(SearchError, match="shape"):
            make_indexer(store, encoder=Wrong()).run(lib)
        assert store.count() == 0

    def test_ctrl_c_keeps_finished_work_and_the_rerun_resumes(self, store, lib):
        populate(lib)
        make_photo(lib / "2020" / "extra.jpg", "yellow")
        encoder = CountingEncoder(fail_on_call=3)  # KeyboardInterrupt on the 3rd batch
        indexer = make_indexer(store, encoder=encoder, batch_size=1, chunk_size=2)
        with pytest.raises(KeyboardInterrupt):
            indexer.run(lib)
        assert store.count() == 2  # one full chunk written

        resumed = make_indexer(store).run(lib)
        assert (resumed.unchanged, resumed.new, resumed.failed) == (2, 3, ())
        assert store.count() == 5

    def test_ctrl_c_flushes_the_partly_filled_chunk(self, store, lib):
        populate(lib)
        encoder = CountingEncoder(fail_on_call=4)
        indexer = make_indexer(store, encoder=encoder, batch_size=1, chunk_size=100)
        with pytest.raises(KeyboardInterrupt):
            indexer.run(lib)
        assert store.count() == 3
        assert make_indexer(store).run(lib).new == 1

    def test_gazetteer_errors_are_not_swallowed(self, store, lib):
        populate(lib)

        class Broken(FakeGazetteer):
            def label_many(self, lats, lons):
                raise RuntimeError("gazetteer exploded")

        with pytest.raises(RuntimeError, match="exploded"):
            make_indexer(store, gazetteer=Broken()).run(lib)
        assert store.count() == 0


# ----------------------------------------------------------------------------- scanning


class TestLibraryScan:
    def test_videos_hidden_and_other_files_are_skipped(self, store, lib):
        make_photo(lib / "ok.jpg")
        make_photo(lib / "UPPER.JPEG", "blue")
        (lib / "clip.mov").write_bytes(b"\x00")
        (lib / "clip.mp4").write_bytes(b"\x00")
        (lib / "notes.txt").write_text("hi")
        make_photo(lib / ".hidden.jpg")
        make_photo(lib / "._apple.jpg")
        make_photo(lib / ".cache" / "inside.jpg")
        report = make_indexer(store).run(lib)
        assert report.scanned == 2
        assert {Path(p).name for p in paths_in(store)} == {"ok.jpg", "UPPER.JPEG"}

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlinked_root_keeps_the_symlink_path(self, store, tmp_path, lib):
        populate(lib)
        link = tmp_path / "link"
        link.symlink_to(lib, target_is_directory=True)
        make_indexer(store).run(link)
        assert all(p.startswith(str(link) + os.sep) for p in paths_in(store))
        assert json.loads(store.get_meta("library_roots")) == [str(link)]
        # the same photos through the real path are different canonical paths: a second
        # index of the real folder does not crash and does not touch the link entries
        report = make_indexer(store).run(link)
        assert report.unchanged == 4

    def test_relative_root_is_made_absolute(self, store, tmp_path, lib, monkeypatch):
        populate(lib)
        monkeypatch.chdir(tmp_path)
        make_indexer(store).run("library")
        assert all(os.path.isabs(p) for p in paths_in(store))
        assert paths_in(store) == {str(p) for p in lib.rglob("*.*") if p.is_file()}
        assert make_indexer(store).run("./library/../library").unchanged == 4

    def test_tilde_root_is_expanded(self, store, tmp_path, lib, monkeypatch):
        populate(lib)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        make_indexer(store).run("~/library")
        assert paths_in(store) == {str(p) for p in lib.rglob("*.*") if p.is_file()}
        assert json.loads(store.get_meta("library_roots")) == [str(lib)]

    def test_trailing_separator_root(self, store, lib):
        populate(lib)
        assert make_indexer(store).run(str(lib) + os.sep).new == 4
        assert json.loads(store.get_meta("library_roots")) == [str(lib)]

    def test_index_folder_inside_the_library_is_skipped(self, tmp_path):
        root = tmp_path / "lib"
        make_photo(root / "a.jpg")
        make_photo(root / "indexdir" / "thumb.jpg", "blue")  # lives next to the index
        make_photo(root / "indexdir" / "deeper" / "t2.jpg", "blue")
        with Store.open(
            root / "indexdir" / "index.sqlite", model_name="fake-color-16", dim=16
        ) as s:
            report = make_indexer(s).run(root)
            assert report.scanned == 1
            assert {Path(p).name for p in paths_in(s)} == {"a.jpg"}

    def test_index_file_directly_in_the_root_does_not_hide_the_library(self, tmp_path):
        root = tmp_path / "lib"
        make_photo(root / "a.jpg")
        make_photo(root / "sub" / "b.jpg")
        with Store.open(root / "index.sqlite", model_name="fake-color-16", dim=16) as s:
            assert make_indexer(s).run(root).new == 2

    def test_exclude_dirs(self, store, lib):
        populate(lib)
        report = make_indexer(store).run(lib, exclude_dirs=[lib / "2023", str(lib / "2021")])
        assert report.scanned == 1
        assert {Path(p).name for p in paths_in(store)} == {"tree.jpg"}

    def test_excluding_a_folder_prunes_what_was_indexed_there(self, store, lib):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        report = indexer.run(lib, exclude_dirs=[lib / "2023"])
        assert (report.removed, report.unchanged) == (2, 2)


# ----------------------------------------------------------------------------- places


class TestPlaces:
    def test_no_gazetteer_means_no_labels(self, store, lib):
        populate(lib)
        make_indexer(store, gazetteer=None).run(lib)
        stats = store.stats()
        assert stats.with_gps == 2 and stats.with_place == 0

    def test_labels_are_looked_up_in_one_vectorised_call(self, store, lib):
        populate(lib)
        gaz = FakeGazetteer()
        make_indexer(store, gazetteer=gaz).run(lib)
        assert gaz.many_calls == [2]  # only the two photos with GPS

    def test_gazetteer_without_label_many_falls_back_to_label_for(self, store, lib):
        populate(lib)
        make_indexer(store, gazetteer=LabelForOnlyGazetteer()).run(lib)
        assert store.stats().with_place == 2

    def test_no_gps_no_lookup(self, store, lib):
        make_photo(lib / "plain.jpg")
        gaz = FakeGazetteer()
        make_indexer(store, gazetteer=gaz).run(lib)
        assert gaz.many_calls == []


# ----------------------------------------------------------------------------- progress


class TestProgress:
    def test_stages_and_counts(self, store, lib):
        populate(lib)
        events: list[tuple[int, int, str]] = []
        make_indexer(store, batch_size=3, chunk_size=3).run(
            lib, progress=lambda d, t, s: events.append((d, t, s))
        )
        stages = [s for _, _, s in events]
        assert set(stages) == {"scan", "metadata", "encode", "store"}
        # scan, then metadata, then encode/store interleave (chunks are written as they fill)
        order = {"scan": 0, "metadata": 1, "encode": 2, "store": 2}
        assert [order[s] for s in stages] == sorted(order[s] for s in stages)
        assert events[0] == (4, 4, "scan")
        assert [e for e in events if e[2] == "metadata"] == [
            (i, 4, "metadata") for i in range(1, 5)
        ]
        assert [e for e in events if e[2] == "encode"] == [(3, 4, "encode"), (4, 4, "encode")]
        assert [e for e in events if e[2] == "store"] == [(3, 4, "store"), (4, 4, "store")]

    def test_scan_progress_while_counting(self, store, lib, monkeypatch):
        monkeypatch.setattr(indexer_module, "_SCAN_PROGRESS_EVERY", 2)
        for i in range(5):
            make_photo(lib / f"{i}.jpg")
        events: list[tuple[int, int, str]] = []
        make_indexer(store).run(lib, progress=lambda d, t, s: events.append((d, t, s)))
        assert [e for e in events if e[2] == "scan"] == [
            (2, 0, "scan"),
            (4, 0, "scan"),
            (5, 5, "scan"),
        ]

    def test_an_unchanged_rerun_reports_only_the_scan(self, store, lib):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        events: list[tuple[int, int, str]] = []
        indexer.run(lib, progress=lambda d, t, s: events.append((d, t, s)))
        assert events == [(4, 4, "scan")]

    def test_progress_counts_failures_in_metadata(self, store, lib):
        make_photo(lib / "a.jpg")
        (lib / "b.jpg").write_bytes(b"junk")
        events: list[tuple[int, int, str]] = []
        make_indexer(store).run(lib, progress=lambda d, t, s: events.append((d, t, s)))
        assert (2, 2, "metadata") in events
        assert (1, 2, "encode") in events and (1, 2, "store") in events


# ----------------------------------------------------------------------------- construction


class TestIndexerConstruction:
    @pytest.mark.parametrize("kw", [{"workers": 0}, {"batch_size": 0}, {"chunk_size": 0}])
    def test_rejects_non_positive_sizes(self, store, kw):
        with pytest.raises(ValueError):
            make_indexer(store, **kw)

    def test_rejects_an_encoder_that_does_not_match_the_index(self, store):
        class Other(FakeImageEncoder):
            name = "something-else"

        with pytest.raises(IndexMismatchError, match="something-else"):
            make_indexer(store, encoder=Other())

        class WrongDim(FakeImageEncoder):
            dim = 8

        with pytest.raises(IndexMismatchError):
            make_indexer(store, encoder=WrongDim())

    def test_context_manager_closes_the_store(self, tmp_path):
        s = Store.open(tmp_path / "i.sqlite", model_name="fake-color-16", dim=16)
        with make_indexer(s) as indexer:
            assert indexer.store is s
        with pytest.raises(SearchError, match="closed"):
            s.count()
        indexer.close()  # twice is fine

    def test_small_batches_and_single_worker_give_the_same_index(self, tmp_path, lib):
        populate(lib)
        results = []
        for n, (workers, batch, chunk) in enumerate([(1, 1, 1), (4, 32, 256), (2, 3, 2)]):
            with Store.open(tmp_path / f"{n}.sqlite", model_name="fake-color-16", dim=16) as s:
                make_indexer(s, workers=workers, batch_size=batch, chunk_size=chunk).run(lib)
                vec = FakeTextEncoder().encode_text(["blue"])[0]
                ids, _ = s.top_k(vec, None, 4)
                results.append([(s.get_photos([i])[0].meta.path, score) for i, score in ids])
        assert results[0] == results[1] == results[2]


# ----------------------------------------------------------------------------- open_indexer


class TestOpenIndexer:
    @pytest.fixture(autouse=True)
    def _cheap_gazetteer(self, monkeypatch):
        from facesorter.search import geo

        self.gaz = FakeGazetteer()
        monkeypatch.setattr(geo, "build_gazetteer", lambda *a, **k: self.gaz)

    def test_creates_the_index_and_its_parent_folder(self, fake_model, tmp_path, lib):
        populate(lib)
        path = tmp_path / "does" / "not" / "exist" / "index.sqlite"
        with open_indexer(path, model=fake_model, workers=2, batch_size=2) as indexer:
            assert indexer.gazetteer is self.gaz
            assert (indexer.workers, indexer.batch_size) == (2, 2)
            assert indexer.store.model_name == "fake-color-16" and indexer.store.dim == 16
            report = indexer.run(lib)
        assert report.new == 4 and path.exists()
        with Store.open(path, create=False) as s:
            assert s.model_name == "fake-color-16" and s.count() == 4

    def test_reopening_continues_incrementally(self, fake_model, tmp_path, lib):
        populate(lib)
        path = tmp_path / "index.sqlite"
        with open_indexer(path, model=fake_model) as indexer:
            indexer.run(lib)
        with open_indexer(path, model=fake_model) as indexer:
            assert indexer.run(lib).unchanged == 4

    def test_expands_tilde(self, fake_model, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        with open_indexer("~/.facesorter/index.sqlite", model=fake_model):
            pass
        assert (tmp_path / ".facesorter" / "index.sqlite").exists()

    def test_other_model_than_the_index_is_a_mismatch(self, fake_model, tmp_path):
        path = tmp_path / "index.sqlite"
        open_indexer(path, model=fake_model).close()
        register_model(
            "fake-other",
            lambda device: _NamedEncoder("fake-other"),
            lambda quantization: FakeTextEncoder(),
            dim=16,
        )
        try:
            with pytest.raises(IndexMismatchError):
                open_indexer(path, model="fake-other")
        finally:
            unregister_model("fake-other")

    def test_unknown_model_is_unavailable_and_creates_nothing(self, tmp_path):
        path = tmp_path / "idx" / "index.sqlite"
        with pytest.raises(ModelUnavailableError):
            open_indexer(path, model="no-such-model")
        assert not path.exists() and not path.parent.exists()

    def test_gazetteer_failure_closes_the_store(self, fake_model, tmp_path, monkeypatch):
        from facesorter.search import geo

        def boom(*a, **k):
            raise ModelUnavailableError("no place data")

        monkeypatch.setattr(geo, "build_gazetteer", boom)
        with pytest.raises(ModelUnavailableError, match="place data"):
            open_indexer(tmp_path / "index.sqlite", model=fake_model)
        # the file can be opened (no lock left behind) and holds an empty index
        with Store.open(tmp_path / "index.sqlite", create=False) as s:
            assert s.count() == 0

    def test_indexer_construction_failure_closes_the_store(self, fake_model, tmp_path):
        with pytest.raises(ValueError):
            open_indexer(tmp_path / "index.sqlite", model=fake_model, workers=0)
        with Store.open(tmp_path / "index.sqlite", create=False) as s:
            assert s.count() == 0


class _NamedEncoder(FakeImageEncoder):
    def __init__(self, name: str) -> None:
        self.name = name


# ----------------------------------------------------------------------------- signatures


class TestChangeDetection:
    def _setup(self, store, lib):
        photos = populate(lib)
        encoder = CountingEncoder()
        indexer = make_indexer(store, encoder=encoder)
        indexer.run(lib)
        return photos, encoder, indexer

    def test_same_size_new_mtime_is_an_update(self, store, lib):
        photos, encoder, indexer = self._setup(store, lib)
        target = photos["folder"]
        make_color_image("green").save(target, comment=b"aaaa")
        indexer.run(lib)  # index the commented version
        stat = target.stat()
        make_color_image("green").save(target, comment=b"bbbb")  # valid, same size, new bytes
        assert target.stat().st_size == stat.st_size
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 7_000_000_000))
        report = indexer.run(lib)
        assert (report.updated, report.unchanged, report.failed) == (1, 3, ())

    def test_new_size_same_mtime_is_an_update(self, store, lib):
        photos, encoder, indexer = self._setup(store, lib)
        target = photos["folder"]
        stat = target.stat()
        target.write_bytes(target.read_bytes() + b"\x00")  # trailing byte: still a valid JPEG
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert target.stat().st_mtime_ns == stat.st_mtime_ns
        report = indexer.run(lib)
        assert (report.updated, report.unchanged) == (1, 3)

    def test_identical_size_and_mtime_is_unchanged(self, store, lib):
        photos, encoder, indexer = self._setup(store, lib)
        target = photos["folder"]
        stat = target.stat()
        target.write_bytes(target.read_bytes())
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert indexer.run(lib).unchanged == 4


# ----------------------------------------------------------------------------- unreadable dirs


class TestSkippedDirectories:
    def _locked_library(self, lib):
        for name in "abc":
            make_photo(lib / "keep" / f"{name}.jpg", "red")
        for name in "wxyz":
            make_photo(lib / "locked" / f"{name}.jpg", "blue")

    def test_photos_of_an_unlistable_folder_are_not_pruned(self, store, lib, monkeypatch, caplog):
        self._locked_library(lib)
        indexer = make_indexer(store)
        assert indexer.run(lib).new == 7

        locked = str(lib / "locked")
        real = indexer_module.iter_library

        def listing(root, exclude_dirs=(), *, on_skip=None):
            for path in real(root, exclude_dirs, on_skip=on_skip):
                if "locked" not in path.parts:
                    yield path
            on_skip(Path(locked), PermissionError(13, "Permission denied"))

        monkeypatch.setattr(indexer_module, "iter_library", listing)
        with caplog.at_level("WARNING"):
            report = indexer.run(lib)
        assert report.skipped_dirs == (locked,)
        assert (report.scanned, report.removed, report.unchanged) == (3, 0, 3)
        assert store.count() == 7
        assert "Permission denied" in caplog.text

    def test_only_the_skipped_folder_is_protected(self, store, lib, monkeypatch):
        self._locked_library(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        (lib / "keep" / "a.jpg").unlink()
        real = indexer_module.iter_library

        def listing(root, exclude_dirs=(), *, on_skip=None):
            for path in real(root, exclude_dirs, on_skip=on_skip):
                if "locked" not in path.parts:
                    yield path
            on_skip(lib / "locked", PermissionError(13, "nope"))

        monkeypatch.setattr(indexer_module, "iter_library", listing)
        report = indexer.run(lib)
        assert report.removed == 1  # keep/a.jpg is gone for real
        assert store.count() == 6

    def test_a_sibling_with_a_common_prefix_is_still_pruned(self, store, lib, monkeypatch):
        make_photo(lib / "locked" / "a.jpg")
        make_photo(lib / "locked2" / "b.jpg")
        make_photo(lib / "other" / "c.jpg")
        indexer = make_indexer(store)
        indexer.run(lib)
        (lib / "locked2" / "b.jpg").unlink()
        real = indexer_module.iter_library

        def listing(root, exclude_dirs=(), *, on_skip=None):
            yield from (
                p for p in real(root, exclude_dirs, on_skip=on_skip) if "locked" not in p.parts
            )
            on_skip(lib / "locked", OSError("io"))

        monkeypatch.setattr(indexer_module, "iter_library", listing)
        assert indexer.run(lib).removed == 1

    def test_an_unlistable_root_is_refused_not_wiped(self, store, lib, monkeypatch):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)

        def listing(root, exclude_dirs=(), *, on_skip=None):
            on_skip(Path(root), PermissionError(13, "nope"))
            return iter(())

        monkeypatch.setattr(indexer_module, "iter_library", listing)
        with pytest.raises(SearchError):
            indexer.run(lib)
        report = indexer.run(lib, allow_empty_prune=True)
        assert report.skipped_dirs == (str(lib),) and report.removed == 0
        assert store.count() == 4

    @pytest.mark.skipif(
        sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="needs POSIX permissions and a non-root user",
    )
    def test_chmod_zero_folder(self, store, lib):
        self._locked_library(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        locked = lib / "locked"
        locked.chmod(0)
        try:
            report = indexer.run(lib)
        finally:
            locked.chmod(0o755)
        assert report.skipped_dirs == (str(locked),)
        assert (report.removed, report.scanned) == (0, 3)
        assert store.count() == 7
        # permissions back: the next run sees everything and nothing is reported
        again = indexer.run(lib)
        assert again.skipped_dirs == () and again.unchanged == 7

    def test_clean_scan_reports_no_skipped_dirs(self, store, lib):
        populate(lib)
        assert make_indexer(store).run(lib).skipped_dirs == ()


# ----------------------------------------------------------------------------- root spelling


def _case_insensitive(path: Path) -> bool:
    return os.path.exists(str(path).swapcase()) and path.name != path.name.swapcase()


class TestRootSpelling:
    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlinked_root_reuses_the_stored_spelling(self, store, tmp_path, lib):
        populate(lib)
        link = tmp_path / "link"
        link.symlink_to(lib, target_is_directory=True)
        indexer = make_indexer(store)
        indexer.run(lib)
        report = indexer.run(link)
        assert (report.new, report.unchanged, report.removed) == (0, 4, 0)
        assert store.count() == 4
        assert all(p.startswith(str(lib) + os.sep) for p in paths_in(store))
        assert json.loads(store.get_meta("library_roots")) == [str(lib)]
        # and a modification seen through the link updates the existing rows
        os.utime(lib / "2021" / "snow.png", ns=(1, 1))
        assert indexer.run(link).updated == 1 and store.count() == 4

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_link_first_then_real_path_keeps_the_link_spelling(self, store, tmp_path, lib):
        populate(lib)
        link = tmp_path / "link"
        link.symlink_to(lib, target_is_directory=True)
        indexer = make_indexer(store)
        indexer.run(link)
        assert indexer.run(lib).unchanged == 4
        assert store.count() == 4
        assert json.loads(store.get_meta("library_roots")) == [str(link)]

    def test_case_variant_root(self, store, tmp_path):
        root = tmp_path / "Photos"
        make_photo(root / "a.jpg")
        make_photo(root / "b.jpg", "blue")
        variant = tmp_path / "pHOTOS"
        if not variant.exists():
            pytest.skip("case-sensitive filesystem")
        indexer = make_indexer(store)
        indexer.run(root)
        report = indexer.run(variant)
        assert (report.new, report.unchanged, report.removed) == (0, 2, 0)
        assert store.count() == 2
        assert json.loads(store.get_meta("library_roots")) == [str(root)]

    def test_unrelated_roots_are_not_merged(self, store, tmp_path, lib):
        other = tmp_path / "other"
        make_photo(other / "a.jpg")
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        indexer.run(other)
        assert len(json.loads(store.get_meta("library_roots"))) == 2

    def test_a_folder_inside_a_stored_root_is_its_own_root(self, store, lib):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        report = indexer.run(lib / "2023")
        assert (report.scanned, report.unchanged, report.removed) == (2, 2, 0)
        assert json.loads(store.get_meta("library_roots")) == [str(lib), str(lib / "2023")]


# ----------------------------------------------------------------------------- remove_root


class TestRemoveRoot:
    def test_removes_entries_and_the_root(self, store, tmp_path, lib):
        other = tmp_path / "other"
        kept = make_photo(other / "keep.jpg")
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        indexer.run(other)
        before = tree_digest(lib)
        assert indexer.remove_root(lib) == 4
        assert paths_in(store) == {str(kept)}
        assert json.loads(store.get_meta("library_roots")) == [str(other)]
        assert tree_digest(lib) == before  # library files are untouched
        assert indexer.remove_root(lib) == 0  # idempotent

    def test_works_for_a_folder_that_no_longer_exists(self, store, tmp_path, lib):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        moved = tmp_path / "library-renamed"
        lib.rename(moved)
        # the ghosts: indexing the new name leaves the old entries behind
        assert indexer.run(moved).new == 4
        assert store.count() == 8
        assert indexer.remove_root(lib) == 4
        assert store.count() == 4
        assert json.loads(store.get_meta("library_roots")) == [str(moved)]

    def test_matches_another_spelling_of_the_stored_root(self, store, tmp_path, lib):
        if sys.platform == "win32":  # pragma: no cover
            pytest.skip("symlinks need privileges on Windows")
        populate(lib)
        link = tmp_path / "link"
        link.symlink_to(lib, target_is_directory=True)
        indexer = make_indexer(store)
        indexer.run(lib)
        assert indexer.remove_root(link) == 4
        assert json.loads(store.get_meta("library_roots")) == []

    def test_unknown_root_and_tilde_and_relative(self, store, tmp_path, lib, monkeypatch):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        assert indexer.remove_root(tmp_path / "never-indexed") == 0
        assert store.count() == 4
        monkeypatch.chdir(tmp_path)
        assert indexer.remove_root("library") == 4

    def test_a_sibling_with_a_common_prefix_survives(self, store, tmp_path, lib):
        sibling = tmp_path / "library-backup"
        kept = make_photo(sibling / "keep.jpg")
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        indexer.run(sibling)
        indexer.remove_root(lib)
        assert paths_in(store) == {str(kept)}

    def test_reindexing_after_removal_starts_fresh(self, store, lib):
        populate(lib)
        indexer = make_indexer(store)
        indexer.run(lib)
        indexer.remove_root(lib)
        assert indexer.run(lib).new == 4
