from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from conftest import ALICE, BOB, CRASH, GROUP, LOW_CONF, NO_FACE

from facesorter import finder
from facesorter.detection import Face
from facesorter.errors import CopyError, InvalidPathError, NoFacesError, NoImagesError
from facesorter.finder import (
    average_embedding,
    cosine_distances,
    find_matches,
    get_reference_embedding,
    largest_face,
    matches_reference,
)

E0, E1 = np.eye(4)[0], np.eye(4)[1]


def face(embedding, score=0.9, box=(0, 0, 10, 10)):
    return Face(box=box, embedding=np.asarray(embedding, dtype=float), score=score)


class TestCosineDistances:
    def test_identical_orthogonal_opposite(self):
        d = cosine_distances(E0, [E0, E1, -E0])
        assert d == pytest.approx([0.0, 1.0, 2.0])

    @pytest.mark.parametrize("scale", [1e-6, 0.5, 3.0, 1e6])
    def test_is_scale_invariant(self, scale):
        ref = np.array([1.0, 2.0, 3.0])
        emb = np.array([[3.0, 1.0, 2.0]])
        expected = cosine_distances(ref, emb)
        assert cosine_distances(ref * scale, emb) == pytest.approx(expected)
        assert cosine_distances(ref, emb * scale) == pytest.approx(expected)

    def test_zero_vector_is_maximally_unrelated(self):
        assert cosine_distances(E0, [np.zeros(4)]) == pytest.approx([1.0])
        assert cosine_distances(np.zeros(4), [E0]) == pytest.approx([1.0])

    def test_single_embedding_and_shape(self):
        assert cosine_distances(E0, E1).shape == (1,)
        assert cosine_distances(E0, np.empty((0, 4))).shape == (0,)

    @pytest.mark.parametrize("seed", range(5))
    def test_always_within_zero_and_two(self, seed):
        rng = np.random.default_rng(seed)
        d = cosine_distances(rng.normal(size=16), rng.normal(size=(50, 16)))
        assert d.min() >= 0.0
        assert d.max() <= 2.0

    def test_matches_manual_formula(self):
        ref, emb = np.array([1.0, 1.0]), np.array([[1.0, 0.0]])
        assert cosine_distances(ref, emb)[0] == pytest.approx(1 - 1 / np.sqrt(2))


class TestMatchesReference:
    def test_close_face_matches(self):
        assert matches_reference([face(E0)], E0, threshold=0.6, confidence=0.8)

    def test_far_face_does_not_match(self):
        assert not matches_reference([face(E1)], E0, threshold=0.6, confidence=0.8)

    def test_no_faces(self):
        assert not matches_reference([], E0, threshold=0.6, confidence=0.8)

    def test_any_face_in_a_group_photo_can_match(self):
        assert matches_reference([face(E1), face(E0)], E0, threshold=0.6, confidence=0.8)

    def test_threshold_is_strict(self):
        emb = np.array([3.0, 4.0, 0.0, 0.0])
        distance = float(cosine_distances(E0, [emb])[0])
        assert not matches_reference([face(emb)], E0, threshold=distance, confidence=0.5)
        above = float(np.nextafter(distance, np.inf))
        assert matches_reference([face(emb)], E0, threshold=above, confidence=0.5)

    def test_confidence_is_inclusive(self):
        assert matches_reference([face(E0, score=0.8)], E0, threshold=0.6, confidence=0.8)
        assert not matches_reference([face(E0, score=0.79)], E0, threshold=0.6, confidence=0.8)

    def test_low_confidence_match_is_ignored_even_if_identical(self):
        faces = [face(E0, score=0.3), face(E1, score=0.99)]
        assert not matches_reference(faces, E0, threshold=0.6, confidence=0.8)


class TestLargestFace:
    def test_picks_biggest_box(self):
        small, big = face(E0, box=(0, 0, 5, 5)), face(E1, box=(0, 0, 20, 20))
        assert largest_face([small, big]) is big
        assert largest_face([big, small]) is big

    def test_single_face(self):
        only = face(E0)
        assert largest_face([only]) is only

    def test_ties_keep_first(self):
        first, second = face(E0), face(E1)
        assert largest_face([first, second]) is first


class TestAverageEmbedding:
    def test_result_is_unit_length(self):
        rng = np.random.default_rng(0)
        centroid = average_embedding(list(rng.normal(size=(5, 16)) * 7))
        assert np.linalg.norm(centroid) == pytest.approx(1.0)

    def test_single_embedding_is_normalised(self):
        assert average_embedding([np.array([3.0, 4.0])]) == pytest.approx([0.6, 0.8])

    def test_is_the_mean_not_the_median(self):
        centroid = average_embedding([E0, E0, E1])
        assert centroid == pytest.approx(np.array([2, 1, 0, 0]) / np.sqrt(5))

    def test_each_embedding_is_normalised_before_averaging(self):
        # A raw mean would be dominated by the long vector: (10, 1) -> mostly e0.
        centroid = average_embedding([10 * E0, E1])
        assert centroid == pytest.approx(np.array([1, 1, 0, 0]) / np.sqrt(2))

    def test_opposite_vectors_cancel_to_zero_without_nan(self):
        centroid = average_embedding([E0, -E0])
        assert np.all(np.isfinite(centroid))
        assert np.linalg.norm(centroid) == 0.0

    def test_zero_vector_among_inputs_does_not_produce_nan(self):
        centroid = average_embedding([np.zeros(4), E0])
        assert centroid == pytest.approx(E0)


class TestGetReferenceEmbedding:
    def test_averages_reference_photos(self, reference_dir, fake_model):
        ref = get_reference_embedding(fake_model, reference_dir, confidence=0.8)
        assert ref is not None
        assert np.linalg.norm(ref) == pytest.approx(1.0)
        assert cosine_distances(ref, [np.eye(8)[0]])[0] < 0.01

    @pytest.mark.parametrize("large_first", [True, False])
    def test_uses_largest_face_per_photo(
        self, large_first, tmp_path, make_image, make_face, make_model
    ):
        alice, bob = np.eye(8)[0], np.eye(8)[1]
        faces = [make_face(alice, size=60), make_face(bob, size=30)]
        model = make_model({GROUP: faces if large_first else faces[::-1]})
        make_image(tmp_path / "ref" / "group.png", GROUP)
        ref = get_reference_embedding(model, str(tmp_path / "ref"), confidence=0.8)
        assert cosine_distances(ref, [alice])[0] == pytest.approx(0.0)

    def test_largest_face_must_also_be_confident(self, tmp_path, make_image, make_face, make_model):
        alice, bob = np.eye(8)[0], np.eye(8)[1]
        faces = [make_face(bob, score=0.3, size=90), make_face(alice, size=20)]
        model = make_model({GROUP: faces})
        make_image(tmp_path / "ref" / "group.png", GROUP)
        ref = get_reference_embedding(model, str(tmp_path / "ref"), confidence=0.8)
        assert cosine_distances(ref, [alice])[0] == pytest.approx(0.0)

    def test_ignores_low_confidence_faces(self, tmp_path, make_image, fake_model):
        make_image(tmp_path / "ref" / "blurry.png", LOW_CONF)
        assert get_reference_embedding(fake_model, str(tmp_path / "ref"), 0.8) is None
        assert get_reference_embedding(fake_model, str(tmp_path / "ref"), 0.4) is not None

    def test_bad_files_are_skipped(self, tmp_path, make_image, fake_model):
        ref = tmp_path / "ref"
        make_image(ref / "ok.png", ALICE)
        make_image(ref / "crash.png", CRASH)
        make_image(ref / "none.png", NO_FACE)
        (ref / "broken.jpg").write_bytes(b"nope")
        assert get_reference_embedding(fake_model, str(ref), 0.8) is not None

    def test_no_usable_face_returns_none(self, tmp_path, make_image, fake_model):
        make_image(tmp_path / "ref" / "none.png", NO_FACE)
        assert get_reference_embedding(fake_model, str(tmp_path / "ref"), 0.8) is None

    def test_empty_directory_raises(self, tmp_path, fake_model):
        with pytest.raises(NoImagesError):
            get_reference_embedding(fake_model, str(tmp_path), 0.8)

    def test_exclude_dir_is_not_scanned(self, tmp_path, make_image, fake_model):
        make_image(tmp_path / "ref" / "ref.png", ALICE)
        make_image(tmp_path / "ref" / "out" / "stale.png", BOB)
        ref = get_reference_embedding(
            fake_model, str(tmp_path / "ref"), 0.8, exclude_dir=str(tmp_path / "ref" / "out")
        )
        assert cosine_distances(ref, [np.eye(8)[0]])[0] == pytest.approx(0.0)


class TestFindMatches:
    EXPECTED = {"alice_1.png", "alice_1_1.png", "alice_2.png", "group.png"}

    def run(self, photo_dir, reference_dir, **kwargs):
        return find_matches(
            photo_dir.source, photo_dir.output, reference_dir, model=photo_dir.model, **kwargs
        )

    def test_copies_only_photos_of_the_person(self, photo_dir, reference_dir, tree):
        assert self.run(photo_dir, reference_dir) == 4
        assert tree(photo_dir.output) == self.EXPECTED

    def test_same_basename_in_different_subdirs_are_both_kept(self, photo_dir, reference_dir):
        self.run(photo_dir, reference_dir)
        assert os.path.isfile(os.path.join(photo_dir.output, "alice_1.png"))
        assert os.path.isfile(os.path.join(photo_dir.output, "alice_1_1.png"))

    def test_copies_are_byte_identical(self, photo_dir, reference_dir):
        self.run(photo_dir, reference_dir)
        out = os.path.join(photo_dir.output, "group.png")
        assert Path(out).read_bytes() == Path(photo_dir.files["group.png"]).read_bytes()

    def test_strict_threshold_finds_nothing_close_enough(self, photo_dir, reference_dir):
        # Alice 2 is slightly off the average of the two references; threshold ~0 drops it.
        assert self.run(photo_dir, reference_dir, threshold=1e-9) == 0

    def test_loose_threshold_matches_everyone(self, photo_dir, reference_dir):
        assert self.run(photo_dir, reference_dir, threshold=1.5) == 6  # 3 Alice + 1 group + 2 Bob

    def test_lower_confidence_includes_low_scoring_faces(self, photo_dir, reference_dir):
        assert self.run(photo_dir, reference_dir, confidence=0.4) == 5  # + blurry.png

    def test_existing_output_files_are_never_overwritten(self, photo_dir, reference_dir):
        os.makedirs(photo_dir.output)
        keep = os.path.join(photo_dir.output, "group.png")
        Path(keep).write_bytes(b"keep me")
        self.run(photo_dir, reference_dir)
        assert Path(keep).read_bytes() == b"keep me"
        assert os.path.isfile(os.path.join(photo_dir.output, "group_1.png"))

    def test_rerun_adds_copies_instead_of_overwriting(self, photo_dir, reference_dir, tree):
        self.run(photo_dir, reference_dir)
        self.run(photo_dir, reference_dir)
        assert len(tree(photo_dir.output)) == 8

    def test_output_inside_source_is_not_rescanned(self, photo_dir, reference_dir):
        photo_dir.output = os.path.join(photo_dir.source, "found")
        self.run(photo_dir, reference_dir)
        first_run_calls = len(photo_dir.model.seen_ids)
        photo_dir.model.seen_ids.clear()
        assert self.run(photo_dir, reference_dir) == 4
        assert len(photo_dir.model.seen_ids) == first_run_calls  # previous output not rescanned

    def test_output_inside_reference_dir_is_skipped(self, photo_dir, reference_dir):
        photo_dir.output = os.path.join(reference_dir, "found")
        self.run(photo_dir, reference_dir)
        first_run_calls = len(photo_dir.model.seen_ids)
        photo_dir.model.seen_ids.clear()
        assert self.run(photo_dir, reference_dir) == 4
        assert len(photo_dir.model.seen_ids) == first_run_calls  # previous output not rescanned

    def test_source_equal_to_output_is_rejected(self, photo_dir, reference_dir):
        with pytest.raises(InvalidPathError):
            find_matches(photo_dir.source, photo_dir.source, reference_dir, model=photo_dir.model)

    def test_source_inside_output_is_rejected(self, photo_dir, reference_dir):
        with pytest.raises(InvalidPathError):
            find_matches(
                photo_dir.source,
                os.path.dirname(photo_dir.source),
                reference_dir,
                model=photo_dir.model,
            )

    def test_missing_source(self, tmp_path, reference_dir, fake_model):
        with pytest.raises(InvalidPathError):
            find_matches(
                str(tmp_path / "nope"), str(tmp_path / "o"), reference_dir, model=fake_model
            )

    def test_empty_source_raises_no_images(self, tmp_path, reference_dir, fake_model):
        (tmp_path / "empty").mkdir()
        with pytest.raises(NoImagesError, match="source"):
            find_matches(
                str(tmp_path / "empty"), str(tmp_path / "o"), reference_dir, model=fake_model
            )

    def test_empty_reference_raises_no_images(self, photo_dir, tmp_path):
        (tmp_path / "empty_ref").mkdir()
        with pytest.raises(NoImagesError, match="reference"):
            self.run(photo_dir, str(tmp_path / "empty_ref"))

    def test_reference_without_faces_raises_no_faces(self, photo_dir, tmp_path, make_image):
        make_image(tmp_path / "ref" / "x.png", NO_FACE)
        with pytest.raises(NoFacesError, match="reference"):
            self.run(photo_dir, str(tmp_path / "ref"))

    def test_all_copies_failing_raises_copy_error(self, photo_dir, reference_dir, monkeypatch):
        def boom(*args, **kwargs):
            raise PermissionError("read-only")

        monkeypatch.setattr(finder, "copy_image", boom)
        with pytest.raises(CopyError, match="4 matching"):
            self.run(photo_dir, reference_dir)

    def test_partial_copy_failure_is_not_fatal(self, photo_dir, reference_dir, monkeypatch):
        real_copy, calls = finder.copy_image, []

        def flaky(src, dest_dir, filename):
            calls.append(filename)
            if len(calls) == 1:
                raise PermissionError("denied")
            return real_copy(src, dest_dir, filename)

        monkeypatch.setattr(finder, "copy_image", flaky)
        assert self.run(photo_dir, reference_dir) == 3

    def test_unreadable_photos_do_not_abort_the_search(self, photo_dir, reference_dir):
        # broken.jpg and crash.png are in the library; the search still finishes.
        assert self.run(photo_dir, reference_dir) == 4

    def test_loads_model_on_demand(self, photo_dir, reference_dir, monkeypatch):
        monkeypatch.setattr(finder, "get_face_model", lambda: photo_dir.model)
        assert find_matches(photo_dir.source, photo_dir.output, reference_dir) == 4

    def test_no_match_returns_zero_and_creates_empty_output(self, tmp_path, make_image, fake_model):
        make_image(tmp_path / "src" / "bob.png", BOB)
        make_image(tmp_path / "ref" / "alice.png", ALICE)
        n = find_matches(
            str(tmp_path / "src"), str(tmp_path / "o"), str(tmp_path / "ref"), model=fake_model
        )
        assert n == 0
        assert os.listdir(tmp_path / "o") == []
