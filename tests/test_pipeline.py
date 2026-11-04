from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from conftest import ALICE, NO_FACE

from facesorter import pipeline
from facesorter.detection import Face
from facesorter.errors import CopyError, InvalidPathError, NoFacesError, NoImagesError
from facesorter.pipeline import (
    ScanResult,
    group_by_person,
    plan_copies,
    process_images,
    scan_images,
    write_report,
)


def names(paths):
    return [os.path.basename(p) for p in paths]


def folder_contents(root, tree):
    """Map each top-level folder of ``root`` to the sorted file names inside it."""
    result: dict[str, list[str]] = {}
    for rel in sorted(tree(root)):
        folder, _, name = rel.partition("/")
        if name:
            result.setdefault(folder, []).append(name)
    return {k: sorted(v) for k, v in result.items()}


class TestScanImages:
    def scan(self, photo_dir, confidence=0.8):
        files = [photo_dir.files[k] for k in sorted(photo_dir.files) if not k.endswith(".txt")]
        return scan_images(photo_dir.model, files, confidence)

    def test_classifies_every_kind_of_file(self, photo_dir):
        result = self.scan(photo_dir)
        assert names(result.no_faces) == ["blurry.png", "nofaces.png"]
        assert names(result.load_errors) == ["broken.jpg"]
        assert names(result.detection_errors) == ["crash.png"]
        assert names(result.multiple_faces) == ["group.png"]

    def test_keeps_only_confident_faces(self, photo_dir):
        result = self.scan(photo_dir)
        assert all(face.score >= 0.8 for _, face in result.faces)
        assert len(result.faces) == 7  # 3 alice, 2 bob and the 2 faces of group.png
        assert "blurry.png" not in names(p for p, _ in result.faces)

    def test_low_confidence_threshold_admits_blurry_photo(self, photo_dir):
        result = self.scan(photo_dir, confidence=0.4)
        assert names(result.no_faces) == ["nofaces.png"]

    def test_confidence_is_inclusive(self, tmp_path, make_image, make_face, make_model):
        path = make_image(tmp_path / "a.png", ALICE)
        model = make_model({ALICE: [make_face(np.ones(4), score=0.8)]})
        assert len(scan_images(model, [path], 0.8).faces) == 1
        assert len(scan_images(model, [path], 0.81).faces) == 0

    def test_empty_file_list(self, fake_model):
        assert scan_images(fake_model, [], 0.8) == ScanResult()


class TestGroupByPerson:
    def test_groups_and_orders_by_label(self):
        groups = group_by_person(["a", "b", "c"], [1, 0, 1])
        assert list(groups) == ["Person_0", "Person_1"]
        assert groups == {"Person_0": ["b"], "Person_1": ["a", "c"]}

    def test_image_with_several_faces_of_one_person_appears_once(self):
        assert group_by_person(["a", "a", "b"], [0, 0, 0]) == {"Person_0": ["a", "b"]}

    def test_multi_person_image_is_in_each_group(self):
        groups = group_by_person(["a", "a"], [0, 1])
        assert groups == {"Person_0": ["a"], "Person_1": ["a"]}

    def test_accepts_numpy_labels(self):
        groups = group_by_person(["a", "b"], np.array([3, 3]))
        assert list(groups) == ["Person_3"]

    def test_empty(self):
        assert group_by_person([], []) == {}


class TestPlanCopies:
    def test_unique_names_are_unchanged(self):
        plan = plan_copies({"Person_0": ["/x/a.jpg", "/y/b.jpg"]})
        assert plan == [("/x/a.jpg", "Person_0", "a.jpg"), ("/y/b.jpg", "Person_0", "b.jpg")]

    def test_same_basename_gets_numeric_suffix(self):
        plan = plan_copies({"Person_0": ["/x/a.jpg", "/y/a.jpg", "/z/A.JPG"]})
        assert [dest for _, _, dest in plan] == ["a.jpg", "a_1.jpg", "A_2.JPG"]

    def test_names_are_tracked_per_person_folder(self):
        plan = plan_copies({"Person_0": ["/x/a.jpg"], "Person_1": ["/y/a.jpg"]})
        assert [dest for _, _, dest in plan] == ["a.jpg", "a.jpg"]


class TestWriteReport:
    def test_sections_and_summary(self, tmp_path):
        scan = ScanResult(
            no_faces=["/p/none.jpg"],
            load_errors=["/p/bad.jpg"],
            detection_errors=["/p/crash.jpg"],
            multiple_faces=["/p/group.jpg"],
        )
        path = str(tmp_path / "report.txt")
        write_report(path, ["Total images processed: 4"], scan)
        text = Path(path).read_text(encoding="utf-8")
        assert text.startswith("Face Sorting Report\n" + "=" * 40 + "\n")
        assert "Total images processed: 4\n" in text
        for title, entry in [
            ("Images with no faces:", "/p/none.jpg"),
            ("Images failed to load:", "/p/bad.jpg"),
            ("Images failed during face detection:", "/p/crash.jpg"),
            ("Images with multiple faces:", "/p/group.jpg"),
        ]:
            assert text.index(title) < text.index(entry)


class TestProcessImages:
    def sort(self, photo_dir, **kwargs):
        kwargs.setdefault("cluster_threshold", 0.6)
        kwargs.setdefault("confidence", 0.8)
        return process_images(photo_dir.source, photo_dir.output, model=photo_dir.model, **kwargs)

    def test_sorts_two_people_into_two_folders(self, photo_dir, tree):
        assert self.sort(photo_dir) == 2
        assert folder_contents(photo_dir.output, tree) == {
            "Person_0": sorted(["alice_1.png", "alice_1_1.png", "alice_2.png", "group.png"]),
            "Person_1": sorted(["bob_1.PNG", "bob_2.png", "group.png"]),
        }

    def test_same_basename_from_different_subdirs_are_both_kept(self, photo_dir):
        self.sort(photo_dir)
        person = os.path.join(photo_dir.output, "Person_0")
        assert os.path.isfile(os.path.join(person, "alice_1.png"))
        assert os.path.isfile(os.path.join(person, "alice_1_1.png"))

    def test_multi_face_photo_is_copied_once_into_each_person_folder(self, photo_dir, tree):
        self.sort(photo_dir)
        copies = [rel for rel in tree(photo_dir.output) if rel.endswith("group.png")]
        assert sorted(copies) == ["Person_0/group.png", "Person_1/group.png"]

    def test_threshold_controls_number_of_people(self, photo_dir):
        assert self.sort(photo_dir, cluster_threshold=1.5) == 1
        photo_dir.output += "_loose"
        assert (
            self.sort(photo_dir, cluster_threshold=1e-4) == 4
        )  # the slightly different shots split off

    def test_output_is_deterministic(self, photo_dir, tree, tmp_path):
        self.sort(photo_dir)
        first = folder_contents(photo_dir.output, tree)
        photo_dir.output = str(tmp_path / "second")
        self.sort(photo_dir)
        assert folder_contents(photo_dir.output, tree) == first

    def test_report_summary_lines(self, photo_dir):
        self.sort(photo_dir)
        report = Path(os.path.join(photo_dir.output, "report.txt")).read_text(encoding="utf-8")
        assert report.splitlines()[2:11] == [
            "Total images processed: 10",
            "Total faces detected: 7",
            "Unique persons found: 2",
            "Images with multiple faces: 1",
            "Images with no faces: 2",
            "Images failed to load: 1",
            "Images failed during face detection: 1",
            "Files copied: 7 of 7",
            "Copy failures: 0",
        ]

    def test_report_keeps_load_and_detection_failures_apart(self, photo_dir):
        Path(photo_dir.source, "broken2.png").write_bytes(b"junk")
        self.sort(photo_dir)
        lines = Path(photo_dir.output, "report.txt").read_text(encoding="utf-8").splitlines()
        assert "Images failed to load: 2" in lines
        assert "Images failed during face detection: 1" in lines
        assert "Total images processed: 11" in lines

    def test_report_lists_problem_photos(self, photo_dir):
        self.sort(photo_dir)
        report = Path(os.path.join(photo_dir.output, "report.txt")).read_text(encoding="utf-8")
        sections = report.split("\n\n")
        by_title = {s.split(":\n")[0]: s for s in sections if ":\n" in s}
        assert "nofaces.png" in by_title["Images with no faces"]
        assert "blurry.png" in by_title["Images with no faces"]
        assert "broken.jpg" in by_title["Images failed to load"]
        assert "crash.png" in by_title["Images failed during face detection"]
        assert "group.png" in by_title["Images with multiple faces"]

    def test_no_runtime_log_is_written_by_the_library(self, photo_dir, tree):
        self.sort(photo_dir)
        assert "processing_log.txt" not in tree(photo_dir.output)
        assert "report.txt" in tree(photo_dir.output)

    def test_non_image_files_are_not_copied(self, photo_dir, tree):
        self.sort(photo_dir)
        assert not any(rel.endswith(".txt") and "/" in rel for rel in tree(photo_dir.output))

    def test_output_inside_source_is_skipped_on_rerun(self, photo_dir, tree):
        photo_dir.output = os.path.join(photo_dir.source, "sorted")
        self.sort(photo_dir)
        first = tree(photo_dir.output)
        self.sort(photo_dir)
        report = Path(os.path.join(photo_dir.output, "report.txt")).read_text(encoding="utf-8")
        assert "Total images processed: 10" in report  # the earlier output was not rescanned
        assert first <= tree(photo_dir.output)

    def test_rerun_never_overwrites_earlier_results(self, photo_dir, tree):
        self.sort(photo_dir)
        before = tree(photo_dir.output)
        marker = os.path.join(photo_dir.output, "Person_0", "alice_2.png")
        Path(marker).write_bytes(b"edited by the user")
        self.sort(photo_dir)
        assert Path(marker).read_bytes() == b"edited by the user"
        assert os.path.isfile(os.path.join(photo_dir.output, "Person_0", "alice_2_1.png"))
        assert before < tree(photo_dir.output)

    def test_preexisting_file_in_person_folder_is_preserved(self, photo_dir):
        existing = os.path.join(photo_dir.output, "Person_1", "bob_2.png")
        os.makedirs(os.path.dirname(existing))
        Path(existing).write_bytes(b"mine")
        self.sort(photo_dir)
        assert Path(existing).read_bytes() == b"mine"
        assert os.path.isfile(os.path.join(photo_dir.output, "Person_1", "bob_2_1.png"))

    def test_source_equal_to_output_is_rejected(self, photo_dir):
        photo_dir.output = photo_dir.source
        with pytest.raises(InvalidPathError):
            self.sort(photo_dir)

    def test_source_inside_output_is_rejected(self, photo_dir):
        photo_dir.output = os.path.dirname(photo_dir.source)
        with pytest.raises(InvalidPathError):
            self.sort(photo_dir)

    def test_missing_source(self, photo_dir, tmp_path):
        photo_dir.source = str(tmp_path / "nope")
        with pytest.raises(InvalidPathError):
            self.sort(photo_dir)

    def test_no_images(self, tmp_path, fake_model):
        (tmp_path / "empty").mkdir()
        (tmp_path / "empty" / "readme.txt").write_text("hi")
        with pytest.raises(NoImagesError):
            process_images(str(tmp_path / "empty"), str(tmp_path / "o"), 0.6, 0.8, model=fake_model)

    def test_no_faces_reports_unprocessable_files(self, tmp_path, make_image, fake_model):
        src = tmp_path / "src"
        make_image(src / "none.png", NO_FACE)
        (src / "broken.jpg").write_bytes(b"x")
        with pytest.raises(NoFacesError, match=r"1 of them could not be processed"):
            process_images(str(src), str(tmp_path / "o"), 0.6, 0.8, model=fake_model)

    def test_no_faces_creates_no_output(self, tmp_path, make_image, fake_model):
        make_image(tmp_path / "src" / "none.png", NO_FACE)
        with pytest.raises(NoFacesError):
            process_images(str(tmp_path / "src"), str(tmp_path / "o"), 0.6, 0.8, model=fake_model)
        assert not os.path.exists(tmp_path / "o")

    def test_confidence_too_high_means_no_faces(self, photo_dir):
        with pytest.raises(NoFacesError, match="confidence >= 0.995"):
            self.sort(photo_dir, confidence=0.995)

    def test_every_copy_failing_raises_copy_error_but_writes_report(self, photo_dir, monkeypatch):
        def boom(*args, **kwargs):
            raise PermissionError("read-only")

        monkeypatch.setattr(pipeline, "copy_image_to_person_folder", boom)
        with pytest.raises(CopyError):
            self.sort(photo_dir)
        report = Path(os.path.join(photo_dir.output, "report.txt")).read_text(encoding="utf-8")
        assert "Files copied: 0 of 7" in report

    def test_partial_copy_failure_is_reported_not_fatal(self, photo_dir, monkeypatch):
        real, calls = pipeline.copy_image_to_person_folder, []

        def flaky(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                raise PermissionError("denied")
            return real(*args, **kwargs)

        monkeypatch.setattr(pipeline, "copy_image_to_person_folder", flaky)
        assert self.sort(photo_dir) == 2
        report = Path(os.path.join(photo_dir.output, "report.txt")).read_text(encoding="utf-8")
        assert "Copy failures: 1" in report

    def test_loads_model_on_demand(self, photo_dir, monkeypatch):
        monkeypatch.setattr(pipeline, "get_face_model", lambda: photo_dir.model)
        assert process_images(photo_dir.source, photo_dir.output, 0.6, 0.8) == 2


def test_scan_result_faces_are_face_objects(photo_dir):
    result = scan_images(photo_dir.model, [photo_dir.files["alice_1.png"]], 0.8)
    assert all(isinstance(face, Face) for _, face in result.faces)
