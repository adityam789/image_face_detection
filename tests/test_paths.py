from __future__ import annotations

import os

import pytest

from facesorter.paths import is_inside, same_path


def _symlink(target, link):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")


class TestSamePath:
    def test_identical(self, tmp_path):
        assert same_path(str(tmp_path), str(tmp_path))

    def test_different_spelling(self, tmp_path):
        (tmp_path / "a").mkdir()
        assert same_path(str(tmp_path / "a"), str(tmp_path / "a" / ".." / "a") + os.sep)

    def test_different_directories(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        assert not same_path(str(tmp_path / "a"), str(tmp_path / "b"))

    def test_symlink_resolves_to_target(self, tmp_path):
        (tmp_path / "real").mkdir()
        _symlink(tmp_path / "real", tmp_path / "link")
        assert same_path(str(tmp_path / "real"), str(tmp_path / "link"))

    def test_nonexistent_paths_are_compared_by_normalised_path(self, tmp_path):
        assert same_path(str(tmp_path / "x" / ".." / "y"), str(tmp_path / "y"))
        assert not same_path(str(tmp_path / "x"), str(tmp_path / "y"))

    def test_nonexistent_path_below_a_symlink(self, tmp_path):
        (tmp_path / "real").mkdir()
        _symlink(tmp_path / "real", tmp_path / "link")
        assert same_path(str(tmp_path / "link" / "new"), str(tmp_path / "real" / "new"))

    def test_prefers_samefile_when_both_exist(self, tmp_path, monkeypatch):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        monkeypatch.setattr(os.path, "samefile", lambda x, y: True)
        assert same_path(str(tmp_path / "a"), str(tmp_path / "b"))

    def test_case_insensitive_filesystem(self, tmp_path):
        (tmp_path / "Out").mkdir()
        if not os.path.exists(tmp_path / "out"):
            pytest.skip("case-sensitive filesystem")
        assert same_path(str(tmp_path / "Out"), str(tmp_path / "out"))

    def test_existing_vs_missing_path(self, tmp_path):
        assert not same_path(str(tmp_path), str(tmp_path / "missing"))


class TestIsInside:
    def test_directory_is_inside_itself(self, tmp_path):
        assert is_inside(str(tmp_path), str(tmp_path))

    def test_direct_and_deep_children(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        assert is_inside(str(tmp_path / "a"), str(tmp_path))
        assert is_inside(str(tmp_path / "a" / "b"), str(tmp_path))

    def test_not_yet_created_child(self, tmp_path):
        assert is_inside(str(tmp_path / "new" / "output"), str(tmp_path))

    def test_parent_is_not_inside_child(self, tmp_path):
        (tmp_path / "a").mkdir()
        assert not is_inside(str(tmp_path), str(tmp_path / "a"))

    def test_shared_name_prefix_is_not_containment(self, tmp_path):
        (tmp_path / "out").mkdir()
        (tmp_path / "out2").mkdir()
        assert not is_inside(str(tmp_path / "out2"), str(tmp_path / "out"))

    def test_child_reached_through_symlink(self, tmp_path):
        (tmp_path / "real" / "inner").mkdir(parents=True)
        _symlink(tmp_path / "real", tmp_path / "link")
        assert is_inside(str(tmp_path / "link" / "inner"), str(tmp_path / "real"))

    def test_relative_paths(self, tmp_path, monkeypatch):
        (tmp_path / "a" / "b").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert is_inside(os.path.join("a", "b"), "a")
        assert not is_inside("a", os.path.join("a", "b"))
