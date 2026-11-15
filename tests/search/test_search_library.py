"""Tests for facesorter.search.library: still-image detection and the library walk."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from facesorter.errors import InvalidPathError
from facesorter.search.library import IMAGE_SUFFIXES, is_still_image, iter_library


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _rel(root: Path, paths) -> list[str]:
    return [p.relative_to(root).as_posix() for p in paths]


class TestIsStillImage:
    @pytest.mark.parametrize(
        "name",
        ["a.jpg", "a.JPG", "a.jpeg", "a.Png", "a.heic", "a.HEIF", "dir/sub/a.HeIc", "a.b.jpg"],
    )
    def test_supported(self, name):
        assert is_still_image(name)

    @pytest.mark.parametrize(
        "name",
        ["a.mov", "a.mp4", "a.gif", "a.txt", "a", "jpg", "a.jpg.bak", "._a.jpg", ".hidden.png"],
    )
    def test_unsupported_or_hidden(self, name):
        assert not is_still_image(name)

    def test_accepts_pathlike(self):
        assert is_still_image(Path("x") / "a.png")

    def test_suffix_set(self):
        assert IMAGE_SUFFIXES == {".heic", ".heif", ".jpg", ".jpeg", ".png"}


class TestIterLibrary:
    def test_recursive_sorted_absolute(self, tmp_path):
        for rel in ["b/2.jpg", "a/z.png", "a/y.JPG", "a/sub/1.heic", "top.jpeg", "b/1.jpg"]:
            _touch(tmp_path / rel)
        found = list(iter_library(tmp_path))
        assert all(p.is_absolute() for p in found)
        assert _rel(tmp_path, found) == [
            "a/sub/1.heic",
            "a/y.JPG",
            "a/z.png",
            "b/1.jpg",
            "b/2.jpg",
            "top.jpeg",
        ]

    def test_order_is_depth_first_name_sorted_not_path_string_sorted(self, tmp_path):
        # "a" < "a b" < "a.jpg" by entry name, so directory "a" is walked completely first,
        # whereas sorting the path strings would give "a b/x.jpg" < "a.jpg" < "a/x.jpg".
        for rel in ["a b/x.jpg", "a/x.jpg", "a.jpg", "a/z/y.jpg"]:
            _touch(tmp_path / rel)
        got = _rel(tmp_path, iter_library(tmp_path))
        assert got == ["a/x.jpg", "a/z/y.jpg", "a b/x.jpg", "a.jpg"]
        assert got != sorted(got)

    def test_order_independent_of_creation_order(self, tmp_path):
        names = ["c.jpg", "a.jpg", "b/x.jpg", "a2/x.jpg", "B.jpg"]
        one = tmp_path / "one"
        two = tmp_path / "two"
        for n in names:
            _touch(one / n)
        for n in reversed(names):
            _touch(two / n)
        assert _rel(one, iter_library(one)) == _rel(two, iter_library(two))

    def test_is_lazy_generator(self, tmp_path):
        _touch(tmp_path / "a.jpg")
        it = iter_library(tmp_path)
        assert iter(it) is it
        assert next(it).name == "a.jpg"

    def test_paths_are_canonical_for_str_roots_trailing_slash_and_tilde(
        self, tmp_path, monkeypatch
    ):
        _touch(tmp_path / "lib" / "sub" / "a.jpg")
        _touch(tmp_path / "lib" / "b.png")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        expected = [str(tmp_path / "lib" / "b.png"), str(tmp_path / "lib" / "sub" / "a.jpg")]
        expected = sorted(expected, key=lambda s: (s.count(os.sep), s))
        for root in (tmp_path / "lib", f"{tmp_path / 'lib'}{os.sep}", "~/lib", Path("~/lib/")):
            found = list(iter_library(root))
            assert all(isinstance(p, Path) for p in found)
            assert sorted(str(p) for p in found) == sorted(expected)

    def test_symlinks_in_root_are_not_resolved(self, tmp_path):
        _touch(tmp_path / "real" / "a.jpg")
        try:
            os.symlink(tmp_path / "real", tmp_path / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        (found,) = list(iter_library(tmp_path / "link"))
        assert str(found) == str(tmp_path / "link" / "a.jpg")

    def test_tilde_in_exclude_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        keep = _touch(tmp_path / "lib" / "keep.jpg")
        _touch(tmp_path / "lib" / "skip" / "b.jpg")
        assert list(iter_library(tmp_path / "lib", exclude_dirs=["~/lib/skip"])) == [keep]

    def test_relative_root_gives_absolute_paths(self, tmp_path, monkeypatch):
        _touch(tmp_path / "lib" / "a.jpg")
        monkeypatch.chdir(tmp_path)
        (found,) = list(iter_library("lib"))
        assert found.is_absolute()
        assert found.samefile(tmp_path / "lib" / "a.jpg")

    def test_skips_hidden_appledouble_videos_and_unsupported(self, tmp_path):
        keep = _touch(tmp_path / "keep.jpg")
        _touch(tmp_path / ".hidden.jpg")
        _touch(tmp_path / "._keep.jpg")
        _touch(tmp_path / "keep.mov")  # Live Photo companion
        _touch(tmp_path / "clip.MP4")
        _touch(tmp_path / "notes.txt")
        _touch(tmp_path / "noext")
        _touch(tmp_path / ".hiddendir" / "inside.jpg")
        _touch(tmp_path / ".Trashes" / "x.png")
        _touch(tmp_path / "ok" / "._junk.png")
        assert list(iter_library(tmp_path)) == [keep]

    def test_directory_named_like_an_image_is_not_yielded(self, tmp_path):
        (tmp_path / "folder.jpg").mkdir()
        inner = _touch(tmp_path / "folder.jpg" / "in.png")
        assert list(iter_library(tmp_path)) == [inner]

    def test_does_not_follow_symlinked_directories(self, tmp_path):
        lib = tmp_path / "lib"
        real = _touch(tmp_path / "outside" / "real.jpg")
        _touch(lib / "own.jpg")
        try:
            os.symlink(real.parent, lib / "link", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        assert _rel(lib, iter_library(lib)) == ["own.jpg"]

    def test_symlinked_file_is_yielded_but_broken_link_is_not(self, tmp_path):
        lib = tmp_path / "lib"
        target = _touch(tmp_path / "outside" / "real.jpg")
        lib.mkdir()
        try:
            os.symlink(target, lib / "link.jpg")
            os.symlink(tmp_path / "missing.jpg", lib / "broken.jpg")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        assert _rel(lib, iter_library(lib)) == ["link.jpg"]

    def test_symlinked_root_is_scanned(self, tmp_path):
        _touch(tmp_path / "real" / "a.jpg")
        try:
            os.symlink(tmp_path / "real", tmp_path / "rootlink", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        assert [p.name for p in iter_library(tmp_path / "rootlink")] == ["a.jpg"]

    def test_exclude_dirs_skips_directory_and_everything_inside(self, tmp_path):
        keep = _touch(tmp_path / "keep" / "a.jpg")
        _touch(tmp_path / "out" / "b.jpg")
        _touch(tmp_path / "out" / "deep" / "c.jpg")
        top = _touch(tmp_path / "top.jpg")
        found = list(iter_library(tmp_path, exclude_dirs=[tmp_path / "out"]))
        assert found == [keep, top]

    def test_exclude_dirs_accepts_strings_nested_and_unnormalised(self, tmp_path):
        keep = _touch(tmp_path / "a" / "keep.jpg")
        _touch(tmp_path / "a" / "nested" / "skip.jpg")
        _touch(tmp_path / "b" / "skip.jpg")
        excludes = [str(tmp_path / "a" / "nested") + os.sep, str(tmp_path / "x" / ".." / "b")]
        assert list(iter_library(str(tmp_path), exclude_dirs=excludes)) == [keep]

    def test_exclude_through_symlink_alias(self, tmp_path):
        lib = tmp_path / "lib"
        keep = _touch(lib / "keep" / "a.jpg")
        _touch(lib / "skip" / "b.jpg")
        try:
            os.symlink(lib / "skip", tmp_path / "alias", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        assert list(iter_library(lib, exclude_dirs=[tmp_path / "alias"])) == [keep]

    def test_exclude_only_matches_whole_directory_names(self, tmp_path):
        a = _touch(tmp_path / "out2" / "a.jpg")
        _touch(tmp_path / "out" / "b.jpg")
        assert list(iter_library(tmp_path, exclude_dirs=[tmp_path / "out"])) == [a]

    def test_nonexistent_exclude_is_ignored(self, tmp_path):
        a = _touch(tmp_path / "a.jpg")
        assert list(iter_library(tmp_path, exclude_dirs=[tmp_path / "nope"])) == [a]

    def test_root_inside_excluded_dir_yields_nothing(self, tmp_path):
        _touch(tmp_path / "out" / "sub" / "a.jpg")
        assert list(iter_library(tmp_path / "out" / "sub", exclude_dirs=[tmp_path / "out"])) == []

    def test_root_equal_to_excluded_dir_yields_nothing(self, tmp_path):
        _touch(tmp_path / "a.jpg")
        assert list(iter_library(tmp_path, exclude_dirs=[tmp_path])) == []

    def test_empty_directory(self, tmp_path):
        assert list(iter_library(tmp_path)) == []

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(InvalidPathError, match="nope"):
            list(iter_library(tmp_path / "nope"))

    def test_error_is_raised_on_first_next(self, tmp_path):
        with pytest.raises(InvalidPathError):
            next(iter_library(tmp_path / "nope"))

    def test_file_root_raises(self, tmp_path):
        f = _touch(tmp_path / "a.jpg")
        with pytest.raises(InvalidPathError):
            list(iter_library(f))

    @pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX non-root")
    def test_unreadable_subdirectory_is_skipped(self, tmp_path):
        ok = _touch(tmp_path / "a_ok" / "a.jpg")
        _touch(tmp_path / "b_locked" / "b.jpg")
        locked = tmp_path / "b_locked"
        locked.chmod(0)
        try:
            assert list(iter_library(tmp_path)) == [ok]
        finally:
            locked.chmod(0o755)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX non-root")
class TestOnSkip:
    @pytest.fixture
    def locked(self):
        dirs: list[Path] = []
        yield dirs
        for d in dirs:  # restore permissions so tmp_path can be cleaned up
            d.chmod(0o755)

    def _lock(self, locked: list[Path], directory: Path) -> Path:
        directory.chmod(0)
        locked.append(directory)
        return directory

    def test_called_with_exactly_the_unreadable_directory(self, tmp_path, locked):
        ok = _touch(tmp_path / "a_ok" / "a.jpg")
        _touch(tmp_path / "b_locked" / "b.jpg")
        bad = self._lock(locked, tmp_path / "b_locked")
        calls: list[tuple[Path, BaseException]] = []
        assert list(iter_library(tmp_path, on_skip=lambda d, e: calls.append((d, e)))) == [ok]
        assert len(calls) == 1
        directory, error = calls[0]
        assert directory == bad
        assert isinstance(directory, Path)
        assert isinstance(error, PermissionError)

    def test_not_called_when_everything_is_readable(self, tmp_path):
        _touch(tmp_path / "a" / "a.jpg")
        calls: list[Path] = []
        assert len(list(iter_library(tmp_path, on_skip=lambda d, e: calls.append(d)))) == 1
        assert calls == []

    def test_not_called_for_excluded_or_hidden_directories(self, tmp_path):
        _touch(tmp_path / "out" / "a.jpg")
        calls: list[Path] = []
        list(
            iter_library(
                tmp_path, exclude_dirs=[tmp_path / "out"], on_skip=lambda d, e: calls.append(d)
            )
        )
        assert calls == []

    def test_nested_unreadable_directories_are_all_reported(self, tmp_path, locked):
        ok = _touch(tmp_path / "ok" / "a.jpg")
        _touch(tmp_path / "x" / "inner" / "b.jpg")
        _touch(tmp_path / "y" / "c.jpg")
        _touch(tmp_path / "z" / "deep" / "d.jpg")
        # lock an inner directory and a top-level one; the nested one under y stays reachable
        inner = self._lock(locked, tmp_path / "x" / "inner")
        y = self._lock(locked, tmp_path / "y")
        deep = self._lock(locked, tmp_path / "z" / "deep")
        calls: list[Path] = []
        found = list(iter_library(tmp_path, on_skip=lambda d, e: calls.append(d)))
        assert found == [ok]
        assert calls == [inner, y, deep]

    def test_raising_callback_does_not_break_the_scan(self, tmp_path, locked, caplog):
        _touch(tmp_path / "a" / "a.jpg")
        _touch(tmp_path / "b" / "b.jpg")
        _touch(tmp_path / "c" / "c.jpg")
        self._lock(locked, tmp_path / "b")
        calls: list[Path] = []

        def boom(directory, error):
            calls.append(directory)
            raise RuntimeError("callback failed")

        with caplog.at_level("ERROR", logger="facesorter.search.library"):
            found = list(iter_library(tmp_path, on_skip=boom))
        assert _rel(tmp_path, found) == ["a/a.jpg", "c/c.jpg"]
        assert len(calls) == 1
        assert "on_skip callback failed" in caplog.text

    def test_without_callback_still_warns_and_skips(self, tmp_path, locked, caplog):
        ok = _touch(tmp_path / "a" / "a.jpg")
        _touch(tmp_path / "b" / "b.jpg")
        self._lock(locked, tmp_path / "b")
        with caplog.at_level("WARNING", logger="facesorter.search.library"):
            assert list(iter_library(tmp_path)) == [ok]
        assert "Skipping unreadable folder" in caplog.text

    def test_callback_is_in_addition_to_the_warning(self, tmp_path, locked, caplog):
        _touch(tmp_path / "b" / "b.jpg")
        self._lock(locked, tmp_path / "b")
        with caplog.at_level("WARNING", logger="facesorter.search.library"):
            list(iter_library(tmp_path, on_skip=lambda d, e: None))
        assert "Skipping unreadable folder" in caplog.text

    def test_is_keyword_only(self, tmp_path):
        with pytest.raises(TypeError):
            iter_library(tmp_path, (), lambda d, e: None)  # type: ignore[misc]
