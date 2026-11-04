from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import ALICE
from PIL import Image

from facesorter.errors import InvalidPathError
from facesorter.utils import (
    copy_image,
    copy_image_to_person_folder,
    get_image_files,
    is_image_file,
    load_image,
    unique_filename,
)


class TestIsImageFile:
    @pytest.mark.parametrize(
        "name", ["a.jpg", "a.JPG", "a.jpeg", "a.JpEg", "a.png", "a.PNG", "a.heic", "a.HEIC"]
    )
    def test_supported_extensions_any_case(self, name):
        assert is_image_file(name)

    @pytest.mark.parametrize("name", ["a.txt", "a.gif", "jpg", "a.jpg.bak", "a.png.txt", ""])
    def test_unsupported_names(self, name):
        assert not is_image_file(name)


class TestLoadImage:
    def test_applies_exif_orientation(self, tmp_path):
        # Stored 40x20 (landscape) with orientation 6 => displayed rotated as 20x40.
        exif = Image.Exif()
        exif[0x0112] = 6
        path = str(tmp_path / "sideways.jpg")
        Image.new("RGB", (40, 20), (200, 10, 10)).save(path, exif=exif)

        assert load_image(path).size == (20, 40)

    def test_exif_transpose_moves_pixels(self, tmp_path):
        # Orientation 3 = rotated 180 degrees: the top-left marker must end up bottom-right.
        img = Image.new("RGB", (8, 8), (0, 0, 0))
        img.putpixel((0, 0), (255, 255, 255))
        exif = Image.Exif()
        exif[0x0112] = 3
        path = str(tmp_path / "upside_down.png")
        img.save(path, exif=exif)

        loaded = load_image(path)
        assert loaded.getpixel((7, 7)) == (255, 255, 255)
        assert loaded.getpixel((0, 0)) == (0, 0, 0)

    def test_without_exif_is_unchanged(self, make_image):
        assert load_image(make_image(os.path.join("x", "a.png"), ALICE, size=(30, 10))).size == (
            30,
            10,
        )

    @pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
    def test_always_returns_rgb(self, tmp_path, mode):
        path = str(tmp_path / f"{mode}.png")
        Image.new(mode, (4, 4)).save(path)
        assert load_image(path).mode == "RGB"

    def test_heic_roundtrip(self, tmp_path):
        path = str(tmp_path / "photo.heic")
        try:
            Image.new("RGB", (32, 16), (90, 120, 30)).save(path, format="HEIF")
        except Exception as exc:  # pillow-heif wheels without an encoder
            pytest.skip(f"cannot write HEIC here: {exc}")
        assert load_image(path).size == (32, 16)

    def test_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "broken.jpg"
        path.write_bytes(b"definitely not a jpeg")
        with pytest.raises(OSError):
            load_image(str(path))


class TestUniqueFilename:
    def test_first_use_is_unchanged(self):
        assert unique_filename("a.jpg", set()) == "a.jpg"

    def test_collisions_get_incrementing_suffix(self):
        taken: set[str] = set()
        names = [unique_filename("a.jpg", taken) for _ in range(3)]
        assert names == ["a.jpg", "a_1.jpg", "a_2.jpg"]

    def test_comparison_is_case_insensitive(self):
        taken: set[str] = set()
        assert unique_filename("Photo.JPG", taken) == "Photo.JPG"
        assert unique_filename("photo.jpg", taken) == "photo_1.jpg"

    def test_skips_suffix_that_is_already_taken(self):
        taken: set[str] = set()
        assert unique_filename("a_1.jpg", taken) == "a_1.jpg"
        assert unique_filename("a.jpg", taken) == "a.jpg"
        assert unique_filename("a.jpg", taken) == "a_2.jpg"

    def test_name_without_extension(self):
        taken: set[str] = set()
        assert [unique_filename("README", taken) for _ in range(2)] == ["README", "README_1"]

    @pytest.mark.parametrize("count", [1, 2, 10, 50])
    def test_n_equal_names_give_n_distinct_names(self, count):
        taken: set[str] = set()
        names = [unique_filename("img.png", taken) for _ in range(count)]
        assert len({n.casefold() for n in names}) == count
        assert len(taken) == count


class TestCopyImage:
    def test_copies_content_and_mtime(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload")
        os.utime(src, (1_000_000_000, 1_000_000_000))

        dest = copy_image(str(src), str(tmp_path / "out"), "dst.bin")

        assert dest == str(tmp_path / "out" / "dst.bin")
        assert Path(dest).read_bytes() == b"payload"
        assert os.path.getmtime(dest) == pytest.approx(1_000_000_000)

    def test_creates_missing_destination_dirs(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x")
        dest = copy_image(str(src), str(tmp_path / "a" / "b" / "c"), "f.bin")
        assert os.path.isfile(dest)

    def test_never_overwrites_existing_file(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"new")
        out = tmp_path / "out"
        out.mkdir()
        (out / "f.bin").write_bytes(b"precious original")

        dest = copy_image(str(src), str(out), "f.bin")

        assert os.path.basename(dest) == "f_1.bin"
        assert (out / "f.bin").read_bytes() == b"precious original"
        assert (out / "f_1.bin").read_bytes() == b"new"

    def test_repeated_runs_keep_every_copy(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x")
        names = [
            os.path.basename(copy_image(str(src), str(tmp_path / "o"), "f.bin")) for _ in range(3)
        ]
        assert names == ["f.bin", "f_1.bin", "f_2.bin"]

    def test_missing_source_raises_and_leaves_nothing(self, tmp_path):
        out = tmp_path / "out"
        with pytest.raises(FileNotFoundError):
            copy_image(str(tmp_path / "nope.bin"), str(out), "f.bin")
        assert list(out.iterdir()) == []

    def test_person_folder_variant(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x")
        dest = copy_image_to_person_folder(str(src), str(tmp_path), "Person_3", "f.bin")
        assert dest == str(tmp_path / "Person_3" / "f.bin")


class TestGetImageFiles:
    def test_recursive_sorted_and_filtered(self, photo_dir):
        found = get_image_files(photo_dir.source)
        rel = [os.path.relpath(p, photo_dir.source).replace(os.sep, "/") for p in found]
        assert rel == [
            "alice_1.png",
            "blurry.png",
            "bob_1.PNG",
            "broken.jpg",
            "crash.png",
            "group.png",
            "nofaces.png",
            "sub/alice_2.png",
            "sub/bob_2.png",
            "sub/deeper/alice_1.png",
        ]

    def test_order_does_not_depend_on_filesystem_order(self, tmp_path, make_image, monkeypatch):
        for folder in ("b", "a", "c"):
            for name in ("x.png", "w.png"):
                make_image(tmp_path / folder / name, ALICE)
        real_walk = os.walk

        def reversed_walk(top, *args, **kwargs):
            for root, dirs, files in real_walk(top, *args, **kwargs):
                dirs.sort(reverse=True)
                files.sort(reverse=True)
                yield root, dirs, files

        monkeypatch.setattr(os, "walk", reversed_walk)
        found = [
            os.path.relpath(p, tmp_path).replace(os.sep, "/")
            for p in get_image_files(str(tmp_path))
        ]
        assert found == ["a/w.png", "a/x.png", "b/w.png", "b/x.png", "c/w.png", "c/x.png"]

    def test_empty_directory(self, tmp_path):
        assert get_image_files(str(tmp_path)) == []

    def test_missing_directory_is_rejected(self, tmp_path):
        with pytest.raises(InvalidPathError, match="Not a directory"):
            get_image_files(str(tmp_path / "missing"))

    def test_file_instead_of_directory_is_rejected(self, tmp_path):
        f = tmp_path / "a.png"
        f.write_bytes(b"")
        with pytest.raises(InvalidPathError):
            get_image_files(str(f))

    def test_exclude_dir_inside_source_is_skipped(self, photo_dir):
        excluded = os.path.join(photo_dir.source, "sub")
        found = get_image_files(photo_dir.source, exclude_dir=excluded + os.sep)
        assert found
        assert not any(os.sep + "sub" + os.sep in p for p in found)

    def test_exclude_dir_skips_nested_output_of_earlier_run(self, photo_dir, make_image):
        out = os.path.join(photo_dir.source, "out")
        make_image(os.path.join(out, "Person_0", "old.png"), ALICE)
        assert not any("Person_0" in p for p in get_image_files(photo_dir.source, exclude_dir=out))
        assert any("Person_0" in p for p in get_image_files(photo_dir.source))

    def test_exclude_dir_nested_several_levels_deep(self, photo_dir, make_image):
        out = os.path.join(photo_dir.source, "sub", "deeper", "out")
        make_image(os.path.join(out, "Person_0", "old.png"), ALICE)
        found = get_image_files(photo_dir.source, exclude_dir=out)
        assert not any(os.sep + "out" + os.sep in p for p in found)
        assert len(found) == 10  # everything else is still found

    def test_exclude_dir_through_symlink(self, photo_dir, tmp_path):
        link = tmp_path / "link_to_sub"
        try:
            os.symlink(os.path.join(photo_dir.source, "sub"), link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        found = get_image_files(photo_dir.source, exclude_dir=str(link))
        assert not any(os.sep + "sub" + os.sep in p for p in found)

    def test_source_equal_to_exclude_dir_is_rejected(self, photo_dir):
        with pytest.raises(InvalidPathError, match="must not be the same as, or contain"):
            get_image_files(photo_dir.source, exclude_dir=photo_dir.source)

    def test_source_inside_exclude_dir_is_rejected(self, photo_dir):
        parent = os.path.dirname(photo_dir.source)
        with pytest.raises(InvalidPathError):
            get_image_files(photo_dir.source, exclude_dir=parent)

    def test_sibling_with_common_prefix_is_not_excluded(self, tmp_path, make_image):
        make_image(tmp_path / "photos" / "a.png", ALICE)
        (tmp_path / "photos_out").mkdir()
        assert len(get_image_files(str(tmp_path / "photos"), str(tmp_path / "photos_out"))) == 1
