"""Tests for facesorter.search.metadata: EXIF dates, folder/mtime fallbacks, GPS, camera, size.

All images are tiny synthetic files built with PIL (and pillow_heif for HEIC).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from facesorter.search import metadata as md  # noqa: E402
from facesorter.search.errors import UnreadableImageError  # noqa: E402
from facesorter.search.metadata import (  # noqa: E402
    date_from_folders,
    parse_exif_datetime,
    parse_gps,
    read_metadata,
)
from facesorter.search.types import PhotoMeta  # noqa: E402

DATE_ORIGINAL = 0x9003
DATE_DIGITIZED = 0x9004
DATE_TIME = 0x0132


def _exif(
    *,
    original=None,
    digitized=None,
    datetime_=None,
    orientation=None,
    make=None,
    model=None,
    gps=None,
) -> Image.Exif:
    ex = Image.Exif()
    if datetime_ is not None:
        ex[DATE_TIME] = datetime_
    if orientation is not None:
        ex[0x0112] = orientation
    if make is not None:
        ex[0x010F] = make
    if model is not None:
        ex[0x0110] = model
    sub = ex.get_ifd(0x8769)
    if original is not None:
        sub[DATE_ORIGINAL] = original
    if digitized is not None:
        sub[DATE_DIGITIZED] = digitized
    if gps:
        ex.get_ifd(0x8825).update(gps)
    return ex


def _jpeg(path: Path, exif: Image.Exif | None = None, size=(40, 20)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (200, 30, 30))
    if exif is None:
        img.save(path)
    else:
        img.save(path, exif=exif)
    return path


def _gps(lat, lat_ref, lon, lon_ref) -> dict:
    return {1: lat_ref, 2: lat, 3: lon_ref, 4: lon}


def _set_mtime(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


# --------------------------------------------------------------------------- EXIF dates


class TestParseExifDatetime:
    def test_standard(self):
        assert parse_exif_datetime("2021:05:06 07:08:09") == datetime(2021, 5, 6, 7, 8, 9)

    def test_bytes_and_nul_padding(self):
        assert parse_exif_datetime(b"2021:05:06 07:08:09\x00") == datetime(2021, 5, 6, 7, 8, 9)

    @pytest.mark.parametrize(
        "text",
        [
            "2021-05-06 07:08:09",
            "2021/05/06 07:08:09",
            "2021-05-06T07:08:09",
            "  2021:05:06 07:08:09",
        ],
    )
    def test_tolerates_other_separators(self, text):
        assert parse_exif_datetime(text) == datetime(2021, 5, 6, 7, 8, 9)

    def test_trailing_junk_ignored(self):
        assert parse_exif_datetime("2021:05:06 07:08:09.123+02:00") == datetime(2021, 5, 6, 7, 8, 9)

    def test_date_only_and_no_seconds(self):
        assert parse_exif_datetime("2021:05:06") == datetime(2021, 5, 6)
        assert parse_exif_datetime("2021:05:06 07:08") == datetime(2021, 5, 6, 7, 8)

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "   ",
            "0000:00:00 00:00:00",
            "    :  :     :  :  ",
            "2021:13:01 00:00:00",
            "2021:02:30 00:00:00",
            "1850:01:01 00:00:00",
            "3000:01:01 00:00:00",
            "garbage",
            12345,
            (2021, 5, 6),
            b"\xff\xfe",
        ],
    )
    def test_missing_or_impossible(self, value):
        assert parse_exif_datetime(value) is None

    @pytest.mark.parametrize(
        "text", ["2023:07:14 25:61:61", "2021:05:06 25:00:00", "2021:05:06 07:61:00"]
    )
    def test_impossible_time_keeps_the_date(self, text):
        got = parse_exif_datetime(text)
        assert got == datetime(*(int(x) for x in text[:10].split(":")))
        assert (got.hour, got.minute, got.second) == (0, 0, 0)

    def test_year_bounds(self):
        assert parse_exif_datetime("1900:01:01 00:00:00") is not None
        assert parse_exif_datetime("1899:12:31 23:59:59") is None
        assert parse_exif_datetime("2100:12:31 23:59:59") is not None
        assert parse_exif_datetime("2101:01:01 00:00:00") is None

    def test_result_is_naive(self):
        assert parse_exif_datetime("2021:05:06 07:08:09").tzinfo is None


class TestExifPrecedence:
    def test_original_wins(self, tmp_path):
        p = _jpeg(
            tmp_path / "a.jpg",
            _exif(
                original="2020:01:02 03:04:05",
                digitized="2021:01:01 00:00:00",
                datetime_="2022:01:01 00:00:00",
            ),
        )
        meta = read_metadata(p)
        assert meta.taken == datetime(2020, 1, 2, 3, 4, 5)
        assert meta.date_source == "exif"

    def test_digitized_beats_datetime(self, tmp_path):
        p = _jpeg(
            tmp_path / "a.jpg",
            _exif(digitized="2021:01:01 00:00:00", datetime_="2022:01:01 00:00:00"),
        )
        assert read_metadata(p).taken == datetime(2021, 1, 1)

    def test_datetime_only(self, tmp_path):
        p = _jpeg(tmp_path / "a.jpg", _exif(datetime_="2022:03:04 05:06:07"))
        meta = read_metadata(p)
        assert meta.taken == datetime(2022, 3, 4, 5, 6, 7)
        assert meta.date_source == "exif"

    def test_invalid_higher_priority_falls_through(self, tmp_path):
        p = _jpeg(
            tmp_path / "a.jpg",
            _exif(original="0000:00:00 00:00:00", datetime_="2022:03:04 05:06:07"),
        )
        assert read_metadata(p).taken == datetime(2022, 3, 4, 5, 6, 7)

    def test_original_in_base_ifd_is_found(self):
        base = {DATE_ORIGINAL: "2019:09:09 09:09:09"}
        assert md._capture_time(base, {}) == datetime(2019, 9, 9, 9, 9, 9)

    def test_all_blank_falls_back_to_mtime(self, tmp_path):
        p = _jpeg(tmp_path / "a.jpg", _exif(original="", datetime_="0000:00:00 00:00:00"))
        _set_mtime(p, datetime(2018, 8, 8, 8, 8, 8))
        meta = read_metadata(p)
        assert meta.date_source == "mtime"
        assert meta.taken == datetime(2018, 8, 8, 8, 8, 8)


# --------------------------------------------------------------------------- folder dates


class TestDateFromFolders:
    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (["2023", "07 - July", "trip"], (datetime(2023, 7, 1), "month")),
            (["2023", "07"], (datetime(2023, 7, 1), "month")),
            (["2023", "7"], (datetime(2023, 7, 1), "month")),
            (["2023", "07-Italy"], (datetime(2023, 7, 1), "month")),
            (["2023", "07 July"], (datetime(2023, 7, 1), "month")),
            (["2023", "12)"], (datetime(2023, 12, 1), "month")),
            (["2023", "July"], (datetime(2023, 7, 1), "month")),
            (["2023", "July - Italy"], (datetime(2023, 7, 1), "month")),
            (["2023", "sept"], (datetime(2023, 9, 1), "month")),
            (["2023-07"], (datetime(2023, 7, 1), "month")),
            (["2023_07"], (datetime(2023, 7, 1), "month")),
            (["2023.07"], (datetime(2023, 7, 1), "month")),
            (["2023-07 Beach"], (datetime(2023, 7, 1), "month")),
            (["2023-07-14"], (datetime(2023, 7, 14), "day")),
            (["2023_07_14 - Wedding"], (datetime(2023, 7, 14), "day")),
            (["Family", "2023-07-14 Beach", "raw"], (datetime(2023, 7, 14), "day")),
            (["Photos", "2019", "12 - December"], (datetime(2019, 12, 1), "month")),
            # a year folder alone is usable, at year precision
            (["2019"], (datetime(2019, 1, 1), "year")),
            (["2019", "Trip", "raw"], (datetime(2019, 1, 1), "year")),
            (["Photos", "2019"], (datetime(2019, 1, 1), "year")),
            (["2023-07-14Trip"], (datetime(2023, 7, 1), "month")),  # day glued to text: month only
            # month text that is not a month leaves the year
            (["2023", "10Photos"], (datetime(2023, 1, 1), "year")),
            (["2023", "2 weeks in Italy"], (datetime(2023, 1, 1), "year")),
            (["2023", "10 days in Rome"], (datetime(2023, 1, 1), "year")),
            (["2023", "13"], (datetime(2023, 1, 1), "year")),
            (["2023", "00"], (datetime(2023, 1, 1), "year")),
            (["2023", "Vacation"], (datetime(2023, 1, 1), "year")),
            (["2023", "July trip"], (datetime(2023, 1, 1), "year")),
            (["2023", "Mayhem"], (datetime(2023, 1, 1), "year")),
            # an impossible day falls back to the month, never to the 1st at day precision
            (["2023-02-30"], (datetime(2023, 2, 1), "month")),
            (["2023-02", "x"], (datetime(2023, 2, 1), "month")),
            # most specific wins, regardless of position
            (["2023", "07", "2023-07-14"], (datetime(2023, 7, 14), "day")),
            (["2019", "2023-07"], (datetime(2023, 7, 1), "month")),
            (["2023-07", "2019"], (datetime(2023, 7, 1), "month")),
            # equally specific: the outermost wins
            (["2023", "07", "2022-03"], (datetime(2023, 7, 1), "month")),
            (["2019", "2020"], (datetime(2019, 1, 1), "year")),
        ],
    )
    def test_recognised(self, parts, expected):
        assert date_from_folders(parts) == expected

    @pytest.mark.parametrize(
        "parts",
        [
            [],
            ["Vacation"],
            ["2023-13"],
            ["2023-10Photos"],
            ["1850", "07"],
            ["1850"],
            ["2023-0712"],
            ["20230714"],
            ["12", "2023-"],
            ["12", "2023 trip"],
            ["10Photos"],
            ["07"],
        ],
    )
    def test_not_recognised(self, parts):
        assert date_from_folders(parts) is None

    def test_invalid_day_in_a_real_file(self, tmp_path):
        p = tmp_path / "2023-02-30" / "a.png"
        p.parent.mkdir()
        Image.new("RGB", (8, 8)).save(p)
        meta = read_metadata(p, tmp_path)
        assert (meta.taken, meta.date_source, meta.date_precision) == (
            datetime(2023, 2, 1),
            "folder",
            "month",
        )


class TestPrecision:
    def test_exif_is_day(self, tmp_path):
        p = _jpeg(tmp_path / "2019" / "a.jpg", _exif(original="2020:01:02 03:04:05"))
        meta = read_metadata(p, tmp_path)
        assert (meta.date_source, meta.date_precision) == ("exif", "day")

    @pytest.mark.parametrize(
        ("folders", "taken", "precision"),
        [
            (("2023-07-14", "x"), datetime(2023, 7, 14), "day"),
            (("2023", "07 - July"), datetime(2023, 7, 1), "month"),
            (("2023-07",), datetime(2023, 7, 1), "month"),
            (("2019", "Trip"), datetime(2019, 1, 1), "year"),
            (("2019",), datetime(2019, 1, 1), "year"),
        ],
    )
    def test_folder_precision(self, tmp_path, folders, taken, precision):
        p = tmp_path.joinpath(*folders, "a.png")
        p.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(p)
        meta = read_metadata(p, tmp_path)
        assert (meta.taken, meta.date_source, meta.date_precision) == (taken, "folder", precision)

    def test_mtime_is_day_precision(self, tmp_path):
        p = tmp_path / "a.png"
        Image.new("RGB", (8, 8)).save(p)
        meta = read_metadata(p, tmp_path)
        assert (meta.date_source, meta.date_precision) == ("mtime", "day")

    def test_year_folder_beats_mtime_but_not_exif(self, tmp_path):
        p = _jpeg(tmp_path / "2019" / "a.jpg")
        _set_mtime(p, datetime(2024, 1, 1))
        assert read_metadata(p, tmp_path).taken == datetime(2019, 1, 1)


class TestFolderFallback:
    def test_png_without_exif_uses_folder(self, tmp_path):
        p = tmp_path / "2023" / "07 - July" / "trip" / "a.png"
        p.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(p)
        meta = read_metadata(p, tmp_path)
        assert meta.date_source == "folder"
        assert meta.taken == datetime(2023, 7, 1)
        assert (meta.lat, meta.lon, meta.camera) == (None, None, None)

    def test_day_folder(self, tmp_path):
        p = tmp_path / "2023-07-14" / "a.png"
        p.parent.mkdir()
        Image.new("RGB", (8, 8)).save(p)
        assert read_metadata(p, tmp_path).taken == datetime(2023, 7, 14)

    def test_exif_beats_folder(self, tmp_path):
        p = _jpeg(tmp_path / "2023" / "07" / "a.jpg", _exif(original="2020:02:03 04:05:06"))
        meta = read_metadata(p, tmp_path)
        assert (meta.taken, meta.date_source) == (datetime(2020, 2, 3, 4, 5, 6), "exif")

    def test_no_library_root_skips_folder_names(self, tmp_path):
        p = _jpeg(tmp_path / "2023-07-14" / "a.jpg")
        assert read_metadata(p).date_source == "mtime"

    def test_folder_names_above_library_root_are_ignored(self, tmp_path):
        root = tmp_path / "2019" / "05" / "library"
        p = _jpeg(root / "plain" / "a.jpg")
        assert read_metadata(p, root).date_source == "mtime"

    def test_file_outside_library_root_is_ignored(self, tmp_path):
        p = _jpeg(tmp_path / "2023-07-14" / "a.jpg")
        other = tmp_path / "elsewhere"
        other.mkdir()
        assert read_metadata(p, other).date_source == "mtime"

    def test_file_directly_in_root(self, tmp_path):
        p = _jpeg(tmp_path / "a.jpg")
        assert read_metadata(p, tmp_path).date_source == "mtime"

    def test_symlinked_library_root_with_resolved_file_path(self, tmp_path):
        real = tmp_path / "real"
        p = _jpeg(real / "2023" / "07 - July" / "a.jpg")
        link = tmp_path / "link"
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        for photo, root in ((p, link), (link / "2023" / "07 - July" / "a.jpg", real)):
            meta = read_metadata(photo, root)
            assert (meta.taken, meta.date_source) == (datetime(2023, 7, 1), "folder")

    def test_outside_root_still_ignored_when_symlinks_involved(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        p = _jpeg(tmp_path / "2023-07-14" / "a.jpg")
        link = tmp_path / "link"
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        assert read_metadata(p, link).date_source == "mtime"

    def test_file_name_is_not_used(self, tmp_path):
        p = _jpeg(tmp_path / "2023-07-14.jpg")
        assert read_metadata(p, tmp_path).date_source == "mtime"

    def test_accepts_str_and_pathlike_root(self, tmp_path):
        p = tmp_path / "2023-07" / "a.png"
        p.parent.mkdir()
        Image.new("RGB", (8, 8)).save(p)
        assert read_metadata(str(p), str(tmp_path)).taken == datetime(2023, 7, 1)


# --------------------------------------------------------------------------- mtime fallback


class TestMtime:
    def test_png_without_exif_uses_mtime(self, tmp_path):
        p = tmp_path / "a.png"
        Image.new("RGB", (8, 8)).save(p)
        when = datetime(2017, 4, 5, 6, 7, 8)
        _set_mtime(p, when)
        meta = read_metadata(p, tmp_path)
        assert meta.date_source == "mtime"
        assert meta.taken == when
        assert meta.mtime_ns == p.stat().st_mtime_ns

    def test_unusable_mtime_gives_none(self, tmp_path):
        p = tmp_path / "a.png"
        Image.new("RGB", (8, 8)).save(p)
        os.utime(p, ns=(0, 0))
        meta = read_metadata(p)
        assert meta.taken is None
        assert meta.date_source == "none"

    def test_mtime_overflow_gives_none(self, monkeypatch):
        class Boom:
            @staticmethod
            def fromtimestamp(_):
                raise OverflowError

        monkeypatch.setattr(md, "datetime", Boom)
        assert md._mtime(10**18) is None


# --------------------------------------------------------------------------- GPS


class TestGps:
    @pytest.mark.parametrize(
        ("lat_ref", "lon_ref", "lat_sign", "lon_sign"),
        [("N", "E", 1, 1), ("N", "W", 1, -1), ("S", "E", -1, 1), ("S", "W", -1, -1)],
    )
    def test_hemispheres(self, tmp_path, lat_ref, lon_ref, lat_sign, lon_sign):
        gps = _gps((40.0, 30.0, 0.0), lat_ref, (73.0, 15.0, 36.0), lon_ref)
        meta = read_metadata(_jpeg(tmp_path / "a.jpg", _exif(gps=gps)))
        assert meta.lat == pytest.approx(lat_sign * 40.5)
        assert meta.lon == pytest.approx(lon_sign * 73.26)

    def test_dms_rationals_survive_jpeg_roundtrip(self, tmp_path):
        gps = _gps((40.0, 26.0, 46.302), "N", (79.0, 58.0, 55.903), "W")
        meta = read_metadata(_jpeg(tmp_path / "a.jpg", _exif(gps=gps)))
        assert meta.lat == pytest.approx(40 + 26 / 60 + 46.302 / 3600, abs=1e-5)
        assert meta.lon == pytest.approx(-(79 + 58 / 60 + 55.903 / 3600), abs=1e-5)

    def test_lowercase_bytes_and_nul_refs(self):
        gps = _gps((10.0, 0.0, 0.0), b"s\x00", (20.0, 0.0, 0.0), " w ")
        assert parse_gps(gps) == (-10.0, -20.0)

    def test_tuple_floats_ints_and_single_value_forms(self):
        assert parse_gps(_gps((10, 30, 0), "N", (20, 0, 0), "E")) == (10.5, 20.0)
        assert parse_gps(_gps((10.5, 0.0), "N", (20.0,), "E")) == (10.5, 20.0)
        assert parse_gps(_gps(10.25, "N", 20.5, "E")) == (10.25, 20.5)
        assert parse_gps(_gps([1.0, 30.0, 0.0], "S", [2.0, 0.0, 0.0], "W")) == (-1.5, -2.0)

    def test_rational_like_objects(self):
        from fractions import Fraction

        gps = _gps((Fraction(40), Fraction(30), Fraction(0)), "N", (Fraction(5),) * 3, "E")
        lat, lon = parse_gps(gps)
        assert lat == pytest.approx(40.5)
        assert lon == pytest.approx(5 + 5 / 60 + 5 / 3600)

    def test_real_ifd_rational_with_zero_denominator(self):
        from PIL.TiffImagePlugin import IFDRational

        bad = IFDRational(1, 0)
        assert parse_gps(_gps((bad, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E")) is None

    @pytest.mark.parametrize(
        "gps",
        [
            {},
            _gps((91.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E"),  # lat out of range
            _gps((10.0, 0.0, 0.0), "N", (181.0, 0.0, 0.0), "E"),  # lon out of range
            _gps((89.0, 59.0, 70.0), "N", (10.0, 0.0, 0.0), "E"),  # rounds above 90
            _gps((-10.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E"),  # negative component
            _gps((10.0, 0.0, 0.0), "X", (10.0, 0.0, 0.0), "E"),  # bad ref
            _gps((10.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "N"),  # lon ref is a lat ref
            _gps((10.0, 0.0, 0.0), None, (10.0, 0.0, 0.0), "E"),  # missing ref
            _gps((10.0, 0.0, 0.0), 5, (10.0, 0.0, 0.0), "E"),  # non-text ref
            _gps((10.0, 0.0, 0.0), "N", None, "E"),  # missing value
            _gps((10.0, 0.0, 0.0), "N", "abc", "E"),
            _gps((1.0, 2.0, 3.0, 4.0), "N", (10.0, 0.0, 0.0), "E"),  # too many parts
            _gps((), "N", (10.0, 0.0, 0.0), "E"),  # no parts
            _gps((float("nan"), 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E"),
            _gps((float("inf"), 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E"),
            _gps((None, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E"),
            _gps(-5.0, "N", 10.0, "E"),  # negative bare number
            {1: "N", 2: (10.0, 0.0, 0.0)},  # longitude absent entirely
        ],
    )
    def test_rejected(self, gps):
        assert parse_gps(gps) is None

    @pytest.mark.parametrize(
        "lat", [(10.0, 60.0, 0.0), (10.0, 0.0, 60.0), (10.0, 75.0, 0.0), (10.0, 0.0, 3600.0)]
    )
    def test_minutes_or_seconds_of_sixty_or_more_are_rejected(self, lat):
        assert parse_gps(_gps(lat, "N", (10.0, 0.0, 0.0), "E")) is None
        assert parse_gps(_gps((10.0, 0.0, 0.0), "N", lat, "E")) is None

    def test_just_below_sixty_is_fine(self):
        got = parse_gps(_gps((10.0, 59.0, 59.9), "N", (10.0, 0.0, 0.0), "E"))
        assert got[0] == pytest.approx(10 + 59 / 60 + 59.9 / 3600)

    def test_exact_zero_zero_is_missing(self):
        assert parse_gps(_gps((0.0, 0.0, 0.0), "N", (0.0, 0.0, 0.0), "E")) is None
        assert parse_gps(_gps((0.0, 0.0, 0.0), "S", (0.0, 0.0, 0.0), "W")) is None

    def test_zero_on_one_axis_is_valid(self):
        assert parse_gps(_gps((0.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E")) == (0.0, 10.0)

    def test_limits_are_inclusive(self):
        assert parse_gps(_gps((90.0, 0.0, 0.0), "S", (180.0, 0.0, 0.0), "W")) == (-90.0, -180.0)

    def test_bad_gps_leaves_other_metadata_intact(self, tmp_path):
        gps = _gps((95.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E")
        p = _jpeg(tmp_path / "a.jpg", _exif(original="2020:01:02 03:04:05", make="Canon", gps=gps))
        meta = read_metadata(p)
        assert (meta.lat, meta.lon) == (None, None)
        assert meta.taken == datetime(2020, 1, 2, 3, 4, 5)
        assert meta.camera == "Canon"


# --------------------------------------------------------------------------- camera, size


class TestCamera:
    @pytest.mark.parametrize(
        ("make", "model", "expected"),
        [
            ("Apple", "iPhone 12 Pro", "Apple iPhone 12 Pro"),
            ("Canon", "Canon EOS 5D Mark IV", "Canon EOS 5D Mark IV"),
            ("CANON", "Canon EOS R5", "Canon EOS R5"),
            ("NIKON CORPORATION", "NIKON D850", "NIKON CORPORATION NIKON D850"),
            ("  Sony  ", " ILCE-7M3 ", "Sony ILCE-7M3"),
            ("Apple", None, "Apple"),
            (None, "Pixel 7", "Pixel 7"),
            ("Apple", "  ", "Apple"),
            ("", "", None),
            (None, None, None),
        ],
    )
    def test_make_model(self, tmp_path, make, model, expected):
        p = _jpeg(tmp_path / "a.jpg", _exif(make=make, model=model, original="2020:01:01 00:00:00"))
        assert read_metadata(p).camera == expected

    def test_bytes_and_nul(self):
        assert md._camera({0x010F: b"Leica\x00", 0x0110: b"Q2\x00"}) == "Leica Q2"

    def test_non_text_values_ignored(self):
        assert md._camera({0x010F: 5, 0x0110: 7}) is None


class TestSize:
    def test_plain(self, tmp_path):
        meta = read_metadata(_jpeg(tmp_path / "a.jpg", size=(40, 20)))
        assert (meta.width, meta.height) == (40, 20)

    @pytest.mark.parametrize("orientation", [1, 2, 3, 4])
    def test_non_rotating_orientations(self, tmp_path, orientation):
        p = _jpeg(tmp_path / "a.jpg", _exif(orientation=orientation), size=(40, 20))
        meta = read_metadata(p)
        assert (meta.width, meta.height) == (40, 20)

    @pytest.mark.parametrize("orientation", [5, 6, 7, 8])
    def test_rotating_orientations_swap(self, tmp_path, orientation):
        p = _jpeg(tmp_path / "a.jpg", _exif(orientation=orientation), size=(40, 20))
        meta = read_metadata(p)
        assert (meta.width, meta.height) == (20, 40)

    def test_bogus_orientation_ignored(self):
        assert md._displayed_size(4, 3, 99) == (4, 3)
        assert md._displayed_size(4, 3, "6") == (4, 3)
        assert md._displayed_size(4, 3, None) == (4, 3)

    def test_png_size(self, tmp_path):
        p = tmp_path / "a.png"
        Image.new("RGB", (12, 7)).save(p)
        meta = read_metadata(p)
        assert (meta.width, meta.height) == (12, 7)


# --------------------------------------------------------------------------- the result


class TestPhotoMeta:
    def test_fields(self, tmp_path):
        p = _jpeg(
            tmp_path / "a.jpg",
            _exif(
                original="2020:01:02 03:04:05",
                make="Apple",
                model="iPhone 12",
                gps=_gps((51.0, 30.0, 0.0), "N", (0.0, 7.5, 0.0), "W"),
            ),
        )
        meta = read_metadata(p)
        assert isinstance(meta, PhotoMeta)
        assert meta.path == str(p)
        assert os.path.isabs(meta.path)
        assert meta.size == p.stat().st_size
        assert meta.mtime_ns == p.stat().st_mtime_ns
        assert meta.camera == "Apple iPhone 12"
        assert meta.lat == pytest.approx(51.5)
        assert meta.lon == pytest.approx(-0.125)

    def test_path_is_canonical_abspath_without_symlink_resolution(self, tmp_path):
        real = tmp_path / "real"
        _jpeg(real / "a.jpg")
        link = tmp_path / "link"
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available")
        messy = f"{link}{os.sep}.{os.sep}sub{os.sep}..{os.sep}a.jpg"
        assert read_metadata(messy).path == str(link / "a.jpg")

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        _jpeg(tmp_path / "a.jpg")
        assert read_metadata("~/a.jpg").path == str(tmp_path / "a.jpg")
        assert read_metadata(Path("~") / "a.jpg").path == str(tmp_path / "a.jpg")

    def test_matches_iter_library_paths(self, tmp_path, monkeypatch):
        from facesorter.search.library import iter_library

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        _jpeg(tmp_path / "lib" / "2023" / "a.jpg")
        _jpeg(tmp_path / "lib" / "b.jpg")
        for root in (tmp_path / "lib", f"{tmp_path / 'lib'}{os.sep}", "~/lib", "~/lib/", "~/lib/."):
            paths = list(iter_library(root))
            assert len(paths) == 2
            for path in paths:
                assert isinstance(path, Path)
                assert read_metadata(path, root).path == str(path)
                assert read_metadata(str(path)).path == str(path)

    def test_relative_path_is_made_absolute(self, tmp_path, monkeypatch):
        _jpeg(tmp_path / "a.jpg")
        monkeypatch.chdir(tmp_path)
        assert read_metadata("a.jpg").path == str(tmp_path / "a.jpg")

    def test_does_not_decode_pixels(self, tmp_path, monkeypatch):
        p = _jpeg(tmp_path / "a.jpg", _exif(original="2020:01:02 03:04:05"))

        def boom(self, *args, **kwargs):
            raise AssertionError("pixels were decoded")

        monkeypatch.setattr(Image.Image, "load", boom)
        assert read_metadata(p).taken == datetime(2020, 1, 2, 3, 4, 5)

    def test_truncated_jpeg_body_still_reads_header(self, tmp_path):
        p = _jpeg(tmp_path / "a.jpg", _exif(original="2020:01:02 03:04:05"), size=(300, 200))
        data = p.read_bytes()
        p.write_bytes(data[: len(data) // 2])
        meta = read_metadata(p)
        assert (meta.width, meta.height) == (300, 200)


# --------------------------------------------------------------------------- failures


class TestUnreadable:
    def test_missing_file(self, tmp_path):
        with pytest.raises(UnreadableImageError, match="missing.jpg"):
            read_metadata(tmp_path / "missing.jpg")

    def test_not_an_image(self, tmp_path):
        p = tmp_path / "a.jpg"
        p.write_text("definitely not a jpeg")
        with pytest.raises(UnreadableImageError) as err:
            read_metadata(p)
        assert str(p) in str(err.value)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "a.png"
        p.write_bytes(b"")
        with pytest.raises(UnreadableImageError, match="a.png"):
            read_metadata(p)

    def test_corrupt_png_header(self, tmp_path):
        p = tmp_path / "a.png"
        Image.new("RGB", (8, 8)).save(p)
        p.write_bytes(p.read_bytes()[:20])
        with pytest.raises(UnreadableImageError):
            read_metadata(p)

    def test_corrupt_heic(self, tmp_path):
        pytest.importorskip("pillow_heif")
        p = tmp_path / "a.heic"
        p.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 40)
        with pytest.raises(UnreadableImageError, match="a.heic"):
            read_metadata(p)

    def test_directory(self, tmp_path):
        with pytest.raises(UnreadableImageError):
            read_metadata(tmp_path)

    def test_is_a_search_error(self, tmp_path):
        from facesorter.errors import FaceSorterError

        with pytest.raises(FaceSorterError):
            read_metadata(tmp_path / "nope.jpg")


class TestMalformedExif:
    def test_getexif_failure_is_not_fatal(self, tmp_path, monkeypatch):
        p = _jpeg(tmp_path / "a.jpg")

        def boom(self):
            raise ValueError("broken exif")

        monkeypatch.setattr(Image.Image, "getexif", boom)
        meta = read_metadata(p)
        assert meta.date_source == "mtime"
        assert (meta.width, meta.height) == (40, 20)

    def test_sub_ifd_failure_keeps_base_ifd(self, tmp_path, monkeypatch):
        p = _jpeg(tmp_path / "a.jpg", _exif(make="Canon", datetime_="2022:03:04 05:06:07"))

        def boom(self, tag):
            raise SyntaxError("bad ifd")

        monkeypatch.setattr(Image.Exif, "get_ifd", boom)
        meta = read_metadata(p)
        assert meta.camera == "Canon"
        assert meta.taken == datetime(2022, 3, 4, 5, 6, 7)
        assert meta.lat is None

    @pytest.mark.filterwarnings("ignore:Corrupt EXIF")
    def test_garbage_exif_segment(self, tmp_path):
        p = tmp_path / "a.jpg"
        Image.new("RGB", (10, 10)).save(p)
        data = p.read_bytes()
        segment = b"Exif\x00\x00" + b"II*\x00\xff\xff\xff\xff" + b"\x01" * 30
        length = (len(segment) + 2).to_bytes(2, "big")
        p.write_bytes(data[:2] + b"\xff\xe1" + length + segment + data[2:])
        meta = read_metadata(p)  # must not raise
        assert (meta.width, meta.height) == (10, 10)


# --------------------------------------------------------------------------- HEIC


class TestHeic:
    def test_heic_with_exif(self, tmp_path):
        pytest.importorskip("pillow_heif")
        p = tmp_path / "a.heic"
        exif = _exif(
            original="2021:05:06 07:08:09",
            make="Apple",
            model="iPhone 13",
            gps=_gps((25.0, 45.0, 0.0), "N", (80.0, 15.0, 0.0), "W"),
        )
        try:
            Image.new("RGB", (64, 32), (10, 200, 10)).save(p, format="HEIF", exif=exif)
        except Exception as exc:  # pragma: no cover - encoder missing in this build
            pytest.skip(f"cannot write HEIC here: {exc}")
        meta = read_metadata(p)
        assert meta.taken == datetime(2021, 5, 6, 7, 8, 9)
        assert meta.date_source == "exif"
        assert meta.camera == "Apple iPhone 13"
        assert meta.lat == pytest.approx(25.75)
        assert meta.lon == pytest.approx(-80.25)
        assert (meta.width, meta.height) == (64, 32)

    @pytest.mark.parametrize("orientation", [1, 3, 6, 8])
    def test_heic_orientation_not_swapped_twice(self, tmp_path, orientation):
        # pillow_heif reports the displayed size with orientation already normalised, so the
        # EXIF orientation tag written into the file must not swap width and height again.
        pytest.importorskip("pillow_heif")
        p = tmp_path / "a.heic"
        try:
            Image.new("RGB", (64, 32)).save(
                p,
                format="HEIF",
                exif=_exif(orientation=orientation, original="2021:05:06 07:08:09"),
            )
        except Exception as exc:  # pragma: no cover - encoder missing in this build
            pytest.skip(f"cannot write HEIC here: {exc}")
        with Image.open(p) as probe:
            displayed = probe.size
            tag = probe.getexif().get(0x0112, 1)
        meta = read_metadata(p)
        assert (meta.width, meta.height) == (displayed if tag in (1, 2, 3, 4) else displayed[::-1])
        assert (meta.width, meta.height) == (64, 32)

    def test_register_is_idempotent(self):
        md._register_heif()
        md._register_heif()
        assert md._heif_registered
