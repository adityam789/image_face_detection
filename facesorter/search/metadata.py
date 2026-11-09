"""
facesorter/search/metadata.py
Read capture time, GPS position, camera and size from a still image without decoding pixels.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from .errors import UnreadableImageError
from .types import PhotoMeta

logger = logging.getLogger(__name__)

# EXIF tag ids.
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_ORIENTATION = 0x0112
_TAG_DATETIME = 0x0132
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME_DIGITIZED = 0x9004
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825
# GPS IFD tag ids.
_GPS_LAT_REF, _GPS_LAT, _GPS_LON_REF, _GPS_LON = 1, 2, 3, 4

_MIN_YEAR = 1900
_MAX_YEAR = 2100

_heif_registered = False


def _register_heif() -> None:
    """Register the HEIC/HEIF opener with PIL once (lazy: pillow_heif is only needed here)."""
    global _heif_registered
    if _heif_registered:
        return
    _heif_registered = True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.warning("HEIC support unavailable: %s", exc)


# ------------------------------------------------------------------------------ EXIF dates

_EXIF_DATE_RE = re.compile(
    r"^\s*(\d{4})[:\-/.](\d{1,2})[:\-/.](\d{1,2})"
    r"(?:[ T]+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?"
)


def parse_exif_datetime(value: object) -> datetime | None:
    """
    Parse an EXIF date string (``"YYYY:MM:DD HH:MM:SS"``) into a naive ``datetime``.

    Tolerates other separators, a missing time and trailing junk (sub-seconds, time zones,
    NUL padding). Blank values, the all-zero placeholder and impossible dates give ``None``;
    a valid date with an impossible time of day (``25:61:61``) keeps the date at 00:00:00.
    """
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    if not isinstance(value, str):
        return None
    match = _EXIF_DATE_RE.match(value.replace("\x00", " "))
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(g) if g else 0 for g in match.groups())
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        pass
    try:  # impossible time of day (e.g. 25:61:61): the date itself is still useful
        return datetime(year, month, day)
    except ValueError:
        return None


def _capture_time(base: Mapping[int, Any], sub: Mapping[int, Any]) -> datetime | None:
    """DateTimeOriginal > DateTimeDigitized > DateTime; the first parsable one wins."""
    for tag, source in (
        (_TAG_DATETIME_ORIGINAL, sub),
        (_TAG_DATETIME_ORIGINAL, base),
        (_TAG_DATETIME_DIGITIZED, sub),
        (_TAG_DATETIME_DIGITIZED, base),
        (_TAG_DATETIME, base),
        (_TAG_DATETIME, sub),
    ):
        parsed = parse_exif_datetime(source.get(tag))
        if parsed is not None:
            return parsed
    return None


# ------------------------------------------------------------------------------ folder dates

_MONTH_NAMES = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
# A folder name that starts with a date must not run straight into other text ("10Photos").
_BOUNDARY = r"(?=$|[\s\-_.)])"
_FOLDER_DAY_RE = re.compile(r"^(\d{4})[-_.](\d{1,2})[-_.](\d{1,2})" + _BOUNDARY)
_FOLDER_MONTH_RE = re.compile(r"^(\d{4})[-_.](\d{1,2})" + _BOUNDARY)
_FOLDER_YEAR_RE = re.compile(r"^(\d{4})$")
# Month folder after a year folder: "07", "7", "07 - July", "07-Italy", "07 July" (but not
# "10Photos" or "2 weeks in Italy": digits followed by a plain space need a month name).
_FOLDER_MONTH_NUM_RE = re.compile(
    r"^(\d{1,2})(?:$|\s*[-_.)]|\s+(?=(?:"
    + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
    + r")\b))",
    re.IGNORECASE,
)
_FOLDER_MONTH_NAME_RE = re.compile(r"^([A-Za-z]+)\s*(?:$|[-_.,)])")

_Precision = Literal["day", "month", "year"]


def _make_date(year: int, month: int = 1, day: int = 1) -> datetime | None:
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def date_from_folders(parts: list[str]) -> tuple[datetime, _Precision] | None:
    """
    Guess a date from folder names (outermost first), e.g. ``["2023", "07 - July", "trip"]``.

    Returns ``(date, precision)`` or ``None``. Recognised, from most to least specific:

    * ``YYYY-MM-DD`` in one folder name (also ``_`` / ``.``): precision ``"day"``. An
      impossible day (``2023-02-30``) falls back to the month: ``(2023-02-01, "month")``.
    * ``YYYY-MM`` in one folder name, or a year folder followed by a month folder
      (``07``, ``07 - July``, ``July``): precision ``"month"``, on the 1st at 00:00.
    * A bare year folder (``2019``): precision ``"year"``, on January 1st at 00:00.

    Digits must not run into other text: ``10Photos`` and ``2 weeks in Italy`` are not months.
    The most specific match wins; ties go to the outermost folder.
    """
    rank = {"day": 3, "month": 2, "year": 1}
    best: tuple[datetime, _Precision] | None = None
    for i, part in enumerate(parts):
        found: tuple[datetime, _Precision] | None = None
        if m := _FOLDER_DAY_RE.match(part):
            value = _make_date(int(m[1]), int(m[2]), int(m[3]))
            found = (value, "day") if value else None
        if found is None and (m := _FOLDER_MONTH_RE.match(part)):
            value = _make_date(int(m[1]), int(m[2]))
            found = (value, "month") if value else None
        if found is None and (m := _FOLDER_YEAR_RE.match(part)):
            month = _month_of(parts[i + 1]) if i + 1 < len(parts) else None
            value = _make_date(int(m[1]), month or 1)
            if value:
                found = (value, "month" if month else "year")
        if found is not None and (best is None or rank[found[1]] > rank[best[1]]):
            best = found
    return best


def _month_of(folder: str) -> int | None:
    """Month number from a folder named ``07``, ``7 - July`` or ``July``; else ``None``."""
    m = _FOLDER_MONTH_NUM_RE.match(folder)
    if m:
        month = int(m[1])
        return month if 1 <= month <= 12 else None
    m = _FOLDER_MONTH_NAME_RE.match(folder)
    return _MONTH_NAMES.get(m[1].lower()) if m else None


def _folder_date(
    path: str, library_root: str | os.PathLike[str] | None
) -> tuple[datetime, _Precision] | None:
    if library_root is None:
        return None
    root = os.path.abspath(os.path.expanduser(os.fspath(library_root)))
    parts = _relative_parts(os.path.dirname(path), root)
    if parts is None:  # maybe only one of the two goes through a symlink
        parts = _relative_parts(os.path.realpath(os.path.dirname(path)), os.path.realpath(root))
    return date_from_folders(parts) if parts is not None else None


def _relative_parts(folder: str, root: str) -> list[str] | None:
    """Folder names of ``folder`` below ``root``, or ``None`` if it is not inside ``root``."""
    rel = os.path.relpath(folder, root)
    parts = [p for p in rel.split(os.sep) if p and p != os.curdir]
    return None if parts[:1] == [os.pardir] else parts


# ------------------------------------------------------------------------------ GPS


def _to_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]  # IFDRational / int / float / str
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def _dms_to_degrees(value: object) -> float | None:
    """Degrees from ``(deg, min, sec)`` (or fewer parts / a bare number).

    Negative components and minutes or seconds of 60 or more give ``None``.
    """
    if isinstance(value, (tuple, list)):
        parts = [_to_float(v) for v in value]
        if not 1 <= len(parts) <= 3 or any(p is None or p < 0 for p in parts):
            return None
        if any(p >= 60 for p in parts[1:]):  # minutes / seconds
            return None
        degrees = 0.0
        for divisor, part in zip((1.0, 60.0, 3600.0), parts):
            degrees += (part or 0.0) / divisor
        return degrees
    number = _to_float(value)
    return number if number is not None and number >= 0 else None


def _signed(value: object, ref: object, positive: str, negative: str, limit: float) -> float | None:
    degrees = _dms_to_degrees(value)
    if degrees is None:
        return None
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", errors="ignore")
    if not isinstance(ref, str):
        return None
    ref = ref.replace("\x00", "").strip().upper()
    if ref == positive:
        sign = 1.0
    elif ref == negative:
        sign = -1.0
    else:
        return None
    result = sign * degrees
    return result if abs(result) <= limit else None


def parse_gps(gps: Mapping[int, Any]) -> tuple[float, float] | None:
    """
    Decimal ``(lat, lon)`` from an EXIF GPS IFD, or ``None`` when absent or implausible.

    Out-of-range values, missing or unknown N/S/E/W references and the exact ``(0, 0)``
    "no fix" placeholder are treated as missing.
    """
    if not gps:
        return None
    lat = _signed(gps.get(_GPS_LAT), gps.get(_GPS_LAT_REF), "N", "S", 90.0)
    lon = _signed(gps.get(_GPS_LON), gps.get(_GPS_LON_REF), "E", "W", 180.0)
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
        return None
    return lat, lon


# ------------------------------------------------------------------------------ camera, size


def _clean_text(value: object) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if not isinstance(value, str):
        return None
    text = value.replace("\x00", " ").strip()
    return text or None


def _camera(base: Mapping[int, Any]) -> str | None:
    make = _clean_text(base.get(_TAG_MAKE))
    model = _clean_text(base.get(_TAG_MODEL))
    if make and model:
        return model if model.lower().startswith(make.lower()) else f"{make} {model}"
    return make or model


def _displayed_size(width: int, height: int, orientation: object) -> tuple[int, int]:
    """Swap width and height for the EXIF orientations that rotate by 90 degrees (5-8)."""
    if isinstance(orientation, int) and 5 <= orientation <= 8:
        return height, width
    return width, height


# ------------------------------------------------------------------------------ public API


def read_metadata(
    path: str | os.PathLike[str], library_root: str | os.PathLike[str] | None = None
) -> PhotoMeta:
    """
    Read the metadata of one still image through PIL, touching only headers and EXIF.

    The capture time comes from EXIF (DateTimeOriginal, then DateTimeDigitized, then
    DateTime), else from the folder name relative to ``library_root`` (``2023/07 - July``,
    ``2023-07-14`` ...), else from the file's modification time; ``date_source`` records which.
    ``width``/``height`` are the displayed size (EXIF orientation applied).

    Missing or malformed EXIF never raises. Raises ``UnreadableImageError`` (naming the
    path) if the file is missing, not an image, or corrupt.
    """
    from PIL import Image

    _register_heif()
    abs_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    try:
        stat = os.stat(abs_path)
    except OSError as exc:
        raise UnreadableImageError(f"Cannot read image {abs_path}: {exc}") from exc

    try:
        with Image.open(abs_path) as img:
            width, height = img.size
            base, sub, gps = _read_exif(img, abs_path)
    except Exception as exc:  # PIL raises OSError, SyntaxError, ValueError, ... on bad files
        raise UnreadableImageError(f"Cannot read image {abs_path}: {exc}") from exc

    taken = _capture_time(base, sub)
    source = "exif"
    precision: _Precision = "day"
    if taken is None:
        folder = _folder_date(abs_path, library_root)
        if folder is not None:
            taken, precision = folder
            source = "folder"
    if taken is None:
        taken, source = _mtime(stat.st_mtime_ns), "mtime"
    if taken is None:
        source = "none"

    coords = parse_gps(gps)
    width, height = _displayed_size(width, height, base.get(_TAG_ORIENTATION))
    return PhotoMeta(
        path=abs_path,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        width=width,
        height=height,
        taken=taken,
        date_source=source,  # type: ignore[arg-type]
        lat=coords[0] if coords else None,
        lon=coords[1] if coords else None,
        camera=_camera(base),
        date_precision=precision,
    )


def _read_exif(img: Any, path: str) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
    """Base, Exif and GPS IFDs as plain dicts; a malformed part comes back empty."""
    try:
        exif = img.getexif()
        base = dict(exif)
    except Exception as exc:
        logger.debug("Ignoring malformed EXIF in %s: %s", path, exc)
        return {}, {}, {}
    return base, _sub_ifd(exif, _IFD_EXIF, path), _sub_ifd(exif, _IFD_GPS, path)


def _sub_ifd(exif: Any, tag: int, path: str) -> dict[int, Any]:
    try:
        return dict(exif.get_ifd(tag))
    except Exception as exc:
        logger.debug("Ignoring malformed EXIF IFD 0x%04x in %s: %s", tag, path, exc)
        return {}


def _mtime(mtime_ns: int) -> datetime | None:
    if mtime_ns <= 0:
        return None
    try:
        return datetime.fromtimestamp(mtime_ns / 1e9)
    except (OverflowError, OSError, ValueError):
        return None
