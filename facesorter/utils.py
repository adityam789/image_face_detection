"""
facesorter/utils.py
Filesystem and image-loading helpers shared by the sort and find pipelines.
"""

from __future__ import annotations

import os
import shutil

import pillow_heif
from PIL import Image, ImageOps

from .errors import InvalidPathError
from .paths import is_inside, same_path

pillow_heif.register_heif_opener()

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic")


def is_image_file(filename: str) -> bool:
    """Return True if the filename has a supported image extension (case-insensitive)."""
    return filename.lower().endswith(SUPPORTED_EXTENSIONS)


def load_image(image_path: str) -> Image.Image:
    """
    Load an image as RGB with its EXIF orientation applied.

    Phone photos are often stored sideways with an EXIF orientation tag; without
    transposing, the face detector would see rotated faces. HEIC is supported
    through pillow_heif.
    """
    with Image.open(image_path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def unique_filename(filename: str, taken: set[str]) -> str:
    """
    Return a filename that does not collide with any name in ``taken``.

    The first use of a name is returned unchanged; later collisions get a numeric
    suffix (``photo.jpg`` -> ``photo_1.jpg``, ``photo_2.jpg``, ...). Comparison is
    case-insensitive so results are also safe on case-insensitive filesystems.
    The chosen name is added to ``taken``. Given the same call order the result is
    deterministic.
    """
    stem, ext = os.path.splitext(filename)
    candidate = filename
    counter = 0
    while candidate.casefold() in taken:
        counter += 1
        candidate = f"{stem}_{counter}{ext}"
    taken.add(candidate.casefold())
    return candidate


def copy_image(src_path: str, dest_dir: str, filename: str) -> str:
    """
    Copy ``src_path`` into ``dest_dir`` without ever overwriting an existing file.

    If ``dest_dir/filename`` already exists on disk (from a previous run, or because
    the source tree overlaps the output), the first free ``name_N.ext`` is used
    instead. The destination is opened exclusively, so this is race-free.

    Returns:
        The path actually written.
    """
    ensure_dir(dest_dir)
    stem, ext = os.path.splitext(filename)
    name = filename
    counter = 0
    while True:
        dest_path = os.path.join(dest_dir, name)
        try:
            with open(src_path, "rb") as fin, open(dest_path, "xb") as fout:
                shutil.copyfileobj(fin, fout)
        except FileExistsError:
            counter += 1
            name = f"{stem}_{counter}{ext}"
            continue
        shutil.copystat(src_path, dest_path)
        return dest_path


def copy_image_to_person_folder(src_path: str, dest_dir: str, person_id: str, filename: str) -> str:
    """Copy an image into ``dest_dir/person_id/`` (never overwriting); returns the path."""
    return copy_image(src_path, os.path.join(dest_dir, person_id), filename)


def get_image_files(source_dir: str, exclude_dir: str | None = None) -> list[str]:
    """
    Recursively find supported image files in ``source_dir``, in sorted order.

    Args:
        source_dir: Directory to scan.
        exclude_dir: Directory to skip entirely (typically the output directory, so
            a previous run's results are not rescanned when output is inside source).

    Raises:
        InvalidPathError: If ``source_dir`` is not a directory, or it is the same as,
            or located inside, ``exclude_dir`` (copying could overwrite the inputs).
    """
    if not os.path.isdir(source_dir):
        raise InvalidPathError(f"Not a directory: {source_dir}")
    if exclude_dir and is_inside(source_dir, exclude_dir):
        raise InvalidPathError(
            f"The output directory ({exclude_dir}) must not be the same as, or contain, "
            f"the input directory ({source_dir})"
        )

    files: list[str] = []
    for root, dirs, filenames in os.walk(source_dir):
        dirs[:] = sorted(
            d for d in dirs if not (exclude_dir and same_path(os.path.join(root, d), exclude_dir))
        )
        files.extend(os.path.join(root, f) for f in sorted(filenames) if is_image_file(f))
    return files
