"""
facesorter/search/library.py
Find the still images in a photo library, in a deterministic order.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from ..errors import InvalidPathError
from ..paths import is_inside, same_path

logger = logging.getLogger(__name__)

SkipCallback = Callable[[Path, BaseException], None]

#: Lower-case extensions of the still-image formats the search index supports.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".heic", ".heif", ".jpg", ".jpeg", ".png"})


def is_still_image(path: str | os.PathLike[str]) -> bool:
    """True if ``path`` has a supported still-image extension (case-insensitive).

    Looks only at the name. Hidden files and macOS AppleDouble files (``._photo.jpg``) have a
    supported extension but are not photos, so they are rejected here too.
    """
    name = os.path.basename(os.fspath(path))
    if name.startswith("."):
        return False
    return os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES


def iter_library(
    root: str | os.PathLike[str],
    exclude_dirs: Iterable[str | os.PathLike[str]] = (),
    *,
    on_skip: SkipCallback | None = None,
) -> Iterator[Path]:
    """
    Yield the absolute path of every still image under ``root``, recursively.

    The order is deterministic and independent of the filesystem: depth-first, with the
    entries of each directory sorted by name (case-sensitive). This is not the same as
    sorting the path strings: ``a/x.jpg`` comes before ``a b/x.jpg``, because the directory
    ``a`` sorts before ``a b`` and is walked completely first.
    Paths are yielded in canonical form, ``os.path.abspath(os.path.expanduser(root))`` joined
    with the relative path (``~`` is expanded, symlinks are not resolved), so
    ``str(path)`` equals ``PhotoMeta.path`` from ``read_metadata`` for the same file.
    Hidden files and directories (leading ``.``), AppleDouble ``._*`` files, videos
    (including Live Photo ``.mov`` companions) and other unsupported types are skipped.
    Symlinked directories are not followed. Any directory equal to or inside one of
    ``exclude_dirs`` is skipped (compared with ``samefile`` semantics, so symlinks and
    case-insensitive filesystems are handled).

    Raises ``InvalidPathError`` if ``root`` is not an existing directory. A directory that
    cannot be listed is logged and skipped, and ``on_skip(directory, error)`` is called for it
    (``directory`` is a canonical ``Path``). Callers that prune what they no longer find
    (an indexer) must use this to avoid treating the photos of an unreadable folder as
    deleted. An exception raised by ``on_skip`` is logged and does not stop the scan.
    """
    root_path = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if not os.path.isdir(root_path):
        raise InvalidPathError(f"Photo library folder does not exist or is not a folder: {root}")
    excluded = [os.path.abspath(os.path.expanduser(os.fspath(d))) for d in exclude_dirs]
    # Only the root can start out inside an excluded folder; below it, an ancestor that is not
    # excluded means a directory is excluded only if it *is* one of the excluded folders.
    if any(is_inside(root_path, ex) for ex in excluded):
        logger.info("Library root %s is inside an excluded folder; nothing to scan", root_path)
        return
    yield from _walk(root_path, excluded, on_skip)


def _walk(directory: str, excluded: list[str], on_skip: SkipCallback | None) -> Iterator[Path]:
    try:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError as exc:
        logger.warning("Skipping unreadable folder %s: %s", directory, exc)
        if on_skip is not None:
            try:
                on_skip(Path(directory), exc)
            except Exception:
                logger.exception("on_skip callback failed for %s", directory)
        return
    for entry in entries:
        name = entry.name
        if name.startswith("."):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                if any(same_path(entry.path, ex) for ex in excluded):
                    logger.debug("Skipping excluded folder %s", entry.path)
                    continue
                yield from _walk(entry.path, excluded, on_skip)
            elif is_still_image(name) and entry.is_file():
                yield Path(entry.path)
        except OSError as exc:  # pragma: no cover - entry vanished or became unreadable
            logger.warning("Skipping %s: %s", entry.path, exc)
