"""
facesorter/paths.py
Filesystem path comparison helpers (dependency-free, so the CLI can import them cheaply).
"""

from __future__ import annotations

import os


def same_path(a: str, b: str) -> bool:
    """
    True if ``a`` and ``b`` refer to the same filesystem location.

    Uses ``os.path.samefile`` when both exist (correct on case-insensitive filesystems
    and through symlinks) and falls back to comparing normalised real paths.
    """
    if os.path.exists(a) and os.path.exists(b):
        return os.path.samefile(a, b)
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def is_inside(child: str, parent: str) -> bool:
    """True if ``child`` is ``parent`` itself or located anywhere beneath it."""
    current = os.path.realpath(child)
    while True:
        if same_path(current, parent):
            return True
        up = os.path.dirname(current)
        if up == current:
            return False
        current = up
