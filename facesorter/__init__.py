"""
facesorter
Cluster photos by person, or find photos of one person, using InsightFace.

The public API is exposed lazily so that ``import facesorter`` (and therefore
``facesorter --help``) does not pull in heavy dependencies such as scikit-learn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Library etiquette: stay silent unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__", "process_images", "find_matches"]

if TYPE_CHECKING:  # pragma: no cover
    from .finder import find_matches
    from .pipeline import process_images


def __getattr__(name: str) -> Any:
    """Import the public API on first access (PEP 562)."""
    if name == "process_images":
        from .pipeline import process_images

        return process_images
    if name == "find_matches":
        from .finder import find_matches

        return find_matches
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
