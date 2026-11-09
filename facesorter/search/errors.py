"""
facesorter/search/errors.py
Typed errors for the photo-search feature.
"""

from __future__ import annotations

from ..errors import FaceSorterError


class SearchError(FaceSorterError):
    """Base class for photo-search failures."""


class IndexNotFoundError(SearchError):
    """The index file does not exist; run ``facesorter index`` first."""


class IndexMismatchError(SearchError):
    """The index was built with a different embedding model or schema than requested."""


class UnreadableImageError(SearchError):
    """A file could not be opened or decoded as an image."""


class ModelUnavailableError(SearchError):
    """An optional model or runtime (encoder, local LLM) could not be loaded."""
