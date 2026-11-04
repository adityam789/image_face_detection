"""
facesorter/errors.py
Exception types raised by the facesorter library and reported by the CLI.
"""

from __future__ import annotations


class FaceSorterError(Exception):
    """Base class for all expected, user-facing facesorter failures."""


class ConfigError(FaceSorterError, ValueError):
    """A configuration file or option value is invalid."""


class InvalidPathError(FaceSorterError):
    """A required directory is missing or an output/source combination is unusable."""


class NoImagesError(FaceSorterError):
    """No supported image files were found in a directory."""


class NoFacesError(FaceSorterError):
    """No usable faces were found, so there is nothing to sort or match against."""


class CopyError(FaceSorterError):
    """Every attempt to copy result files into the output directory failed."""
