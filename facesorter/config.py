"""
facesorter/config.py
Settings resolution with a clear precedence: CLI flags > config file > built-in defaults.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass

from .errors import ConfigError

DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_CONFIDENCE = 0.8
DEFAULT_CLUSTER_THRESHOLD = 0.6
DEFAULT_MATCH_THRESHOLD = 0.6


# Allowed (low, high) range for each setting. Cosine distance lies in [0, 2].
RANGES = {
    "confidence": (0.0, 1.0),
    "cluster_threshold": (0.0, 2.0),
    "match_threshold": (0.0, 2.0),
}


@dataclass(frozen=True)
class Settings:
    """Resolved, validated settings."""

    confidence: float = DEFAULT_CONFIDENCE
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD
    match_threshold: float = DEFAULT_MATCH_THRESHOLD


def _validate(name: str, value: float, low: float, high: float, source: str) -> float:
    """Return ``value`` if within [low, high], otherwise raise ConfigError."""
    if not low <= value <= high:  # also rejects NaN
        raise ConfigError(f"{name} must be between {low} and {high} (got {value}) in {source}")
    return value


def _read_file_values(config_path: str | None) -> dict[str, float]:
    """Read and validate known numeric keys from the DEFAULT section of an INI file."""
    if not config_path or not os.path.isfile(config_path):
        return {}  # a missing config file is fine
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with open(config_path, encoding="utf-8") as fh:
            parser.read_file(fh)
    except (configparser.Error, OSError, UnicodeDecodeError) as exc:
        detail = " ".join(str(exc).split())  # configparser messages span several lines
        raise ConfigError(f"Cannot read config file {config_path}: {detail}") from exc

    values: dict[str, float] = {}
    for key, (low, high) in RANGES.items():
        raw = parser.defaults().get(key)
        if raw is None:
            continue
        try:
            number = float(raw)
        except ValueError:
            raise ConfigError(
                f"Invalid value for '{key}' in {config_path}: {raw!r} (expected a number)"
            ) from None
        values[key] = _validate(key, number, low, high, config_path)
    return values


def load_settings(
    config_path: str | None = None,
    *,
    confidence: float | None = None,
    cluster_threshold: float | None = None,
    match_threshold: float | None = None,
) -> Settings:
    """
    Resolve settings from defaults, an optional INI file, and CLI overrides.

    Precedence (highest first): explicit keyword arguments (CLI flags), the config
    file's ``[DEFAULT]`` section, built-in defaults. ``match_threshold`` falls back to
    the file's ``cluster_threshold`` when absent (legacy configs only had that key).
    Unknown keys (e.g. the removed ``nms_threshold``) are ignored.

    Raises:
        ConfigError: If the file is unparsable or any value is not a number in range.
    """
    from_file = _read_file_values(config_path)

    overrides = {
        "confidence": confidence,
        "cluster_threshold": cluster_threshold,
        "match_threshold": match_threshold,
    }
    for name, value in overrides.items():
        if value is not None:
            _validate(name, float(value), *RANGES[name], "command-line options")

    file_cluster = from_file.get("cluster_threshold", DEFAULT_CLUSTER_THRESHOLD)
    file_match = from_file.get("match_threshold", from_file.get("cluster_threshold"))
    if file_match is None:
        file_match = DEFAULT_MATCH_THRESHOLD

    return Settings(
        confidence=float(confidence)
        if confidence is not None
        else from_file.get("confidence", DEFAULT_CONFIDENCE),
        cluster_threshold=float(cluster_threshold)
        if cluster_threshold is not None
        else file_cluster,
        match_threshold=float(match_threshold) if match_threshold is not None else file_match,
    )
