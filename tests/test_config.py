from __future__ import annotations

import dataclasses

import pytest

from facesorter.config import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_CONFIDENCE,
    DEFAULT_MATCH_THRESHOLD,
    Settings,
    load_settings,
)
from facesorter.errors import ConfigError


def test_documented_defaults_are_pinned():
    """The defaults are documented in the README and config.ini; changing them is breaking."""
    s = load_settings(None)
    assert (s.confidence, s.cluster_threshold, s.match_threshold) == (0.8, 0.6, 0.6)
    assert Settings() == s


def test_defaults_without_file():
    assert load_settings(None) == Settings(
        DEFAULT_CONFIDENCE, DEFAULT_CLUSTER_THRESHOLD, DEFAULT_MATCH_THRESHOLD
    )


def test_missing_file_is_not_an_error(tmp_path):
    assert load_settings(str(tmp_path / "absent.ini")) == Settings()


def test_settings_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().confidence = 0.1  # type: ignore[misc]


def test_file_overrides_defaults(write_config):
    path = write_config(
        "[DEFAULT]\nconfidence = 0.5\ncluster_threshold = 0.4\nmatch_threshold = 0.3\n"
    )
    assert load_settings(path) == Settings(0.5, 0.4, 0.3)


def test_cli_overrides_file(write_config):
    path = write_config(
        "[DEFAULT]\nconfidence = 0.5\ncluster_threshold = 0.4\nmatch_threshold = 0.3\n"
    )
    s = load_settings(path, confidence=0.9, cluster_threshold=0.1, match_threshold=0.2)
    assert s == Settings(0.9, 0.1, 0.2)


def test_cli_override_applies_per_key(write_config):
    path = write_config("[DEFAULT]\nconfidence = 0.5\ncluster_threshold = 0.4\n")
    s = load_settings(path, confidence=0.7)
    assert (s.confidence, s.cluster_threshold) == (0.7, 0.4)


def test_cli_zero_is_an_override_not_unset(write_config):
    path = write_config("[DEFAULT]\nconfidence = 0.5\n")
    assert load_settings(path, confidence=0.0).confidence == 0.0


def test_match_threshold_falls_back_to_file_cluster_threshold(write_config):
    path = write_config("[DEFAULT]\ncluster_threshold = 0.35\n")
    s = load_settings(path)
    assert s.cluster_threshold == s.match_threshold == 0.35


def test_explicit_match_threshold_beats_fallback(write_config):
    path = write_config("[DEFAULT]\ncluster_threshold = 0.35\nmatch_threshold = 0.45\n")
    assert load_settings(path).match_threshold == 0.45


def test_cli_cluster_threshold_does_not_change_match_threshold(write_config):
    path = write_config("[DEFAULT]\ncluster_threshold = 0.35\n")
    s = load_settings(path, cluster_threshold=0.1)
    assert (s.cluster_threshold, s.match_threshold) == (0.1, 0.35)


def test_match_threshold_default_when_nothing_configured(write_config):
    assert load_settings(write_config("[DEFAULT]\n")).match_threshold == DEFAULT_MATCH_THRESHOLD


def test_unknown_keys_and_other_sections_are_ignored(write_config):
    path = write_config(
        "[DEFAULT]\nnms_threshold = 0.4\nconfidence = 0.55\nnote = 100% sure\n"
        "[other]\nconfidence = 0.1\n"
    )
    assert load_settings(path).confidence == 0.55


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("confidence", "1.01"),
        ("confidence", "-0.01"),
        ("cluster_threshold", "2.5"),
        ("cluster_threshold", "-1"),
        ("match_threshold", "3"),
        ("confidence", "nan"),
        ("confidence", "abc"),
        ("match_threshold", ""),
    ],
)
def test_invalid_file_values_raise_config_error(write_config, key, value):
    with pytest.raises(ConfigError, match=key):
        load_settings(write_config(f"[DEFAULT]\n{key} = {value}\n"))


@pytest.mark.parametrize(
    ("key", "value"),
    [("confidence", 1.5), ("confidence", -0.1), ("cluster_threshold", 2.1),
     ("match_threshold", -2.0), ("confidence", float("nan"))],
)  # fmt: skip
def test_invalid_cli_values_raise_config_error(key, value):
    with pytest.raises(ConfigError, match="command-line options"):
        load_settings(None, **{key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [("confidence", 0.0), ("confidence", 1.0), ("cluster_threshold", 2.0),
     ("match_threshold", 0.0)],
)  # fmt: skip
def test_range_endpoints_are_accepted(key, value):
    assert getattr(load_settings(None, **{key: value}), key) == value


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


@pytest.mark.parametrize(
    "body", ["confidence = 0.5\n", "[DEFAULT\nconfidence = 0.5\n", "[DEFAULT]\nconfidence\n"]
)
def test_unparsable_file_raises_config_error(write_config, body):
    with pytest.raises(ConfigError, match="Cannot read config file"):
        load_settings(write_config(body))


def test_non_utf8_file_raises_config_error(tmp_path):
    path = tmp_path / "latin.ini"
    path.write_bytes(b"[DEFAULT]\n# caf\xe9\nconfidence = 0.5\n")
    with pytest.raises(ConfigError):
        load_settings(str(path))


@pytest.mark.parametrize("key", ["confidence", "cluster_threshold", "match_threshold"])
@pytest.mark.parametrize("value", [-0.01, -0.5])
def test_slightly_negative_values_are_rejected(write_config, key, value):
    with pytest.raises(ConfigError):
        load_settings(write_config(f"[DEFAULT]\n{key} = {value}\n"))
    with pytest.raises(ConfigError, match="command-line options"):
        load_settings(None, **{key: value})
