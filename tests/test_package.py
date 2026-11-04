"""Packaging-level behaviour: lazy imports, ``python -m facesorter`` and the public API."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import facesorter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_python(*args):
    """Run ``python *args`` in a fresh interpreter that can import the in-repo package."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, env=env, timeout=60
    )


def test_version_string():
    assert facesorter.__version__.count(".") == 2


def test_importing_cli_does_not_load_heavy_dependencies():
    code = (
        "import sys, facesorter.cli\n"
        "heavy = ['insightface', 'sklearn', 'onnxruntime', 'scipy',"
        " 'facesorter.pipeline', 'facesorter.finder', 'facesorter.clustering']\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "print(','.join(loaded))\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    result = run_python("-c", code)
    assert result.returncode == 0, f"unexpectedly imported: {result.stdout.strip()}"


def test_importing_package_does_not_import_cli_or_pipeline():
    code = "import sys, facesorter; sys.exit(int('facesorter.pipeline' in sys.modules))"
    assert run_python("-c", code).returncode == 0


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_python_dash_m_runs(flag):
    result = run_python("-m", "facesorter", flag)
    assert result.returncode == 0
    assert "facesorter" in result.stdout


def test_python_dash_m_reports_exit_code(tmp_path):
    result = run_python(
        *["-m", "facesorter", "sort", "--source", str(tmp_path / "nope")],
        *["--output", str(tmp_path / "out")],
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_lazy_public_api():
    from facesorter import finder, pipeline

    assert facesorter.process_images is pipeline.process_images
    assert facesorter.find_matches is finder.find_matches


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_thing"):
        facesorter.no_such_thing  # noqa: B018


def test_library_logger_is_silent_by_default():
    import logging

    assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger("facesorter").handlers)


def test_main_module_exits_with_cli_status(monkeypatch, capsys):
    """In-process run of ``python -m facesorter`` so ``__main__`` is covered too."""
    import runpy

    monkeypatch.setattr(sys, "argv", ["facesorter", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("facesorter", run_name="__main__")
    assert exc.value.code == 0
    assert facesorter.__version__ in capsys.readouterr().out
