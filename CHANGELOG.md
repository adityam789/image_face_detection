# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-03

First packaged release, following a refactor of the initial scripts. Changes below are relative to the pre-release scripts.

### Added

- `facesorter` console script (also `python -m facesorter` and `python face_sorter.py`) with `sort` and `find` subcommands, `--version`, `-v/--verbose`, and validated `--config` handling.
- Installable package with `pyproject.toml` (Python 3.9+, optional `dev` extras).
- `config.py` with validated settings and the precedence CLI flags > config file > defaults; separate `cluster_threshold` and `match_threshold` keys.
- Typed errors (`ConfigError`, `InvalidPathError`, `NoImagesError`, `NoFacesError`, `CopyError`) mapped to exit codes 1 and 2 (130 for Ctrl+C).
- Separate `report.txt` for `sort`, listing images with no faces, load failures, detection failures and multi-face photos.
- Continuous integration on GitHub Actions (ruff lint, ruff format check, pytest with coverage on Python 3.9 to 3.13).
- A pytest suite that runs the pipelines against a fake face model, without model weights.
- `scripts/visualize_detections.py` (replaces `test_face_detection.py`) for drawing detections on one image.
- MIT `LICENSE`, `CONTRIBUTING.md` and `docs/architecture.md`.

### Changed

- `find` uses the centroid of unit-normalised reference embeddings, and `--threshold` applies to `match_threshold` for `find` and `cluster_threshold` for `sort`.
- Log output goes to stderr (was stdout).
- `sort` exits with code 1 (was 0) when no faces are found.
- Image discovery is sorted, so output layout is deterministic.
- Heavy imports (scikit-learn, InsightFace) are deferred until a command runs, so `--help` is fast.
- `face_sorter.py` is now a thin wrapper around `facesorter.cli`.
- Documented the `min_samples=1` single-linkage chaining trade-off in `clustering.py`.

### Fixed

- EXIF orientation is now applied when loading photos, so sideways phone photos are detected correctly.
- Files with the same name from different folders no longer collide in the output; later ones get `_1`, `_2`, ... suffixes.
- The sort report no longer overwrites the runtime log (they are now `report.txt` and `processing_log.txt`).
- Copying never overwrites existing files, including in `find`, and an output directory inside the source is skipped when scanning.
- A source directory equal to or inside the output directory is rejected, as are missing directories and invalid config values, before any files are created.
- In `sort`, a model failure on a single image no longer aborts the run; it is logged and listed in `report.txt`.

### Removed

- The `--nms` option and `nms_threshold` config key (unknown config keys are ignored).

[Unreleased]: https://github.com/adityam789/image_face_detection/commits/main
[0.1.0]: https://github.com/adityam789/image_face_detection/commits/main
