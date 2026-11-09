# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Natural-language photo search, in the new `facesorter.search` package with six commands: `facesorter index`, `search`, `serve`, `people`, `stats` and `eval`. Needs Python 3.10+ and `pip install "facesorter[search]"`; `sort` and `find` still need only Python 3.9+. Guide: `docs/search.md`; design: `docs/search-design.md`.
- `index`: incremental indexing of a photo folder into one local SQLite file (`~/.facesorter/index.sqlite`, or `FACESORTER_INDEX` / `--index`) with EXIF dates, folder-name dates (month and year precision), GPS place labels (city, region, country) and SigLIP 2 image embeddings. Pruning is per library folder; an unreadable folder keeps its photos; an empty scan of a known folder is refused unless `--allow-empty-prune`. `index --remove-root DIR` forgets a library folder (for example after renaming it) without touching any file.
- `search`: queries are split into exact filters (date, place, people) and visual text ranked by SigLIP 2, using an ONNX text tower that needs no PyTorch. Flags `--from`, `--to`, `--place`, `--person`, `-k`, `--json`, `--paths-only`, `--copy-to`, `--no-ask`.
- Query understanding by a small local language model (Apple-silicon MLX, `search-mlx`, or `llama-cpp-python`, `search-llama`) or by deterministic rules when none is installed (`--llm auto|mlx|llama-cpp|none`, `--no-llm`, `--llm-model`). The model only copies the date phrase; code resolves it, and every extracted span must appear in the query.
- A date grammar (seasons, holidays, "N ago", ranges) that refuses what it cannot apply (`since`, `before`, `early`, ambiguous numeric dates, lunar holidays, non-English words) with a question or a notice instead of a silent filter. "Ask once, then tell and search without."
- An offline gazetteer (GeoNames through `geonamescache`) with region names for the US, Canada, Australia, the UK, India, Germany, Italy, France, Spain and Austria, a curated list of destinations, and library-aware choice between namesakes (for example Amherst).
- `people scan`, `cluster`, `list` and `name`: face grouping with the existing InsightFace pipeline, with names kept across re-clustering and a 10,000-face cap.
- `serve`: a local Gradio web app on `127.0.0.1` that serves only a thumbnail cache (kept in a private directory, together with Gradio's cache copies), checks the `Host` and `Origin` headers against an allowlist to block DNS rebinding, shows folder names but never the library path, and has thumbs up/down ratings.
- `stats`, and `eval`: a caption-free evaluation report (metadata probes, safety battery, rating precision) that prints aggregates only unless `--show-paths`.
- Documentation: a photo-search section in the README, `docs/search.md`, `docs/search-design.md`, and a search section in `docs/architecture.md`.
- Test suite for the search package (fake encoders, fake LLM and the `tests/search/` directory); real-model tests are skipped unless `FACESORTER_MODEL_TESTS=1`.

### Changed

- CI installs `.[dev,search-core]` (it was `.[dev]`) so the search tests run on Python 3.9 to 3.13; those needing a missing package skip.
- `.gitignore` excludes the search index and derived data (`.facesorter/`, `*.sqlite`, `*.sqlite3`, `*.db`, `feedback*.jsonl`).
- `facesorter` now lists the six new subcommands in `--help`. The behaviour, flags, defaults and exit codes of `sort` and `find` are unchanged.
- README and `docs/architecture.md` describe the new network access (one-time model downloads for search).

### Notes

- New optional dependency groups in `pyproject.toml`: `search-core`, `search`, `search-mlx`, `search-llama`. On Python 3.9 they install nothing.
- Models downloaded on first use: SigLIP 2 B/16 image weights (about 1.5 GB) and the ONNX text tower (about 0.6 GB) from the Hugging Face Hub, an optional Qwen2.5-1.5B-Instruct (about 0.9 to 1.1 GB), and InsightFace's `buffalo_l` for people. SigLIP 2 and Qwen are Apache-2.0; InsightFace's pretrained models are non-commercial research only.
- Figures in the docs: COCO 5k text-to-image Recall@1 is a public benchmark; query-parsing accuracy comes from small hand-labelled sets and is optimistic.

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
