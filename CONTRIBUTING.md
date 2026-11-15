# Contributing

Thanks for your interest. Bug reports and small, focused pull requests are welcome.

## Setup

Python 3.9 or newer is required for the core tool (`sort`, `find`). Photo search needs Python 3.10 or newer.

```bash
git clone https://github.com/adityam789/image_face_detection.git
cd image_face_detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core tool only
pip install -e ".[dev,search-core]"     # also the search tests (what CI installs; Python 3.10+)
```

Before opening a pull request, run the same checks CI runs:

```bash
ruff check .
ruff format --check .     # `ruff format .` fixes formatting
pytest
```

## Writing tests

Tests must not need model weights or network access. The pipeline functions (`process_images`, `find_matches`, `scan_images`, ...) take a `model=` argument; pass a small fake whose `get(img_bgr)` returns objects with `bbox`, `embedding` and `det_score` attributes, and generate images with Pillow into `tmp_path`. A complete example is in [docs/architecture.md](docs/architecture.md#testing-with-a-fake-model). Use PNG for synthetic images whose pixel values matter. Please don't add real photos of people (including in tests, docs and screenshots); tests build synthetic images instead.

## Working on photo search

The photo-search feature lives in `facesorter/search/` with tests in `tests/search/`; read [docs/search-design.md](docs/search-design.md) first, and [docs/search.md](docs/search.md) for the behaviour users see.

**Running the tests.** No search test needs model weights, a network or real photos. Image and text encoders are replaced by the fakes in `facesorter/search/testing.py` (a solid-colour image embeds near the text of its colour name, so results can be asserted exactly) and the language model by `ScriptedLLM`. To use a fake where code loads a model by name, register it and unregister it afterwards:

```python
from facesorter.search import testing
from facesorter.search.encoders import register_model, unregister_model

register_model(
    testing.FAKE_MODEL_NAME,
    lambda device: testing.FakeImageEncoder(),
    lambda quantization: testing.FakeTextEncoder(),
    dim=testing.FAKE_DIM,
)
# ... index and search with the model name "fake-color-16" ...
unregister_model(testing.FAKE_MODEL_NAME)
```

```bash
pytest tests/search                     # the search tests only
pytest -k "dates or parser"             # a subset by name
```

Tests that need an optional package (`gradio`, `geonamescache`, `pillow-heif`, `mlx_lm`, ...) skip when it is missing. The web-app tests need `pip install -e ".[dev,search]"`.

**Real-model tests.** A few tests load the real SigLIP 2 weights or the real MLX model. They are skipped by default and run only with `FACESORTER_MODEL_TESTS=1`; they download models from the Hugging Face Hub on first run, so do not make CI depend on them.

```bash
FACESORTER_MODEL_TESTS=1 pytest tests/search/test_search_encoders.py tests/search/test_search_llm.py
```

**Coverage gotcha.** `--cov` takes a dotted module or package name, or a directory path. A file path ending in `.py` is reported as "Module ... was never imported" and no data is collected:

```bash
pytest tests/search/test_search_dates.py --cov=facesorter.search.dates --cov-report=term-missing   # works
pytest tests/search --cov=facesorter/search --cov-report=term-missing                              # works
pytest tests/search --cov=facesorter/search/dates.py                                                # no data
```

**Lint and format.** `ruff check .` and `ruff format --check .` cover the search package too (line length 100). The package keeps `from __future__ import annotations` so it stays importable on Python 3.9; heavy imports (torch, onnxruntime, open_clip, gradio, mlx_lm, llama_cpp, insightface, scikit-learn) go inside the function that needs them, so `facesorter --help` stays fast.

**Mutation checking.** When reviewing or fixing search code, check that the tests really guard the behaviour: break the line on purpose (flip `<` to `<=`, drop a guard, return early), run the relevant tests, confirm that at least one fails, then restore the line. If nothing fails, the behaviour is untested; add a test that fails against the broken line. This is done by hand; there is no mutation-testing tool in the project.

**Privacy rules for tests, docs and examples.** Photos and place names from a real personal library must never appear in this repository: not in tests, fixtures, docs, screenshots, example output or commit messages. Tests build synthetic images with Pillow (solid colours, generated EXIF dates and GPS values) in `tmp_path`. Examples in docs use generic paths such as `~/Pictures` and well-known places (Lisbon, Kyoto, Banff). Never commit an index file, thumbnail cache or evaluation report made from a real library (`.gitignore` excludes `.facesorter/`, `*.sqlite`, `*.sqlite3`, `*.db` and `feedback*.jsonl`). `facesorter eval` prints aggregates only by default; keep it that way, and do not paste `--show-paths` output into issues.

## Pull requests

Keep a pull request to one change, describe what it does and why, and add or update tests for behavior changes. If a change affects flags, defaults, exit codes or output layout, update the README (and [docs/search.md](docs/search.md) for search commands) and add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md). CI (lint, format check and tests on Python 3.9 to 3.13, with `.[dev,search-core]`) must pass.

Please open an issue first for larger changes such as a different clustering strategy or new output formats, so the approach can be discussed before you invest time. The trade-offs behind current choices are written down in the README's "Design decisions and trade-offs" section.
