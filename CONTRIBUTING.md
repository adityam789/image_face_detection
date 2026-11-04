# Contributing

Thanks for your interest. Bug reports and small, focused pull requests are welcome.

## Setup

Python 3.9 or newer is required.

```bash
git clone https://github.com/adityam789/image_face_detection.git
cd image_face_detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Before opening a pull request, run the same checks CI runs:

```bash
ruff check .
ruff format --check .     # `ruff format .` fixes formatting
pytest
```

## Writing tests

Tests must not need model weights or network access. The pipeline functions (`process_images`, `find_matches`, `scan_images`, ...) take a `model=` argument; pass a small fake whose `get(img_bgr)` returns objects with `bbox`, `embedding` and `det_score` attributes, and generate images with Pillow into `tmp_path`. A complete example is in [docs/architecture.md](docs/architecture.md#testing-with-a-fake-model). Use PNG for synthetic images whose pixel values matter. Please don't add real photos of people (including in tests, docs and screenshots); tests build synthetic images instead.

## Pull requests

Keep a pull request to one change, describe what it does and why, and add or update tests for behavior changes. If a change affects flags, defaults, exit codes or output layout, update the README and add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md). CI (lint, format check and tests on Python 3.9 to 3.13) must pass.

Please open an issue first for larger changes such as a different clustering strategy or new output formats, so the approach can be discussed before you invest time. The trade-offs behind current choices are written down in the README's "Design decisions and trade-offs" section.
