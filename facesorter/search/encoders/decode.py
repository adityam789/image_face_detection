"""
facesorter/search/encoders/decode.py
Fast, parallel image decoding for the embedding model.

Indexing time is dominated by decoding 12-megapixel HEIC/JPEG files, not by the model, and
the model only looks at a 224x224 view. So we decode as little as possible:

* JPEG: ``Image.draft`` makes libjpeg decode at 1/2, 1/4 or 1/8 scale (DCT-domain scaling),
  which is several times cheaper than a full decode.
* HEIC: HEVC cannot be decoded at reduced size (libheif has no such mode), so the full
  decode is unavoidable. The embedded thumbnail is deliberately NOT used: it is typically
  320x240, too small, and can be stale after an edit. We still avoid extra full-resolution
  work (colour conversion and EXIF transposition happen after the downscale).
* Everything else (PNG, ...): normal decode, then downscale.

The downscale keeps the aspect ratio and stops at ``target`` pixels on the shorter side
(default 448, twice SigLIP's 224 input), so the model's own resize still sees an
over-sampled image and the embedding is practically unchanged.

``iter_decoded`` runs this in a thread pool (PIL and libheif release the GIL while
decoding), keeping order and bounding memory.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import UnreadableImageError

logger = logging.getLogger(__name__)

# Shorter-side size kept after decoding: 2x SigLIP's 224 px input.
DEFAULT_TARGET = 448

_EXIF_ORIENTATION = 0x0112
_TRANSPOSE_FOR_ORIENTATION = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}

_heif_registered = False


def _ensure_heif() -> None:
    """Register the HEIC opener on first use (keeps ``import`` cheap)."""
    global _heif_registered
    if not _heif_registered:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _heif_registered = True


def _to_8bit(img: Image.Image) -> Image.Image:
    """Bring high-bit-depth / float grayscale (``I;16*``, ``I``, ``F``) to 8-bit ``L``.

    Pillow's own conversion clips these modes (16-bit PNGs come out all white, float TIFFs
    all black), so scale explicitly: 16-bit values by 1/256, floats in [0, 1] by 255,
    anything else by a min-max stretch.
    """
    arr = np.asarray(img).astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if img.mode.startswith("I;16") or (img.mode == "I" and 255 < hi <= 65535):
        arr = arr / 256.0
    elif lo >= 0.0 and hi <= 1.0 and img.mode == "F":
        arr = arr * 255.0
    elif lo < 0.0 or hi > 255.0:
        arr = (arr - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(arr)
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8), "L")


def _downscale(img: Image.Image, target: int) -> Image.Image:
    """Shrink so the shorter side is ``target`` px; never upscale."""
    short = min(img.size)
    if short <= target:
        return img
    scale = target / short
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    # reducing_gap does a cheap box-reduce first, then an antialiased resample.
    return img.resize(size, Image.Resampling.LANCZOS, reducing_gap=2.0)


def _decode(path: Path, target: int) -> Image.Image:
    _ensure_heif()
    with Image.open(path) as img:
        orientation = img.getexif().get(_EXIF_ORIENTATION)
        if img.format in ("JPEG", "MPO"):  # MPO: JPEG-compatible multi-picture camera files
            # Ask libjpeg for the cheapest scale that still has >= target px on both sides.
            img.draft("RGB", (target, target))
        img.load()
        if img.mode in ("I", "F") or img.mode.startswith("I;16"):
            img = _to_8bit(img)
        small = _downscale(img, target)
        if small.mode != "RGB":
            small = small.convert("RGB")
        elif small is img:
            small = img.copy()  # detach from the file before it closes
    op = _TRANSPOSE_FOR_ORIENTATION.get(orientation) if isinstance(orientation, int) else None
    return small.transpose(op) if op is not None else small


def _check_target(target: int) -> None:
    if target < 1:
        raise ValueError(f"target must be positive, got {target}")


def load_for_model(path: str | os.PathLike[str], target: int = DEFAULT_TARGET) -> Image.Image:
    """Decode ``path`` to an RGB image for the embedding model, as cheaply as possible.

    EXIF orientation is applied and the image is downscaled (aspect ratio kept) to at least
    ``target`` pixels on the shorter side; smaller images are returned at full size. Raises
    :class:`UnreadableImageError` if the file cannot be opened or decoded.
    """
    _check_target(target)
    p = Path(path)
    try:
        return _decode(p, target)
    except Exception as exc:  # PIL raises OSError, ValueError, DecompressionBombError, ...
        raise UnreadableImageError(f"Cannot read image {p}: {exc}") from exc


def iter_decoded(
    paths: Iterable[str | os.PathLike[str]],
    workers: int = 4,
    prefetch: int | None = None,
    target: int = DEFAULT_TARGET,
) -> Iterator[tuple[Path, Image.Image | Exception]]:
    """Decode ``paths`` in a thread pool, yielding ``(path, image_or_error)`` in input order.

    At most ``prefetch`` (default ``2 * workers``) images are decoded or waiting at once, so
    memory stays bounded however many paths there are. A file that cannot be read yields an
    :class:`UnreadableImageError` in place of the image instead of raising. Abandoning the
    iterator early cancels the queued work. Bad ``workers`` / ``target`` raise ``ValueError``
    immediately, not on first iteration.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    _check_target(target)
    return _iter_decoded(paths, workers, prefetch, target)


def _iter_decoded(
    paths: Iterable[str | os.PathLike[str]],
    workers: int,
    prefetch: int | None,
    target: int,
) -> Iterator[tuple[Path, Image.Image | Exception]]:
    depth = max(1, prefetch if prefetch is not None else 2 * workers)
    source = iter(paths)
    pending: deque[tuple[Path, Future[Image.Image]]] = deque()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="decode") as pool:

        def fill() -> None:
            while len(pending) < depth:
                try:
                    p = Path(next(source))
                except StopIteration:
                    return
                pending.append((p, pool.submit(load_for_model, p, target)))

        try:
            fill()
            while pending:
                p, fut = pending.popleft()
                try:
                    result: Image.Image | Exception = fut.result()
                except Exception as exc:
                    logger.debug("Skipping unreadable image %s: %s", p, exc)
                    result = exc
                fill()
                yield p, result
        finally:
            for _, fut in pending:
                fut.cancel()
