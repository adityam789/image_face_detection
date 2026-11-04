"""
facesorter/finder.py
Logic for finding photos of one specific person in a directory of images.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from .config import DEFAULT_CONFIDENCE, DEFAULT_MATCH_THRESHOLD
from .detection import Face, detect_faces_and_embeddings, get_face_model
from .errors import CopyError, NoFacesError, NoImagesError
from .utils import copy_image, ensure_dir, get_image_files, load_image, unique_filename

if TYPE_CHECKING:  # pragma: no cover
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


def cosine_distances(reference: np.ndarray, embeddings: Any) -> np.ndarray:
    """
    Cosine distance (1 - cosine similarity) from ``reference`` to each embedding.

    Zero-norm vectors have no direction and get the maximum "unrelated" distance of 1.0.
    Returns an array of shape (N,).
    """
    ref = np.asarray(reference, dtype=float).reshape(-1)
    emb = np.atleast_2d(np.asarray(embeddings, dtype=float))
    denom = np.linalg.norm(emb, axis=1) * np.linalg.norm(ref)
    sims = np.divide(emb @ ref, denom, out=np.zeros(len(emb)), where=denom > 0)
    return np.clip(1.0 - sims, 0.0, 2.0)


def matches_reference(
    faces: Sequence[Face], reference: np.ndarray, threshold: float, confidence: float
) -> bool:
    """True if any face scoring at least ``confidence`` is closer than ``threshold``."""
    candidates = [f for f in faces if f.score >= confidence]
    if not candidates:
        return False
    distances = cosine_distances(reference, [f.embedding for f in candidates])
    return bool(np.any(distances < threshold))


def largest_face(faces: Sequence[Face]) -> Face:
    """Return the face with the biggest bounding box (probably the photo's subject)."""
    return max(faces, key=lambda f: f.area)


def average_embedding(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """
    Mean direction of the embeddings, as a unit vector.

    Each embedding is normalised to unit length before averaging so that every
    reference photo contributes equally regardless of its raw embedding norm (this
    differs slightly from averaging the raw vectors). The centroid is normalised
    again, as cosine distance requires.
    """
    vectors = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, norms, out=vectors.copy(), where=norms > 0)
    centroid = np.mean(vectors, axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


def get_reference_embedding(
    model: FaceAnalysis,
    reference_dir: str,
    confidence: float = DEFAULT_CONFIDENCE,
    exclude_dir: str | None = None,
) -> np.ndarray | None:
    """
    Compute one embedding for a person from a directory of their photos.

    In each photo the largest face above ``confidence`` is used; the result is the
    normalised average of those embeddings. ``exclude_dir`` (normally the output
    directory) is skipped while scanning.

    Returns:
        The reference embedding, or None if no usable face was found.

    Raises:
        NoImagesError: If the directory contains no supported images.
    """
    files = get_image_files(reference_dir, exclude_dir=exclude_dir)
    if not files:
        raise NoImagesError(f"No image files found in reference directory: {reference_dir}")

    logger.info("Computing reference embedding from %d images...", len(files))
    embeddings = []
    for file_path in tqdm(files, desc="Analyzing reference images"):
        try:
            faces = detect_faces_and_embeddings(model, load_image(file_path))
        except Exception as exc:  # one bad file must not abort the run
            logger.warning("Error processing reference image %s: %s", file_path, exc)
            continue
        valid = [f for f in faces if f.score >= confidence]
        if not valid:
            logger.warning("No valid faces found in %s", file_path)
            continue
        if len(valid) > 1:
            logger.debug("Multiple faces in %s, using largest one.", file_path)
        embeddings.append(largest_face(valid).embedding)

    if not embeddings:
        return None
    logger.info("Reference embedding computed from %d faces.", len(embeddings))
    return average_embedding(embeddings)


def find_matches(
    source_dir: str,
    output_dir: str,
    reference_dir: str,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    model: FaceAnalysis | None = None,
) -> int:
    """
    Copy photos of the reference person from ``source_dir`` into ``output_dir``.

    A photo matches if any face above ``confidence`` has cosine distance to the
    reference embedding below ``threshold``. Files with the same name from different
    subfolders are kept by giving later ones a numeric suffix, and existing files in
    ``output_dir`` are never overwritten.

    Args:
        source_dir: Directory to search (recursively).
        output_dir: Where matching photos are copied. May be inside ``source_dir`` or
            ``reference_dir``; it is skipped while scanning.
        reference_dir: Directory of photos of the target person.
        threshold: Cosine distance threshold (lower = stricter).
        confidence: Minimum face detection score.
        model: Optional pre-built face model (loaded on demand if omitted).

    Returns:
        Number of photos copied.

    Raises:
        InvalidPathError: If a directory is missing or overlaps the output directory.
        NoImagesError: If the source or reference directory has no images.
        NoFacesError: If no reference face could be found.
        CopyError: If matches were found but none could be copied.
    """
    image_files = get_image_files(source_dir, exclude_dir=output_dir)
    if not image_files:
        raise NoImagesError(f"No image files found in source directory: {source_dir}")

    model = model if model is not None else get_face_model()
    reference = get_reference_embedding(model, reference_dir, confidence, exclude_dir=output_dir)
    if reference is None:
        raise NoFacesError(
            f"Could not find a face (confidence >= {confidence}) in any reference image "
            f"under {reference_dir}"
        )

    logger.info("Scanning %d images in %s...", len(image_files), source_dir)
    ensure_dir(output_dir)
    used_names: set[str] = set()
    matched = copied = copy_failures = image_errors = 0
    for img_path in tqdm(image_files, desc="Searching for matches"):
        try:
            faces = detect_faces_and_embeddings(model, load_image(img_path))
            is_match = matches_reference(faces, reference, threshold, confidence)
        except Exception as exc:  # one bad file must not abort the run
            logger.error("Error processing %s: %s", img_path, exc)
            image_errors += 1
            continue
        if not is_match:
            continue
        matched += 1
        try:
            copy_image(
                img_path, output_dir, unique_filename(os.path.basename(img_path), used_names)
            )
            copied += 1
        except OSError as exc:
            logger.error("Failed to copy %s: %s", img_path, exc)
            copy_failures += 1

    logger.info(
        "Search complete: %d images scanned, %d matched, %d copied to %s "
        "(%d copy failures, %d images could not be processed)",
        len(image_files),
        matched,
        copied,
        output_dir,
        copy_failures,
        image_errors,
    )
    if matched and not copied:
        raise CopyError(f"Found {matched} matching images but could not copy any to {output_dir}")
    return copied
