"""
facesorter/pipeline.py
Main pipeline for sorting photos into one folder per person.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tqdm import tqdm

from .clustering import cluster_embeddings
from .detection import Face, detect_faces_and_embeddings, get_face_model
from .errors import CopyError, NoFacesError, NoImagesError
from .utils import (
    copy_image_to_person_folder,
    ensure_dir,
    get_image_files,
    load_image,
    unique_filename,
)

if TYPE_CHECKING:  # pragma: no cover
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

REPORT_NAME = "report.txt"


@dataclass
class ScanResult:
    """Outcome of detecting faces across a list of images."""

    faces: list[tuple[str, Face]] = field(default_factory=list)  # (image_path, face)
    no_faces: list[str] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)  # file could not be opened/decoded
    detection_errors: list[str] = field(default_factory=list)  # the face model failed
    multiple_faces: list[str] = field(default_factory=list)


def scan_images(model: FaceAnalysis, image_files: Sequence[str], confidence: float) -> ScanResult:
    """Detect faces scoring at least ``confidence`` in every image."""
    result = ScanResult()
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            image = load_image(img_path)
        except Exception as exc:  # one bad file must not abort the run
            logger.warning("Failed to load image %s: %s", img_path, exc)
            result.load_errors.append(img_path)
            continue
        try:
            all_faces = detect_faces_and_embeddings(model, image)
        except Exception as exc:  # one bad file must not abort the run
            logger.warning("Face detection failed for %s: %s", img_path, exc)
            result.detection_errors.append(img_path)
            continue
        faces = [f for f in all_faces if f.score >= confidence]
        if not faces:
            result.no_faces.append(img_path)
            best = max((f.score for f in all_faces), default=None)
            if best is None:
                logger.info("No faces detected in %s", img_path)
            else:
                logger.info("No faces >= %.2f in %s (best score %.2f)", confidence, img_path, best)
            continue
        if len(faces) > 1:
            result.multiple_faces.append(img_path)
        result.faces.extend((img_path, face) for face in faces)
    return result


def person_folder_name(label: int) -> str:
    """Folder name for a cluster label."""
    return f"Person_{label}"


def group_by_person(image_paths: Sequence[str], labels: Sequence[int]) -> dict[str, list[str]]:
    """
    Map person folder name -> images containing that person.

    ``image_paths[i]`` is the image that face ``i`` came from and ``labels[i]`` its
    cluster. Persons are ordered by label, and each image appears once per person, in
    first-seen order, so the result is deterministic.
    """
    by_label: dict[int, dict[str, None]] = {}  # dict as an ordered set
    for path, label in zip(image_paths, labels):
        by_label.setdefault(int(label), {})[path] = None
    return {person_folder_name(label): list(by_label[label]) for label in sorted(by_label)}


def plan_copies(groups: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """
    Decide ``(source_path, person_folder, destination_filename)`` for every copy.

    Files sharing a basename within one person folder get numeric suffixes instead of
    overwriting each other.
    """
    plan = []
    for person_id, paths in groups.items():
        taken: set[str] = set()
        for path in paths:
            plan.append((path, person_id, unique_filename(os.path.basename(path), taken)))
    return plan


def write_report(path: str, summary: Sequence[str], scan: ScanResult) -> None:
    """Write the structured sort report (kept separate from the runtime log)."""
    sections = [
        ("Images with no faces", scan.no_faces),
        ("Images failed to load", scan.load_errors),
        ("Images failed during face detection", scan.detection_errors),
        ("Images with multiple faces", scan.multiple_faces),
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Face Sorting Report\n" + "=" * 40 + "\n")
        fh.writelines(line + "\n" for line in summary)
        for title, paths in sections:
            fh.write(f"\n{title}:\n")
            fh.writelines(p + "\n" for p in paths)


def process_images(
    source_dir: str,
    output_dir: str,
    cluster_threshold: float,
    confidence: float,
    model: FaceAnalysis | None = None,
) -> int:
    """
    Sort the images in ``source_dir`` into ``output_dir/Person_N`` folders.

    1. Scan source images (skipping ``output_dir`` if it lies inside the source).
    2. Detect faces and extract embeddings, keeping those above ``confidence``.
    3. Cluster embeddings to find unique identities.
    4. Copy each image into the folder of every person it contains.
    5. Write ``report.txt`` into ``output_dir``.

    Args:
        source_dir: Input directory containing images.
        output_dir: Output directory for results.
        cluster_threshold: DBSCAN cosine distance threshold.
        confidence: Face detection confidence filter (0.0-1.0).
        model: Optional pre-built face model (loaded on demand if omitted).

    Returns:
        The number of distinct persons found.

    Raises:
        InvalidPathError: If ``source_dir`` is missing or equals ``output_dir``.
        NoImagesError: If no supported images are found.
        NoFacesError: If no face above ``confidence`` is found in any image.
        CopyError: If every attempt to copy a result file failed.
    """
    image_files = get_image_files(source_dir, exclude_dir=output_dir)
    if not image_files:
        raise NoImagesError(f"No image files found in {source_dir}")
    logger.info("Found %d image files in %s", len(image_files), source_dir)

    model = model if model is not None else get_face_model()
    scan = scan_images(model, image_files, confidence)
    if not scan.faces:
        n_errors = len(scan.load_errors) + len(scan.detection_errors)
        message = f"No faces with confidence >= {confidence} found in {len(image_files)} images"
        if n_errors:
            message += f" ({n_errors} of them could not be processed; see the log)"
        raise NoFacesError(message)

    labels = cluster_embeddings([face.embedding for _, face in scan.faces], cluster_threshold)
    groups = group_by_person([path for path, _ in scan.faces], labels)
    n_persons = len(groups)
    logger.info("Clustering complete. Found %d unique persons.", n_persons)

    ensure_dir(output_dir)
    plan = plan_copies(groups)
    copy_failures = 0
    for src, person_id, filename in tqdm(plan, desc="Copying images"):
        try:
            copy_image_to_person_folder(src, output_dir, person_id, filename)
        except OSError as exc:
            copy_failures += 1
            logger.warning("Failed to copy %s to %s/%s: %s", src, person_id, filename, exc)

    summary = [
        f"Total images processed: {len(image_files)}",
        f"Total faces detected: {len(scan.faces)}",
        f"Unique persons found: {n_persons}",
        f"Images with multiple faces: {len(scan.multiple_faces)}",
        f"Images with no faces: {len(scan.no_faces)}",
        f"Images failed to load: {len(scan.load_errors)}",
        f"Images failed during face detection: {len(scan.detection_errors)}",
        f"Files copied: {len(plan) - copy_failures} of {len(plan)}",
        f"Copy failures: {copy_failures}",
    ]
    logger.info("\n".join(summary))
    write_report(os.path.join(output_dir, REPORT_NAME), summary, scan)
    if plan and copy_failures == len(plan):
        raise CopyError(f"Could not copy any files to {output_dir}")
    return n_persons
