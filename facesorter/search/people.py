"""
facesorter/search/people.py
Faces and named people for photo search: detect faces in indexed photos, cluster them into
"people", and keep the names the user gave across re-clustering.

Two steps, both incremental and both safe to interrupt:

* ``scan_faces`` runs the existing InsightFace pipeline (``facesorter.detection``) over the
  photos that have not been scanned yet (or whose file changed) and stores each face's box and
  embedding. Work is committed photo by photo, so Ctrl-C loses at most the photo in progress
  and the next run continues where it stopped.
* ``cluster_faces`` groups every stored embedding with the existing cosine DBSCAN
  (``facesorter.clustering``) and writes the cluster ids back **without** touching the
  pictures again. Names the user attached to clusters (``Store.name_cluster``) are carried
  over to the new clusters, so "Bob" survives adding more photos.

Naming a cluster is what makes ``with Bob`` work: ``Filters(people=("Bob",))`` selects photos
containing a face of the cluster named Bob.

insightface and scikit-learn are imported lazily (inside the functions that need them), so
importing this module is cheap.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .errors import ModelUnavailableError, SearchError
from .store import ClusterInfo, FaceRow, Store

logger = logging.getLogger(__name__)

#: Photos larger than this (longest side, pixels) are shrunk before face detection. A 12 MP
#: phone photo is ~4000 px wide; the detector works on a 640 px canvas anyway, so 1600 px keeps
#: small background faces detectable while making detection several times faster.
DETECTION_MAX_SIDE = 1600

#: ``cluster_faces`` refuses more faces than this. DBSCAN is given a dense N x N distance
#: matrix; measured peak memory is about 12.6 bytes per N squared (10 000 faces: ~1.3 GB,
#: 20 000 faces: ~5 GB), and it grows with N squared.
MAX_FACES = 10_000

_INSTALL_HINT = (
    "Face recognition needs InsightFace, which could not be imported. Install the search "
    'extras with:  pip install "facesorter[search]"  (insightface and onnxruntime are '
    "dependencies of the core package; if their installation failed, install them directly "
    "with: pip install insightface onnxruntime)."
)


# --------------------------------------------------------------------------- reports


@dataclass(frozen=True)
class ScanReport:
    """Outcome of ``scan_faces``.

    ``photos_scanned`` counts photos that were processed successfully (including those where
    no face was found); ``failed`` lists ``(path, error message)`` for the others, which stay
    unscanned and are retried next time.
    """

    photos_scanned: int
    faces_found: int
    photos_without_faces: int
    failed: tuple[tuple[str, str], ...]
    seconds: float


@dataclass(frozen=True)
class ClusterReport:
    """Outcome of ``cluster_faces``.

    ``faces`` is the number of faces that were clustered; ``clusters`` how many clusters
    (people candidates) they now form; ``named_kept`` how many user-given names were carried
    over, ``names_dropped`` the names that could not be (their cluster was merged into a
    bigger named one, or dissolved); ``singletons`` how many groups of exactly one face the
    clustering found (kept as one-face clusters when ``min_cluster_faces == 1``, left
    unassigned otherwise).
    """

    faces: int
    clusters: int
    named_kept: int
    names_dropped: tuple[str, ...]
    singletons: int
    seconds: float


# --------------------------------------------------------------------------- scanning


def _load_model() -> Any:
    """Load the InsightFace model, turning every failure into ``ModelUnavailableError``."""
    try:
        from facesorter.detection import get_face_model

        return get_face_model()
    except ImportError as exc:
        raise ModelUnavailableError(f"{_INSTALL_HINT} ({exc})") from exc
    except Exception as exc:  # noqa: BLE001 - e.g. the weights cannot be downloaded offline
        raise ModelUnavailableError(
            f"Could not load the face model ({type(exc).__name__}: {exc}). InsightFace "
            "downloads its 'buffalo_l' weights on first use; check the network connection."
        ) from exc


def _shrink(image: Image.Image, max_side: int) -> tuple[Image.Image, float, float]:
    """``(image to detect on, x scale, y scale)``; multiply detected coordinates by the scales
    to get back to the original image. Images within ``max_side`` are returned unchanged."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, 1.0, 1.0
    ratio = max_side / longest
    small = image.resize(
        (max(1, round(width * ratio)), max(1, round(height * ratio))),
        Image.Resampling.BILINEAR,
        reducing_gap=2.0,
    )
    return small, width / small.width, height / small.height


def _scan_one(
    store: Store, photo_id: int, path: str, model: Any, confidence: float, max_side: int
) -> int:
    """Detect, store and mark one photo; return the number of faces kept."""
    from facesorter.detection import detect_faces_and_embeddings
    from facesorter.utils import load_image

    image = load_image(path)
    width, height = image.size
    small, sx, sy = _shrink(image, max_side)
    rows = []
    for face in detect_faces_and_embeddings(model, small):
        if face.score < confidence:
            continue
        x1, y1, x2, y2 = face.box
        rows.append(
            FaceRow(
                cluster_id=None,
                x1=min(max(x1 * sx, 0.0), float(width)),
                y1=min(max(y1 * sy, 0.0), float(height)),
                x2=min(max(x2 * sx, 0.0), float(width)),
                y2=min(max(y2 * sy, 0.0), float(height)),
                embedding=face.embedding,
            )
        )
    store.set_faces(photo_id, rows)
    store.mark_faces_scanned(photo_id)
    return len(rows)


def scan_faces(
    store: Store,
    *,
    model: Any = None,
    confidence: float = 0.8,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ScanReport:
    """Detect faces in every photo that has not been scanned yet, and store them.

    Photos are those from ``store.photos_needing_face_scan(limit)``: never scanned, or whose
    size / modification time changed since (the store drops their old faces on re-indexing).
    Each photo is loaded with ``facesorter.utils.load_image`` (EXIF orientation applied, HEIC
    supported). Photos whose longest side exceeds ``DETECTION_MAX_SIDE`` (1600 px) are
    downscaled before detection for speed, and the boxes are scaled back, so stored boxes are
    in the coordinates of the full-size, EXIF-corrected image. Only faces with a detection
    score of at least ``confidence`` are kept. Faces and the "scanned" mark are written per
    photo, and the scan is sequential (the InsightFace session is not assumed thread-safe).

    Args:
        store: The photo index.
        model: A face model with ``get(img_bgr)`` (InsightFace ``FaceAnalysis`` or a fake).
            ``None`` loads InsightFace lazily, and only if there is something to scan.
        confidence: Minimum detection score, 0..1.
        limit: Scan at most this many photos (``None`` = all pending). Photos that failed
            earlier are still pending and are picked first, so with a tiny ``limit`` and
            many unreadable files a run can make no progress.
        progress: Called as ``progress(done, total)`` after every photo, failed or not.

    Returns:
        A ``ScanReport``. A photo that cannot be read or on which the model raises is listed
        in ``failed`` (not raised) and left unscanned, so it is retried on the next run.

    Raises:
        ModelUnavailableError: ``model`` is ``None`` and InsightFace or its weights cannot be
            loaded (the message says how to install it).
        ValueError: ``confidence`` outside 0..1, or a negative ``limit``.
        KeyboardInterrupt: propagated. Every photo finished before it is saved; running the
            scan again continues with the rest.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    start = time.perf_counter()
    pending = store.photos_needing_face_scan(limit)
    if pending and model is None:
        model = _load_model()

    scanned = faces_found = no_faces = 0
    failed: list[tuple[str, str]] = []
    total = len(pending)
    for done, photo in enumerate(pending, start=1):
        path = photo.meta.path
        try:
            count = _scan_one(store, photo.id, path, model, confidence, DETECTION_MAX_SIDE)
        except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the scan
            logger.warning("Face scan failed for %s: %s", path, exc)
            failed.append((path, f"{type(exc).__name__}: {exc}"))
        else:
            scanned += 1
            faces_found += count
            no_faces += count == 0
        if progress is not None:
            progress(done, total)

    report = ScanReport(
        photos_scanned=scanned,
        faces_found=faces_found,
        photos_without_faces=no_faces,
        failed=tuple(failed),
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Scanned %d photos: %d faces, %d without faces, %d failed",
        scanned,
        faces_found,
        no_faces,
        len(failed),
    )
    return report


# --------------------------------------------------------------------------- clustering


def _old_membership(store: Store) -> dict[tuple[int, int], int]:
    """``(photo_id, face_idx) -> cluster id`` for every currently clustered face."""
    membership = {}
    for photo_id in store.candidate_ids(None):
        for face_idx, row in enumerate(store.get_faces(int(photo_id))):
            if row.cluster_id is not None:
                membership[(int(photo_id), face_idx)] = row.cluster_id
    return membership


def _match_old_to_new(old_of_face: np.ndarray, labels: np.ndarray) -> list[tuple[int, int, int]]:
    """``(overlap, old cluster, new label)`` for every pair sharing faces, biggest overlap
    first; ties by lower old id, then lower new label (fully deterministic)."""
    keep = (old_of_face >= 0) & (labels >= 0)
    if not keep.any():
        return []
    pairs, counts = np.unique(
        np.stack([old_of_face[keep], labels[keep]], axis=1), axis=0, return_counts=True
    )
    triples = [(int(c), int(o), int(n)) for (o, n), c in zip(pairs, counts)]
    triples.sort(key=lambda t: (-t[0], t[1], t[2]))
    return triples


def cluster_faces(
    store: Store, *, threshold: float = 0.6, min_cluster_faces: int = 1
) -> ClusterReport:
    """Group all stored faces into people and keep the user's names.

    Uses ``facesorter.clustering.cluster_embeddings`` (DBSCAN, cosine distance, same as
    ``facesorter sort``). With the underlying ``min_samples=1`` every face joins some cluster
    and clusters grow by **single-linkage chaining**: if A is within ``threshold`` of B and B
    within ``threshold`` of C, all three are one person even if A and C are far apart, so one
    ambiguous face can merge two people (lower ``threshold`` to split, at the cost of
    splitting one person into several clusters). A cluster is only ever a candidate: the user
    decides who it is by naming it.

    ``min_cluster_faces`` is the stricter option: clusters with fewer faces stay unassigned
    (``cluster_id`` None), so one-off faces in crowds do not become "people". The default 1
    keeps every face in a cluster.

    **Names survive re-clustering.** Before clustering, the faces of each named cluster are
    recorded, and names are handed over by one deterministic global assignment. Every pair
    (old named cluster, new cluster) that shares faces is a candidate, weighted by the number
    of shared faces. Candidates are taken greedily by descending overlap (ties: the lower old
    cluster id, then the lower new cluster label; labels follow the order of each cluster's
    earliest face by photo id and face index). A candidate is accepted when neither the name
    nor the new cluster is used yet, so a new cluster gets at most one name and a name goes to
    at most one cluster. A name's best-overlap cluster is always acceptable; when that one was
    taken by a stronger claim (the clusters were merged), the name falls back to its next-best
    unclaimed cluster, but only if the shared faces are at least half of that cluster's
    faces, so a name never lands on a mostly foreign cluster. Names left over (merged away, or
    their cluster dissolved, for example by ``min_cluster_faces``) are dropped and reported in
    ``names_dropped``. New clusters reuse the id of the old cluster they overlap most (each id
    once, same tie-breaks), so ids stay stable for people who did not change; other clusters
    get fresh ids.

    Memory: DBSCAN gets a dense N x N matrix, so more than ``MAX_FACES`` (10 000) faces raise
    ``SearchError``. A higher clustering ``threshold`` does not help (the matrix size depends
    only on the number of faces); the remedy is fewer faces, e.g. scan a part of the library
    with ``scan_faces(limit=...)``, index fewer photos, or use a higher scan ``confidence``.

    Raises:
        SearchError: more than ``MAX_FACES`` faces, or stored embeddings of mixed sizes.
        ValueError: ``threshold`` negative or ``min_cluster_faces`` below 1.
    """
    if not threshold >= 0.0:  # also rejects NaN
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if min_cluster_faces < 1:
        raise ValueError(f"min_cluster_faces must be >= 1, got {min_cluster_faces}")
    start = time.perf_counter()
    photo_ids, face_idx, embeddings = store.face_embeddings()
    n = len(photo_ids)
    if n == 0:
        return ClusterReport(0, 0, 0, (), 0, time.perf_counter() - start)
    if n > MAX_FACES:
        raise SearchError(
            f"{n} faces is too many to cluster at once (limit {MAX_FACES}: clustering builds "
            "an N x N distance matrix, so memory grows with the square). Index or scan fewer "
            "photos (--limit), or use a higher face-detection confidence. A higher clustering "
            "threshold does not reduce the memory needed."
        )

    from facesorter.clustering import cluster_embeddings  # lazy: imports scikit-learn

    old_clusters: list[ClusterInfo] = store.clusters()
    old_names = {c.cluster_id: c.name for c in old_clusters if c.name}
    membership = _old_membership(store) if old_clusters else {}

    raw = cluster_embeddings(embeddings, threshold)
    sizes = np.bincount(raw)
    labels = np.where(sizes[raw] >= min_cluster_faces, raw, -1)
    singletons = int((sizes == 1).sum())

    # Match old clusters to new ones: ids first (every old cluster), then names.
    old_of_face = np.array(
        [membership.get((int(p), int(i)), -1) for p, i in zip(photo_ids, face_idx)],
        dtype=np.int64,
    )
    triples = _match_old_to_new(old_of_face, labels)
    new_labels = sorted(int(x) for x in np.unique(labels[labels >= 0]))

    new_id: dict[int, int] = {}
    used_old: set[int] = set()
    for _, old, new in triples:
        if old not in used_old and new not in new_id:
            new_id[new] = old
            used_old.add(old)
    next_id = max((c.cluster_id for c in old_clusters), default=0) + 1
    for label in new_labels:
        if label not in new_id:
            new_id[label] = next_id
            next_id += 1

    # Names: greedy global assignment (see the docstring).
    carried: dict[int, int] = {}  # new label -> old named cluster that gives it its name
    named_done: set[int] = set()
    seen_old: set[int] = set()
    for overlap, old, new in triples:
        if old not in old_names:
            continue
        first_choice = old not in seen_old  # triples are sorted: its best-overlap cluster
        seen_old.add(old)
        if old in named_done or new in carried:
            continue
        if first_choice or 2 * overlap >= sizes[new]:
            carried[new] = old
            named_done.add(old)
    names = {new_id[new]: old_names[old] for new, old in carried.items()}
    kept_old = named_done
    dropped = tuple(old_names[old] for old in sorted(old_names) if old not in kept_old)

    assignments: dict[tuple[int, int], int | None] = {
        (int(p), int(i)): (None if label < 0 else new_id[int(label)])
        for p, i, label in zip(photo_ids, face_idx, labels)
    }
    for key in membership:  # clustered faces without an embedding cannot be re-clustered:
        assignments.setdefault(key, None)  # leave them unassigned, never under a stale id
    store.recluster(assignments, names=names)

    report = ClusterReport(
        faces=n,
        clusters=len(new_labels),
        named_kept=len(names),
        names_dropped=dropped,
        singletons=singletons,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Clustered %d faces into %d clusters (%d names kept, %d dropped)",
        n,
        report.clusters,
        report.named_kept,
        len(dropped),
    )
    return report


def cluster_summary(store: Store) -> list[ClusterInfo]:
    """The face clusters, biggest first (by face count, then cluster id), for listing."""
    return sorted(store.clusters(), key=lambda c: (-c.face_count, c.cluster_id))
