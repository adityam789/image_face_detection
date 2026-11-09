"""
facesorter/search/indexer.py
Incremental indexing pipeline: scan a photo library, read metadata, label places, embed the
pixels and write everything to the :class:`~facesorter.search.store.Store`.

The sequence (cheapest work first, so a re-run over an unchanged library is just ``stat``):

1. ``iter_library`` lists the still images (canonical paths, deterministic order).
2. Each file is ``os.stat``-ed and compared with ``store.signatures()``; unchanged files are
   skipped without opening them (``force=True`` redoes everything). ``limit`` caps how many
   new / changed files are processed in one run.
3. ``read_metadata`` for the new / changed files (unreadable ones are reported, not fatal),
   then one vectorised gazetteer lookup labels the photos that have GPS.
4. ``iter_decoded`` decodes in a thread pool; images are embedded ``batch_size`` at a time and
   written ``chunk_size`` photos per transaction. A crash or Ctrl-C therefore keeps every
   finished chunk (the partly filled one is flushed best-effort) and the next run only does
   the remainder.
5. After a complete, unlimited scan, photos under the library root that no longer exist are
   removed from the index. A scan that finds nothing while the index holds photos from that
   root is refused (an unmounted drive must never wipe the index) unless the caller opts in.

Library files are only ever read. Heavy imports (torch, the gazetteer data) happen lazily in
``open_indexer``.

Lifecycle limits worth knowing:

* Identity is the canonical path (``abspath`` of the root plus the relative path; symlinks
  are not resolved). A renamed or moved file is therefore a new photo: it is embedded again
  and the faces / feedback attached to the old path are lost with it.
* Renaming a whole library folder leaves the entries of the old root in the index as ghosts,
  because only the root being indexed is ever pruned. Remove them with
  :meth:`Indexer.remove_root` (on the old spelling) before or after indexing the new one.
* A root that is spelled differently but is the same folder (other case on a case-insensitive
  filesystem, or through a symlink) reuses the spelling already stored in ``library_roots``,
  so one photo never gets two rows.
* A folder that cannot be listed makes the scan incomplete: its photos are kept in the index
  and the folder is reported in ``IndexReport.skipped_dirs``.
* Two indexers running on the same root at once leave a consistent index but each may report
  inflated ``new`` / ``updated`` counts for photos the other also did.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..paths import is_inside, same_path
from .encoders import DEFAULT_MODEL, iter_decoded, load_image_encoder
from .errors import IndexMismatchError, SearchError, UnreadableImageError
from .library import iter_library
from .metadata import read_metadata
from .store import Store, canonical_path
from .types import Gazetteer, ImageEncoder, PhotoMeta, PlaceLabel

logger = logging.getLogger(__name__)

__all__ = ["IndexReport", "Indexer", "open_indexer"]

_SCAN_PROGRESS_EVERY = 500
_META_ROOTS = "library_roots"
_META_INDEXED_AT = "indexed_at"

Progress = Callable[[int, int, str], None]
_Record = tuple[PhotoMeta, "PlaceLabel | None", np.ndarray]


@dataclass(frozen=True)
class IndexReport:
    """What one :meth:`Indexer.run` did.

    ``scanned`` counts every still image found under the root. Each is ``unchanged``
    (skipped), ``new`` / ``updated`` (embedded and stored), in ``failed`` (``(path, reason)``;
    retried by the next run), or, with ``limit``, deferred to a later run. ``removed`` counts
    index entries deleted because their file is gone. ``skipped_dirs`` lists folders that could
    not be listed (permissions, I/O error): the scan was incomplete there, so nothing under
    them was pruned.
    """

    scanned: int
    new: int
    updated: int
    unchanged: int
    removed: int
    failed: tuple[tuple[str, str], ...]
    seconds: float
    model_name: str
    skipped_dirs: tuple[str, ...] = ()


class Indexer:
    """Indexes photo libraries into a ``Store`` with a given image encoder.

    The indexer takes ownership of ``store``: :meth:`close` closes it. ``gazetteer`` may be
    ``None`` (no place labels). ``workers`` is the decode / metadata thread count,
    ``batch_size`` the number of images per encoder call, ``chunk_size`` the number of photos
    per store transaction.
    """

    def __init__(
        self,
        store: Store,
        image_encoder: ImageEncoder,
        gazetteer: Gazetteer | None,
        *,
        workers: int = 4,
        batch_size: int = 32,
        chunk_size: int = 256,
    ) -> None:
        for label, value in (("workers", workers), ("batch_size", batch_size)):
            if value < 1:
                raise ValueError(f"{label} must be at least 1, got {value}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
        if image_encoder.name != store.model_name or image_encoder.dim != store.dim:
            raise IndexMismatchError(
                f"The index was built with model {store.model_name!r} (dim {store.dim}) but the "
                f"image encoder is {image_encoder.name!r} (dim {image_encoder.dim}). Re-index "
                "with the same model, or into a new index file."
            )
        self.store = store
        self.encoder = image_encoder
        self.gazetteer = gazetteer
        self.workers = workers
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Close the store. Safe to call twice."""
        self.store.close()

    def __enter__(self) -> Indexer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ run

    def run(
        self,
        root: str | os.PathLike[str],
        *,
        exclude_dirs: Iterable[str | os.PathLike[str]] = (),
        force: bool = False,
        limit: int | None = None,
        progress: Progress | None = None,
        allow_empty_prune: bool = False,
    ) -> IndexReport:
        """Index (or incrementally update) the photos under ``root``.

        ``force`` re-embeds every photo, even unchanged ones. ``limit`` processes at most that
        many new / changed photos (a trial run; it never prunes). ``progress(done, total,
        stage)`` is called with stage ``"scan"`` (``total`` is 0 while still counting, then
        equals ``done``), ``"metadata"``, ``"encode"`` and ``"store"`` (``total`` = photos to
        process). The index's own directory is skipped when it lies inside the library.

        Raises ``InvalidPathError`` if ``root`` is not a folder and ``SearchError`` if the scan
        finds no photos although the index holds photos from this root (pass
        ``allow_empty_prune=True`` to really remove them). A photo that cannot be read goes in
        ``IndexReport.failed``; an encoder failure aborts the run (finished chunks are kept).
        """
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        started = time.perf_counter()
        root_path = self._canonical_root(canonical_path(root))
        report = progress or _no_progress

        existing, unchanged, todo, failed, skipped = self._scan(
            root_path, exclude_dirs, force, report
        )
        indexed_here = _under(self.store.signatures(), root_path)
        if not existing and indexed_here and not allow_empty_prune:
            raise SearchError(
                f"No photos were found under {root_path}, but the index holds "
                f"{len(indexed_here)} photo(s) from there. Is the drive unmounted or the "
                "folder moved? Nothing was changed. Pass allow_empty_prune=True (CLI: "
                "--allow-empty-prune) to remove them from the index anyway."
            )
        self._record_root(root_path)

        if limit is not None:
            todo = todo[:limit]
        new = updated = 0
        if todo:
            known = set(indexed_here)
            new, updated, bad = self._process(todo, root_path, known, report)
            failed.extend(bad)

        removed = 0
        if limit is None:
            stale = [
                p
                for p in indexed_here
                if p not in existing and not any(_is_under(p, d) for d in skipped)
            ]
            if stale:
                removed = self.store.delete_paths(stale)
                logger.info("Removed %d vanished photo(s) from the index", removed)
        self.store.set_meta(
            _META_INDEXED_AT, datetime.now().astimezone().isoformat(timespec="seconds")
        )
        return IndexReport(
            scanned=len(existing),
            new=new,
            updated=updated,
            unchanged=unchanged,
            removed=removed,
            failed=tuple(failed),
            seconds=time.perf_counter() - started,
            model_name=self.encoder.name,
            skipped_dirs=tuple(skipped),
        )

    def remove_root(self, root: str | os.PathLike[str]) -> int:
        """Forget a library folder: delete every index entry under ``root`` and drop it from
        ``library_roots``. Returns the number of photos removed.

        ``root`` does not need to exist any more (this is how the ghosts of a renamed or
        deleted library are cleaned up); a spelling of the same folder as a stored root is
        matched to it. Photos of a root nested inside ``root`` go too. Library files are not
        touched.
        """
        root_path = self._canonical_root(canonical_path(root))
        doomed = _under(self.store.signatures(), root_path)
        removed = self.store.delete_paths(doomed) if doomed else 0
        roots = self._stored_roots()
        if root_path in roots:
            self.store.set_meta(_META_ROOTS, json.dumps([r for r in roots if r != root_path]))
        logger.info("Removed %d photo(s) of %s from the index", removed, root_path)
        return removed

    # ------------------------------------------------------------------ stages

    def _scan(
        self,
        root_path: str,
        exclude_dirs: Iterable[str | os.PathLike[str]],
        force: bool,
        progress: Progress,
    ) -> tuple[set[str], int, list[str], list[tuple[str, str]], list[str]]:
        """List the library and split it into unchanged files and files to (re)index.

        The last item lists folders that could not be listed (the scan is incomplete there)."""
        excluded = [os.fspath(d) for d in exclude_dirs]
        index_dir = str(self.store.path.parent)
        # The index's own folder is skipped, unless that is the library root itself: then
        # excluding it would exclude every photo (the index file is not an image anyway).
        if is_inside(index_dir, root_path) and not same_path(index_dir, root_path):
            excluded.append(index_dir)
        signatures = self.store.signatures()
        existing: set[str] = set()
        todo: list[str] = []
        failed: list[tuple[str, str]] = []
        skipped: list[str] = []
        unchanged = 0

        def on_skip(directory: Path, exc: OSError) -> None:
            logger.warning("Could not list %s (%s); its photos stay in the index", directory, exc)
            skipped.append(canonical_path(directory))

        for found in iter_library(root_path, exclude_dirs=excluded, on_skip=on_skip):
            path = canonical_path(found)
            existing.add(path)
            try:
                st = os.stat(path)
            except OSError as exc:  # vanished or unreadable since the directory listing
                failed.append((path, f"Cannot read {path}: {exc}"))
                continue
            if not force and signatures.get(path) == (st.st_size, st.st_mtime_ns):
                unchanged += 1
            else:
                todo.append(path)
            if len(existing) % _SCAN_PROGRESS_EVERY == 0:
                progress(len(existing), 0, "scan")
        progress(len(existing), len(existing), "scan")
        return existing, unchanged, todo, failed, skipped

    def _stored_roots(self) -> list[str]:
        """The library roots recorded in the index (tolerating a damaged setting)."""
        try:
            stored = json.loads(self.store.get_meta(_META_ROOTS) or "[]")
        except ValueError:
            logger.warning("Ignoring unreadable %r setting in the index", _META_ROOTS)
            return []
        return [r for r in stored if isinstance(r, str)] if isinstance(stored, list) else []

    def _canonical_root(self, root_path: str) -> str:
        """The spelling to use for ``root_path``: the stored one if it is the same folder."""
        roots = self._stored_roots()
        if root_path in roots:
            return root_path
        for known in roots:
            if same_path(root_path, known):
                logger.info(
                    "%s is the library %s already in the index; using that", root_path, known
                )
                return known
        return root_path

    def _record_root(self, root_path: str) -> None:
        """Remember the library root in the index (merged with earlier roots, deduplicated)."""
        roots = self._stored_roots()
        if root_path not in roots:
            roots.append(root_path)
        self.store.set_meta(_META_ROOTS, json.dumps(roots))

    def _process(
        self, todo: list[str], root_path: str, known: set[str], progress: Progress
    ) -> tuple[int, int, list[tuple[str, str]]]:
        """Metadata -> places -> decode -> embed -> store. Returns ``(new, updated, failed)``."""
        total = len(todo)
        failed: list[tuple[str, str]] = []
        metas = self._read_all(todo, root_path, failed, progress)
        places = self._label_places(metas)
        by_path = {m.path: (m, p) for m, p in zip(metas, places)}

        pending: list[_Record] = []
        batch_paths: list[str] = []
        batch_images: list[object] = []
        counts = {"new": 0, "updated": 0, "encoded": 0, "stored": 0}

        def flush() -> None:
            if not pending:
                return
            self.store.upsert_photos(pending)
            for meta, _, _ in pending:
                counts["updated" if meta.path in known else "new"] += 1
            counts["stored"] += len(pending)
            pending.clear()
            progress(counts["stored"], total, "store")

        def encode_batch() -> None:
            if not batch_images:
                return
            vectors = self._encode(batch_images, batch_paths)
            for path, vec in zip(batch_paths, vectors):
                meta, place = by_path[path]
                pending.append((meta, place, vec))
            counts["encoded"] += len(batch_images)
            batch_images.clear()
            batch_paths.clear()
            progress(counts["encoded"], total, "encode")
            if len(pending) >= self.chunk_size:
                flush()

        stream = iter_decoded([m.path for m in metas], workers=self.workers)
        try:
            for path, image in stream:
                key = str(path)
                if isinstance(image, Exception):
                    failed.append((key, str(image)))
                    continue
                batch_paths.append(key)
                batch_images.append(image)
                if len(batch_images) >= self.batch_size:
                    encode_batch()
            encode_batch()
            flush()
        except BaseException:
            # Ctrl-C or a crash: keep what is already embedded so the next run resumes.
            try:
                flush()
            except Exception:  # pragma: no cover - the original error matters more
                logger.warning("Could not save the last partial chunk", exc_info=True)
            raise
        finally:
            stream.close()
        return counts["new"], counts["updated"], failed

    def _read_all(
        self, todo: list[str], root_path: str, failed: list[tuple[str, str]], progress: Progress
    ) -> list[PhotoMeta]:
        """Read metadata for ``todo`` in a thread pool; unreadable files go to ``failed``."""
        total = len(todo)
        metas: list[PhotoMeta] = []
        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="meta")
        try:
            for done, result in enumerate(pool.map(self._read_one, todo, [root_path] * total), 1):
                if isinstance(result, tuple):
                    failed.append(result)
                else:
                    metas.append(result)
                progress(done, total, "metadata")
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return metas

    @staticmethod
    def _read_one(path: str, root: str) -> PhotoMeta | tuple[str, str]:
        try:
            return read_metadata(path, library_root=root)
        except UnreadableImageError as exc:
            logger.warning("Skipping unreadable image: %s", exc)
            return (path, str(exc))

    def _label_places(self, metas: list[PhotoMeta]) -> list[PlaceLabel | None]:
        """Place label per photo (``None`` without GPS, without a gazetteer, or off the map)."""
        labels: list[PlaceLabel | None] = [None] * len(metas)
        if self.gazetteer is None:
            return labels
        gps = [i for i, m in enumerate(metas) if m.lat is not None and m.lon is not None]
        if not gps:
            return labels
        lats = [float(metas[i].lat) for i in gps]  # type: ignore[arg-type]
        lons = [float(metas[i].lon) for i in gps]  # type: ignore[arg-type]
        many = getattr(self.gazetteer, "label_many", None)
        found = many(lats, lons) if callable(many) else [
            self.gazetteer.label_for(a, b) for a, b in zip(lats, lons)
        ]  # fmt: skip
        for i, label in zip(gps, found):
            labels[i] = label
        return labels

    def _encode(self, images: list[object], paths: list[str]) -> np.ndarray:
        """Embed one batch; a failing or misshapen encoder is fatal (it is not the file's fault)."""
        try:
            vectors = np.asarray(self.encoder.encode_images(images))  # type: ignore[arg-type]
        except Exception as exc:
            raise SearchError(
                f"The image encoder failed on a batch starting at {paths[0]}: {exc}"
            ) from exc
        if vectors.shape != (len(images), self.encoder.dim):
            raise SearchError(
                f"The image encoder returned shape {vectors.shape}, expected "
                f"{(len(images), self.encoder.dim)}."
            )
        return vectors


# ---------------------------------------------------------------------------- helpers


def _no_progress(done: int, total: int, stage: str) -> None:
    return None


def _is_under(path: str, directory: str) -> bool:
    """Whether canonical ``path`` is inside canonical ``directory`` (string prefix)."""
    prefix = directory if directory.endswith(os.sep) else directory + os.sep
    return path.startswith(prefix)


def _under(signatures: dict[str, tuple[int, int]], root_path: str) -> list[str]:
    """Indexed paths located inside ``root_path`` (string prefix on canonical paths)."""
    return [p for p in signatures if _is_under(p, root_path)]


def open_indexer(
    index_path: str | os.PathLike[str],
    *,
    device: str = "auto",
    model: str = DEFAULT_MODEL,
    workers: int = 4,
    batch_size: int = 32,
) -> Indexer:
    """Create (or open) the index at ``index_path`` and return a ready :class:`Indexer`.

    Loads the image encoder for ``model`` on ``device`` and the offline gazetteer, and creates
    the index file (and its parent folder) if needed with the encoder's name and dimension.
    An existing index built with another model raises ``IndexMismatchError``; a missing model
    or runtime raises ``ModelUnavailableError``. Call ``close()`` when done.
    """
    from .geo import build_gazetteer

    encoder = load_image_encoder(model, device)
    store = Store.open(Path(index_path), model_name=encoder.name, dim=encoder.dim, create=True)
    try:
        return Indexer(store, encoder, build_gazetteer(), workers=workers, batch_size=batch_size)
    except BaseException:
        store.close()
        raise
