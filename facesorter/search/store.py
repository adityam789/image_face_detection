"""
facesorter/search/store.py
The photo index: SQLite metadata + embedding vectors + filtered exact cosine search.

One SQLite file (WAL, transactional) holds everything: photo metadata, float16 embedding
BLOBs, face clusters, person names and search feedback. Embeddings live in the same database
(not a side ``.npy``) so a crash can never leave metadata and vectors out of step.

* **Search.** Embeddings are L2-normalised, so cosine similarity is a dot product. They are
  loaded lazily into one in-memory ``float32`` matrix (row ``i`` <-> photo ``_ids[i]``) and
  searched exactly with a NumPy matmul (single-digit to low tens of milliseconds for
  100k-300k vectors). There is no ANN index. Filters (date, place, people) are applied in SQL
  first; if they leave under ``gather_threshold`` of the library the surviving rows are
  gathered and multiplied, otherwise the whole matrix is multiplied and the survivors picked.
  Both plans give the same answer.
* **Concurrency.** One SQLite connection (``check_same_thread=False``) and one re-entrant
  lock: every public method takes the lock, so a Store can be shared by a UI thread pool and
  an indexing thread. Searches are therefore serialised (each is ~10 ms). Other *processes*
  (for example ``facesorter index`` while ``serve`` runs) are detected through
  ``PRAGMA data_version`` and trigger a lazy reload of the matrix.
* **Determinism.** Scores are rounded to 6 decimals before ranking and ties are broken by
  ascending photo id, so results do not depend on BLAS blocking or on the search plan.

Only the standard library and NumPy are used.
"""

from __future__ import annotations

import calendar
import logging
import math
import os
import sqlite3
import threading
import unicodedata
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .errors import IndexMismatchError, IndexNotFoundError, SearchError
from .types import Filters, IndexedPhoto, PhotoMeta, PlaceFilter, PlaceLabel

logger = logging.getLogger(__name__)

__all__ = [
    "ClusterInfo",
    "DuplicateNameError",
    "FaceRow",
    "FeedbackRow",
    "FeedbackSummary",
    "QueryFeedback",
    "SCHEMA_VERSION",
    "Store",
    "StoreStats",
    "fold",
    "canonical_path",
    "haversine_km",
]

SCHEMA_VERSION = 3
_APPLICATION_ID = 0x46535243  # "FSRC": marks the file as a facesorter search index
_EARTH_RADIUS_KM = 6371.0088
_TRUSTED_DATE_SOURCES = ("exif", "folder")  # mtime-derived dates are not capture dates
_DATE_SOURCES = ("exif", "folder", "mtime", "none")
_DATE_PRECISIONS = ("day", "month", "year")
_BROWSE_ORDER_SQL = {
    "newest": "taken IS NULL, taken DESC, id",
    "oldest": "taken IS NULL, taken, id",
    "id": "id",
}
_SQL_CHUNK = 500  # stays under SQLite's historical 999 host-parameter limit
_GATHER_BLOCK = 20_000  # rows per matmul block on the gather path (bounds temporary memory)
_SCORE_DECIMALS = 6
_MIN_NORM = 1e-6
_SAMPLE_PHOTOS = 6
_RESERVED_META_KEYS = frozenset({"model_name", "dim", "created", "schema", "schema_version"})

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    taken TEXT,
    day INTEGER,
    day_lo INTEGER,
    day_hi INTEGER,
    date_precision TEXT NOT NULL DEFAULT 'day',
    date_source TEXT NOT NULL,
    lat REAL,
    lon REAL,
    camera TEXT,
    city TEXT, region TEXT, country TEXT, country_code TEXT,
    city_key TEXT, region_key TEXT, country_key TEXT
);
CREATE INDEX photos_day ON photos(day_lo, day_hi, date_source);  -- covering for date filters
CREATE INDEX photos_lat ON photos(lat, lon);  -- covering for the near bounding box
CREATE INDEX photos_city ON photos(city_key);
CREATE INDEX photos_region ON photos(region_key);
CREATE INDEX photos_country ON photos(country_key);
CREATE TABLE embeddings (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    vec BLOB NOT NULL
);
CREATE TABLE faces (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    face_idx INTEGER NOT NULL,
    cluster_id INTEGER,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    embedding BLOB,  -- little-endian float16, unit length, sized by the face model
    PRIMARY KEY (photo_id, face_idx)
);
CREATE INDEX faces_cluster ON faces(cluster_id);
CREATE TABLE face_scan (
    photo_id INTEGER PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    scanned_ts TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL
);
CREATE TABLE people (
    cluster_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE
);
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    query TEXT NOT NULL,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    relevant INTEGER NOT NULL
);
CREATE INDEX feedback_query ON feedback(query, photo_id);
"""


# --------------------------------------------------------------------------- public types


class DuplicateNameError(SearchError, ValueError):
    """A person name is already attached to a different face cluster."""


@dataclass(frozen=True)
class FaceRow:
    """One detected face: its cluster (``None`` = not clustered), pixel box and, optionally,
    its face-recognition embedding (any length; stored as unit-length float16 and returned as
    unit-length float32). The embedding is excluded from equality comparisons."""

    cluster_id: int | None
    x1: float
    y1: float
    x2: float
    y2: float
    embedding: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ClusterInfo:
    """A group of faces believed to be one person.

    ``sample_photo_ids`` are up to 6 photos, those with the largest face of this cluster first.
    """

    cluster_id: int
    face_count: int
    photo_count: int
    name: str | None
    sample_photo_ids: tuple[int, ...]


@dataclass(frozen=True)
class FeedbackRow:
    id: int
    ts: str  # UTC, ISO-8601
    query: str
    photo_id: int
    relevant: bool


@dataclass(frozen=True)
class QueryFeedback:
    """Precision of the rated hits of one query (latest rating per photo wins)."""

    query: str
    rated: int
    relevant: int
    precision: float


@dataclass(frozen=True)
class FeedbackSummary:
    """``precision`` pools every rating; ``mean_precision`` averages the per-query values."""

    per_query: tuple[QueryFeedback, ...]
    rated: int
    relevant: int
    precision: float | None
    mean_precision: float | None


@dataclass(frozen=True)
class StoreStats:
    """Counts describing the index. Dates span only exif/folder-dated photos."""

    photos: int
    with_gps: int
    with_place: int
    by_date_source: dict[str, int]
    date_min: date | None
    date_max: date | None
    faces: int
    named_people: int
    feedback: int


# --------------------------------------------------------------------------- helpers


_FOLD_MAP = str.maketrans(
    {"ø": "o", "đ": "d", "ł": "l", "æ": "ae", "œ": "oe", "ı": "i", "ð": "d", "þ": "th"}
)


def fold(text: str) -> str:
    """Case-, accent- and whitespace-insensitive key: ``"  São  Paulo"`` -> ``"sao paulo"``.

    Also maps letters that have no decomposition (``ø æ œ ß đ ł ı ð þ``) so that a hand-typed
    ``Tromso`` matches a stored ``Tromsø`` (same folding as the gazetteer).
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold()).translate(_FOLD_MAP)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.split())


def canonical_path(path: str | os.PathLike[str]) -> str:
    """``os.path.abspath(os.path.expanduser(path))`` as ``str`` (symlinks are not resolved).

    The form every path in the index is stored and compared in. Matching is case-sensitive.
    """
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError(f"photo paths must be str or PathLike[str], got {type(path).__name__}")
    return os.path.abspath(os.path.expanduser(raw))


def _day_span(taken: datetime, precision: str) -> tuple[int, int]:
    """First and last calendar day (ordinals) that a photo with this precision covers."""
    d = taken.date()
    if precision == "month":
        last = calendar.monthrange(d.year, d.month)[1]
        return d.replace(day=1).toordinal(), d.replace(day=last).toordinal()
    if precision == "year":
        return date(d.year, 1, 1).toordinal(), date(d.year, 12, 31).toordinal()
    return d.toordinal(), d.toordinal()


def haversine_km(lat1: object, lon1: object, lat2: object, lon2: object) -> np.ndarray:
    """Great-circle distance in km (broadcasts over NumPy arrays; inputs in degrees)."""
    p1 = np.radians(np.asarray(lat1, dtype=np.float64))
    p2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _key_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    key = fold(value)
    return key or None


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _chunks(items: Sequence[object], size: int = _SQL_CHUNK) -> Iterator[Sequence[object]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def _normalise(vec: object, dim: int | None, what: str) -> np.ndarray:
    """Validate shape / dtype / finiteness and return a unit-length float32 vector.

    ``dim=None`` accepts any non-empty 1-D length (face embeddings)."""
    arr = np.asarray(vec)
    if arr.dtype.kind != "f":
        raise ValueError(f"{what}: expected a floating-point array, got dtype {arr.dtype}")
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if dim is None:
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{what}: expected a non-empty 1-D vector, got shape {arr.shape}")
    elif arr.shape != (dim,):
        raise ValueError(f"{what}: expected shape ({dim},), got {arr.shape}")
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{what}: contains NaN or infinity")
    norm = float(np.linalg.norm(arr))
    if norm < _MIN_NORM:
        raise ValueError(f"{what}: zero vector cannot be normalised")
    return arr / norm


def _bbox_clauses(lat: float, lon: float, radius_km: float) -> tuple[str, list[float]]:
    """SQL (without ``WHERE``) selecting a lat/lon box that contains the search circle."""
    ang = radius_km / _EARTH_RADIUS_KM  # angular radius in radians
    pad = 1e-6
    dlat = math.degrees(ang) + pad
    lat_lo, lat_hi = max(-90.0, lat - dlat), min(90.0, lat + dlat)
    sql = "lat BETWEEN ? AND ?"
    params: list[float] = [lat_lo, lat_hi]
    cos_lat = math.cos(math.radians(lat))
    if lat - dlat <= -90.0 or lat + dlat >= 90.0 or ang >= math.pi / 2 or cos_lat <= 1e-12:
        return sql, params  # circle reaches a pole: every longitude can match
    ratio = math.sin(ang) / cos_lat
    if ratio >= 1.0:  # pragma: no cover - guards float edge cases of the pole test above
        return sql, params
    dlon = math.degrees(math.asin(ratio)) + pad
    lo, hi = lon - dlon, lon + dlon
    if lo < -180.0:
        sql += " AND (lon >= ? OR lon <= ?)"
        params += [lo + 360.0, hi]
    elif hi > 180.0:
        sql += " AND (lon >= ? OR lon <= ?)"
        params += [lo, hi - 360.0]
    else:
        sql += " AND lon BETWEEN ? AND ?"
        params += [lo, hi]
    return sql, params


def _select_top(scores: np.ndarray, ids: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Best ``k`` (id, score) pairs; ties (after rounding) broken by ascending id."""
    keys = np.round(scores.astype(np.float64), _SCORE_DECIMALS)
    m = len(keys)
    if m > k:
        threshold = np.partition(keys, m - k)[m - k]
        keep = np.flatnonzero(keys >= threshold)
    else:
        keep = np.arange(m)
    order = np.lexsort((ids[keep], -keys[keep]))[:k]
    chosen = keep[order]
    return [(int(ids[i]), float(keys[i])) for i in chosen]


# --------------------------------------------------------------------------- the store


class Store:
    """The photo index. Create with :meth:`open`; use as a context manager or call ``close``.

    ``gather_threshold`` (default 0.2) is the candidate fraction of the library below which
    filtered search gathers the candidate rows instead of multiplying the whole matrix.
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        path: Path,
        model_name: str,
        dim: int,
        gather_threshold: float = 0.2,
    ) -> None:
        self._db: sqlite3.Connection | None = db
        self._path = path
        self._model_name = model_name
        self._dim = dim
        self.gather_threshold = gather_threshold
        self._lock = threading.RLock()
        self._loaded = False
        self._data_version = -1
        self._mat = np.zeros((0, dim), dtype=np.float32)
        self._ids = np.zeros(0, dtype=np.int64)
        self._row_of = np.zeros(0, dtype=np.int64)  # photo id -> matrix row, -1 if absent
        self._n = 0

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        model_name: str | None = None,
        dim: int | None = None,
        create: bool = True,
        gather_threshold: float = 0.2,
    ) -> Store:
        """Open (or create) the index at ``path``.

        ``model_name`` and ``dim`` must match an existing index; pass ``None`` to adopt the
        index's own values (how the search engine learns which encoder to load). Both are
        required to create a new index.

        Raises ``IndexNotFoundError`` (missing file and ``create=False``, or an empty file
        with ``create=False``) and ``IndexMismatchError`` (not a facesorter index, a different
        schema version, model or dimension - the message says to re-index).
        """
        p = Path(path).expanduser()
        if p.is_dir():
            raise SearchError(f"{p} is a directory; give the path of the index file.")
        existed = p.exists()
        if not existed and not create:
            raise IndexNotFoundError(f"No photo index at {p}. Run `facesorter index` first.")
        if not existed:
            cls._check_create_args(model_name, dim)  # before any file is made
            p.parent.mkdir(parents=True, exist_ok=True)
        try:
            db = sqlite3.connect(str(p), isolation_level=None, check_same_thread=False, timeout=30)
        except sqlite3.Error as exc:  # pragma: no cover - connect is lazy; defensive
            raise SearchError(f"Cannot open the photo index at {p}: {exc}") from exc
        try:
            return cls._initialise(db, p, model_name, dim, create, gather_threshold)
        except BaseException:
            db.close()
            # Never leave a half-made (empty) file behind. A non-empty file belongs to
            # whichever process won a creation race, so it is left alone.
            if not existed and p.exists() and p.stat().st_size == 0:
                for suffix in ("", "-wal", "-shm"):
                    Path(str(p) + suffix).unlink(missing_ok=True)
            raise

    @staticmethod
    def _check_create_args(model_name: str | None, dim: int | None) -> None:
        if model_name is None or dim is None:
            raise ValueError("model_name and dim are required to create a new index")
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")

    @staticmethod
    def _read_state(db: sqlite3.Connection, p: Path) -> tuple[int, int, bool]:
        """``(application_id, user_version, has_any_table)``; unreadable files are refused."""
        try:
            db.execute("BEGIN")  # one snapshot: a concurrent creator must not be seen half-way
            try:
                app_id = db.execute("PRAGMA application_id").fetchone()[0]
                version = db.execute("PRAGMA user_version").fetchone()[0]
                has_tables = db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] > 0
            finally:
                if db.in_transaction:
                    db.execute("ROLLBACK")
        except sqlite3.DatabaseError as exc:
            raise IndexMismatchError(
                f"{p} is not a readable facesorter photo index ({exc}). "
                "Delete it and re-index to rebuild."
            ) from exc
        return app_id, version, has_tables

    @classmethod
    def _initialise(
        cls,
        db: sqlite3.Connection,
        p: Path,
        model_name: str | None,
        dim: int | None,
        create: bool,
        gather_threshold: float,
    ) -> Store:
        app_id, version, has_tables = cls._read_state(db, p)
        if not has_tables and app_id == 0 and version == 0:  # brand new / empty file
            if not create:
                raise IndexNotFoundError(f"No photo index at {p}. Run `facesorter index` first.")
            cls._check_create_args(model_name, dim)
            assert model_name is not None and dim is not None
            if cls._create_schema(db, model_name, dim):
                logger.info("created photo index %s (model %s, dim %d)", p, model_name, dim)
            else:
                logger.debug("lost a creation race for %s; validating the winner's index", p)
            # Validate what is on disk: our own index, or the race winner's.
            app_id, version, _ = cls._read_state(db, p)
        if app_id != _APPLICATION_ID:
            raise IndexMismatchError(
                f"{p} is not a facesorter photo index. Choose another path, or delete it "
                "and re-index."
            )
        if version != SCHEMA_VERSION:
            raise IndexMismatchError(
                f"{p} has schema version {version}, expected {SCHEMA_VERSION}. "
                "Delete it and re-index to rebuild."
            )
        try:
            meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
            stored_model, stored_dim = meta["model_name"], int(meta["dim"])
        except (sqlite3.DatabaseError, KeyError, ValueError) as exc:
            raise IndexMismatchError(
                f"{p} is damaged ({exc}). Delete it and re-index to rebuild."
            ) from exc
        if (model_name is not None and model_name != stored_model) or (
            dim is not None and dim != stored_dim
        ):
            raise IndexMismatchError(
                f"{p} was built with model {stored_model!r} (dim {stored_dim}) but "
                f"{model_name!r} (dim {dim}) was requested. Re-index to switch models."
            )
        model_name, dim = stored_model, stored_dim

        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA synchronous = NORMAL")
        try:
            db.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:  # pragma: no cover - read-only or exotic filesystem
            logger.debug("could not enable WAL for %s", p)
        assert model_name is not None and dim is not None
        return cls(db, p, model_name, dim, gather_threshold)

    @staticmethod
    def _create_schema(db: sqlite3.Connection, model_name: str, dim: int) -> bool:
        """Create the schema in one ``BEGIN IMMEDIATE`` transaction; race-safe.

        The emptiness check is repeated once the write lock is held, so when two processes
        create the same new index only one does the work. Returns ``False`` (having changed
        nothing) when someone else got there first.
        """
        db.execute("BEGIN IMMEDIATE")
        try:
            empty = (
                db.execute("PRAGMA application_id").fetchone()[0] == 0
                and db.execute("PRAGMA user_version").fetchone()[0] == 0
                and db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
            )
            if not empty:
                db.execute("ROLLBACK")
                return False
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    db.execute(statement)
            db.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [
                    ("model_name", model_name),
                    ("dim", str(dim)),
                    ("created", datetime.now(timezone.utc).isoformat(timespec="seconds")),
                ],
            )
            db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("COMMIT")
        except BaseException:  # pragma: no cover - a fresh file cannot fail here
            db.execute("ROLLBACK")
            raise
        return True

    def close(self) -> None:
        """Close the database and drop the in-memory matrix. Safe to call twice."""
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None
            self._invalidate()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------ plumbing

    def _require(self) -> sqlite3.Connection:
        if self._db is None:
            raise SearchError("The photo index is closed.")
        return self._db

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._require()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """A locked ``BEGIN IMMEDIATE`` ... ``COMMIT`` block (rolled back on error)."""
        with self._lock:
            db = self._require()
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            try:
                db.execute("COMMIT")
            except BaseException:
                self._invalidate()  # memory may be ahead of the database now
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover - already rolled back
                    pass
                raise

    def _invalidate(self) -> None:
        self._loaded = False
        self._mat = np.zeros((0, self._dim), dtype=np.float32)
        self._ids = np.zeros(0, dtype=np.int64)
        self._row_of = np.zeros(0, dtype=np.int64)
        self._n = 0

    def _ensure_loaded(self, db: sqlite3.Connection) -> None:
        """Load the matrix on first use; reload if another process changed the database."""
        version = db.execute("PRAGMA data_version").fetchone()[0]
        if self._loaded and version != self._data_version:
            logger.info("index changed on disk; reloading vectors")
            self._invalidate()
        if self._loaded:
            return
        total = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        capacity = max(1024, int(total * 1.2) + 1)
        mat = np.zeros((capacity, self._dim), dtype=np.float32)
        ids = np.zeros(capacity, dtype=np.int64)
        n = 0
        cursor = db.execute("SELECT photo_id, vec FROM embeddings ORDER BY photo_id")
        while True:
            rows = cursor.fetchmany(4096)
            if not rows:
                break
            blob = b"".join(r[1] for r in rows)
            if len(blob) != len(rows) * self._dim * 2:
                raise IndexMismatchError(
                    f"{self._path} has embeddings of the wrong size. Delete it and re-index."
                )
            block = np.frombuffer(blob, dtype="<f2").reshape(len(rows), self._dim)
            mat[n : n + len(rows)] = block
            ids[n : n + len(rows)] = [r[0] for r in rows]
            n += len(rows)
        self._mat, self._ids, self._n = mat, ids, n
        self._row_of = np.full(int(ids[:n].max()) + 1 if n else 0, -1, dtype=np.int64)
        self._row_of[ids[:n]] = np.arange(n)
        self._data_version = version
        self._loaded = True
        logger.debug("loaded %d vectors (%d x %d)", n, n, self._dim)

    def _ensure_row_map(self, max_id: int) -> None:
        if max_id >= len(self._row_of):
            grown = np.full(max(max_id + 1, 2 * len(self._row_of), 1024), -1, dtype=np.int64)
            grown[: len(self._row_of)] = self._row_of
            self._row_of = grown

    def _apply_vectors(self, items: list[tuple[int, np.ndarray]]) -> None:
        """Put (id, float32 vector) pairs into the loaded matrix (overwrite or append)."""
        if not self._loaded:
            return
        fresh = []
        for photo_id, vec in dict(items).items():  # one entry per id, the last vector wins
            self._ensure_row_map(photo_id)
            row = int(self._row_of[photo_id])
            if row >= 0:
                self._mat[row] = vec
            else:
                fresh.append((photo_id, vec))
        if not fresh:
            return
        need = self._n + len(fresh)
        if need > len(self._mat):
            capacity = max(need, int(len(self._mat) * 1.5) + 1)
            mat = np.zeros((capacity, self._dim), dtype=np.float32)
            mat[: self._n] = self._mat[: self._n]
            ids = np.zeros(capacity, dtype=np.int64)
            ids[: self._n] = self._ids[: self._n]
            self._mat, self._ids = mat, ids
        for offset, (photo_id, vec) in enumerate(fresh):
            row = self._n + offset
            self._mat[row] = vec
            self._ids[row] = photo_id
            self._row_of[photo_id] = row
        self._n = need

    def _remove_vectors(self, photo_ids: Sequence[int]) -> None:
        """Delete rows from the loaded matrix by moving tail rows into the holes."""
        if not self._loaded:
            return
        valid = [i for i in photo_ids if i < len(self._row_of) and self._row_of[i] >= 0]
        if not valid:
            return
        dead = np.sort(self._row_of[np.array(valid, dtype=np.int64)])
        new_n = self._n - len(dead)
        holes = dead[dead < new_n]
        tail = np.arange(new_n, self._n)
        movers = tail[~np.isin(tail, dead)]
        self._row_of[self._ids[dead]] = -1
        if len(holes):
            self._mat[holes] = self._mat[movers]
            self._ids[holes] = self._ids[movers]
            self._row_of[self._ids[holes]] = holes
        self._n = new_n

    # ------------------------------------------------------------------ writing photos

    def upsert_photos(
        self, records: Sequence[tuple[PhotoMeta, PlaceLabel | None, np.ndarray]]
    ) -> list[int]:
        """Insert or replace photos by ``path`` in one transaction; return their ids.

        The id of an existing path is kept. The embedding must be a floating-point vector of
        shape ``(dim,)`` with finite values; it is re-normalised and stored as float16. If a
        photo's size or mtime changed, its stored faces are dropped (they described old
        pixels). Every record is validated before anything is written, so a bad batch changes
        nothing (``ValueError``). If a path appears twice, the later record wins (both entries
        of the returned list then hold the same id).

        ``meta.path`` is canonicalised (``canonical_path``) on write, so ``Path`` / ``str`` /
        ``~`` / relative spellings of one file cannot create duplicate rows. Path uniqueness is
        case-sensitive. ``meta.date_precision`` is stored and used by the date filter.
        """
        prepared = []
        for i, (meta, place, vec) in enumerate(records):
            if meta.date_source not in _DATE_SOURCES:
                raise ValueError(f"record {i}: unknown date_source {meta.date_source!r}")
            if meta.date_precision not in _DATE_PRECISIONS:
                raise ValueError(f"record {i}: unknown date_precision {meta.date_precision!r}")
            prepared.append(
                (
                    replace(meta, path=canonical_path(meta.path)),
                    place,
                    _normalise(vec, self._dim, f"record {i} embedding"),
                )
            )
        if not prepared:
            return []
        with self._write() as db:
            ids = []
            for meta, place, vec in prepared:
                ids.append(self._upsert_one(db, meta, place, vec))
            try:
                self._apply_vectors(
                    [
                        (i, v.astype(np.float16).astype(np.float32))
                        for i, (_, _, v) in zip(ids, prepared)
                    ]
                )
            except BaseException:  # pragma: no cover - e.g. MemoryError
                self._invalidate()
                raise
        return ids

    def _upsert_one(
        self, db: sqlite3.Connection, meta: PhotoMeta, place: PlaceLabel | None, vec: np.ndarray
    ) -> int:
        taken = meta.taken
        place = place or PlaceLabel()
        day_lo, day_hi = (
            _day_span(taken, meta.date_precision) if taken is not None else (None, None)
        )
        values = {
            "size": int(meta.size),
            "mtime_ns": int(meta.mtime_ns),
            "width": meta.width,
            "height": meta.height,
            "taken": taken.isoformat() if taken is not None else None,
            "day": taken.date().toordinal() if taken is not None else None,
            "day_lo": day_lo,
            "day_hi": day_hi,
            "date_precision": meta.date_precision,
            "date_source": meta.date_source,
            "lat": _finite_or_none(meta.lat),
            "lon": _finite_or_none(meta.lon),
            "camera": meta.camera,
            "city": place.city,
            "region": place.region,
            "country": place.country,
            "country_code": place.country_code,
            "city_key": _key_or_none(place.city),
            "region_key": _key_or_none(place.region),
            "country_key": _key_or_none(place.country),
        }
        blob = vec.astype("<f2").tobytes()
        row = db.execute(
            "SELECT id, size, mtime_ns FROM photos WHERE path = ?", (meta.path,)
        ).fetchone()
        if row is None:
            columns = ", ".join(["path", *values])
            cursor = db.execute(
                f"INSERT INTO photos({columns}) VALUES ({_placeholders(len(values) + 1)})",
                (meta.path, *values.values()),
            )
            photo_id = int(cursor.lastrowid or 0)
            db.execute("INSERT INTO embeddings(photo_id, vec) VALUES (?, ?)", (photo_id, blob))
            return photo_id
        photo_id, old_size, old_mtime = row
        assignments = ", ".join(f"{c} = ?" for c in values)
        db.execute(f"UPDATE photos SET {assignments} WHERE id = ?", (*values.values(), photo_id))
        db.execute("UPDATE embeddings SET vec = ? WHERE photo_id = ?", (blob, photo_id))
        if (old_size, old_mtime) != (values["size"], values["mtime_ns"]):
            db.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            db.execute("DELETE FROM face_scan WHERE photo_id = ?", (photo_id,))
        return int(photo_id)

    def delete_paths(self, paths: Iterable[str | os.PathLike[str]]) -> int:
        """Remove the photos at these paths (and their faces / feedback); return how many.

        Paths may be ``str`` or ``PathLike`` and are canonicalised first.
        """
        wanted = list(dict.fromkeys(canonical_path(p) for p in paths))
        with self._write() as db:
            ids: list[int] = []
            for chunk in _chunks(wanted):
                rows = db.execute(
                    f"SELECT id FROM photos WHERE path IN ({_placeholders(len(chunk))})", chunk
                ).fetchall()
                ids.extend(r[0] for r in rows)
            self._delete_ids(db, ids)
        return len(ids)

    def prune(
        self, existing_paths: Collection[str | os.PathLike[str]], *, allow_empty: bool = False
    ) -> int:
        """Remove every photo whose path is not in ``existing_paths``; return how many.

        Paths may be ``str`` or ``PathLike`` and are canonicalised first. An empty collection
        raises ``ValueError`` unless ``allow_empty=True``: a failed library scan must never
        wipe the index.
        """
        if len(existing_paths) == 0 and not allow_empty:
            raise ValueError("prune() got no paths; pass allow_empty=True to empty the index")
        keep = {canonical_path(p) for p in existing_paths}
        with self._write() as db:
            ids = [r[0] for r in db.execute("SELECT id, path FROM photos") if r[1] not in keep]
            self._delete_ids(db, ids)
        return len(ids)

    def _delete_ids(self, db: sqlite3.Connection, ids: Sequence[int]) -> None:
        for chunk in _chunks(ids):
            db.execute(f"DELETE FROM photos WHERE id IN ({_placeholders(len(chunk))})", chunk)
        self._remove_vectors(ids)

    # ------------------------------------------------------------------ reading photos

    def signatures(self) -> dict[str, tuple[int, int]]:
        """``path -> (size, mtime_ns)`` for every indexed photo (to detect changed files)."""
        with self._read() as db:
            return {p: (s, m) for p, s, m in db.execute("SELECT path, size, mtime_ns FROM photos")}

    def count(self) -> int:
        with self._read() as db:
            return int(db.execute("SELECT COUNT(*) FROM photos").fetchone()[0])

    def get_photos(self, ids: Iterable[int]) -> list[IndexedPhoto]:
        """Photos for ``ids`` in the order given (repeats kept, unknown ids skipped)."""
        wanted = [int(i) for i in ids]
        found: dict[int, IndexedPhoto] = {}
        with self._read() as db:
            for chunk in _chunks(list(dict.fromkeys(wanted))):
                cursor = db.execute(
                    f"SELECT * FROM photos WHERE id IN ({_placeholders(len(chunk))})", chunk
                )
                names = [d[0] for d in cursor.description]
                for row in cursor:
                    photo = self._photo_from_row(dict(zip(names, row)))
                    found[photo.id] = photo
        return [found[i] for i in wanted if i in found]

    @staticmethod
    def _photo_from_row(r: dict[str, Any]) -> IndexedPhoto:
        taken = r["taken"]
        meta = PhotoMeta(
            path=str(r["path"]),
            size=int(r["size"]),
            mtime_ns=int(r["mtime_ns"]),
            width=r["width"],
            height=r["height"],
            taken=datetime.fromisoformat(str(taken)) if taken is not None else None,
            date_source=r["date_source"],
            date_precision=r["date_precision"],
            lat=r["lat"],
            lon=r["lon"],
            camera=r["camera"],
        )
        place = None
        if any(r[c] is not None for c in ("city", "region", "country", "country_code")):
            place = PlaceLabel(
                city=r["city"],
                region=r["region"],
                country=r["country"],
                country_code=r["country_code"],
            )
        return IndexedPhoto(id=int(r["id"]), meta=meta, place=place)

    def stats(self) -> StoreStats:
        """Counts for ``facesorter index --stats`` and the UI."""
        with self._read() as db:
            one = lambda sql: int(db.execute(sql).fetchone()[0])  # noqa: E731
            by_source = dict.fromkeys(_DATE_SOURCES, 0)
            by_source.update(
                db.execute("SELECT date_source, COUNT(*) FROM photos GROUP BY date_source")
            )
            marks = _placeholders(len(_TRUSTED_DATE_SOURCES))
            lo, hi = db.execute(
                f"SELECT MIN(day), MAX(day) FROM photos WHERE date_source IN ({marks})",
                _TRUSTED_DATE_SOURCES,
            ).fetchone()
            return StoreStats(
                photos=one("SELECT COUNT(*) FROM photos"),
                with_gps=one("SELECT COUNT(*) FROM photos WHERE lat IS NOT NULL"),
                with_place=one(
                    "SELECT COUNT(*) FROM photos WHERE city IS NOT NULL OR region IS NOT NULL "
                    "OR country IS NOT NULL OR country_code IS NOT NULL"
                ),
                by_date_source=by_source,
                date_min=date.fromordinal(lo) if lo is not None else None,
                date_max=date.fromordinal(hi) if hi is not None else None,
                faces=one("SELECT COUNT(*) FROM faces"),
                named_people=one("SELECT COUNT(*) FROM people"),
                feedback=one("SELECT COUNT(*) FROM feedback"),
            )

    # ------------------------------------------------------------------ filtering + search

    def candidate_ids(self, filters: Filters | None = None) -> np.ndarray:
        """Ids (ascending, ``int64``) of photos that satisfy every constraint in ``filters``.

        * **date_range**: each photo covers an interval by its ``date_precision`` (``day``: that
          day; ``month``: the whole calendar month; ``year``: the whole year) and matches when
          that interval overlaps the inclusive filter range. Only photos whose ``date_source``
          is ``"exif"`` or ``"folder"`` count; mtime-derived and undated photos are excluded
          whenever a date filter is active.
        * **place**: ``city`` / ``region`` / ``country`` equal, ignoring case, accents and
          extra spaces; ``near=(lat, lon, radius_km)`` keeps photos with GPS within the
          radius (inclusive, great-circle). Photos without a place label never match a named
          level; photos without GPS never match ``near``. A ``PlaceFilter`` with no level and
          no ``near`` is a no-op.
        * **people**: every name must belong to some face in the photo whose cluster is
          named (case/accent-insensitive).
        """
        with self._read() as db:
            return self._candidate_ids(db, filters)

    def _filter_sql(
        self, filters: Filters | None
    ) -> tuple[list[str], list[object], tuple[float, float, float] | None]:
        where: list[str] = []
        params: list[object] = []
        near = None
        if filters is not None:
            if filters.date_range is not None:
                marks = _placeholders(len(_TRUSTED_DATE_SOURCES))
                where.append(f"date_source IN ({marks}) AND day_lo <= ? AND day_hi >= ?")
                params += [*_TRUSTED_DATE_SOURCES]
                params += [filters.date_range.end.toordinal(), filters.date_range.start.toordinal()]
            if filters.place is not None:
                near = self._place_clauses(filters.place, where, params)
            for key in dict.fromkeys(fold(n) for n in filters.people):
                where.append(
                    "EXISTS (SELECT 1 FROM faces f JOIN people p ON p.cluster_id = f.cluster_id "
                    "WHERE f.photo_id = photos.id AND p.name_key = ?)"
                )
                params.append(key)
        return where, params, near

    def _candidate_ids(self, db: sqlite3.Connection, filters: Filters | None) -> np.ndarray:
        where, params, near = self._filter_sql(filters)
        columns = "id, lat, lon" if near else "id"
        sql = f"SELECT {columns} FROM photos"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id"
        rows = db.execute(sql, params).fetchall()
        if near is None:
            return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        if not rows:
            return np.zeros(0, dtype=np.int64)
        arr = np.array(rows, dtype=np.float64)
        dist = haversine_km(near[0], near[1], arr[:, 1], arr[:, 2])
        return arr[dist <= near[2] + 1e-9, 0].astype(np.int64)

    def browse(
        self, filters: Filters | None = None, k: int = 24, order: str = "newest"
    ) -> tuple[list[int], int]:
        """Photo ids passing ``filters`` without any visual ranking; returns ``(ids, candidates)``.

        For queries with no visual text. ``order``: ``"newest"`` (capture time descending,
        undated photos last), ``"oldest"`` (ascending, undated last) or ``"id"`` (ascending);
        ties are broken by ascending id. ``k`` must be at least 1. Month / year precision
        photos sort by their stored ``taken`` (the first day of the period).
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if order not in _BROWSE_ORDER_SQL:
            raise ValueError(f"order must be one of {sorted(_BROWSE_ORDER_SQL)}, got {order!r}")
        with self._read() as db:
            where, params, near = self._filter_sql(filters)
            suffix = " WHERE " + " AND ".join(where) if where else ""
            if near is None:
                total = int(
                    db.execute("SELECT COUNT(*) FROM photos" + suffix, params).fetchone()[0]
                )
                rows = db.execute(
                    f"SELECT id FROM photos{suffix} ORDER BY {_BROWSE_ORDER_SQL[order]} LIMIT ?",
                    [*params, k],
                )
                return [r[0] for r in rows], total
            rows = db.execute(
                f"SELECT id, lat, lon, taken FROM photos{suffix} ORDER BY id", params
            ).fetchall()
        if not rows:
            return [], 0
        arr = np.array([r[1:3] for r in rows], dtype=np.float64)
        within = haversine_km(near[0], near[1], arr[:, 0], arr[:, 1]) <= near[2] + 1e-9
        kept = [r for r, ok in zip(rows, within) if ok]  # still in ascending id order
        if order != "id":
            dated = [r for r in kept if r[3] is not None]  # sorted() is stable: ties keep id order
            undated = [r for r in kept if r[3] is None]
            kept = sorted(dated, key=lambda r: r[3], reverse=order == "newest") + undated
        return [r[0] for r in kept[:k]], len(kept)

    @staticmethod
    def _place_clauses(
        place: PlaceFilter, where: list[str], params: list[object]
    ) -> tuple[float, float, float] | None:
        for column, value in (
            ("city_key", place.city),
            ("region_key", place.region),
            ("country_key", place.country),
        ):
            key = _key_or_none(value)
            if key is not None:
                where.append(f"{column} = ?")
                params.append(key)
        if place.near is None:
            return None
        lat, lon, radius = (float(x) for x in place.near)
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(radius)):
            raise ValueError(f"near={place.near!r} must be finite numbers")
        if not -90.0 <= lat <= 90.0 or radius < 0:
            raise ValueError(f"near={place.near!r}: need -90 <= lat <= 90 and radius_km >= 0")
        lon = (lon + 180.0) % 360.0 - 180.0
        clause, bbox = _bbox_clauses(lat, lon, radius)
        where.append("lat IS NOT NULL AND lon IS NOT NULL AND " + clause)
        params += bbox
        return lat, lon, radius

    def _plan(self, candidates: int, total: int) -> str:
        """``"gather"`` (multiply only the candidate rows) or ``"scan"`` (multiply everything)."""
        return "gather" if candidates <= self.gather_threshold * total else "scan"

    def top_k(
        self, query_vec: np.ndarray, filters: Filters | None = None, k: int = 10
    ) -> tuple[list[tuple[int, float]], int]:
        """Best ``k`` photos by cosine similarity among those passing ``filters``.

        Returns ``([(photo_id, score), ...] best first, candidates_searched)``. Scores are
        rounded to 6 decimals; ties are broken by ascending id, so the order is deterministic.
        ``k`` larger than the candidate pool just returns the whole pool; no candidates gives
        ``([], 0)``. The query is validated (shape, finite) and re-normalised.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        q = _normalise(query_vec, self._dim, "query vector")
        with self._read() as db:
            self._ensure_loaded(db)
            n = self._n
            if filters is None or filters.is_empty:
                if n == 0:
                    return [], 0
                return _select_top(self._mat[:n] @ q, self._ids[:n], k), n
            cand = self._candidate_ids(db, filters)
            cand = cand[cand < len(self._row_of)]
            rows = self._row_of[cand]
            rows = rows[rows >= 0]
            if len(rows) == 0:
                return [], 0
            if self._plan(len(rows), n) == "gather":
                scores = np.concatenate(
                    [self._mat[block] @ q for block in _chunks_array(rows, _GATHER_BLOCK)]
                )
            else:
                scores = (self._mat[:n] @ q)[rows]
            return _select_top(scores, self._ids[rows], k), int(len(rows))

    # ------------------------------------------------------------------ faces + people

    def set_faces(self, photo_id: int, rows: Sequence[FaceRow]) -> None:
        """Replace the faces of one photo (``face_idx`` is the position in ``rows``).

        Embeddings, when given, must be floating-point, 1-D, finite and non-zero; they are
        normalised to unit length and stored as float16. Invalid input raises ``ValueError``
        and changes nothing. This does not mark the photo as scanned (see
        ``mark_faces_scanned``).
        """
        blobs = []
        for i, r in enumerate(rows):
            if r.embedding is None:
                blobs.append(None)
            else:
                unit = _normalise(r.embedding, None, f"face {i} embedding")
                blobs.append(unit.astype("<f2").tobytes())
        with self._write() as db:
            if db.execute("SELECT 1 FROM photos WHERE id = ?", (photo_id,)).fetchone() is None:
                raise ValueError(f"unknown photo id {photo_id}")
            db.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            db.executemany(
                "INSERT INTO faces(photo_id, face_idx, cluster_id, x1, y1, x2, y2, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (photo_id, i, r.cluster_id, r.x1, r.y1, r.x2, r.y2, blob)
                    for i, (r, blob) in enumerate(zip(rows, blobs))
                ],
            )

    def get_faces(self, photo_id: int) -> list[FaceRow]:
        """The photo's faces in ``face_idx`` order, with unit-length float32 embeddings."""
        with self._read() as db:
            cursor = db.execute(
                "SELECT cluster_id, x1, y1, x2, y2, embedding FROM faces WHERE photo_id = ? "
                "ORDER BY face_idx",
                (photo_id,),
            )
            return [
                FaceRow(
                    c,
                    x1,
                    y1,
                    x2,
                    y2,
                    None if blob is None else np.frombuffer(blob, dtype="<f2").astype(np.float32),
                )
                for c, x1, y1, x2, y2, blob in cursor
            ]

    def mark_faces_scanned(self, photo_id: int) -> None:
        """Record that face detection ran on this photo (as it is now: size and mtime)."""
        with self._write() as db:
            row = db.execute(
                "SELECT size, mtime_ns FROM photos WHERE id = ?", (photo_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown photo id {photo_id}")
            db.execute(
                "INSERT OR REPLACE INTO face_scan(photo_id, scanned_ts, size, mtime_ns) "
                "VALUES (?, ?, ?, ?)",
                (photo_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), *row),
            )

    def photos_needing_face_scan(self, limit: int | None = None) -> list[IndexedPhoto]:
        """Photos never scanned, or whose size / mtime differ from when they were scanned.

        Ascending id; at most ``limit`` of them (``None`` = all, negative raises ``ValueError``).
        """
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be None or >= 0, got {limit}")
        with self._read() as db:
            cursor = db.execute(
                "SELECT photos.* FROM photos LEFT JOIN face_scan s ON s.photo_id = photos.id "
                "WHERE s.photo_id IS NULL OR s.size != photos.size "
                "OR s.mtime_ns != photos.mtime_ns "
                "ORDER BY photos.id LIMIT ?",
                (-1 if limit is None else limit,),
            )
            names = [d[0] for d in cursor.description]
            return [self._photo_from_row(dict(zip(names, row))) for row in cursor]

    def face_embeddings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(photo_ids int64 (n,), face_idx int32 (n,), embeddings float32 (n, d))`` for every
        face that has an embedding, ordered by (photo id, face_idx).

        With no embeddings the arrays are empty and ``d`` is 0. Mixed embedding lengths (faces
        stored by different face models) raise ``SearchError``: rescan the library.
        """
        with self._read() as db:
            rows = db.execute(
                "SELECT photo_id, face_idx, embedding FROM faces WHERE embedding IS NOT NULL "
                "ORDER BY photo_id, face_idx"
            ).fetchall()
        if not rows:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
                np.zeros((0, 0), dtype=np.float32),
            )
        size = len(rows[0][2])
        if any(len(r[2]) != size for r in rows):
            raise SearchError("Stored face embeddings have different sizes; rescan the faces.")
        emb = np.frombuffer(b"".join(r[2] for r in rows), dtype="<f2").reshape(len(rows), -1)
        return (
            np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows)),
            np.fromiter((r[1] for r in rows), dtype=np.int32, count=len(rows)),
            emb.astype(np.float32),
        )

    def recluster(
        self,
        assignments: Mapping[tuple[int, int], int | None],
        *,
        names: Mapping[int, str] | None = None,
    ) -> None:
        """Set ``cluster_id`` for the given ``(photo_id, face_idx)`` faces in one transaction.

        Faces not mentioned keep their current cluster, so pass every face you clustered. An
        unknown face raises ``ValueError``.

        ``names`` (cluster id -> name) *replaces* the people table: exactly those names remain
        (blank names are skipped; a name used twice, ignoring case and accents, raises
        ``DuplicateNameError``; a cluster with no faces afterwards raises ``ValueError``).
        Without ``names``, existing names stay with their cluster id, except that names of
        clusters that no longer have faces are removed. Any error rolls everything back.
        """
        with self._write() as db:
            for (photo_id, face_idx), cluster_id in assignments.items():
                cursor = db.execute(
                    "UPDATE faces SET cluster_id = ? WHERE photo_id = ? AND face_idx = ?",
                    (cluster_id, photo_id, face_idx),
                )
                if cursor.rowcount == 0:
                    raise ValueError(f"unknown face ({photo_id}, {face_idx})")
            if names is None:
                db.execute(
                    "DELETE FROM people WHERE cluster_id NOT IN "
                    "(SELECT DISTINCT cluster_id FROM faces WHERE cluster_id IS NOT NULL)"
                )
                return
            wanted: dict[int, tuple[str, str]] = {}
            seen: dict[str, int] = {}
            for cluster_id, name in names.items():
                clean = " ".join(name.split())
                if not clean:
                    continue
                key = fold(clean)
                if key in seen and seen[key] != cluster_id:
                    raise DuplicateNameError(f"The name {clean!r} is given to two clusters.")
                seen[key] = cluster_id
                has_faces = db.execute(
                    "SELECT 1 FROM faces WHERE cluster_id = ? LIMIT 1", (cluster_id,)
                ).fetchone()
                if has_faces is None:
                    raise ValueError(f"unknown face cluster {cluster_id}")
                wanted[cluster_id] = (clean, key)
            db.execute("DELETE FROM people")
            db.executemany(
                "INSERT INTO people(cluster_id, name, name_key) VALUES (?, ?, ?)",
                [(cid, name, key) for cid, (name, key) in wanted.items()],
            )

    def clear_faces(self, *, keep_names: bool = False, keep_scans: bool = False) -> None:
        """Delete every face (before re-detecting). Names are dropped too, because cluster ids
        are not stable across clusterings, unless ``keep_names``; scan marks are dropped
        (so every photo needs a new scan) unless ``keep_scans``."""
        with self._write() as db:
            db.execute("DELETE FROM faces")
            if not keep_names:
                db.execute("DELETE FROM people")
            if not keep_scans:
                db.execute("DELETE FROM face_scan")

    # ------------------------------------------------------------------ generic metadata

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """A free-form string setting stored in the index (also readable: model_name, dim)."""
        with self._read() as db:
            row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        """Store a string setting (for example ``library_roots`` as JSON, ``indexed_at``).

        Internal keys (``model_name``, ``dim``, ``created``, ``schema``, ``schema_version``)
        are reserved and raise ``ValueError``, as do an empty key or a non-string value.
        """
        if not isinstance(key, str) or not key or key in _RESERVED_META_KEYS:
            raise ValueError(f"invalid or reserved meta key {key!r}")
        if not isinstance(value, str):
            raise ValueError(f"meta value must be a string, got {type(value).__name__}")
        with self._write() as db:
            db.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def clusters(self) -> list[ClusterInfo]:
        """Face clusters, biggest (by face count) first; unclustered faces are not listed."""
        with self._read() as db:
            rows = db.execute(
                "SELECT f.cluster_id, COUNT(*), COUNT(DISTINCT f.photo_id), p.name "
                "FROM faces f LEFT JOIN people p ON p.cluster_id = f.cluster_id "
                "WHERE f.cluster_id IS NOT NULL GROUP BY f.cluster_id "
                "ORDER BY COUNT(*) DESC, f.cluster_id"
            ).fetchall()
            out = []
            for cluster_id, faces, photos, name in rows:
                samples = db.execute(
                    "SELECT photo_id FROM faces WHERE cluster_id = ? GROUP BY photo_id "
                    "ORDER BY MAX((x2 - x1) * (y2 - y1)) DESC, photo_id LIMIT ?",
                    (cluster_id, _SAMPLE_PHOTOS),
                ).fetchall()
                out.append(
                    ClusterInfo(cluster_id, faces, photos, name, tuple(r[0] for r in samples))
                )
            return out

    def name_cluster(self, cluster_id: int, name: str | None) -> None:
        """Attach ``name`` to a cluster; ``""`` or ``None`` removes its name.

        Names are unique ignoring case and accents: using one that belongs to another cluster
        raises ``DuplicateNameError`` (re-naming the same cluster, e.g. to fix capitalisation,
        is fine). Unknown clusters raise ``ValueError``.
        """
        clean = " ".join((name or "").split())
        with self._write() as db:
            exists = db.execute(
                "SELECT 1 FROM faces WHERE cluster_id = ? LIMIT 1", (cluster_id,)
            ).fetchone()
            if exists is None:
                raise ValueError(f"unknown face cluster {cluster_id}")
            if not clean:
                db.execute("DELETE FROM people WHERE cluster_id = ?", (cluster_id,))
                return
            key = fold(clean)
            owner = db.execute(
                "SELECT p.cluster_id, EXISTS(SELECT 1 FROM faces f WHERE f.cluster_id = "
                "p.cluster_id) FROM people p WHERE p.name_key = ?",
                (key,),
            ).fetchone()
            if owner is not None and owner[0] != cluster_id:
                if owner[1]:
                    raise DuplicateNameError(
                        f"The name {clean!r} is already used by face cluster {owner[0]}."
                    )
                db.execute("DELETE FROM people WHERE cluster_id = ?", (owner[0],))  # orphan
            db.execute(
                "INSERT INTO people(cluster_id, name, name_key) VALUES (?, ?, ?) "
                "ON CONFLICT(cluster_id) DO UPDATE SET name = excluded.name, "
                "name_key = excluded.name_key",
                (cluster_id, clean, key),
            )

    def known_names(self) -> list[str]:
        """Sorted names of clusters that still have faces (satisfies ``PeopleDirectory``)."""
        with self._read() as db:
            rows = db.execute(
                "SELECT name FROM people WHERE EXISTS "
                "(SELECT 1 FROM faces f WHERE f.cluster_id = people.cluster_id) "
                "ORDER BY name_key"
            )
            return [r[0] for r in rows]

    def names_for_photos(self, photo_ids: Iterable[int]) -> dict[int, list[str]]:
        """Named people present in each photo (sorted); photos with none are omitted."""
        wanted = list(dict.fromkeys(int(i) for i in photo_ids))
        out: dict[int, set[str]] = {}
        with self._read() as db:
            for chunk in _chunks(wanted):
                rows = db.execute(
                    "SELECT f.photo_id, p.name FROM faces f JOIN people p "
                    f"ON p.cluster_id = f.cluster_id WHERE f.photo_id IN "
                    f"({_placeholders(len(chunk))})",
                    chunk,
                )
                for photo_id, name in rows:
                    out.setdefault(photo_id, set()).add(name)
        return {i: sorted(names, key=fold) for i, names in out.items()}

    # ------------------------------------------------------------------ feedback

    def record_feedback(self, query: str, photo_id: int, relevant: bool) -> int:
        """Store a thumbs-up/down for ``photo_id`` as a hit of ``query``; return the row id."""
        with self._write() as db:
            if db.execute("SELECT 1 FROM photos WHERE id = ?", (photo_id,)).fetchone() is None:
                raise ValueError(f"unknown photo id {photo_id}")
            cursor = db.execute(
                "INSERT INTO feedback(ts, query, photo_id, relevant) VALUES (?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    query.strip(),
                    photo_id,
                    int(bool(relevant)),
                ),
            )
            return int(cursor.lastrowid or 0)

    def feedback_rows(self) -> list[FeedbackRow]:
        """Every rating, oldest first."""
        with self._read() as db:
            cursor = db.execute(
                "SELECT id, ts, query, photo_id, relevant FROM feedback ORDER BY id"
            )
            return [FeedbackRow(i, ts, q, p, bool(r)) for i, ts, q, p, r in cursor]

    def feedback_summary(self) -> FeedbackSummary:
        """Per-query precision of rated hits (the latest rating of a photo for a query wins)."""
        with self._read() as db:
            rows = db.execute(
                "SELECT query, COUNT(*), SUM(relevant) FROM feedback f WHERE id = "
                "(SELECT MAX(id) FROM feedback WHERE query = f.query AND photo_id = f.photo_id) "
                "GROUP BY query ORDER BY query"
            ).fetchall()
        per_query = tuple(QueryFeedback(q, n, int(rel), rel / n) for q, n, rel in rows)
        rated = sum(q.rated for q in per_query)
        relevant = sum(q.relevant for q in per_query)
        return FeedbackSummary(
            per_query=per_query,
            rated=rated,
            relevant=relevant,
            precision=relevant / rated if rated else None,
            mean_precision=(
                sum(q.precision for q in per_query) / len(per_query) if per_query else None
            ),
        )


def _chunks_array(arr: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(arr), size):
        yield arr[start : start + size]
