"""
facesorter/search/types.py
Shared contracts for the photo-search feature.

Everything here is plain data (frozen dataclasses) or ``typing.Protocol`` interfaces, so
modules can be built and tested independently against the same vocabulary. Nothing in this
module imports heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from PIL import Image


# --------------------------------------------------------------------------- places


@dataclass(frozen=True)
class PlaceLabel:
    """Where a photo was taken, as a city / region / country hierarchy.

    ``region`` is the state or province. Any level may be ``None``.
    """

    city: str | None = None
    region: str | None = None
    country: str | None = None
    country_code: str | None = None

    @property
    def display(self) -> str:
        """``"City, Region, Country"`` with missing levels skipped."""
        return ", ".join(p for p in (self.city, self.region, self.country) if p)


@dataclass(frozen=True)
class PlaceFilter:
    """A resolved place constraint. Level fields are matched case-insensitively and exactly.

    Use ``city``/``region``/``country`` for named places; ``near`` is a fallback for
    landmarks or names the gazetteer only knows as a point: ``(lat, lon, radius_km)``.
    ``text`` is the user's wording, kept for display ("in Key West").
    """

    city: str | None = None
    region: str | None = None
    country: str | None = None
    near: tuple[float, float, float] | None = None
    text: str = ""


# --------------------------------------------------------------------------- photos


@dataclass(frozen=True)
class PhotoMeta:
    """Metadata read from one still image (no pixels, no embedding).

    ``path`` is a ``str`` in canonical form: ``os.path.abspath(os.path.expanduser(...))``
    (symlinks are NOT resolved). Every API that takes or returns photo paths uses this
    form so that ``Path`` and ``str`` values never silently disagree.

    ``date_precision`` says how exact ``taken`` is: ``"day"`` (EXIF or a full date), or
    ``"month"`` / ``"year"`` when the date came from a folder name and ``taken`` is just
    the first day of that month / year. A date filter must treat such a photo as
    spanning the whole month / year, not as one instant.
    """

    path: str  # canonical absolute path (see above)
    size: int  # bytes
    mtime_ns: int
    width: int | None = None
    height: int | None = None
    taken: datetime | None = None  # naive local capture time
    date_source: Literal["exif", "folder", "mtime", "none"] = "none"
    lat: float | None = None
    lon: float | None = None
    camera: str | None = None
    date_precision: Literal["day", "month", "year"] = "day"


@dataclass(frozen=True)
class IndexedPhoto:
    """A photo as stored in the index."""

    id: int
    meta: PhotoMeta
    place: PlaceLabel | None = None


# --------------------------------------------------------------------------- filters


@dataclass(frozen=True)
class DateRange:
    """Inclusive calendar-date range."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"DateRange end {self.end} is before start {self.start}")


@dataclass(frozen=True)
class Filters:
    """Structured constraints. All present constraints must hold (logical AND).

    ``people`` are names of people the user has named (every name must be in the photo).
    """

    date_range: DateRange | None = None
    place: PlaceFilter | None = None
    people: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.date_range is None and self.place is None and not self.people


# --------------------------------------------------------------------------- queries


@dataclass(frozen=True)
class ParsedQuery:
    """What was extracted from the user's words, before anything is resolved.

    Every non-``None`` span (``date_phrase``, ``place``, each person) must appear verbatim
    in ``raw`` (case-insensitive) - the grounding rule that stops a model inventing filters.
    ``text`` is the visual remainder used for the embedding search: ``raw`` minus the
    grounded spans and filler words.
    """

    raw: str
    text: str
    date_phrase: str | None = None
    place: str | None = None
    people: tuple[str, ...] = ()
    source: Literal["llm", "rules", "flags"] = "rules"


@dataclass(frozen=True)
class Unresolved:
    """A phrase the system could not turn into a filter."""

    kind: Literal["date", "place", "person"]
    phrase: str
    reason: str
    question: str  # human-readable, e.g. 'I did not understand "Easter 2024" as a date.'


@dataclass(frozen=True)
class ParseResult:
    """A parsed query with every phrase either resolved into ``filters`` or listed as
    ``unresolved``."""

    parsed: ParsedQuery
    filters: Filters
    unresolved: tuple[Unresolved, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return self.parsed.text


@dataclass(frozen=True)
class SearchRequest:
    """One search. ``filters`` (explicit flags) override what the parser infers per field.

    Clarification protocol ("ask once, then tell and search without"):

    * ``allow_clarification=True`` and something is unresolved -> the response has
      ``needs_clarification`` set and no hits.
    * The caller asks the user and calls again with ``clarifications`` mapping each
      original phrase to the user's reworded phrase (``""`` = skip it).
    * A phrase still unresolved after that is reported in ``dropped`` and the search runs
      without that filter. With ``allow_clarification=False`` this happens immediately.
    """

    query: str = ""
    filters: Filters | None = None
    k: int = 24
    use_llm: bool = True
    allow_clarification: bool = True
    clarifications: Mapping[str, str] = field(default_factory=dict)
    today: date | None = None  # for tests; defaults to date.today()


@dataclass(frozen=True)
class Hit:
    photo: IndexedPhoto
    score: float  # cosine similarity of the query text to the image (higher = better)


@dataclass(frozen=True)
class SearchResponse:
    hits: tuple[Hit, ...]
    parse: ParseResult
    applied: Filters  # the filters actually used for this search
    candidates: int  # photos that passed the filters (the pool searched)
    needs_clarification: tuple[Unresolved, ...] = ()
    dropped: tuple[Unresolved, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- protocols


@runtime_checkable
class ImageEncoder(Protocol):
    """Turns images into L2-normalised embeddings."""

    name: str  # model identifier recorded in the index, e.g. "siglip2-b16-webli"
    dim: int

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return ``float32`` array of shape ``(len(images), dim)``, rows unit length."""
        ...


@runtime_checkable
class TextEncoder(Protocol):
    """Turns text into L2-normalised embeddings in the same space as the image encoder."""

    name: str  # must equal the paired ImageEncoder.name
    dim: int

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``float32`` array of shape ``(len(texts), dim)``, rows unit length."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """A small local chat model."""

    name: str

    def chat(self, messages: Sequence[Mapping[str, str]], max_tokens: int = 160) -> str:
        """Greedy-decode a reply for OpenAI-style ``{"role", "content"}`` messages."""
        ...


@runtime_checkable
class Gazetteer(Protocol):
    """Offline place knowledge."""

    def label_for(self, lat: float, lon: float) -> PlaceLabel | None:
        """Label a coordinate, or ``None`` if nothing plausible is near."""
        ...

    def resolve(self, name: str) -> PlaceFilter | None:
        """Turn a place name from a query into a filter, or ``None`` if not understood."""
        ...


@runtime_checkable
class PeopleDirectory(Protocol):
    """Names the user has attached to face clusters."""

    def known_names(self) -> Collection[str]: ...


@runtime_checkable
class SearchEngine(Protocol):
    """What the CLI and the UI talk to."""

    def search(self, request: SearchRequest) -> SearchResponse: ...
