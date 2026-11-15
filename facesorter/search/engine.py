"""
facesorter/search/engine.py
The search engine behind the CLI and the UI: parse -> filters + visual text -> ranked hits.

``Engine.search`` follows the clarification protocol documented on ``SearchRequest``:

* Phrases the parser could not resolve are either returned for the caller to ask about
  (``needs_clarification``, no hits) or, when they were already asked about or clarification
  is not allowed, reported in ``dropped`` while the search runs without that filter.
* Visual text is embedded by the text encoder and ranked exactly against the candidates that
  pass the structured filters. With no visual text the filtered photos are listed newest first.

``explain`` and ``describe_filters`` turn a response into the plain-language notices the CLI
and the UI show. ``open_engine`` assembles everything from an index file.
"""

from __future__ import annotations

import logging
import operator
import os
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

from .encoders import load_text_encoder
from .errors import IndexMismatchError, SearchError
from .parser import QueryParser, is_asked
from .store import Store, fold
from .types import (
    Filters,
    Gazetteer,
    Hit,
    PlaceFilter,
    PlaceLabel,
    SearchRequest,
    SearchResponse,
    TextEncoder,
    Unresolved,
)

logger = logging.getLogger(__name__)

__all__ = [
    "INDEX_ENV",
    "Engine",
    "LibraryGazetteer",
    "default_index_path",
    "describe_filters",
    "explain",
    "open_engine",
]

#: Environment variable that overrides the default index location.
INDEX_ENV = "FACESORTER_INDEX"
_MAX_TEXT_CHARS = 2000  # the text tower truncates to 64 tokens anyway
_MAX_K = 2**31 - 1  # far beyond any library; keeps k inside SQLite's integer range
_TOP_PLACES = 5
# Notices added by the engine when a search finds nothing; explain() prints them last.
_HINT_PREFIXES = ("Your library covers photos from", "Places in your library include:")


class LibraryGazetteer:
    """A gazetteer that prefers the namesake which actually has photos in the index.

    A bare place name resolves to the biggest place of that name ("Amherst" is Amherst, New
    York; "Monterey" is Monterrey, Mexico), which is rarely the one in a personal library.
    ``resolve`` weighs the gazetteer's own answer against the other readings (``candidates``)
    whose own name equals the query and picks the one with the most photos in the index (ties
    keep the gazetteer's order, so its answer wins). It keeps the gazetteer's answer when no
    reading has a photo, or when the query carries a qualifier ("Amherst, New York", "Paris
    Texas": no other reading is then named exactly like the query).
    ``label_for``, ``label_many`` and ``candidates`` pass through. Answers are cached per
    name and dropped when the index changes (its photo count or ``indexed_at`` setting).
    """

    def __init__(self, gazetteer: Gazetteer, store: Store) -> None:
        self.inner = gazetteer
        self.store = store
        self._lock = threading.Lock()
        self._cache: dict[str, PlaceFilter | None] = {}
        self._stamp: tuple[int, str | None] | None = None

    def label_for(self, lat: float, lon: float) -> PlaceLabel | None:
        return self.inner.label_for(lat, lon)

    def label_many(self, lats: Sequence[float], lons: Sequence[float]) -> list[PlaceLabel | None]:
        many = getattr(self.inner, "label_many", None)
        if callable(many):
            return list(many(lats, lons))
        return [self.inner.label_for(a, b) for a, b in zip(lats, lons)]

    def candidates(self, name: str) -> list[PlaceFilter]:
        many = getattr(self.inner, "candidates", None)
        if callable(many):
            return list(many(name))
        found = self.inner.resolve(name)
        return [found] if found is not None else []

    def resolve(self, name: str) -> PlaceFilter | None:
        key = fold(" ".join(str(name).split()))
        try:
            stamp = (self.store.count(), self.store.get_meta("indexed_at"))
        except SearchError:  # closed index: behave like the plain gazetteer
            return self.inner.resolve(name)
        with self._lock:
            if stamp != self._stamp:
                self._cache.clear()
                self._stamp = stamp
            if key in self._cache:
                return self._cache[key]
        chosen = self._choose(name, key)
        with self._lock:
            if stamp == self._stamp:
                self._cache[key] = chosen
        return chosen

    def _choose(self, name: str, key: str) -> PlaceFilter | None:
        default = self.inner.resolve(name)
        if default is None:
            return None
        # The gazetteer's own answer first, then every other reading with the same name, so
        # a tie keeps the gazetteer's choice.
        pool = [default]
        pool += [c for c in self.candidates(name) if c != default and fold(_own_name(c)) == key]
        if len(pool) < 2:
            return default
        counts = [len(self.store.candidate_ids(Filters(place=c))) for c in pool]
        best = max(counts)
        return pool[counts.index(best)] if best > 0 else default


def _own_name(place: PlaceFilter) -> str:
    """The name a reading is known by: its most specific level (``text`` for a bare point)."""
    return place.city or place.region or place.country or place.text


class Engine:
    """Answers :class:`SearchRequest` objects from an index (satisfies ``SearchEngine``).

    The engine takes ownership of ``store``: :meth:`close` closes it. It can be shared by
    several threads: parsing and text encoding are serialised (local LLM runtimes and
    tokenisers are not guaranteed to be re-entrant) and the store is lock-protected.
    ``search`` never raises for odd user input, only for a closed index or a failed model.

    A ``gazetteer`` that is not already a :class:`LibraryGazetteer` is wrapped in one, so
    ``engine.gazetteer`` is library-aware; if the ``parser`` was built with that same
    gazetteer object it is switched to the wrapper too (``open_engine`` does this for you).
    """

    def __init__(
        self,
        store: Store,
        text_encoder: TextEncoder,
        parser: QueryParser,
        gazetteer: Gazetteer | None = None,
    ) -> None:
        if text_encoder.name != store.model_name or text_encoder.dim != store.dim:
            raise IndexMismatchError(
                f"The index was built with model {store.model_name!r} (dim {store.dim}) but the "
                f"text encoder is {text_encoder.name!r} (dim {text_encoder.dim})."
            )
        self.store = store
        self.text_encoder = text_encoder
        self.parser = parser
        if gazetteer is not None and not isinstance(gazetteer, LibraryGazetteer):
            wrapped = LibraryGazetteer(gazetteer, store)
            if getattr(parser, "_gazetteer", None) is gazetteer:
                parser._gazetteer = wrapped
            gazetteer = wrapped
        self.gazetteer = gazetteer
        self._places_cache: tuple[tuple[int, str | None], list[str]] | None = None
        self._parse_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Close the store. Safe to call twice."""
        self.store.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def has_llm(self) -> bool:
        """Whether queries are understood by a local LLM (otherwise by rules only)."""
        return getattr(self.parser, "_llm", None) is not None

    # ------------------------------------------------------------------ search

    def search(self, request: SearchRequest) -> SearchResponse:
        """Run one search. ``request.k`` must be at least 1 (``ValueError`` otherwise)."""
        try:
            k = operator.index(request.k)
        except TypeError:
            raise ValueError(f"k must be a whole number, got {request.k!r}") from None
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        k = min(k, _MAX_K)

        t0 = time.perf_counter()
        with self._parse_lock:
            result = self.parser.parse(request)
        applied = result.filters

        unasked = [u for u in result.unresolved if not is_asked(u.phrase, request.clarifications)]
        if unasked and request.allow_clarification:
            logger.debug("Needs clarification for %d phrase(s)", len(unasked))
            return SearchResponse(
                hits=(),
                parse=result,
                applied=applied,
                candidates=0,
                needs_clarification=tuple(unasked),
                timings_ms={"parse": _ms(t0), "encode": 0.0, "search": 0.0},
            )
        dropped = tuple(result.unresolved)  # asked, or may not be asked: search without them
        t1 = time.perf_counter()

        text = result.text.strip()
        vector = None
        if text:
            with self._encode_lock:
                vector = self.text_encoder.encode_text([text[:_MAX_TEXT_CHARS]])[0]
        t2 = time.perf_counter()

        if vector is None:
            ids, candidates = self.store.browse(applied, k, "newest")
            scored = [(i, 0.0) for i in ids]
        else:
            scored, candidates = self.store.top_k(vector, applied, k)
        score_of = dict(scored)
        hits = tuple(
            Hit(photo=photo, score=float(score_of[photo.id]))
            for photo in self.store.get_photos(i for i, _ in scored)
        )
        if candidates == 0:
            hints = self._empty_hints(applied)
            if hints:
                result = replace(result, notes=(*result.notes, *hints))
        t3 = time.perf_counter()
        return SearchResponse(
            hits=hits,
            parse=result,
            applied=applied,
            candidates=int(candidates),
            dropped=dropped,
            timings_ms={
                "parse": (t1 - t0) * 1000.0,
                "encode": (t2 - t1) * 1000.0 if vector is not None else 0.0,
                "search": (t3 - t2) * 1000.0,
            },
        )

    def _empty_hints(self, applied: Filters) -> list[str]:
        """Why-nothing hints for a search without candidates: only for a filter that, taken
        alone, already matches no photo."""
        store = self.store
        hints: list[str] = []
        if store.count() == 0:
            return hints
        if applied.date_range is not None:
            if store.candidate_ids(Filters(date_range=applied.date_range)).size == 0:
                stats = store.stats()
                if stats.date_min is not None and stats.date_max is not None:
                    hints.append(
                        f"Your library covers photos from {stats.date_min.isoformat()} "
                        f"to {stats.date_max.isoformat()}."
                    )
        if applied.place is not None:
            if store.candidate_ids(Filters(place=applied.place)).size == 0:
                names = self._top_places()
                if names:
                    hints.append("Places in your library include: " + ", ".join(names) + ".")
        return hints

    def _top_places(self) -> list[str]:
        """The most common cities in the index (countries if no photo has a city)."""
        stamp = (self.store.count(), self.store.get_meta("indexed_at"))
        cached = self._places_cache
        if cached is not None and cached[0] == stamp:
            return cached[1]
        ids = [int(i) for i in self.store.candidate_ids(None)]
        cities: Counter[str] = Counter()
        countries: Counter[str] = Counter()
        for start in range(0, len(ids), 2000):
            for photo in self.store.get_photos(ids[start : start + 2000]):
                if photo.place is not None:
                    cities[photo.place.city or ""] += 1
                    countries[photo.place.country or ""] += 1
        cities.pop("", None)
        countries.pop("", None)
        ranked = cities or countries
        names = [name for name, _ in sorted(ranked.items(), key=lambda t: (-t[1], t[0]))]
        self._places_cache = (stamp, names[:_TOP_PLACES])
        return names[:_TOP_PLACES]


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0


# ---------------------------------------------------------------------------- wording


def describe_filters(filters: Filters) -> str:
    """One line describing a filter set, e.g. ``dates 2023-06-01 to 2023-08-31 · place Key
    West, Florida, United States · with Bob``; ``""`` when there is no constraint."""
    parts: list[str] = []
    if filters.date_range is not None:
        start, end = filters.date_range.start, filters.date_range.end
        if start == date.min and end == date.max:
            pass  # no bound at all
        elif end == date.max:  # open-ended ("--from 2023-06")
            parts.append(f"from {start.isoformat()}")
        elif start == date.min:
            parts.append(f"up to {end.isoformat()}")
        elif start == end:
            parts.append(f"date {start.isoformat()}")
        else:
            parts.append(f"dates {start.isoformat()} to {end.isoformat()}")
    if filters.place is not None:
        place = filters.place
        named = ", ".join(p for p in (place.city, place.region, place.country) if p)
        if named:
            parts.append(f"place {named}")
        elif place.near is not None:
            lat, lon, radius = place.near
            parts.append(f"place within {radius:g} km of {lat:.4f}, {lon:.4f}")
        elif place.text:
            parts.append(f"place {place.text}")
    if filters.people:
        seen: set[str] = set()
        names = [n for n in filters.people if not (n.casefold() in seen or seen.add(n.casefold()))]
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        parts.append(f"with {who}")
    return " · ".join(parts)


_KIND_NOUN = {"date": "a date", "place": "a place", "person": "a person's name"}


def _dropped_notice(item: Unresolved) -> str:
    noun = _KIND_NOUN.get(item.kind, "a filter")
    return f'I couldn\'t understand "{item.phrase}" as {noun}, so I searched without it.'


def explain(response: SearchResponse) -> list[str]:
    """Human-readable notices for a response, in the order a user should read them.

    Parse notes first, then the questions to ask (``needs_clarification``), a line for each
    filter that was dropped, and finally what the search itself found: ``No photos match:
    <filters>`` for an empty pool, or ``No visual description, showing newest first`` when
    the query had only filters. After ``No photos match`` come the engine's hints (what dates
    and places the library does have), which ``Engine.search`` stores in ``parse.notes``.
    """
    notices = [n for n in response.parse.notes if not n.startswith(_HINT_PREFIXES)]
    hints = [n for n in response.parse.notes if n.startswith(_HINT_PREFIXES)]
    notices.extend(u.question for u in response.needs_clarification)
    notices.extend(_dropped_notice(u) for u in response.dropped)
    if response.needs_clarification:
        return notices
    if response.candidates == 0:
        described = describe_filters(response.applied)
        notices.append(f"No photos match: {described}" if described else "The index is empty")
    elif not response.parse.text.strip():
        notices.append("No visual description, showing newest first")
    return notices + hints


# ---------------------------------------------------------------------------- assembly


def default_index_path() -> Path:
    """``$FACESORTER_INDEX`` if set, else ``~/.facesorter/index.sqlite``. Creates nothing."""
    override = os.environ.get(INDEX_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(os.path.expanduser("~")) / ".facesorter" / "index.sqlite"


def open_engine(
    index_path: str | os.PathLike[str],
    *,
    llm: str = "auto",
    llm_model: str | None = None,
    use_llm: bool = True,
    text_quantization: str = "fp16",
) -> Engine:
    """Open an existing index and build an :class:`Engine` for it.

    The index tells which embedding model to load for the text side. ``use_llm=False`` (or
    ``llm="none"``, or no installed runtime with ``"auto"``) selects rules-only parsing.
    Raises ``IndexNotFoundError`` (no index there), ``IndexMismatchError`` (not a compatible
    index) or ``ModelUnavailableError`` (missing model / runtime); the index is closed again
    on any failure. Call ``close()`` when done.
    """
    from .geo import build_gazetteer
    from .llm import load_llm

    store = Store.open(Path(index_path), create=False)
    try:
        text_encoder = load_text_encoder(store.model_name, text_quantization)
        gazetteer = build_gazetteer()
        backend = load_llm(llm, llm_model) if use_llm else None
        gazetteer = LibraryGazetteer(gazetteer, store)
        parser = QueryParser(backend, gazetteer, store)
        return Engine(store, text_encoder, parser, gazetteer)
    except BaseException:
        store.close()
        raise
