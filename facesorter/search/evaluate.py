"""
facesorter/search/evaluate.py
Caption-free evaluation of photo search: no labels, no captions, no annotation sessions.

Quality is measured from two sources the library already has:

* **Metadata the photos carry** (EXIF date, GPS-derived place). Every probe query is built
  from a photo's *own* metadata or checked against it, so the right answer is known:

  - *self-retrieval probes*: "photos from March 2024", "photos in Paris", ... built from a
    sampled photo; the parsed filters must equal the expected ones, the filters must reach
    the engine, every hit must satisfy them, and the photo must be among the candidates;
  - *lift probes*: "snow on the ground" should return photos taken disproportionately in
    winter months, "the eiffel tower" photos disproportionately labelled France. The share of
    top-k hits meeting the metadata expectation is compared with the library's base rate;
  - *safety battery*: tricky phrasings ("since 2020", "03/04/2024", an unknown place, ...)
    run in rules mode must never silently apply a wrong filter.

* **Thumbs up/down ratings** made in the UI (``Store.record_feedback``): precision of the
  rated hits per query and overall. ``suggest_probe_queries`` proposes generic queries worth
  rating.

Privacy: reports contain aggregates only (counts, rates, histograms in which any bucket
smaller than ``MIN_BUCKET`` photos is folded into "other"). File paths, photo content and the
text of feedback queries appear only when ``show_paths`` / ``--show-paths`` is requested.

``add_arguments`` and ``run`` are the glue for ``facesorter eval``. Heavy dependencies are
never imported here; the engine is created lazily by ``run``.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .errors import SearchError
from .store import fold
from .types import (
    DateRange,
    Filters,
    IndexedPhoto,
    PlaceFilter,
    SearchRequest,
    SearchResponse,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import Store

__all__ = [
    "DEFAULT_LIFT_QUERIES",
    "EvalReport",
    "LiftQuery",
    "ProbeQueryError",
    "add_arguments",
    "load_lift_queries",
    "run",
    "run_probes",
    "suggest_probe_queries",
]

MIN_BUCKET = 5  # histogram buckets smaller than this are folded into "other"
MIN_BASE_SUPPORT = 30  # library photos needed to judge a lift probe
MIN_HIT_SUPPORT = 5  # top-k hits carrying the metadata needed to judge a lift probe
MIN_RATED = 10  # ratings below this are reported as "insufficient data"
MIN_EFFECT = 0.10  # a 'pass' needs the share to beat the base rate by this much
ROUND_TO = 5  # library-wide counts are rounded to this when small buckets are hidden
_Z = 1.96  # 95% Wilson interval
_TRUSTED = ("exif", "folder")
_MONTHS = (
    "January February March April May June July August September October November December"
).split()
_LIMIT_BUCKETS = 10

LLM_CHOICES = ("auto", "mlx", "llama-cpp", "none")


class ProbeQueryError(SearchError, ValueError):
    """The ``--queries`` file is unreadable or invalid."""


class _EvalEngine(Protocol):
    store: Store

    def search(self, request: SearchRequest) -> SearchResponse: ...


# --------------------------------------------------------------------------- report types


@dataclass
class _Report:
    """Base for report sections: JSON-serialisable through ``to_dict``."""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)  # type: ignore[no-any-return]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, date):
        return value.isoformat()
    return value


@dataclass
class Histogram(_Report):
    """Counts per bucket; buckets under ``MIN_BUCKET`` photos are folded into "other".

    Only the *number* of folded buckets is reported, never their photo count, and no total is
    stored, so a hidden size cannot be recovered by subtraction.
    """

    buckets: dict[str, int] = field(default_factory=dict)
    other_buckets: int = 0


@dataclass
class LibraryProfile(_Report):
    """Counts are exact unless ``counts_rounded``: when any small bucket is hidden, every
    library-wide count in the report is rounded to the nearest 5 (and rates to 2 decimals)
    so the hidden size cannot be recovered from a total."""

    counts_rounded: bool = False
    photos: int = 0
    dated: int = 0  # trusted (exif / folder) date
    month_known: int = 0  # trusted date of day or month precision
    with_place: int = 0
    with_city: int = 0
    months: Histogram = field(default_factory=Histogram)
    years: Histogram = field(default_factory=Histogram)
    cities: Histogram = field(default_factory=Histogram)
    countries: Histogram = field(default_factory=Histogram)


@dataclass
class ProbeTypeReport(_Report):
    """Self-retrieval results for one probe type."""

    type: str = ""
    probes: int = 0
    successes: int = 0
    success_rate: float | None = None
    parsed_ok_rate: float | None = None
    filter_correct_rate: float | None = None  # probes whose hits all satisfy the applied filters
    ambiguous: int = 0  # place name resolved to a namesake elsewhere (not a parser failure)
    # Reserved: recall@k is meaningful only for queries with visual text; these date / place
    # only probes cannot rank, so it is always None and rendered "n/a (no visual text)".
    recall_at_k: float | None = None
    candidates_mean: float | None = None
    candidates_median: float | None = None
    failures: dict[str, int] = field(default_factory=dict)
    examples: list[dict[str, str]] = field(default_factory=list)  # only with show_paths


@dataclass
class LiftResult(_Report):
    query: str = ""
    expectation: str = ""
    status: str = "ok"  # ok | insufficient_data | filtered | error
    reason: str = ""
    verdict: str = ""  # pass | weak | fail ('' when not ok)
    k: int = 0
    hits: int = 0
    hits_eligible: int = 0
    hits_matching: int = 0
    share: float | None = None
    share_ci_low: float | None = None
    share_ci_high: float | None = None
    base_eligible: int = 0
    base_matching: int = 0
    base_rate: float | None = None
    lift: float | None = None


@dataclass
class PlaceRoundTrip(_Report):
    """Does the bare city name of every stored place label resolve back to that label?"""

    labels: int = 0
    labels_ok: int = 0
    label_rate: float | None = None
    photos: int = 0
    photos_ok: int = 0
    photo_rate: float | None = None
    failing: Histogram = field(default_factory=Histogram)  # by label, only with show_paths


@dataclass
class FeedbackQueryReport(_Report):
    query: str = ""
    rated: int = 0
    relevant: int = 0
    precision: float = 0.0


@dataclass
class FeedbackReport(_Report):
    status: str = "none"  # none | insufficient_data | ok
    queries: int = 0
    rated: int = 0
    relevant: int = 0
    precision: float | None = None
    precision_ci_low: float | None = None
    precision_ci_high: float | None = None
    mean_precision: float | None = None
    per_query: list[FeedbackQueryReport] = field(default_factory=list)


@dataclass
class LatencyReport(_Report):
    samples: int = 0
    p50_ms: float | None = None
    p95_ms: float | None = None
    stages: dict[str, dict[str, float]] = field(default_factory=dict)  # name -> p50 / p95


@dataclass
class SafetyReport(_Report):
    """Counts of the tricky-query battery (every case runs with and without clarification)."""

    cases: int = 0
    runs: int = 0
    safe_runs: int = 0
    unsafe_runs: int = 0
    reported_runs: int = 0  # safe because the engine asked / reported what it did not understand
    quiet_runs: int = 0  # safe because no filter was applied
    errors: int = 0
    controls: int = 0
    controls_ok: int = 0
    unsafe_cases: list[dict[str, str]] = field(default_factory=list)
    failed_controls: list[str] = field(default_factory=list)


@dataclass
class EvalReport(_Report):
    params: dict[str, Any] = field(default_factory=dict)
    library: LibraryProfile = field(default_factory=LibraryProfile)
    self_retrieval: list[ProbeTypeReport] = field(default_factory=list)
    lift: list[LiftResult] = field(default_factory=list)
    place_roundtrip: PlaceRoundTrip = field(default_factory=PlaceRoundTrip)
    feedback: FeedbackReport = field(default_factory=FeedbackReport)
    latency: LatencyReport = field(default_factory=LatencyReport)
    safety: SafetyReport = field(default_factory=SafetyReport)
    warnings: list[str] = field(default_factory=list)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return _markdown(self)


# --------------------------------------------------------------------------- small maths


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0..100); ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _wilson(successes: int, n: int) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + _Z**2 / n
    centre = (p + _Z**2 / (2 * n)) / denom
    half = _Z * math.sqrt(p * (1 - p) / n + _Z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- photo facts


@dataclass(frozen=True)
class _Facts:
    month: int | None  # trusted day / month-precision date
    place_text: str  # folded "City, Region, Country"
    country: str
    country_code: str
    has_place: bool


def _facts(photo: IndexedPhoto) -> _Facts:
    meta = photo.meta
    month = None
    if meta.taken is not None and meta.date_source in _TRUSTED and meta.date_precision != "year":
        month = meta.taken.month
    place = photo.place
    return _Facts(
        month=month,
        place_text=fold(place.display) if place else "",
        country=fold(place.country or "") if place else "",
        country_code=(place.country_code or "").casefold() if place else "",
        has_place=bool(place and place.display),
    )


def _trusted_date(photo: IndexedPhoto) -> bool:
    return photo.meta.taken is not None and photo.meta.date_source in _TRUSTED


def _city(photo: IndexedPhoto) -> str | None:
    return photo.place.city if photo.place and photo.place.city else None


def _load_photos(store: Store) -> list[IndexedPhoto]:
    ids = [int(i) for i in store.candidate_ids(None)]
    return store.get_photos(ids)


# --------------------------------------------------------------------------- library profile


def _histogram(counts: Mapping[str, int], limit: int | None = None) -> Histogram:
    shown = sorted(
        ((k, n) for k, n in counts.items() if n >= MIN_BUCKET), key=lambda kv: (-kv[1], kv[0])
    )
    if limit is not None:
        shown = shown[:limit]
    return Histogram(buckets=dict(shown), other_buckets=len(counts) - len(shown))


def _round_count(n: int) -> int:
    return (n + ROUND_TO // 2) // ROUND_TO * ROUND_TO


def _profile(photos: Sequence[IndexedPhoto]) -> LibraryProfile:
    months: Counter[str] = Counter()
    years: Counter[str] = Counter()
    cities: Counter[str] = Counter()
    countries: Counter[str] = Counter()
    profile = LibraryProfile(photos=len(photos))
    for photo in photos:
        facts = _facts(photo)
        if _trusted_date(photo):
            profile.dated += 1
            years[str(photo.meta.taken.year)] += 1  # type: ignore[union-attr]
        if facts.month is not None:
            profile.month_known += 1
            months[_MONTHS[facts.month - 1][:3]] += 1
        if facts.has_place:
            profile.with_place += 1
        if _city(photo):
            profile.with_city += 1
            cities[_city(photo)] += 1  # type: ignore[index]
        if photo.place and photo.place.country:
            countries[photo.place.country] += 1
    profile.months = _histogram(months)
    profile.years = _histogram(years)
    profile.cities = _histogram(cities, _LIMIT_BUCKETS)
    profile.countries = _histogram(countries, _LIMIT_BUCKETS)
    return profile


# --------------------------------------------------------------------------- probe queries


def suggest_probe_queries(store: Store, n: int = 10) -> list[str]:
    """Generic queries worth rating in the UI, derived from the library's place and month
    distribution. Only places, and (year, month) pairs, with at least ``MIN_BUCKET`` photos
    are used and no path or file name is ever included. Deterministic for a given library.
    """
    if n <= 0:
        return []
    photos = _load_photos(store)
    places = list(_profile(photos).cities.buckets)
    pairs: Counter[tuple[int, int]] = Counter()
    for photo in photos:
        month = _facts(photo).month
        if month is not None:
            pairs[(photo.meta.taken.year, month)] += 1  # type: ignore[union-attr]
    months = [
        f"photos from {_MONTHS[m - 1]} {y}"
        for (y, m), count in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        if count >= MIN_BUCKET
    ]
    visual = ["people smiling", "food on a table", "a sunset", "a dog", "snow", "the beach"]
    suggestions: list[str] = []
    for i in range(max(len(visual), len(places), len(months))):
        if i < len(visual):
            suggestions.append(visual[i])
        if i < len(places):
            suggestions.append(f"{visual[(i + 1) % len(visual)]} in {places[i]}")
        if i < len(months):
            suggestions.append(months[i])
    return list(dict.fromkeys(suggestions))[:n]


# --- lift queries


@dataclass(frozen=True)
class LiftQuery:
    """A visual query plus a metadata expectation about the photos it should return.

    All given dimensions must hold; within ``places`` / ``countries`` / ``months`` any entry
    may match. ``places`` are folded substrings of "City, Region, Country".
    """

    query: str
    months: frozenset[int] = frozenset()
    places: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = []
        if self.months:
            parts.append("months=" + ",".join(_MONTHS[m - 1][:3] for m in sorted(self.months)))
        if self.places:
            parts.append("place~" + "|".join(self.places))
        if self.countries:
            parts.append("country=" + "|".join(self.countries))
        return " and ".join(parts)

    def applicable(self, facts: _Facts) -> bool:
        """Does the photo carry the metadata this expectation needs?"""
        if self.months and facts.month is None:
            return False
        return not ((self.places or self.countries) and not facts.has_place)

    def holds(self, facts: _Facts) -> bool:
        if self.months and facts.month not in self.months:
            return False
        if self.places and not any(fold(p) in facts.place_text for p in self.places):
            return False
        if self.countries:
            keys = {fold(c) for c in self.countries}
            if facts.country not in keys and facts.country_code not in keys:
                return False
        return True


# Seasonal expectations assume the northern hemisphere; add your own via --queries.
DEFAULT_LIFT_QUERIES: tuple[LiftQuery, ...] = (
    LiftQuery("snow on the ground", months=frozenset({12, 1, 2, 3})),
    LiftQuery("people skiing on a snowy mountain", months=frozenset({12, 1, 2, 3})),
    LiftQuery("a christmas tree", months=frozenset({12})),
    LiftQuery("colourful autumn leaves", months=frozenset({9, 10, 11})),
    LiftQuery("carved pumpkins", months=frozenset({10})),
    LiftQuery("cherry blossoms", months=frozenset({3, 4})),
    LiftQuery("the eiffel tower", countries=("France",)),
    LiftQuery("the golden gate bridge", places=("san francisco", "california")),
    LiftQuery("the colosseum", countries=("Italy",)),
)


def _month_number(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise ProbeQueryError(f"{where}: months must be 1-12 or month names, got {value!r}")
    if isinstance(value, int):
        if 1 <= value <= 12:
            return value
        raise ProbeQueryError(f"{where}: month {value} is not in 1-12")
    if isinstance(value, str):
        key = value.strip().casefold()
        for number, name in enumerate(_MONTHS, start=1):
            if key in (name.casefold(), name[:3].casefold()):
                return number
    raise ProbeQueryError(f"{where}: months must be 1-12 or month names, got {value!r}")


def _string_list(value: Any, name: str, where: str) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items:
        raise ProbeQueryError(f"{where}: '{name}' must be a non-empty string or list of strings")
    out = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ProbeQueryError(f"{where}: '{name}' entries must be non-empty strings")
        out.append(item.strip())
    return tuple(out)


def load_lift_queries(path: str | Path) -> list[LiftQuery]:
    """Read ``[{"query": ..., "months"?: [...], "places"?: [...], "countries"?: [...]}]``.

    ``months`` are numbers 1-12 or month names; ``places`` are substrings of the photo's
    "City, Region, Country" label; ``countries`` are country names or ISO codes. At least one
    expectation is required per entry. Raises ``ProbeQueryError`` with a precise message.
    """
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProbeQueryError(f"Cannot read the queries file: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ProbeQueryError(f"The queries file is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ProbeQueryError("The queries file must contain a JSON list of objects.")
    allowed = {"query", "months", "places", "countries"}
    out: list[LiftQuery] = []
    for i, entry in enumerate(data):
        where = f"entry {i + 1}"
        if not isinstance(entry, dict):
            raise ProbeQueryError(f"{where}: expected an object, got {type(entry).__name__}")
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise ProbeQueryError(f"{where}: unknown key(s) {', '.join(unknown)}")
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ProbeQueryError(f"{where}: 'query' must be a non-empty string")
        months: frozenset[int] = frozenset()
        if "months" in entry:
            raw = entry["months"]
            if not isinstance(raw, list) or not raw:
                raise ProbeQueryError(f"{where}: 'months' must be a non-empty list")
            months = frozenset(_month_number(m, where) for m in raw)
        places = _string_list(entry["places"], "places", where) if "places" in entry else ()
        countries = (
            _string_list(entry["countries"], "countries", where) if "countries" in entry else ()
        )
        if not (months or places or countries):
            raise ProbeQueryError(f"{where}: give at least one of months, places, countries")
        out.append(LiftQuery(query.strip(), months, places, countries))
    return out


# --------------------------------------------------------------------------- engine wrapper


class _Recorder:
    """Calls the engine and remembers wall-clock and per-stage timings."""

    def __init__(self, engine: _EvalEngine, clock: Callable[[], float]) -> None:
        self.engine = engine
        self.clock = clock
        self.walls: list[float] = []
        self.stages: dict[str, list[float]] = defaultdict(list)

    def search(self, request: SearchRequest, *, record: bool = True) -> SearchResponse:
        start = self.clock()
        response = self.engine.search(request)
        if record:
            self.walls.append((self.clock() - start) * 1000.0)
            for name, value in response.timings_ms.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.stages[str(name)].append(float(value))
        return response

    def latency(self) -> LatencyReport:
        return LatencyReport(
            samples=len(self.walls),
            p50_ms=_percentile(self.walls, 50),
            p95_ms=_percentile(self.walls, 95),
            stages={
                name: {"p50": _percentile(vals, 50), "p95": _percentile(vals, 95)}  # type: ignore[dict-item]
                for name, vals in sorted(self.stages.items())
            },
        )


class _Candidates:
    """Small cache of ``store.candidate_ids`` keyed by filters."""

    def __init__(self, store: Store, limit: int = 32) -> None:
        self.store = store
        self.limit = limit
        self._cache: dict[Filters, np.ndarray] = {}

    def get(self, filters: Filters) -> np.ndarray:
        if filters not in self._cache:
            if len(self._cache) >= self.limit:
                self._cache.clear()
            self._cache[filters] = self.store.candidate_ids(filters)
        return self._cache[filters]


# --------------------------------------------------------------------------- self-retrieval


def _label_matches(place: PlaceFilter | None, city: str, region: str | None, country: str | None):
    """Does a resolved place filter denote the stored label ``(city, region, country)``?

    The city must agree; region and country must agree wherever the filter states them.
    """
    if place is None or fold(place.city or "") != fold(city):
        return False
    return all(
        not given or not stored or fold(given) == fold(stored)
        for given, stored in ((place.region, region), (place.country, country))
    )


@dataclass(frozen=True)
class _Expect:
    date_range: DateRange | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None

    def _place_ok(self, place: PlaceFilter | None) -> bool:
        if self.city is None:
            return place is None
        return _label_matches(place, self.city, self.region, self.country)

    def matches(self, filters: Filters) -> bool:
        return (
            not filters.people
            and filters.date_range == self.date_range
            and self._place_ok(filters.place)
        )

    def is_namesake(self, filters: Filters) -> bool:
        """The city name resolved, but to a same-named place elsewhere."""
        place = filters.place
        return (
            self.city is not None
            and place is not None
            and not filters.people
            and filters.date_range == self.date_range
            and fold(place.city or "") == fold(self.city)
            and not self._place_ok(place)
        )


@dataclass
class _Outcome:
    type: str
    query: str
    path: str
    reason: str | None
    parsed_ok: bool = False
    filter_ok: bool = True
    candidates: int | None = None


AMBIGUOUS = "ambiguous_place_name"  # not a failure: the name really denotes several places


_PROBE_TYPES = ("month_year", "month_year_bare", "city", "city_month")


def _month_range(photo: IndexedPhoto) -> tuple[DateRange, str]:
    taken = photo.meta.taken
    assert taken is not None
    last = calendar.monthrange(taken.year, taken.month)[1]
    rng = DateRange(date(taken.year, taken.month, 1), date(taken.year, taken.month, last))
    return rng, f"{_MONTHS[taken.month - 1]} {taken.year}"


def _probes_for(photo: IndexedPhoto) -> list[tuple[str, str, _Expect]]:
    probes: list[tuple[str, str, _Expect]] = []
    facts = _facts(photo)
    month = _month_range(photo) if facts.month is not None else None
    city = _city(photo)
    if month:
        rng, label = month
        probes.append(("month_year", f"photos from {label}", _Expect(date_range=rng)))
        probes.append(("month_year_bare", label, _Expect(date_range=rng)))
    if city:
        place = photo.place
        assert place is not None
        where = {"city": city, "region": place.region, "country": place.country}
        probes.append(("city", f"photos in {city}", _Expect(**where)))
        if month:
            rng, label = month
            probes.append(
                ("city_month", f"photos in {city} in {label}", _Expect(date_range=rng, **where))
            )
    return probes


def _run_self_retrieval(
    rec: _Recorder,
    cands: _Candidates,
    photos: Sequence[IndexedPhoto],
    *,
    k: int,
    use_llm: bool,
    today: date,
    progress: Callable[[str], None],
) -> list[_Outcome]:
    outcomes: list[_Outcome] = []
    for n, photo in enumerate(photos, start=1):
        for ptype, query, expect in _probes_for(photo):
            outcomes.append(_probe(rec, cands, photo, ptype, query, expect, k, use_llm, today))
        if n % 25 == 0:
            progress(f"self-retrieval probes: {n}/{len(photos)} photos")
    return outcomes


def _probe(
    rec: _Recorder,
    cands: _Candidates,
    photo: IndexedPhoto,
    ptype: str,
    query: str,
    expect: _Expect,
    k: int,
    use_llm: bool,
    today: date,
) -> _Outcome:
    out = _Outcome(ptype, query, photo.meta.path, None)
    request = SearchRequest(
        query=query, k=k, use_llm=use_llm, allow_clarification=False, today=today
    )
    try:
        response = rec.search(request)
    except Exception as exc:
        out.reason = "engine_error"
        out.path = f"{photo.meta.path} ({type(exc).__name__})"
        return out
    out.candidates = response.candidates
    out.parsed_ok = expect.matches(response.parse.filters)
    applied_ok = expect.matches(response.applied)
    pool = cands.get(response.applied)
    hit_ids = [h.photo.id for h in response.hits]
    out.filter_ok = bool(np.isin(hit_ids, pool).all()) if hit_ids else True
    if expect.is_namesake(response.parse.filters):
        out.reason = AMBIGUOUS
    elif not out.parsed_ok:
        out.reason = "parser_misunderstood"
    elif not applied_ok:
        out.reason = "dropped_filter"
    elif response.candidates == 0:
        out.reason = "zero_candidates"
    elif not out.filter_ok:
        out.reason = "filter_violation"
    elif not bool(np.isin(photo.id, pool)):
        out.reason = "photo_not_in_candidates"
    return out


def _summarise_probes(outcomes: Sequence[_Outcome], show_paths: bool) -> list[ProbeTypeReport]:
    reports = []
    for ptype in _PROBE_TYPES:
        group = [o for o in outcomes if o.type == ptype]
        if not group:
            continue
        scored = [o for o in group if o.reason != AMBIGUOUS]
        answered = [o for o in scored if o.candidates is not None]
        failed = [o for o in scored if o.reason]
        cand = [float(o.candidates) for o in group if o.candidates is not None]
        successes = len(scored) - len(failed)
        examples = []
        if show_paths:
            examples = [
                {"query": o.query, "path": o.path, "reason": o.reason or ""} for o in failed
            ][:5]
        reports.append(
            ProbeTypeReport(
                type=ptype,
                probes=len(group),
                successes=successes,
                success_rate=_rate(successes, len(scored)),
                parsed_ok_rate=_rate(sum(o.parsed_ok for o in scored), len(scored)),
                filter_correct_rate=_rate(sum(o.filter_ok for o in answered), len(answered)),
                ambiguous=len(group) - len(scored),
                candidates_mean=_mean(cand),
                candidates_median=_percentile(cand, 50),
                failures=dict(sorted(Counter(o.reason for o in failed).items())),
                examples=examples,
            )
        )
    return reports


# --------------------------------------------------------------------------- lift probes


def _verdict(share: float, ci_low: float, base_rate: float) -> str:
    """``pass``: the whole 95% interval of the hit share beats the base rate and the share
    beats it by ``MIN_EFFECT``; ``weak``: above the base rate but not convincingly; ``fail``.

    A fixed lift threshold would be unreachable when the base rate is high (lift <= 1/base).
    """
    if ci_low > base_rate and share - base_rate >= MIN_EFFECT:
        return "pass"
    return "weak" if share > base_rate else "fail"


def _run_lift(
    rec: _Recorder,
    queries: Sequence[LiftQuery],
    facts: Sequence[_Facts],
    *,
    k: int,
    use_llm: bool,
    today: date,
) -> list[LiftResult]:
    results = []
    for lq in queries:
        eligible = [f for f in facts if lq.applicable(f)]
        base_n = len(eligible)
        base_m = sum(1 for f in eligible if lq.holds(f))
        res = LiftResult(
            query=lq.query,
            expectation=lq.describe(),
            k=k,
            base_eligible=base_n,
            base_matching=base_m,
            base_rate=_rate(base_m, base_n),
        )
        request = SearchRequest(
            query=lq.query, k=k, use_llm=use_llm, allow_clarification=False, today=today
        )
        try:
            response = rec.search(request)
        except Exception as exc:
            res.status, res.reason = "error", f"engine raised {type(exc).__name__}"
            results.append(res)
            continue
        res.hits = len(response.hits)
        if not response.applied.is_empty:
            res.status = "filtered"
            res.reason = "the query was parsed into filters, so the hits are not a visual ranking"
            results.append(res)
            continue
        hit_facts = [f for f in (_facts(h.photo) for h in response.hits) if lq.applicable(f)]
        res.hits_eligible = len(hit_facts)
        res.hits_matching = sum(1 for f in hit_facts if lq.holds(f))
        problem = None
        if base_n < MIN_BASE_SUPPORT:
            problem = f"too few library photos carry the needed metadata (need {MIN_BASE_SUPPORT})"
        elif base_m == 0:
            problem = "no library photo meets the expectation"
        elif base_m == base_n:
            problem = "every library photo meets the expectation, so lift is undefined"
        elif res.hits_eligible < MIN_HIT_SUPPORT:
            problem = (
                f"only {res.hits_eligible} top-{k} hits carry the needed metadata "
                f"(need {MIN_HIT_SUPPORT})"
            )
        if problem:
            res.status, res.reason = "insufficient_data", problem
            results.append(res)
            continue
        share = res.hits_matching / res.hits_eligible
        low, high = _wilson(res.hits_matching, res.hits_eligible)  # type: ignore[misc]
        base_rate = base_m / base_n
        res.share, res.share_ci_low, res.share_ci_high = share, low, high
        res.lift = share / base_rate
        res.verdict = _verdict(share, low, base_rate)
        results.append(res)
    return results


# --------------------------------------------------------------------------- feedback


def _feedback(store: Store, show_details: bool) -> FeedbackReport:
    summary = store.feedback_summary()
    if summary.rated == 0:
        return FeedbackReport(status="none")
    ci = _wilson(summary.relevant, summary.rated)
    per_query = [
        FeedbackQueryReport(
            query=q.query if show_details else f"query {i}",
            rated=q.rated,
            relevant=q.relevant,
            precision=q.precision,
        )
        for i, q in enumerate(summary.per_query, start=1)
    ]
    return FeedbackReport(
        status="ok" if summary.rated >= MIN_RATED else "insufficient_data",
        queries=len(summary.per_query),
        rated=summary.rated,
        relevant=summary.relevant,
        precision=summary.precision,
        precision_ci_low=ci[0] if ci else None,
        precision_ci_high=ci[1] if ci else None,
        mean_precision=summary.mean_precision,
        per_query=per_query,
    )


# --------------------------------------------------------------------------- safety battery


@dataclass(frozen=True)
class _SafetyCase:
    name: str
    query: str
    safe: Callable[[Filters, date], bool]
    control: bool = False  # a well-formed query that SHOULD produce the filter


def _no_date(f: Filters, _today: date) -> bool:
    return f.date_range is None


def _no_place(f: Filters, _today: date) -> bool:
    return f.place is None


def _no_people(f: Filters, _today: date) -> bool:
    return not f.people


def _open_from(start: date) -> Callable[[Filters, date], bool]:
    def check(f: Filters, today: date) -> bool:
        r = f.date_range
        return r is None or (r.start == start and r.end >= today)

    return check


def _open_until(end: date) -> Callable[[Filters, date], bool]:
    def check(f: Filters, _today: date) -> bool:
        r = f.date_range
        return r is None or r.end == end

    return check


def _either_year(f: Filters, _today: date) -> bool:
    r = f.date_range
    return r is None or (r.start == date(2019, 1, 1) and r.end == date(2020, 12, 31))


def _narrower_than_year(f: Filters, _today: date) -> bool:
    r = f.date_range
    return r is None or (r.start.year == r.end.year == 2019 and (r.end - r.start).days <= 124)


def _whole_2019(f: Filters, _today: date) -> bool:
    return f.date_range == DateRange(date(2019, 1, 1), date(2019, 12, 31))


def _visible_place(f: Filters, _today: date) -> bool:
    """A real but ambiguous namesake must still filter, and the filter must be visible: the
    one-line description the CLI / UI show names the place that was chosen."""
    from .engine import describe_filters

    return f.place is not None and "place " in describe_filters(f)


SAFETY_CASES: tuple[_SafetyCase, ...] = (
    _SafetyCase("operator_since", "photos since 2020", _open_from(date(2020, 1, 1))),
    _SafetyCase("operator_except", "photos except 2019", _no_date),
    _SafetyCase("operator_not_season", "not in summer", _no_date),
    _SafetyCase("operator_before", "photos before 2015", _open_until(date(2014, 12, 31))),
    _SafetyCase("operator_after", "after March 2022", _open_from(date(2022, 4, 1))),
    _SafetyCase("ambiguous_numeric_date", "03/04/2024", _no_date),
    _SafetyCase("two_dates", "2019 or 2020", _either_year),
    _SafetyCase("vague_time", "a while ago", _no_date),
    _SafetyCase("impossible_date", "31 February 2020", _no_date),
    _SafetyCase("other_language_date", "im Sommer 2019", _no_date),
    _SafetyCase("season_not_widened", "summer 2019", _narrower_than_year),
    _SafetyCase("unknown_place", "photos in Xyzzyville", _no_place),
    _SafetyCase("unknown_place_near", "near Xyzzyville", _no_place),
    _SafetyCase("unknown_place_two", "in Qwertyland", _no_place),
    _SafetyCase("generic_location_word", "in the kitchen", _no_place),
    _SafetyCase("scenery_not_place", "at the beach", _no_place),
    _SafetyCase("home_not_place", "at home", _no_place),
    _SafetyCase("unknown_person", "with Zorblax", _no_people),
    _SafetyCase("control_year", "photos from 2019", _whole_2019, control=True),
    _SafetyCase("control_namesake_place", "in Amherst", _visible_place, control=True),
)


def _run_safety(rec: _Recorder, today: date, progress: Callable[[str], None]) -> SafetyReport:
    """Run the battery. Controls are counted apart: they are not safety runs, they check that
    the guard rails do not suppress a well-formed query."""
    report = SafetyReport(
        cases=sum(1 for c in SAFETY_CASES if not c.control),
        controls=sum(1 for c in SAFETY_CASES if c.control),
    )
    for case in SAFETY_CASES:
        control_ok = True
        for clarify in (True, False):
            request = SearchRequest(
                query=case.query, k=1, use_llm=False, allow_clarification=clarify, today=today
            )
            try:
                response = rec.search(request, record=False)
            except Exception:
                verdict, reported, quiet = "error", False, False
            else:
                verdict = "safe" if case.safe(response.applied, today) else "wrong filter applied"
                reported = bool(
                    response.needs_clarification or response.dropped or response.parse.unresolved
                )
                quiet = response.applied.is_empty
            if case.control:
                control_ok = control_ok and verdict == "safe"
                continue
            report.runs += 1
            if verdict == "safe":
                report.safe_runs += 1
                report.reported_runs += reported
                report.quiet_runs += quiet and not reported
                continue
            report.unsafe_runs += 1
            report.errors += verdict == "error"
            report.unsafe_cases.append({"case": case.name, "query": case.query, "why": verdict})
        if case.control:
            report.controls_ok += control_ok
            if not control_ok:
                report.failed_controls.append(case.name)
    progress(f"safety battery: {report.runs} runs")
    return report


# --------------------------------------------------------------------------- place labels


def _run_roundtrip(
    rec: _Recorder,
    photos: Sequence[IndexedPhoto],
    *,
    use_llm: bool,
    today: date,
    show_paths: bool,
    progress: Callable[[str], None],
) -> PlaceRoundTrip:
    """For every distinct stored (city, region, country) label, parse ``photos in <city>``
    through the engine and check the resolved place is that label (so library-aware
    disambiguation between namesakes is included)."""
    labels: Counter[tuple[str, str | None, str | None]] = Counter()
    for photo in photos:
        if photo.place and photo.place.city:
            labels[(photo.place.city, photo.place.region, photo.place.country)] += 1
    report = PlaceRoundTrip(labels=len(labels), photos=sum(labels.values()))
    failing: Counter[str] = Counter()
    for n, ((city, region, country), count) in enumerate(sorted(labels.items()), start=1):
        request = SearchRequest(
            query=f"photos in {city}",
            k=1,
            use_llm=use_llm,
            allow_clarification=False,
            today=today,
        )
        try:
            place = rec.search(request, record=False).parse.filters.place
        except Exception:
            place = None
        if _label_matches(place, city, region, country):
            report.labels_ok += 1
            report.photos_ok += count
        else:
            failing[", ".join(p for p in (city, region, country) if p)] += count
        if n % 50 == 0:
            progress(f"place labels: {n}/{len(labels)}")
    report.label_rate = _rate(report.labels_ok, report.labels)
    report.photo_rate = _rate(report.photos_ok, report.photos)
    if show_paths:
        report.failing = _histogram(failing)
    return report


def _blur(report: EvalReport) -> None:
    """Hide the size of folded buckets: if any histogram folded a bucket, every library-wide
    count is rounded to ``ROUND_TO`` and the rates derived from them to 2 decimals, so no
    total minus the visible buckets gives back a hidden count."""
    lib = report.library
    hidden = any(h.other_buckets for h in (lib.months, lib.years, lib.cities, lib.countries))
    if not hidden and not report.place_roundtrip.failing.other_buckets:
        return
    lib.counts_rounded = True
    for name in ("photos", "dated", "month_known", "with_place", "with_city"):
        setattr(lib, name, _round_count(getattr(lib, name)))
    two = lambda x: None if x is None else round(x, 2)  # noqa: E731
    for t in report.self_retrieval:
        t.probes, t.successes, t.ambiguous = (
            _round_count(n) for n in (t.probes, t.successes, t.ambiguous)
        )
        t.success_rate, t.parsed_ok_rate = two(t.success_rate), two(t.parsed_ok_rate)
        t.filter_correct_rate = two(t.filter_correct_rate)
        t.failures = {name: max(ROUND_TO, _round_count(n)) for name, n in t.failures.items()}
    for r in report.lift:
        r.base_eligible, r.base_matching = (
            _round_count(r.base_eligible),
            _round_count(r.base_matching),
        )
        r.base_rate, r.lift = two(r.base_rate), two(r.lift)
    rt = report.place_roundtrip
    rt.photos, rt.photos_ok = _round_count(rt.photos), _round_count(rt.photos_ok)
    rt.photo_rate = two(rt.photo_rate)


# --------------------------------------------------------------------------- driver


def run_probes(
    engine: _EvalEngine,
    *,
    sample: int = 200,
    k: int = 24,
    seed: int = 0,
    today: date | None = None,
    queries_file: str | Path | None = None,
    show_paths: bool = False,
    use_llm: bool = True,
    include_default_queries: bool = True,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EvalReport:
    """Run every probe family against ``engine`` and return the report.

    ``sample`` is the number of photos used for self-retrieval probes (each yields up to four
    queries), ``k`` the result size, ``seed`` makes the sampling reproducible. ``queries_file``
    adds lift probes (see ``load_lift_queries``; raises ``ProbeQueryError``). Nothing here
    writes to the index. Paths and feedback query text appear in the report only when
    ``show_paths`` is true. The safety battery always runs in rules mode.
    """
    if sample < 1 or k < 1:
        raise ValueError("sample and k must be at least 1")
    today = today or date.today()
    say = progress or (lambda _msg: None)
    lift_queries: list[LiftQuery] = list(DEFAULT_LIFT_QUERIES) if include_default_queries else []
    if queries_file is not None:
        extra = {q.query.casefold(): q for q in load_lift_queries(queries_file)}
        lift_queries = [q for q in lift_queries if q.query.casefold() not in extra]
        lift_queries += list(extra.values())

    store = engine.store
    photos = _load_photos(store)
    report = EvalReport(
        params={
            "sample": sample,
            "k": k,
            "seed": seed,
            "today": today,
            "use_llm": use_llm,
            "show_paths": show_paths,
            "lift_queries": len(lift_queries),
        }
    )
    report.library = _profile(photos)
    if not photos:
        report.warnings.append("The index is empty: nothing to probe.")
    rec = _Recorder(engine, clock)
    cands = _Candidates(store)

    pool = sorted(
        (p for p in photos if _facts(p).month is not None or _city(p)), key=lambda p: p.id
    )
    if len(pool) > sample:
        pool = sorted(random.Random(seed).sample(pool, sample), key=lambda p: p.id)
    if photos and not pool:
        report.warnings.append("No photo has a usable date or city: self-retrieval skipped.")
    outcomes = _run_self_retrieval(
        rec, cands, pool, k=k, use_llm=use_llm, today=today, progress=say
    )
    report.self_retrieval = _summarise_probes(outcomes, show_paths)

    say("lift probes")
    report.lift = _run_lift(
        rec, lift_queries, [_facts(p) for p in photos], k=k, use_llm=use_llm, today=today
    )
    report.latency = rec.latency()
    say("place labels")
    report.place_roundtrip = _run_roundtrip(
        rec, photos, use_llm=use_llm, today=today, show_paths=show_paths, progress=say
    )
    say("safety battery")
    report.safety = _run_safety(rec, today, say)
    report.feedback = _feedback(store, show_paths)
    if report.feedback.status == "insufficient_data":
        report.warnings.append(
            f"Only {report.feedback.rated} rated hits; rate at least {MIN_RATED} for a usable "
            "precision estimate."
        )
    if any(r.status == "insufficient_data" for r in report.lift):
        report.warnings.append(
            "Some lift probes have insufficient data; they are reported but not scored."
        )
    _blur(report)
    return report


# --------------------------------------------------------------------------- markdown


def _recall(value: float | None) -> str:
    return "n/a (no visual text)" if value is None else _pct(value)


def _hist_line(name: str, h: Histogram) -> str:
    parts = [f"{key} {n}" for key, n in h.buckets.items()]
    if h.other_buckets:
        parts.append(f"other ({h.other_buckets} buckets under {MIN_BUCKET} photos or not shown)")
    return f"- {name}: " + (", ".join(parts) if parts else "none")


def _markdown(r: EvalReport) -> str:
    lib = r.library
    out = ["# Photo search evaluation", ""]
    p = r.params
    out.append(
        f"Sample {p.get('sample')} photos, k={p.get('k')}, seed={p.get('seed')}, "
        f"today={p.get('today')}, llm={'on' if p.get('use_llm') else 'off'}. "
        + (
            "File paths are shown."
            if p.get("show_paths")
            else f"Aggregates only; buckets under {MIN_BUCKET} photos are folded into 'other'."
        )
    )
    for w in r.warnings:
        out += ["", f"> Warning: {w}"]
    out += ["", "## Library", ""]
    about = "about " if lib.counts_rounded else ""
    out.append(
        f"- photos {about}{lib.photos}; with a trusted date {about}{lib.dated} "
        f"({_pct(_rate(lib.dated, lib.photos))}); with a place {about}{lib.with_place} "
        f"({_pct(_rate(lib.with_place, lib.photos))}); with a city {about}{lib.with_city}"
    )
    if lib.counts_rounded:
        out.append(
            f"- Library-wide counts are rounded to the nearest {ROUND_TO} because some small "
            "buckets are hidden."
        )
    out += [
        _hist_line("months", lib.months),
        _hist_line("years", lib.years),
        _hist_line("cities", lib.cities),
        _hist_line("countries", lib.countries),
    ]
    out += ["", "## Metadata self-retrieval", ""]
    if r.self_retrieval:
        out.append(
            "| probe | n | success | parsed ok | filters respected | recall@k "
            "| cand. mean | cand. median | ambiguous name |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|")
        for t in r.self_retrieval:
            out.append(
                f"| {t.type} | {t.probes} | {_pct(t.success_rate)} | {_pct(t.parsed_ok_rate)} "
                f"| {_pct(t.filter_correct_rate)} | {_recall(t.recall_at_k)} "
                f"| {_num(t.candidates_mean)} | {_num(t.candidates_median)} | {t.ambiguous} |"
            )
        out.append(
            "Filter correctness is the signal here; recall@k is only meaningful when the query "
            "has visual text, and these date / place-only probes cannot rank. 'Ambiguous name' "
            "probes (a city name that really denotes several places) are excluded from the "
            "rates."
        )
        for t in r.self_retrieval:
            if t.failures:
                why = ", ".join(f"{name} {n}" for name, n in t.failures.items())
                out.append(f"- {t.type} failures: {why}")
            for ex in t.examples:
                out.append(f"  - {ex['reason']}: {ex['query']!r} ({ex['path']})")
    else:
        out.append("No probes ran (no photo with a date or city).")
    out += ["", "## Lift probes (share of top-k meeting a metadata expectation)", ""]
    out.append("| query | expectation | status | share | base rate | lift | verdict |")
    out.append("|---|---|---|---|---|---|---|")
    for lr in r.lift:
        if lr.status == "ok":
            status = f"ok ({lr.hits_matching}/{lr.hits_eligible})"
        elif lr.status == "insufficient_data":
            status = "insufficient data"
        else:
            status = lr.status
        out.append(
            f"| {lr.query} | {lr.expectation} | {status} | {_pct(lr.share)} "
            f"| {_pct(lr.base_rate)} | {_num(lr.lift, 2)} | {lr.verdict or '-'} |"
        )
    notes = [f"- {lr.query!r}: {lr.reason}" for lr in r.lift if lr.reason]
    out += notes
    rt = r.place_roundtrip
    out += ["", "## Place label round-trip", ""]
    if rt.labels:
        about = "about " if lib.counts_rounded else ""
        out.append(
            f"- labels resolving back to themselves: {rt.labels_ok}/{rt.labels} "
            f"({_pct(rt.label_rate)}); photos covered: {about}{rt.photos_ok}/{about}{rt.photos} "
            f"({_pct(rt.photo_rate)})"
        )
        if not r.params.get("show_paths"):
            out.append(f"- {rt.labels - rt.labels_ok} labels failed (names hidden; --show-paths)")
        else:
            out.append(_hist_line("failing labels (photos)", rt.failing))
    else:
        out.append("No photo has a city label.")
    fb = r.feedback
    out += ["", "## Feedback (thumbs up/down)", ""]
    if fb.status == "none":
        out.append("No ratings yet. Rate hits in the UI, or try `suggest_probe_queries`.")
    else:
        out.append(
            f"- {fb.rated} rated hits over {fb.queries} queries; precision "
            f"{_pct(fb.precision)} (95% CI {_pct(fb.precision_ci_low)} to "
            f"{_pct(fb.precision_ci_high)}); mean per-query precision "
            f"{_pct(fb.mean_precision)}"
            + ("; insufficient data" if fb.status == "insufficient_data" else "")
        )
        out.append("- Precision covers rated hits only, which are not a random sample.")
        for q in fb.per_query:
            out.append(f"  - {q.query}: {q.relevant}/{q.rated} ({_pct(q.precision)})")
    lat = r.latency
    out += ["", "## Latency", ""]
    out.append(f"- {lat.samples} searches: p50 {_num(lat.p50_ms)} ms, p95 {_num(lat.p95_ms)} ms")
    for name, v in lat.stages.items():
        out.append(f"  - {name}: p50 {_num(v['p50'])} ms, p95 {_num(v['p95'])} ms")
    s = r.safety
    out += ["", "## Safety checks (rules mode)", ""]
    out.append(
        f"- {s.runs} runs over {s.cases} tricky queries: safe {s.safe_runs} "
        f"(reported {s.reported_runs}, no filter {s.quiet_runs}), unsafe {s.unsafe_runs}, "
        f"engine errors {s.errors}"
    )
    out.append(f"- controls (well-formed queries that must filter): {s.controls_ok}/{s.controls}")
    for u in s.unsafe_cases:
        out.append(f"  - UNSAFE {u['case']}: {u['query']!r} ({u['why']})")
    for name in s.failed_controls:
        out.append(f"  - control failed: {name}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- CLI glue


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``facesorter eval`` options on ``parser``."""
    parser.add_argument("--index", metavar="PATH", default=None, help="photo index file")
    parser.add_argument(
        "--sample", type=int, default=200, metavar="N", help="photos to sample (default 200)"
    )
    parser.add_argument("-k", type=int, default=24, metavar="N", help="results per query")
    parser.add_argument("--seed", type=int, default=0, metavar="N", help="sampling seed")
    parser.add_argument(
        "--queries",
        metavar="FILE",
        default=None,
        help='JSON list of extra lift probes: [{"query", "months"?, "places"?, "countries"?}]',
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--markdown", metavar="FILE", default=None, help="also write markdown")
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="include file paths and feedback query text (default: aggregates only)",
    )
    parser.add_argument(
        "--llm",
        choices=LLM_CHOICES,
        default="auto",
        help="query-understanding backend used by the probes (default auto)",
    )
    parser.add_argument("--no-llm", action="store_true", help="same as --llm none")


def _default_engine_factory(index: str, **kwargs: Any) -> Any:
    from .engine import open_engine

    return open_engine(index, **kwargs)


def _index_path(args: argparse.Namespace) -> str:
    given = getattr(args, "index", None)
    if given:
        return str(given)
    from .engine import default_index_path

    return str(default_index_path())


def run(
    args: argparse.Namespace,
    *,
    engine_factory: Callable[..., Any] | None = None,
) -> int:
    """Run the evaluation for the CLI. Returns 0, 1 (harness failure) or 2 (bad arguments).

    ``engine_factory(index_path, llm=mode, use_llm=bool)`` has the ``open_engine`` signature
    and must return an object with ``search`` and ``store`` (and optionally ``close``); it
    defaults to ``facesorter.search.engine.open_engine``.
    """

    def err(message: str) -> None:
        print(message, file=sys.stderr)

    sample = getattr(args, "sample", 200)
    k = getattr(args, "k", 24)
    llm_arg = getattr(args, "llm", "auto")
    no_llm = bool(getattr(args, "no_llm", False))
    if sample < 1 or k < 1:
        err("error: --sample and -k must be at least 1.")
        return 2
    if llm_arg not in LLM_CHOICES:
        err(f"error: --llm must be one of {', '.join(LLM_CHOICES)}.")
        return 2
    if no_llm and llm_arg not in ("auto", "none"):
        err("error: --no-llm conflicts with --llm.")
        return 2
    llm = "none" if no_llm else llm_arg
    queries = getattr(args, "queries", None)
    if queries is not None:
        try:
            load_lift_queries(queries)
        except ProbeQueryError as exc:
            err(f"error: {exc}")
            return 2

    factory = engine_factory or _default_engine_factory
    engine = None
    try:
        engine = factory(_index_path(args), llm=llm, use_llm=llm != "none")
        report = run_probes(
            engine,
            sample=sample,
            k=k,
            seed=getattr(args, "seed", 0),
            queries_file=queries,
            show_paths=getattr(args, "show_paths", False),
            use_llm=llm != "none",
            progress=lambda msg: err(f"[eval] {msg}"),
        )
        markdown = report.to_markdown()
        md_path = getattr(args, "markdown", None)
        if md_path:
            Path(md_path).expanduser().write_text(markdown, encoding="utf-8")
        print(report.to_json() if getattr(args, "json", False) else markdown, end="")
        return 0
    except Exception as exc:
        err(f"error: evaluation failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
