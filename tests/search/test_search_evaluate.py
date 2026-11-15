"""Tests for facesorter.search.evaluate: the caption-free evaluation harness.

The synthetic library has a deliberate structure (colour follows the month, places follow the
row number) so every number in the report can be derived independently in the test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from facesorter.search import engine as engine_module
from facesorter.search import evaluate as ev
from facesorter.search.engine import Engine
from facesorter.search.errors import IndexNotFoundError
from facesorter.search.evaluate import (
    LiftQuery,
    ProbeQueryError,
    add_arguments,
    load_lift_queries,
    run,
    run_probes,
    suggest_probe_queries,
)
from facesorter.search.parser import QueryParser
from facesorter.search.store import Store
from facesorter.search.testing import (
    FAKE_DIM,
    FAKE_MODEL_NAME,
    FakeImageEncoder,
    FakeTextEncoder,
    make_color_image,
)
from facesorter.search.types import (
    DateRange,
    Filters,
    ParsedQuery,
    ParseResult,
    PhotoMeta,
    PlaceFilter,
    PlaceLabel,
    SearchRequest,
    SearchResponse,
    Unresolved,
)

TODAY = date(2026, 11, 8)
PARIS = PlaceLabel("Paris", "Ile-de-France", "France", "FR")
GOA = PlaceLabel("Goa", "Goa", "India", "IN")
BERLIN = PlaceLabel("Berlin", "Berlin", "Germany", "DE")
ROME = PlaceLabel("Rome", "Lazio", "Italy", "IT")
ZAGREB = PlaceLabel("Zagreb", None, "Croatia", "HR")
OSLO = PlaceLabel("Oslo", None, "Norway", "NO")
AMHERST = PlaceLabel("Amherst", "Massachusetts", "United States", "US")
WINTER = {12, 1, 2, 3}


# --------------------------------------------------------------------------- the library


@dataclass(frozen=True)
class Spec:
    path: str
    taken: datetime | None
    source: str
    precision: str
    place: PlaceLabel | None
    color: str

    @property
    def month(self) -> int | None:
        """The month a trusted date puts the photo in (what the harness may rely on)."""
        if self.taken is None or self.source not in ("exif", "folder"):
            return None
        return None if self.precision == "year" else self.taken.month


def build_specs() -> list[Spec]:
    specs: list[Spec] = []
    for i in range(96):
        month = i % 12 + 1
        year = 2023 + (i // 12) % 2
        taken = datetime(year, month, 1 + (i % 7) * 3 + i // 48, i % 24, i % 60)
        color = "white" if month in WINTER else "yellow" if month in (6, 7, 8) else "green"
        place = (PARIS, PARIS, GOA, None)[i % 4]
        specs.append(Spec(f"/lib/photo_{i:03d}.jpg", taken, "exif", "day", place, color))
    for j in range(5):  # a small city, exactly at the minimum bucket size
        taken = datetime(2023, 6, 2 + j, 12, 0)
        specs.append(Spec(f"/lib/berlin_{j}.jpg", taken, "exif", "day", BERLIN, "yellow"))
    for j in range(2):  # undated (mtime) photos in Paris
        taken = datetime(2022, 1, 1 + j, 9, 0)
        specs.append(Spec(f"/lib/mtime_{j}.jpg", taken, "mtime", "day", PARIS, "blue"))
    specs.append(
        Spec("/lib/folder_may.jpg", datetime(2023, 5, 1), "folder", "month", None, "green")
    )
    return specs


class FakeGazetteer:
    names = {
        "paris": PARIS,
        "goa": GOA,
        "berlin": BERLIN,
        "amherst": AMHERST,
        "rome": ROME,
    }

    def label_for(self, lat: float, lon: float) -> PlaceLabel | None:  # pragma: no cover
        return None

    def resolve(self, name: str) -> PlaceFilter | None:
        label = self.names.get(name.strip().lower())
        if label is None:
            return None
        return PlaceFilter(label.city, label.region, label.country, text=name)


def fill_store(store: Store, specs: list[Spec]) -> list[int]:
    encoder = FakeImageEncoder()
    vectors = {c: encoder.encode_images([make_color_image(c)])[0] for c in {s.color for s in specs}}
    records = [
        (
            PhotoMeta(
                path=s.path,
                size=10,
                mtime_ns=1,
                taken=s.taken,
                date_source=s.source,  # type: ignore[arg-type]
                date_precision=s.precision,  # type: ignore[arg-type]
            ),
            s.place,
            vectors[s.color],
        )
        for s in specs
    ]
    return store.upsert_photos(records)


def new_store(tmp_path: Path, name: str = "index.db") -> Store:
    return Store.open(tmp_path / name, model_name=FAKE_MODEL_NAME, dim=FAKE_DIM)


def make_engine(store: Store) -> Engine:
    parser = QueryParser(None, FakeGazetteer(), store)
    return Engine(store, FakeTextEncoder(), parser, FakeGazetteer())


@pytest.fixture()
def specs() -> list[Spec]:
    return build_specs()


@pytest.fixture()
def lib(tmp_path, specs):
    store = new_store(tmp_path)
    ids = fill_store(store, specs)
    yield store, ids
    store.close()


@pytest.fixture()
def engine(lib):
    return make_engine(lib[0])


def fake_clock() -> Callable[[], float]:
    counter = itertools.count()
    return lambda: next(counter) * 0.002  # every search measures 2 ms


def probe(engine, **kw):
    kw.setdefault("today", TODAY)
    kw.setdefault("clock", fake_clock())
    kw.setdefault("include_default_queries", False)
    return run_probes(engine, **kw)


def by_type(report):
    return {t.type: t for t in report.self_retrieval}


# --------------------------------------------------------------------------- stub engines


class StubEngine:
    """A real store behind a scripted ``search``."""

    def __init__(self, store: Store, fn: Callable[[SearchRequest], SearchResponse]) -> None:
        self.store = store
        self.fn = fn
        self.requests: list[SearchRequest] = []
        self.closed = False

    def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        return self.fn(request)

    def close(self) -> None:
        self.closed = True


def make_response(
    query: str = "",
    *,
    parse_filters: Filters | None = None,
    applied: Filters | None = None,
    hits=(),
    candidates: int = 0,
    dropped=(),
    needs=(),
    unresolved=(),
    timings=None,
) -> SearchResponse:
    parsed = ParsedQuery(raw=query, text="")
    parse = ParseResult(parsed, parse_filters or Filters(), tuple(unresolved))
    return SearchResponse(
        hits=tuple(hits),
        parse=parse,
        applied=applied if applied is not None else (parse_filters or Filters()),
        candidates=candidates,
        needs_clarification=tuple(needs),
        dropped=tuple(dropped),
        timings_ms=timings or {},
    )


def quiet(real: Engine) -> StubEngine:
    """The real engine with its (non-deterministic) stage timings removed."""
    return wrap(real, lambda req, resp: replace(resp, timings_ms={}))


def wrap(real: Engine, mutate: Callable[[SearchRequest, SearchResponse], SearchResponse]):
    return StubEngine(real.store, lambda req: mutate(req, real.search(req)))


# --------------------------------------------------------------------------- library profile


def test_profile_counts_and_histograms(engine, specs):
    report = probe(engine, sample=500)
    lib = report.library
    assert not lib.counts_rounded  # no bucket is hidden, so counts are exact
    assert lib.photos == len(specs) == 104
    assert lib.dated == 102  # 96 exif + 5 berlin + 1 folder; mtime photos are not trusted
    assert lib.month_known == 102
    assert lib.with_place == 79 and lib.with_city == 79

    months, years = {}, {}
    for s in specs:
        if s.month:
            name = ev._MONTHS[s.month - 1][:3]
            months[name] = months.get(name, 0) + 1
            years[str(s.taken.year)] = years.get(str(s.taken.year), 0) + 1
    assert lib.months.buckets == months
    assert lib.months.other_buckets == 0
    assert lib.years.buckets == years
    assert lib.cities.buckets == {"Paris": 50, "Goa": 24, "Berlin": 5}
    assert lib.countries.buckets == {"France": 50, "India": 24, "Germany": 5}
    assert lib.cities.other_buckets == 0


# A library with hidden buckets: Paris 20, Goa 14, Rome 6 are shown; Zagreb 3 and Oslo 4 are
# folded; 6 photos have no place. Months are Jan-Mar 2023 only, so only cities hide anything.
HIDDEN = {"Zagreb": 3, "Oslo": 4}


def month_of(i: int) -> int:
    return 1 if i < 22 else 2 if i < 38 else 3  # 22 / 16 / 15 photos


class HiddenGazetteer(FakeGazetteer):
    """Also knows the two small cities."""

    names = {**FakeGazetteer.names, "zagreb": ZAGREB, "oslo": OSLO}


def privacy_engine(store, *, knows_small_cities: bool = True) -> Engine:
    gaz = HiddenGazetteer() if knows_small_cities else FakeGazetteer()
    return Engine(store, FakeTextEncoder(), QueryParser(None, gaz, store), gaz)


def privacy_store(tmp_path):
    labels = (
        [PARIS] * 20
        + [GOA] * 14
        + [ROME] * 6
        + [ZAGREB] * HIDDEN["Zagreb"]
        + [OSLO] * HIDDEN["Oslo"]
        + [None] * 6
    )
    specs = [
        Spec(
            f"/lib/p{i}.jpg",
            datetime(2023, month_of(i), 1 + i % 20, 10),
            "exif",
            "day",
            pl,
            "green",
        )
        for i, pl in enumerate(labels)
    ]
    store = new_store(tmp_path, "privacy.db")
    fill_store(store, specs)
    return store


def visible_numbers(obj, skip=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in skip:
                yield from visible_numbers(value, skip)
    elif isinstance(obj, list):
        for value in obj:
            yield from visible_numbers(value, skip)
    elif isinstance(obj, int) and not isinstance(obj, bool):
        yield obj


def library_markdown(md: str) -> str:
    """The sections that describe the library (everything before the lift table)."""
    keep, on = [], False
    for line in md.splitlines():
        if line.startswith("## Lift") or line.startswith("## Feedback"):
            on = False
        if line.startswith("## Place label") or line.startswith("## Library"):
            on = True
        if on:
            keep.append(line)
    return "\n".join(keep)


@pytest.mark.parametrize("show_paths", [False, True])
def test_hidden_bucket_sizes_cannot_be_recovered_by_subtraction(tmp_path, show_paths):
    store = privacy_store(tmp_path)
    report = probe(privacy_engine(store), sample=500, show_paths=show_paths)
    lib = report.library
    assert lib.counts_rounded
    assert lib.cities.buckets == {"Paris": 20, "Goa": 14, "Rome": 6}
    assert lib.cities.other_buckets == 2
    assert lib.countries.other_buckets == 2
    assert not hasattr(lib.cities, "total") and not hasattr(lib.cities, "other_photos")
    # library-wide counts are rounded to the nearest 5
    assert (lib.photos, lib.dated, lib.month_known) == (55, 55, 55)  # exact value is 53
    assert (lib.with_place, lib.with_city) == (45, 45)  # exact value is 47
    md = report.to_markdown()
    assert "about 45" in md and "rounded to the nearest 5" in md

    sections = {
        k: v
        for k, v in report.to_dict().items()
        if k in ("library", "self_retrieval", "lift", "place_roundtrip")
    }
    numbers = set(visible_numbers(sections))
    numbers |= {int(n) for n in re.findall(r"(?<![\d.])\d+(?![\d.])", library_markdown(md))}
    visible = [20, 14, 6]
    sums = {sum(c) for r in range(4) for c in itertools.combinations(visible, r)}
    secrets = {HIDDEN["Zagreb"], HIDDEN["Oslo"], sum(HIDDEN.values())}
    recovered = {(x, s_) for x in numbers for s_ in sums if x - s_ in secrets}
    assert recovered == set()
    # neither name nor country of a hidden place leaks
    text = report.to_json() + md
    assert "Zagreb" not in text and "Croatia" not in text and "Oslo" not in text
    if show_paths:
        assert report.place_roundtrip.failing.other_buckets == 0
    store.close()


def test_failing_small_labels_stay_hidden_and_failure_counts_are_rounded_up(tmp_path):
    store = privacy_store(tmp_path)
    engine = privacy_engine(store, knows_small_cities=False)
    for show_paths in (False, True):
        report = probe(engine, sample=500, show_paths=show_paths)
        rt = report.place_roundtrip
        assert (rt.labels, rt.labels_ok) == (5, 3)  # Zagreb and Oslo do not resolve
        assert (rt.photos, rt.photos_ok) == (45, 40)  # exact 47 and 40
        assert rt.failing.buckets == {}
        assert rt.failing.other_buckets == (2 if show_paths else 0)
        hidden = report.to_dict()
        hidden.pop("self_retrieval")  # failure *examples* (show_paths) quote the queries
        text = json.dumps(hidden) + report.place_roundtrip.failing.__repr__()
        md = report.to_markdown().split("## Metadata self-retrieval")[0]
        md += report.to_markdown().split("## Place label round-trip")[1]
        for name in ("Oslo", "Zagreb", "Croatia", "Norway"):
            assert name not in text and name not in md
        for t in report.self_retrieval:
            for count in t.failures.values():
                assert count >= 5 and count % 5 == 0  # never 3, 4 or their sum 7
    store.close()


def test_rates_are_blurred_when_counts_are_rounded(tmp_path):
    store = privacy_store(tmp_path)
    report = probe(privacy_engine(store), sample=500)
    for t in report.self_retrieval:
        assert t.probes % 5 == 0 and t.successes % 5 == 0
        for rate in (t.success_rate, t.parsed_ok_rate, t.filter_correct_rate):
            assert rate == round(rate, 2)
    store.close()


def test_histogram_counts_buckets_without_sizes():
    h = ev._histogram({"a": 9, "b": 1, "c": 2, "d": 7}, limit=1)
    assert h.buckets == {"a": 9}
    assert h.other_buckets == 3
    empty = ev._histogram({})
    assert empty.buckets == {} and empty.other_buckets == 0
    assert ev._hist_line("x", empty) == "- x: none"


def test_histogram_minimum_bucket_boundary():
    h = ev._histogram({"five": 5, "four": 4})
    assert h.buckets == {"five": 5} and h.other_buckets == 1
    assert "four" not in ev._hist_line("x", h)


def test_round_count():
    assert [ev._round_count(n) for n in (0, 1, 2, 3, 7, 8, 22, 23)] == [0, 0, 0, 5, 5, 10, 20, 25]


# --------------------------------------------------------------------------- self-retrieval


def oracle_matches(specs, kind, spec):
    out = []
    for i, s in enumerate(specs):
        if kind in ("month_year", "month_year_bare"):
            ok = s.month and (s.taken.year, s.taken.month) == (spec.taken.year, spec.taken.month)
        elif kind == "city":
            ok = s.place and s.place.city == spec.place.city
        else:
            ok = (
                s.month
                and (s.taken.year, s.taken.month) == (spec.taken.year, spec.taken.month)
                and s.place
                and s.place.city == spec.place.city
            )
        if ok:
            out.append(i)
    return out


def eligible(spec, kind):
    month = spec.month is not None
    city = spec.place is not None
    return {
        "month_year": month,
        "month_year_bare": month,
        "city": city,
        "city_month": month and city,
    }[kind]


def test_self_retrieval_exact_numbers_with_large_k(engine, specs):
    report = probe(engine, sample=500, k=500)
    types = by_type(report)
    assert list(types) == ["month_year", "month_year_bare", "city", "city_month"]
    for kind, t in types.items():
        pool = [s for s in specs if eligible(s, kind)]
        sizes = [len(oracle_matches(specs, kind, s)) for s in pool]
        assert t.probes == len(pool) and t.successes == len(pool)
        assert t.success_rate == 1.0 and t.parsed_ok_rate == 1.0
        assert t.filter_correct_rate == 1.0
        assert t.recall_at_k is None and t.ambiguous == 0
        assert t.failures == {} and t.examples == []
        assert t.candidates_mean == pytest.approx(sum(sizes) / len(sizes))
        assert t.candidates_median == float(np.median(sizes))
    assert (types["month_year"].probes, types["city"].probes, types["city_month"].probes) == (
        102,
        79,
        77,
    )


def test_recall_is_reported_as_not_applicable_for_filter_only_probes(engine):
    report = probe(engine, sample=500, k=3)
    assert all(t.recall_at_k is None for t in report.self_retrieval)
    md = report.to_markdown()
    assert md.count("n/a (no visual text)") == len(report.self_retrieval)
    assert "recall@k is only meaningful when the query has visual text" in md
    assert "chance" not in md


def test_probe_queries_use_prepositions(engine):
    stub = wrap(engine, lambda req, resp: resp)
    probe(stub, sample=500)
    queries = {r.query for r in stub.requests}
    assert "photos in Paris" in queries
    assert "photos in Paris in January 2023" in queries
    assert "photos from January 2023" in queries and "January 2023" in queries
    assert not any(q.startswith("Paris ") for q in queries)


def test_sampling_is_deterministic_per_seed(engine):
    def queries(seed):
        stub = wrap(engine, lambda req, resp: resp)
        probe(stub, sample=10, seed=seed, include_default_queries=False)
        return [r.query for r in stub.requests]

    first, again, other = queries(1), queries(1), queries(2)
    assert first == again
    assert first != other
    reports = [probe(quiet(engine), sample=10, seed=1) for _ in range(2)]
    assert reports[0].to_dict() == reports[1].to_dict()
    assert reports[0].to_json() == reports[1].to_json()


def test_sample_caps_number_of_photos(engine):
    stub = wrap(engine, lambda req, resp: resp)
    report = probe(stub, sample=4)
    # at most four probes per sampled photo
    assert sum(t.probes for t in report.self_retrieval) <= 16
    assert report.params["sample"] == 4


# --------------------------------------------------------------------------- broken engines


def test_engine_that_drops_filters_is_caught(engine):
    def drop(req, resp):
        return replace(resp, applied=Filters(), candidates=resp.candidates)

    report = probe(wrap(engine, drop), sample=500, k=500)
    for t in report.self_retrieval:
        assert t.success_rate == 0.0
        assert t.failures == {"dropped_filter": t.probes}
        assert t.parsed_ok_rate == 1.0  # the parser was fine; the engine lost the filter


def test_engine_that_ignores_filters_when_ranking_is_caught(engine):
    everything = engine.search(SearchRequest(query="", k=500))

    def ignore(req, resp):
        return replace(resp, hits=everything.hits)  # applied still claims the filters

    report = probe(wrap(engine, ignore), sample=500, k=500)
    for t in report.self_retrieval:
        assert t.failures == {"filter_violation": t.probes}
        assert t.filter_correct_rate == 0.0


def test_parser_misunderstanding_is_reported(engine):
    def blank(req, resp):
        garbled = engine.search(replace(req, query="hello there"))
        return garbled

    report = probe(wrap(engine, lambda req, resp: blank(req, resp)), sample=500)
    for t in report.self_retrieval:
        assert t.failures == {"parser_misunderstood": t.probes}
        assert t.parsed_ok_rate == 0.0


def test_wrong_date_or_extra_people_is_not_the_expected_filter():
    month = DateRange(date(2023, 3, 1), date(2023, 3, 31))
    e = ev._Expect(date_range=month)
    assert e.matches(Filters(date_range=month))
    assert not e.matches(Filters(date_range=DateRange(date(2023, 3, 1), date(2023, 4, 30))))
    assert not e.matches(Filters(date_range=month, people=("Bob",)))
    assert not e.matches(Filters(date_range=month, place=PlaceFilter(city="Paris")))
    c = ev._Expect(city="São Paulo")
    assert c.matches(Filters(place=PlaceFilter(city="sao paulo")))
    assert not c.matches(Filters(place=PlaceFilter(country="Brazil")))
    assert not c.matches(Filters())
    assert not ev._label_matches(None, "Paris", None, None)
    paris = ev._Expect(city="Paris", region="Ile-de-France", country="France")
    assert paris.matches(Filters(place=PlaceFilter(city="Paris")))
    assert paris.matches(Filters(place=PlaceFilter(city="Paris", country="France")))
    assert not paris.matches(Filters(place=PlaceFilter(city="Paris", region="Texas")))
    assert paris.is_namesake(Filters(place=PlaceFilter(city="Paris", region="Texas")))
    assert not paris.is_namesake(Filters(place=PlaceFilter(city="Paris")))
    assert not paris.is_namesake(Filters(place=PlaceFilter(city="Lyon")))
    assert not paris.is_namesake(Filters())
    assert not e.is_namesake(Filters(date_range=month))
    assert not paris.is_namesake(
        Filters(place=PlaceFilter(city="Paris", region="Texas"), people=("B",))
    )


def test_namesakes_are_reported_separately_not_as_parser_failures(engine):
    texas = PlaceFilter(city="Paris", region="Texas", country="United States", text="Paris")

    def to_texas(req, resp):
        filters = replace(resp.parse.filters, place=texas) if resp.parse.filters.place else None
        if filters is None or resp.parse.filters.place.city != "Paris":
            return resp
        parse = replace(resp.parse, filters=filters)
        return replace(resp, parse=parse, applied=filters)

    report = probe(wrap(engine, to_texas), sample=500)
    types = by_type(report)
    assert types["city"].ambiguous == 50 and types["city_month"].ambiguous == 50 - 2
    assert types["city"].failures == {}  # Goa and Berlin probes are fine
    assert types["city"].success_rate == 1.0
    assert types["month_year"].ambiguous == 0
    assert "50" in report.to_markdown()


def test_dictionary_word_city_resolves_with_a_preposition(tmp_path):
    reading = PlaceLabel("Reading", "England", "United Kingdom", "GB")

    class Gaz(FakeGazetteer):
        names = {"reading": reading}

    store = new_store(tmp_path, "reading.db")
    specs = [
        Spec(f"/lib/r{i}.jpg", datetime(2023, 3, 1 + i, 9), "exif", "day", reading, "green")
        for i in range(6)
    ]
    fill_store(store, specs)
    parser = QueryParser(None, Gaz(), store)
    engine = Engine(store, FakeTextEncoder(), parser, Gaz())
    report = probe(engine, sample=50)
    types = by_type(report)
    assert types["city"].success_rate == 1.0 and types["city_month"].success_rate == 1.0
    assert report.place_roundtrip.labels_ok == 1
    store.close()


def test_zero_candidates_and_missing_photo_and_engine_errors(lib, engine):
    store, _ = lib

    def zero(req, resp):
        return replace(resp, hits=(), candidates=0)

    report = probe(wrap(engine, zero), sample=500)
    assert all(t.failures == {"zero_candidates": t.probes} for t in report.self_retrieval)

    class ProxyStore:
        """Claims an unrelated candidate pool for every filter."""

        def __init__(self, inner):
            self.inner = inner

        def candidate_ids(self, filters=None):
            if filters is None:
                return self.inner.candidate_ids(None)
            return np.array([10**6], dtype=np.int64)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    nohits = StubEngine(ProxyStore(store), lambda req: replace(engine.search(req), hits=()))
    report = probe(nohits, sample=500)
    assert all(t.failures == {"photo_not_in_candidates": t.probes} for t in report.self_retrieval)
    assert all(t.filter_correct_rate == 1.0 for t in report.self_retrieval)

    def boom(req):
        raise RuntimeError("model fell over")

    report = probe(StubEngine(store, boom), sample=500, show_paths=True)
    for t in report.self_retrieval:
        assert t.failures == {"engine_error": t.probes}
        assert t.filter_correct_rate is None and t.candidates_mean is None
        assert "RuntimeError" in t.examples[0]["path"]
    assert report.safety.errors == report.safety.runs
    assert [r.status for r in report.lift] == []  # defaults are off in this helper


def test_paths_only_appear_with_show_paths(engine):
    def drop(req, resp):
        return replace(resp, applied=Filters())

    quiet = probe(wrap(engine, drop), sample=20)
    assert all(t.examples == [] for t in quiet.self_retrieval)
    assert "photo_0" not in quiet.to_json() and "/lib/" not in quiet.to_markdown()

    loud = probe(wrap(engine, drop), sample=20, show_paths=True)
    examples = [e for t in loud.self_retrieval for e in t.examples]
    assert examples and all(e["path"].startswith("/lib/") for e in examples)
    assert all(len(t.examples) <= 5 for t in loud.self_retrieval)
    assert "/lib/" in loud.to_json() and "/lib/" in loud.to_markdown()
    assert "dropped_filter" in loud.to_markdown()


# --------------------------------------------------------------------------- lift probes


def write_queries(tmp_path, data) -> Path:
    path = tmp_path / "q.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def lift_for(engine, tmp_path, entries, **kw):
    path = write_queries(tmp_path, entries)
    report = probe(engine, queries_file=path, sample=1, **kw)
    return {r.query: r for r in report.lift}


def test_lift_exact_numbers(engine, specs, tmp_path):
    winter = [s for s in specs if s.month]
    base_matching = sum(1 for s in winter if s.month in WINTER)
    results = lift_for(
        engine,
        tmp_path,
        [
            {"query": "white snow", "months": [12, 1, 2, 3]},
            {"query": "white snow again", "months": ["dec", "January", 2, "mar"]},
            {"query": "white powder", "months": [6, 7, 8]},
            {"query": "yellow fluff", "months": list(range(1, 12))},
        ],
        k=24,
    )
    r = results["white snow"]
    assert r.status == "ok" and r.verdict == "pass"
    assert (r.hits, r.hits_eligible, r.hits_matching) == (24, 24, 24)
    assert (r.base_eligible, r.base_matching) == (len(winter), base_matching) == (102, 32)
    assert r.base_rate == pytest.approx(32 / 102)
    assert r.share == 1.0 and r.lift == pytest.approx(102 / 32)
    assert r.share_ci_low == pytest.approx(0.8620, abs=1e-3)
    assert r.expectation == "months=Jan,Feb,Mar,Dec"
    assert results["white snow again"].lift == r.lift

    summer = results["white powder"]  # white photos are never in June-August
    assert summer.status == "ok" and summer.verdict == "fail"
    assert summer.share == 0.0 and summer.lift == 0.0

    weak = results["yellow fluff"]  # almost everything is Jan-Nov: lift barely above 1
    assert weak.verdict == "weak"
    assert weak.base_matching == 94  # all but the 8 December photos
    assert weak.lift == pytest.approx(102 / 94)


def test_lift_with_places_and_countries(engine, specs, tmp_path):
    results = lift_for(
        engine,
        tmp_path,
        [
            {"query": "white snow", "places": "paris"},
            {"query": "yellow sun", "countries": ["IN", "Germany"]},
            {"query": "green leaf", "months": [5], "places": ["goa", "berlin"]},
        ],
        k=40,
    )
    paris = results["white snow"]
    assert paris.expectation == "place~paris"
    assert paris.base_eligible == 79 and paris.base_matching == 50
    assert paris.status == "ok"
    country = results["yellow sun"]
    assert country.expectation == "country=IN|Germany"
    assert country.base_matching == 29  # 24 Goa + 5 Berlin
    both = results["green leaf"]
    assert both.expectation == "months=May and place~goa|berlin"
    labelled = [s for s in specs if s.month and s.place]
    assert both.base_eligible == len(labelled) == 77
    assert both.base_matching == sum(
        1 for s in labelled if s.month == 5 and s.place.city in ("Goa", "Berlin")
    )


def test_lift_insufficient_data_cases(engine, tmp_path):
    results = lift_for(
        engine,
        tmp_path,
        [
            {"query": "white nothing", "places": ["atlantis"]},
            {"query": "white all", "months": list(range(1, 13))},
            {"query": "white few", "months": [1]},
        ],
        k=3,
    )
    assert "no library photo" in results["white nothing"].reason
    assert results["white nothing"].status == "insufficient_data"
    assert results["white nothing"].lift is None
    assert "every library photo" in results["white all"].reason
    assert "top-3 hits" in results["white few"].reason
    assert results["white few"].verdict == ""


def test_lift_with_a_tiny_base_set_says_insufficient(tmp_path):
    store = new_store(tmp_path, "tiny.db")
    fill_store(store, build_specs()[:10])
    report = probe(
        make_engine(store),
        queries_file=write_queries(tmp_path, [{"query": "white snow", "months": [1]}]),
    )
    (r,) = report.lift
    assert (
        r.status == "insufficient_data"
        and "too few library photos" in r.reason
        and "10" not in r.reason
    )
    assert "insufficient data" in report.to_markdown()
    assert any("insufficient data" in w for w in report.warnings)
    store.close()


def test_lift_query_that_parses_into_filters_is_not_scored(engine, tmp_path):
    results = lift_for(engine, tmp_path, [{"query": "white snow March 2023", "months": [3]}])
    r = results["white snow March 2023"]
    assert r.status == "filtered" and r.share is None
    assert "filters" in r.reason


def test_lift_engine_error_is_contained(lib, tmp_path):
    store, _ = lib
    boom = StubEngine(store, lambda req: (_ for _ in ()).throw(ValueError("bad")))
    report = probe(boom, queries_file=write_queries(tmp_path, [{"query": "white", "months": [1]}]))
    assert report.lift[0].status == "error" and "ValueError" in report.lift[0].reason


def test_default_queries_ship_and_file_overrides_by_query(engine, tmp_path):
    report = run_probes(engine, sample=1, today=TODAY, clock=fake_clock())
    assert [r.query for r in report.lift] == [q.query for q in ev.DEFAULT_LIFT_QUERIES]
    assert len(report.lift) >= 8
    assert all(r.status in ("ok", "insufficient_data", "filtered") for r in report.lift)
    path = write_queries(tmp_path, [{"query": "A Christmas Tree", "months": [1]}])
    report = run_probes(engine, sample=1, today=TODAY, queries_file=path, clock=fake_clock())
    christmas = [r for r in report.lift if r.query.lower() == "a christmas tree"]
    assert len(christmas) == 1 and christmas[0].expectation == "months=Jan"
    assert report.params["lift_queries"] == len(ev.DEFAULT_LIFT_QUERIES)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"query": "x"}, "JSON list"),
        (["x"], "entry 1: expected an object"),
        ([{"query": ""}], "'query' must be a non-empty string"),
        ([{"months": [1]}], "'query' must be"),
        ([{"query": "x"}], "at least one of"),
        ([{"query": "x", "months": [1], "colour": 1}], "unknown key"),
        ([{"query": "x", "months": []}], "'months' must be a non-empty list"),
        ([{"query": "x", "months": "dec"}], "'months' must be a non-empty list"),
        ([{"query": "x", "months": [13]}], "not in 1-12"),
        ([{"query": "x", "months": [True]}], "must be 1-12 or month names"),
        ([{"query": "x", "months": ["smarch"]}], "must be 1-12 or month names"),
        ([{"query": "x", "months": [1.5]}], "must be 1-12 or month names"),
        ([{"query": "x", "places": []}], "'places' must be a non-empty"),
        ([{"query": "x", "places": [3]}], "'places' entries"),
        ([{"query": "x", "countries": ["  "]}], "'countries' entries"),
        ([{"query": "x", "months": [1]}, {"query": "y"}], "entry 2"),
    ],
)
def test_queries_file_validation(tmp_path, data, message):
    with pytest.raises(ProbeQueryError, match=message):
        load_lift_queries(write_queries(tmp_path, data))


def test_queries_file_unreadable_or_not_json(tmp_path):
    with pytest.raises(ProbeQueryError, match="Cannot read"):
        load_lift_queries(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(ProbeQueryError, match="not valid JSON"):
        load_lift_queries(bad)


def test_queries_file_roundtrip(tmp_path):
    (q,) = load_lift_queries(
        write_queries(
            tmp_path,
            [{"query": " x ", "months": ["Dec", 1], "places": "Paris", "countries": ["fr"]}],
        )
    )
    assert q == LiftQuery("x", frozenset({12, 1}), ("Paris",), ("fr",))


def test_run_probes_validates_arguments(engine):
    with pytest.raises(ValueError):
        run_probes(engine, sample=0)
    with pytest.raises(ValueError):
        run_probes(engine, k=0)


# --------------------------------------------------------------------------- feedback


def test_feedback_metrics(lib, engine):
    store, ids = lib
    for pid, ok in zip(ids[:6], (True, True, True, False, True, False)):
        store.record_feedback("secret beach query", pid, ok)
    for pid, ok in zip(ids[:4], (True, False, False, False)):
        store.record_feedback("another private query", pid, ok)
    store.record_feedback("another private query", ids[1], True)  # latest rating wins

    fb = probe(engine).feedback
    assert fb.status == "ok"
    assert (fb.queries, fb.rated, fb.relevant) == (2, 10, 6)
    assert fb.precision == pytest.approx(0.6)
    # (4/6 + 2/4) / 2 with ids[1] flipped to relevant in the second query
    assert fb.mean_precision == pytest.approx((4 / 6 + 2 / 4) / 2)
    assert 0 < fb.precision_ci_low < 0.6 < fb.precision_ci_high < 1
    assert [q.query for q in fb.per_query] == ["query 1", "query 2"]  # redacted by default
    md = probe(engine).to_markdown()
    assert "secret beach query" not in md and "6 rated" not in md and "10 rated hits" in md

    shown = probe(engine, show_paths=True).feedback
    assert [q.query for q in shown.per_query] == ["another private query", "secret beach query"]


def test_feedback_none_and_insufficient(lib, engine):
    store, ids = lib
    report = probe(engine)
    assert report.feedback.status == "none" and "No ratings yet" in report.to_markdown()
    store.record_feedback("q", ids[0], True)
    report = probe(engine)
    assert report.feedback.status == "insufficient_data"
    assert any("rate at least" in w for w in report.warnings)
    assert "insufficient data" in report.to_markdown()


# --------------------------------------------------------------------------- suggestions


def test_suggest_probe_queries_are_generic_and_deterministic(tmp_path):
    store = privacy_store(tmp_path)
    suggestions = suggest_probe_queries(store, 12)
    assert suggestions == suggest_probe_queries(store, 12)
    assert len(suggestions) == 12 and len(set(suggestions)) == 12
    joined = " ".join(suggestions)
    assert "Paris" in joined and "Goa" in joined and "Rome" in joined
    assert "Zagreb" not in joined and "Oslo" not in joined  # under the minimum bucket size
    assert "/lib" not in joined and ".jpg" not in joined
    months = [x for x in suggestions if x.startswith("photos from ")]
    assert months == [
        "photos from January 2023",
        "photos from February 2023",
        "photos from March 2023",
    ]  # most photos first
    assert len(suggest_probe_queries(store, 3)) == 3
    assert suggest_probe_queries(store, 0) == []
    store.close()


def test_suggest_probe_queries_on_an_empty_store(tmp_path):
    store = new_store(tmp_path, "empty.db")
    out = suggest_probe_queries(store, 4)
    assert len(out) == 4 and not any("photos from" in s for s in out)
    store.close()


def test_suggest_pairs_each_month_with_its_own_year(tmp_path):
    def spec(i, year, month):
        return Spec(
            f"/lib/s{year}{month}{i}.jpg", datetime(year, month, 1 + i), "exif", "day", None, "red"
        )

    specs = (
        [spec(i, 2020, 1) for i in range(6)]
        + [spec(i, 2021, 2) for i in range(5)]
        + [spec(i, 2022, 3) for i in range(4)]  # under the minimum: never suggested
    )
    store = new_store(tmp_path, "pairs.db")
    fill_store(store, specs)
    out = [q for q in suggest_probe_queries(store, 40) if q.startswith("photos from")]
    assert out == ["photos from January 2020", "photos from February 2021"]
    store.close()


# --------------------------------------------------------------------------- latency


def test_percentile():
    assert ev._percentile([], 50) is None
    assert ev._percentile([4.0], 95) == 4.0
    assert ev._percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert ev._percentile([4.0, 1.0, 3.0, 2.0], 100) == 4.0
    assert ev._percentile(list(map(float, range(1, 101))), 95) == pytest.approx(95.05)


def test_latency_uses_wall_clock_and_engine_stage_timings(lib, engine):
    store, _ = lib
    n = itertools.count(1)

    def fn(req):
        i = next(n)
        return replace(engine.search(req), timings_ms={"parse": float(i), "search": 2.0 * i})

    report = probe(StubEngine(store, fn), sample=500)
    lat = report.latency
    expected = sum(t.probes for t in report.self_retrieval)
    assert lat.samples == expected
    assert lat.p50_ms == pytest.approx(2.0) and lat.p95_ms == pytest.approx(2.0)
    assert set(lat.stages) == {"parse", "search"}
    assert lat.stages["parse"]["p50"] == pytest.approx((expected + 1) / 2)
    assert lat.stages["search"]["p95"] == pytest.approx(2.0 * (1 + (expected - 1) * 0.95))
    md = report.to_markdown()
    assert f"{expected} searches: p50 2.0 ms, p95 2.0 ms" in md


def test_latency_ignores_non_numeric_stage_timings(lib, engine):
    store, _ = lib
    stub = StubEngine(
        store,
        lambda req: replace(engine.search(req), timings_ms={"ok": 1.0, "bad": float("nan")}),
    )
    assert set(probe(stub, sample=2).latency.stages) == {"ok"}


# --------------------------------------------------------------------------- safety


def perfect_filters(request: SearchRequest) -> Filters:
    q = request.query
    if q == "photos since 2020":
        return Filters(date_range=DateRange(date(2020, 1, 1), request.today))
    if q == "photos before 2015":
        return Filters(date_range=DateRange(date(2014, 1, 1), date(2014, 12, 31)))
    if q == "after March 2022":
        return Filters(date_range=DateRange(date(2022, 4, 1), request.today))
    if q == "2019 or 2020":
        return Filters(date_range=DateRange(date(2019, 1, 1), date(2020, 12, 31)))
    if q == "summer 2019":
        return Filters(date_range=DateRange(date(2019, 6, 1), date(2019, 8, 31)))
    if q == "photos from 2019":
        return Filters(date_range=DateRange(date(2019, 1, 1), date(2019, 12, 31)))
    if q == "in Amherst":
        return Filters(place=PlaceFilter("Amherst", "Massachusetts", "United States"))
    return Filters()


def test_safety_with_a_careful_engine(lib):
    store, _ = lib
    asked = Unresolved("date", "x", "why", "q?")

    def fn(req):
        filters = perfect_filters(req)
        if req.query == "photos except 2019":
            if req.allow_clarification:
                return make_response(req.query, needs=[asked], unresolved=[asked])
            return make_response(req.query, dropped=[asked], unresolved=[asked])
        return make_response(req.query, parse_filters=filters)

    s = probe(StubEngine(store, fn), sample=1).safety
    battery = [c for c in ev.SAFETY_CASES if not c.control]
    n_cases = len(battery)
    assert n_cases == len(ev.SAFETY_CASES) - 2
    assert (s.cases, s.runs) == (n_cases, 2 * n_cases)
    assert s.unsafe_runs == 0 and s.errors == 0 and s.unsafe_cases == []
    assert s.safe_runs == 2 * n_cases
    assert s.reported_runs == 2  # 'except 2019', with and without clarification
    perfect_runs = 2 * sum(
        1
        for c in battery
        if not perfect_filters(SearchRequest(query=c.query, today=TODAY)).is_empty
    )
    assert s.quiet_runs == s.safe_runs - s.reported_runs - perfect_runs
    assert (s.controls, s.controls_ok, s.failed_controls) == (2, 2, [])


def test_safety_catches_a_wrong_filter_everywhere(lib):
    store, _ = lib
    wrong = Filters(
        date_range=DateRange(date(2000, 1, 1), date(2000, 1, 2)),
        place=PlaceFilter(city="Paris"),
        people=("Zorblax",),
    )
    s = probe(
        StubEngine(store, lambda req: make_response(req.query, parse_filters=wrong)), sample=1
    ).safety
    assert s.unsafe_runs == s.runs and s.safe_runs == 0
    assert len(s.unsafe_cases) == s.runs
    assert {u["why"] for u in s.unsafe_cases} == {"wrong filter applied"}
    # a wrong date filter fails the date control; a visible place filter passes the other
    assert s.controls_ok == 1 and s.failed_controls == ["control_year"]
    md = run_probes(
        StubEngine(store, lambda req: make_response(req.query, parse_filters=wrong)),
        sample=1,
        today=TODAY,
        include_default_queries=False,
    ).to_markdown()
    assert "UNSAFE operator_since: 'photos since 2020'" in md
    assert "control failed: control_year" in md


def test_safety_a_silent_engine_is_safe_but_fails_the_control(lib):
    store, _ = lib
    s = probe(StubEngine(store, lambda req: make_response(req.query)), sample=1).safety
    assert s.unsafe_runs == 0
    assert s.quiet_runs == s.runs and s.safe_runs == s.runs and s.reported_runs == 0
    assert s.controls_ok == 0


def test_safety_engine_errors_count_as_unsafe(lib):
    store, _ = lib

    def boom(req):
        raise RuntimeError("x")

    s = probe(StubEngine(store, boom), sample=1).safety
    assert s.errors == s.runs and s.unsafe_runs == s.runs
    assert {u["why"] for u in s.unsafe_cases} == {"error"}
    assert s.failed_controls == ["control_year", "control_namesake_place"]


def test_safety_requests_use_rules_mode_and_both_clarification_paths(lib):
    store, _ = lib
    stub = StubEngine(store, lambda req: make_response(req.query))
    probe(stub, sample=1)
    battery = [r for r in stub.requests if r.query in {c.query for c in ev.SAFETY_CASES}]
    assert battery and all(not r.use_llm for r in battery)
    assert {r.allow_clarification for r in battery} == {True, False}
    assert all(r.today == TODAY for r in battery)


def test_safety_with_the_real_engine_never_widens_silently(engine):
    s = probe(engine, sample=1).safety
    # Known gap in the rules parser, tracked separately: an impossible date ("31 February")
    # is read as the whole month. Everything else in the battery must be safe.
    assert {u["case"] for u in s.unsafe_cases} <= {"impossible_date"}
    assert s.controls_ok == 2


# --------------------------------------------------------------------------- output stability


def test_json_and_markdown_are_stable_and_complete(engine):
    a = probe(quiet(engine), sample=30, seed=3)
    b = probe(quiet(engine), sample=30, seed=3)
    assert a.to_json() == b.to_json() and a.to_markdown() == b.to_markdown()
    data = json.loads(a.to_json())
    assert data == a.to_dict()
    assert list(data) == sorted(data)  # sort_keys
    assert set(data) == {
        "params",
        "library",
        "self_retrieval",
        "lift",
        "place_roundtrip",
        "feedback",
        "latency",
        "safety",
        "warnings",
    }
    assert data["params"]["today"] == "2026-11-08"
    md = a.to_markdown()
    headings = [line for line in md.splitlines() if line.startswith("#")]
    assert headings == [
        "# Photo search evaluation",
        "## Library",
        "## Metadata self-retrieval",
        "## Lift probes (share of top-k meeting a metadata expectation)",
        "## Place label round-trip",
        "## Feedback (thumbs up/down)",
        "## Latency",
        "## Safety checks (rules mode)",
    ]
    assert md.endswith("\n")
    assert "Aggregates only" in md and "File paths are shown" not in md
    assert "| month_year |" in md and "100.0%" in md


def test_to_dict_rounds_floats_and_handles_dates():
    assert ev._jsonable({"x": 1 / 3, "d": date(2024, 1, 2), "t": (1, [2.0])}) == {
        "x": 0.333333,
        "d": "2024-01-02",
        "t": [1, [2.0]],
    }
    assert ev._pct(None) == "n/a" and ev._pct(0.5) == "50.0%"
    assert ev._num(None) == "n/a" and ev._num(2.345, 2) == "2.35"
    assert ev._wilson(0, 0) is None
    low, high = ev._wilson(10, 10)
    assert high == 1.0 and 0.7 < low < 1.0


# --------------------------------------------------------------------------- edge cases


def test_empty_index(tmp_path):
    store = new_store(tmp_path, "empty.db")
    report = probe(make_engine(store))
    assert report.library.photos == 0
    assert report.self_retrieval == []
    assert report.warnings and "empty" in report.warnings[0]
    md = report.to_markdown()
    assert "No probes ran" in md
    assert report.safety.runs == 2 * report.safety.cases
    assert json.loads(report.to_json())["library"]["photos"] == 0
    default = run_probes(make_engine(store), today=TODAY, clock=fake_clock())
    assert all(r.status == "insufficient_data" for r in default.lift if r.status != "filtered")
    store.close()


def test_one_photo_library(tmp_path):
    store = new_store(tmp_path, "one.db")
    fill_store(store, build_specs()[:1])
    report = probe(make_engine(store), sample=50)
    types = by_type(report)
    assert set(types) == {"month_year", "month_year_bare", "city", "city_month"}
    for t in types.values():
        assert t.success_rate == 1.0 and t.recall_at_k is None
        assert t.candidates_mean == 1.0
    # the lone city is a hidden bucket, so library-wide counts are rounded (1 -> 0)
    assert report.library.cities.buckets == {} and report.library.counts_rounded
    assert report.library.photos == 0
    rt = report.place_roundtrip
    assert (rt.labels, rt.labels_ok) == (1, 1)
    store.close()


def test_library_without_usable_metadata(tmp_path):
    store = new_store(tmp_path, "bare.db")
    spec = Spec("/lib/x.jpg", None, "none", "day", None, "red")
    fill_store(store, [spec])
    report = probe(make_engine(store))
    assert report.self_retrieval == []
    assert any("No photo has a usable date" in w for w in report.warnings)
    store.close()


def test_year_precision_photos_are_not_month_probes(tmp_path):
    store = new_store(tmp_path, "y.db")
    spec = Spec("/lib/y.jpg", datetime(2020, 1, 1), "folder", "year", PARIS, "red")
    fill_store(store, [spec])
    report = probe(make_engine(store))
    assert set(by_type(report)) == {"city"}
    store.close()


# --------------------------------------------------------------------------- CLI glue


def parse_args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(list(argv))


def test_add_arguments_defaults_and_flags():
    d = parse_args()
    assert (d.index, d.sample, d.k, d.seed, d.queries) == (None, 200, 24, 0, None)
    assert (d.json, d.markdown, d.show_paths, d.llm, d.no_llm) == (
        False,
        None,
        False,
        "auto",
        False,
    )
    a = parse_args(
        "--index",
        "i.db",
        "--sample",
        "5",
        "-k",
        "7",
        "--seed",
        "9",
        "--queries",
        "q.json",
        "--json",
        "--markdown",
        "o.md",
        "--show-paths",
        "--llm",
        "llama-cpp",
        "--no-llm",
    )
    assert (a.index, a.sample, a.k, a.seed, a.queries) == ("i.db", 5, 7, 9, "q.json")
    assert (a.json, a.markdown, a.show_paths, a.llm, a.no_llm) == (
        True,
        "o.md",
        True,
        "llama-cpp",
        True,
    )
    with pytest.raises(SystemExit):
        parse_args("--llm", "gpt")


class Factory:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.calls: list[tuple] = []

    def __call__(self, index, **kw):
        self.calls.append((index, kw))
        return self.engine


def test_run_prints_markdown_and_progress(engine, capsys, tmp_path):
    stub = wrap(engine, lambda req, resp: resp)
    factory = Factory(stub)
    out_md = tmp_path / "report.md"
    code = run(
        parse_args("--index", "x.db", "--sample", "30", "--markdown", str(out_md), "--no-llm"),
        engine_factory=factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert factory.calls == [("x.db", {"llm": "none", "use_llm": False})]
    assert stub.closed
    assert captured.out.startswith("# Photo search evaluation")
    assert out_md.read_text(encoding="utf-8") == captured.out
    assert "[eval]" in captured.err and "# Photo" not in captured.err
    assert all(not r.use_llm for r in stub.requests)
    assert "/lib/" not in captured.out


def test_run_json_output(engine, capsys):
    code = run(
        parse_args("--index", "x.db", "--json", "--llm", "mlx"), engine_factory=Factory(engine)
    )
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["params"]["use_llm"] is True and data["library"]["photos"] == 104


def test_run_show_paths(engine, capsys):
    def drop(req, resp):
        return replace(resp, applied=Filters())

    code = run(
        parse_args("--index", "x.db", "--sample", "5", "--show-paths"),
        engine_factory=Factory(wrap(engine, drop)),
    )
    assert code == 0 and "/lib/" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ("--sample", "0"),
        ("-k", "0"),
        ("--queries", "{missing}"),
        ("--llm", "mlx", "--no-llm"),
    ],
)
def test_run_bad_arguments_return_2(argv, capsys, tmp_path):
    argv = tuple(str(tmp_path / "nope.json") if a == "{missing}" else a for a in argv)
    factory = Factory(None)
    assert run(parse_args(*argv), engine_factory=factory) == 2
    assert factory.calls == []  # nothing was opened
    assert "error:" in capsys.readouterr().err


def test_run_rejects_an_unknown_llm_value(capsys):
    ns = argparse.Namespace(llm="gpt", sample=5, k=5)
    assert run(ns, engine_factory=Factory(None)) == 2
    assert "--llm must be one of" in capsys.readouterr().err


def test_run_invalid_queries_file_is_a_usage_error(capsys, tmp_path):
    bad = write_queries(tmp_path, [{"query": "x"}])
    assert run(parse_args("--queries", str(bad)), engine_factory=Factory(None)) == 2
    assert "at least one of" in capsys.readouterr().err


def test_run_returns_1_on_harness_failure(capsys, tmp_path):
    def factory(index, **kw):
        raise IndexNotFoundError("No photo index at nowhere.")

    assert run(parse_args("--index", "nowhere"), engine_factory=factory) == 1
    err = capsys.readouterr().err
    assert "evaluation failed" in err and "IndexNotFoundError" in err


def test_run_returns_1_when_the_report_cannot_be_written(engine, capsys, tmp_path):
    stub = wrap(engine, lambda req, resp: resp)
    args = parse_args("--index", "x", "--sample", "2", "--markdown", str(tmp_path / "no" / "r.md"))
    assert run(args, engine_factory=Factory(stub)) == 1
    assert stub.closed
    assert "evaluation failed" in capsys.readouterr().err


def test_run_with_a_queries_file_scores_it(engine, capsys, tmp_path):
    path = write_queries(tmp_path, [{"query": "white snow", "months": [12, 1, 2, 3]}])
    code = run(
        parse_args("--index", "x", "--sample", "2", "--json", "--queries", str(path), "-k", "10"),
        engine_factory=Factory(engine),
    )
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(r["query"] == "white snow" and r["status"] == "ok" for r in data["lift"])


def test_default_engine_factory_and_index_path(monkeypatch, engine, capsys):
    seen = {}

    def fake_open(index, **kw):
        seen["call"] = (index, kw)
        return wrap(engine, lambda req, resp: resp)

    monkeypatch.setattr(engine_module, "open_engine", fake_open)
    monkeypatch.setattr(engine_module, "default_index_path", lambda: Path("/default/index.sqlite"))
    assert run(parse_args("--sample", "2")) == 0
    assert seen["call"] == ("/default/index.sqlite", {"llm": "auto", "use_llm": True})
    assert run(parse_args("--sample", "2", "--index", "mine.db")) == 0
    assert seen["call"][0] == "mine.db"
    capsys.readouterr()
    # a bare namespace (as a hand-built caller might pass) works too
    assert run(argparse.Namespace(index="z.db")) == 0
    assert seen["call"][0] == "z.db"


def test_markdown_shows_filtered_and_error_lift_statuses(engine, tmp_path):
    path = write_queries(
        tmp_path,
        [
            {"query": "white snow March 2023", "months": [3]},
            {"query": "explode", "months": [3]},
        ],
    )

    def fn(req, resp):
        if req.query == "explode":
            raise RuntimeError("no")
        return resp

    md = probe(wrap(engine, fn), queries_file=path, sample=1).to_markdown()
    assert "| filtered |" in md and "| error |" in md
    assert "engine raised RuntimeError" in md


# --------------------------------------------------------------------------- verdicts


def test_verdict_uses_the_wilson_bound_not_a_fixed_lift():
    def verdict(hits, n, base):
        return ev._verdict(hits / n, ev._wilson(hits, n)[0], base)

    # base rate 0.1: a 2x lift is reachable and convincing
    assert verdict(8, 24, 0.1) == "pass"
    assert verdict(5, 24, 0.1) == "weak"  # 21% but the interval still reaches below 10%
    # base rate 0.5
    assert verdict(18, 24, 0.5) == "pass"
    assert verdict(13, 24, 0.5) == "weak"
    assert verdict(10, 24, 0.5) == "fail"
    # base rate 0.8: lift can never exceed 1.25, yet a clear win is still a pass
    assert verdict(24, 24, 0.8) == "pass"
    assert verdict(22, 24, 0.8) == "weak"
    assert verdict(24, 24, 0.95) == "weak"
    # a significant but tiny effect is not a pass
    assert verdict(900, 1000, 0.895) == "weak"
    assert verdict(960, 1000, 0.85) == "pass"


def test_small_hit_sets_cannot_pass_even_with_a_big_lift(engine, specs, lib, tmp_path):
    """5 eligible hits, 2 matching, base rate ~0.16: lift 2.5 but the interval is wide."""
    store, ids = lib
    jan_feb = [i for i, s in zip(ids, specs) if s.month in (1, 2)][:2]
    others = [i for i, s in zip(ids, specs) if s.month in (6, 7)][:3]
    photos = store.get_photos(jan_feb + others)

    def fn(req):
        return make_response(
            req.query, hits=[type(engine.search(req).hits[0])(p, 0.5) for p in photos]
        )

    path = write_queries(tmp_path, [{"query": "white snow", "months": [1, 2]}])
    report = probe(StubEngine(store, fn), queries_file=path, sample=1)
    (r,) = report.lift
    assert (r.hits_eligible, r.hits_matching) == (5, 2)
    assert r.share == pytest.approx(0.4)
    assert r.lift > 2.0 and r.share_ci_low == pytest.approx(0.118, abs=2e-3)
    assert r.verdict == "weak"


# --------------------------------------------------------------------------- safety extras


def test_since_2020_answered_with_just_2020_is_flagged_unsafe(lib):
    store, _ = lib

    def fn(req):
        if req.query == "photos since 2020":
            return make_response(
                req.query,
                parse_filters=Filters(date_range=DateRange(date(2020, 1, 1), date(2020, 12, 31))),
            )
        return make_response(req.query)

    s = probe(StubEngine(store, fn), sample=1).safety
    assert {u["case"] for u in s.unsafe_cases} == {"operator_since"}
    assert s.unsafe_runs == 2
    # ... while the genuinely open-ended reading is accepted
    ok = probe(
        StubEngine(
            store,
            lambda r: make_response(
                r.query,
                parse_filters=Filters(date_range=DateRange(date(2020, 1, 1), TODAY))
                if r.query == "photos since 2020"
                else None,
            ),
        ),
        sample=1,
    ).safety
    assert "operator_since" not in {u["case"] for u in ok.unsafe_cases}


def test_battery_has_no_atlantis_and_two_made_up_places():
    queries = {c.query for c in ev.SAFETY_CASES}
    assert "in Atlantis" not in queries
    assert {"photos in Xyzzyville", "in Qwertyland"} <= queries
    assert "in Amherst" in queries


def test_namesake_control_needs_a_visible_place_filter():
    check = ev._visible_place
    assert check(Filters(place=PlaceFilter("Amherst", "Massachusetts", "United States")), TODAY)
    assert check(Filters(place=PlaceFilter(near=(1.0, 2.0, 5.0))), TODAY)  # radius is shown
    assert not check(Filters(place=PlaceFilter()), TODAY)  # nothing a user could see
    assert not check(Filters(), TODAY)


# --------------------------------------------------------------------------- place round-trip


def test_place_roundtrip_exact(engine, specs):
    rt = probe(engine, sample=1).place_roundtrip
    assert (rt.labels, rt.labels_ok, rt.label_rate) == (3, 3, 1.0)
    assert (rt.photos, rt.photos_ok, rt.photo_rate) == (79, 79, 1.0)
    assert rt.failing.buckets == {} and rt.failing.other_buckets == 0
    md = probe(engine, sample=1).to_markdown()
    assert "labels resolving back to themselves: 3/3 (100.0%); photos covered: 79/79" in md
    assert "0 labels failed (names hidden; --show-paths)" in md


def test_place_roundtrip_reports_namesake_failures_by_count_then_by_name(engine):
    texas = PlaceFilter(city="Paris", region="Texas", country="United States", text="Paris")

    def to_texas(req, resp):
        place = resp.parse.filters.place
        if place is None or place.city != "Paris":
            return resp
        filters = replace(resp.parse.filters, place=texas)
        return replace(resp, parse=replace(resp.parse, filters=filters), applied=filters)

    quiet_report = probe(wrap(engine, to_texas), sample=1)
    rt = quiet_report.place_roundtrip
    assert (rt.labels, rt.labels_ok) == (3, 2)
    assert (rt.photos, rt.photos_ok) == (79, 29)
    assert rt.failing.buckets == {}  # names only with show_paths
    md = quiet_report.to_markdown()
    assert "1 labels failed (names hidden; --show-paths)" in md
    assert "Ile-de-France" not in md + quiet_report.to_json()

    loud = probe(wrap(engine, to_texas), sample=1, show_paths=True).place_roundtrip
    assert loud.failing.buckets == {"Paris, Ile-de-France, France": 50}
    md = probe(wrap(engine, to_texas), sample=1, show_paths=True).to_markdown()
    assert "failing labels (photos): Paris, Ile-de-France, France 50" in md


def test_place_roundtrip_unresolved_and_error_count_as_failures(engine):
    def nothing(req, resp):
        if req.query.startswith("photos in Goa") and "in " not in req.query[len("photos in Goa") :]:
            raise RuntimeError("boom")
        filters = replace(resp.parse.filters, place=None)
        return replace(resp, parse=replace(resp.parse, filters=filters))

    rt = probe(wrap(engine, nothing), sample=1).place_roundtrip
    assert rt.labels_ok == 0 and rt.photos_ok == 0 and rt.label_rate == 0.0


def test_place_roundtrip_with_no_city_labels(tmp_path):
    store = new_store(tmp_path, "nolabels.db")
    fill_store(store, [Spec("/lib/a.jpg", datetime(2023, 1, 1), "exif", "day", None, "red")])
    report = probe(make_engine(store))
    assert report.place_roundtrip.labels == 0 and report.place_roundtrip.label_rate is None
    assert "No photo has a city label." in report.to_markdown()
    store.close()


def test_place_roundtrip_progress_for_large_label_sets(tmp_path):
    store = new_store(tmp_path, "many.db")
    specs = [
        Spec(f"/lib/c{i}.jpg", datetime(2023, 1, 1), "exif", "day", PlaceLabel(f"City{i}"), "red")
        for i in range(51)
    ]
    fill_store(store, specs)
    messages: list[str] = []
    probe(make_engine(store), sample=1, progress=messages.append)
    assert "place labels: 50/51" in messages
    store.close()
