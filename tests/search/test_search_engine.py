"""Tests for facesorter.search.engine: search, clarification, wording helpers, assembly."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facesorter.search import engine as engine_module
from facesorter.search import testing
from facesorter.search.encoders import register_model, unregister_model
from facesorter.search.engine import (
    INDEX_ENV,
    Engine,
    LibraryGazetteer,
    default_index_path,
    describe_filters,
    explain,
    open_engine,
)
from facesorter.search.errors import (
    IndexMismatchError,
    IndexNotFoundError,
    ModelUnavailableError,
    SearchError,
)
from facesorter.search.parser import QueryParser
from facesorter.search.store import FaceRow, Store
from facesorter.search.testing import (
    COLORS,
    FakeImageEncoder,
    FakeTextEncoder,
    ScriptedLLM,
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
    SearchEngine,
    SearchRequest,
    SearchResponse,
    Unresolved,
)

TODAY = date(2026, 11, 8)
PARIS = PlaceLabel("Paris", "Île-de-France", "France", "FR")
KEY_WEST = PlaceLabel("Key West", "Florida", "United States", "US")
PARIS_GPS = (48.8566, 2.3522)
KEY_WEST_GPS = (24.5551, -81.78)


# ----------------------------------------------------------------------------- fixtures


class FakeGazetteer:
    """Knows Paris and Key West by name (and labels nothing)."""

    places = {
        "paris": PlaceFilter(city="Paris", region="Île-de-France", country="France", text="Paris"),
        "key west": PlaceFilter(
            city="Key West", region="Florida", country="United States", text="Key West"
        ),
    }

    def label_for(self, lat, lon):  # pragma: no cover - protocol filler
        return None

    def resolve(self, name):
        return self.places.get(name.lower())


def request(query: str = "", **kw) -> SearchRequest:
    kw.setdefault("today", TODAY)
    return SearchRequest(query=query, **kw)


# name -> (color, taken, place, people)
DATASET = {
    "paris_red_summer": ("red", datetime(2026, 7, 4, 12), PARIS, ()),
    "paris_blue_summer": ("blue", datetime(2026, 7, 10, 9), PARIS, ()),
    "paris_red_2024": ("red", datetime(2024, 7, 4, 12), PARIS, ()),
    "kw_red_summer": ("red", datetime(2026, 7, 5, 8), KEY_WEST, ()),
    "kw_green_summer": ("green", datetime(2026, 6, 15, 18), KEY_WEST, ("Bob",)),
    "undated_yellow": ("yellow", None, None, ()),
}


def photo_meta(path: str, taken: datetime | None) -> PhotoMeta:
    return PhotoMeta(
        path=path,
        size=1,
        mtime_ns=1,
        taken=taken,
        date_source="exif" if taken else "none",
    )


def fill_store(store: Store, dataset=DATASET) -> dict[str, int]:
    encoder = FakeImageEncoder()
    names = list(dataset)
    records = []
    for name in names:
        color, taken, place, _ = dataset[name]
        vec = encoder.encode_images([make_color_image(color)])[0]
        records.append((photo_meta(f"/photos/{name}.jpg", taken), place, vec))
    ids = dict(zip(names, store.upsert_photos(records)))
    cluster = 0
    people: dict[str, int] = {}
    for name in names:
        for person in dataset[name][3]:
            if person not in people:
                cluster += 1
                people[person] = cluster
            store.set_faces(ids[name], [FaceRow(people[person], 0, 0, 5, 5)])
    for person, cid in people.items():
        store.name_cluster(cid, person)
    return ids


@pytest.fixture
def store(tmp_path: Path):
    with Store.open(tmp_path / "index.sqlite", model_name="fake-color-16", dim=16) as s:
        yield s


@pytest.fixture
def ids(store) -> dict[str, int]:
    return fill_store(store)


def make_engine(store: Store, llm=None, gazetteer="default") -> Engine:
    gaz = FakeGazetteer() if gazetteer == "default" else gazetteer
    return Engine(store, FakeTextEncoder(), QueryParser(llm, gaz, store), gaz)


@pytest.fixture
def engine(store, ids) -> Engine:
    return make_engine(store)


def names_of(response: SearchResponse) -> list[str]:
    return [Path(h.photo.meta.path).stem for h in response.hits]


@pytest.fixture
def fake_model():
    register_model(
        testing.FAKE_MODEL_NAME,
        lambda device: FakeImageEncoder(),
        lambda quantization: FakeTextEncoder(),
        dim=testing.FAKE_DIM,
    )
    yield testing.FAKE_MODEL_NAME
    unregister_model(testing.FAKE_MODEL_NAME)


# ----------------------------------------------------------------------------- search


class TestSearch:
    def test_satisfies_the_protocol(self, engine):
        assert isinstance(engine, SearchEngine)

    def test_filters_and_visual_text(self, engine):
        res = engine.search(request("red cars in Paris last summer"))
        assert names_of(res)[0] == "paris_red_summer"
        assert names_of(res) == ["paris_red_summer", "paris_blue_summer"]
        assert res.candidates == 2
        assert res.applied.place.city == "Paris"
        assert res.applied.date_range == DateRange(date(2026, 6, 1), date(2026, 8, 31))
        assert res.needs_clarification == () and res.dropped == ()
        assert res.parse.text == "red cars"
        scores = [h.score for h in res.hits]
        assert scores == sorted(scores, reverse=True) and scores[0] > scores[1]
        assert all(isinstance(h.score, float) for h in res.hits)
        assert set(res.timings_ms) == {"parse", "encode", "search"}
        assert all(v >= 0 for v in res.timings_ms.values())

    def test_hits_carry_the_stored_photo(self, engine, store, ids):
        res = engine.search(request("green"))
        hit = res.hits[0]
        assert hit.photo.id == ids["kw_green_summer"]
        assert hit.photo.meta.path == "/photos/kw_green_summer.jpg"
        assert hit.photo.place == KEY_WEST
        assert hit.photo == store.get_photos([hit.photo.id])[0]

    def test_pure_visual_query_ranks_the_whole_library(self, engine):
        res = engine.search(request("blue"))
        assert res.candidates == len(DATASET) and len(res.hits) == len(DATASET)
        assert names_of(res)[0] == "paris_blue_summer"
        assert res.applied == Filters()

    def test_people_filter_from_the_known_names(self, engine):
        res = engine.search(request("red with Bob"))
        assert res.applied.people == ("Bob",)
        assert names_of(res) == ["kw_green_summer"]

    def test_llm_extraction(self, store, ids):
        llm = ScriptedLLM(
            {
                "red cars in Paris last summer": json.dumps(
                    {
                        "text": "red cars",
                        "place": "Paris",
                        "people": [],
                        "date_phrase": "last summer",
                    }
                )
            }
        )
        eng = make_engine(store, llm=llm)
        res = eng.search(request("red cars in Paris last summer"))
        assert res.parse.parsed.source == "llm" and len(llm.calls) == 1
        assert names_of(res) == ["paris_red_summer", "paris_blue_summer"]

    def test_use_llm_false_never_calls_the_model(self, store, ids):
        llm = ScriptedLLM(default='{"text": "WRONG", "place": null, "people": []}')
        res = make_engine(store, llm=llm).search(request("red in Paris", use_llm=False))
        assert llm.calls == [] and res.parse.parsed.source == "rules"
        assert res.applied.place.city == "Paris"

    def test_explicit_filters_override_the_query(self, engine):
        flags = Filters(place=PlaceFilter(city="Key West"))
        res = engine.search(request("red cars in Paris", filters=flags))
        assert res.applied.place.city == "Key West"
        assert names_of(res)[0] == "kw_red_summer"
        assert any("explicit place filter" in n for n in res.parse.notes)

    def test_flags_alone_browse(self, engine):
        flags = Filters(date_range=DateRange(date(2026, 7, 1), date(2026, 7, 31)))
        res = engine.search(request("", filters=flags))
        assert names_of(res) == ["paris_blue_summer", "kw_red_summer", "paris_red_summer"]
        assert res.parse.parsed.source == "flags"

    def test_k_limits_hits_but_not_candidates(self, engine):
        res = engine.search(request("red", k=2))
        assert len(res.hits) == 2 and res.candidates == len(DATASET)
        assert len(engine.search(request("red", k=1000)).hits) == len(DATASET)

    def test_k_accepts_numpy_integers(self, engine):
        assert len(engine.search(request("red", k=np.int64(3))).hits) == 3

    @pytest.mark.parametrize("k", [0, -1, None, "3", 2.5])
    def test_bad_k_is_a_value_error(self, engine, k):
        with pytest.raises(ValueError, match="k must be"):
            engine.search(request("red", k=k))

    def test_hit_for_a_photo_removed_meanwhile_is_skipped(self, engine, store, ids, monkeypatch):
        real = store.top_k

        def racy(*args, **kwargs):
            scored, n = real(*args, **kwargs)
            store.delete_paths(["/photos/paris_red_summer.jpg"])
            return scored, n

        monkeypatch.setattr(store, "top_k", racy)
        res = engine.search(request("red cars in Paris last summer"))
        assert names_of(res) == ["paris_blue_summer"]

    def test_text_is_clipped_before_encoding(self, store, ids):
        seen: list[str] = []

        class Spy(FakeTextEncoder):
            def encode_text(self, texts):
                seen.extend(texts)
                return super().encode_text(texts)

        eng = Engine(store, Spy(), QueryParser(None, FakeGazetteer(), store))
        eng.search(request("red " + "word " * 2000))
        assert len(seen) == 1 and len(seen[0]) == engine_module._MAX_TEXT_CHARS


class TestBrowse:
    def test_no_visual_text_lists_newest_first_with_zero_scores(self, engine):
        res = engine.search(request("in Paris"))
        assert res.parse.text == ""
        assert names_of(res) == ["paris_blue_summer", "paris_red_summer", "paris_red_2024"]
        assert [h.score for h in res.hits] == [0.0, 0.0, 0.0]
        assert res.candidates == 3
        assert res.timings_ms["encode"] == 0.0

    def test_empty_query_browses_the_whole_index_newest_first(self, engine):
        res = engine.search(request(""))
        assert names_of(res)[0] == "paris_blue_summer"
        assert names_of(res)[-1] == "undated_yellow"  # undated last
        assert res.candidates == len(DATASET)
        assert engine.search(request("   \t ")).candidates == len(DATASET)

    def test_browse_does_not_load_the_text_encoder(self, store, ids):
        class Boom(FakeTextEncoder):
            def encode_text(self, texts):
                raise AssertionError("browse must not embed text")

        eng = Engine(store, Boom(), QueryParser(None, FakeGazetteer(), store))
        assert eng.search(request("in Key West")).hits

    def test_browse_respects_k(self, engine):
        assert len(engine.search(request("", k=2)).hits) == 2

    def test_empty_index(self, store):
        res = make_engine(store).search(request("red"))
        assert res.hits == () and res.candidates == 0
        assert res.applied == Filters()
        assert explain(res) == ["The index is empty"]
        browse = make_engine(store).search(request(""))
        assert browse.hits == () and browse.candidates == 0


class TestZeroCandidates:
    def test_filters_matching_nothing(self, engine):
        res = engine.search(request("red cars in Paris in 2019"))
        assert res.hits == () and res.candidates == 0
        assert explain(res) == [
            "No photos match: dates 2019-01-01 to 2019-12-31 · place Paris, Île-de-France, France",
            "Your library covers photos from 2024-07-04 to 2026-07-10.",
        ]

    def test_filters_matching_nothing_without_visual_text(self, engine):
        res = engine.search(request("Bob in Paris"))
        assert res.hits == () and res.candidates == 0
        assert explain(res)[-1] == (
            "No photos match: place Paris, Île-de-France, France · with Bob"
        )


class TestClarification:
    QUERY = "green dinner Diwali 2023"

    def test_first_call_asks_and_returns_no_hits(self, engine):
        res = engine.search(request(self.QUERY))
        assert res.hits == () and res.candidates == 0
        (u,) = res.needs_clarification
        assert (u.kind, u.phrase) == ("date", "Diwali 2023")
        assert res.dropped == ()
        assert res.applied.date_range is None
        assert explain(res) == [u.question]
        assert set(res.timings_ms) == {"parse", "encode", "search"}

    def test_reworded_answer_resolves(self, engine):
        res = engine.search(request(self.QUERY, clarifications={"Diwali 2023": "June 2026"}))
        assert res.needs_clarification == () and res.dropped == ()
        assert res.applied.date_range == DateRange(date(2026, 6, 1), date(2026, 6, 30))
        assert names_of(res) == ["kw_green_summer"]

    def test_still_not_understood_is_dropped_and_searched_without(self, engine):
        res = engine.search(request(self.QUERY, clarifications={"Diwali 2023": "the festival"}))
        assert res.needs_clarification == ()
        (u,) = res.dropped
        assert u.phrase == "Diwali 2023"
        assert res.applied.date_range is None
        assert names_of(res)[0] == "kw_green_summer" and res.candidates == len(DATASET)
        assert (
            'I couldn\'t understand "Diwali 2023" as a date, so I searched without it.'
            in explain(res)
        )

    def test_skipping_with_an_empty_answer_reports_nothing(self, engine):
        res = engine.search(request(self.QUERY, clarifications={"Diwali 2023": ""}))
        assert res.dropped == () and res.needs_clarification == ()
        assert names_of(res)[0] == "kw_green_summer"

    def test_asked_check_ignores_case_and_spacing(self, engine):
        res = engine.search(request(self.QUERY, clarifications={"  diwali   2023": "huh"}))
        assert res.needs_clarification == () and len(res.dropped) == 1

    def test_no_clarification_allowed_drops_immediately(self, engine):
        res = engine.search(request(self.QUERY, allow_clarification=False))
        assert res.needs_clarification == ()
        assert [u.phrase for u in res.dropped] == ["Diwali 2023"]
        assert names_of(res)[0] == "kw_green_summer"

    def test_full_round_trip_with_the_llm_parsing_once(self, store, ids):
        llm = ScriptedLLM(
            {self.QUERY: json.dumps({"date_phrase": "Diwali 2023", "place": None, "people": []})}
        )
        eng = make_engine(store, llm=llm)
        first = eng.search(request(self.QUERY))
        (u,) = first.needs_clarification
        second = eng.search(request(self.QUERY, clarifications={u.phrase: "June 2026"}))
        assert names_of(second) == ["kw_green_summer"]
        assert len(llm.calls) == 1  # the parser reuses the model's reply

    def test_one_phrase_asked_the_other_still_unasked(self, store, ids):
        q = "red in Atlantis with Zed"
        llm = ScriptedLLM({q: json.dumps({"place": "Atlantis", "people": ["Zed"]})})
        engine = make_engine(store, llm=llm)
        first = engine.search(request(q))
        assert {u.phrase for u in first.needs_clarification} == {"Atlantis", "Zed"}
        second = engine.search(request(q, clarifications={"Zed": "nobody"}))
        assert [u.phrase for u in second.needs_clarification] == ["Atlantis"]
        assert second.hits == () and second.dropped == ()
        third = engine.search(request(q, clarifications={"Zed": "nobody", "Atlantis": "nowhere"}))
        assert third.needs_clarification == ()
        assert {u.phrase for u in third.dropped} == {"Atlantis", "Zed"}
        assert len(third.hits) == len(DATASET)

    def test_a_flag_for_the_field_means_nothing_to_ask(self, engine):
        flags = Filters(date_range=DateRange(date(2026, 6, 1), date(2026, 6, 30)))
        res = engine.search(request(self.QUERY, filters=flags))
        assert res.needs_clarification == () and res.dropped == ()
        assert names_of(res) == ["kw_green_summer"]


class TestRobustness:
    ODD = [
        "",
        " ",
        "\x00\x01",
        "!!!",
        "???? in ????",
        "🙂 in Paris",
        "العربية في Paris",
        "'; DROP TABLE photos; --",
        "a" * 50_000,
        "in in in in",
        "last last last summer summer",
        "red " * 3000,
        "Paris " * 500,
        "Bob and Bob and Bob",
        "2026-13-45",
        "since 2020 not in summer",
    ]

    @pytest.mark.parametrize("query", ODD, ids=[f"odd{i}" for i in range(len(ODD))])
    def test_odd_queries_never_raise(self, engine, query):
        for allow in (True, False):
            res = engine.search(request(query, allow_clarification=allow))
            assert isinstance(res, SearchResponse)
            assert len(res.hits) <= 24
            explain(res)  # must not raise either

    def test_llm_garbage_never_raises(self, store, ids):
        eng = make_engine(store, llm=ScriptedLLM(default="not json at all"))
        res = eng.search(request("red cars in Paris"))
        assert res.parse.parsed.source == "rules"
        assert any("not usable" in n for n in res.parse.notes)
        assert names_of(res)[:2] == ["paris_red_summer", "paris_red_2024"]

    def test_none_query_is_treated_as_empty(self, engine):
        res = engine.search(SearchRequest(query=None, today=TODAY))  # type: ignore[arg-type]
        assert res.candidates == len(DATASET)

    def test_closed_index_raises_a_search_error(self, store, ids):
        eng = make_engine(store)
        eng.close()
        with pytest.raises(SearchError, match="closed"):
            eng.search(request("red"))
        with pytest.raises(SearchError, match="closed"):
            eng.search(request(""))

    def test_failing_text_encoder_propagates(self, store, ids):
        class Broken(FakeTextEncoder):
            def encode_text(self, texts):
                raise RuntimeError("model crashed")

        eng = Engine(store, Broken(), QueryParser(None, FakeGazetteer(), store))
        with pytest.raises(RuntimeError, match="crashed"):
            eng.search(request("red"))


class TestAgainstBruteForce:
    def test_filters_restrict_exactly_like_a_python_scan(self, tmp_path):
        rng = np.random.default_rng(7)
        colors = list(COLORS)
        places = [PARIS, KEY_WEST, None]
        with Store.open(tmp_path / "bf.sqlite", model_name="fake-color-16", dim=16) as store:
            rows = {}
            records = []
            enc = FakeImageEncoder()
            for i in range(150):
                color = colors[int(rng.integers(len(colors)))]
                place = places[int(rng.integers(3))]
                taken = (
                    datetime(2024 + int(rng.integers(3)), 1 + int(rng.integers(12)), 15)
                    if rng.random() > 0.1
                    else None
                )
                path = f"/p/{i:03d}.jpg"
                rows[path] = (color, place, taken)
                records.append(
                    (
                        photo_meta(path, taken),
                        place,
                        enc.encode_images([make_color_image(color)])[0],
                    )
                )
            store.upsert_photos(records)
            eng = make_engine(store)

            def brute(text_color, city, rng_dates):
                pool = []
                for path, (_color, place, taken) in rows.items():
                    if city and (place is None or place.city != city):
                        continue
                    if rng_dates:
                        if taken is None or not (rng_dates[0] <= taken.date() <= rng_dates[1]):
                            continue
                    pool.append(path)
                vec = FakeTextEncoder().encode_text([text_color])[0]
                scored = [
                    (round(float(enc.encode_images([make_color_image(rows[p][0])])[0] @ vec), 6), p)
                    for p in pool
                ]
                scored.sort(key=lambda t: (-t[0], t[1]))
                return [p for _, p in scored], len(pool)

            cases = [
                ("red", None, None, "red"),
                ("blue in Paris", "Paris", None, "blue"),
                (
                    "green in Key West in 2025",
                    "Key West",
                    (date(2025, 1, 1), date(2025, 12, 31)),
                    "green",
                ),
                ("white in March 2026", None, (date(2026, 3, 1), date(2026, 3, 31)), "white"),
                (
                    "yellow in Paris in 2024",
                    "Paris",
                    (date(2024, 1, 1), date(2024, 12, 31)),
                    "yellow",
                ),
            ]
            for query, city, dates, color in cases:
                expected, n = brute(color, city, dates)
                res = eng.search(request(query, k=15))
                assert res.candidates == n, query
                got = [h.photo.meta.path for h in res.hits]
                assert got == expected[:15], query
                assert n > 0


class TestThreads:
    def test_concurrent_searches_match_serial_results(self, store, ids):
        llm = ScriptedLLM(default='{"text": "", "place": null, "people": [], "date_phrase": null}')
        eng = make_engine(store, llm=llm)
        queries = [
            "red cars in Paris last summer",
            "blue",
            "green with Bob",
            "in Key West",
            "",
        ] * 12
        serial = [names_of(eng.search(request(q))) for q in queries]
        with ThreadPoolExecutor(8) as pool:
            parallel = list(pool.map(lambda q: names_of(eng.search(request(q))), queries))
        assert parallel == serial

    def test_parsing_is_serialised(self, store, ids):
        inside = 0
        worst = 0
        lock = threading.Lock()

        class SlowParser(QueryParser):
            def parse(self, req, today=None):
                nonlocal inside, worst
                with lock:
                    inside += 1
                    worst = max(worst, inside)
                try:
                    return super().parse(req, today)
                finally:
                    with lock:
                        inside -= 1

        eng = Engine(store, FakeTextEncoder(), SlowParser(None, FakeGazetteer(), store))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda q: eng.search(request(q)), ["red cars in Paris"] * 24))
        assert worst == 1


class TestConstruction:
    def test_text_encoder_must_match_the_index(self, store):
        class Other(FakeTextEncoder):
            name = "other-model"

        with pytest.raises(IndexMismatchError, match="other-model"):
            Engine(store, Other(), QueryParser(None, None, store))

        class WrongDim(FakeTextEncoder):
            dim = 4

        with pytest.raises(IndexMismatchError):
            Engine(store, WrongDim(), QueryParser(None, None, store))

    def test_attributes_and_context_manager(self, store, ids):
        parser = QueryParser(None, None, store)
        gaz = FakeGazetteer()
        with Engine(store, FakeTextEncoder(), parser, gaz) as eng:
            assert eng.store is store and eng.parser is parser
            assert isinstance(eng.gazetteer, LibraryGazetteer) and eng.gazetteer.inner is gaz
            assert parser._gazetteer is None  # the parser had its own (no) gazetteer
        with pytest.raises(SearchError, match="closed"):
            store.count()
        eng.close()  # twice is fine

    def test_gazetteer_is_optional(self, store, ids):
        eng = Engine(store, FakeTextEncoder(), QueryParser(None, None, store))
        res = eng.search(request("red in Paris"))
        assert res.applied.place is None and len(res.hits) == len(DATASET)


# ----------------------------------------------------------------------------- wording


def parse_result(text: str = "red", notes: tuple[str, ...] = ()) -> ParseResult:
    return ParseResult(ParsedQuery(raw=text, text=text), Filters(), notes=notes)


def response(**kw) -> SearchResponse:
    base = dict(hits=(), parse=parse_result(), applied=Filters(), candidates=5)
    base.update(kw)
    return SearchResponse(**base)


class TestDescribeFilters:
    def test_empty(self):
        assert describe_filters(Filters()) == ""

    def test_full_example(self):
        f = Filters(
            date_range=DateRange(date(2023, 6, 1), date(2023, 8, 31)),
            place=PlaceFilter(
                city="Key West", region="Florida", country="United States", text="Key West"
            ),
            people=("Bob",),
        )
        assert describe_filters(f) == (
            "dates 2023-06-01 to 2023-08-31 · place Key West, Florida, United States · with Bob"
        )

    def test_single_day(self):
        f = Filters(date_range=DateRange(date(2023, 6, 1), date(2023, 6, 1)))
        assert describe_filters(f) == "date 2023-06-01"

    @pytest.mark.parametrize(
        ("place", "expected"),
        [
            (PlaceFilter(country="France"), "place France"),
            (
                PlaceFilter(region="Florida", country="United States"),
                "place Florida, United States",
            ),
            (PlaceFilter(city="Paris", text="paris"), "place Paris"),
            (
                PlaceFilter(near=(48.8584, 2.2945, 5.0), text="Eiffel Tower"),
                "place within 5 km of 48.8584, 2.2945",
            ),
            (PlaceFilter(near=(1.0, 2.0, 12.5)), "place within 12.5 km of 1.0000, 2.0000"),
            (PlaceFilter(text="somewhere"), "place somewhere"),
            (PlaceFilter(), ""),
        ],
    )
    def test_places(self, place, expected):
        assert describe_filters(Filters(place=place)) == expected

    @pytest.mark.parametrize(
        ("people", "expected"),
        [
            (("Bob",), "with Bob"),
            (("Bob", "Alice"), "with Bob and Alice"),
            (("Bob", "Alice", "Mom"), "with Bob, Alice and Mom"),
        ],
    )
    def test_people(self, people, expected):
        assert describe_filters(Filters(people=people)) == expected

    def test_people_only_and_place_only_have_no_separators(self):
        assert " · " not in describe_filters(Filters(people=("Bob",)))
        assert describe_filters(Filters(place=PlaceFilter(city="Rome"))) == "place Rome"


class TestExplain:
    def test_nothing_to_say(self):
        assert explain(response()) == []

    def test_parse_notes_come_first(self):
        res = response(parse=parse_result(notes=("note one", "note two")))
        assert explain(res) == ["note one", "note two"]

    def test_questions_for_clarification(self):
        u = Unresolved("date", "Easter 2024", "unknown", 'I did not understand "Easter 2024".')
        res = response(needs_clarification=(u,), candidates=0, parse=parse_result(notes=("n",)))
        assert explain(res) == ["n", 'I did not understand "Easter 2024".']

    @pytest.mark.parametrize(
        ("kind", "noun"),
        [("date", "a date"), ("place", "a place"), ("person", "a person's name")],
    )
    def test_dropped_filters(self, kind, noun):
        u = Unresolved(kind, "X y", "why", "q?")
        assert explain(response(dropped=(u,))) == [
            f'I couldn\'t understand "X y" as {noun}, so I searched without it.'
        ]

    def test_unknown_kind_still_has_wording(self):
        u = Unresolved("mood", "sad", "why", "q?")  # type: ignore[arg-type]
        assert explain(response(dropped=(u,))) == [
            'I couldn\'t understand "sad" as a filter, so I searched without it.'
        ]

    def test_zero_candidates(self):
        applied = Filters(place=PlaceFilter(city="Rome"), people=("Bob",))
        assert explain(response(candidates=0, applied=applied)) == [
            "No photos match: place Rome · with Bob"
        ]

    def test_zero_candidates_without_filters_means_empty_index(self):
        assert explain(response(candidates=0)) == ["The index is empty"]

    def test_browse_notice(self):
        assert explain(response(parse=parse_result(text=""))) == [
            "No visual description, showing newest first"
        ]
        assert explain(response(parse=parse_result(text="  "))) == [
            "No visual description, showing newest first"
        ]

    def test_zero_candidates_replaces_the_browse_notice(self):
        res = response(parse=parse_result(text=""), candidates=0, applied=Filters(people=("Bob",)))
        assert explain(res) == ["No photos match: with Bob"]

    def test_order_notes_dropped_then_outcome(self):
        u = Unresolved("place", "Atlantis", "why", "q?")
        res = response(
            parse=parse_result(text="", notes=("a note",)),
            dropped=(u,),
        )
        assert explain(res) == [
            "a note",
            'I couldn\'t understand "Atlantis" as a place, so I searched without it.',
            "No visual description, showing newest first",
        ]


# ----------------------------------------------------------------------------- assembly


class TestDefaultIndexPath:
    def test_default_location_creates_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv(INDEX_ENV, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        path = default_index_path()
        assert path == tmp_path / ".facesorter" / "index.sqlite"
        assert not (tmp_path / ".facesorter").exists()

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv(INDEX_ENV, str(tmp_path / "mine.sqlite"))
        assert default_index_path() == tmp_path / "mine.sqlite"

    def test_env_override_expands_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv(INDEX_ENV, "~/idx/x.sqlite")
        assert default_index_path() == tmp_path / "idx" / "x.sqlite"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_env_is_ignored(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv(INDEX_ENV, value)
        assert default_index_path() == tmp_path / ".facesorter" / "index.sqlite"

    def test_returns_a_path(self):
        assert isinstance(default_index_path(), Path)


class TestOpenEngine:
    @pytest.fixture(autouse=True)
    def _cheap_gazetteer(self, monkeypatch):
        from facesorter.search import geo

        monkeypatch.setattr(geo, "build_gazetteer", lambda *a, **k: FakeGazetteer())

    @pytest.fixture
    def index_file(self, tmp_path, fake_model) -> Path:
        path = tmp_path / "index.sqlite"
        with Store.open(path, model_name=fake_model, dim=16) as s:
            fill_store(s)
        return path

    def test_rules_mode(self, index_file):
        with open_engine(index_file, use_llm=False) as eng:
            assert isinstance(eng, Engine)
            res = eng.search(request("red cars in Paris last summer"))
            assert names_of(res)[0] == "paris_red_summer"
            assert eng.parser._llm is None

    def test_accepts_str_and_tilde(self, index_file, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        with open_engine("~/index.sqlite", use_llm=False) as eng:
            assert eng.store.count() == len(DATASET)
        with open_engine(str(index_file), use_llm=False):
            pass

    def test_llm_arguments_are_passed_to_load_llm(self, index_file, monkeypatch):
        import facesorter.search.llm as llm_module

        scripted = ScriptedLLM(default='{"text": "red", "place": null, "people": []}')
        calls = []

        def fake_load(backend, model):
            calls.append((backend, model))
            return scripted

        monkeypatch.setattr(llm_module, "load_llm", fake_load)
        with open_engine(index_file, llm="mlx", llm_model="some/model") as eng:
            assert eng.parser._llm is scripted
            eng.search(request("red"))
        assert calls == [("mlx", "some/model")] and len(scripted.calls) == 1

    def test_use_llm_false_does_not_even_load_one(self, index_file, monkeypatch):
        import facesorter.search.llm as llm_module

        monkeypatch.setattr(llm_module, "load_llm", lambda *a: pytest.fail("must not load"))
        with open_engine(index_file, use_llm=False):
            pass

    def test_auto_without_a_runtime_is_rules_only(self, index_file, monkeypatch):
        monkeypatch.setenv("FACESORTER_LLM_BACKEND", "none")
        with open_engine(index_file) as eng:
            assert eng.parser._llm is None

    def test_people_come_from_the_index(self, index_file):
        with open_engine(index_file, use_llm=False) as eng:
            res = eng.search(request("green with Bob"))
            assert res.applied.people == ("Bob",) and names_of(res) == ["kw_green_summer"]

    def test_text_quantization_is_forwarded(self, index_file, fake_model):
        seen = []
        unregister_model(fake_model)
        register_model(
            fake_model,
            lambda device: FakeImageEncoder(),
            lambda q: seen.append(q) or FakeTextEncoder(),
            dim=16,
        )
        with open_engine(index_file, use_llm=False, text_quantization="int8"):
            pass
        assert seen == ["int8"]

    def test_missing_index(self, tmp_path):
        path = tmp_path / "sub" / "missing.sqlite"
        with pytest.raises(IndexNotFoundError, match="facesorter index"):
            open_engine(path)
        assert not path.exists() and not path.parent.exists()

    def test_not_an_index(self, tmp_path):
        junk = tmp_path / "junk.sqlite"
        junk.write_bytes(b"this is not a database" * 100)
        with pytest.raises(SearchError):
            open_engine(junk)

    def test_unknown_model_in_the_index_is_unavailable_and_closes_the_store(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "index.sqlite"
        Store.open(path, model_name="ghost-model", dim=8).close()
        closed = []
        real_close = Store.close
        monkeypatch.setattr(Store, "close", lambda self: (closed.append(self), real_close(self)))
        with pytest.raises(ModelUnavailableError, match="ghost-model"):
            open_engine(path, use_llm=False)
        assert len(closed) == 1

    def test_unknown_llm_backend_closes_the_store(self, index_file, monkeypatch):
        closed = []
        real_close = Store.close
        monkeypatch.setattr(Store, "close", lambda self: (closed.append(self), real_close(self)))
        with pytest.raises(ModelUnavailableError, match="bogus"):
            open_engine(index_file, llm="bogus")
        assert len(closed) == 1

    def test_gazetteer_failure_closes_the_store(self, index_file, monkeypatch):
        from facesorter.search import geo

        def boom(*a, **k):
            raise ModelUnavailableError("no place data")

        monkeypatch.setattr(geo, "build_gazetteer", boom)
        with pytest.raises(ModelUnavailableError, match="place data"):
            open_engine(index_file, use_llm=False)


# ----------------------------------------------------------------------------- end to end


def exif(taken: str, gps: tuple[float, float] | None = None) -> Image.Exif:
    ex = Image.Exif()
    ex.get_ifd(0x8769)[0x9003] = taken

    def dms(v: float) -> tuple[float, float, float]:
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        return float(d), float(m), round((v - d - m / 60) * 3600, 4)

    if gps:
        lat, lon = gps
        ex.get_ifd(0x8825).update(
            {1: "N" if lat >= 0 else "S", 2: dms(lat), 3: "E" if lon >= 0 else "W", 4: dms(lon)}
        )
    return ex


def save(path: Path, color: str, taken: str | None = None, gps=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if taken:
        make_color_image(color).save(path, exif=exif(taken, gps))
    else:
        make_color_image(color).save(path)
    return path


@pytest.fixture(scope="module")
def e2e_library(tmp_path_factory):
    """A synthetic library: year / month / place folders, EXIF dates and GPS, one undated."""
    root = tmp_path_factory.mktemp("e2e") / "Pictures"
    save(
        root / "2026" / "07 - July" / "Paris" / "car_red.jpg",
        "red",
        "2026:07:04 10:00:00",
        PARIS_GPS,
    )
    save(
        root / "2026" / "07 - July" / "Paris" / "car_blue.jpg",
        "blue",
        "2026:07:10 11:00:00",
        PARIS_GPS,
    )
    save(
        root / "2024" / "07 - July" / "Paris" / "old_red.jpg",
        "red",
        "2024:07:04 10:00:00",
        PARIS_GPS,
    )
    save(
        root / "2026" / "06 - June" / "Key West" / "beach_red.jpg",
        "red",
        "2026:06:20 09:00:00",
        KEY_WEST_GPS,
    )
    save(
        root / "2026" / "06 - June" / "Key West" / "boat_green.jpg",
        "green",
        "2026:06:21 09:00:00",
        KEY_WEST_GPS,
    )
    save(root / "2022" / "03 - March" / "tree_green.jpg", "green")  # date from the folder only
    save(root / "misc" / "yellow.png", "yellow")
    return root


@pytest.fixture(scope="module")
def e2e_index(e2e_library, tmp_path_factory):
    # the real offline gazetteer needs the search-core extras (absent on the core-only CI leg)
    pytest.importorskip("geonamescache")
    pytest.importorskip("scipy")
    from facesorter.search.indexer import open_indexer

    register_model(
        testing.FAKE_MODEL_NAME,
        lambda device: FakeImageEncoder(),
        lambda quantization: FakeTextEncoder(),
        dim=testing.FAKE_DIM,
    )
    try:
        path = tmp_path_factory.mktemp("e2e_idx") / "nested" / "index.sqlite"
        with open_indexer(path, model=testing.FAKE_MODEL_NAME) as indexer:  # real gazetteer
            report = indexer.run(e2e_library)
    finally:
        unregister_model(testing.FAKE_MODEL_NAME)
    assert (report.scanned, report.new, report.failed) == (7, 7, ())
    return path


@pytest.fixture
def e2e_engine(e2e_index, fake_model):
    with open_engine(e2e_index, use_llm=False) as eng:  # real gazetteer
        yield eng


class TestEndToEnd:
    def test_red_cars_in_paris_last_summer(self, e2e_engine):
        res = e2e_engine.search(request("red cars in Paris last summer"))
        assert names_of(res) == ["car_red", "car_blue"]
        assert res.applied.place.city == "Paris"
        assert res.applied.date_range == DateRange(date(2026, 6, 1), date(2026, 8, 31))
        assert res.candidates == 2 and explain(res) == []

    def test_same_visual_text_different_place(self, e2e_engine):
        res = e2e_engine.search(request("red in Key West"))
        assert names_of(res)[0] == "beach_red" and res.candidates == 2

    def test_other_year_picks_the_other_red_photo(self, e2e_engine):
        res = e2e_engine.search(request("red car in Paris in 2024"))
        assert names_of(res) == ["old_red"]

    def test_folder_dated_photo_is_found_by_its_month(self, e2e_engine):
        res = e2e_engine.search(request("green tree in March 2022"))
        assert names_of(res) == ["tree_green"]
        res = e2e_engine.search(request("green in 2022"))
        assert names_of(res) == ["tree_green"]

    def test_unfiltered_ranking_puts_the_colour_first(self, e2e_engine):
        assert names_of(e2e_engine.search(request("yellow")))[0] == "yellow"
        assert set(names_of(e2e_engine.search(request("green")))[:2]) == {
            "boat_green",
            "tree_green",
        }

    def test_date_only_query_browses_newest_first(self, e2e_engine):
        res = e2e_engine.search(request("last summer"))
        assert names_of(res) == ["car_blue", "car_red", "boat_green", "beach_red"]
        assert "No visual description, showing newest first" in explain(res)

    def test_unknown_date_asks_then_drops(self, e2e_engine):
        q = "green Diwali 2023"
        first = e2e_engine.search(request(q))
        (u,) = first.needs_clarification
        assert first.hits == () and u.phrase == "Diwali 2023"
        reworded = e2e_engine.search(request(q, clarifications={u.phrase: "June 2026"}))
        assert names_of(reworded) == ["boat_green", "beach_red"]
        dropped = e2e_engine.search(request(q, clarifications={u.phrase: "that festival"}))
        assert [d.phrase for d in dropped.dropped] == [u.phrase] and set(names_of(dropped)[:2]) == {
            "boat_green",
            "tree_green",
        }
        assert any("searched without it" in n for n in explain(dropped))

    def test_nothing_in_that_summer(self, e2e_engine):
        res = e2e_engine.search(request("red in Paris summer 2019"))
        assert res.hits == () and explain(res)[0].startswith("No photos match")
        assert explain(res)[1].startswith("Your library covers photos from 2022-03-01")

    def test_library_roots_recorded(self, e2e_engine, e2e_library):
        assert json.loads(e2e_engine.store.get_meta("library_roots")) == [str(e2e_library)]
        assert os.path.isabs(e2e_engine.store.get_meta("library_roots").strip('[]"'))


def test_indexer_and_engine_agree_on_incremental_changes(tmp_path, fake_model, monkeypatch):
    """Index, change the library, re-index: searches see exactly the new state."""
    from facesorter.search import geo

    monkeypatch.setattr(geo, "build_gazetteer", lambda *a, **k: FakeGazetteer())
    root = tmp_path / "pics"
    save(root / "a.jpg", "red", "2026:07:01 10:00:00")
    save(root / "b.jpg", "blue", "2026:07:02 10:00:00")
    path = tmp_path / "index.sqlite"
    from facesorter.search.indexer import open_indexer

    with open_indexer(path, model=fake_model) as indexer:
        indexer.run(root)
        (root / "a.jpg").unlink()
        save(root / "c.jpg", "red", "2026:07:03 10:00:00")
        report = indexer.run(root)
    assert (report.new, report.removed) == (1, 1)
    with open_engine(path, use_llm=False) as eng:
        assert names_of(eng.search(request("red"))) == ["c", "b"]


# ----------------------------------------------------------------------------- library-aware places


def pf(city=None, region=None, country=None, **kw) -> PlaceFilter:
    return PlaceFilter(
        city=city, region=region, country=country, text=kw.pop("text", city or ""), **kw
    )


AMHERST_NY = pf("Amherst", "New York", "United States")
AMHERST_MA = pf("Amherst", "Massachusetts", "United States")
MONTERREY = pf("Monterrey", "Nuevo León", "Mexico")
MONTEREY = pf("Monterey", "California", "United States")
PARIS_FR = pf("Paris", "Île-de-France", "France")
PARIS_TX = pf("Paris", "Texas", "United States")
SPRINGFIELD_IL = pf("Springfield", "Illinois", "United States")
SPRINGFIELD_MA = pf("Springfield", "Massachusetts", "United States")
SANTA_CRUZ_PH = pf("Santa Cruz", "Calabarzon", "Philippines")
SANTA_CRUZ_POINT = PlaceFilter(near=(36.97, -122.03, 10.0), text="Santa Cruz")


class AmbiguousGazetteer:
    """Readings per name, biggest namesake first, like ``OfflineGazetteer.candidates``."""

    table = {
        "amherst": [AMHERST_NY, AMHERST_MA],
        "amherst, new york": [AMHERST_NY],
        "monterey": [MONTERREY, MONTEREY],
        "paris": [PARIS_FR, PARIS_TX],
        "paris texas": [PARIS_TX, PARIS_FR],
        "springfield": [SPRINGFIELD_IL, SPRINGFIELD_MA],
        "santa cruz": [SANTA_CRUZ_PH, SANTA_CRUZ_POINT],
        "atlantis": [pf("Atlantis", None, "South Africa")],
    }

    def __init__(self) -> None:
        self.candidate_calls: list[str] = []

    def candidates(self, name):
        self.candidate_calls.append(name)
        return list(self.table.get(" ".join(name.lower().split()), []))

    def resolve(self, name):
        found = self.candidates(name)
        return found[0] if found else None

    def label_for(self, lat, lon):
        return PlaceLabel("Somewhere")

    def label_many(self, lats, lons):
        return [PlaceLabel("Many")] * len(lats)


def add_photos(store: Store, spec: dict[str, tuple[PlaceFilter | None, int]], gps=None) -> None:
    """Photos labelled with the places of ``spec`` (name -> (place, how many))."""
    enc = FakeImageEncoder()
    vec = enc.encode_images([make_color_image("red")])[0]
    records = []
    for name, (place, n) in spec.items():
        label = None if place is None else PlaceLabel(place.city, place.region, place.country)
        for i in range(n):
            meta = photo_meta(f"/lib/{name}-{i}.jpg", datetime(2025, 1, 1 + i))
            if gps:
                meta = replace(meta, lat=gps[0], lon=gps[1])
            records.append((meta, label, vec))
    store.upsert_photos(records)


class TestLibraryGazetteer:
    def make(self, store, **spec) -> tuple[LibraryGazetteer, AmbiguousGazetteer]:
        inner = AmbiguousGazetteer()
        add_photos(store, spec)
        return LibraryGazetteer(inner, store), inner

    def test_the_namesake_with_photos_beats_the_bigger_one(self, store):
        lib, _ = self.make(store, ma=(AMHERST_MA, 3))
        assert lib.resolve("Amherst") == AMHERST_MA

    def test_most_photos_wins(self, store):
        lib, _ = self.make(store, ny=(AMHERST_NY, 1), ma=(AMHERST_MA, 3))
        assert lib.resolve("amherst") == AMHERST_MA

    def test_the_gazetteers_own_choice_wins_a_tie(self, store):
        lib, _ = self.make(store, ny=(AMHERST_NY, 2), ma=(AMHERST_MA, 2))
        assert lib.resolve("Amherst") == AMHERST_NY

    def test_a_smaller_spelling_variant_with_photos_beats_the_default(self, store):
        lib, _ = self.make(store, ca=(MONTEREY, 2))
        assert lib.resolve("Monterey") == MONTEREY  # not Monterrey, Mexico

    def test_more_photos_in_the_default_keeps_it(self, store):
        lib, _ = self.make(store, mx=(MONTERREY, 5), ca=(MONTEREY, 2))
        assert lib.resolve("Monterey") == MONTERREY

    def test_nothing_in_the_library_means_the_default(self, store):
        lib, _ = self.make(store, other=(PARIS_FR, 1))
        assert lib.resolve("Springfield") == SPRINGFIELD_IL
        assert lib.resolve("Monterey") == MONTERREY

    def test_an_empty_index_means_the_default(self, store):
        lib, _ = self.make(store)
        assert lib.resolve("Amherst") == AMHERST_NY

    def test_explicit_qualifier_keeps_the_default_behaviour(self, store):
        lib, _ = self.make(store, ma=(AMHERST_MA, 5))
        assert lib.resolve("Amherst, New York") == AMHERST_NY
        tx = AmbiguousGazetteer.table["paris texas"][0]
        add_photos(store, {"fr": (PARIS_FR, 9)})
        assert lib.resolve("Paris Texas") == tx  # the qualified reading, not the busier Paris

    def test_unknown_names_and_names_with_one_reading(self, store):
        lib, _ = self.make(store, x=(AMHERST_MA, 1))
        assert lib.resolve("Nowhereville") is None
        assert lib.resolve("Atlantis").country == "South Africa"

    def test_regions_and_countries_count_too(self, store):
        class G(AmbiguousGazetteer):
            table = {
                "washington": [
                    pf("Washington", "District of Columbia", "United States"),
                    PlaceFilter(region="Washington", country="United States", text="Washington"),
                ]
            }

        add_photos(store, {"seattle": (pf("Seattle", "Washington", "United States"), 4)})
        assert LibraryGazetteer(G(), store).resolve("Washington").region == "Washington"

    def test_a_point_reading_is_counted_by_its_radius(self, store):
        inner = AmbiguousGazetteer()
        add_photos(store, {"sc": (None, 2)}, gps=(36.97, -122.03))
        assert LibraryGazetteer(inner, store).resolve("Santa Cruz") == SANTA_CRUZ_POINT

    def test_case_accent_and_spacing_share_one_cache_entry(self, store):
        lib, inner = self.make(store, ma=(AMHERST_MA, 3))
        for spelling in ("Amherst", " amherst ", "AMHERST", "Amherst"):
            assert lib.resolve(spelling) == AMHERST_MA
        assert len(inner.candidate_calls) == 2  # computed once (resolve + candidates)

    def test_cache_is_dropped_when_the_index_changes(self, store):
        lib, inner = self.make(store, ma=(AMHERST_MA, 1))
        assert lib.resolve("Amherst") == AMHERST_MA
        calls = len(inner.candidate_calls)
        assert lib.resolve("Amherst") == AMHERST_MA
        assert len(inner.candidate_calls) == calls  # cached
        add_photos(store, {"ny": (AMHERST_NY, 4)})  # re-index: the count changed
        assert lib.resolve("Amherst") == AMHERST_NY
        assert len(inner.candidate_calls) > calls

    def test_cache_is_dropped_when_indexed_at_changes(self, store):
        lib, inner = self.make(store, ma=(AMHERST_MA, 1))
        lib.resolve("Amherst")
        calls = len(inner.candidate_calls)
        store.set_meta("indexed_at", "2025-01-01T00:00:00")
        lib.resolve("Amherst")
        assert len(inner.candidate_calls) > calls

    def test_none_is_cached_too(self, store):
        lib, inner = self.make(store, ma=(AMHERST_MA, 1))
        assert lib.resolve("Nowhere") is None and lib.resolve("Nowhere") is None
        assert len(inner.candidate_calls) == 1

    def test_closed_index_falls_back_to_the_wrapped_gazetteer(self, tmp_path):
        s = Store.open(tmp_path / "c.sqlite", model_name="fake-color-16", dim=16)
        lib = LibraryGazetteer(AmbiguousGazetteer(), s)
        s.close()
        assert lib.resolve("Amherst") == AMHERST_NY

    def test_pass_through(self, store):
        lib, inner = self.make(store)
        assert lib.label_for(1, 2) == PlaceLabel("Somewhere")
        assert lib.label_many([1, 2], [3, 4]) == [PlaceLabel("Many")] * 2
        assert lib.candidates("Amherst") == [AMHERST_NY, AMHERST_MA]

    def test_pass_through_for_a_minimal_gazetteer(self, store):
        class Minimal:
            def label_for(self, lat, lon):
                return PlaceLabel("Only")

            def resolve(self, name):
                return PlaceFilter(city=name) if name == "Rome" else None

        lib = LibraryGazetteer(Minimal(), store)
        assert lib.label_many([1, 2], [3, 4]) == [PlaceLabel("Only")] * 2
        assert lib.candidates("Rome") == [PlaceFilter(city="Rome")]
        assert lib.candidates("Oslo") == []
        assert lib.resolve("Rome") == PlaceFilter(city="Rome")  # no candidates() on the inner
        assert lib.resolve("Oslo") is None


class TestLibraryAwareSearch:
    def test_query_uses_the_namesake_in_the_library(self, store):
        add_photos(store, {"ma": (AMHERST_MA, 3)})
        eng = Engine(store, FakeTextEncoder(), QueryParser(None, None, store), None)
        gaz = AmbiguousGazetteer()
        eng = Engine(store, FakeTextEncoder(), QueryParser(None, gaz, store), gaz)
        res = eng.search(request("red in Amherst"))
        assert res.applied.place.region == "Massachusetts" and len(res.hits) == 3
        assert res.candidates == 3

    def test_engine_switches_a_parser_built_on_the_same_gazetteer(self, store):
        gaz = AmbiguousGazetteer()
        parser = QueryParser(None, gaz, store)
        eng = Engine(store, FakeTextEncoder(), parser, gaz)
        assert eng.gazetteer.inner is gaz and parser._gazetteer is eng.gazetteer

    def test_an_already_wrapped_gazetteer_is_not_wrapped_twice(self, store):
        wrapped = LibraryGazetteer(AmbiguousGazetteer(), store)
        parser = QueryParser(None, wrapped, store)
        eng = Engine(store, FakeTextEncoder(), parser, wrapped)
        assert eng.gazetteer is wrapped and parser._gazetteer is wrapped

    def test_flag_places_resolve_the_same_way_through_engine_gazetteer(self, store):
        add_photos(store, {"ca": (MONTEREY, 2)})
        gaz = AmbiguousGazetteer()
        eng = Engine(store, FakeTextEncoder(), QueryParser(None, gaz, store), gaz)
        assert eng.gazetteer.resolve("Monterey") == MONTEREY  # what the UI's place box uses

    def test_open_engine_wires_the_wrapper_into_the_parser(self, tmp_path, fake_model, monkeypatch):
        from facesorter.search import geo

        monkeypatch.setattr(geo, "build_gazetteer", lambda *a, **k: AmbiguousGazetteer())
        path = tmp_path / "i.sqlite"
        with Store.open(path, model_name=fake_model, dim=16) as s:
            add_photos(s, {"ma": (AMHERST_MA, 2)})
        with open_engine(path, use_llm=False) as eng:
            assert isinstance(eng.gazetteer, LibraryGazetteer)
            assert eng.parser._gazetteer is eng.gazetteer
            assert eng.search(request("red in Amherst")).applied.place == AMHERST_MA


# ----------------------------------------------------------------------------- hints, wording, misc


class TestEmptyResultHints:
    def test_place_hint_lists_the_most_common_cities(self, engine):
        res = engine.search(request("", filters=Filters(place=PlaceFilter(city="Rome"))))
        assert res.candidates == 0
        assert explain(res) == [
            "No photos match: place Rome",
            "Places in your library include: Paris, Key West.",
        ]

    def test_both_hints(self, engine):
        flags = Filters(
            date_range=DateRange(date(2019, 1, 1), date(2019, 12, 31)),
            place=PlaceFilter(city="Rome"),
        )
        assert explain(engine.search(request("red", filters=flags))) == [
            "No photos match: dates 2019-01-01 to 2019-12-31 · place Rome",
            "Your library covers photos from 2024-07-04 to 2026-07-10.",
            "Places in your library include: Paris, Key West.",
        ]

    def test_hints_only_for_the_filter_that_matches_nothing(self, engine):
        # each filter alone matches photos; only the combination is empty
        flags = Filters(
            date_range=DateRange(date(2024, 1, 1), date(2024, 12, 31)),
            place=PlaceFilter(city="Key West"),
        )
        res = engine.search(request("red", filters=flags))
        assert res.candidates == 0
        (only,) = explain(res)
        assert only.startswith("No photos match: dates 2024-01-01 to 2024-12-31 · place Key West")

    def test_date_hint_only_when_the_date_is_the_problem(self, engine):
        flags = Filters(
            date_range=DateRange(date(2019, 1, 1), date(2019, 12, 31)),
            place=PlaceFilter(city="Paris"),
        )
        notes = explain(engine.search(request("red", filters=flags)))
        assert len(notes) == 2 and notes[1].startswith("Your library covers")

    def test_hints_land_in_parse_notes_but_print_after_the_outcome(self, engine):
        res = engine.search(request("red cars in Paris in 2019"))
        assert res.parse.notes[-1].startswith("Your library covers")
        assert explain(res)[0].startswith("No photos match")

    def test_no_hints_for_an_empty_index(self, store):
        res = make_engine(store).search(request("red in Paris in 2019"))
        assert explain(res) == [
            "No photos match: dates 2019-01-01 to 2019-12-31 · place Paris, Île-de-France, France"
        ]

    def test_no_date_hint_without_dated_photos(self, store):
        add_photos(store, {"x": (None, 0)})
        store.upsert_photos(
            [
                (
                    photo_meta("/p/u.jpg", None),
                    PARIS,
                    FakeImageEncoder().encode_images([make_color_image("red")])[0],
                )
            ]
        )
        res = make_engine(store).search(request("red in 2019"))
        assert explain(res) == ["No photos match: dates 2019-01-01 to 2019-12-31"]

    def test_countries_are_listed_when_no_photo_has_a_city(self, store):
        vec = FakeImageEncoder().encode_images([make_color_image("red")])[0]
        label = PlaceLabel(None, None, "France", "FR")
        store.upsert_photos([(photo_meta("/p/a.jpg", datetime(2025, 1, 1)), label, vec)])
        res = make_engine(store).search(
            request("", filters=Filters(place=PlaceFilter(city="Rome")))
        )
        assert explain(res)[-1] == "Places in your library include: France."

    def test_no_place_hint_when_nothing_is_labelled(self, store):
        vec = FakeImageEncoder().encode_images([make_color_image("red")])[0]
        store.upsert_photos([(photo_meta("/p/a.jpg", datetime(2025, 1, 1)), None, vec)])
        res = make_engine(store).search(
            request("", filters=Filters(place=PlaceFilter(city="Rome")))
        )
        assert explain(res) == ["No photos match: place Rome"]

    def test_at_most_five_places_and_the_list_is_cached(self, store, monkeypatch):
        vec = FakeImageEncoder().encode_images([make_color_image("red")])[0]
        records = []
        for i, city in enumerate("ABCDEFG"):
            for j in range(8 - i):
                records.append(
                    (photo_meta(f"/p/{city}{j}.jpg", datetime(2025, 1, 1)), PlaceLabel(city), vec)
                )
        store.upsert_photos(records)
        eng = make_engine(store)
        flags = Filters(place=PlaceFilter(city="Rome"))
        assert explain(eng.search(request("", filters=flags)))[-1] == (
            "Places in your library include: A, B, C, D, E."
        )
        calls = []
        real = store.get_photos
        monkeypatch.setattr(store, "get_photos", lambda ids: calls.append(list(ids)) or real(ids))
        eng.search(request("", filters=flags))
        assert [c for c in calls if c] == []  # the second explanation came from the cache
        store.upsert_photos([(photo_meta("/p/new.jpg", None), PlaceLabel("Z"), vec)])
        eng.search(request("", filters=flags))
        assert [c for c in calls if c]  # the index changed, so it was recomputed

    def test_explain_prints_hints_last(self):
        res = response(
            candidates=0,
            applied=Filters(place=PlaceFilter(city="Rome")),
            parse=ParseResult(
                ParsedQuery(raw="x", text="x"),
                Filters(),
                notes=("Places in your library include: A.", "a real note"),
            ),
        )
        assert explain(res) == [
            "a real note",
            "No photos match: place Rome",
            "Places in your library include: A.",
        ]


class TestDescribeOpenEnded:
    def test_open_end(self):
        f = Filters(date_range=DateRange(date(2023, 6, 1), date.max))
        assert describe_filters(f) == "from 2023-06-01"

    def test_open_start(self):
        f = Filters(date_range=DateRange(date.min, date(2024, 2, 29)))
        assert describe_filters(f) == "up to 2024-02-29"

    def test_no_bound_at_all_prints_nothing(self):
        assert describe_filters(Filters(date_range=DateRange(date.min, date.max))) == ""

    def test_combined_with_other_filters(self):
        f = Filters(
            date_range=DateRange(date(2023, 6, 1), date.max),
            place=PlaceFilter(city="Rome"),
            people=("Bob",),
        )
        assert describe_filters(f) == "from 2023-06-01 · place Rome · with Bob"
        assert "9999" not in describe_filters(f)

    def test_people_are_deduplicated_ignoring_case(self):
        assert describe_filters(Filters(people=("BOB", "bob"))) == "with BOB"
        assert describe_filters(Filters(people=("Bob", "bob", "Alice"))) == "with Bob and Alice"

    def test_open_ended_search_filters_work_and_print_sanely(self, engine):
        flags = Filters(date_range=DateRange(date(2026, 7, 1), date.max))
        res = engine.search(request("red", filters=flags))
        assert res.candidates == 3  # the three photos of July 2026; nothing later exists
        assert names_of(res)[0] == "paris_red_summer"
        empty = engine.search(
            request("red", filters=Filters(date_range=DateRange(date(2030, 1, 1), date.max)))
        )
        assert explain(empty)[0] == "No photos match: from 2030-01-01"


class TestHasLlmAndLimits:
    def test_has_llm(self, store, ids):
        assert make_engine(store).has_llm is False
        assert make_engine(store, llm=ScriptedLLM()).has_llm is True

    def test_has_llm_from_open_engine(self, tmp_path, fake_model, monkeypatch):
        import facesorter.search.llm as llm_module
        from facesorter.search import geo

        monkeypatch.setattr(geo, "build_gazetteer", lambda *a, **k: FakeGazetteer())
        monkeypatch.setattr(llm_module, "load_llm", lambda *a: ScriptedLLM())
        path = tmp_path / "i.sqlite"
        Store.open(path, model_name=fake_model, dim=16).close()
        with open_engine(path) as eng:
            assert eng.has_llm
        with open_engine(path, use_llm=False) as eng:
            assert not eng.has_llm

    @pytest.mark.parametrize("k", [2**31, 2**63, 2**63 + 5, 10**30])
    def test_huge_k_is_clamped(self, engine, k):
        assert len(engine.search(request("", k=k)).hits) == len(DATASET)  # browse path
        assert len(engine.search(request("red", k=k)).hits) == len(DATASET)  # ranked path

    @pytest.mark.parametrize("k", [0, -(2**70)])
    def test_small_k_still_rejected(self, engine, k):
        with pytest.raises(ValueError):
            engine.search(request("red", k=k))

    def test_text_encoder_is_never_entered_concurrently(self, store, ids):
        state = {"now": 0, "worst": 0, "calls": 0}
        lock = threading.Lock()

        class Counting(FakeTextEncoder):
            def encode_text(self, texts):
                with lock:
                    state["now"] += 1
                    state["calls"] += 1
                    state["worst"] = max(state["worst"], state["now"])
                time.sleep(0.01)
                try:
                    return super().encode_text(texts)
                finally:
                    with lock:
                        state["now"] -= 1

        eng = Engine(store, Counting(), QueryParser(None, FakeGazetteer(), store))
        barrier = threading.Barrier(8)

        def go(i):
            barrier.wait()
            return eng.search(request(f"red {i}"))

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(go, range(8)))
        assert state["calls"] == 8 and state["worst"] == 1
