"""Tests for facesorter.search.ui_logic: form -> request, clarification flow, thumbnails.

Everything uses synthetic images and a scripted engine over a real (temporary) ``Store``;
no model weights, network or real photos.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pillow_heif
import pytest
from PIL import Image

from facesorter.search import ui_logic
from facesorter.search.parser import is_asked
from facesorter.search.store import Store
from facesorter.search.types import (
    DateRange,
    Filters,
    Hit,
    ParsedQuery,
    ParseResult,
    PhotoMeta,
    PlaceFilter,
    PlaceLabel,
    SearchRequest,
    SearchResponse,
    Unresolved,
)
from facesorter.search.ui_logic import (
    HitCard,
    Session,
    ThumbnailCache,
    UIInputError,
    answer_clarifications,
    clamp_k,
    date_filter,
    parse_date_bound,
    record_feedback,
    run_search,
    skip_clarifications,
    start_search,
)

DIM = 8
MODEL = "fake-model"


# --------------------------------------------------------------------------- helpers


def make_jpeg(path: Path, size: tuple[int, int] = (64, 48), color=(200, 40, 40), **save) -> Path:
    Image.new("RGB", size, color).save(path, "JPEG", **save)
    return path


def photo_meta(path: Path, taken: str | None = "2023-07-14T10:00:00", **kw) -> PhotoMeta:
    st = path.stat()
    when = datetime.fromisoformat(taken) if taken else None
    return PhotoMeta(
        path=str(path),
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        taken=when,
        date_source=kw.pop("date_source", "exif" if when else "none"),
        **kw,
    )


def unresolved(kind: str, phrase: str) -> Unresolved:
    return Unresolved(kind, phrase, "unknown", f'What do you mean by "{phrase}"?')  # type: ignore[arg-type]


class FakeGazetteer:
    def __init__(self, known: dict[str, PlaceFilter]) -> None:
        self.known = known

    def resolve(self, name: str) -> PlaceFilter | None:
        return self.known.get(name.casefold())


class FakeEngine:
    """Scripted ``SearchEngine`` over a real store; mimics the engine's ask-vs-drop rule."""

    def __init__(self, store: Store, ask: Sequence[Unresolved] = (), gazetteer=None) -> None:
        self.store = store
        self.ask = list(ask)
        self.gazetteer = gazetteer
        self.requests: list[SearchRequest] = []
        self.raises: Exception | None = None
        self.respond: Callable[[SearchRequest], SearchResponse] | None = None

    def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.respond is not None:
            return self.respond(request)
        applied = request.filters or Filters()
        parse = ParseResult(
            ParsedQuery(raw=request.query, text=request.query), applied, tuple(self.ask)
        )
        unasked = [u for u in self.ask if not is_asked(u.phrase, request.clarifications)]
        if unasked and request.allow_clarification:
            return SearchResponse((), parse, applied, 0, needs_clarification=tuple(unasked))
        ids = [p for p in range(1, self.store.count() + 1)][: request.k]
        hits = tuple(
            Hit(photo, score=1.0 / (i + 1)) for i, photo in enumerate(self.store.get_photos(ids))
        )
        return SearchResponse(
            hits,
            parse,
            applied,
            self.store.count(),
            dropped=tuple(self.ask),
            timings_ms={"parse": 1.0, "encode": 2.0, "search": 3.0},
        )


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root = tmp_path / "library" / "trip"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def store(tmp_path: Path, library: Path):
    s = Store.open(tmp_path / "index.sqlite", model_name=MODEL, dim=DIM)
    rng = np.random.default_rng(0)
    records = []
    for i in range(5):
        path = make_jpeg(library / f"IMG_{i}.jpg", color=(40 * i, 90, 200))
        place = PlaceLabel(city="Key West", region="Florida", country="United States")
        records.append(
            (
                photo_meta(path, f"2023-07-{10 + i:02d}T10:00:00", camera="Pixel 8"),
                place if i % 2 == 0 else None,
                rng.standard_normal(DIM).astype(np.float32),
            )
        )
    s.upsert_photos(records)
    yield s
    s.close()


@pytest.fixture
def engine(store: Store) -> FakeEngine:
    return FakeEngine(store)


@pytest.fixture
def thumbs(tmp_path: Path) -> ThumbnailCache:
    return ThumbnailCache(tmp_path / "thumbs")


# --------------------------------------------------------------------------- dates


@pytest.mark.parametrize(
    ("text", "end", "expected"),
    [
        ("2024", False, date(2024, 1, 1)),
        ("2024", True, date(2024, 12, 31)),
        ("2024-02", False, date(2024, 2, 1)),
        ("2024-02", True, date(2024, 2, 29)),  # leap year
        ("2023-02", True, date(2023, 2, 28)),
        ("1900-02", True, date(1900, 2, 28)),  # century, not leap
        ("2000-02", True, date(2000, 2, 29)),  # divisible by 400, leap
        ("2024-12", True, date(2024, 12, 31)),
        ("2024-2-5", False, date(2024, 2, 5)),
        ("2024-02-29", True, date(2024, 2, 29)),
        (" 2024/07/14 ", False, date(2024, 7, 14)),
        ("2024.07.14", True, date(2024, 7, 14)),
        ("", False, None),
        ("   ", True, None),
        (None, True, None),
    ],
)
def test_parse_date_bound(text, end, expected) -> None:
    assert parse_date_bound(text, end=end) == expected


@pytest.mark.parametrize(
    "text",
    [
        "2023-13",
        "2023-00",
        "2023-02-30",
        "2023-02-29",
        "0000",
        "abcd",
        "23",
        "last summer",
        "2023-07-14-1",
        "2023-07-00",
        "2023-1-1x",
        "0000-01-01",
    ],
)
def test_parse_date_bound_rejects_bad_input_with_a_friendly_message(text) -> None:
    for end in (False, True):
        with pytest.raises(UIInputError) as info:
            parse_date_bound(text, end=end)
        assert "YYYY" in str(info.value)


def test_date_filter_expands_each_side_and_leaves_open_ends() -> None:
    assert date_filter("", "") is None
    assert date_filter(None, None) is None
    assert date_filter("2023", "2023") == DateRange(date(2023, 1, 1), date(2023, 12, 31))
    assert date_filter("2024-02", "2024-02") == DateRange(date(2024, 2, 1), date(2024, 2, 29))
    assert date_filter("2023-06", "") == DateRange(date(2023, 6, 1), date.max)
    assert date_filter("", "2020-01") == DateRange(date.min, date(2020, 1, 31))
    assert date_filter("2023-03-05", "2023-03-05") == DateRange(date(2023, 3, 5), date(2023, 3, 5))


def test_date_filter_rejects_from_after_to() -> None:
    with pytest.raises(UIInputError, match="after"):
        date_filter("2024", "2023")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (24, 24),
        (1, 1),
        (0, 1),
        (-5, 1),
        (10**9, ui_logic.MAX_K),
        ("12", 12),
        (7.9, 7),
        (None, 24),
        ("x", 24),
        (float("nan"), 24),
        (float("inf"), 24),
        ([], 24),
        (ui_logic.MAX_K, ui_logic.MAX_K),
        (ui_logic.MAX_K + 1, ui_logic.MAX_K),
    ],
)
def test_clamp_k(value, expected) -> None:
    assert clamp_k(value) == expected


# --------------------------------------------------------------------------- run_search


def test_run_search_builds_the_request_and_cards(engine: FakeEngine, thumbs: ThumbnailCache):
    result = run_search(engine, "  blue   sea ", k=3, thumbs=thumbs)
    (req,) = engine.requests
    assert (req.query, req.k, req.filters, req.use_llm) == ("blue sea", 3, None, True)
    assert req.allow_clarification and dict(req.clarifications) == {}
    assert result.error is None and len(result.cards) == 3
    card = result.cards[0]
    assert card.photo_id == 1 and card.filename == "IMG_0.jpg"
    assert card.caption == "2023-07-10 · Key West, Florida, United States · 1.00"
    assert result.cards[1].caption == "2023-07-11 · 0.50"  # no place label
    assert card.camera == "Pixel 8" and card.thumb_ok
    assert card.thumb_path and Path(card.thumb_path).parent == thumbs.directory
    assert result.candidates == 5 and result.elapsed_ms > 0
    assert result.timings_ms == {"parse": 1.0, "encode": 2.0, "search": 3.0}
    assert result.status.startswith("Showing 3 photos of 5 candidates")
    assert 'Looking for: "blue sea".' in result.reading
    assert result.reading.splitlines()[0] == "Filters: none."
    assert result.gallery_items()[0] == (card.thumb_path, card.caption)


def test_cards_work_without_a_thumbnail_cache(engine: FakeEngine):
    result = run_search(engine, "x", k=2)
    assert [c.thumb_path for c in result.cards] == [None, None]
    assert result.gallery_items() == [] and result.notices == []


def test_explicit_fields_override_per_field(engine: FakeEngine):
    run_search(
        engine,
        "beach",
        date_from="2023-06",
        date_to="2023-08",
        place="",
        people=["Bob", "bob", " ", "Alice"],
        use_llm=False,
    )
    (req,) = engine.requests
    assert req.filters == Filters(
        date_range=DateRange(date(2023, 6, 1), date(2023, 8, 31)),
        place=None,
        people=("Bob", "Alice"),
    )
    assert req.use_llm is False and req.query == "beach"


def test_only_set_fields_are_sent(engine: FakeEngine):
    run_search(engine, "beach", people="Bob")  # a single string is one name
    assert engine.requests[0].filters == Filters(people=("Bob",))
    run_search(engine, "beach", people=None)
    assert engine.requests[1].filters is None


def test_place_is_resolved_through_the_gazetteer(store: Store):
    kw = PlaceFilter(city="Key West", region="Florida", country="United States")
    engine = FakeEngine(store, gazetteer=FakeGazetteer({"key west": kw}))
    result = run_search(engine, "beach", place="Key West")
    (req,) = engine.requests
    assert req.query == "beach"  # not added to the query when resolved
    assert req.filters is not None and req.filters.place == PlaceFilter(
        "Key West", "Florida", "United States", text="Key West"
    )
    assert "place Key West, Florida, United States" in result.reading


def test_unresolved_or_unavailable_place_becomes_part_of_the_query(store: Store, engine):
    withgaz = FakeEngine(store, gazetteer=FakeGazetteer({}))
    run_search(withgaz, "beach", place="Atlantis")
    assert withgaz.requests[0].query == "beach in Atlantis"
    assert withgaz.requests[0].filters is None
    run_search(engine, "", place="  Paris ")  # no gazetteer attribute value
    assert engine.requests[0].query == "in Paris"


def test_gazetteer_errors_and_odd_answers_fall_back_to_the_query(store: Store):
    class Boom:
        def resolve(self, name):
            raise RuntimeError("no")

    class Odd:
        def resolve(self, name):
            return "Paris"

    for gaz in (Boom(), Odd()):
        engine = FakeEngine(store, gazetteer=gaz)
        run_search(engine, "cafe", place="Paris")
        assert engine.requests[0].query == "cafe in Paris"


def test_k_is_clamped_with_a_notice(engine: FakeEngine):
    result = run_search(engine, "x", k=10**6)
    assert engine.requests[0].k == ui_logic.MAX_K
    assert f"Showing at most {ui_logic.MAX_K} results." in result.notices
    result = run_search(engine, "x", k=0)
    assert engine.requests[1].k == 1 and len(result.cards) == 1
    run_search(engine, "x", k=None)  # type: ignore[arg-type]
    assert engine.requests[2].k == ui_logic.DEFAULT_K


def test_long_queries_are_cut(engine: FakeEngine):
    run_search(engine, "word " * 1000)
    assert len(engine.requests[0].query) <= ui_logic.MAX_QUERY_CHARS


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"date_from": "2023-02-30"}, "not a real calendar date"),
        ({"date_to": "next week"}, "not a date"),
        ({"date_from": "2024", "date_to": "2023"}, "after"),
    ],
)
def test_bad_dates_give_a_message_and_skip_the_engine(engine: FakeEngine, kwargs, fragment):
    result = run_search(engine, "beach", **kwargs)
    assert result.error and fragment in result.error
    assert result.status == result.error and result.cards == []
    assert engine.requests == []


def test_engine_failure_becomes_an_error_result(engine: FakeEngine, caplog):
    engine.raises = RuntimeError("model exploded")
    result = run_search(engine, "beach", k=1000)
    assert result.error == "Search failed: model exploded"
    assert f"Showing at most {ui_logic.MAX_K} results." in result.notices
    engine.raises = ValueError()
    assert run_search(engine, "beach").error == "Search failed: ValueError"


def test_empty_results_and_single_photo_status(engine: FakeEngine, store: Store):
    engine.respond = lambda req: SearchResponse(
        (), ParseResult(ParsedQuery("q", ""), Filters()), Filters(), 0
    )
    assert run_search(engine, "q").status == "No photos found (searched 0 candidate photos)."
    result = run_search(FakeEngine(store), "x", k=1)
    assert result.status.startswith("Showing 1 photo of 5")
    assert "No visual description" in result.reading or 'Looking for: "x"' in result.reading


def test_reading_text_uses_engine_wording_and_open_ended_dates(engine: FakeEngine):
    result = run_search(engine, "", date_from="2023-06", people=["Bob"])
    first = result.reading.splitlines()[0]
    assert first == "Filters: dates on or after 2023-06-01 · with Bob."
    assert "No visual description in the query." in result.reading
    result = run_search(engine, "x", date_to="2020")
    assert result.reading.splitlines()[0] == "Filters: dates on or before 2020-12-31."
    result = run_search(engine, "x", date_from="2023", date_to="2023")
    assert result.reading.splitlines()[0] == "Filters: dates 2023-01-01 to 2023-12-31."


def test_helper_failures_fall_back_without_breaking_the_page(engine: FakeEngine, monkeypatch):
    def boom(*_a):
        raise RuntimeError("bug")

    engine.describe_filters = boom  # type: ignore[attr-defined]
    engine.explain = boom  # type: ignore[attr-defined]
    result = run_search(engine, "x", people=["Bob"])
    assert result.reading.splitlines()[0] == "Filters: with Bob."
    assert result.error is None and result.notices == []


def test_fallback_wording_when_the_engine_module_is_unavailable(engine, monkeypatch):
    import facesorter.search as package

    monkeypatch.setitem(__import__("sys").modules, "facesorter.search.engine", None)
    monkeypatch.delattr(package, "engine", raising=False)
    place = PlaceFilter(city="Oslo", country="Norway", text="oslo")
    only_text = PlaceFilter(text="the lake")
    near = PlaceFilter(near=(1.0, 2.0, 3.0))
    for filters, expected in [
        (
            Filters(DateRange(date(2020, 1, 1), date(2020, 1, 2)), place, ("A", "B")),
            "dates 2020-01-01 to 2020-01-02 · place Oslo, Norway · with A, B",
        ),
        (Filters(place=only_text), "place the lake"),
        (Filters(place=near), "place (by distance)"),
        (Filters(), ""),
    ]:
        assert ui_logic._describe(engine, filters) == expected
    empty = SearchResponse((), ParseResult(ParsedQuery("q", ""), Filters()), Filters(), 0)
    assert ui_logic._notices(engine, empty) == [
        "No photos match the filters.",
        "No visual description, so photos are listed newest first.",
    ]


def test_engine_module_helpers_are_used_by_default(engine: FakeEngine):
    result = run_search(engine, "", date_from="2023", date_to="2023")
    assert "dates 2023-01-01 to 2023-12-31" in result.reading
    assert any("No visual description" in n for n in result.notices)


# --------------------------------------------------------------------------- clarification


def test_first_run_asks_and_the_answer_run_never_does(engine: FakeEngine, thumbs):
    engine.ask = [unresolved("date", "Diwali 2023"), unresolved("place", "Narnia")]
    session = Session()
    first = start_search(engine, session, ui_logic.SearchParams(query="lights"), thumbs=thumbs)
    assert first.needs_clarification and first.cards == []
    assert first.questions == [
        ("Diwali 2023", 'What do you mean by "Diwali 2023"?'),
        ("Narnia", 'What do you mean by "Narnia"?'),
    ]
    assert session.pending == first.questions and not session.answered
    assert first.status == "I need a little help understanding your query."
    assert engine.requests[0].allow_clarification is True

    second = answer_clarifications(engine, session, ["November 2023", ""], thumbs=thumbs)
    req = engine.requests[1]
    assert req.allow_clarification is False
    assert dict(req.clarifications) == {"Diwali 2023": "November 2023", "Narnia": ""}
    assert not second.needs_clarification and second.questions == []
    assert len(second.cards) == 5
    assert second.dropped == [
        'I did not understand "Diwali 2023", so I searched without it.',
        'I did not understand "Narnia", so I searched without it.',
    ]
    assert session.pending == [] and session.answered and session.result is second
    # notices already covered by the dropped banner are not repeated
    assert not any("Diwali" in n for n in second.notices)


def test_questions_are_not_repeated_in_the_notices(engine: FakeEngine):
    engine.ask = [unresolved("date", "Diwali 2023")]
    engine.explain = lambda r: [  # type: ignore[attr-defined]
        "kept note",
        *[q.question for q in r.needs_clarification],
    ]
    first = run_search(engine, "lights")
    assert first.notices == ["kept note"]


def test_skip_sends_every_phrase_as_asked_with_blank_replies(engine: FakeEngine):
    engine.ask = [unresolved("person", "Zed")]
    session = Session()
    start_search(engine, session, ui_logic.SearchParams(query="x", k=2))
    result = skip_clarifications(engine, session)
    req = engine.requests[-1]
    assert dict(req.clarifications) == {"Zed": ""} and req.allow_clarification is False
    assert len(result.cards) == 2 and result.dropped


def test_replies_as_mapping_short_lists_and_none(engine: FakeEngine):
    engine.ask = [unresolved("date", "Diwali 2023"), unresolved("place", "Narnia")]
    session = Session()
    start_search(engine, session, ui_logic.SearchParams(query="x"))
    answer_clarifications(engine, session, {"  diwali   2023 ": "  Nov  2023 "})
    assert dict(engine.requests[-1].clarifications) == {"Diwali 2023": "Nov 2023", "Narnia": ""}
    session.pending = [("Narnia", "q")]
    answer_clarifications(engine, session, None)
    assert dict(engine.requests[-1].clarifications) == {"Narnia": ""}
    session.pending = [("a", "q"), ("b", "q")]
    answer_clarifications(engine, session, ["only one"])
    assert dict(engine.requests[-1].clarifications) == {"a": "only one", "b": ""}


def test_second_run_reports_even_a_misbehaving_engine_that_asks_again(engine: FakeEngine):
    ask = unresolved("date", "Diwali 2023")

    def always_asks(req: SearchRequest) -> SearchResponse:
        parse = ParseResult(ParsedQuery(req.query, "x"), Filters(), (ask,))
        return SearchResponse((), parse, Filters(), 0, needs_clarification=(ask,))

    engine.respond = always_asks
    session = Session()
    first = start_search(engine, session, ui_logic.SearchParams(query="x"))
    assert first.needs_clarification
    second = answer_clarifications(engine, session, ["still odd"])
    assert second.questions == [] and not second.needs_clarification
    assert second.dropped == ['I did not understand "Diwali 2023", so I searched without it.']


def test_answering_with_nothing_pending_is_a_friendly_error(engine: FakeEngine):
    session = Session()
    result = answer_clarifications(engine, session, ["x"])
    assert result.error and "nothing to answer" in result.error
    assert engine.requests == []


def test_a_new_search_resets_the_clarification_state(engine: FakeEngine):
    engine.ask = [unresolved("date", "Diwali 2023")]
    session = Session()
    start_search(engine, session, ui_logic.SearchParams(query="x"))
    answer_clarifications(engine, session, [""])
    assert session.answered
    start_search(engine, session, ui_logic.SearchParams(query="y"))
    assert not session.answered and session.pending and session.query == "y"


def test_bad_date_on_start_leaves_nothing_pending(engine: FakeEngine):
    session = Session()
    result = start_search(engine, session, ui_logic.SearchParams(query="x", date_from="zzzz"))
    assert result.error and session.pending == [] and session.result is result


# --------------------------------------------------------------------------- feedback


def test_record_feedback_writes_to_the_store(engine: FakeEngine, store: Store):
    assert record_feedback(engine, " blue sea ", 2, True) == "Thanks, marked as relevant."
    assert record_feedback(engine, "blue sea", 3, False) == "Thanks, marked as not relevant."
    rows = store.feedback_rows()
    assert [(r.query, r.photo_id, r.relevant) for r in rows] == [
        ("blue sea", 2, True),
        ("blue sea", 3, False),
    ]


def test_record_feedback_failures_are_messages(engine: FakeEngine):
    assert "unknown photo id 999" in record_feedback(engine, "q", 999, True)
    engine.store.close()
    assert record_feedback(engine, "q", 1, True).startswith("Could not save your rating (")


# --------------------------------------------------------------------------- people, details


def test_people_names_appear_on_cards_and_details(engine: FakeEngine):
    engine.store.names_for_photos = lambda ids: {1: ["Bob", "Alice"]}  # type: ignore[method-assign]
    result = run_search(engine, "x", k=2)
    assert result.cards[0].people == ("Bob", "Alice") and result.cards[1].people == ()
    rows = dict(result.cards[0].details())
    assert rows["People"] == "Bob, Alice" and rows["Score"] == "1.000"
    assert rows["File"] == "IMG_0.jpg" and rows["Folder"] == "trip"  # the name, not the path
    assert os.path.dirname(result.cards[0].path) not in "".join(rows.values())
    assert rows["Camera"] == "Pixel 8"

    def boom(ids):
        raise RuntimeError("db")

    engine.store.names_for_photos = boom  # type: ignore[method-assign]
    assert run_search(engine, "x", k=1).cards[0].people == ()


def test_date_text_by_precision_and_source(tmp_path: Path):
    p = make_jpeg(tmp_path / "a.jpg")

    def text(**kw) -> str:
        from facesorter.search.types import IndexedPhoto

        return ui_logic._date_text(IndexedPhoto(1, photo_meta(p, **kw)))

    assert text() == "2023-07-14"
    assert text(date_precision="month") == "2023-07"
    assert text(date_precision="year") == "2023"
    assert text(date_source="mtime") == "2023-07-14 (file date)"
    assert text(taken=None) == ""
    assert text(date_source="none") == ""


def test_hitcard_details_skip_empty_values():
    card = HitCard(1, "/x/a/b.jpg", "b.jpg", "a", None, "cap", 0.5)
    assert card.details() == [("File", "b.jpg"), ("Score", "0.500"), ("Folder", "a")]


def test_md_escape_neutralises_markdown_and_html():
    nasty = "![x](http://evil/a.png) <b>*_`#"
    escaped = ui_logic.md_escape(nasty)
    assert "](" not in escaped.replace("\\]\\(", "")
    assert "<" not in escaped.replace("\\<", "")
    assert ui_logic.md_escape("plain words") == "plain words"


def test_md_escape_neutralises_newlines_and_autolinks():
    hostile = "a.jpg\n\n# Heading\r\n- item\t| cell\x0b\u2028```\nhttp://localhost/x me@host"
    escaped = ui_logic.md_escape(hostile)
    assert not any(c in escaped for c in "\r\n\t\x0b\x0c\x85\u2028\u2029")
    assert "\n" not in escaped and "://" not in escaped and "@" not in escaped.replace("\\@", "")
    assert escaped.startswith("a\\.jpg ")  # separators collapse to one space
    assert ui_logic.md_escape("http://localhost/x") == "http\\:\\/\\/localhost\\/x"


# --------------------------------------------------------------------------- unreadable files


def test_missing_photos_get_a_placeholder_and_a_notice(engine: FakeEngine, thumbs, library):
    (library / "IMG_1.jpg").unlink()
    (library / "IMG_2.jpg").write_bytes(b"not an image")
    result = run_search(engine, "x", k=4, thumbs=thumbs)
    ok = [c.thumb_ok for c in result.cards]
    assert ok == [True, False, False, True]
    assert result.cards[1].thumb_path == str(thumbs.placeholder())  # and it is a placeholder
    assert any(n.startswith("2 photos could not be read") for n in result.notices)
    (library / "IMG_0.jpg").unlink()
    one = run_search(engine, "x", k=1, thumbs=ThumbnailCache(thumbs.directory.parent / "t2"))
    assert any(n.startswith("1 photo could not be read") for n in one.notices)


# --------------------------------------------------------------------------- footer + safety


def test_index_summary_shows_counts_and_dates_but_never_the_library_location(
    engine: FakeEngine, store: Store
):
    store.set_meta("library_roots", '["/Users/me/Pictures/private"]')
    store.set_meta("indexed_at", "2025-01-02")
    assert ui_logic.index_summary(engine) == "5 photos indexed, dated 2023-07-10 to 2023-07-14."
    assert "private" not in ui_logic.index_summary(engine)
    store.close()
    assert ui_logic.index_summary(engine) == "Index statistics are not available."
    assert ui_logic.index_summary(object()) == "Index statistics are not available."


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("::1", True),
        ("[::1]", True),
        ("127.5.5.5", True),
        ("0.0.0.0", False),
        ("192.168.1.5", False),
        ("", False),
        ("example.com", False),
        ("::", False),
    ],
)
def test_is_loopback_host(host, loopback):
    assert ui_logic.is_loopback_host(host) is loopback


def test_ensure_outside_library(tmp_path: Path):
    root = tmp_path / "photos"
    root.mkdir()
    cache = tmp_path / "cache"
    assert ui_logic.ensure_outside_library(cache, [str(root)]) == cache.resolve()
    for bad in (root, root / "thumbs", tmp_path):  # same, inside, and an ancestor of the library
        with pytest.raises(ValueError, match="overlaps the photo library"):
            ui_logic.ensure_outside_library(bad, [str(root)])
    assert ui_logic.ensure_outside_library(cache, []) == cache.resolve()


def test_default_thumbs_dir_is_under_home_and_not_created(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert ui_logic.default_thumbs_dir() == tmp_path / ".facesorter" / "thumbs"
    assert not (tmp_path / ".facesorter").exists()
    cache = ThumbnailCache()
    assert cache.directory == (tmp_path / ".facesorter" / "thumbs").resolve()
    assert cache.directory.is_dir()


# --------------------------------------------------------------------------- thumbnails


def test_thumbnail_is_a_small_jpeg_named_by_hash(thumbs: ThumbnailCache, tmp_path: Path):
    src = make_jpeg(tmp_path / "big.jpg", size=(2000, 1500))
    meta = photo_meta(src)
    out = thumbs.get(meta)
    assert out is not None and out == thumbs.path_for(meta)
    assert out.name == f"{ThumbnailCache.key(meta.path, meta.mtime_ns, meta.size)}.jpg"
    with Image.open(out) as img:
        assert img.format == "JPEG" and img.size == (512, 384)
    assert os.stat(out).st_mode & 0o077 == 0 or os.name == "nt"
    assert thumbs.directory.stat().st_mode & 0o777 == 0o700 or os.name == "nt"


def test_key_changes_with_path_mtime_and_size_and_canonicalises():
    base = ThumbnailCache.key("/a/b.jpg", 1, 2)
    assert base == ThumbnailCache.key("/a/./b.jpg", 1, 2)
    assert (
        len(
            {
                base,
                ThumbnailCache.key("/a/c.jpg", 1, 2),
                ThumbnailCache.key("/a/b.jpg", 2, 2),
                ThumbnailCache.key("/a/b.jpg", 1, 3),
            }
        )
        == 4
    )


def test_unchanged_photo_is_decoded_once_and_changed_photo_gets_a_new_thumbnail(
    thumbs: ThumbnailCache, tmp_path: Path, monkeypatch
):
    calls: list[str] = []
    real = ui_logic.load_for_model
    monkeypatch.setattr(
        ui_logic, "load_for_model", lambda p, target: (calls.append(p), real(p, target))[1]
    )
    src = make_jpeg(tmp_path / "a.jpg")
    meta = photo_meta(src)
    first = thumbs.get(meta)
    assert thumbs.get(meta) == first and len(calls) == 1
    src.unlink()  # the cached thumbnail still answers for the indexed version
    assert thumbs.get(meta) == first and len(calls) == 1
    edited = photo_meta(make_jpeg(src, color=(1, 2, 3)))
    edited = PhotoMeta(**{**edited.__dict__, "mtime_ns": meta.mtime_ns + 1})
    assert thumbs.get(edited) not in (None, first) and len(calls) == 2


def test_exif_rotation_is_applied(thumbs: ThumbnailCache, tmp_path: Path):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 clockwise to display
    src = make_jpeg(tmp_path / "rot.jpg", size=(80, 40), exif=exif)
    out = thumbs.get(photo_meta(src))
    with Image.open(out) as img:
        assert img.size == (40, 80)


def test_heic_photos_get_thumbnails(thumbs: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "x.heic"
    pillow_heif.from_pillow(Image.new("RGB", (1280, 960), (10, 200, 30))).save(str(src), quality=80)
    out = thumbs.get(photo_meta(src))
    assert out is not None
    with Image.open(out) as img:
        assert img.format == "JPEG" and img.size == (512, 384)


def test_thumbnails_fit_a_square_box_whatever_the_shape(tmp_path: Path):
    cache = ThumbnailCache(tmp_path / "t", size=64)
    for name, size, expected in (
        ("pano", (2000, 100), (64, 3)),
        ("tall", (300, 900), (21, 64)),
        ("small", (40, 30), (40, 30)),  # never enlarged
    ):
        src = make_jpeg(tmp_path / f"{name}.jpg", size=size)
        with Image.open(cache.get(photo_meta(src))) as img:
            assert img.size == expected, name


def test_unreadable_and_missing_files_return_none_and_leave_no_debris(
    thumbs: ThumbnailCache, tmp_path: Path
):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8 this is not a jpeg")
    assert thumbs.get(photo_meta(bad)) is None
    assert thumbs.get(photo_meta(bad, taken=None)) is None
    gone = make_jpeg(tmp_path / "gone.jpg")
    meta = photo_meta(gone)
    gone.unlink()
    assert thumbs.get(meta) is None
    assert list(thumbs.directory.iterdir()) == []


def test_write_is_atomic_when_the_final_rename_fails(
    thumbs: ThumbnailCache, tmp_path: Path, monkeypatch
):
    src = make_jpeg(tmp_path / "a.jpg")

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(ui_logic.os, "replace", broken)
    assert thumbs.get(photo_meta(src)) is None
    assert list(thumbs.directory.iterdir()) == []  # neither a final file nor a stray temp file


def test_write_failure_while_encoding_leaves_nothing(
    thumbs: ThumbnailCache, tmp_path: Path, monkeypatch
):
    src = make_jpeg(tmp_path / "a.jpg")

    def broken(self, *a, **k):
        raise OSError("encoder died")

    monkeypatch.setattr(Image.Image, "save", broken)
    assert thumbs.get(photo_meta(src)) is None
    assert list(thumbs.directory.iterdir()) == []


def test_temp_files_never_look_like_thumbnails(thumbs: ThumbnailCache, tmp_path: Path):
    src = make_jpeg(tmp_path / "a.jpg")
    seen: list[list[str]] = []
    real = os.replace

    def spy(a, b):
        seen.append(sorted(os.listdir(thumbs.directory)))
        return real(a, b)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ui_logic.os, "replace", spy)
        thumbs.get(photo_meta(src))
    (listing,) = seen
    assert len(listing) == 1 and listing[0].startswith(".tmp-") and listing[0].endswith(".part")
    assert not ui_logic._THUMB_NAME.match(listing[0])


def test_stale_temp_files_are_removed_on_start_fresh_ones_kept(tmp_path: Path):
    d = tmp_path / "t"
    d.mkdir()
    old, fresh = d / ".tmp-old.part", d / ".tmp-new.part"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    long_ago = time.time() - 7200
    os.utime(old, (long_ago, long_ago))
    ThumbnailCache(d)
    assert not old.exists() and fresh.exists()


def _fill(cache: ThumbnailCache, tmp_path: Path, n: int) -> list[PhotoMeta]:
    rng = np.random.default_rng(1)
    metas = []
    for i in range(n):
        path = tmp_path / f"n{i}.png"  # noise: roughly the same compressed size each time
        Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(path)
        metas.append(photo_meta(path))
    return metas


def test_eviction_removes_least_recently_used_first(tmp_path: Path):
    cache = ThumbnailCache(tmp_path / "t", max_bytes=10**9, size=128)
    metas = _fill(cache, tmp_path, 5)
    paths = [cache.get(m) for m in metas[:4]]
    assert all(paths)
    for age, p in zip((400, 300, 200, 100), paths):  # a is oldest ... d is newest
        os.utime(p, (time.time() - age,) * 2)
    unit = max(p.stat().st_size for p in paths)
    cache.get(metas[0])  # touching `a` makes it the most recently used
    cache.max_bytes = int(unit * 4.3)  # the fifth thumbnail pushes past the cap
    fifth = cache.get(metas[4])
    survivors = {p.name for p in cache.directory.glob("*.jpg")}
    assert fifth.name in survivors and paths[0].name in survivors
    assert paths[1].name not in survivors  # the least recently used went first
    assert cache.total_bytes() <= cache.max_bytes
    cache.max_bytes = 1  # a cap smaller than one thumbnail: keep only what was just written
    last = cache.get(_fill(cache, tmp_path / ".", 6)[5])
    assert {p.name for p in cache.directory.glob("*.jpg")} == {last.name}


def test_eviction_survives_a_file_that_cannot_be_deleted(tmp_path: Path, monkeypatch):
    cache = ThumbnailCache(tmp_path / "t", max_bytes=10**9, size=128)
    metas = _fill(cache, tmp_path, 3)
    for m in metas[:2]:
        cache.get(m)
    cache.max_bytes = 1
    monkeypatch.setattr(ui_logic.os, "unlink", lambda p: (_ for _ in ()).throw(PermissionError()))
    assert cache.get(metas[2]) is not None  # eviction fails quietly; the new thumbnail is fine


def test_eviction_ignores_foreign_files_and_the_placeholder(tmp_path: Path):
    cache = ThumbnailCache(tmp_path / "t", max_bytes=1, size=128)
    keep = cache.directory / "notes.txt"
    keep.write_text("mine")
    placeholder = cache.placeholder()
    (src,) = _fill(cache, tmp_path, 1)
    assert cache.get(src) is not None
    assert keep.exists() and placeholder.exists()
    assert cache.placeholder() == placeholder
    with Image.open(placeholder) as img:
        assert img.format == "JPEG"


def test_total_bytes_tracks_existing_files_on_restart(tmp_path: Path):
    cache = ThumbnailCache(tmp_path / "t", size=128)
    for m in _fill(cache, tmp_path, 3):
        cache.get(m)
    used = cache.total_bytes()
    assert used > 0 and ThumbnailCache(tmp_path / "t")._total == used


def test_invalid_cache_settings():
    with pytest.raises(ValueError, match="max_bytes"):
        ThumbnailCache(max_bytes=0)
    with pytest.raises(ValueError, match="size"):
        ThumbnailCache(size=4)


def test_concurrent_requests_build_each_thumbnail_once(tmp_path: Path, monkeypatch):
    cache = ThumbnailCache(tmp_path / "t", size=128)
    metas = _fill(cache, tmp_path, 4)
    lock = threading.Lock()
    calls: list[str] = []
    real = ui_logic.load_for_model

    def slow(path, target):
        with lock:
            calls.append(str(path))
        time.sleep(0.02)
        return real(path, target)

    monkeypatch.setattr(ui_logic, "load_for_model", slow)
    barrier = threading.Barrier(12)
    results: list[Path | None] = []
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait()
            got = cache.get(metas[i % 4])
            with lock:
                results.append(got)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 12 and all(results)
    assert sorted(calls) == sorted(m.path for m in metas)  # once per photo, not per thread
    for p in cache.directory.glob("*.jpg"):
        with Image.open(p) as img:
            img.load()
    assert not list(cache.directory.glob(".tmp-*"))


def test_get_many_keeps_order_and_runs_in_parallel(thumbs: ThumbnailCache, tmp_path: Path):
    metas = _fill(thumbs, tmp_path, 6)
    gone = photo_meta(make_jpeg(tmp_path / "x.jpg"))
    (tmp_path / "x.jpg").unlink()
    out = thumbs.get_many([*metas[:3], gone, *metas[3:]])
    assert [p is None for p in out] == [False] * 3 + [True] + [False] * 3
    assert out[:3] == [thumbs.path_for(m) for m in metas[:3]]
    assert thumbs.get_many([]) == []
    assert thumbs.get_many(metas[:1]) == [thumbs.path_for(metas[0])]


def test_eviction_under_concurrent_writers_stays_consistent(tmp_path: Path):
    cache = ThumbnailCache(tmp_path / "t", size=128)
    metas = _fill(cache, tmp_path, 12)
    one = cache.get(metas[0]).stat().st_size
    cache.max_bytes = one * 5
    threads = [threading.Thread(target=cache.get_many, args=(metas[i::3],)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.total_bytes() <= cache.max_bytes
    assert cache._total == cache.total_bytes()


# --------------------------------------------------------------------------- real engine


def test_with_the_real_engine_end_to_end(tmp_path: Path, library: Path, thumbs):
    pytest.importorskip("dateparser")
    from facesorter.search.engine import Engine
    from facesorter.search.parser import QueryParser
    from facesorter.search.testing import (
        COLORS,
        FAKE_DIM,
        FAKE_MODEL_NAME,
        FakeImageEncoder,
        FakeTextEncoder,
        make_color_image,
    )

    store = Store.open(tmp_path / "real.sqlite", model_name=FAKE_MODEL_NAME, dim=FAKE_DIM)
    encoder = FakeImageEncoder()
    records = []
    for i, color in enumerate(COLORS):
        path = library / f"{color}.jpg"
        make_color_image(color, (64, 48)).save(path)
        vec = encoder.encode_images([make_color_image(color)])[0]
        records.append((photo_meta(path, f"2022-0{1 + i % 8}-15T09:00:00"), None, vec))
    store.upsert_photos(records)
    real = Engine(store, FakeTextEncoder(), QueryParser(None, None, store))
    try:
        session = Session()
        params = ui_logic.SearchParams(query="red lights at Diwali 2023", k=3)
        first = start_search(real, session, params, thumbs=thumbs)
        assert first.questions and first.questions[0][0] == "Diwali 2023"
        assert first.cards == []
        second = answer_clarifications(real, session, [""], thumbs=thumbs)
        assert second.questions == [] and len(second.cards) == 3
        assert second.cards[0].filename == "red.jpg"
        assert second.dropped == [] and any(
            'Skipped the date "Diwali 2023"' in n for n in second.notices
        )
        start_search(real, session, params, thumbs=thumbs)
        reworded = answer_clarifications(real, session, ["March 2022"], thumbs=thumbs)
        assert reworded.reading.splitlines()[0] == "Filters: dates 2022-03-01 to 2022-03-31."
        assert [c.filename for c in reworded.cards] == ["blue.jpg"]
        start_search(real, session, params, thumbs=thumbs)
        again = answer_clarifications(real, session, ["gibberish blorp"], thumbs=thumbs)
        assert again.questions == [] and "Diwali 2023" in again.dropped[0]
        assert len(again.cards) == 3
        filtered = run_search(real, "blue", date_from="2022-03", date_to="2022-03", thumbs=thumbs)
        assert [c.filename for c in filtered.cards] == ["blue.jpg"]
        assert filtered.reading.splitlines()[0] == "Filters: dates 2022-03-01 to 2022-03-31."
        assert record_feedback(real, "blue", filtered.cards[0].photo_id, True).startswith("Thanks")
        assert ui_logic.index_summary(real).startswith("8 photos indexed")
    finally:
        real.close()


# --------------------------------------------------------------------------- unwritable cache


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)
def test_a_read_only_thumbnail_directory_degrades_instead_of_raising(
    engine: FakeEngine, tmp_path: Path
):
    cache = ThumbnailCache(tmp_path / "ro")
    cache.directory.chmod(0o500)
    try:
        assert cache.placeholder() is None  # cannot be written, and must not raise
        result = run_search(engine, "x", k=3, thumbs=cache)
    finally:
        cache.directory.chmod(0o700)
    assert result.error is None and len(result.cards) == 3
    assert all(c.thumb_path is None and not c.thumb_ok for c in result.cards)
    assert result.gallery_items() == []
    assert any("3 photos could not be read" in n and "left out" in n for n in result.notices)


def test_a_failing_cache_never_breaks_a_search(engine: FakeEngine, thumbs: ThumbnailCache):
    def boom(metas, workers=4):
        raise OSError("disk exploded")

    thumbs.get_many = boom  # type: ignore[method-assign]
    result = run_search(engine, "x", k=2, thumbs=thumbs)
    assert result.error is None and len(result.cards) == 2
    assert [c.thumb_ok for c in result.cards] == [False, False]
    assert all(c.thumb_path == str(thumbs.placeholder()) for c in result.cards)


def test_eviction_errors_do_not_lose_the_new_thumbnail(
    thumbs: ThumbnailCache, tmp_path: Path, monkeypatch
):
    def broken(written, keep):
        raise OSError("scan failed")

    monkeypatch.setattr(thumbs, "_account", broken)
    meta = photo_meta(make_jpeg(tmp_path / "a.jpg"))
    assert thumbs.get(meta) == thumbs.path_for(meta) and thumbs.path_for(meta).exists()


def test_placeholder_creation_failure_returns_none(thumbs: ThumbnailCache, monkeypatch):
    monkeypatch.setattr(thumbs, "_write_jpeg", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert thumbs.placeholder() is None


# --------------------------------------------------------------------------- host guard


def test_host_policy_for_a_loopback_bind():
    policy = ui_logic.HostPolicy.for_bind("127.0.0.1", 7860)
    for good in (
        "localhost", "LOCALHOST", "Localhost:7860", "127.0.0.1", "127.0.0.1:7860",
        "[::1]", "[::1]:7860", "localhost.", "localhost.:7860",
    ):  # fmt: skip
        assert policy.host_allowed(good), good
    for bad in (
        None, "", "evil.com", "evil.com:7860", "127.0.0.1:1", "localhost:80", "[::1]:99",
        "127.0.0.1.evil.com", "evil.com#127.0.0.1", "127.0.0.1@evil.com", "a b",
        "localhost/x", "127.0.0.2", "192.168.1.5", "localhost:notaport", "[::1", ":7860",
        "localhost:99999",
    ):  # fmt: skip
        assert not policy.host_allowed(bad), bad


def test_host_policy_without_a_known_port_accepts_any_port():
    policy = ui_logic.HostPolicy.for_bind("localhost")
    assert policy.host_allowed("localhost:1") and policy.host_allowed("127.0.0.1:65535")
    assert not policy.host_allowed("evil.com:1")
    policy.port = 8000
    assert not policy.host_allowed("localhost:1") and policy.host_allowed("localhost:8000")


def test_host_policy_adds_the_configured_non_loopback_host():
    policy = ui_logic.HostPolicy.for_bind("Photos.Example.Org", 9000)
    assert policy.host_allowed("photos.example.org:9000") and policy.host_allowed("localhost")
    assert not policy.host_allowed("other.example.org:9000")
    assert not policy.host_allowed("192.168.1.9:9000")  # not a wildcard bind: no IP literals
    v6 = ui_logic.HostPolicy.for_bind("[fd00::5]", 9000)
    assert v6.host_allowed("[fd00::5]:9000") and not v6.host_allowed("[fd00::6]:9000")


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", ""])
def test_a_wildcard_bind_accepts_ip_literals_and_this_machine_but_never_other_names(
    wildcard, monkeypatch
):
    monkeypatch.setattr("socket.gethostname", lambda: "Laptop.Home")
    policy = ui_logic.HostPolicy.for_bind(wildcard, 9000)
    for good in ("192.168.1.9:9000", "[fd00::5]:9000", "10.0.0.1", "laptop.home:9000",
                 "laptop.local", "localhost:9000"):  # fmt: skip
        assert policy.host_allowed(good), good
    for bad in ("evil.com:9000", "192.168.1.9:1", "attacker.example", "laptop.evil.com"):
        assert not policy.host_allowed(bad), bad


def test_host_policy_origins():
    policy = ui_logic.HostPolicy.for_bind("127.0.0.1", 7860)
    assert policy.origin_allowed(None)  # no header: a non-browser client
    for good in ("http://127.0.0.1:7860", "http://localhost:7860", "http://[::1]:7860/",
                 "https://localhost:7860"):  # fmt: skip
        assert policy.origin_allowed(good), good
    for bad in ("null", "", "http://evil.com", "http://evil.com:7860", "http://127.0.0.1:9",
                "file://", "ftp://localhost", "http://localhost:7860/path",
                "http://localhost:7860?x", "http://localhost:7860#x", "http://[::1"):  # fmt: skip
        assert not policy.origin_allowed(bad), bad


def asgi(app, *, host="127.0.0.1:7860", origin=None, scope_type="http", method="GET",
         extra=(), hosts=None):  # fmt: skip
    """Drive ``app`` with a hand-made scope; return ``(status_or_None, body, reached_app)``."""
    import asyncio

    headers = [(b"host", h.encode()) for h in ([host] if hosts is None else hosts)]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    headers.extend(extra)
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        sent.append(message)

    scope = {"type": scope_type, "method": method, "path": "/", "headers": headers}
    asyncio.run(app(scope, receive, send))
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return (start["status"] if start else None), body, sent


class _Marker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, scope, receive, send):
        self.calls.append(scope["type"])
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"hello"})


def guarded(port=7860, host="127.0.0.1"):
    inner = _Marker()
    return inner, ui_logic.HostGuardMiddleware(inner, ui_logic.HostPolicy.for_bind(host, port))


def test_guard_passes_requests_addressed_to_this_server():
    inner, app = guarded()
    for host in ("127.0.0.1:7860", "localhost:7860", "[::1]:7860", "LOCALHOST", "127.0.0.1"):
        status, body, _ = asgi(app, host=host)
        assert (status, body) == (200, b"hello"), host
    for method in ("POST", "PUT", "DELETE", "GET"):
        origin = "http://localhost:7860"
        assert asgi(app, method=method, origin=origin)[0] == 200
    assert len(inner.calls) == 9


def test_guard_refuses_foreign_hosts_on_every_method_without_reaching_the_app():
    inner, app = guarded()
    for method in ("GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS"):
        status, body, sent = asgi(app, host="evil.com:7860", method=method)
        assert status == 403 and b"Forbidden" in body, method
        headers = dict(sent[0]["headers"])
        assert headers[b"content-length"] == str(len(body)).encode()
    assert inner.calls == []


def test_guard_refuses_missing_duplicate_and_garbled_host_headers():
    inner, app = guarded()
    assert asgi(app, hosts=[])[0] == 403
    assert asgi(app, hosts=["127.0.0.1:7860", "127.0.0.1:7860"])[0] == 403
    assert asgi(app, hosts=["evil.com:7860", "127.0.0.1:7860"])[0] == 403
    assert asgi(app, host="127.0.0.1:7860@evil.com")[0] == 403
    assert inner.calls == []


def test_guard_refuses_foreign_origins_even_with_a_good_host():
    inner, app = guarded()
    for origin in ("http://evil.com:7860", "http://evil.com", "null", "http://127.0.0.1:81"):
        for method in ("POST", "GET"):
            assert asgi(app, method=method, origin=origin)[0] == 403, (method, origin)
    status, _, _ = asgi(app, extra=[(b"origin", b"http://localhost:7860"),
                                    (b"origin", b"http://localhost:7860")])  # fmt: skip
    assert status == 403  # two Origin headers are never legitimate
    assert inner.calls == []


def test_guard_covers_websockets_and_passes_lifespan_through():
    inner, app = guarded()
    status, _, sent = asgi(app, host="evil.com", scope_type="websocket")
    assert status is None and sent == [{"type": "websocket.close", "code": 1008}]
    _, _, sent = asgi(app, scope_type="websocket", origin="http://evil.com")
    assert sent == [{"type": "websocket.close", "code": 1008}]
    assert inner.calls == []
    asgi(app, scope_type="websocket")
    assert inner.calls == ["websocket"]

    import asyncio

    asyncio.run(app({"type": "lifespan"}, None, None))
    assert inner.calls == ["websocket", "lifespan"]


def test_guard_follows_a_port_that_becomes_known_later():
    policy = ui_logic.HostPolicy.for_bind("127.0.0.1")
    app = ui_logic.HostGuardMiddleware(_Marker(), policy)
    assert asgi(app, host="127.0.0.1:5000")[0] == 200
    policy.port = 6000
    assert asgi(app, host="127.0.0.1:5000")[0] == 403
    assert asgi(app, host="127.0.0.1:6000")[0] == 200
