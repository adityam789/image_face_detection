"""Tests for facesorter.search.parser: extraction, grounding, resolution, flags, clarification."""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta

import pytest

from facesorter.search.parser import (
    QueryParser,
    _People,
    _Query,
    clarification_key,
    is_asked,
)
from facesorter.search.prompts import SYSTEM_PROMPT, build_messages
from facesorter.search.testing import ScriptedLLM
from facesorter.search.types import (
    DateRange,
    Filters,
    ParseResult,
    PlaceFilter,
    SearchRequest,
)

TODAY = date(2026, 11, 7)  # a Saturday


# ----------------------------------------------------------------- fakes


_PLACE_NAMES = (
    "Goa Tokyo Switzerland Paris Berlin Bangkok Como Whistler Nice Reading Orange Hope Mobile "
    "London Kyoto Rome Springfield"
).split() + ["Key West", "St. Louis", "Isle of Skye", "Paris, France", "New York"]


class FakeGazetteer:
    """Exact (case-insensitive) name lookup; counts calls."""

    def __init__(self, extra: tuple[str, ...] = (), fail: bool = False) -> None:
        self.known = {n.lower(): PlaceFilter(city=n, text=n) for n in (*_PLACE_NAMES, *extra)}
        self.fail = fail
        self.calls: list[str] = []

    def label_for(self, lat, lon):  # pragma: no cover - protocol filler
        return None

    def resolve(self, name):
        self.calls.append(name)
        if self.fail:
            raise RuntimeError("gazetteer exploded")
        return self.known.get(name.lower())


class FakePeople:
    def __init__(self, *names: str, fail: bool = False) -> None:
        self.names = list(names)
        self.fail = fail

    def known_names(self):
        if self.fail:
            raise RuntimeError("people exploded")
        return list(self.names)


NAMES = ("Bob", "Alice", "Sarah", "Mom", "John", "Mike", "Dave", "Priya", "Mary Ann")


def make_parser(llm=None, gazetteer="default", people="default", **kw) -> QueryParser:
    gaz = FakeGazetteer() if gazetteer == "default" else gazetteer
    ppl = FakePeople(*NAMES) if people == "default" else people
    return QueryParser(llm, gaz, ppl, **kw)


def run(query: str, parser: QueryParser | None = None, **kw) -> ParseResult:
    parser = parser or make_parser()
    kw.setdefault("today", TODAY)
    return parser.parse(SearchRequest(query=query, **kw))


def rng(res: ParseResult):
    dr = res.filters.date_range
    return (dr.start.isoformat(), dr.end.isoformat()) if dr else None


# ---------------------------------------------------------- labelled queries (research/qset.py)

QSET = [
    # (query, text, place, people, date_from, date_to)
    ("photos from yesterday", None, None, [], "2026-11-06", "2026-11-06"),
    ("last month", None, None, [], "2026-10-01", "2026-10-31"),
    ("in March 2024", None, None, [], "2024-03-01", "2024-03-31"),
    ("during 2023", None, None, [], "2023-01-01", "2023-12-31"),
    ("last summer", None, None, [], "2026-06-01", "2026-08-31"),
    ("summer 2019", None, None, [], "2019-06-01", "2019-08-31"),
    ("Christmas 2022", None, None, [], "2022-12-24", "2022-12-26"),
    ("last year", None, None, [], "2025-01-01", "2025-12-31"),
    ("this week", None, None, [], "2026-11-02", "2026-11-08"),
    ("between June and August 2021", None, None, [], "2021-06-01", "2021-08-31"),
    ("New Year's Eve 2024", None, None, [], "2024-12-31", "2025-01-01"),
    ("last weekend", None, None, [], "2026-10-31", "2026-11-01"),
    ("Halloween 2023", None, None, [], "2023-10-30", "2023-11-01"),
    (
        "sunset at the beach in Goa, December 2022",
        "sunset at the beach",
        "Goa",
        [],
        "2022-12-01",
        "2022-12-31",
    ),
    (
        "Bob and Alice hiking last summer",
        "hiking",
        None,
        ["Bob", "Alice"],
        "2026-06-01",
        "2026-08-31",
    ),
    (
        "my dog playing in the snow in 2021",
        "dog playing in the snow",
        None,
        [],
        "2021-01-01",
        "2021-12-31",
    ),
    ("food in Tokyo", "food", "Tokyo", [], None, None),
    ("photos of Sarah at a wedding", "wedding", None, ["Sarah"], None, None),
    ("cats", "cats", None, [], None, None),
    (
        "mountain views from Switzerland in July 2023",
        "mountain views",
        "Switzerland",
        [],
        "2023-07-01",
        "2023-07-31",
    ),
    ("graduation ceremony May 2025", "graduation ceremony", None, [], "2025-05-01", "2025-05-31"),
    (
        "Mom at the Eiffel Tower in Paris last year",
        "Eiffel Tower",
        "Paris",
        ["Mom"],
        "2025-01-01",
        "2025-12-31",
    ),
    ("red car", "red car", None, [], None, None),
    (
        "birthday cake photos from November 2025",
        "birthday cake",
        None,
        [],
        "2025-11-01",
        "2025-11-30",
    ),
    (
        "a group photo with John, Mike and Dave in Berlin",
        "group photo",
        "Berlin",
        ["John", "Mike", "Dave"],
        None,
        None,
    ),
    ("screenshots from last week", "screenshots", None, [], "2026-10-26", "2026-11-01"),
    (
        "kids playing at the park yesterday",
        "kids playing at the park",
        None,
        [],
        "2026-11-06",
        "2026-11-06",
    ),
    ("street food in Bangkok 2019", "street food", "Bangkok", [], "2019-01-01", "2019-12-31"),
    ("the lake in Como, 2020", "lake", "Como", [], "2020-01-01", "2020-12-31"),
    (
        "Priya skiing in Whistler in February 2024",
        "skiing",
        "Whistler",
        ["Priya"],
        "2024-02-01",
        "2024-02-29",
    ),
]


# fmt: off
# ruff: disable[E501]
# Real replies from local models on the 30 labelled queries (research spike, greedy decoding).
_REPLAYS: dict[str, dict[str, dict[str, object]]] = {
    "qwen25_15b": {
        "photos from yesterday": {'text': '', 'place': None, 'people': [], 'date_phrase': 'yesterday'},
        "last month": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last month'},
        "in March 2024": {'text': '', 'place': None, 'people': [], 'date_phrase': 'in March 2024'},
        "during 2023": {'text': '', 'place': None, 'people': [], 'date_phrase': '2023'},
        "last summer": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last summer'},
        "summer 2019": {'text': '', 'place': None, 'people': [], 'date_phrase': 'summer 2019'},
        "Christmas 2022": {'text': '', 'place': None, 'people': [], 'date_phrase': 'Christmas 2022'},
        "last year": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last year'},
        "this week": {'text': '', 'place': None, 'people': [], 'date_phrase': 'this week'},
        "between June and August 2021": {'text': '', 'place': None, 'people': [], 'date_phrase': 'between June and August 2021'},
        "New Year's Eve 2024": {'text': "New Year's Eve", 'place': None, 'people': [], 'date_phrase': "New Year's Eve 2024"},
        "last weekend": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last weekend'},
        "Halloween 2023": {'text': 'Halloween', 'place': None, 'people': [], 'date_phrase': 'Halloween 2023'},
        "sunset at the beach in Goa, December 2022": {'text': 'sunset', 'place': 'Goa', 'people': [], 'date_phrase': 'December 2022'},
        "Bob and Alice hiking last summer": {'text': '', 'place': None, 'people': ['Bob', 'Alice'], 'date_phrase': 'last summer'},
        "my dog playing in the snow in 2021": {'text': 'my dog playing in the snow', 'place': None, 'people': [], 'date_phrase': '2021'},
        "food in Tokyo": {'text': 'food', 'place': 'Tokyo', 'people': [], 'date_phrase': None},
        "photos of Sarah at a wedding": {'text': 'Sarah at a wedding', 'place': None, 'people': ['Sarah'], 'date_phrase': None},
        "cats": {'text': 'cats', 'place': None, 'people': [], 'date_phrase': None},
        "mountain views from Switzerland in July 2023": {'text': 'mountain views', 'place': 'Switzerland', 'people': [], 'date_phrase': 'July 2023'},
        "graduation ceremony May 2025": {'text': 'graduation ceremony', 'place': None, 'people': [], 'date_phrase': 'May 2025'},
        "Mom at the Eiffel Tower in Paris last year": {'text': 'Mom at the Eiffel Tower in Paris last year', 'place': 'Paris', 'people': ['Mom'], 'date_phrase': 'last year'},
        "red car": {'text': 'red car', 'place': None, 'people': [], 'date_phrase': None},
        "birthday cake photos from November 2025": {'text': '', 'place': None, 'people': [], 'date_phrase': 'November 2025'},
        "a group photo with John, Mike and Dave in Berlin": {'text': 'a group photo', 'place': 'Berlin', 'people': ['John', 'Mike', 'Dave'], 'date_phrase': None},
        "screenshots from last week": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last week'},
        "kids playing at the park yesterday": {'text': 'kids playing', 'place': 'park', 'people': [], 'date_phrase': 'yesterday'},
        "street food in Bangkok 2019": {'text': 'street food', 'place': 'Bangkok', 'people': [], 'date_phrase': '2019'},
        "the lake in Como, 2020": {'text': 'lake', 'place': 'Como', 'people': [], 'date_phrase': '2020'},
        "Priya skiing in Whistler in February 2024": {'text': 's skiing', 'place': 'Whistler', 'people': ['Priya'], 'date_phrase': 'February 2024'},
    },
    "qwen3_06b": {
        "photos from yesterday": {'text': 'photos', 'place': None, 'people': [], 'date_phrase': 'yesterday'},
        "last month": {'text': 'last month', 'place': None, 'people': [], 'date_phrase': 'last month'},
        "in March 2024": {'text': 'March 2024', 'place': None, 'people': [], 'date_phrase': 'March 2024'},
        "during 2023": {'text': '', 'place': None, 'people': [], 'date_phrase': '2023'},
        "last summer": {'text': 'last summer', 'place': None, 'people': [], 'date_phrase': 'last summer'},
        "summer 2019": {'text': 'summer', 'place': None, 'people': [], 'date_phrase': '2019'},
        "Christmas 2022": {'text': 'Christmas', 'place': None, 'people': [], 'date_phrase': '2022'},
        "last year": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last year'},
        "this week": {'text': 'this week', 'place': None, 'people': [], 'date_phrase': 'this week'},
        "between June and August 2021": {'text': 'between June and August 2021', 'place': None, 'people': [], 'date_phrase': 'June 2021 to August 2021'},
        "New Year's Eve 2024": {'text': "New Year's Eve", 'place': None, 'people': [], 'date_phrase': '2024'},
        "last weekend": {'text': 'last weekend', 'place': None, 'people': [], 'date_phrase': 'last weekend'},
        "Halloween 2023": {'text': 'Halloween', 'place': None, 'people': [], 'date_phrase': '2023'},
        "sunset at the beach in Goa, December 2022": {'text': 'sunset', 'place': 'Goa', 'people': [], 'date_phrase': 'December 2022'},
        "Bob and Alice hiking last summer": {'text': 'hiking', 'place': None, 'people': ['Bob', 'Alice'], 'date_phrase': 'last summer'},
        "my dog playing in the snow in 2021": {'text': 'playing in the snow', 'place': None, 'people': [], 'date_phrase': '2021'},
        "food in Tokyo": {'text': 'food', 'place': 'Tokyo', 'people': [], 'date_phrase': None},
        "photos of Sarah at a wedding": {'text': 'photos', 'place': None, 'people': ['Sarah'], 'date_phrase': None},
        "cats": {'text': 'cats', 'place': None, 'people': [], 'date_phrase': None},
        "mountain views from Switzerland in July 2023": {'text': 'mountain views', 'place': 'Switzerland', 'people': [], 'date_phrase': 'July 2023'},
        "graduation ceremony May 2025": {'text': 'graduation ceremony', 'place': None, 'people': [], 'date_phrase': 'May 2025'},
        "Mom at the Eiffel Tower in Paris last year": {'text': 'Mom', 'place': 'Paris', 'people': ['Mom'], 'date_phrase': 'last year'},
        "red car": {'text': 'red', 'place': None, 'people': [], 'date_phrase': None},
        "birthday cake photos from November 2025": {'text': 'birthday cake photos', 'place': None, 'people': [], 'date_phrase': 'November 2025'},
        "a group photo with John, Mike and Dave in Berlin": {'text': 'group photo', 'place': 'Berlin', 'people': ['John', 'Mike', 'Dave'], 'date_phrase': None},
        "screenshots from last week": {'text': 'screenshots', 'place': None, 'people': [], 'date_phrase': 'last week'},
        "kids playing at the park yesterday": {'text': 'kids playing', 'place': None, 'people': [], 'date_phrase': 'yesterday'},
        "street food in Bangkok 2019": {'text': 'street food', 'place': 'Bangkok', 'people': [], 'date_phrase': '2019'},
        "the lake in Como, 2020": {'text': 'lake', 'place': 'Como', 'people': [], 'date_phrase': '2020'},
        "Priya skiing in Whistler in February 2024": {'text': 'skiing', 'place': 'Whistler', 'people': ['Priya'], 'date_phrase': 'February 2024'},
    },
    "qwen3_17b": {
        "photos from yesterday": {'text': '', 'place': None, 'people': [], 'date_phrase': 'yesterday'},
        "last month": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last month'},
        "in March 2024": {'text': '', 'place': None, 'people': [], 'date_phrase': 'March 2024'},
        "during 2023": {'text': '', 'place': None, 'people': [], 'date_phrase': '2023'},
        "last summer": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last summer'},
        "summer 2019": {'text': '', 'place': None, 'people': [], 'date_phrase': 'summer 2019'},
        "Christmas 2022": {'text': '', 'place': None, 'people': [], 'date_phrase': 'Christmas 2022'},
        "last year": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last year'},
        "this week": {'text': '', 'place': None, 'people': [], 'date_phrase': 'this week'},
        "between June and August 2021": {'text': '', 'place': None, 'people': [], 'date_phrase': 'between June and August 2021'},
        "New Year's Eve 2024": {'text': '', 'place': None, 'people': [], 'date_phrase': "New Year's Eve 2024"},
        "last weekend": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last weekend'},
        "Halloween 2023": {'text': '', 'place': None, 'people': [], 'date_phrase': 'Halloween 2023'},
        "sunset at the beach in Goa, December 2022": {'text': 'sunset', 'place': 'Goa', 'people': [], 'date_phrase': 'December 2022'},
        "Bob and Alice hiking last summer": {'text': '', 'place': None, 'people': ['Bob', 'Alice'], 'date_phrase': 'last summer'},
        "my dog playing in the snow in 2021": {'text': 'dog playing', 'place': None, 'people': ['my dog'], 'date_phrase': '2021'},
        "food in Tokyo": {'text': 'food', 'place': 'Tokyo', 'people': [], 'date_phrase': None},
        "photos of Sarah at a wedding": {'text': 'photos', 'place': None, 'people': ['Sarah'], 'date_phrase': 'at a wedding'},
        "cats": {'text': '', 'place': None, 'people': [], 'date_phrase': None},
        "mountain views from Switzerland in July 2023": {'text': 'mountain views', 'place': 'Switzerland', 'people': [], 'date_phrase': 'July 2023'},
        "graduation ceremony May 2025": {'text': '', 'place': None, 'people': [], 'date_phrase': 'May 2025'},
        "Mom at the Eiffel Tower in Paris last year": {'text': '', 'place': 'Paris', 'people': ['Mom'], 'date_phrase': 'last year'},
        "red car": {'text': '', 'place': None, 'people': [], 'date_phrase': 'red car'},
        "birthday cake photos from November 2025": {'text': '', 'place': None, 'people': [], 'date_phrase': 'November 2025'},
        "a group photo with John, Mike and Dave in Berlin": {'text': 'group photo', 'place': 'Berlin', 'people': ['John', 'Mike', 'Dave'], 'date_phrase': 'last winter'},
        "screenshots from last week": {'text': '', 'place': None, 'people': [], 'date_phrase': 'last week'},
        "kids playing at the park yesterday": {'text': '', 'place': None, 'people': [], 'date_phrase': 'yesterday'},
        "street food in Bangkok 2019": {'text': 'street food', 'place': 'Bangkok', 'people': [], 'date_phrase': '2019'},
        "the lake in Como, 2020": {'text': 'lake', 'place': 'Como', 'people': [], 'date_phrase': '2020'},
        "Priya skiing in Whistler in February 2024": {'text': 'skiing', 'place': 'Whistler', 'people': ['Priya'], 'date_phrase': 'February 2024'},
    },
}
# ruff: enable[E501]
# fmt: on


def llm_for(model: str) -> ScriptedLLM:
    return ScriptedLLM({q: json.dumps(p) for q, p in _REPLAYS[model].items()})


def tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def text_ok(got: str, want: str | None) -> bool:
    g, w = tokens(got), tokens(want)
    if not g and not w:
        return True
    inter = len(g & w)
    if not inter:
        return False
    p, r = inter / len(g), inter / len(w)
    return 2 * p * r / (p + r) >= 0.5  # the spike's token-F1 criterion


def correct(res: ParseResult, expected) -> bool:
    _query, text, place, people, start, end = expected
    got_range = rng(res)
    want_range = (start, end) if start else None
    got_place = res.filters.place.text.lower() if res.filters.place else None
    return (
        text_ok(res.parsed.text, text)
        and got_place == (place.lower() if place else None)
        and list(res.filters.people) == people
        and got_range == want_range
        and not res.unresolved
    )


def accuracy(parser: QueryParser) -> int:
    return sum(correct(run(q[0], parser), q) for q in QSET)


# ----------------------------------------------------------------- end to end on the labelled set


def test_rules_mode_labelled_queries():
    parser = make_parser(None)
    failures = [q[0] for q in QSET if not correct(run(q[0], parser), q)]
    assert failures == []


def test_rules_mode_text_is_exact_for_all_but_one():
    parser = make_parser(None)
    for query, text, *_ in QSET:
        got = run(query, parser).parsed.text
        if query.startswith("a group photo"):
            assert got == "group"  # "photo" is edge filler, same as "photos"
        else:
            assert got == (text or ""), query


@pytest.mark.parametrize(
    ("model", "minimum"), [("qwen25_15b", 30), ("qwen3_17b", 30), ("qwen3_06b", 29)]
)
def test_replayed_llm_outputs(model, minimum):
    """Real small-model replies: grounding + rebuilt text recover (nearly) every query."""
    parser = make_parser(llm_for(model))
    assert accuracy(parser) >= minimum


def test_replay_marks_source_llm():
    res = run("food in Tokyo", make_parser(llm_for("qwen25_15b")))
    assert res.parsed.source == "llm"
    assert res.filters.place == PlaceFilter(city="Tokyo", text="Tokyo")
    assert res.parsed.text == "food"


def test_replay_fixes_empty_text():
    # raw model reply: {"text": "", "date_phrase": "last month"}; "birthday cake photos ..."
    res = run("birthday cake photos from November 2025", make_parser(llm_for("qwen25_15b")))
    assert res.parsed.text == "birthday cake"
    assert rng(res) == ("2025-11-01", "2025-11-30")
    res = run("screenshots from last week", make_parser(llm_for("qwen3_17b")))
    assert res.parsed.text == "screenshots"  # model said text == ""


def test_replay_fixes_garbled_text():
    # Qwen2.5 answered text="s skiing" (it ate "Priya" mid-token)
    res = run("Priya skiing in Whistler in February 2024", make_parser(llm_for("qwen25_15b")))
    assert res.parsed.text == "skiing"
    # Qwen2.5 answered text="Mom at the Eiffel Tower in Paris last year"
    res = run("Mom at the Eiffel Tower in Paris last year", make_parser(llm_for("qwen25_15b")))
    assert res.parsed.text == "Eiffel Tower"
    assert res.filters.people == ("Mom",)


def test_replay_drops_hallucinated_date():
    # Qwen3-1.7B answered date_phrase "last winter" (copied from the few-shot example)
    query = "a group photo with John, Mike and Dave in Berlin"
    res = run(query, make_parser(llm_for("qwen3_17b")))
    assert res.filters.date_range is None
    assert res.parsed.date_phrase is None
    assert not res.unresolved
    assert any("last winter" in n and "does not appear" in n for n in res.notes)
    assert res.filters.people == ("John", "Mike", "Dave")
    assert res.filters.place.text == "Berlin"


def test_replay_red_car_is_not_a_date():
    # Qwen3-1.7B answered date_phrase "red car": grounded, but not a date at all
    res = run("red car", make_parser(llm_for("qwen3_17b")))
    assert res.filters.is_empty and not res.unresolved
    assert res.parsed.text == "red car"
    assert res.parsed.date_phrase is None
    assert any("not a date" in n for n in res.notes)


def test_replay_other_invented_spans():
    parser = make_parser(llm_for("qwen3_17b"))
    res = run("my dog playing in the snow in 2021", parser)  # people: ["my dog"]
    assert res.filters.people == () and not res.unresolved
    assert res.parsed.text == "dog playing in the snow"  # edge filler "my" is stripped
    assert any('"my dog"' in n for n in res.notes)
    res = run("photos of Sarah at a wedding", parser)  # date_phrase: "at a wedding"
    assert res.filters.date_range is None and not res.unresolved
    assert res.parsed.text == "wedding"
    res = run("kids playing at the park yesterday", make_parser(llm_for("qwen25_15b")))
    assert res.filters.place is None and not res.unresolved  # place "park": not a place
    assert res.parsed.text == "kids playing at the park"


def test_replay_widens_date_fragments():
    # Qwen3-0.6B answered date_phrase "2019" for "summer 2019": that would be the whole year
    res = run("summer 2019", make_parser(llm_for("qwen3_06b")))
    assert rng(res) == ("2019-06-01", "2019-08-31")
    assert res.parsed.date_phrase == "summer 2019"
    assert any("Widened" in n for n in res.notes)
    res = run("Christmas 2022", make_parser(llm_for("qwen3_06b")))
    assert rng(res) == ("2022-12-24", "2022-12-26")


def test_replay_reworded_date_is_rescued_from_the_query():
    # Qwen3-0.6B reworded "between June and August 2021" as "June 2021 to August 2021"
    res = run("between June and August 2021", make_parser(llm_for("qwen3_06b")))
    assert rng(res) == ("2021-06-01", "2021-08-31")
    assert any("does not appear" in n for n in res.notes)
    assert any("Used" in n for n in res.notes)


def test_grounding_invariant_over_all_replays():
    for model in _REPLAYS:
        parser = make_parser(llm_for(model))
        for query, *_ in QSET:
            res = run(query, parser)
            lowered = query.lower()
            for span in (res.parsed.date_phrase, res.parsed.place, *res.parsed.people):
                assert span is None or span.lower() in lowered, (model, query, span)


# ----------------------------------------------------------------- LLM prompt and reply handling


def reply_parser(reply: str) -> tuple[QueryParser, ScriptedLLM]:
    llm = ScriptedLLM(default=reply)
    return make_parser(llm), llm


def test_prompt_shape():
    msgs = build_messages("hello Tokyo")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[-1] == {"role": "user", "content": "hello Tokyo"}
    roles = [m["role"] for m in msgs]
    assert roles[1:-1] == ["user", "assistant"] * 5
    for m in msgs[2:-1:2]:
        assert set(json.loads(m["content"])) == {"text", "place", "people", "date_phrase"}


def test_llm_receives_the_stripped_query():
    llm = ScriptedLLM({"food in Tokyo": '{"place": "Tokyo", "people": [], "date_phrase": null}'})
    res = run("  food in Tokyo \n", make_parser(llm))
    assert res.parsed.source == "llm" and res.filters.place.text == "Tokyo"
    assert llm.calls[0][-1] == {"role": "user", "content": "food in Tokyo"}


@pytest.mark.parametrize(
    "reply",
    [
        '{"text": "x", "place": "Tokyo", "people": [], "date_phrase": null}',
        'Sure! Here you go:\n```json\n{"text": "x", "place": "Tokyo", "people": [], '
        '"date_phrase": null}\n```\nHope that helps.',
        '{"place": "Tokyo", "people": [], "date_phrase": null} trailing text {not json',
        "{'place': 'Tokyo', 'people': [], 'date_phrase': None}",
        'noise {oops} then {"Place": "Tokyo", "People": "", "Date_Phrase": "null"}',
        '{"place": "Tokyo", "people": 5, "date_phrase": ["x"]}',
    ],
)
def test_reply_parsing_is_robust(reply):
    parser, _ = reply_parser(reply)
    res = run("food in Tokyo", parser)
    assert res.parsed.source == "llm"
    assert res.filters.place.text == "Tokyo"
    assert res.parsed.text == "food"
    assert res.filters.date_range is None


@pytest.mark.parametrize(
    "reply",
    ["", "I cannot help with that.", "{broken", "[]", "{}", '{"unrelated": 1}', "null", "{'a': "],
)
def test_unusable_reply_falls_back_to_rules(reply):
    parser, _ = reply_parser(reply)
    res = run("food in Tokyo last month", parser)
    assert res.parsed.source == "rules"
    assert res.filters.place.text == "Tokyo"
    assert rng(res) == ("2026-10-01", "2026-10-31")
    assert any("not usable JSON" in n for n in res.notes)


def test_non_string_reply_falls_back_to_rules():
    class Weird:
        name = "weird"

        def chat(self, messages, max_tokens=160):
            return None

    res = run("food in Tokyo", make_parser(Weird()))
    assert res.parsed.source == "rules" and res.filters.place.text == "Tokyo"


def test_llm_exception_falls_back_to_rules():
    class Boom:
        name = "boom"

        def chat(self, messages, max_tokens=160):
            raise OSError("model crashed")

    res = run("food in Tokyo", make_parser(Boom()))
    assert res.parsed.source == "rules" and res.filters.place.text == "Tokyo"
    assert any("OSError" in n for n in res.notes)


def test_use_llm_false_or_missing_llm_never_calls_the_model():
    llm = ScriptedLLM(default='{"place": "Berlin"}')
    res = run("food in Tokyo", make_parser(llm), use_llm=False)
    assert llm.calls == [] and res.parsed.source == "rules"
    assert res.filters.place.text == "Tokyo"
    res = run("food in Tokyo", make_parser(None), use_llm=True)
    assert res.parsed.source == "rules"


def test_very_long_query_skips_the_llm():
    llm = ScriptedLLM(default="{}")
    res = run("beach " * 150 + "in Tokyo", make_parser(llm))
    assert llm.calls == []
    assert res.filters.place.text == "Tokyo"
    assert any("very long" in n for n in res.notes)


def test_blank_query_never_calls_the_llm():
    llm = ScriptedLLM(default="{}")
    res = run("   ", make_parser(llm))
    assert llm.calls == [] and res.parsed.text == "" and res.filters.is_empty


def test_llm_people_string_and_duplicates():
    parser, _ = reply_parser('{"people": "bob and Alice, BOB", "place": null, "date_phrase": null}')
    res = run("Bob and Alice at the lake", parser)
    assert res.filters.people == ("Bob", "Alice")
    assert res.parsed.people == ("Bob", "Alice")
    assert res.parsed.text == "lake"


# ----------------------------------------------------------------- grounding


def test_ungrounded_spans_are_dropped_with_notes():
    parser, _ = reply_parser(
        '{"place": "Paris", "people": ["Bob", "Zelda"], "date_phrase": "last winter"}'
    )
    res = run("Bob at the beach", parser)
    assert res.parsed.place is None and res.filters.place is None
    assert res.parsed.date_phrase is None and res.filters.date_range is None
    assert res.parsed.people == ("Bob",)
    joined = " ".join(res.notes)
    assert '"Paris"' in joined and '"Zelda"' in joined and '"last winter"' in joined
    assert res.parsed.text == "at the beach".replace("at the ", "")  # edge filler stripped


def test_grounding_is_case_insensitive_and_whole_word():
    parser, _ = reply_parser(
        '{"place": "tokyo", "people": ["Al", "BOB"], "date_phrase": "LAST YEAR"}'
    )
    res = run("Bob in Tokyo last year, alpine hike", parser)
    assert res.parsed.place == "Tokyo"  # the user's own spelling
    assert res.parsed.date_phrase == "last year"
    assert res.parsed.people == ("Bob",)  # "Al" is inside "alpine"
    assert res.filters.people == ("Bob",)
    assert res.parsed.text == "alpine hike"


def test_grounding_tolerates_whitespace_and_punctuation_noise():
    parser, _ = reply_parser('{"place": "Key  West.", "date_phrase": " last   summer "}')
    gaz = FakeGazetteer()
    res = run("Key West, last summer sunsets", make_parser(parser._llm, gaz))
    assert res.filters.place.text == "Key West"
    assert rng(res) == ("2026-06-01", "2026-08-31")
    assert res.parsed.text == "sunsets"


def test_possessive_and_connectors_are_removed_from_text():
    parser, _ = reply_parser('{"people": ["Anna"], "place": "Kyoto", "date_phrase": "autumn 2018"}')
    res = run("Anna's wedding photos from autumn 2018 in Kyoto", make_parser(parser._llm))
    assert res.parsed.text == "wedding"
    assert res.filters.people == ()  # Anna is not a known person
    assert [u.kind for u in res.unresolved] == ["person"]


def test_repeated_spans_are_all_removed():
    parser, _ = reply_parser('{"people": ["Bob"], "place": "Paris"}')
    res = run("Bob in Paris, Bob and the dog in Paris", parser)
    assert res.parsed.text == "the dog".replace("the ", "")
    assert res.filters.people == ("Bob",)


# ----------------------------------------------------------------- resolution and unresolved


def test_unsupported_date_in_rules_mode_becomes_unresolved():
    res = run("dinner at Diwali 2023 with Bob")
    assert res.filters.date_range is None
    (u,) = res.unresolved
    assert (u.kind, u.phrase) == ("date", "Diwali 2023")
    assert "lunar" in u.reason
    assert u.question == (
        'I didn\'t understand "Diwali 2023" as a date. Could you reword it, '
        'e.g. "March 2024" or "2024-03-31"?'
    )
    assert res.parsed.date_phrase == "Diwali 2023"
    assert res.parsed.text == "dinner"  # the phrase is stripped even while unresolved
    assert res.filters.people == ("Bob",)


def test_llm_date_unresolved_vs_not_a_date():
    parser, _ = reply_parser('{"date_phrase": "Diwali"}')
    res = run("lights at Diwali", parser)
    assert [u.kind for u in res.unresolved] == ["date"]
    parser, _ = reply_parser('{"date_phrase": "the early 1990s"}')
    res = run("photos from the early 1990s", parser)
    assert [u.phrase for u in res.unresolved] == ["the early 1990s"]
    assert res.unresolved[0].reason


def test_unknown_place_and_person_questions():
    parser, _ = reply_parser(
        '{"place": "Atlantis", "people": ["Bobb", "Zed"], "date_phrase": null}'
    )
    res = run("Bobb and Zed in Atlantis", parser)
    kinds = {u.kind: u for u in res.unresolved}
    assert set(kinds) == {"place", "person"} and len(res.unresolved) == 3
    assert kinds["place"].question.startswith('I don\'t know the place "Atlantis"')
    persons = [u for u in res.unresolved if u.kind == "person"]
    assert 'Did you mean "Bob"?' in persons[0].question
    assert "Did you mean" not in persons[1].question
    assert res.filters.is_empty


def test_lowercase_unknown_place_or_person_is_search_text():
    parser, _ = reply_parser('{"place": "beach", "people": ["the dog"], "date_phrase": null}')
    res = run("the dog at the beach", parser)
    assert not res.unresolved
    assert res.parsed.place is None and res.parsed.people == ()
    assert res.parsed.text == "dog at the beach"
    assert any("not a known place" in n for n in res.notes)
    assert any("not a known person" in n for n in res.notes)


def test_llm_names_are_normalised_to_the_known_spelling():
    parser, _ = reply_parser('{"people": ["mary ann", "MOM"]}')
    res = run("mary ann and MOM cooking", parser)
    assert res.filters.people == ("Mary Ann", "Mom")
    assert res.parsed.people == ("mary ann", "MOM")  # the user's spelling in ParsedQuery
    assert res.parsed.text == "cooking"


def test_llm_place_without_gazetteer_stays_in_text():
    parser = QueryParser(ScriptedLLM(default='{"place": "Tokyo", "people": ["Bob"]}'), None, None)
    res = run("Bob in Tokyo", parser)
    assert res.filters.is_empty and not res.unresolved
    assert res.parsed.text == "Bob in Tokyo"
    assert res.parsed.place is None and res.parsed.people == ()
    assert any("No place database" in n for n in res.notes)
    assert any("No people directory" in n for n in res.notes)


def test_rules_mode_without_gazetteer_or_people():
    parser = QueryParser(None)
    res = run("Bob in Tokyo last month", parser)
    assert res.filters.place is None and res.filters.people == ()
    assert rng(res) == ("2026-10-01", "2026-10-31")
    assert res.parsed.text == "Bob in Tokyo"


def test_failing_collaborators_do_not_break_parsing():
    parser = make_parser(None, FakeGazetteer(fail=True), FakePeople(fail=True))
    res = run("Bob in Tokyo last month", parser)
    assert res.filters.place is None and res.filters.people == ()
    assert rng(res) == ("2026-10-01", "2026-10-31")
    parser, _ = reply_parser('{"place": "Tokyo", "people": ["Bob"]}')
    parser = QueryParser(parser._llm, FakeGazetteer(fail=True), FakePeople(fail=True))
    res = run("Bob in Tokyo", parser)
    assert [u.kind for u in res.unresolved] == ["place", "person"] or res.unresolved
    assert res.filters.is_empty


# ----------------------------------------------------------------- explicit flags


def test_flags_override_each_field_but_spans_are_still_stripped():
    flags = Filters(
        date_range=DateRange(date(2020, 1, 1), date(2020, 1, 31)),
        place=PlaceFilter(country="Japan", text="Japan"),
        people=("Alice",),
    )
    res = run("Bob in Tokyo last month", make_parser(None), filters=flags)
    assert res.filters == flags
    assert res.parsed.text == ""
    assert res.parsed.date_phrase == "last month"
    assert res.parsed.place == "Tokyo" and res.parsed.people == ("Bob",)
    assert not res.unresolved
    assert len([n for n in res.notes if "overrides" in n]) == 3


def test_flags_apply_per_field():
    flags = Filters(date_range=DateRange(date(2020, 1, 1), date(2020, 1, 31)))
    res = run("Bob in Tokyo last month", make_parser(None), filters=flags)
    assert rng(res) == ("2020-01-01", "2020-01-31")
    assert res.filters.place.text == "Tokyo" and res.filters.people == ("Bob",)
    flags = Filters(people=("Sarah",))
    res = run("Bob in Tokyo last month", make_parser(None), filters=flags)
    assert res.filters.people == ("Sarah",) and rng(res) == ("2026-10-01", "2026-10-31")
    flags = Filters(place=PlaceFilter(city="Rome", text="Rome"))
    res = run("Bob in Tokyo", make_parser(None), filters=flags)
    assert res.filters.place.city == "Rome" and res.filters.people == ("Bob",)


def test_flag_wins_over_an_unresolved_phrase_without_asking():
    flags = Filters(date_range=DateRange(date(2020, 1, 1), date(2020, 1, 31)))
    res = run("Diwali 2023 dinner", make_parser(None), filters=flags)
    assert not res.unresolved and res.parsed.text == "dinner"
    assert rng(res) == ("2020-01-01", "2020-01-31")


def test_flag_does_not_swallow_non_date_text():
    flags = Filters(date_range=DateRange(date(2020, 1, 1), date(2020, 1, 31)))
    parser, _ = reply_parser('{"date_phrase": "red car"}')
    res = run("red car", parser, filters=flags)
    assert res.parsed.text == "red car" and res.parsed.date_phrase is None
    assert not [n for n in res.notes if "overrides" in n]


def test_only_flags_no_query():
    flags = Filters(people=("Bob",))
    res = run("", make_parser(ScriptedLLM()), filters=flags)
    assert res.parsed.source == "flags"
    assert res.filters == flags and res.parsed.text == "" and res.parsed.raw == ""
    res = run("", make_parser(None))
    assert res.parsed.source == "rules" and res.filters.is_empty
    res = run("cats", make_parser(None), filters=flags)
    assert res.parsed.source == "rules" and res.filters.people == ("Bob",)
    assert res.parsed.text == "cats"


def test_empty_flags_object_is_ignored():
    res = run("food in Tokyo", make_parser(None), filters=Filters())
    assert res.filters.place.text == "Tokyo"


# ----------------------------------------------------------------- clarifications


def unresolved_request(**kw) -> ParseResult:
    return run("dinner at Diwali 2023 with Bob", **kw)


def test_clarification_reword_resolves_the_filter():
    res = unresolved_request(clarifications={"Diwali 2023": "November 2023"})
    assert not res.unresolved
    assert rng(res) == ("2023-11-01", "2023-11-30")
    assert res.parsed.date_phrase == "Diwali 2023"  # the original span stays grounded
    assert res.parsed.text == "dinner"
    assert any('Understood "November 2023"' in n for n in res.notes)
    assert res.filters.people == ("Bob",)


def test_clarification_empty_string_skips():
    res = unresolved_request(clarifications={"Diwali 2023": ""})
    assert not res.unresolved and res.filters.date_range is None
    assert any("Skipped" in n for n in res.notes)
    assert res.parsed.text == "dinner"
    res = unresolved_request(clarifications={"Diwali 2023": "   "})
    assert not res.unresolved


def test_clarification_still_not_understood_keeps_original_phrase():
    res = unresolved_request(clarifications={"Diwali 2023": "around the festival"})
    (u,) = res.unresolved
    assert u.phrase == "Diwali 2023" and u.kind == "date"  # the engine matches on this
    assert "around the festival" in u.reason
    assert 'still didn\'t understand "around the festival"' in u.question
    assert res.filters.date_range is None


def test_clarification_keys_are_normalised_and_extra_keys_ignored():
    res = unresolved_request(
        clarifications={"  diwali   2023 ": "2023-11-12", "unrelated": "whatever", "": "x"}
    )
    assert rng(res) == ("2023-11-12", "2023-11-12") and not res.unresolved


def test_clarification_non_string_values_are_treated_as_skip():
    res = unresolved_request(clarifications={"Diwali 2023": None})  # type: ignore[dict-item]
    assert not res.unresolved and res.filters.date_range is None


def test_clarification_place_and_person():
    parser, _ = reply_parser('{"place": "Atlantis", "people": ["Zed"]}')
    request = dict(clarifications={"Atlantis": "Rome", "Zed": "dave"})
    res = run("Zed in Atlantis", parser, **request)
    assert not res.unresolved
    assert res.filters.place.text == "Rome" and res.filters.people == ("Dave",)
    res = run("Zed in Atlantis", parser, clarifications={"Atlantis": "Narnia", "Zed": "Nobody"})
    assert {(u.kind, u.phrase) for u in res.unresolved} == {
        ("place", "Atlantis"),
        ("person", "Zed"),
    }
    res = run("Zed in Atlantis", parser, clarifications={"Atlantis": "", "Zed": ""})
    assert not res.unresolved and res.filters.is_empty
    # skipped place/person words are not filters any more: they go back into the search text
    assert res.parsed.text == "Zed in Atlantis"


def test_clarification_answers_only_the_phrases_it_names():
    parser, _ = reply_parser('{"place": "Atlantis", "people": ["Zed"]}')
    res = run("Zed in Atlantis", parser, clarifications={"Zed": "Bob"})
    assert res.filters.people == ("Bob",)
    assert [(u.kind, u.phrase) for u in res.unresolved] == [("place", "Atlantis")]


def test_clarification_person_with_several_names():
    parser, _ = reply_parser('{"people": ["Zed"]}')
    res = run("Zed at the lake", parser, clarifications={"Zed": "Bob and Alice"})
    assert res.filters.people == ("Bob", "Alice")
    res = run("Zed at the lake", parser, clarifications={"Zed": "Bob and Nobody"})
    assert [u.phrase for u in res.unresolved] == ["Zed"]


def test_clarification_does_not_override_a_flag():
    flags = Filters(date_range=DateRange(date(2020, 1, 1), date(2020, 1, 31)))
    res = unresolved_request(clarifications={"Diwali 2023": "March 2024"}, filters=flags)
    assert rng(res) == ("2020-01-01", "2020-01-31")


def test_clarification_round_trip_matches_search_request_protocol():
    parser = make_parser(None)
    first = run("dinner at Diwali 2023", parser)
    answers = {u.phrase: "" for u in first.unresolved}  # what the engine would send back
    second = run("dinner at Diwali 2023", parser, clarifications=answers)
    assert first.unresolved and not second.unresolved
    assert second.parsed.text == first.parsed.text == "dinner"


# ----------------------------------------------------------------- rules-mode precision


def plain(query: str, gazetteer_extra: tuple[str, ...] = (), **kw) -> ParseResult:
    gaz = FakeGazetteer(extra=gazetteer_extra)
    return run(query, make_parser(None, gaz, **kw))


@pytest.mark.parametrize(
    "query",
    [
        "photos that may be blurry",
        "the march of the penguins",
        "women's march downtown",
        "a fall in the snow",
        "summer vacation",
        "winter coat",
        "spring flowers",
        "Christmas tree",
        "Halloween costume",
        "weekend trip",
        "monday blues",
        "autumn leaves",
        "the sun on the water",
        "Mom with the kids",
        "red car",
        "five dogs",
        "1800 people at a concert",
        "cats",
        "Easter eggs",
        "Diwali lights",
        "my bob haircut",
    ],
)
def test_rules_false_positive_dates_stay_in_text(query):
    res = plain(query)
    assert res.filters.date_range is None and not res.unresolved
    assert res.parsed.date_phrase is None
    if query not in {"my bob haircut", "Mom with the kids"}:
        assert res.parsed.text and query.endswith(res.parsed.text)  # only edge filler is lost


def test_rules_prepositions_enable_ambiguous_words():
    assert rng(plain("lake in summer")) == ("2026-06-01", "2026-08-31")
    assert rng(plain("a hike in May")) == ("2026-05-01", "2026-05-31")
    assert rng(plain("a hike in march")) == ("2026-03-01", "2026-03-31")
    assert rng(plain("tree at Christmas")) == ("2025-12-24", "2025-12-26")
    assert rng(plain("dinner on Monday")) == ("2026-11-02", "2026-11-02")
    assert rng(plain("coffee over the weekend")) == ("2026-10-31", "2026-11-01")
    assert rng(plain("photos from fall 2019")) == ("2019-09-01", "2019-11-30")
    assert plain("lake in summer").parsed.text == "lake"


def test_rules_date_features():
    res = plain("Easter 2024 egg hunt")
    assert rng(res) == ("2024-03-30", "2024-04-01") and res.parsed.text == "egg hunt"
    res = plain("trip 3 weeks ago")
    assert rng(res) == ("2026-10-12", "2026-10-18") and res.parsed.text == "trip"
    res = plain("Thanksgiving 2023 dinner with Alice")
    assert rng(res) == ("2023-11-22", "2023-11-24") and res.filters.people == ("Alice",)
    res = plain("the summer of 2017 at the lake")
    assert rng(res) == ("2017-06-01", "2017-08-31") and res.parsed.text == "lake"
    assert plain("winter 2020", southern_hemisphere=True).filters.date_range.start == date(
        2020, 6, 1
    )


def test_rules_year_first_dates_are_not_split():
    res = plain("trip to Japan 2018 summer")
    assert rng(res) == ("2018-06-01", "2018-08-31")
    assert res.parsed.text == "trip to Japan"


def test_rules_picks_the_longest_date_phrase():
    res = plain("New Year's Eve 2024 fireworks")
    assert res.parsed.date_phrase == "New Year's Eve 2024"
    assert rng(res) == ("2024-12-31", "2025-01-01")


def test_rules_place_preferences_and_guards():
    gaz = FakeGazetteer()
    parser = make_parser(None, gaz)
    assert run("Nice weather in Paris", parser).filters.place.text == "Paris"
    assert run("photos in Nice", parser).filters.place.text == "Nice"
    assert run("Nice weather today", parser).filters.place is None  # common word, no preposition
    assert run("Reading books at home", parser).filters.place is None
    assert run("a day in Reading", parser).filters.place.text == "Reading"
    assert run("Orange juice", parser).filters.place is None
    assert run("an orange cat in Tokyo", parser).filters.place.text == "Tokyo"
    assert run("Hope and joy", parser).filters.place is None
    assert run("mobile phone", parser).filters.place is None
    assert run("Tokyo food", parser).filters.place.text == "Tokyo"  # sentence-initial, no rival
    assert run("Sunset in Goa", parser).filters.place.text == "Goa"
    assert run("Sunset Goa", parser).filters.place.text == "Goa"
    assert run("The beach", parser).filters.place is None
    assert run("on a trip to Japan and Goa", parser).filters.place.text == "Goa"
    assert run("Visiting Key West with friends", parser).filters.place.text == "Key West"
    assert run("dinner at St. Louis", parser).filters.place.text == "St. Louis"
    assert run("hiking in Isle of Skye", parser).filters.place.text == "Isle of Skye"
    assert run("a week in Paris, France", parser).filters.place.text == "Paris, France"
    assert run("a week in New York", parser).filters.place.text == "New York"
    assert run("New York pizza", parser).parsed.text == "pizza"
    assert run("in Springfield. Then Tokyo", parser).filters.place.text == "Springfield"


def test_rules_place_is_removed_from_text_with_its_preposition():
    res = plain("street food in Bangkok")
    assert res.parsed.text == "street food" and res.parsed.place == "Bangkok"
    res = plain("Berlin")
    assert res.filters.place.text == "Berlin" and res.parsed.text == ""


def test_rules_place_prefers_prepositioned_over_longer():
    gaz = FakeGazetteer(extra=("Eiffel Tower",))
    parser = make_parser(None, gaz)
    assert run("pics at Eiffel Tower in Paris", parser).filters.place.text == "Eiffel Tower"
    assert run("Paris trip", parser).filters.place.text == "Paris"


def test_rules_place_word_limit():
    gaz = FakeGazetteer(extra=("Alpha Beta Gamma Delta",))
    parser = make_parser(None, gaz)
    assert run("in Alpha Beta Gamma Delta", parser).filters.place.text == "Alpha Beta Gamma Delta"
    res = run("in Alpha Beta Gamma Delta Epsilon Zeta", parser)
    assert res.filters.place.text == "Alpha Beta Gamma Delta"  # five capitalised words: too long


def test_rules_place_does_not_overlap_date_or_people():
    gaz = FakeGazetteer(extra=("December", "John"))
    parser = make_parser(None, gaz)
    res = run("John in Goa, December 2022", parser)
    assert res.filters.place.text == "Goa" and res.filters.people == ("John",)
    assert rng(res) == ("2022-12-01", "2022-12-31")
    assert res.parsed.text == ""


def test_rules_people_whole_word_possessive_and_case():
    res = plain("Bob's dog and ALICE at the park")
    assert res.filters.people == ("Bob", "Alice") and res.parsed.text == "dog at the park"
    assert res.parsed.people == ("Bob", "ALICE")
    res = plain("Bobby and Bobcat")
    assert res.filters.people == ()
    res = plain("mary ann and Mary")
    assert res.filters.people == ("Mary Ann",)  # "Mary" alone is not a known name
    assert plain("Bob, Bob and Bob").filters.people == ("Bob",)


def test_rules_person_and_month_name_conflict():
    parser = make_parser(None, people=FakePeople("April", "June"))
    res = run("April at the beach", parser)
    assert res.filters.people == ("April",) and res.filters.date_range is None
    res = run("trip in June with April", parser)
    assert rng(res) == ("2026-06-01", "2026-06-30")
    assert res.filters.people == ("April",)  # the date span wins the overlap for "June"


def test_rules_text_when_everything_is_filler():
    assert plain("photos of").parsed.text == "photos of"
    assert plain("the").parsed.text == "the"
    res = plain("photos of my dog")
    assert res.parsed.text == "dog"


# ----------------------------------------------------------------- odd input and contract


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   \n\t ",
        "!!!",
        "🎄🎅 in 2022",
        "café à Zürich",
        "日本 2020年",
        "a" * 50_000,
        "(((" * 1000,
        "\x00\x01",
        "in in in the the",
        "last last last",
        "2022 2022 2022 2022 2022",
        "Mom, Mom, Mom",
    ],
)
def test_odd_input_never_raises(query):
    for llm in (None, ScriptedLLM(default='{"place": "Tokyo", "people": ["Bob"]}')):
        res = run(query, make_parser(llm))
        assert isinstance(res, ParseResult)
        assert res.parsed.raw == query
        # contract: spans appear verbatim in raw
        for span in (res.parsed.date_phrase, res.parsed.place, *res.parsed.people):
            assert span is None or span.lower() in query.lower()


def test_unicode_query_is_preserved():
    parser = make_parser(None, FakeGazetteer(extra=("Zürich",)))
    res = run("café à Zürich last year", parser)
    assert res.filters.place.text == "Zürich"
    assert res.parsed.text == "café à"
    assert rng(res) == ("2025-01-01", "2025-12-31")


def test_very_long_query_is_fast():
    start = time.perf_counter()
    res = run("beach sunset Tokyo " * 3000 + "last year", make_parser(None))
    assert time.perf_counter() - start < 5
    assert isinstance(res, ParseResult)


def test_non_string_query_is_treated_as_empty():
    parser = make_parser(None)
    res = parser.parse(SearchRequest(query=None, today=TODAY))  # type: ignore[arg-type]
    assert res.parsed.text == "" and res.filters.is_empty


def test_today_precedence_and_default(monkeypatch):
    parser = make_parser(None)
    other = date(2030, 5, 20)
    req = SearchRequest(query="yesterday", today=TODAY)
    assert rng(parser.parse(req)) == ("2026-11-06", "2026-11-06")
    assert rng(parser.parse(req, today=other)) == ("2030-05-19", "2030-05-19")
    real = date.today()
    res = parser.parse(SearchRequest(query="yesterday"))
    assert res.filters.date_range.start in {real - timedelta(1), date.today() - timedelta(1)}


def test_parse_is_deterministic_and_pure():
    parser = make_parser(llm_for("qwen25_15b"))
    first = [run(q[0], parser) for q in QSET]
    second = [run(q[0], parser) for q in QSET]
    assert first == second
    request = SearchRequest(query="Bob in Tokyo", today=TODAY, clarifications={"x": "y"})
    assert parser.parse(request) == parser.parse(request)
    assert request.clarifications == {"x": "y"}


def test_southern_hemisphere_option():
    parser = QueryParser(None, southern_hemisphere=True)
    res = parser.parse(SearchRequest(query="last summer", today=TODAY))
    assert rng(res) == ("2025-12-01", "2026-02-28")


def test_parse_result_text_property():
    res = plain("food in Tokyo")
    assert res.text == res.parsed.text == "food"


# ----------------------------------------------------------------- review round 2: regressions
# Rules mode and LLM mode must never silently change the meaning of a date, and every phrase that
# is not understood must come back as an Unresolved instead of vanishing or widening a filter.


def only_unresolved(res: ParseResult, kind: str = "date"):
    assert res.filters.date_range is None, res.filters
    (u,) = [x for x in res.unresolved if x.kind == kind]
    return u


@pytest.mark.parametrize(
    ("query", "phrase"),
    [
        ("photos since 2020", "since 2020"),
        ("pictures before 2015", "before 2015"),
        ("until June 2021", "until June 2021"),
        ("from 2019 onwards", "2019 onwards"),
        ("after Christmas 2022", "after Christmas 2022"),
        ("everything except 2019", "except 2019"),
        ("not in summer", "not in summer"),
        ("not yesterday", "not yesterday"),
        ("early 2020 hiking", "early 2020"),
        ("2 or 3 years ago", "2 or 3 years ago"),
        ("June 2020 and December 2021", "June 2020 and December 2021"),
        ("2019 or 2020", "2019 or 2020"),
        ("hiking in June or July 2020", "June or July 2020"),
        ("Bob in 2019 and 2020", "2019 and 2020"),
        ("last year or this year", "last year or this year"),
    ],
)
def test_rules_operators_and_lists_are_unresolved_never_applied(query, phrase):
    res = plain(query)
    u = only_unresolved(res)
    assert u.phrase == phrase and u.question and u.reason
    assert res.parsed.date_phrase == phrase
    assert phrase.lower() not in res.parsed.text.lower()  # stripped; nothing is kept as a filter


def test_operator_text_is_cleaned_and_questions_are_specific():
    res = plain("hiking photos since 2020 with Bob")
    u = only_unresolved(res)
    assert res.parsed.text == "hiking" and res.filters.people == ("Bob",)
    assert "open-ended" in u.question and '"since 2020"' in u.question
    u = only_unresolved(plain("June 2020 and December 2021"))
    assert 'Which should I use: "June 2020" or "December 2021"?' in u.question


def test_operator_clarification_round_trip():
    parser = make_parser(None)
    first = run("hiking since 2020", parser)
    (u,) = first.unresolved
    answer = run("hiking since 2020", parser, clarifications={u.phrase: "2020 to 2026"})
    assert rng(answer) == ("2020-01-01", "2026-12-31") and not answer.unresolved
    assert answer.parsed.text == "hiking"
    skipped = run("hiking since 2020", parser, clarifications={u.phrase: ""})
    assert skipped.filters.date_range is None and not skipped.unresolved


@pytest.mark.parametrize(
    ("query", "phrase", "text"),
    [
        ("party on 03/04/2024", "03/04/2024", "party"),
        ("trip on 2024-02-30", "2024-02-30", "trip"),
        ("party 03.04.2019", "03.04.2019", "party"),
    ],
)
def test_rules_bad_numeric_dates_are_unresolved(query, phrase, text):
    res = plain(query)
    u = only_unresolved(res)
    assert u.phrase == phrase and res.parsed.text == text


@pytest.mark.parametrize(
    ("query", "phrase"),
    [
        ("Eid with Priya 2023", "2023"),
        ("plage à Paris été 2019", "été 2019"),
        ("Weihnachten 2019 in Berlin", "Weihnachten 2019"),
        ("verano 2021", "verano 2021"),
        ("Easter eggs 2024", "2024"),
        ("Halloween costumes 2022", "2022"),
    ],
)
def test_rules_unknown_date_word_next_to_a_year_is_not_a_year_filter(query, phrase):
    res = plain(query)
    u = only_unresolved(res)
    assert u.phrase == phrase and u.question
    assert res.filters.date_range is None


def test_foreign_date_word_reason_says_english_only():
    assert "English" in only_unresolved(plain("verano 2021")).reason
    assert "English" in only_unresolved(plain("Weihnachten 2019")).reason
    assert plain("verano en la playa").filters.date_range is None


@pytest.mark.parametrize(
    "query",
    [
        "lunch at a Chinese restaurant",
        "at the lunar eclipse",
        "photo of the lunar eclipse",
        "Chinese food",
        "Chinese New Year dinner",
        "Windows 2000 screenshot",
        "jersey number 2012",
        "iPhone 15",
        "a June bug",
    ],
)
def test_rules_ordinary_text_with_date_looking_words(query):
    res = plain(query)
    assert res.filters.date_range is None and not res.unresolved and not res.parsed.date_phrase


def test_lunar_names_keep_working_with_an_anchor():
    assert (
        only_unresolved(plain("dinner at Chinese New Year 2024")).phrase == "Chinese New Year 2024"
    )
    assert only_unresolved(plain("Diwali 2022 in Mumbai")).phrase == "Diwali 2022"


def test_possessive_dates_leave_clean_text():
    res = plain("pictures from last year's trip to Rome")
    assert rng(res) == ("2025-01-01", "2025-12-31") and res.parsed.text == "trip"
    res = plain("Alice yesterday's lunch")
    assert rng(res) == ("2026-11-06", "2026-11-06")
    assert res.parsed.text == "lunch" and res.filters.people == ("Alice",)


def test_through_is_removed_like_other_connectors():
    res = plain("road trip through Canada summer 2019", gazetteer_extra=("Canada",))
    assert res.parsed.text == "road trip" and res.filters.place.text == "Canada"
    assert plain("road trip through").parsed.text == "road trip"


def test_skipped_phrase_gives_its_words_back_unless_it_was_a_date():
    parser, _ = reply_parser(
        '{"place": "Atlantis", "people": ["Zed"], "date_phrase": "Diwali 2023"}'
    )
    q = "Zed in Atlantis at Diwali 2023"
    res = run(q, parser, clarifications={"Zed": "", "Atlantis": "", "Diwali 2023": ""})
    assert not res.unresolved and res.filters.is_empty
    assert res.parsed.text == "Zed in Atlantis"  # the date words are gone, the others are back
    res = run(q, parser, clarifications={"Diwali 2023": ""})
    assert [u.kind for u in res.unresolved] == ["place", "person"]


def test_whole_query_skipped_real_date_is_not_left_empty():
    res = run("Diwali 2023", make_parser(None), clarifications={"Diwali 2023": ""})
    assert res.parsed.text == "Diwali 2023" and res.filters.is_empty
    assert any("search text" in n for n in res.notes)


# ----------------------------------------------------------------- LLM slots and fallbacks

Q1 = "Bob at the beach last summer in Key West"


def j(**kw) -> str:
    return json.dumps({"text": "", "place": None, "people": [], "date_phrase": None, **kw})


def llm_run(reply: str, query: str = Q1, **kw) -> ParseResult:
    return run(query, make_parser(ScriptedLLM({query: reply})), **kw)


def full(res: ParseResult):
    return (
        rng(res),
        res.filters.place.text if res.filters.place else None,
        res.filters.people,
        res.parsed.text,
    )


SUMMER = ("2026-06-01", "2026-08-31")
GOOD = (SUMMER, "Key West", ("Bob",), "beach")


@pytest.mark.parametrize(
    "reply",
    [
        j(place="Key West", people=["Bob"]),  # date missing
        j(place="last summer", people=["Bob"]),  # date in the place slot
        j(place="last summer", people=["Bob"], date_phrase="last summer"),
        j(place="last summer", people=["Bob"], date_phrase="Key West"),  # swapped
        j(place="Key West", people=["Bob", "last summer", "Key West"], date_phrase="last summer"),
        j(place="Key West", people=["Bob", "Key West"], date_phrase="last summer"),
        j(date_phrase="last summer in Key West", people=["Bob"]),  # too wide
        j(date_phrase="beach last summer", people=["Bob"], place="Key West"),
        j(place="Key West", people=["Bob"], date_phrase="summer"),  # fragment
        j(place="Key West", people=["Bob"], date_phrase="summer 2026"),  # reworded
        j(place="Key West", people=["Bob"], date_phrase="2026-06-01 to 2026-08-31"),
        j(place="Paris", people=["Bob"], date_phrase="last summer"),  # invented place
        j(place="Key West", people=["Bob", "Carol"], date_phrase="last summer"),
        '{"text": 5, "place": ["Key West"], "people": {"a": 1}, "date_phrase": 2026}',
        j(place="null", people=["none"], date_phrase="N/A"),
        j(date_phrase=Q1, place=Q1, people=[Q1]),  # the whole query in every slot
        j(place="West", people=["Bob"], date_phrase="last summer"),  # place fragment
        j(place="Key", people=["Bob"], date_phrase="last summer"),
        j(),
    ],
)
def test_llm_slot_confusions_end_in_the_same_correct_filters(reply):
    res = llm_run(reply)
    assert full(res) == GOOD, res.notes
    assert not res.unresolved


def test_llm_null_fields_fall_back_to_rules_with_a_note():
    res = llm_run(j(), "kids playing last weekend")
    assert rng(res) == ("2026-10-31", "2026-11-01") and res.parsed.text == "kids playing"
    assert res.parsed.source == "llm"
    assert any('no date; found "last weekend"' in n for n in res.notes)
    res = llm_run(j(), "Priya hiking in Tokyo")
    assert res.filters.people == ("Priya",) and res.filters.place.text == "Tokyo"
    assert any("no usable place" in n for n in res.notes)
    assert any("no usable person" in n for n in res.notes)


def test_llm_all_null_reply_on_plain_query_has_no_spurious_note():
    res = llm_run(j(place="null", people=["none"], date_phrase="N/A"), "red car")
    assert res.filters.is_empty and res.notes == () and res.parsed.text == "red car"
    res = llm_run('{"place":"null","people":["none"],"date_phrase":"N/A"}', "cats")
    assert res.notes == ()


def test_known_person_named_as_place_is_a_person():
    res = llm_run(j(place="Alice", date_phrase="last year"), "Alice in Paris last year")
    assert res.filters.people == ("Alice",) and res.filters.place.text == "Paris"
    assert any("known person, not a place" in n for n in res.notes)
    res = llm_run(j(people=["Alice", "Paris"], date_phrase="last year"), "Alice in Paris last year")
    assert res.filters.people == ("Alice",) and res.filters.place.text == "Paris"
    assert not res.unresolved


def test_llm_person_that_is_a_date_or_place_is_rerouted():
    res = llm_run(j(people=["last year"]), "photos last year")
    assert rng(res) == ("2025-01-01", "2025-12-31") and not res.unresolved
    res = llm_run(j(people=["Tokyo"]), "dinner in Tokyo")
    assert res.filters.place.text == "Tokyo" and not res.unresolved
    res = llm_run(j(people=["Tokyo", "Paris"]), "Tokyo or Paris")
    assert not res.unresolved
    res = llm_run(
        j(place="Tokyo", people=["yesterday"], date_phrase="last year"), "Tokyo yesterday"
    )
    assert res.filters.place.text == "Tokyo" and not res.unresolved


def test_llm_date_that_is_really_a_person_or_place():
    res = llm_run(j(date_phrase="Bob"), "Bob at the lake")
    assert res.filters.people == ("Bob",) and not res.unresolved
    res = llm_run(j(date_phrase="Tokyo"), "lake in Tokyo")
    assert res.filters.place.text == "Tokyo" and res.filters.date_range is None


def test_llm_operator_fragment_is_read_as_the_whole_phrase():
    res = llm_run(j(date_phrase="2020"), "photos since 2020")
    u = only_unresolved(res)
    assert u.phrase == "since 2020" and any('Read "2020"' in n for n in res.notes)
    res = llm_run(j(date_phrase="since 2020"), "photos since 2020")
    assert only_unresolved(res).phrase == "since 2020"
    res = llm_run(j(date_phrase="June 2020"), "June 2020 and December 2021")
    assert only_unresolved(res).phrase == "June 2020 and December 2021"
    res = llm_run(j(date_phrase="2022"), "Diwali 2022 dinner")
    assert only_unresolved(res).phrase == "Diwali 2022"


def test_llm_chinese_is_not_a_date():
    res = llm_run(j(date_phrase="Chinese"), "lunch at a Chinese restaurant")
    assert not res.unresolved and res.parsed.text == "lunch at a Chinese restaurant"


def test_llm_place_fragment_is_widened_and_long_names_dropped():
    res = llm_run(j(place="West", people=["Bob"], date_phrase="last summer"))
    assert res.filters.place.text == "Key West"
    assert any("Widened the place" in n for n in res.notes)
    res = llm_run(j(people=["one two three four five six"]), "one two three four five six")
    assert not res.unresolved


def test_llm_whole_query_slots_do_not_empty_the_text():
    res = llm_run(j(date_phrase="x y z", place="x y z", people=["x y z"]), "x y z")
    assert res.parsed.text == "x y z" and not res.unresolved


def test_llm_reply_is_cached_per_query():
    llm = ScriptedLLM({"Zed in Atlantis": j(place="Atlantis", people=["Zed"])})
    parser = make_parser(llm)
    first = run("Zed in Atlantis", parser)
    assert len(first.unresolved) == 2 and len(llm.calls) == 1
    second = run("Zed in Atlantis", parser, clarifications={"Atlantis": "Rome", "Zed": "Bob"})
    assert len(llm.calls) == 1 and not second.unresolved
    run("  Zed in Atlantis ", parser)
    assert len(llm.calls) == 1  # keyed on the stripped query
    unusable = ScriptedLLM(default="no json")
    parser = make_parser(unusable)
    for _ in range(2):
        res = run("food in Tokyo", parser)
        assert any("not usable" in n for n in res.notes)
    assert len(unusable.calls) == 1


def test_llm_cache_is_bounded_and_errors_are_not_cached():
    llm = ScriptedLLM(default=j())
    parser = make_parser(llm)
    for i in range(70):
        run(f"query number {i}", parser)
    assert len(parser._cache) == 64
    run("query number 0", parser)  # evicted: asks the model again
    assert len(llm.calls) == 71
    calls = []

    class Flaky:
        name = "flaky"

        def chat(self, messages, max_tokens=160):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("once")
            return j(place="Tokyo")

    parser = make_parser(Flaky())
    assert run("food in Tokyo", parser).parsed.source == "rules"
    assert run("food in Tokyo", parser).parsed.source == "llm"


# ----------------------------------------------------------------- people matching


def people_parser(*names: str) -> QueryParser:
    return QueryParser(None, FakeGazetteer(), FakePeople(*names))


def test_first_name_matches_a_unique_full_name():
    parser = people_parser("Bob Smith", "Alice Jones")
    res = run("Bob at the beach", parser)
    assert res.filters.people == ("Bob Smith",) and res.parsed.text == "beach"
    assert res.parsed.people == ("Bob",)
    res = run("bob smith and ALICE", parser)
    assert res.filters.people == ("Bob Smith", "Alice Jones")


def test_ambiguous_first_name_asks_which_one():
    parser = people_parser("Bob Smith", "Bob Jones", "Alice")
    res = run("Bob at the beach", parser)
    (u,) = res.unresolved
    assert u.kind == "person" and u.phrase == "Bob"
    assert u.question == 'Which "Bob" did you mean: Bob Jones or Bob Smith?'
    assert res.filters.people == ()
    res = run("Bob at the beach", parser, clarifications={"Bob": "Bob Smith"})
    assert res.filters.people == ("Bob Smith",) and not res.unresolved
    res = run("Bob at the beach", parser, clarifications={"Bob": "bob"})
    assert [u.phrase for u in res.unresolved] == ["Bob"]
    assert "more than one person" in res.unresolved[0].reason
    res = run("Bob Jones and Bob Smith", parser)
    assert res.filters.people == ("Bob Smith", "Bob Jones") or set(res.filters.people) == {
        "Bob Smith",
        "Bob Jones",
    }


def test_exact_name_wins_over_first_name_prefix():
    parser = people_parser("Bob", "Bob Smith")
    assert run("Bob at the beach", parser).filters.people == ("Bob",)
    assert run("Bob Smith at the beach", parser).filters.people == ("Bob Smith",)


def test_accent_and_case_insensitive_names():
    parser = people_parser("José", "Zoë Müller")
    assert run("jose at the beach", parser).filters.people == ("José",)
    assert run("JOSÉ and zoe muller", parser).filters.people == ("José", "Zoë Müller")
    assert run("José's dog", parser).parsed.text == "dog"
    llm = ScriptedLLM({"Jose in Paris": j(people=["Jose"], place="Paris")})
    res = run("Jose in Paris", QueryParser(llm, FakeGazetteer(), FakePeople("José")))
    assert res.filters.people == ("José",) and res.parsed.people == ("Jose",)
    res = run("x", parser, clarifications={})
    assert res.filters.is_empty


def test_llm_first_name_and_clarification_reply_use_the_same_matching():
    parser = QueryParser(
        ScriptedLLM({"Bob Smith's dog": j(people=["Bob"]), "Zed hiking": j(people=["Zed"])}),
        FakeGazetteer(),
        FakePeople("Bob Smith", "José"),
    )
    res = run("Bob Smith's dog", parser)
    assert res.filters.people == ("Bob Smith",)
    res = run("Zed hiking", parser, clarifications={"Zed": "jose"})
    assert res.filters.people == ("José",)


def test_people_without_directory_reword_is_unresolved():
    parser = QueryParser(ScriptedLLM(default=j()), FakeGazetteer(), FakePeople())
    res = run(
        "Zed hiking", QueryParser(ScriptedLLM(default=j(people=["Zed"])), None, FakePeople("A"))
    )
    assert [u.phrase for u in res.unresolved] == ["Zed"]
    other = QueryParser(ScriptedLLM(default=j(people=["Zed"])), None, FakePeople("A"))
    res = run("Zed hiking", other, clarifications={"Zed": "Amy"})
    assert [u.phrase for u in res.unresolved] == ["Zed"]
    assert parser is not None


# ----------------------------------------------------------------- limits and performance


def test_pathological_repetition_is_fast_and_capped():
    parser = make_parser(None)
    start = time.perf_counter()
    res = run("Bob " * 50_000, parser)
    assert time.perf_counter() - start < 3
    assert res.filters.people == ("Bob",)
    assert any("long" in n for n in res.notes)


def test_long_query_note_when_the_date_is_beyond_the_scan_limit():
    res = run("beach " * 500 + "last summer", make_parser(None))
    assert res.filters.date_range is None
    assert any("first part" in n for n in res.notes)
    assert run("beach in Tokyo last summer", make_parser(None)).notes == ()


def test_clarification_key_and_is_asked():
    assert clarification_key("  Diwali   2023 ") == "diwali 2023"
    assert clarification_key("STRASSE") == clarification_key("strasse")
    assert is_asked("Diwali 2023", {"  diwali  2023": "x"})
    assert is_asked("Zed", {"Zed": ""})
    assert not is_asked("Zed", {"Bob": "x"})
    assert not is_asked("Zed", None) and not is_asked("Zed", {})
    assert not is_asked("Zed", {1: "x"})  # type: ignore[dict-item]
    first = run("dinner at Diwali 2023", make_parser(None))
    assert not is_asked(first.unresolved[0].phrase, {})
    second = run("dinner at Diwali 2023", make_parser(None), clarifications={"diwali 2023": "bad"})
    assert is_asked(second.unresolved[0].phrase, {"diwali 2023": "bad"})


def test_landmark_does_not_beat_a_named_city():
    gaz = FakeGazetteer()
    gaz.known["eiffel tower"] = PlaceFilter(near=(48.85, 2.29, 1.5), text="Eiffel Tower")
    res = run("Bob and Mom at the Eiffel Tower in Paris yesterday", make_parser(None, gaz))
    assert res.filters.place.text == "Paris"
    res = run("pics at the Eiffel Tower", make_parser(None, gaz))
    assert res.filters.place.text == "Eiffel Tower"  # only a landmark: still used


def test_reviewer_regression_table_in_rules_mode():
    cases = {
        "Mom's cooking": ("Mom", "cooking"),
        "Dear Mom": ("Mom", "Dear"),
        "bob": ("Bob", ""),
        "Priya": ("Priya", ""),
    }
    parser = make_parser(None, people=FakePeople("Bob", "Alice", "Priya", "Mom"))
    for query, (person, text) in cases.items():
        res = run(query, parser)
        assert res.filters.people == (person,) and res.parsed.text == text, query


def test_internal_helpers_handle_empty_input():
    assert _Query("abc").find_all("") == []
    assert _Query("abc").find(None) is None
    assert _People(["Bob"]).resolve("   ") == (None, [])
    assert _People([]).scan(_Query("Bob"), 3, []) == []
    assert _People(["", "  "]).by_fold == {}


def test_date_slot_dropped_as_a_person_is_refilled_by_rules_with_its_hint():
    res = llm_run(j(date_phrase="Bob"), "Bob since 2020")
    assert res.filters.people == ("Bob",)
    assert only_unresolved(res).phrase == "since 2020"
    assert any("given as a date but is a known person" in n for n in res.notes)
    assert any('no usable date; found "since 2020"' in n for n in res.notes)


def test_person_that_is_a_bare_weekday_stays_text_but_with_a_preposition_is_a_date():
    res = llm_run(j(people=["Monday"]), "dinner Monday")
    assert res.filters.date_range is None and not res.unresolved
    assert res.parsed.text == "dinner Monday"
    assert any('Treated "Monday" as search text' in n for n in res.notes)
    res = llm_run(j(people=["Monday"]), "dinner on Monday")
    assert rng(res) == ("2026-11-02", "2026-11-02") and not res.unresolved
    assert res.filters.people == ()


def test_bare_word_in_the_place_slot_is_text_too():
    res = llm_run(j(place="Summer"), "Summer dress")
    assert res.filters.is_empty and not res.unresolved and res.parsed.text == "Summer dress"
    res = llm_run(j(place="summer"), "dress for the summer")
    assert rng(res) == ("2026-06-01", "2026-08-31")


# ----------------------------------------------------------------- bare words in LLM mode

BARE_QUERIES = [
    ("autumn leaves in Amherst", "autumn"),
    ("winter coat", "winter"),
    ("summer dress", "summer"),
    ("spring flowers", "spring"),
    ("christmas tree", "christmas"),
    ("a fall in the snow", "fall"),
    ("march of the penguins", "march"),
    ("halloween costume", "halloween"),
    ("easter eggs", "easter"),
    ("may flowers", "may"),
    ("Thanksgiving turkey", "Thanksgiving"),
    ("monday blues", "monday"),
    ("Autumn colours", "Autumn"),
]


def test_real_qwen_reply_for_autumn_leaves_in_amherst():
    """Real misbehaviour: Qwen2.5-1.5B said date_phrase "autumn", which hid every photo."""
    reply = '{"text":"leaves","place":"Amherst","people":[],"date_phrase":"autumn"}'
    gaz = FakeGazetteer(extra=("Amherst",))
    llm = ScriptedLLM({"autumn leaves in Amherst": reply})
    res = run("autumn leaves in Amherst", QueryParser(llm, gaz, FakePeople(*NAMES)))
    assert res.filters.date_range is None and res.parsed.date_phrase is None
    assert res.filters.place.text == "Amherst" and not res.unresolved
    assert res.parsed.text == "autumn leaves"
    assert any('Treated "autumn" as search text' in n for n in res.notes)


@pytest.mark.parametrize(("query", "word"), BARE_QUERIES)
def test_llm_bare_season_holiday_month_words_stay_search_text(query, word):
    res = llm_run(j(date_phrase=word), query)
    assert res.filters.date_range is None and not res.unresolved
    assert res.parsed.date_phrase is None and res.parsed.source == "llm"
    assert word.lower() in res.parsed.text.lower()
    assert any(f'Treated "{word}" as search text' in n for n in res.notes)


@pytest.mark.parametrize(
    ("query", "phrase", "expected"),
    [
        ("leaves in autumn", "autumn", ("2026-09-01", "2026-11-30")),
        ("leaves in autumn", "in autumn", ("2026-09-01", "2026-11-30")),
        ("leaves last autumn", "last autumn", ("2025-09-01", "2025-11-30")),
        ("leaves autumn 2023", "autumn 2023", ("2023-09-01", "2023-11-30")),
        ("leaves in the autumn", "autumn", ("2026-09-01", "2026-11-30")),
        ("a trip during the summer", "summer", ("2026-06-01", "2026-08-31")),
        ("tree at Christmas", "Christmas", ("2025-12-24", "2025-12-26")),
        ("tree Christmas 2022", "Christmas 2022", ("2022-12-24", "2022-12-26")),
        ("hike in May", "May", ("2026-05-01", "2026-05-31")),
        ("hike of March", "March", ("2026-03-01", "2026-03-31")),
        ("dinner on Monday", "Monday", ("2026-11-02", "2026-11-02")),
        ("parties from December", "December", ("2025-12-01", "2025-12-31")),
        ("hike around Easter", "Easter", ("2026-04-04", "2026-04-06")),
        ("costumes for Halloween", "Halloween", ("2026-10-30", "2026-11-01")),
        ("cats over the weekend", "weekend", ("2026-10-31", "2026-11-01")),
        ("ski this winter", "this winter", ("2026-12-01", "2027-02-28")),
        ("ski winter 2021", "winter 2021", ("2020-12-01", "2021-02-28")),
        ("photos 2019", "2019", ("2019-01-01", "2019-12-31")),
        ("photos yesterday", "yesterday", ("2026-11-06", "2026-11-06")),
    ],
)
def test_llm_bare_words_count_with_a_year_qualifier_or_preposition(query, phrase, expected):
    res = llm_run(j(date_phrase=phrase), query)
    assert rng(res) == expected, res.notes
    assert not any("as search text" in n for n in res.notes)


def test_llm_bare_word_in_one_place_but_with_a_preposition_in_another():
    res = llm_run(j(date_phrase="autumn"), "autumn leaves in autumn")
    # the model's phrase is grounded on its first occurrence, which is bare; the grammar
    # finds the prepositioned one and the rules fill the date
    assert rng(res) == ("2026-09-01", "2026-11-30")


def test_rules_mode_agrees_with_llm_mode_on_bare_words():
    for query, word in BARE_QUERIES:
        assert plain(query).filters.date_range is None
        assert llm_run(j(date_phrase=word), query).filters.date_range is None


# ----------------------------------------------------------------- impossible dates


@pytest.mark.parametrize(
    ("query", "phrase", "reason", "text"),
    [
        ("party 31 February 2020", "31 February 2020", "February 2020 has no day 31", "party"),
        ("30 Feb 2020 trip", "30 Feb 2020", "February 2020 has no day 30", "trip"),
        ("hike 31 April 2021", "31 April 2021", "April 2021 has no day 31", "hike"),
        ("trip on 2020-02-30", "2020-02-30", "February 2020 has no day 30", "trip"),
        ("Feb 30", "Feb 30", "February has no day 30", ""),
        ("party on 31/04/2021", "31/04/2021", "April 2021 has no day 31", "party"),
        ("29 February 2021 wedding", "29 February 2021", "February 2021 has no day 29", "wedding"),
    ],
)
def test_impossible_dates_are_unresolved_in_both_modes(query, phrase, reason, text):
    for res in (plain(query), llm_run(j(date_phrase=phrase), query), llm_run(j(), query)):
        u = only_unresolved(res)
        assert u.phrase == phrase and reason in u.reason and reason in u.question
        assert res.parsed.text == text and res.filters.date_range is None


def test_llm_partial_date_phrase_of_an_impossible_date_is_replaced():
    res = llm_run(j(date_phrase="February 2020"), "party 31 February 2020")
    assert only_unresolved(res).phrase == "31 February 2020"
    res = llm_run(j(date_phrase="2020"), "party Feb 31, 2020")
    assert only_unresolved(res).phrase == "Feb 31, 2020"


def test_impossible_date_clarification_round_trip():
    parser = make_parser(None)
    q = "party 31 February 2020"
    res = run(q, parser, clarifications={"31 February 2020": "29 February 2020"})
    assert rng(res) == ("2020-02-29", "2020-02-29") and not res.unresolved
    res = run(q, parser, clarifications={"31 February 2020": "31 February 2020"})
    assert only_unresolved(res).phrase == "31 February 2020"
    res = run(q, parser, clarifications={"31 February 2020": ""})
    assert not res.unresolved and res.filters.date_range is None and res.parsed.text == "party"


def test_person_slot_holding_a_plain_year_becomes_the_date():
    res = llm_run(j(people=["2019"]), "Windows 2019 screenshot")
    assert rng(res) == ("2019-01-01", "2019-12-31") and res.filters.people == ()
    assert any("listed as a person but is a date" in n for n in res.notes)


# ----------------------------------------------------------------- operators via every entry point

OPERATOR_REPLIES = [
    *(f"{w} 2019" for w in (
        "since", "before", "after", "until", "till", "by", "pre", "post", "prior", "early",
        "late", "mid", "middle", "end", "beginning", "start", "not", "except", "excluding",
        "without",
    )),
    *(f"2019 {w}" for w in ("onwards", "onward", "later", "earlier", "afterwards", "forward")),
    "2019 or 2020",
    "June 2020 and December 2021",
]  # fmt: skip


@pytest.mark.parametrize("reply", OPERATOR_REPLIES)
def test_operator_clarification_reply_stays_unresolved_with_the_original_phrase(reply):
    parser = make_parser(None)
    query = "dinner at Diwali 2023"
    res = run(query, parser, clarifications={"Diwali 2023": reply})
    u = only_unresolved(res)
    assert u.phrase == "Diwali 2023"  # the engine matches on the original phrase
    assert reply in u.reason and res.filters.date_range is None
    assert res.parsed.text == "dinner"
    assert is_asked(u.phrase, {"Diwali 2023": reply})
    # the follow-up question is the specific one, not a generic retry
    assert (
        "still didn't understand" not in u.question
        and "2019" in u.question
        or "Which" in u.question
    )


def test_after_2019_reply_is_never_a_silent_whole_year_filter():
    res = run(
        "dinner at Diwali 2023", make_parser(None), clarifications={"Diwali 2023": "after 2019"}
    )
    assert res.filters.date_range is None
    assert only_unresolved(res).question.startswith("I can't filter on open-ended dates")


@pytest.mark.parametrize("reply", OPERATOR_REPLIES)
def test_llm_phrases_with_operators_are_unresolved(reply):
    res = llm_run(j(date_phrase=reply), f"photos {reply}")
    u = only_unresolved(res)
    assert u.phrase == reply and res.filters.date_range is None


@pytest.mark.parametrize("word", ["after", "by", "since", "before", "not", "early"])
def test_operator_phrase_as_a_direct_phrase_gets_the_specific_question(word):
    parser, _ = reply_parser(j(date_phrase=f"{word} 2019"))
    res = run(f"hike {word} 2019", parser)
    u = only_unresolved(res)
    assert u.question and "didn't understand" not in u.question
    # a reworded phrase outside any query: the LLM said "<word> 2019" but the query differs
    res = llm_run(j(date_phrase=f"{word} 2019"), f"hike (see {word} 2019)")
    assert res.filters.date_range is None


def test_model_phrase_with_an_operator_the_query_scan_missed_still_gets_the_specific_question():
    # "version 2019" makes the scanner treat 2019 as a number, but the phrase is still an operator
    res = llm_run(j(date_phrase="2019 or earlier"), "version 2019 or earlier")
    u = only_unresolved(res)
    assert u.phrase == "2019 or earlier" and "open-ended" in u.question
