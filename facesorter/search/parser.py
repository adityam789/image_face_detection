"""
facesorter/search/parser.py
``QueryParser``: natural-language query -> structured filters + visual search text.

Flow (see docs/search-design.md, "Query flow"):

1. **Extract** ``{place, people, date_phrase}`` with a small LLM (few-shot JSON, the model
   copies the date phrase, it never computes dates) or, with no LLM, deterministic rules. In
   LLM mode a field the model left empty, got wrong or invented is filled from the rules for
   that field (with a note); a phrase put in the wrong slot ("last summer" as the place, a
   place among the people) is moved to the right one.
2. **Ground** every extracted span: it must appear verbatim (case- and accent-insensitive, whole
   words) in the query, otherwise it is dropped. ``text`` is then *rebuilt* as the query minus
   the grounded spans and filler words at its edges; the model's own ``text`` is never used.
3. **Resolve** the date phrase (``dates.py``), place (``Gazetteer.resolve``) and people (must
   be known names). Whatever is not understood becomes an ``Unresolved`` with a friendly
   question instead of silently widening or narrowing the search: this includes operators
   ("since 2020", "not in summer"), several dates ("2019 or 2020"), ambiguous numeric dates and
   date words of other languages.
4. Explicit flags (``request.filters``) override the inferred filter per field, and
   ``request.clarifications`` re-resolve the user's reworded phrases (see ``SearchRequest``).
   ``clarification_key`` / ``is_asked`` let the engine decide ask-vs-drop with the same
   normalisation.

Parsing is pure and deterministic for a given (request, LLM replies, gazetteer, people, today)
and never raises on odd input; every surprising decision is recorded in ``ParseResult.notes``.
The model's reply is cached per query (bounded), so the engine's second ``parse`` call after a
clarification does not run the model again.
"""

from __future__ import annotations

import ast
import difflib
import json
import logging
import re
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from .dates import (
    DateSpan,
    analyse_dates,
    bare_date_word,
    looks_like_date,
    preposition_before,
    resolve_date_phrase,
)
from .prompts import build_messages
from .types import (
    Filters,
    Gazetteer,
    LLMBackend,
    ParsedQuery,
    ParseResult,
    PeopleDirectory,
    PlaceFilter,
    SearchRequest,
    Unresolved,
)

logger = logging.getLogger(__name__)

__all__ = ["QueryParser", "clarification_key", "is_asked"]

# Words stripped from the edges of the rebuilt visual text.
_FILLER = frozenset(
    "photos photo pictures picture pics pic images image of in at from on the a an with and my "
    "during between for through around near by".split()
)
_WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*")
_POSSESSIVE_RE = re.compile(r"['’]s(?!\w)", re.IGNORECASE)
# A single connector word swallowed together with the span it introduces ("in Goa", "with Bob").
_DATE_PLACE_CONNECTORS = "in|on|at|from|during|near|around|for|of|by|since|to|through"
_PEOPLE_CONNECTORS = "with|and|of"
_CONNECTOR_BEFORE = {
    "span": re.compile(rf"(?<!\w)(?:{_DATE_PLACE_CONNECTORS})\s+$", re.IGNORECASE),
    "person": re.compile(rf"(?<!\w)(?:{_PEOPLE_CONNECTORS})\s+$", re.IGNORECASE),
}
_MAX_TOKENS = 400  # rules scanning is capped; a photo query is a handful of words
_MAX_SCAN_CHARS = 4000
_MAX_LLM_QUERY_CHARS = 600
_MAX_PEOPLE = 12
_MAX_PEOPLE_MATCHES = 40
_MAX_CUTS = 2000  # occurrences of one span removed from the text
_MAX_NAME_WORDS = 5
_CACHE_SIZE = 64

# Rules-mode place detection.
_PLACE_PREPS = frozenset("in at near from around outside inside to".split())
_PLACE_CONNECTORS = frozenset("of de del da di la le el al du van von der".split())
_ABBREVIATIONS = frozenset({"st", "mt", "ft", "dr"})
_COMMON_WORD_PLACES = frozenset(
    "nice reading mobile orange hope bath normal surprise paradise independence".split()
)
_NOT_PLACES = (
    _FILLER
    | frozenset("january february march april may june july august september october".split())
    | frozenset("november december monday tuesday wednesday thursday friday saturday".split())
    | frozenset("sunday spring summer autumn fall winter i".split())
)
_MAX_PLACE_TRIES = 40

_NULLISH = frozenset({"", "null", "none", "n/a", "na", "nil", "unknown"})
_FIELDS = ("date", "place", "people")


def clarification_key(phrase: str) -> str:
    """The normalised form under which a phrase is matched against ``request.clarifications``.

    Whitespace is collapsed and case ignored, so ``"  Diwali   2023 "`` and ``"diwali 2023"``
    are the same phrase.
    """
    return " ".join(str(phrase).split()).casefold()


def is_asked(phrase: str, clarifications: Mapping[str, str] | None) -> bool:
    """Whether the user was already asked about ``phrase`` (it has a key in ``clarifications``).

    The engine's rule: an unresolved phrase that was asked is *dropped* (reported in
    ``SearchResponse.dropped``, search runs without that filter); one that was not asked is
    returned in ``needs_clarification``. An answer of ``""`` also counts as asked.
    """
    if not clarifications:
        return False
    key = clarification_key(phrase)
    return any(clarification_key(k) == key for k in clarifications if isinstance(k, str))


# --------------------------------------------------------------------------- text helpers


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Raw:
    """What the model said, before grounding. ``None`` means "no usable value"."""

    date_phrase: str | None = None
    place: str | None = None
    people: tuple[str, ...] = ()


@dataclass
class _Slots:
    """Grounded spans: each is a verbatim piece of the query."""

    date: _Span | None = None
    hint: DateSpan | None = None  # why the date span is not understood, with its question
    place: _Span | None = None
    people: list[_Span] = field(default_factory=list)


@dataclass
class _Item:
    """The fate of one grounded span."""

    status: Literal["ok", "unresolved", "ignored", "skipped"]
    value: Any = None  # DateRange | PlaceFilter | canonical name(s)
    unresolved: Unresolved | None = None
    restore: bool = False  # a skipped span that was not a real filter: give its words back


def _fold(text: str) -> str:
    """Lower-case, accent-free copy of ``text`` with exactly the same length."""
    out: list[str] = []
    for ch in text:
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        base = base.lower()
        out.append(base if len(base) == 1 else (ch.lower() if len(ch.lower()) == 1 else ch))
    return "".join(out)


def _phrase_regex(folded_phrase: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in folded_phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?![\w\-])", re.IGNORECASE)


class _Query:
    """The query with an accent-folded twin of identical length for matching."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.folded = _fold(raw)

    def find_all(self, phrase: str) -> list[_Span]:
        cleaned = _clean_phrase(phrase)
        if not cleaned:
            return []
        regex = _phrase_regex(_fold(cleaned))
        return [_Span(m.start(), m.end(), self.raw[m.start() : m.end()]) for m in
                regex.finditer(self.folded)]  # fmt: skip

    def find(self, phrase: str | None) -> _Span | None:
        found = self.find_all(phrase) if phrase else []
        return found[0] if found else None


def _clean_phrase(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip(" \t\"'.,;:!?")
    return None if cleaned.lower() in _NULLISH else cleaned


def _overlaps(span: tuple[int, int], taken: Iterable[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in taken)


def _strip_edges(words: list[str]) -> list[str]:
    start, end = 0, len(words)
    while start < end and words[start].lower() in _FILLER:
        start += 1
    while end > start and words[end - 1].lower() in _FILLER:
        end -= 1
    return words[start:end]


def _residual(q: _Query, spans: list[tuple[str, Literal["span", "person"]]]) -> str:
    """The query minus the grounded spans (and the connector word before each) and edge filler."""
    cuts: list[tuple[int, int]] = []
    seen: set[tuple[str, str]] = set()
    for phrase, kind in spans:
        if (_fold(phrase), kind) in seen:
            continue
        seen.add((_fold(phrase), kind))
        for found in q.find_all(phrase)[:_MAX_CUTS]:
            start, end = found.start, found.end
            possessive = _POSSESSIVE_RE.match(q.raw, end)
            if possessive:
                end = possessive.end()
            connector = _CONNECTOR_BEFORE[kind].search(q.raw, max(0, start - 16), start)
            if connector:
                start = connector.start()
            cuts.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(cuts):
        if start > cursor:
            pieces.append(q.raw[cursor:start])
        cursor = max(cursor, end)
    pieces.append(q.raw[cursor:])
    words = _strip_edges(_WORD_RE.findall(" ".join(pieces)))
    if not words and not cuts:
        return " ".join(_WORD_RE.findall(q.raw))  # only filler words: keep them
    return " ".join(words)


def _has_capital(phrase: str) -> bool:
    return any(w[:1].isupper() for w in phrase.split())


def _is_capitalised(word: str) -> bool:
    return len(word) > 1 and word[0].isupper()


# --------------------------------------------------------------------------- people


class _People:
    """Known names with accent/case-insensitive and first-name matching."""

    def __init__(self, names: Iterable[str]) -> None:
        by_fold: dict[str, str] = {}
        for name in names:
            cleaned = " ".join(name.split())
            if cleaned:
                by_fold.setdefault(_fold(cleaned), cleaned)
        self.by_fold = by_fold
        patterns = set(by_fold)
        patterns |= {f.split()[0] for f in by_fold if len(f.split()[0]) >= 2}
        ordered = sorted(patterns, key=len, reverse=True)
        self._scanner = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(r"\s+".join(map(re.escape, p.split())) for p in ordered)
                + r")(?![\w\-])"
            )
            if ordered
            else None
        )  # fmt: skip

    def resolve(self, typed: str) -> tuple[str | None, list[str]]:
        """``(canonical name, [])`` if exactly one known name fits; ``(None, [names])`` if
        several do ("Bob" for Bob Smith and Bob Jones); ``(None, [])`` if none."""
        words = _fold(" ".join(typed.split())).split()
        if not words:
            return None, []
        exact = self.by_fold.get(" ".join(words))
        if exact is not None:
            return exact, []
        fits = [n for f, n in self.by_fold.items() if f.split()[: len(words)] == words]
        if len(fits) == 1:
            return fits[0], []
        return None, sorted(fits)

    def scan(self, q: _Query, limit: int, taken: list[tuple[int, int]]) -> list[_Span]:
        if self._scanner is None:
            return []
        found: list[_Span] = []
        for match in self._scanner.finditer(q.folded, 0, limit):
            if len(found) >= _MAX_PEOPLE_MATCHES:
                break
            if _overlaps(match.span(), taken):
                continue
            taken.append(match.span())
            found.append(_Span(match.start(), match.end(), q.raw[match.start() : match.end()]))
        return found


# --------------------------------------------------------------------------- LLM reply


def _json_objects(reply: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", reply):
        tail = reply[match.start() :]
        try:
            obj, _ = decoder.raw_decode(tail)
        except ValueError:
            try:  # models sometimes answer with a Python-style dict
                end = tail.index("}") + 1
                obj = ast.literal_eval(tail[:end])
            except (ValueError, SyntaxError):
                continue
        if isinstance(obj, dict):
            yield obj


def _first_string(value: object) -> str | None:
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if isinstance(v, str)), None)
    return _clean_phrase(value)


def _parse_reply(reply: object) -> _Raw | None:
    """Pull the extraction object out of a chatty/fenced/trailing-text reply, or ``None``."""
    if not isinstance(reply, str):
        return None
    for obj in _json_objects(reply):
        keys = {str(k).lower(): v for k, v in obj.items()}
        if not keys.keys() & {"date_phrase", "place", "people"}:
            continue
        people_raw = keys.get("people")
        if isinstance(people_raw, str):
            people_raw = re.split(r"\s*(?:,|;|&|\band\b)\s*", people_raw)
        if not isinstance(people_raw, (list, tuple)):
            people_raw = []
        people = [p for p in (_clean_phrase(x) for x in people_raw) if p]
        return _Raw(
            date_phrase=_first_string(keys.get("date_phrase")),
            place=_first_string(keys.get("place")),
            people=tuple(people[:_MAX_PEOPLE]),
        )
    return None


# --------------------------------------------------------------------------- the parser


class QueryParser:
    """Turn a ``SearchRequest`` into a ``ParseResult``.

    ``llm=None`` (or ``request.use_llm=False``) selects the deterministic rules extractor.
    ``gazetteer`` / ``people`` may be ``None``: place and people filters are then simply not
    produced (an LLM-named place or person is left in the visual text). ``southern_hemisphere``
    flips the meteorological seasons used for "summer", "winter", ...
    """

    def __init__(
        self,
        llm: LLMBackend | None,
        gazetteer: Gazetteer | None = None,
        people: PeopleDirectory | None = None,
        *,
        southern_hemisphere: bool = False,
    ) -> None:
        self._llm = llm
        self._gazetteer = gazetteer
        self._people = people
        self._south = southern_hemisphere
        self._cache: OrderedDict[str, tuple[_Raw | None, str]] = OrderedDict()

    # ------------------------------------------------------------------ public

    def parse(self, request: SearchRequest, today: date | None = None) -> ParseResult:
        """Parse ``request``. ``today`` beats ``request.today``, which beats the clock."""
        today = today or request.today or date.today()
        raw_query = request.query if isinstance(request.query, str) else ""
        q = _Query(raw_query)
        flags = request.filters or Filters()
        notes: list[str] = []
        known = self._known_people()
        places: dict[str, PlaceFilter | None] = {}
        limit = self._scan_limit(raw_query, notes)

        slots = _Slots()
        source: Literal["llm", "rules", "flags"] = "rules"
        if not raw_query.strip():
            source = "flags" if not flags.is_empty else "rules"
        else:
            extracted = self._extract_llm(request, raw_query, notes)
            if extracted is not None:
                source = "llm"
                slots = self._ground_llm(q, extracted, today, limit, known, places, notes)
            else:
                slots = self._rules_slots(q, today, limit, known, places, set(_FIELDS), [], notes)

        date_item = self._date_item(slots, today, notes)
        place_item = self._place_item(slots.place, places, notes)
        person_items = [(s, self._person_item(s.text, known, notes)) for s in slots.people]

        self._apply_clarifications(
            request.clarifications, date_item, place_item, person_items, known, places, today, notes
        )

        date_range = date_item.value if date_item.status == "ok" else None
        place_filter = place_item.value if place_item.status == "ok" else None
        names: list[str] = []
        for _, it in person_items:
            if it.status == "ok":
                names.extend([it.value] if isinstance(it.value, str) else it.value)
        unresolved = [
            it.unresolved
            for it in (date_item, place_item, *(it for _, it in person_items))
            if it.status == "unresolved" and it.unresolved is not None
        ]
        # Explicit flags win, per field.
        if flags.date_range is not None:
            date_range, unresolved = flags.date_range, [u for u in unresolved if u.kind != "date"]
            if slots.date and date_item.status != "ignored":
                notes.append(
                    f'The explicit date filter overrides "{slots.date.text}" found in the query.'
                )
        if flags.place is not None:
            place_filter, unresolved = flags.place, [u for u in unresolved if u.kind != "place"]
            if slots.place and place_item.status != "ignored":
                notes.append(
                    f'The explicit place filter overrides "{slots.place.text}" found in the query.'
                )
        if flags.people:
            names, unresolved = list(flags.people), [u for u in unresolved if u.kind != "person"]
            if any(it.status != "ignored" for _, it in person_items):
                notes.append("The explicit people filter overrides the people named in the query.")

        def stripped(item: _Item) -> bool:
            return item.status != "ignored" and not (item.status == "skipped" and item.restore)

        spans: list[tuple[str, Literal["span", "person"]]] = []
        if slots.date and stripped(date_item):
            spans.append((slots.date.text, "span"))
        if slots.place and stripped(place_item):
            spans.append((slots.place.text, "span"))
        kept_people = [s.text for s, it in person_items if it.status != "ignored"]
        spans.extend((s.text, "person") for s, it in person_items if stripped(it))

        text = _residual(q, spans) if raw_query.strip() else ""
        filters = Filters(
            date_range=date_range, place=place_filter, people=tuple(dict.fromkeys(names))
        )
        if not text and spans and filters.is_empty and not unresolved and _covers(q, spans):
            text = " ".join(_WORD_RE.findall(raw_query))
            notes.append("Everything in the query was read as a filter that is not usable; "
                         "the words are used as search text instead.")  # fmt: skip
        parsed = ParsedQuery(
            raw=raw_query,
            text=text,
            date_phrase=slots.date.text if slots.date and date_item.status != "ignored" else None,
            place=slots.place.text if slots.place and place_item.status != "ignored" else None,
            people=tuple(kept_people),
            source=source,
        )
        return ParseResult(
            parsed=parsed, filters=filters, unresolved=tuple(unresolved), notes=tuple(notes)
        )

    # ------------------------------------------------------------------ extraction

    def _known_people(self) -> _People | None:
        if self._people is None:
            return None
        try:
            names = [n for n in self._people.known_names() if isinstance(n, str)]
        except Exception:
            logger.warning("PeopleDirectory.known_names() failed", exc_info=True)
            names = []
        return _People(names)

    @staticmethod
    def _scan_limit(query: str, notes: list[str]) -> int:
        """How much of the query the rules look at (first 400 words / 4000 characters)."""
        limit = min(len(query), _MAX_SCAN_CHARS)
        words = list(re.finditer(r"\S+", query[:limit]))
        if len(words) > _MAX_TOKENS:
            limit = words[_MAX_TOKENS - 1].end()
        if limit < len(query):
            notes.append(
                "The query is long, so only its first part was searched for dates, places "
                "and names."
            )
        return limit

    def _extract_llm(self, request: SearchRequest, query: str, notes: list[str]) -> _Raw | None:
        if not request.use_llm or self._llm is None:
            return None
        if len(query) > _MAX_LLM_QUERY_CHARS:
            notes.append("The query is very long, so the language model was skipped.")
            return None
        key = query.strip()
        if key in self._cache:
            self._cache.move_to_end(key)
            raw, failure = self._cache[key]
        else:
            try:
                reply = self._llm.chat(build_messages(key), max_tokens=160)
            except Exception as exc:
                logger.warning("LLM query parsing failed: %s", exc)
                notes.append(
                    f"The language model failed ({type(exc).__name__}); used rules instead."
                )
                return None
            raw = _parse_reply(reply)
            failure = "" if raw is not None else "unusable"
            self._cache[key] = (raw, failure)
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        if raw is None:
            logger.info("Unusable LLM reply for query %r", query)
            notes.append("The language model's reply was not usable JSON; used rules instead.")
        return raw

    def _ground_llm(
        self,
        q: _Query,
        raw: _Raw,
        today: date,
        limit: int,
        known: _People | None,
        places: dict[str, PlaceFilter | None],
        notes: list[str],
    ) -> _Slots:
        """Ground the model's spans, repair them, and fill what it missed from the rules."""
        slots = _Slots()
        finding = analyse_dates(q.raw[:limit], today, self._south)
        slots.date, slots.hint = self._refine_date(q, raw.date_phrase, finding, today, notes)
        slots.place = self._grounded(q, raw.place, "place", notes)
        for name in raw.people:
            span = self._grounded(q, name, "person", notes)
            if span and span.text.lower() not in {p.text.lower() for p in slots.people}:
                slots.people.append(span)
        slots.people.sort(key=lambda s: s.start)

        self._reconcile(slots, q, known, places, today, notes)

        if slots.place is not None and self._gazetteer is not None:
            if self._resolve_place(slots.place.text, places) is None:
                taken = [(s.start, s.end) for s in (slots.date, *slots.people) if s]
                better = self._rules_place(q, limit, taken, places)
                if better is not None and _overlaps(
                    (better.start, better.end), [(slots.place.start, slots.place.end)]
                ):
                    notes.append(f'Widened the place "{slots.place.text}" to "{better.text}".')
                    slots.place = better

        missing = {f for f in _FIELDS if (f == "people" and not slots.people) or
                   (f != "people" and getattr(slots, f) is None)}  # fmt: skip
        if missing:
            taken = [(s.start, s.end) for s in (slots.date, slots.place, *slots.people) if s]
            filled = self._rules_slots(q, today, limit, known, places, missing, taken, [])
            for name, label in (("date", "date"), ("place", "place"), ("people", "person")):
                value = getattr(filled, name)
                if not value:
                    continue
                shown = ", ".join(s.text for s in value) if name == "people" else value.text
                notes.append(
                    f'The language model gave no usable {label}; found "{shown}" with rules.'
                )
                setattr(slots, name, value)
            if filled.hint is not None and slots.date is filled.date:
                slots.hint = filled.hint
        return slots

    @staticmethod
    def _grounded(q: _Query, phrase: str | None, label: str, notes: list[str]) -> _Span | None:
        cleaned = _clean_phrase(phrase)
        if not cleaned:
            return None
        span = q.find(cleaned)
        if span is None:
            notes.append(f'Ignored the {label} "{phrase}": it does not appear in your query.')
        return span

    def _refine_date(
        self,
        q: _Query,
        given: str | None,
        finding: DateSpan | None,
        today: date,
        notes: list[str],
    ) -> tuple[_Span | None, DateSpan | None]:
        """Repair the model's date phrase using what the date grammar finds in the query.

        * It is replaced or widened to the full wording when the grammar sees an operator, a
          second date or a longer phrase around it ("2020" for "since 2020", "2019" for
          "summer 2019"): resolving the fragment would silently change the meaning.
        * A phrase that is not understood but contains one that is ("last summer in Key West")
          is narrowed to it.
        * A phrase that is not in the query at all is replaced by the grammar's finding.
        """
        span = self._grounded(q, given, "date phrase", notes)
        if span is None:
            if finding is None:
                return None, None
            if _clean_phrase(given):
                notes.append(f'Used "{finding.phrase}" from the query as the date phrase.')
            else:
                notes.append(
                    f'The language model gave no date; found "{finding.phrase}" with rules.'
                )
            return _Span(finding.start, finding.end, finding.phrase), _hint(finding)
        if finding is not None and _touches(finding, span):
            if not finding.resolution.understood:
                if finding.phrase != span.text:
                    notes.append(f'Read "{span.text}" as "{finding.phrase}".')
                return _Span(finding.start, finding.end, finding.phrase), finding
            if (
                finding.start <= span.start
                and span.end <= finding.end
                and finding.phrase != span.text
            ):
                notes.append(f'Widened the date phrase "{span.text}" to "{finding.phrase}".')
                return _Span(finding.start, finding.end, finding.phrase), None
        if not resolve_date_phrase(span.text, today, self._south).understood:
            inner = analyse_dates(span.text, today, self._south)
            if inner is not None and inner.resolution.understood:
                start = span.start + inner.start
                notes.append(f'Narrowed the date phrase "{span.text}" to "{inner.phrase}".')
                return _Span(start, start + len(inner.phrase), inner.phrase), None
        elif self._bare_blocked(q, span, today):
            notes.append(_bare_note(span.text))
            return None, None
        return span, None

    def _bare_blocked(self, q: _Query, span: _Span, today: date) -> bool:
        """A bare season/holiday/month/weekday word with nothing marking it as a date."""
        return bare_date_word(span.text, today, self._south) and not preposition_before(
            q.raw, span.start
        )

    def _reconcile(
        self,
        slots: _Slots,
        q: _Query,
        known: _People | None,
        places: dict[str, PlaceFilter | None],
        today: date,
        notes: list[str],
    ) -> None:
        """Fix phrases the model put in the wrong slot, duplicated or made up as too long."""

        def is_person(text: str) -> bool:
            if known is None:
                return False
            name, candidates = known.resolve(text)
            return name is not None or bool(candidates)

        def is_place(text: str) -> bool:
            return self._gazetteer is not None and self._resolve_place(text, places) is not None

        def is_date(text: str) -> bool:
            return resolve_date_phrase(text, today, self._south).understood

        def too_long(span: _Span) -> bool:
            return len(span.text.split()) > _MAX_NAME_WORDS

        covers_all = len(q.raw.split()) >= 3

        def whole_query(span: _Span) -> bool:
            return covers_all and len(span.text.split()) >= len(q.raw.split())

        place, date_span = slots.place, slots.date
        if place is not None:
            if is_person(place.text):
                if not any(p.text.lower() == place.text.lower() for p in slots.people):
                    slots.people.append(place)
                notes.append(f'"{place.text}" is a known person, not a place.')
                slots.place = None
            elif is_place(place.text):
                pass
            elif is_date(place.text):
                if self._bare_blocked(q, place, today):
                    notes.append(_bare_note(place.text))
                elif date_span is None or date_span.text.lower() == place.text.lower():
                    slots.date, slots.hint = place, None
                    notes.append(f'"{place.text}" was given as a place but is a date.')
                else:
                    notes.append(f'Ignored the place "{place.text}": it is a date.')
                slots.place = None
            elif too_long(place) or whole_query(place):
                notes.append(f'Ignored the place "{place.text}": too long to be a place name.')
                slots.place = None
        date_span = slots.date
        if date_span is not None and slots.hint is None and not is_date(date_span.text):
            if looks_like_date(date_span.text):
                pass
            elif whole_query(date_span):
                notes.append(f'Ignored the date phrase "{date_span.text}": it is the whole query.')
                slots.date = None
            elif slots.place is None and is_place(date_span.text):
                slots.place, slots.date = date_span, None
                notes.append(f'"{date_span.text}" was given as a date but is a place.')
            elif is_person(date_span.text):
                slots.people.append(date_span)
                slots.date = None
                notes.append(f'"{date_span.text}" was given as a date but is a known person.')
        kept: list[_Span] = []
        for person in slots.people:
            low = person.text.lower()
            if is_person(person.text):
                pass
            elif slots.place is not None and low == slots.place.text.lower():
                notes.append(f'Ignored "{person.text}" as a person: it is the place.')
                continue
            elif slots.date is not None and low == slots.date.text.lower():
                notes.append(f'Ignored "{person.text}" as a person: it is the date.')
                continue
            elif is_place(person.text) and slots.place is None:
                slots.place = person
                notes.append(f'"{person.text}" was listed as a person but is a place.')
                continue
            elif is_place(person.text):
                notes.append(f'Ignored "{person.text}" as a person: it is a place.')
                continue
            elif is_date(person.text):
                if self._bare_blocked(q, person, today):
                    notes.append(_bare_note(person.text))
                    continue
                if slots.date is None:
                    slots.date, slots.hint = person, None
                notes.append(f'"{person.text}" was listed as a person but is a date.')
                continue
            elif too_long(person) or whole_query(person):
                notes.append(f'Ignored "{person.text}" as a person: too long to be a name.')
                continue
            kept.append(person)
        # a person inside the date or place span is part of that phrase
        fixed = [s for s in (slots.date, slots.place) if s is not None]
        slots.people = [
            p for p in kept if not _overlaps((p.start, p.end), [(s.start, s.end) for s in fixed])
        ]

    def _rules_slots(
        self,
        q: _Query,
        today: date,
        limit: int,
        known: _People | None,
        places: dict[str, PlaceFilter | None],
        want: set[str],
        taken: list[tuple[int, int]],
        notes: list[str],
    ) -> _Slots:
        """Deterministic extraction of the ``want`` fields from the first ``limit`` characters."""
        slots = _Slots()
        taken = list(taken)
        if "date" in want:
            finding = analyse_dates(q.raw[:limit], today, self._south)
            if finding is not None and not _overlaps((finding.start, finding.end), taken):
                slots.date = _Span(finding.start, finding.end, finding.phrase)
                slots.hint = _hint(finding)
                taken.append((finding.start, finding.end))
        if "people" in want and known is not None:
            slots.people = known.scan(q, limit, taken)
        if "place" in want:
            slots.place = self._rules_place(q, limit, taken, places)
        return slots

    def _rules_place(
        self,
        q: _Query,
        limit: int,
        taken: list[tuple[int, int]],
        places: dict[str, PlaceFilter | None],
    ) -> _Span | None:
        """Longest capitalised span (<= 4 words) the gazetteer resolves, preferring "in X"."""
        if self._gazetteer is None:
            return None
        query = q.raw
        tokens = [m for m in _WORD_RE.finditer(query, 0, limit)][:_MAX_TOKENS]
        candidates: list[tuple[tuple[bool, bool, int, int], _Span]] = []
        for i, first in enumerate(tokens):
            if not _is_capitalised(first.group()):
                continue
            caps = 0
            for j in range(i, min(i + 6, len(tokens))):
                word = tokens[j].group()
                if j > i and not _joins(query, tokens[j - 1], tokens[j]):
                    break
                if _is_capitalised(word):
                    caps += 1
                elif j == i or word.lower() not in _PLACE_CONNECTORS:
                    break
                if caps > 4:
                    break
                if not _is_capitalised(word):
                    continue
                span = (first.start(), tokens[j].end())
                if _overlaps(span, taken):
                    break
                text = query[span[0] : span[1]]
                n_words = j - i + 1
                after_prep = _after_preposition(query, tokens, i)
                sentence_start = i == 0 or query[tokens[i - 1].end() : first.start()].strip() in {
                    ".",
                    "!",
                    "?",
                }
                weak = n_words == 1 and sentence_start and not after_prep
                lowered = text.lower()
                if n_words == 1 and (lowered in _NOT_PLACES):
                    continue
                if lowered in _COMMON_WORD_PLACES and not after_prep:
                    continue
                candidates.append(((weak, not after_prep, -n_words, -i), _Span(*span, text)))
        candidates.sort(key=lambda c: c[0])
        landmark: _Span | None = None
        for _, found in candidates[:_MAX_PLACE_TRIES]:
            resolved = self._resolve_place(found.text, places)
            if resolved is None:
                continue
            if resolved.city or resolved.region or resolved.country:
                return found  # a named city/region/country beats a point-only landmark
            landmark = landmark or found
        return landmark

    # ------------------------------------------------------------------ resolution

    def _resolve_place(self, name: str, cache: dict[str, PlaceFilter | None]) -> PlaceFilter | None:
        key = name.lower()
        if key not in cache:
            try:
                cache[key] = self._gazetteer.resolve(name) if self._gazetteer else None
            except Exception:
                logger.warning("Gazetteer.resolve(%r) failed", name, exc_info=True)
                cache[key] = None
        return cache[key]

    def _date_item(self, slots: _Slots, today: date, notes: list[str]) -> _Item:
        span = slots.date
        if span is None:
            return _Item("ignored")
        phrase = span.text
        hint = slots.hint
        if hint is not None and not hint.resolution.understood and hint.phrase == phrase:
            return _Item(
                "unresolved",
                unresolved=Unresolved(
                    "date",
                    phrase,
                    hint.resolution.reason,
                    hint.question or _date_question(phrase, today, self._south),
                ),
            )
        result = resolve_date_phrase(phrase, today, self._south)
        if result.understood:
            return _Item("ok", result.range)
        if not looks_like_date(phrase):
            notes.append(f'Treated "{phrase}" as search text: it is not a date.')
            return _Item("ignored")
        return _Item(
            "unresolved",
            unresolved=Unresolved(
                "date", phrase, result.reason, _date_question(phrase, today, self._south)
            ),
        )

    def _place_item(
        self, span: _Span | None, cache: dict[str, PlaceFilter | None], notes: list[str]
    ) -> _Item:
        if span is None:
            return _Item("ignored")
        phrase = span.text
        if self._gazetteer is None:
            notes.append(f'No place database is available, so "{phrase}" is used as search text.')
            return _Item("ignored")
        found = self._resolve_place(phrase, cache)
        if found is not None:
            return _Item("ok", found)
        if not _has_capital(phrase):
            notes.append(f'Treated "{phrase}" as search text: it is not a known place.')
            return _Item("ignored")
        return _Item(
            "unresolved", unresolved=self._place_unresolved(phrase, "not in the gazetteer")
        )

    @staticmethod
    def _place_unresolved(phrase: str, reason: str) -> Unresolved:
        question = (
            f'I don\'t know the place "{phrase}". Could you name a city, region or country, '
            "or leave it blank to search without a place?"
        )
        return Unresolved("place", phrase, f"Place not recognised ({reason}).", question)

    def _person_item(self, phrase: str, known: _People | None, notes: list[str]) -> _Item:
        if known is None:
            notes.append(f'No people directory is available, so "{phrase}" is used as search text.')
            return _Item("ignored")
        name, candidates = known.resolve(phrase)
        if name is not None:
            return _Item("ok", name)
        if candidates:
            return _Item("unresolved", unresolved=self._ambiguous_person(phrase, candidates))
        if not _has_capital(phrase):
            notes.append(f'Treated "{phrase}" as search text: it is not a known person.')
            return _Item("ignored")
        return _Item("unresolved", unresolved=self._person_unresolved(phrase, known))

    @staticmethod
    def _ambiguous_person(phrase: str, candidates: list[str]) -> Unresolved:
        shown = " or ".join(candidates[:4])
        question = f'Which "{phrase}" did you mean: {shown}?'
        return Unresolved("person", phrase, f"More than one person fits: {shown}.", question)

    @staticmethod
    def _person_unresolved(phrase: str, known: _People) -> Unresolved:
        close = difflib.get_close_matches(_fold(phrase), list(known.by_fold), n=1, cutoff=0.75)
        hint = f' Did you mean "{known.by_fold[close[0]]}"?' if close else ""
        question = (
            f'I don\'t know anyone called "{phrase}".{hint} Which person did you mean, '
            "or leave it blank to search without them?"
        )
        return Unresolved("person", phrase, "No person with that name.", question)

    # ------------------------------------------------------------------ clarification

    def _apply_clarifications(
        self,
        clarifications: Mapping[str, str],
        date_item: _Item,
        place_item: _Item,
        person_items: list[tuple[_Span, _Item]],
        known: _People | None,
        places: dict[str, PlaceFilter | None],
        today: date,
        notes: list[str],
    ) -> None:
        """Re-resolve unresolved items with the user's replacements, in place."""
        if not clarifications:
            return
        replies = {
            clarification_key(k): v
            for k, v in clarifications.items()
            if isinstance(k, str) and k.strip()
        }
        for item in (date_item, place_item, *(it for _, it in person_items)):
            original = item.unresolved
            if item.status != "unresolved" or original is None:
                continue
            key = clarification_key(original.phrase)
            if key not in replies:
                continue
            reply = replies[key] if isinstance(replies[key], str) else ""
            if not reply.strip():
                item.status, item.value, item.unresolved = "skipped", None, None
                # give the words back unless the phrase really was a date
                item.restore = original.kind != "date"
                notes.append(f'Skipped the {original.kind} "{original.phrase}" at your request.')
                continue
            reply = " ".join(reply.split())
            value, reason = self._reword(original.kind, reply, known, places, today)
            if value is not None:
                item.status, item.value, item.unresolved = "ok", value, None
                notes.append(f'Understood "{reply}" for the {original.kind} "{original.phrase}".')
            else:
                item.unresolved = Unresolved(
                    original.kind,
                    original.phrase,
                    f'Replacement "{reply}" was not understood either: {reason}',
                    self._still_unclear(original.kind, reply, today),
                )

    def _still_unclear(self, kind: str, reply: str, today: date) -> str:
        if kind == "date":
            found = analyse_dates(reply, today, self._south)
            if found is not None and not found.resolution.understood and found.question:
                return found.question
        return (
            f'I still didn\'t understand "{reply}" as a {kind}. '
            "Could you try different wording, or leave it blank to skip it?"
        )

    def _reword(
        self,
        kind: str,
        reply: str,
        known: _People | None,
        places: dict[str, PlaceFilter | None],
        today: date,
    ) -> tuple[Any, str]:
        if kind == "date":
            result = resolve_date_phrase(reply, today, self._south)
            return (result.range, result.reason) if result.understood else (None, result.reason)
        if kind == "place":
            found = self._resolve_place(reply, places)
            return (found, "") if found is not None else (None, "not a known place")
        if known is None:  # pragma: no cover - unresolved people imply a directory
            return None, "no people directory"
        name, candidates = known.resolve(reply)
        if name is not None:
            return name, ""
        if candidates:
            return None, f"more than one person fits: {' or '.join(candidates[:4])}"
        parts = [known.resolve(p)[0] for p in re.split(r"\s*(?:,|&|\band\b)\s*", reply)]
        if len(parts) > 1 and all(parts):
            return tuple(parts), ""  # "Bob and Alice": several people for one slot
        return None, "not a known person"


def _bare_note(phrase: str) -> str:
    return (
        f'Treated "{phrase}" as search text: a bare season, holiday, month or weekday only '
        'counts as a date with a year, "last/this/next" or a word like "in" before it.'
    )


def _date_question(phrase: str, today: date, south: bool = False) -> str:
    """The question for a date phrase that was not understood.

    Operator phrases ("after 2019", "2019 or 2020") get the same specific question the query
    scanner gives them, so a phrase typed directly (a clarification reply, an LLM phrase) is
    treated like the same words inside a query.
    """
    found = analyse_dates(phrase, today, south)
    if found is not None and not found.resolution.understood and found.question:
        return found.question
    return (
        f'I didn\'t understand "{phrase}" as a date. Could you reword it, '
        'e.g. "March 2024" or "2024-03-31"?'
    )


def _hint(finding: DateSpan) -> DateSpan | None:
    return None if finding.resolution.understood else finding


def _touches(finding: DateSpan, span: _Span) -> bool:
    ranges = [(finding.start, finding.end), *finding.parts]
    return _overlaps((span.start, span.end), ranges)


def _covers(q: _Query, spans: list[tuple[str, Literal["span", "person"]]]) -> bool:
    """Whether the spans account for every word of the query."""
    rest = q.raw
    for phrase, _ in spans:
        rest = re.sub(re.escape(phrase), " ", rest, flags=re.IGNORECASE)
    return not _strip_edges(_WORD_RE.findall(rest))


def _joins(query: str, left: re.Match[str], right: re.Match[str]) -> bool:
    gap = query[left.end() : right.start()].strip()
    return gap in {"", ","} or (gap == "." and left.group().lower() in _ABBREVIATIONS)


def _after_preposition(query: str, tokens: list[re.Match[str]], i: int) -> bool:
    j = i - 1
    if j >= 0 and tokens[j].group().lower() in {"the", "a", "an"}:
        j -= 1
    if j < 0 or query[tokens[j].end() : tokens[j + 1].start()].strip():
        return False
    return tokens[j].group().lower() in _PLACE_PREPS
