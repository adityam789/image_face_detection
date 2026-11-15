"""
facesorter/search/dates.py
Date phrase -> inclusive ``DateRange``, with an explicit "understood?" signal.

The resolver is a small, deterministic grammar (seasons, holidays, relative phrases, months,
years, explicit dates, "between X and Y") with ``dateparser`` as a strictly guarded fallback
for odd absolute spellings. The one rule that shapes everything: **never silently widen**.
If any word of the phrase is not understood the result is ``understood=False`` with a reason,
so the caller can ask the user instead of searching the wrong period ("Easter 2024" must not
become "all of 2024").

Conventions (all documented here so callers and users get the same answers):

* Ranges are inclusive calendar dates. Weeks run Monday-Sunday.
* Seasons are meteorological: northern hemisphere spring Mar-May, summer Jun-Aug, autumn/fall
  Sep-Nov, winter Dec-Feb; with ``southern_hemisphere=True`` summer is Dec-Feb, autumn
  Mar-May, winter Jun-Aug, spring Sep-Nov. A winter/summer that spans New Year is labelled by
  the year it ends in: "winter 2020" is Dec 2019 - Feb 2020.
* ``last <X>`` for a month, season, holiday or weekday is the most recent *completed*
  occurrence (it ended before ``today``); a bare ``<X>`` is the most recent occurrence that has
  *started* (so it may still be in progress); ``next <X>`` is the first occurrence that starts
  after ``today``; ``this <X>`` is the occurrence containing ``today`` or, failing that, the
  one starting in today's calendar year ("this Christmas" in November is the coming one).
  "Last summer" in November 2026 is summer 2026, in July 2026 it is summer 2025.
* ``this week/month/year`` and ``next ...`` return the whole calendar period, so the range may
  end after ``today`` (harmless for photos: nothing was taken in the future).
* ``last week/month/year`` is the previous calendar period. ``last weekend`` is the Saturday
  and Sunday of the previous Monday-Sunday week; ``this weekend`` is this week's Saturday and
  Sunday. Bare ``weekend`` means ``last weekend``.
* ``last/past N days|weeks|months|years`` is a rolling window ending today.
  ``N units ago`` is the calendar unit (day / ISO week / month / year) containing that moment.
* Holidays use United States conventions and get a small window around the day: Christmas,
  Halloween, Easter, Thanksgiving +/-1 day; Memorial Day and Labor Day are the Saturday-Monday
  long weekend. Lunar and religious-calendar holidays (Diwali, Eid, Lunar New Year, Hanukkah,
  ...) are recognised but reported as not understood: a future ``holidays`` dependency would
  let them be computed.
* Times of day ("yesterday evening") are ignored: the result is the whole day.
* Ambiguous numeric dates such as ``3/4/2024`` or ``03.04.2019`` are rejected; use
  ``2024-03-04``.
* Operators change a date's meaning, so they are never ignored: "since 2020", "before June",
  "except 2019", "not in summer", "2019 or 2020", "early 2020" are all reported as not
  understood (only one closed range can be applied), never reduced to the bare date.
* Only English date words are understood; other languages ("été", "Weihnachten") are
  reported rather than guessed.

Heavy imports (``dateparser``) are lazy and optional: without it only the fallback is lost.
"""

from __future__ import annotations

import calendar
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from .types import DateRange

logger = logging.getLogger(__name__)

__all__ = [
    "DateResolution",
    "DateSpan",
    "analyse_dates",
    "bare_date_word",
    "find_date_phrases",
    "looks_like_date",
    "preposition_before",
    "resolve_date_phrase",
]


@dataclass(frozen=True)
class DateResolution:
    """Outcome of resolving a phrase. ``range`` is ``None`` exactly when not understood."""

    range: DateRange | None
    understood: bool
    reason: str = ""


@dataclass(frozen=True)
class DateSpan:
    """A date phrase found inside a longer query: ``text[start:end] == phrase``.

    ``question`` is a ready-made, friendly question for a span that is not understood (empty:
    use a generic one). ``kind`` is the grammar's category (``"year"``, ``"season"``, ...,
    ``"operator"``, ``"conflict"``). ``parts`` lists the ``(start, end)`` offsets of every
    date mention involved in a conflict or operator span.
    """

    start: int
    end: int
    phrase: str
    resolution: DateResolution
    question: str = ""
    kind: str = ""
    parts: tuple[tuple[int, int], ...] = ()


# --------------------------------------------------------------------------- vocabulary

_MONTHS: dict[str, int] = {}
for _i in range(1, 13):
    _MONTHS[calendar.month_name[_i].lower()] = _i
    _MONTHS[calendar.month_abbr[_i].lower()] = _i
_MONTHS["sept"] = 9

_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}

_SEASONS_NH = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2),
}
_SEASONS_SH = {
    "summer": (12, 2),
    "autumn": (3, 5),
    "fall": (3, 5),
    "winter": (6, 8),
    "spring": (9, 11),
}

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

# Words dropped wherever they appear / at the start of a phrase, before matching.
_DROP_ANYWHERE = {"the", "of"}
_DROP_LEADING = {"in", "on", "at", "from", "during", "around", "for"}
_TIME_OF_DAY = {"morning", "afternoon", "evening", "night"}
_YEAR = r"(?:19|20)\d{2}"
_UNITS = ("day", "week", "month", "year")

# Lunar / religious calendars: recognised, deliberately not computed. Single words trigger
# only next to a year or last/this/next; the multi-word names are needed for words that are
# ordinary English on their own ("Chinese restaurant", "lunar eclipse").
_UNSUPPORTED_WORDS = {
    "diwali",
    "deepavali",
    "eid",
    "ramadan",
    "hanukkah",
    "chanukah",
    "hannukah",
    "passover",
    "pesach",
    "purim",
    "sukkot",
    "shavuot",
    "hashanah",
    "kippur",
    "holi",
    "navratri",
    "dussehra",
    "vesak",
}
_UNSUPPORTED_PHRASES = (
    ("chinese", "new", "year"),
    ("lunar", "new", "year"),
    ("rosh", "hashanah"),
    ("yom", "kippur"),
)
# Date words of other languages (accent-folded): reported, never guessed.
_FOREIGN_DATE_WORDS = {
    "ete", "verano", "primavera", "invierno", "otono", "inverno", "outono", "sommer",
    "fruhling", "herbst", "automne", "hiver", "printemps", "autunno", "zomer", "herfst",
    "weihnachten", "navidad", "nochebuena", "natale", "ostern", "pascua", "paques",
    "pasqua", "kerst", "silvester",
}  # fmt: skip
_YEAR_BLOCKERS = {
    "number", "no", "num", "nr", "version", "ver", "v", "model", "route", "room", "windows",
    "office", "iphone", "ipad", "galaxy", "pixel", "type", "series", "edition", "class", "grade",
    "gen", "generation", "rev", "release", "level", "chapter", "page", "section", "unit",
    "apartment", "apt", "suite", "flight", "bus", "line", "track", "bmw", "audi", "ford", "ps",
    "xbox", "playstation",
}  # fmt: skip


def _eom(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) ``weekday`` of a month, or the last one for ``n == -1``."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = _eom(year, month)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Western (Gregorian) Easter Sunday, by the anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _around(day: Callable[[int], date], before: int = 0, after: int = 0) -> _RangeFn:
    return lambda y: (day(y) - timedelta(days=before), day(y) + timedelta(days=after))


def _fixed(month: int, day: int, before: int = 0, after: int = 0) -> _RangeFn:
    return _around(lambda y: date(y, month, day), before, after)


def _new_years_eve(year: int) -> tuple[date, date]:
    return date(year, 12, 31), date(year + 1, 1, 1)


def _long_weekend(day: Callable[[int], date]) -> _RangeFn:
    """Saturday through the holiday Monday."""
    return _around(day, before=2)


_RangeFn = Callable[[int], "tuple[date, date]"]

_HOLIDAYS: dict[str, _RangeFn] = {
    "christmas": _fixed(12, 25, 1, 1),
    "christmas eve": _fixed(12, 24),
    "boxing day": _fixed(12, 26),
    "new years eve": _new_years_eve,
    "new years": _fixed(1, 1),
    "halloween": _fixed(10, 31, 1, 1),
    "valentines day": _fixed(2, 14),
    "independence day": _fixed(7, 4),
    "st patricks day": _fixed(3, 17),
    "cinco de mayo": _fixed(5, 5),
    "juneteenth": _fixed(6, 19),
    "easter": _around(_easter, 1, 1),
    "good friday": _around(lambda y: _easter(y) - timedelta(days=2)),
    "thanksgiving": _around(lambda y: _nth_weekday(y, 11, 3, 4), 1, 1),
    "black friday": _around(lambda y: _nth_weekday(y, 11, 3, 4) + timedelta(days=1)),
    "memorial day": _long_weekend(lambda y: _nth_weekday(y, 5, 0, -1)),
    "labor day": _long_weekend(lambda y: _nth_weekday(y, 9, 0, 1)),
    "mothers day": _around(lambda y: _nth_weekday(y, 5, 6, 2)),
    "fathers day": _around(lambda y: _nth_weekday(y, 6, 6, 3)),
}
_HOLIDAY_ALIASES = {
    "christmas day": "christmas",
    "xmas": "christmas",
    "xmas eve": "christmas eve",
    "new year eve": "new years eve",
    "new years day": "new years",
    "new year": "new years",
    "valentines": "valentines day",
    "valentine day": "valentines day",
    "fourth july": "independence day",
    "july fourth": "independence day",
    "st patricks": "st patricks day",
    "saint patricks day": "st patricks day",
    "easter sunday": "easter",
    "thanksgiving day": "thanksgiving",
    "mother day": "mothers day",
    "father day": "fathers day",
}
_HOLIDAY_NAMES = {**{name: name for name in _HOLIDAYS}, **_HOLIDAY_ALIASES}

_Q = r"(?P<q>last|this|next)"
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_SEASON_ALT = "|".join(_SEASONS_NH)
_WEEKDAY_ALT = "|".join(_WEEKDAYS)
_HOLIDAY_ALT = "|".join(re.escape(n) for n in sorted(_HOLIDAY_NAMES, key=len, reverse=True))
_NUM_ALT = "|".join(["\\d{1,4}", *_NUMBER_WORDS])
_UNIT_ALT = "|".join(_UNITS)

# Every word the grammar can use; the scanner only tries spans made of these.
_VOCAB = (
    set(_MONTHS)
    | set(_WEEKDAYS)
    | set(_SEASONS_NH)
    | set(_NUMBER_WORDS)
    | _DROP_ANYWHERE
    | _DROP_LEADING
    | _TIME_OF_DAY
    | _UNSUPPORTED_WORDS
    | {w for phrase in _UNSUPPORTED_PHRASES for w in phrase}
    | {w for name in _HOLIDAY_NAMES for w in name.split()}
    | {"last", "this", "next", "past", "ago", "between", "and", "to", "through", "thru"}
    | {"until", "today", "yesterday", "tomorrow", "tonight", "before", "weekend", "new", "year"}
    | {u + s for u in (*_UNITS, "hour", "minute") for s in ("", "s")}
)
# Words that make a phrase "date-like" even if it cannot be resolved (so it is worth asking
# the user about instead of silently ignoring it).
_DATE_SIGNAL = (
    set(_MONTHS)
    | set(_WEEKDAYS)
    | set(_SEASONS_NH)
    | _UNSUPPORTED_WORDS
    | {w for name in _HOLIDAY_NAMES for w in name.split() if w not in {"day", "new", "de"}}
    | {"yesterday", "today", "tomorrow", "tonight", "weekend", "ago", "weeks", "months", "years"}
    | {"week", "month", "year"}
)
_RELATIVE_WORDS = {"ago", "last", "next", "this", "past", "now", "today", "tomorrow", "yesterday"}
_OPERATOR_WORDS = {"or", "and"}  # joining two dates; the other operators are listed below
_NOT_SUPPORTED_WORDS = {
    "since",
    "until",  # only a range separator ("June until August") when the grammar matched
    "before",
    "after",
    "till",
    "by",
    "except",
    "excluding",
    "not",
    "without",
    "onwards",
    "onward",
    "afterwards",
    "later",
    "earlier",
    "pre",
    "post",
    "prior",
    "early",
    "late",
    "mid",
    "middle",
    "end",
    "beginning",
    "start",
    "about",
    "approximately",
    "circa",
}
_EDGE_START = _DROP_ANYWHERE | _DROP_LEADING | {"and", "to"}
_EDGE_WORDS = _EDGE_START | {"between"}
_NUMERIC = re.compile(r"\d+(?:st|nd|rd|th)?|[\d/.\-]+")
_PREPS = {"in", "on", "at", "during", "for", "from", "since", "until", "by", "over", "around"}
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|\w+(?:['’\-/]\w+)*")
_MAX_SCAN_TOKENS = 400
_MAX_SPAN_TOKENS = 8
_MIN_YEAR, _MAX_YEAR_AHEAD = 1900, 10


# --------------------------------------------------------------------------- internals


@dataclass(frozen=True)
class _Hit:
    start: date
    end: date
    kind: str  # season | holiday | month | weekday | year | relative | range | date | fallback
    bare: bool = False  # no year / last / this / next given: an ordinary word could be meant


class _NotUnderstood(Exception):
    def __init__(self, reason: str, *, bare: bool = False, invalid: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.bare = bare
        self.invalid = invalid  # a plausible date that cannot exist ("31 February")


@dataclass(frozen=True)
class _Outcome:
    hit: _Hit | None
    reason: str = ""
    bare: bool = False  # for not-understood outcomes that are still "recognised"
    invalid: bool = False  # recognised as a date, but an impossible one


@dataclass(frozen=True)
class _Ctx:
    today: date
    south: bool

    @property
    def seasons(self) -> dict[str, tuple[int, int]]:
        return _SEASONS_SH if self.south else _SEASONS_NH


_KEEP_S = {"valentine", "mother", "father", "patrick"}  # holiday names keep their "s"
_ORDINAL = re.compile(r"(\d{1,2})(?:st|nd|rd|th)")


def _canon_words(raw_tokens: list[str]) -> list[str]:
    """Lower-case words with apostrophes resolved: "year's" -> "year", "New Year's" -> "years"."""
    out: list[str] = []
    for raw in raw_tokens:
        word = raw.lower().replace("\u2019", "'").replace("\u2018", "'")
        if word.endswith("'s"):
            base = word[:-2]
            keep = base in _KEEP_S or (base == "year" and bool(out) and out[-1] == "new")
            word = base + "s" if keep else base
        else:
            word = word.replace("'", "")
        match = _ORDINAL.fullmatch(word)
        out.append(match.group(1) if match else word)
    return out


def _normalise(phrase: str) -> list[str]:
    text = re.sub(r"(?<=\d)\.(?=\d)", "/", phrase)  # 03.04.2019 -> 03/04/2019
    text = re.sub(r"[^\w/\-'\u2019\u2018]+", " ", text)
    tokens = [t for t in _canon_words(text.split()) if t and t not in _DROP_ANYWHERE]
    while tokens and tokens[0] in _DROP_LEADING:
        tokens.pop(0)
    if len(tokens) > 1 and tokens[-1] in _TIME_OF_DAY:
        tokens.pop()
        if tokens == ["last"]:
            tokens = ["yesterday"]
        elif tokens == ["this"]:
            tokens = ["today"]
    return tokens


def _shift_months(day: date, n: int) -> date:
    year, month0 = divmod(day.year * 12 + day.month - 1 + n, 12)
    month = month0 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _pick(fn: _RangeFn, today: date, mode: str, what: str) -> tuple[date, date]:
    """Choose the occurrence of ``fn(year)`` that ``mode`` asks for.

    ``completed``: latest that ended before today. ``started``: latest that began on or before
    today. ``upcoming``: earliest that begins after today. ``this``: the one containing today,
    else the one that begins in today's calendar year.
    """
    years = range(today.year - 1, today.year + 3)
    if mode != "upcoming":
        years = range(today.year + 2, today.year - 120, -1)
    instances: list[tuple[date, date]] = []
    for year in years:
        try:
            instances.append(fn(year))
        except (ValueError, OverflowError):  # e.g. Feb 29 in a common year
            continue
    if mode == "this":  # prefer the instance containing today over one starting this year
        instances = [i for i in instances if i[0] <= today <= i[1]] + instances
    for start, end in instances:
        if (
            (mode == "this" and (start <= today <= end or start.year == today.year))
            or (mode == "completed" and end < today)
            or (mode == "started" and start <= today)
            or (mode == "upcoming" and start > today)
        ):
            return start, end
    raise _NotUnderstood(f'No matching occurrence of "{what}" was found.')


def _mode(qualifier: str | None) -> str:
    return {"last": "completed", "next": "upcoming", "this": "this"}.get(qualifier or "", "started")


def _season_fn(ctx: _Ctx, season: str) -> _RangeFn:
    first, last = ctx.seasons[season]

    def fn(year: int) -> tuple[date, date]:
        start = date(year - 1 if first > last else year, first, 1)
        return start, _eom(year, last)

    return fn


def _month_fn(month: int) -> _RangeFn:
    return lambda y: (date(y, month, 1), _eom(y, month))


def _qualified(
    fn: _RangeFn, qualifier: str | None, year: str | None, ctx: _Ctx, kind: str, what: str
) -> _Hit:
    """Shared handling of ``[last|this|next] <thing> [YEAR]`` for months/seasons/holidays."""
    if year is not None:
        if qualifier:
            raise _NotUnderstood(f'"{qualifier}" and a year cannot be combined for "{what}".')
        start, end = fn(int(year))
        return _Hit(start, end, kind)
    start, end = _pick(fn, ctx.today, _mode(qualifier), what)
    return _Hit(start, end, kind, bare=qualifier is None)


def _h_simple_day(m: re.Match[str], ctx: _Ctx) -> _Hit:
    word = m.group(0)
    offset = {
        "today": 0,
        "tonight": 0,
        "yesterday": -1,
        "tomorrow": 1,
        "day before yesterday": -2,
    }[word]
    day = ctx.today + timedelta(days=offset)
    return _Hit(day, day, "relative")


def _number(text: str) -> int:
    return int(text) if text.isdigit() else _NUMBER_WORDS[text]


def _h_ago(m: re.Match[str], ctx: _Ctx) -> _Hit:
    n, unit = _number(m.group("n")), m.group("u")
    today = ctx.today
    if unit in ("hour", "minute"):
        return _Hit(today, today, "relative")
    if unit == "day":
        day = today - timedelta(days=n)
        return _Hit(day, day, "relative")
    if unit == "week":
        monday = _monday(today - timedelta(weeks=n))
        return _Hit(monday, monday + timedelta(days=6), "relative")
    if unit == "month":
        day = _shift_months(today, -n)
        return _Hit(date(day.year, day.month, 1), _eom(day.year, day.month), "relative")
    year = today.year - n
    return _Hit(date(year, 1, 1), date(year, 12, 31), "relative")


def _h_rolling(m: re.Match[str], ctx: _Ctx) -> _Hit:
    n = _number(m.group("n")) if m.groupdict().get("n") else 1
    unit = m.group("u")
    today = ctx.today
    if n < 1:
        raise _NotUnderstood("A rolling window needs at least one unit.")
    if unit == "day":
        start = today - timedelta(days=n - 1)
    elif unit == "week":
        start = today - timedelta(days=7 * n - 1)
    elif unit == "month":
        start = _shift_months(today, -n) + timedelta(days=1)
    else:
        start = _shift_months(today, -12 * n) + timedelta(days=1)
    return _Hit(start, today, "relative")


def _h_unit(m: re.Match[str], ctx: _Ctx) -> _Hit:
    q, unit, today = m.group("q"), m.group("u"), ctx.today
    step = {"last": -1, "this": 0, "next": 1}[q]
    if unit == "week":
        monday = _monday(today) + timedelta(weeks=step)
        return _Hit(monday, monday + timedelta(days=6), "relative")
    if unit == "month":
        day = _shift_months(today, step)
        return _Hit(date(day.year, day.month, 1), _eom(day.year, day.month), "relative")
    year = today.year + step
    return _Hit(date(year, 1, 1), date(year, 12, 31), "relative")


def _h_weekend(m: re.Match[str], ctx: _Ctx) -> _Hit:
    q = m.groupdict().get("q") or "last"
    offset = {"last": -2, "this": 5, "next": 12}[q]
    saturday = _monday(ctx.today) + timedelta(days=offset)
    bare = m.group("q") is None and "past" not in m.group(0)
    return _Hit(saturday, saturday + timedelta(days=1), "relative", bare=bare)


def _h_season(m: re.Match[str], ctx: _Ctx) -> _Hit:
    season = m.group("s")
    qualifier = m.groupdict().get("q") or ("last" if "past" in m.group(0).split() else None)
    return _qualified(_season_fn(ctx, season), qualifier, m.group("y"), ctx, "season", season)


def _h_holiday(m: re.Match[str], ctx: _Ctx) -> _Hit:
    name = _HOLIDAY_NAMES[m.group("h")]
    return _qualified(_HOLIDAYS[name], m.groupdict().get("q"), m.group("y"), ctx, "holiday", name)


def _h_weekday(m: re.Match[str], ctx: _Ctx) -> _Hit:
    q, w, today = m.group("q"), _WEEKDAYS[m.group("w")], ctx.today
    if q == "this":
        day = _monday(today) + timedelta(days=w)
    elif q == "next":
        day = today + timedelta(days=(w - today.weekday()) % 7 or 7)
    elif q == "last":
        day = today - timedelta(days=(today.weekday() - w) % 7 or 7)
    else:
        day = today - timedelta(days=(today.weekday() - w) % 7)
    return _Hit(day, day, "weekday", bare=q is None)


def _h_month(m: re.Match[str], ctx: _Ctx) -> _Hit:
    month = _MONTHS[m.group("m")]
    what = m.group("m")
    return _qualified(_month_fn(month), m.groupdict().get("q"), m.group("y"), ctx, "month", what)


def _valid_day(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError:
        if 1 <= month <= 12 and 1 <= day <= 31:
            name = calendar.month_name[month]
            raise _NotUnderstood(f"{name} {year} has no day {day}.", invalid=True) from None
        raise _NotUnderstood(f"{year:04d}-{month:02d}-{day:02d} is not a real date.") from None


def _h_month_day(m: re.Match[str], ctx: _Ctx) -> _Hit:
    month, day = _MONTHS[m.group("m")], int(m.group("d"))
    if m.group("y"):
        found = _valid_day(int(m.group("y")), month, day)
        return _Hit(found, found, "date")
    if not 1 <= day <= calendar.monthrange(2024, month)[1]:  # 2024: leap year, Feb has 29
        if 1 <= day <= 31:
            raise _NotUnderstood(f"{calendar.month_name[month]} has no day {day}.", invalid=True)
        raise _NotUnderstood(f"{m.group('m')} {day} is not a real date.")
    start, end = _pick(
        lambda y: (date(y, month, day), date(y, month, day)), ctx.today, "started", "date"
    )
    return _Hit(start, end, "date")


def _h_iso(m: re.Match[str], ctx: _Ctx) -> _Hit:
    year, month = int(m.group("y")), int(m.group("m"))
    if not 1 <= month <= 12:
        raise _NotUnderstood(f"{month} is not a month.")
    if m.group("d") is None:
        return _Hit(date(year, month, 1), _eom(year, month), "month")
    found = _valid_day(year, month, int(m.group("d")))
    return _Hit(found, found, "date")


def _h_numeric(m: re.Match[str], ctx: _Ctx) -> _Hit:
    a, b, year = int(m.group("a")), int(m.group("b")), int(m.group("y"))
    if a > 12 and b <= 12:
        day, month = a, b
    elif b > 12 and a <= 12:
        month, day = a, b
    elif a == b:
        month = day = a
    elif a <= 12 and b <= 12:
        raise _NotUnderstood(
            f'"{a}/{b}/{year}" is ambiguous (day/month or month/day); write it as {year}-MM-DD.'
        )
    else:
        raise _NotUnderstood(f'"{a}/{b}/{year}" is not a real date.')
    found = _valid_day(year, month, day)
    return _Hit(found, found, "date")


def _h_year(m: re.Match[str], ctx: _Ctx) -> _Hit:
    year = int(m.group("y"))
    return _Hit(date(year, 1, 1), date(year, 12, 31), "year", bare=True)


_Handler = Callable[["re.Match[str]", _Ctx], _Hit]


_RULES: list[tuple[re.Pattern[str], _Handler]] = [
    (re.compile(r"today|tonight|yesterday|tomorrow|day before yesterday"), _h_simple_day),
    (
        re.compile(rf"(?P<n>{_NUM_ALT}) (?P<u>{_UNIT_ALT}|hour|minute)s? ago"),
        _h_ago,
    ),
    (re.compile(rf"(?:last|past) (?P<n>{_NUM_ALT}) (?P<u>{_UNIT_ALT})s?"), _h_rolling),
    (re.compile(rf"past (?P<u>{_UNIT_ALT})"), _h_rolling),
    (re.compile(rf"{_Q} (?P<u>week|month|year)"), _h_unit),
    (re.compile(rf"(?:{_Q} |(?:this )?past )?weekend"), _h_weekend),
    (
        re.compile(rf"(?:{_Q} |(?:this )?past )?(?P<s>{_SEASON_ALT})(?: (?P<y>{_YEAR}))?"),
        _h_season,
    ),
    (re.compile(rf"(?:{_Q} )?(?P<h>{_HOLIDAY_ALT})(?: (?P<y>{_YEAR}))?"), _h_holiday),
    (re.compile(rf"(?P<y>{_YEAR}) (?P<s>{_SEASON_ALT})"), _h_season),
    (re.compile(rf"(?P<y>{_YEAR}) (?P<h>{_HOLIDAY_ALT})"), _h_holiday),
    (re.compile(rf"(?P<y>{_YEAR}) (?P<m>{_MONTH_ALT})"), _h_month),
    (re.compile(rf"(?:{_Q} )?(?P<w>{_WEEKDAY_ALT})"), _h_weekday),
    (re.compile(rf"(?:{_Q} )?(?P<m>{_MONTH_ALT})(?: (?P<y>{_YEAR}))?"), _h_month),
    (re.compile(rf"(?P<m>{_MONTH_ALT}) (?P<d>\d{{1,2}})(?: (?P<y>{_YEAR}))?"), _h_month_day),
    (re.compile(rf"(?P<d>\d{{1,2}}) (?P<m>{_MONTH_ALT})(?: (?P<y>{_YEAR}))?"), _h_month_day),
    (re.compile(rf"(?P<y>{_YEAR})[-/](?P<m>\d{{2}})(?:[-/](?P<d>\d{{2}}))?"), _h_iso),
    (re.compile(rf"(?P<a>\d{{1,2}})[/-](?P<b>\d{{1,2}})[/-](?P<y>{_YEAR})"), _h_numeric),
    (re.compile(rf"(?:year )?(?P<y>{_YEAR})"), _h_year),
]


def _endpoint(tokens: list[str]) -> tuple[int | None, int | None] | None:
    """``(month, year)`` for ``june`` / ``june 2021`` / ``2021``; ``None`` if anything else."""
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in _MONTHS:
            return _MONTHS[tok], None
        if re.fullmatch(_YEAR, tok):
            return None, int(tok)
    elif len(tokens) == 2 and tokens[0] in _MONTHS and re.fullmatch(_YEAR, tokens[1]):
        return _MONTHS[tokens[0]], int(tokens[1])
    return None


def _range(tokens: list[str], ctx: _Ctx) -> _Hit | None:
    between = tokens[0] == "between"
    body = tokens[1:] if between else tokens
    for i, word in enumerate(body):
        if word in ("to", "through", "thru", "until") or (word == "and" and between):
            left, right = _endpoint(body[:i]), _endpoint(body[i + 1 :])
            break
    else:
        return None
    if left is None or right is None:
        return None
    (m1, y1), (m2, y2) = left, right
    if m1 is None and m2 is None:
        assert y1 is not None and y2 is not None
        if y1 > y2:
            raise _NotUnderstood("The date range runs backwards.")
        return _Hit(date(y1, 1, 1), date(y2, 12, 31), "range")
    if m1 is None or m2 is None:
        raise _NotUnderstood("A range cannot mix a month and a bare year.")
    if y1 is None and y2 is None:

        def fn(year: int) -> tuple[date, date]:
            return date(year if m1 <= m2 else year - 1, m1, 1), _eom(year, m2)

        start, end = _pick(fn, ctx.today, "started", "range")
        return _Hit(start, end, "range", bare=True)
    if y1 is None:
        assert y2 is not None
        y1 = y2 if m1 <= m2 else y2 - 1
    if y2 is None:
        y2 = y1 if m2 >= m1 else y1 + 1
    start, end = date(y1, m1, 1), _eom(y2, m2)
    if end < start:
        raise _NotUnderstood("The date range runs backwards.")
    return _Hit(start, end, "range")


def _unsupported_name(tokens: list[str]) -> str | None:
    """The lunar/religious holiday named in ``tokens``, if any."""
    for word in tokens:
        if word in _UNSUPPORTED_WORDS:
            return word
    for phrase in _UNSUPPORTED_PHRASES:
        size = len(phrase)
        for k in range(len(tokens) - size + 1):
            if tuple(tokens[k : k + size]) == phrase:
                return " ".join(phrase)
    return None


def _grammar(tokens: list[str], ctx: _Ctx) -> _Hit | None:
    name = _unsupported_name(tokens)
    if name is not None:
        anchored = any(re.fullmatch(_YEAR, t) or t in {"last", "this", "next"} for t in tokens)
        raise _NotUnderstood(
            f'"{name}" depends on a lunar or religious calendar, which is not supported yet.',
            bare=not anchored,
        )
    text = " ".join(tokens)
    for pattern, handler in _RULES:
        m = pattern.fullmatch(text)
        if m:
            return handler(m, ctx)
    return _range(tokens, ctx)


def _fallback(tokens: list[str], ctx: _Ctx) -> _Hit | None:
    """Ask ``dateparser`` only for single, fully covered absolute-looking phrases."""
    # Only absolute-looking text: dateparser reads "5 months" as "5 months ago" and similar.
    has_anchor = any(t in _MONTHS or re.fullmatch(_YEAR, t) for t in tokens)
    if not has_anchor or any(
        t in _RELATIVE_WORDS | _OPERATOR_WORDS | _NOT_SUPPORTED_WORDS for t in tokens
    ):
        return None  # operators ("after 2019") change the meaning: dateparser must not drop them
    numbers = [t for t in tokens if t[:1].isdigit()]
    if any(("-" in t or "/" in t) for t in tokens) or len(numbers) > 2:
        return None  # numeric forms are the grammar's job; dateparser guesses US order
    if all(t[:1].isdigit() for t in tokens):
        return None
    years = [k for k, t in enumerate(tokens) if re.fullmatch(_YEAR, t)]
    if years and any(len(t) <= 2 and t.isdigit() for t in tokens[years[0] + 1 :]):
        return None  # "may 2024 5": a stray number after the year
    try:
        from dateparser.date import DateDataParser
        from dateparser.search import search_dates
    except ImportError:
        return None
    text = " ".join(tokens)
    settings = {
        "RELATIVE_BASE": datetime(ctx.today.year, ctx.today.month, ctx.today.day),
        "PREFER_DATES_FROM": "past",
    }
    try:
        hits = search_dates(text, languages=["en"], settings=settings)
        if not hits or len(hits) != 1:
            return None
        piece, found = hits[0]
        if piece.lower() not in text or text.replace(piece.lower(), " ", 1).split():
            return None  # words outside the match: do not guess
        period = DateDataParser(languages=["en"], settings=settings).get_date_data(piece).period
    except Exception:  # dateparser raises a zoo of errors on odd input
        logger.debug("dateparser failed on %r", text, exc_info=True)
        return None
    day = found.date()
    if not _MIN_YEAR <= day.year <= ctx.today.year + _MAX_YEAR_AHEAD:
        return None
    if period == "year":
        return _Hit(date(day.year, 1, 1), date(day.year, 12, 31), "fallback")
    if period == "month":
        return _Hit(date(day.year, day.month, 1), _eom(day.year, day.month), "fallback")
    if period == "week":
        monday = _monday(day)
        return _Hit(monday, monday + timedelta(days=6), "fallback")
    return _Hit(day, day, "fallback")


def _explain(tokens: list[str]) -> str:
    if "or" in tokens or "and" in tokens:
        return "Only a single date or one range can be applied, not a list or a choice."
    vague = [t for t in tokens if t in _NOT_SUPPORTED_WORDS]
    if vague:
        return f'"{vague[0]}" changes what the date means; only one closed range is supported.'
    unknown = [t for t in tokens if t not in _VOCAB and not t[:1].isdigit()]
    if unknown:
        return f'Not recognised as part of a date: "{" ".join(unknown)}".'
    return "The wording is not a date form I know."


def _evaluate(phrase: str, ctx: _Ctx, fallback: bool) -> _Outcome:
    try:
        tokens = _normalise(phrase)
        if not tokens:
            return _Outcome(None, "The date phrase is empty.")
        hit = _grammar(tokens, ctx)
        if hit is None and fallback:
            hit = _fallback(tokens, ctx)
        if hit is None:
            return _Outcome(None, _explain(tokens))
        return _Outcome(hit)
    except _NotUnderstood as exc:
        return _Outcome(None, exc.reason, exc.bare, exc.invalid)
    except (ValueError, OverflowError) as exc:
        return _Outcome(None, f"The date is out of the supported range ({exc}).")


# --------------------------------------------------------------------------- public API


def resolve_date_phrase(
    phrase: str, today: date, southern_hemisphere: bool = False
) -> DateResolution:
    """Resolve a date phrase to an inclusive ``DateRange``.

    Returns ``understood=False`` (with a human-readable ``reason`` and ``range=None``) when any
    part of the phrase is not understood; it never guesses and never raises.
    """
    outcome = _evaluate(str(phrase or ""), _Ctx(today, southern_hemisphere), fallback=True)
    if outcome.hit is None:
        return DateResolution(None, False, outcome.reason)
    return DateResolution(DateRange(outcome.hit.start, outcome.hit.end), True, "")


def bare_date_word(phrase: str, today: date, southern_hemisphere: bool = False) -> bool:
    """True if ``phrase`` is just a season, holiday, month, weekday or weekend word.

    "autumn", "Christmas", "May", "Monday", "weekend", "June to August" are bare: with no year,
    no last/this/next and no leading preposition they are most likely ordinary words ("autumn
    leaves", "Christmas tree"). "in autumn", "last autumn", "autumn 2023" are not bare, and a
    plain year ("2019") is never bare. Callers that accept model-supplied date phrases use this
    with ``preposition_before`` to apply the same policy as the rules scanner.
    """
    text = str(phrase or "")
    first = _TOKEN_RE.search(text)
    if first and first.group().lower() in _PREPS | {"of"}:
        return False
    hit = _evaluate(text, _Ctx(today, southern_hemisphere), fallback=False).hit
    return hit is not None and hit.bare and hit.kind != "year"


def preposition_before(text: str, start: int) -> bool:
    """Whether ``text[start:]`` is immediately preceded by in/on/at/during/for/from/of/... .

    A single "the"/"a"/"an" in between is skipped ("during the summer").
    """
    words = [m for m in _TOKEN_RE.finditer(text[:start])]
    if not words or text[words[-1].end() : start].strip(" "):
        return False
    prev = words[-1].group().lower()
    if prev in {"the", "a", "an"} and len(words) > 1:
        prev = words[-2].group().lower()
    return prev in _PREPS | {"of"}


def looks_like_date(phrase: str) -> bool:
    """True if ``phrase`` contains any date-bearing word or digit (resolvable or not).

    Used to tell "a date I do not understand" (worth asking about) from "not a date at all".
    """
    tokens = _normalise(str(phrase or ""))
    return any(t in _DATE_SIGNAL or any(c.isdigit() for c in t) for t in tokens)


# --------------------------------------------------------------------------- scanner

_BEFORE_OPS = {
    "since": "open", "before": "open", "after": "open", "until": "open", "till": "open",
    "by": "open", "pre": "open", "post": "open", "prior": "open",
    "early": "part", "late": "part", "mid": "part", "middle": "part", "end": "part",
    "beginning": "part", "start": "part",
    "not": "exclude", "except": "exclude", "excluding": "exclude", "without": "exclude",
}  # fmt: skip
_AFTER_OPS = {"onwards": "open", "onward": "open", "later": "open", "earlier": "open",
              "afterwards": "open", "forward": "open"}  # fmt: skip
_OP_SKIP = {"of", "the", "in", "on", "at", "during", "a", "an", "from"}
_JOINERS = {"and", "or", "&", ",", "to", "vs", "then", "plus"}
_DATELIKE_NUMERIC = re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}[./-]\d{1,2}(?:[./-]\d{1,2})?")
_MAX_WEAK_DISTANCE = 120


@dataclass(frozen=True)
class _Tok:
    start: int
    end: int
    word: str


@dataclass(frozen=True)
class _Scan:
    spans: list[DateSpan]  # accepted, longest first
    weak: list[DateSpan]  # date-ish words that were rejected as ordinary text
    toks: list[_Tok]


def _tokenise(text: str) -> list[_Tok]:
    matches = list(_TOKEN_RE.finditer(text))[:_MAX_SCAN_TOKENS]
    words = _canon_words([m.group() for m in matches])
    return [_Tok(m.start(), m.end(), w) for m, w in zip(matches, words)]


def _fold_word(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _year_context_ok(text: str, toks: list[_Tok], i: int, j: int) -> bool:
    """A bare year is a date unless it looks like a number: "Windows 2000", "route 2050"."""
    first, last = toks[i], toks[j]
    if first.start > 0 and text[first.start - 1] == "#":
        return False
    prev = toks[i - 1] if i > 0 and not text[toks[i - 1].end : first.start].strip(" ") else None
    nxt = (
        toks[j + 1]
        if j + 1 < len(toks) and not text[last.end : toks[j + 1].start].strip(" ")
        else None
    )
    loose = toks[i - 1] if i > 0 and not text[toks[i - 1].end : first.start].strip(" .") else None
    if loose is not None and loose.word in _YEAR_BLOCKERS:
        return False
    if prev is not None and (prev.word in _YEAR_BLOCKERS or prev.word[:1].isdigit()):
        return False
    if prev is not None and prev.word in _BEFORE_OPS:
        return True  # "early 2020", "since 2020": analyse_dates reports the operator
    after = toks[j + 2] if j + 2 < len(toks) and toks[j + 1].word in {"and", "or"} else None
    if nxt is not None and nxt.word in _AFTER_OPS or after is not None and after.word in _AFTER_OPS:
        return True  # "2020 onwards", "2020 or earlier"
    connective = _PREPS | {"the", "a", "an", "of", "and", "to"}
    # in the middle of a sentence, between two ordinary words: "Windows 2000 screenshot"
    return not (
        prev is not None and prev.word not in connective and nxt is not None
        and nxt.word not in connective
    )  # fmt: skip


def _scan(text: str, ctx: _Ctx) -> _Scan:
    toks = _tokenise(text)
    words = [t.word for t in toks]
    spans: list[DateSpan] = []
    weak: list[DateSpan] = []
    for i, first in enumerate(toks):
        shown = text[first.start : first.end]
        if _fold_word(first.word) in _FOREIGN_DATE_WORDS:
            reason = f'"{shown}" is not an English date word; only English is supported.'
            weak.append(
                DateSpan(
                    first.start,
                    first.end,
                    shown,
                    DateResolution(None, False, reason),
                    kind="foreign",
                )
            )
        if words[i] in _EDGE_START:
            continue
        prev = None
        if i > 0 and not text[toks[i - 1].end : first.start].strip(" "):
            prev = words[i - 1]
            if prev in {"the", "a", "an"} and i > 1:
                prev = words[i - 2]
        for j in range(i, min(i + _MAX_SPAN_TOKENS, len(toks))):
            gap = text[toks[j - 1].end : toks[j].start] if j > i else ""
            if gap.strip(" ,."):
                break
            if words[j] not in _VOCAB and not _NUMERIC.fullmatch(words[j]):
                break
            if words[j] in _EDGE_WORDS:
                continue
            span = _classify(text, ctx, toks, i, j, prev)
            if span is None:
                continue
            if span.kind.startswith("weak"):  # rejected bare word: remember, do not accept
                weak.append(span)
            else:
                spans.append(span)
    spans.sort(key=lambda s: (-(s.end - s.start), s.start))
    return _Scan(spans, weak, toks)


def _classify(
    text: str, ctx: _Ctx, toks: list[_Tok], i: int, j: int, prev: str | None
) -> DateSpan | None:
    """The span for tokens ``i..j`` if it is a date (or a recognised non-date), else ``None``.

    A bare season/holiday/lunar word rejected as ordinary text comes back with kind ``weak``.
    """
    start, end = toks[i].start, toks[j].end
    phrase = text[start:end]
    outcome = _evaluate(phrase, ctx, fallback=False)
    hit = outcome.hit
    if hit is not None:
        span = DateSpan(
            start, end, phrase, DateResolution(DateRange(hit.start, hit.end), True), kind=hit.kind
        )
        if hit.kind == "year":
            plausible = 1970 <= hit.start.year <= ctx.today.year + 1
            return span if plausible and _year_context_ok(text, toks, i, j) else None
        if hit.bare and prev not in _PREPS:
            weak_kinds = {"season": "weak", "holiday": "weak", "month": "weak_month"}
            return replace(span, kind=weak_kinds[hit.kind]) if hit.kind in weak_kinds else None
        return span
    tokens = _normalise(phrase)
    unsupported = _unsupported_name(tokens) is not None
    numeric = outcome.invalid or (
        i == j and len(tokens) == 1 and _DATELIKE_NUMERIC.fullmatch(tokens[0])
    )
    if not (unsupported or numeric):
        return None
    if unsupported and toks[i].word in {"a", "an"}:
        return None  # "a Chinese restaurant": an unsupported span never starts with an article
    span = DateSpan(
        start,
        end,
        phrase,
        DateResolution(None, False, outcome.reason),
        question=(
            f"{outcome.reason} Which date did you mean?"
            if outcome.invalid and not unsupported
            else ""
        ),
        kind="unsupported" if unsupported else "numeric",
    )
    if unsupported and outcome.bare and prev not in _PREPS:
        return replace(span, kind="weak_unsupported")
    return span


def find_date_phrases(text: str, today: date, southern_hemisphere: bool = False) -> list[DateSpan]:
    """Find date phrases inside free text, best first (longest, then earliest).

    Only phrases the grammar understands - or recognises but cannot compute (lunar holidays,
    impossible numeric dates such as ``2024-02-30``, returned with ``understood=False``) - are
    returned. Ordinary words that merely look like dates are filtered out:

    * a bare season, holiday, month, weekday or weekend only counts after a preposition
      ("in summer", "at Christmas", "on Monday"); "summer vacation" and "Christmas tree" do not;
    * a bare year must be plausible (1970 to next year), must not follow a number-like word
      ("Windows 2000", "route 2050", "#2012") and must not sit between two ordinary words
      ("Windows 2000 screenshot"); "Bangkok 2019", "in 2019" and "2019 hiking" are fine;
    * spans never start or end with a filler word ("in", "the", ...).

    ``dateparser`` is never consulted here. Use ``analyse_dates`` for the single answer a
    query parser should act on.
    """
    return _scan(text, _Ctx(today, southern_hemisphere)).spans


def _contained(inner: DateSpan, outer: DateSpan) -> bool:
    return inner is not outer and outer.start <= inner.start and inner.end <= outer.end


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _quote(text: str) -> str:
    return '"' + text + '"'


def _conflict(text: str, maximal: list[DateSpan]) -> DateSpan:
    ordered = sorted(maximal, key=lambda s: s.start)
    chain = [ordered[0]]
    for nxt in ordered[1:]:
        gap = [w.lower() for w in re.findall(r"\w+|[&,]", text[chain[-1].end : nxt.start])]
        if nxt.start >= chain[-1].end and all(g in _JOINERS for g in gap):
            chain.append(nxt)
        else:
            break
    if len(chain) >= 2:
        start, end = chain[0].start, chain[-1].end
        options = chain
    else:
        start, end = ordered[0].start, ordered[0].end
        options = ordered[:2]
    shown = " or ".join(_quote(o.phrase) for o in options[:3])
    return DateSpan(
        start,
        end,
        text[start:end],
        DateResolution(
            None, False, f"More than one date was given ({shown}); only one range can be applied."
        ),
        question=(
            f"I can only filter on one date range. Which should I use: {shown}? "
            "(Or give one range covering both, e.g. "
            f'"{options[0].phrase} to {options[1].phrase}".)'
        ),
        kind="conflict",
        parts=tuple((o.start, o.end) for o in ordered),
    )


def _operator_span(text: str, span: DateSpan, toks: list[_Tok]) -> DateSpan | None:
    """``span`` extended over an adjacent operator word ("since", "not", "onwards"), if any."""
    inside = [k for k, t in enumerate(toks) if t.start >= span.start and t.end <= span.end]
    lo, hi = inside[0], inside[-1]
    start, end, ops = span.start, span.end, []
    k, skipped = lo - 1, 0
    while k >= 0 and toks[k].word in _OP_SKIP and skipped < 2:
        k, skipped = k - 1, skipped + 1
    if (
        k >= 0
        and toks[k].word in _BEFORE_OPS
        and not text[toks[k].end : toks[k + 1].start].strip(" ,")
    ):
        start, ops = toks[k].start, [toks[k].word]
    elif (
        lo >= 2
        and toks[lo - 1].word == "or"
        and (toks[lo - 2].word[:1].isdigit() or toks[lo - 2].word in _NUMBER_WORDS)
    ):
        start, ops = toks[lo - 2].start, ["or"]  # "2 or 3 years ago"
    k = hi + 1
    if k < len(toks) and toks[k].word in {"and", "or"}:
        k += 1
    if k < len(toks) and toks[k].word in _AFTER_OPS:
        end, ops = toks[k].end, [*ops, toks[k].word]
    if not ops:
        return None
    kind = _BEFORE_OPS.get(ops[0]) or _AFTER_OPS.get(ops[0]) or "choice"
    phrase = text[start:end]
    if kind == "exclude":
        question = (
            f'I can\'t exclude dates ("{phrase}"). Please say which dates you do want, '
            'e.g. "2020 to 2022", or leave it blank.'
        )
    elif kind == "part":
        question = (
            f'I can\'t narrow to part of a period ("{phrase}"). Could you give a month or a '
            f'range, e.g. "June {span.phrase}"?'
            if not any(c.isalpha() for c in span.phrase)
            else f'I can\'t narrow to part of a period ("{phrase}"). Could you give a month or '
            'a range, e.g. "June 2019" or "June to August 2019"?'
        )
    else:
        question = (
            f'I can\'t filter on open-ended dates like "{phrase}". Could you give a start '
            'and an end, e.g. "2020 to 2026", or leave it blank?'
        )
    return DateSpan(
        start,
        end,
        phrase,
        DateResolution(
            None,
            False,
            f'"{ops[0]}" changes what the date means; only one closed range is supported.',
        ),
        question=question,
        kind="operator",
        parts=((span.start, span.end),),
    )


def _weak_span(text: str, year: DateSpan, weak: list[DateSpan]) -> DateSpan | None:
    """A date word that was rejected as ordinary text, sitting next to a bare year."""
    near = [
        w
        for w in weak
        if w.kind != "weak_month"
        and not _overlap((w.start, w.end), (year.start, year.end))
        and min(abs(w.start - year.end), abs(year.start - w.end)) <= _MAX_WEAK_DISTANCE
    ]
    if not near:
        return None
    word = min(near, key=lambda w: min(abs(w.start - year.end), abs(year.start - w.end)))
    lo, hi = sorted([word, year], key=lambda s: s.start)
    between = text[lo.end : hi.start]
    why = word.resolution.reason or "is read as ordinary text here"
    if not re.search(r"\w", between):  # directly next to each other: one phrase
        phrase = text[lo.start : hi.end]
        reason = (
            why
            if word.kind in {"foreign", "weak_unsupported"}
            else f'"{word.phrase}" and "{year.phrase}" together are not a date I know.'
        )
        return DateSpan(lo.start, hi.end, phrase, DateResolution(None, False, reason), kind="weak")
    reason = (
        why
        if word.kind in {"foreign", "weak_unsupported"}
        else f'"{word.phrase}" and "{year.phrase}" were found apart, so the date is unclear.'
    )
    question = (
        f'I found "{word.phrase}" and "{year.phrase}" in separate places. Did you mean '
        f'"{word.phrase} {year.phrase}"? Please reword the date, e.g. "{word.phrase} '
        f'{year.phrase}", or leave it blank.'
    )
    if word.kind in {"foreign", "weak_unsupported"}:
        question = (
            f'I didn\'t understand "{word.phrase}" next to "{year.phrase}" as a date '
            f"({why.rstrip('.')}). Could you reword it, e.g. "
            f'"November {year.phrase}" or "{year.phrase}"?'
        )
    return DateSpan(
        year.start,
        year.end,
        year.phrase,
        DateResolution(None, False, reason),
        question=question,
        kind="weak",
        parts=((word.start, word.end), (year.start, year.end)),
    )


def analyse_dates(text: str, today: date, southern_hemisphere: bool = False) -> DateSpan | None:
    """The one date finding a query parser should act on, or ``None``.

    Unlike ``find_date_phrases`` this never reduces what the user wrote to a smaller, wrongly
    meaning date:

    * two or more separate date mentions ("June 2020 and December 2021", "2019 or 2020",
      "Bob last summer, birthday 2019") -> a not-understood ``conflict`` span asking which one;
    * an operator next to the date ("since 2020", "before June", "not in summer",
      "2020 onwards", "early 2020") -> a not-understood ``operator`` span;
    * a bare year next to or near a season / holiday word that was not accepted as a date
      ("Easter eggs 2024", "Eid with Priya 2023", "Weihnachten 2019") -> a not-understood span,
      because the year alone would silently widen the search;
    * otherwise the best span (possibly itself not understood: lunar holidays, invalid numeric
      dates such as ``2024-02-30``).

    The returned span's ``phrase`` is the exact text to strip from the query. Spans that are not
    understood carry a friendly ``question``.
    """
    scan = _scan(text, _Ctx(today, southern_hemisphere))
    if not scan.spans:
        return None
    maximal = [s for s in scan.spans if not any(_contained(s, o) for o in scan.spans)]
    if len(maximal) > 1:
        return _conflict(text, maximal)
    best = maximal[0]
    if not best.resolution.understood:
        return best
    joined = _joined_weak(text, best, scan.weak)
    if joined is not None:
        return _conflict(text, [joined, best])
    return (
        _operator_span(text, best, scan.toks)
        or (_weak_span(text, best, scan.weak) if best.kind == "year" else None)
        or best
    )


def _joined_weak(text: str, best: DateSpan, weak: list[DateSpan]) -> DateSpan | None:
    """A rejected date word tied to ``best`` by "and"/"or"/"," ("Christmas and Easter 2022")."""
    for word in weak:
        lo, hi = sorted([word, best], key=lambda s: s.start)
        if lo.end > hi.start:
            continue
        gap = [g.lower() for g in re.findall(r"\w+|[&,]", text[lo.end : hi.start])]
        if gap and all(g in _JOINERS for g in gap):
            return word
    return None
