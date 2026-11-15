"""Tests for facesorter.search.dates: the date-phrase resolver and the in-text scanner."""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import date, datetime, timedelta

import pytest

from facesorter.search.dates import (
    DateResolution,
    analyse_dates,
    bare_date_word,
    find_date_phrases,
    looks_like_date,
    preposition_before,
    resolve_date_phrase,
)

TODAY = date(2026, 11, 7)  # a Saturday

needs_dateparser = pytest.mark.skipif(
    importlib.util.find_spec("dateparser") is None, reason="dateparser not installed"
)


def span(phrase: str, today: date = TODAY, south: bool = False):
    res = resolve_date_phrase(phrase, today, south)
    assert res.understood, f"{phrase!r}: {res.reason}"
    assert res.range is not None
    return res.range.start.isoformat(), res.range.end.isoformat()


def not_understood(phrase: str, today: date = TODAY) -> str:
    res = resolve_date_phrase(phrase, today)
    assert not res.understood, f"{phrase!r} unexpectedly understood as {res.range}"
    assert res.range is None
    assert res.reason
    return res.reason


# The labelled expectations from the research spike (research/qset.py and heldout.py),
# reproduced with TODAY = 2026-11-07.
LABELLED = [
    ("yesterday", "2026-11-06", "2026-11-06"),
    ("last month", "2026-10-01", "2026-10-31"),
    ("in March 2024", "2024-03-01", "2024-03-31"),
    ("during 2023", "2023-01-01", "2023-12-31"),
    ("last summer", "2026-06-01", "2026-08-31"),
    ("summer 2019", "2019-06-01", "2019-08-31"),
    ("Christmas 2022", "2022-12-24", "2022-12-26"),
    ("last year", "2025-01-01", "2025-12-31"),
    ("this week", "2026-11-02", "2026-11-08"),
    ("between June and August 2021", "2021-06-01", "2021-08-31"),
    ("New Year's Eve 2024", "2024-12-31", "2025-01-01"),
    ("last weekend", "2026-10-31", "2026-11-01"),
    ("Halloween 2023", "2023-10-30", "2023-11-01"),
    ("December 2022", "2022-12-01", "2022-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("July 2023", "2023-07-01", "2023-07-31"),
    ("May 2025", "2025-05-01", "2025-05-31"),
    ("November 2025", "2025-11-01", "2025-11-30"),
    ("last week", "2026-10-26", "2026-11-01"),
    ("February 2024", "2024-02-01", "2024-02-29"),
    ("2019", "2019-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    # held-out set
    ("last autumn", "2025-09-01", "2025-11-30"),
    ("spring 2018", "2018-03-01", "2018-05-31"),
    ("winter 2020", "2019-12-01", "2020-02-29"),
    ("Fourth of July 2021", "2021-07-04", "2021-07-04"),
    ("Valentine's Day 2023", "2023-02-14", "2023-02-14"),
    ("this year", "2026-01-01", "2026-12-31"),
    ("June 2022", "2022-06-01", "2022-06-30"),
    ("last Christmas", "2025-12-24", "2025-12-26"),
    ("between March and May 2019", "2019-03-01", "2019-05-31"),
    ("the summer of 2017", "2017-06-01", "2017-08-31"),
    ("yesterday evening", "2026-11-06", "2026-11-06"),
    ("Christmas", "2025-12-24", "2025-12-26"),
    ("Easter 2024", "2024-03-30", "2024-04-01"),
    ("Thanksgiving 2023", "2023-11-22", "2023-11-24"),
]


@pytest.mark.parametrize(("phrase", "start", "end"), LABELLED)
def test_labelled_phrases(phrase, start, end):
    assert span(phrase) == (start, end)


def test_result_type_and_reason():
    ok = resolve_date_phrase("last year", TODAY)
    assert isinstance(ok, DateResolution)
    assert ok.understood and ok.reason == "" and ok.range is not None
    with pytest.raises(AttributeError):
        ok.understood = False  # frozen


# ----------------------------------------------------------------- no hard-coded dates


@pytest.mark.parametrize("offset", range(0, 800, 7))
def test_relative_phrases_follow_today(offset):
    today = date(2023, 1, 1) + timedelta(days=offset)
    assert span("yesterday", today) == ((today - timedelta(1)).isoformat(),) * 2
    assert span("today", today) == (today.isoformat(),) * 2
    assert span("tomorrow", today) == ((today + timedelta(1)).isoformat(),) * 2
    week = resolve_date_phrase("this week", today).range
    assert week.start.weekday() == 0 and week.end.weekday() == 6
    assert week.start <= today <= week.end
    last = resolve_date_phrase("last week", today).range
    assert last.end == week.start - timedelta(1)
    assert span("this year", today) == (f"{today.year}-01-01", f"{today.year}-12-31")
    assert span("last year", today) == (f"{today.year - 1}-01-01", f"{today.year - 1}-12-31")
    month = resolve_date_phrase("this month", today).range
    assert month.start.day == 1 and month.start <= today <= month.end
    # the weekend is always Saturday+Sunday, before this Monday-Sunday week
    weekend = resolve_date_phrase("last weekend", today).range
    assert weekend.start.weekday() == 5 and weekend.end.weekday() == 6
    assert weekend.end < week.start
    assert weekend.start >= week.start - timedelta(2)


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 11, 7), ("2026-06-01", "2026-08-31")),
        (date(2026, 7, 15), ("2025-06-01", "2025-08-31")),  # this summer is not complete
        (date(2026, 8, 31), ("2025-06-01", "2025-08-31")),  # ends today: not completed
        (date(2026, 9, 1), ("2026-06-01", "2026-08-31")),
        (date(2020, 1, 10), ("2019-06-01", "2019-08-31")),
    ],
)
def test_last_summer_is_most_recent_completed(today, expected):
    assert span("last summer", today) == expected


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 11, 7), ("2025-12-01", "2026-02-28")),
        (date(2026, 1, 15), ("2024-12-01", "2025-02-28")),  # this winter is not over yet
        (date(2024, 5, 1), ("2023-12-01", "2024-02-29")),  # leap February
    ],
)
def test_last_winter(today, expected):
    assert span("last winter", today) == expected


def test_this_and_next_season():
    assert span("this summer", date(2026, 7, 4)) == ("2026-06-01", "2026-08-31")
    assert span("this summer", TODAY) == ("2026-06-01", "2026-08-31")  # most recent started
    assert span("this winter", date(2026, 1, 15)) == ("2025-12-01", "2026-02-28")
    assert span("this winter", TODAY) == ("2026-12-01", "2027-02-28")  # the coming one
    assert span("this spring", TODAY) == ("2026-03-01", "2026-05-31")
    assert span("this december", TODAY) == ("2026-12-01", "2026-12-31")
    assert span("this march", TODAY) == ("2026-03-01", "2026-03-31")
    assert span("next summer", TODAY) == ("2027-06-01", "2027-08-31")
    assert span("next winter", TODAY) == ("2026-12-01", "2027-02-28")
    assert span("this past summer", TODAY) == ("2026-06-01", "2026-08-31")
    assert span("past summer", date(2026, 7, 4)) == ("2025-06-01", "2025-08-31")
    assert span("summer", TODAY) == ("2026-06-01", "2026-08-31")  # bare = most recent started
    assert span("fall 2019", TODAY) == ("2019-09-01", "2019-11-30")


def test_southern_hemisphere_seasons():
    assert span("summer 2020", TODAY, south=True) == ("2019-12-01", "2020-02-29")
    assert span("winter 2020", TODAY, south=True) == ("2020-06-01", "2020-08-31")
    assert span("last winter", TODAY, south=True) == ("2026-06-01", "2026-08-31")
    assert span("autumn 2021", TODAY, south=True) == ("2021-03-01", "2021-05-31")
    assert span("spring 2021", TODAY, south=True) == ("2021-09-01", "2021-11-30")
    assert span("last summer", TODAY, south=True) == ("2025-12-01", "2026-02-28")


def test_christmas_follows_today():
    assert span("Christmas", date(2026, 12, 25)) == ("2026-12-24", "2026-12-26")
    assert span("Christmas", date(2026, 12, 20)) == ("2025-12-24", "2025-12-26")
    assert span("last Christmas", date(2026, 12, 27)) == ("2026-12-24", "2026-12-26")
    assert span("last Christmas", date(2026, 12, 26)) == ("2025-12-24", "2025-12-26")
    assert span("next Christmas", TODAY) == ("2026-12-24", "2026-12-26")
    assert span("this Christmas", TODAY) == ("2026-12-24", "2026-12-26")
    assert span("this Christmas", date(2027, 12, 31)) == ("2027-12-24", "2027-12-26")
    assert span("this Christmas", date(2027, 12, 25)) == ("2027-12-24", "2027-12-26")
    assert span("this new years eve", date(2027, 1, 1)) == ("2026-12-31", "2027-01-01")


def test_future_phrases_are_allowed():
    assert span("tomorrow") == ("2026-11-08", "2026-11-08")
    assert span("next week") == ("2026-11-09", "2026-11-15")
    assert span("next month") == ("2026-12-01", "2026-12-31")
    assert span("next year") == ("2027-01-01", "2027-12-31")
    assert span("this weekend") == ("2026-11-07", "2026-11-08")
    assert span("next weekend") == ("2026-11-14", "2026-11-15")
    assert span("next friday") == ("2026-11-13", "2026-11-13")
    assert span("december") == ("2025-12-01", "2025-12-31")  # bare month: most recent started
    assert span("next december") == ("2026-12-01", "2026-12-31")
    assert span("2030") == ("2030-01-01", "2030-12-31")


# ----------------------------------------------------------------- relative phrases


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("3 days ago", ("2026-11-04", "2026-11-04")),
        ("a day ago", ("2026-11-06", "2026-11-06")),
        ("2 weeks ago", ("2026-10-19", "2026-10-25")),
        ("a week ago", ("2026-10-26", "2026-11-01")),
        ("three weeks ago", ("2026-10-12", "2026-10-18")),
        ("2 months ago", ("2026-09-01", "2026-09-30")),
        ("a month ago", ("2026-10-01", "2026-10-31")),
        ("11 months ago", ("2025-12-01", "2025-12-31")),
        ("2 years ago", ("2024-01-01", "2024-12-31")),
        ("an hour ago", ("2026-11-07", "2026-11-07")),
        ("30 minutes ago", ("2026-11-07", "2026-11-07")),
        ("last 3 days", ("2026-11-05", "2026-11-07")),
        ("past 2 weeks", ("2026-10-25", "2026-11-07")),
        ("last 2 months", ("2026-09-08", "2026-11-07")),
        ("past 1 years", ("2025-11-08", "2026-11-07")),
        ("past week", ("2026-11-01", "2026-11-07")),
        ("past year", ("2025-11-08", "2026-11-07")),
        ("last night", ("2026-11-06", "2026-11-06")),
        ("this morning", ("2026-11-07", "2026-11-07")),
        ("tonight", ("2026-11-07", "2026-11-07")),
        ("yesterday morning", ("2026-11-06", "2026-11-06")),
        ("day before yesterday", ("2026-11-05", "2026-11-05")),
        ("Christmas morning", ("2025-12-24", "2025-12-26")),
    ],
)
def test_relative_phrases(phrase, expected):
    assert span(phrase) == expected


def test_months_ago_clamps_day_of_month():
    assert span("1 month ago", date(2026, 3, 31)) == ("2026-02-01", "2026-02-28")
    assert span("12 months ago", date(2024, 2, 29)) == ("2023-02-01", "2023-02-28")
    assert span("1 year ago", date(2024, 2, 29)) == ("2023-01-01", "2023-12-31")


def test_weekdays():
    # TODAY is a Saturday
    assert span("saturday") == ("2026-11-07", "2026-11-07")
    assert span("last saturday") == ("2026-10-31", "2026-10-31")
    assert span("friday") == ("2026-11-06", "2026-11-06")
    assert span("on Monday") == ("2026-11-02", "2026-11-02")
    assert span("last monday") == ("2026-11-02", "2026-11-02")
    assert span("this sunday") == ("2026-11-08", "2026-11-08")
    assert span("this monday") == ("2026-11-02", "2026-11-02")
    assert span("next monday") == ("2026-11-09", "2026-11-09")
    assert span("next saturday") == ("2026-11-14", "2026-11-14")
    assert span("sunday") == ("2026-11-01", "2026-11-01")


def test_weekend_variants():
    assert span("weekend") == span("last weekend") == ("2026-10-31", "2026-11-01")
    assert span("this past weekend") == ("2026-10-31", "2026-11-01")
    assert span("the weekend") == ("2026-10-31", "2026-11-01")
    sunday = date(2026, 11, 8)
    assert span("last weekend", sunday) == ("2026-10-31", "2026-11-01")
    assert span("this weekend", sunday) == ("2026-11-07", "2026-11-08")
    monday = date(2026, 11, 9)
    assert span("last weekend", monday) == ("2026-11-07", "2026-11-08")


# ----------------------------------------------------------------- months, dates, ranges


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("march", ("2026-03-01", "2026-03-31")),
        ("in November", ("2026-11-01", "2026-11-30")),  # in progress counts as started
        ("last november", ("2025-11-01", "2025-11-30")),
        ("last march", ("2026-03-01", "2026-03-31")),
        ("sept 2020", ("2020-09-01", "2020-09-30")),
        ("Mar 2021", ("2021-03-01", "2021-03-31")),
        ("2024-03", ("2024-03-01", "2024-03-31")),
        ("2024-03-31", ("2024-03-31", "2024-03-31")),
        ("2024/03/31", ("2024-03-31", "2024-03-31")),
        ("March 3, 2024", ("2024-03-03", "2024-03-03")),
        ("March 3rd 2024", ("2024-03-03", "2024-03-03")),
        ("3 March 2024", ("2024-03-03", "2024-03-03")),
        ("3rd of March 2021", ("2021-03-03", "2021-03-03")),
        ("March 3", ("2026-03-03", "2026-03-03")),
        ("December 25", ("2025-12-25", "2025-12-25")),
        ("on the 4th of July", ("2026-07-04", "2026-07-04")),
        ("Feb 29", ("2024-02-29", "2024-02-29")),
        ("15/4/2024", ("2024-04-15", "2024-04-15")),
        ("4/15/2024", ("2024-04-15", "2024-04-15")),
        ("5/5/2024", ("2024-05-05", "2024-05-05")),
        ("the year 2020", ("2020-01-01", "2020-12-31")),
        ("in 1999", ("1999-01-01", "1999-12-31")),
        ("2018 summer", ("2018-06-01", "2018-08-31")),
        ("2022 December", ("2022-12-01", "2022-12-31")),
        ("2022 Christmas", ("2022-12-24", "2022-12-26")),
    ],
)
def test_months_and_explicit_dates(phrase, expected):
    assert span(phrase) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("between June and August 2021", ("2021-06-01", "2021-08-31")),
        ("between March and May 2019", ("2019-03-01", "2019-05-31")),
        ("between November and February 2021", ("2020-11-01", "2021-02-28")),
        ("between June 2020 and August 2021", ("2020-06-01", "2021-08-31")),
        ("between June 2020 and August", ("2020-06-01", "2020-08-31")),
        ("between November 2020 and February", ("2020-11-01", "2021-02-28")),
        ("from June to August 2021", ("2021-06-01", "2021-08-31")),
        ("june through august", ("2026-06-01", "2026-08-31")),
        ("between November and February", ("2026-11-01", "2027-02-28")),
        ("between 2019 and 2021", ("2019-01-01", "2021-12-31")),
        ("2019 to 2021", ("2019-01-01", "2021-12-31")),
    ],
)
def test_ranges(phrase, expected):
    assert span(phrase) == expected


# ----------------------------------------------------------------- holidays


def test_movable_holidays_known_dates():
    assert span("Easter 2024") == ("2024-03-30", "2024-04-01")
    assert span("Easter 2025") == ("2025-04-19", "2025-04-21")
    assert span("easter sunday 2026") == ("2026-04-04", "2026-04-06")
    assert span("Good Friday 2024") == ("2024-03-29", "2024-03-29")
    assert span("Thanksgiving 2024") == ("2024-11-27", "2024-11-29")
    assert span("Black Friday 2024") == ("2024-11-29", "2024-11-29")
    assert span("Memorial Day 2024") == ("2024-05-25", "2024-05-27")
    assert span("Labor Day 2024") == ("2024-08-31", "2024-09-02")
    assert span("Mother's Day 2024") == ("2024-05-12", "2024-05-12")
    assert span("Father's Day 2024") == ("2024-06-16", "2024-06-16")
    assert span("Easter") == ("2026-04-04", "2026-04-06")
    assert span("last thanksgiving") == ("2025-11-26", "2025-11-28")
    assert span("this thanksgiving") == ("2026-11-25", "2026-11-27")


def test_easter_matches_dateutil_for_many_years():
    easter = pytest.importorskip("dateutil.easter")
    for year in range(1900, 2100):
        res = resolve_date_phrase(f"Easter {year}", TODAY).range
        assert res.start + timedelta(days=1) == easter.easter(year)
        assert res.end - timedelta(days=1) == easter.easter(year)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Christmas Eve 2023", ("2023-12-24", "2023-12-24")),
        ("xmas 2023", ("2023-12-24", "2023-12-26")),
        ("Boxing Day 2023", ("2023-12-26", "2023-12-26")),
        ("New Year's Day 2024", ("2024-01-01", "2024-01-01")),
        ("new year 2024", ("2024-01-01", "2024-01-01")),
        ("New Years Eve 2023", ("2023-12-31", "2024-01-01")),
        ("St. Patrick's Day 2022", ("2022-03-17", "2022-03-17")),
        ("Cinco de Mayo 2022", ("2022-05-05", "2022-05-05")),
        ("Juneteenth 2022", ("2022-06-19", "2022-06-19")),
        ("Independence Day 2022", ("2022-07-04", "2022-07-04")),
        ("valentines 2022", ("2022-02-14", "2022-02-14")),
        ("Halloween", ("2026-10-30", "2026-11-01")),
        ("last halloween", ("2026-10-30", "2026-11-01")),
        ("next halloween", ("2027-10-30", "2027-11-01")),
    ],
)
def test_fixed_holidays(phrase, expected):
    assert span(phrase) == expected


@pytest.mark.parametrize(
    "phrase",
    ["Diwali 2023", "Eid", "Lunar New Year 2024", "Chinese New Year", "Hanukkah", "last Passover"],
)
def test_lunar_holidays_are_not_understood(phrase):
    assert "lunar or religious" in not_understood(phrase)


# ----------------------------------------------------------------- never silently widen


@pytest.mark.parametrize(
    "phrase",
    [
        "",
        "   ",
        "red car",
        "birthday 2024",
        "wedding June 2021",
        "early March",
        "late 2019",
        "mid June 2020",
        "since 2020",
        "before Christmas",
        "end of May",
        "last summer 2019",
        "last March 2020",
        "3/4/2024",
        "February 30 2024",
        "Feb 30",
        "2024-02-30",
        "2024-13",
        "13/13/2024",
        "13/14/2024",
        "last 0 days",
        "5 months",
        "june and august",
        "march to 2021",
        "2021 to 2019",
        "between June 2021 and March 2021",
        "between August and June 2021 2020",
        "this past week",
        "summer 2019 and winter 2020",
        "Easter Bunny 2024",
        "2024 beach",
        "1850",
        "99999 years ago",
        "5000 years ago",
        "february 2000000",
    ],
)
def test_unrecognised_words_are_never_widened(phrase):
    assert not_understood(phrase)


def test_reasons_are_helpful():
    assert "empty" in not_understood("")
    assert "birthday" in not_understood("birthday 2024")
    assert "early" in not_understood("early March")
    assert "ambiguous" in not_understood("3/4/2024")
    assert "has no day 30" in not_understood("2024-02-30")
    assert "backwards" in not_understood("2021 to 2019")
    assert "mix a month" in not_understood("march to 2021")
    assert "cannot be combined" in not_understood("last summer 2019")
    assert not_understood("june and august")


def test_non_string_input_does_not_raise():
    for bad in (None, 5, 3.5, [], b"x"):
        res = resolve_date_phrase(bad, TODAY)  # type: ignore[arg-type]
        assert not res.understood


def test_extreme_today_does_not_raise():
    for today in (date.min, date.max):
        for phrase in ("last year", "yesterday", "last summer", "2 years ago", "Christmas"):
            res = resolve_date_phrase(phrase, today)
            assert isinstance(res.understood, bool)


def test_unicode_and_punctuation_noise():
    assert not_understood("märz 2024")
    assert not_understood("🎄")
    assert span("  IN   MARCH,   2024!! ") == ("2024-03-01", "2024-03-31")
    assert span("“Christmas” 2022") == ("2022-12-24", "2022-12-26")
    assert span("Valentine’s Day 2023") == ("2023-02-14", "2023-02-14")


# ----------------------------------------------------------------- looks_like_date


def test_looks_like_date():
    for phrase in ("Diwali", "red 2024", "monday", "last year", "3 weeks ago", "June", "xmas"):
        assert looks_like_date(phrase), phrase
    for phrase in ("red car", "at a wedding", "my dog", "", "the beach"):
        assert not looks_like_date(phrase), phrase
    assert not looks_like_date(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------- scanner


def found(text: str, today: date = TODAY):
    spans = find_date_phrases(text, today)
    return [s.phrase for s in spans]


def best(text: str, today: date = TODAY):
    spans = find_date_phrases(text, today)
    return spans[0].phrase if spans else None


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("photos from yesterday", "yesterday"),
        ("last month", "last month"),
        ("in March 2024", "March 2024"),
        ("during 2023", "2023"),
        ("Bob hiking last summer", "last summer"),
        ("sunset in Goa, December 2022", "December 2022"),
        ("between June and August 2021 at the lake", "between June and August 2021"),
        ("New Year's Eve 2024 party", "New Year's Eve 2024"),
        ("Halloween 2023", "Halloween 2023"),
        ("the summer of 2017", "summer of 2017"),
        ("fireworks on the 4th of July", "4th of July"),
        ("graduation ceremony May 2025", "May 2025"),
        ("kids at the park yesterday evening", "yesterday evening"),
        ("pics from 3 weeks ago", "3 weeks ago"),
        ("a week ago at the beach", "a week ago"),
        ("dinner on Monday", "Monday"),
        ("at Christmas", "Christmas"),
        ("in summer", "summer"),
        ("in the winter", "winter"),
        ("over the weekend", "weekend"),
        ("Sept. 2020 hike", "Sept. 2020"),
        ("St. Patrick's Day 2022", "St. Patrick's Day 2022"),
        ("Easter 2024 egg hunt", "Easter 2024"),
        ("2019-05-04 party", "2019-05-04"),
        ("trip in 2018", "2018"),
        ("last 3 days", "last 3 days"),
    ],
)
def test_scanner_finds_longest_phrase(text, phrase):
    assert best(text) == phrase


@pytest.mark.parametrize(
    "text",
    [
        "photos that may be blurry",
        "the march of the penguins",
        "women's march",
        "a fall in the snow",
        "people who fall down",
        "summer vacation",
        "winter coat",
        "spring flowers",
        "Christmas tree",
        "Halloween costume",
        "Easter eggs",
        "weekend trip",
        "monday blues",
        "autumn leaves",
        "I will march",
        "the sun and the moon",
        "Mom with the kids",
        "1800 people",
        "year 3000 prediction",
        "red car",
        "cats",
        "the last supper",
        "this is nice",
        "next to the sea",
        "five dogs",
        "a to z",
        "",
        "   ",
        "!!!",
    ],
)
def test_scanner_ignores_ordinary_words(text):
    assert found(text) == [], text


def test_scanner_requires_preposition_for_ambiguous_months():
    assert best("in may") == "may"
    assert best("during March") == "March"
    assert best("from the 5th of May") == "5th of May"
    assert best("May 2024") == "May 2024"
    assert best("may") is None
    assert best("march") is None
    assert best("Wedding in March") == "March"
    assert best("fall") is None
    assert best("in fall") == "fall"
    assert best("last fall") == "last fall"
    assert best("fall 2019") == "fall 2019"


def test_scanner_reports_unsupported_holidays_with_context():
    spans = find_date_phrases("dinner at Diwali 2023 with family", TODAY)
    assert spans[0].phrase == "Diwali 2023"
    assert not spans[0].resolution.understood and spans[0].resolution.range is None
    assert "lunar" in spans[0].resolution.reason
    assert found("Diwali lights") == []  # bare: probably visual
    assert found("at Diwali") == ["Diwali"]
    assert found("last Eid") == ["last Eid"]


def test_scanner_does_not_join_across_other_punctuation():
    assert sorted(found("in march; 2024")) == ["2024", "march"]  # never "march; 2024"
    assert best("Goa. Yesterday") == "Yesterday"


def test_scanner_span_offsets():
    text = "Bob and the dog in Goa, December 2022 at sunset"
    (first, *_) = find_date_phrases(text, TODAY)
    assert text[first.start : first.end] == first.phrase == "December 2022"
    assert first.resolution.understood
    assert first.resolution.range.start == date(2022, 12, 1)


def test_scanner_orders_longest_first_and_offers_alternatives():
    spans = find_date_phrases("Christmas 2022 or last summer", TODAY)
    lengths = [s.end - s.start for s in spans]
    assert lengths == sorted(lengths, reverse=True)
    phrases = {s.phrase for s in spans}
    assert {"Christmas 2022", "last summer"} <= phrases


def test_scanner_is_bounded_on_huge_input():
    text = "word " * 50_000 + "yesterday"
    assert find_date_phrases(text, TODAY) == []  # scanning is capped; and it is quick
    text = "march 2024 " * 5000
    assert find_date_phrases(text, TODAY)


def test_scanner_southern_hemisphere():
    spans = find_date_phrases("winter 2020", TODAY, southern_hemisphere=True)
    assert spans[0].resolution.range.start == date(2020, 6, 1)


# ----------------------------------------------------------------- dateparser fallback


@needs_dateparser
def test_dateparser_fallback_handles_odd_absolute_forms():
    assert span("Sunday, March 3, 2024") == ("2024-03-03", "2024-03-03")


@needs_dateparser
def test_dateparser_cannot_widen_unknown_words():
    # dateparser alone would happily return the whole year for these
    assert not_understood("Shabbat 2024")
    assert not_understood("red Mar 3 2024")
    assert not_understood("fortnight 2024")


def _stub_dateparser(monkeypatch, hits, period="day", boom=False):
    pkg = types.ModuleType("dateparser")
    date_mod = types.ModuleType("dateparser.date")
    search_mod = types.ModuleType("dateparser.search")

    class DateData:
        def __init__(self, period):
            self.period = period

    class DateDataParser:
        def __init__(self, languages=None, settings=None):
            pass

        def get_date_data(self, text):
            return DateData(period)

    def search_dates(text, languages=None, settings=None):
        if boom:
            raise RuntimeError("boom")
        return hits(text) if callable(hits) else hits

    date_mod.DateDataParser = DateDataParser
    search_mod.search_dates = search_dates
    for name, mod in (
        ("dateparser", pkg),
        ("dateparser.date", date_mod),
        ("dateparser.search", search_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("year", ("2020-01-01", "2020-12-31")),
        ("month", ("2020-02-01", "2020-02-29")),
        ("week", ("2020-02-10", "2020-02-16")),
        ("day", ("2020-02-12", "2020-02-12")),
        ("time", ("2020-02-12", "2020-02-12")),
    ],
)
def test_fallback_period_mapping(monkeypatch, period, expected):
    _stub_dateparser(monkeypatch, [("zzz 2020", datetime(2020, 2, 12))], period)
    assert span("zzz 2020") == expected


def test_fallback_requires_full_coverage_and_single_hit(monkeypatch):
    _stub_dateparser(monkeypatch, [("2020", datetime(2020, 2, 12))])
    assert not_understood("zzz 2020")  # "zzz" is not covered by the match
    _stub_dateparser(
        monkeypatch, [("zzz 2020", datetime(2020, 2, 12)), ("1", datetime(2020, 1, 1))]
    )
    assert not_understood("zzz 2020")
    _stub_dateparser(monkeypatch, [])
    assert not_understood("zzz 2020")
    _stub_dateparser(monkeypatch, [("other 2020", datetime(2020, 2, 12))])
    assert not_understood("zzz 2020")


def test_fallback_rejects_implausible_years_and_errors(monkeypatch):
    _stub_dateparser(monkeypatch, [("zzz 2020", datetime(1066, 2, 12))])
    assert not_understood("zzz 2020")
    _stub_dateparser(monkeypatch, [("zzz 2020", datetime(2020, 2, 12))], boom=True)
    assert not_understood("zzz 2020")


def test_fallback_skipped_without_date_words(monkeypatch):
    calls = []
    _stub_dateparser(monkeypatch, lambda text: calls.append(text) or [])
    assert not_understood("red car")
    assert calls == []


def test_fallback_missing_dateparser_is_graceful(monkeypatch):
    monkeypatch.setitem(sys.modules, "dateparser", None)
    monkeypatch.setitem(sys.modules, "dateparser.date", None)
    monkeypatch.setitem(sys.modules, "dateparser.search", None)
    assert not_understood("zzz 2020")
    assert span("last summer") == ("2026-06-01", "2026-08-31")  # the grammar needs none of it


# ----------------------------------------------------------------- review round 2: regressions
# Operators must never be dropped, several dates never reduced to one, unknown date words never
# left next to a bare year, and ordinary words never read as dates.


def finding(text: str, today: date = TODAY):
    return analyse_dates(text, today)


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("photos since 2020", "since 2020"),
        ("pictures before 2015", "before 2015"),
        ("after 2019", "after 2019"),
        ("until June 2021", "until June 2021"),
        ("by 2019", "by 2019"),
        ("till March 2020", "till March 2020"),
        ("from 2019 onwards", "2019 onwards"),
        ("2019 and later", "2019 and later"),
        ("2019 or earlier", "2019 or earlier"),
        ("before Christmas 2022", "before Christmas 2022"),
        ("after Christmas 2022", "after Christmas 2022"),
        ("everything except 2019", "except 2019"),
        ("excluding summer 2019", "excluding summer 2019"),
        ("without 2019", "without 2019"),
        ("not in summer", "not in summer"),
        ("not yesterday", "not yesterday"),
        ("early 2020 hiking", "early 2020"),
        ("late summer 2019", "late summer 2019"),
        ("mid June 2020", "mid June 2020"),
        ("the end of 2019", "end of 2019"),
        ("beginning of 2020", "beginning of 2020"),
        ("2 or 3 years ago", "2 or 3 years ago"),
    ],
)
def test_operators_are_reported_not_dropped(text, phrase):
    got = finding(text)
    assert got is not None and got.kind == "operator"
    assert got.phrase == phrase and text[got.start : got.end] == phrase
    assert not got.resolution.understood and got.resolution.range is None
    assert got.resolution.reason and got.question
    assert phrase in got.question or got.phrase in got.question


def test_operator_questions_depend_on_the_kind():
    assert "open-ended" in finding("since 2020").question
    assert "exclude" in finding("except 2019").question
    assert "part of a period" in finding("early 2020").question
    assert "part of a period" in finding("early March 2020").question


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("June 2020 and December 2021", "June 2020 and December 2021"),
        ("2019 or 2020", "2019 or 2020"),
        ("Bob in 2019 and 2020", "2019 and 2020"),
        ("hiking in June or July 2020", "June or July 2020"),
        ("last year or this year", "last year or this year"),
        ("March 2024 and April 2024", "March 2024 and April 2024"),
        ("Christmas and Easter 2022", "Christmas and Easter 2022"),
        ("March, April 2024", "March, April 2024"),
    ],
)
def test_several_dates_are_a_conflict(text, phrase):
    got = finding(text)
    assert got is not None and got.kind == "conflict" and got.phrase == phrase
    assert not got.resolution.understood
    assert "Which should I use" in got.question and len(got.parts) >= 2


def test_distant_dates_conflict_on_the_first_only():
    got = finding("Bob last summer, birthday 2019")
    assert got.kind == "conflict" and got.phrase == "last summer"
    assert '"last summer" or "2019"' in got.question


def test_between_and_to_ranges_are_single_dates_not_conflicts():
    for text in ("between June and August 2021", "from March to May hiking", "2019 to 2021"):
        got = finding(text)
        assert got.resolution.understood and got.kind == "range"


@pytest.mark.parametrize(
    ("text", "phrase", "reason"),
    [
        ("party on 03/04/2024", "03/04/2024", "ambiguous"),
        ("party 03.04.2019", "03.04.2019", "ambiguous"),
        ("04.03.2020", "04.03.2020", "ambiguous"),
        ("trip on 2024-02-30", "2024-02-30", "has no day 30"),
        ("trip 2024-13", "2024-13", "not a month"),
        ("13/14/2024 trip", "13/14/2024", "not a real date"),
    ],
)
def test_bad_numeric_dates_are_reported(text, phrase, reason):
    got = finding(text)
    assert got.kind == "numeric" and got.phrase == phrase
    assert not got.resolution.understood and reason in got.resolution.reason


def test_valid_numeric_dates_and_numbers_that_are_not_dates():
    assert finding("photos 15.04.2019").resolution.range.start == date(2019, 4, 15)
    assert finding("2024-03-31 party").resolution.range.start == date(2024, 3, 31)
    for text in ("version 3.14", "build 1.5.2", "iPhone 15", "5 people", "pi is 3.14159"):
        assert finding(text) is None, text


@pytest.mark.parametrize(
    ("text", "phrase", "english_only"),
    [
        ("plage à Paris été 2019", "été 2019", True),
        ("Weihnachten 2019 in Berlin", "Weihnachten 2019", True),
        ("verano 2021", "verano 2021", True),
        ("Sommer 2019", "Sommer 2019", True),
        ("2019 Frühling", "2019 Frühling", True),
    ],
)
def test_foreign_date_words_next_to_a_year_are_reported(text, phrase, english_only):
    got = finding(text)
    assert got.kind == "weak" and got.phrase == phrase and not got.resolution.understood
    assert ("English" in got.resolution.reason) is english_only


def test_foreign_word_alone_or_without_year_is_not_a_date():
    for text in ("Weihnachten", "verano en la playa", "été"):
        assert finding(text) is None


@pytest.mark.parametrize(
    "text",
    ["Easter eggs 2024", "Halloween costumes 2022", "summer vacation 2019", "Christmas tree 2022"],
)
def test_ordinary_date_word_apart_from_a_bare_year_is_unclear(text):
    got = finding(text)
    year = text.split()[-1]
    assert got.kind == "weak" and got.phrase == year and not got.resolution.understood
    assert "apart" in got.resolution.reason
    assert f'Did you mean "{text.split()[0]} {year}"' in got.question


def test_unsupported_holiday_far_from_a_year_is_reported_with_its_reason():
    got = finding("Eid with Priya 2023")
    assert got.kind == "weak" and got.phrase == "2023"
    assert "lunar" in got.resolution.reason and "Eid" in got.question
    assert finding("Diwali 2022 in Mumbai").kind == "unsupported"


def test_bare_year_next_to_its_season_or_holiday_word_is_fine():
    assert finding("Easter 2024 eggs").resolution.understood
    assert finding("summer 2019 vacation").resolution.range.start == date(2019, 6, 1)
    assert finding("street food in Bangkok 2019").resolution.range.start == date(2019, 1, 1)


@pytest.mark.parametrize(
    "text",
    [
        "lunch at a Chinese restaurant",
        "at the lunar eclipse",
        "Chinese food",
        "photo of the lunar eclipse",
        "Chinese New Year dinner",
        "Diwali lights",
        "Eid mubarak balloons",
    ],
)
def test_lunar_words_used_normally_are_not_dates(text):
    assert finding(text) is None


def test_lunar_names_need_the_full_name_and_an_anchor():
    for text, phrase in (
        ("dinner at Chinese New Year 2024", "Chinese New Year 2024"),
        ("last Lunar New Year", "Lunar New Year"),
        ("on Rosh Hashanah", "Rosh Hashanah"),
        ("at Yom Kippur", "Yom Kippur"),
        ("Diwali 2022", "Diwali 2022"),
    ):
        got = finding(text)
        assert got.kind == "unsupported" and got.phrase.endswith(phrase.split()[-1])
        assert not text.lower().startswith(got.phrase.lower() + " a")
    assert not any(s.phrase.lower().startswith(("a ", "an ", "the ")) for s in
                   find_date_phrases("at a Chinese New Year 2024 party", TODAY))  # fmt: skip
    assert "lunar" in not_understood("Chinese New Year")
    assert not_understood("Chinese")
    assert not looks_like_date("Chinese") and not looks_like_date("lunar")
    assert looks_like_date("Hanukkah") and looks_like_date("Yom Kippur")


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("last year's trip to Rome", "last year's", ("2025-01-01", "2025-12-31")),
        ("last year’s trip", "last year’s", ("2025-01-01", "2025-12-31")),
        ("yesterday's lunch", "yesterday's", ("2026-11-06", "2026-11-06")),
        ("this week's photos", "this week's", ("2026-11-02", "2026-11-08")),
        ("New Year's Eve 2024 party", "New Year's Eve 2024", ("2024-12-31", "2025-01-01")),
        ("Mother's Day 2024", "Mother's Day 2024", ("2024-05-12", "2024-05-12")),
        ("St. Patrick's Day 2022", "St. Patrick's Day 2022", ("2022-03-17", "2022-03-17")),
    ],
)
def test_possessives(text, phrase, expected):
    got = finding(text)
    assert got.phrase == phrase and got.resolution.understood
    assert (
        got.resolution.range.start.isoformat(),
        got.resolution.range.end.isoformat(),
    ) == expected
    assert span(phrase) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Windows 2000 screenshot",
        "jersey number 2012",
        "route 2050",
        "1955 Chevy on route 2050",
        "photo 1969",
        "room 2019",
        "version 2020",
        "model 2018 sedan",
        "#2012",
        "no. 2015",
        "5 2019",
        "3 4 2019",
        "iPhone 2019 box photos",
    ],
)
def test_bare_years_that_are_numbers_are_not_dates(text):
    assert find_date_phrases(text, TODAY) == [], text


@pytest.mark.parametrize(
    ("text", "year"),
    [
        ("2019", 2019),
        ("in 2019", 2019),
        ("Bangkok 2019", 2019),
        ("2019 hiking", 2019),
        ("trip, 2019, was fun", 2019),
        ("2019 in Berlin", 2019),
        ("Hawaii 2018", 2018),
        ("Rio de Janeiro carnival 2020", 2020),
        ("the 2019 trip", 2019),
    ],
)
def test_bare_years_that_are_dates(text, year):
    spans = find_date_phrases(text, TODAY)
    assert spans and spans[0].resolution.range.start.year == year


def test_year_in_the_middle_between_ordinary_words_is_not_a_date():
    assert find_date_phrases("Bangkok 2019 street food", TODAY) == []
    assert find_date_phrases("Windows 2000 screenshot", TODAY) == []


@pytest.mark.parametrize(
    "phrase", ["yesterday 2019", "3 days ago 2019", "now 2020", "next month 2019"]
)
def test_relative_phrases_do_not_accept_a_stray_year(phrase):
    not_understood(phrase)


@pytest.mark.parametrize(
    "phrase", ["03.04.2019", "3 4 2019", "1 2 2020", "04.03.2020", "march 2024-04", "may 2024 5"]
)
def test_ambiguous_or_garbled_numerics_are_not_guessed(phrase):
    not_understood(phrase)


def test_ambiguous_numerics_even_if_dateparser_would_guess(monkeypatch):
    called = []
    _stub_dateparser(
        monkeypatch, lambda text: called.append(text) or [(text, datetime(2019, 3, 4))]
    )
    for phrase in ("3 4 2019", "1 2 2020", "march 2024-04", "may 2024 5", "1 2 3 2020"):
        not_understood(phrase)
    assert called == []  # rejected before dateparser is even asked


def test_bare_number_pairs_are_not_guessed(monkeypatch):
    _stub_dateparser(monkeypatch, lambda text: [(text, datetime(2019, 4, 1))])
    not_understood("4 2019")


def test_unambiguous_dotted_dates_work():
    assert span("15.04.2019") == ("2019-04-15", "2019-04-15")
    assert span("2019.04.15") == ("2019-04-15", "2019-04-15")
    assert span("last 1 year") == ("2025-11-08", "2026-11-07")
    assert span("last 1 month") == ("2026-10-08", "2026-11-07")


def test_operator_words_resolve_as_not_understood():
    for phrase in ("since 2020", "until June", "except 2019", "not yesterday", "2019 onwards"):
        assert "changes what the date means" in not_understood(phrase) or not_understood(phrase)
    assert "single date" in not_understood("2019 or 2020")
    assert "single date" in not_understood("June 2020 and December 2021")


def test_analyse_dates_plain_cases():
    assert analyse_dates("", TODAY) is None
    assert analyse_dates("red car", TODAY) is None
    got = analyse_dates("sunset in Goa, December 2022", TODAY)
    assert got.phrase == "December 2022" and got.resolution.understood and got.question == ""
    south = analyse_dates("winter 2020", TODAY, southern_hemisphere=True)
    assert south.resolution.range.start == date(2020, 6, 1)


def test_analyse_dates_keeps_gaps_with_other_words_apart():
    got = analyse_dates("June 2020 was great, December 2021 too", TODAY)
    assert got.kind == "conflict" and got.phrase == "June 2020"


def test_analyse_dates_far_weak_word_is_ignored():
    text = "Christmas tree " + "x " * 100 + "in 2019"
    assert analyse_dates(text, TODAY).resolution.understood


# ----------------------------------------------------------------- impossible dates, bare words


@pytest.mark.parametrize(
    ("text", "phrase", "reason"),
    [
        ("31 February 2020", "31 February 2020", "February 2020 has no day 31"),
        ("photos 30 Feb 2020", "30 Feb 2020", "February 2020 has no day 30"),
        ("31 April 2021", "31 April 2021", "April 2021 has no day 31"),
        ("February 31 2020", "February 31 2020", "February 2020 has no day 31"),
        ("Feb 31, 2020 party", "Feb 31, 2020", "February 2020 has no day 31"),
        ("29 February 2021", "29 February 2021", "February 2021 has no day 29"),
        ("Feb 30", "Feb 30", "February has no day 30"),
        ("trip on 31 Feb", "31 Feb", "February has no day 31"),
        ("31 June", "31 June", "June has no day 31"),
        ("31/04/2021", "31/04/2021", "April 2021 has no day 31"),
        ("2020-02-30", "2020-02-30", "February 2020 has no day 30"),
        ("2021-04-31", "2021-04-31", "April 2021 has no day 31"),
    ],
)
def test_impossible_dates_are_never_widened_to_the_month_or_year(text, phrase, reason):
    got = finding(text)
    assert got is not None and got.phrase == phrase and got.kind == "numeric"
    assert not got.resolution.understood and got.resolution.range is None
    assert got.resolution.reason == reason + "." and reason in got.question
    assert not_understood(phrase) == reason + "."


def test_possible_dates_and_non_dates_around_impossible_ones():
    assert span("29 February 2020") == ("2020-02-29", "2020-02-29")
    assert span("31 March 2020") == ("2020-03-31", "2020-03-31")
    assert finding("Feb 29").resolution.understood
    # numbers that are too big to be a day are just numbers
    assert finding("may 50 people") is None
    assert not_understood("March 32 2020")
    assert not_understood("0 March 2020")


def test_bare_date_word():
    for word in ("autumn", "fall", "summer", "Christmas", "Halloween", "Easter", "May", "march",
                 "Monday", "weekend", "June to August"):  # fmt: skip
        assert bare_date_word(word, TODAY), word
    for phrase in (
        "in autumn", "during the summer", "at Christmas", "of May", "last autumn", "this winter",
        "autumn 2023", "Christmas 2022", "2019", "yesterday", "3 weeks ago", "March 2024",
        "red car", "", "on Monday",
    ):  # fmt: skip
        assert not bare_date_word(phrase, TODAY), phrase
    assert not bare_date_word(None, TODAY)  # type: ignore[arg-type]
    assert bare_date_word("summer", TODAY, southern_hemisphere=True)


def test_preposition_before():
    text = "leaves in autumn, during the summer, at Christmas, autumn leaves, of May"
    for word in ("autumn", "summer", "Christmas", "May"):
        assert preposition_before(text, text.index(word)), word
    assert not preposition_before(text, text.rindex("autumn"))
    assert not preposition_before("autumn leaves", 0)
    assert not preposition_before("in. autumn", 4)  # a full stop separates them
    assert not preposition_before("a autumn", 2)  # "a" alone is not a preposition
    assert preposition_before("in a autumn", 5)


# ----------------------------------------------------------------- operators, every entry point

BEFORE_OPERATORS = [
    "since", "before", "after", "until", "till", "by", "pre", "post", "prior", "early", "late",
    "mid", "middle", "end", "beginning", "start", "not", "except", "excluding", "without",
]  # fmt: skip
AFTER_OPERATORS = ["onwards", "onward", "later", "earlier", "afterwards", "forward"]


@pytest.mark.parametrize("word", BEFORE_OPERATORS)
@pytest.mark.parametrize("date_text", ["2019", "June 2020", "summer 2019", "Christmas 2022"])
def test_leading_operator_is_refused_by_both_entry_points(word, date_text):
    phrase = f"{word} {date_text}"
    direct = resolve_date_phrase(phrase, TODAY)
    assert not direct.understood and direct.range is None, phrase
    scanned = analyse_dates(f"photos {phrase} hiking", TODAY)
    assert scanned is not None and scanned.kind == "operator", phrase
    assert not scanned.resolution.understood and word in scanned.phrase.lower()
    assert scanned.question
    # the same specific reason from both
    assert word in direct.reason and word in scanned.resolution.reason


@pytest.mark.parametrize("word", AFTER_OPERATORS)
@pytest.mark.parametrize("date_text", ["2019", "June 2020"])
def test_trailing_operator_is_refused_by_both_entry_points(word, date_text):
    phrase = f"{date_text} {word}"
    direct = resolve_date_phrase(phrase, TODAY)
    assert not direct.understood and direct.range is None, phrase
    scanned = analyse_dates(f"photos {phrase} hiking", TODAY)
    assert scanned is not None and scanned.kind == "operator", phrase
    assert not scanned.resolution.understood and scanned.question


@pytest.mark.parametrize(
    "phrase",
    [
        "after 2019", "by 2019", "2019 after", "2019 by", "the after of 2019", "the by of 2019",
        "after june 2020", "by june 2020",
    ],
)  # fmt: skip
def test_dateparser_fallback_cannot_drop_an_operator(phrase):
    # dateparser reads "after 2019" as "2019"; the fallback must never be asked
    assert not resolve_date_phrase(phrase, TODAY).understood


@pytest.mark.parametrize(
    "phrase",
    [
        "2019 or 2020",
        "June 2020 and December 2021",
        "June and August",
        "2019 and later",
        "2019 or earlier",
        "last year or this year",
    ],
)
def test_lists_and_choices_are_refused_by_both_entry_points(phrase):
    direct = resolve_date_phrase(phrase, TODAY)
    assert not direct.understood and "single date" in direct.reason
    scanned = analyse_dates(f"photos {phrase}", TODAY)
    # two bare months ("June and August") are not dates in a sentence, so nothing is applied
    assert scanned is None or not scanned.resolution.understood


def test_operator_words_do_not_break_valid_phrases():
    for phrase in ("between June and August 2021", "June until August 2021", "from June to August"):
        assert resolve_date_phrase(phrase, TODAY).understood, phrase
    assert resolve_date_phrase("the weekend", TODAY).understood
