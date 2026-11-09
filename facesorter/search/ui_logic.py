"""
facesorter/search/ui_logic.py
Pure (gradio-free) logic behind the local search UI.

``ui.py`` wires widgets to the functions here and does nothing else, so everything that
can go wrong is testable without a browser (and on Python 3.9, where gradio is absent).

* ``run_search`` turns the form fields into a :class:`SearchRequest` (explicit fields
  override what the parser infers, per field), calls the engine and returns a
  :class:`UIResult`: hit cards, the "How I read your query" text, pending clarification
  questions, dropped-filter notices, counts and timings. Bad input (an impossible date, an
  engine failure) comes back as ``UIResult.error``, never as an exception.
* The clarification protocol is "ask once, then tell and search without": a first search may
  return questions; ``answer_clarifications`` re-runs with the user's replies (an empty reply
  skips a phrase) and always with ``allow_clarification=False``, so the second run can never
  ask again. Phrases that are still not understood are reported as dropped.
* ``ThumbnailCache`` renders ~512 px JPEG thumbnails of the originals (EXIF-correct, HEIC
  included) into a private directory. The UI serves only that directory, so original photos
  are never exposed to the browser.
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .encoders.decode import load_for_model
from .types import (
    DateRange,
    Filters,
    IndexedPhoto,
    PlaceFilter,
    SearchRequest,
    SearchResponse,
    Unresolved,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_THUMB_CAP_BYTES",
    "MAX_K",
    "HitCard",
    "Session",
    "SearchParams",
    "HostGuardMiddleware",
    "HostPolicy",
    "ThumbnailCache",
    "UIInputError",
    "UIResult",
    "answer_clarifications",
    "clamp_k",
    "date_filter",
    "default_thumbs_dir",
    "ensure_outside_library",
    "index_summary",
    "is_loopback_host",
    "md_escape",
    "parse_date_bound",
    "record_feedback",
    "run_search",
    "skip_clarifications",
    "start_search",
]

DEFAULT_K = 24
MAX_K = 200
MAX_QUERY_CHARS = 1000
THUMB_SIZE = 512
DEFAULT_THUMB_CAP_BYTES = 2 * 1024**3
_THUMB_WORKERS = 4
_STRIPES = 16
_PLACEHOLDER_NAME = "_missing.jpg"
_THUMB_NAME = re.compile(r"^[0-9a-f]{40}\.jpg$")
_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")
_DATE_HELP = "Use YYYY, YYYY-MM or YYYY-MM-DD, for example 2023, 2023-07 or 2023-07-14."


class UIInputError(ValueError):
    """The user typed something the form cannot use; the message is shown as is."""


# --------------------------------------------------------------------------- dates


def parse_date_bound(text: str | None, *, end: bool) -> date | None:
    """Expand ``'YYYY'``, ``'YYYY-MM'`` or ``'YYYY-MM-DD'`` to the first (or last) day it covers.

    ``end=False`` gives the start of the period (``2024`` -> 2024-01-01), ``end=True`` its
    last day (``2024-02`` -> 2024-02-29, leap years respected). Blank text means "unset" and
    returns ``None``. Anything else raises :class:`UIInputError` with a friendly message.
    """
    raw = (text or "").strip().replace("/", "-").replace(".", "-")
    if not raw:
        return None
    m = _DATE_RE.match(raw)
    if m is None:
        raise UIInputError(f'"{text.strip() if text else raw}" is not a date. {_DATE_HELP}')
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else None
    day = int(m.group(3)) if m.group(3) else None
    try:
        if month is None:
            return date(year, 12, 31) if end else date(year, 1, 1)
        if day is None:
            last = calendar.monthrange(year, month)[1]  # raises for month 0 or 13
            return date(year, month, last if end else 1)
        return date(year, month, day)
    except ValueError as exc:
        raise UIInputError(f'"{raw}" is not a real calendar date ({exc}). {_DATE_HELP}') from exc


def date_filter(date_from: str | None, date_to: str | None) -> DateRange | None:
    """The date range for the two form fields, or ``None`` when both are blank.

    The "from" field expands to the start of its period and "to" to the end of its period,
    so ``2023`` to ``2023`` is the whole year. A missing side is open (``date.min`` /
    ``date.max``). A "from" after the "to" raises :class:`UIInputError`.
    """
    start = parse_date_bound(date_from, end=False)
    stop = parse_date_bound(date_to, end=True)
    if start is None and stop is None:
        return None
    start = start or date.min
    stop = stop or date.max
    if stop < start:
        raise UIInputError(f'"From" ({start.isoformat()}) is after "to" ({stop.isoformat()}).')
    return DateRange(start, stop)


def clamp_k(k: object, default: int = DEFAULT_K) -> int:
    """``k`` as an int in ``1..MAX_K``; anything that is not a finite number gives ``default``."""
    try:
        value = int(float(k))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(MAX_K, value))


# --------------------------------------------------------------------------- view models


@dataclass(frozen=True)
class HitCard:
    """One search hit prepared for display. Paths are display text, never served."""

    photo_id: int
    path: str
    filename: str
    folder: str
    thumb_path: str | None
    caption: str
    score: float
    date_text: str = ""
    place_text: str = ""
    camera: str = ""
    people: tuple[str, ...] = ()
    thumb_ok: bool = True

    def details(self) -> list[tuple[str, str]]:
        """``(label, value)`` rows for the detail panel; empty values are skipped."""
        rows = [
            ("File", self.filename),
            ("Date", self.date_text),
            ("Place", self.place_text),
            ("Camera", self.camera),
            ("People", ", ".join(self.people)),
            ("Score", f"{self.score:.3f}"),
            ("Folder", self.folder),
        ]
        return [(label, value) for label, value in rows if value]


@dataclass(frozen=True)
class SearchParams:
    """The form values of one search (kept so a clarification can re-run it)."""

    query: str = ""
    date_from: str = ""
    date_to: str = ""
    place: str = ""
    people: tuple[str, ...] = ()
    k: int = DEFAULT_K
    use_llm: bool = True


@dataclass
class UIResult:
    """Everything the page needs to show after one search."""

    query: str = ""
    cards: list[HitCard] = field(default_factory=list)
    reading: str = ""
    notices: list[str] = field(default_factory=list)
    questions: list[tuple[str, str]] = field(default_factory=list)  # (phrase, question)
    dropped: list[str] = field(default_factory=list)
    candidates: int = 0
    elapsed_ms: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def needs_clarification(self) -> bool:
        return bool(self.questions)

    @property
    def status(self) -> str:
        """One line under the search box."""
        if self.error:
            return self.error
        if self.questions:
            return "I need a little help understanding your query."
        pool = f"{self.candidates:,}"
        if not self.cards:
            return f"No photos found (searched {pool} candidate photos)."
        n = len(self.cards)
        noun = "photo" if n == 1 else "photos"
        return f"Showing {n} {noun} of {pool} candidates in {self.elapsed_ms / 1000:.2f} s."

    def gallery_cards(self) -> list[HitCard]:
        """The cards that have an image to show, in gallery order (index = gallery index)."""
        return [c for c in self.cards if c.thumb_path]

    def gallery_items(self) -> list[tuple[str, str]]:
        """``(thumbnail path, caption)`` pairs for the gallery."""
        return [(c.thumb_path or "", c.caption) for c in self.gallery_cards()]


@dataclass
class Session:
    """Per-browser state: the last search and what the user may do next."""

    params: SearchParams = field(default_factory=SearchParams)
    result: UIResult | None = None
    pending: list[tuple[str, str]] = field(default_factory=list)
    answered: bool = False
    selected: HitCard | None = None

    @property
    def query(self) -> str:
        return self.params.query


# --------------------------------------------------------------------------- helpers

# ``:`` ``/`` and ``@`` are escaped too so bare URLs and e-mail addresses are not auto-linked.
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>&$:/@])")
_LINE_BREAKS = re.compile(r"[\r\n\t\x0b\x0c\x85\u2028\u2029]+")


def md_escape(text: str) -> str:
    """Escape ``text`` so Markdown shows it literally (file names and queries are untrusted).

    Line breaks and tabs become a space, so a hostile file name cannot start a new Markdown
    block (a heading, list, code fence or table row).
    """
    return _MD_SPECIAL.sub(r"\\\1", _LINE_BREAKS.sub(" ", str(text)))


def _engine_helper(engine: Any, name: str) -> Callable[..., Any] | None:
    """``describe_filters`` / ``explain`` from the engine object or its module, if present."""
    fn = getattr(engine, name, None)
    if callable(fn):
        return fn  # type: ignore[no-any-return]
    try:
        from . import engine as engine_module
    except ImportError:  # the engine module is optional for the UI's own tests
        return None
    fn = getattr(engine_module, name, None)
    return fn if callable(fn) else None


def _describe_range(r: DateRange) -> str:
    if r.start == date.min:
        return f"on or before {r.end.isoformat()}"
    if r.end == date.max:
        return f"on or after {r.start.isoformat()}"
    return f"{r.start.isoformat()} to {r.end.isoformat()}"


def _fallback_describe(filters: Filters) -> str:
    parts = []
    if filters.date_range is not None:
        parts.append("dates " + _describe_range(filters.date_range))
    if filters.place is not None:
        p = filters.place
        named = ", ".join(x for x in (p.city, p.region, p.country) if x)
        parts.append("place " + (named or p.text or "(by distance)"))
    if filters.people:
        parts.append("with " + ", ".join(filters.people))
    return " · ".join(parts)


def _fallback_explain(response: SearchResponse) -> list[str]:
    notes = []
    if response.candidates == 0:
        notes.append("No photos match the filters.")
    if not response.parse.text.strip():
        notes.append("No visual description, so photos are listed newest first.")
    return notes


def _describe(engine: Any, filters: Filters) -> str:
    """Engine wording for ``filters``; an open-ended date side is worded here instead."""
    open_ended = filters.date_range is not None and (
        filters.date_range.start == date.min or filters.date_range.end == date.max
    )
    fn = _engine_helper(engine, "describe_filters")
    try:
        if open_ended and filters.date_range is not None:
            rest = replace(filters, date_range=None)
            tail = fn(rest) if fn else _fallback_describe(rest)
            head = "dates " + _describe_range(filters.date_range)
            return head + (f" · {tail}" if tail else "")
        text = fn(filters) if fn else _fallback_describe(filters)
    except Exception:  # a helper bug must not blank the page
        logger.exception("describe_filters failed")
        text = _fallback_describe(filters)
    return str(text)


def _notices(engine: Any, response: SearchResponse) -> list[str]:
    fn = _engine_helper(engine, "explain")
    try:
        lines = fn(response) if fn else _fallback_explain(response)
    except Exception:
        logger.exception("explain failed")
        lines = _fallback_explain(response)
    return [str(x) for x in lines if str(x).strip()]


def _clean_people(people: Iterable[str] | str | None) -> tuple[str, ...]:
    if not people:
        return ()
    items = [people] if isinstance(people, str) else list(people)
    seen: dict[str, str] = {}
    for name in items:
        clean = " ".join(str(name).split())
        if clean:
            seen.setdefault(clean.casefold(), clean)
    return tuple(seen.values())


def _date_text(photo: IndexedPhoto) -> str:
    meta = photo.meta
    if meta.taken is None or meta.date_source == "none":
        return ""
    if meta.date_precision == "year":
        text = f"{meta.taken:%Y}"
    elif meta.date_precision == "month":
        text = f"{meta.taken:%Y-%m}"
    else:
        text = f"{meta.taken:%Y-%m-%d}"
    return f"{text} (file date)" if meta.date_source == "mtime" else text


def _make_card(
    hit_photo: IndexedPhoto,
    score: float,
    thumb: Path | None,
    placeholder: Path | None,
    people: Sequence[str],
) -> HitCard:
    meta = hit_photo.meta
    date_text = _date_text(hit_photo)
    place_text = hit_photo.place.display if hit_photo.place else ""
    caption = " · ".join(x for x in (date_text, place_text, f"{score:.2f}") if x)
    shown = thumb or placeholder
    return HitCard(
        photo_id=hit_photo.id,
        path=meta.path,
        filename=os.path.basename(meta.path),
        folder=os.path.basename(os.path.dirname(meta.path)),  # the name only, never the full path
        thumb_path=str(shown) if shown else None,
        caption=caption,
        score=float(score),
        date_text=date_text,
        place_text=place_text,
        camera=meta.camera or "",
        people=tuple(people),
        thumb_ok=thumb is not None,
    )


def _people_for(engine: Any, ids: list[int]) -> dict[int, list[str]]:
    fn = getattr(getattr(engine, "store", None), "names_for_photos", None)
    if not callable(fn) or not ids:
        return {}
    try:
        return dict(fn(ids))
    except Exception:
        logger.debug("names_for_photos failed", exc_info=True)
        return {}


def _resolve_place(engine: Any, place: str) -> PlaceFilter | None:
    gazetteer = getattr(engine, "gazetteer", None)
    resolve = getattr(gazetteer, "resolve", None)
    if not callable(resolve):
        return None
    try:
        found = resolve(place)
    except Exception:
        logger.debug("gazetteer failed on %r", place, exc_info=True)
        return None
    if isinstance(found, PlaceFilter):
        return found if found.text else replace(found, text=place)
    return None


def build_request(
    engine: Any,
    params: SearchParams,
    clarifications: Mapping[str, str] | None,
    allow_clarification: bool,
) -> tuple[SearchRequest, list[str]]:
    """The :class:`SearchRequest` for ``params`` plus any notes about how fields were used.

    Raises :class:`UIInputError` for an unusable date. A place that the gazetteer cannot
    resolve (or no gazetteer at all) is appended to the query as ``in <place>`` so the parser
    handles it and the clarification protocol applies.
    """
    notes: list[str] = []
    query = " ".join((params.query or "").split())[:MAX_QUERY_CHARS]
    place_text = " ".join((params.place or "").split())
    place_filter = _resolve_place(engine, place_text) if place_text else None
    if place_text and place_filter is None:
        query = f"{query} in {place_text}".strip()
    date_range = date_filter(params.date_from, params.date_to)
    people = _clean_people(params.people)
    filters = None
    if date_range is not None or place_filter is not None or people:
        filters = Filters(date_range=date_range, place=place_filter, people=people)
    k = clamp_k(params.k)
    if params.k is not None and k != params.k:
        notes.append(f"Showing at most {k} results.")
    request = SearchRequest(
        query=query,
        filters=filters,
        k=k,
        use_llm=bool(params.use_llm),
        allow_clarification=allow_clarification,
        clarifications=dict(clarifications or {}),
    )
    return request, notes


def _dropped_text(item: Unresolved) -> str:
    return f'I did not understand "{item.phrase}", so I searched without it.'


def _build_result(
    engine: Any,
    query: str,
    response: SearchResponse,
    thumbs: ThumbnailCache | None,
    notes: list[str],
    allow_clarification: bool,
) -> UIResult:
    result = UIResult(query=query, notices=list(notes))
    dropped = list(response.dropped)
    questions = list(response.needs_clarification)
    if questions and not allow_clarification:
        # A misbehaving engine must not make the UI ask twice: report instead.
        dropped.extend(q for q in questions if q.phrase not in {d.phrase for d in dropped})
        questions = []
    result.questions = [(q.phrase, q.question) for q in questions]
    result.dropped = [_dropped_text(d) for d in dropped]
    result.candidates = int(response.candidates)
    result.timings_ms = {str(k): float(v) for k, v in dict(response.timings_ms).items()}

    shown_phrases = [d.phrase.casefold() for d in dropped]
    asked = {q.question for q in questions}
    for line in _notices(engine, response):
        # The dropped banner and the question boxes already say this; do not repeat it.
        if line in asked or any(p and p in line.casefold() for p in shown_phrases):
            continue
        result.notices.append(line)

    visual = response.parse.text.strip()
    described = _describe(engine, response.applied)
    lines = [f"Filters: {described}." if described else "Filters: none."]
    lines.append(f'Looking for: "{visual}".' if visual else "No visual description in the query.")
    result.reading = "\n".join(lines)

    hits = list(response.hits)
    metas = [h.photo.meta for h in hits]
    thumb_paths: list[Path | None] = [None] * len(hits)
    if thumbs is not None:
        try:
            thumb_paths = thumbs.get_many(metas)
        except Exception:  # never let a cache problem break the search
            logger.exception("thumbnail cache failed")
    placeholder: Path | None = None
    placeholder_tried = False
    names = _people_for(engine, [h.photo.id for h in hits])
    missing = 0
    for hit, thumb in zip(hits, thumb_paths):
        if thumbs is not None and thumb is None:
            missing += 1
            if not placeholder_tried:
                placeholder_tried = True
                placeholder = thumbs.placeholder()
        result.cards.append(
            _make_card(hit.photo, hit.score, thumb, placeholder, names.get(hit.photo.id, ()))
        )
    if missing:
        noun = "photo" if missing == 1 else "photos"
        outcome = "a placeholder is shown" if placeholder else "it is left out of the gallery"
        result.notices.append(
            f"{missing} {noun} could not be read from disk (moved, deleted or damaged), "
            f"so {outcome}."
        )
    return result


# --------------------------------------------------------------------------- search


def run_search(
    engine: Any,
    query: str,
    date_from: str = "",
    date_to: str = "",
    place: str = "",
    people: Iterable[str] | None = (),
    k: int = DEFAULT_K,
    clarifications: Mapping[str, str] | None = None,
    allow_clarification: bool = True,
    use_llm: bool = True,
    *,
    thumbs: ThumbnailCache | None = None,
) -> UIResult:
    """Run one search from form values and return what the page should show.

    Explicit fields (dates, place, people) override the parser's guess per field. Never
    raises for bad input or an engine failure: ``UIResult.error`` carries a friendly message.
    """
    params = SearchParams(
        query=query or "",
        date_from=date_from or "",
        date_to=date_to or "",
        place=place or "",
        people=_clean_people(people),
        k=k,
        use_llm=use_llm,
    )
    return _run(engine, params, clarifications, allow_clarification, thumbs)


def _run(
    engine: Any,
    params: SearchParams,
    clarifications: Mapping[str, str] | None,
    allow_clarification: bool,
    thumbs: ThumbnailCache | None,
) -> UIResult:
    started = time.perf_counter()
    try:
        request, notes = build_request(engine, params, clarifications, allow_clarification)
    except UIInputError as exc:
        return UIResult(query=params.query, error=str(exc))
    try:
        response = engine.search(request)
    except Exception as exc:
        logger.exception("search failed")
        detail = str(exc) or type(exc).__name__
        return UIResult(query=params.query, error=f"Search failed: {detail}", notices=list(notes))
    result = _build_result(engine, params.query, response, thumbs, notes, allow_clarification)
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


def start_search(
    engine: Any,
    session: Session,
    params: SearchParams,
    *,
    thumbs: ThumbnailCache | None = None,
) -> UIResult:
    """Search for ``params`` as a fresh question: resets the clarification state of ``session``."""
    result = _run(engine, params, None, True, thumbs)
    session.params = params
    session.result = result
    session.pending = list(result.questions)
    session.answered = False
    return result


def answer_clarifications(
    engine: Any,
    session: Session,
    replies: Sequence[str] | Mapping[str, str] | None,
    *,
    thumbs: ThumbnailCache | None = None,
) -> UIResult:
    """Re-run the last search with the user's replies; this run never asks again.

    ``replies`` is either a sequence aligned with ``session.pending`` or a mapping from the
    original phrase to the reply. A missing or blank reply skips that phrase. Every pending
    phrase is sent (so the engine counts it as asked) and ``allow_clarification`` is
    ``False``, so anything still not understood comes back as dropped.
    """
    if not session.pending:
        return UIResult(
            query=session.query, error="There is nothing to answer; run a search first."
        )
    if isinstance(replies, Mapping):
        by_key = {" ".join(str(k).split()).casefold(): v for k, v in replies.items()}
        ordered = [by_key.get(" ".join(p.split()).casefold(), "") for p, _ in session.pending]
    else:
        ordered = list(replies or [])
    answers = {
        phrase: " ".join(str(ordered[i] or "").split()) if i < len(ordered) else ""
        for i, (phrase, _) in enumerate(session.pending)
    }
    result = _run(engine, session.params, answers, False, thumbs)
    session.result = result
    session.pending = []
    session.answered = True
    return result


def skip_clarifications(
    engine: Any, session: Session, *, thumbs: ThumbnailCache | None = None
) -> UIResult:
    """Search without the phrases the engine asked about (every reply blank)."""
    return answer_clarifications(engine, session, [], thumbs=thumbs)


def record_feedback(engine: Any, query: str, photo_id: int, relevant: bool) -> str:
    """Store a thumbs-up / thumbs-down for ``photo_id`` and return a message for the user."""
    try:
        engine.store.record_feedback(query, int(photo_id), bool(relevant))
    except ValueError as exc:  # unknown photo id
        return f"Could not save your rating: {exc}"
    except Exception as exc:
        logger.exception("record_feedback failed")
        return f"Could not save your rating ({type(exc).__name__})."
    return "Thanks, marked as relevant." if relevant else "Thanks, marked as not relevant."


# --------------------------------------------------------------------------- index footer


def index_summary(engine: Any) -> str:
    """The page footer: how many photos are indexed and the span of their dates.

    Deliberately says nothing about where the library is.
    """
    try:
        stats = engine.store.stats()
        text = f"{stats.photos:,} photos indexed"
        if stats.date_min and stats.date_max:
            text += f", dated {stats.date_min.isoformat()} to {stats.date_max.isoformat()}"
        return text + "."
    except Exception:
        logger.debug("index_summary failed", exc_info=True)
        return "Index statistics are not available."


# --------------------------------------------------------------------------- safety checks


def is_loopback_host(host: str) -> bool:
    """Whether binding to ``host`` keeps the server reachable from this machine only."""
    import ipaddress

    name = (host or "").strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})
_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
_BAD_AUTHORITY = re.compile(r"[\s/\\?#@]")


def _split_authority(value: str) -> tuple[str, int | None] | None:
    """``"Host:80"`` -> ``("host", 80)``; ``None`` if ``value`` is not a clean authority."""
    from urllib.parse import urlsplit

    if not value or _BAD_AUTHORITY.search(value):
        return None
    try:
        parts = urlsplit("//" + value)
        hostname, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not hostname:
        return None
    return hostname.rstrip(".").lower(), port


@dataclass
class HostPolicy:
    """Which ``Host`` / ``Origin`` values the server answers (the DNS-rebinding defence).

    A web page can make its own domain resolve to 127.0.0.1 and then talk to a local server
    from the user's browser; such requests still carry the attacker's name in ``Host`` and
    ``Origin``. Refusing every name that is not ours stops that. ``port`` is the port the
    server is bound to (``None`` until known: any port is accepted).
    """

    hostnames: frozenset[str] = frozenset(_LOOPBACK_NAMES)
    any_ip: bool = False  # a wildcard bind: any IP literal is a legitimate way to reach us
    port: int | None = None

    @classmethod
    def for_bind(cls, host: str, port: int | None = None) -> HostPolicy:
        """The policy for a server bound to ``host``; non-loopback hosts are added."""
        import socket

        name = (host or "").strip().strip("[]").lower()
        names = set(_LOOPBACK_NAMES)
        any_ip = False
        if name in _WILDCARD_HOSTS:
            any_ip = True
            machine = socket.gethostname().lower()
            names.update({machine, machine.split(".")[0] + ".local"})
        elif name:
            names.add(name)
        return cls(frozenset(names), any_ip, port or None)

    def host_allowed(self, value: str | None) -> bool:
        """Whether a ``Host`` header (or an Origin's authority) names this server."""
        split = _split_authority(value or "")
        if split is None:
            return False
        hostname, port = split
        if self.port is not None and port is not None and port != self.port:
            return False
        if hostname in self.hostnames:
            return True
        if self.any_ip:
            import ipaddress

            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                return False
            return True
        return False

    def origin_allowed(self, origin: str | None) -> bool:
        """Whether an ``Origin`` header is this server's own origin (``None``: no header)."""
        if origin is None:
            return True
        from urllib.parse import urlsplit

        try:
            parts = urlsplit(origin)
        except ValueError:
            return False
        if parts.scheme not in ("http", "https") or parts.path not in ("", "/"):
            return False
        if parts.query or parts.fragment:
            return False
        return self.host_allowed(parts.netloc)


class HostGuardMiddleware:
    """Pure ASGI middleware: 403 for every request whose Host or Origin is not ours.

    Applies to every route (pages, API, files, event streams, websockets) because it sits in
    front of the whole application. Lifespan events pass through.
    """

    def __init__(self, app: Any, policy: HostPolicy) -> None:
        self.app = app
        self.policy = policy

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        hosts, origins = [], []
        for key, value in scope.get("headers") or ():
            if key == b"host":
                hosts.append(value.decode("latin-1"))
            elif key == b"origin":
                origins.append(value.decode("latin-1"))
        ok = len(hosts) == 1 and self.policy.host_allowed(hosts[0])
        ok = (
            ok and len(origins) <= 1 and self.policy.origin_allowed(origins[0] if origins else None)
        )
        if ok:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = b"Forbidden: this server only answers requests addressed to it directly.\n"
        headers = [(b"content-type", b"text/plain; charset=utf-8")]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({"type": "http.response.start", "status": 403, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def ensure_outside_library(thumbs_dir: str | os.PathLike[str], roots: Iterable[str]) -> Path:
    """Resolve ``thumbs_dir`` and refuse it if it overlaps a library root (either way round).

    The thumbnail directory is the only directory the UI serves, so it must never be, contain
    or sit inside the photo library. Returns the resolved directory.
    """
    thumbs = Path(thumbs_dir).expanduser().resolve()
    for root in roots:
        resolved = Path(root).expanduser().resolve()
        if _inside(thumbs, resolved) or _inside(resolved, thumbs):
            raise ValueError(
                f"The thumbnail directory {thumbs} overlaps the photo library {resolved}. "
                "Choose a thumbnail directory outside the library."
            )
    return thumbs


def default_thumbs_dir() -> Path:
    """``~/.facesorter/thumbs``."""
    return Path.home() / ".facesorter" / "thumbs"


# --------------------------------------------------------------------------- thumbnails


class ThumbnailCache:
    """On-disk cache of small JPEG thumbnails, safe to use from several threads.

    A thumbnail is named by the SHA-1 of ``canonical path | mtime_ns | size``, so an edited
    photo gets a new thumbnail and an unchanged one is never decoded twice. Files are written
    to a temporary name and renamed into place (readers never see a partial file). When the
    directory grows past ``max_bytes`` the least recently used thumbnails are deleted until it
    is back under 90% of the cap. A photo that is missing or cannot be decoded yields ``None``;
    :meth:`placeholder` provides a stand-in image to show instead.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        max_bytes: int = DEFAULT_THUMB_CAP_BYTES,
        size: int = THUMB_SIZE,
    ) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        if size < 16:
            raise ValueError(f"size must be at least 16 pixels, got {size}")
        root = Path(directory) if directory is not None else default_thumbs_dir()
        root = root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        try:
            root.chmod(0o700)
        except OSError:  # pragma: no cover - e.g. not our directory
            logger.debug("could not restrict permissions on %s", root)
        self.directory = root.resolve()
        self.max_bytes = int(max_bytes)
        self.size = int(size)
        self._lock = threading.Lock()  # guards _total and eviction
        self._stripes = [threading.Lock() for _ in range(_STRIPES)]
        self._total = 0
        self._remove_stale_temp_files()
        with self._lock:
            self._total = self._scan()[1]

    # ------------------------------------------------------------------ naming

    @staticmethod
    def key(path: str, mtime_ns: int, size: int) -> str:
        """The cache key (hex SHA-1) for a photo's path, mtime and size."""
        canonical = os.path.abspath(os.path.expanduser(path))
        return hashlib.sha1(f"{canonical}|{mtime_ns}|{size}".encode()).hexdigest()

    def path_for(self, meta: Any) -> Path:
        """Where the thumbnail of ``meta`` (a ``PhotoMeta``) lives, whether or not it exists."""
        return self.directory / f"{self.key(meta.path, meta.mtime_ns, meta.size)}.jpg"

    # ------------------------------------------------------------------ public API

    def get(self, meta: Any) -> Path | None:
        """The thumbnail path for ``meta``, creating it if needed; ``None`` if unreadable."""
        target = self.path_for(meta)
        stripe = self._stripes[int(target.stem[:8], 16) % _STRIPES]
        with stripe:  # one thread builds a given thumbnail; others wait and reuse it
            if self._touch(target):
                return target
            try:
                written = self._build(meta.path, target)
            except Exception as exc:  # unreadable, vanished, disk full ...
                logger.info("no thumbnail for %s: %s", meta.path, exc)
                return None
        try:
            self._account(written, keep=target)
        except OSError as exc:  # eviction trouble must not lose the thumbnail we just made
            logger.info("thumbnail eviction failed: %s", exc)
        return target

    def get_many(self, metas: Sequence[Any], workers: int = _THUMB_WORKERS) -> list[Path | None]:
        """:meth:`get` for each item, decoding in parallel; the result keeps the input order."""
        if len(metas) <= 1:
            return [self.get(m) for m in metas]
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="thumb") as pool:
            return list(pool.map(self.get, metas))

    def placeholder(self) -> Path | None:
        """A grey "photo unavailable" image (created once, never evicted).

        ``None`` if it cannot be written (read-only or full disk); never raises.
        """
        target = self.directory / _PLACEHOLDER_NAME
        if not target.exists():
            try:
                img = Image.new("RGB", (self.size, self.size * 3 // 4), (58, 58, 64))
                ImageDraw.Draw(img).text((16, 16), "photo unavailable", fill=(200, 200, 205))
                self._write_jpeg(img, target)
            except Exception as exc:
                logger.info("no placeholder image: %s", exc)
                return None
        return target

    def total_bytes(self) -> int:
        """Bytes currently used by thumbnails (a fresh scan)."""
        with self._lock:
            self._total = self._scan()[1]
            return self._total

    # ------------------------------------------------------------------ internals

    def _touch(self, target: Path) -> bool:
        """Mark ``target`` as recently used; ``False`` if it does not exist."""
        try:
            os.utime(target)
        except OSError:
            return False
        return True

    def _build(self, source: str, target: Path) -> int:
        img = load_for_model(source, target=self.size)
        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)  # fit a square box
        return self._write_jpeg(img, target)

    @staticmethod
    def _write_jpeg(img: Image.Image, target: Path) -> int:
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                img.convert("RGB").save(handle, "JPEG", quality=82, optimize=True)
            size = os.path.getsize(tmp_name)
            os.replace(tmp_name, target)  # atomic on POSIX and Windows
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - already gone
                pass
            raise
        return size

    def _scan(self) -> tuple[list[tuple[int, int, str]], int]:
        entries: list[tuple[int, int, str]] = []
        total = 0
        with os.scandir(self.directory) as it:
            for entry in it:
                if not _THUMB_NAME.match(entry.name):
                    continue
                try:
                    st = entry.stat()
                except OSError:  # pragma: no cover - deleted while scanning
                    continue
                entries.append((st.st_mtime_ns, st.st_size, entry.path))
                total += st.st_size
        return entries, total

    def _account(self, written: int, keep: Path) -> None:
        with self._lock:
            self._total += written
            if self._total <= self.max_bytes:
                return
            entries, total = self._scan()
            goal = int(self.max_bytes * 0.9)
            for _, size, path in sorted(entries):
                if total <= goal:
                    break
                if path == str(keep):
                    continue
                try:
                    os.unlink(path)
                except OSError:
                    continue
                total -= size
            self._total = total

    def _remove_stale_temp_files(self) -> None:
        """Delete leftovers of a crash (temporary files older than an hour)."""
        cutoff = time.time() - 3600
        for entry in self.directory.glob(".tmp-*.part"):
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
            except OSError:  # pragma: no cover - raced with another process
                continue
