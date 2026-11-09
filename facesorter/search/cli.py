"""
facesorter/search/cli.py
Command-line front end for photo search: ``index``, ``search``, ``serve``, ``people``,
``stats`` and ``eval``.

:func:`register` adds the subcommands to the ``facesorter`` parser and :func:`dispatch` runs
the chosen one. Importing this module needs only the standard library: every handler imports
its heavy modules (engine, store, indexer, UI, face model) when it runs, so ``facesorter
--help`` stays fast and works even on Python 3.9 without any search dependency.

Every ``main_*`` entry point takes the parsed arguments plus optional factories
(``engine_factory``, ``indexer_factory``, ...) so tests can substitute fakes. Exit codes follow
the rest of the CLI: 0 success, 1 runtime failure, 2 usage / configuration / path / missing
dependency, 130 Ctrl-C.
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import dataclasses
import importlib
import importlib.util
import ipaddress
import json
import logging
import os
import re
import sys
import traceback
import types
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from typing import Any

from ..errors import CopyError, FaceSorterError, InvalidPathError
from ..paths import is_inside
from .errors import IndexNotFoundError, ModelUnavailableError, SearchError

__all__ = [
    "COMMANDS",
    "MissingDependencyError",
    "UsageError",
    "dispatch",
    "main_eval",
    "main_index",
    "main_people",
    "main_search",
    "main_serve",
    "main_stats",
    "parse_date_flag",
    "register",
]

COMMANDS = ("index", "search", "serve", "people", "stats", "eval")

logger = logging.getLogger("facesorter")

_INSTALL_HINT = 'pip install "facesorter[search]"'
_FACE_HINT = "pip install insightface onnxruntime"
_LLM_CHOICES = ("auto", "mlx", "llama-cpp", "none")
_DEVICE_CHOICES = ("auto", "mps", "cuda", "cpu")
_DEFAULT_K = 10
_DEFAULT_PORT = 7860
_DEFAULT_FACE_CONFIDENCE = 0.8
_DEFAULT_CLUSTER_THRESHOLD = 0.6
_MAX_SAMPLE_PATHS = 3
_MAX_NAME_CHARS = 100


class UsageError(FaceSorterError):
    """The command line asked for something impossible (exit code 2)."""


class MissingDependencyError(SearchError):
    """An optional dependency of the search feature is not installed (exit code 2)."""


# --------------------------------------------------------------------------- parser


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port: {text!r}") from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"port must be between 0 and 65535, got {value}")
    return value


def _add_common(parser: argparse.ArgumentParser, *, index: bool = True) -> None:
    # SUPPRESS: a -v given before the subcommand is not overwritten by the subparser.
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show progress details and tracebacks",
    )
    if index:
        parser.add_argument(
            "--index", metavar="PATH", default=None, help="Index file (default: per-user location)"
        )


def _add_llm(parser: argparse.ArgumentParser, *, no_llm_help: str) -> None:
    parser.add_argument("--no-llm", action="store_true", help=no_llm_help)
    parser.add_argument(
        "--llm",
        choices=_LLM_CHOICES,
        default="auto",
        help="Local LLM backend for query understanding (default: auto)",
    )
    parser.add_argument("--llm-model", metavar="MODEL", default=None, help="LLM model name or path")


def register(subparsers: Any) -> None:
    """Add the search subcommands to the ``facesorter`` subparsers action."""
    p = subparsers.add_parser("index", help="Index a photo library for search")
    _add_common(p)
    p.add_argument("library", nargs="?", metavar="LIBRARY", help="Folder of photos to index")
    p.add_argument(
        "--remove-root",
        metavar="DIR",
        help="Instead of indexing, forget a library folder (its photos leave the index; "
        "the files are not touched). The folder need not exist any more",
    )
    p.add_argument("--workers", type=_positive_int, default=None, help="Decoder threads")
    p.add_argument("--batch-size", type=_positive_int, default=None, help="Images per model batch")
    p.add_argument("--limit", type=_positive_int, default=None, help="Index at most N new photos")
    p.add_argument("--force", action="store_true", help="Re-index photos even if unchanged")
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DIR",
        help="Skip this folder (repeatable)",
    )
    p.add_argument("--device", choices=_DEVICE_CHOICES, default="auto", help="Compute device")
    p.add_argument(
        "--allow-empty-prune",
        action="store_true",
        help="Allow removing every photo from the index when the library looks empty",
    )

    p = subparsers.add_parser("search", help="Search the indexed photos in natural language")
    _add_common(p)
    p.add_argument("query", nargs="*", metavar="QUERY", help="What to look for")
    p.add_argument("--from", dest="date_from", metavar="DATE", help="YYYY-MM-DD, YYYY-MM or YYYY")
    p.add_argument("--to", dest="date_to", metavar="DATE", help="YYYY-MM-DD, YYYY-MM or YYYY")
    p.add_argument("--place", metavar="NAME", help="Only photos taken at this place")
    p.add_argument(
        "--person", action="append", default=[], metavar="NAME", help="Named person (repeatable)"
    )
    p.add_argument(
        "-k", type=_positive_int, default=_DEFAULT_K, metavar="N", help="Number of results"
    )
    _add_llm(p, no_llm_help="Understand the query with rules only (no local LLM)")
    p.add_argument("--no-ask", action="store_true", help="Never ask follow-up questions")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="One JSON object per result line")
    out.add_argument("--paths-only", action="store_true", help="Print only the file paths")
    p.add_argument(
        "--copy-to",
        metavar="DIR",
        help="Copy the result photos into this folder. Never overwrites: copying the same "
        "photo again adds _1, _2, ... to the name",
    )

    p = subparsers.add_parser("serve", help="Start the local search web app")
    _add_common(p)
    p.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1)")
    p.add_argument("--port", type=_port, default=_DEFAULT_PORT, help="Port (default: 7860)")
    _add_llm(p, no_llm_help="Understand queries with rules only (no local LLM)")
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow a non-loopback --host (exposes your photo library to the network)",
    )
    p.add_argument("--open-browser", action="store_true", help="Open the page in a browser")

    p = subparsers.add_parser("people", help="Detect, group and name the people in your photos")
    people = p.add_subparsers(
        dest="people_command", required=True, metavar="{scan,cluster,list,name}"
    )
    q = people.add_parser("scan", help="Detect faces in indexed photos")
    _add_common(q)
    q.add_argument(
        "--confidence", type=_unit_float, default=_DEFAULT_FACE_CONFIDENCE, help="Detection score"
    )
    q.add_argument("--limit", type=_positive_int, default=None, help="Scan at most N photos")
    q = people.add_parser("cluster", help="Group detected faces into people")
    _add_common(q)
    q.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_CLUSTER_THRESHOLD,
        help="Clustering threshold (cosine distance)",
    )
    q = people.add_parser("list", help="List the face clusters")
    _add_common(q)
    q = people.add_parser("name", help="Name (or un-name) a face cluster")
    _add_common(q)
    q.add_argument("cluster_id", type=int, metavar="CLUSTER_ID")
    q.add_argument("name", nargs="?", default=None, metavar="NAME", help="Empty removes the name")
    q.add_argument("--clear", action="store_true", help="Remove the cluster's name")

    p = subparsers.add_parser("stats", help="Show what the index contains")
    _add_common(p)

    # eval's options belong to the evaluation module, which is imported only when it runs, so
    # this parser accepts everything after "eval" verbatim (even option-looking words).
    p = subparsers.add_parser(
        "eval", help="Evaluate search quality (see `facesorter eval --help`)", add_help=False
    )
    p.parse_known_args = types.MethodType(_keep_all_arguments, p)  # type: ignore[method-assign]


def _keep_all_arguments(
    self: argparse.ArgumentParser,
    args: Sequence[str] | None = None,
    namespace: argparse.Namespace | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    ns = argparse.Namespace() if namespace is None else namespace
    ns.eval_args = list(args or [])
    return ns, []


# --------------------------------------------------------------------------- plumbing


@contextlib.contextmanager
def _logging(verbose: bool) -> Iterator[None]:
    """Log to stderr (INFO with ``-v``, warnings otherwise); restore the logger afterwards."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def _old_python_message() -> str | None:
    """Why search cannot work here, on Python < 3.10 (a pip suggestion would not help)."""
    if sys.version_info >= (3, 10):
        return None
    return (
        "Photo search needs Python 3.10 or newer "
        f"(this is Python {sys.version_info[0]}.{sys.version_info[1]})."
    )


def _missing_message(exc: ImportError) -> str:
    root = (exc.name or "").split(".")[0]
    if root in {"insightface", "onnxruntime"}:
        return f"The face model needs {root}, which is not installed. Install it with: {_FACE_HINT}"
    old = _old_python_message()
    if old:
        return old
    what = f"{exc.name!r}" if exc.name else str(exc)
    return (
        f"Photo search needs an optional package that is missing ({what}). Install: {_INSTALL_HINT}"
    )


def _exit_code(exc: FaceSorterError) -> int:
    usage = (InvalidPathError, IndexNotFoundError, ModelUnavailableError, UsageError)
    return 2 if isinstance(exc, (*usage, MissingDependencyError)) else 1


def _tolerate_odd_encodings() -> None:
    """Never crash on a character the terminal's encoding cannot show (print an escape)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="backslashreplace")


def _stdout_shows_unicode() -> bool:
    try:
        "\u00ce".encode(getattr(sys.stdout, "encoding", None) or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _guarded(args: argparse.Namespace, run: Callable[[], int]) -> int:
    """Run a handler with logging set up and every failure turned into a message + exit code."""
    verbose = bool(getattr(args, "verbose", False))
    _tolerate_odd_encodings()
    with _logging(verbose):
        try:
            return run()
        except FaceSorterError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return _exit_code(exc)
        except ImportError as exc:
            print(f"error: {_missing_message(exc)}", file=sys.stderr)
            if verbose:
                traceback.print_exc()
            return 2
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130
        except BrokenPipeError:  # e.g. `facesorter search ... | head`
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - keeps the exit-time flush quiet
            return 0
        except Exception as exc:  # last resort: a clean one-liner, traceback only with -v
            print(f"error: Unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
            if verbose:
                traceback.print_exc()
            return 1


def _import(name: str) -> Any:
    return importlib.import_module(name, package=__package__)


def _default_index_path() -> str:
    return str(_import(".engine").default_index_path())


def _index_path(args: argparse.Namespace) -> str:
    return str(args.index) if getattr(args, "index", None) else _default_index_path()


@contextlib.contextmanager
def _closing(obj: Any) -> Iterator[Any]:
    try:
        yield obj
    finally:
        close = getattr(obj, "close", None)
        if callable(close):
            close()


def _open_store(args: argparse.Namespace, store_factory: Callable[..., Any] | None) -> Any:
    """Open the index for reading and writing without loading any model."""
    path = _index_path(args)
    if store_factory is None:
        store_factory = _import(".store").Store.open
    return store_factory(path, create=False)


def _echo(text: str = "", *, err: bool = False) -> None:
    print(text, file=sys.stderr if err else sys.stdout)


# --------------------------------------------------------------------------- dates


_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")


def parse_date_flag(text: str, *, end: bool, flag: str = "date") -> date:
    """Parse ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` to the first (or, with ``end``, last) day.

    ``--from 2024-02`` starts on 2024-02-01; ``--to 2024-02`` ends on 2024-02-29 in a leap
    year; ``--to 2023`` ends on 2023-12-31.
    """
    m = _DATE_RE.match(text.strip())
    if m is None:
        raise UsageError(f"{flag}: {text!r} is not a date; use YYYY-MM-DD, YYYY-MM or YYYY.")
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else None
    day = int(m.group(3)) if m.group(3) else None
    try:
        if month is None:
            return date(year, 12, 31) if end else date(year, 1, 1)
        if day is None:
            if end:
                return date(year, month, calendar.monthrange(year, month)[1])
            return date(year, month, 1)
        return date(year, month, day)
    except (ValueError, calendar.IllegalMonthError):
        raise UsageError(f"{flag}: {text!r} is not a valid calendar date.") from None


def _date_range(args: argparse.Namespace) -> Any:
    """The ``DateRange`` for ``--from`` / ``--to`` (either may be omitted), or ``None``."""
    if not args.date_from and not args.date_to:
        return None
    from .types import DateRange

    start = parse_date_flag(args.date_from, end=False, flag="--from") if args.date_from else None
    stop = parse_date_flag(args.date_to, end=True, flag="--to") if args.date_to else None
    if start and stop and stop < start:
        raise UsageError(f"--from ({start}) is after --to ({stop}).")
    return DateRange(start or date.min, stop or date.max)


# --------------------------------------------------------------------------- index


class _Progress:
    """Adapts an ``(done, total[, stage])`` progress callback to a tqdm bar on stderr.

    A new bar starts whenever the stage changes. The indexer's ``store`` stage trails the
    ``encode`` stage by one chunk, so it is not drawn separately; a ``total`` of 0 (the scan
    is still counting) shows a bar without a total.
    """

    _LABELS = {"scan": "Scanning", "metadata": "Reading metadata", "encode": "Embedding"}

    def __init__(
        self, bar_factory: Callable[..., Any] | None = None, desc: str = "Working"
    ) -> None:
        self._factory = bar_factory
        self._desc = desc
        self._bar: Any = None
        self._stage: str | None = None
        self._done = 0

    def __call__(self, done: int, total: int = 0, stage: str | None = None) -> None:
        if stage == "store":
            return
        if self._bar is None or stage != self._stage:
            self.close()
            if self._factory is None:
                self._factory = _import_tqdm()
            self._bar = self._factory(
                total=total or None,
                unit="photo",
                desc=self._LABELS.get(stage or "", self._desc),
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
            )
            self._stage, self._done = stage, 0
        elif total and getattr(self._bar, "total", None) != total:
            self._bar.total = total
        self._bar.update(max(0, done - self._done))
        self._done = done

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _import_tqdm() -> Callable[..., Any]:
    from tqdm import tqdm

    return tqdm


def _print_failures(failed: Sequence[tuple[str, str]], what: str, verbose: bool) -> None:
    """Warn about ``(path, reason)`` failures; list them only with ``-v`` (paths are private)."""
    if not failed:
        return
    hint = "" if verbose else " (run with -v to list them)"
    _echo(f"warning: {len(failed)} photo(s) could not be {what}{hint}", err=True)
    if verbose:
        for path, reason in failed:
            _echo(f"  {path}: {reason}", err=True)


def _indexer_options(args: argparse.Namespace) -> dict[str, int]:
    return {
        key: value
        for key, value in (("workers", args.workers), ("batch_size", args.batch_size))
        if value is not None
    }


def _remove_root(args: argparse.Namespace, indexer_factory: Callable[..., Any] | None) -> int:
    path = _index_path(args)
    if not os.path.isfile(path):
        raise IndexNotFoundError(f"No photo index at {path}. Run `facesorter index` first.")
    root = os.path.abspath(os.path.expanduser(args.remove_root))
    if indexer_factory is None:
        indexer_factory = _import(".indexer").open_indexer
    with _closing(indexer_factory(path, device=args.device, **_indexer_options(args))) as indexer:
        roots_before = indexer.store.get_meta("library_roots")
        removed = indexer.remove_root(root)
        roots_after = indexer.store.get_meta("library_roots")
    if not removed and roots_before == roots_after:
        raise UsageError(f"{args.remove_root} is not an indexed library folder; nothing removed.")
    _echo(f"Removed {removed} photo(s) of {root} from the index {path}")
    return 0


def main_index(
    args: argparse.Namespace,
    *,
    indexer_factory: Callable[..., Any] | None = None,
    progress_factory: Callable[..., Any] | None = None,
) -> int:
    """``facesorter index LIBRARY``: scan, embed and store new or changed photos."""
    return _guarded(args, lambda: _index(args, indexer_factory, progress_factory))


def _index(
    args: argparse.Namespace,
    indexer_factory: Callable[..., Any] | None,
    progress_factory: Callable[..., Any] | None,
) -> int:
    if args.remove_root is not None:
        if args.library:
            raise UsageError("--remove-root cannot be combined with a LIBRARY to index.")
        return _remove_root(args, indexer_factory)
    if not args.library:
        raise UsageError("Give the LIBRARY folder to index, or --remove-root DIR.")
    library = os.path.abspath(os.path.expanduser(args.library))
    if os.path.exists(library) and not os.path.isdir(library):
        raise InvalidPathError(f"The library path is not a directory: {args.library}")
    if not os.path.isdir(library):
        raise InvalidPathError(f"The library folder does not exist: {args.library}")
    excludes = tuple(os.path.abspath(os.path.expanduser(d)) for d in args.exclude)
    path = _index_path(args)
    if is_inside(path, library):
        _echo(
            f"warning: the index file ({path}) is inside the library; it is skipped while "
            "indexing, but keeping it elsewhere is safer.",
            err=True,
        )
    if indexer_factory is None:
        indexer_factory = _import(".indexer").open_indexer

    options = _indexer_options(args)
    progress = _Progress(progress_factory)
    try:
        with _closing(indexer_factory(path, device=args.device, **options)) as indexer:
            report = indexer.run(
                library,
                exclude_dirs=excludes,
                force=args.force,
                limit=args.limit,
                progress=progress,
                allow_empty_prune=args.allow_empty_prune,
            )
    finally:
        progress.close()

    _echo(f"Index: {path}")
    for label in ("scanned", "new", "updated", "unchanged", "removed"):
        _echo(f"  {label:<10}{getattr(report, label)}")
    _echo(f"  {'failed':<10}{len(report.failed)}")
    _echo(f"  {'time':<10}{report.seconds:.1f} s (model: {report.model_name})")
    _print_failures(report.failed, "indexed", args.verbose)
    return 0


# --------------------------------------------------------------------------- search


def _open_engine(args: argparse.Namespace, engine_factory: Callable[..., Any] | None) -> Any:
    if engine_factory is None:
        engine_factory = _import(".engine").open_engine
    llm = "none" if args.no_llm else args.llm
    return engine_factory(
        _index_path(args),
        llm=llm,
        llm_model=args.llm_model,
        use_llm=llm != "none",
    )


def _resolve_place(engine: Any, name: str, gazetteer_factory: Callable[[], Any] | None) -> Any:
    gazetteer = getattr(engine, "gazetteer", None)
    if gazetteer is None:
        factory = gazetteer_factory or _import(".geo").build_gazetteer
        gazetteer = factory()
    place = gazetteer.resolve(name)
    if place is None:
        raise UsageError(
            f"--place: {name!r} is not a place I know. Try a city, region or country name."
        )
    return place


def _resolve_people(engine: Any, names: Sequence[str]) -> tuple[str, ...]:
    known = list(engine.store.known_names())
    fold = _import(".store").fold
    by_key = {fold(n): n for n in known}
    resolved: list[str] = []
    for name in names:
        match = by_key.get(fold(" ".join(name.split())))
        if match is None:
            if known:
                raise UsageError(
                    f"--person: {name!r} is not a known person. Known: {', '.join(known)}"
                )
            raise UsageError(
                f"--person: {name!r} is not a known person; no one has been named yet "
                "(see `facesorter people name`)."
            )
        if match not in resolved:
            resolved.append(match)
    return tuple(resolved)


def _explicit_filters(
    args: argparse.Namespace, engine: Any, gazetteer_factory: Callable[[], Any] | None
) -> Any:
    from .types import Filters

    date_range = _date_range(args)
    place = _resolve_place(engine, args.place, gazetteer_factory) if args.place else None
    people = _resolve_people(engine, args.person) if args.person else ()
    filters = Filters(date_range=date_range, place=place, people=people)
    return None if filters.is_empty else filters


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def _ask(response: Any, input_fn: Callable[[str], str]) -> dict[str, str]:
    """Put each unresolved phrase to the user once; an empty reply (or EOF) skips it."""
    answers: dict[str, str] = {}
    eof = False
    for item in response.needs_clarification:
        if eof:
            answers[item.phrase] = ""
            continue
        print(item.question, file=sys.stderr)
        print(
            "Reword it, or press Enter to search without it: ", end="", file=sys.stderr, flush=True
        )
        try:
            answers[item.phrase] = input_fn("").strip()
        except EOFError:
            eof = True
            answers[item.phrase] = ""
    return answers


def _when(photo: Any) -> str:
    taken = photo.meta.taken
    if taken is None:
        return "-"
    precision = getattr(photo.meta, "date_precision", "day")
    text = taken.date().isoformat()
    return text[:4] if precision == "year" else text[:7] if precision == "month" else text


def _where(photo: Any) -> str:
    return photo.place.display if photo.place is not None and photo.place.display else "-"


def _print_table(hits: Sequence[Any]) -> None:
    rows = [
        (str(i), f"{h.score:.3f}", _when(h.photo), _where(h.photo), h.photo.meta.path)
        for i, h in enumerate(hits, 1)
    ]
    head = ("#", "score", "date", "place", "path")
    widths = [max(len(head[c]), *(len(r[c]) for r in rows)) for c in range(4)]
    for row in (head, *rows):
        cells = [row[c].rjust(widths[c]) if c < 2 else row[c].ljust(widths[c]) for c in range(4)]
        print("  ".join(cells) + "  " + row[4])


def _print_json(hits: Sequence[Any]) -> None:
    unicode_ok = _stdout_shows_unicode()  # keep names readable, but always emit valid JSON
    for i, hit in enumerate(hits, 1):
        meta = hit.photo.meta
        record = {
            "rank": i,
            "score": round(float(hit.score), 4),
            "id": hit.photo.id,
            "path": meta.path,
            "date": meta.taken.isoformat() if meta.taken else None,
            "date_precision": meta.date_precision,
            "place": _where(hit.photo) if hit.photo.place else None,
        }
        print(json.dumps(record, ensure_ascii=not unicode_ok))


def _copy_hits(hits: Sequence[Any], dest: str, *, verbose: bool, err: bool) -> None:
    from ..utils import copy_image

    copied = 0
    failed: list[tuple[str, str]] = []
    for hit in hits:
        src = hit.photo.meta.path
        try:
            copy_image(src, dest, os.path.basename(src))
            copied += 1
        except OSError as exc:
            failed.append((src, exc.strerror or type(exc).__name__))
    if hits and not copied:
        if verbose:
            for src, why in failed:
                _echo(f"  {src}: {why}", err=True)
        reason = failed[0][1]
        raise CopyError(f"Could not copy any of the {len(hits)} photos into {dest}: {reason}")
    _echo(f"Copied {copied} photo(s) to {dest}", err=err)
    if failed:
        _echo(f"warning: {len(failed)} photo(s) could not be copied", err=True)
        if verbose:
            for src, why in failed:
                _echo(f"  {src}: {why}", err=True)


def main_search(
    args: argparse.Namespace,
    *,
    engine_factory: Callable[..., Any] | None = None,
    gazetteer_factory: Callable[[], Any] | None = None,
    is_interactive: Callable[[], bool] | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    """``facesorter search [QUERY ...]``: ranked photos for a query and/or explicit filters."""
    return _guarded(
        args, lambda: _search(args, engine_factory, gazetteer_factory, is_interactive, input_fn)
    )


def _search(
    args: argparse.Namespace,
    engine_factory: Callable[..., Any] | None,
    gazetteer_factory: Callable[[], Any] | None,
    is_interactive: Callable[[], bool] | None,
    input_fn: Callable[[str], str] | None,
) -> int:
    from .types import SearchRequest

    query = " ".join(args.query).strip()
    if not query and not (args.date_from or args.date_to or args.place or args.person):
        raise UsageError("Give a query, or at least one of --from, --to, --place, --person.")
    _date_range(args)  # reject bad dates before anything heavy is loaded
    dest = None
    if args.copy_to:
        dest = os.path.abspath(os.path.expanduser(args.copy_to))
        if os.path.exists(dest) and not os.path.isdir(dest):
            raise InvalidPathError(f"The --copy-to path exists and is not a folder: {args.copy_to}")

    with _closing(_open_engine(args, engine_factory)) as engine:
        filters = _explicit_filters(args, engine, gazetteer_factory)
        interactive = not args.no_ask and (is_interactive or _stdin_is_tty)()
        request = SearchRequest(
            query=query,
            filters=filters,
            k=args.k,
            use_llm=not args.no_llm and args.llm != "none",
            allow_clarification=interactive,
        )
        response = engine.search(request)
        if response.needs_clarification and interactive:
            answers = _ask(response, input_fn or input)
            request = dataclasses.replace(request, clarifications=answers)
            response = engine.search(request)

    engine_mod = _import(".engine")
    plain = args.json or args.paths_only
    _echo(
        f"Searching: {engine_mod.describe_filters(response.applied) or '(no filters)'}", err=plain
    )
    for line in engine_mod.explain(response):
        _echo(line, err=True)

    hits = response.hits
    if not hits:
        _echo("No matching photos.", err=True)
    elif args.paths_only:
        for hit in hits:
            print(hit.photo.meta.path)
    elif args.json:
        _print_json(hits)
    else:
        _print_table(hits)
    if dest is not None:
        _copy_hits(hits, dest, verbose=args.verbose, err=plain)
    return 0


# --------------------------------------------------------------------------- serve


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def main_serve(
    args: argparse.Namespace,
    *,
    engine_factory: Callable[..., Any] | None = None,
    serve_fn: Callable[..., None] | None = None,
) -> int:
    """``facesorter serve``: run the local web app (blocks until interrupted)."""
    return _guarded(args, lambda: _serve(args, engine_factory, serve_fn))


def _serve(
    args: argparse.Namespace,
    engine_factory: Callable[..., Any] | None,
    serve_fn: Callable[..., None] | None,
) -> int:
    if not _is_loopback(args.host):
        if not args.allow_remote:
            raise UsageError(
                f"Refusing to listen on {args.host}: the web app would expose your photo library "
                "to the network. Use the default 127.0.0.1, or pass --allow-remote if you really "
                "mean it."
            )
        _echo(
            f"warning: listening on {args.host} exposes your photo library to the network; "
            "anyone who can reach this address can browse it.",
            err=True,
        )
    path = _index_path(args)
    if not os.path.isfile(path):
        raise IndexNotFoundError(f"No photo index at {path}. Run `facesorter index` first.")
    if serve_fn is None:
        if importlib.util.find_spec("gradio") is None:
            raise MissingDependencyError(
                _old_python_message() or f"The web app needs gradio. Install: {_INSTALL_HINT}"
            )
        serve_fn = _import(".ui").serve

    with _closing(_open_engine(args, engine_factory)) as engine:
        try:
            roots = json.loads(engine.store.get_meta("library_roots") or "[]")
        except ValueError:
            roots = []
        if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots) or not roots:
            raise IndexNotFoundError(
                "The index does not record which folders it covers; run `facesorter index` first."
            )
        host = args.host.strip("[]")
        shown = f"[{host}]" if ":" in host else host
        where = f"http://{shown}:{args.port}" if args.port else f"{shown} (port chosen by the OS)"
        _echo(f"Serving {len(roots)} photo folder(s) at {where} (Ctrl-C to stop)")
        serve_fn(
            engine,
            library_roots=roots,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
    return 0


# --------------------------------------------------------------------------- stats


def main_stats(args: argparse.Namespace, *, store_factory: Callable[..., Any] | None = None) -> int:
    """``facesorter stats``: what the index contains."""
    return _guarded(args, lambda: _stats(args, store_factory))


def _stats(args: argparse.Namespace, store_factory: Callable[..., Any] | None) -> int:
    with _closing(_open_store(args, store_factory)) as store:
        s = store.stats()
        model = getattr(store, "model_name", None)
    _echo(f"Index:        {_index_path(args)}" + (f" (model: {model})" if model else ""))
    _echo(f"Photos:       {s.photos}")
    _echo(f"With GPS:     {s.with_gps}")
    _echo(f"With a place: {s.with_place}")
    if s.date_min and s.date_max:
        _echo(f"Date span:    {s.date_min} to {s.date_max}")
    else:
        _echo("Date span:    none")
    sources = ", ".join(f"{k} {v}" for k, v in s.by_date_source.items()) or "none"
    _echo(f"Date source:  {sources}")
    _echo(f"Faces:        {s.faces}")
    _echo(f"Named people: {s.named_people}")
    _echo(f"Feedback:     {s.feedback}")
    return 0


# --------------------------------------------------------------------------- people


def main_people(
    args: argparse.Namespace,
    *,
    store_factory: Callable[..., Any] | None = None,
    people_module: Any = None,
) -> int:
    """``facesorter people scan|cluster|list|name``."""
    return _guarded(args, lambda: _people(args, store_factory, people_module))


def _people(args: argparse.Namespace, store_factory: Callable[..., Any] | None, mod: Any) -> int:
    sub = args.people_command
    with _closing(_open_store(args, store_factory)) as store:
        if sub == "list":
            return _people_list(args, store)
        if sub == "name":
            return _people_name(args, store)
        if mod is None:
            mod = _import(".people")
        if sub == "scan":
            progress = _Progress(desc="Scanning faces")
            try:
                scan = mod.scan_faces(
                    store, confidence=args.confidence, limit=args.limit, progress=progress
                )
            finally:
                progress.close()
            _echo(
                f"Scanned {scan.photos_scanned} photo(s): {scan.faces_found} face(s), "
                f"{scan.photos_without_faces} photo(s) without faces, "
                f"{len(scan.failed)} failed ({scan.seconds:.1f} s)"
            )
            _print_failures(scan.failed, "scanned", args.verbose)
            return 0
        done = mod.cluster_faces(store, threshold=args.threshold)
    _echo(
        f"Grouped {done.faces} face(s) into {done.clusters} cluster(s) "
        f"({done.singletons} with a single face); kept {done.named_kept} name(s) "
        f"({done.seconds:.1f} s)"
    )
    if done.names_dropped:
        _echo(
            f"warning: names that no longer fit a cluster: {', '.join(done.names_dropped)}",
            err=True,
        )
    return 0


def _people_list(args: argparse.Namespace, store: Any) -> int:
    clusters = store.clusters()
    if not clusters:
        _echo(
            "No face clusters yet. Run `facesorter people scan`, then `facesorter people cluster`."
        )
        return 0
    _echo(f"{'id':>5}  {'faces':>6}  {'photos':>6}  name")
    for c in clusters:
        _echo(f"{c.cluster_id:>5}  {c.face_count:>6}  {c.photo_count:>6}  {c.name or '-'}")
        if args.verbose:
            for photo in store.get_photos(c.sample_photo_ids[:_MAX_SAMPLE_PATHS]):
                _echo(f"{'':>5}  {photo.meta.path}")
    return 0


def _people_name(args: argparse.Namespace, store: Any) -> int:
    if args.clear and args.name:
        raise UsageError("Give a NAME or --clear, not both.")
    if args.name is None and not args.clear:
        raise UsageError("Give a NAME for the cluster, or --clear to remove its name.")
    name = None if args.clear else " ".join((args.name or "").split())
    if name and len(name) > _MAX_NAME_CHARS:
        raise UsageError(f"The name is too long ({len(name)} characters; the limit is 100).")
    duplicate = _import(".store").DuplicateNameError
    try:
        store.name_cluster(args.cluster_id, name or None)
    except duplicate as exc:
        raise UsageError(str(exc)) from None
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    _echo(f"Cluster {args.cluster_id}: " + (f"named {name!r}" if name else "name removed"))
    return 0


# --------------------------------------------------------------------------- eval


def main_eval(
    args: argparse.Namespace,
    *,
    evaluate_module: Any = None,
    engine_factory: Callable[..., Any] | None = None,
) -> int:
    """``facesorter eval ...``: delegated to :mod:`facesorter.search.evaluate`."""
    return _guarded(args, lambda: _eval(args, evaluate_module, engine_factory))


def _eval(args: argparse.Namespace, mod: Any, engine_factory: Callable[..., Any] | None) -> int:
    if mod is None:
        mod = _import(".evaluate")
    parser = argparse.ArgumentParser(prog="facesorter eval")
    mod.add_arguments(parser)
    ns = parser.parse_args(list(args.eval_args))
    ns.verbose = bool(getattr(args, "verbose", False)) or bool(getattr(ns, "verbose", False))
    if engine_factory is None:
        return int(mod.run(ns))
    return int(mod.run(ns, engine_factory=engine_factory))


# --------------------------------------------------------------------------- dispatch


def dispatch(args: argparse.Namespace) -> int:
    """Run the parsed search subcommand and return its exit code."""
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "index": main_index,
        "search": main_search,
        "serve": main_serve,
        "people": main_people,
        "stats": main_stats,
        "eval": main_eval,
    }
    return handlers[args.command](args)
