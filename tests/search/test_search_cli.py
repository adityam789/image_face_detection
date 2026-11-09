"""Tests for facesorter.search.cli: index, search, serve, people, stats and eval commands.

Everything runs against fakes (engine, indexer, UI, evaluation module) injected through the
``main_*`` factory parameters, or against a real ``Store`` in a temporary directory. No model
weights, network or real photos are involved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from facesorter.cli import build_parser, main
from facesorter.errors import FaceSorterError, InvalidPathError
from facesorter.search import cli as search_cli
from facesorter.search.errors import (
    IndexMismatchError,
    IndexNotFoundError,
    ModelUnavailableError,
    SearchError,
)
from facesorter.search.indexer import IndexReport
from facesorter.search.people import ClusterReport, ScanReport
from facesorter.search.store import FaceRow, Store
from facesorter.search.types import (
    DateRange,
    Filters,
    Hit,
    IndexedPhoto,
    ParsedQuery,
    ParseResult,
    PhotoMeta,
    PlaceFilter,
    PlaceLabel,
    SearchResponse,
    Unresolved,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIM = 8


def parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


@pytest.fixture(autouse=True)
def index_env(tmp_path, monkeypatch):
    """The default index path is a temporary one, never the user's."""
    path = tmp_path / "default-index.sqlite"
    monkeypatch.setenv("FACESORTER_INDEX", str(path))
    return path


# --------------------------------------------------------------------------- fakes


def make_photo(
    pid: int = 1,
    path: str = "/lib/a.jpg",
    taken: datetime | None = datetime(2023, 7, 4, 12, 0),
    city: str | None = "Key West",
    precision: str = "day",
) -> IndexedPhoto:
    meta = PhotoMeta(path=path, size=1, mtime_ns=1, taken=taken, date_precision=precision)
    place = PlaceLabel(city=city, region="Florida", country="United States") if city else None
    return IndexedPhoto(id=pid, meta=meta, place=place)


def make_response(
    hits=(),
    *,
    applied: Filters | None = None,
    needs=(),
    dropped=(),
    text: str = "beach",
    notes=(),
    candidates: int | None = None,
) -> SearchResponse:
    filters = applied or Filters()
    parsed = ParsedQuery(raw=text, text=text)
    return SearchResponse(
        hits=tuple(hits),
        parse=ParseResult(parsed=parsed, filters=filters, notes=tuple(notes)),
        applied=filters,
        candidates=len(hits) if candidates is None else candidates,
        needs_clarification=tuple(needs),
        dropped=tuple(dropped),
    )


def unresolved(phrase: str = "Easter 2024", kind: str = "date") -> Unresolved:
    return Unresolved(kind, phrase, "no", f'I did not understand "{phrase}" as a {kind}.')


class FakeStore:
    def __init__(self, names=(), meta=None):
        self.names = list(names)
        self.meta = dict(meta or {})
        self.closed = False

    def known_names(self):
        return list(self.names)

    def get_meta(self, key, default=None):
        return self.meta.get(key, default)

    def close(self):
        self.closed = True


class FakeGazetteer:
    def __init__(self):
        self.calls = []

    def resolve(self, name):
        self.calls.append(name)
        if name.lower() == "key west":
            return PlaceFilter(city="Key West", region="Florida", country="United States")
        return None


class FakeEngine:
    """A ``SearchEngine`` returning canned responses (one per call, the last one repeats)."""

    def __init__(self, responses, *, names=(), meta=None, gazetteer="default"):
        self.responses = list(responses)
        self.requests = []
        self.store = FakeStore(names, meta)
        self.gazetteer = FakeGazetteer() if gazetteer == "default" else gazetteer
        if self.gazetteer is None:
            del self.gazetteer
        self.closed = False

    def search(self, request):
        self.requests.append(request)
        index = min(len(self.requests), len(self.responses)) - 1
        return self.responses[index]

    def close(self):
        self.closed = True


class Factory:
    """Records how the engine was opened; returns ``engine``."""

    def __init__(self, engine):
        self.engine = engine
        self.calls = []

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self.engine


def run_search(argv, engine, **kwargs):
    factory = Factory(engine)
    args = parse("search", *argv)
    kwargs.setdefault("is_interactive", lambda: False)
    code = search_cli.main_search(args, engine_factory=factory, **kwargs)
    return code, factory


# --------------------------------------------------------------------------- parser / hook


@pytest.mark.parametrize(
    "argv",
    [
        ["index", "--help"],
        ["search", "--help"],
        ["serve", "--help"],
        ["people", "--help"],
        ["people", "scan", "--help"],
        ["people", "cluster", "--help"],
        ["people", "list", "--help"],
        ["people", "name", "--help"],
        ["stats", "--help"],
        ["eval", "--help"],
    ],
)
def test_help_for_every_subcommand_needs_no_search_dependency(argv):
    code = subprocess.run(
        [sys.executable, "-m", "facesorter", *argv],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONDONTWRITEBYTECODE="1"),
        timeout=60,
    )
    assert code.returncode == 0, code.stderr
    assert "usage: facesorter" in code.stdout


def run_isolated(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )


def test_building_the_parser_imports_nothing_heavy():
    code = (
        "import sys, facesorter.cli\n"
        "facesorter.cli.build_parser()\n"
        "heavy = ['numpy', 'PIL', 'torch', 'sklearn', 'insightface', 'onnxruntime', 'scipy',\n"
        "         'tqdm', 'gradio', 'dateparser', 'geonamescache', 'tokenizers',\n"
        "         'huggingface_hub', 'gradio_client', 'pillow_heif', 'sqlite3',\n"
        "         'facesorter.search.engine', 'facesorter.search.store',\n"
        "         'facesorter.search.indexer', 'facesorter.search.people', 'facesorter.utils']\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "print(','.join(loaded))\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    result = run_isolated(code)
    assert result.returncode == 0, f"unexpectedly imported: {result.stdout.strip()}"


BLOCKER = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {
            'numpy', 'PIL', 'scipy', 'pillow_heif', 'tqdm', 'torch', 'tokenizers', 'dateparser',
            'geonamescache', 'onnxruntime', 'huggingface_hub',
        }:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, Block())
from facesorter.cli import main
"""


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "a", "beach"],
        ["index", "."],
        ["stats"],
        ["people", "list"],
        ["serve"],
    ],
)
def test_commands_without_search_dependencies_exit_2_with_install_hint(argv, tmp_path):
    # the index file must exist for `serve` to get as far as loading the engine
    (tmp_path / "idx").write_text("x")
    code = BLOCKER + f"sys.exit(main({argv + ['--index', str(tmp_path / 'idx')]!r}))\n"
    result = run_isolated(code)
    assert result.returncode == 2, result.stderr
    if sys.version_info >= (3, 10):
        assert 'pip install "facesorter[search]"' in result.stderr
    else:  # nothing to pip install there: say what is needed instead
        assert "Python 3.10 or newer" in result.stderr and "pip install" not in result.stderr
    assert "Traceback" not in result.stderr


def test_help_still_works_without_search_dependencies():
    result = run_isolated(BLOCKER + "sys.exit(main(['search', '--help']))\n")
    assert result.returncode == 0 and "usage: facesorter search" in result.stdout


def test_missing_message_on_old_python(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 9, 18, "final", 0))
    text = search_cli._missing_message(ModuleNotFoundError("x", name="numpy"))
    assert "Python 3.10 or newer" in text and "this is Python 3.9" in text
    assert "pip install" not in text  # installing the extra cannot help on 3.9


def test_missing_message_for_face_model():
    text = search_cli._missing_message(ModuleNotFoundError("x", name="insightface.app"))
    assert "insightface" in text and "pip install insightface onnxruntime" in text


def test_missing_message_without_module_name(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 4, "final", 0))
    text = search_cli._missing_message(ImportError("DLL load failed"))
    assert "optional package" in text and 'pip install "facesorter[search]"' in text


def test_missing_package_on_a_new_python_has_no_version_complaint(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 4, "final", 0))
    text = search_cli._missing_message(ModuleNotFoundError("x", name="open_clip"))
    assert "'open_clip'" in text and "3.10" not in text


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "--json", "--paths-only", "x"],
        ["search", "-k", "0", "x"],
        ["search", "-k", "abc", "x"],
        ["index", ".", "--workers", "0"],
        ["index", ".", "--device", "tpu"],
        ["serve", "--port", "70000"],
        ["serve", "--port", "http"],
        ["people"],
        ["people", "name"],
        ["people", "name", "x", "Bob"],
        ["people", "scan", "--confidence", "2"],
        ["people", "scan", "--confidence", "high"],
        ["search", "--llm", "gpt"],
    ],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_verbose_before_and_after_subcommand():
    assert parse("-v", "stats").verbose is True
    assert parse("stats", "-v").verbose is True
    assert parse("stats").verbose is False
    assert parse("-v", "eval", "-x").verbose is True
    assert parse("eval", "-x").verbose is False


def test_existing_commands_still_registered():
    assert parse("sort", "--source", "a", "--output", "b").command == "sort"
    assert parse("find", "--source", "a", "--reference", "r", "--output", "b").command == "find"


def test_search_commands_skip_the_sort_path_checks(index_env, capsys):
    # `sort` would demand --source/--output directories; search commands never see that code.
    assert main(["stats"]) == 2
    assert "facesorter index" in capsys.readouterr().err


# --------------------------------------------------------------------------- date flags


@pytest.mark.parametrize(
    ("text", "end", "expected"),
    [
        ("2024", False, date(2024, 1, 1)),
        ("2024", True, date(2024, 12, 31)),
        ("2024-02", False, date(2024, 2, 1)),
        ("2024-02", True, date(2024, 2, 29)),  # leap year
        ("2023-02", True, date(2023, 2, 28)),
        ("1900-02", True, date(1900, 2, 28)),  # century, not a leap year
        ("2024-3", True, date(2024, 3, 31)),
        ("2024-12", True, date(2024, 12, 31)),
        ("2024-03-09", False, date(2024, 3, 9)),
        ("2024-03-09", True, date(2024, 3, 9)),
        (" 2024-06 ", False, date(2024, 6, 1)),
    ],
)
def test_parse_date_flag(text, end, expected):
    assert search_cli.parse_date_flag(text, end=end) == expected


@pytest.mark.parametrize(
    "text",
    ["", "24", "2024-13", "2024-00", "2024-02-30", "0000", "2024/03", "last year", "2024-1-2-3"],
)
def test_parse_date_flag_rejects(text):
    with pytest.raises(search_cli.UsageError, match="--from"):
        search_cli.parse_date_flag(text, end=False, flag="--from")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--from", "2024", "--to", "2024"], DateRange(date(2024, 1, 1), date(2024, 12, 31))),
        (["--from", "2024-02", "--to", "2024-02"], DateRange(date(2024, 2, 1), date(2024, 2, 29))),
        (["--from", "2023-02", "--to", "2023-02"], DateRange(date(2023, 2, 1), date(2023, 2, 28))),
        (["--from", "2020-05-03", "--to", "2021"], DateRange(date(2020, 5, 3), date(2021, 12, 31))),
        (["--from", "2022-06"], DateRange(date(2022, 6, 1), date.max)),
        (["--to", "2022-06"], DateRange(date.min, date(2022, 6, 30))),
    ],
)
def test_date_flags_become_a_date_range(argv, expected):
    engine = FakeEngine([make_response()])
    code, _ = run_search(argv, engine)
    assert code == 0
    assert engine.requests[0].filters == Filters(date_range=expected)


def test_from_after_to_exits_2(capsys):
    engine = FakeEngine([make_response()])
    code, factory = run_search(["--from", "2024-05", "--to", "2024-04", "x"], engine)
    assert code == 2
    assert "is after --to" in capsys.readouterr().err
    assert factory.calls == []


def test_bad_date_exits_2_before_loading_anything(capsys):
    engine = FakeEngine([make_response()])
    code, factory = run_search(["--from", "yesterday", "x"], engine)
    assert code == 2
    assert "not a date" in capsys.readouterr().err
    assert factory.calls == []


# --------------------------------------------------------------------------- search output


def hits_fixture():
    return [
        Hit(make_photo(1, "/lib/a.jpg"), 0.31234),
        Hit(make_photo(2, "/lib/b é.jpg", taken=None, city=None), 0.2),
        Hit(make_photo(3, "/lib/c.jpg", taken=datetime(2019, 7, 1), precision="month"), 0.1),
        Hit(make_photo(4, "/lib/d.jpg", taken=datetime(2018, 1, 1), precision="year"), 0.05),
    ]


def test_table_output(capsys):
    applied = Filters(date_range=DateRange(date(2023, 6, 1), date(2023, 8, 31)), people=("Bob",))
    engine = FakeEngine([make_response(hits_fixture(), applied=applied)])
    code, factory = run_search(["bob", "beach", "-k", "4"], engine)
    out, err = capsys.readouterr()
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "Searching: dates 2023-06-01 to 2023-08-31 · with Bob"
    assert lines[1].split() == ["#", "score", "date", "place", "path"]
    assert lines[2].split()[:3] == ["1", "0.312", "2023-07-04"]
    assert "Key West, Florida, United States" in lines[2] and lines[2].endswith("/lib/a.jpg")
    assert lines[3].split()[2:4] == ["-", "-"] and lines[3].endswith("/lib/b é.jpg")
    assert lines[4].split()[2] == "2019-07" and lines[5].split()[2] == "2018"
    assert err == ""
    path, kwargs = factory.calls[0]
    assert path == str(os.environ["FACESORTER_INDEX"])
    assert kwargs == {"llm": "auto", "llm_model": None, "use_llm": True}
    assert engine.requests[0].query == "bob beach" and engine.requests[0].k == 4
    assert engine.closed and engine.store is not None


def test_columns_are_aligned(capsys):
    engine = FakeEngine([make_response(hits_fixture())])
    run_search(["x"], engine)
    rows = capsys.readouterr().out.splitlines()[1:]
    assert {row.index("/lib/") for row in rows[1:]} == {rows[0].index("path")}
    assert {row.index(" ", 2) for row in rows[1:]} == {rows[0].index(" ", 2)}  # score column


def test_searching_line_with_no_filters(capsys):
    engine = FakeEngine([make_response([Hit(make_photo(), 0.5)])])
    run_search(["beach"], engine)
    assert capsys.readouterr().out.splitlines()[0] == "Searching: (no filters)"


def test_zero_results_exit_0(capsys):
    engine = FakeEngine([make_response([], applied=Filters(people=("Bob",)))])
    code, _ = run_search(["x"], engine)
    out, err = capsys.readouterr()
    assert code == 0
    assert "No matching photos." in err
    assert "No photos match: with Bob" in err  # explain() notice
    assert out.splitlines() == ["Searching: with Bob"]


def test_notices_from_explain_go_to_stderr(capsys):
    response = make_response(
        [Hit(make_photo(), 0.4)], notes=["note one"], dropped=[unresolved("Easter 2024")]
    )
    run_search(["x"], FakeEngine([response]))
    out, err = capsys.readouterr()
    assert "note one" in err and "Easter 2024" in err and "searched without it" in err
    assert "note one" not in out


def test_json_lines(capsys):
    engine = FakeEngine([make_response(hits_fixture()[:2])])
    code, _ = run_search(["--json", "x"], engine)
    out, err = capsys.readouterr()
    assert code == 0
    records = [json.loads(line) for line in out.splitlines()]
    assert records[0] == {
        "rank": 1,
        "score": 0.3123,
        "id": 1,
        "path": "/lib/a.jpg",
        "date": "2023-07-04T12:00:00",
        "date_precision": "day",
        "place": "Key West, Florida, United States",
    }
    assert records[1]["date"] is None and records[1]["place"] is None
    assert records[1]["path"] == "/lib/b é.jpg"  # not escaped
    assert err.startswith("Searching:")  # stdout stays machine readable


def test_json_with_no_results_prints_nothing_on_stdout(capsys):
    run_search(["--json", "x"], FakeEngine([make_response([])]))
    out, err = capsys.readouterr()
    assert out == "" and "No matching photos." in err


def test_paths_only(capsys):
    code, _ = run_search(["--paths-only", "x"], FakeEngine([make_response(hits_fixture()[:2])]))
    out, err = capsys.readouterr()
    assert code == 0
    assert out.splitlines() == ["/lib/a.jpg", "/lib/b é.jpg"]
    assert err.startswith("Searching:")


# --------------------------------------------------------------------------- search flags


def test_no_query_and_no_flags_exits_2(capsys):
    engine = FakeEngine([make_response()])
    code, factory = run_search([], engine)
    assert code == 2
    assert "Give a query" in capsys.readouterr().err
    assert factory.calls == []


def test_flags_alone_are_enough():
    engine = FakeEngine([make_response()], names=["Bob"])
    code, _ = run_search(["--person", "Bob"], engine)
    assert code == 0
    assert engine.requests[0].query == "" and engine.requests[0].filters.people == ("Bob",)


def test_place_resolved_with_the_engines_gazetteer():
    engine = FakeEngine([make_response()])
    code, _ = run_search(["--place", "Key West", "x"], engine)
    assert code == 0
    assert engine.requests[0].filters.place.city == "Key West"
    assert engine.gazetteer.calls == ["Key West"]


def test_place_falls_back_to_a_gazetteer_factory():
    engine = FakeEngine([make_response()], gazetteer=None)
    gaz = FakeGazetteer()
    code, _ = run_search(["--place", "key west"], engine, gazetteer_factory=lambda: gaz)
    assert code == 0 and gaz.calls == ["key west"]
    assert engine.requests[0].filters.place.city == "Key West"


def test_unknown_place_exits_2_and_does_not_search(capsys):
    engine = FakeEngine([make_response()])
    code, _ = run_search(["--place", "Atlantis", "x"], engine)
    err = capsys.readouterr().err
    assert code == 2
    assert "'Atlantis' is not a place I know" in err
    assert engine.requests == [] and engine.closed


def test_people_resolved_case_and_accent_insensitively():
    engine = FakeEngine([make_response()], names=["Zoë", "Bob Smith"])
    code, _ = run_search(["--person", "zoe", "--person", "bob  smith", "--person", "ZOË"], engine)
    assert code == 0
    assert engine.requests[0].filters.people == ("Zoë", "Bob Smith")


def test_unknown_person_lists_the_known_names(capsys):
    engine = FakeEngine([make_response()], names=["Alice", "Bob"])
    code, _ = run_search(["--person", "Carol", "x"], engine)
    err = capsys.readouterr().err
    assert code == 2
    assert "'Carol' is not a known person. Known: Alice, Bob" in err


def test_unknown_person_when_nobody_is_named(capsys):
    engine = FakeEngine([make_response()])
    code, _ = run_search(["--person", "Carol", "x"], engine)
    err = capsys.readouterr().err
    assert code == 2 and "no one has been named yet" in err and "people name" in err


def test_all_flags_combine_into_one_filters_object():
    engine = FakeEngine([make_response()], names=["Bob"])
    run_search(
        ["--from", "2023-06", "--to", "2023-08", "--place", "Key West", "--person", "Bob"], engine
    )
    f = engine.requests[0].filters
    assert f.date_range == DateRange(date(2023, 6, 1), date(2023, 8, 31))
    assert f.place.city == "Key West" and f.people == ("Bob",)


def test_no_flags_means_no_explicit_filters():
    engine = FakeEngine([make_response()])
    run_search(["beach"], engine)
    assert engine.requests[0].filters is None


@pytest.mark.parametrize(
    ("flags", "llm", "use_llm"),
    [
        ([], "auto", True),
        (["--no-llm"], "none", False),
        (["--llm", "none"], "none", False),
        (["--llm", "mlx", "--llm-model", "m"], "mlx", True),
        (["--no-llm", "--llm", "mlx"], "none", False),
    ],
)
def test_llm_flags(flags, llm, use_llm):
    engine = FakeEngine([make_response()])
    _, factory = run_search([*flags, "x"], engine)
    assert factory.calls[0][1]["llm"] == llm and factory.calls[0][1]["use_llm"] is use_llm
    assert engine.requests[0].use_llm is use_llm
    if "--llm-model" in flags:
        assert factory.calls[0][1]["llm_model"] == "m"


def test_index_flag_is_used():
    _, factory = run_search(["--index", "/x/my.sqlite", "q"], FakeEngine([make_response()]))
    assert factory.calls[0][0] == "/x/my.sqlite"


# --------------------------------------------------------------------------- clarification


def clarifying_engine(*, second=None):
    first = make_response(needs=[unresolved("Easter 2024"), unresolved("Atlantis", "place")])
    final = second or make_response(
        [Hit(make_photo(), 0.4)], dropped=[unresolved("Atlantis", "place")]
    )
    return FakeEngine([first, final])


def test_interactive_clarification_asks_once_and_searches_again(capsys):
    engine = clarifying_engine()
    answers = iter(["April 2024", ""])
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return next(answers)

    code, _ = run_search(["x"], engine, is_interactive=lambda: True, input_fn=fake_input)
    out, err = capsys.readouterr()
    assert code == 0
    assert len(engine.requests) == 2
    assert engine.requests[0].allow_clarification is True
    assert engine.requests[1].clarifications == {"Easter 2024": "April 2024", "Atlantis": ""}
    assert 'did not understand "Easter 2024"' in err and 'did not understand "Atlantis"' in err
    assert "press Enter" in err
    assert len(prompts) == 2
    assert "Searching:" in out and "/lib/a.jpg" in out
    assert "Atlantis" in err and "searched without it" in err  # dropped, reported by explain()


def test_clarification_uses_builtin_input_and_strips(monkeypatch, capsys):
    engine = clarifying_engine()
    replies = iter(["  spring 2024  ", "   "])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    run_search(["x"], engine, is_interactive=lambda: True)
    assert engine.requests[1].clarifications == {"Easter 2024": "spring 2024", "Atlantis": ""}


def test_eof_during_clarification_skips_the_rest(capsys):
    engine = clarifying_engine()
    calls = []

    def fake_input(prompt):
        calls.append(prompt)
        raise EOFError

    code, _ = run_search(["x"], engine, is_interactive=lambda: True, input_fn=fake_input)
    assert code == 0 and len(calls) == 1  # asked once, EOF skips the remaining item
    assert engine.requests[1].clarifications == {"Easter 2024": "", "Atlantis": ""}


def test_no_ask_never_prompts():
    engine = FakeEngine([make_response([Hit(make_photo(), 0.4)], dropped=[unresolved()])])

    def boom(prompt):
        raise AssertionError("must not prompt")

    code, _ = run_search(["--no-ask", "x"], engine, is_interactive=lambda: True, input_fn=boom)
    assert code == 0
    assert len(engine.requests) == 1 and engine.requests[0].allow_clarification is False


def test_non_interactive_never_prompts():
    engine = FakeEngine([make_response([], dropped=[unresolved()])])
    code, _ = run_search(["x"], engine, is_interactive=lambda: False)
    assert code == 0 and engine.requests[0].allow_clarification is False


def test_default_interactivity_follows_stdin(monkeypatch):
    class Tty:
        def isatty(self):
            return True

    class Closed:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stdin", Tty())
    assert search_cli._stdin_is_tty() is True
    monkeypatch.setattr(sys, "stdin", Closed())
    assert search_cli._stdin_is_tty() is False
    monkeypatch.setattr(sys, "stdin", None)
    assert search_cli._stdin_is_tty() is False


def test_default_interactivity_used_by_main_search(monkeypatch):
    monkeypatch.setattr(search_cli, "_stdin_is_tty", lambda: True)
    engine = FakeEngine([make_response()])
    factory = Factory(engine)
    search_cli.main_search(parse("search", "x"), engine_factory=factory)
    assert engine.requests[0].allow_clarification is True


# --------------------------------------------------------------------------- copy-to


def real_photos(tmp_path: Path, names=("a.jpg", "b.jpg")) -> list[Hit]:
    src = tmp_path / "lib"
    src.mkdir(exist_ok=True)
    hits = []
    for i, name in enumerate(names, 1):
        (src / name).write_bytes(f"photo {name}".encode())
        hits.append(Hit(make_photo(i, str(src / name)), 0.5))
    return hits


def test_copy_to_copies_and_reports(tmp_path, capsys):
    hits = real_photos(tmp_path)
    dest = tmp_path / "out"
    engine = FakeEngine([make_response(hits)])
    code, _ = run_search(["--copy-to", str(dest), "x"], engine)
    out = capsys.readouterr().out
    assert code == 0
    assert sorted(os.listdir(dest)) == ["a.jpg", "b.jpg"]
    assert (dest / "a.jpg").read_bytes() == b"photo a.jpg"
    assert f"Copied 2 photo(s) to {dest}" in out


def test_copy_to_never_overwrites(tmp_path):
    hits = real_photos(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "a.jpg").write_bytes(b"precious")
    run_search(["--copy-to", str(dest), "x"], FakeEngine([make_response(hits)]))
    assert (dest / "a.jpg").read_bytes() == b"precious"
    assert (dest / "a_1.jpg").read_bytes() == b"photo a.jpg"
    run_search(["--copy-to", str(dest), "x"], FakeEngine([make_response(hits)]))
    assert sorted(os.listdir(dest)) == ["a.jpg", "a_1.jpg", "a_2.jpg", "b.jpg", "b_1.jpg"]


def test_copy_to_message_goes_to_stderr_in_machine_modes(tmp_path, capsys):
    hits = real_photos(tmp_path)
    run_search(
        ["--paths-only", "--copy-to", str(tmp_path / "o"), "x"], FakeEngine([make_response(hits)])
    )
    out, err = capsys.readouterr()
    assert "Copied" not in out and "Copied 2 photo(s)" in err


def test_copy_to_with_no_hits_creates_nothing(tmp_path, capsys):
    dest = tmp_path / "out"
    code, _ = run_search(["--copy-to", str(dest), "x"], FakeEngine([make_response([])]))
    assert code == 0 and not dest.exists()
    assert "Copied 0 photo(s)" in capsys.readouterr().out


def test_copy_to_a_file_exits_2_before_loading(tmp_path, capsys):
    target = tmp_path / "file.txt"
    target.write_text("x")
    code, factory = run_search(["--copy-to", str(target), "x"], FakeEngine([make_response()]))
    assert code == 2 and factory.calls == []
    assert "not a folder" in capsys.readouterr().err


def test_copy_partial_failure_warns_and_verbose_lists(tmp_path, capsys):
    hits = real_photos(tmp_path, ("a.jpg",)) + [Hit(make_photo(9, str(tmp_path / "gone.jpg")), 0.1)]
    dest = tmp_path / "out"
    code, _ = run_search(["--copy-to", str(dest), "x"], FakeEngine([make_response(hits)]))
    err = capsys.readouterr().err
    assert code == 0 and os.listdir(dest) == ["a.jpg"]
    assert "1 photo(s) could not be copied" in err and "gone.jpg" not in err  # private by default
    code, _ = run_search(["-v", "--copy-to", str(dest), "x"], FakeEngine([make_response(hits)]))
    assert code == 0
    # args built by parse() carry -v only via the parser, so run through main_search directly
    args = parse("-v", "search", "--copy-to", str(dest), "x")
    factory = Factory(FakeEngine([make_response(hits)]))
    search_cli.main_search(args, engine_factory=factory, is_interactive=lambda: False)
    assert "gone.jpg" in capsys.readouterr().err


def test_copy_total_failure_exits_1(tmp_path, capsys):
    hits = [Hit(make_photo(1, str(tmp_path / "missing.jpg")), 0.1)]
    code, _ = run_search(["--copy-to", str(tmp_path / "o"), "x"], FakeEngine([make_response(hits)]))
    assert code == 1
    err = capsys.readouterr().err
    assert "Could not copy any of the 1 photos" in err
    assert "No such file or directory" in err  # the first underlying reason
    assert "missing.jpg" not in err  # per-file detail (a path) only with -v


def test_copy_total_failure_lists_every_reason_with_verbose(tmp_path, capsys):
    hits = [Hit(make_photo(i, str(tmp_path / f"m{i}.jpg")), 0.1) for i in (1, 2)]
    args = parse("-v", "search", "--copy-to", str(tmp_path / "o"), "x")
    code = search_cli.main_search(
        args,
        engine_factory=Factory(FakeEngine([make_response(hits)])),
        is_interactive=lambda: False,
    )
    err = capsys.readouterr().err
    assert code == 1 and "m1.jpg" in err and "m2.jpg" in err


def test_copy_to_help_mentions_rerun_suffixes(capsys):
    with pytest.raises(SystemExit):
        main(["search", "--help"])
    assert "_1, _2" in " ".join(capsys.readouterr().out.split())


# --------------------------------------------------------------------------- errors / exit codes


class Boom:
    def __init__(self, exc):
        self.exc = exc

    def __call__(self, *a, **k):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (IndexNotFoundError("No photo index at x. Run `facesorter index` first."), 2),
        (ModelUnavailableError("no model"), 2),
        (InvalidPathError("bad"), 2),
        (IndexMismatchError("different model"), 1),
        (SearchError("broken"), 1),
        (FaceSorterError("generic"), 1),
        (RuntimeError("kaboom"), 1),
        (KeyboardInterrupt(), 130),
        (ModuleNotFoundError("x", name="open_clip"), 2),
    ],
)
def test_search_error_exit_codes(exc, code, capsys):
    result = search_cli.main_search(parse("search", "x"), engine_factory=Boom(exc))
    err = capsys.readouterr().err
    assert result == code
    assert "Traceback" not in err
    if code != 130:
        assert err.startswith("error: ")


def test_unexpected_error_is_one_line_without_verbose(capsys):
    search_cli.main_search(parse("search", "x"), engine_factory=Boom(RuntimeError("kaboom")))
    err = capsys.readouterr().err
    assert err.strip() == "error: Unexpected RuntimeError: kaboom"


def test_unexpected_error_traceback_with_verbose(capsys):
    args = parse("-v", "search", "x")
    assert search_cli.main_search(args, engine_factory=Boom(RuntimeError("kaboom"))) == 1
    assert "Traceback" in capsys.readouterr().err


def test_missing_dependency_traceback_with_verbose(capsys, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 4, "final", 0))
    args = parse("search", "-v", "x")
    exc = ModuleNotFoundError("x", name="open_clip")
    assert search_cli.main_search(args, engine_factory=Boom(exc)) == 2
    err = capsys.readouterr().err
    assert "Traceback" in err and "pip install" in err


def test_keyboard_interrupt_message(capsys):
    search_cli.main_search(parse("search", "x"), engine_factory=Boom(KeyboardInterrupt()))
    assert "Interrupted." in capsys.readouterr().err


def test_broken_pipe_is_quiet(monkeypatch, capsys):
    real = sys.stdout
    monkeypatch.setattr(sys, "stdout", real)  # restored by monkeypatch afterwards
    code = search_cli.main_search(parse("search", "x"), engine_factory=Boom(BrokenPipeError()))
    sys.stdout.close()
    assert code == 0
    monkeypatch.setattr(sys, "stdout", real)


def test_engine_is_closed_even_when_search_fails():
    class Failing(FakeEngine):
        def search(self, request):
            raise RuntimeError("model crashed")

    engine = Failing([])
    code, _ = run_search(["x"], engine)
    assert code == 1 and engine.closed


def test_logger_is_left_pristine(capsys):
    logger = logging.getLogger("facesorter")
    before = list(logger.handlers)
    for flags in ([], ["-v"]):
        run_search([*flags, "x"], FakeEngine([make_response()]))
        assert logger.level == logging.NOTSET and logger.propagate is True
        assert logger.handlers == before


def test_verbose_logs_info_to_stderr(capsys):
    class Chatty(FakeEngine):
        def search(self, request):
            logging.getLogger("facesorter.search.engine").info("hello from the engine")
            logging.getLogger("facesorter.search.engine").debug("too detailed")
            return super().search(request)

    args = parse("-v", "search", "x")
    search_cli.main_search(args, engine_factory=Factory(Chatty([make_response()])))
    err = capsys.readouterr().err
    assert "INFO: hello from the engine" in err and "too detailed" not in err
    search_cli.main_search(parse("search", "x"), engine_factory=Factory(Chatty([make_response()])))
    assert "hello from the engine" not in capsys.readouterr().err


# --------------------------------------------------------------------------- main() end to end


def test_main_search_uses_open_engine_from_the_engine_module(monkeypatch, capsys):
    engine = FakeEngine([make_response([Hit(make_photo(), 0.9)])])
    factory = Factory(engine)
    monkeypatch.setattr("facesorter.search.engine.open_engine", factory)
    assert main(["search", "--no-ask", "--paths-only", "a", "red", "car"]) == 0
    assert capsys.readouterr().out == "/lib/a.jpg\n"
    assert factory.calls[0][0] == os.environ["FACESORTER_INDEX"]  # default index path
    assert engine.requests[0].query == "a red car"


def test_main_search_missing_index_exits_2(capsys):
    assert main(["search", "x"]) == 2
    err = capsys.readouterr().err
    assert "facesorter index" in err and "Traceback" not in err


# --------------------------------------------------------------------------- index


class FakeIndexer:
    def __init__(self, report=None, exc=None, steps=()):
        self.report = report or IndexReport(10, 4, 1, 5, 2, (), 3.25, "fake-model")
        self.exc = exc
        self.steps = steps
        self.run_calls = []
        self.closed = False
        self.store = FakeStore(meta={"library_roots": '["/lib"]'})
        self.removed = {}

    def remove_root(self, root):
        count = self.removed.get(root, 0)
        if root in self.removed:
            self.store.meta["library_roots"] = "[]"
        return count

    def run(self, root, **kwargs):
        self.run_calls.append((root, kwargs))
        for step in self.steps:
            kwargs["progress"](*step)
        if self.exc:
            raise self.exc
        return self.report

    def close(self):
        self.closed = True


class IndexFactory:
    def __init__(self, indexer):
        self.indexer = indexer
        self.calls = []

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self.indexer


class FakeBar:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.total = kwargs.get("total")
        self.n = 0
        self.closed = False
        FakeBar.instances.append(self)

    def update(self, n):
        self.n += n

    def close(self):
        self.closed = True


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / "Library"
    lib.mkdir()
    return lib


def run_index(argv, indexer, **kwargs):
    FakeBar.instances = []
    factory = IndexFactory(indexer)
    code = search_cli.main_index(
        parse("index", *argv), indexer_factory=factory, progress_factory=FakeBar, **kwargs
    )
    return code, factory


def test_index_summary_and_defaults(library, capsys, index_env):
    indexer = FakeIndexer()
    code, factory = run_index([str(library)], indexer)
    out, err = capsys.readouterr()
    assert code == 0 and err == ""
    rows = [line.split(None, 1) for line in out.splitlines()[1:]]
    assert rows == [
        ["scanned", "10"],
        ["new", "4"],
        ["updated", "1"],
        ["unchanged", "5"],
        ["removed", "2"],
        ["failed", "0"],
        ["time", "3.2 s (model: fake-model)"],
    ]
    assert f"Index: {index_env}" in out
    assert factory.calls == [(str(index_env), {"device": "auto"})]  # G's defaults are kept
    root, kwargs = indexer.run_calls[0]
    assert root == str(library)
    assert kwargs["exclude_dirs"] == () and kwargs["force"] is False and kwargs["limit"] is None
    assert kwargs["allow_empty_prune"] is False
    assert indexer.closed


def test_index_passes_every_flag(library, tmp_path):
    indexer = FakeIndexer()
    argv = [
        str(library),
        *["--index", str(tmp_path / "i.sqlite"), "--workers", "3", "--batch-size", "8"],
        *["--limit", "5", "--force", "--exclude", "skip me", "--exclude", str(tmp_path / "x")],
        *["--device", "cpu", "--allow-empty-prune"],
    ]
    code, factory = run_index(argv, indexer)
    assert code == 0
    assert factory.calls == [
        (str(tmp_path / "i.sqlite"), {"device": "cpu", "workers": 3, "batch_size": 8})
    ]
    _, kwargs = indexer.run_calls[0]
    assert kwargs["exclude_dirs"] == (os.path.abspath("skip me"), str(tmp_path / "x"))
    assert kwargs["force"] is True and kwargs["limit"] == 5 and kwargs["allow_empty_prune"] is True


def test_index_relative_library_is_made_absolute(library, monkeypatch):
    monkeypatch.chdir(library.parent)
    indexer = FakeIndexer()
    run_index(["Library"], indexer)
    assert indexer.run_calls[0][0] == str(library)


def test_index_missing_library_exits_2(tmp_path, capsys):
    factory = IndexFactory(FakeIndexer())
    code = search_cli.main_index(parse("index", str(tmp_path / "nope")), indexer_factory=factory)
    assert code == 2 and factory.calls == []
    assert "library folder does not exist" in capsys.readouterr().err


def test_index_failures_are_counted_and_listed_only_with_verbose(library, capsys):
    report = IndexReport(
        5, 3, 0, 0, 0, (("/lib/bad.heic", "corrupt"), ("/lib/x.jpg", "truncated")), 1.0, "m"
    )
    code, _ = run_index([str(library)], FakeIndexer(report))
    out, err = capsys.readouterr()
    assert code == 0
    assert ["failed", "2"] in [line.split() for line in out.splitlines()]
    assert "warning: 2 photo(s) could not be indexed (run with -v to list them)" in err
    assert "bad.heic" not in err

    args = parse("-v", "index", str(library))
    search_cli.main_index(
        args, indexer_factory=IndexFactory(FakeIndexer(report)), progress_factory=FakeBar
    )
    err = capsys.readouterr().err
    assert "/lib/bad.heic: corrupt" in err and "/lib/x.jpg: truncated" in err
    assert "could not be indexed\n" in err


def test_index_progress_bars_follow_the_stages(library):
    steps = [
        (500, 0, "scan"),
        (900, 900, "scan"),
        (3, 3, "metadata"),
        (2, 3, "encode"),
        (3, 3, "encode"),
        (2, 3, "store"),  # ignored
    ]
    code, _ = run_index([str(library)], FakeIndexer(steps=steps))
    assert code == 0
    bars = FakeBar.instances
    assert [b.kwargs["desc"] for b in bars] == ["Scanning", "Reading metadata", "Embedding"]
    assert bars[0].kwargs["total"] is None and bars[0].total == 900 and bars[0].n == 900
    assert bars[2].n == 3 and bars[2].total == 3
    assert all(b.closed and b.kwargs["file"] is sys.stderr for b in bars)


def test_progress_without_stage_uses_its_own_description():
    FakeBar.instances = []
    progress = search_cli._Progress(FakeBar, desc="Scanning faces")
    progress(1, 4)
    progress(4, 4)
    progress.close()
    progress.close()
    assert FakeBar.instances[0].kwargs["desc"] == "Scanning faces" and FakeBar.instances[0].n == 4


def test_progress_defaults_to_tqdm_on_stderr(library, capsys):
    steps = [(1, 2, "metadata"), (2, 2, "metadata")]
    code = search_cli.main_index(
        parse("index", str(library)), indexer_factory=IndexFactory(FakeIndexer(steps=steps))
    )
    assert code == 0
    assert "Reading metadata" in capsys.readouterr().err


def test_index_bar_is_closed_on_failure(library):
    code, _ = run_index(
        [str(library)], FakeIndexer(exc=SearchError("boom"), steps=[(1, 5, "encode")])
    )
    assert code == 1
    assert FakeBar.instances[0].closed


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (SearchError("No photos were found under x"), 1),
        (ModelUnavailableError("model missing"), 2),
        (IndexMismatchError("other model"), 1),
        (KeyboardInterrupt(), 130),
        (RuntimeError("x"), 1),
    ],
)
def test_index_error_exit_codes(library, exc, code, capsys):
    result, factory = run_index([str(library)], FakeIndexer(exc=exc))
    assert result == code
    assert factory.indexer.closed


def test_index_uses_open_indexer_from_the_indexer_module(library, monkeypatch, capsys):
    factory = IndexFactory(FakeIndexer())
    monkeypatch.setattr("facesorter.search.indexer.open_indexer", factory)
    assert main(["index", str(library)]) == 0
    assert factory.calls[0][0] == os.environ["FACESORTER_INDEX"]
    assert "scanned" in capsys.readouterr().out


def test_index_open_failure_exit_code(library):
    code = search_cli.main_index(
        parse("index", str(library)), indexer_factory=Boom(ModelUnavailableError("no torch"))
    )
    assert code == 2


# --------------------------------------------------------------------------- serve


def serve_args(*argv):
    return parse("serve", *argv)


class ServeRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, engine, **kwargs):
        self.calls.append((engine, kwargs))


@pytest.fixture
def index_file(tmp_path):
    path = tmp_path / "index.sqlite"
    path.write_text("not read")
    return path


def run_serve(argv, engine, index_file, **kwargs):
    recorder = ServeRecorder()
    factory = Factory(engine)
    code = search_cli.main_serve(
        serve_args("--index", str(index_file), *argv),
        engine_factory=factory,
        serve_fn=recorder,
        **kwargs,
    )
    return code, recorder, factory


def test_serve_defaults(index_file, capsys):
    engine = FakeEngine([], meta={"library_roots": json.dumps(["/lib/a", "/lib/b"])})
    code, recorder, factory = run_serve([], engine, index_file)
    assert code == 0
    assert recorder.calls == [
        (
            engine,
            {
                "library_roots": ["/lib/a", "/lib/b"],
                "host": "127.0.0.1",
                "port": 7860,
                "open_browser": False,
            },
        )
    ]
    assert factory.calls[0][1] == {"llm": "auto", "llm_model": None, "use_llm": True}
    assert "http://127.0.0.1:7860" in capsys.readouterr().out and engine.closed


def test_serve_options(index_file):
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    code, recorder, factory = run_serve(
        ["--host", "localhost", "--port", "9000", "--open-browser", "--no-llm"], engine, index_file
    )
    assert code == 0
    kwargs = recorder.calls[0][1]
    assert (kwargs["host"], kwargs["port"], kwargs["open_browser"]) == ("localhost", 9000, True)
    assert factory.calls[0][1]["use_llm"] is False


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.5.5.5", "localhost", "LOCALHOST", "::1", "[::1]"]
)
def test_loopback_hosts_are_allowed(host, index_file, capsys):
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    code, recorder, _ = run_serve(["--host", host], engine, index_file)
    assert code == 0 and len(recorder.calls) == 1
    assert "warning" not in capsys.readouterr().err


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "photos.example.com", ""])
def test_non_loopback_host_is_refused(host, index_file, capsys):
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    code, recorder, factory = run_serve(["--host", host], engine, index_file)
    err = capsys.readouterr().err
    assert code == 2
    assert recorder.calls == [] and factory.calls == []  # nothing loaded, nothing served
    assert "Refusing to listen" in err and "--allow-remote" in err


def test_allow_remote_serves_with_a_warning(index_file, capsys):
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    code, recorder, _ = run_serve(["--host", "0.0.0.0", "--allow-remote"], engine, index_file)
    err = capsys.readouterr().err
    assert code == 0 and recorder.calls[0][1]["host"] == "0.0.0.0"
    assert "exposes your photo library to the network" in err


def test_serve_without_an_index_exits_2(tmp_path, capsys):
    recorder = ServeRecorder()
    factory = Factory(FakeEngine([]))
    code = search_cli.main_serve(
        serve_args("--index", str(tmp_path / "none.sqlite")),
        engine_factory=factory,
        serve_fn=recorder,
    )
    assert code == 2 and factory.calls == [] and recorder.calls == []
    assert "Run `facesorter index` first" in capsys.readouterr().err


@pytest.mark.parametrize("meta", [None, "[]", "not json", '"/lib"', "[1, 2]", "{}"])
def test_serve_without_library_roots_exits_2(meta, index_file, capsys):
    engine = FakeEngine([], meta={} if meta is None else {"library_roots": meta})
    code, recorder, _ = run_serve([], engine, index_file)
    assert code == 2 and recorder.calls == [] and engine.closed
    assert "facesorter index" in capsys.readouterr().err


def test_serve_without_gradio_fails_before_loading_anything(index_file, monkeypatch, capsys):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    monkeypatch.setattr(sys, "version_info", (3, 11, 4, "final", 0))
    factory = Factory(FakeEngine([], meta={"library_roots": '["/lib"]'}))
    code = search_cli.main_serve(serve_args("--index", str(index_file)), engine_factory=factory)
    out, err = capsys.readouterr()
    assert code == 2 and factory.calls == []
    assert "gradio" in err and "Serving" not in out and "Traceback" not in err


def test_serve_without_gradio_on_old_python(index_file, monkeypatch, capsys):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    monkeypatch.setattr(sys, "version_info", (3, 9, 6, "final", 0))
    code = search_cli.main_serve(
        serve_args("--index", str(index_file)), engine_factory=Factory(FakeEngine([]))
    )
    err = capsys.readouterr().err
    assert code == 2 and "Python 3.10 or newer" in err and "pip install" not in err


def test_serve_ui_model_unavailable_is_one_clean_line(index_file, capsys):
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    code = search_cli.main_serve(
        serve_args("--index", str(index_file)),
        engine_factory=Factory(engine),
        serve_fn=Boom(ModelUnavailableError("The search UI needs gradio")),
    )
    err = capsys.readouterr().err
    assert code == 2 and err.strip() == "error: The search UI needs gradio"


@pytest.mark.parametrize(
    ("argv", "shown"),
    [
        (["--host", "::1"], "http://[::1]:7860"),
        (["--host", "[::1]", "--port", "9"], "http://[::1]:9"),
        (["--host", "localhost", "--port", "0"], "localhost (port chosen by the OS)"),
        (["--port", "0"], "127.0.0.1 (port chosen by the OS)"),
    ],
)
def test_serving_line_formats_hosts_and_ports(argv, shown, index_file, capsys):
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    run_serve(argv, engine, index_file)
    assert f"at {shown} (Ctrl-C to stop)" in capsys.readouterr().out


def test_serve_ui_failure_is_reported(index_file, capsys, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 4, "final", 0))
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    code = search_cli.main_serve(
        serve_args("--index", str(index_file)),
        engine_factory=Factory(engine),
        serve_fn=Boom(ModuleNotFoundError("x", name="gradio")),
    )
    assert code == 2 and 'pip install "facesorter[search]"' in capsys.readouterr().err
    assert engine.closed


def test_serve_default_ui_is_imported_lazily(index_file, monkeypatch):
    seen = []
    fake_ui = SimpleNamespace(serve=lambda engine, **kw: seen.append(kw))
    monkeypatch.setitem(sys.modules, "facesorter.search.ui", fake_ui)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    engine = FakeEngine([], meta={"library_roots": '["/lib"]'})
    code = search_cli.main_serve(
        serve_args("--index", str(index_file)), engine_factory=Factory(engine)
    )
    assert code == 0 and seen[0]["library_roots"] == ["/lib"]


# --------------------------------------------------------------------------- stats / people


def vec(seed):
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


@pytest.fixture
def store_path(tmp_path):
    path = tmp_path / "real.sqlite"
    with Store.open(path, model_name="fake", dim=DIM) as store:
        records = []
        for i in range(4):
            meta = PhotoMeta(
                path=f"/lib/p{i}.jpg",
                size=i + 1,
                mtime_ns=1,
                taken=datetime(2020 + i, 5, 1),
                date_source="exif",
                lat=24.5 if i < 2 else None,
                lon=-81.8 if i < 2 else None,
            )
            place = PlaceLabel(city="Key West", region="Florida", country="United States")
            records.append((meta, place if i < 2 else None, vec(i)))
        ids = store.upsert_photos(records)
        store.set_faces(ids[0], [FaceRow(1, 0, 0, 10, 10), FaceRow(2, 0, 0, 4, 4)])
        store.set_faces(ids[1], [FaceRow(1, 0, 0, 12, 12)])
        store.set_faces(ids[2], [FaceRow(1, 0, 0, 3, 3)])
        store.name_cluster(2, "Bob")
        store.record_feedback("beach", ids[0], True)
    return path


def test_stats_lines(store_path, capsys):
    assert main(["stats", "--index", str(store_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"Index:        {store_path} (model: fake)",
        "Photos:       4",
        "With GPS:     2",
        "With a place: 2",
        "Date span:    2020-05-01 to 2023-05-01",
        "Date source:  exif 4, folder 0, mtime 0, none 0",
        "Faces:        4",
        "Named people: 1",
        "Feedback:     1",
    ]


def test_stats_of_an_empty_index(tmp_path, capsys):
    path = tmp_path / "empty.sqlite"
    Store.open(path, model_name="fake", dim=DIM).close()
    assert main(["stats", "--index", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Photos:       0" in out and "Date span:    none" in out


def test_stats_uses_the_default_index_path(index_env, capsys):
    Store.open(index_env, model_name="fake", dim=DIM).close()
    assert main(["stats"]) == 0
    assert str(index_env) in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["stats"], ["people", "list"], ["people", "name", "1", "x"]])
def test_commands_on_a_missing_index_exit_2(argv, tmp_path, capsys):
    assert main([*argv, "--index", str(tmp_path / "none.sqlite")]) == 2
    assert "facesorter index" in capsys.readouterr().err
    assert not (tmp_path / "none.sqlite").exists()  # never creates an index by accident


def test_people_list(store_path, capsys):
    assert main(["people", "list", "--index", str(store_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["id", "faces", "photos", "name"]
    assert out[1].split() == ["1", "3", "3", "-"]
    assert out[2].split() == ["2", "1", "1", "Bob"]
    assert len(out) == 3 and "/lib/" not in "".join(out)  # paths are private without -v


def test_people_list_verbose_shows_three_sample_paths(store_path, capsys):
    assert main(["people", "list", "-v", "--index", str(store_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert sum("/lib/p" in line for line in out) == 3 + 1


def test_people_list_when_empty(tmp_path, capsys):
    path = tmp_path / "e.sqlite"
    Store.open(path, model_name="fake", dim=DIM).close()
    assert main(["people", "list", "--index", str(path)]) == 0
    assert "people scan" in capsys.readouterr().out


def test_people_name_and_clear(store_path, capsys):
    base = ["--index", str(store_path)]
    assert main(["people", "name", "1", "  Alice   Smith ", *base]) == 0
    assert "named 'Alice Smith'" in capsys.readouterr().out
    with Store.open(store_path, create=False) as store:
        assert store.known_names() == ["Alice Smith", "Bob"]
    assert main(["people", "name", "1", "--clear", *base]) == 0
    assert "name removed" in capsys.readouterr().out
    assert main(["people", "name", "2", "", *base]) == 0  # empty NAME removes too
    with Store.open(store_path, create=False) as store:
        assert store.known_names() == []


def test_people_name_duplicate_exits_2(store_path, capsys):
    assert main(["people", "name", "1", "bob", "--index", str(store_path)]) == 2
    err = capsys.readouterr().err
    assert "already used by face cluster 2" in err and "Traceback" not in err


def test_people_name_unknown_cluster_exits_2(store_path, capsys):
    assert main(["people", "name", "99", "Zed", "--index", str(store_path)]) == 2
    assert "unknown face cluster 99" in capsys.readouterr().err


def test_people_name_needs_a_name_or_clear(store_path, capsys):
    assert main(["people", "name", "1", "--index", str(store_path)]) == 2
    assert "Give a NAME" in capsys.readouterr().err
    assert main(["people", "name", "1", "X", "--clear", "--index", str(store_path)]) == 2
    assert "not both" in capsys.readouterr().err


class FakePeople:
    def __init__(self, scan=None, cluster=None):
        self.scan = scan or ScanReport(7, 12, 2, (("/lib/bad.jpg", "boom"),), 1.5)
        self.cluster = cluster or ClusterReport(12, 3, 2, ("Old",), 1, 0.5)
        self.calls = []

    def scan_faces(self, store, **kwargs):
        self.calls.append(("scan", kwargs))
        kwargs["progress"](1, 2)
        kwargs["progress"](2, 2)
        return self.scan

    def cluster_faces(self, store, **kwargs):
        self.calls.append(("cluster", kwargs))
        return self.cluster


def test_people_scan(store_path, capsys):
    people = FakePeople()
    args = parse(
        "people", "scan", "--index", str(store_path), "--confidence", "0.5", "--limit", "9"
    )
    assert search_cli.main_people(args, people_module=people) == 0
    out, err = capsys.readouterr()
    assert "Scanned 7 photo(s): 12 face(s), 2 photo(s) without faces, 1 failed (1.5 s)" in out
    assert "1 photo(s) could not be scanned (run with -v to list them)" in err
    assert "bad.jpg" not in err
    ((name, kwargs),) = people.calls
    assert name == "scan" and kwargs["confidence"] == 0.5 and kwargs["limit"] == 9


def test_people_scan_defaults_and_verbose(store_path, capsys):
    people = FakePeople()
    args = parse("-v", "people", "scan", "--index", str(store_path))
    assert search_cli.main_people(args, people_module=people) == 0
    assert people.calls[0][1]["confidence"] == 0.8 and people.calls[0][1]["limit"] is None
    assert "/lib/bad.jpg: boom" in capsys.readouterr().err


def test_people_cluster(store_path, capsys):
    people = FakePeople()
    args = parse("people", "cluster", "--index", str(store_path), "--threshold", "0.4")
    assert search_cli.main_people(args, people_module=people) == 0
    out, err = capsys.readouterr()
    assert "Grouped 12 face(s) into 3 cluster(s) (1 with a single face); kept 2 name(s)" in out
    assert "names that no longer fit a cluster: Old" in err
    assert people.calls == [("cluster", {"threshold": 0.4})]
    search_cli.main_people(
        parse("people", "cluster", "--index", str(store_path)),
        people_module=FakePeople(cluster=ClusterReport(1, 1, 0, (), 0, 0.1)),
    )
    assert "warning" not in capsys.readouterr().err


def test_people_scan_with_the_real_module_and_nothing_to_scan(tmp_path, capsys):
    path = tmp_path / "e.sqlite"
    Store.open(path, model_name="fake", dim=DIM).close()  # no photos: no face model is loaded
    assert main(["people", "scan", "--index", str(path)]) == 0
    assert "Scanned 0 photo(s)" in capsys.readouterr().out


def test_people_scan_model_unavailable_exits_2(store_path, capsys):
    class NoModel(FakePeople):
        def scan_faces(self, store, **kwargs):
            raise ModelUnavailableError("Face recognition needs InsightFace")

    args = parse("people", "scan", "--index", str(store_path))
    assert search_cli.main_people(args, people_module=NoModel()) == 2
    assert "InsightFace" in capsys.readouterr().err


def test_people_scan_missing_insightface_import_exits_2(store_path, capsys):
    class NoImport(FakePeople):
        def scan_faces(self, store, **kwargs):
            raise ModuleNotFoundError("No module named 'insightface'", name="insightface")

    args = parse("people", "scan", "--index", str(store_path))
    assert search_cli.main_people(args, people_module=NoImport()) == 2
    assert "pip install insightface onnxruntime" in capsys.readouterr().err


def test_people_scan_closes_the_store(store_path):
    closed = []

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            closed.append(True)
            self.inner.close()

    def factory(path, **kwargs):
        return Spy(Store.open(path, **kwargs))

    args = parse("people", "list", "--index", str(store_path))
    assert search_cli.main_people(args, store_factory=factory) == 0
    assert closed == [True]


# --------------------------------------------------------------------------- eval


class FakeEvaluate:
    def __init__(self, code=0):
        self.code = code
        self.run_calls = []

    def add_arguments(self, parser):
        parser.add_argument("--suite", default="all")
        parser.add_argument("--queries", type=int, default=5)

    def run(self, args, *, engine_factory=None):
        self.run_calls.append((args, engine_factory))
        return self.code


def test_eval_forwards_everything_after_the_command():
    assert parse("eval", "--suite", "dates", "-k", "3").eval_args == ["--suite", "dates", "-k", "3"]
    assert parse("eval").eval_args == []


def test_eval_delegates_to_the_evaluation_module():
    fake = FakeEvaluate(code=0)
    args = parse("-v", "eval", "--suite", "dates", "--queries", "9")
    assert search_cli.main_eval(args, evaluate_module=fake) == 0
    ns, factory = fake.run_calls[0]
    assert (ns.suite, ns.queries, ns.verbose) == ("dates", 9, True)
    assert factory is None


def test_eval_passes_an_injected_engine_factory_and_the_exit_code():
    fake = FakeEvaluate(code=1)
    engine_factory = Factory(None)
    code = search_cli.main_eval(parse("eval"), evaluate_module=fake, engine_factory=engine_factory)
    assert code == 1 and fake.run_calls[0][1] is engine_factory


def test_eval_option_errors_come_from_the_evaluation_parser(capsys):
    with pytest.raises(SystemExit) as exc:
        search_cli.main_eval(parse("eval", "--bogus"), evaluate_module=FakeEvaluate())
    assert exc.value.code == 2
    assert "usage: facesorter eval" in capsys.readouterr().err


def test_eval_help_is_the_evaluation_modules(capsys):
    with pytest.raises(SystemExit) as exc:
        search_cli.main_eval(parse("eval", "--help"), evaluate_module=FakeEvaluate())
    assert exc.value.code == 0
    assert "--suite" in capsys.readouterr().out


def test_eval_through_main_uses_the_real_module_name(monkeypatch):
    fake = FakeEvaluate(code=0)
    monkeypatch.setitem(sys.modules, "facesorter.search.evaluate", fake)
    assert main(["eval", "--queries", "2"]) == 0
    assert fake.run_calls[0][0].queries == 2


# --------------------------------------------------------------------------- misc


def test_dispatch_routes_every_command(monkeypatch):
    seen = []
    for name in ("index", "search", "serve", "people", "stats", "eval"):
        monkeypatch.setattr(search_cli, f"main_{name}", lambda a, n=name: seen.append(n) or 0)
    for argv in (
        ["index", "."],
        ["search", "x"],
        ["serve"],
        ["people", "list"],
        ["stats"],
        ["eval"],
    ):
        assert search_cli.dispatch(parse(*argv)) == 0
    assert seen == list(search_cli.COMMANDS)


def test_replace_keeps_other_request_fields():
    # the clarification retry must not lose k / filters / use_llm
    engine = clarifying_engine()
    run_search(
        ["--no-llm", "-k", "7", "--from", "2020", "x"],
        engine,
        is_interactive=lambda: True,
        input_fn=lambda p: "",
    )
    first, second = engine.requests
    assert second == replace(first, clarifications=second.clarifications)
    assert second.k == 7 and second.use_llm is False and second.filters == first.filters


def test_eval_with_the_real_evaluation_module(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0
    assert "--sample" in capsys.readouterr().out
    assert main(["eval", "--sample", "0"]) == 2  # validated by evaluate.run
    assert "at least 1" in capsys.readouterr().err


# --------------------------------------------------------------------------- encodings / json


def ile_hit():
    photo = make_photo(1, "/lib/Île é.jpg", city="Paris")
    photo = replace(photo, place=PlaceLabel(city="Paris", region="Île-de-France", country="France"))
    return Hit(photo, 0.5)


def test_json_keeps_non_ascii_literal_on_a_utf8_terminal(capsys):
    run_search(["--json", "x"], FakeEngine([make_response([ile_hit()])]))
    out = capsys.readouterr().out
    assert "Île-de-France" in out and "\\u00ce" not in out
    assert json.loads(out)["place"] == "Paris, Île-de-France, France"


def test_json_month_precision_is_reported(capsys):
    hit = Hit(make_photo(1, taken=datetime(2019, 7, 1), precision="month"), 0.5)
    run_search(["--json", "x"], FakeEngine([make_response([hit])]))
    assert json.loads(capsys.readouterr().out)["date_precision"] == "month"


class AsciiStream:
    """A strict ASCII text stream (like PYTHONIOENCODING=ascii) that records its bytes."""

    def __init__(self):
        import io

        self.raw = io.BytesIO()
        self.stream = io.TextIOWrapper(self.raw, encoding="ascii", errors="strict")

    def text(self):
        self.stream.flush()
        return self.raw.getvalue().decode("ascii")


@pytest.fixture
def ascii_streams(monkeypatch):
    """Call the returned function inside the test (pytest swaps streams at the start of a call)."""

    def install():
        out, err = AsciiStream(), AsciiStream()
        monkeypatch.setattr(sys, "stdout", out.stream)
        monkeypatch.setattr(sys, "stderr", err.stream)
        return out, err

    return install


def test_table_does_not_crash_on_an_ascii_terminal(ascii_streams):
    out, _ = ascii_streams()
    code, _ = run_search(["x"], FakeEngine([make_response([ile_hit(), *hits_fixture()])]))
    assert code == 0
    text = out.text()
    assert "\\xce" in text and "Searching:" in text and "/lib/d.jpg" in text  # all rows printed


def test_json_is_valid_ascii_json_on_an_ascii_terminal(ascii_streams):
    out, _ = ascii_streams()
    run_search(["--json", "x"], FakeEngine([make_response([ile_hit()])]))
    record = json.loads(out.text())
    assert record["place"] == "Paris, Île-de-France, France"  # escaped, but still correct JSON


def test_messages_on_stderr_survive_an_ascii_terminal(ascii_streams):
    _, err = ascii_streams()
    engine = FakeEngine([make_response([], dropped=[unresolved("Zürich 2024", "place")])])
    assert run_search(["x"], engine)[0] == 0
    assert "Z\\xfcrich 2024" in err.text()


def test_odd_streams_are_left_alone(monkeypatch):
    class Frozen:
        def reconfigure(self, **kwargs):
            raise ValueError("cannot reconfigure")

    monkeypatch.setattr(sys, "stdout", Frozen())
    search_cli._tolerate_odd_encodings()  # must not raise
    monkeypatch.setattr(sys, "stdout", object())  # no reconfigure at all
    search_cli._tolerate_odd_encodings()
    assert search_cli._stdout_shows_unicode() is False  # no encoding attribute: assume ASCII


def test_stdout_shows_unicode_for_utf8_and_unknown_codec(monkeypatch):
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(encoding="utf-8"))
    assert search_cli._stdout_shows_unicode() is True
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(encoding="no-such-codec"))
    assert search_cli._stdout_shows_unicode() is False


# --------------------------------------------------------------------------- nits


def test_index_of_a_file_says_not_a_directory(tmp_path, capsys):
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"x")
    code, factory = run_index([str(target)], FakeIndexer())
    assert code == 2 and factory.calls == []
    assert "not a directory" in capsys.readouterr().err


def test_index_file_inside_the_library_warns(library, capsys):
    code, _ = run_index(
        [str(library), "--index", str(library / ".facesorter" / "i.sqlite")], FakeIndexer()
    )
    err = capsys.readouterr().err
    assert code == 0 and "inside the library" in err
    run_index([str(library), "--index", str(library.parent / "elsewhere.sqlite")], FakeIndexer())
    assert "inside the library" not in capsys.readouterr().err


def test_people_name_length_is_capped(store_path, capsys):
    base = ["--index", str(store_path)]
    assert main(["people", "name", "1", "x" * 101, *base]) == 2
    assert "too long (101 characters" in capsys.readouterr().err
    assert main(["people", "name", "1", "x" * 100, *base]) == 0


# --------------------------------------------------------------------------- index --remove-root


@pytest.fixture
def existing_index(tmp_path):
    path = tmp_path / "exists.sqlite"
    path.write_text("x")
    return path


def test_remove_root_reports_the_count(existing_index, tmp_path, capsys):
    indexer = FakeIndexer()
    root = tmp_path / "old library"  # need not exist any more
    indexer.removed[str(root)] = 12
    factory = IndexFactory(indexer)
    code = search_cli.main_index(
        parse(
            "index", "--remove-root", str(root), "--index", str(existing_index), "--device", "cpu"
        ),
        indexer_factory=factory,
    )
    out = capsys.readouterr().out
    assert code == 0 and f"Removed 12 photo(s) of {root}" in out
    assert indexer.closed and indexer.run_calls == []  # nothing is indexed
    assert factory.calls == [(str(existing_index), {"device": "cpu"})]


def test_remove_root_with_no_photos_left_still_succeeds(existing_index, tmp_path, capsys):
    indexer = FakeIndexer()
    indexer.removed["/gone"] = 0  # known root without photos: only the setting is dropped
    code = search_cli.main_index(
        parse("index", "--remove-root", "/gone", "--index", str(existing_index)),
        indexer_factory=IndexFactory(indexer),
    )
    assert code == 0 and "Removed 0 photo(s)" in capsys.readouterr().out


def test_remove_root_unknown_root_exits_2(existing_index, capsys):
    indexer = FakeIndexer()
    code = search_cli.main_index(
        parse("index", "--remove-root", "/never/indexed", "--index", str(existing_index)),
        indexer_factory=IndexFactory(indexer),
    )
    assert code == 2 and "not an indexed library folder" in capsys.readouterr().err


def test_remove_root_cannot_be_combined_with_a_library(existing_index, library, capsys):
    factory = IndexFactory(FakeIndexer())
    argv = ["index", str(library), "--remove-root", "/x", "--index", str(existing_index)]
    assert search_cli.main_index(parse(*argv), indexer_factory=factory) == 2
    assert "cannot be combined" in capsys.readouterr().err and factory.calls == []


def test_index_needs_a_library_or_remove_root(capsys):
    assert search_cli.main_index(parse("index"), indexer_factory=IndexFactory(FakeIndexer())) == 2
    assert "--remove-root" in capsys.readouterr().err


def test_remove_root_needs_an_existing_index(tmp_path, capsys):
    factory = IndexFactory(FakeIndexer())
    argv = ["index", "--remove-root", "/x", "--index", str(tmp_path / "none.sqlite")]
    assert search_cli.main_index(parse(*argv), indexer_factory=factory) == 2
    assert "Run `facesorter index` first" in capsys.readouterr().err
    assert factory.calls == [] and not (tmp_path / "none.sqlite").exists()


def test_remove_root_uses_open_indexer_by_default(existing_index, monkeypatch, capsys):
    indexer = FakeIndexer()
    indexer.removed[os.path.abspath("/r")] = 3
    monkeypatch.setattr("facesorter.search.indexer.open_indexer", IndexFactory(indexer))
    assert main(["index", "--remove-root", "/r", "--index", str(existing_index)]) == 0
    assert "Removed 3 photo(s)" in capsys.readouterr().out
