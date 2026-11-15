"""Tests for facesorter.search.ui: the Gradio app, run for real on a loopback port.

Skipped when gradio is not installed (it needs Python 3.10+). The engine is scripted
(``FakeEngine``) over a temporary store of synthetic photos; nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

gr = pytest.importorskip("gradio")  # needs Python >= 3.10; everything below needs it too
gradio_client = pytest.importorskip("gradio_client")
httpx = pytest.importorskip("httpx")

from PIL import Image  # noqa: E402
from test_search_ui_logic import FakeEngine, make_jpeg, photo_meta, unresolved  # noqa: E402

from facesorter.search import ui  # noqa: E402
from facesorter.search.errors import ModelUnavailableError  # noqa: E402
from facesorter.search.store import Store  # noqa: E402
from facesorter.search.ui_logic import Session, UIResult  # noqa: E402

# Positions in the lists of outputs the search / answer / skip events return (see
# ``_Handlers.render``). gradio_client leaves out the State and layout groups; the raw queue
# API (class ``Raw``) returns every output, so its indices are shifted.
Q = ui.MAX_QUESTIONS
STATUS, READING, NOTICES, DROPPED, Q0 = 0, 1, 2, 3, 4
GALLERY, DETAIL, FEEDBACK = 4 + Q, 5 + Q, 6 + Q
RAW_BOX, RAW_Q0, RAW_GALLERY = 5, 6, 6 + Q


@pytest.fixture(autouse=True)
def _keep_gradio_cache_env():
    """Apps point GRADIO_TEMP_DIR at their own folder; do not leak that between tests."""
    saved = os.environ.get("GRADIO_TEMP_DIR")
    yield
    if saved is None:
        os.environ.pop("GRADIO_TEMP_DIR", None)
    else:
        os.environ["GRADIO_TEMP_DIR"] = saved


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("ui")
    library = root / "library" / "trip"
    library.mkdir(parents=True)
    store = Store.open(root / "index.sqlite", model_name="fake", dim=8)
    rng = np.random.default_rng(0)
    records = [
        (
            photo_meta(
                make_jpeg(library / f"IMG_{i}.jpg", size=(900, 600)), f"2023-07-1{i}T10:00:00"
            ),
            None,
            rng.standard_normal(8).astype(np.float32),
        )
        for i in range(4)
    ]
    store.upsert_photos(records)
    store.known_names = lambda: ["Alice", "Bob"]  # type: ignore[method-assign]
    engine = FakeEngine(store)
    thumbs = root / ".facesorter" / "thumbs"  # a dot directory, like the real default
    demo = ui.build_app(engine, thumbs_dir=thumbs, library_roots=[str(library.parent)])
    port = free_port()
    ui.launch_app(demo, thumbs_dir=thumbs, port=port, block=False)
    yield SimpleNamespace(
        demo=demo, engine=engine, store=store, library=library, thumbs=thumbs.resolve(),
        base=f"http://127.0.0.1:{port}", port=port,
    )  # fmt: skip
    demo.close()
    store.close()


@pytest.fixture
def w(world):
    world.engine.requests.clear()
    world.engine.ask = []
    world.engine.raises = None
    world.engine.respond = None
    return world


def new_client(w) -> gradio_client.Client:
    return gradio_client.Client(w.base, verbose=False)


def search(client, query="blue", date_from="", date_to="", place="", people=(), k=3, llm=True):
    return client.predict(
        query, date_from, date_to, place, list(people), k, llm, api_name="/search"
    )


# --------------------------------------------------------------------------- launch settings


def test_launch_is_local_private_and_telemetry_free(w):
    demo = w.demo
    assert demo.share is False and demo.share_url is None
    assert demo.analytics_enabled is False
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert demo.allowed_paths == [str(w.thumbs)]  # the thumbnail cache and nothing else
    assert demo.blocked_paths == [] or str(w.library) not in demo.allowed_paths
    assert demo.local_url.startswith(f"http://127.0.0.1:{w.port}")
    assert demo.server_name == "127.0.0.1"
    page = httpx.get(w.base)
    assert page.status_code == 200 and "facesorter photo search" in page.text
    assert demo._queue is not None  # queue enabled so a slow LLM call does not freeze the page


def test_no_connection_leaves_the_machine(tmp_path: Path, monkeypatch):
    seen: list[object] = []
    real_connect = socket.socket.connect

    def spy(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            seen.append(address)
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", spy)
    store = Store.open(tmp_path / "i.sqlite", model_name="fake", dim=8)
    try:
        engine = FakeEngine(store)
        thumbs = tmp_path / "t"
        demo = ui.build_app(engine, thumbs_dir=thumbs)
        ui.launch_app(demo, thumbs_dir=thumbs, port=free_port(), block=False)
        try:
            search(gradio_client.Client(demo.local_url, verbose=False))
        finally:
            demo.close()
    finally:
        store.close()
    assert seen == []


def test_server_is_not_reachable_on_a_non_loopback_address(w):
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.254.254.254", 9))  # UDP connect sends nothing; it only picks a route
        lan_ip = probe.getsockname()[0]
    except OSError:
        pytest.skip("no network interface to probe")
    finally:
        probe.close()
    if lan_ip.startswith("127."):
        pytest.skip("only a loopback interface exists")
    with pytest.raises(OSError):
        socket.create_connection((lan_ip, w.port), timeout=2).close()
    socket.create_connection(("127.0.0.1", w.port), timeout=2).close()


# --------------------------------------------------------------------------- what is served


def file_url(w, path) -> str:
    return f"{w.base}/gradio_api/file={path}"


def test_thumbnails_are_served_but_originals_and_other_files_are_not(w):
    client = new_client(w)
    search(client)
    thumbs = sorted(w.thumbs.glob("*.jpg"))
    assert len(thumbs) == 3
    ok = httpx.get(file_url(w, thumbs[0]))
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg"
    assert Image.open(__import__("io").BytesIO(ok.content)).size == (512, 341)

    originals = sorted(w.library.glob("*.jpg"))
    assert originals
    for forbidden in (*originals, w.library, w.store.path, Path("/etc/hosts")):
        assert httpx.get(file_url(w, forbidden)).status_code == 403, forbidden
    # path tricks stay inside the sandbox too
    sneaky = f"{w.thumbs}/../../../library/trip/{originals[0].name}"
    assert httpx.get(file_url(w, sneaky)).status_code == 403
    assert httpx.get(f"{w.base}/gradio_api/file=http://example.com/x.jpg").status_code in (403, 404)
    # the root, config and docs do not list any file path of the library
    config = httpx.get(f"{w.base}/config").text
    assert str(w.library) not in config


def test_gallery_items_returned_to_the_browser_are_the_thumbnails(w):
    client = new_client(w)
    out = search(client, k=2)
    items = out[GALLERY]["value"]
    assert len(items) == 2
    for item in items:
        with Image.open(item["image"]) as img:  # the client's downloaded copy
            assert img.format == "JPEG" and max(img.size) <= 1024
    assert [i["caption"] for i in items] == ["2023-07-10 · 1.00", "2023-07-11 · 0.50"]


# --------------------------------------------------------------------------- end to end


def test_search_through_the_api(w):
    out = new_client(w).predict(
        "blue sea", "2023-06", "2023-08", "", ["Bob"], 2, False, api_name="/search"
    )
    (req,) = w.engine.requests
    assert req.query == "blue sea" and req.k == 2 and req.use_llm is False
    assert req.filters is not None and req.filters.people == ("Bob",)
    assert out[STATUS].startswith("Showing 2 photos of 4 candidates")
    assert out[READING]["visible"] is True
    assert "dates 2023\\-06\\-01 to 2023\\-08\\-31" in out[READING]["value"]
    assert "Bob" in out[READING]["value"]
    assert out[DROPPED]["visible"] is False
    assert all(out[Q0 + i]["visible"] is False for i in range(ui.MAX_QUESTIONS))


def test_a_bad_date_is_a_friendly_message(w):
    out = search(new_client(w), date_from="2023-02-30")
    assert "not a real calendar date" in out[STATUS].replace("\\", "")
    assert out[GALLERY]["value"] == [] and w.engine.requests == []


def test_an_engine_failure_does_not_break_the_page(w):
    w.engine.raises = RuntimeError("boom")
    out = search(new_client(w))
    assert out[STATUS].replace("\\", "") == "Search failed: boom"


def test_clarification_round_trip_asks_once_then_tells(w):
    w.engine.ask = [unresolved("date", "Diwali 2023"), unresolved("place", "Narnia")]
    client = new_client(w)
    first = search(client, query="lights")
    assert first[GALLERY]["value"] == []
    assert first[Q0]["visible"] and first[Q0]["label"] == 'What do you mean by "Diwali 2023"?'
    assert first[Q0 + 1]["visible"] and not first[Q0 + 2]["visible"]
    assert w.engine.requests[-1].allow_clarification is True

    second = client.predict("March 2023", "", "", "", "", "", api_name="/answer")
    req = w.engine.requests[-1]
    assert req.allow_clarification is False
    assert dict(req.clarifications) == {"Diwali 2023": "March 2023", "Narnia": ""}
    assert not any(second[Q0 + i]["visible"] for i in range(Q))
    assert second[DROPPED]["visible"] is True and "Narnia" in second[DROPPED]["value"]
    assert len(second[GALLERY]["value"]) == 3

    # skipping is the other button, and a fresh search asks again
    assert search(client, query="lights")[Q0]["visible"] is True
    skipped = client.predict(api_name="/skip")
    assert dict(w.engine.requests[-1].clarifications) == {"Diwali 2023": "", "Narnia": ""}
    assert not skipped[Q0]["visible"] and len(skipped[GALLERY]["value"]) == 3


def test_the_clarification_block_appears_only_when_the_engine_asks(w):
    raw = Raw(w)
    plain = raw.call("search", ["blue", "", "", "", [], 3, True])
    assert plain[RAW_BOX]["visible"] is False and len(plain[RAW_GALLERY]["value"]) == 3
    w.engine.ask = [unresolved("person", "Zed")]
    asking = raw.call("search", ["blue", "", "", "", [], 3, True])
    assert asking[RAW_BOX]["visible"] is True and asking[RAW_Q0]["visible"] is True
    assert asking[RAW_GALLERY]["value"] == []
    done = raw.call("skip", [None])
    assert done[RAW_BOX]["visible"] is False and len(done[RAW_GALLERY]["value"]) == 3


def test_answering_without_a_search_is_a_message(w):
    out = new_client(w).predict("x", "", "", "", "", "", api_name="/answer")
    assert "nothing to answer" in out[STATUS]
    assert w.engine.requests == []


# --- the select event carries event data, which only the raw queue API can send ---------


class Raw:
    """Minimal client for Gradio's queue API (used for events gradio_client cannot call)."""

    def __init__(self, w) -> None:
        self.base = w.base
        self.hash = uuid.uuid4().hex
        self.ids = {}
        for dep in w.demo.config["dependencies"]:
            name = dep.get("api_name")
            if name:
                self.ids[name] = dep["id"]
            for _, event in dep["targets"]:
                if event == "select":
                    self.ids["select"] = dep["id"]

    def call(self, name, data, event_data=None):
        body = {
            "data": data, "fn_index": self.ids[name], "session_hash": self.hash,
            "event_data": event_data, "trigger_id": None,
        }  # fmt: skip
        httpx.post(f"{self.base}/gradio_api/queue/join", json=body).raise_for_status()
        params = {"session_hash": self.hash}
        with httpx.stream("GET", f"{self.base}/gradio_api/queue/data", params=params) as stream:
            for line in stream.iter_lines():
                if line.startswith("data:"):
                    msg = json.loads(line[5:])
                    if msg["msg"] == "process_completed":
                        assert msg["success"], msg
                        return msg["output"]["data"]
        raise AssertionError("no result")


def test_select_a_photo_shows_details_and_feedback_is_recorded(w):
    raw = Raw(w)
    raw.call("search", ["blue sea", "", "", "", [], 3, True])
    detail = raw.call("select", [None], {"index": 1, "value": None, "selected": True})
    assert detail[1]["visible"] is True
    text = detail[2].replace("\\", "").replace("*", "")
    assert "File: IMG_1.jpg" in text and "Date: 2023-07-11" in text and "Score: 0.500" in text
    assert "Folder: trip" in text  # the folder NAME only; never the full path
    assert str(w.library) not in text and str(w.library.parent) not in text

    before = len(w.store.feedback_rows())
    assert raw.call("rate_up", [None]) == ["Thanks, marked as relevant\\."]
    assert raw.call("rate_down", [None]) == ["Thanks, marked as not relevant\\."]
    rows = w.store.feedback_rows()[before:]
    assert [(r.query, r.photo_id, r.relevant) for r in rows] == [
        ("blue sea", 2, True),
        ("blue sea", 2, False),
    ]


def test_rating_before_selecting_a_photo_is_refused(w):
    raw = Raw(w)
    before = len(w.store.feedback_rows())
    assert raw.call("rate_up", [None]) == ["Select a photo first\\."]
    assert len(w.store.feedback_rows()) == before


# --------------------------------------------------------------------------- layout


def test_layout_has_the_documented_controls(w):
    blocks = list(w.demo.blocks.values())
    buttons = {b.value for b in blocks if isinstance(b, gr.Button)}
    assert {"Search", "Search with these answers", "Skip and search without"} <= buttons
    assert {"👍 Relevant", "👎 Not relevant"} <= buttons
    accordions = {b.label for b in blocks if isinstance(b, gr.Accordion)}
    assert "How I read your query" in accordions
    assert any(a.startswith("Filters") for a in accordions)
    (people,) = [b for b in blocks if isinstance(b, gr.Dropdown)]
    assert people.multiselect and [c[1] for c in people.choices] == ["Alice", "Bob"]
    (gallery,) = [b for b in blocks if isinstance(b, gr.Gallery)]
    assert gallery.allow_preview is False
    sliders = [b for b in blocks if isinstance(b, gr.Slider)]
    assert (sliders[0].minimum, sliders[0].maximum) == (1, ui.MAX_K)
    (checkbox,) = [b for b in blocks if isinstance(b, gr.Checkbox)]
    assert checkbox.visible is True
    assert w.demo.title == "facesorter photo search"


def test_page_load_refreshes_names_and_footer(w):
    update, footer = w.demo.facesorter_handlers.load()
    assert update["choices"] == ["Alice", "Bob"]
    assert footer.replace("\\", "") == "4 photos indexed, dated 2023-07-10 to 2023-07-13."
    assert str(w.library) not in footer and "Library" not in footer
    w.store.known_names = lambda: (_ for _ in ()).throw(RuntimeError("closed"))  # type: ignore
    try:
        assert w.demo.facesorter_handlers.load()[0]["choices"] == []
    finally:
        w.store.known_names = lambda: ["Alice", "Bob"]  # type: ignore[method-assign]


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({}, True),
        ({"has_llm": False}, False),
        ({"llm_available": True}, True),
        ({"llm": None}, False),
        ({"llm": object()}, True),
        ({"parser": SimpleNamespace(_llm=None)}, False),
        ({"parser": SimpleNamespace(_llm=object())}, True),
        ({"parser": SimpleNamespace()}, True),
    ],
)
def test_llm_checkbox_visibility(attrs, expected):
    assert ui._has_llm(SimpleNamespace(**attrs)) is expected


def test_the_checkbox_is_hidden_without_an_llm(tmp_path: Path):
    store = Store.open(tmp_path / "i.sqlite", model_name="fake", dim=8)
    try:
        engine = FakeEngine(store)
        engine.has_llm = False  # type: ignore[attr-defined]
        demo = ui.build_app(engine, thumbs_dir=tmp_path / "t")
        (box,) = [b for b in demo.blocks.values() if isinstance(b, gr.Checkbox)]
        assert box.visible is False
    finally:
        store.close()


# --------------------------------------------------------------------------- handlers


def test_handler_edge_cases(w):
    h = w.demo.facesorter_handlers
    # no session yet: friendly results, never an exception
    assert "nothing to answer" in h.answer(None, "x")[1]
    assert "nothing to answer" in h.skip(None)[1]
    assert h.rate(None, True) == "Select a photo first\\."
    assert h.select(None, SimpleNamespace(index=0))[1]["visible"] is False

    out = h.search("sea", "", "", "", None, None, True)  # k=None, people=None
    session = out[0]
    assert isinstance(session, Session) and w.engine.requests[-1].k == ui.DEFAULT_K
    for bad in (None, -1, 99, "x", [], True and 1.5):
        assert h.select(session, SimpleNamespace(index=bad))[1]["visible"] is False
    ok = h.select(session, SimpleNamespace(index=[2, 0]))
    assert ok[1]["visible"] is True and session.selected is session.result.gallery_cards()[2]
    assert h.rate(session, True) == "Thanks, marked as relevant\\."


def test_many_questions_are_capped_with_a_note(w):
    h = w.demo.facesorter_handlers
    w.engine.ask = [unresolved("person", f"Person{i}") for i in range(ui.MAX_QUESTIONS + 2)]
    out = h.search("x", "", "", "", None, 5, True)
    assert "Only the first" in out[1]
    assert sum(1 for q in out[6 : 6 + ui.MAX_QUESTIONS] if q["visible"]) == ui.MAX_QUESTIONS
    session = out[0]
    h.answer(session, *[""] * ui.MAX_QUESTIONS)
    assert len(w.engine.requests[-1].clarifications) == ui.MAX_QUESTIONS + 2  # extras: skipped


def test_user_text_is_escaped_in_markdown(w):
    h = w.demo.facesorter_handlers
    w.engine.ask = []
    out = h.search("![x](http://evil.example/a.png)", "", "", "", None, 2, True)
    assert "![x](http" not in out[2]["value"]
    assert "\\!\\[x\\]\\(http" in out[2]["value"]


# --------------------------------------------------------------------------- serve + guards


class FakeDemo:
    server_port = 4321

    def __init__(self) -> None:
        self.kwargs: dict | None = None
        self.closed = False

    def launch(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        self.closed = True


def test_launch_app_passes_safe_settings(tmp_path: Path, recwarn, capsys):
    demo = FakeDemo()
    ui.launch_app(demo, thumbs_dir=tmp_path, port=1234, open_browser=True, block=True)
    k = demo.kwargs
    assert k["server_name"] == "127.0.0.1" and k["server_port"] == 1234
    assert k["share"] is False and k["inbrowser"] is True and k["prevent_thread_lock"] is False
    assert k["allowed_paths"] == [str(tmp_path.resolve())]
    assert not recwarn.list and capsys.readouterr().err == ""
    ui.launch_app(demo, thumbs_dir=tmp_path, block=False)
    assert demo.kwargs["prevent_thread_lock"] is True and demo.kwargs["inbrowser"] is False


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "myhost.local"])
def test_a_non_loopback_host_warns_loudly(tmp_path: Path, host, capsys, caplog):
    demo = FakeDemo()
    with pytest.warns(UserWarning, match="NOT loopback"):
        ui.launch_app(demo, thumbs_dir=tmp_path, host=host)
    assert demo.kwargs["server_name"] == host and demo.kwargs["share"] is False
    assert "SECURITY WARNING" in capsys.readouterr().err
    assert "NOT loopback" in caplog.text


def test_serve_builds_launches_and_closes(tmp_path: Path, monkeypatch):
    store = Store.open(tmp_path / "i.sqlite", model_name="fake", dim=8)
    calls = []
    monkeypatch.setattr(ui, "launch_app", lambda demo, **kw: calls.append((demo, kw)))
    try:
        engine = FakeEngine(store)
        library = tmp_path / "photos"
        library.mkdir()
        ui.serve(engine, library_roots=[str(library)], port=5555, thumbs_dir=tmp_path / "thumbs")
        ((demo, kw),) = calls
        assert kw == {
            "thumbs_dir": (tmp_path / "thumbs").resolve(), "host": "127.0.0.1", "port": 5555,
            "open_browser": False,
        }  # fmt: skip
        assert (tmp_path / "thumbs").is_dir()
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        ui.serve(engine, library_roots=[], open_browser=True)
        assert calls[1][1]["thumbs_dir"] == (tmp_path / "home" / ".facesorter" / "thumbs").resolve()
        assert calls[1][1]["open_browser"] is True
    finally:
        store.close()


def test_serve_closes_the_server_even_if_launch_fails(tmp_path: Path, monkeypatch):
    store = Store.open(tmp_path / "i.sqlite", model_name="fake", dim=8)
    closed = []

    def boom(demo, **kw):
        demo.close = lambda: closed.append(True)
        raise OSError("port in use")

    monkeypatch.setattr(ui, "launch_app", boom)
    try:
        with pytest.raises(OSError, match="port in use"):
            ui.serve(FakeEngine(store), library_roots=[], thumbs_dir=tmp_path / "t")
    finally:
        store.close()
    assert closed == [True]


def test_the_thumbnail_directory_may_not_overlap_the_library(tmp_path: Path):
    store = Store.open(tmp_path / "i.sqlite", model_name="fake", dim=8)
    library = tmp_path / "photos"
    library.mkdir()
    try:
        engine = FakeEngine(store)
        for bad in (library, library / "thumbs", tmp_path):
            with pytest.raises(ValueError, match="overlaps the photo library"):
                ui.build_app(engine, thumbs_dir=bad, library_roots=[str(library)])
            with pytest.raises(ValueError, match="overlaps the photo library"):
                ui.serve(engine, library_roots=[str(library)], thumbs_dir=bad)
    finally:
        store.close()
    assert not (library / "thumbs").exists()


def test_import_hint_when_gradio_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "gradio", None)
    with pytest.raises(ModelUnavailableError, match=r"3\.10.*facesorter\[search\]"):
        ui._import_gradio()


def test_render_returns_one_value_per_output(w):
    h = w.demo.facesorter_handlers
    out = h.render(Session(), UIResult(query="q"))
    expected = 1 + 4 + 1 + ui.MAX_QUESTIONS + 1 + 3  # session, 4 texts+box, questions, gallery, 3
    assert len(out) == expected + 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h.render(Session(), UIResult(query="q", error="bad"))


# --------------------------------------------------------------------------- DNS rebinding


def url_for(w, path: str) -> str:
    return f"{w.base}{path}"


def get(w, path: str, host: str | None = None, origin: str | None = None, method="GET", **kw):
    headers = {}
    if host is not None:
        headers["Host"] = host
    if origin is not None:
        headers["Origin"] = origin
    return httpx.request(method, url_for(w, path), headers=headers, timeout=20, **kw)


def a_thumbnail(w) -> Path:
    search(new_client(w))
    w.engine.requests.clear()  # only requests made after this point count as attacks
    return sorted(w.thumbs.glob("*.jpg"))[0]


@pytest.mark.parametrize("host", ["evil.com", "evil.com:{port}", "127.0.0.1.evil.com:{port}",
                                  "192.168.1.9:{port}", "localhost:1"])  # fmt: skip
def test_rebound_hosts_get_403_on_every_route(w, host):
    host = host.format(port=w.port)
    thumb = a_thumbnail(w)
    for path in ("/", "/config", "/gradio_api/info", "/gradio_api/heartbeat/abc",
                 f"/gradio_api/file={thumb}", "/gradio_api/queue/data?session_hash=x",
                 "/gradio_api/startup-events", "/theme.css", "/nosuchroute"):  # fmt: skip
        response = get(w, path, host=host, origin=f"http://{host}")
        assert response.status_code == 403, (host, path, response.status_code)
        assert "Forbidden" in response.text
    join = get(w, "/gradio_api/queue/join", host=host, origin=f"http://{host}", method="POST",
               json={"data": [], "fn_index": 0, "session_hash": "x"})  # fmt: skip
    assert join.status_code == 403
    assert w.engine.requests == []  # no search ran


def test_a_rebound_page_cannot_search_or_read_thumbnails(w):
    """The full attack: evil.com resolves to 127.0.0.1 and its page drives the API."""
    host, origin = f"evil.com:{w.port}", f"http://evil.com:{w.port}"
    thumb = a_thumbnail(w)
    w.engine.requests.clear()
    assert get(w, "/config", host=host, origin=origin).status_code == 403
    body = {"data": ["red", "", "", "", [], 3, True], "fn_index": 0, "session_hash": "deadbeef"}
    assert get(w, "/gradio_api/queue/join", host=host, origin=origin, method="POST",
               json=body).status_code == 403  # fmt: skip
    assert get(w, f"/gradio_api/file={thumb}", host=host, origin=origin).status_code == 403
    assert w.engine.requests == []


@pytest.mark.parametrize(
    "host",
    ["{ip}:{port}", "localhost:{port}", "LocalHost:{port}", "LOCALHOST", "localhost",
     "127.0.0.1", "[::1]:{port}", "[::1]", "localhost.:{port}"],
)  # fmt: skip
def test_host_variants_for_this_machine_are_served(w, host):
    host = host.format(ip="127.0.0.1", port=w.port)
    thumb = a_thumbnail(w)
    assert get(w, "/config", host=host).status_code == 200
    assert get(w, f"/gradio_api/file={thumb}", host=host).status_code == 200
    assert get(w, "/", host=host, origin=f"http://{host}").status_code == 200


def test_a_foreign_origin_cannot_post_even_to_a_good_host(w):
    body = {"data": ["red", "", "", "", [], 3, True], "fn_index": 0, "session_hash": "cafe"}
    for origin in ("http://evil.com", f"http://evil.com:{w.port}", "null",
                   f"http://127.0.0.1:{w.port + 1}"):  # fmt: skip
        response = get(w, "/gradio_api/queue/join", host=f"127.0.0.1:{w.port}", origin=origin,
                       method="POST", json=body)  # fmt: skip
        assert response.status_code == 403, origin
    assert w.engine.requests == []
    own = f"http://127.0.0.1:{w.port}"
    ok = get(w, "/gradio_api/queue/join", host=f"127.0.0.1:{w.port}", origin=own, method="POST",
             json=body)  # fmt: skip
    assert ok.status_code == 200  # the page's own origin still works


def test_normal_flows_still_work_through_the_guard(w):
    out = search(new_client(w), k=2)
    assert len(out[GALLERY]["value"]) == 2
    raw = Raw(w)
    raw.call("search", ["blue sea", "", "", "", [], 3, True])
    assert raw.call("select", [None], {"index": 0, "value": None, "selected": True})[1]["visible"]


def test_launch_installs_the_guard_before_the_server_starts(tmp_path: Path):
    demo = FakeDemo()
    ui.launch_app(demo, thumbs_dir=tmp_path, port=1234)
    (mw,) = demo.kwargs["app_kwargs"]["middleware"]
    assert mw.cls is ui.HostGuardMiddleware
    policy = mw.kwargs["policy"]
    assert policy.hostnames == frozenset({"localhost", "127.0.0.1", "::1"})
    assert policy.port == 4321  # the port the server actually bound (reported by gradio)
    assert demo.kwargs["ssr_mode"] is False and demo.kwargs["share"] is False


def test_a_configured_non_loopback_host_joins_the_allowlist(tmp_path: Path):
    demo = FakeDemo()
    with pytest.warns(UserWarning):
        ui.launch_app(demo, thumbs_dir=tmp_path, host="photos.lan", port=1234)
    policy = demo.kwargs["app_kwargs"]["middleware"][0].kwargs["policy"]
    assert policy.host_allowed("photos.lan:4321") and policy.host_allowed("localhost")
    assert not policy.host_allowed("evil.com:4321") and not policy.host_allowed("10.0.0.3:4321")
    wildcard = FakeDemo()
    with pytest.warns(UserWarning):
        ui.launch_app(wildcard, thumbs_dir=tmp_path, host="0.0.0.0", port=1234)
    policy = wildcard.kwargs["app_kwargs"]["middleware"][0].kwargs["policy"]
    assert policy.host_allowed("10.0.0.3:4321") and not policy.host_allowed("evil.com:4321")


# --------------------------------------------------------------------------- private cache


def test_gradio_copies_of_thumbnails_live_inside_the_private_thumbnail_directory(w):
    cache = w.thumbs / ".gradio"
    assert os.environ["GRADIO_TEMP_DIR"] == str(cache)
    raw = Raw(w)
    out = raw.call("search", ["blue", "", "", "", [], 3, True])
    paths = [item["image"]["path"] for item in out[RAW_GALLERY]["value"]]
    assert len(paths) == 3
    for path in paths:
        assert Path(path).resolve().is_relative_to(cache.resolve()), path
        assert Path(path).is_file()
    if os.name != "nt":
        assert w.thumbs.stat().st_mode & 0o777 == 0o700
        assert cache.stat().st_mode & 0o777 == 0o700
    # and the browser can fetch those copies (they are the app's own cache)
    for item in out[RAW_GALLERY]["value"]:
        assert httpx.get(item["image"]["url"]).status_code == 200


def test_the_cache_folder_is_created_and_exported(tmp_path: Path):
    folder = ui._use_private_gradio_cache(tmp_path / "t")
    assert folder == tmp_path / "t" / ".gradio" and folder.is_dir()
    assert os.environ["GRADIO_TEMP_DIR"] == str(folder)
    if os.name != "nt":
        assert folder.stat().st_mode & 0o777 == 0o700


# --------------------------------------------------------------------------- mutants


def test_engine_calls_share_one_serial_concurrency_group(w):
    """Search, answer, skip and the query box's Enter key must never run at the same time."""
    fns = w.demo.fns.values()
    engine_calls = [f for f in fns if f.api_name in ("search", "answer", "skip") or
                    any(event == "submit" for _, event in f.targets)]  # fmt: skip
    assert len(engine_calls) == 4
    for fn in engine_calls:
        assert fn.concurrency_limit == 1 and fn.concurrency_id == "engine", fn.api_name
    others = [f for f in fns if f not in engine_calls]
    assert others and all(f.concurrency_id != "engine" for f in others)


def test_hub_telemetry_is_forced_off_even_after_the_hub_was_imported(monkeypatch):
    constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", False)
    for key in ("GRADIO_ANALYTICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        monkeypatch.delenv(key, raising=False)
    ui._disable_telemetry()
    assert constants.HF_HUB_DISABLE_TELEMETRY is True
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1" and os.environ["DO_NOT_TRACK"] == "1"
