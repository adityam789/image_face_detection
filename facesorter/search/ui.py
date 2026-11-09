"""
facesorter/search/ui.py
The local Gradio front end for photo search.

``build_app`` assembles the page and ``serve`` runs it. Handlers are thin: they call
``ui_logic`` and shape the answer into widget updates.

Security and privacy choices, all deliberate:

* Only the thumbnail cache directory is passed to Gradio's ``allowed_paths``. Library roots
  are never served and original photos never reach the browser; file paths are shown as
  text only. The thumbnail directory must not overlap the library.
* The server binds to 127.0.0.1 by default and warns loudly for any other host.
* ``share`` is always ``False`` and Gradio analytics are switched off.
* The queue is on and engine calls run one at a time (SQLite and a local LLM are not made
  for parallel use), so a slow LLM call never freezes the page.

``gradio`` needs Python 3.10+, so it is imported only inside the functions here.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ModelUnavailableError
from .ui_logic import (
    DEFAULT_K,
    MAX_K,
    HostGuardMiddleware,
    HostPolicy,
    SearchParams,
    Session,
    ThumbnailCache,
    UIResult,
    answer_clarifications,
    default_thumbs_dir,
    ensure_outside_library,
    index_summary,
    is_loopback_host,
    md_escape,
    record_feedback,
    skip_clarifications,
    start_search,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gradio as gr

logger = logging.getLogger(__name__)

__all__ = ["build_app", "launch_app", "serve"]

MAX_QUESTIONS = 6  # clarification text boxes; further questions are skipped
EXAMPLE_QUERY = "Bob at the beach last summer in Lisbon"


def _disable_telemetry() -> None:
    """Switch off Gradio analytics and the Hugging Face Hub usage ping.

    Gradio's own flag is read from the environment, but ``huggingface_hub`` reads its flag
    once at import time and Gradio sends it a ping on launch, so a variable set now is too
    late if the hub was already imported (the search stack imports it). Patch that module's
    constant as well.
    """
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None:
        hub_constants.HF_HUB_DISABLE_TELEMETRY = True


def _use_private_gradio_cache(thumbs_dir: Path) -> Path:
    """Point Gradio's file cache at a folder inside the (0700) thumbnail directory.

    Gradio copies every file a handler returns into its cache, by default a world-readable
    folder under the system temp directory. Gradio reads ``GRADIO_TEMP_DIR`` when components
    are created and when the server starts, so this is called before both.
    """
    cache = Path(thumbs_dir) / ".gradio"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(cache)
    return cache


def _import_gradio() -> Any:
    """Import gradio with telemetry off."""
    _disable_telemetry()
    try:
        import gradio
    except ImportError as exc:
        raise ModelUnavailableError(
            "The search UI needs gradio, which requires Python 3.10 or newer. "
            'Install it with: pip install "facesorter[search]"'
        ) from exc
    return gradio


def _has_llm(engine: Any) -> bool:
    """Whether the engine can parse queries with an LLM (unknown counts as yes)."""
    for holder in (engine, getattr(engine, "parser", None)):
        if holder is None:
            continue
        for name in ("has_llm", "llm_available"):
            value = getattr(holder, name, None)
            if isinstance(value, bool):
                return value
        for name in ("llm", "_llm"):
            if hasattr(holder, name):
                return getattr(holder, name) is not None
    return True


def _known_names(engine: Any) -> list[str]:
    try:
        return list(engine.store.known_names())
    except Exception:
        logger.debug("known_names failed", exc_info=True)
        return []


def _lines(items: Sequence[str]) -> str:
    return "\n\n".join(md_escape(x) for x in items)


def _details_markdown(card: Any) -> str:
    rows = [f"**{md_escape(label)}:** {md_escape(value)}" for label, value in card.details()]
    return "\n\n".join(rows)


class _Handlers:
    """The callbacks behind the widgets (kept as plain methods so tests can call them)."""

    def __init__(self, gr: Any, engine: Any, thumbs: ThumbnailCache) -> None:
        self.gr = gr
        self.engine = engine
        self.thumbs = thumbs

    # The order of ``render`` outputs; ``build_app`` wires the components in the same order.
    # session, status, reading, notices, dropped, clarify box, questions..., gallery,
    # detail box, detail text, feedback message

    def render(self, session: Session, result: UIResult) -> tuple[Any, ...]:
        gr = self.gr
        asking = result.needs_clarification
        reading = ""
        if result.reading:
            reading = _lines(result.reading.splitlines())
        notices = _lines(result.notices)
        dropped = _lines(result.dropped)
        questions = []
        for i in range(MAX_QUESTIONS):
            if asking and i < len(result.questions):
                phrase, question = result.questions[i]
                questions.append(
                    gr.update(
                        visible=True,
                        value="",
                        label=question or f'What did you mean by "{phrase}"?',
                        placeholder="Reword it, or leave blank to skip",
                    )
                )
            else:
                questions.append(gr.update(visible=False, value=""))
        session.selected = None
        gallery = gr.update(value=result.gallery_items() if not asking else [])
        status = md_escape(result.status)
        if asking and len(result.questions) > MAX_QUESTIONS:
            status += " " + md_escape(
                f"Only the first {MAX_QUESTIONS} questions are shown; the rest will be skipped."
            )
        return (
            session,
            status,
            gr.update(value=reading, visible=bool(reading)),
            gr.update(value=notices, visible=bool(notices)),
            gr.update(value=dropped, visible=bool(dropped)),
            gr.update(visible=asking),
            *questions,
            gallery,
            gr.update(visible=False),
            "",
            "",
        )

    def search(
        self,
        query: str,
        date_from: str,
        date_to: str,
        place: str,
        people: Sequence[str] | None,
        k: Any,
        use_llm: bool,
    ) -> tuple[Any, ...]:
        params = SearchParams(
            query=query or "",
            date_from=date_from or "",
            date_to=date_to or "",
            place=place or "",
            people=tuple(people or ()),
            k=k if k is not None else DEFAULT_K,
            use_llm=bool(use_llm),
        )
        session = Session()
        result = start_search(self.engine, session, params, thumbs=self.thumbs)
        return self.render(session, result)

    def answer(self, session: Session | None, *replies: str) -> tuple[Any, ...]:
        session = session or Session()
        result = answer_clarifications(
            self.engine, session, [r or "" for r in replies], thumbs=self.thumbs
        )
        return self.render(session, result)

    def skip(self, session: Session | None) -> tuple[Any, ...]:
        session = session or Session()
        result = skip_clarifications(self.engine, session, thumbs=self.thumbs)
        return self.render(session, result)

    def select(self, session: Session | None, evt: Any) -> tuple[Any, ...]:
        gr = self.gr
        cards = session.result.gallery_cards() if session and session.result else []
        index = getattr(evt, "index", None)
        if isinstance(index, (list, tuple)):  # (row, column) from some components
            index = index[0] if index else None
        if session is None or not isinstance(index, int) or not 0 <= index < len(cards):
            return session, gr.update(visible=False), "", ""
        session.selected = cards[index]
        return session, gr.update(visible=True), _details_markdown(cards[index]), ""

    def rate(self, session: Session | None, relevant: bool) -> str:
        if session is None or session.selected is None:
            return md_escape("Select a photo first.")
        message = record_feedback(self.engine, session.query, session.selected.photo_id, relevant)
        return md_escape(message)

    def load(self) -> tuple[Any, ...]:
        gr = self.gr
        return gr.update(choices=_known_names(self.engine)), md_escape(index_summary(self.engine))


def build_app(
    engine: Any,
    *,
    thumbs_dir: str | os.PathLike[str] | None = None,
    library_roots: Sequence[str] = (),
) -> gr.Blocks:
    """Build the search page for ``engine``.

    Thumbnails are cached in ``thumbs_dir`` (default ``~/.facesorter/thumbs``), which must
    not overlap ``library_roots``. Pass the same directory to :func:`launch_app` /
    :func:`serve`, which expose that directory and nothing else.
    """
    gr = _import_gradio()
    directory = ensure_outside_library(thumbs_dir or default_thumbs_dir(), library_roots)
    thumbs = ThumbnailCache(directory)
    _use_private_gradio_cache(thumbs.directory)
    h = _Handlers(gr, engine, thumbs)

    with gr.Blocks(
        title="facesorter photo search",
        analytics_enabled=False,
        delete_cache=(3600, 24 * 3600),  # Gradio keeps copies of served thumbnails; tidy them
    ) as demo:
        session = gr.State(None)
        gr.Markdown("# Photo search\nEverything runs on this computer.")
        with gr.Row():
            query = gr.Textbox(
                label="What are you looking for?",
                placeholder=EXAMPLE_QUERY,
                lines=1,
                max_lines=1,
                scale=5,
                autofocus=True,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)
        with gr.Accordion("Filters (optional, these override the query)", open=False):
            with gr.Row():
                date_from = gr.Textbox(label="Date from", placeholder="2023 or 2023-07")
                date_to = gr.Textbox(label="Date to", placeholder="2023 or 2023-07-14")
            place = gr.Textbox(label="Place", placeholder="Lisbon")
            people = gr.Dropdown(label="People", choices=_known_names(engine), multiselect=True)
            k = gr.Slider(label="Results", minimum=1, maximum=MAX_K, step=1, value=DEFAULT_K)
            use_llm = gr.Checkbox(
                label="Use the local language model to read my query",
                value=True,
                visible=_has_llm(engine),
            )

        status = gr.Markdown()
        with gr.Accordion("How I read your query", open=True):
            reading = gr.Markdown(visible=False)
            notices = gr.Markdown(visible=False)
        dropped = gr.Markdown(visible=False, elem_id="dropped-banner")

        with gr.Group(visible=False) as clarify_box:
            gr.Markdown("**I did not understand part of your query.** Reword it, or skip.")
            questions = [gr.Textbox(visible=False, max_lines=1) for _ in range(MAX_QUESTIONS)]
            with gr.Row():
                answer_btn = gr.Button("Search with these answers", variant="primary")
                skip_btn = gr.Button("Skip and search without")

        gallery = gr.Gallery(
            label="Results",
            columns=4,
            height="auto",
            object_fit="contain",
            allow_preview=False,
            show_label=False,
            interactive=False,
        )
        with gr.Group(visible=False) as detail_box:
            detail_text = gr.Markdown()
            with gr.Row():
                up_btn = gr.Button("👍 Relevant")
                down_btn = gr.Button("👎 Not relevant")
            feedback_msg = gr.Markdown()
        footer = gr.Markdown(elem_id="index-footer")

        # Engine calls share one concurrency group so they run one at a time.
        engine_call = {"concurrency_limit": 1, "concurrency_id": "engine"}
        search_inputs = [query, date_from, date_to, place, people, k, use_llm]
        render_outputs = [
            session,
            status,
            reading,
            notices,
            dropped,
            clarify_box,
            *questions,
            gallery,
            detail_box,
            detail_text,
            feedback_msg,
        ]
        search_btn.click(h.search, search_inputs, render_outputs, api_name="search", **engine_call)
        query.submit(h.search, search_inputs, render_outputs, api_name=False, **engine_call)
        answer_btn.click(
            h.answer, [session, *questions], render_outputs, api_name="answer", **engine_call
        )
        skip_btn.click(h.skip, [session], render_outputs, api_name="skip", **engine_call)

        def on_select(state: Any, evt: Any) -> tuple[Any, ...]:
            return h.select(state, evt)

        # Gradio injects the event data only when the annotation is the real class; the
        # module uses lazy annotations and gradio is imported here, so set it explicitly.
        on_select.__annotations__ = {"evt": gr.SelectData}
        gallery.select(
            on_select,
            [session],
            [session, detail_box, detail_text, feedback_msg],
            api_name=False,
        )
        up_btn.click(lambda s: h.rate(s, True), [session], [feedback_msg], api_name="rate_up")
        down_btn.click(lambda s: h.rate(s, False), [session], [feedback_msg], api_name="rate_down")
        demo.load(h.load, None, [people, footer], api_name=False)

    demo.queue()
    demo.facesorter_handlers = h  # type: ignore[attr-defined]  # used by tests
    return demo


def launch_app(
    demo: gr.Blocks,
    *,
    thumbs_dir: str | os.PathLike[str],
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = False,
    block: bool = True,
) -> Any:
    """Start ``demo`` serving only ``thumbs_dir`` (``block=False`` returns once it is up)."""
    if not is_loopback_host(host):
        message = (
            f"SECURITY WARNING: the search UI is binding to {host!r}, which is NOT loopback. "
            "Anyone who can reach this machine on the network can search and see your photo "
            "thumbnails, file names and folders. Use 127.0.0.1 unless you really mean it."
        )
        logger.warning(message)
        warnings.warn(message, stacklevel=2)
        print(message, file=sys.stderr, flush=True)
    thumbs_path = Path(thumbs_dir).expanduser().resolve()
    allowed = [str(thumbs_path)]
    _use_private_gradio_cache(thumbs_path)
    # Answer only requests addressed to this server (defence against DNS rebinding): the
    # guard is passed in as FastAPI middleware, which is in place before the server starts.
    policy = HostPolicy.for_bind(host, port)
    middleware = [_starlette_middleware(HostGuardMiddleware, policy=policy)]
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=open_browser,
        allowed_paths=allowed,
        prevent_thread_lock=not block,
        show_error=False,
        quiet=not block,
        ssr_mode=False,  # no separate Node front end: every request reaches the guard
        app_kwargs={"middleware": middleware},
    )
    policy.port = getattr(demo, "server_port", None) or policy.port
    return demo


def _starlette_middleware(cls: Any, **options: Any) -> Any:
    from starlette.middleware import Middleware

    return Middleware(cls, **options)


def serve(
    engine: Any,
    *,
    library_roots: Sequence[str],
    host: str = "127.0.0.1",
    port: int = 7860,
    thumbs_dir: Path | None = None,
    open_browser: bool = False,
) -> None:
    """Run the UI until interrupted (blocking). Called by ``facesorter serve``."""
    directory = ensure_outside_library(thumbs_dir or default_thumbs_dir(), library_roots)
    demo = build_app(engine, thumbs_dir=directory, library_roots=library_roots)
    try:
        launch_app(demo, thumbs_dir=directory, host=host, port=port, open_browser=open_browser)
    finally:
        demo.close()
