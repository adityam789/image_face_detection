"""
tests/search/test_search_llm.py
Local LLM backends and auto-selection, tested against stub ``mlx_lm`` / ``llama_cpp`` modules.

The default run needs no network, model weights or real runtime. Set FACESORTER_MODEL_TESTS=1
to also run the smoke test against the real MLX model already in the Hugging Face cache.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import types

import pytest

from facesorter.search import llm as llm_pkg
from facesorter.search.errors import ModelUnavailableError
from facesorter.search.llm import (
    DEFAULT_MODELS,
    LlamaCppBackend,
    MLXBackend,
    _hub,
    available_backends,
    load_llm,
)
from facesorter.search.llm._base import normalise_messages, strip_thinking
from facesorter.search.types import LLMBackend

MESSAGES = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "Bob at the beach last summer"},
]
REPLY = '{"text": "beach", "place": null, "people": ["Bob"], "date_phrase": "last summer"}'


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(llm_pkg.BACKEND_ENV, raising=False)
    monkeypatch.delenv(llm_pkg.MODEL_ENV, raising=False)


# --------------------------------------------------------------------------- stubs


class StubTokenizer:
    def __init__(self, chat_template="plain template"):
        self.chat_template = chat_template
        self.template_calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        self.template_calls.append((list(messages), add_generation_prompt, kwargs))
        return [1, 2, 3]


class StubMLX:
    """Injects fake ``mlx_lm`` and ``mlx.core`` modules and records what they were asked."""

    def __init__(self, monkeypatch, chat_template="plain template", reply=REPLY, delay=0.0):
        self.tokenizer = StubTokenizer(chat_template)
        self.reply = reply
        self.delay = delay
        self.loads = []
        self.generates = []
        self.cache_cleared = 0
        self.load_error = None
        self.generate_error = None

        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.load = self._load
        mlx_lm.generate = self._generate
        mlx_pkg = types.ModuleType("mlx")
        core = types.ModuleType("mlx.core")
        core.clear_cache = self._clear_cache
        mlx_pkg.core = core
        monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
        monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", core)

    def _load(self, model):
        self.loads.append(model)
        if self.delay:
            time.sleep(self.delay)
        if self.load_error:
            raise self.load_error
        return "model-object", self.tokenizer

    def _generate(self, model, tokenizer, prompt=None, max_tokens=None, verbose=None, **kw):
        if self.generate_error:
            raise self.generate_error
        self.generates.append(
            {"model": model, "prompt": prompt, "max_tokens": max_tokens, "verbose": verbose, **kw}
        )
        return self.reply

    def _clear_cache(self):
        self.cache_cleared += 1


class StubLlama:
    """A fake ``llama_cpp`` module; ``Llama`` instances are recorded in ``instances``."""

    def __init__(self, monkeypatch, reply=REPLY, gpu=True, with_gpu_probe=True):
        outer = self
        self.instances = []
        self.pretrained_calls = []
        self.fail_with = None

        class Llama:
            def __init__(self, model_path=None, **kwargs):
                if outer.fail_with:
                    raise outer.fail_with
                self.model_path = model_path
                self.kwargs = kwargs
                self.closed = False
                self.chat_calls = []
                outer.instances.append(self)

            @classmethod
            def from_pretrained(cls, repo_id, filename, **kwargs):
                outer.pretrained_calls.append((repo_id, filename, kwargs))
                return cls(model_path=f"/cache/{repo_id}/{filename}", **kwargs)

            def create_chat_completion(self, messages, max_tokens, temperature):
                self.chat_calls.append((messages, max_tokens, temperature))
                return {"choices": [{"message": {"content": reply}}]}

            def close(self):
                self.closed = True

        mod = types.ModuleType("llama_cpp")
        mod.Llama = Llama
        if with_gpu_probe:
            mod.llama_supports_gpu_offload = lambda: gpu
        monkeypatch.setitem(sys.modules, "llama_cpp", mod)


def block_import(monkeypatch, name):
    """Make ``import name`` raise ImportError."""
    monkeypatch.setitem(sys.modules, name, None)


# --------------------------------------------------------------------------- helpers


def test_strip_thinking():
    assert strip_thinking("<think>\n\n</think>\n\n{}") == "{}"
    assert strip_thinking("<think>reasoning\nmore</think> answer ") == "answer"
    assert strip_thinking('  {"a": 1} ') == '{"a": 1}'
    assert strip_thinking("x <think>kept</think>") == "x <think>kept</think>"


def test_normalise_messages():
    assert normalise_messages([{"role": "user", "content": "hi", "extra": "x"}]) == [
        {"role": "user", "content": "hi"}
    ]
    with pytest.raises(ValueError, match="at least one"):
        normalise_messages([])
    with pytest.raises(ValueError, match="content"):
        normalise_messages([{"role": "user"}])


# --------------------------------------------------------------------------- MLX


def test_mlx_is_an_llm_backend():
    backend = MLXBackend()
    assert isinstance(backend, LLMBackend)
    assert backend.name == "mlx:mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    assert "not loaded" in repr(backend)


def test_mlx_construction_is_lazy(monkeypatch):
    stub = StubMLX(monkeypatch)
    backend = MLXBackend("some/model")
    assert stub.loads == []
    assert not backend.loaded


def test_mlx_chat_formats_prompt_and_generates_greedily(monkeypatch):
    stub = StubMLX(monkeypatch)
    backend = MLXBackend("some/model")
    assert backend.chat(MESSAGES, max_tokens=77) == REPLY
    assert stub.loads == ["some/model"]
    ((messages, add_gen, kwargs),) = stub.tokenizer.template_calls
    assert messages == MESSAGES and add_gen is True and kwargs == {}
    (gen,) = stub.generates
    assert gen == {
        "model": "model-object",
        "prompt": [1, 2, 3],
        "max_tokens": 77,
        "verbose": False,
    }
    assert "sampler" not in gen and "temp" not in gen  # default mlx_lm decoding is greedy
    assert backend.loaded and "loaded" in repr(backend)


def test_mlx_default_max_tokens(monkeypatch):
    stub = StubMLX(monkeypatch)
    MLXBackend().chat(MESSAGES)
    assert stub.generates[0]["max_tokens"] == 160


def test_mlx_loads_once_across_calls(monkeypatch):
    stub = StubMLX(monkeypatch)
    backend = MLXBackend("m")
    backend.chat(MESSAGES)
    backend.chat(MESSAGES)
    assert len(stub.loads) == 1 and len(stub.generates) == 2


def test_mlx_qwen3_template_disables_thinking(monkeypatch):
    stub = StubMLX(monkeypatch, chat_template="{% if enable_thinking %}...{% endif %}")
    MLXBackend("mlx-community/Qwen3-1.7B-4bit").chat(MESSAGES)
    assert stub.tokenizer.template_calls[0][2] == {"enable_thinking": False}


def test_mlx_enable_thinking_only_when_template_accepts_it(monkeypatch):
    for template in ("plain", None, {"default": "x", "tool_use": "y"}):
        stub = StubMLX(monkeypatch, chat_template=template)
        MLXBackend("m").chat(MESSAGES)
        assert stub.tokenizer.template_calls[0][2] == {}
    stub = StubMLX(monkeypatch, chat_template={"default": "{{ enable_thinking }}"})
    MLXBackend("m").chat(MESSAGES)
    assert stub.tokenizer.template_calls[0][2] == {"enable_thinking": False}


def test_mlx_strips_think_block_from_reply(monkeypatch):
    StubMLX(monkeypatch, reply="<think>\n\n</think>\n\n" + REPLY)
    assert MLXBackend("m").chat(MESSAGES) == REPLY


def test_mlx_missing_runtime_maps_to_model_unavailable(monkeypatch):
    block_import(monkeypatch, "mlx_lm")
    with pytest.raises(ModelUnavailableError, match=r"facesorter\[search-mlx\].*Python >= 3\.10"):
        MLXBackend("m").chat(MESSAGES)


def test_mlx_load_failure_maps_to_model_unavailable_and_retries(monkeypatch):
    stub = StubMLX(monkeypatch)
    stub.load_error = OSError("repo not found")
    backend = MLXBackend("bad/model")
    with pytest.raises(ModelUnavailableError, match="bad/model.*repo not found") as info:
        backend.chat(MESSAGES)
    assert isinstance(info.value.__cause__, OSError)
    assert not backend.loaded
    stub.load_error = None  # a transient failure (e.g. network) is retried on the next call
    assert backend.chat(MESSAGES) == REPLY


def test_mlx_generation_failure_maps_to_model_unavailable(monkeypatch):
    stub = StubMLX(monkeypatch)
    stub.generate_error = RuntimeError("out of memory")
    with pytest.raises(ModelUnavailableError, match="out of memory"):
        MLXBackend("m").chat(MESSAGES)


def test_typed_errors_from_the_runtime_pass_through(monkeypatch):
    stub = StubMLX(monkeypatch)
    stub.generate_error = ModelUnavailableError("already typed")
    with pytest.raises(ModelUnavailableError, match="^already typed$"):
        MLXBackend("m").chat(MESSAGES)


def test_chat_rejects_bad_arguments(monkeypatch):
    stub = StubMLX(monkeypatch)
    backend = MLXBackend("m")
    with pytest.raises(ValueError):
        backend.chat([])
    with pytest.raises(ValueError):
        backend.chat(MESSAGES, max_tokens=0)
    assert stub.loads == []  # bad input does not trigger a load


def test_mlx_close_releases_and_chat_reloads(monkeypatch):
    stub = StubMLX(monkeypatch)
    backend = MLXBackend("m")
    backend.chat(MESSAGES)
    backend.close()
    assert not backend.loaded and stub.cache_cleared == 1
    backend.close()  # idempotent
    assert stub.cache_cleared == 1
    assert backend.chat(MESSAGES) == REPLY  # documented: chat after close reloads
    assert len(stub.loads) == 2


def test_close_before_load_is_a_noop(monkeypatch):
    stub = StubMLX(monkeypatch)
    MLXBackend("m").close()
    assert stub.loads == [] and stub.cache_cleared == 0


def test_mlx_context_manager_closes(monkeypatch):
    stub = StubMLX(monkeypatch)
    with MLXBackend("m") as backend:
        backend.chat(MESSAGES)
        assert backend.loaded
    assert not backend.loaded and stub.cache_cleared == 1


def test_mlx_close_survives_cache_failure(monkeypatch):
    stub = StubMLX(monkeypatch)
    backend = MLXBackend("m")
    backend.chat(MESSAGES)
    sys.modules["mlx.core"].clear_cache = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    backend.close()  # must not raise
    assert not backend.loaded and stub.loads == ["m"]


def test_mlx_release_memory_falls_back_to_metal_namespace(monkeypatch):
    StubMLX(monkeypatch)
    core = sys.modules["mlx.core"]
    del core.clear_cache
    cleared = []
    core.metal = types.SimpleNamespace(clear_cache=lambda: cleared.append(1))
    backend = MLXBackend("m")
    backend.chat(MESSAGES)
    backend.close()
    assert cleared == [1]


def test_mlx_release_memory_without_any_clear_function(monkeypatch):
    StubMLX(monkeypatch)
    core = sys.modules["mlx.core"]
    del core.clear_cache
    core.metal = types.SimpleNamespace()
    backend = MLXBackend("m")
    backend.chat(MESSAGES)
    backend.close()  # nothing to call, nothing raised
    assert not backend.loaded


def test_lazy_load_is_thread_safe(monkeypatch):
    stub = StubMLX(monkeypatch, delay=0.2)
    backend = MLXBackend("m")
    start = threading.Barrier(2)
    results, errors = [], []

    def worker():
        try:
            start.wait()
            results.append(backend.chat(MESSAGES))
        except Exception as exc:  # pragma: no cover - failure path of the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert results == [REPLY, REPLY]
    assert len(stub.loads) == 1


def test_generation_is_serialised(monkeypatch):
    stub = StubMLX(monkeypatch)
    active, peak = [0], [0]
    guard = threading.Lock()

    def slow_generate(model, tokenizer, **kwargs):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return REPLY

    sys.modules["mlx_lm"].generate = slow_generate
    backend = MLXBackend("m")
    threads = [threading.Thread(target=backend.chat, args=(MESSAGES,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert peak[0] == 1 and len(stub.loads) == 1


# --------------------------------------------------------------------------- llama.cpp


def test_llama_default_model_is_official_qwen_q4_k_m():
    backend = LlamaCppBackend()
    assert isinstance(backend, LLMBackend)
    assert backend.repo_id == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    assert backend.filename == "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    assert (
        backend.name
        == "llama-cpp:Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    assert (
        DEFAULT_MODELS["llama-cpp"]
        == "Qwen/Qwen2.5-1.5B-Instruct-GGUF:qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )


def test_llama_constructor_validation():
    with pytest.raises(ValueError, match="either"):
        LlamaCppBackend("a.gguf", repo_id="x/y")
    with pytest.raises(ValueError, match="repo_id"):
        LlamaCppBackend(filename="*.gguf")
    backend = LlamaCppBackend(repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    assert backend.filename == "*[qQ]4_[kK]_[mM].gguf"


def test_llama_local_path(monkeypatch, tmp_path):
    stub = StubLlama(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    backend = LlamaCppBackend(str(gguf), n_ctx=1024)
    assert stub.instances == []  # lazy
    assert backend.chat(MESSAGES, max_tokens=50) == REPLY
    (inst,) = stub.instances
    assert inst.model_path == str(gguf)
    assert inst.kwargs == {"n_ctx": 1024, "n_gpu_layers": -1, "verbose": False}
    assert inst.chat_calls == [(MESSAGES, 50, 0.0)]  # greedy: temperature 0
    assert stub.pretrained_calls == []


def test_llama_from_pretrained(monkeypatch):
    stub = StubLlama(monkeypatch)
    backend = LlamaCppBackend(repo_id="org/repo", filename="*q4_k_m.gguf")
    assert backend.chat(MESSAGES) == REPLY
    assert stub.pretrained_calls == [
        ("org/repo", "*q4_k_m.gguf", {"n_ctx": 2048, "n_gpu_layers": -1, "verbose": False})
    ]


def test_llama_gpu_layers_selection(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    stub = StubLlama(monkeypatch, gpu=False)
    LlamaCppBackend(str(gguf)).chat(MESSAGES)
    assert stub.instances[0].kwargs["n_gpu_layers"] == 0
    stub = StubLlama(monkeypatch, with_gpu_probe=False)  # old build without the probe
    LlamaCppBackend(str(gguf)).chat(MESSAGES)
    assert stub.instances[0].kwargs["n_gpu_layers"] == 0
    stub = StubLlama(monkeypatch, gpu=False)
    LlamaCppBackend(str(gguf), n_gpu_layers=12).chat(MESSAGES)  # explicit value wins
    assert stub.instances[0].kwargs["n_gpu_layers"] == 12


def test_llama_gpu_probe_failure_falls_back_to_cpu(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    stub = StubLlama(monkeypatch)
    sys.modules["llama_cpp"].llama_supports_gpu_offload = lambda: 1 / 0
    LlamaCppBackend(str(gguf)).chat(MESSAGES)
    assert stub.instances[0].kwargs["n_gpu_layers"] == 0


def test_llama_missing_runtime(monkeypatch):
    block_import(monkeypatch, "llama_cpp")
    with pytest.raises(ModelUnavailableError, match=r"facesorter\[search-llama\].*3\.10"):
        LlamaCppBackend().chat(MESSAGES)


def test_llama_missing_file_maps_to_model_unavailable(monkeypatch, tmp_path):
    StubLlama(monkeypatch)
    backend = LlamaCppBackend(str(tmp_path / "absent.gguf"))
    with pytest.raises(ModelUnavailableError, match="absent.gguf"):
        backend.chat(MESSAGES)
    assert not backend.loaded


def test_llama_load_error_maps_to_model_unavailable(monkeypatch):
    stub = StubLlama(monkeypatch)
    stub.fail_with = ValueError("Failed to load model from file")
    with pytest.raises(ModelUnavailableError, match="Failed to load"):
        LlamaCppBackend(repo_id="org/repo", filename="x.gguf").chat(MESSAGES)


def test_llama_empty_content_and_generation_error(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    stub = StubLlama(monkeypatch, reply=None)
    backend = LlamaCppBackend(str(gguf))
    assert backend.chat(MESSAGES) == ""
    stub.instances[0].create_chat_completion = lambda **kw: (_ for _ in ()).throw(
        RuntimeError("context overflow")
    )
    with pytest.raises(ModelUnavailableError, match="context overflow"):
        backend.chat(MESSAGES)


def test_llama_close_context_manager_and_reload(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    stub = StubLlama(monkeypatch)
    with LlamaCppBackend(str(gguf)) as backend:
        backend.chat(MESSAGES)
    assert stub.instances[0].closed and not backend.loaded
    assert backend.chat(MESSAGES) == REPLY
    assert len(stub.instances) == 2


def test_llama_close_swallows_runtime_errors(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    stub = StubLlama(monkeypatch)
    backend = LlamaCppBackend(str(gguf))
    backend.chat(MESSAGES)
    stub.instances[0].close = lambda: 1 / 0
    backend.close()
    assert not backend.loaded


def test_llama_lazy_load_is_thread_safe(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    stub = StubLlama(monkeypatch)
    original_init = sys.modules["llama_cpp"].Llama.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(0.2)
        original_init(self, *args, **kwargs)

    sys.modules["llama_cpp"].Llama.__init__ = slow_init
    backend = LlamaCppBackend(str(gguf))
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        backend.chat(MESSAGES)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(stub.instances) == 1


@pytest.mark.parametrize(
    ("spec", "expect"),
    [
        ("models/qwen.gguf", {"model_path": "models/qwen.gguf"}),
        ("C:\\models\\qwen.gguf", {"model_path": "C:\\models\\qwen.gguf"}),
        ("C:/models/qwen", {"model_path": "C:/models/qwen"}),
        ("org/repo:file-*.gguf", {"repo_id": "org/repo", "filename": "file-*.gguf"}),
        ("org/repo:", {"repo_id": "org/repo", "filename": "*[qQ]4_[kK]_[mM].gguf"}),
        ("org/repo", {"repo_id": "org/repo", "filename": "*[qQ]4_[kK]_[mM].gguf"}),
    ],
)
def test_llama_from_spec(spec, expect):
    backend = LlamaCppBackend.from_spec(spec)
    for key, value in expect.items():
        assert getattr(backend, key) == value


def test_llama_from_spec_existing_file(tmp_path):
    path = tmp_path / "weights"  # no .gguf suffix, but it exists
    path.write_bytes(b"x")
    assert LlamaCppBackend.from_spec(str(path)).model_path == str(path)


# --------------------------------------------------------------------------- hub notice


def test_download_notice_logged_when_not_cached(monkeypatch, caplog):
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda repo, filename: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    StubMLX(monkeypatch)
    with caplog.at_level(logging.INFO, logger="facesorter.search.llm"):
        MLXBackend().chat(MESSAGES)
    notices = [r.getMessage() for r in caplog.records if "Downloading" in r.getMessage()]
    assert len(notices) == 1 and "0.9 GB" in notices[0] and "Qwen2.5-1.5B" in notices[0]


def test_download_notice_for_gguf_names_file_and_size(monkeypatch, caplog):
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda repo, filename: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    StubLlama(monkeypatch)
    with caplog.at_level(logging.INFO, logger="facesorter.search.llm"):
        LlamaCppBackend().chat(MESSAGES)
    notice = next(r.getMessage() for r in caplog.records if "Downloading" in r.getMessage())
    assert "qwen2.5-1.5b-instruct-q4_k_m.gguf" in notice and "1.1 GB" in notice


def test_no_download_notice_when_cached_or_local(monkeypatch, caplog, tmp_path):
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda repo, filename: str(tmp_path / filename)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    StubMLX(monkeypatch)
    StubLlama(monkeypatch)
    with caplog.at_level(logging.INFO, logger="facesorter.search.llm"):
        MLXBackend().chat(MESSAGES)
        LlamaCppBackend().chat(MESSAGES)
        MLXBackend(str(tmp_path)).chat(MESSAGES)  # a local directory is never downloaded
    assert not [r for r in caplog.records if "Downloading" in r.getMessage()]


def test_hub_is_cached_edge_cases(monkeypatch):
    block_import(monkeypatch, "huggingface_hub")
    assert _hub.is_cached("a/b") is None
    _hub.announce_download("a/b", "1 GB")  # no hub, no crash, no notice
    hub = types.ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda repo, filename: (_ for _ in ()).throw(RuntimeError())
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert _hub.is_cached("a/b", "*.gguf") is None  # a glob cannot be checked
    assert _hub.is_cached("a/b", "f.gguf") is None  # unexpected error: unknown
    hub.try_to_load_from_cache = lambda repo, filename: f"/cache/{filename}"
    assert _hub.is_cached("a/b") is True  # no filename: config.json stands in for the repo


# --------------------------------------------------------------------------- selection


def set_platform(monkeypatch, system, machine):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)


def set_installed(monkeypatch, mlx_lm, llama_cpp):
    for name, present in (("mlx_lm", mlx_lm), ("llama_cpp", llama_cpp)):
        if present:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        else:
            monkeypatch.setitem(sys.modules, name, None)


PLATFORMS = {
    "mac-arm": ("Darwin", "arm64"),
    "mac-intel": ("Darwin", "x86_64"),
    "linux": ("Linux", "x86_64"),
    "windows": ("Windows", "AMD64"),
}


@pytest.mark.parametrize("plat", sorted(PLATFORMS))
@pytest.mark.parametrize("mlx_lm", [False, True])
@pytest.mark.parametrize("llama_cpp", [False, True])
def test_auto_selection_matrix(monkeypatch, plat, mlx_lm, llama_cpp):
    set_platform(monkeypatch, *PLATFORMS[plat])
    set_installed(monkeypatch, mlx_lm, llama_cpp)
    expect_mlx = plat == "mac-arm" and mlx_lm
    assert available_backends() == {"mlx": expect_mlx, "llama-cpp": llama_cpp}
    backend = load_llm()
    if expect_mlx:
        assert isinstance(backend, MLXBackend)
    elif llama_cpp:
        assert isinstance(backend, LlamaCppBackend)
    else:
        assert backend is None  # rules-only mode is not an error


def test_load_llm_is_lazy(monkeypatch):
    set_platform(monkeypatch, "Darwin", "arm64")
    stub = StubMLX(monkeypatch)
    backend = load_llm("mlx")
    assert isinstance(backend, MLXBackend) and not backend.loaded and stub.loads == []


def test_availability_check_does_not_import_runtimes(monkeypatch):
    for name in ("mlx_lm", "llama_cpp"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: types.SimpleNamespace() if name else None
    )
    set_platform(monkeypatch, "Darwin", "arm64")
    assert available_backends() == {"mlx": True, "llama-cpp": True}
    assert "mlx_lm" not in sys.modules and "llama_cpp" not in sys.modules


def test_availability_handles_broken_find_spec(monkeypatch):
    for name in ("mlx_lm", "llama_cpp"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def boom(name):
        raise ValueError("spec is None")

    monkeypatch.setattr("importlib.util.find_spec", boom)
    assert available_backends()["llama-cpp"] is False


def test_none_backend_returns_none(monkeypatch):
    set_platform(monkeypatch, "Darwin", "arm64")
    set_installed(monkeypatch, True, True)
    assert load_llm("none") is None
    assert load_llm("off") is None


def test_explicit_mlx_unavailable_gives_install_hint(monkeypatch):
    set_platform(monkeypatch, "Darwin", "arm64")
    set_installed(monkeypatch, False, True)
    with pytest.raises(ModelUnavailableError, match=r"facesorter\[search-mlx\].*3\.10"):
        load_llm("mlx")


def test_explicit_mlx_on_wrong_platform_suggests_llama(monkeypatch):
    set_platform(monkeypatch, "Linux", "x86_64")
    set_installed(monkeypatch, True, False)
    with pytest.raises(ModelUnavailableError, match=r"Apple silicon.*search-llama.*3\.10"):
        load_llm("mlx")


def test_explicit_llama_unavailable_gives_install_hint(monkeypatch):
    set_platform(monkeypatch, "Linux", "x86_64")
    set_installed(monkeypatch, False, False)
    with pytest.raises(ModelUnavailableError, match=r"facesorter\[search-llama\].*3\.10"):
        load_llm("llama-cpp")


@pytest.mark.parametrize("alias", ["llama_cpp", "llamacpp", "LLAMA-CPP", " gguf "])
def test_backend_aliases(monkeypatch, alias):
    set_installed(monkeypatch, False, True)
    assert isinstance(load_llm(alias), LlamaCppBackend)


def test_unknown_backend_is_a_typed_error():
    with pytest.raises(ModelUnavailableError, match="Unknown LLM backend 'ollama'"):
        load_llm("ollama")


def test_default_models_used_when_no_model_given(monkeypatch):
    set_platform(monkeypatch, "Darwin", "arm64")
    set_installed(monkeypatch, True, True)
    assert load_llm("mlx").model == DEFAULT_MODELS["mlx"]
    llama = load_llm("llama-cpp")
    assert (llama.repo_id, llama.filename) == (
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )


def test_model_argument(monkeypatch):
    set_platform(monkeypatch, "Darwin", "arm64")
    set_installed(monkeypatch, True, True)
    assert load_llm("mlx", "mlx-community/Qwen3-1.7B-4bit").model == "mlx-community/Qwen3-1.7B-4bit"
    assert load_llm("llama-cpp", "/models/q.gguf").model_path == "/models/q.gguf"


def test_env_overrides(monkeypatch):
    set_platform(monkeypatch, "Darwin", "arm64")
    set_installed(monkeypatch, True, True)
    monkeypatch.setenv(llm_pkg.BACKEND_ENV, "llama-cpp")
    monkeypatch.setenv(llm_pkg.MODEL_ENV, "org/repo:f.gguf")
    backend = load_llm()
    assert isinstance(backend, LlamaCppBackend)
    assert (backend.repo_id, backend.filename) == ("org/repo", "f.gguf")
    monkeypatch.setenv(llm_pkg.BACKEND_ENV, "none")
    assert load_llm() is None
    monkeypatch.setenv(llm_pkg.BACKEND_ENV, "mlx")
    monkeypatch.setenv(llm_pkg.MODEL_ENV, "my/mlx-model")
    assert load_llm().model == "my/mlx-model"


def test_explicit_arguments_win_over_env(monkeypatch):
    set_platform(monkeypatch, "Darwin", "arm64")
    set_installed(monkeypatch, True, True)
    monkeypatch.setenv(llm_pkg.BACKEND_ENV, "llama-cpp")
    monkeypatch.setenv(llm_pkg.MODEL_ENV, "env/model")
    backend = load_llm("mlx", "arg/model")
    assert isinstance(backend, MLXBackend) and backend.model == "arg/model"
    assert load_llm("none") is None  # explicit "none" beats an env backend too


def test_bad_env_backend_is_a_typed_error(monkeypatch):
    monkeypatch.setenv(llm_pkg.BACKEND_ENV, "bogus")
    with pytest.raises(ModelUnavailableError, match=llm_pkg.BACKEND_ENV):
        load_llm()


def test_empty_env_values_are_ignored(monkeypatch):
    set_platform(monkeypatch, "Linux", "x86_64")
    set_installed(monkeypatch, False, True)
    monkeypatch.setenv(llm_pkg.BACKEND_ENV, "")
    monkeypatch.setenv(llm_pkg.MODEL_ENV, "")
    backend = load_llm()
    assert (
        isinstance(backend, LlamaCppBackend)
        and backend.repo_id == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    )


# --------------------------------------------------------------------------- real model (gated)

real_models = pytest.mark.skipif(
    os.environ.get("FACESORTER_MODEL_TESTS") != "1",
    reason="set FACESORTER_MODEL_TESTS=1 to run against the real MLX model",
)


@real_models
def test_real_mlx_smoke():
    pytest.importorskip("mlx_lm")
    system = (
        'Return ONLY a JSON object with keys "text", "place" (string or null), "people" '
        '(list of first names) and "date_phrase" (copied verbatim from the query, or null).'
    )
    shots = [
        {"role": "system", "content": system},
        {"role": "user", "content": "pizza night with Tom in Rome in April 2022"},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "text": "pizza night",
                    "place": "Rome",
                    "people": ["Tom"],
                    "date_phrase": "April 2022",
                }
            ),
        },
    ]
    queries = ["Bob and Alice hiking last summer", "cats", "sunset in Lisbon in May 2021"]
    with MLXBackend() as backend:
        for query in queries:
            reply = backend.chat([*shots, {"role": "user", "content": query}], max_tokens=120)
            data = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
            assert isinstance(data, dict) and {"text", "place", "people", "date_phrase"} <= set(
                data
            )
        # Called from another thread than the one that loaded the model.
        out = []
        t = threading.Thread(
            target=lambda: out.append(
                backend.chat(shots[:2] + [{"role": "user", "content": "dogs"}], max_tokens=60)
            )
        )
        t.start()
        t.join(timeout=120)
        assert out and "{" in out[0]
