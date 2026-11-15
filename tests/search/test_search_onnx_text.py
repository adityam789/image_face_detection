"""Tests for facesorter.search.encoders.onnx_text with fake onnxruntime/tokenizers/hub."""

from __future__ import annotations

import builtins
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facesorter.search.encoders import onnx_text
from facesorter.search.encoders.onnx_text import CONTEXT_LENGTH, OnnxTextEncoder, canonicalize
from facesorter.search.errors import ModelUnavailableError

DIM = 768


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Maps each character to its code point; records what text it was given."""

    seen: list[str] = []
    from_file_calls: list[str] = []
    eos: int | None = 1

    @classmethod
    def from_file(cls, path: str) -> FakeTokenizer:
        cls.from_file_calls.append(path)
        return cls()

    def token_to_id(self, token: str) -> int | None:
        assert token == "<eos>"
        return FakeTokenizer.eos

    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        FakeTokenizer.seen.extend(texts)
        return [_Encoding([ord(c) for c in t] + [1]) for t in texts]  # 1 = EOS


class _Output:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    """Embedding = hash-like function of the ids; outputs fp16 like the real fp16 model."""

    instances: list[FakeSession] = []
    construct_delay = 0.0

    def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
        time.sleep(FakeSession.construct_delay)
        self.path = path
        self.providers = providers
        self.runs: list[np.ndarray] = []
        FakeSession.instances.append(self)

    def get_outputs(self) -> list[_Output]:
        return [_Output("last_hidden_state"), _Output("pooler_output")]

    def run(self, outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        ids = feed["input_ids"]
        assert ids.dtype == np.int64
        assert ids.shape[1] == CONTEXT_LENGTH
        self.runs.append(ids)
        pooled = np.zeros((ids.shape[0], DIM), dtype=np.float16)
        for row, seq in enumerate(ids):
            rng = np.random.default_rng(int(seq.sum()) + 7 * int(np.count_nonzero(seq)))
            pooled[row] = (rng.standard_normal(DIM) * 3).astype(np.float16)
        return [np.zeros((ids.shape[0], CONTEXT_LENGTH, 4), np.float16), pooled]


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.SimpleNamespace:
    """Install fake onnxruntime, tokenizers and huggingface_hub modules."""
    FakeTokenizer.seen = []
    FakeTokenizer.from_file_calls = []
    FakeTokenizer.eos = 1
    FakeSession.instances = []
    FakeSession.construct_delay = 0.0
    downloads: list[tuple[str, str, bool]] = []
    state = types.SimpleNamespace(cached=True, network_ok=True, downloads=downloads)

    def hf_hub_download(
        repo: str, filename: str, cache_dir: Any = None, local_files_only: bool = False
    ) -> str:
        downloads.append((repo, filename, local_files_only))
        if local_files_only and not state.cached:
            raise FileNotFoundError("not cached")
        if not local_files_only and not state.network_ok:
            raise ConnectionError("offline")
        return str(tmp_path / filename.replace("/", "_"))

    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = FakeSession  # type: ignore[attr-defined]
    ort.SessionOptions = types.SimpleNamespace  # type: ignore[attr-defined]
    tok = types.ModuleType("tokenizers")
    tok.Tokenizer = FakeTokenizer  # type: ignore[attr-defined]
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", tok)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return state


# --------------------------------------------------------------------------- canonicalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A Dog, on the Beach!", "a dog on the beach"),
        ("  two   people\tand\na cat  ", "two people and a cat"),
        ("sunset_over_the-sea", "sunset over thesea"),
        ("Bob's 3rd birthday (2019)", "bobs 3rd birthday 2019"),
        ('say "hi" & wave; ok?', "say hi wave ok"),
        ("ÉCOLE Café", "école café"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_canonicalize(raw: str, expected: str) -> None:
    assert canonicalize(raw) == expected


# --------------------------------------------------------------------------- encoding


def test_identity_attributes() -> None:
    enc = OnnxTextEncoder()
    assert enc.name == "siglip2-b16-webli"
    assert enc.dim == 768


def test_construction_is_lazy(fakes: types.SimpleNamespace) -> None:
    OnnxTextEncoder()
    assert fakes.downloads == []
    assert FakeSession.instances == []


def test_encode_returns_unit_float32(fakes: types.SimpleNamespace) -> None:
    enc = OnnxTextEncoder()
    out = enc.encode_text(["a red car", "a dog"])
    assert out.shape == (2, DIM)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(out[0], out[1])


def test_text_is_canonicalised_before_tokenising(fakes: types.SimpleNamespace) -> None:
    OnnxTextEncoder().encode_text(["A Dog_on the BEACH!!"])
    assert FakeTokenizer.seen == ["a dog on the beach"]


def test_ids_padded_with_zero_to_64(fakes: types.SimpleNamespace) -> None:
    enc = OnnxTextEncoder()
    enc.encode_text(["ab"])
    ids = FakeSession.instances[0].runs[0]
    assert ids.shape == (1, 64)
    assert ids[0, :3].tolist() == [ord("a"), ord("b"), 1]
    assert not ids[0, 3:].any()


def test_long_text_truncated_to_64_keeping_eos(fakes: types.SimpleNamespace) -> None:
    OnnxTextEncoder().encode_text(["x" * 200])
    ids = FakeSession.instances[0].runs[0]
    assert ids.shape == (1, 64)
    assert (ids[0, :63] == ord("x")).all()
    assert ids[0, 63] == 1  # EOS survives truncation, like open_clip


def test_text_of_exactly_64_tokens_is_untouched(fakes: types.SimpleNamespace) -> None:
    OnnxTextEncoder().encode_text(["x" * 63])  # 63 chars + EOS = 64 tokens
    ids = FakeSession.instances[0].runs[0][0]
    assert (ids[:63] == ord("x")).all() and ids[63] == 1


def test_eos_id_is_taken_from_the_tokenizer(fakes: types.SimpleNamespace) -> None:
    FakeTokenizer.eos = 7
    OnnxTextEncoder().encode_text(["x" * 200])
    assert FakeSession.instances[0].runs[0][0, 63] == 7


def test_eos_defaults_to_one_when_tokenizer_lacks_it(fakes: types.SimpleNamespace) -> None:
    FakeTokenizer.eos = None
    OnnxTextEncoder().encode_text(["x" * 200])
    assert FakeSession.instances[0].runs[0][0, 63] == 1


def test_batching_matches_one_by_one(fakes: types.SimpleNamespace) -> None:
    texts = [f"photo number {i} of a cat" for i in range(7)]
    enc = OnnxTextEncoder(batch_size=3)
    batched = enc.encode_text(texts)
    singles = np.concatenate([enc.encode_text([t]) for t in texts])
    assert np.allclose(batched, singles, atol=1e-6)
    assert [r.shape[0] for r in FakeSession.instances[0].runs[:3]] == [3, 3, 1]


def test_order_is_preserved(fakes: types.SimpleNamespace) -> None:
    enc = OnnxTextEncoder()
    ab = enc.encode_text(["alpha", "beta"])
    ba = enc.encode_text(["beta", "alpha"])
    assert np.allclose(ab[0], ba[1]) and np.allclose(ab[1], ba[0])


def test_empty_input_needs_no_model(fakes: types.SimpleNamespace) -> None:
    out = OnnxTextEncoder().encode_text([])
    assert out.shape == (0, DIM) and out.dtype == np.float32
    assert fakes.downloads == []


def test_empty_string_is_encodable(fakes: types.SimpleNamespace) -> None:
    out = OnnxTextEncoder().encode_text([""])
    assert np.isfinite(out).all()


def test_zero_vector_does_not_produce_nan(fakes: types.SimpleNamespace, monkeypatch) -> None:
    monkeypatch.setattr(
        FakeSession,
        "run",
        lambda self, o, f: [None, np.zeros((len(f["input_ids"]), DIM), np.float16)],
    )
    out = OnnxTextEncoder().encode_text(["x"])
    assert np.isfinite(out).all()


def test_single_string_rejected() -> None:
    with pytest.raises(TypeError):
        OnnxTextEncoder().encode_text("a dog")  # type: ignore[arg-type]


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="quantization"):
        OnnxTextEncoder(quantization="int4")
    with pytest.raises(ValueError, match="batch_size"):
        OnnxTextEncoder(batch_size=0)


# --------------------------------------------------------------------------- loading


def test_session_uses_cpu_provider_and_fp16_file(fakes: types.SimpleNamespace) -> None:
    OnnxTextEncoder().encode_text(["x"])
    session = FakeSession.instances[0]
    assert session.providers == ["CPUExecutionProvider"]
    assert session.path.endswith("onnx_text_model_fp16.onnx")


def test_int8_file_selected(fakes: types.SimpleNamespace) -> None:
    OnnxTextEncoder(quantization="int8").load()
    assert FakeSession.instances[0].path.endswith("onnx_text_model_int8.onnx")


def test_cache_hit_does_not_touch_network(fakes: types.SimpleNamespace) -> None:
    OnnxTextEncoder().load()
    assert all(local_only for _, _, local_only in fakes.downloads)
    assert {f for _, f, _ in fakes.downloads} == {"onnx/text_model_fp16.onnx", "tokenizer.json"}


def test_cache_miss_falls_back_to_download(fakes: types.SimpleNamespace) -> None:
    fakes.cached = False
    OnnxTextEncoder().load()
    assert any(not local_only for _, _, local_only in fakes.downloads)
    assert len(FakeSession.instances) == 1


def test_offline_and_uncached_gives_actionable_error(fakes: types.SimpleNamespace) -> None:
    fakes.cached = False
    fakes.network_ok = False
    with pytest.raises(ModelUnavailableError, match="download"):
        OnnxTextEncoder().load()


def test_load_is_idempotent(fakes: types.SimpleNamespace) -> None:
    enc = OnnxTextEncoder()
    enc.load()
    enc.load()
    enc.encode_text(["x"])
    assert len(FakeSession.instances) == 1
    assert len(FakeTokenizer.from_file_calls) == 1


def test_concurrent_first_use_loads_once(fakes: types.SimpleNamespace) -> None:
    FakeSession.construct_delay = 0.05  # widen the race window
    enc = OnnxTextEncoder()
    barrier = threading.Barrier(8)
    results: list[np.ndarray] = []
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(enc.encode_text([f"query {i % 3}"]))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(FakeSession.instances) == 1
    assert len(FakeTokenizer.from_file_calls) == 1
    assert len(results) == 8


def test_concurrent_explicit_load_creates_one_session(fakes: types.SimpleNamespace) -> None:
    FakeSession.construct_delay = 0.05
    enc = OnnxTextEncoder()
    barrier = threading.Barrier(8)

    def work() -> None:
        barrier.wait(timeout=5)
        enc.load()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(FakeSession.instances) == 1


def test_close_then_reuse_reloads(fakes: types.SimpleNamespace) -> None:
    enc = OnnxTextEncoder()
    first = enc.encode_text(["x"])
    enc.close()
    again = enc.encode_text(["x"])
    assert len(FakeSession.instances) == 2
    assert np.allclose(first, again)


def test_pooler_output_found_by_name(fakes: types.SimpleNamespace, monkeypatch) -> None:
    monkeypatch.setattr(
        FakeSession, "get_outputs", lambda self: [_Output("pooler_output"), _Output("other")]
    )
    monkeypatch.setattr(
        FakeSession,
        "run",
        lambda self, o, f: [np.ones((1, DIM), np.float16), np.full((1, DIM), 9, np.float16)],
    )
    out = OnnxTextEncoder().encode_text(["x"])
    assert np.allclose(out, 1 / np.sqrt(DIM))  # picked output 0 (the all-ones one)


def test_missing_output_name_defaults_to_second(fakes: types.SimpleNamespace, monkeypatch) -> None:
    monkeypatch.setattr(FakeSession, "get_outputs", lambda self: [_Output("a"), _Output("b")])
    enc = OnnxTextEncoder()
    enc.load()
    assert enc._output_index == 1


def test_session_creation_failure_is_typed(fakes: types.SimpleNamespace, monkeypatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("corrupt onnx")

    monkeypatch.setattr(sys.modules["onnxruntime"], "InferenceSession", boom)
    with pytest.raises(ModelUnavailableError, match="corrupt onnx"):
        OnnxTextEncoder().load()


@pytest.mark.parametrize("missing", ["onnxruntime", "tokenizers", "huggingface_hub"])
def test_missing_dependency_is_actionable(
    fakes: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setitem(sys.modules, missing, None)  # makes ``import`` raise ImportError
    with pytest.raises(ModelUnavailableError, match=r"Python 3\.10.*facesorter\[search\]"):
        OnnxTextEncoder().load()


@pytest.mark.parametrize("failure", [OSError("dlopen failed"), RuntimeError("bad abi")])
@pytest.mark.parametrize("module", ["onnxruntime", "tokenizers", "huggingface_hub"])
def test_any_import_time_failure_is_wrapped(
    fakes: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    failure: Exception,
) -> None:
    real_import = builtins.__import__

    def broken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module:
            raise failure
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModelUnavailableError, match="Python 3.10") as info:
        OnnxTextEncoder().load()
    assert info.value.__cause__ is failure


def test_module_import_is_cheap() -> None:
    # onnx_text must not import heavy packages at module level.
    src = Path(onnx_text.__file__).read_text()
    top_level = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any(("onnxruntime" in ln) or ("tokenizers" in ln) for ln in top_level)
