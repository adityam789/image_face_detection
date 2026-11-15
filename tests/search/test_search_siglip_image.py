"""Tests for facesorter.search.encoders.siglip_image with fake torch / open_clip."""

from __future__ import annotations

import builtins
import logging
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest
from PIL import Image

from facesorter.search.encoders import siglip_image
from facesorter.search.encoders.siglip_image import SigLIPImageEncoder, resolve_device
from facesorter.search.errors import ModelUnavailableError

DIM = 768


class FakeTensor:
    """Just enough of torch.Tensor: a numpy array that remembers device and dtype."""

    def __init__(self, arr: np.ndarray, device: str = "cpu", dtype: str = "float32") -> None:
        self.arr, self.device, self.dtype = arr, device, dtype

    def to(self, device: str | None = None, dtype: str | None = None) -> FakeTensor:
        return FakeTensor(self.arr, device or self.device, dtype or self.dtype)

    def float(self) -> FakeTensor:
        return FakeTensor(self.arr.astype(np.float32), self.device, "float32")

    def cpu(self) -> FakeTensor:
        return FakeTensor(self.arr, "cpu", self.dtype)

    def numpy(self) -> np.ndarray:
        return self.arr


class FakeModel:
    def __init__(self, env: Env | None = None) -> None:
        self.env = env
        self.device = "cpu"
        self.dtype = "float32"
        self.batches: list[FakeTensor] = []
        self.evaluating = False

    def to(self, device: str, dtype: str) -> FakeModel:
        if self.env is not None and self.env.fail_to_gpu and device != "cpu":
            raise RuntimeError("MPS backend out of memory")
        self.device, self.dtype = device, dtype
        return self

    def eval(self) -> FakeModel:
        self.evaluating = True
        return self

    def encode_image(self, batch: FakeTensor, normalize: bool = False) -> FakeTensor:
        env = self.env
        if env is not None:
            if env.fail_gpu_forward and self.device != "cpu":
                raise RuntimeError("unsupported fp16 op")
            with env.stats_lock:
                env.active += 1
                env.max_active = max(env.max_active, env.active)
            time.sleep(env.forward_delay)
            with env.stats_lock:
                env.active -= 1
        self.batches.append(batch)
        assert normalize
        # embedding = fixed random projection of the mean pixel; deliberately NOT unit length
        proj = np.random.default_rng(0).standard_normal((3, DIM)).astype(np.float32)
        feats = batch.arr.mean(axis=(2, 3)) @ proj * 5
        return FakeTensor(feats, batch.device, batch.dtype)


class Env:
    def __init__(self) -> None:
        self.mps = False
        self.cuda = False
        self.created: list[tuple[str, str]] = []
        self.model = FakeModel(self)
        self.create_delay = 0.0
        self.forward_delay = 0.0
        self.fail_to_gpu = False
        self.fail_gpu_forward = False
        self.active = 0
        self.max_active = 0
        self.stats_lock = threading.Lock()
        self.preprocess_modes: list[str] = []
        self.cache_cleared: list[str] = []
        self.fail_create: Exception | None = None


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Env:
    e = Env()

    torch = types.ModuleType("torch")
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: e.mps)
    )
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: e.cuda, empty_cache=lambda: e.cache_cleared.append("cuda")
    )
    torch.mps = types.SimpleNamespace(  # type: ignore[attr-defined]
        empty_cache=lambda: e.cache_cleared.append("mps")
    )
    torch.stack = lambda items: FakeTensor(np.stack([t.arr for t in items]))  # type: ignore[attr-defined]

    class _Inference:
        def __enter__(self) -> None: ...
        def __exit__(self, *a: Any) -> None: ...

    torch.inference_mode = _Inference  # type: ignore[attr-defined]

    open_clip = types.ModuleType("open_clip")

    def create(arch: str, pretrained: str) -> tuple[FakeModel, None, Any]:
        time.sleep(e.create_delay)
        if e.fail_create:
            raise e.fail_create
        e.created.append((arch, pretrained))

        def preprocess(img: Image.Image) -> FakeTensor:
            e.preprocess_modes.append(img.mode)
            arr = np.asarray(img.resize((8, 8)), dtype=np.float32).transpose(2, 0, 1) / 255.0
            return FakeTensor(arr)

        return e.model, None, preprocess

    open_clip.create_model_and_transforms = create  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    return e


def _img(color: tuple[int, int, int], mode: str = "RGB") -> Image.Image:
    return Image.new("RGB", (16, 12), color).convert(mode)


# --------------------------------------------------------------------------- device


def test_resolve_device_explicit_needs_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_priority(env: Env) -> None:
    assert resolve_device("auto") == "cpu"
    env.cuda = True
    assert resolve_device("auto") == "cuda"
    env.mps = True
    assert resolve_device("auto") == "mps"


def test_resolve_device_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ModelUnavailableError, match=r"Python 3\.10.*facesorter\[search\]"):
        resolve_device("auto")


def test_resolve_device_torch_without_mps_backend(env: Env) -> None:
    del sys.modules["torch"].backends
    sys.modules["torch"].backends = types.SimpleNamespace()
    assert resolve_device("auto") == "cpu"


# --------------------------------------------------------------------------- encoding


def test_identity_attributes() -> None:
    enc = SigLIPImageEncoder()
    assert (enc.name, enc.dim) == ("siglip2-b16-webli", 768)


def test_construction_is_lazy(env: Env) -> None:
    SigLIPImageEncoder()
    assert env.created == []


def test_encode_shape_dtype_and_unit_norm(env: Env) -> None:
    enc = SigLIPImageEncoder(device="cpu")
    out = enc.encode_images([_img((200, 10, 10)), _img((10, 10, 200)), _img((10, 200, 10))])
    assert out.shape == (3, DIM)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert env.created == [("ViT-B-16-SigLIP2", "webli")]


def test_cpu_uses_fp32_and_gpu_uses_fp16(env: Env) -> None:
    cpu = SigLIPImageEncoder(device="cpu")
    cpu.encode_images([_img((1, 2, 3))])
    assert (env.model.device, env.model.dtype) == ("cpu", "float32")
    assert env.model.evaluating
    gpu = SigLIPImageEncoder(device="mps")
    gpu.encode_images([_img((1, 2, 3))])
    assert (env.model.device, env.model.dtype) == ("mps", "float16")
    assert env.model.batches[-1].dtype == "float16" and env.model.batches[-1].device == "mps"


def test_auto_device_selected(env: Env) -> None:
    env.mps = True
    enc = SigLIPImageEncoder()
    enc.load()
    assert enc.device == "mps"


def test_batches_are_chunked_and_order_kept(env: Env) -> None:
    colors = [(i * 20, 255 - i * 20, 40) for i in range(7)]
    enc = SigLIPImageEncoder(device="cpu", batch_size=3)
    batched = enc.encode_images([_img(c) for c in colors])
    assert [b.arr.shape[0] for b in env.model.batches] == [3, 3, 1]
    singles = np.concatenate([enc.encode_images([_img(c)]) for c in colors])
    assert np.allclose(batched, singles, atol=1e-6)


def test_non_rgb_images_are_converted(env: Env) -> None:
    SigLIPImageEncoder(device="cpu").encode_images([_img((9, 9, 9), "L"), _img((9, 9, 9), "RGBA")])
    assert env.preprocess_modes == ["RGB", "RGB"]


def test_empty_input_does_not_load_model(env: Env) -> None:
    out = SigLIPImageEncoder(device="cpu").encode_images([])
    assert out.shape == (0, DIM) and out.dtype == np.float32
    assert env.created == []


def test_concurrent_first_use_loads_once_and_never_overlaps_forward(env: Env) -> None:
    env.create_delay = 0.05  # widen the race window
    env.forward_delay = 0.02
    enc = SigLIPImageEncoder(device="cpu")
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            barrier.wait(timeout=5)
            enc.encode_images([_img((5, 5, 5))])
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(env.created) == 1
    assert env.max_active == 1  # the forward pass is never entered concurrently


def test_concurrent_explicit_load_creates_one_model(env: Env) -> None:
    env.create_delay = 0.05
    enc = SigLIPImageEncoder(device="cpu")
    barrier = threading.Barrier(8)

    def work() -> None:
        barrier.wait(timeout=5)
        enc.load()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(env.created) == 1


def test_close_frees_and_reloads(env: Env) -> None:
    enc = SigLIPImageEncoder(device="mps")
    enc.encode_images([_img((1, 1, 1))])
    enc.close()
    assert env.cache_cleared == ["mps"]
    assert enc._model is None
    enc.encode_images([_img((1, 1, 1))])
    assert len(env.created) == 2


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_close_clears_matching_cache(env: Env, device: str) -> None:
    enc = SigLIPImageEncoder(device=device)
    enc.load()
    enc.close()
    assert env.cache_cleared == (["cuda"] if device == "cuda" else [])


def test_close_before_load_is_a_noop(env: Env) -> None:
    SigLIPImageEncoder().close()
    assert env.cache_cleared == []


# --------------------------------------------------------------------------- failures


def test_missing_dependencies_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "open_clip", None)
    with pytest.raises(
        ModelUnavailableError, match=r"Python 3\.10.*pip install .*facesorter\[search\]"
    ):
        SigLIPImageEncoder(device="cpu").load()


def test_weight_download_failure_is_typed(env: Env) -> None:
    env.fail_create = OSError("no route to host")
    with pytest.raises(ModelUnavailableError, match="no route to host"):
        SigLIPImageEncoder(device="cpu").load()


def test_batch_size_validated() -> None:
    with pytest.raises(ValueError):
        SigLIPImageEncoder(batch_size=0)


def test_module_does_not_import_heavy_packages_at_top_level() -> None:
    src = open(siglip_image.__file__).read()
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any(("torch" in ln) or ("open_clip" in ln) for ln in top)


@pytest.mark.parametrize("failure", [OSError("dlopen failed"), RuntimeError("bad abi")])
@pytest.mark.parametrize("module", ["torch", "open_clip"])
def test_any_import_time_failure_is_wrapped(
    env: Env, monkeypatch: pytest.MonkeyPatch, module: str, failure: Exception
) -> None:
    real_import = builtins.__import__

    def broken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module:
            raise failure
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModelUnavailableError, match="Python 3.10") as info:
        SigLIPImageEncoder(device="cpu").load()
    assert info.value.__cause__ is failure


def test_resolve_device_wraps_broken_torch(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def broken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise OSError("dlopen failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModelUnavailableError, match="Python 3.10") as info:
        resolve_device("auto")
    assert isinstance(info.value.__cause__, OSError)


# --------------------------------------------------------------------------- CPU fallback


def test_falls_back_to_cpu_when_gpu_load_fails(env: Env, caplog: pytest.LogCaptureFixture) -> None:
    env.fail_to_gpu = True
    enc = SigLIPImageEncoder(device="mps")
    with caplog.at_level(logging.WARNING):
        out = enc.encode_images([_img((9, 9, 9))])
    assert enc.device == "cpu"
    assert (env.model.device, env.model.dtype) == ("cpu", "float32")
    assert out.shape == (1, DIM)
    assert "falling back to the CPU" in caplog.text


def test_falls_back_to_cpu_when_gpu_forward_fails(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    env.fail_gpu_forward = True
    enc = SigLIPImageEncoder(device="mps", batch_size=2)
    with caplog.at_level(logging.WARNING):
        out = enc.encode_images([_img((i * 30, 5, 5)) for i in range(5)])
    assert enc.device == "cpu"
    assert out.shape == (5, DIM)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert "falling back to the CPU" in caplog.text
    # later calls stay on the CPU without further warnings
    assert enc.encode_images([_img((1, 2, 3))]).shape == (1, DIM)


def test_cpu_forward_failure_is_not_swallowed(env: Env) -> None:
    enc = SigLIPImageEncoder(device="cpu")
    enc.load()

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("broken")

    env.model.encode_image = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="broken"):
        enc.encode_images([_img((1, 2, 3))])


def test_gpu_and_cpu_load_both_failing_is_typed(env: Env) -> None:
    env.fail_create = OSError("weights missing")
    with pytest.raises(ModelUnavailableError, match="weights missing"):
        SigLIPImageEncoder(device="mps").load()
