"""Tests for facesorter.search.encoders.decode (synthetic images only)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pillow_heif
import pytest
from PIL import Image, ImageOps, JpegImagePlugin

from facesorter.search.encoders import decode
from facesorter.search.encoders.decode import iter_decoded, load_for_model
from facesorter.search.errors import UnreadableImageError


def _scene(size: tuple[int, int]) -> Image.Image:
    """Asymmetric test pattern: red left, blue top-right, green bottom-right."""
    w, h = size
    img = Image.new("RGB", size, (0, 200, 0))
    img.paste((220, 20, 20), (0, 0, w // 2, h))
    img.paste((20, 20, 220), (w // 2, 0, w, h // 2))
    return img


def _exif(orientation: int) -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif


def test_jpeg_is_downscaled_to_target_short_side(tmp_path: Path) -> None:
    path = tmp_path / "big.jpg"
    _scene((2400, 1800)).save(path, quality=90)
    img = load_for_model(path, target=448)
    assert img.mode == "RGB"
    assert min(img.size) >= 448
    assert min(img.size) < 448 * 2  # draft + resize got close to the target
    assert img.size[0] / img.size[1] == pytest.approx(4 / 3, rel=0.02)


def test_jpeg_content_survives_draft(tmp_path: Path) -> None:
    path = tmp_path / "big.jpg"
    _scene((2400, 1800)).save(path, quality=95)
    img = load_for_model(str(path), target=300)  # str paths work too
    w, h = img.size
    assert img.getpixel((w // 4, h // 2))[0] > 180  # red half
    assert img.getpixel((3 * w // 4, h // 4))[2] > 180  # blue
    assert img.getpixel((3 * w // 4, 3 * h // 4))[1] > 150  # green


def test_small_image_is_not_upscaled(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    _scene((120, 80)).save(path)
    img = load_for_model(path)
    assert img.size == (120, 80)
    assert np.array_equal(np.asarray(img), np.asarray(_scene((120, 80))))


def test_png_is_downscaled(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    _scene((1000, 700)).save(path)
    assert load_for_model(path, target=350).size == (500, 350)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_matches_pil(tmp_path: Path, orientation: int) -> None:
    path = tmp_path / "o.png"
    _scene((60, 40)).save(path, exif=_exif(orientation))
    expected = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    got = load_for_model(path)
    assert got.size == expected.size
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_exif_orientation_applied_after_downscale(tmp_path: Path) -> None:
    path = tmp_path / "rot.jpg"
    _scene((2000, 1000)).save(path, quality=90, exif=_exif(6))
    img = load_for_model(path, target=400)
    assert img.height > img.width  # portrait after rotation
    assert min(img.size) >= 400


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_other_modes_become_rgb(tmp_path: Path, mode: str) -> None:
    path = tmp_path / f"{mode}.tiff"
    _scene((50, 40)).convert(mode).save(path)
    assert load_for_model(path).mode == "RGB"


def test_cmyk_jpeg_with_draft(tmp_path: Path) -> None:
    path = tmp_path / "cmyk.jpg"
    _scene((1200, 900)).convert("CMYK").save(path)
    img = load_for_model(path, target=300)
    assert img.mode == "RGB"
    assert min(img.size) >= 300


def test_heic_roundtrip_downscaled(tmp_path: Path) -> None:
    path = tmp_path / "photo.heic"
    pillow_heif.from_pillow(_scene((1600, 1200))).save(path, quality=80)
    img = load_for_model(path, target=400)
    assert img.mode == "RGB"
    assert img.size == (533, 400)
    assert img.getpixel((img.width // 4, img.height // 2))[0] > 150


def test_unreadable_files_raise_typed_error(tmp_path: Path) -> None:
    garbage = tmp_path / "bad.jpg"
    garbage.write_bytes(b"this is not a jpeg")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    for path in (garbage, empty, tmp_path / "missing.jpg"):
        with pytest.raises(UnreadableImageError) as info:
            load_for_model(path)
        assert str(path) in str(info.value)
        assert info.value.__cause__ is not None


def test_truncated_jpeg_is_unreadable(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    _scene((800, 600)).save(good)
    truncated = tmp_path / "cut.jpg"
    truncated.write_bytes(good.read_bytes()[:300])
    with pytest.raises(UnreadableImageError):
        load_for_model(truncated)


def test_invalid_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_for_model(tmp_path / "x.jpg", target=0)


def test_heif_opener_registered_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(decode, "_heif_registered", False)
    monkeypatch.setattr(pillow_heif, "register_heif_opener", lambda: calls.append(1))
    decode._ensure_heif()
    decode._ensure_heif()
    assert calls == [1]


# --------------------------------------------------------------------------- iter_decoded


def _make_files(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i:02d}.png"
        Image.new("RGB", (20 + i, 10), (i * 10, 0, 0)).save(p)
        paths.append(p)
    return paths


def test_iter_decoded_preserves_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_files(tmp_path, 12)
    real = decode.load_for_model

    def slow_first(path: Path, target: int) -> Image.Image:
        if path.name == "img00.png":
            time.sleep(0.2)  # finishes last but must still be yielded first
        return real(path, target)

    monkeypatch.setattr(decode, "load_for_model", slow_first)
    out = list(iter_decoded(paths, workers=4))
    assert [p for p, _ in out] == paths
    assert [img.size[0] for _, img in out] == [20 + i for i in range(12)]  # type: ignore[union-attr]


def test_iter_decoded_yields_errors_not_raises(tmp_path: Path) -> None:
    paths = _make_files(tmp_path, 3)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    items = list(iter_decoded([paths[0], bad, str(paths[1]), tmp_path / "gone.png"], workers=2))
    assert [p.name for p, _ in items] == ["img00.png", "bad.jpg", "img01.png", "gone.png"]
    assert isinstance(items[0][1], Image.Image)
    assert isinstance(items[1][1], UnreadableImageError)
    assert isinstance(items[2][1], Image.Image)
    assert isinstance(items[3][1], UnreadableImageError)


def test_iter_decoded_empty() -> None:
    assert list(iter_decoded([])) == []


def test_iter_decoded_bounds_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = 0
    lock = threading.Lock()

    def fake_load(path: Path, target: int) -> Image.Image:
        nonlocal started
        with lock:
            started += 1
        return Image.new("RGB", (4, 4))

    monkeypatch.setattr(decode, "load_for_model", fake_load)
    paths = (tmp_path / f"{i}.png" for i in range(100))  # a generator: never fully realised
    it = iter_decoded(paths, workers=2, prefetch=3)
    next(it)
    time.sleep(0.1)
    # one consumed + at most prefetch (3) more submitted, however long the input is
    assert started <= 1 + 3
    it.close()


def test_iter_decoded_default_prefetch_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets: list[int] = []

    def fake_load(path: Path, target: int) -> Image.Image:
        targets.append(target)
        return Image.new("RGB", (4, 4))

    monkeypatch.setattr(decode, "load_for_model", fake_load)
    out = list(iter_decoded([tmp_path / "a.png", tmp_path / "b.png"], target=123))
    assert len(out) == 2
    assert targets == [123, 123]


def test_iter_decoded_invalid_arguments_raise_immediately() -> None:
    # No iteration needed: the ValueError comes from the call itself, once.
    with pytest.raises(ValueError, match="workers"):
        iter_decoded([], workers=0)
    for bad in (0, -5):
        with pytest.raises(ValueError, match="target"):
            iter_decoded(["a.jpg", "b.jpg"], target=bad)


def test_iter_decoded_early_close_cancels_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers = 2
    started: list[str] = []
    lock = threading.Lock()
    gate = threading.Event()

    def fake_load(path: Path, target: int) -> Image.Image:
        with lock:
            started.append(path.name)
        if path.name != "0.png":
            gate.wait(timeout=5)  # everything but the first file blocks
        return Image.new("RGB", (4, 4))

    monkeypatch.setattr(decode, "load_for_model", fake_load)
    it = iter_decoded([tmp_path / f"{i}.png" for i in range(60)], workers=workers, prefetch=30)
    assert next(it)[0].name == "0.png"
    threading.Timer(0.3, gate.set).start()  # lets running loaders finish after cancellation
    it.close()
    time.sleep(0.2)
    # Only running loaders may have started: queued futures were cancelled, not run.
    assert len(started) <= workers + 1
    assert "59.png" not in started


# --------------------------------------------------------------------------- draft / MPO


def _record_draft(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []
    real = JpegImagePlugin.JpegImageFile.draft

    def spy(self: Image.Image, mode: str | None, size: tuple[int, int] | None):  # type: ignore[no-untyped-def]
        result = real(self, mode, size)
        calls.append((mode, size, self.size))  # size after draft = size libjpeg will decode
        return result

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy)
    return calls


def test_large_jpeg_uses_draft_with_target_square(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), (30, 90, 160)).save(path, quality=85)
    calls = _record_draft(monkeypatch)
    out = load_for_model(path, target=448)
    assert len(calls) == 1
    mode, size, decoded = calls[0]
    assert (mode, size) == ("RGB", (448, 448))
    # decoded at reduced scale (never the full 12 MP), yet still >= target on both sides
    assert decoded[0] <= 1008 and decoded[1] <= 756  # type: ignore[index]
    assert min(decoded) >= 448  # type: ignore[type-var]
    assert min(out.size) == 448


def test_mpo_files_also_use_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "camera.mpo"
    main = Image.new("RGB", (4000, 3000), (200, 50, 50))
    main.save(path, format="MPO", save_all=True, append_images=[Image.new("RGB", (400, 300))])
    with Image.open(path) as probe:
        assert probe.format == "MPO"
    calls = _record_draft(monkeypatch)
    out = load_for_model(path, target=448)
    assert len(calls) == 1 and calls[0][:2] == ("RGB", (448, 448))
    assert calls[0][2][0] <= 1008  # type: ignore[index]
    assert out.size == (597, 448)
    assert out.getpixel((10, 10))[0] > 150


def test_png_never_calls_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.png"
    Image.new("RGB", (900, 700)).save(path)
    calls = _record_draft(monkeypatch)
    load_for_model(path, target=300)
    assert calls == []


# --------------------------------------------------------------------------- bit depth


def test_16bit_grayscale_png_keeps_mid_grey(tmp_path: Path) -> None:
    path = tmp_path / "g16.png"
    Image.fromarray(np.full((40, 50), 32768, dtype=np.uint16)).save(path)
    with Image.open(path) as probe:
        assert probe.mode.startswith("I")
    img = load_for_model(path)
    assert img.mode == "RGB"
    assert img.getpixel((5, 5)) == (128, 128, 128)


def test_16bit_gradient_is_monotonic(tmp_path: Path) -> None:
    path = tmp_path / "ramp.png"
    ramp = np.tile(np.linspace(0, 65535, 64).astype(np.uint16), (8, 1))
    Image.fromarray(ramp).save(path)
    row = np.asarray(load_for_model(path))[0, :, 0].astype(int)
    assert row[0] <= 2 and row[-1] >= 253
    assert (np.diff(row) >= 0).all()


def test_float_tiff_unit_range_maps_to_grey(tmp_path: Path) -> None:
    path = tmp_path / "f.tiff"
    Image.fromarray(np.full((30, 30), 0.5, dtype=np.float32)).save(path)
    assert load_for_model(path).getpixel((3, 3)) == (128, 128, 128)


def test_float_tiff_wide_range_is_stretched(tmp_path: Path) -> None:
    path = tmp_path / "wide.tiff"
    data = np.tile(np.linspace(-1000.0, 3000.0, 40, dtype=np.float32), (10, 1))
    Image.fromarray(data).save(path)
    row = np.asarray(load_for_model(path))[0, :, 0].astype(int)
    assert row[0] == 0 and row[-1] == 255
    assert 120 <= row[len(row) // 2] <= 135


def test_float_tiff_already_in_8bit_range_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "f255.tiff"
    Image.fromarray(np.full((20, 20), 100.0, dtype=np.float32)).save(path)
    assert load_for_model(path).getpixel((1, 1)) == (100, 100, 100)


def test_constant_float_image_outside_range_does_not_divide_by_zero(tmp_path: Path) -> None:
    path = tmp_path / "const.tiff"
    Image.fromarray(np.full((20, 20), 5000.0, dtype=np.float32)).save(path)
    assert load_for_model(path).getpixel((1, 1)) == (0, 0, 0)


def test_32bit_integer_image_scaled(tmp_path: Path) -> None:
    img = Image.fromarray(np.full((10, 10), 40000, dtype=np.int32), mode="I")
    assert decode._to_8bit(img).getpixel((2, 2)) == 156  # 40000 / 256
    big = Image.fromarray(np.tile(np.array([0, 100000], dtype=np.int32), (4, 5)), mode="I")
    out = np.asarray(decode._to_8bit(big))
    assert out.min() == 0 and out.max() == 255


# --------------------------------------------------------------------------- failures


class _Boom(Exception):
    """Not an OSError: stands in for whatever a codec decides to raise."""


@pytest.mark.parametrize(
    "exc", [SyntaxError("bad png"), ValueError("bad"), _Boom("odd"), MemoryError()]
)
def test_non_oserror_failures_become_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    def boom(path: Path, target: int) -> Image.Image:
        raise exc

    monkeypatch.setattr(decode, "_decode", boom)
    with pytest.raises(UnreadableImageError) as info:
        load_for_model(tmp_path / "x.jpg")
    assert info.value.__cause__ is exc
    # and through the iterator they are yielded, not raised
    ((path, result),) = iter_decoded([tmp_path / "x.jpg"])
    assert isinstance(result, UnreadableImageError) and result.__cause__ is exc


def test_decompression_bomb_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bomb.png"
    Image.new("RGB", (200, 200)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # 40000 px > 2 * 100: hard error
    with pytest.raises(UnreadableImageError) as info:
        load_for_model(path)
    assert isinstance(info.value.__cause__, Image.DecompressionBombError)
    assert not isinstance(info.value.__cause__, OSError)


def test_corrupt_png_data_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)).save(path)
    raw = bytearray(path.read_bytes())
    start = raw.index(b"IDAT") + 20
    raw[start : start + 200] = bytes(200)  # zero a run in the middle of the deflate stream
    path.write_bytes(bytes(raw))
    with pytest.raises(UnreadableImageError) as info:
        load_for_model(path)
    assert info.value.__cause__ is not None
