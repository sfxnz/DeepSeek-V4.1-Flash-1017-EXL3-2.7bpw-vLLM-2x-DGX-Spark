#!/usr/bin/env python3
"""16-bit grayscale decode checks inside the serve image (docker/patch/image_16bit_gray.py), CPU only.

Run it after sitecustomize applied (DSV41_IMAGE_16BIT_GRAY=1) or restored (=0) the rewrite, with docker/patch
mounted the way run.sh mounts it:

    docker run --rm --network none --memory 4g -e DSV41_IMAGE_16BIT_GRAY=1 \\
      -v $PWD/docker/patch:/opt/dsv41-patch:ro \\
      -v $PWD/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \\
      -v $PWD/tests:/checks:ro --entrypoint python3 <image> /checks/image_16bit_gray_image_check.py

Each image goes through every serve decode path:
- ImageMediaIO.load_bytes (HTTP fetches end there);
- a data URL and a file:// URL through MediaConnector.fetch_image (load_base64, load_file);
- MultiModalDataParser for a PIL item;
- the V4.1 processor on an unconverted image (media_io_kwargs image_mode null).
"Stock" is the image's own code, rebuilt from the unpatched text of the two rewritten files.
With =1: a 16-bit gradient holding all 65536 values decodes to round(v / 257) on every path, an EXIF
orientation is applied as stock applies it, and a tRNS key's pixels composite onto the background as stock
composites them (a key below 255; for a higher key stock picks the wrong pixels, see image_16bit_gray.py).
The processor's patches equal those of the expected 8-bit image. 16-bit RGB / RGBA PNGs, 8-bit L, LA, RGBA
and an EXIF-rotated JPEG decode byte-identical to stock.
With =0 the files are the image's text again and 16-bit gray still clamps to 255.
Prints one line per check and exits 1 on any failure.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import image_16bit_gray as fix
import vllm.models.deepseek_v4_1.common.mm_preprocess as pre
import vllm.multimodal.image as mm_image
import vllm.multimodal.media.image as media_image
import vllm.multimodal.parse as mm_parse
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.parse import MultiModalDataParser

BG = (0, 0, 255)  # load paths composite alpha onto this; no gray pixel has it
WHITE = (255, 255, 255)  # MultiModalDataParser composites onto vLLM's default white
VISION = types.SimpleNamespace(  # the 2.7bpw-mcg-hq pack's vision_config
    vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=1024, vision_min_pixels=295936,
    vision_max_wh_ratio=None)
ROTATE_6 = Image.Transpose.ROTATE_270  # what ImageOps.exif_transpose applies for orientation 6


def stock_module(module, unpatch, name):
    """The image's stock module, executed from the unpatched text of the (possibly) rewritten file."""
    mod = types.ModuleType(name)
    exec(compile(unpatch(Path(module.__file__).read_text()), f"stock {module.__file__}", "exec"), mod.__dict__)
    return mod


def png(arr) -> bytes:
    ok, enc = cv2.imencode(".png", arr)
    assert ok
    return enc.tobytes()


def saved(img, fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def exif(orientation: int) -> bytes:
    e = Image.Exif()
    e[0x0112] = orientation
    return e.tobytes()


def gray8(v16):
    """The API's 16-bit -> 8-bit scaling, round(v / 257), computed independently of the patch."""
    return np.rint(v16.astype(np.float64) / 257.0).astype(np.uint8)


def rgb(gray, transparent=None, bg=WHITE):
    out = np.repeat(gray[..., None], 3, axis=2)
    if transparent is not None:
        out[transparent] = bg
    return out


def cases():
    """(name, encoded bytes, expected RGB of a 16-bit gray image as fn(background) or None, processor expected
    or None, (mask of the tRNS key pixels, whether stock composites exactly those) or None)."""
    every = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    ramp = (np.arange(32 * 48, dtype=np.uint64).reshape(32, 48) * 65535 // (32 * 48 - 1)).astype(np.uint16)
    rot = np.asarray(Image.fromarray(gray8(ramp)).transpose(ROTATE_6))
    low = ramp.copy()
    low[3, 7] = 160  # no other pixel is 160, or clamps to it
    hit = np.zeros(ramp.shape, bool)
    hit[3, 7] = True
    high = int(ramp[3, 7])  # 6446 = 0x192E; no pixel is 0x2E = 46
    photo = np.dstack([np.linspace(0, 255, 64 * 40).reshape(40, 64).astype(np.uint8)] * 3)
    photo[..., 1] = photo[..., 1][::-1]
    alpha8 = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (40, 1))
    return [
        ("16-bit gray, all 65536 values", png(every), lambda bg: rgb(gray8(every)), rgb(gray8(every)), None),
        ("16-bit gray, EXIF orientation 6", saved(Image.fromarray(ramp), exif=exif(6)),
         lambda bg: rgb(rot), rgb(rot), None),
        ("16-bit gray, tRNS key 160", saved(Image.fromarray(low), transparency=160),
         lambda bg: rgb(gray8(low), hit, bg), rgb(gray8(low)), (hit, True)),
        ("16-bit gray, tRNS key 6446", saved(Image.fromarray(ramp), transparency=high),
         lambda bg: rgb(gray8(ramp), hit, bg), rgb(gray8(ramp)), (hit, False)),
        ("16-bit RGB PNG", png(np.dstack([ramp, ramp[::-1], ramp])), None, None, None),
        ("16-bit RGBA PNG", png(np.dstack([ramp, ramp, ramp[::-1], ramp])), None, None, None),
        ("8-bit L", saved(Image.fromarray(photo[..., 0])), None, None, None),
        ("8-bit LA", saved(Image.fromarray(np.dstack([photo[..., 0], alpha8]))), None, None, None),
        ("8-bit RGBA", saved(Image.fromarray(np.dstack([photo, alpha8]))), None, None, None),
        ("JPEG, EXIF orientation 6", saved(Image.fromarray(photo), "JPEG", quality=95, exif=exif(6)), None, None, None),
    ]


def paths(tmp: Path):
    """(name, background the path composites onto, fn(bytes) -> RGB array) for each serve decode path."""

    def load_bytes(data):
        return ImageMediaIO(rgba_background_color=BG).load_bytes(data).media

    def data_url(data):
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        return MediaConnector({"image": {"rgba_background_color": list(BG)}}).fetch_image(url)

    def file_url(data):
        path = tmp / f"img{len(list(tmp.iterdir()))}.bin"
        path.write_bytes(data)
        conn = MediaConnector({"image": {"rgba_background_color": list(BG)}}, allowed_local_media_path=str(tmp))
        return conn.fetch_image(path.as_uri())

    def parser(data):
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return MultiModalDataParser()._parse_image_data(img).get(0)

    return [
        ("ImageMediaIO.load_bytes", BG, load_bytes),
        ("data URL (load_base64)", BG, data_url),
        ("file URL (load_file)", BG, file_url),
        ("MultiModalDataParser", WHITE, parser),
    ]


def as_rgb(img) -> np.ndarray:
    arr = np.asarray(img)
    assert arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3, (arr.dtype, arr.shape)
    return arr


def with_stock_convert(stock, fn, data):
    """fn(data) with the image's stock convert_image_mode bound where ImageMediaIO and the parser look it up."""
    saved_bindings = media_image.convert_image_mode, mm_parse.convert_image_mode
    media_image.convert_image_mode = mm_parse.convert_image_mode = stock.convert_image_mode
    try:
        return as_rgb(fn(data))
    finally:
        media_image.convert_image_mode, mm_parse.convert_image_mode = saved_bindings


def same_patches(a, b) -> bool:
    return all(torch.equal(x, y) if torch.is_tensor(x) else x == y for x, y in zip(a, b))


def main() -> int:
    on = os.environ.get(fix.ENV) == "1"
    results = []
    rewritten = [fix.MARK in Path(m.__file__).read_text() for m in (mm_image, pre)]
    results.append((f"{fix.ENV}={'1' if on else '0'}: both files {'rewritten' if on else 'stock'}",
                    rewritten == [on, on], f"multimodal/image.py, mm_preprocess.py marker: {rewritten}"))
    stock = stock_module(mm_image, fix.unpatch_convert_source, "dsv41_stock_multimodal_image")
    stock_pre = stock_module(pre, fix.unpatch_load_source, "dsv41_stock_mm_preprocess")
    proc, stock_proc = pre.DeepseekV4VLImageProcessor(VISION), stock_pre.DeepseekV4VLImageProcessor(VISION)
    with tempfile.TemporaryDirectory() as d:
        for name, data, want16, want_proc, transparent in cases():
            for pname, bg, fn in paths(Path(d)):
                got, ref = as_rgb(fn(data)), with_stock_convert(stock, fn, data)
                label = f"{name} via {pname}"
                if want16 is None:
                    results.append((f"{label}: identical to stock", np.array_equal(got, ref), ""))
                elif on:
                    want = want16(bg)
                    diff = int(np.abs(got.astype(int) - want.astype(int)).max()) if got.shape == want.shape else "shape"
                    results.append((f"{label}: round(v / 257)", np.array_equal(got, want), f"max |diff| {diff}"))
                    results.append((f"{label}: stock clamps (control)", not np.array_equal(ref, want), ""))
                    if transparent is not None:
                        mask, stock_agrees = transparent
                        on_bg = bool((got[mask] == bg).all()) and np.array_equal(got[~mask], want[~mask])
                        if stock_agrees:
                            results.append((f"{label}: key pixels composite as stock", on_bg
                                            and np.array_equal(got[mask], ref[mask]), ""))
                        elif bg != WHITE:  # a missed key pixel clamps to 255, which is white
                            results.append((f"{label}: key pixels composite; stock misses the key (control)",
                                            on_bg and not bool((ref[mask] == bg).all()), ""))
                else:
                    results.append((f"{label}: stock clamp kept", np.array_equal(got, ref) and int(got.max()) == 255
                                    and not np.array_equal(got, want16(bg)), ""))
            raw = ImageMediaIO(image_mode=None).load_bytes(data)  # media_io_kwargs image_mode null
            got = proc(raw)
            if want_proc is None:
                results.append((f"{name} via processor (unconverted {raw.mode}): identical to stock",
                                same_patches(got, stock_proc(raw)), ""))
            elif on:
                results.append((f"{name} via processor (unconverted {raw.mode}): patches of round(v / 257)",
                                same_patches(got, proc(Image.fromarray(want_proc))), ""))
                results.append((f"{name} via processor: stock clamps (control)",
                                not same_patches(stock_proc(raw), got), ""))
            else:
                results.append((f"{name} via processor: stock clamp kept", same_patches(got, stock_proc(raw)), ""))
    bad = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" -- {detail}" if detail and not ok else ""))
        bad += not ok
    print(f"{len(results) - bad}/{len(results)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
