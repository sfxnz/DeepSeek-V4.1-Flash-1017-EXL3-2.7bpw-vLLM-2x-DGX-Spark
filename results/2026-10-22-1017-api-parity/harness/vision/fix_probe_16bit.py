"""Prototype of the 16-bit grayscale fix, applied in-process only (the recipe is not touched).

Run in the image: harness/vision/run_in_image.sh /ap/harness/vision/fix_probe_16bit.py

Pillow opens a 16-bit grayscale PNG as mode I;16 and Image.convert("RGB") clips every value
above 255 to 255, so vLLM's ImageMediaIO hands the processor an almost white image. The API
(OpenCV) scales 16-bit to 8-bit by 1/257 (opencv.rs:151-158). The candidate fix maps I;16* to L
with the same rounding before vLLM's convert_image_mode; this probe shows ours then matches the API.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path("/ap/harness/vision")
sys.path.insert(0, str(HERE))
import spec_port  # noqa: E402
from pixel_helpers import cmp  # noqa: E402

import vllm.multimodal.image as vimg  # noqa: E402
import vllm.multimodal.media.image as vmimg  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402

stock = vimg.convert_image_mode


def convert_image_mode_16bit(image, to_mode, background_color=(255, 255, 255)):
    if image.mode.startswith("I;16") and to_mode != image.mode:
        arr = np.asarray(image, dtype=np.float64)
        image = Image.fromarray(np.clip(np.rint(arr / 257.0), 0, 255).astype(np.uint8))  # mode L
    return stock(image, to_mode, background_color)


def ours(data):
    img = ImageMediaIO().load_bytes(data).media
    plan = spec_port.vllm_plan(*img.size)
    return np.asarray(ImageOps.pad(img.convert("RGB"), (plan["w"], plan["h"]), color=(127, 127, 127)), dtype=np.uint8)


up = json.loads((HERE / "out" / "upstream.json").read_text())
res = {}
for name in ("gray16.png", "rgb16.png", "gray_l.png"):
    data = (HERE / "images" / name).read_bytes()
    with Image.open(HERE / up["per_image"][name]["high"]["file"]) as w:
        api = np.asarray(w.convert("RGB"), dtype=np.uint8)
    before = cmp(ours(data), api)
    vmimg.convert_image_mode = convert_image_mode_16bit
    after = cmp(ours(data), api)
    vmimg.convert_image_mode = stock
    with Image.open(io.BytesIO(data)) as im:
        mode = im.mode
    res[name] = {"pil_mode": mode, "stock_vs_api": before, "fixed_vs_api": after}
    print(f"{name:12s} PIL mode {mode:6s} stock~api {before}\n{'':12s} {'':11s} fixed~api {after}")
(HERE / "out" / "fix_probe_16bit.json").write_text(json.dumps(res, indent=1))
