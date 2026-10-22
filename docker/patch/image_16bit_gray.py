"""16-bit grayscale images keep their gray levels (DSV41_IMAGE_16BIT_GRAY=1).

Pillow opens a 16-bit grayscale PNG as mode I;16, and its conversion to an 8-bit mode clamps every value to
255: a 16-bit scan, depth map or medical image reached the vision tower almost white. The DeepSeek API scales
16-bit images to 8-bit by 1/257 (deepseek-recipe opencv.rs:151-158). api-parity vision V1: PSNR against the
API's pixels 4.86 dB stock, 51.5 dB with this mapping (the rest is resize differences).

gray16_to_8bit() maps I;16* to L by round(v / 257), computed exactly in integers as (v + 128) // 257. A tRNS
colour key (PNG gray with transparency) becomes an LA alpha mask taken from the 16-bit values, so the stock
alpha handling (composite onto the background colour) still applies to the key's pixels. Pillow's own I;16 to
RGBA conversion compares the key's low byte with the clamped values: for a key below 255 it marks the same
pixels, for a higher key the wrong ones (key 6446 = 0x192E marks v = 46; key 255 marks every v >= 255). The
API ignores a gray tRNS key. Pillow already decodes 16-bit RGB and RGBA PNGs to 8-bit RGB and RGBA; every
mode other than I;16* is untouched.

Two source rewrites of the image's vLLM call it for I;16* images only:
- multimodal/image.py convert_image_mode, when the target mode is 8-bit. Every decode path converts through
  it after the EXIF transpose: ImageMediaIO.load_bytes (and load_base64 / load_file, which call it),
  encode_base64, and the processor's MultiModalDataParser for PIL items.
- models/deepseek_v4_1/common/mm_preprocess.py load_image, the V4.1 processor's own convert("RGB"), which an
  image reaches unconverted when a request sets media_io_kwargs image_mode to null or I;16.

Applied from sitecustomize at interpreter start, before anything imports vLLM. Each anchor must match once.
Idempotent (MARK). DSV41_IMAGE_16BIT_GRAY=0 keeps the image's text, and puts it back in a container whose
files were already rewritten. Top-level imports are stdlib only.
"""

from __future__ import annotations

import os
import sys

MARK = "dsv41-image-16bit-gray"
ENV = "DSV41_IMAGE_16BIT_GRAY"
LOG_ENGAGED = "dsv41: image 16-bit gray engaged"
LOG_DISARMED = "dsv41: image 16-bit gray FAILED"

CONVERT_REL = "multimodal/image.py"
CONVERT_MODULE = "vllm.multimodal.image"
CONVERT_OLD = """    if image.mode == to_mode:
        return image

    if to_mode == "RGB" and _has_transparency(image):
"""
CONVERT_NEW = """    if image.mode == to_mode:
        return image

    # dsv41-image-16bit-gray: Pillow clamps 16-bit gray (I;16*) to 255 in an 8-bit mode; scale by 1/257.
    if image.mode.startswith("I;16") and not to_mode.startswith(("I", "F")):
        from image_16bit_gray import gray16_to_8bit

        return convert_image_mode(gray16_to_8bit(image), to_mode, background_color)

    if to_mode == "RGB" and _has_transparency(image):
"""

LOAD_REL = "models/deepseek_v4_1/common/mm_preprocess.py"
LOAD_MODULE = "vllm.models.deepseek_v4_1.common.mm_preprocess"
LOAD_OLD = """    p = patch_size
    image = image.convert("RGB")
"""
LOAD_NEW = """    p = patch_size
    # dsv41-image-16bit-gray: Pillow clamps 16-bit gray (I;16*) to 255 in RGB; scale by 1/257.
    if image.mode.startswith("I;16"):
        from image_16bit_gray import gray16_to_8bit

        image = gray16_to_8bit(image)
    image = image.convert("RGB")
"""


def gray16_to_8bit(image):
    """I;16* -> L by round(v / 257), the API's 16-bit scaling; a tRNS key -> LA with an exact alpha mask."""
    import numpy as np
    from PIL import Image

    v = np.asarray(image).astype(np.uint32)
    out = Image.fromarray(((v + 128) // 257).astype(np.uint8))
    key = image.info.get("transparency")
    if isinstance(key, int):
        out = Image.merge("LA", (out, Image.fromarray(np.where(v == key, 0, 255).astype(np.uint8))))
    return out


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(ENV, "0") == "1"


def _swap(src: str, old: str, new: str, rel: str) -> str:
    if src.count(old) != 1:
        raise ValueError(f"anchor found {src.count(old)} times in {rel}, need 1")
    return src.replace(old, new)


def patch_convert_source(src: str) -> str:
    """multimodal/image.py with convert_image_mode scaling I;16* (idempotent)."""
    return src if MARK in src else _swap(src, CONVERT_OLD, CONVERT_NEW, CONVERT_REL)


def patch_load_source(src: str) -> str:
    """mm_preprocess.py with load_image scaling I;16* (idempotent)."""
    return src if MARK in src else _swap(src, LOAD_OLD, LOAD_NEW, LOAD_REL)


def unpatch_convert_source(src: str) -> str:
    return _swap(src, CONVERT_NEW, CONVERT_OLD, CONVERT_REL) if MARK in src else src


def unpatch_load_source(src: str) -> str:
    return _swap(src, LOAD_NEW, LOAD_OLD, LOAD_REL) if MARK in src else src


# (file under the vllm root, module, rewrite, inverse)
REWRITES = (
    (CONVERT_REL, CONVERT_MODULE, patch_convert_source, unpatch_convert_source),
    (LOAD_REL, LOAD_MODULE, patch_load_source, unpatch_load_source),
)


def install(vllm_root, rewrite, modules=None) -> str:
    """Rewrite both files with sitecustomize's `rewrite(path, fn)`; print the engagement line."""
    modules = sys.modules if modules is None else modules
    done = []
    try:
        for rel, module, patch, _ in REWRITES:
            state = rewrite(os.path.join(str(vllm_root), rel), patch)
            if state == "rewritten" and module in modules:
                raise RuntimeError(f"{module} was imported before the rewrite; this process converts with the stock code")
            done.append(f"{os.path.basename(rel)} {state}")
    except BaseException as exc:
        print(f"dsv41: image 16-bit gray FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
    detail = ", ".join(done)
    print(f"dsv41: image 16-bit gray engaged ({detail})", file=sys.stderr, flush=True)
    return detail


def restore(vllm_root, rewrite) -> list[str]:
    """DSV41_IMAGE_16BIT_GRAY=0: put the image's text back where this container's files were rewritten."""
    states = []
    for rel, _, _, unpatch in REWRITES:
        path = os.path.join(str(vllm_root), rel)
        if os.path.isfile(path):
            states.append(rewrite(path, unpatch))
    return states
