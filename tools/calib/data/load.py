"""Loader for the DeepSeek-V4.1-Flash-1017 calibration (`calib`) and held-out (`panel`) token sets.

Files (default root /home/sfxnz/projects/data/dsv41-1017/calib, override with DSV41_CALIB_ROOT):
    calib.jsonl, panel.jsonl   one JSON record per sequence (schema below)
    images/<sha256>.<ext>      original image bytes, referenced by records
    manifest.json              counts per set/domain, sources, revisions, seeds

Record schema (one line of calib.jsonl / panel.jsonl):
    id              str   "calib-00042" / "panel-00007"; unique across both sets
    set             str   "calib" | "panel"
    domain          str   chat | code | math | tool | multilingual | long | encyclopedic | image
    subdomain       str   finer tag, e.g. "code/agentic-swe", "image/ocr", "multilingual/zh"
    lang            str   ISO 639-1 of the conversation text ("en", "zh", ...)
    thinking_mode   str   "chat" | "thinking"  (as passed to the reference encode_messages)
    reasoning_effort int|null  numeric budget 1-100 rendered in thinking mode (null in chat mode)
    n_tokens        int   len(tokens), <= 32768
    tokens          list[int]  full input ids, BOS first, image placeholders already expanded
    loss_spans      list[[a, b]]  half-open token ranges produced by the assistant (reasoning,
                    </think>, content, DSML tool calls, EOS). The generation header is not included.
    images          list[dict] in prompt order, each:
                      path      relative to root, e.g. "images/<sha256>.jpg"
                      sha256    of the image file bytes
                      width, height   decoded size (what image_processor.load_image sees)
                      start     index of the span's first token (IMAGE_START) in tokens
                      n_tokens  span length = n_llm_h * (n_llm_w + 1) + 2
                      n_llm_h, n_llm_w, n_vit_h, n_vit_w   reference plan_image_grid result
    messages        list  the OpenAI-style conversation that was encoded (image urls = paths above)
    source          dict  dataset, revision, file, row, license, generator
    tokens_sha256   str   sha256 of the little-endian int32 token array (used for the disjointness check)

Every position inside an image span carries image_token_id 129264; token_types() rebuilds the
reference TEXT/IMAGE_START/IMAGE/IMAGE_NEW_LINE/IMAGE_END types (TEXT = -1), so
`image_mask = token_types >= 0` is the bias_vl / engram mask from inference/model.py.

Typical harness use:
    from load import load_set, loss_mask, token_types, ref_image_inputs
    for rec in load_set("panel"):
        ids = rec["tokens"]; types = token_types(rec)
        imgs = ref_image_inputs(rec, image_processor_module, model_args)  # None when no images
        logits = model.forward(torch.tensor([ids]), 0, images=[imgs] if imgs else None,
                               token_types=torch.tensor([types]) if imgs else None)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

DEFAULT_ROOT = Path(os.environ.get("DSV41_CALIB_ROOT", "/home/sfxnz/projects/data/dsv41-1017/calib"))
IMAGE_TOKEN_ID = 129264
TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = -1, 0, 1, 2, 3
DOMAINS = ("chat", "code", "math", "tool", "multilingual", "long", "encyclopedic", "image")


def iter_set(name: str, root: Path | str = DEFAULT_ROOT, domains=None) -> Iterator[dict]:
    """Yield records of `calib` or `panel` in file order, optionally only the given domains."""
    if name not in ("calib", "panel"):
        raise ValueError(f"unknown set {name!r}")
    root = Path(root)
    with open(root / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if domains is None or rec["domain"] in domains:
                rec["_root"] = str(root)
                yield rec


def load_set(name: str, root: Path | str = DEFAULT_ROOT, domains=None) -> list[dict]:
    return list(iter_set(name, root, domains))


def loss_mask(rec: dict) -> list[int]:
    """1 on assistant-produced tokens, 0 elsewhere (prompt, headers, image spans)."""
    m = [0] * rec["n_tokens"]
    for a, b in rec["loss_spans"]:
        m[a:b] = [1] * (b - a)
    return m


def token_types(rec: dict) -> list[int]:
    """Reference token types (inference/image_processor.py layout); TEXT = -1 outside image spans."""
    t = [TEXT] * rec["n_tokens"]
    for im in rec["images"]:
        span = [IMAGE_START] + ([IMAGE] * im["n_llm_w"] + [IMAGE_NEW_LINE]) * im["n_llm_h"] + [IMAGE_END]
        t[im["start"]:im["start"] + len(span)] = span
    return t


def image_path(rec: dict, i: int) -> Path:
    return Path(rec.get("_root", DEFAULT_ROOT)) / rec["images"][i]["path"]


def ref_image_inputs(rec: dict, image_processor, args):
    """Build the reference `ImageInput` list for model.forward(images=[...]).

    image_processor: the module inference/image_processor.py of the 1017 repo; args: its ModelArgs
    (vision_* fields). Pixels go through the reference load_image; the grid is checked against the record.
    Returns None for text-only records.
    """
    if not rec["images"]:
        return None
    out = []
    for i, im in enumerate(rec["images"]):
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_processor.load_image({"url": str(image_path(rec, i))}, args)
        got = (n_vit_h, n_vit_w, n_llm_h, n_llm_w)
        want = (im["n_vit_h"], im["n_vit_w"], im["n_llm_h"], im["n_llm_w"])
        if got != want:
            raise ValueError(f"{rec['id']} image {i}: grid {got} != recorded {want}")
        types = image_processor.image_token_types(n_llm_h, n_llm_w)
        out.append(image_processor.ImageInput(im["start"], patches, n_vit_h, n_vit_w, types))
    return out


def validate(rec: dict) -> None:
    """Cheap structural checks; raises AssertionError on a malformed record."""
    n = rec["n_tokens"]
    assert n == len(rec["tokens"]) <= 32768, rec["id"]
    assert rec["tokens"][0] == 0, "BOS first"
    prev = 0
    for a, b in rec["loss_spans"]:
        assert prev <= a < b <= n, (rec["id"], a, b)
        prev = b
    n_img = 0
    for im in rec["images"]:
        s, k = im["start"], im["n_tokens"]
        assert k == im["n_llm_h"] * (im["n_llm_w"] + 1) + 2
        assert all(t == IMAGE_TOKEN_ID for t in rec["tokens"][s:s + k]), rec["id"]
        n_img += k
    assert sum(t == IMAGE_TOKEN_ID for t in rec["tokens"]) == n_img, rec["id"]
