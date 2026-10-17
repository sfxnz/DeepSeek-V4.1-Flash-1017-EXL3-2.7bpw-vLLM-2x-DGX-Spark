"""Encode a conversation into DeepSeek-V4.1-Flash-1017 token ids, exactly as the reference does.

The prompt string comes from the official `encoding/encoding.py` (HF deepseek-ai/DeepSeek-V4.1-Flash-1017
@ 1bae0e1088993b9b82de08f59de5be5a34e42170). Token ids come from that revision's tokenizer.json, the same
call `inference/generate.py` makes (`tokenizer.encode(prompt)`; add_bos_token is false and the post-processor
is plain ByteLevel, so nothing is added). Each `<｜deepseek_image｜>` is expanded into the span that
`inference/image_processor.prepare_vl_inputs` builds, with the grid from the reference `plan_image_grid`.

The loss mask marks every character the assistant renders (reasoning, `</think>`, content, DSML tool calls,
EOS); the generation header (`<｜Assistant｜><think>` / `<｜Assistant｜></think>`) is prompt. Spans come from
calling the reference `render_message` per message with the reference preprocessing, and the concatenation
is asserted equal to the reference `encode_messages` output.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
REF_DIR = Path(os.environ.get("DSV41_1017_REF", "/home/sfxnz/projects/data/dsv41-1017/calib/ref"))
IMAGE_TOKEN_ID = 129264
TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = -1, 0, 1, 2, 3
MAX_SEQ_LEN = 32768


def _load_module(name: str, path: Path, stub_torch: bool = False):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    injected = False
    if stub_torch and "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            # image_processor imports torch at module level; plan_image_grid (all we call) is pure math.
            stub = types.ModuleType("torch")
            stub.Tensor = object  # only used in a dataclass annotation
            sys.modules["torch"] = stub
            injected = True
    try:
        spec.loader.exec_module(mod)
    finally:
        if injected:
            del sys.modules["torch"]
    return mod


class Encoder:
    def __init__(self, ref_dir: Path = REF_DIR):
        self.ref_dir = Path(ref_dir)
        self.enc = _load_module("dsv41_1017_encoding", self.ref_dir / "encoding" / "encoding.py")
        sys.modules.setdefault("encoding", self.enc)  # image_processor.prepare_vl_inputs imports it lazily
        self.ip = _load_module("dsv41_1017_image_processor", self.ref_dir / "inference" / "image_processor.py",
                               stub_torch=True)
        cfg = json.loads((self.ref_dir / "inference" / "config.json").read_text())
        self.vision_args = types.SimpleNamespace(**{k: v for k, v in cfg.items() if k.startswith("vision_")})
        assert cfg["image_token_id"] == IMAGE_TOKEN_ID
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(self.ref_dir / "tokenizer.json"))
        assert self.tok.token_to_id(self.enc.IMAGE_PLACEHOLDER) == IMAGE_TOKEN_ID

    # ---------------------------------------------------------------- text
    def render(self, messages, thinking_mode, reasoning_effort=None):
        """Return (prompt, image_records, assistant_char_spans) via the reference functions."""
        enc = self.enc
        prompt_ref, media = enc.encode_messages(messages, thinking_mode=thinking_mode,
                                                reasoning_effort=reasoning_effort, return_multi_modal_data=True)
        # Same steps as encode_messages / _encode_messages_text (no context, default drop_thinking=True).
        for m in messages:
            enc._validate_no_image_sp_tokens(m)
        msgs = enc.merge_tool_messages(messages)
        msgs = enc.sort_tool_results_by_call_order(msgs)
        msgs, _ = enc.process_image_messages(msgs)
        drop = not any(m.get("tools") for m in msgs)
        if thinking_mode == "thinking" and drop:
            msgs = enc._drop_thinking_messages(msgs)
        prompt, spans = enc.bos_token, []
        for i in range(len(msgs)):
            piece = enc.render_message(i, msgs, thinking_mode=thinking_mode, drop_thinking=drop,
                                       reasoning_effort=reasoning_effort)
            if msgs[i].get("role") == "assistant":
                spans.append((len(prompt), len(prompt) + len(piece)))
            prompt += piece
        if prompt != prompt_ref:
            raise AssertionError("span rendering diverged from reference encode_messages")
        return prompt, media["images"], spans

    def tokenize(self, prompt, spans):
        e = self.tok.encode(prompt, add_special_tokens=False)
        ids, offs = e.ids, e.offsets
        starts = {o[0] for o in offs}
        ends = {o[1] for o in offs}
        mask = [0] * len(ids)
        for a, b in spans:
            if a not in starts or b not in ends:
                raise AssertionError(f"assistant span {a}:{b} not on a token boundary")
        si = 0
        sp = sorted(spans)
        for t, (a, _) in enumerate(offs):
            while si < len(sp) and a >= sp[si][1]:
                si += 1
            if si < len(sp) and sp[si][0] <= a < sp[si][1]:
                mask[t] = 1
        return ids, mask

    # --------------------------------------------------------------- image
    def image_grid(self, width, height):
        n_llm_h, n_llm_w, bh, bw = self.ip.plan_image_grid(width, height, self.vision_args)
        p = self.vision_args.vision_patch_size
        return dict(n_llm_h=n_llm_h, n_llm_w=n_llm_w, n_vit_h=bh // p, n_vit_w=bw // p,
                    n_tokens=self.ip.num_image_tokens(n_llm_h, n_llm_w))

    # ---------------------------------------------------------------- full
    def encode(self, messages, thinking_mode, reasoning_effort=None, image_bytes=None):
        """image_bytes: dict url -> raw bytes for every image block url used in messages.

        Returns dict(tokens, loss_mask, images, prompt). images[i] = {url, sha256, width, height, start,
        n_tokens, n_llm_h, n_llm_w, n_vit_h, n_vit_w}; tokens[start:start+n_tokens] are all IMAGE_TOKEN_ID.
        """
        prompt, recs, spans = self.render(messages, thinking_mode, reasoning_effort)
        ids, mask = self.tokenize(prompt, spans)
        n_ph = sum(1 for t in ids if t == IMAGE_TOKEN_ID)
        if n_ph != len(recs):
            raise ValueError(f"{n_ph} image tokens for {len(recs)} images")
        out_ids, out_mask, images = [], [], []
        it = iter(recs)
        for t, m in zip(ids, mask):
            if t != IMAGE_TOKEN_ID:
                out_ids.append(t)
                out_mask.append(m)
                continue
            rec = next(it)
            raw = image_bytes[rec["url"]]
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as im:  # load_image uses the decoded size (no EXIF transpose)
                w, h = im.convert("RGB").size
            g = self.image_grid(w, h)
            images.append(dict(url=rec["url"], sha256=hashlib.sha256(raw).hexdigest(), width=w, height=h,
                               start=len(out_ids), **g))
            out_ids += [IMAGE_TOKEN_ID] * g["n_tokens"]
            out_mask += [0] * g["n_tokens"]
        return dict(tokens=out_ids, loss_mask=out_mask, images=images, prompt=prompt)


def token_types(n_tokens, images):
    """Reference layout: TEXT=-1 elsewhere; each image span is START, (IMAGE*w + NEW_LINE)*h, END."""
    types_ = [TEXT] * n_tokens
    for im in images:
        span = [IMAGE_START] + ([IMAGE] * im["n_llm_w"] + [IMAGE_NEW_LINE]) * im["n_llm_h"] + [IMAGE_END]
        assert len(span) == im["n_tokens"]
        types_[im["start"]:im["start"] + len(span)] = span
    return types_
