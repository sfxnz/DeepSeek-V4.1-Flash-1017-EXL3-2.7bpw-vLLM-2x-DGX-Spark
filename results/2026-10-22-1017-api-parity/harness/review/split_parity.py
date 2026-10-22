#!/usr/bin/env python3
"""Per-image resolve split (gateway) vs the API's single resolve call: token adjustment and strict-mode error
text, for a few image mixes.  venv/bin/python -I split_parity.py"""
import base64
import os
import struct
import sys
import zlib

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-gateway/gateway")
import deepseek_recipe as d  # noqa: E402
import dsv41_gateway as G  # noqa: E402

TOK = os.path.expanduser("~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/"
                         "2.7bpw-mcg-hq/tokenizer.json")


def png(w, h):
    raw = zlib.compress(b"".join(b"\x00" + b"\x10\x80\xf0" * w for _ in range(h)), 9)

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    return "data:image/png;base64," + base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", raw) + chunk(b"IEND", b"")).decode()


gw = G.Gateway(TOK, "m", strict=True)
api = d.ImageResolver(d.ReqwestImageFetcher(), d.OpenCvImagePreprocessor())
sizes = [(64, 48), (1920, 1080), (5000, 300), (300, 7000), (1024, 1024)]
for name, srcs in (
        ("5 mixed sizes/details", [d.DataUrlImageSource(png(w, h), det) for (w, h), det in zip(sizes, ("low", "high", "auto", "original", "low"))]),
        ("14 mixed", [d.DataUrlImageSource(png(*sizes[i % 5]), ("low", "high")[i % 2]) for i in range(14)]),
        ("15 mixed (one 5000 px)", [d.DataUrlImageSource(png(*sizes[i % 5]), "high") for i in range(15)]),
        ("3 in one message, 3rd broken", [d.DataUrlImageSource(png(32, 32), path=".messages[0]"),
                                          d.DataUrlImageSource(png(32, 32), path=".messages[0]"),
                                          d.DataUrlImageSource("data:image/png;base64,AAAA", path=".messages[0]")])):
    try:
        want = api.resolve(srcs, d.ImageQuota()).image_token_adjustment()
    except (d.ImageError, d.CalcResizeError) as e:
        want = f"error: {e}"
    try:
        got = gw.images(srcs)[0]
    except G.ApiError as e:
        got = f"error: {e.body['error']['message']}"
    print(f"{name:32s} API single call: {want!s:60.60s} | gateway: {got!s:.60s} | same: {want == got}")
