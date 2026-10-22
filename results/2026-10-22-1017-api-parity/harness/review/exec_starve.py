#!/usr/bin/env python3
"""K concurrent image requests (N photos of 4032x3024 each) through asyncio.to_thread(gw.prepare), as the
gateway runs them, and a text-only request 0.5 s later: IMAGE_LOCK waiters hold the default executor's
workers (min(32, os.cpu_count() + 4)).  .../exec_starve.py K N"""
import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, "/gw")
sys.path.insert(0, "/rv")
from mem2 import png_solid, TOK  # noqa: E402


async def main(k, n):
    import dsv41_gateway as G

    gw = G.Gateway(TOK, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    url = "data:image/png;base64," + base64.b64encode(png_solid(4032, 3024)).decode()
    img = json.dumps({"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "thinking": {"type": "disabled"},
                      "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}] * n}]})
    txt = json.dumps({"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "thinking": {"type": "disabled"},
                      "messages": [{"role": "user", "content": "hi"}]})

    async def timed(raw, delay):
        await asyncio.sleep(delay)
        t0 = time.monotonic()
        await asyncio.to_thread(gw.prepare, "chat", json.loads(raw))
        return time.monotonic() - t0

    print(f"os.cpu_count()={os.cpu_count()} -> default executor workers {min(32, os.cpu_count() + 4)}", flush=True)
    res = await asyncio.gather(*(timed(img, 0) for _ in range(k)), timed(txt, 0.5))
    print(f"{k} image requests x {n} photos: slowest {max(res[:-1], default=0):.1f} s; text-only request started 0.5 s later: "
          f"prepare {res[-1]:.2f} s")


asyncio.run(main(*map(int, sys.argv[1:])))
