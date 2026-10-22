#!/usr/bin/env python3
"""IMAGE_LOCK: one request with N images of WxH (>= 15: a single resolve call holding the lock) and, 0.5 s
later, a request with one 64x64 image. Prints both prepare times.  .../lock_starve.py N W H"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/gw")
sys.path.insert(0, "/rv")
from mem2 import png_solid, TOK  # noqa: E402
import base64  # noqa: E402


def body(n, w, h):
    url = "data:image/png;base64," + base64.b64encode(png_solid(w, h)).decode()
    return {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe."}] + [
                {"type": "image_url", "image_url": {"url": url}}] * n}]}


async def main(n, w, h):
    import dsv41_gateway as G

    gw = G.Gateway(TOK, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    heavy, light = json.dumps(body(n, w, h)), json.dumps(body(1, 64, 64))
    print(f"heavy body {len(heavy) / 2**20:.1f} MiB ({n} x {w}x{h})", flush=True)

    async def timed(raw, delay):
        await asyncio.sleep(delay)
        t0 = time.monotonic()
        try:
            await asyncio.to_thread(gw.prepare, "chat", json.loads(raw))
            out = "ok"
        except Exception as e:  # noqa: BLE001
            out = f"{type(e).__name__}: {str(e)[:60]}"
        return round(time.monotonic() - t0, 2), out

    (th, oh), (tl, ol) = await asyncio.gather(timed(heavy, 0), timed(light, 0.5))
    print(f"heavy request prepare {th} s ({oh}); 1-image request started 0.5 s later: prepare {tl} s ({ol})")


asyncio.run(main(*map(int, sys.argv[1:])))
