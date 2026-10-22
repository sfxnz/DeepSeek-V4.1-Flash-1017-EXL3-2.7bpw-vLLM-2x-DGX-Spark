#!/usr/bin/env python3
"""K concurrent Gateway.prepare() calls (as asyncio.to_thread runs them for K concurrent clients), each a Chat
request with N solid-colour WxH PNGs (default 8 x 8192x8192, 2.2 MiB). Args: K [W H N]. Run inside the serve image with the gateway's
container memory cap (run.sh GATEWAY_MEMORY=2g):

  docker run --rm --network none --memory 2g --cpus 4 -v /dev/null:/usr/lib/python3.12/sitecustomize.py:ro \
    -v <venv site-packages>/deepseek_recipe:/opt/dsr/deepseek_recipe:ro \
    -v <venv site-packages>/deepseek_recipe.libs:/opt/dsr/deepseek_recipe.libs:ro \
    -v <worktree>/gateway:/gw:ro -v <this dir>:/rv:ro -v ~/.cache/huggingface/hub:/hub:ro \
    -e PYTHONPATH=/opt/dsr -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 <image> /rv/mem_concurrent.py K
"""
import asyncio
import base64
import json
import resource
import struct
import sys
import time
import zlib

sys.path.insert(0, "/gw")
TOK = "/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq/tokenizer.json"


def png_solid(w, h):
    comp = zlib.compressobj(9)
    row = b"\x00" + bytes([200, 30, 30]) * w
    idat = b"".join(comp.compress(row) for _ in range(h)) + comp.flush()

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


async def main(k: int, w: int = 8192, h: int = 8192, n: int = 8):
    import dsv41_gateway as G

    url = "data:image/png;base64," + base64.b64encode(png_solid(w, h)).decode()
    raw = json.dumps({"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "thinking": {"type": "disabled"},
                      "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe."}] + [
                          {"type": "image_url", "image_url": {"url": url}}] * n}]}).encode()
    gw = G.Gateway(TOK, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    print(f"request body {len(raw) / 2**20:.1f} MiB; idle peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB",
          flush=True)
    t0 = time.monotonic()
    preps = await asyncio.gather(*(asyncio.to_thread(gw.prepare, "chat", json.loads(raw)) for _ in range(k)))
    print(f"{k} concurrent prepares done in {time.monotonic() - t0:.2f} s, {[p.n_images for p in preps]} images; "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB", flush=True)


if __name__ == "__main__":
    asyncio.run(main(*map(int, sys.argv[1:])))
