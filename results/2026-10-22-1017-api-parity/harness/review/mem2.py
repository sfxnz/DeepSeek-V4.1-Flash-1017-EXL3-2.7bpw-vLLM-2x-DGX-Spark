#!/usr/bin/env python3
"""K concurrent Gateway.prepare() calls (asyncio.to_thread, as K concurrent clients), each a Chat request with
N solid-colour WxH PNGs of bit depth BITS (8|16) and CH channels (3 RGB | 4 RGBA). Prints peak RSS, the wall
time, and each request's prepare time (IMAGE_LOCK waits included). Run in the serve image with --memory 2g:
  .../mem2.py K W H N [BITS] [CH]
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


def png_solid(w, h, bits=8, ch=3):
    comp = zlib.compressobj(9)
    px = (bytes([200, 30, 30, 255][:ch]) if bits == 8 else b"".join(struct.pack(">H", v) for v in (51400, 7710, 7710, 65535)[:ch]))
    row = b"\x00" + px * w
    idat = b"".join(comp.compress(row) for _ in range(h)) + comp.flush()

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    ctype = 2 if ch == 3 else 6
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, ctype, 0, 0, 0))
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


async def main(k, w, h, n, bits=8, ch=3):
    import dsv41_gateway as G

    url = "data:image/png;base64," + base64.b64encode(png_solid(w, h, bits, ch)).decode()
    raw = json.dumps({"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "thinking": {"type": "disabled"},
                      "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe."}] + [
                          {"type": "image_url", "image_url": {"url": url}}] * n}]}).encode()
    gw = G.Gateway(TOK, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    print(f"K={k} x {n} images {w}x{h} {bits}-bit {ch}ch, body {len(raw) / 2**20:.1f} MiB; idle peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB", flush=True)

    async def one():
        t0 = time.monotonic()
        try:
            p = await asyncio.to_thread(gw.prepare, "chat", json.loads(raw))
            out = f"{p.n_images} images, prompt_tokens {p.prompt_tokens}"
        except (G.Fallback, G.ApiError) as e:
            out = f"{type(e).__name__}: {str(e)[:80]}"
        return time.monotonic() - t0, out

    t0 = time.monotonic()
    res = await asyncio.gather(*(one() for _ in range(k)))
    print(f"done in {time.monotonic() - t0:.2f} s; per request {[round(t, 2) for t, _ in res]} s; {res[0][1]}; "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB", flush=True)


if __name__ == "__main__":
    asyncio.run(main(*map(int, sys.argv[1:])))
