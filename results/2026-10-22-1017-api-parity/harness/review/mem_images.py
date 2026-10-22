#!/usr/bin/env python3
"""Peak RSS and wall time of Gateway.prepare() + the upstream JSON body for requests with large images.

Usage: venv/bin/python -I mem_images.py <case> [n]
  case big:   n solid-colour 8192x8192 PNGs (~200 KB each on the wire: header dims pass the API's 8192 cap)
  case noise: n incompressible 2400x1400 PNGs (~10 MB each, 4 MiB images within the API's 64 MiB total)
Runs the same code path as the gateway's worker thread (gw.prepare) and then the event-loop step
json.dumps(prep.greq).encode(), and prints ru_maxrss before/after. GATEWAY_MEMORY in run.sh is 2g.
"""
import base64
import json
import os
import resource
import struct
import sys
import time
import zlib

GW = "/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-gateway/gateway"
TOK = os.path.expanduser("~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/"
                         "2.7bpw-mcg-hq/tokenizer.json")
sys.dont_write_bytecode = True
sys.path.insert(0, GW)


def png_rows(w, h, row_fn):
    comp = zlib.compressobj(9)
    out = []
    for y in range(h):
        out.append(comp.compress(b"\x00" + row_fn(y)))
    out.append(comp.flush())
    idat = b"".join(out)

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    case, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if case == "big":
        row = bytes([200, 30, 30]) * 8192
        png = png_rows(8192, 8192, lambda y: row)
        pngs = [png] * n
    else:
        pngs = [png_rows(2400, 1400, lambda y: os.urandom(2400 * 3)) for _ in range(n)]
    parts = [{"type": "text", "text": "Describe the images."}]
    parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p).decode()}}
              for p in pngs]
    body = {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": parts}]}
    raw = json.dumps(body).encode()
    del pngs, parts
    import dsv41_gateway as G

    gw = G.Gateway(TOK, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    before = rss_mib()
    t0 = time.monotonic()
    prep = gw.prepare("chat", json.loads(raw))
    t1 = time.monotonic()
    wire = json.dumps(prep.greq).encode()
    t2 = time.monotonic()
    print(f"case={case} n={n} request_body={len(raw) / 2**20:.1f} MiB upstream_body={len(wire) / 2**20:.1f} MiB "
          f"images={prep.n_images} prompt_tokens={prep.prompt_tokens}")
    print(f"peak RSS before prepare {before:.0f} MiB, after prepare+json.dumps {rss_mib():.0f} MiB "
          f"(prepare {t1 - t0:.2f} s in the worker thread, json.dumps+encode {1e3 * (t2 - t1):.0f} ms on the event loop)")


if __name__ == "__main__":
    main()
