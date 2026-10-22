#!/usr/bin/env python3
"""100 concurrent long-lived proxied streams through a fresh gateway (httpx's default pool: 100 connections,
pool timeout None in the gateway's client): /health and a new request meanwhile."""
import asyncio
import os
import subprocess
import sys
import time

sys.path.insert(0, "/rv")
import http_probe as hp  # noqa: E402


async def main():
    import httpx

    procs = [subprocess.Popen([sys.executable, "/rv/http_probe.py", "upstream"]),
             subprocess.Popen([sys.executable, "/gw/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18100",
                               "--tokenizer", hp.TOK, "--served-model-name", hp.SERVED],
                              env={**os.environ, "DSV41_GATEWAY_UPSTREAM": hp.UP}, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)]
    ac = httpx.AsyncClient(timeout=httpx.Timeout(60), limits=httpx.Limits(max_connections=300))
    for _ in range(200):
        try:
            if (await ac.get(hp.GW + "/health")).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.1)
    stack = []
    for _ in range(100):
        cm = ac.stream("GET", hp.GW + "/hold")
        stack.append((cm, await cm.__aenter__()))
    t0 = time.monotonic()
    h = await ac.get(hp.GW + "/health", timeout=30)
    print(f"100 open proxied streams: gateway /health -> {h.status_code} after {time.monotonic() - t0:.1f} s: {h.text[:100]}")
    t0 = time.monotonic()
    try:
        r = await ac.post(hp.GW + "/v1/chat/completions", json={"model": hp.SERVED, "thinking": {"type": "disabled"},
                                                                 "messages": [{"role": "user", "content": "hi"}]},
                          timeout=15)
        out = str(r.status_code)
    except httpx.HTTPError as e:
        out = repr(e)[:40]
    print(f"  an API request meanwhile: {out} after {time.monotonic() - t0:.1f} s; upstream /health direct: "
          f"{(await ac.get(hp.UP + '/health')).status_code}")
    for p in procs:
        p.kill()


asyncio.run(main())
