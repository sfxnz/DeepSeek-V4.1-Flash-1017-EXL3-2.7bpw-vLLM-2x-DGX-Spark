#!/usr/bin/env python3
"""Where does the gateway sit after a stream client leaves during prefill (after the grace window)?
Runs the gateway app in-process (uvicorn, same loop) against http_probe.py's upstream in slow_prefill mode,
disconnects a raw-socket client at 4 s, then dumps every asyncio task's stack."""
import asyncio
import io
import json
import subprocess
import sys
import time

sys.path.insert(0, "/rv")
sys.path.insert(0, "/gw")
import http_probe as hp  # noqa: E402


async def client(leave_after):
    body = json.dumps({"model": hp.SERVED, "messages": [{"role": "user", "content": "hi"}],
                       "thinking": {"type": "disabled"}, "stream": True}).encode()
    r, w = await asyncio.open_connection("127.0.0.1", 18100)
    w.write(b"POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n"
            + f"content-length: {len(body)}\r\n\r\n".encode() + body)
    await w.drain()
    try:
        got = await asyncio.wait_for(r.read(4096), leave_after)
        await asyncio.sleep(max(0.0, leave_after - 2.1))
    except asyncio.TimeoutError:
        got = b""
    w.close()
    return got.split(b"\r\n", 1)[0]


async def main():
    import httpx
    import uvicorn

    import dsv41_gateway as G

    orig_aclose = httpx.Response.aclose

    async def aclose(self):
        print(f"[dbg] Response.aclose (is_closed={self.is_closed}) url={self.request.url.path}", flush=True)
        try:
            await orig_aclose(self)
        except BaseException as e:
            print(f"[dbg] Response.aclose raised {e!r}", flush=True)
            raise
        print("[dbg] Response.aclose returned", flush=True)

    httpx.Response.aclose = aclose
    orig_after = G.events_after

    async def events_after(first):
        print("[dbg] events_after: awaiting first", flush=True)
        try:
            async for ev in orig_after(first):
                yield ev
        except BaseException as e:
            print(f"[dbg] events_after raised {e!r}; first: done={first.done()} cancelled={first.cancelled()}", flush=True)
            raise
        print("[dbg] events_after ended normally", flush=True)

    G.events_after = events_after
    import starlette.responses as SR

    orig_call = SR.StreamingResponse.__call__

    async def call(self, scope, receive, send):
        print("[dbg] StreamingResponse.__call__ enter", flush=True)
        try:
            await orig_call(self, scope, receive, send)
        except BaseException as e:
            print(f"[dbg] StreamingResponse.__call__ raised {e!r}", flush=True)
            raise
        print("[dbg] StreamingResponse.__call__ returned", flush=True)

    SR.StreamingResponse.__call__ = call

    up = subprocess.Popen([sys.executable, "/rv/http_probe.py", "upstream"])
    gw = G.Gateway(hp.TOK, hp.SERVED, upstream=hp.UP)
    server = uvicorn.Server(uvicorn.Config(G.make_app(gw), host="127.0.0.1", port=18100, log_level="warning"))
    srv_task = asyncio.create_task(server.serve())
    async with httpx.AsyncClient() as c:
        for _ in range(100):
            try:
                if (await c.get(hp.GW + "/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
        await c.post(hp.UP + "/mock/mode", json={"mode": "slow_prefill"})
        print("client saw", await client(4.0), flush=True)
        await asyncio.sleep(3.0)
        print("upstream record 3 s after the client left:", (await c.get(hp.UP + "/mock/log")).json()[-1], flush=True)
    for t in asyncio.all_tasks():
        if t is asyncio.current_task() or t is srv_task:
            continue
        buf = io.StringIO()
        t.print_stack(file=buf)
        text = buf.getvalue()
        print("----", t.get_name(), t.get_coro(), "\n", text[-1800:], flush=True)
    server.should_exit = True
    up.terminate()
    await asyncio.sleep(0.5)


asyncio.run(main())
