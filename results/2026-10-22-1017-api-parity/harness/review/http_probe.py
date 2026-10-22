#!/usr/bin/env python3
"""HTTP-level probes of gateway/dsv41_gateway.py (run as run.sh runs it) against a scripted upstream.

One container of the serve image, no network: the fake upstream (Starlette/uvicorn, 127.0.0.1:18101), the
gateway process (127.0.0.1:18100), and the client checks in this process.

  docker run --rm --network none --memory 4g --cpus 4 -v /dev/null:/usr/lib/python3.12/sitecustomize.py:ro \
    -v <venv>/deepseek_recipe:/opt/dsr/deepseek_recipe:ro -v <venv>/deepseek_recipe.libs:/opt/dsr/deepseek_recipe.libs:ro \
    -v <worktree>/gateway:/gw:ro -v <this dir>:/rv:ro -v ~/.cache/huggingface/hub:/hub:ro \
    -e PYTHONPATH=/opt/dsr -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 <image> /rv/http_probe.py
"""
import asyncio
import json
import os
import subprocess
import sys
import time

UP, GW = "http://127.0.0.1:18101", "http://127.0.0.1:18100"
TOK = "/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq/tokenizer.json"
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def upstream():
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    state = {"mode": "ok", "log": [], "prefill_s": 30}

    def ev(rid, ids, finish=None):
        return "data: " + json.dumps({"request_id": rid, "choices": [
            {"index": 0, "logprobs": None, "finish_reason": finish, "token_ids": ids}], "usage": None}) + "\n\n"

    async def generate(request):
        body = await request.json()
        rid, mode = body["request_id"], state["mode"]
        rec = {"mode": mode, "cancelled_after_s": None, "sent": 0, "done": False}
        state["log"].append(rec)
        t0 = time.monotonic()

        async def gen():
            try:
                if mode == "slow_prefill":
                    await asyncio.sleep(state["prefill_s"])
                if mode in ("slow_drop", "slow_malformed"):
                    await asyncio.sleep(state["prefill_s"])
                    if mode == "slow_malformed":
                        yield "data: [1, 2]\n\n"
                        return
                    raise RuntimeError("simulated upstream death during prefill")
                for _ in range(3):
                    yield ev(rid, [16])
                    rec["sent"] += 1
                    await asyncio.sleep(0.05)
                if mode == "drop":
                    raise RuntimeError("simulated upstream death mid-stream")
                if mode == "malformed":
                    yield 'data: {"choices": [\n\n'
                yield ev(rid, [16, 1], "stop")
                yield "data: " + json.dumps({"request_id": rid, "choices": [], "usage": {
                    "prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}) + "\n\n"
                yield "data: [DONE]\n\n"
                rec["done"] = True
            except asyncio.CancelledError:
                rec["cancelled_after_s"] = round(time.monotonic() - t0, 2)
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def hold(request):
        async def gen():
            while True:
                yield b"."
                await asyncio.sleep(1)
        return StreamingResponse(gen(), media_type="text/plain")

    async def slow(request):
        rec = {"mode": "proxied_slow", "cancelled_after_s": None, "sent": 0, "done": False}
        state["log"].append(rec)
        t0 = time.monotonic()

        async def gen():
            try:
                await asyncio.sleep(30)
                yield b"data: late\n\n"
                rec["done"] = True
            except asyncio.CancelledError:
                rec["cancelled_after_s"] = round(time.monotonic() - t0, 2)
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    def with_cancellation(handler):
        """vLLM's entrypoints/serve/utils/api_utils.py:with_cancellation (non-stream generation is
        cancelled when its client disconnects)."""
        async def wrapper(request):
            await request.body()
            work = asyncio.ensure_future(handler(request))

            async def gone():
                while (await request.receive())["type"] != "http.disconnect":
                    pass

            watch = asyncio.ensure_future(gone())
            done, pending = await asyncio.wait([work, watch], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            return work.result() if work in done else Response(status_code=499)
        return wrapper

    async def slow_nonstream(request):
        rec = {"mode": "slow_nonstream " + request.url.path, "cancelled_after_s": None, "sent": 0, "done": False}
        state["log"].append(rec)
        t0 = time.monotonic()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            rec["cancelled_after_s"] = round(time.monotonic() - t0, 2)
            raise
        rec["done"] = True
        return JSONResponse({"choices": []})

    async def chat(request):
        if state["mode"] == "slow_chat":
            return await slow_nonstream(request)
        return await echo(request)

    async def echo(request):
        raw = await request.body()
        return JSONResponse({"method": request.method, "path": request.scope["path"],
                             "raw_path": request.scope.get("raw_path", b"").decode(),
                             "query": request.scope["query_string"].decode(), "len": len(raw),
                             "host": request.headers.get("host")})

    async def mode(request):
        state.update(await request.json())
        return JSONResponse({})

    async def log(request):
        return JSONResponse(state["log"])

    app = Starlette(routes=[
        Route("/health", lambda r: Response(status_code=200), methods=["GET"]),
        Route("/inference/v1/generate", generate, methods=["POST"]),
        Route("/hold", hold, methods=["GET"]), Route("/slow", slow, methods=["GET"]),
        Route("/mock/mode", mode, methods=["POST"]), Route("/mock/log", log, methods=["GET"]),
        Route("/v1/chat/completions", with_cancellation(chat), methods=["POST"]),
        Route("/v1/completions", with_cancellation(slow_nonstream), methods=["POST"]),
        Route("/v1/responses/{rid:path}", echo, methods=["GET"]),
    ])
    uvicorn.run(app, host="127.0.0.1", port=18101, log_level="critical")


def main():
    import httpx

    procs = [subprocess.Popen([sys.executable, __file__, "upstream"])]
    env = {**os.environ, "DSV41_GATEWAY_UPSTREAM": UP}
    procs.append(subprocess.Popen([sys.executable, "/gw/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18100",
                                   "--tokenizer", TOK, "--served-model-name", SERVED], env=env,
                                  stdout=open("/tmp/gw.log", "w"), stderr=subprocess.STDOUT))
    c = httpx.Client(timeout=60, limits=httpx.Limits(max_connections=300))
    for _ in range(200):
        try:
            if c.get(GW + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    chat = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}

    def setmode(m):
        c.post(UP + "/mock/mode", json={"mode": m})

    def post(body, stream):
        try:
            if stream:
                with c.stream("POST", GW + "/v1/chat/completions", json=body) as r:
                    data = b""
                    try:
                        for b in r.iter_raw():
                            data += b
                        err = None
                    except httpx.HTTPError as e:
                        err = repr(e)[:110]
                    return r.status_code, r.headers.get("content-type"), r.headers.get("x-dsv41-gateway"), data, err
            r = c.post(GW + "/v1/chat/completions", json=body)
            return r.status_code, r.headers.get("content-type"), r.headers.get("x-dsv41-gateway"), r.content, None
        except httpx.HTTPError as e:
            return None, None, None, b"", repr(e)[:110]

    for m in ("ok", "drop", "malformed"):
        for stream in (False, True):
            setmode(m)
            st, ct, hdr, data, err = post({**chat, "stream": stream}, stream)
            tail = data.decode(errors="replace").strip().split("\n\n")[-1][-110:]
            print(f"upstream {m:9s} {'stream    ' if stream else 'non-stream'}: status={st} type={ct} "
                  f"x-dsv41-gateway={hdr} client_error={err} last={tail!r}")

    # client disconnect within / after the 2 s grace window while vLLM is still in prefill
    for wait in (1.0, 4.0):
        setmode("slow_prefill")
        t0 = time.monotonic()
        with httpx.Client(timeout=60) as c2:
            with c2.stream("POST", GW + "/v1/chat/completions", json={**chat, "stream": True}) as r:
                time.sleep(max(0.0, wait - (time.monotonic() - t0)))
        time.sleep(1.0)
        rec = c.get(UP + "/mock/log").json()[-1]
        print(f"stream, client gone at {wait:.0f} s during a 30 s prefill: upstream generator cancelled after "
              f"{rec['cancelled_after_s']} s (sent {rec['sent']} events)")

    # proxy: trailing-slash redirect issued by the upstream router, and a percent-encoded path
    r = c.post(GW + "/v1/chat/completions/", json={**chat, "echo": True})
    print(f"POST /v1/chat/completions/ via gateway: {r.status_code} location={r.headers.get('location')}")
    r = c.post(UP + "/v1/chat/completions/", json={**chat, "echo": True}, headers={"host": "spark1.lan:8000"})
    print(f"  same request to the upstream directly with the client's Host: {r.status_code} "
          f"location={r.headers.get('location')}")
    r = c.get(GW + "/v1/responses/resp%2Fx%3Fy%23z?a=1")
    print(f"GET /v1/responses/resp%2Fx%3Fy%23z?a=1 via gateway -> upstream saw {r.text}")
    r = c.post(GW + "/v1/chat/completions", json={**chat, "echo": True}, headers={"expect": "100-continue"})
    print(f"fallback with Expect: 100-continue: {r.status_code} {r.headers.get('x-dsv41-gateway')} {r.text[:80]}")

    # httpx pool: 100 long-lived proxied streams (the client default limit), then /health and an API call
    async def pool():
        ac = httpx.AsyncClient(timeout=httpx.Timeout(60), limits=httpx.Limits(max_connections=300))
        stack = []
        for _ in range(100):
            cm = ac.stream("GET", GW + "/hold")
            stack.append((cm, await cm.__aenter__()))
        t0 = time.monotonic()
        h = await ac.get(GW + "/health", timeout=30)
        print(f"100 open proxied streams: gateway /health -> {h.status_code} after {time.monotonic() - t0:.1f} s: {h.text[:120]}")
        t0 = time.monotonic()
        try:
            r = await ac.post(GW + "/v1/chat/completions", json=chat, timeout=15)
            out = f"{r.status_code}"
        except httpx.HTTPError as e:
            out = repr(e)[:60]
        print(f"  an API request meanwhile: {out} after {time.monotonic() - t0:.1f} s; upstream /health direct: "
              f"{(await ac.get(UP + '/health')).status_code}")
        for cm, _ in stack:
            await cm.__aexit__(None, None, None)
        await ac.aclose()

    setmode("ok")
    asyncio.run(pool())
    for p in procs:
        p.terminate()
    print("--- gateway log (tail) ---")
    print("\n".join(open("/tmp/gw.log").read().splitlines()[-12:]))


if __name__ == "__main__":
    if sys.argv[1:] == ["upstream"]:
        upstream()
    else:
        sys.path.insert(0, "/gw")
        main()
