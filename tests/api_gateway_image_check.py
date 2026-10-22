#!/usr/bin/env python3
"""In-image integration check for gateway/dsv41_gateway.py: the gateway process over HTTP against
vLLM's own /inference/v1/generate route.

Two containers of the serve image share one network namespace (no network access):

  upstream  vLLM's token-in/token-out route (its router, ServingTokens and exception handlers, built
            from the pack's config on CPU) over a scripted engine, with docker/patch mounted as run.sh
            mounts it, plus /mock/* controls and a recorder that answers every other route:
              python3 api_gateway_image_check.py upstream --port 18001 --model <pack snapshot>
  check     the gateway as run.sh runs it (image sitecustomize masked, deepseek_recipe on PYTHONPATH)
            and the checks:
              python3 api_gateway_image_check.py check --upstream http://127.0.0.1:18001 \\
                  --gateway <gateway dir>/dsv41_gateway.py --tokenizer <pack snapshot>/tokenizer.json

The checks: prompt token IDs, max_tokens and sampling of every fixture request (/dsv41/render); the SSE
frames and bodies of every stream fixture through vLLM's real token stream (multi-token deltas, stop
sequences that abort vLLM's request, length, logprobs, cache hits); images bound in placeholder order
through vLLM's processor; vLLM errors before and after the first event in the API's error shape;
fallbacks and the proxy byte for byte; client disconnects abort vLLM's request; /health; strict mode,
private-URL blocking and the context-limit 400; four concurrent streams; startup time, RSS and the
convert+encode latency of the ~74K-token agentic request. tests/test_api_gateway.py ImageTests runs
both containers (DSV41_IMAGE_TESTS=1). Prints one PASS/FAIL line per check and the measurements;
exits 1 on any failure.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/api_gateway"
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
IMAGE_TOKEN = 129264


# ---------------------------------------------------------------- upstream (vLLM image, patches on)

class ScriptedEngine:
    """EngineClient stand-in: replays the posted script as V1 delta outputs. Logprobs are built as
    LogprobsProcessor builds them (the sampled token first, then the top k); `raise_at` raises a
    ValueError before that chunk, as an engine-side validation error would."""

    def __init__(self, vllm_config, renderer):
        self.vllm_config, self.model_config, self.renderer = vllm_config, vllm_config.model_config, renderer
        self.input_processor = self.io_processor = None
        self.errored, self.dead_error = False, RuntimeError("engine dead")
        self.script = {"chunks": [[16], [1]], "finish": "stop"}
        self.log = []

    def check_admission(self, n=1):
        return None

    async def is_tracing_enabled(self):
        return False

    async def abort(self, *args, **kwargs):
        return None

    async def notify_kv_transfer_request_rejected(self, *args, **kwargs):
        return None

    def generate(self, engine_input, sampling_params, request_id, **kwargs):
        placeholders = engine_input.get("mm_placeholders") or {}
        entry = {"request_id": request_id, "prompt_token_ids": list(engine_input.get("prompt_token_ids") or []),
                 "mm_placeholders": [[p.offset, p.length] for p in placeholders.get("image", [])],
                 "sent": 0, "finished": False, "aborted": False}
        self.log.append(entry)
        return self._replay(entry, sampling_params, dict(self.script))

    async def _replay(self, entry, sp, script):
        from vllm.logprobs import Logprob
        from vllm.outputs import CompletionOutput, RequestOutput

        chunks, pos = script["chunks"], 0
        try:
            for i, ids in enumerate(chunks):
                await asyncio.sleep(script.get("delay", 0.0) + (script.get("first_delay", 0.0) if i == 0 else 0.0))
                if script.get("raise_at") == i:
                    raise ValueError(script.get("error", "scripted engine error"))
                lps = None
                if sp.logprobs is not None:
                    lps = []
                    for tok, (lp, ranked) in zip(ids, script["logprobs"][pos:pos + len(ids)]):
                        alts = {tok: Logprob(logprob=lp, rank=1, decoded_token=None)}
                        for rank, (alt, alt_lp) in enumerate(ranked[:sp.logprobs], 1):
                            alts.setdefault(alt, Logprob(logprob=alt_lp, rank=rank, decoded_token=None))
                        lps.append(alts)
                pos += len(ids)
                final = i == len(chunks) - 1
                out = CompletionOutput(index=0, text="", token_ids=ids, cumulative_logprob=None, logprobs=lps,
                                       finish_reason=script["finish"] if final else None, stop_reason=None)
                entry["sent"] = i + 1
                yield RequestOutput(request_id=entry["request_id"], prompt=None,
                                    prompt_token_ids=entry["prompt_token_ids"], prompt_logprobs=None, outputs=[out],
                                    finished=final, num_cached_tokens=script.get("cached", 0))
            entry["finished"] = True
        finally:
            entry["aborted"] = not entry["finished"]


async def vllm_state(model: str):
    """The API-server objects `vllm serve` builds for the run.sh flags that shape request handling."""
    from starlette.datastructures import State
    from vllm import AsyncEngineArgs, envs
    from vllm.config import VllmConfig
    from vllm.entrypoints.generate.api_router import init_generate_state
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.entrypoints.launchers.render.app_state import init_render_app_state
    from vllm.entrypoints.scale_out.factories import init_scale_out_state
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    args = make_arg_parser(FlexibleArgumentParser()).parse_args([
        "--model", model, "--max-model-len", "500000", "--tokenizer-mode", "deepseek_v41",
        "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice", "--reasoning-parser", "deepseek_v41",
        "--served-model-name", SERVED, "--trust-remote-code", "--mm-processor-cache-gb", "1",
        "--enable-prompt-tokens-details", "--max-logprobs", "21"])
    model_config = AsyncEngineArgs.from_cli_args(args).create_model_config()
    model_config.quantization = None  # no weights are loaded (launchers/render/entry.py)
    envs.VLLM_CPU_KVCACHE_SPACE = 0
    vllm_config = VllmConfig(model_config=model_config)
    state = State()
    await init_render_app_state(vllm_config, state, args)
    engine = ScriptedEngine(vllm_config, state.online_renderer.renderer)
    state.engine_client = engine
    args.structured_outputs_config.reasoning_parser = args.reasoning_parser  # as `vllm serve` does
    await init_generate_state(engine, state, args, None, ("generate",))
    init_scale_out_state(state, args, engine, None)
    return state, engine


def run_upstream(port: int, model: str) -> None:
    import contextlib

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from vllm.entrypoints.scale_out.token_in_token_out.api_router import router
    from vllm.entrypoints.serve.exception_handling.register import init_exception_handler

    bodies, echoes, health = {}, [], {"status": 200}
    engine = None

    @contextlib.asynccontextmanager
    async def lifespan(app):
        nonlocal engine
        state, engine = await vllm_state(model)
        app.state.serving_tokens, app.state.args = state.serving_tokens, state.args
        print("mock upstream ready", file=sys.stderr, flush=True)
        yield

    class Tee:
        """Record /inference/v1/generate bodies as sent (plain ASGI: disconnects pass through)."""

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http" or scope["path"] != "/inference/v1/generate":
                return await self.app(scope, receive, send)
            parts = []

            async def tee_receive():
                msg = await receive()
                if msg["type"] == "http.request":
                    parts.append(msg.get("body", b""))
                    if not msg.get("more_body"):
                        body = json.loads(b"".join(parts))
                        bodies[body.get("request_id")] = body
                return msg

            if engine.script.get("raw_sse") is None:
                return await self.app(scope, tee_receive, send)
            # A broken upstream: the scripted SSE bytes, then optionally a dropped connection.
            while (await tee_receive()).get("more_body"):
                pass
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            for frame in engine.script["raw_sse"]:
                await send({"type": "http.response.body", "body": frame.encode(), "more_body": True})
            if engine.script.get("drop"):
                raise RuntimeError("scripted connection drop")  # uvicorn cuts the chunked body
            await send({"type": "http.response.body", "body": b""})

    app = FastAPI(lifespan=lifespan)
    init_exception_handler(app)
    app.add_middleware(Tee)

    async def get_health(request):
        return Response(status_code=health["status"])

    async def set_health(request):
        health.update(await request.json())
        return JSONResponse({})

    async def set_script(request):
        engine.script = await request.json()
        return JSONResponse({})

    async def reset(request):
        bodies.clear()
        echoes.clear()
        engine.log.clear()
        return JSONResponse({})

    async def get_log(request):
        return JSONResponse({"engine": engine.log, "bodies": bodies, "echoes": echoes})

    async def red_png(request):
        from api_gateway_fixtures import RED  # noqa: PLC0415
        return Response(base64.b64decode(RED.split(",", 1)[1]), media_type="image/png")

    for path, fn, method in (("/health", get_health, "GET"), ("/mock/health", set_health, "POST"),
                             ("/mock/script", set_script, "POST"), ("/mock/reset", reset, "POST"),
                             ("/mock/log", get_log, "GET"), ("/mock/red.png", red_png, "GET")):
        app.add_route(path, fn, methods=[method])
    app.include_router(router)

    async def record(request):
        raw = await request.body()
        rec = {"method": request.method, "path": request.url.path, "raw_path": request.scope["raw_path"].decode(),
               "query": request.scope["query_string"].decode(), "body_sha256": hashlib.sha256(raw).hexdigest(),
               "headers": [[k.decode("latin-1"), v.decode("latin-1")] for k, v in request.headers.raw]}
        echoes.append(rec)
        t0, delay = time.monotonic(), float(request.headers.get("x-mock-delay", 0))
        while time.monotonic() - t0 < delay:  # hold the answer, watching for the caller to go away
            if await request.is_disconnected():
                rec["client_gone_after_s"] = round(time.monotonic() - t0, 2)
                return Response(status_code=499)
            await asyncio.sleep(0.05)
        headers = [("x-mock", "1"), ("x-multi", "a"), ("x-multi", "b")]
        if b'"stream": true' in raw or b'"stream":true' in raw:
            frames = [f"data: {{\"echo\": {i}, \"path\": \"{request.url.path}\"}}\n\n".encode() for i in range(3)]
            frames.append(b"data: [DONE]\n\n")
            rec["response"] = b"".join(frames).decode()

            async def gen():
                for f in frames:
                    yield f
                    await asyncio.sleep(0.01)

            out = StreamingResponse(gen(), media_type="text/event-stream")
        else:
            payload = json.dumps({"echo": rec["path"], "query": rec["query"], "sha": rec["body_sha256"]})
            rec["response"] = payload
            out = Response(payload, status_code=200 if request.method != "DELETE" else 202,
                           media_type="application/json")
        out.raw_headers += [(k.encode(), v.encode()) for k, v in headers]
        return out

    app.add_route("/{path:path}", record, methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    sys.path.insert(0, str(ROOT / "tools"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ---------------------------------------------------------------- check (gateway image, no sitecustomize)

class Results:
    def __init__(self):
        self.rows, self.numbers = [], {}

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok or not detail else f" -- {detail[:600]}"), flush=True)
        return bool(ok)


def rss_kib(pid: int) -> dict:
    status = Path(f"/proc/{pid}/status").read_text()
    return {k: int(status.split(f"{k}:")[1].split()[0]) for k in ("VmRSS", "VmHWM")}


def first_diff(got: str, want: str) -> str:
    i = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    return f"at {i}: got {got[max(0, i - 80):i + 120]!r} want {want[max(0, i - 80):i + 120]!r}"


def upstream_sockets(pid: int, port: int) -> int:
    """ESTABLISHED TCP connections that process `pid` holds to `port`."""
    inodes = set()
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if link.startswith("socket:["):
            inodes.add(link[8:-1])
    return sum(1 for table in ("/proc/net/tcp", "/proc/net/tcp6") for f in
               (line.split() for line in Path(table).read_text().splitlines()[1:])
               if f[3] == "01" and int(f[2].rsplit(":", 1)[1], 16) == port and f[9] in inodes)


class Checker:
    def __init__(self, args):
        import httpx

        sys.path.insert(0, str(ROOT / "tools"))
        sys.path.insert(0, str(Path(args.gateway).parent))
        import api_gateway_fixtures as fx

        self.fx, self.args, self.r = fx, args, Results()
        self.http = httpx.Client(timeout=120)
        self.up = args.upstream.rstrip("/")
        self.prompts = json.loads((FIXTURES / "prompts.json").read_text())
        self.streams = json.loads((FIXTURES / "streams.json").read_text())["cases"]
        self.fallbacks = json.loads((FIXTURES / "fallbacks.json").read_text())["cases"]
        self.procs = []

    # ---- plumbing
    def start_gateway(self, port: int, **env) -> tuple:
        log = tempfile.NamedTemporaryFile("w+", prefix="gateway-", suffix=".log", delete=False)
        t0 = time.monotonic()
        proc = subprocess.Popen(
            [sys.executable, self.args.gateway, "--host", "127.0.0.1", "--port", str(port), "--tokenizer",
             self.args.tokenizer, "--served-model-name", SERVED],
            env={**os.environ, "DSV41_GATEWAY_UPSTREAM": self.up, **env}, stdout=log, stderr=subprocess.STDOUT)
        self.procs.append(proc)
        url = f"http://127.0.0.1:{port}"
        while time.monotonic() - t0 < 60 and proc.poll() is None:
            try:
                resp = self.http.get(url + "/health")
                if resp.status_code == 200 and resp.json().get("gateway") == "ok":
                    return proc, url, time.monotonic() - t0, log.name
            except Exception:  # noqa: BLE001 - not listening yet
                pass
            time.sleep(0.05)
        raise RuntimeError(f"gateway did not start: {Path(log.name).read_text()[-3000:]}")

    def script(self, **script):
        self.http.post(self.up + "/mock/script", json=script).raise_for_status()
        self.http.post(self.up + "/mock/reset").raise_for_status()

    def mock_log(self) -> dict:
        return self.http.get(self.up + "/mock/log").json()

    def wait_aborted(self, timeout: float = 5.0) -> dict:
        t0 = time.monotonic()
        while True:
            log = self.mock_log()
            entry = log["engine"][-1] if log["engine"] else {}
            if entry.get("aborted") or entry.get("finished") or time.monotonic() - t0 > timeout:
                return entry
            time.sleep(0.05)

    def post(self, url: str, body, stream: bool, **kw):
        """(status, headers, raw bytes) of one request."""
        if stream:
            with self.http.stream("POST", url, json=body, **kw) as resp:
                return resp.status_code, resp.headers, b"".join(resp.iter_raw())
        resp = self.http.post(url, json=body, **kw)
        return resp.status_code, resp.headers, resp.content

    # ---- checks
    def prompts_render(self, gw: str):
        bad = []
        for c in self.prompts["cases"]:
            got = self.http.post(f"{gw}/dsv41/render/{c['protocol']}", json=c["body"]).json()
            sp = got.get("sampling") or {}
            want = (c["ids"], c["max_tokens"], c["temperature"], c["top_p"], c["n_images"], c["thinking"], c["effort"])
            have = (got.get("prompt_token_ids"), got.get("max_tokens"), sp.get("temperature"), sp.get("top_p"),
                    got.get("image_count"), got.get("thinking"), got.get("effort"))
            # The prompt text is the API's except for the image placeholder's spelling (pack vs bundled).
            prompt_ok = c["n_images"] or got.get("prompt") == c["prompt"]
            if have != want or got.get("fallback_reason") is not None or not prompt_ok:
                bad.append(c["name"])
        self.r.add(f"render: prompt token IDs, max_tokens, sampling, thinking, effort = the API's "
                   f"({len(self.prompts['cases']) - len(bad)}/{len(self.prompts['cases'])})", not bad, repr(bad))

    def stream_fixtures(self, gw: str):
        route = {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}
        for s in self.streams:
            early = s["consumed"] < len(s["chunks"])
            self.script(chunks=s["chunks"], finish=s["finish"], logprobs=s["logprobs"], cached=s["cached"],
                        delay=0.05 if early else 0.0)
            status, headers, raw = self.post(gw + route[s["protocol"]], s["body"], s["stream"])
            log = self.mock_log()
            body = next(iter(log["bodies"].values()), {})
            rid = body.get("request_id", "?")
            text = raw.decode()
            if s["stream"]:
                frames = [f + "\n\n" for f in text.split("\n\n") if f]
                got, want = [self.fx.normalize(f, rid) for f in frames], s["frames"]
                same = got == want
                detail = "" if same else next((first_diff(a, b) for a, b in zip(got, want) if a != b),
                                              f"{len(got)} vs {len(want)} frames")
            else:
                got, want = self.fx.normalize(text, rid), s["response"]
                same, detail = got == want, first_diff(got, want) if got != want else ""
            sp = body.get("sampling_params", {})
            sent_ok = (sp.get("detokenize") is False and body.get("stream") is True and "stop" not in sp
                       and len(body.get("token_ids", [])) == s["prompt_tokens"])
            ok = status == 200 and headers.get("x-dsv41-gateway") == "api" and same and sent_ok
            name = f"stream fixture {s['name']}: {'SSE' if s['stream'] else 'body'} = the API's"
            if early:
                entry = self.wait_aborted()
                ok = ok and entry.get("aborted") and entry.get("sent", 99) < len(s["chunks"])
                name += f"; stop sequence aborts vLLM after {entry.get('sent')}/{len(s['chunks'])} chunks"
            self.r.add(name, ok, f"status {status} hdr {headers.get('x-dsv41-gateway')} sent_ok {sent_ok} {detail}")

    def images(self, gw: str):
        fx = self.fx
        red, blue = (base64.b64decode(u.split(",", 1)[1]) for u in (fx.RED, fx.BLUE))
        order = {"chat_images": [red, blue], "chat_tool_result_images": [red, blue], "msg_images": [red],
                 "msg_tool_result_image": [blue], "resp_input_image": [red]}
        for c in self.prompts["cases"]:
            if c["name"] not in order:
                continue
            self.script(chunks=[[16], [1]], finish="stop")
            status, _, _ = self.post(gw + {"chat": "/v1/chat/completions", "messages": "/v1/messages",
                                           "responses": "/v1/responses"}[c["protocol"]], c["body"], False)
            log = self.mock_log()
            body, entry = next(iter(log["bodies"].values()), {}), (log["engine"] or [{}])[-1]
            parts = [base64.b64decode(p["url"].split(",", 1)[1]) for p in body.get("content_parts", [])]
            spans = entry.get("mm_placeholders", [])
            ids = entry.get("prompt_token_ids", [])
            spans_ok = len(spans) == c["n_images"] and all(ids[o:o + n].count(IMAGE_TOKEN) >= n - 1 for o, n in spans)
            if c["name"].startswith("chat_") and len(spans) == 2:
                spans_ok = spans_ok and spans[0][1] < spans[1][1]  # the 32x24 image binds first
            vllm_tokens = sum(n for _, n in spans)
            api_tokens = c["n_images"] + c["image_adjust"]
            self.r.add(f"images {c['name']}: original bytes in placeholder order, vLLM spans {spans} "
                       f"(vLLM {vllm_tokens} vs API {api_tokens} image tokens)",
                       status == 200 and parts == order[c["name"]] and spans_ok, f"status {status} spans {spans}")

    def upstream_errors(self, gw: str):
        url = gw + "/v1/chat/completions"
        base = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
        for stream in (False, True):
            kind = "stream" if stream else "non-stream"
            self.script(chunks=[[16], [1]], finish="stop")
            status, headers, raw = self.post(url, {**base, "top_k": -5, "stream": stream}, stream)
            body = json.loads(raw or b"{}")
            self.r.add(f"vLLM 400 before the stream ({kind}): 400 in the API's error shape",
                       status == 400 and set(body.get("error", {})) == {"message", "type", "param", "code"}
                       and body["error"]["type"] == "invalid_request_error" and headers.get("x-dsv41-gateway") == "api"
                       and "top_k" in body["error"]["message"], f"{status} {raw[:300]!r}")
            self.script(chunks=[[16], [1]], finish="stop", raise_at=0, error="grammar rejected: multipleOf")
            status, headers, raw = self.post(url, {**base, "stream": stream}, stream)
            body = json.loads(raw or b"{}") if raw[:1] == b"{" else {}
            self.r.add(f"vLLM rejection as its first stream event, at once ({kind}): 400 in the API's error shape, "
                       f"no SSE", status == 400 and body.get("error", {}).get("message") == "grammar rejected: multipleOf"
                       and body["error"]["code"] == "invalid_request_error", f"{status} {raw[:300]!r}")
            self.script(chunks=[[16], [16], [16], [1]], finish="stop", raise_at=2, error="engine failed mid-stream")
            status, headers, raw = self.post(url, {**base, "stream": stream}, stream)
            if stream:
                frames = [f for f in raw.decode().split("\n\n") if f]
                ok = (status == 200 and len(frames) >= 2 and json.loads(frames[-1][6:])["error"]["message"]
                      == "engine failed mid-stream" and '"content":"' in frames[0])
            else:
                ok = status == 400 and json.loads(raw)["error"]["message"] == "engine failed mid-stream"
            self.r.add(f"vLLM error after output started ({kind}): the API's error shape ends it", ok,
                       f"{status} {raw[-300:]!r}")

    def fallbacks_and_proxy(self, gw: str, log_path: str):
        headers = {"authorization": "Bearer t0k3n", "x-test": "1", "user-agent": "gateway-check"}
        bad = []
        for c in self.fallbacks:
            self.script(chunks=[[16], [1]], finish="stop")
            raw_body = json.dumps(c["body"]).encode()
            resp = self.http.post(gw + {"chat": "/v1/chat/completions", "messages": "/v1/messages",
                                        "responses": "/v1/responses"}[c["protocol"]], content=raw_body,
                                  headers={**headers, "content-type": "application/json"})
            echo = (self.mock_log()["echoes"] or [{}])[-1]
            sent = dict((k, v) for k, v in echo.get("headers", []))
            if not (resp.headers.get("x-dsv41-gateway") == "fallback" and resp.content == echo.get("response", "").encode()
                    and echo.get("body_sha256") == hashlib.sha256(raw_body).hexdigest() and echo.get("method") == "POST"
                    and all(sent.get(k) == v for k, v in headers.items())
                    and sent.get("content-length") == str(len(raw_body))
                    and resp.headers.get_list("x-multi") == ["a", "b"]):
                bad.append(c["name"])
        self.r.add(f"fallbacks reach vLLM unchanged and come back byte for byte ({len(self.fallbacks) - len(bad)}/"
                   f"{len(self.fallbacks)})", not bad, repr(bad))
        logged = Path(log_path).read_text()
        self.r.add("one log line per fallback with its reason", logged.count("dsv41-gateway: fallback ") >= len(
            self.fallbacks) and "fallback chat: vLLM field guided_json" in logged, logged[-500:])
        raw = b'{"model": "x", "messages": [}'
        resp = self.http.post(gw + "/v1/chat/completions?a=1", content=raw, headers={"content-type": "application/json"})
        echo = (self.mock_log()["echoes"] or [{}])[-1]
        self.r.add("a body that is not JSON falls back unchanged",
                   resp.headers.get("x-dsv41-gateway") == "fallback" and echo.get("query") == "a=1"
                   and echo.get("body_sha256") == hashlib.sha256(raw).hexdigest(), repr(echo)[:300])
        body = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "echo": True, "stream": True}
        status, hdrs, got = self.post(gw + "/v1/chat/completions", body, True)
        echo = (self.mock_log()["echoes"] or [{}])[-1]
        self.r.add("a streamed fallback comes back byte for byte", status == 200 and got == echo.get("response", "").encode()
                   and hdrs.get("content-type", "").startswith("text/event-stream"), repr(got[:200]))
        for method, path in (("GET", "/v1/models?x=1"), ("POST", "/tokenize"), ("DELETE", "/v1/responses/r1"),
                             ("GET", "/v1/messages/count_tokens"), ("PUT", "/metrics")):
            resp = self.http.request(method, gw + path, content=b'{"a": 1}' if method != "GET" else None, headers=headers)
            echo = (self.mock_log()["echoes"] or [{}])[-1]
            p, _, q = path.partition("?")
            self.r.add(f"proxy {method} {path}: request and response unchanged",
                       resp.content == echo.get("response", "").encode() and echo.get("path") == p
                       and echo.get("query") == q and echo.get("method") == method
                       and resp.status_code == (202 if method == "DELETE" else 200)
                       and "x-dsv41-gateway" not in resp.headers and resp.headers.get_list("x-multi") == ["a", "b"],
                       f"{resp.status_code} {resp.headers} {echo}")
        resp = self.http.head(gw + "/v1/models")
        self.r.add("proxy HEAD", resp.status_code == 200 and resp.headers.get("x-mock") == "1", repr(resp.headers))

    def grace_window(self, gw: str, grace: float):
        """A stream's headers wait for vLLM's first event up to the grace window, not for the prefill."""
        url = gw + "/v1/chat/completions"
        s = next(x for x in self.streams if x["name"] == "chat_text_stream")
        self.script(chunks=s["chunks"], finish=s["finish"], first_delay=grace + 1.5)  # a long prefill
        t0 = time.monotonic()
        with self.http.stream("POST", url, json=s["body"]) as resp:
            headers_s = time.monotonic() - t0
            raw = b"".join(resp.iter_raw())
        rid = next(iter(self.mock_log()["bodies"]), "?")
        frames = [self.fx.normalize(f + "\n\n", rid) for f in raw.decode().split("\n\n") if f]
        self.r.add(f"stream, first token after the grace window: headers after {headers_s:.2f} s, then the API's SSE",
                   resp.status_code == 200 and grace - 0.3 < headers_s < grace + 1.0 and frames == s["frames"],
                   f"{resp.status_code} {frames[:2]}")
        base = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
        for stream in (True, False):
            self.script(chunks=[[16], [1]], finish="stop", first_delay=grace + 1.5, raise_at=0, error="late rejection")
            status, headers, raw = self.post(url, {**base, "stream": stream}, stream)
            if stream:  # headers went out: the start chunk, then the API-shaped error frame
                frames = [f for f in raw.decode().split("\n\n") if f]
                ok = (status == 200 and len(frames) == 2 and '"role":"assistant"' in frames[0]
                      and json.loads(frames[1][6:]) == {"error": {"message": "late rejection", "type":
                                                         "invalid_request_error", "param": None,
                                                         "code": "invalid_request_error"}})
            else:  # a non-stream response waits for the first event
                ok = status == 400 and json.loads(raw)["error"]["message"] == "late rejection"
            self.r.add(f"vLLM rejection after the grace window ({'stream: 200, start chunk, error frame' if stream else 'non-stream: 400'})",
                       ok and headers.get("x-dsv41-gateway") == "api", f"{status} {raw[:400]!r}")

    def disconnects(self, gw: str):
        body = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
        self.script(chunks=[[16]] * 200 + [[1]], finish="stop", delay=0.05)
        with self.http.stream("POST", gw + "/v1/chat/completions", json={**body, "stream": True}) as resp:
            it = resp.iter_raw()
            got = next(it) + next(it)
        entry = self.wait_aborted()
        self.r.add(f"stream: client disconnect aborts vLLM's request (after {entry.get('sent')}/201 chunks)",
                   entry.get("aborted") and entry.get("sent", 999) < 50 and got.startswith(b"data: "), repr(entry)[:200])
        self.script(chunks=[[16]] * 200 + [[1]], finish="stop", delay=0.05)
        try:
            self.http.post(gw + "/v1/chat/completions", json=body, timeout=1.0)
        except Exception:  # noqa: BLE001 - the client gives up
            pass
        entry = self.wait_aborted()
        self.r.add(f"non-stream: client disconnect aborts vLLM's request (after {entry.get('sent')}/201 chunks)",
                   entry.get("aborted") and entry.get("sent", 999) < 60, repr(entry)[:200])

    def health(self, gw: str):
        ok = self.http.get(gw + "/health")
        self.http.post(self.up + "/mock/health", json={"status": 503})
        down = self.http.get(gw + "/health")
        self.http.post(self.up + "/mock/health", json={"status": 200})
        self.r.add("/health follows vLLM's /health", ok.status_code == 200 and down.status_code == 503
                   and ok.json() == {"gateway": "ok", "upstream": "ok"}, f"{ok.text} {down.text}")

    def url_images(self, gw: str, strict_gw: str):
        img_url = self.up + "/mock/red.png"
        body = {"model": SERVED, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "colour?"}, {"type": "image_url", "image_url": {"url": img_url}}]}]}
        self.script(chunks=[[16], [1]], finish="stop")
        status, _, _ = self.post(gw + "/v1/chat/completions", body, False)
        log = self.mock_log()
        parts = next(iter(log["bodies"].values()), {}).get("content_parts", [])
        red = base64.b64decode(self.fx.RED.split(",", 1)[1])
        self.r.add("URL image: fetched once by the gateway, sent to vLLM as a data URL",
                   status == 200 and len(parts) == 1 and base64.b64decode(parts[0]["url"].split(",", 1)[1]) == red
                   and [e["path"] for e in log["echoes"]] == [], f"{status} {log['echoes']}")
        self.script(chunks=[[16], [1]], finish="stop")
        status, _, raw = self.post(strict_gw + "/v1/chat/completions", body, False)
        log = self.mock_log()
        self.r.add("DSV41_GATEWAY_BLOCK_PRIVATE_URLS=1: a loopback image URL is refused (400), nothing reaches vLLM",
                   status == 400 and not log["bodies"] and not log["echoes"] and b"127.0.0.1" in raw, raw[:300].decode())

    def strict(self, strict_gw: str):
        bad = []
        for c in self.fallbacks:
            if c["api_error"] is None:
                continue
            resp = self.http.post(strict_gw + {"chat": "/v1/chat/completions", "messages": "/v1/messages",
                                               "responses": "/v1/responses"}[c["protocol"]], json=c["body"])
            if (resp.status_code, resp.json()) != (c["api_error"]["status"], c["api_error"]["body"]):
                bad.append(c["name"])
        n = sum(c["api_error"] is not None for c in self.fallbacks)
        self.r.add(f"DSV41_GATEWAY_STRICT=1: the API's error body for requests it rejects ({n - len(bad)}/{n})",
                   not bad, repr(bad))
        resp = self.http.post(strict_gw + "/v1/chat/completions",
                              json={"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "echo": True})
        self.r.add("DSV41_GATEWAY_STRICT=1: vLLM-only fields still go to vLLM",
                   resp.headers.get("x-dsv41-gateway") == "fallback", resp.text[:200])

    def context_limit(self, strict_gw: str, limit: int):
        import deepseek_recipe as d

        long_text = "lorem ipsum " * 3000
        cases = [("chat", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": limit}),
                 ("messages", "/v1/messages", {"messages": [{"role": "user", "content": long_text}], "max_tokens": 64}),
                 ("responses", "/v1/responses", {"input": "hi", "max_output_tokens": limit})]
        api = self.fx.Api(d, self.args.tokenizer)
        for protocol, path, body in cases:
            body = {"model": SERVED, **body}
            _, cr = api.convert(protocol, body)
            ids = api.enc.encode(cr.conversation)
            try:
                cr.resolve_max_tokens(len(ids), limit)
                want = None
            except d.ConversionError as e:
                want = (e.status_code, json.loads(e.body))
            for stream in (False, True):
                self.script(chunks=[[16], [1]], finish="stop")
                resp = self.http.post(strict_gw + path, json={**body, "stream": stream})
                got = (resp.status_code, resp.json() if resp.content[:1] == b"{" else resp.text)
                self.r.add(f"context limit {protocol} ({'stream' if stream else 'non-stream'}): the API's 400 body, "
                           f"before any upstream call", want is not None and got == want
                           and resp.headers.get("x-dsv41-gateway") == "api" and not self.mock_log()["bodies"],
                           f"{got} want {want}")
        got = self.http.post(strict_gw + "/dsv41/render/messages",
                             json={"model": SERVED, "messages": [{"role": "user", "content": long_text}]}).json()
        self.r.add("render of a prompt past the context limit: 200, max_tokens null",
                   got.get("max_tokens") is None and len(got.get("prompt_token_ids") or []) > limit, repr(got)[:300])

    def concurrency(self, gw: str):
        s = next(x for x in self.streams if x["name"] == "chat_text_stream")
        self.script(chunks=s["chunks"], finish=s["finish"], cached=0, delay=0.02)
        out = [None] * 4

        def one(i):
            out[i] = self.post(gw + "/v1/chat/completions", s["body"], True)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log = self.mock_log()
        ok = len(log["bodies"]) == 4
        for status, _, raw in out:
            frames = [f + "\n\n" for f in raw.decode().split("\n\n") if f]
            rid = json.loads(frames[0][6:])["id"]
            ok = ok and status == 200 and [self.fx.normalize(f, rid) for f in frames] == s["frames"]
        self.r.add("four concurrent streams: each = the API's SSE", ok)

    # ---- regressions of the independent review (harness/review)
    def review_leaving_streams(self, gw: str, pid: int):  # 1
        n, statuses = 20, []
        body = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
        self.script(chunks=[[16], [1]], finish="stop", first_delay=600.0)  # a prefill past the grace window

        def leave():
            with self.http.stream("POST", gw + "/v1/chat/completions", json={**body, "stream": True}) as resp:
                statuses.append(resp.status_code)  # the headers came after the grace window
                time.sleep(1.0)

        threads = [threading.Thread(target=leave) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t0 = time.monotonic()
        while True:
            entries = self.mock_log()["engine"]
            aborted = sum(e["aborted"] for e in entries)
            if aborted == n or time.monotonic() - t0 > 10:
                break
            time.sleep(0.2)
        time.sleep(3.0)  # idle upstream connections expire (keepalive_expiry 2 s)
        left = upstream_sockets(pid, int(self.up.rsplit(":", 1)[1]))
        self.script(chunks=[[16], [1]], finish="stop")
        after = self.http.post(gw + "/v1/chat/completions", json=body).status_code
        self.r.add(f"review 1: {n} stream clients leave after the grace window, before the first token: {aborted}/"
                   f"{len(entries)} vLLM requests aborted, {left} upstream connections left, then a request: {after}",
                   statuses == [200] * n and aborted == n == len(entries) and left == 0 and after == 200, repr(statuses))

    def review_proxy_disconnects(self, gw: str):  # 2
        for name, path, body in (("proxied /v1/completions", "/v1/completions", {"model": SERVED, "prompt": "hi"}),
                                 ("fallback (echo: true)", "/v1/chat/completions",
                                  {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "echo": True})):
            self.script(chunks=[[16], [1]], finish="stop")
            try:
                self.http.post(gw + path, json=body, headers={"x-mock-delay": "30"}, timeout=1.0)
            except Exception:  # noqa: BLE001 - the client gives up
                pass
            gone, t0 = None, time.monotonic()
            while gone is None and time.monotonic() - t0 < 5:
                gone = (self.mock_log()["echoes"] or [{}])[-1].get("client_gone_after_s")
                time.sleep(0.1)
            self.r.add(f"review 2: non-stream {name}: the client's disconnect reaches vLLM after {gone} s",
                       gone is not None and gone < 3.0)

    def review_broken_upstream(self, gw: str):  # 4, 5
        url, base = gw + "/v1/chat/completions", {"model": SERVED, "messages": [{"role": "user", "content": "hi"}],
                                                  "thinking": {"type": "disabled"}}
        tok = ('data: {"request_id": "r", "choices": [{"index": 0, "logprobs": null, "finish_reason": null, '
               '"token_ids": [16]}], "usage": null}\n\n')
        usage = 'data: {"request_id": "r", "choices": [], "usage": {"prompt_tokens": 5, "total_tokens": 6}}\n\n'
        done = "data: [DONE]\n\n"
        for name, script, needle in (
                ("a malformed event", {"raw_sse": [tok, "data: {not json\n\n", done]}, "malformed event"),
                ("a dropped connection", {"raw_sse": [tok], "drop": True}, "connection failed"),
                ("no finish reason", {"raw_sse": [tok, usage, done]}, "without a stop or length finish"),
                ("an engine abort (vLLM drops its token-less output)", {"chunks": [[16], [16], []], "finish": "abort"},
                 "without a stop or length finish"),
                ("no event at all", {"raw_sse": [done]}, "without output")):
            for stream in (False, True):
                self.script(**{"chunks": [[16], [1]], "finish": "stop", **script})
                try:
                    status, headers, raw = self.post(url, {**base, "stream": stream}, stream)
                except Exception as e:  # noqa: BLE001 - a cut stream
                    status, headers, raw = None, {}, repr(e).encode()
                text = raw.decode(errors="replace")
                started = stream and name != "no event at all"  # the stream ends with the error frame
                payload = ([f for f in text.split("\n\n") if f] or [""])[-1][6:] if started else text
                err = json.loads(payload) if payload[:1] == "{" else {}
                self.r.add(f"review 4/5: {name} ({'stream' if stream else 'non-stream'}): "
                           f"{'an error frame' if started else '502'} in the API's error shape",
                           status == (200 if started else 502) and headers.get("x-dsv41-gateway") == "api"
                           and err.get("error", {}).get("type") == "internal_error"
                           and needle in err["error"]["message"], f"{status} {text[-300:]}")

    def review_routes_and_media(self, gw: str, block_gw: str):  # 6
        bad = []
        for path in ("/inference/v1/generate", "/v1/chat/completions/render", "/v1/completions/render",
                     "/v1/chat/completions/derender", "/v1/completions/derender", "/v1/messages/render"):
            self.script(chunks=[[16], [1]], finish="stop")
            resp = self.http.post(gw + path, json={"model": SERVED, "token_ids": [1, 2], "sampling_params": {}})
            log = self.mock_log()
            if resp.status_code != 404 or log["echoes"] or log["bodies"] or log["engine"]:
                bad.append(path)
        self.r.add("review 6: vLLM's scale-out routes are not served on the public port (404, nothing reaches vLLM)",
                   not bad, repr(bad))
        user = [{"role": "user", "content": [{"type": "image_url", "image_url": {
            "url": "http://169.254.169.254/latest/meta-data/x.png?sig=s"}}]}]
        for name, path, body in (("a fallback (echo: true)", "/v1/chat/completions", {"messages": user, "echo": True}),
                                 ("an API-rejected request (n: 2)", "/v1/chat/completions", {"messages": user, "n": 2}),
                                 ("proxied /tokenize", "/tokenize", {"messages": user})):
            self.script(chunks=[[16], [1]], finish="stop")
            resp = self.http.post(block_gw + path, json={"model": SERVED, **body})
            self.r.add(f"review 6: BLOCK_PRIVATE_URLS=1, {name} with an http(s) image URL: 400, nothing reaches vLLM",
                       resp.status_code == 400 and resp.json()["error"]["type"] == "invalid_request_error"
                       and not self.mock_log()["echoes"], resp.text[:200])
        self.script(chunks=[[16], [1]], finish="stop")
        resp = self.http.post(block_gw + "/v1/chat/completions", json={"model": SERVED, "echo": True, "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": self.fx.RED}}]}]})
        self.r.add("review 6: BLOCK_PRIVATE_URLS=1 still falls back without remote URLs (a data URL image)",
                   resp.status_code == 200 and resp.headers.get("x-dsv41-gateway") == "fallback", resp.text[:200])

    def review_pool(self, gw: str):  # 8
        import httpx

        n, codes = 110, []
        with httpx.Client(timeout=60, limits=httpx.Limits(max_connections=None)) as many:
            def hold():
                codes.append(many.get(gw + "/v1/models", headers={"x-mock-delay": "4"}).status_code)

            threads = [threading.Thread(target=hold) for _ in range(n)]
            for t in threads:
                t.start()
            time.sleep(1.5)
            t0 = time.monotonic()
            health = self.http.get(gw + "/health", timeout=10).status_code
            took = time.monotonic() - t0
            for t in threads:
                t.join()
        self.r.add(f"review 8: with {n} proxied requests open, /health answers {health} in {took:.2f} s; "
                   f"{codes.count(200)}/{n} complete", health == 200 and took < 1.0 and codes.count(200) == n)

    def review_raw_path(self, gw: str):  # 9
        self.script(chunks=[[16], [1]], finish="stop")
        self.http.get(gw + "/v1/responses/resp%2Fx%3Fy%23z?a=1", headers={"host": "client.example:8000"})
        echo = (self.mock_log()["echoes"] or [{}])[-1]
        self.r.add("review 9: the proxy keeps the raw path, the query and the client's Host",
                   echo.get("raw_path") == "/v1/responses/resp%2Fx%3Fy%23z" and echo.get("query") == "a=1"
                   and dict(echo.get("headers", [])).get("host") == "client.example:8000", repr(echo)[:300])

    def review_image_memory(self, port: int):  # 3, pass 2 A/B
        """A request's images decode in the API's one resolve call; requests take turns, so three at once
        peak like one."""
        import cv2
        import numpy as np

        png = cv2.imencode(".png", np.full((4096, 4096, 3), 180, np.uint8))[1].tobytes()
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        body = {"model": SERVED, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "x"}] + [{"type": "image_url", "image_url": {"url": url}}] * 8}]}
        proc, gwm, _, _ = self.start_gateway(port)  # a fresh gateway: its peak RSS is these requests'
        idle = rss_kib(proc.pid)["VmHWM"] >> 10
        counts = [self.http.post(gwm + "/dsv41/render/chat", json=body).json().get("image_count")]
        single = rss_kib(proc.pid)["VmHWM"] >> 10
        threads = [threading.Thread(target=lambda: counts.append(
            self.http.post(gwm + "/dsv41/render/chat", json=body).json().get("image_count"))) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        triple = rss_kib(proc.pid)["VmHWM"] >> 10
        proc.terminate()
        self.r.numbers["review 3: 8 PNGs of 4096 px per request, gateway peak RSS MiB (idle, 1 request, then "
                       "3 at once)"] = (idle, single, triple)
        self.r.add(f"review 3, pass 2 B: 8 PNGs of 4096 px per request: one request peaks at {single} MiB, three "
                   f"at once at {triple} MiB (requests take turns on the image executor)",
                   counts == [8] * 4 and triple < single + (single - idle) // 2, repr(counts))

    def review_text_beside_images(self, gw: str):  # pass 2 A
        import cv2
        import numpy as np

        png = cv2.imencode(".png", np.full((3024, 4032, 3), 120, np.uint8))[1].tobytes()
        img = {"model": SERVED, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}] * 14}]}
        txt = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}]}
        out = []

        def one(body):
            t0 = time.monotonic()
            status = self.http.post(gw + "/dsv41/render/chat", json=body).status_code
            out.append((body is txt, time.monotonic() - t0, status))

        threads = [threading.Thread(target=one, args=(img,)) for _ in range(30)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        one(txt)
        for t in threads:
            t.join()
        text_s = next(s for is_txt, s, _ in out if is_txt)
        slowest = max(s for is_txt, s, _ in out if not is_txt)
        self.r.add(f"pass 2 A: a text request beside 30 queued 14-photo requests renders in {text_s:.2f} s "
                   f"(image requests take up to {slowest:.1f} s, one at a time)",
                   text_s < 1.0 and all(status == 200 for _, _, status in out), repr(out)[:300])

    def review_logs_and_fields(self, gw: str, block_gw: str, log_path: str):  # pass 2 C, D, E
        user = [{"role": "user", "content": "hi"}]
        for body in ({"model": SERVED, "messages": "my secret prompt text"},
                     {"model": SERVED, "messages": user, "response_format": {
                         "type": "json_schema", "json_schema": {"name": "x", "schema": "secret-schema-text"}}},
                     {"model": SERVED, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
                         {"type": "image_url", "image_url": {
                             "url": "HTTPS://user:hunter2@img.invalid/x.png?X-Amz-Signature=sekrit"}}]}]}):
            self.script(chunks=[[16], [1]], finish="stop")
            self.http.post(gw + "/v1/chat/completions", json=body)
        log = Path(log_path).read_text()
        leaked = [s for s in ("my secret prompt text", "secret-schema-text", "hunter2", "sekrit", "img.invalid/x.png?")
                  if s in log]
        self.r.add("pass 2 C: fallback log lines name error classes and field paths, no request values or URL "
                   "credentials", not leaked and "fallback chat: API 400: invalid request body" in log
                   and "fallback chat: API 400: Invalid json schema" in log
                   and re.search(r"fallback chat: image \w+\n", log), f"leaked {leaked}: {log[-600:]}")
        self.script(chunks=[[16], [1]], finish="stop")
        resp = self.http.post(gw + "/v1/chat/completions", json={
            "model": SERVED, "messages": user, "logit_bias": {}, "documents": [], "allowed_token_ids": [],
            "cache_salt": "", "thinking": {"type": "disabled"}})
        self.r.add("pass 2 D: empty no-op vLLM fields (logit_bias {}, documents [], cache_salt \"\") stay on the API path",
                   resp.status_code == 200 and resp.headers.get("x-dsv41-gateway") == "api", resp.text[:200])
        for name, body in (("metadata.url", {"messages": user, "echo": True, "metadata": {"url": "https://example.com/d"}}),
                           ("a schema's examples URL", {"messages": user, "structured_outputs": {"json": {
                               "type": "object", "examples": [{"url": "https://example.com"}]}}})):
            self.script(chunks=[[16], [1]], finish="stop")
            resp = self.http.post(block_gw + "/v1/chat/completions", json={"model": SERVED, **body})
            self.r.add(f"pass 2 E: BLOCK_PRIVATE_URLS=1 lets a fallback with {name} (no media part) through",
                       resp.status_code == 200 and resp.headers.get("x-dsv41-gateway") == "fallback"
                       and len(self.mock_log()["echoes"]) == 1, resp.text[:200])

    def costs(self, gw: str, pid: int):
        import deepseek_recipe as d

        import dsv41_gateway as G

        maps = Path(f"/proc/{pid}/maps").read_text()
        self.r.add("sitecustomize masked: the gateway process maps no torch or vLLM library, the check process "
                   "imported neither", "libtorch" not in maps and "/vllm/" not in maps and "vllm" not in sys.modules
                   and "torch" not in sys.modules and Path("/usr/lib/python3.12/sitecustomize.py").read_text() == "")
        big = self.fx.agentic_chat_body(40)
        meta = self.prompts["meta"]["agentic_40"]
        gw_core = G.Gateway(self.args.tokenizer, SERVED)
        times, enc, loop = [], [], asyncio.new_event_loop()
        for _ in range(7):
            t0 = time.perf_counter()
            prep = loop.run_until_complete(gw_core.prepare("chat", json.loads(json.dumps(big))))
            times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            gw_core.encoding.encode(prep.converted.conversation)
            enc.append(time.perf_counter() - t0)
        ids = prep.greq["token_ids"]
        self.r.add(f"~74K-token agentic request: {len(ids)} prompt token IDs = the API's (sha256)",
                   hashlib.sha256(json.dumps(ids).encode()).hexdigest() == meta["ids_sha256"])
        self.r.numbers["prepare_74k_ms (convert+render+encode+sampling, median of 7)"] = round(1e3 * statistics.median(times), 1)
        self.r.numbers["encode_74k_ms (render+tokenize, median of 7)"] = round(1e3 * statistics.median(enc), 1)
        t0 = time.perf_counter()
        got = self.http.post(gw + "/dsv41/render/chat", json=big).json()
        self.r.numbers["render_74k_http_ms (body parse + prepare + 74K-id JSON reply)"] = round(1e3 * (time.perf_counter() - t0), 1)
        self.r.add("render endpoint, ~74K-token request: same IDs", got.get("prompt_token_ids") == ids)
        small = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
        e2e = {}
        for name, body in (("small", small), ("74k", big)):
            samples = []
            for _ in range(5):
                self.script(chunks=[[16], [1]], finish="stop")
                t0 = time.perf_counter()
                resp = self.http.post(gw + "/v1/chat/completions", json=body)
                samples.append(time.perf_counter() - t0)
                assert resp.status_code == 200, resp.text
            e2e[name] = statistics.median(samples)
        self.r.numbers["e2e_small_ms (client -> gateway -> vLLM route -> scripted engine, 2 tokens)"] = round(1e3 * e2e["small"], 1)
        self.r.numbers["e2e_74k_ms"] = round(1e3 * e2e["74k"], 1)
        self.r.numbers["gateway_rss_after_74k_kib"] = rss_kib(pid)
        print(f"deepseek-recipe {d.__version__}")


def run_check(args) -> int:
    ck = Checker(args)
    try:
        t0 = time.monotonic()
        while True:  # the upstream builds vLLM's state first
            try:
                if ck.http.get(ck.up + "/health").status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            if time.monotonic() - t0 > 600:
                raise RuntimeError("mock upstream did not come up")
            time.sleep(1)
        proc, gw, startup, log_path = ck.start_gateway(args.port)
        ck.r.numbers["gateway_startup_s (spawn -> /health 200)"] = round(startup, 2)
        ck.r.numbers["gateway_rss_idle_kib"] = rss_kib(proc.pid)
        line = Path(log_path).read_text().splitlines()[0]
        ck.r.add(f"startup line: {line}", line == f"dsv41-gateway: engaged deepseek-recipe 0.2.0 upstream={ck.up} "
                                                    f"model={SERVED}")
        limit = 4096
        _, strict_gw, _, _ = ck.start_gateway(args.port + 10, DSV41_GATEWAY_STRICT="1",
                                              DSV41_GATEWAY_BLOCK_PRIVATE_URLS="1",
                                              DSV41_GATEWAY_MAX_MODEL_LEN=str(limit))
        _, block_gw, _, _ = ck.start_gateway(args.port + 20, DSV41_GATEWAY_BLOCK_PRIVATE_URLS="1")
        ck.review_image_memory(args.port + 30)
        ck.prompts_render(gw)
        ck.stream_fixtures(gw)
        ck.images(gw)
        ck.upstream_errors(gw)
        ck.fallbacks_and_proxy(gw, log_path)
        import dsv41_gateway  # on sys.path since Checker()

        ck.grace_window(gw, dsv41_gateway.FIRST_EVENT_GRACE_S)
        ck.disconnects(gw)
        ck.health(gw)
        ck.url_images(gw, strict_gw)
        ck.strict(strict_gw)
        ck.context_limit(strict_gw, limit)
        ck.concurrency(gw)
        ck.review_leaving_streams(gw, proc.pid)
        ck.review_proxy_disconnects(gw)
        ck.review_broken_upstream(gw)
        ck.review_routes_and_media(gw, block_gw)
        ck.review_pool(gw)
        ck.review_raw_path(gw)
        ck.review_logs_and_fields(gw, block_gw, log_path)
        ck.costs(gw, proc.pid)
        ck.review_text_beside_images(gw)  # after costs: its image decodes raise RSS
    finally:
        for p in ck.procs:
            p.terminate()
            p.wait(timeout=10)
    bad = sum(not ok for _, ok, _ in ck.r.rows)
    for k, v in ck.r.numbers.items():
        print(f"NUMBER {k}: {v}")
    print(f"{len(ck.r.rows) - bad}/{len(ck.r.rows)} passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upstream")
    up.add_argument("--port", type=int, default=18001)
    up.add_argument("--model", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--upstream", default="http://127.0.0.1:18001")
    ck.add_argument("--gateway", required=True, help="path of dsv41_gateway.py")
    ck.add_argument("--tokenizer", required=True)
    ck.add_argument("--port", type=int, default=18000)
    args = ap.parse_args(argv)
    if args.cmd == "upstream":
        run_upstream(args.port, args.model)
        return 0
    return run_check(args)


if __name__ == "__main__":
    sys.exit(main())
