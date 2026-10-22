#!/usr/bin/env python3
"""gateway/dsv41_gateway.py: the DeepSeek API front end (deepseek-recipe 0.2.0 over vLLM's /inference/v1/generate).

Host, stdlib: the golden fixtures (tools/api_gateway_fixtures.py, from deepseek-recipe itself) cover what
the gateway must reproduce, and the module keeps what run.sh and the audit rely on: top-level imports are
stdlib, deepseek_recipe and constraints only (no vLLM, torch or web stack), the env names, the startup marker.

CoreTests (skipped unless deepseek_recipe imports and the pack tokenizer is in the HF cache): every fixture
request -> the API's prompt token IDs, max_tokens and sampling; every fallback fixture falls back, or returns
the API's error under DSV41_GATEWAY_STRICT=1; every stream fixture -> the API's SSE frames and response body
from vLLM-format token deltas (emulated as ServingTokens formats them), a stop sequence ending the read
early; vLLM errors in the API's error shape; the helpers.

    PYTHONPATH=<dir holding deepseek_recipe> python3 -m unittest tests.test_api_gateway -q

ImageTests (opt-in DSV41_IMAGE_TESTS=1, docker, no GPU, the wheel unpacked at DSV41_GATEWAY_WHEEL_DIR):
CoreTests inside the serve image with its sitecustomize masked, then tests/api_gateway_image_check.py
(vLLM's own token route over a scripted engine in one container, the gateway process over HTTP in another).
"""
from __future__ import annotations

import ast
import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = ROOT / "gateway/dsv41_gateway.py"
FIXTURES = ROOT / "tests/fixtures/api_gateway"
HUB = Path.home() / ".cache/huggingface/hub"
PACK = HUB / "models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq"
TOKENIZER = Path(os.environ.get("DSV41_TEST_TOKENIZER", PACK / "tokenizer.json"))
WHEEL = Path(os.environ.get("DSV41_GATEWAY_WHEEL_DIR", Path.home() / ".cache/dsv41-gateway/deepseek_recipe-0.2.0"))
IMAGE = os.environ.get("DSV41_IMAGE", "dsv41-flash-exl3-sm121:e16-rows")
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
sys.path.insert(0, str(ROOT / "tools"))
import api_gateway_fixtures as fx  # noqa: E402  (stdlib at import)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def have_wheel() -> bool:
    return importlib.util.find_spec("deepseek_recipe") is not None


def module_constants() -> dict:
    """The gateway module's literal top-level constants, read with ast (nothing imported)."""
    out = {}
    for node in ast.parse(GATEWAY.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


class FixtureTests(unittest.TestCase):
    def test_requests_cover_the_api_shapes(self) -> None:
        names = {c["name"] for c in load("prompts.json")["cases"]}
        for want in ("chat_thinking_enabled", "chat_thinking_disabled", "chat_effort_minimal", "chat_effort_medium",
                     "chat_effort_xhigh", "chat_ctk_thinking_true", "chat_ctk_effort_only",
                     "chat_ctk_thinking_false_top_level_effort", "chat_max_completion_tokens",
                     "chat_max_tokens_over_max_completion_tokens", "chat_tools_history_thinking",
                     "chat_tool_choice_required", "chat_json_schema", "chat_regex", "chat_prefix",
                     "chat_special_spellings", "chat_tool_result_images", "msg_thinking_adaptive_effort_medium",
                     "msg_tool_result_image", "resp_developer_instructions", "resp_input_text_parts",
                     "resp_special_spellings"):
            self.assertIn(want, names)
        self.assertEqual({c["protocol"] for c in load("fallbacks.json")["cases"]}, {"chat", "messages", "responses"})

    def test_streams_cover_the_output_paths(self) -> None:
        cases = load("streams.json")["cases"]
        sizes = {len(chunk) for c in cases for chunk in c["chunks"]}
        self.assertEqual(max(sizes), 6)
        self.assertTrue({1, 5, 6} <= sizes)
        self.assertTrue(any(c["consumed"] < len(c["chunks"]) for c in cases), "a stop sequence that ends early")
        self.assertTrue(any(c["finish"] == "length" and "tool" in c["name"] for c in cases))
        self.assertTrue(any(c["logprobs"] for c in cases) and any(c["cached"] for c in cases))
        self.assertEqual({(c["protocol"], c["stream"]) for c in cases},
                         {(p, s) for p in ("chat", "messages", "responses") for s in (False, True)})


class SourceTests(unittest.TestCase):
    def test_top_level_imports_leave_out_vllm_torch_and_the_web_stack(self) -> None:
        tree = ast.parse(GATEWAY.read_text())
        mods = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods - set(sys.stdlib_module_names), {"deepseek_recipe", "constraints"})

    def test_env_names_and_startup_marker(self) -> None:
        consts = module_constants()
        envs = {v for k, v in consts.items() if k.startswith("ENV_")}
        self.assertEqual(envs, {"DSV41_GATEWAY_UPSTREAM", "DSV41_GATEWAY_STRICT", "DSV41_GATEWAY_BLOCK_PRIVATE_URLS",
                                "DSV41_GATEWAY_MAX_MODEL_LEN"})
        self.assertEqual(set(re.findall(r"DSV41_[A-Z_]+", GATEWAY.read_text())), envs)
        self.assertEqual(consts["LOG_ENGAGED"], "dsv41-gateway: engaged deepseek-recipe")

    def test_fallback_fields_are_the_spec_list(self) -> None:
        self.assertEqual(set(module_constants()["FALLBACK_FIELDS"]), {
            "continue_final_message", "add_generation_prompt", "chat_template", "documents", "structured_outputs",
            "guided_json", "guided_regex", "guided_choice", "guided_grammar", "guided_decoding_backend",
            "prompt_logprobs", "echo", "vllm_xargs", "mm_processor_kwargs", "media_io_kwargs", "return_token_ids",
            "return_tokens_as_token_ids", "logits_processors", "bad_words", "allowed_token_ids",
            "truncate_prompt_tokens", "stop_token_ids", "skip_special_tokens", "include_stop_str_in_output",
            "spaces_between_special_tokens",
            # review: fields this vLLM honours that the API path would drop
            "previous_response_id", "conversation", "logit_bias", "thinking_token_budget", "include_reasoning",
            "add_special_tokens", "repetition_detection", "logprob_token_ids", "return_token_offsets",
            "return_prompt_text", "return_assistant_tokens_mask", "use_beam_search", "length_penalty", "cache_salt"})


def vllm_logprob(tok: int, entry: list, k: int) -> dict:
    """One token's logprobs as /inference/v1/generate streams them for sampling_params.logprobs = k: the
    sampled token first, then the top k, deduplicated, cut at max(k, 1) (serving.py _create_tokens_logprobs)."""
    lp, ranked = entry
    alts = {tok: lp}
    for t, alt_lp in ranked[:k]:
        alts.setdefault(t, alt_lp)
    return {"token": f"token_id:{tok}", "logprob": lp, "bytes": None,
            "top_logprobs": [{"token": f"token_id:{t}", "logprob": v, "bytes": None}
                             for t, v in list(alts.items())[:max(k, 1)]]}


async def vllm_events(case: dict, greq: dict, read: list):
    """A stream fixture as vLLM's token stream (stream_options.include_usage, prompt-tokens-details)."""
    k, pos = greq["sampling_params"].get("logprobs"), 0
    for i, ids in enumerate(case["chunks"]):
        read.append(i)
        lps = None
        if k is not None:
            lps = {"content": [vllm_logprob(t, case["logprobs"][pos + j], k) for j, t in enumerate(ids)]}
        pos += len(ids)
        last = i == len(case["chunks"]) - 1
        yield {"request_id": greq["request_id"], "choices": [{
            "index": 0, "logprobs": lps, "finish_reason": case["finish"] if last else None, "token_ids": ids,
            "routed_experts": None, "sampling_mask": None}], "usage": None}
    n = sum(map(len, case["chunks"]))
    yield {"request_id": greq["request_id"], "choices": [], "usage": {
        "prompt_tokens": case["prompt_tokens"], "total_tokens": case["prompt_tokens"] + n, "completion_tokens": n,
        "prompt_tokens_details": {"cached_tokens": case["cached"]}}}


async def listed(*events):
    for ev in events:
        yield ev


@unittest.skipUnless(have_wheel() and TOKENIZER.is_file(),
                     f"needs deepseek_recipe on PYTHONPATH and the pack tokenizer ({TOKENIZER})")
class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(GATEWAY.parent))
        import dsv41_gateway as G

        cls.G = G
        cls.gw = G.Gateway(str(TOKENIZER), SERVED)
        cls.strict = G.Gateway(str(TOKENIZER), SERVED, strict=True)

    def prepare(self, case: dict, gw=None):
        return asyncio.run((gw or self.gw).prepare(case["protocol"], json.loads(json.dumps(case["body"]))))

    def test_prompts_match_the_api(self) -> None:
        for c in load("prompts.json")["cases"]:
            with self.subTest(c["name"]):
                p = self.prepare(c)
                sp, conv = p.greq["sampling_params"], p.converted.conversation
                self.assertEqual(p.greq["token_ids"], c["ids"])
                self.assertEqual((sp["max_tokens"], sp["temperature"], sp["top_p"], p.n_images, p.prompt_tokens),
                                 (c["max_tokens"], c["temperature"], c["top_p"], c["n_images"], c["prompt_tokens"]))
                self.assertEqual((conv.thinking_mode, conv.reasoning_effort), (c["thinking"], c["effort"]))
                if not c["n_images"]:  # the pack spells the image placeholder <｜deepseek_image｜>
                    self.assertEqual(p.prompt, c["prompt"])

    def test_generate_request(self) -> None:
        cases = {c["name"]: c for c in load("prompts.json")["cases"]}
        p = self.prepare(cases["chat_stop_max_tokens"])
        self.assertEqual(p.greq["request_id"], p.id)
        self.assertEqual((p.greq["model"], p.greq["stream"], p.greq["stream_options"]),
                         (SERVED, True, {"include_usage": True}))
        self.assertEqual(p.greq["sampling_params"], {"max_tokens": 77, "temperature": 1.0, "top_p": 1.0,
                                                     "detokenize": False})  # stop strings stay in the processor
        self.assertEqual(p.converted.parsing_options.stop_sequences, ["323", "\n\n"])
        self.assertEqual(self.prepare(cases["chat_vllm_noop_fields"]).greq["sampling_params"]["top_k"], 20)
        self.assertEqual(self.prepare(cases["chat_sampling"]).greq["sampling_params"]["seed"], 7)
        self.assertEqual(self.prepare(cases["chat_logprobs"]).greq["sampling_params"]["logprobs"], 4)
        self.assertEqual(self.prepare(cases["msg_sampling_stops"]).greq["sampling_params"]["top_k"], 5)

    def test_images_follow_the_placeholders(self) -> None:
        case = next(c for c in load("prompts.json")["cases"] if c["name"] == "chat_tool_result_images")
        parts = self.prepare(case).greq["content_parts"]
        got = [base64.b64decode(p["url"].split(",", 1)[1]) for p in parts]
        # Tool results arrive b then a; the prompt renders them in call order a (red) then b (blue).
        self.assertEqual(got, [base64.b64decode(u.split(",", 1)[1]) for u in (fx.RED, fx.BLUE)])

    def test_fallbacks_and_strict_errors(self) -> None:
        for c in load("fallbacks.json")["cases"]:
            with self.subTest(c["name"]):
                with self.assertRaises(self.G.Fallback):
                    self.prepare(c)
                if c["api_error"] is None:  # vLLM-only fields: vLLM serves them in strict mode too
                    with self.assertRaises(self.G.Fallback):
                        self.prepare(c, self.strict)
                    continue
                with self.assertRaises(self.G.ApiError) as err:
                    self.prepare(c, self.strict)
                self.assertEqual((err.exception.status, err.exception.body),
                                 (c["api_error"]["status"], c["api_error"]["body"]))

    def test_streams_match_the_api(self) -> None:
        for c in load("streams.json")["cases"]:
            with self.subTest(c["name"]):
                p, read = self.prepare(c), []

                async def run():
                    events = vllm_events(c, p.greq, read)
                    if c["stream"]:
                        return [fx.normalize(f, p.id) async for f in self.gw.sse_frames(p, events)]
                    return fx.normalize(await self.gw.full_body(p, events), p.id)

                got = asyncio.run(run())
                self.assertEqual(got, c["frames"] if c["stream"] else c["response"])
                self.assertEqual(len(read), c["consumed"], "chunks read from vLLM")

    def test_vllm_errors_take_the_api_error_shape(self) -> None:
        G = self.G
        err = G.upstream_error(422, {"detail": [{"msg": "bad top_k"}]})
        self.assertEqual((err.status, err.body["error"]["type"], err.body["error"]["code"]),
                         (400, "invalid_request_error", "invalid_request_error"))
        self.assertIn("bad top_k", err.body["error"]["message"])
        err = G.upstream_error(503, "Service Unavailable")
        self.assertEqual((err.status, err.body), (503, G.error_body("Service Unavailable", "internal_error")))
        vllm_err = {"error": {"message": "grammar rejected", "type": "BadRequestError", "param": None, "code": 400}}
        with self.assertRaises(G.ApiError) as ctx:
            asyncio.run(G.Gateway.first_event(listed(vllm_err)))
        self.assertEqual((ctx.exception.status, ctx.exception.body), (400, G.error_body("grammar rejected")))

        async def drain():
            stream = await G.Gateway.first_event(listed({"choices": [{"token_ids": [16]}]}))
            return [ev async for ev in stream]

        self.assertEqual(asyncio.run(drain()), [{"choices": [{"token_ids": [16]}]}])
        case = next(c for c in load("streams.json")["cases"] if c["name"] == "chat_text_stream")
        p = self.prepare(case)
        tokens = {"choices": [{"index": 0, "logprobs": None, "finish_reason": None, "token_ids": [16]}]}
        failed = {"error": {"message": "engine died", "type": "InternalServerError", "param": None, "code": 500}}

        async def frames():
            return [f async for f in self.gw.sse_frames(p, listed(tokens, failed))]

        got = asyncio.run(frames())
        self.assertEqual(json.loads(got[-1][len("data: "):]), G.error_body("engine died", "internal_error"))
        self.assertTrue(got[0].startswith('data: {"id":"chatcmpl-'))
        with self.assertRaises(G.ApiError):
            asyncio.run(self.gw.full_body(self.prepare({**case, "body": {**case["body"], "stream": False}}),
                                          listed(tokens, failed)))

    def test_context_limit_is_carried_to_the_route(self) -> None:
        gw = self.G.Gateway(str(TOKENIZER), SERVED, max_model_len=64)
        body = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
        p = asyncio.run(gw.prepare("chat", body))
        self.assertEqual(p.context_error.status, 400)
        self.assertIn("maximum context length is 64 tokens", p.context_error.body["error"]["message"])
        self.assertIsNone(gw.render_info(p)["max_tokens"])
        self.assertIsNone(asyncio.run(self.gw.prepare("chat", dict(body))).context_error)

    def test_structured_outputs_come_from_constraints(self) -> None:
        G = self.G
        body = {"model": SERVED, "messages": [{"role": "user", "content": "Reply in JSON"}],
                "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}}
        self.assertEqual(asyncio.run(self.gw.prepare("chat", dict(body))).greq["sampling_params"]["structured_outputs"],
                         {"json": "{}"})  # the API's json_object: any JSON value
        real = G.structured_outputs_for
        try:
            G.structured_outputs_for = lambda converted, *, thinking: G.FALLBACK
            with self.assertRaises(G.Fallback):
                asyncio.run(self.gw.prepare("chat", dict(body)))
        finally:
            G.structured_outputs_for = real

    def test_helpers(self) -> None:
        G = self.G
        vllm = [vllm_logprob(5, [-0.5, [[7, -0.1], [5, -0.5], [9, -0.7], [3, -0.9]]], 3),
                vllm_logprob(4, [-2.0, [[7, -0.1], [8, -0.2], [9, -0.3]]], 3)]
        self.assertEqual(G.api_logprobs([dict(e) for e in vllm], 2), [(-0.5, [(7, -0.1), (5, -0.5)]),
                                                                       (-2.0, [(7, -0.1), (8, -0.2)])])
        self.assertEqual(G.api_logprobs(vllm[:1], 0), [(-0.5, [])])
        self.assertEqual(G.data_url_bytes("data:image/png;base64,aGVs\nbG8"), b"hello")
        self.assertEqual(G.data_url_bytes("data:text/plain,a%20b"), b"a b")
        for address, ok in (("8.8.8.8", True), ("2001:4860:4860::8888", True), ("127.0.0.1", False),
                            ("10.0.0.1", False), ("169.254.169.254", False), ("100.64.0.1", False),
                            ("224.0.0.1", False), ("::ffff:127.0.0.1", False), ("::1", False), ("fe80::1", False)):
            self.assertEqual(G.public_address("h", address), ok, address)
        self.assertIsNone(G.fallback_reason({"echo": False, "add_generation_prompt": True, "stop_token_ids": [],
                                             "skip_special_tokens": True, "documents": None}))
        self.assertEqual(G.fallback_reason({"add_generation_prompt": False}), "vLLM field add_generation_prompt")

    def test_chat_shim_keeps_todays_meaning(self) -> None:
        on, off = {"type": "enabled"}, {"type": "disabled"}
        for ctk, top, want in (
                ({"enable_thinking": False, "thinking": True}, {}, {"thinking": on}),  # vLLM ORs the switches
                ({"thinking": True}, {}, {"thinking": on}),  # effort: the API default (75)
                ({"thinking": True, "reasoning_effort": "low"}, {}, {"thinking": on, "reasoning_effort": "low"}),
                ({"reasoning_effort": "high"}, {}, {"thinking": off, "reasoning_effort": "high"}),  # no switch: off
                ({"thinking": True, "reasoning_effort": "none"}, {}, {"thinking": off, "reasoning_effort": "none"}),
                ({"thinking": False}, {"reasoning_effort": "high"}, {"reasoning_effort": "high"}),  # API meaning
                ({"thinking": True, "reasoning_effort": "max"}, {"thinking": off},
                 {"thinking": off, "reasoning_effort": "max"})):
            body = {**top, "chat_template_kwargs": ctk}
            self.assertIsNone(self.G.shim_chat(body), ctk)
            self.assertEqual(body, want, (ctk, top))
        self.assertIn("chat_template_kwargs", self.G.shim_chat({"chat_template_kwargs": {"thinking": "no"}}))
        for body, want in (({"max_completion_tokens": 9}, 9), ({"max_tokens": 5, "max_completion_tokens": 9}, 5),
                           ({"max_tokens": None, "max_completion_tokens": 9}, 9), ({}, None)):
            self.assertIsNone(self.G.shim_chat(body))
            self.assertEqual(body.get("max_tokens"), want)

    def test_events_after_a_pending_first_event(self) -> None:
        G = self.G

        async def drain(outcome):
            first = asyncio.get_running_loop().create_future()
            setter = first.set_exception if isinstance(outcome, BaseException) else first.set_result
            asyncio.get_running_loop().call_later(0.01, setter, outcome)
            return [ev async for ev in G.events_after(first)]

        self.assertEqual(asyncio.run(drain(listed({"a": 1}, {"b": 2}))), [{"a": 1}, {"b": 2}])
        with self.assertRaises(G.ApiError):
            asyncio.run(drain(G.ApiError(400, G.error_body("late rejection"))))

    def test_broken_upstream_streams_end_with_a_502(self) -> None:  # review 4, 5
        G = self.G
        case = next(c for c in load("streams.json")["cases"] if c["name"] == "chat_text_stream")
        tokens = {"choices": [{"index": 0, "logprobs": None, "finish_reason": None, "token_ids": [16]}]}
        usage = {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12}}
        aborted = {"choices": [{"index": 0, "logprobs": None, "finish_reason": "abort", "token_ids": [16]}]}

        async def lines(*items):
            for line in items:
                yield line

        async def parse(*items):
            return [ev async for ev in G.sse_events(lines(*items))]

        for bad in ("data: {not json", "data: 5"):
            with self.assertRaises(G.ApiError) as ctx:
                asyncio.run(parse("data: {}", bad))
            self.assertEqual((ctx.exception.status, ctx.exception.body["error"]["type"]), (502, "internal_error"))
        with self.assertRaises(G.ApiError) as ctx:  # no event at all
            asyncio.run(G.Gateway.first_event(listed()))
        self.assertEqual(ctx.exception.status, 502)
        for events in ((tokens, usage), (aborted, usage)):  # no stop or length finish: an engine abort
            p = self.prepare(case)
            got = asyncio.run(self._frames(p, listed(*events)))
            self.assertEqual(json.loads(got[-1][len("data: "):])["error"]["type"], "internal_error")
            self.assertIn("without a stop or length finish", got[-1])
            with self.assertRaises(G.ApiError) as ctx:
                asyncio.run(self.gw.full_body(self.prepare({**case, "body": {**case["body"], "stream": False}}),
                                              listed(*events)))
            self.assertEqual(ctx.exception.status, 502)
        # A malformed event shape becomes the API-shaped error frame, not a cut stream.
        p = self.prepare({**case, "body": {**case["body"], "logprobs": True}})
        got = asyncio.run(self._frames(p, listed(tokens)))
        self.assertEqual(json.loads(got[-1][len("data: "):])["error"]["type"], "internal_error")

    async def _frames(self, prep, events):
        return [f async for f in self.gw.sse_frames(prep, events)]

    def test_images_stay_within_the_api_limits(self) -> None:  # review 3
        G, d = self.G, self.G.d
        gw = G.Gateway(str(TOKENIZER), SERVED)

        class Fetcher:
            calls = 0

            def fetch(self, url):
                self.calls += 1
                return b"\x89PNG" + b"\0" * (20 << 20)

        class Resolver:
            groups = []

            def resolve(self, group, quota):
                self.groups.append(len(group))
                return type("MM", (), {"image_token_adjustment": lambda self: 2})()

        gw.fetcher, gw.resolver = Fetcher(), Resolver()
        with self.assertRaises(G.Fallback) as ctx:  # 5 x 20 MiB: fetching stops at the 64 MiB total
            gw.images([d.UrlImageSource(f"https://img.example/{i}.png?sig=secret") for i in range(5)])
        self.assertEqual((str(ctx.exception), gw.fetcher.calls, gw.resolver.groups), ("image bytes", 4, []))
        with self.assertRaises(G.Fallback) as ctx:  # past 600 images nothing is fetched or decoded
            gw.images([d.UrlImageSource("https://img.example/x.png")] * 601)
        self.assertEqual((str(ctx.exception), gw.fetcher.calls), ("image count", 4))
        red = d.DataUrlImageSource(fx.RED)
        self.assertEqual(gw.images([red] * 3)[0], 2)  # one resolve call: the API's own (8-way decode)
        self.assertEqual(gw.resolver.groups, [3])
        with self.assertRaises(G.ApiError) as ctx:
            G.Gateway(str(TOKENIZER), SERVED, strict=True).images([red] * 601)
        self.assertIn("Too many images: max 600 images per request, got 601", ctx.exception.body["error"]["message"])

    def test_logs_and_fallback_reasons_leave_out_secrets(self) -> None:  # review 10, pass 2 C
        G = self.G
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            G.log("fallback chat: https://bucket.s3.example/a/x.png?X-Amz-Signature=abc&X-Amz-Credential=k failed")
            G.log("fetch HTTPS://user:hunter2@img.example/x.png#frag?sig=1 and http://u@h/p?q failed")
        self.assertEqual(err.getvalue(), "dsv41-gateway: fallback chat: https://bucket.s3.example/a/x.png failed\n"
                                         "dsv41-gateway: fetch HTTPS://img.example/x.png and http://h/p failed\n")
        self.assertEqual(G.shim_chat({"chat_template_kwargs": {"drop_thinking": "token-123"}}),
                         "chat_template_kwargs ['drop_thinking']")
        for body, reason in (
                ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,AAAA"}}]}]}, r"^image \w+$"),
                ({"messages": "my secret prompt text"}, r"^API 400: invalid request body$"),
                ({"messages": [{"role": "user", "content": "hi"}], "response_format": {
                    "type": "json_schema", "json_schema": {"name": "x", "schema": "secret-schema-text"}}},
                 r"^API 400: Invalid json schema$"),
                ({"messages": [{"role": "user", "content": "hi"}], "model": "secret-model-name"},
                 r"^API 400: The supported API model names are DeepSeek-V4.1-Flash$")):
            with self.assertRaises(G.Fallback) as ctx:
                asyncio.run(self.gw.prepare("chat", {"model": SERVED, **body}))
            self.assertRegex(str(ctx.exception), reason)
        self.assertEqual(G.error_kind("tools[0]: server tools are not supported"), "tools[0]: server tools are not supported")
        self.assertEqual(G.error_kind("Unsupported custom tool: 'grep'. Only 'apply_patch' is supported."),
                         "Unsupported custom tool")

    def test_empty_values_do_not_fall_back(self) -> None:  # pass 2 D
        self.assertIsNone(self.G.fallback_reason({"logit_bias": {}, "documents": [], "allowed_token_ids": [],
                                                  "cache_salt": "", "bad_words": [], "chat_template": ""}))
        self.assertEqual(self.G.fallback_reason({"logit_bias": {"16": -100}}), "vLLM field logit_bias")

    def test_image_work_never_holds_the_default_executor(self) -> None:  # pass 2 A
        gw, seen, busy = self.G.Gateway(str(TOKENIZER), SERVED), [], []

        def slow_images(sources):  # 0.1 s of image work per request
            busy.append(1)
            seen.append((threading.current_thread().name, len(busy)))
            time.sleep(0.1)
            busy.pop()
            return 0, [{"type": "image_url", "url": "data:application/octet-stream;base64,"}]

        gw.images = slow_images
        img = {"model": SERVED, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": fx.RED}}]}]}
        txt = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}]}

        async def timed(body, delay):
            await asyncio.sleep(delay)
            t0 = time.monotonic()
            await gw.prepare("chat", json.loads(json.dumps(body)))
            return time.monotonic() - t0

        async def main():
            return await asyncio.gather(*(timed(img, 0) for _ in range(40)), timed(txt, 0.1))

        times = asyncio.run(main())  # 40 image requests (4 s of image work; more than the default pool's
        # workers) and a text request
        self.assertLess(times[-1], 0.5, "a text request waited behind image work")
        self.assertGreater(max(times[:-1]), 3.5)
        self.assertEqual({name.rsplit("_", 1)[0] for name, _ in seen}, {"dsv41-images"})
        self.assertEqual(max(n for _, n in seen), 1, "one request's image work at a time")

    def test_media_urls_and_internal_routes(self) -> None:  # review 6, pass 2 E
        G = self.G
        for body in ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://a/x"}}]}]},
                     {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "HTTP://a/x"}]}]},
                     {"input": [{"role": "user", "content": [{"type": "input_file", "file_url": "https://a/f.pdf"}]}]},
                     {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "http://a"}}]}]},
                     {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": [
                         {"type": "image", "source": {"type": "url", "url": "http://a"}}]}]}]}):
            self.assertTrue(G.remote_media(body), body)
        for body in ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": fx.RED}}]}]},
                     {"tools": [{"function": {"parameters": {"properties": {"url": {"default": "https://a"}}}}}]},
                     {"messages": [{"role": "user", "content": "see https://a/x.png"}]},
                     {"messages": [{"role": "user", "content": "hi"}], "metadata": {"url": "https://example.com/doc"}},
                     {"structured_outputs": {"json": {"type": "object", "examples": [{"url": "https://example.com"}]}}},
                     {"response_format": {"type": "json_schema", "json_schema": {"schema": {
                         "type": "object", "properties": {"image_url": {"type": "string", "default": "https://a"}}}}}},
                     {"messages": [{"role": "assistant", "content": [{"type": "tool_use", "name": "fetch",
                                                                      "input": {"url": "https://example.com"}}]}]}):
            self.assertFalse(G.remote_media(body), body)
        for path in ("/inference/v1/generate", "/v1/chat/completions/render", "/v1/completions/derender",
                     "/v1/messages/render/"):
            self.assertTrue(G.internal_route(path), path)
        for path in ("/v1/chat/completions", "/v1/models", "/tokenize", "/dsv41/render/chat"):
            self.assertFalse(G.internal_route(path), path)


@unittest.skipUnless(os.environ.get("DSV41_IMAGE_TESTS") == "1" and shutil.which("docker") and WHEEL.is_dir()
                     and PACK.is_dir(), "set DSV41_IMAGE_TESTS=1 (needs docker, the serve image, "
                                        "DSV41_GATEWAY_WHEEL_DIR and the pack snapshot)")
class ImageTests(unittest.TestCase):
    # The gateway container as run.sh starts it: the image's sitecustomize (it imports vLLM and torch)
    # masked, the unpacked deepseek-recipe wheel on PYTHONPATH, the HF cache read-only.
    GATEWAY_MOUNTS = ["-v", "/dev/null:/usr/lib/python3.12/sitecustomize.py:ro", "-v", f"{WHEEL}:/opt/dsr:ro",
                      "-v", f"{ROOT}/gateway:/opt/dsv41-gateway:ro", "-v", f"{ROOT}:/w:ro", "-v", f"{HUB}:/hub:ro",
                      "-e", "PYTHONPATH=/opt/dsr", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    PACK_IN = f"/hub/{PACK.relative_to(HUB)}"

    def test_core_tests_inside_the_image(self) -> None:
        proc = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--memory", "4g", "--cpus", "4", *self.GATEWAY_MOUNTS,
             "-e", f"DSV41_TEST_TOKENIZER={self.PACK_IN}/tokenizer.json", "-w", "/w/tests", "--entrypoint", "python3",
             IMAGE, "-m", "unittest", "test_api_gateway", "-v"], capture_output=True, text=True, timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        self.assertRegex(proc.stderr, r"test_streams_match_the_api \(test_api_gateway\.CoreTests.*\) \.\.\. ok")

    def test_gateway_over_http_inside_the_image(self) -> None:
        name = f"dsv41-api-gateway-check-{os.getpid()}"
        subprocess.run(
            ["docker", "run", "-d", "--name", name, "--network", "none", "--memory", "8g", "--cpus", "4",
             "-v", f"{ROOT}/docker/patch:/opt/dsv41-patch:ro",
             "-v", f"{ROOT}/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro",
             "-v", f"{ROOT}:/w:ro", "-v", f"{HUB}:/hub:ro", "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_PLUGINS=vllm_exl3",
             "-e", "VLLM_TARGET_DEVICE=cpu", "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python3", IMAGE,
             "/w/tests/api_gateway_image_check.py", "upstream", "--port", "18001", "--model", self.PACK_IN],
            check=True, capture_output=True, timeout=120)
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--network", f"container:{name}", "--memory", "2g", "--cpus", "4",
                 *self.GATEWAY_MOUNTS, "--entrypoint", "python3", IMAGE, "/w/tests/api_gateway_image_check.py",
                 "check", "--gateway", "/opt/dsv41-gateway/dsv41_gateway.py",
                 "--tokenizer", f"{self.PACK_IN}/tokenizer.json"], capture_output=True, text=True, timeout=1200)
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout[-4000:] + proc.stderr[-2000:])
        self.assertNotIn("FAIL", proc.stdout)
        self.assertIn("PASS render: prompt token IDs", proc.stdout)


if __name__ == "__main__":
    unittest.main()
