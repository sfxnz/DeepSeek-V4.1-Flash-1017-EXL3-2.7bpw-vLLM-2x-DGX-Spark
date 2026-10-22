#!/usr/bin/env python3
"""Offline tests for tests/api_eval.py (quality_eval --api): response-shape and
stream checks, render comparison, the 400 text, API gates, the render fixture
and its generator, and main() wiring against a fake gateway that answers with
deepseek-recipe 0.2.0's response shapes. The generator check needs
deepseek-recipe and the pack tokenizer.json, and skips without them."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import api_eval as api  # noqa: E402
import api_eval_fixtures as gen  # noqa: E402
import quality_eval as q  # noqa: E402

API_HDR = {"status": 200, "gateway": "api"}
USAGE = {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12,
         "prompt_tokens_details": {"cached_tokens": 0}, "prompt_cache_hit_tokens": 0,
         "prompt_cache_miss_tokens": 11}
CALL = {"index": 0, "id": "call_00_EwT0WQ8ykv9TXgqfl9ol4575", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}


def _chat(msg: dict, finish: str = "stop", usage: dict | None = None) -> dict:
    """A full Chat Completions reply as deepseek-recipe 0.2.0 shapes it."""
    return {**API_HDR, "body": {"id": "rid-1", "object": "chat.completion", "created": 0, "model": "m",
                                "choices": [{"index": 0, "message": {"role": "assistant", **msg},
                                             "logprobs": None, "finish_reason": finish}],
                                "usage": usage or USAGE}}


THINK_USAGE = {**USAGE, "completion_tokens": 9, "total_tokens": 20,
               "completion_tokens_details": {"reasoning_tokens": 7}}


def _sse(chunks: list, done: bool = True) -> list:
    return [(None, json.dumps(c)) for c in chunks] + ([(None, "[DONE]")] if done else [])


def _chunk(delta: dict, finish=None, **extra) -> dict:
    return {"id": "rid-1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "system_fingerprint": None,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}], **extra}


THINK_STREAM = [  # thinking on, include_usage (probe of deepseek-recipe 0.2.0)
    _chunk({"role": "assistant", "content": None, "reasoning_content": ""}, usage=None),
    _chunk({"content": None, "reasoning_content": "ok 47"}, usage=None),
    _chunk({"content": None, "reasoning_content": " here."}, usage=None),
    _chunk({"content": "1081", "reasoning_content": None}, usage=None),
    _chunk({"content": "", "reasoning_content": None}, "stop", usage={
        **USAGE, "completion_tokens": 8, "total_tokens": 19, "completion_tokens_details": {"reasoning_tokens": 4}}),
]
TOOL_STREAM = [  # thinking off, no include_usage: usage only on the last chunk
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"tool_calls": [{**CALL, "function": {"name": "get_weather", "arguments": ""}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city"'}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ': "Paris"}'}}]}),
    _chunk({"content": ""}, "tool_calls", usage=USAGE),
]


class TransportTests(unittest.TestCase):
    def test_parse_sse(self) -> None:
        text = ('event: message_start\ndata: {"a": 1}\n\n: comment\ndata: {"s": "x y"}\n\n'
                "data: line1\r\ndata:line2\r\n\r\ndata: [DONE]\n\n")
        self.assertEqual(api.parse_sse(text), [
            ("message_start", '{"a": 1}'), (None, '{"s": "x y"}'), (None, "line1\nline2"), (None, "[DONE]")])


class ChatShapeTests(unittest.TestCase):
    def test_api_shapes_pass(self) -> None:
        self.assertEqual(api.chat_violations(_chat({"content": "323"}), thinking=False), [])
        think = _chat({"content": "323", "reasoning_content": "17*19 is 323."}, usage=THINK_USAGE)
        self.assertEqual(api.chat_violations(think, thinking=True), [])
        tool = _chat({"content": "", "tool_calls": [CALL]}, "tool_calls")
        self.assertEqual(api.chat_violations(tool, thinking=False), [])
        # max_tokens inside a call: the API returns the cut call with finish length.
        cut = _chat({"content": "", "tool_calls": [{**CALL, "function": {"name": "get_weather", "arguments": ""}}]},
                    "length")
        self.assertEqual(api.chat_violations(cut, thinking=False), [])

    def _one(self, reply: dict, thinking: bool = False) -> str:
        v = api.chat_violations(reply, thinking=thinking)
        self.assertEqual(len(v), 1, v)
        return v[0]

    def test_violations(self) -> None:
        self.assertIn("reasoning", self._one(_chat({"content": "323", "reasoning": "r"}, usage=THINK_USAGE)))
        self.assertIn("no reasoning_content", self._one(_chat({"content": "323"}, usage=THINK_USAGE), True))
        self.assertIn("thinking off", self._one(_chat({"content": "1", "reasoning_content": "r"})))
        self.assertIn("finish_reason stop", self._one(_chat({"content": "", "tool_calls": [CALL]})))
        self.assertIn("without tool_calls", self._one(_chat({"content": ""}, "tool_calls")))
        bad_id = {**CALL, "id": "chatcmpl-tool-1"}
        self.assertIn("tool call shape", self._one(_chat({"content": "", "tool_calls": [bad_id]}, "tool_calls")))
        bad_args = {**CALL, "function": {"name": "get_weather", "arguments": "[1]"}}
        self.assertIn("not a JSON object", self._one(_chat({"content": "", "tool_calls": [bad_args]}, "tool_calls")))
        twice = [CALL, {**CALL, "index": 1}]
        self.assertIn("duplicate", self._one(_chat({"content": "", "tool_calls": twice}, "tool_calls")))
        self.assertIn("markup", self._one(_chat({"content": "x</think>323"})))
        self.assertIn("reasoning_tokens", self._one(
            _chat({"content": "1", "reasoning_content": "r"}, usage=USAGE), True))
        no_cache = {k: v for k, v in USAGE.items() if not k.startswith("prompt_cache")}
        self.assertIn("prompt_cache", self._one(_chat({"content": "323"}, usage=no_cache)))
        self.assertIn("want api", self._one({**_chat({"content": "323"}), "gateway": "fallback"}))
        self.assertIn("HTTP 400", self._one({"status": 400, "gateway": "api", "body": {"error": {}}}))

    def test_stream_assembly_matches_the_api_structure(self) -> None:
        msg, finish, usage, v = api.assemble_stream(_sse(THINK_STREAM), include_usage=True)
        self.assertEqual(v, [])
        self.assertEqual((msg["reasoning_content"], msg["content"], finish), ("ok 47 here.", "1081", "stop"))
        self.assertEqual(api.message_violations(msg, finish, True), [])
        self.assertEqual(api.usage_violations(usage, True, msg["reasoning_content"]), [])
        msg, finish, usage, v = api.assemble_stream(_sse(TOOL_STREAM), include_usage=False)
        self.assertEqual((v, finish), ([], "tool_calls"))
        self.assertEqual(api.calls_of(msg), [("get_weather", {"city": "Paris"})])
        self.assertEqual(api.message_violations(msg, finish, False), [])

    def test_stream_violations(self) -> None:
        def viol(chunks, include_usage=False, done=True):
            return api.assemble_stream(_sse(chunks, done), include_usage=include_usage)[3]

        self.assertIn("[DONE]", viol(TOOL_STREAM, done=False)[0])
        two = copy.deepcopy(TOOL_STREAM)
        two[3]["choices"][0]["finish_reason"] = "tool_calls"
        self.assertIn("finish_reason on chunks [3, 4]", viol(two)[0])
        early = copy.deepcopy(TOOL_STREAM)
        early[1]["usage"] = USAGE
        self.assertIn("usage before the last chunk", viol(early)[0])
        no_null = copy.deepcopy(THINK_STREAM)
        del no_null[1]["usage"]
        self.assertIn("must be null", viol(no_null, include_usage=True)[0])
        late = copy.deepcopy(THINK_STREAM)
        late.insert(4, _chunk({"content": None, "reasoning_content": "more"}, usage=None))
        self.assertIn("reasoning delta after content", viol(late, include_usage=True)[0])
        last = copy.deepcopy(TOOL_STREAM)
        del last[-1]["usage"]
        self.assertIn("no usage", viol(last)[0])
        norole = copy.deepcopy(TOOL_STREAM)
        del norole[0]["choices"][0]["delta"]["role"]
        self.assertIn("role", viol(norole)[0])


class ProtocolShapeTests(unittest.TestCase):
    MSG_USAGE = {"input_tokens": 37, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                 "output_tokens": 9, "service_tier": "standard"}

    def _msg(self, blocks, stop="end_turn"):
        return {**API_HDR, "body": {"id": "rid-1", "type": "message", "role": "assistant", "model": "m",
                                    "content": blocks, "stop_reason": stop, "stop_sequence": None,
                                    "usage": self.MSG_USAGE}}

    def test_messages(self) -> None:
        think = [{"type": "thinking", "thinking": "17*19 is 323.", "signature": "rid-1"},
                 {"type": "text", "text": "323"}]
        self.assertEqual(api.messages_violations(self._msg(think), thinking=True), [])
        use = [{"type": "tool_use", "id": "call_00_N", "name": "get_weather", "input": {"city": "Paris"}}]
        self.assertEqual(api.messages_violations(self._msg(use, "tool_use"), thinking=False), [])
        self.assertTrue(api.messages_violations(self._msg(think[1:]), thinking=True))
        self.assertTrue(api.messages_violations(self._msg(think), thinking=False))
        self.assertTrue(api.messages_violations(self._msg(use, "end_turn"), thinking=False))
        self.assertTrue(api.messages_violations(self._msg([{"type": "text", "text": "<｜DSML｜ calls>"}]),
                                                thinking=False))

    def _resp(self, output, usage=None):
        return {**API_HDR, "body": {"id": "rid-1", "object": "response", "status": "completed", "output": output,
                                    "usage": usage or {"input_tokens": 37, "input_tokens_details": {"cached_tokens": 0},
                                                       "output_tokens": 9,
                                                       "output_tokens_details": {"reasoning_tokens": 7},
                                                       "total_tokens": 46}}}

    def test_responses(self) -> None:
        reasoning = {"type": "reasoning", "id": "rs_rid-1_0", "status": "completed", "summary": [],
                     "content": [{"type": "reasoning_text", "text": "17*19 is 323."}], "encrypted_content": "rid-1-0"}
        message = {"type": "message", "id": "msg_rid-1_1", "status": "completed", "role": "assistant",
                   "phase": "final_answer",
                   "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "323"}]}
        call = {"type": "function_call", "id": "fc_rid-1_0", "status": "completed",
                "call_id": "call_00_R", "name": "get_weather", "arguments": '{"city": "Paris"}'}
        self.assertEqual(api.responses_violations(self._resp([reasoning, message]), thinking=True), [])
        self.assertEqual(api.output_text(self._resp([reasoning, message])["body"], "reasoning"), "17*19 is 323.")
        self.assertEqual(api.responses_violations(self._resp([call]), thinking=False), [])
        # Empty reasoning: the API emits no reasoning item.
        self.assertEqual(api.responses_violations(self._resp([message]), thinking=True), [])
        self.assertTrue(api.responses_violations(self._resp([message, reasoning]), thinking=True))
        self.assertTrue(api.responses_violations(self._resp([reasoning, message]), thinking=False))
        self.assertTrue(api.responses_violations(self._resp([{**call, "arguments": "{"}]), thinking=False))


class PureLogicTests(unittest.TestCase):
    def test_schema_errors(self) -> None:
        ok = {"name": "Ada", "age": 36, "languages": ["C"]}
        self.assertEqual(api.schema_errors(ok, api.PERSON), [])
        self.assertEqual(api.schema_errors({**ok, "age": True}, api.PERSON), ["$.age: expected integer"])
        self.assertEqual(api.schema_errors({"name": "Ada", "languages": [1]}, api.PERSON),
                         ["$.age: missing", "$.languages[0]: expected string"])
        self.assertEqual(api.schema_errors({**ok, "x": 1}, api.PERSON), ["$.x: not allowed"])
        self.assertEqual(api.schema_errors("b", {"type": "string", "enum": ["a"]}), ["$: not in enum"])
        self.assertEqual(api.schema_errors(1.5, {"type": "number"}), [])

    def test_context_limit_error_is_the_api_body(self) -> None:
        # deepseek-recipe 0.2.0: ConversionError.body of resolve_max_tokens(106785, 500000), max_tokens 393216.
        body = ('{"error":{"message":"This model\'s maximum context length is 500000 tokens. However, you '
                'requested 500001 tokens (106785 in the messages, 393216 in the completion). Please reduce the '
                'length of the messages or completion.","type":"invalid_request_error","param":null,'
                '"code":"invalid_request_error"}}')
        self.assertEqual(api.context_limit_error(500000, 106785, 393216), json.loads(body))

    def test_calls_and_tool_output(self) -> None:
        calls = [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Paris, France"})]
        self.assertTrue(api.calls_match(calls, [("get_weather", "paris"), ("get_weather", "tokyo")]))
        self.assertFalse(api.calls_match(calls[:1], [("get_weather", "paris")]))
        self.assertFalse(api.calls_match([("get_time", {"city": "Paris"})], [("get_weather", "paris")]))
        self.assertFalse(api.calls_match([("get_weather", None)], [("get_weather", "paris")]))
        self.assertIn('"temperature_c": 18', api.tool_output("get_weather", {"city": "Paris, France"}))
        self.assertIn("14:05", api.tool_output("get_time", {"city": "Paris"}))

    def test_render_diffs(self) -> None:
        case = next(c for c in _fixture()["cases"] if c["id"] == "chat_image_one")
        want = case["expect"]
        good = {"prompt": want["prompt"], "prompt_token_ids": list(want["prompt_token_ids"]),
                "image_count": 1, "thinking": False, "effort": None, "fallback_reason": None,
                "sampling": {"temperature": 1.0, "top_p": 1.0}, "max_tokens": want["max_tokens"]}
        self.assertEqual(api.render_diffs({**API_HDR, "body": good}, want, 500000), [])
        bad = copy.deepcopy(good)
        bad["prompt_token_ids"][3] += 1
        bad["effort"] = "high"
        diffs = api.render_diffs({**API_HDR, "body": bad}, want, 500000)
        self.assertIn("prompt_token_ids differ at 3", diffs[0])
        self.assertIn("effort 'high'", diffs[1])
        # The context left caps max_tokens; the image adds one token of margin.
        limit = want["prompt_tokens"] + 1 + 100
        self.assertEqual(api.render_diffs({**API_HDR, "body": {**good, "max_tokens": 100}}, want, limit), [])
        self.assertTrue(api.render_diffs({**API_HDR, "body": good}, want, limit))
        self.assertEqual(api.render_diffs({**API_HDR, "body": {"fallback_reason": "x"}}, {"fallback": True}, 1), [])
        self.assertIn("want fallback", api.render_diffs({**API_HDR, "body": good}, {"fallback": True}, 1)[0])
        self.assertIn("fallback: x", api.render_diffs({**API_HDR, "body": {"fallback_reason": "x"}}, want, 1)[0])
        self.assertIn("HTTP 404", api.render_diffs({"status": 404, "body": None, "text": "nf"}, want, 1)[0])


class ApiGateTests(unittest.TestCase):
    def _gate(self, rows, name):
        return next(r for r in rows if r["gate"] == name)

    def test_invariants_are_absolute(self) -> None:
        cur = {"components": {"render": {"invariants": []}, "errors": {"invariants": ["chat: HTTP 200"]},
                              "toolflow": {"invariants": [], "acc": q.rate(1, 6)}}}
        rows = q.gates(cur, None)
        self.assertTrue(self._gate(rows, "render.invariants")["pass"])
        self.assertEqual((self._gate(rows, "errors.invariants")["pass"],
                          self._gate(rows, "errors.invariants")["value"]), (False, 1))
        self.assertFalse(any(r["gate"] == "toolflow.acc" for r in rows))   # rates need a baseline

    def test_rates_use_the_wilson_rule(self) -> None:
        base = {"components": {"toolflow": {"acc": q.rate(6, 6)}, "stream": {"text_equal": q.rate(3, 3)}}}
        for path, k, n, ok in (("toolflow.acc", 5, 6, True), ("toolflow.acc", 4, 6, False),
                               ("stream.text_equal", 2, 3, True)):
            group, name = path.split(".")
            cur = {"components": {group: {name: q.rate(k, n), "invariants": []}}}
            self.assertEqual(self._gate(q.gates(cur, base), path)["pass"], ok, (path, k, n))
        self.assertNotIn("reasoning.acc", [r["gate"] for r in q.gates(
            {"components": {"reasoning": {"acc": q.rate(0, 4)}}}, base)])

    def test_summary_skips_unscored_rows(self) -> None:
        s = api.summary([{"id": "a", "ok": True, "violations": []}, {"id": "b", "ok": None, "violations": ["x"]},
                         {"id": "c", "ok": False, "violations": []}])
        self.assertEqual((s["acc"]["k"], s["acc"]["n"], s["invariants"]), (1, 2, ["b: x"]))


def _fixture() -> dict:
    return json.loads(api.FIXTURE.read_text(encoding="utf-8"))


class FixtureTests(unittest.TestCase):
    def test_fixture_is_the_generator_corpus(self) -> None:
        spec, corpus = _fixture(), gen.cases()
        self.assertEqual(spec["header"]["deepseek_recipe"], gen.VERSION)
        self.assertEqual(spec["header"]["tokenizer_sha256"], gen.TOKENIZER_SHA256)
        self.assertEqual([c["id"] for c in spec["cases"]], [c["id"] for c in corpus])
        self.assertEqual(len({c["id"] for c in corpus}), len(corpus))
        for row, case in zip(spec["cases"], corpus):
            self.assertEqual({k: v for k, v in row.items() if k != "expect"}, case, case["id"])

    def test_expectations_are_well_formed(self) -> None:
        image_id = 129264   # <｜deepseek_image｜> in the pack tokenizer
        keys = {"prompt_token_ids", "prompt", "image_count", "thinking", "effort", "temperature", "top_p",
                "prompt_tokens", "max_tokens"}
        for case in _fixture()["cases"]:
            e = case["expect"]
            self.assertIn(case["protocol"], ("chat", "messages", "responses"))
            if e.get("fallback"):
                self.assertTrue(case.get("fallback"), case["id"])
                continue
            self.assertEqual(set(e), keys, case["id"])
            self.assertTrue(all(isinstance(i, int) and 0 <= i < 130000 for i in e["prompt_token_ids"]))
            self.assertEqual(e["prompt_token_ids"][0], 0, case["id"])   # BOS once
            self.assertEqual(e["prompt_token_ids"].count(image_id), e["image_count"], case["id"])
            self.assertEqual(e["thinking"], e["effort"] is not None, case["id"])

    def test_corpus_covers_the_parity_shapes(self) -> None:
        ids = {c["id"] for c in gen.cases()}
        for want in ("chat_thinking_enabled", "chat_effort_medium", "chat_effort_ultra", "chat_ctk_harness",
                     "chat_tool_roundtrip_think", "chat_tool_choice_none", "chat_tool_choice_required",
                     "chat_tool_choice_named", "chat_json_schema", "chat_prefix", "chat_special_spellings",
                     "chat_image_one", "msg_system_blocks", "msg_thinking_enabled", "resp_input_text",
                     "resp_instructions_developer", "resp_effort_medium", "chat_developer_fallback"):
            self.assertIn(want, ids)

    @unittest.skipUnless(importlib.util.find_spec("deepseek_recipe") and gen.TOKENIZER.is_file(),
                         "needs deepseek-recipe 0.2.0 and the pack tokenizer.json")
    def test_generator_is_deterministic_and_reproduces_the_fixture(self) -> None:
        text = gen.build(gen.TOKENIZER)
        self.assertEqual(text, gen.build(gen.TOKENIZER))
        self.assertEqual(text, api.FIXTURE.read_text(encoding="utf-8"))


# ------------------------------------------------- fake gateway (API shapes)

LIMIT = 400_000   # context the fake reports: the errors filler stays small


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n\n".join(p.get("text") or "" for p in content or [] if isinstance(p, dict))


def _parse(proto: str, body: dict):
    if proto == "responses":
        inp = body["input"]
        items = [{"role": "user", "content": inp}] if isinstance(inp, str) else inp
        think = "reasoning" in body
        results = [i["output"] for i in items if i.get("type") == "function_call_output"]
        fmt = None
    else:
        items = body["messages"]
        kind = (body.get("thinking") or {}).get("type")
        think = kind in ("enabled", "adaptive") or (kind is None and body.get("reasoning_effort") not in (None, "none"))
        if proto == "chat":
            results = [m["content"] for m in items if m.get("role") == "tool"]
        else:
            last = items[-1]["content"]
            results = [b["content"] for b in last if b.get("type") == "tool_result"] if isinstance(last, list) else []
        fmt = body.get("response_format")
    users = [i for i in items if i.get("role") == "user"]
    return think, _text(users[-1]["content"]) if users else "", results, body.get("tools"), \
        body.get("tool_choice"), fmt, (body.get("stop") or [None])[0]


def _brain(think, text, results, tools, choice, fmt, stop):
    """Scripted model: (reasoning, content, calls)."""
    reasoning = "Work it out: 23*47 = 1081 and 38*29 = 1102." if think else None
    calls, low = [], text.lower()
    if results:
        content = "Here you go: " + "; ".join(results)
    elif tools and choice != "none" and (choice not in (None, "auto") or "tool" in low):
        name = choice["function"]["name"] if isinstance(choice, dict) and "function" in choice else "get_weather"
        calls = [(name, {"city": c}) for c in ("Paris", "Tokyo") if c.lower() in low] or [(name, {"city": "Paris"})]
        content = ""
    elif fmt and fmt["type"] == "json_schema":
        content = json.dumps({"name": "Ada", "age": 36, "languages": ["C", "Rust"]})
    elif fmt and fmt["type"] == "json_object":
        content = json.dumps({"colors": ["red", "yellow", "blue"]})
    else:
        content = next((a for k, a in (("17*19", "323"), ("23*47", "1081"), ("38*29", "1102")) if k in text),
                       "Paris is mild and rainy in autumn.")
    if stop and stop in content:
        content = content[:content.index(stop)]
    return reasoning, content, calls


def _usage(reasoning, content, think):
    n = len((reasoning or "").split()) + len(content.split()) + 1
    u = {**USAGE, "completion_tokens": n, "total_tokens": 11 + n}
    if think:
        u["completion_tokens_details"] = {"reasoning_tokens": len(reasoning.split())}
    return u


def _tool_calls(calls):
    return [{"index": i, "id": f"call_{i:02d}_abcdefghijklmnopqrst1234", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]


def _chat_full(reasoning, content, calls, think, vllm):
    msg = {"role": "assistant", "content": content}
    if think:
        msg["reasoning" if vllm else "reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = _tool_calls(calls)
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "fake-model",
            "choices": [{"index": 0, "message": msg, "logprobs": None,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": _usage(reasoning, content, think)}


def _chat_stream(reasoning, content, calls, think, include_usage):
    def chunk(delta, finish=None, usage=False):
        ch = _chunk(delta, finish)
        if usage is not False:
            ch["usage"] = usage
        elif include_usage:
            ch["usage"] = None
        return ch

    out = [chunk({"role": "assistant", "content": None, "reasoning_content": ""} if think
                 else {"role": "assistant", "content": ""})]
    if think:
        out.append(chunk({"content": None, "reasoning_content": reasoning}))
    if content:
        out.append(chunk({"content": content}))
    for tc in _tool_calls(calls):
        out.append(chunk({"tool_calls": [{**tc, "function": {"name": tc["function"]["name"], "arguments": ""}}]}))
        out.append(chunk({"tool_calls": [{"index": tc["index"], "function": {
            "arguments": tc["function"]["arguments"]}}]}))
    out.append(chunk({"content": ""}, "tool_calls" if calls else "stop", _usage(reasoning, content, think)))
    return "".join(f"data: {json.dumps(x)}\n\n" for x in out) + "data: [DONE]\n\n"


def _messages(reasoning, content, calls, think, vllm):
    blocks = [{"type": "thinking", "thinking": reasoning, "signature": "x"}] if think else []
    blocks += [{"type": "text", "text": content}] if content else []
    blocks += [{"type": "tool_use", "id": f"call_{i:02d}_x", "name": n, "input": a} for i, (n, a) in enumerate(calls)]
    return {"id": "x", "type": "message", "role": "assistant", "model": "fake-model", "content": blocks,
            "stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None,
            "usage": ProtocolShapeTests.MSG_USAGE}


def _responses(reasoning, content, calls, think, vllm):
    out = [{"type": "reasoning", "id": "rs", "status": "completed", "summary": [], "encrypted_content": "x",
            "content": [{"type": "reasoning_text", "text": reasoning}]}] if think else []
    out += [{"type": "message", "id": "m", "status": "completed", "role": "assistant", "phase": "final_answer",
             "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": content}]}] if content else []
    out += [{"type": "function_call", "id": f"fc_{i}", "status": "completed", "call_id": f"call_{i:02d}_x",
             "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(calls)]
    return {"id": "x", "object": "response", "status": "completed", "output": out,
            "usage": {"input_tokens": 11, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 5,
                      "output_tokens_details": {"reasoning_tokens": 3 if think else 0}, "total_tokens": 16}}


def _render(body: dict, drift: bool) -> dict:
    case = next((c for c in _fixture()["cases"] if c["body"] == body), None)
    if case is None:   # errors filler, or the toolflow turn-2 reasoning check
        text = _text(body["input"] if "input" in body else body["messages"][-1]["content"])
        prompt = text if len(text) > 5000 else json.dumps(body)
        return {"prompt": prompt, "prompt_token_ids": [7] * len(prompt.split()), "fallback_reason": None}
    e = case["expect"]
    if e.get("fallback"):
        return {"fallback_reason": "API 400: unsupported request", "prompt_token_ids": None}
    ids = list(e["prompt_token_ids"])
    if drift and case["id"] == "chat_special_spellings":
        ids[2] = 128803   # a typed <｜User｜> encoded as the control token
    return {"prompt": e["prompt"], "prompt_token_ids": ids, "image_count": e["image_count"],
            "thinking": e["thinking"], "effort": e["effort"],
            "sampling": {"temperature": e["temperature"], "top_p": e["top_p"]},
            "max_tokens": e["max_tokens"], "fallback_reason": None}


class _Gateway(BaseHTTPRequestHandler):
    vllm = False    # answer like vLLM direct: no gateway header, `reasoning` field
    drift = False   # one render case drifts from deepseek-recipe

    def log_message(self, fmt: str, *args) -> None:  # noqa: ARG002
        return

    def _send(self, status: int, obj, ctype: str = "application/json") -> None:
        out = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        if not self.vllm:
            self.send_header("x-dsv41-gateway", "api")
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, {"data": [{"id": "fake-model", "root": "/snap/x", "max_model_len": LIMIT}]})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body.pop("model") == "fake-model"
        if self.path.startswith("/dsv41/render/"):
            return self._send(200, _render(body, self.drift))
        proto = {"/v1/chat/completions": "chat", "/v1/messages": "messages", "/v1/responses": "responses"}[self.path]
        requested = body.get("max_tokens", body.get("max_output_tokens"))
        if requested == api.API_MAX_TOKENS:
            text = _text(body["input"] if "input" in body else body["messages"][-1]["content"])
            return self._send(400, api.context_limit_error(LIMIT, len(text.split()), requested))
        think, *rest = _parse(proto, body)
        reasoning, content, calls = _brain(think, *rest)
        if proto == "chat" and body.get("stream"):
            include = bool((body.get("stream_options") or {}).get("include_usage"))
            return self._send(200, _chat_stream(reasoning, content, calls, think, include), "text/event-stream")
        build = {"chat": _chat_full, "messages": _messages, "responses": _responses}[proto]
        return self._send(200, build(reasoning, content, calls, think, self.vllm))


class ApiWiringTests(unittest.TestCase):
    def _run(self, **flags) -> tuple[int, dict, str]:
        handler = type("G", (_Gateway,), flags)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with mock.patch.object(q.subprocess, "run", side_effect=FileNotFoundError), \
                    tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "api.json"
                with redirect_stdout(io.StringIO()) as buf:
                    rc = q.main(["--url", f"http://127.0.0.1:{httpd.server_address[1]}", "--api",
                                 "--out", str(out)])
                return rc, json.loads(out.read_text()), buf.getvalue()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_api_mode_passes_against_api_shapes(self) -> None:
        rc, res, log = self._run()
        self.assertEqual(rc, 0, log)
        self.assertEqual(res["mode"], "api")
        self.assertEqual(list(res["components"]), q.API)
        comps = res["components"]
        self.assertEqual((comps["render"]["same"], comps["render"]["cases"]), (52, 52))
        for name in ("toolflow", "reasoning", "structured", "messages", "responses"):
            self.assertEqual(comps[name]["acc"]["k"], comps[name]["acc"]["n"], name)
        self.assertEqual(comps["stream"]["text_equal"]["n"], 3)
        self.assertEqual(comps["reasoning"]["acc"]["n"], 4)   # the stop item is not vacuous here
        self.assertEqual([r["status"] for r in comps["errors"]["items"]], [400, 400, 400])
        self.assertIn('"render_violations": 0', log)
        # Self-gating against its own result passes; a weaker run fails on rates only.
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            base, weak = Path(tmp) / "base.json", Path(tmp) / "weak.json"
            base.write_text(json.dumps(res))
            res["components"]["toolflow"]["acc"] = q.rate(3, 6)
            weak.write_text(json.dumps(res))
            self.assertEqual(q.main(["--result", str(base), "--baseline", str(base)]), 0)
            self.assertEqual(q.main(["--result", str(weak), "--baseline", str(base)]), 1)

    def test_render_drift_fails_render_only(self) -> None:
        rc, res, _ = self._run(drift=True)
        self.assertEqual(rc, 1)
        failed = [g["gate"] for g in res["gates"] if not g["pass"]]
        self.assertEqual(failed, ["render.invariants"])
        self.assertIn("chat_special_spellings: prompt_token_ids differ at 2",
                      res["components"]["render"]["invariants"][0])

    def test_vllm_direct_fails_the_response_components(self) -> None:
        rc, res, _ = self._run(vllm=True)
        self.assertEqual(rc, 1)
        failed = {g["gate"] for g in res["gates"] if not g["pass"]}
        self.assertEqual(failed, {f"{n}.invariants" for n in
                                  ("toolflow", "reasoning", "structured", "stream", "messages", "responses")})
        self.assertTrue(any("vLLM field `reasoning`" in x for x in res["components"]["reasoning"]["invariants"]))


if __name__ == "__main__":
    unittest.main()
