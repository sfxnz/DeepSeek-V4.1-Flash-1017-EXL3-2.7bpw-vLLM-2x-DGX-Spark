#!/usr/bin/env python3
"""API-parity components for `tests/quality_eval.py --api` (stdlib, HTTP only).

They check the dsv41 API gateway (deepseek-recipe in front of vLLM) against
the DeepSeek API as deepseek-recipe 0.2.0 encodes it. Every component returns
`invariants`: violations of checks that hold for any model output (render IDs,
response shape, stream structure, gateway path, 400 text). quality_eval gates
them absolutely. Components that score sampled model behaviour also return
`acc` (stream: `text_equal`), gated against a baseline with the Wilson rule.

  render      POST /dsv41/render/{chat,messages,responses} for the corpus in
              tests/fixtures/api_eval/render.json (tools/api_eval_fixtures.py)
              and compare prompt_token_ids, thinking, effort, temperature,
              top_p, max_tokens and image_count. Fallback cases must fall back.
  toolflow    Chat tool loops: call -> result -> answer, parallel calls,
              tool_choice required / named / none, thinking-on tool use (the
              turn-2 prompt must keep the turn-1 reasoning).
  reasoning   thinking on (thinking:{type}, effort alias medium) returns
              reasoning_content and clean content; a stop string that occurs
              in the reasoning does not cut the answer; thinking off has none.
  structured  json_schema output validates, json_object parses, and
              response_format + tools still allows a tool call.
  stream      stream vs non-stream: same finish_reason and fields, usage on
              the final chunk, equal text for short greedy answers.
  messages    /v1/messages: basic, thinking, tool round trip.
  responses   /v1/responses: input_text basic, reasoning.effort medium,
              function call round trip.
  errors      prompt + max_tokens over the context limit returns the API's 400
              body on all three routes. Sent with stream=true: a 200 is closed
              unread, which aborts the generation.

Thinking off uses temperature 0. Thinking on is sampled (the API forces T=1.0),
so those checks only feed `acc` and structural invariants. Every request sets
max_tokens. Concurrency <= 2.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import quality_eval as q

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "api_eval" / "render.json"
GATEWAY_HEADER = "x-dsv41-gateway"           # "api" on the deepseek-recipe path
MARKUP = ("<｜", "｜DSML｜", "<think>", "</think>")
CALL_ID = re.compile(r"^call_\d{2}_\w+$")     # deepseek-recipe tool_call_id.rs
API_MAX_TOKENS = 393216                      # largest max_tokens the API accepts
MAX_THINK = 4096

OFF = {"thinking": {"type": "disabled"}, "temperature": 0}
THINK_LOW = {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
_CITY = {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
         "required": ["city"]}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.", "parameters": _CITY}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Get the current local time in a city.", "parameters": _CITY}}
WEATHER_DATA = {"paris": {"temperature_c": 18, "condition": "light rain"},
                "tokyo": {"temperature_c": 24, "condition": "sunny"}}
WEATHER_PROMPT = "What's the weather in Paris right now? Use the get_weather tool."
PRODUCT = ("What is 23*47? Reply with only the final number.", "1081")
SMOKE = ("What is 17*19? Reply with only the integer.", "323")
PERSON = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer"},
    "languages": {"type": "array", "items": {"type": "string"}}},
    "required": ["name", "age", "languages"], "additionalProperties": False}


# ----------------------------------------------------------------- transport

def parse_sse(text: str) -> list[tuple[str | None, str]]:
    """(event, data) per SSE message. Split on \\n only: JSON may hold U+2028."""
    events, name, data = [], None, []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line:
            if data:
                events.append((name, "\n".join(data)))
            name, data = None, []
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        events.append((name, "\n".join(data)))
    return events


def call(c, path: str, body: dict, *, timeout: float | None = None, drop_ok: bool = False) -> dict:
    """POST {model, **body}. HTTP errors are returned, not raised: a 4xx is a
    finding for the item. drop_ok closes a 2xx response unread."""
    req = urllib.request.Request(
        c.url + path, data=json.dumps({"model": c.model, **body}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout or c.timeout)
    except urllib.error.HTTPError as exc:
        resp = exc
    with resp:
        out = {"status": resp.getcode(), "gateway": resp.headers.get(GATEWAY_HEADER)}
        if drop_ok and out["status"] < 400:
            return out
        raw = resp.read().decode()
    if "text/event-stream" in (resp.headers.get("Content-Type") or ""):
        out["events"] = parse_sse(raw)
        return out
    try:
        out["body"] = json.loads(raw)
    except ValueError:
        out["body"], out["text"] = None, raw[:300]
    return out


def chat(c, messages: list, *, max_tokens: int, **extra) -> dict:
    return call(c, "/v1/chat/completions", {"messages": messages, "max_tokens": max_tokens, **extra})


# --------------------------------------------------------------- pure checks

def clean(text: str | None) -> bool:
    """No template markup leaked into user-visible text."""
    return not any(m in (text or "") for m in MARKUP)


def json_obj(text) -> dict | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def schema_errors(value, schema: dict, path: str = "$") -> list[str]:
    """Violations of the JSON Schema subset used here: type, properties,
    required, additionalProperties false, items, enum."""
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
             "integer": int, "number": (int, float)}
    if kind and (not isinstance(value, types[kind])
                 or (isinstance(value, bool) and kind in ("integer", "number"))):
        return [f"{path}: expected {kind}"]
    errs = []
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: not in enum")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errs += [f"{path}.{k}: missing" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errs += [f"{path}.{k}: not allowed" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errs += schema_errors(value[k], sub, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, x in enumerate(value):
            errs += schema_errors(x, schema["items"], f"{path}[{i}]")
    return errs


def path_violations(reply: dict) -> list[str]:
    """2xx from the gateway's deepseek-recipe path (not its vLLM fallback)."""
    if reply["status"] != 200:
        return [f"HTTP {reply['status']}: {json.dumps(reply.get('body') or reply.get('text'))[:200]}"]
    if reply.get("gateway") != "api":
        return [f"{GATEWAY_HEADER}: {reply.get('gateway')} (want api)"]
    return []


def message_violations(msg: dict, finish: str | None, thinking: bool) -> list[str]:
    """Chat message invariants, shared by full and streamed responses."""
    v = []
    calls = msg.get("tool_calls") or []
    if finish not in ("stop", "length", "tool_calls"):
        v.append(f"finish_reason {finish!r}")
    if finish == "tool_calls" and not calls:
        v.append("finish_reason tool_calls without tool_calls")
    if finish == "stop" and calls:
        v.append("tool_calls with finish_reason stop")
    if finish == "tool_calls":   # a length finish may carry a cut call
        for tc in calls:
            fn = tc.get("function") or {}
            if not CALL_ID.match(str(tc.get("id"))) or tc.get("type") != "function" or not fn.get("name"):
                v.append(f"tool call shape {json.dumps(tc)[:160]}")
            if json_obj(fn.get("arguments")) is None:
                v.append(f"arguments not a JSON object: {str(fn.get('arguments'))[:80]}")
    if len({tc.get("id") for tc in calls}) != len(calls):
        v.append("duplicate tool call ids")
    if not clean(msg.get("content")):
        v.append(f"markup in content: {str(msg.get('content'))[:120]!r}")
    if "reasoning" in msg:
        v.append("vLLM field `reasoning` (API: reasoning_content)")
    if thinking and not isinstance(msg.get("reasoning_content"), str):
        v.append("thinking on but no reasoning_content")
    if not thinking and msg.get("reasoning_content"):
        v.append("reasoning_content with thinking off")
    return v


def usage_violations(usage, thinking: bool, reasoning: str | None) -> list[str]:
    if not isinstance(usage, dict):
        return ["no usage"]
    v = []
    p, n = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not (isinstance(p, int) and p > 0 and isinstance(n, int) and usage.get("total_tokens") == p + n):
        v.append(f"usage counts {usage}")
    hit, miss = usage.get("prompt_cache_hit_tokens"), usage.get("prompt_cache_miss_tokens")
    if not (isinstance(hit, int) and isinstance(miss, int) and hit + miss == p):
        v.append(f"prompt_cache_hit/miss_tokens {hit}/{miss} vs prompt_tokens {p}")
    if thinking:
        r = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if not isinstance(r, int) or (reasoning and r <= 0):
            v.append(f"reasoning_tokens {r!r} for {len(reasoning or '')} reasoning chars")
    return v


def choice_of(reply: dict) -> tuple[dict, str | None, dict | None]:
    """(message, finish_reason, usage) of a full chat response, {} when malformed."""
    try:
        body = reply["body"]
        ch = body["choices"][0]
        return ch["message"], ch.get("finish_reason"), body.get("usage")
    except (KeyError, IndexError, TypeError):
        return {}, None, None


def chat_violations(reply: dict, *, thinking: bool) -> list[str]:
    v = path_violations(reply)
    if reply["status"] != 200:
        return v
    msg, finish, usage = choice_of(reply)
    if not msg:
        return v + [f"not a chat.completion: {json.dumps(reply.get('body'))[:200]}"]
    return (v + message_violations(msg, finish, thinking)
            + usage_violations(usage, thinking, msg.get("reasoning_content")))


def assemble_stream(events: list, *, include_usage: bool) -> tuple[dict, str | None, dict | None, list[str]]:
    """Chat chunks -> (message, finish_reason, usage, structural violations).

    API structure: one id; the first delta carries role; reasoning deltas come
    before content deltas; exactly one finish_reason, on the last chunk, which
    also carries the usage (with include_usage the others have usage null);
    data: [DONE] ends the stream."""
    v = []
    datas = [d for _, d in events]
    if not datas or datas[-1] != "[DONE]":
        v.append("stream does not end with data: [DONE]")
    chunks = []
    for d in datas:
        if d != "[DONE]":
            try:
                chunks.append(json.loads(d))
            except ValueError:
                v.append(f"non-JSON data: {d[:80]}")
    if not chunks:
        return {}, None, None, v + ["no chunks"]
    if len({ch.get("id") for ch in chunks}) != 1 or any(
            ch.get("object") != "chat.completion.chunk" for ch in chunks):
        v.append("chunks differ in id or are not chat.completion.chunk")
    if ((chunks[0].get("choices") or [{}])[0].get("delta") or {}).get("role") != "assistant":
        v.append("first delta has no role assistant")
    msg = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
    finishes, in_content = [], False
    for i, ch in enumerate(chunks):
        if i < len(chunks) - 1 and (ch.get("usage") is not None or (include_usage and "usage" not in ch)):
            v.append(f"chunk {i}: usage before the last chunk must be {'null' if include_usage else 'absent'}")
        for choice in ch.get("choices") or []:
            if choice.get("finish_reason") is not None:
                finishes.append((i, choice["finish_reason"]))
            d = choice.get("delta") or {}
            if d.get("reasoning_content"):
                if in_content:
                    v.append(f"chunk {i}: reasoning delta after content")
                msg["reasoning_content"] += d["reasoning_content"]
            if d.get("content"):
                in_content = True
                msg["content"] += d["content"]
            for tc in d.get("tool_calls") or []:
                calls = msg["tool_calls"]
                while len(calls) <= tc.get("index", 0):
                    calls.append({"id": None, "type": None, "function": {"name": "", "arguments": ""}})
                slot, fn = calls[tc.get("index", 0)], tc.get("function") or {}
                slot["id"] = tc.get("id") or slot["id"]
                slot["type"] = tc.get("type") or slot["type"]
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
    if [i for i, _ in finishes] != [len(chunks) - 1]:
        v.append(f"finish_reason on chunks {[i for i, _ in finishes]} of {len(chunks)} (want only the last)")
    usage = chunks[-1].get("usage")
    if not isinstance(usage, dict):
        v.append("last chunk carries no usage")
    return msg, (finishes[-1][1] if finishes else None), usage, v


def context_limit_error(limit: int, prompt_tokens: int, requested: int) -> dict:
    """The API's 400 body when prompt + max_tokens exceeds the context
    (deepseek-recipe request/mod.rs resolve_max_tokens, error_response.rs)."""
    return {"error": {"message": (
        f"This model's maximum context length is {limit} tokens. However, you requested "
        f"{prompt_tokens + requested} tokens ({prompt_tokens} in the messages, {requested} in the "
        "completion). Please reduce the length of the messages or completion."),
        "type": "invalid_request_error", "param": None, "code": "invalid_request_error"}}


def render_diffs(reply: dict, want: dict, limit: int | None) -> list[str]:
    """Render endpoint reply vs a fixture's `expect`; [] when they match."""
    if reply["status"] != 200 or not isinstance(reply.get("body"), dict):
        return [f"HTTP {reply['status']}: {json.dumps(reply.get('body') or reply.get('text'))[:200]}"]
    got = reply["body"]
    if want.get("fallback"):
        return [] if got.get("fallback_reason") else ["API render; the API rejects this request (want fallback)"]
    if got.get("fallback_reason"):
        return [f"fallback: {got['fallback_reason']}"]
    diffs = []
    ids, ref = got.get("prompt_token_ids") or [], want["prompt_token_ids"]
    d = q.first_divergence(ids, ref)
    if d is not None:
        p, r = got.get("prompt") or "", want["prompt"]
        at = next((i for i, (x, y) in enumerate(zip(p, r)) if x != y), min(len(p), len(r)))
        diffs.append(f"prompt_token_ids differ at {d} ({len(ids)} vs {len(ref)} ids): "
                     f"got {ids[d:d + 6]} want {ref[d:d + 6]}; text at {at}: "
                     f"got {p[max(0, at - 20):at + 40]!r} want {r[max(0, at - 20):at + 40]!r}")
    sampling = got.get("sampling") or {}
    for key, value in (("thinking", got.get("thinking")), ("effort", got.get("effort")),
                       ("image_count", got.get("image_count")),
                       ("temperature", sampling.get("temperature")), ("top_p", sampling.get("top_p"))):
        if value != want[key]:
            diffs.append(f"{key} {value!r} (want {want[key]!r})")
    # The gateway clamps max_tokens to the context left (one token per image of
    # margin for vLLM's alignment pad); the fixture holds the unclamped value.
    cap = want["max_tokens"] if not limit else max(
        1, min(want["max_tokens"], limit - want["prompt_tokens"] - want["image_count"]))
    if got.get("max_tokens") != cap:
        diffs.append(f"max_tokens {got.get('max_tokens')!r} (want {cap})")
    return diffs


def summary(rows: list[dict], key: str = "ok", name: str = "acc") -> dict:
    """Component result: invariant violations (absolute gate) and the rate of
    rows[key] over the rows that scored it (Wilson gate vs the baseline)."""
    scored = [r for r in rows if r.get(key) is not None]
    return {name: q.rate(sum(bool(r[key]) for r in scored), len(scored)),
            "invariants": [f"{r['id']}: {x}" for r in rows for x in r["violations"]], "items": rows}


# --------------------------------------------------------------- components

def run_render(c) -> dict:
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    limit = c.models["data"][0].get("max_model_len")

    def one(case):
        diffs = render_diffs(call(c, f"/dsv41/render/{case['protocol']}", case["body"]), case["expect"], limit)
        return {"id": case["id"], "violations": diffs}

    rows = q._pmap(one, spec["cases"])
    return {"cases": len(rows), "same": sum(not r["violations"] for r in rows),
            "fixture": spec["header"], "context_limit": limit,
            "invariants": [f"{r['id']}: {x}" for r in rows for x in r["violations"]], "items": rows}


def _city(args: dict | None) -> str:
    return str((args or {}).get("city", "")).strip().casefold()


def tool_output(name: str, args: dict | None) -> str:
    city = _city(args)
    if name == "get_time":
        return json.dumps({"city": city, "local_time": "14:05"})
    data = next((v for k, v in WEATHER_DATA.items() if k in city), {"temperature_c": 11, "condition": "cloudy"})
    return json.dumps({"city": city, **data})


def calls_of(msg: dict) -> list[tuple[str, dict | None]]:
    return [((tc.get("function") or {}).get("name"), json_obj((tc.get("function") or {}).get("arguments")))
            for tc in msg.get("tool_calls") or []]


def calls_match(calls: list, want: list[tuple[str, str]]) -> bool:
    """Same tool names with the wanted city in each call's `city`, any order."""
    got = sorted((n or "", _city(a)) for n, a in calls)
    return len(got) == len(want) and all(n == wn and wc in city for (n, city), (wn, wc) in zip(got, sorted(want)))


TOOLFLOW = [
    {"id": "single", "prompt": WEATHER_PROMPT, "tools": [WEATHER, TIME],
     "want": [("get_weather", "paris")], "final": ["18"]},
    {"id": "parallel", "tools": [WEATHER], "want": [("get_weather", "paris"), ("get_weather", "tokyo")],
     "prompt": ("Use the get_weather tool to get the current weather in Paris and in Tokyo. "
                "Call it once for each city, both in this turn."), "final": ["18", "24"]},
    {"id": "required", "prompt": "Tell me something useful about Paris.", "tools": [WEATHER, TIME],
     "tool_choice": "required", "city": "paris"},
    {"id": "named", "prompt": "What's the weather in Paris right now?", "tools": [WEATHER, TIME],
     "tool_choice": {"type": "function", "function": {"name": "get_time"}}, "only": "get_time", "city": "paris"},
    {"id": "none", "prompt": "What is the weather usually like in Paris in autumn? Answer in one sentence.",
     "tools": [WEATHER, TIME], "tool_choice": "none"},
    {"id": "think", "prompt": WEATHER_PROMPT, "tools": [WEATHER, TIME], "think": True,
     "want": [("get_weather", "paris")], "final": ["18"]},
]


def toolflow_item(c, item: dict) -> dict:
    think = item.get("think", False)
    mode = {"tools": item["tools"], **(THINK_LOW if think else OFF)}
    if "tool_choice" in item:
        mode["tool_choice"] = item["tool_choice"]
    budget = MAX_THINK if think else 256
    msgs = [{"role": "user", "content": item["prompt"]}]
    r1 = chat(c, msgs, max_tokens=budget, **mode)
    v = chat_violations(r1, thinking=think)
    msg, finish, _ = choice_of(r1)
    calls = calls_of(msg)
    row = {"id": item["id"], "violations": v, "finish": finish, "calls": calls}
    choice = item.get("tool_choice")
    if r1["status"] == 200:   # what tool_choice guarantees (tools dropped / prefill + grammar)
        if choice == "none" and calls:
            v.append("tool calls with tool_choice none")
        if choice not in (None, "none") and (finish != "tool_calls" or not calls):
            v.append(f"tool_choice {json.dumps(choice)}: finish {finish} with {len(calls)} calls")
        if item.get("only") and any(n != item["only"] for n, _ in calls):
            v.append(f"named tool_choice {item['only']} but called {[n for n, _ in calls]}")
    if "want" in item:
        ok = finish == "tool_calls" and calls_match(calls, item["want"])
    elif "city" in item:
        ok = bool(calls) and all(item["city"] in _city(a) for _, a in calls)
    else:
        ok = not calls and bool((msg.get("content") or "").strip()) and finish == "stop"
    if ok and "final" in item:
        turn = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]}
        if think:
            turn["reasoning_content"] = msg.get("reasoning_content") or ""
        msgs2 = msgs + [turn] + [{"role": "tool", "tool_call_id": tc["id"], "content": tool_output(n, a)}
                                 for tc, (n, a) in zip(msg["tool_calls"], calls)]
        if think and turn["reasoning_content"].strip():
            # Thinking + tools: the current turn's reasoning stays in the prompt.
            rr = call(c, "/dsv41/render/chat", {"messages": msgs2, "max_tokens": budget, **mode})
            if turn["reasoning_content"] not in str((rr.get("body") or {}).get("prompt")):
                v.append(f"turn-1 reasoning missing from the turn-2 prompt (render HTTP {rr['status']})")
        r2 = chat(c, msgs2, max_tokens=budget, **mode)
        v += chat_violations(r2, thinking=think)
        msg2, finish2, _ = choice_of(r2)
        row["final"] = (msg2.get("content") or "")[:200]
        ok = finish2 == "stop" and all(s in row["final"] for s in item["final"])
    row["ok"] = ok
    return row


def run_toolflow(c) -> dict:
    return summary(q._pmap(lambda it: toolflow_item(c, it), TOOLFLOW))


REASONING = [
    {"id": "effort_medium", "q": PRODUCT, "mode": {"reasoning_effort": "medium"}, "think": True},
    {"id": "thinking_enabled", "q": ("What is 38*29? Reply with only the final number.", "1102"),
     "mode": {"thinking": {"type": "enabled"}}, "think": True},
    # API: stop strings match in the answer only. "47" is in the reasoning
    # almost surely, never in a bare "1081".
    {"id": "stop_in_reasoning", "q": PRODUCT, "mode": THINK_LOW, "think": True, "stop": "47"},
    {"id": "thinking_off", "q": PRODUCT, "mode": OFF, "think": False},
]


def reasoning_item(c, item: dict) -> dict:
    (prompt, answer), think = item["q"], item["think"]
    extra = {"stop": [item["stop"]]} if "stop" in item else {}
    r = chat(c, [{"role": "user", "content": prompt}], max_tokens=MAX_THINK if think else 32,
             **item["mode"], **extra)
    v = chat_violations(r, thinking=think)
    msg, finish, usage = choice_of(r)
    reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
    row = {"id": item["id"], "violations": v, "finish": finish, "content": content[:120],
           "reasoning_chars": len(reasoning),
           "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")}
    if "stop" in item and item["stop"] in content:
        v.append(f"stop string {item['stop']!r} left in content")
    row["ok"] = answer in content and finish == "stop" and (bool(reasoning.strip()) or not think)
    if "stop" in item and item["stop"] not in reasoning:
        row["ok"], row["vacuous"] = None, True   # the stop string never hit the reasoning
    return row


def run_reasoning(c) -> dict:
    return summary(q._pmap(lambda it: reasoning_item(c, it), REASONING))


STRUCTURED = [
    {"id": "json_schema", "schema": PERSON,
     "prompt": "Invent a fictional software engineer: name, age and the programming languages they know.",
     "format": {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": PERSON}}},
    {"id": "json_object", "prompt": 'List three primary colors as a JSON object with the key "colors".',
     "format": {"type": "json_object"}},
    {"id": "format_plus_tools", "prompt": WEATHER_PROMPT, "tools": [WEATHER],
     "format": {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON}}},
]


def structured_item(c, item: dict) -> dict:
    extra = {"tools": item["tools"]} if "tools" in item else {}
    r = chat(c, [{"role": "user", "content": item["prompt"]}], max_tokens=256,
             response_format=item["format"], **extra, **OFF)
    v = chat_violations(r, thinking=False)
    msg, finish, _ = choice_of(r)
    content = (msg.get("content") or "").strip()
    row = {"id": item["id"], "violations": v, "finish": finish, "content": content[:200]}
    if "tools" in item:   # API: the tool constraint wins over response_format
        row["ok"] = finish == "tool_calls" and calls_match(calls_of(msg), [("get_weather", "paris")])
        return row
    try:
        value = json.loads(content)
    except ValueError:
        value = None
        if finish == "stop":   # a grammar-constrained answer that finished must parse
            v.append(f"{item['format']['type']} output is not JSON: {content[:80]!r}")
    errs = schema_errors(value, item["schema"]) if value is not None and "schema" in item else []
    if errs and finish == "stop":
        v.append(f"json_schema output does not validate: {errs[:3]}")
    row["ok"] = finish == "stop" and value is not None and not errs and (
        "schema" in item or isinstance(value, dict) and len(value.get("colors") or []) == 3)
    return row


def run_structured(c) -> dict:
    return summary(q._pmap(lambda it: structured_item(c, it), STRUCTURED))


STREAM = [
    {"id": "answer", "prompt": SMOKE[0], "max_tokens": 32},
    {"id": "answer_usage", "prompt": SMOKE[0], "max_tokens": 32, "include_usage": True},
    {"id": "tool", "prompt": WEATHER_PROMPT, "max_tokens": 256, "tools": [WEATHER]},
    {"id": "think", "prompt": PRODUCT[0], "max_tokens": MAX_THINK, "think": True},   # stream only
]


def stream_item(c, item: dict) -> dict:
    think = item.get("think", False)
    body = {"messages": [{"role": "user", "content": item["prompt"]}], "max_tokens": item["max_tokens"],
            **(THINK_LOW if think else OFF), **({"tools": item["tools"]} if "tools" in item else {})}
    s = call(c, "/v1/chat/completions", {
        **body, "stream": True, **({"stream_options": {"include_usage": True}} if item.get("include_usage") else {})})
    v = path_violations(s)
    row = {"id": item["id"], "violations": v}
    if s["status"] != 200:
        return row
    if "events" not in s:
        v.append("not an event stream")
        return row
    smsg, sfinish, susage, sv = assemble_stream(s["events"], include_usage=item.get("include_usage", False))
    v += sv + message_violations(smsg, sfinish, think) + usage_violations(
        susage, think, smsg.get("reasoning_content"))
    if think:   # sampled: structure only
        row["reasoning_chars"], row["content"] = len(smsg["reasoning_content"]), smsg["content"][:120]
        return row
    n = call(c, "/v1/chat/completions", body)
    v += chat_violations(n, thinking=False)
    nmsg, nfinish, nusage = choice_of(n)
    if nfinish != sfinish:
        v.append(f"finish_reason {nfinish} vs stream {sfinish}")
    if (nusage or {}).get("prompt_tokens") != (susage or {}).get("prompt_tokens"):
        v.append("prompt_tokens differ between stream and non-stream")
    for key in ("content", "reasoning_content", "tool_calls"):
        if bool(nmsg.get(key)) != bool(smsg.get(key)):
            v.append(f"{key} in only one of stream / non-stream")
    # Equal text is model behaviour (greedy is not bit-reproducible here).
    row["equal"] = ((nmsg.get("content") or "") == smsg["content"] and calls_of(nmsg) == calls_of(smsg)
                    and (nusage or {}).get("completion_tokens") == (susage or {}).get("completion_tokens"))
    return row


def run_stream(c) -> dict:
    return summary(q._pmap(lambda it: stream_item(c, it), STREAM), key="equal", name="text_equal")


def _texts(blocks: list, kind: str, field: str) -> str:
    return "".join(b.get(field) or "" for b in blocks if b.get("type") == kind)


def messages_violations(reply: dict, *, thinking: bool) -> list[str]:
    v = path_violations(reply)
    if reply["status"] != 200:
        return v
    body = reply.get("body") or {}
    blocks = body.get("content")
    if body.get("type") != "message" or body.get("role") != "assistant" or not isinstance(blocks, list):
        return v + [f"not a message: {json.dumps(body)[:200]}"]
    stop, kinds = body.get("stop_reason"), [b.get("type") for b in blocks]
    uses = [b for b in blocks if b.get("type") == "tool_use"]
    if stop not in ("end_turn", "max_tokens", "stop_sequence", "tool_use"):
        v.append(f"stop_reason {stop!r}")
    if (stop == "tool_use") != bool(uses) and stop != "max_tokens":
        v.append(f"stop_reason {stop} with {len(uses)} tool_use blocks")
    if thinking and (kinds[:1] != ["thinking"] or not isinstance(blocks[0].get("signature"), str)):
        v.append(f"thinking on: blocks {kinds} (want a signed thinking block first)")
    if not thinking and "thinking" in kinds:
        v.append("thinking block with thinking disabled")
    for b in uses:
        if not b.get("id") or not b.get("name") or not isinstance(b.get("input"), dict):
            v.append(f"tool_use shape {json.dumps(b)[:160]}")
    if len({b.get("id") for b in uses}) != len(uses):
        v.append("duplicate tool_use ids")
    if not clean(_texts(blocks, "text", "text")):
        v.append("markup in text blocks")
    u = body.get("usage") or {}
    if not (isinstance(u.get("input_tokens"), int) and u["input_tokens"] > 0 and isinstance(u.get("output_tokens"), int)):
        v.append(f"usage {u}")
    return v


def run_messages(c) -> dict:
    def text_msg(t):
        return [{"role": "user", "content": [{"type": "text", "text": t}]}]

    def basic():
        r = call(c, "/v1/messages", {"messages": text_msg(SMOKE[0]), "max_tokens": 32, **OFF})
        body = r.get("body") or {}
        return {"id": "basic", "violations": messages_violations(r, thinking=False),
                "ok": SMOKE[1] in _texts(body.get("content") or [], "text", "text")
                and body.get("stop_reason") == "end_turn"}

    def thinking():
        r = call(c, "/v1/messages", {
            "system": [{"type": "text", "text": "You are a careful calculator."},
                       {"type": "text", "text": "Answer tersely."}],
            "messages": text_msg(PRODUCT[0]), "max_tokens": MAX_THINK,
            "thinking": {"type": "enabled", "budget_tokens": 2048}})
        body = r.get("body") or {}
        blocks = body.get("content") or []
        return {"id": "thinking", "violations": messages_violations(r, thinking=True),
                "ok": bool(_texts(blocks, "thinking", "thinking").strip())
                and PRODUCT[1] in _texts(blocks, "text", "text") and body.get("stop_reason") == "end_turn"}

    def tool():
        tools = [{"name": "get_weather", "description": "Get the current weather for a city.", "input_schema": _CITY}]
        msgs = text_msg(WEATHER_PROMPT)
        r1 = call(c, "/v1/messages", {"messages": msgs, "tools": tools, "max_tokens": 256, **OFF})
        v = messages_violations(r1, thinking=False)
        blocks = (r1.get("body") or {}).get("content") or []
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        row = {"id": "tool", "violations": v, "calls": [(b.get("name"), b.get("input")) for b in uses]}
        row["ok"] = calls_match(row["calls"], [("get_weather", "paris")])
        if row["ok"]:
            msgs2 = msgs + [{"role": "assistant", "content": blocks}, {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b["id"], "content": tool_output(b["name"], b["input"])}
                for b in uses]}]
            r2 = call(c, "/v1/messages", {"messages": msgs2, "tools": tools, "max_tokens": 256, **OFF})
            v += messages_violations(r2, thinking=False)
            body2 = r2.get("body") or {}
            row["final"] = _texts(body2.get("content") or [], "text", "text")[:200]
            row["ok"] = "18" in row["final"] and body2.get("stop_reason") == "end_turn"
        return row

    return summary(q._pmap(lambda f: f(), [basic, thinking, tool]))


def responses_violations(reply: dict, *, thinking: bool) -> list[str]:
    v = path_violations(reply)
    if reply["status"] != 200:
        return v
    body = reply.get("body") or {}
    items = body.get("output")
    if body.get("object") != "response" or not isinstance(items, list):
        return v + [f"not a response: {json.dumps(body)[:200]}"]
    kinds = [i.get("type") for i in items]
    if body.get("status") not in ("completed", "incomplete"):
        v.append(f"status {body.get('status')!r}")
    # The API omits the reasoning item when the reasoning is empty.
    if "reasoning" in kinds and (not thinking or kinds[0] != "reasoning"):
        v.append(f"output {kinds}: reasoning item {'not first' if thinking else 'with thinking off'}")
    calls = [i for i in items if i.get("type") == "function_call"]
    for fc in calls:
        if not fc.get("call_id") or not fc.get("name") or json_obj(fc.get("arguments")) is None:
            v.append(f"function_call shape {json.dumps(fc)[:160]}")
    if len({fc.get("call_id") for fc in calls}) != len(calls):
        v.append("duplicate call_ids")
    if not clean(output_text(body)):
        v.append("markup in output_text")
    u = body.get("usage") or {}
    if not (isinstance(u.get("input_tokens"), int) and u["input_tokens"] > 0
            and u.get("total_tokens") == u["input_tokens"] + (u.get("output_tokens") or 0)):
        v.append(f"usage {u}")
    if thinking and not isinstance((u.get("output_tokens_details") or {}).get("reasoning_tokens"), int):
        v.append("no output_tokens_details.reasoning_tokens")
    return v


def output_text(body: dict, kind: str = "message") -> str:
    """Text of the output items of one kind (message -> output_text, reasoning -> reasoning_text)."""
    part = "output_text" if kind == "message" else "reasoning_text"
    return "".join(p.get("text") or "" for i in body.get("output") or [] if i.get("type") == kind
                   for p in i.get("content") or [] if p.get("type") == part)


def run_responses(c) -> dict:
    def user(t):
        return [{"role": "user", "content": [{"type": "input_text", "text": t}]}]

    def basic():
        r = call(c, "/v1/responses", {"input": user(SMOKE[0]), "max_output_tokens": 32, "temperature": 0})
        body = r.get("body") or {}
        return {"id": "basic", "violations": responses_violations(r, thinking=False),
                "ok": SMOKE[1] in output_text(body) and body.get("status") == "completed"}

    def thinking():
        r = call(c, "/v1/responses", {"input": user(PRODUCT[0]), "max_output_tokens": MAX_THINK,
                                      "reasoning": {"effort": "medium"}})
        body = r.get("body") or {}
        return {"id": "thinking", "violations": responses_violations(r, thinking=True),
                "ok": bool(output_text(body, "reasoning").strip()) and PRODUCT[1] in output_text(body)
                and body.get("status") == "completed"}

    def tool():
        tools = [{"type": "function", "name": "get_weather",
                  "description": "Get the current weather for a city.", "parameters": _CITY}]
        inp = user(WEATHER_PROMPT)
        r1 = call(c, "/v1/responses", {"input": inp, "tools": tools, "max_output_tokens": 256, "temperature": 0})
        v = responses_violations(r1, thinking=False)
        out = (r1.get("body") or {}).get("output") or []
        fcs = [i for i in out if i.get("type") == "function_call"]
        row = {"id": "tool", "violations": v, "calls": [(i.get("name"), json_obj(i.get("arguments"))) for i in fcs]}
        row["ok"] = calls_match(row["calls"], [("get_weather", "paris")])
        if row["ok"]:
            inp2 = inp + out + [{"type": "function_call_output", "call_id": i["call_id"],
                                 "output": tool_output(i["name"], json_obj(i["arguments"]))} for i in fcs]
            r2 = call(c, "/v1/responses", {"input": inp2, "tools": tools, "max_output_tokens": 256,
                                           "temperature": 0})
            v += responses_violations(r2, thinking=False)
            row["final"] = output_text(r2.get("body") or {})[:200]
            row["ok"] = "18" in row["final"] and (r2.get("body") or {}).get("status") == "completed"
        return row

    return summary(q._pmap(lambda f: f(), [basic, thinking, tool]))


def render_count(c, protocol: str, body: dict) -> int:
    r = call(c, f"/dsv41/render/{protocol}", body)
    ids = (r.get("body") or {}).get("prompt_token_ids")
    if r["status"] != 200 or not isinstance(ids, list):
        raise RuntimeError(f"render/{protocol} HTTP {r['status']}: {json.dumps(r.get('body'))[:200]}")
    return len(ids)


def run_errors(c) -> dict:
    limit = c.models["data"][0].get("max_model_len")
    if not limit:
        return {"invariants": ["/v1/models reports no max_model_len"], "items": []}
    floor = limit - API_MAX_TOKENS + 1   # prompt tokens that put prompt + max over the limit
    n, text = max(8, floor // 60), ""
    for _ in range(4):
        text = "\n\n".join(q.filler_paragraphs(4242, n))
        got = render_count(c, "chat", {"messages": [{"role": "user", "content": text}]})
        if got > floor:
            break
        n = n * (floor + 512) // got + 1
    else:
        raise RuntimeError(f"filler reached only {got} of {floor} prompt tokens")
    routes = (("chat", "/v1/chat/completions", {"messages": [{"role": "user", "content": text}]}, "max_tokens"),
              ("messages", "/v1/messages", {"messages": [{"role": "user", "content": text}]}, "max_tokens"),
              ("responses", "/v1/responses", {"input": text}, "max_output_tokens"))
    rows = []
    for proto, path, body, field in routes:
        prompt = render_count(c, proto, body)
        want = context_limit_error(limit, prompt, API_MAX_TOKENS)
        r = call(c, path, {**body, field: API_MAX_TOKENS, "stream": True}, timeout=120, drop_ok=True)
        diffs = [] if r["status"] == 400 and r.get("body") == want else [
            f"HTTP {r['status']} {json.dumps(r.get('body'))[:300]} (want 400 {json.dumps(want)[:120]}...)"]
        rows.append({"id": proto, "prompt_tokens": prompt, "status": r["status"], "gateway": r["gateway"],
                     "violations": diffs})
    return {"context_limit": limit, "invariants": [f"{r['id']}: {x}" for r in rows for x in r["violations"]],
            "items": rows}
