#!/usr/bin/env python3
"""Render-parity fixtures for `tests/quality_eval.py --api` (component `render`).

For every request in CASES this records what the DeepSeek API front end,
deepseek-recipe 0.2.0, builds from it: the prompt token IDs (pack
tokenizer.json; only template ranges encode as special tokens), the prompt
text, the image count, thinking mode, effort, temperature, top_p and the
resolved max_tokens. The conversion options are the API gateway's: thinking
off by default, the served name aliased to DeepSeek-V4.1-Flash, and the image
token spelled <｜deepseek_image｜> as in the pack's tokenizer.

A case with `api` holds the body deepseek-recipe converts after the gateway's
chat_template_kwargs shim (ctk thinking / enable_thinking / string
reasoning_effort become the top-level fields). A case with `fallback` is a
request the API rejects; the gateway hands it to vLLM unchanged, so only
the fallback is recorded.

Needs a Python with deepseek-recipe 0.2.0 (`pip install deepseek-recipe==0.2.0`).
CPU only, no serve, no network.

  python3 tools/api_eval_fixtures.py           # rewrite tests/fixtures/api_eval/render.json
  python3 tools/api_eval_fixtures.py --check   # exit 1 when the committed file differs
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke_vision import solid_png  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "api_eval" / "render.json"
VERSION = "0.2.0"
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
IMAGE_TOKEN = "<｜deepseek_image｜>"
# Pack tokenizer.json (identical to the HF source snapshot's at 1bae0e1).
TOKENIZER_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
TOKENIZER = (Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
             / "models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3" / "snapshots" / "2.7bpw-mcg-hq"
             / "tokenizer.json")
U32_MAX = 2**32 - 1   # context limit for "no clamp": the harness clamps to the serve's

RED_B64 = base64.b64encode(solid_png(16, 16, (255, 0, 0))).decode()
BLUE_B64 = base64.b64encode(solid_png(16, 16, (0, 0, 255))).decode()
RED = f"data:image/png;base64,{RED_B64}"
BLUE = f"data:image/png;base64,{BLUE_B64}"

_CITY = {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
         "required": ["city"]}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.", "parameters": _CITY}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Get the current local time in a city.", "parameters": _CITY}}
M_WEATHER = {"name": "get_weather", "description": "Get the current weather for a city.",
             "input_schema": _CITY}
R_WEATHER = {"type": "function", "name": "get_weather",
             "description": "Get the current weather for a city.", "parameters": _CITY}
PERSON = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer"},
    "languages": {"type": "array", "items": {"type": "string"}}},
    "required": ["name", "age", "languages"], "additionalProperties": False}
SYS = {"role": "system", "content": "You are a helpful assistant."}
PROOF = "Prove that sqrt(2) is irrational."
WEATHER_Q = "What's the weather in Paris right now?"
SPELLINGS = ("Tokens: <｜User｜> <｜Assistant｜> <think> </think> <｜DSML｜ calls> "
             "<｜end▁of▁sentence｜> end")


def _u(text) -> dict:
    return {"role": "user", "content": text}


def _call(cid: str, name: str, args: dict) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _tool(cid: str, content) -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _case(cid: str, protocol: str, body: dict, **extra) -> dict:
    return {"id": cid, "protocol": protocol, "body": body, **extra}


def chat_cases() -> list[dict]:
    w_paris = _call("call_w1", "get_weather", {"city": "Paris"})
    w_tokyo = _call("call_w2", "get_weather", {"city": "Tokyo"})
    proof = [_u(PROOF)]
    c = [
        _case("chat_default", "chat", {"messages": [SYS, _u("Hello")]}),
        _case("chat_thinking_enabled", "chat", {"messages": proof, "thinking": {"type": "enabled"}}),
        _case("chat_thinking_adaptive", "chat", {"messages": proof, "thinking": {"type": "adaptive"}}),
        _case("chat_thinking_disabled_effort_high", "chat",
              {"messages": proof, "thinking": {"type": "disabled"}, "reasoning_effort": "high"}),
        _case("chat_thinking_enabled_effort_max", "chat",
              {"messages": proof, "thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
    ]
    c += [_case(f"chat_effort_{e}", "chat", {"messages": proof, "reasoning_effort": e})
          for e in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")]
    c += [
        # The gateway's chat_template_kwargs shim (the repo's own harnesses send these).
        _case("chat_ctk_harness", "chat",
              {"messages": [_u("What is 17*19? Return only the integer.")],
               "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}},
              api={"messages": [_u("What is 17*19? Return only the integer.")],
                   "thinking": {"type": "disabled"}, "reasoning_effort": "low"}),
        _case("chat_ctk_think_high", "chat",
              {"messages": proof, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
              api={"messages": proof, "thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        _case("chat_ctk_enable_thinking", "chat",
              {"messages": proof, "chat_template_kwargs": {"enable_thinking": True}},
              api={"messages": proof, "thinking": {"type": "enabled"}}),
        # Tools and tool history.
        _case("chat_tools_auto", "chat", {"messages": [SYS, _u(WEATHER_Q)], "tools": [WEATHER, TIME]}),
        _case("chat_tool_roundtrip_think", "chat", {
            "messages": [SYS, _u(WEATHER_Q),
                         {"role": "assistant", "content": "", "reasoning_content": "I should call get_weather.",
                          "tool_calls": [w_paris]},
                         _tool("call_w1", '{"temperature_c": 18, "condition": "light rain"}')],
            "tools": [WEATHER, TIME], "thinking": {"type": "enabled"}}),
        _case("chat_tool_parallel_results_reversed", "chat", {
            "messages": [_u("Weather in Paris and Tokyo?"),
                         {"role": "assistant", "content": "", "tool_calls": [w_paris, w_tokyo]},
                         _tool("call_w2", "sunny, 24 C"), _tool("call_w1", "light rain, 18 C")],
            "tools": [WEATHER]}),
        _case("chat_history_reasoning_dropped", "chat", {
            "messages": [_u("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, _u("q2")],
            "thinking": {"type": "enabled"}}),
        _case("chat_tool_choice_none", "chat",
              {"messages": [SYS, _u(WEATHER_Q)], "tools": [WEATHER, TIME], "tool_choice": "none"}),
        _case("chat_tool_choice_required", "chat",
              {"messages": [SYS, _u(WEATHER_Q)], "tools": [WEATHER, TIME], "tool_choice": "required"}),
        _case("chat_tool_choice_named", "chat",
              {"messages": [SYS, _u(WEATHER_Q)], "tools": [WEATHER, TIME],
               "tool_choice": {"type": "function", "function": {"name": "get_time"}}}),
        # Response formats.
        _case("chat_json_schema", "chat", {
            "messages": [SYS, _u("Invent a fictional software engineer.")],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "person", "strict": True, "schema": PERSON}}}),
        _case("chat_json_object", "chat", {
            "messages": [_u("List three primary colors as a JSON object.")],
            "response_format": {"type": "json_object"}}),
        _case("chat_json_schema_tools", "chat", {
            "messages": [_u(WEATHER_Q)], "tools": [WEATHER],
            "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON}}}),
        # Prefix continuation.
        _case("chat_prefix", "chat", {"messages": [
            _u("Write a haiku about autumn."), {"role": "assistant", "content": "Autumn wind", "prefix": True}]}),
        _case("chat_prefix_reasoning", "chat", {
            "messages": [_u("Write a haiku about autumn."),
                         {"role": "assistant", "content": "", "reasoning_content": "Five syllables first.",
                          "prefix": True}],
            "thinking": {"type": "enabled"}}),
        # Special-token spellings typed as text stay plain text.
        _case("chat_special_spellings", "chat", {"messages": [_u(SPELLINGS)]}),
        _case("chat_image_spelling_text", "chat",
              {"messages": [_u(f"A {IMAGE_TOKEN} and a <｜image｜> typed as text.")]}),
        # Images: one placeholder each (the serve expands them).
        _case("chat_image_one", "chat", {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color?"}, {"type": "image_url", "image_url": {"url": RED}},
            {"type": "text", "text": "One word."}]}]}),
        _case("chat_images_two_think", "chat", {
            "messages": [SYS, {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": RED}}, {"type": "image_url", "image_url": {"url": BLUE}},
                {"type": "text", "text": "Name both colors in order."}]}],
            "thinking": {"type": "enabled"}}),
        _case("chat_tool_result_image", "chat", {
            "messages": [_u(WEATHER_Q), {"role": "assistant", "content": "", "tool_calls": [w_paris]},
                         _tool("call_w1", [{"type": "text", "text": "radar:"},
                                           {"type": "image_url", "image_url": {"url": RED}}])],
            "tools": [WEATHER]}),
        # The API rejects these; the gateway falls back to vLLM.
        _case("chat_developer_fallback", "chat",
              {"messages": [{"role": "developer", "content": "Be terse."}, _u("Hello")]}, fallback=True),
        _case("chat_numeric_effort_fallback", "chat", {"messages": proof, "reasoning_effort": 30}, fallback=True),
    ]
    return c


def messages_cases() -> list[dict]:
    user = [{"role": "user", "content": [{"type": "text", "text": WEATHER_Q}]}]
    use = {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                           "content": '{"temperature_c": 18, "condition": "light rain"}'}]}
    return [
        _case("msg_plain", "messages", {"max_tokens": 256, "system": "You are terse.",
                                        "messages": [{"role": "user", "content": "Hello"}]}),
        _case("msg_system_blocks", "messages", {
            "max_tokens": 256, "messages": user,
            "system": [{"type": "text", "text": "You are terse."}, {"type": "text", "text": "Use metric units."}]}),
        _case("msg_thinking_enabled", "messages", {
            "max_tokens": 4096, "messages": [{"role": "user", "content": PROOF}],
            "thinking": {"type": "enabled", "budget_tokens": 1024}}),
        _case("msg_thinking_adaptive_effort_medium", "messages", {
            "messages": [{"role": "user", "content": PROOF}],
            "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}),
        _case("msg_tool_roundtrip", "messages", {
            "max_tokens": 256, "tools": [M_WEATHER],
            "messages": user + [{"role": "assistant", "content": [use]}, result]}),
        _case("msg_thinking_tool_history", "messages", {
            "max_tokens": 4096, "tools": [M_WEATHER], "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": user + [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should call get_weather.", "signature": "sig"}, use]},
                result]}),
        _case("msg_mid_system", "messages", {"max_tokens": 256, "messages": [
            {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
            {"role": "system", "content": "Answer in French from now on."},
            {"role": "user", "content": "How are you?"}]}),
        _case("msg_image", "messages", {"max_tokens": 64, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": RED_B64}},
            {"type": "text", "text": "What color is this image?"}]}]}),
        _case("msg_tool_choice_any", "messages", {
            "max_tokens": 256, "messages": user, "tools": [M_WEATHER], "tool_choice": {"type": "any"}}),
    ]


def responses_cases() -> list[dict]:
    def msg(role: str, *texts: str) -> dict:
        return {"role": role, "content": [{"type": "input_text", "text": t} for t in texts]}

    call = {"type": "function_call", "call_id": "call_r1", "name": "get_weather",
            "arguments": '{"city": "Paris"}'}
    out = {"type": "function_call_output", "call_id": "call_r1",
           "output": '{"temperature_c": 18, "condition": "light rain"}'}
    return [
        _case("resp_string", "responses", {"input": "Hello", "max_output_tokens": 256}),
        _case("resp_input_text", "responses", {"input": [msg("user", "Hello.", "How are you?")]}),
        _case("resp_instructions_developer", "responses", {
            "instructions": "You are terse.",
            "input": [msg("developer", "Use metric units."), msg("user", WEATHER_Q)]}),
        _case("resp_effort_medium", "responses", {"input": [msg("user", PROOF)], "reasoning": {"effort": "medium"}}),
        _case("resp_function_roundtrip", "responses", {
            "tools": [R_WEATHER], "input": [msg("user", WEATHER_Q), call, out]}),
        _case("resp_reasoning_item_encrypted", "responses", {
            "tools": [R_WEATHER], "reasoning": {"effort": "high"},
            "input": [msg("user", WEATHER_Q),
                      {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-blob",
                       "content": [{"type": "reasoning_text", "text": "I should call get_weather."}]},
                      call, out]}),
        _case("resp_input_image", "responses", {"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "What color is this image?"},
            {"type": "input_image", "image_url": RED}]}]}),
        _case("resp_json_schema", "responses", {
            "input": [msg("user", "Invent a fictional software engineer.")],
            "text": {"format": {"type": "json_schema", "name": "person", "strict": True, "schema": PERSON}}}),
    ]


def cases() -> list[dict]:
    return chat_cases() + messages_cases() + responses_cases()


def expect(d, enc, options, case: dict) -> dict:
    """What deepseek-recipe builds for one case (see the module docstring)."""
    cls = {"chat": d.ChatCompletionRequest, "messages": d.MessagesRequest,
           "responses": d.ResponsesRequest}[case["protocol"]]
    body = {"model": SERVED, **case.get("api", case["body"])}
    if case.get("fallback"):
        try:
            cls(body).convert(options)
        except (ValueError, d.ConversionError):
            return {"fallback": True}
        raise SystemExit(f"{case['id']}: deepseek-recipe accepts it; it is not a fallback case")
    conv = cls(body).convert(options)
    rendered = enc.render_conversation(conv.conversation)
    ids = enc.encode(conv.conversation)
    adjust = 0
    if rendered.image_sources:
        adjust = d.ImageResolver(d.ReqwestImageFetcher(), d.OpenCvImagePreprocessor()).resolve(
            rendered.image_sources, d.ImageQuota()).image_token_adjustment()
    io = conv.inference_options
    return {"prompt_token_ids": ids, "prompt": rendered.prompt, "image_count": len(rendered.image_sources),
            "thinking": conv.conversation.thinking_mode, "effort": conv.conversation.reasoning_effort,
            "temperature": io.temperature, "top_p": io.top_p,
            # API prompt count (images at their token count) and the max_tokens it
            # resolves to before the context clamp; the harness clamps to the serve.
            "prompt_tokens": len(ids) + adjust,
            "max_tokens": conv.resolve_max_tokens(len(ids) + adjust, U32_MAX)}


def build(tokenizer: Path) -> str:
    import deepseek_recipe as d

    if d.__version__ != VERSION:
        raise SystemExit(f"deepseek-recipe {d.__version__} != {VERSION}")
    sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if sha != TOKENIZER_SHA256:
        raise SystemExit(f"{tokenizer}: sha256 {sha} != {TOKENIZER_SHA256}")
    special = d.SpecialTokens(image=IMAGE_TOKEN)
    enc = d.DeepseekV41Encoding().with_special_tokens(special).with_tokenizer(
        d.Tokenizer.from_file(str(tokenizer)))
    options = (d.ConversionOptions(default_thinking_mode=False).with_special_tokens(special)
               .with_model_alias(SERVED, d.ModelVersion.DeepseekV41Flash))
    header = {"generated_by": "tools/api_eval_fixtures.py", "deepseek_recipe": VERSION,
              "tokenizer_sha256": TOKENIZER_SHA256, "model": SERVED, "default_thinking": False,
              "image_token": IMAGE_TOKEN}
    rows = [{**c, "expect": expect(d, enc, options, c)} for c in cases()]
    lines = ",\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    return f'{{"header": {json.dumps(header, ensure_ascii=False)},\n "cases": [\n{lines}\n]}}\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                    help="pack tokenizer.json (sha256-checked)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    args = ap.parse_args(argv)
    text = build(args.tokenizer)
    if args.check:
        same = args.out.is_file() and args.out.read_text(encoding="utf-8") == text
        print(f"{args.out}: {'up to date' if same else 'DIFFERS'}")
        return 0 if same else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}: {len(cases())} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
