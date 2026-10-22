#!/usr/bin/env python3
"""Golden fixtures for the API gateway tests (tests/test_api_gateway.py, tests/api_gateway_image_check.py).

The DeepSeek API's own behaviour, recorded from deepseek-recipe 0.2.0 with the tokenizer it bundles
(static/tokenizers/v41/tokenizer.json of deepseek-ai/deepseek-recipe @935817c; it spells id 129264
<｜image｜> where the pack spells it <｜deepseek_image｜> and is otherwise identical), the API's
default spellings and no gateway code:

  prompts.json    request -> prompt token IDs, image count and token adjustment, thinking, effort,
                  temperature/top_p/seed/logprobs, and the max_tokens the gateway sends
                  (resolve_max_tokens over the prompt plus one token per image, context 500000).
                  A request that uses the Chat fields vLLM honours today (chat_template_kwargs,
                  max_completion_tokens) carries `api_body`: the request with the DeepSeek fields of
                  the same meaning, which the API encoded.
  fallbacks.json  requests the gateway hands to vLLM unchanged, with the API's error (status, body)
                  where the API rejects them or their `api_body` (DSV41_GATEWAY_STRICT=1 returns it).
  streams.json    scripted model output (token IDs in vLLM delta chunks of 1-6 tokens, the finish
                  reason, ranked logprob candidates, prompt cache hits) -> the SSE frames and the
                  response body of the API's StreamProcessor and chunk generators, normalized:
                  response id RID, timestamps 0, tool call ids call_ID. `consumed` is the number of
                  upstream chunks read before the API stops (a stop sequence ends it early).

    PYTHONPATH=<dir holding deepseek_recipe 0.2.0> python3 tools/api_gateway_fixtures.py \\
        --api-tokenizer <deepseek-recipe checkout>/static/tokenizers/v41/tokenizer.json

Deterministic: a rerun writes the same files. agentic_chat_body() is also the image check's
long-request body.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/api_gateway"
M = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
CONTEXT = 500_000
EOS = 1
RID = "RID"

WEATHER = {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
           "required": ["city"]}
CHAT_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
                                              "parameters": WEATHER}}
MSG_TOOL = {"name": "get_weather", "description": "Get the weather", "input_schema": WEATHER}
RESP_TOOL = {"type": "function", "name": "get_weather", "description": "Get the weather", "parameters": WEATHER}
Q = "What is 17*19?"
W = "Weather in Paris?"
U = [{"role": "user", "content": Q}]
NT = {"thinking": {"type": "disabled"}}
# Every special-token spelling a client might type: the API encodes them as plain text.
SPELLINGS = ("<｜begin▁of▁sentence｜> <｜User｜> <｜Assistant｜> <think> </think> <｜end▁of▁sentence｜> "
             "<｜DSML｜tool_calls> </｜DSML｜tool_calls> <｜deepseek_image｜> <｜image｜> <｜System｜>")


def png(w: int, h: int, rgb: tuple[int, int, int]) -> str:
    """A solid-colour PNG as a data URL (stdlib only)."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(t: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)

    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


RED = png(32, 24, (255, 0, 0))
BLUE = png(640, 480, (0, 0, 255))
LOREM = ("The quick brown fox jumps over the lazy dog while the compiler reports warning W{0} in module "
         "src/pkg_{0}/file_{0}.rs at line {0}; retry the build with --verbose and check the cache. ")


def agentic_chat_body(rounds: int, *, lines: int = 36, tools: int = 24) -> dict:
    """A coding-agent Chat request: long system prompt, tools, `rounds` tool round trips.

    rounds=40 is the ~74K-token request of the protocols study (proto_bench.chat_body)."""
    def tool(i):
        return {"type": "function", "function": {
            "name": f"tool_{i:02d}", "description": f"Tool number {i} reads and writes project files. " * 6,
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path"},
                "mode": {"type": "string", "enum": ["read", "write", "append"]},
                "limit": {"type": "integer"}}, "required": ["path"]}}}

    msgs = [{"role": "system", "content": "You are a coding agent. " + "Follow the repository conventions exactly. " * 300},
            {"role": "user", "content": "Fix the build."}]
    for i in range(rounds):
        call = {"id": f"call_{i}", "type": "function", "function": {
            "name": f"tool_{i % tools:02d}", "arguments": json.dumps({"path": f"src/f{i}.rs", "mode": "read"})}}
        msgs += [{"role": "assistant", "content": f"Step {i}.", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": f"call_{i}",
                  "content": "".join(LOREM.format(i * 100 + j) for j in range(lines))}]
    msgs.append({"role": "user", "content": "Summarize what you found."})
    return {"model": M, "messages": msgs, "tools": [tool(i) for i in range(tools)], **NT}


def calls(*names: str) -> list:
    return [{"id": f"call_{n}", "type": "function", "function": {"name": "get_weather",
                                                                  "arguments": json.dumps({"city": n})}}
            for n in names]


# ---- request corpus: (name, protocol, body[, api_body]) ----
PROMPTS = [
    ("chat_plain", "chat", {"messages": U}),
    ("chat_thinking_enabled", "chat", {"messages": U, "thinking": {"type": "enabled"}}),
    ("chat_thinking_disabled", "chat", {"messages": U, **NT}),
    *[(f"chat_effort_{e}", "chat", {"messages": U, "reasoning_effort": e})
      for e in ("none", "minimal", "low", "medium", "high", "xhigh", "max")],
    ("chat_thinking_disabled_effort_high", "chat", {"messages": U, "reasoning_effort": "high", **NT}),
    ("chat_ctk_thinking_true", "chat", {"messages": U, "chat_template_kwargs": {"thinking": True}},
     {"messages": U, "thinking": {"type": "enabled"}}),
    ("chat_ctk_thinking_false_effort_low", "chat",
     {"messages": U, "chat_template_kwargs": {"thinking": False, "reasoning_effort": "low"}},
     {"messages": U, **NT, "reasoning_effort": "low"}),
    ("chat_ctk_enable_thinking_effort_max", "chat",
     {"messages": U, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "max"}},
     {"messages": U, "thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
    ("chat_ctk_effort_with_top_level_thinking", "chat",
     {"messages": U, **NT, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
     {"messages": U, **NT, "reasoning_effort": "high"}),
    # vLLM today: a kwargs effort without a true switch leaves thinking off (serve default kwargs).
    ("chat_ctk_effort_only", "chat", {"messages": U, "chat_template_kwargs": {"reasoning_effort": "high"}},
     {"messages": U, **NT, "reasoning_effort": "high"}),
    ("chat_ctk_thinking_true_effort_none", "chat",
     {"messages": U, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "none"}},
     {"messages": U, **NT, "reasoning_effort": "none"}),
    # A top-level reasoning_effort keeps the API's meaning (vLLM ORs its switch with the kwargs' too).
    ("chat_ctk_thinking_false_top_level_effort", "chat",
     {"messages": U, "reasoning_effort": "high", "chat_template_kwargs": {"thinking": False}},
     {"messages": U, "reasoning_effort": "high"}),
    ("chat_max_completion_tokens", "chat", {"messages": U, "max_completion_tokens": 33, **NT},
     {"messages": U, "max_tokens": 33, **NT}),
    ("chat_max_tokens_over_max_completion_tokens", "chat",
     {"messages": U, "max_tokens": 20, "max_completion_tokens": 33, **NT}, {"messages": U, "max_tokens": 20, **NT}),
    ("chat_vllm_noop_fields", "chat", {"messages": U, **NT, "add_generation_prompt": True, "echo": False,
                                       "skip_special_tokens": True, "stop_token_ids": [], "top_k": 20}),
    ("chat_multi_turn_system", "chat", {"messages": [
        {"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello."}, {"role": "user", "content": Q}], **NT}),
    ("chat_tools_auto", "chat", {"messages": [{"role": "user", "content": W}], "tools": [CHAT_TOOL], **NT}),
    ("chat_tools_history", "chat", {"messages": [
        {"role": "user", "content": W}, {"role": "assistant", "content": "", "tool_calls": calls("Paris")},
        {"role": "tool", "tool_call_id": "call_Paris", "content": "Sunny, 21C"}], "tools": [CHAT_TOOL], **NT}),
    ("chat_tools_history_thinking", "chat", {"messages": [
        {"role": "user", "content": W},
        {"role": "assistant", "content": None, "reasoning_content": "Need the weather tool.",
         "tool_calls": calls("Paris")},
        {"role": "tool", "tool_call_id": "call_Paris", "content": "Sunny"}],
        "tools": [CHAT_TOOL], "reasoning_effort": "high"}),
    ("chat_tool_choice_none", "chat", {"messages": U, "tools": [CHAT_TOOL], "tool_choice": "none", **NT}),
    ("chat_tool_choice_required", "chat", {"messages": U, "tools": [CHAT_TOOL], "tool_choice": "required", **NT}),
    ("chat_tool_choice_named", "chat", {"messages": U, "tools": [
        CHAT_TOOL, {"type": "function", "function": {**CHAT_TOOL["function"], "name": "other"}}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, **NT}),
    ("chat_tools_strict", "chat", {"messages": U, "tools": [
        {"type": "function", "function": {**CHAT_TOOL["function"], "strict": True}}], **NT}),
    ("chat_json_object", "chat", {"messages": [{"role": "user", "content": "Reply in JSON: 17*19"}],
                                  "response_format": {"type": "json_object"}, **NT}),
    ("chat_json_schema", "chat", {"messages": U, "response_format": {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True, "schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                                                     "required": ["value"]}}}, **NT}),
    ("chat_json_schema_thinking", "chat", {"messages": U, "response_format": {"type": "json_schema", "json_schema": {
        "name": "answer", "schema": {"type": "object", "properties": {"value": {"type": "integer"}}}}},
        "reasoning_effort": "high"}),
    ("chat_regex", "chat", {"messages": U, "response_format": {"type": "regex", "regex": "[0-9]+"}, **NT}),
    ("chat_prefix", "chat", {"messages": U + [{"role": "assistant", "content": "17*19 = ", "prefix": True}], **NT}),
    ("chat_prefix_reasoning", "chat", {"messages": U + [
        {"role": "assistant", "content": "", "reasoning_content": "17*19 is", "prefix": True}],
        "reasoning_effort": "high"}),
    ("chat_special_spellings", "chat", {"messages": [
        {"role": "system", "content": "Literal tokens: " + SPELLINGS},
        {"role": "user", "content": "Repeat these: " + SPELLINGS},
        {"role": "assistant", "content": "", "tool_calls": calls("Oslo")},
        {"role": "tool", "tool_call_id": "call_Oslo", "content": "tool says " + SPELLINGS},
        {"role": "user", "content": [{"type": "text", "text": "and " + SPELLINGS}]}], "tools": [CHAT_TOOL], **NT}),
    ("chat_images", "chat", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Compare the colours."}, {"type": "image_url", "image_url": {"url": RED}},
        {"type": "image_url", "image_url": {"url": BLUE, "detail": "high"}}]}], **NT}),
    ("chat_tool_result_images", "chat", {"messages": [
        {"role": "user", "content": "Look at both."},
        {"role": "assistant", "content": "", "tool_calls": calls("a", "b")},
        {"role": "tool", "tool_call_id": "call_b", "content": [{"type": "text", "text": "image b"},
                                                               {"type": "image_url", "image_url": {"url": BLUE}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": [{"type": "text", "text": "image a"},
                                                               {"type": "image_url", "image_url": {"url": RED}}]}],
        "tools": [CHAT_TOOL], **NT}),
    ("chat_sampling", "chat", {"messages": U, "temperature": 0.25, "top_p": 0.5, "seed": 7, **NT}),
    ("chat_sampling_thinking", "chat", {"messages": U, "temperature": 0.25, "top_p": 0.5, "reasoning_effort": "low"}),
    ("chat_logprobs", "chat", {"messages": U, "logprobs": True, "top_logprobs": 3, **NT}),
    ("chat_stop_max_tokens", "chat", {"messages": U, "stop": ["323", "\n\n"], "max_tokens": 77, **NT}),
    ("chat_latest_reminder", "chat", {"messages": U + [{"role": "latest_reminder", "content": "Answer in one line."}],
                                      **NT}),
    ("chat_billing_header", "chat", {"messages": [
        {"role": "system", "content": "x-anthropic-billing-header: cc_version=1.2.3; cch=abcd\nYou are helpful."}] + U,
        **NT}),
    ("chat_agentic_6", "chat", agentic_chat_body(6, lines=4, tools=6)),
    ("msg_plain", "messages", {"max_tokens": 64, "messages": U, **NT}),
    ("msg_no_max_tokens", "messages", {"messages": U, **NT}),
    ("msg_system_blocks", "messages", {"max_tokens": 64, "system": [{"type": "text", "text": "You are terse."},
                                                                    {"type": "text", "text": "Use metric units."}],
                                       "messages": U, **NT}),
    ("msg_thinking_enabled", "messages", {"max_tokens": 4096, "messages": U,
                                          "thinking": {"type": "enabled", "budget_tokens": 2048}}),
    ("msg_thinking_adaptive_effort_medium", "messages", {"max_tokens": 4096, "messages": U,
                                                         "thinking": {"type": "adaptive"},
                                                         "output_config": {"effort": "medium"}}),
    ("msg_effort_max", "messages", {"max_tokens": 4096, "messages": U, "output_config": {"effort": "max"}}),
    ("msg_tools_roundtrip", "messages", {"max_tokens": 256, "tools": [MSG_TOOL], **NT, "messages": [
        {"role": "user", "content": W},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                                          {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                           "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": [{"type": "text", "text": "Sunny"}, {"type": "text", "text": "21C"}]}]}]}),
    ("msg_mid_system", "messages", {"max_tokens": 64, **NT, "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."},
        {"role": "system", "content": "Now answer in French."}, {"role": "user", "content": Q}]}),
    ("msg_images", "messages", {"max_tokens": 64, **NT, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": RED.split(",", 1)[1]}},
        {"type": "text", "text": "What colour?"}]}]}),
    ("msg_tool_result_image", "messages", {"max_tokens": 64, "tools": [MSG_TOOL], **NT, "messages": [
        {"role": "user", "content": "Show me."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                           "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [
            {"type": "text", "text": "radar"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": BLUE.split(",", 1)[1]}}]}]}]}),
    ("msg_special_spellings", "messages", {"max_tokens": 64, "system": "sys " + SPELLINGS, **NT,
                                           "messages": [{"role": "user", "content": SPELLINGS}]}),
    ("msg_tool_choice_any", "messages", {"max_tokens": 64, "tools": [MSG_TOOL], "tool_choice": {"type": "any"},
                                         "messages": [{"role": "user", "content": W}], **NT}),
    ("msg_sampling_stops", "messages", {"max_tokens": 64, "temperature": 0.5, "top_k": 5,
                                        "stop_sequences": ["a", "b", "c", "d", "e"], "messages": U, **NT}),
    ("resp_string", "responses", {"input": Q, "reasoning": {"effort": "none"}}),
    ("resp_default", "responses", {"input": Q}),
    ("resp_input_text_parts", "responses", {"reasoning": {"effort": "none"}, "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "First part."},
                                     {"type": "input_text", "text": Q}]}]}),
    ("resp_developer_instructions", "responses", {"instructions": "You are terse.", "reasoning": {"effort": "none"},
                                                  "input": [{"role": "developer", "content": "Answer in French."},
                                                            {"role": "user", "content": Q}]}),
    ("resp_effort_medium", "responses", {"input": Q, "reasoning": {"effort": "medium"}}),
    ("resp_effort_xhigh", "responses", {"input": Q, "reasoning": {"effort": "xhigh"}}),
    ("resp_function_roundtrip", "responses", {"tools": [RESP_TOOL], "reasoning": {"effort": "none"}, "input": [
        {"role": "user", "content": W},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"}]}),
    ("resp_reasoning_item", "responses", {"tools": [RESP_TOOL], "reasoning": {"effort": "high"}, "input": [
        {"role": "user", "content": W},
        {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "Need it."}],
         "encrypted_content": "opaque"},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"}]}),
    ("resp_namespace_tool", "responses", {"reasoning": {"effort": "none"}, "input": W, "tools": [
        {"type": "namespace", "name": "weather", "description": "Weather tools.", "tools": [RESP_TOOL]}]}),
    ("resp_apply_patch", "responses", {"reasoning": {"effort": "none"}, "input": "Fix the typo.",
                                       "tools": [{"type": "custom", "name": "apply_patch"}]}),
    ("resp_input_image", "responses", {"reasoning": {"effort": "none"}, "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "What colour?"}, {"type": "input_image", "image_url": RED}]}]}),
    ("resp_text_json_schema", "responses", {"input": Q, "reasoning": {"effort": "none"}, "text": {"format": {
        "type": "json_schema", "name": "answer", "schema": {"type": "object", "properties": {"value": {"type": "integer"}}}}}}),
    ("resp_top_logprobs", "responses", {"input": Q, "reasoning": {"effort": "none"}, "top_logprobs": 2}),
    ("resp_special_spellings", "responses", {"instructions": SPELLINGS, "input": SPELLINGS,
                                             "reasoning": {"effort": "none"}}),
]

# ---- requests the gateway hands to vLLM: (name, protocol, body[, api_body]) ----
FALLBACKS = [
    ("chat_max_completion_tokens_400k", "chat", {"messages": U, "max_completion_tokens": 400000},
     {"messages": U, "max_tokens": 400000}),
    ("chat_continue_final_message", "chat", {"messages": U + [{"role": "assistant", "content": "17*19 = "}],
                                             "continue_final_message": True, "add_generation_prompt": False}),
    ("chat_guided_json", "chat", {"messages": U, "guided_json": {"type": "object"}}),
    ("chat_structured_outputs", "chat", {"messages": U, "structured_outputs": {"regex": "[0-9]+"}}),
    ("chat_skip_special_tokens_false", "chat", {"messages": U, "skip_special_tokens": False}),
    ("chat_echo", "chat", {"messages": U, "echo": True}),
    ("chat_stop_token_ids", "chat", {"messages": U, "stop_token_ids": [128803]}),
    ("chat_ctk_drop_thinking", "chat", {"messages": U, "chat_template_kwargs": {"drop_thinking": False}}),
    ("chat_ctk_numeric_effort", "chat", {"messages": U, "chat_template_kwargs": {"reasoning_effort": 30}}),
    ("chat_unknown_model", "chat", {"messages": U, "model": "gpt-4o"}),
    ("chat_missing_model", "chat", {"messages": U, "model": None}),
    ("chat_n2", "chat", {"messages": U, "n": 2}),
    ("chat_developer_role", "chat", {"messages": [{"role": "developer", "content": "Be terse."}] + U}),
    ("chat_required_thinking", "chat", {"messages": U, "tools": [CHAT_TOOL], "tool_choice": "required",
                                        "reasoning_effort": "high"}),
    ("chat_top_logprobs_21", "chat", {"messages": U, "logprobs": True, "top_logprobs": 21}),
    ("chat_empty_messages", "chat", {"messages": []}),
    ("chat_max_tokens_400k", "chat", {"messages": U, "max_tokens": 400000}),
    ("msg_chat_template_kwargs", "messages", {"max_tokens": 64, "messages": U,
                                              "chat_template_kwargs": {"thinking": True}}),
    ("msg_web_search_tool", "messages", {"max_tokens": 64, "messages": U,
                                         "tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
    ("resp_custom_tool", "responses", {"input": Q, "tools": [{"type": "custom", "name": "grep"}]}),
    # Fields this vLLM honours that the API path would drop.
    ("chat_logit_bias", "chat", {"messages": U, "logit_bias": {"16": -100}}),
    ("chat_include_reasoning_false", "chat", {"messages": U, "reasoning_effort": "high", "include_reasoning": False}),
    ("chat_thinking_token_budget", "chat", {"messages": U, "reasoning_effort": "high", "thinking_token_budget": 64}),
    ("chat_use_beam_search", "chat", {"messages": U, "use_beam_search": True}),
    ("chat_cache_salt", "chat", {"messages": U, "cache_salt": "tenant-a"}),
    ("resp_previous_response_id", "responses", {"input": Q, "previous_response_id": "resp_123"}),
    ("resp_conversation", "responses", {"input": Q, "conversation": "conv_123"}),
]


def ranked(i: int, sampled: int, k: int = 25) -> list:
    """Deterministic logprob candidates for output position i, best first, with `sampled` at rank
    0, 2, 1 or outside the top 21 (vLLM returns at most 20 + the sampled token)."""
    rank = (0, 2, 1, k - 1)[i % 4]
    others = [1000 + (i * 37 + j * 101) % 50000 for j in range(k)]
    others = [t for t in others if t != sampled][:k - 1]
    ids = others[:rank] + [sampled] + others[rank:]
    return [[t, -(j + 1) * (1 + i % 3) / 64] for j, t in enumerate(ids)]


class Api:
    """The DeepSeek API side: deepseek-recipe with its bundled tokenizer and default spellings."""

    def __init__(self, d, tokenizer_path: str):
        self.d = d
        self.tok = d.Tokenizer.from_file(tokenizer_path)
        self.enc = d.DeepseekV41Encoding().with_tokenizer(self.tok)
        # The serve's thinking-off default and the model names the gateway accepts (no V4-Pro).
        self.opts = d.ConversionOptions(default_thinking_mode=False).with_model_versions_by_name(
            dict.fromkeys((M, "DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash-1017"), d.ModelVersion.DeepseekV41Flash))
        self.protocols = {"chat": (d.ChatCompletionRequest, d.ChatCompletionResponse),
                          "messages": (d.MessagesRequest, d.MessagesResponse),
                          "responses": (d.ResponsesRequest, d.ResponsesResponse)}

    def convert(self, protocol: str, body: dict):
        """(request model, ConversationRequest) or raises (status, error body)."""
        d = self.d
        try:
            req = self.protocols[protocol][0](copy.deepcopy(body))
            return req.model, req.convert(self.opts)
        except d.ConversionError as e:
            raise ApiReject(e.status_code, json.loads(e.body)) from None
        except ValueError as e:  # deserialization, answered 400 like server-py
            raise ApiReject(400, {"error": {"message": str(e), "type": "invalid_request_error", "param": None,
                                            "code": "invalid_request_error"}}) from None

    def prompt(self, protocol: str, body: dict) -> dict:
        d = self.d
        _, cr = self.convert(protocol, body)
        rendered = self.enc.render_conversation(cr.conversation)
        ids = self.enc.encode(cr.conversation)
        adjust = 0
        if rendered.image_sources:
            mm = d.ImageResolver(d.ReqwestImageFetcher(), d.OpenCvImagePreprocessor()).resolve(
                rendered.image_sources, d.ImageQuota())
            adjust = mm.image_token_adjustment()
        n = len(rendered.image_sources)
        io = cr.inference_options
        return {"ids": ids, "prompt": rendered.prompt, "n_images": n, "image_adjust": adjust,
                "prompt_tokens": len(ids) + adjust, "thinking": cr.conversation.thinking_mode,
                "effort": cr.conversation.reasoning_effort,
                "max_tokens": cr.resolve_max_tokens(len(ids) + adjust + n, CONTEXT),
                "temperature": io.temperature, "top_p": io.top_p, "seed": io.seed, "logprobs": io.logprobs,
                "top_logprobs": io.top_logprobs, "stream": cr.stream}

    def respond(self, protocol: str, body: dict, chunks: list, finish: str, *, prompt_tokens: int, cached: int,
                logprobs: list | None) -> dict:
        """Feed vLLM-style chunks (EOS kept on a stop) through the API's StreamProcessor."""
        d = self.d
        req_cls, resp_cls = self.protocols[protocol]
        model, cr = self.convert(protocol, body)
        top_n = (cr.inference_options.top_logprobs or 0) if cr.inference_options.logprobs else None
        proc = d.StreamProcessor(req_cls.chunk_generator(cr, RID, model), cr.parsing_options, self.tok)
        acc = resp_cls(RID, model, 0, 0, 0)
        frames, consumed, pos = [], 0, 0

        def emit(chs):
            for ch in chs:
                name = resp_cls.sse_event_name(ch)
                frames.append((f"event: {name}\n" if name else "") + f"data: {ch.to_json()}\n\n")
                acc.accumulate(ch)

        emit(proc.push(d.InferenceChunk.ready(prompt_usage=d.PromptUsage(
            prompt_tokens=prompt_tokens, prompt_cache_hit_tokens=cached))))
        for i, ids in enumerate(chunks):
            consumed = i + 1
            lps = None
            if top_n is not None:
                lps = [(lp, [tuple(c) for c in cands[:top_n]]) for lp, cands in logprobs[pos:pos + len(ids)]]
            pos += len(ids)
            if finish == "stop" and i == len(chunks) - 1 and ids[-1] == EOS:
                ids, lps = ids[:-1], lps and lps[:-1]  # the API never pushes the stop token
            if ids:
                emit(proc.push(d.InferenceChunk.tokens(ids, lps)))
            if proc.finished:
                break
        if not proc.finished:
            emit(proc.push(d.InferenceChunk.finish(
                d.InferenceFinishReason.Length if finish == "length" else d.InferenceFinishReason.Stop)))
        emit(proc.finish())
        proc.close()
        done = resp_cls.sse_done_data()
        if done:
            frames.append(f"data: {done}\n\n")
        return {"frames": [normalize(f) for f in frames], "body": normalize(acc.to_json()), "consumed": consumed,
                "stream": cr.stream}


class ApiReject(Exception):
    def __init__(self, status: int, body: dict):
        super().__init__(status)
        self.status, self.body = status, body


def normalize(text: str, rid: str = RID) -> str:
    """Blank what differs per response: the response id, timestamps, tool call ids."""
    text = text.replace(rid, RID)
    text = re.sub(r'"(created|created_at|completed_at)":\d+', r'"\1":0', text)
    return re.sub(r"call_\d\d_(ET_)?[A-Za-z0-9]{20}\d{4}", "call_ID", text)


def assistant_text(api: Api, *, content: str = "", tool_calls=None, tools=None, user: str = Q) -> str:
    """The answer part of an assistant turn (content, DSML calls) as the V4.1 encoder renders it."""
    d = api.d
    conv = d.Conversation([d.UserMessage(user), d.AssistantMessage(content, tool_calls=tool_calls)],
                          thinking_mode=False, tools=tools or [])
    p = api.enc.render_conversation(conv).prompt
    head = "<｜Assistant｜></think>"
    return p[p.index(head) + len(head):p.rindex("<｜end▁of▁sentence｜>")]


def assistant_ids(api: Api, *, thinking: bool, reasoning: str = "", **kw) -> list:
    """Model output token IDs: [reasoning </think>] answer, tokenized with the full vocabulary so
    </think> and the DSML tags are their special-token IDs."""
    return api.tok.encode((reasoning + "</think>" if thinking else "") + assistant_text(api, **kw))


def split(ids: list, sizes=(1, 6, 2, 5, 3, 4)) -> list:
    out, pos, i = [], 0, 0
    while pos < len(ids):
        out.append(ids[pos:pos + sizes[i % len(sizes)]])
        pos += sizes[i % len(sizes)]
        i += 1
    return out


def streams(api: Api) -> list:
    d = api.d
    tool = d.ToolDefinition("get_weather", "Get the weather", WEATHER)
    tc = [d.ToolCall("call_x", "get_weather", "{\"city\": \"Paris\"}")]
    text_nt = assistant_ids(api, thinking=False, content="17 × 19 = 323. Done here.")
    text_th = assistant_ids(api, thinking=True, reasoning="17*19 = 17*20 - 17 = 323.", content="323")
    tool_nt = assistant_ids(api, thinking=False, content="Let me check.", tool_calls=tc, tools=[tool], user=W)
    tool_th = assistant_ids(api, thinking=True, reasoning="Need the weather tool.", tool_calls=tc, tools=[tool], user=W)
    utf8 = assistant_ids(api, thinking=False, content="你好，世界！🙂 Ünïcødé ✓")
    after = tool_nt + api.tok.encode("\n\nIgnored trailing text.")
    calls_text = assistant_text(api, tool_calls=tc, tools=[tool], user=W)
    prefill = "\n\n<｜DSML｜ calls>\n"  # a required tool choice ends the prompt with it
    assert calls_text.startswith(prefill), calls_text[:40]
    req_calls = api.tok.encode(calls_text[len(prefill):])
    json_out = api.tok.encode("```json\n{\"value\": 323}\n```")
    cw = [{"role": "user", "content": W}]
    out = []

    def add(name, protocol, body, ids, *, finish="stop", sizes=(1, 6, 2, 5, 3, 4), cached=0, logprobs=False):
        body = {"model": M, **body}
        p = api.prompt(protocol, body)
        full = ids + [EOS] if finish == "stop" else ids
        chunks = split(full, sizes)
        lps = None
        if logprobs:  # [sampled logprob, ranked candidates] per output token, the EOS included
            lps = [[next(lp for t, lp in c if t == tok), c] for i, tok in enumerate(full) for c in [ranked(i, tok)]]
        r0 = api.respond(protocol, body, chunks, finish, prompt_tokens=p["prompt_tokens"], cached=0, logprobs=lps)
        frames, resp_body = r0["frames"], r0["body"]
        if cached and r0["consumed"] == len(chunks):
            # vLLM reports cache hits in its final usage chunk: the gateway writes them into the final
            # usage event (and the body); events sent before carry the 0 the processor started with.
            rc = api.respond(protocol, body, chunks, finish, prompt_tokens=p["prompt_tokens"], cached=cached,
                             logprobs=lps)
            last = max(i for i, (a, b) in enumerate(zip(r0["frames"], rc["frames"])) if a != b)
            frames = r0["frames"][:last] + [rc["frames"][last]] + r0["frames"][last + 1:]
            resp_body = rc["body"]
        out.append({"name": name, "protocol": protocol, "body": body, "chunks": chunks, "finish": finish,
                    "logprobs": lps, "cached": cached, "prompt_tokens": p["prompt_tokens"], "stream": r0["stream"],
                    "consumed": r0["consumed"], "frames": frames, "response": resp_body})

    add("chat_text_full", "chat", {"messages": U, **NT}, text_nt, cached=5)
    add("chat_text_stream", "chat", {"messages": U, "stream": True, **NT}, text_nt, sizes=(6, 5, 4, 3, 2, 1))
    add("chat_text_stream_usage", "chat", {"messages": U, "stream": True, "stream_options": {"include_usage": True},
                                           **NT}, text_nt, cached=7)
    add("chat_think_stream", "chat", {"messages": U, "stream": True, "reasoning_effort": "high"}, text_th, cached=3)
    add("chat_think_full", "chat", {"messages": U, "thinking": {"type": "enabled"}}, text_th)
    add("chat_tool_stream", "chat", {"messages": cw, "tools": [CHAT_TOOL], "stream": True, **NT}, tool_nt)
    add("chat_tool_full", "chat", {"messages": cw, "tools": [CHAT_TOOL], **NT}, tool_nt, sizes=(6,))
    add("chat_think_tool_stream", "chat", {"messages": cw, "tools": [CHAT_TOOL], "stream": True,
                                           "reasoning_effort": "max"}, tool_th, sizes=(5, 6))
    add("chat_logprobs_full", "chat", {"messages": U, "logprobs": True, "top_logprobs": 3, **NT}, text_nt,
        logprobs=True)
    add("chat_logprobs_stream", "chat", {"messages": U, "stream": True, "logprobs": True, **NT}, text_nt,
        logprobs=True)
    add("chat_logprobs_think_full", "chat", {"messages": U, "logprobs": True, "top_logprobs": 20,
                                             "reasoning_effort": "high"}, text_th, logprobs=True)
    add("chat_length_full", "chat", {"messages": U, "max_tokens": 7, **NT}, text_nt[:7], finish="length")
    add("chat_length_mid_tool_full", "chat", {"messages": cw, "tools": [CHAT_TOOL], "max_tokens": 16, **NT},
        tool_nt[:16], finish="length")
    add("chat_length_mid_tool_stream", "chat", {"messages": cw, "tools": [CHAT_TOOL], "max_tokens": 16,
                                                "stream": True, **NT}, tool_nt[:16], finish="length")
    add("chat_stop_sequence_stream", "chat", {"messages": U, "stop": ["323"], "stream": True, **NT}, text_nt,
        sizes=(2,), cached=4)
    add("chat_stop_sequence_full", "chat", {"messages": U, "stop": ["= 3"], **NT}, text_nt, sizes=(3,))
    add("chat_stop_in_reasoning_full", "chat", {"messages": U, "stop": ["17"], "reasoning_effort": "high"}, text_th)
    add("chat_content_after_tools_full", "chat", {"messages": cw, "tools": [CHAT_TOOL], **NT}, after)
    add("chat_required_full", "chat", {"messages": cw, "tools": [CHAT_TOOL], "tool_choice": "required", **NT},
        req_calls)
    add("chat_json_full", "chat", {"messages": [{"role": "user", "content": "JSON please"}],
                                   "response_format": {"type": "json_object"}, **NT}, json_out)
    add("chat_utf8_stream", "chat", {"messages": U, "stream": True, **NT}, utf8, sizes=(1,))
    assert len(text_nt) == 11, len(text_nt)
    add("chat_eos_alone_stream", "chat", {"messages": U, "stream": True, **NT}, text_nt, sizes=(6, 5, 1))
    add("msg_text_full", "messages", {"messages": U, "max_tokens": 1024, **NT}, text_nt, cached=6)
    add("msg_text_stream", "messages", {"messages": U, "max_tokens": 1024, "stream": True, **NT}, text_nt, cached=6)
    add("msg_think_stream", "messages", {"messages": U, "max_tokens": 4096, "stream": True,
                                         "thinking": {"type": "enabled", "budget_tokens": 2048}}, text_th)
    add("msg_tool_stream", "messages", {"messages": cw, "tools": [MSG_TOOL], "max_tokens": 1024, "stream": True,
                                        **NT}, tool_nt)
    add("msg_think_tool_full", "messages", {"messages": cw, "tools": [MSG_TOOL], "max_tokens": 4096,
                                            "thinking": {"type": "enabled"}, "output_config": {"effort": "high"}},
        tool_th)
    add("msg_length_full", "messages", {"messages": U, "max_tokens": 5, **NT}, text_nt[:5], finish="length")
    add("msg_stop_sequence_stream", "messages", {"messages": U, "max_tokens": 1024, "stream": True,
                                                 "stop_sequences": ["19"], **NT}, text_nt, sizes=(4,))
    add("resp_text_full", "responses", {"input": Q, "reasoning": {"effort": "none"}}, text_nt, cached=2)
    add("resp_text_stream", "responses", {"input": Q, "stream": True, "reasoning": {"effort": "none"}}, text_nt,
        cached=2)
    add("resp_think_stream", "responses", {"input": Q, "stream": True, "reasoning": {"effort": "high"}}, text_th)
    add("resp_tool_stream", "responses", {"input": W, "tools": [RESP_TOOL], "stream": True,
                                          "reasoning": {"effort": "none"}}, tool_nt)
    add("resp_think_tool_full", "responses", {"input": W, "tools": [RESP_TOOL], "reasoning": {"effort": "high"}},
        tool_th)
    add("resp_length_full", "responses", {"input": Q, "max_output_tokens": 4, "reasoning": {"effort": "none"}},
        text_nt[:4], finish="length")
    add("resp_logprobs_stream", "responses", {"input": Q, "stream": True, "top_logprobs": 2,
                                              "reasoning": {"effort": "none"}}, text_nt, logprobs=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api-tokenizer", required=True,
                    help="deepseek-recipe's static/tokenizers/v41/tokenizer.json (the API's own tokenizer)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    import deepseek_recipe as d

    api = Api(d, args.api_tokenizer)
    meta = {"deepseek_recipe": d.__version__, "api_tokenizer_sha256":
            hashlib.sha256(Path(args.api_tokenizer).read_bytes()).hexdigest(), "context": CONTEXT}
    prompts = []
    for name, protocol, body, *api_body in PROMPTS:
        body = {"model": M, **body}
        ref = {"model": M, **api_body[0]} if api_body else body
        prompts.append({"name": name, "protocol": protocol, "body": body, "api_body": ref, **api.prompt(protocol, ref)})
    big = agentic_chat_body(40)
    p = api.prompt("chat", big)
    meta["agentic_40"] = {"prompt_tokens": p["prompt_tokens"],
                          "ids_sha256": hashlib.sha256(json.dumps(p["ids"]).encode()).hexdigest()}
    fallbacks = []
    for name, protocol, body, *api_body in FALLBACKS:
        body = {"model": M, **body}
        if body["model"] is None:
            del body["model"]
        try:
            api.convert(protocol, {"model": M, **api_body[0]} if api_body else body)
            error = None
        except ApiReject as e:
            error = {"status": e.status, "body": e.body}
        fallbacks.append({"name": name, "protocol": protocol, "body": body, "api_error": error})
    args.out.mkdir(parents=True, exist_ok=True)
    for fname, items in (("prompts.json", prompts), ("fallbacks.json", fallbacks), ("streams.json", streams(api))):
        lines = ",\n".join(json.dumps(c, ensure_ascii=False) for c in items)  # one case per line
        (args.out / fname).write_text(f'{{"meta": {json.dumps(meta)},\n"cases": [\n{lines}\n]}}\n')
        print(f"wrote {args.out / fname}: {len(items)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
