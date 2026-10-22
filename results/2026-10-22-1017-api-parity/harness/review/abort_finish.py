#!/usr/bin/env python3
"""A vLLM token stream that ends without a finish_reason, as /inference/v1/generate sends it when the
engine aborts the request (shutdown with in-flight requests, pause(mode="abort"), ...): the abort output
carries no token IDs, so serve_tokens_stream_generator skips it (`if not delta_token_ids: continue`,
token_in_token_out/serving.py:433) and sends the usage chunk and [DONE]. Also a stream with no events at
all (Gateway.first_event's anext(events, {}) default).

  venv/bin/python -I abort_finish.py
"""
import asyncio
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-gateway/gateway")
import dsv41_gateway as G  # noqa: E402

TOK = os.path.expanduser("~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/"
                         "2.7bpw-mcg-hq/tokenizer.json")
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
gw = G.Gateway(TOK, SERVED)
text_ids = gw.tokenizer.encode("The answer is 3")  # a truncated answer; the model was going to say 323


async def listed(*events):
    for ev in events:
        yield ev


def vllm_events(rid):
    yield {"request_id": rid, "choices": [{"index": 0, "logprobs": None, "finish_reason": None,
                                           "token_ids": text_ids}], "usage": None}
    # the abort output (finish_reason "abort", no token IDs) is skipped by vLLM; then usage, then [DONE]
    yield {"request_id": rid, "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": len(text_ids),
                                                      "total_tokens": 12 + len(text_ids)}}


async def main():
    for protocol, body in (("chat", {"messages": [{"role": "user", "content": "17*19?"}]}),
                           ("messages", {"max_tokens": 64, "messages": [{"role": "user", "content": "17*19?"}]}),
                           ("responses", {"input": "17*19?"})):
        base = {"model": SERVED, "thinking": {"type": "disabled"}, **body}
        if protocol == "responses":
            base = {"model": SERVED, "reasoning": {"effort": "none"}, **body}
        prep = gw.prepare(protocol, dict(base))
        stream = await G.Gateway.first_event(listed(*vllm_events(prep.id)))
        out = json.loads(await gw.full_body(prep, stream))
        if protocol == "chat":
            print(f"chat     non-stream: finish_reason={out['choices'][0]['finish_reason']!r} "
                  f"content={out['choices'][0]['message']['content']!r}")
        elif protocol == "messages":
            print(f"messages non-stream: stop_reason={out['stop_reason']!r} content={out['content']}")
        else:
            print(f"responses non-stream: status={out['status']!r} incomplete_details={out.get('incomplete_details')!r}")
        prep = gw.prepare(protocol, dict(base, stream=True))
        stream = await G.Gateway.first_event(listed(*vllm_events(prep.id)))
        frames = [f async for f in gw.sse_frames(prep, stream)]
        print(f"{protocol:8s} stream last frames: {[f.strip()[-140:] for f in frames[-2:]]}")
    prep = gw.prepare("chat", {"model": SERVED, "thinking": {"type": "disabled"},
                               "messages": [{"role": "user", "content": "hi"}]})
    out = json.loads(await gw.full_body(prep, await G.Gateway.first_event(listed())))
    print(f"chat, upstream stream with no events at all: HTTP 200 body finish_reason="
          f"{out['choices'][0]['finish_reason']!r} content={out['choices'][0]['message']['content']!r}")


asyncio.run(main())
