#!/usr/bin/env python3
"""DSV41_GATEWAY_BLOCK_PRIVATE_URLS=1 only guards the gateway's own fetch: requests that fall back reach
vLLM unchanged, URL included, and vLLM fetches it (follows redirects, no address filter).
  venv/bin/python -I block_private_bypass.py"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-gateway/gateway")
import dsv41_gateway as G  # noqa: E402

TOK = os.path.expanduser("~/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/"
                         "2.7bpw-mcg-hq/tokenizer.json")
gw = G.Gateway(TOK, "m", block_private_urls=True)
img = {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/meta-data/iam/x.png"}}
base = {"model": "m", "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "describe"}, img]}]}
for name, extra in (("plain", {}), ("echo: true", {"echo": True}),
                    ("chat_template_kwargs drop_thinking", {"chat_template_kwargs": {"drop_thinking": False}}),
                    ("n: 2 (API rejects)", {"n": 2}), ("unknown model (API rejects)", {"model": "x"})):
    try:
        gw.prepare("chat", {**base, **extra})
        print(f"{name:36s} -> API path")
    except G.ApiError as e:
        print(f"{name:36s} -> {e.status} {e.body['error']['message'][:90]}")
    except G.Fallback as e:
        print(f"{name:36s} -> Fallback ({str(e)[:60]}): body with the 169.254.169.254 URL goes to vLLM unchanged")
