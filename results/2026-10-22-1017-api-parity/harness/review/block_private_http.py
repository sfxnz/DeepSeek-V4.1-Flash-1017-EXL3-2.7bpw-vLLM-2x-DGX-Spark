#!/usr/bin/env python3
"""DSV41_GATEWAY_BLOCK_PRIVATE_URLS=1 at the HTTP level (http_probe.py upstream echoes /v1/chat/completions):
requests bound for vLLM that name http(s) media URLs, internal scale-out routes, and a false-positive check."""
import os
import subprocess
import sys
import time

sys.path.insert(0, "/rv")
import http_probe as hp  # noqa: E402


def main():
    import httpx

    procs = [subprocess.Popen([sys.executable, "/rv/http_probe.py", "upstream"]),
             subprocess.Popen([sys.executable, "/gw/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18100",
                               "--tokenizer", hp.TOK, "--served-model-name", hp.SERVED],
                              env={**os.environ, "DSV41_GATEWAY_UPSTREAM": hp.UP, "DSV41_GATEWAY_BLOCK_PRIVATE_URLS": "1"},
                              stdout=open("/tmp/gw.log", "w"), stderr=subprocess.STDOUT)]
    c = httpx.Client(timeout=30)
    for _ in range(200):
        try:
            if c.get(hp.GW + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    img = {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/x.png"}}
    user = [{"role": "user", "content": [{"type": "text", "text": "hi"}, img]}]
    cases = [
        ("chat echo:true + private image URL", "POST", "/v1/chat/completions", {"model": hp.SERVED, "messages": user, "echo": True}),
        ("chat n:2 (API rejects) + private URL", "POST", "/v1/chat/completions", {"model": hp.SERVED, "messages": user, "n": 2}),
        ("messages ctk fallback + source.url", "POST", "/v1/messages", {"model": hp.SERVED, "max_tokens": 9, "chat_template_kwargs": {"x": 1},
                                                                       "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "http://10.0.0.1/x.png"}}]}]}),
        ("/tokenize with an image URL", "POST", "/tokenize", {"model": hp.SERVED, "messages": user}),
        ("/inference/v1/generate", "POST", "/inference/v1/generate", {"token_ids": [1], "sampling_params": {}}),
        ("/v1/chat/completions/render", "POST", "/v1/chat/completions/render", {"model": hp.SERVED, "messages": user}),
        ("FALSE POS? fallback, metadata.url, no image", "POST", "/v1/chat/completions",
         {"model": hp.SERVED, "messages": [{"role": "user", "content": "hi"}], "echo": True, "metadata": {"url": "https://example.com/doc"}}),
        ("FALSE POS? fallback, schema example url", "POST", "/v1/chat/completions",
         {"model": hp.SERVED, "messages": [{"role": "user", "content": "hi"}], "structured_outputs": {"json": {
             "type": "object", "properties": {"link": {"type": "string"}}, "examples": [{"url": "https://example.com"}]}}}),
        ("fallback with a data URL image", "POST", "/v1/chat/completions",
         {"model": hp.SERVED, "echo": True, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}),
    ]
    for name, method, path, body in cases:
        r = c.request(method, hp.GW + path, json=body)
        src = "upstream" if '"raw_path"' in r.text or r.headers.get("content-type", "").startswith("text/event-stream") else "gateway"
        print(f"{name:44s} -> {r.status_code} from {src}: {r.text[:90]}")
    for p in procs:
        p.kill()
    print("--- gateway log lines mentioning refused:")
    print("\n".join(l for l in open("/tmp/gw.log").read().splitlines() if "refused" in l)[:600])


if __name__ == "__main__":
    main()
