#!/usr/bin/env python3
"""Non-stream requests the gateway proxies (fallbacks, /v1/completions, every other route): the gateway awaits
the upstream response with no disconnect watch, so a client that leaves does not abort vLLM's request.
The fake upstream (http_probe.py) wraps these routes like vLLM's with_cancellation and takes 30 s to answer.
Reference: the same request sent to the upstream directly, client gone at 3 s."""
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, "/rv")
import http_probe as hp  # noqa: E402


def leave(port, path, body, after):
    data = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(f"POST {path} HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {len(data)}\r\n\r\n".encode() + data)
    time.sleep(after)
    s.close()


def main():
    import httpx

    procs = [subprocess.Popen([sys.executable, "/rv/http_probe.py", "upstream"]),
             subprocess.Popen([sys.executable, "/gw/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18100",
                               "--tokenizer", hp.TOK, "--served-model-name", hp.SERVED],
                              env={**os.environ, "DSV41_GATEWAY_UPSTREAM": hp.UP}, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)]
    c = httpx.Client(timeout=60)
    for _ in range(200):
        try:
            if c.get(hp.GW + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    c.post(hp.UP + "/mock/mode", json={"mode": "slow_chat"})
    chat = {"model": hp.SERVED, "messages": [{"role": "user", "content": "hi"}], "echo": True}
    cmpl = {"model": hp.SERVED, "prompt": "hi", "max_tokens": 5}
    for name, port, path, body in (("direct to upstream  /v1/completions", 18101, "/v1/completions", cmpl),
                                   ("via gateway (proxy) /v1/completions", 18100, "/v1/completions", cmpl),
                                   ("via gateway (fallback, echo:true) /v1/chat/completions", 18100,
                                    "/v1/chat/completions", chat)):
        leave(port, path, body, 3.0)
        time.sleep(2.0)
        rec = c.get(hp.UP + "/mock/log").json()[-1]
        print(f"{name}: client gone at 3 s -> upstream cancelled after {rec['cancelled_after_s']} s")
    for p in procs:
        p.kill()


if __name__ == "__main__":
    main()
