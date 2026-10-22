#!/usr/bin/env python3
"""A proxied streaming response (fallbacks and every other route) whose client leaves before the upstream
sent anything: is the upstream request cancelled? (http_probe.py upstream: GET /slow sleeps 30 s first.)"""
import os
import socket
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
    s = socket.create_connection(("127.0.0.1", 18100))
    s.sendall(b"GET /slow HTTP/1.1\r\nhost: x\r\n\r\n")
    time.sleep(3.0)
    s.close()
    time.sleep(2.0)
    rec = c.get(hp.UP + "/mock/log").json()[-1]
    print(f"proxied stream, client gone at 3 s before any upstream byte: upstream cancelled after "
          f"{rec['cancelled_after_s']} s")
    for p in procs:
        p.terminate()


if __name__ == "__main__":
    main()
