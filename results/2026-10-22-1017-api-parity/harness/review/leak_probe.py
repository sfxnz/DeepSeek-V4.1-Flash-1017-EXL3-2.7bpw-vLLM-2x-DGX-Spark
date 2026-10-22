#!/usr/bin/env python3
"""Stream clients that leave while vLLM is still in prefill, after FIRST_EVENT_GRACE_S (the events_after path).

Fake upstream = http_probe.py's (slow_prefill: no event for prefill_s, then 4 events and [DONE]). Gateway run as
run.sh runs it. Measures: is the upstream request cancelled (vLLM abort)? does the gateway keep its upstream
TCP connection (/proc/net/tcp, ESTABLISHED to :18101) after the upstream finished? after 100 such clients
(httpx's default max_connections), do /health and an API request still work?
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/rv")
import http_probe as hp  # noqa: E402


def leave_stream(leave_after: float) -> bytes:
    body = json.dumps({"model": hp.SERVED, "messages": [{"role": "user", "content": "hi"}],
                       "thinking": {"type": "disabled"}, "stream": True}).encode()
    s = socket.create_connection(("127.0.0.1", 18100))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n"
              + f"content-length: {len(body)}\r\n\r\n".encode() + body)
    s.settimeout(0.2)
    got, t0 = b"", time.monotonic()
    while time.monotonic() - t0 < leave_after:
        try:
            got += s.recv(65536)
        except socket.timeout:
            pass
    s.close()
    return got.split(b"\r\n", 1)[0]


def established_to_upstream() -> int:
    n = 0
    for line in open("/proc/net/tcp").read().splitlines()[1:]:
        f = line.split()
        if f[3] == "01" and int(f[2].split(":")[1], 16) == 18101:  # ESTABLISHED, remote port 18101
            n += 1
    return n


def main():
    import httpx

    procs = [subprocess.Popen([sys.executable, "/rv/http_probe.py", "upstream"])]
    env = {**os.environ, "DSV41_GATEWAY_UPSTREAM": hp.UP}
    procs.append(subprocess.Popen([sys.executable, "/gw/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18100",
                                   "--tokenizer", hp.TOK, "--served-model-name", hp.SERVED], env=env,
                                  stdout=open("/tmp/gw.log", "w"), stderr=subprocess.STDOUT))
    c = httpx.Client(timeout=60)
    for _ in range(200):
        try:
            if c.get(hp.GW + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    c.close()
    c = httpx.Client(timeout=60)

    def up(path, **kw):
        with httpx.Client(timeout=10) as u:
            return u.request("POST" if kw else "GET", hp.UP + path, **kw).json()

    up("/mock/mode", json={"mode": "slow_prefill", "prefill_s": 6})
    print("one stream client, gone at 4 s (prefill 6 s):", leave_stream(4.0))
    time.sleep(6.0)
    rec = up("/mock/log")[-1]
    print(f"  upstream: cancelled_after_s={rec['cancelled_after_s']} events_sent={rec['sent']} finished={rec['done']}; "
          f"gateway sockets ESTABLISHED to the upstream port: {established_to_upstream()}")

    up("/mock/mode", json={"mode": "slow_prefill", "prefill_s": 600})
    threads = [threading.Thread(target=leave_stream, args=(3.0,)) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(1.0)
    log = up("/mock/log")[1:]
    print(f"100 stream clients gone at 3 s (prefill 600 s): upstream requests cancelled "
          f"{sum(r['cancelled_after_s'] is not None for r in log)}/{len(log)}; gateway sockets to the upstream: "
          f"{established_to_upstream()}")
    t0 = time.monotonic()
    h = c.get(hp.GW + "/health", timeout=30)
    print(f"  gateway /health: {h.status_code} after {time.monotonic() - t0:.1f} s {h.text[:110]}")
    up("/mock/mode", json={"mode": "ok"})
    t0 = time.monotonic()
    try:
        r = c.post(hp.GW + "/v1/chat/completions", json={"model": hp.SERVED, "thinking": {"type": "disabled"},
                                                         "messages": [{"role": "user", "content": "hi"}]}, timeout=20)
        out = str(r.status_code)
    except httpx.HTTPError as e:
        out = repr(e)[:50]
    print(f"  a new API request (upstream fast again): {out} after {time.monotonic() - t0:.1f} s")
    for p in procs:
        p.terminate()


if __name__ == "__main__":
    main()
