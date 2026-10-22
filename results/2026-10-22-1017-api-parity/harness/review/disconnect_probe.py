#!/usr/bin/env python3
"""Client disconnect while vLLM is still in prefill (no first event yet): does the gateway close its
upstream stream so vLLM aborts? Same container layout as http_probe.py (fake upstream on 18101 whose
/inference/v1/generate waits PREFILL_S before its first event and records when its generator is cancelled).

Cases: stream request, client leaves at T s (headers arrive after the 2 s grace window);
non-stream request, client leaves at T s.
"""
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, "/rv")
import http_probe as hp  # noqa: E402  (reuses its upstream)

SERVED = hp.SERVED


def raw_request(body: dict, leave_after: float) -> str:
    """Send one request over a plain socket, read what arrives for leave_after s, then close."""
    data = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", 18100))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n"
              + f"content-length: {len(data)}\r\n\r\n".encode() + data)
    s.settimeout(0.2)
    got, t0 = b"", time.monotonic()
    while time.monotonic() - t0 < leave_after:
        try:
            chunk = s.recv(65536)
            if not chunk:
                break
            got += chunk
        except socket.timeout:
            pass
    s.close()
    return got.decode(errors="replace").split("\r\n", 1)[0]


def main():
    import httpx

    procs = [subprocess.Popen([sys.executable, "/rv/http_probe.py", "upstream"])]
    env = {**os.environ, "DSV41_GATEWAY_UPSTREAM": hp.UP}
    procs.append(subprocess.Popen([sys.executable, "/gw/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18100",
                                   "--tokenizer", hp.TOK, "--served-model-name", SERVED], env=env,
                                  stdout=open("/tmp/gw.log", "w"), stderr=subprocess.STDOUT))
    c = httpx.Client(timeout=60)
    for _ in range(200):
        try:
            if c.get(hp.GW + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    chat = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
    for stream, leave in ((True, 1.0), (True, 4.0), (False, 1.0), (False, 4.0)):
        c.post(hp.UP + "/mock/mode", json={"mode": "slow_prefill"})
        t0 = time.monotonic()
        status_line = raw_request({**chat, "stream": stream}, leave)
        left = time.monotonic() - t0
        rec, waited = None, 0.0
        while waited < 12:
            rec = c.get(hp.UP + "/mock/log").json()[-1]
            if rec["cancelled_after_s"] is not None:
                break
            time.sleep(0.25)
            waited += 0.25
        print(f"{'stream    ' if stream else 'non-stream'} client left at {left:.1f} s (saw {status_line!r}): "
              f"upstream cancelled after {rec['cancelled_after_s']} s, events sent {rec['sent']}")
    for p in procs:
        p.terminate()
    print("--- gateway log (tail) ---")
    print("\n".join(open("/tmp/gw.log").read().splitlines()[-15:]))


if __name__ == "__main__":
    main()
