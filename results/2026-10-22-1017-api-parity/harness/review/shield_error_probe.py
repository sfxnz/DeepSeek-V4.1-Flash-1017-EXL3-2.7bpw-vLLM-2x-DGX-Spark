#!/usr/bin/env python3
"""The shielded first-event task when the upstream fails after the grace window: (a) the client stays (expect
200, start chunk, API-shaped error frame); (b) the client left at 3 s, the upstream dies at 4 s. Then the
gateway log is checked for unhandled-exception noise."""
import json
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
             subprocess.Popen([sys.executable, "-X", "dev", "/gw/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18100",
                               "--tokenizer", hp.TOK, "--served-model-name", hp.SERVED],
                              env={**os.environ, "DSV41_GATEWAY_UPSTREAM": hp.UP}, stdout=open("/tmp/gw.log", "w"),
                              stderr=subprocess.STDOUT)]
    c = httpx.Client(timeout=60)
    for _ in range(200):
        try:
            if c.get(hp.GW + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    chat = {"model": hp.SERVED, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}, "stream": True}
    for mode in ("slow_drop", "slow_malformed"):
        c.post(hp.UP + "/mock/mode", json={"mode": mode, "prefill_s": 4})
        with c.stream("POST", hp.GW + "/v1/chat/completions", json=chat) as r:
            raw = b"".join(r.iter_raw()).decode()
        frames = [f for f in raw.split("\n\n") if f]
        print(f"{mode}: client stays -> {r.status_code}, {len(frames)} frames, last: {frames[-1][:150]!r}")
    c.post(hp.UP + "/mock/mode", json={"mode": "slow_drop", "prefill_s": 4})
    data = json.dumps(chat).encode()
    s = socket.create_connection(("127.0.0.1", 18100))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n"
              + f"content-length: {len(data)}\r\n\r\n".encode() + data)
    time.sleep(3.0)
    s.close()
    time.sleep(3.0)
    rec = c.get(hp.UP + "/mock/log").json()[-1]
    print(f"slow_drop: client left at 3 s -> upstream cancelled after {rec['cancelled_after_s']} s")
    for p in procs:
        p.terminate()
    time.sleep(0.5)
    log = open("/tmp/gw.log").read()
    noise = [l for l in log.splitlines() if any(k in l for k in ("Traceback", "never retrieved", "Exception in ASGI", "Task was destroyed", "ResourceWarning", "ERROR"))]
    print(f"gateway log: {len(noise)} error/noise lines", *noise[:8], sep="\n  ")


if __name__ == "__main__":
    main()
