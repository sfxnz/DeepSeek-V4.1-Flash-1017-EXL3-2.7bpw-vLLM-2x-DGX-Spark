"""Gateway startup with and without the sitecustomize mask (the only difference between the two runs)."""
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/home/sfxnz/projects/data/dsv41-1017/api-parity/harness/wiring")
import gateway_smoke as g  # noqa: E402

argv = g.gateway_argv()
srv = ThreadingHTTPServer(("127.0.0.1", g.VLLM_PORT), g.Upstream)
threading.Thread(target=srv.serve_forever, daemon=True).start()
mask = argv.index("/dev/null:/usr/lib/python3.12/sitecustomize.py:ro")
for label, args in (("masked (run.sh)", argv), ("unmasked", argv[: mask - 1] + argv[mask + 1 :])):
    subprocess.run(["docker", "rm", "-f", g.NAME], capture_output=True)
    t0 = time.time()
    subprocess.run(["docker", *args], check=True, capture_output=True)
    try:
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{g.PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            assert time.time() - t0 < 120
            time.sleep(0.05)
        dt = time.time() - t0
        time.sleep(1)
        st = subprocess.run(["docker", "exec", g.NAME, "cat", "/proc/1/status"], capture_output=True, text=True).stdout
        maps = subprocess.run(["docker", "exec", g.NAME, "cat", "/proc/1/maps"], capture_output=True, text=True).stdout
        logs = subprocess.run(["docker", "logs", g.NAME], capture_output=True, text=True)
        rss = int(re.search(r"VmRSS:\s+(\d+)", st).group(1)) / 1024
        print(f"{label:16s} /health 200 after {dt:.2f} s, VmRSS {rss:.0f} MiB, libtorch maps {maps.count('libtorch')}, "
              f"dsv41-patch log lines {(logs.stdout + logs.stderr).count('dsv41-patch')}")
    finally:
        subprocess.run(["docker", "rm", "-f", g.NAME], capture_output=True)
srv.shutdown()
