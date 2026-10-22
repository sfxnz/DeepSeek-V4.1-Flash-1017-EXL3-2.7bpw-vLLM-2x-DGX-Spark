"""Gateway VmHWM on /dsv41/render requests carrying large images (no upstream needed)."""
import base64, io, os, subprocess, sys, time, httpx
import numpy as np
import cv2
P = "/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq/tokenizer.json"
proc = subprocess.Popen([sys.executable, "/opt/dsv41-gateway/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18060",
                         "--tokenizer", P, "--served-model-name", "m"], env={**os.environ, "DSV41_GATEWAY_UPSTREAM": "http://127.0.0.1:1"},
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while True:
    try:
        httpx.get("http://127.0.0.1:18060/health", timeout=1); break
    except Exception:
        time.sleep(0.05)
def hwm():
    s = open(f"/proc/{proc.pid}/status").read()
    return {k: int(s.split(k + ":")[1].split()[0]) // 1024 for k in ("VmRSS", "VmHWM")}
print("idle MiB", hwm())
rng = np.random.default_rng(0)
img = rng.integers(0, 255, size=(3000, 4000, 3), dtype=np.uint8)
ok, png = cv2.imencode(".png", img)
url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
print("png MiB", round(len(png) / 2**20, 1))
for n in (1, 2):
    body = {"model": "m", "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content":
            [{"type": "text", "text": "x"}] + [{"type": "image_url", "image_url": {"url": url}}] * n}]}
    t0 = time.perf_counter()
    r = httpx.post("http://127.0.0.1:18060/dsv41/render/chat", json=body, timeout=120)
    j = r.json()
    print(f"{n} x 4000x3000 noise PNG: status {r.status_code} images {j.get('image_count')} fallback {str(j.get('fallback_reason'))[:80]} "
          f"render {1e3*(time.perf_counter()-t0):.0f} ms  MiB {hwm()}")
proc.terminate(); proc.wait()
