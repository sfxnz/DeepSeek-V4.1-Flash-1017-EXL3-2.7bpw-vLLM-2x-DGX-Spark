"""Gateway startup time and RSS with and without the image's sitecustomize (run in the image)."""
import os, subprocess, sys, time, httpx
P = "/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq/tokenizer.json"
t0 = time.monotonic()
proc = subprocess.Popen([sys.executable, "/opt/dsv41-gateway/dsv41_gateway.py", "--host", "127.0.0.1", "--port", "18050",
                         "--tokenizer", P, "--served-model-name", "x"], env={**os.environ, "DSV41_GATEWAY_UPSTREAM": "http://127.0.0.1:1"},
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while True:
    try:
        httpx.get("http://127.0.0.1:18050/health", timeout=1)  # 503 (no upstream) once it serves
        break
    except Exception:
        time.sleep(0.05)
dt = time.monotonic() - t0
st = open(f"/proc/{proc.pid}/status").read()
maps = open(f"/proc/{proc.pid}/maps").read()
print(f"sitecustomize={'masked' if not open('/usr/lib/python3.12/sitecustomize.py').read() else 'baked'} startup_s={dt:.2f} "
      f"VmRSS_kib={st.split('VmRSS:')[1].split()[0]} libtorch_mapped={'libtorch' in maps}")
proc.terminate(); proc.wait()
