"""CPU-only smoke of the gateway container exactly as run.sh starts it (no GPU, no serve).

1. Provision: run.sh's own ensure_gateway() body (extracted verbatim) with HOME = harness/wiring/home: real
   download from the pinned URL, sha256 check, python3 zipfile unpack, .sha256 record. Run twice: the
   second call must be a no-op.
2. Take the gateway `docker run` argv from tests/run_sh_harness.py (that HOME, the real HF cache, PORT 18000,
   VLLM_PORT 18001). Its gateway/ mount points at the dry run's temp repo copy (a symlink to the worktree's
   gateway/), so that one mount source is swapped for the worktree path; everything else is run.sh's argv.
3. A stub upstream on 127.0.0.1:18001 answers /health and /v1/models. Start the container, time /health,
   check the startup line, that no vLLM/torch is loaded, RSS, /v1/models through the proxy, a render
   (deepseek_recipe convert + render + encode with the pack tokenizer), and /health 503 once the upstream
   is gone. Then remove the container and run the engagement audit on its log.
"""
import json
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WT = Path("/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-wiring")
HERE = Path(__file__).resolve().parent
HOME = HERE / "home"
HF_CACHE = Path.home() / ".cache/huggingface"
PORT, VLLM_PORT = 18000, 18001
NAME = "dsv41-api-gateway-smoke"
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
sys.path.insert(0, str(WT / "tests"))
import run_sh_harness as h  # noqa: E402


def step(msg):
    print(f"== {msg}", flush=True)


def provision():
    run = (WT / "run.sh").read_text()
    body = re.search(r"^ensure_gateway\(\) \{\n(.*?)^\}", run, re.M | re.S).group(0)
    env = h._serve_env()
    script = "\n".join([
        "set -euo pipefail",
        'log() { printf "==> %s\\n" "$*"; }',
        "API_GATEWAY=1 SKIP_DOWNLOAD=0 PORT=8000",
        f"GATEWAY_DIR={shlex.quote(str(WT / 'gateway'))}",
        'GATEWAY_CACHE="$HOME/.cache/dsv41-gateway"',
        f"GATEWAY_WHEEL_URL={shlex.quote(env['GATEWAY_WHEEL_URL'])}",
        f"GATEWAY_WHEEL_SHA256={shlex.quote(env['GATEWAY_WHEEL_SHA256'])}",
        body,
        "ensure_gateway",
        'echo "GATEWAY_SITE=$GATEWAY_SITE"',
    ])
    for attempt in (1, 2):
        t0 = time.time()
        p = subprocess.run(["bash", "-c", script], env={"HOME": str(HOME), "PATH": "/usr/bin:/bin"},
                           capture_output=True, text=True)
        print(f"ensure_gateway call {attempt}: rc={p.returncode} {time.time() - t0:.1f} s")
        print("  " + (p.stdout + p.stderr).strip().replace("\n", "\n  "))
        assert p.returncode == 0
    site = HOME / ".cache/dsv41-gateway/deepseek_recipe-0.2.0"
    whl = HOME / ".cache/dsv41-gateway" / env["GATEWAY_WHEEL_URL"].rsplit("/", 1)[1]
    got = subprocess.run(["sha256sum", str(whl)], capture_output=True, text=True).stdout.split()[0]
    print(f"wheel {whl.name}: {whl.stat().st_size} bytes, sha256 {got} (pinned {env['GATEWAY_WHEEL_SHA256']})")
    print(f"unpacked top level: {sorted(p.name for p in site.iterdir())}")
    return site


def gateway_argv():
    res = h.dry_run(make_snapshot=False, HOME=str(HOME), HF_CACHE=str(HF_CACHE), PORT=str(PORT),
                    VLLM_PORT=str(VLLM_PORT), GATEWAY_CONTAINER_NAME=NAME, SKIP_DOWNLOAD="1")
    assert res["returncode"] == 0, res["stdout"] + res["stderr"]
    argv = res["gateway"]
    tmp_gateway = str(Path(res["env"]["STUB_RUN_SH"]).parent / "gateway")
    i = argv.index(f"{tmp_gateway}:/opt/dsv41-gateway:ro")
    argv[i] = f"{WT / 'gateway'}:/opt/dsv41-gateway:ro"
    return argv


class Upstream(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("content-length", "0")
            self.end_headers()
        elif self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": SERVED, "object": "model", "max_model_len": 500000}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


def get(path, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def main():
    step("1. provision the pinned wheel with run.sh's ensure_gateway (real download)")
    provision()
    step("2. the gateway docker run argv from the run.sh dry run")
    argv = gateway_argv()
    print("docker " + shlex.join(argv))
    step("3. stub upstream on 127.0.0.1:%d, then the container" % VLLM_PORT)
    srv = ThreadingHTTPServer(("127.0.0.1", VLLM_PORT), Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    t0 = time.time()
    subprocess.run(["docker", *argv], check=True, capture_output=True)
    try:
        while True:
            try:
                status, _, body = get("/health")
                if status == 200:
                    break
            except (urllib.error.URLError, ConnectionError):
                pass
            assert time.time() - t0 < 60, "no /health 200 within 60 s"
            time.sleep(0.05)
        print(f"/health 200 after {time.time() - t0:.2f} s (docker run to first 200): {body.decode()}")
        logs = subprocess.run(["docker", "logs", NAME], capture_output=True, text=True)
        log_text = logs.stdout + logs.stderr
        print("container log:\n  " + log_text.strip().replace("\n", "\n  "))
        print(f"dsv41-patch lines (the image's sitecustomize): {log_text.count('dsv41-patch')}")
        status, headers, body = get("/v1/models")
        print(f"/v1/models via the gateway: {status} {body.decode()}")
        req = {"model": SERVED, "messages": [{"role": "user", "content": "What is 17*19?"}]}
        status, _, body = get("/dsv41/render/chat", json.dumps(req).encode())
        info = json.loads(body)
        print(f"/dsv41/render/chat: {status} thinking={info.get('thinking')} effort={info.get('effort')} "
              f"max_tokens={info.get('max_tokens')} sampling={info.get('sampling')} "
              f"prompt_token_ids[{len(info.get('prompt_token_ids') or [])}]={info.get('prompt_token_ids')} "
              f"fallback_reason={info.get('fallback_reason')}")
        req["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": "xhigh"}
        status, _, body = get("/dsv41/render/chat", json.dumps(req).encode())
        info = json.loads(body)
        print(f"/dsv41/render/chat (ctk thinking + effort xhigh): {status} thinking={info.get('thinking')} "
              f"effort={info.get('effort')} max_tokens={info.get('max_tokens')} sampling={info.get('sampling')}")
        status1 = subprocess.run(["docker", "exec", NAME, "cat", "/proc/1/status"], capture_output=True, text=True).stdout
        rss = re.search(r"VmRSS:\s+(\d+) kB", status1).group(1)
        hwm = re.search(r"VmHWM:\s+(\d+) kB", status1).group(1)
        maps = subprocess.run(["docker", "exec", NAME, "cat", "/proc/1/maps"], capture_output=True, text=True).stdout
        cmd = subprocess.run(["docker", "exec", NAME, "cat", "/proc/1/cmdline"], capture_output=True, text=True).stdout
        print(f"PID 1: {cmd.replace(chr(0), ' ').strip()}")
        print(f"PID 1 VmRSS {int(rss) / 1024:.1f} MiB, VmHWM {int(hwm) / 1024:.1f} MiB")
        print(f"mapped libs: libtorch {maps.count('libtorch')}, vllm {maps.count('/vllm/')}, "
              f"deepseek_recipe {len(set(l.split()[-1] for l in maps.splitlines() if 'deepseek_recipe' in l))} files")
        stats = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}} cpu={{.CPUPerc}}", NAME],
                               capture_output=True, text=True).stdout.strip()
        print(f"docker stats: {stats}")
        insp = subprocess.run(["docker", "inspect", "-f",
                               "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}} {{json .HostConfig.DeviceRequests}} "
                               "{{.HostConfig.NetworkMode}}", NAME], capture_output=True, text=True).stdout.strip()
        print(f"inspect (memory bytes, nano cpus, device requests, network): {insp}")
        srv.shutdown()
        srv.server_close()
        status, _, body = get("/health")
        print(f"/health with the upstream gone: {status} {body.decode()}")
        log_file = HERE / "gateway_smoke_container.log"
        logs = subprocess.run(["docker", "logs", NAME], capture_output=True, text=True)
        log_file.write_text(logs.stdout + logs.stderr)
    finally:
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    step("4. engagement audit on the container log (API_GATEWAY=1, AUDIT=strict)")
    p = subprocess.run([sys.executable, str(WT / "tools/engagement_audit.py"), "--mode", "strict", f"gateway={log_file}"],
                       env={"API_GATEWAY": "1", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    print(f"audit rc={p.returncode}: {(p.stdout + p.stderr).strip()}")
    remaining = subprocess.run(["docker", "ps", "-a", "--filter", f"name={NAME}", "--format", "{{.Names}}"],
                               capture_output=True, text=True).stdout.strip()
    print(f"containers left named {NAME}: {remaining or 'none'}")


if __name__ == "__main__":
    main()
