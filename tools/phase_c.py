#!/usr/bin/env python3
"""Phase C driver (decision B-4): Hessian capture + real-H encode of one node's layers, no GPU sharing.

Per node, on the host (no torch), holding that node's GPU lock (flock) for the whole run:

  loop:  encode every layer whose Hessians are complete but which has no done record
           (build_pack_1017.py run --layers <ready> --hessian-mode real, one container)
         delete the Hessians of encoded layers
         capture the next N layers of this node (tools/calib/capture_h.py: the forward resumes from
           the stream checkpoint after the previous segment's last layer, stops after the segment's
           last layer, writes a new checkpoint, exits)
  until every layer of the node has a done record; then the stream checkpoints are deleted.

N = how many layers of Hessians fit on disk now: (free - reserve - this node's not yet written shard
bytes - 2 stream checkpoints) / bytes per layer of Hessians, capped by --max-layers.
State is the disk itself, so a crash or reboot resumes at unit granularity (a capture segment or one
layer's encode): meta.json complete = captured, state/done/<unit>.json = encoded (the builder's own
claim/done store, on --state-host for spark2), CURRENT.json = stream position. A capture segment that
died is re-run from the last checkpoint; a dead encode resumes its builder claim.
Containers are --rm, named dsv41-1017-phasec-<node>-<step>; leftovers of a killed driver are removed
at start (only names this driver uses). The containers get no --gpu-lock (the driver holds it; a
nested flock on the same file from a container would deadlock).

  tools/phase_c.py run --node spark2 --layers 1,3,5-39:2 --alloc A.json --code CODE_DIR --work W \\
      --state-host spark1 [--lmhead] [--experts 0-7 --limit 8 --calib-limit 16]   (last three: tests)
  tools/phase_c.py status --node spark2 --work W ...
  tools/phase_c.py split --alloc A.json      balanced layer sets for spark1/spark2 by predicted encode time

Measured on spark2 (full calib, 410,542 tokens; data/dsv41-1017/hess/RUNS.txt): capture forward ~40 s per
layer (Engram layers 1 and 14 ~150 s), +~30 s per captured layer (384 experts, 48.4 GB written), stream
checkpoint 17 GB saved in ~15 s / loaded in ~5 s, peak GPU 62.5 GiB; real-H encode s/tensor in
SEC_PER_TENSOR below (~35 min per layer at K_gu/K_d = 3/2).

Assembly happens on spark1 (build_pack_1017.py assemble --node spark1): for every done record whose
node is spark2 it rsyncs OUT/shards/<file> from spark2 (same path, ssh alias spark2 = 10.100.8.2) and
refuses unless size and sha256 equal the done record's.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_pack_1017 as B  # noqa: E402

DATA = Path("/home/sfxnz/projects/data/dsv41-1017")
SNAP = Path("/home/sfxnz/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
            "1bae0e1088993b9b82de08f59de5be5a34e42170")
CKPT_BYTES = 20e9  # full-calib stream checkpoint (measured 16.98 GB), used until a real one exists


def parse_layers(spec: str) -> list[int]:
    """'1,3,5-9' or 'a-b:step' -> sorted layers."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rng, _, step = part.partition(":")
        a, _, b = rng.partition("-")
        out.update(range(int(a), int(b or a) + 1, int(step or 1)))
    return sorted(out)


class Driver:
    def __init__(self, a):
        self.a = a
        self.node = a.node
        self.work = Path(a.work)
        self.hess = self.work / "hessians"
        self.stream = self.work / "stream"
        self.logs = self.work / "logs"
        for d in (self.hess, self.stream, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self.layers = parse_layers(a.layers)
        self.out = Path(a.out)
        self.code = Path(a.code).resolve()
        self.child = None
        self.logf = open(self.logs / f"phase_c-{self.node}.log", "a")
        self.src = Path(a.src)
        self.units = B.plan_units(json.loads((self.src / "model.safetensors.index.json").read_text())["weight_map"])
        self.unit_of = {info["layer"]: u for u, info in self.units.items() if info["kind"] == "experts"}

    # ------------------------------------------------------------------ util
    def log(self, msg: str):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{self.node}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def store(self):
        root = self.out / "state"
        if self.a.state_host:
            ssh = B.ssh_argv(self.a.state_host, Path(self.a.ssh_dir).expanduser())
            subprocess.run(ssh + [f"mkdir -p {shlex.quote(str(root))}"], check=True, capture_output=True, timeout=60)
            return B.SshStore(ssh, root)
        root.mkdir(parents=True, exist_ok=True)
        return B.LocalStore(root)

    def captured(self, L: int) -> bool:
        return B.hessian_ready(self.hess, L)

    def encoded(self, L: int, done: set[str]) -> bool:
        u = self.unit_of[L]
        if not self.a.limit:
            return f"{u}.json" in done
        # test mode: the builder writes a partial record only; it counts when newer than the Hessians
        p = self.out / "units" / f"{u}.partial.json"
        m = self.hess / f"layer_{L:02d}" / "meta.json"
        return p.is_file() and (not m.is_file() or p.stat().st_mtime >= m.stat().st_mtime)

    def stream_layer(self) -> int:
        p = self.stream / "CURRENT.json"
        return json.loads(p.read_text())["layer_done"] if p.is_file() else -1

    def ckpt_bytes(self) -> float:
        p = self.stream / "CURRENT.json"
        if p.is_file():
            cur = json.loads(p.read_text())
            m = self.stream / cur["dir"] / "meta.json"
            if m.is_file():
                return float(json.loads(m.read_text())["bytes"])
        return CKPT_BYTES * (1.0 if not self.a.calib_limit else 0.1)

    def shard_bytes(self, L: int) -> int:
        info = self.units[self.unit_of[L]]
        hdr = B.read_header(self.src / info["file"])[1]
        copied, encoded = B.expected_layout(hdr, self.alloc)
        return B.shard_bytes({**copied, **encoded})

    def hess_layer_bytes(self) -> float:
        n = len(parse_layers(self.a.experts)) if self.a.experts != "all" else 384
        return n * 4 * (5120 ** 2 + 2304 ** 2)

    def choose_n(self, todo: list[int]) -> int:
        free = shutil.disk_usage(self.work).free
        pending_shards = 0 if self.a.limit else sum(self.shard_bytes(L) for L in todo)
        budget = free - self.a.reserve_gb * 1e9 - pending_shards - 2 * self.ckpt_bytes()
        n = int(budget // self.hess_layer_bytes())
        self.log(f"disk: free {free / 1e9:.0f} GB, reserve {self.a.reserve_gb} GB, shards to write "
                 f"{pending_shards / 1e9:.0f} GB, checkpoints {2 * self.ckpt_bytes() / 1e9:.0f} GB -> "
                 f"{max(n, 0)} layers of Hessians fit ({self.hess_layer_bytes() / 1e9:.1f} GB each)")
        if n < 1:
            raise SystemExit("not enough disk for one layer of Hessians")
        return min(n, self.a.max_layers, len(todo))

    # ------------------------------------------------------------------ containers
    def cname(self, step: str) -> str:
        return f"dsv41-1017-phasec-{self.node}-{step}"

    def kill_leftovers(self):
        r = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^dsv41-1017-phasec-{self.node}-",
                            "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
        for name in r.stdout.split():
            self.log(f"removing leftover container {name} (from a killed driver)")
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    def docker(self, step: str, mem: str, argv: list[str], network: str = "none", extra=()) -> int:
        name = self.cname(step)
        # --init: docker stop's SIGTERM reaches python
        cmd = ["docker", "run", "--rm", "--init", "--name", name, "--gpus", "all", "--memory", mem, "--memory-swap", mem,
               "--network", network, "--ipc", "host", "--ulimit", "memlock=-1", "--user", "1000:1000",
               "-e", "HOME=/home/sfxnz", "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONUNBUFFERED=1",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
               "-v", "/home/sfxnz:/home/sfxnz", "-w", str(self.code), *extra,
               "--entrypoint", "python3", self.a.image, *argv]
        logp = self.logs / f"{step}.log"
        self.log(f"start {name}: {shlex.join(argv)[:400]} (log {logp})")
        t0 = time.time()
        with open(logp, "a") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} host {os.uname().nodename}\n# {shlex.join(cmd)}\n")
            lf.flush()
            self.child = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            rc = self.child.wait()
            self.child = None
            lf.write(f"# rc={rc} {time.time() - t0:.0f}s\n")
        self.log(f"end {name}: rc={rc} in {time.time() - t0:.0f}s")
        return rc

    # ------------------------------------------------------------------ steps
    def capture(self, seg: list[int]):
        last = self.stream_layer()
        if last >= min(seg):
            # the stream is past a layer whose Hessians are missing: only a fresh forward can recover
            self.log(f"stream checkpoint after layer {last} >= {min(seg)}: restarting the forward from layer 0")
            fresh = ["--fresh"]
        else:
            fresh = []
        for L in seg:  # an interrupted segment left incomplete layer dirs: start them clean
            d = self.hess / f"layer_{L:02d}"
            if d.exists() and not self.captured(L):
                shutil.rmtree(d)
        argv = ["tools/calib/capture_h.py", "--ckpt", str(self.src), "--calib", self.a.calib,
                "--out", str(self.hess), "--state", str(self.stream), "--layers", ",".join(map(str, seg)),
                "--stop-after", str(max(seg)), "--experts", self.a.experts,
                "--summary", str(self.logs / f"capture-{min(seg):02d}-{max(seg):02d}.json"), *fresh]
        if self.a.calib_limit:
            argv += ["--limit", str(self.a.calib_limit)]
        step = f"cap{min(seg):02d}-{max(seg):02d}"
        if self.docker(step, self.a.capture_mem, argv) != 0:
            raise SystemExit(f"capture of {seg} failed (log {self.logs / step}.log)")
        missing = [L for L in seg if not self.captured(L)]
        if missing:
            raise SystemExit(f"capture exited 0 but layers {missing} are not complete")

    def encode(self, ready: list[int]):
        argv = ["-S", "tools/build_pack_1017.py", "run", "--node", self.node, "--order", "asc",
                "--out", str(self.out), "--src", str(self.src), "--alloc", str(self.a.alloc),
                "--hessian-mode", "real", "--hessians", str(self.hess), "--layers", ",".join(map(str, ready)),
                "--batch", str(self.a.batch)]
        if self.a.gates:
            argv += ["--gates", self.a.gates]
        if self.a.revision:
            argv += ["--revision", self.a.revision]
        if self.a.lmhead:
            argv += ["--lmhead"]
        if self.a.limit:
            argv += ["--limit", str(self.a.limit)]
        net, extra = "none", ["-e", "PYTHONPATH=/usr/local/lib/python3.12/dist-packages"]
        if self.a.state_host:  # the builder's SshStore runs the host's ssh (as launch_argv mounts it)
            argv += ["--state-host", self.a.state_host, "--ssh-dir", str(Path(self.a.ssh_dir).expanduser())]
            net = "host"
            extra += ["-v", "/usr/bin/ssh:/usr/bin/ssh:ro"]
        step = f"enc{min(ready):02d}-{max(ready):02d}"
        rc = self.docker(step, self.a.encode_mem, argv, network=net, extra=extra)
        if rc != 0:
            raise SystemExit(f"encode of {ready} failed rc={rc} (log {self.logs / step}.log)")

    def cleanup(self, done_layers: list[int]):
        for L in done_layers:
            d = self.hess / f"layer_{L:02d}"
            if d.exists():
                gb = sum(f.stat().st_size for f in d.iterdir()) / 1e9
                shutil.rmtree(d)
                self.log(f"deleted Hessians of layer {L} ({gb:.1f} GB)")

    def write_status(self, done: set[str], phase: str):
        st = {"node": self.node, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": phase,
              "layers": self.layers, "encoded": [L for L in self.layers if self.encoded(L, done)],
              "captured_pending": [L for L in self.layers if self.captured(L) and not self.encoded(L, done)],
              "stream_layer_done": self.stream_layer(), "free_gb": round(shutil.disk_usage(self.work).free / 1e9, 1)}
        tmp = self.work / f".status-{self.node}.json.tmp"
        tmp.write_text(json.dumps(st, indent=1) + "\n")
        os.replace(tmp, self.work / f"status-{self.node}.json")
        return st

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        self.alloc = B.load_alloc(self.a.alloc)
        pin = self.work / f"driver-{self.node}.json"
        cfg = {"layers": self.layers, "alloc": self.alloc, "code": str(self.code), "experts": self.a.experts,
               "limit": self.a.limit, "calib": self.a.calib, "calib_limit": self.a.calib_limit, "out": str(self.out)}
        if pin.is_file() and json.loads(pin.read_text()) != cfg:
            raise SystemExit(f"{pin} pins different settings (resume with the same arguments): {pin.read_text()[:400]}")
        pin.write_text(json.dumps(cfg, indent=1) + "\n")
        lockf = open(self.a.gpu_lock, "a")
        self.log(f"waiting for GPU lock {self.a.gpu_lock}")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        self.log(f"holding GPU lock; layers {self.layers}; code {self.code}; out {self.out}")
        self.kill_leftovers()
        store = self.store()
        t0 = time.time()
        while True:
            done = set(store.ls("done"))
            todo = [L for L in self.layers if not self.encoded(L, done)]
            self.write_status(done, "loop")
            self.cleanup([L for L in self.layers if L not in todo])
            if not todo:
                break
            ready = [L for L in todo if self.captured(L)]
            if ready:
                self.write_status(done, f"encode {ready}")
                self.encode(ready)
                done = set(store.ls("done"))
                left = [L for L in ready if not self.encoded(L, done)]
                if left:
                    raise SystemExit(f"encode run ended without done records for layers {left}")
                continue
            n = self.choose_n(todo)
            seg = todo[:n]
            self.write_status(done, f"capture {seg}")
            self.capture(seg)
        if not self.a.keep_stream:
            for d in self.stream.glob("ckpt-*"):
                shutil.rmtree(d)
            (self.stream / "CURRENT.json").unlink(missing_ok=True)
        self.write_status(set(store.ls("done")), "complete")
        self.log(f"all {len(self.layers)} layers encoded in {(time.time() - t0) / 3600:.2f} h this run")
        return 0

    def stop_child(self, *_):
        if self.child is not None:
            name = [x for x in self.child.args if x.startswith("dsv41-1017-phasec-")][0]
            self.log(f"signal: stopping {name}")
            subprocess.run(["docker", "stop", "-t", "30", name], capture_output=True)
        raise SystemExit(130)


# Real-H encode s/tensor on the GB10 (dsv41-quant151, batch 8; data/dsv41-1017/hess/RUNS.txt, r_eff runs,
# 8 layers x 16 experts): batch size 4..32 changes it by < 5%. GU = w1/w3 [5120 in], D = w2 [2304 in].
SEC_PER_TENSOR = {"gu": {2: 2.69, 3: 1.72, 4: 1.70}, "d": {2: 2.32, 3: 1.44, 4: 1.29}}


def layer_cost(alloc: dict, L: int, spt=SEC_PER_TENSOR, experts: int = 384) -> float:
    """Encode seconds of one layer: 2 GU tensors + 1 D tensor per expert at that layer's K."""
    k_gu, k_d = alloc[str(L)]
    return experts * (2 * spt["gu"][k_gu] + spt["d"][k_d])


def balanced_split(alloc: dict, layers=range(40), spt=SEC_PER_TENSOR) -> dict:
    """Two layer sets with near-equal encode time: start from even/odd (both nodes then forward the full
    depth anyway), then move single layers / swap pairs while that lowers the larger side."""
    a = [L for L in layers if L % 2 == 0]
    b = [L for L in layers if L % 2 == 1]
    cost = {L: layer_cost(alloc, L, spt) for L in layers}
    tot = lambda s: sum(cost[L] for L in s)  # noqa: E731
    improved = True
    while improved:
        improved = False
        big, small = (a, b) if tot(a) >= tot(b) else (b, a)
        gap = tot(big) - tot(small)
        best = None
        for x in big:
            if 0 < 2 * cost[x] < 2 * gap and (best is None or abs(gap - 2 * cost[x]) < best[0]):
                best = (abs(gap - 2 * cost[x]), x, None)
            for y in small:
                d = cost[x] - cost[y]
                if 0 < d < gap and (best is None or abs(gap - 2 * d) < best[0]):
                    best = (abs(gap - 2 * d), x, y)
        if best is not None and best[0] < gap:
            _, x, y = best
            big.remove(x)
            small.append(x)
            if y is not None:
                small.remove(y)
                big.append(y)
            improved = True
    a, b = sorted(a), sorted(b)
    return {"spark1": a, "spark2": b, "hours": {"spark1": tot(a) / 3600, "spark2": tot(b) / 3600},
            "even_odd_hours": {"even": tot([L for L in layers if L % 2 == 0]) / 3600,
                               "odd": tot([L for L in layers if L % 2 == 1]) / 3600}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split", help="balanced layer sets for the two nodes from an allocation (encode time)")
    p.add_argument("--alloc", required=True)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "split":
        a = ap.parse_args(argv)
        print(json.dumps(balanced_split(B.load_alloc(a.alloc)), indent=1))
        return 0
    for name in ("run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--node", required=True)
        p.add_argument("--layers", required=True, help="this node's layers, e.g. 1-39:2 (odd) or 0-38:2 (even)")
        p.add_argument("--alloc", required=True)
        p.add_argument("--work", required=True, help="Hessians, stream checkpoints, logs, status")
        p.add_argument("--out", default=str(B.OUT), help="builder OUT (same path on both nodes)")
        p.add_argument("--src", default=str(SNAP))
        p.add_argument("--code", default=str(ROOT), help="frozen copy of the recipe (containers run from it)")
        p.add_argument("--state-host", default=None, help="spark1 for spark2 (builder state lives on spark1)")
        p.add_argument("--ssh-dir", default="/home/sfxnz/.ssh")
        p.add_argument("--gpu-lock", default=None)
        p.add_argument("--calib", default=str(DATA / "calib/calib.jsonl"))
        p.add_argument("--calib-limit", type=int, default=0, help="test: first N calib rows")
        p.add_argument("--experts", default="all", help="test: capture only these experts (e.g. 0-7)")
        p.add_argument("--limit", type=int, default=0, help="test: builder --limit N (partial records)")
        p.add_argument("--lmhead", action="store_true", help="this node also encodes the lm_head unit")
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--gates", default=None)
        p.add_argument("--revision", default=None)
        p.add_argument("--image", default="dsv41-quant151")
        p.add_argument("--capture-mem", default="110g")
        p.add_argument("--encode-mem", default="48g")
        p.add_argument("--reserve-gb", type=float, default=25.0)
        p.add_argument("--max-layers", type=int, default=6, help="cap on layers per capture segment")
        p.add_argument("--keep-stream", action="store_true")
    a = ap.parse_args(argv)
    if a.gpu_lock is None:
        a.gpu_lock = str(DATA / f"{a.node}-gpu.lock")
    if a.experts != "all" and not a.limit:
        ap.error("--experts subset needs --limit (a real build encodes all 384 experts)")
    d = Driver(a)
    if a.cmd == "status":
        d.alloc = B.load_alloc(a.alloc)
        print(json.dumps(d.write_status(set(d.store().ls("done")), "status"), indent=1))
        return 0
    signal.signal(signal.SIGTERM, d.stop_child)
    signal.signal(signal.SIGINT, d.stop_child)
    return d.run()


if __name__ == "__main__":
    raise SystemExit(main())
