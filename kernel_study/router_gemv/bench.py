#!/usr/bin/env python3
"""router_gemv on the GPU: bitwise vs cuBLAS and timing, M = 1..24 (DSV41_ROUTER_GEMV).

Runs in the serving image (e15-kbits) on one GB10, nothing else on the GPU. Real 1017 weights:
target router gates (bf16 [384, 5120], layers --layers), the 3 DSpark draft gates ([128, 5120]),
compressor fused wkv|wgate of layers 2/8/14 ([1024, 5120]) and wkv of layer 20 ([512, 5120]).

1. Discovery: router_gemv.selftest per shape (the serve's own first-call self-test), its wall time,
   the per-M (KS, slices, order) table and the M left to cuBLAS.
2. Bitwise: every weight, every M in the table, x = randn * {1, 8, 30} and an RMS-normed row
   distribution, kernel (serving config) vs torch.mm(x, w.T, out_dtype=fp32) eager; then cuBLAS
   captured in a CUDA graph vs cuBLAS eager (the self-test is eager), and the kernel captured in a
   graph, replayed 20x with new rows, vs eager cuBLAS.
3. Wrapper: router_gemv.make_gate_forward on a stand-in gate (verify, fallback above MAX_M, the
   (out, None) return), and mm_f32 with a flipped-bit kernel (verify must return cuBLAS's output and
   disarm).
4. Timing (graph replays, >= --iters, arms interleaved, ~90 us sleep in front so the launch hides):
   cold L2 (128 MiB write between replays), warm, and beside a DRAM-saturating copy; plus the FFN
   prologue model: the gate GEMM and a 12.2 MB side-stream copy (stand-in for the shared expert's
   gate_up GEMV) launched together, time until both are done.
Prints one JSON line; exit 1 on any bitwise failure.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types

import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker" / "patch"))  # this worktree's patch

SNAP = "/hf/snapshots/2.7bpw-mcg-hq"  # run_gpu.sh mounts models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3 at /hf
TIMING_MS = (1, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def summarize(v):
    return dict(n=len(v), median=round(statistics.median(v), 2), p10=round(pct(v, 0.10), 2),
                p90=round(pct(v, 0.90), 2))


def load(snap, layers):
    from safetensors import safe_open

    idx = json.load(open(snap + "/model.safetensors.index.json"))["weight_map"]

    def get(name):
        with safe_open(snap + "/" + idx[name], framework="pt") as f:
            return f.get_tensor(name)

    shapes = {
        "router384": [get(f"layers.{i}.ffn.gate.weight") for i in layers],
        "draft128": [get(f"mtp.{i}.ffn.gate.weight") for i in range(3)],
        "kv1024": [torch.cat([get(f"layers.{i}.attn.compressor.wkv.weight"),
                              get(f"layers.{i}.attn.compressor.wgate.weight")]) for i in (2, 8, 14)],
        "kv512": [get("layers.20.attn.compressor.wkv.weight")],
    }
    return {k: [w.to(torch.bfloat16).cuda().contiguous() for w in v] for k, v in shapes.items()}


def stock(x, w):
    return torch.mm(x, w.T, out_dtype=torch.float32)


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def inputs(m, k, g, dev):
    base = torch.randn(m, k, device=dev, generator=g)
    rms = base * torch.rsqrt(base.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + 0.5 * torch.rand(k, device=dev, generator=g))
    return [(base * s).to(torch.bfloat16) for s in (1.0, 8.0, 30.0)] + [rms.to(torch.bfloat16)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=SNAP)
    ap.add_argument("--layers", default="0,1,20,39")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--configs", default="", help="extra serving configs to time, 'bk:warps:stages,...'")
    args = ap.parse_args()

    import router_gemv as rg
    from vllm.triton_utils import tl, triton

    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    kernel = rg._build_kernel(tl, triton)
    weights = load(args.snap, [int(v) for v in args.layers.split(",")])
    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(), "shapes": {}, "bad": []}

    # 1. discovery (the serve's self-test) per shape
    tables = {}
    for name, ws in weights.items():
        w = ws[0]
        t0 = time.time()
        table, stock_ms = rg.selftest(torch, triton, kernel, w, lambda x, w=w: stock(x, w), dev)
        torch.cuda.synchronize()
        tables[name] = table
        out["shapes"][name] = {
            "n": int(w.shape[0]), "k": int(w.shape[1]), "selftest_s": round(time.time() - t0, 2),
            "table": {m: [ks, rg.cdiv(w.shape[1], ks), order, list(cfg)] for m, (ks, order, cfg) in sorted(table.items())},
            "cublas_keeps": stock_ms,
        }

    # 2. bitwise: every weight, every served M, 4 input distributions; eager and graph
    for name, ws in weights.items():
        table = tables[name]
        n_cases = n_bad = 0
        for li, w in enumerate(ws):
            for m, (ks, order, cfg) in sorted(table.items()):
                for x in inputs(m, w.shape[1], g, dev):
                    o = torch.empty(m, w.shape[0], device=dev, dtype=torch.float32)
                    rg.launch(kernel, triton, x, w, o, ks, order, *cfg)
                    ref = stock(x, w)
                    torch.cuda.synchronize()
                    n_cases += 1
                    if not bits_equal(o, ref):
                        n_bad += 1
                        out["bad"].append(f"{name} w{li} m={m} ks={ks} order={order}")
        out["shapes"][name]["bitwise_eager"] = {"cases": n_cases, "mismatch": n_bad}
        # graph: cuBLAS captured vs eager, and the kernel captured + replayed with new rows
        w = ws[0]
        graph_bad = []
        for m, (ks, order, cfg) in sorted(table.items()):
            xs = torch.randn(m, w.shape[1], device=dev, generator=g).to(torch.bfloat16)
            o = torch.empty(m, w.shape[0], device=dev, dtype=torch.float32)
            stock(xs, w)
            rg.launch(kernel, triton, xs, w, o, ks, order, *cfg)
            torch.cuda.synchronize()
            gs, gk = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(gs):
                s_out = stock(xs, w)
            with torch.cuda.graph(gk):
                rg.launch(kernel, triton, xs, w, o, ks, order, *cfg)
            for rep in range(20 if m in (3, 4, 12, 16, 24) else 3):
                xs.copy_(torch.randn(m, w.shape[1], device=dev, generator=g).to(torch.bfloat16))
                gs.replay()
                gk.replay()
                ref = stock(xs, w)
                torch.cuda.synchronize()
                if not bits_equal(s_out, ref):
                    graph_bad.append(f"cublas graph != eager m={m} rep={rep}")
                if not bits_equal(o, ref):
                    graph_bad.append(f"kernel graph != cublas m={m} rep={rep}")
            del gs, gk
        out["shapes"][name]["graph_mismatch"] = graph_bad
        out["bad"] += [f"{name} {b}" for b in graph_bad]

    # 3. wrapper and sabotage
    rg._STATE.update(armed=True, verify_left=4, tables={}, kernel=kernel)
    w = weights["router384"][0]
    gate = types.SimpleNamespace(weight=w, allow_cublas_router_gemm=True)
    fwd = rg.make_gate_forward(torch, lambda self, x: (stock(x, self.weight), None))
    for m in (1, 4, 16, 24, 25, 64):
        x = torch.randn(m, w.shape[1], device=dev, generator=g).to(torch.bfloat16)
        got, b = fwd(gate, x)
        torch.cuda.synchronize()
        if b is not None or not bits_equal(got, stock(x, w)):
            out["bad"].append(f"wrapper m={m}")
    out["wrapper_state"] = {"armed": rg._STATE["armed"], "verify_left": rg._STATE["verify_left"],
                            "shapes": sorted(map(list, rg._STATE["tables"]))}

    class Corrupt:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(x, w, o, *a, **kw):
                launch(x, w, o, *a, **kw)
                o.view(torch.int32).view(-1)[:1].bitwise_xor_(1)

            return run

    sab = {}
    rg._STATE.update(armed=True, verify_left=2, tables=dict(rg._STATE["tables"]), kernel=Corrupt(kernel))
    x = torch.randn(4, w.shape[1], device=dev, generator=g).to(torch.bfloat16)
    first = rg.mm_f32(x, w)
    sab["verify_disarmed"] = not rg._STATE["armed"]
    second = rg.mm_f32(x, w)
    ref = stock(x, w)
    torch.cuda.synchronize()
    sab["verify_returned_stock"] = bits_equal(first, ref)
    sab["next_call_stock"] = bits_equal(second, ref)
    rg._STATE.update(armed=True, verify_left=2, tables={}, kernel=Corrupt(kernel))
    third = rg.mm_f32(x, w)
    torch.cuda.synchronize()
    sab["selftest_kept_cublas"] = rg._STATE["tables"].get(tuple(w.shape)) == {} and bits_equal(third, ref)
    out["sabotage"] = sab
    if not all(sab.values()):
        out["bad"].append(f"sabotage {sab}")
    rg._STATE.update(armed=True, verify_left=0, tables=dict(tables_by_key(tables, weights)), kernel=kernel)

    # 4. timing in CUDA graphs
    flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    side = torch.cuda.Stream()
    big_src = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    big_dst = torch.empty_like(big_src)
    hog_src = torch.empty(int(12.2e6), dtype=torch.uint8, device=dev)
    hog_dst = torch.empty_like(hog_src)
    extra = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",") if c]
    timing = {}
    for name, ws in weights.items():
        w = ws[0]
        table = tables[name]
        for m in TIMING_MS:
            x = torch.randn(m, w.shape[1], device=dev, generator=g).to(torch.bfloat16)
            o = torch.empty(m, w.shape[0], device=dev, dtype=torch.float32)
            arms = {"cublas": lambda x=x, w=w: stock(x, w)}
            if m in table:
                ks, order, cfg = table[m]
                arms["kernel"] = (lambda x=x, w=w, o=o, ks=ks, order=order, cfg=cfg:
                                  rg.launch(kernel, triton, x, w, o, ks, order, *cfg))
                for bk, warps, stages in extra:
                    arms[f"kernel_{bk}:{warps}:{stages}"] = (lambda x=x, w=w, o=o, ks=ks, order=order, c=(bk, warps, stages):
                                                             rg.launch(kernel, triton, x, w, o, ks, order, *c))
            graphs = {}
            for an, fn in arms.items():
                fn()
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                graphs[an] = gr
            names = list(graphs)
            for mode in ("cold", "warm", "beside_copy", "prologue"):
                t = {an: [] for an in names}
                for it in range(args.warmup + args.iters):
                    order_ = names[it % len(names):] + names[: it % len(names)]
                    for an in order_:
                        if mode in ("cold", "prologue"):
                            flush.fill_(it & 0xFF)
                        if mode == "beside_copy":
                            side.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(side):
                                big_dst.copy_(big_src)
                            torch.cuda._sleep(20000)
                        else:
                            torch.cuda._sleep(200_000)  # ~90 us: the graph launch lands behind it
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        if mode == "prologue":
                            side.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(side):
                                hog_dst.copy_(hog_src)
                            graphs[an].replay()
                            torch.cuda.current_stream().wait_stream(side)
                        else:
                            graphs[an].replay()
                        e1.record()
                        torch.cuda.synchronize()
                        if it >= args.warmup:
                            t[an].append(e0.elapsed_time(e1) * 1e3)
                timing[f"{name}_m{m}_{mode}"] = {an: summarize(v) for an, v in t.items()}
    # the hog alone, for the prologue model
    t = []
    for it in range(args.warmup + args.iters):
        flush.fill_(it & 0xFF)
        torch.cuda._sleep(200_000)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        hog_dst.copy_(hog_src)
        e1.record()
        torch.cuda.synchronize()
        if it >= args.warmup:
            t.append(e0.elapsed_time(e1) * 1e3)
    timing["hog_alone_cold"] = summarize(t)
    out["timing_us"] = timing
    print(json.dumps(out))
    return 1 if out["bad"] else 0


def tables_by_key(tables, weights):
    return {(int(ws[0].shape[0]), int(ws[0].shape[1])): tables[name] for name, ws in weights.items()}


if __name__ == "__main__":
    sys.exit(main())
