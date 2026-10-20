#!/usr/bin/env python3
"""FFN prologue model: router gate (cuBLAS vs router_gemv) beside the shared expert's gate_up GEMM.

In the serve the cooperative p2b MoE launch can only start when both the router chain and the shared
gate_up GEMV are done (B0 c=1: p2b starts -0.1 us after the gate_up end in 62.5% of layers, the gate_up
starts 13.1 us after the router because cuBLAS's 384 one-warp CTAs fill the SMs). This harness replays
that fork/join per layer with real 1017 weights (rank 0's shared gate_up half, the layer's router gate),
R layers per CUDA graph so every layer's weights are cold:
  fork -> side stream: router logits (cuBLAS torch.mm out_dtype=fp32, or router_gemv's kernel)
          main stream: shared gate_up (dense_gemv production GEMV at M <= 8; b12x mm_mxfp8 above)
  join (main waits on side)
Arms per M: router alone, gate_up alone, both with cuBLAS, both with the kernel. Reports us per layer
(replay / R; ABBA order, 20 warmup + --iters) and the router outputs bitwise (kernel vs cuBLAS).
Needs nvcc in the image (dense_gemv JIT, ~1 min). Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docker" / "patch"))
sys.path.insert(0, str(REPO / "kernel_study" / "dense_gemv"))
os.environ.setdefault("DGEMV_PACK", "/hf/snapshots/2.7bpw-mcg-hq")

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ms", default="3,4,8,12,16")
    args = ap.parse_args()

    import dense_gemv
    import router_gemv as rg
    import weights
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
    from vllm.triton_utils import tl, triton
    from vllm.utils import flashinfer as vfi

    ext = dense_gemv._ext()
    kernel = rg._build_kernel(tl, triton)
    L = []
    for li in range(args.layers):
        w, s2d = weights.load("shared_gate_up", li)
        _, N, K, _ = weights.SHAPES["shared_gate_up"]
        nm, kc, smode, buckets = dense_gemv.CONFIGS[(K, N)]
        L.append(dict(gu=(w, swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous(),
                          dense_gemv.build_scales(s2d, N, K, kc, smode), kc, smode, buckets, N, K),
                      router=weights.load("router_gate", li)[0]))
    wr = L[0]["router"]
    table, cublas_ms = rg.selftest(torch, triton, kernel, wr, lambda x: torch.mm(x, wr.T, out_dtype=torch.float32),
                                   wr.device)
    side = torch.cuda.Stream()
    out = {"layers": args.layers, "router_table": {m: list(v[:2]) + [list(v[2])] for m, v in sorted(table.items())},
           "router_cublas_keeps": cublas_ms, "rows": {}, "bad": []}

    for m in (int(v) for v in args.ms.split(",")):
        x = torch.randn(m, 5120, device="cuda").to(torch.bfloat16)

        def gate_up(per):
            w, wsw, sc, kc, smode, buckets, n, k = per["gu"]
            if m <= dense_gemv.MAX_M:
                W, S, MR = dense_gemv.pick(buckets, m)
                grid = int(ext.plan_grid(W, S, kc, MR, smode, 0, n, k))
                y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                ext.gemv(x, None, None, w.view(torch.uint8), sc, smode, y, W, S, kc, MR, grid, False)
                return y
            q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            return vfi.mm_mxfp8(q, w.t(), s, wsw, out_dtype=torch.bfloat16, backend="auto")

        def router(per, arm):
            w = per["router"]
            if arm == "kernel" and m in table:
                ks, order, cfg = table[m]
                o = torch.empty(m, w.shape[0], dtype=torch.float32, device="cuda")
                rg.launch(kernel, triton, x, w, o, ks, order, *cfg)
                return o
            return torch.mm(x, w.T, out_dtype=torch.float32)

        def chain(mode):
            res = []
            for per in L:
                logits = None
                if mode in ("both_cublas", "both_kernel", "router_cublas", "router_kernel"):
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        logits = router(per, "kernel" if mode.endswith("kernel") else "cublas")
                y = gate_up(per) if mode.startswith("both") or mode == "gate_up" else None
                if logits is not None:
                    torch.cuda.current_stream().wait_stream(side)
                res.append((y, logits))
            return res

        modes = ["gate_up", "router_cublas", "both_cublas"] + (["router_kernel", "both_kernel"] if m in table else [])
        graphs, outs = {}, {}
        for md in modes:
            chain(md)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outs[md] = chain(md)
            graphs[md] = g
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        if "both_kernel" in outs:
            same = all(torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))
                       for a, b in zip(outs["both_kernel"], outs["both_cublas"]))
            if not same:
                out["bad"].append(f"m={m} router logits kernel != cuBLAS in graph")
        for _ in range(20):
            for md in modes:
                graphs[md].replay()
        torch.cuda.synchronize()
        ev = {md: [] for md in modes}
        for i in range(args.iters):
            for md in (modes if i % 2 == 0 else modes[::-1]):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[md].replay()
                b.record()
                ev[md].append((a, b))
        torch.cuda.synchronize()
        row = {}
        for md, pairs in ev.items():
            t = sorted(a.elapsed_time(b) * 1e3 / args.layers for a, b in pairs)
            row[md] = {"median_us_per_layer": round(statistics.median(t), 2), "p10": round(t[len(t) // 10], 2),
                       "p90": round(t[9 * len(t) // 10], 2)}
        if "both_kernel" in row:
            row["saving_us_per_layer"] = round(row["both_cublas"]["median_us_per_layer"]
                                               - row["both_kernel"]["median_us_per_layer"], 2)
            row["saving_ms_per_step_40_layers"] = round(row["saving_us_per_layer"] * 40 / 1e3, 3)
        out["rows"][m] = row
    print(json.dumps(out))
    return 1 if out["bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
