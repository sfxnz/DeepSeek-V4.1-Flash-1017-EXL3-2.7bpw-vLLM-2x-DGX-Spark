#!/usr/bin/env python3
"""Interleaved A/B of the dataflow kernel's launch bounds for K != 2: (256, 3) vs (256, 2) (GPU).

Both bench modules (bench_kbits_lb3.cu = 3 blocks/SM for every K, bench_kbits.cu = the patch: 2 for K != 2) in one process, same layer data and routing per
iteration, cold (L2 flush) calls alternating A/B order per iteration; per (pair, m): medians and the
paired per-call difference (mean +- 95% CI). Also checks both arms give bitwise-equal outputs
(launch bounds change register allocation only).

  python3 kernel_study/p2b_kbits/ab_lb.py --pairs 3:3,4:4,3:2,2:4 --ms 1,4,8 --iters 400 --out /data/runs/kb-ab-lb.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_kbits as bk  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="3:3,4:4,3:2,2:3,3:4,4:3,4:2,2:4")
    ap.add_argument("--ms", default="1,3,4,6,8")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ext = {"lb3": bk.build_ext(True), "lb2": bk.build_ext(False)}
    flush = bk.R.Flusher("cuda")
    res = {}
    for pair in args.pairs.split(","):
        kgu, kd = (int(k) for k in pair.split(":"))
        layer = bk.Layer(20, 0, kgu, kd, "cuda")
        routes = bk.R.Routings(20, random.Random(11))
        b = {k: bk.Bench(e, layer, "cuda", 11) for k, e in ext.items()}
        for m in (int(x) for x in args.ms.split(",")):
            x, rw = b["lb3"].x(m), b["lb3"].rw(m)
            outs = {k: torch.empty_like(x) for k in b}
            same = True
            evs = {k: [] for k in b}
            for i in range(args.iters + 10):
                ids = b["lb3"].ids(routes.draw("census", m))
                for k in (("lb3", "lb2") if i % 2 == 0 else ("lb2", "lb3")):
                    flush()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    b[k].call(2, x, ids, rw, outs[k])
                    e1.record()
                    if i >= 10:
                        evs[k].append((e0, e1))
                torch.cuda.synchronize()
                same &= bk.R.bitwise(outs["lb3"], outs["lb2"])
            t = {k: [a.elapsed_time(c) * 1000.0 for a, c in v] for k, v in evs.items()}
            d = [p - q for p, q in zip(t["lb3"], t["lb2"])]
            ci = 1.96 * statistics.stdev(d) / len(d) ** 0.5
            r = {"lb3_median_us": statistics.median(t["lb3"]), "lb2_median_us": statistics.median(t["lb2"]),
                 "saving_mean_us": statistics.mean(d), "saving_ci95_us": ci,
                 "saving_pct": 100 * statistics.mean(d) / statistics.mean(t["lb3"]), "bitwise_equal": same,
                 "n": len(d)}
            res[f"{pair}/m{m}"] = r
            bk.log(f"[ab] {pair} m={m}: lb3 {r['lb3_median_us']:.1f} lb2 {r['lb2_median_us']:.1f} us, lb2 saves "
                   f"{r['saving_mean_us']:.1f} +- {ci:.1f} us ({r['saving_pct']:.1f}%), bitwise {same}")
        del b, layer
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
