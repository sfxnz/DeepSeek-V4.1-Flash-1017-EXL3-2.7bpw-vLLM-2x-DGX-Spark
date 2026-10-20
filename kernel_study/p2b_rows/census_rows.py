#!/usr/bin/env python3
"""CPU only: distinct experts, 8-row coop chunks and rows per expert per routed layer for the c-session verify
batches (S sequences x consecutive decode tokens, bench_rows.py's c4 source), from the s10 decode census.

  python3 kernel_study/p2b_rows/census_rows.py [--census DIR] [--draws 400]

Chunks <= 8 rows: the dataflow kernels stream an expert's B once per chunk, so chunks - distinct is the extra
B traffic of experts with > 8 rows (the wide kernel keeps 8-row chunks).
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT = Path("/home/sfxnz/projects/data/dsv41-1017/kernels/census_npy")


def seqs_for(m: int) -> int:  # = bench_rows.seqs_for
    return 1 if m <= 4 else 2 if m <= 8 else 3 if m <= 12 else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=DEFAULT)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--ms", default="1,4,8,12,16,20,24")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    caps = [np.load(p) for p in sorted(args.census.glob("*.npy"))]
    rng = random.Random(args.seed)
    layers = caps[0].shape[1]
    print(f"{len(caps)} captures x {caps[0].shape[0]} tokens x {layers} layers x top-{caps[0].shape[2]}")
    for m in (int(v) for v in args.ms.split(",")):
        s = seqs_for(m)
        sizes = [m // s + (i < m % s) for i in range(s)]
        dist = ch8 = over8 = 0
        mx = 0
        for it in range(args.draws):
            layer = it % layers
            rows = []
            for cap, n in zip(rng.sample(caps, s), sizes):
                t0 = rng.randrange(cap.shape[0] - n + 1)
                rows += cap[t0 : t0 + n, layer, :].tolist()
            c = Counter(x for r in rows for x in r)
            dist += len(c)
            ch8 += sum((v + 7) // 8 for v in c.values())
            over8 += sum(v > 8 for v in c.values())
            mx = max(mx, max(c.values()))
        n = args.draws
        print(f"m={m:2d} ({s} seq x {sizes}): distinct {dist / n:5.1f}  chunks<=8 {ch8 / n:5.1f} "
              f"(+{100 * (ch8 - dist) / dist:.1f}% B)  experts with >8 rows {over8 / n:.2f}  max rows/expert {mx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
