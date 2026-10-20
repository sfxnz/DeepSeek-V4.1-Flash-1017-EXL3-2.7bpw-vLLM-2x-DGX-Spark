#!/usr/bin/env python3
"""Target process for Nsight Compute: moe_prefill_tc on one real layer (pack), m rows, uniform routing.

  kernel_study/moe_prefill/ncu_run.sh <tag> --kernel-name regex:'gu_kernel|dn_kernel' --launch-skip 2 \
      --launch-count 2 --set full -o /data/<tag> python3 kernel_study/moe_prefill/ncu_target.py --pair 3:3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_prefill_tc as B  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=Path("/pack"))
    ap.add_argument("--pair", default="3:3")
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--calls", type=int, default=2)
    args = ap.parse_args()
    import torch
    import vllm_exl3.exl3 as P

    tag = B.tc.source_tag()
    mod = B.tc._load_prebuilt(tag) or B.tc.build(HERE / "build" / tag, B.tc.MODULE_PREFIX + tag)
    lay, inners, norm, keep = B.load_layer(P, args.pack, B.PAIR_LAYER[args.pair], 0, 2, "cuda")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(3)
    x, w = B.make_x(args.m, norm, gen, "cuda"), B.make_w(args.m, gen, "cuda")
    ids = B.routing("uniform", args.m, random.Random(3), gen, "cuda")
    for _ in range(args.calls + 1):
        B.tc_call(mod, P, x, ids, w, lay, inners)
    torch.cuda.synchronize()
    print("ncu target done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
