#!/usr/bin/env python3
"""router_gemv kernel structure under Triton's CPU interpreter (no GPU).

  CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 OMP_NUM_THREADS=2 python3 interp_check.py

The interpreter runs tl.dot as numpy fp32 matmuls, so this cannot prove bit-exactness with cuBLAS
(the GPU harness bench.py does). It proves the schedule: the interpreted kernel equals, bit for bit, a
numpy re-implementation of router_gemv.schedule (per slice a chain of BK-block matmuls from zero,
masked slice tails, slices summed in SEQ or REV order) for slicings 1/9/16, orders SEQ/REV, every
launch config, M 1/4/17/24 and an N that is not a multiple of 16 (masked columns). K = 1024 keeps the
interpreter run short (the loop logic does not depend on K). It also checks the
slicing changes the result (the structure is really applied) and that out-of-range rows/columns are
never written. The interpreter mishandles bf16 dot operands (garbage, while fp32 is exact), so the
operands are fp32 tensors holding bf16-rounded values: the schedule and mask logic is the same code.
Prints one JSON line; exit 1 on a mismatch.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker" / "patch"))
import router_gemv as rg  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


def reference(x, w, ks, order, bk):
    """numpy mirror of the kernel loop over rg.schedule: per (BM x 16) output tile, fp32 block matmuls
    of the zero-padded tile operands (the interpreter's tl.dot shapes), slice resets, ordered sum."""
    m, k = x.shape
    n = w.shape[0]
    bm = rg.bm_for(m)
    out = np.zeros((m, n), np.float32)
    for c0 in range(0, n, rg.BN):
        c1 = min(c0 + rg.BN, n)
        total = np.zeros((bm, rg.BN), np.float32)
        acc = np.zeros_like(total)
        for s, k0, kend, first in rg.schedule(k, ks, bk, order):
            if first:
                total = total + acc
                acc = np.zeros_like(acc)
            hi = min(k0 + bk, kend)
            a = np.zeros((bm, bk), np.float32)
            b = np.zeros((bk, rg.BN), np.float32)
            a[:m, : hi - k0] = x[:, k0:hi]
            b[: hi - k0, : c1 - c0] = w[c0:c1, k0:hi].T
            acc = np.matmul(a, b) + acc
        out[:, c0:c1] = (total + acc)[:m, : c1 - c0]
    return out


def main() -> int:
    assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
    kernel = rg._build_kernel(tl, triton)
    g = torch.Generator().manual_seed(0)
    k, n = 1024, 40  # n not a multiple of 16: the last tile's columns 40..47 are masked
    w = torch.randn(n, k, generator=g).to(torch.bfloat16).float()
    out = {"cases": 0, "bitwise_vs_schedule_mirror": 0, "bad": [], "slicing_changes_bits": None}
    configs = sorted({rg.DISCOVERY, rg.SERVE_SPLIT, rg.SERVE_UNSPLIT})
    for ks in (1024, 120, 64):  # 1, 9, 16 slices (120 = round_up(ceil(1024 / 9), 8): a masked slice tail)
        for order in (rg.SEQ, rg.REV):
            if order == rg.REV and ks == k:
                continue
            for m in (1, 4, 17, 24):
                x = torch.randn(m, k, generator=g).to(torch.bfloat16).float()
                xf, wf = x.numpy(), w.numpy()
                cfgs = configs + [rg.serve_config(k, ks, m)]
                for cfg in sorted(set(cfgs)):
                    bk = cfg[0]
                    o = torch.full((m + 2, n + 8), 7.0, dtype=torch.float32)  # sentinel beyond the output
                    view = o[:m, :n]
                    rg.launch(kernel, triton, x, w, view, ks, order, *cfg)
                    got = view.numpy()
                    ref = reference(xf, wf, ks, order, bk)
                    out["cases"] += 1
                    if np.array_equal(got.view(np.int32), ref.view(np.int32)):
                        out["bitwise_vs_schedule_mirror"] += 1
                    else:
                        err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
                        out["bad"].append(f"ks={ks} order={order} m={m} cfg={cfg} rel={err:.3g}")
                    if not (torch.all(o[m:, :] == 7.0) and torch.all(o[:, n:] == 7.0)):
                        out["bad"].append(f"wrote outside the output ks={ks} m={m} cfg={cfg}")
    # the slicing is really applied: 8 slices of 2 blocks vs 1 slice differ in the last bits (in fp32
    # emulation a one-block slice is the same addition as the unsplit chain, so 2 blocks per slice)
    x = torch.randn(4, k, generator=g).to(torch.bfloat16).float()
    a = torch.empty(4, n)
    b = torch.empty(4, n)
    rg.launch(kernel, triton, x, w, a, 128, rg.SEQ, *rg.DISCOVERY)
    rg.launch(kernel, triton, x, w, b, k, rg.SEQ, *rg.DISCOVERY)
    out["slicing_changes_bits"] = bool(not torch.equal(a.view(torch.int32), b.view(torch.int32)))
    out["max_rel_diff_1_vs_8_slices"] = float(((a - b).abs().max() / b.abs().max()).item())
    if not out["slicing_changes_bits"]:
        out["bad"].append("slicing did not change the bits")
    out["torch"], out["triton"] = torch.__version__, triton.__version__
    print(json.dumps(out))
    return 1 if out["bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
