#!/usr/bin/env python3
"""GPU: the mma.sync tensor ceiling on this device, to read moe_prefill_tc's TFLOPS against.

  mma f32acc   m16n8k16 f16 x f16 -> f32, register-only loop, 8 independent accumulators per warp
  mma f16acc   the same with f16 accumulate
  cuBLAS       torch.matmul fp16 / bf16 (fp32 accumulate) at the routed-MoE gate/up shape, m = 8192 * 6 rows

Per variant: warps per CTA x CTAs per SM sweep; TFLOPS and FLOP/clk/SM at the clock nvidia-smi reports.
  python3 kernel_study/moe_prefill/mma_peak.py
"""

from __future__ import annotations

import subprocess
import time

SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>

template <bool F32ACC>
__global__ void mma_loop(float* out, int iters)
{
    uint32_t a[4] = {0x3c003c00u ^ threadIdx.x, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
    uint32_t b[2] = {0x3c003c00u, 0x3c003c00u ^ threadIdx.x};
    float c[8][4] = {};
    uint32_t h[8][2] = {};
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (F32ACC) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            } else {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(h[j][0]), "+r"(h[j][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += c[j][0] + c[j][3] + __half2float(__ushort_as_half((unsigned short) h[j][0]));
    if (s == 12345.f) out[threadIdx.x] = s;
}

void run(torch::Tensor out, int64_t grid, int64_t block, int64_t iters, bool f32acc)
{
    if (f32acc) mma_loop<true><<<grid, block>>>(out.data_ptr<float>(), (int) iters);
    else mma_loop<false><<<grid, block>>>(out.data_ptr<float>(), (int) iters);
}
"""


def sm_clock_mhz() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return float(out[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> int:
    import torch
    from torch.utils.cpp_extension import load_inline

    cc = torch.cuda.get_device_capability()
    arch = f"{cc[0]}{cc[1]}a"
    mod = load_inline("dsv41_mma_peak", cpp_sources="void run(torch::Tensor out, int64_t grid, int64_t block, int64_t iters, bool f32acc);",
                      cuda_sources=SRC, functions=["run"], extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                      verbose=False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    out = torch.zeros(1024, device="cuda")
    iters = 20000
    for f32acc in (True, False):
        for warps, per_sm in ((4, 1), (8, 1), (8, 2), (16, 1), (16, 2)):
            grid, block = sms * per_sm, 32 * warps
            mod.run(out, grid, block, 100, f32acc)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            mod.run(out, grid, block, iters, f32acc)
            e1.record()
            torch.cuda.synchronize()
            clk = sm_clock_mhz()
            ms = e0.elapsed_time(e1)
            flops = grid * warps * iters * 8 * 2 * 16 * 8 * 16
            tf = flops / (ms / 1e3) / 1e12
            print(f"mma {'f32acc' if f32acc else 'f16acc'} warps/CTA {warps:2d} CTAs/SM {per_sm}: {tf:6.1f} TFLOPS, "
                  f"{flops / (ms / 1e3) / sms / (clk * 1e6):6.0f} FLOP/clk/SM at {clk:.0f} MHz", flush=True)
    for dt in (torch.float16, torch.bfloat16):
        a = torch.randn(8192 * 6 // 4, 5120, device="cuda", dtype=dt)
        b = torch.randn(5120, 2304, device="cuda", dtype=dt)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        flops = 2 * a.shape[0] * 5120 * 2304
        print(f"cuBLAS {dt} [{a.shape[0]} x 5120] @ [5120 x 2304]: {flops / (ms / 1e3) / 1e12:6.1f} TFLOPS ({ms:.2f} ms)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
