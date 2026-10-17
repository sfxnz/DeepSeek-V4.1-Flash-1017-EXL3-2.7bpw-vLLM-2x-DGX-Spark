#!/usr/bin/env python3
"""K = 3 / 4 and per-projection K (K_gu != K_d) for the p2b decode MoE kernels (1017 pack).

Before this patch the round-3 kernels were K = 2 MCG only: DSV41_P2B_COOP=1 (SORT=2 coop) and
DSV41_P2B_COOP=2 (p2b_coop_df_kernel) fell back to the SORT=0 p2b for any other K, and the host
required K_gate == K_up == K_down. The 1017 pack stores routed experts at K in {2, 3, 4} per
(layer, projection group): gate/up (w1, w3) at K_gu, down (w2) at K_d.

Kernel side (csrc/p2b_moe.cu, apply after widen_p2b_dataflow.py):
- The BITS template argument of p2b_moe_batched_kernel / p2b_coop_df_kernel / launch_moe_batched
  now encodes both K: BITS = K when K_gu == K_d (2, 3, 4: the existing instantiations, same names,
  same machine code), else BITS = 10 * K_gu + K_d (23, 24, 32, 34, 42, 43). p2b_kgu(BITS) and
  p2b_kd(BITS) pick the tile K of the gate/up and down phases.
- p2b_df_tile and p2b_df_prefetch_tile are templated on the tile K. The K-specific lines are the
  ones run_gemv_tile_coop already has (TWORDS = 8K, LOADS, LSTRIDE, the K = 3 lane < 24 load guard,
  the per-K lane shuffles and dq8_regs_{2,3,4}bits); the L2 prefetch covers the 8K words x 4 n-tiles
  of a k-slice with K 128-byte lines. For K = 2 every expression is the old one.
- The coop (SORT=2) and dataflow kernels are selected for every MCG K (was: K = 2 MCG only).
- Host: K_gate == K_up in {2, 3, 4}, K_down in {2, 3, 4}; K_gu != K_d needs MCG. Mixed K dispatches
  launch_moe_batched<10 * K_gu + K_d, 1> (SORT=0/1 p2b, SORT=2 coop, dataflow, as for uniform K).
- The dataflow kernel runs at 2 blocks/SM (__launch_bounds__(256, 2), <= 128 registers) when BITS != 2:
  at 3 blocks/SM (80 registers) the K = 3 / 4 MMA loops spill (4..12 LDL per iteration); at 2 they are
  spill-free and measured 0.4-10.6 % faster per call (kernel_study/p2b_kbits/ab_lb.py, interleaved,
  bitwise-equal outputs). K = 2 keeps 3 blocks/SM.
Bit-exactness: per (row, expert) every kernel runs the same tile code for the same K, so the
dataflow kernel stays bitwise equal to DSV41_P2B_COOP=1 and, for one-hot routing weights, to p2b.

Plugin side (src/vllm_exl3/exl3.py):
- quantization_config.expert_bits = {"<layer>": [K_gu, K_d]} (ints in {2, 3, 4}); layers without an
  entry keep layer_bits / bits for both groups.
- Exl3MoEMethod allocates w13_trellis with K_gu words and w2_trellis with K_d words; the native
  call passes (K_gu, K_gu, K_d); exl3_moe gets the same triple (its K = 0 instance switches K at
  runtime when they differ); the fat prefill path uses the down tensor's own K for w2 (the
  exl3_fat_gemm kernels are K = 4 only and are used per projection only when that K is 4).
Prefill at K_gu != K_d: exllamav3's exl3_moe has compile-time K only for K_gate == K_up == K_down; its
K = 0 runtime-K instance measured ~2x slower. The patch adds csrc/exl3_moe_kmix.cu to vllm_exl3_c: the same
kernel generated from the installed exllamav3's quant/exl3_moe_kernel.cuh with (K_gu, K_d) as template
arguments (KMIX_KERNEL_EDITS), exl3_moe's host signature, and the plugin calls it for mixed-K layers.
The anchors of the runtime levers that rewrite exl3.py (moe_prep_fused.py, pfg8_loader_reindex.py)
are left verbatim. Idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARK = "// --- widen_p2b_kbits"
MIXED = ((2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3))

HELPERS = r'''// --- widen_p2b_kbits: BITS = K, or 10 * K_gu + K_d when gate/up and down K differ ---
__host__ __device__ constexpr int p2b_kgu(int bits) { return bits >= 10 ? bits / 10 : bits; }
__host__ __device__ constexpr int p2b_kd(int bits) { return bits >= 10 ? bits % 10 : bits; }

// Shared arrays per BITS: BITS = 2 keeps the round-2/3 arrays, whose only user is then still one
// kernel each (<2, 1, 2>, p2b_coop_df_kernel<2, 1>), so NVVM keeps them kernel-local and those
// kernels' shared layout, and with it their machine code, stays canonical-e14's.
template <int BITS>
__device__ __forceinline__ int* p2b_coop_smem_k()
{
    if constexpr (BITS == 2) {
        return p2b_coop_smem();
    } else {
        __shared__ int s[2 * P2B_SORT_CAP + 1];
        return s;
    }
}

template <int BITS>
__device__ __forceinline__ P2bCoopRed* p2b_coop_red_k()
{
    if constexpr (BITS == 2) {
        return p2b_coop_red();
    } else {
        __shared__ P2bCoopRed s[8];
        return s;
    }
}

'''

DF_SMEM = r'''// --- widen_p2b_kbits: dataflow shared arrays per BITS (see p2b_coop_smem_k) ---
template <int BITS>
__device__ __forceinline__ int* p2b_df_sort_smem_k()
{
    if constexpr (BITS == 2) {
        return p2b_df_sort_smem();
    } else {
        __shared__ int s[3 * P2B_SORT_CAP];
        return s;
    }
}

template <int BITS>
__device__ __forceinline__ int* p2b_df_chunk_smem_k()
{
    if constexpr (BITS == 2) {
        return p2b_df_chunk_smem();
    } else {
        __shared__ int s[2 * P2B_SORT_CAP + 1];
        return s;
    }
}

template <int BITS>
__device__ __forceinline__ P2bCoopRed* p2b_df_red_k()
{
    if constexpr (BITS == 2) {
        return p2b_df_red();
    } else {
        __shared__ P2bCoopRed s[8];
        return s;
    }
}

'''

# Builders copied per BITS from the chain's text, reading the per-BITS arrays.
BUILDERS = (
    ("__device__ __forceinline__ void p2b_coop_build(int pairs)\n", "p2b_coop_build", "p2b_coop_build_k",
     (("p2b_coop_smem()", "p2b_coop_smem_k<BITS>()"),)),
    ("__device__ __forceinline__ void p2b_df_build(const int32_t* __restrict__ ids, int pairs)\n", "p2b_df_build",
     "p2b_df_build_k", (("p2b_df_sort_smem()", "p2b_df_sort_smem_k<BITS>()"),
                        ("p2b_df_chunk_smem()", "p2b_df_chunk_smem_k<BITS>()"))),
)

SORT_KERNEL_SMEM = (
    ("p2b_coop_build(m * experts)", "p2b_coop_build_k<BITS>(m * experts)", 1),
    ("p2b_coop_smem()", "p2b_coop_smem_k<BITS>()", 2),
    ("p2b_coop_red()", "p2b_coop_red_k<BITS>()", 2),
)
DF_KERNEL_SMEM = (
    ("p2b_df_build(ids, m * experts)", "p2b_df_build_k<BITS>(ids, m * experts)", 1),
    ("p2b_df_chunk_smem()", "p2b_df_chunk_smem_k<BITS>()", 1),
    ("p2b_df_sort_smem()", "p2b_df_sort_smem_k<BITS>()", 4),
    ("p2b_df_red()", "p2b_df_red_k<BITS>()", 2),
)
HELPERS_ANCHOR = "template <int BITS, int CB, int SORT = 0>\n__global__ __launch_bounds__(256, 4)\nvoid p2b_moe_batched_kernel("

SORT_KERNEL_EDITS = (
    ("                run_gemv_tile_coop<BITS, CB, 1>(B32, is_up ? had_up : had_gate,",
     "                run_gemv_tile_coop<p2b_kgu(BITS), CB, 1>(B32, is_up ? had_up : had_gate,", "coop gate/up tile"),
    ("                run_gemv_tile_coop<BITS, CB, 1>(B32, had_down, down,",
     "                run_gemv_tile_coop<p2b_kd(BITS), CB, 1>(B32, had_down, down,", "coop down tile"),
    ("            run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_gate,",
     "            run_gemv_tile<p2b_kgu(BITS), CB, 1>(B32, A2, C, kslices_gate,", "p2b gate/up tile"),
    ("            run_gemv_tile<BITS, CB, 1>(B32, A2, C, kslices_down,",
     "            run_gemv_tile<p2b_kd(BITS), CB, 1>(B32, A2, C, kslices_down,", "p2b down tile"),
)

TILE_START = "// run_gemv_tile_coop for K=2 MCG, CFG 1"
TILE_END = "// Output (row, 128-block hb): down output Hadamard"

TILE_NEW = r'''// run_gemv_tile_coop for K = bits MCG (widen_p2b_kbits: 2, 3, 4), CFG 1 (8 k-split warps x 4
// n-tiles, register ring 2, fold 2): same loads, decode, MMA, fold and cross-warp reduction order,
// so the same bits. Also prefetches k-slice i + 2 + P2B_DF_PFL2 into L2 at k-slice i (the 4 n-tiles
// of a k-slice are 8 * bits words x 4 = bits 128-byte lines) and loads the A fragment one k-slice ahead.
template <int bits>
__device__ __forceinline__ void p2b_df_tile(
    const uint32_t* __restrict__ B32,
    const half* __restrict__ A,
    half* __restrict__ C,
    int kslices,
    int size_k,
    int group,
    int ntiles,
    int warp,
    int lane,
    P2bCoopRed* sh_red,
    const int* __restrict__ mem,
    int len,
    int m,
    int experts)
{
    constexpr int WK = 8;
    constexpr int WNT = 4;
    constexpr int PF = 2;
    constexpr int FOLD = 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;
    constexpr int PFLINES = bits;  // 128-byte L2 lines per k-slice of the tile
    constexpr int PFSTRIDE = 32;   // words per line

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    const int r0 = lane >> 2;
    const bool r0_ok = r0 < len;
    const half2* A2 = reinterpret_cast<const half2*>(A + p2b_coop_em(mem[r0_ok ? r0 : 0], m, experts) * size_k);
    const half2 hzero = __half2half2(__ushort_as_half(0));

    int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        x_src_b = lane >> 1;
        x_src_a = (x_src_b + 15) & 15;
    } else if constexpr (bits == 3) {
        const int t_offset = lane << 3;
        const int b1 = (t_offset + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
    const uint32_t* lp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane * PFSTRIDE;

    // K = 3: lanes 24..31 hold no trellis word (run_gemv_tile_coop's ld_b).
    auto ld_b = [&] (int i, int l) -> uint32_t {
        if constexpr (bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);
    if (lane < PFLINES)
        for (int d = PF; d < min(PF + P2B_DF_PFL2, myn); ++d)
            p2b_df_prefetch_l2(lp + (size_t) d * slice_stride);

    half2 an0 = hzero, an2 = hzero;
    if (myn > 0) {
        const size_t c0 = (size_t) ks0 * 8 + (lane & 3);
        an0 = A2[c0];
        an2 = A2[c0 + 4];
    }

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }
            if (lane < PFLINES && i + PF + P2B_DF_PFL2 < myn)
                p2b_df_prefetch_l2(lp + (size_t) (i + PF + P2B_DF_PFL2) * slice_stride);

            const half2 a0v = an0;
            const half2 a2v = an2;
            if (i + 1 < myn) {
                const size_t cn = (size_t) (ks0 + i + 1) * 8 + (lane & 3);
                an0 = A2[cn];
                an2 = A2[cn + 4];
            }
            FragB a01, a23;
            a01[0] = r0_ok ? a0v : hzero;
            a23[0] = r0_ok ? a2v : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    bench_fshift::dq8_regs_2bits_fs<1>(awv, bwv, lane << 3, f0, f1);
                } else if constexpr (bits == 4) {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<1>(aw, bw[t], f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<1>(awv, bwv, x_s2, f0, f1);
                }
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Warp reduction
    if (r0_ok) {
        #pragma unroll
        for (int t = 0; t < WNT; ++t) {
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = t * 16 + f * 8 + (lane & 3) * 2;
                sh_red[warp][r0][col + 0] = acc0[t][f].x;
                sh_red[warp][r0][col + 1] = acc0[t][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < COLS * len; idx += THREADS) {
        const int r = idx / COLS;
        const int c = idx % COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WK; ++j)
            sum += sh_red[j][r][c];
        C[p2b_coop_em(mem[r], m, experts) * (size_t) (ntiles * 16) + group * COLS + c] = __float2half_rn(sum);
    }
    __syncthreads();
}

// L2 prefetch of the first n k-slices of warp's K-range of a coop tile (bits lines per slice).
template <int bits>
__device__ __forceinline__ void p2b_df_prefetch_tile(const uint32_t* __restrict__ B32, int kslices, int ntiles,
                                                     int group, int warp, int lane, int n)
{
    const int chunk = CEIL_DIVIDE(kslices, 8);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    if (lane < bits) {
        const size_t stride = (size_t) ntiles * (8 * bits);
        const uint32_t* lp = B32 + (size_t) ks0 * stride + group * (32 * bits) + lane * 32;
        for (int d = 0; d < min(n, myn); ++d)
            p2b_df_prefetch_l2(lp + (size_t) d * stride);
    }
}

'''

DF_EDITS = (
    (
        "__global__ __launch_bounds__(256, 3)\nvoid p2b_coop_df_kernel(",
        "__global__ __launch_bounds__(256, BITS == 2 ? 3 : 2)  // widen_p2b_kbits: K != 2 at 2 blocks/SM\n"
        "void p2b_coop_df_kernel(",
        "dataflow launch bounds",
    ),
    (
        """            p2b_df_prefetch_tile(reinterpret_cast<const uint32_t*>(base), gu ? kslices_gate : kslices_down,
                                 gu ? ntiles_gate : ntiles_down,
                                 gu ? (r >> 2) * 2 + ((r >> 1) & 1) : task - gu_tasks - u * num_groups_down, warp,
                                 lane, 2 + P2B_DF_PFL2);
""",
        """            if constexpr (p2b_kgu(BITS) == p2b_kd(BITS))
                p2b_df_prefetch_tile<p2b_kgu(BITS)>(reinterpret_cast<const uint32_t*>(base), gu ? kslices_gate : kslices_down,
                                     gu ? ntiles_gate : ntiles_down,
                                     gu ? (r >> 2) * 2 + ((r >> 1) & 1) : task - gu_tasks - u * num_groups_down, warp,
                                     lane, 2 + P2B_DF_PFL2);
            else if (gu)
                p2b_df_prefetch_tile<p2b_kgu(BITS)>(reinterpret_cast<const uint32_t*>(base), kslices_gate, ntiles_gate,
                                                    (r >> 2) * 2 + ((r >> 1) & 1), warp, lane, 2 + P2B_DF_PFL2);
            else
                p2b_df_prefetch_tile<p2b_kd(BITS)>(reinterpret_cast<const uint32_t*>(base), kslices_down, ntiles_down,
                                                   task - gu_tasks - u * num_groups_down, warp, lane, 2 + P2B_DF_PFL2);
""",
        "dataflow prologue prefetch",
    ),
    (
        "                p2b_df_tile(reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]),",
        "                p2b_df_tile<p2b_kgu(BITS)>(reinterpret_cast<const uint32_t*>(is_up ? ut_ptrs[src] : gt_ptrs[src]),",
        "dataflow gate/up tile",
    ),
    (
        "            p2b_df_tile(reinterpret_cast<const uint32_t*>(dt_ptrs[ids[mem[0]]]), had_down, down, kslices_down, inter,",
        "            p2b_df_tile<p2b_kd(BITS)>(reinterpret_cast<const uint32_t*>(dt_ptrs[ids[mem[0]]]), had_down, down, kslices_down, inter,",
        "dataflow down tile",
    ),
)

HOST_EDITS = (
    (
        "    if constexpr (BITS == 2 && CB == 1) {\n        if (p2b_coop_enabled() && m * e <= P2B_SORT_CAP)\n",
        "    if constexpr (CB == 1) {  // widen_p2b_kbits: every MCG K (was K = 2 only)\n"
        "        if (p2b_coop_enabled() && m * e <= P2B_SORT_CAP)\n",
        "coop / dataflow selection",
    ),
    (
        '    TORCH_CHECK(kg == ku && ku == kd && (kg == 2 || kg == 3 || kg == 4), "unsupported fused MoE K");',
        "    // widen_p2b_kbits: gate == up K in {2, 3, 4}, down K in {2, 3, 4}; K_gu != K_d needs MCG.\n"
        "    TORCH_CHECK(kg == ku && (kg == 2 || kg == 3 || kg == 4) && (kd == 2 || kd == 3 || kd == 4) && (kd == kg || mcg),\n"
        '                "unsupported fused MoE K");',
        "K check",
    ),
    (
        "    const bool df = mcg && kg == 2 && p2b_coop_mode() == 2",
        "    const bool df = mcg && p2b_coop_mode() == 2",
        "dataflow host condition",
    ),
)

ARGS = ("(x, gt, gu, gv, ut, uu, uv, dt, du, dv, ids, rw, out, gate, up, down, had_gate, had_up, had_down, "
        "accum, e, m, hidden, inter, swiglu_limit);")
DISPATCH_ANCHOR = "    if (mcg && kg == 2) launch_moe_batched<2, 1>" + ARGS + "\n"
DISPATCH_NEW = (
    "    if (kg != kd) {\n"
    "        // widen_p2b_kbits: gate/up K != down K (MCG): BITS = 10 * K_gu + K_d.\n"
    "        switch (10 * kg + kd) {\n"
    + "".join(f"        case {10 * a + b}: launch_moe_batched<{10 * a + b}, 1>{ARGS} break;\n" for a, b in MIXED)
    + "        default: TORCH_CHECK(false, \"unsupported fused MoE K\");\n"
    "        }\n"
    "        return out;\n"
    "    }\n"
)


def _sub1(src: str, old: str, new: str, label: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"widen_p2b_kbits: {label}: expected 1 match, found {src.count(old)}")
    return src.replace(old, new, 1)


def _function(src: str, start: str) -> str:
    """The text of the function whose signature line starts with `start`, through its closing brace."""
    if src.count(start) != 1:
        raise SystemExit(f"widen_p2b_kbits: {start.strip()}: expected 1 match, found {src.count(start)}")
    a = src.index(start)
    return src[a : src.index("\n}\n", a) + 3]


def _builder_k(src: str, start: str, name: str, new_name: str, subs) -> str:
    """`name` copied as `template <int BITS> new_name`, reading the per-BITS shared arrays."""
    body = _function(src, start).replace(f"void {name}(", f"void {new_name}(", 1)
    for old, new in subs:
        if old not in body:
            raise SystemExit(f"widen_p2b_kbits: {name}: {old} not found")
        body = body.replace(old, new)
    return "template <int BITS>\n" + body + "\n"


def _region_subs(src: str, start: str, subs) -> str:
    """Replace each `old` exactly `n` times inside the kernel whose signature starts with `start`."""
    if src.count(start) != 1:
        raise SystemExit(f"widen_p2b_kbits: {start}: expected 1 match, found {src.count(start)}")
    a = src.index(start)
    b = src.index("\n}\n", a)
    region = src[a:b]
    for old, new, n in subs:
        if region.count(old) != n:
            raise SystemExit(f"widen_p2b_kbits: {start} {old}: expected {n}, found {region.count(old)}")
        region = region.replace(old, new)
    return src[:a] + region + src[b:]


def patch_cu(src: str) -> str:
    if MARK in src:
        return src
    if "// --- widen_p2b_dataflow" not in src:
        raise SystemExit("widen_p2b_kbits: apply widen_p2b_dataflow.py first")
    builders = {name: _builder_k(src, start, name, new_name, subs) for start, name, new_name, subs in BUILDERS}
    out = _sub1(src, HELPERS_ANCHOR, HELPERS + builders["p2b_coop_build"] + HELPERS_ANCHOR, "SORT kernel template")
    for old, new, label in SORT_KERNEL_EDITS:
        out = _sub1(out, old, new, label)
    out = _region_subs(out, "void p2b_moe_batched_kernel(", SORT_KERNEL_SMEM)
    if out.count(TILE_START) != 1 or out.count(TILE_END) != 1:
        raise SystemExit("widen_p2b_kbits: dataflow tile block not found")
    a, b = out.index(TILE_START), out.index(TILE_END)
    out = out[:a] + DF_SMEM + builders["p2b_df_build"] + TILE_NEW + out[b:]
    out = _region_subs(out, "void p2b_coop_df_kernel(", DF_KERNEL_SMEM)
    for old, new, label in DF_EDITS + HOST_EDITS:
        out = _sub1(out, old, new, label)
    out = _sub1(out, DISPATCH_ANCHOR, DISPATCH_NEW + DISPATCH_ANCHOR, "launch dispatch")
    if out.count("grid.sync()") != src.count("grid.sync()"):
        raise SystemExit("widen_p2b_kbits: barrier count changed")
    return out


# ----------------------------------------------------------------------------- prefill: exl3_moe at mixed K
# exllamav3's exl3_moe takes one K per projection at compile time only when K_gate == K_up == K_down; any
# other triple runs its K = 0 instance, which switches over all of K = 1..8 per GEMM inside the kernel and
# measured ~2x slower (kernel_study/p2b_kbits/prefill_kbits.py). vllm_exl3_c gets the same kernel with the
# gate/up and down K as template arguments (exl3_moe_kmix_kernel<K_gu, K_d, N, cb = 1>, the 6 mixed pairs
# x N 128 / 256), generated at build time from the installed exllamav3's quant/exl3_moe_kernel.cuh by the
# KMIX_KERNEL_EDITS below and nothing else, plus a host entry point with exl3_moe's signature and its own
# zeroed, self-resetting lock buffer.

KMIX_KERNEL_EDITS = (
    ("template<int t_bits, int MOE_TILESIZE_N, int cb>\n", "template<int t_bits_gu, int t_bits_d, int MOE_TILESIZE_N, int cb>\n", 1),
    ("void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)\n", "void exl3_moe_kmix_kernel(EXL3_MOE_KERNEL_ARGS)\n", 1),
    ('#include "../util.h"\n', '#include "util.h"\n', 1),
    ('#include "../util.cuh"\n', '#include "util.cuh"\n', 1),
    ('#include "../ptx.cuh"\n', '#include "ptx.cuh"\n', 1),
)
KMIX_GEMM_OLD = ("                if constexpr (t_bits)\n"
                 "                    exl3_gemm_kernel_inner<t_bits, false, cb, SHAPE_ARGS, false>(ARGS);\n")


def kmix_kernel(src: str) -> str:
    """exllamav3 quant/exl3_moe_kernel.cuh -> exl3_moe_kmix_kernel<t_bits_gu, t_bits_d, N, cb>."""
    out = src
    for old, new, n in KMIX_KERNEL_EDITS:
        if out.count(old) != n:
            raise SystemExit(f"widen_p2b_kbits: exl3_moe_kernel.cuh {old.strip()}: expected {n}, found {out.count(old)}")
        out = out.replace(old, new)
    if out.count(KMIX_GEMM_OLD) != 2 or out.index("auto gemm_up") > out.index(KMIX_GEMM_OLD) \
            or out.index("auto gemm_down") > out.rindex(KMIX_GEMM_OLD):
        raise SystemExit("widen_p2b_kbits: exl3_moe_kernel.cuh gate/up and down GEMM selection not found")
    a = out.index(KMIX_GEMM_OLD)
    out = out[:a] + KMIX_GEMM_OLD.replace("t_bits", "t_bits_gu") + out[a + len(KMIX_GEMM_OLD):]
    b = out.index(KMIX_GEMM_OLD)
    out = out[:b] + KMIX_GEMM_OLD.replace("t_bits", "t_bits_d") + out[b + len(KMIX_GEMM_OLD):]
    if "t_bits)" in out or "<t_bits," in out:
        raise SystemExit("widen_p2b_kbits: exl3_moe_kernel.cuh: stray t_bits")
    return "// Generated by docker/patch/widen_p2b_kbits.py from exllamav3 quant/exl3_moe_kernel.cuh.\n" + out


KMIX_CU = r"""// --- widen_p2b_kbits: exl3_moe with compile-time gate/up K != down K (MCG) ---
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <mutex>
#include "exl3_moe_kmix_kernel.cuh"

typedef void (*fp_exl3_moe_kmix) (EXL3_MOE_KERNEL_ARGS);

// Same size and zero initialisation as exllamav3's DevCtx lock buffer; the kernel's barriers and scheduler
// reset themselves at the end of every launch.
static int* exl3_moe_kmix_locks(int device)
{
    static std::mutex mtx;
    static int* locks[MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks[device]) {
        const size_t size = (MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS) * sizeof(int);
        cuda_check(cudaMalloc(&locks[device], size));
        cuda_check(cudaMemset(locks[device], 0, size));
    }
    return locks[device];
}

static fp_exl3_moe_kmix exl3_moe_kmix_select(int kgu, int kd, int n256)
{
    #define KMIX(A, B) if (kgu == A && kd == B) return n256 ? exl3_moe_kmix_kernel<A, B, 256, 1> : exl3_moe_kmix_kernel<A, B, 128, 1>
    KMIX(2, 3); KMIX(2, 4); KMIX(3, 2); KMIX(3, 4); KMIX(4, 2); KMIX(4, 3);
    #undef KMIX
    return nullptr;
}

// exllamav3 exl3_moe's signature and launch rule (vllm_exl3.exl3.apply_exl3_fused_moe calls either).
void exl3_moe_kmix(
    const at::Tensor& hidden_state, const at::Tensor& output_state, const at::Tensor& expert_count,
    const at::Tensor& token_sorted, const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g, const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g, const at::Tensor& temp_intermediate_u,
    int64_t act_function, int64_t K_gate, int64_t K_up, int64_t K_down,
    const at::Tensor& gate_ptrs_trellis, const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis, const at::Tensor& up_ptrs_suh, const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis, const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,
    bool gate_mcg, bool gate_mul1, bool up_mcg, bool up_mul1, bool down_mcg, bool down_mul1,
    double act_limit_d, int64_t num_active)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (num_active == 0) return;

    TORCH_CHECK(gate_mcg && up_mcg && down_mcg && !gate_mul1 && !up_mul1 && !down_mul1,
                "exl3_moe_kmix: MCG codebook only");
    TORCH_CHECK(K_gate == K_up, "exl3_moe_kmix: gate and up K must match");
    fp_exl3_moe_kmix kernel = nullptr;
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DIM(hidden_state, 2);
    int bsz = hidden_state.size(0);
    int hidden_dim = hidden_state.size(1);
    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);
    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    int num_experts = expert_count.size(0) - 1;
    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    int num_experts_per_tok = token_sorted.size(0) / bsz;
    TORCH_CHECK_DTYPE(temp_state_g, kHalf);
    TORCH_CHECK_DIM(temp_state_g, 3);
    TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
    TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
    int max_tokens_per_expert = temp_state_g.size(1);
    int concurrency = temp_state_g.size(0);
    TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf);
    TORCH_CHECK_DIM(temp_intermediate_g, 3);
    TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
    TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
    int intermediate_dim = temp_intermediate_g.size(2);
    for (const auto* t : {&gate_ptrs_suh, &gate_ptrs_svh, &up_ptrs_trellis, &up_ptrs_suh, &up_ptrs_svh,
                          &down_ptrs_trellis, &down_ptrs_suh, &down_ptrs_svh})
        TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, (*t));
    TORCH_CHECK(gate_ptrs_trellis.dim() == 1 && gate_ptrs_trellis.size(0) == num_experts,
                "exl3_moe_kmix: number of gate tensors doesn't match num_experts");

    int device;
    cudaGetDevice(&device);
    int num_sms = 0;
    cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    int* locks = exl3_moe_kmix_locks(device);

    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= num_sms, "exl3_moe_kmix: concurrency too high for device num_sms");
    int num_groups = MIN(concurrency, MOE_MAX_GROUPS);
    int group_size = MOE_SMS_PER_EXPERT;
    if (num_active > 0) {
        num_groups = MIN(num_groups, (int) num_active);
        group_size = MIN(num_sms / num_groups, MOE_MAX_SMS_PER_EXPERT);
    }
    dim3 grid_dim(group_size, 1, num_groups);
    const int n256 = hidden_dim % 256 == 0 && intermediate_dim % 256 == 0;
    kernel = exl3_moe_kmix_select((int) K_gate, (int) K_down, n256);
    TORCH_CHECK(kernel != nullptr, "exl3_moe_kmix: unsupported K pair (gate/up K != down K, each in 2..4)");

    static std::mutex attr_mtx;
    {
        std::lock_guard<std::mutex> lock(attr_mtx);
        cuda_check(cudaFuncSetAttribute((void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
    }

    const void* _hidden_state = hidden_state.data_ptr();
    void* _temp_state_g = temp_state_g.data_ptr();
    void* _temp_state_u = temp_state_u.data_ptr();
    void* _temp_intermediate_g = temp_intermediate_g.data_ptr();
    void* _temp_intermediate_u = temp_intermediate_u.data_ptr();
    void* _output_state = output_state.data_ptr();
    void* _gate_ptrs_trellis = gate_ptrs_trellis.data_ptr();
    void* _gate_ptrs_suh = gate_ptrs_suh.data_ptr();
    void* _gate_ptrs_svh = gate_ptrs_svh.data_ptr();
    void* _up_ptrs_trellis = up_ptrs_trellis.data_ptr();
    void* _up_ptrs_suh = up_ptrs_suh.data_ptr();
    void* _up_ptrs_svh = up_ptrs_svh.data_ptr();
    void* _down_ptrs_trellis = down_ptrs_trellis.data_ptr();
    void* _down_ptrs_suh = down_ptrs_suh.data_ptr();
    void* _down_ptrs_svh = down_ptrs_svh.data_ptr();
    void* _expert_count = expert_count.data_ptr();
    void* _token_sorted = token_sorted.data_ptr();
    void* _weight_sorted = weight_sorted.data_ptr();
    float act_limit = (float) act_limit_d;
    int act = (int) act_function, kg = (int) K_gate, ku = (int) K_up, kd = (int) K_down;
    void* kernelArgs[] = {
        &_hidden_state, &_temp_state_g, &_temp_state_u, &_temp_intermediate_g, &_temp_intermediate_u, &_output_state,
        &_gate_ptrs_trellis, &_gate_ptrs_suh, &_gate_ptrs_svh, &_up_ptrs_trellis, &_up_ptrs_suh, &_up_ptrs_svh,
        &_down_ptrs_trellis, &_down_ptrs_suh, &_down_ptrs_svh, &_expert_count, &_token_sorted, &_weight_sorted,
        (void*) &hidden_dim, (void*) &intermediate_dim, (void*) &num_experts, (void*) &num_experts_per_tok,
        (void*) &max_tokens_per_expert, (void*) &num_groups, (void*) &act_limit, (void*) &act,
        (void*) &kg, (void*) &ku, (void*) &kd, (void*) &locks
    };
    cuda_check(cudaLaunchKernel((void*) kernel, grid_dim, block_dim, kernelArgs, SMEM_MAX, stream));
    cuda_check(cudaPeekAtLastError());
}
"""

KMIX_SETUP_OLD = '                    "csrc/exl3_fat_gemm.cu",\n'
KMIX_SETUP_NEW = KMIX_SETUP_OLD + '                    "csrc/exl3_moe_kmix.cu",  # widen_p2b_kbits\n'
KMIX_BIND_DECL_OLD = '#include "exl3_fat_gemm.cuh"\n'
KMIX_BIND_DECL_NEW = KMIX_BIND_DECL_OLD + """
// widen_p2b_kbits: csrc/exl3_moe_kmix.cu
void exl3_moe_kmix(
    const at::Tensor& hidden_state, const at::Tensor& output_state, const at::Tensor& expert_count,
    const at::Tensor& token_sorted, const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g, const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g, const at::Tensor& temp_intermediate_u,
    int64_t act_function, int64_t K_gate, int64_t K_up, int64_t K_down,
    const at::Tensor& gate_ptrs_trellis, const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis, const at::Tensor& up_ptrs_suh, const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis, const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,
    bool gate_mcg, bool gate_mul1, bool up_mcg, bool up_mul1, bool down_mcg, bool down_mul1,
    double act_limit, int64_t num_active);
"""
KMIX_BIND_DEF_OLD = '    m.attr("P2B_MOE_ABI_VERSION") = 2;\n'
KMIX_BIND_DEF_NEW = KMIX_BIND_DEF_OLD + (
    '    m.def("exl3_moe_kmix", &exl3_moe_kmix,\n'
    '          "exl3_moe with compile-time gate/up K != down K (MCG); exllamav3 exl3_moe arguments, num_active last");\n')


def exllamav3_kernel_header() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("widen_p2b_kbits: exllamav3 not installed (needed for exl3_moe_kernel.cuh)")
    return Path(list(spec.submodule_search_locations)[0]) / "exllamav3_ext" / "quant" / "exl3_moe_kernel.cuh"


def patch_setup(src: str) -> str:
    if "csrc/exl3_moe_kmix.cu" in src:
        return src
    return _sub1(src, KMIX_SETUP_OLD, KMIX_SETUP_NEW, "setup.py sources (apply fix_vllm_exl3_setup.py first)")


def patch_bindings(src: str) -> str:
    if '"exl3_moe_kmix"' in src:
        return src
    out = _sub1(src, KMIX_BIND_DECL_OLD, KMIX_BIND_DECL_NEW, "bindings includes")
    return _sub1(out, KMIX_BIND_DEF_OLD, KMIX_BIND_DEF_NEW, "bindings module")


# ----------------------------------------------------------------------------- plugin (exl3.py)

PY_EDITS = (
    (
        """        for layer_idx, layer_k in self.layer_bits.items():
            if layer_k not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported EXL3 bits={layer_k} for layer {layer_idx}"
                )
""",
        """        for layer_idx, layer_k in self.layer_bits.items():
            if layer_k not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported EXL3 bits={layer_k} for layer {layer_idx}"
                )
        # widen_p2b_kbits: routed-expert K per projection group, {"<layer>": [K_gu, K_d]}: K_gu for
        # gate/up (w1, w3), K_d for down (w2). Layers absent from the map use layer_bits / bits.
        raw_expert_bits = kwargs.pop("expert_bits", None) or {}
        self.expert_bits: dict[int, tuple[int, int]] = {}
        for layer_idx, pair in dict(raw_expert_bits).items():
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError(f"expert_bits[{layer_idx}] must be [K_gu, K_d], got {pair!r}")
            k_gu, k_d = (int(v) for v in pair)
            if k_gu not in (2, 3, 4) or k_d not in (2, 3, 4):
                raise ValueError(
                    f"unsupported EXL3 expert_bits={[k_gu, k_d]} for layer {layer_idx}"
                )
            self.expert_bits[int(layer_idx)] = (k_gu, k_d)
""",
        "Exl3Config expert_bits",
    ),
    (
        """        return self.layer_bits.get(int(m.group(1)), self.bits)

""",
        """        return self.layer_bits.get(int(m.group(1)), self.bits)

    def expert_bits_for_prefix(self, prefix: str) -> tuple[int, int]:
        \"\"\"Routed-expert (K_gu, K_d): `expert_bits` entry for this layer, else bits_for_prefix twice.\"\"\"
        m = self._LAYER_RE.search(prefix or "")
        if m is not None and int(m.group(1)) in self.expert_bits:
            return self.expert_bits[int(m.group(1))]
        k = self.bits_for_prefix(prefix)
        return k, k

""",
        "expert_bits_for_prefix",
    ),
    (
        """            return Exl3MoEMethod(
                layer.moe_config, self, bits=self.bits_for_prefix(prefix)
            )
""",
        """            k_gu, k_d = self.expert_bits_for_prefix(prefix)  # widen_p2b_kbits
            return Exl3MoEMethod(layer.moe_config, self, bits=k_gu, bits_d=k_d)
""",
        "get_quant_method MoE K",
    ),
    (
        """        self, moe, quant_config: Exl3Config, bits: int | None = None
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        # One method instance per RoutedExperts layer, so this is per-layer K.
        self.bits = int(bits) if bits is not None else quant_config.bits
""",
        """        self, moe, quant_config: Exl3Config, bits: int | None = None, bits_d: int | None = None
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        # One method instance per RoutedExperts layer, so this is per-layer K.
        self.bits = int(bits) if bits is not None else quant_config.bits
        # widen_p2b_kbits: down (w2) K; self.bits is the gate/up (w13) K.
        self.bits_d = int(bits_d) if bits_d is not None else self.bits
""",
        "Exl3MoEMethod K_d",
    ),
    (
        """        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),
""",
        """        k_words = self.bits_d * 16  # widen_p2b_kbits: w2 holds K_d words per tile row
        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),
""",
        "w2 trellis K_d",
    ),
    (
        """        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits
""",
        """        layer._exl3_k_words = self.bits * 16
        layer._exl3_bits = self.bits
        layer._exl3_k_words_d = self.bits_d * 16  # widen_p2b_kbits
        layer._exl3_bits_d = self.bits_d
""",
        "layer K attributes",
    ),
    (
        """        layer._exl3_k = int(layer._exl3_bits)
        return
""",
        """        layer._exl3_k = int(layer._exl3_bits)
        layer._exl3_k_d = int(getattr(layer, "_exl3_bits_d", layer._exl3_bits))  # widen_p2b_kbits
        return
""",
        "fused state K (native only)",
    ),
    (
        """    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)
""",
        """    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)
    layer._exl3_k_d = int(getattr(layer, "_exl3_bits_d", layer._exl3_bits))  # widen_p2b_kbits
""",
        "fused state K",
    ),
    (
        """    bits = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", -1)))
    if not (
""",
        """    bits = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", -1)))
    bits_d = int(getattr(layer, "_exl3_k_d", bits))  # widen_p2b_kbits
    if not (
""",
        "native dims K_d",
    ),
    (
        """        and bits in (2, 3, 4)
        and len(inners) > 0
""",
        """        and bits in (2, 3, 4)
        and bits_d in (2, 3, 4)
        and len(inners) > 0
""",
        "native dims K_d check",
    ),
    (
        """    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))
    fn = module.p2b_fused_moe
    flags = getattr(layer, "_exl3_codebook_flags", (True, False, True, False, True, False))
    mcg = bool(flags[0])
    mul1 = bool(flags[1])
    if mcg == mul1:
        return None
""",
        """    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))
    k_d = int(getattr(layer, "_exl3_k_d", k))  # widen_p2b_kbits: down K
    fn = module.p2b_fused_moe
    flags = getattr(layer, "_exl3_codebook_flags", (True, False, True, False, True, False))
    mcg = bool(flags[0])
    mul1 = bool(flags[1])
    if mcg == mul1:
        return None
    if k_d != k and not mcg:
        return None  # mixed gate/up and down K: MCG only in p2b
""",
        "native call K_d",
    ),
    (
        """        k,
        k,
        k,
        mcg,
        *extra_args,""",
        """        k,
        k,
        k_d,
        mcg,
        *extra_args,""",
        "native call K triple",
    ),
    (
        """    k = int(getattr(layer, "_exl3_k", 4))
    args = (
""",
        """    k = int(getattr(layer, "_exl3_k", 4))
    k_d = int(getattr(layer, "_exl3_k_d", k))  # widen_p2b_kbits
    if k_d != k and bool(getattr(layer, "_exl3_codebook_flags", (True,))[0]):
        # exl3_moe's K = 0 instance switches K at runtime (~2x slower); vllm_exl3_c carries the same
        # kernel with compile-time (K_gu, K_d), exl3_moe's arguments.
        fn = getattr(_load_native_exl3_ext(), "exl3_moe_kmix", None) or fn
    args = (
""",
        "exl3_moe K_d",
    ),
    (
        """        MOE_ACT_SILU,
        k,
        k,
        k,
""",
        """        MOE_ACT_SILU,
        k,
        k,
        k_d,
""",
        "exl3_moe K triple",
    ),
    (
        """        k = int(getattr(gate, "K", 4))
        mcg = bool(getattr(gate, "mcg", True))
""",
        """        k = int(getattr(gate, "K", 4))
        k_d = int(getattr(down, "K", k))  # widen_p2b_kbits: w2 has its own K
        mcg = bool(getattr(gate, "mcg", True))
""",
        "fat path K_d",
    ),
    (
        """            and use_kernel
            and k == 4
            and mcg
            and not mul1
            and native_c is not None
            and hasattr(native_c, "exl3_fat_gemm_scatter")
        ):
            native_c.exl3_fat_gemm_scatter(
                h2,
                down.trellis,
                out,
                down.svh,
                token_idx,
                weight_sorted[start:offset],
                k,
                mcg,
                mul1,
            )
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, k, mcg, mul1)
""",
        """            and use_kernel
            and k_d == 4
            and mcg
            and not mul1
            and native_c is not None
            and hasattr(native_c, "exl3_fat_gemm_scatter")
        ):
            native_c.exl3_fat_gemm_scatter(
                h2,
                down.trellis,
                out,
                down.svh,
                token_idx,
                weight_sorted[start:offset],
                k_d,
                mcg,
                mul1,
            )
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, k_d, mcg, mul1)
""",
        "fat path down K",
    ),
)


def patch_py(src: str) -> str:
    if "raw_expert_bits = kwargs.pop(\"expert_bits\"" in src:
        return src
    out = src
    for old, new, label in PY_EDITS:
        out = _sub1(out, old, new, label)
    return out


def apply(root: Path) -> None:
    cu = root / "csrc" / "p2b_moe.cu"
    cu.write_text(patch_cu(cu.read_text()))
    print(f"patched {cu}")
    hdr = exllamav3_kernel_header()
    (root / "csrc" / "exl3_moe_kmix_kernel.cuh").write_text(kmix_kernel(hdr.read_text()))
    (root / "csrc" / "exl3_moe_kmix.cu").write_text(KMIX_CU)
    print(f"wrote {root / 'csrc' / 'exl3_moe_kmix.cu'} (kernel from {hdr})")
    for path, fn in ((root / "setup.py", patch_setup), (root / "csrc" / "bindings.cpp", patch_bindings)):
        path.write_text(fn(path.read_text()))
        print(f"patched {path}")
    py = root / "src" / "vllm_exl3" / "exl3.py"
    py.write_text(patch_py(py.read_text()))
    print(f"patched {py}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    apply(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
