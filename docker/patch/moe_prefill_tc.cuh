// dsv41 moe_prefill_tc: routed-expert MoE prefill on tensor cores, EXL3 MCG trellis decoded once per
// (expert, row chunk, column tile) and reused for every routed row of the chunk (DSV41_MOE_PREFILL_TC=1).
//
// Replaces, for prefill-sized calls, vllm_exl3's apply_exl3_fused_moe: exllamav3 exl3_moe (decodes every
// 16x16 trellis tile once per 16 routed rows) plus the fat path (reconstructs fat experts to fp16, cuBLAS).
//
// Two persistent kernels after a one-block work-list kernel, no host sync:
//   gu_kernel<KG>  unit = (expert, <= 64 sorted rows, 256 or 128 intermediate columns of gate AND up)
//                  K loop over the hidden dim in 128-blocks. The input rotation of block kb + 1
//                  (fp16(x) * suh, 128-point Hadamard in fp32, / sqrt(128), fp16) for gate and up is
//                  computed by the warps between the MMAs of block kb into the other A buffer, so the
//                  rotation work hides under the tensor work. Each warp decodes its 16x16 gate and up
//                  tiles once per k16 step (ExLlamaV3's dq_dispatch: the same fp16 weights as every
//                  other EXL3 path) and feeds them to all row blocks of the chunk (mma.sync m16n8k16,
//                  fp32 accumulate). Epilogue per 128-column block: output Hadamard (fp32) * svh,
//                  swiglu clamp, silu(g) * u in fp32 (the fat path's math), fp16, * down suh,
//                  Hadamard -> A2 [sorted row, intermediate] fp16.
//   dn_kernel<KD>  unit = (expert, <= 64 sorted rows, 128 hidden columns), 2 CTAs per SM (<= 128 registers:
//                  the second CTA's MMAs cover one CTA's prologue, barriers and epilogue); K = intermediate;
//                  epilogue: output Hadamard (fp32) * svh * routing weight, fp32 atomic add into out.
// Per output element the K sum runs in ascending k16 order inside one CTA (no split-K).
//
// Device code only (no torch): kernel_study/moe_prefill/host_compile.sh compiles it standalone on the
// host for ptxas resource checks. Build against ExLlamaV3's extension headers (MIT, Turboderp):
// include dirs <exllamav3>/exllamav3_ext and <exllamav3>/exllamav3_ext/quant.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "util.cuh"            // half4, half2_uint32, half_uint16
#include "ptx.cuh"             // FragA/B/C, ptx_mma_m16n8k16, ldsm4, cp_async*
#include "exl3_dq.cuh"         // dq_dispatch<K, cb>: ExLlamaV3's trellis decode in MMA B-fragment layout

namespace dsv41_tc {

constexpr int kThreads = 256;          // 8 warps
constexpr int kMt1 = 64;               // gate/up rows per unit
constexpr int kMt2 = 64;               // down rows per unit (2 CTAs per SM)
constexpr int kRowsW2 = kMt2 / 8;      // down epilogue rows per warp
constexpr int kKb = 128;               // k-block: one Hadamard block of the input dim
constexpr int kCpitch = 136;           // fp32 epilogue tile row pitch (floats; conflict-free float2 stores)
constexpr int kCbMcg = 1;              // exl3_dq codebook index for MCG
constexpr float kRScale = 0.088388347648f;  // 1 / sqrt(128), ExLlamaV3's constant

// ctrl[]: written by worklist_kernel, read by the two persistent kernels.
constexpr int kCtrlItems1 = 0;
constexpr int kCtrlItems2 = 1;
constexpr int kCtrlTicket1 = 2;
constexpr int kCtrlTicket2 = 3;
constexpr int kCtrlInts = 4;

// Shared memory (bytes). gu: A [2 buf][2 proj][64 rows][256 B] + B [2 stages][2 mat][4 k16][16 tiles][32 K]
// + s_tok + s_unit. dn: A2 [2 stages][128 rows][256 B] + B [2 stages][8 k16][8 tiles][32 K] + s_tok + s_w + s_unit.
constexpr int kGuA = 2 * 2 * kMt1 * 256;
constexpr int kDnA = 2 * kMt2 * 256;
__host__ __device__ constexpr int gu_stage_bytes(int kg) { return 2 * 4 * 16 * 32 * kg; }
__host__ __device__ constexpr int dn_stage_bytes(int kd) { return 8 * 8 * 32 * kd; }
__host__ __device__ constexpr int gu_smem_bytes(int kg) { return kGuA + 2 * gu_stage_bytes(kg) + kMt1 * 4 + 16; }
__host__ __device__ constexpr int dn_smem_bytes(int kd) { return kDnA + 2 * dn_stage_bytes(kd) + kMt2 * 4 + kMt2 * 2 + 16; }
static_assert(2 * kMt1 * kCpitch * 4 <= kGuA + 2 * gu_stage_bytes(2), "gu epilogue tile must fit below s_tok");
static_assert(kMt2 * kCpitch * 4 <= kDnA + 2 * dn_stage_bytes(2), "dn epilogue tile must fit below s_tok");
static_assert(gu_smem_bytes(4) <= 101376 && dn_smem_bytes(4) <= 101376, "sm_12x: 99 KiB dynamic smem per block");

// Per-expert pointer tables (vllm_exl3's layer._exl3_ptrs, int64 data_ptr per local expert).
struct Tables {
    const uint16_t* const* gate_trellis;
    const half* const* gate_suh;
    const half* const* gate_svh;
    const uint16_t* const* up_trellis;
    const half* const* up_suh;
    const half* const* up_svh;
    const uint16_t* const* down_trellis;
    const half* const* down_suh;
    const half* const* down_svh;
};

// Row-major [rows][128 halves] tile, 16-byte chunks XOR-swizzled by row so ldmatrix is conflict-free.
__device__ __forceinline__ int a_off(int r, int chunk) { return r * 256 + ((chunk ^ (r & 7)) << 4); }

// The 5 cross-lane butterfly stages of ExLlamaV3's shuffle_had_f4x32 / shuffle_had_f2x32 (lane l holds
// elements 4l..4l+3). There a lane whose stage bit is set flips the sign bit of its own value and adds
// the partner's: round(-v + p). fmaf(-1, v, p) is that same single rounding (the product is exact), so
// one FFMA replaces the LOP3 + FADD pair and the results are bit-identical, NaN and signed zero included.
__device__ __forceinline__ void had_x32(float& h0, float& h1, float& h2, float& h3, int lane)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        const float p0 = __shfl_xor_sync(0xffffffffu, h0, i);
        const float p1 = __shfl_xor_sync(0xffffffffu, h1, i);
        const float p2 = __shfl_xor_sync(0xffffffffu, h2, i);
        const float p3 = __shfl_xor_sync(0xffffffffu, h3, i);
        const float sg = (lane & i) ? -1.0f : 1.0f;
        h0 = fmaf(sg, h0, p0);
        h1 = fmaf(sg, h1, p1);
        h2 = fmaf(sg, h2, p2);
        h3 = fmaf(sg, h3, p3);
    }
}

// fp16 in -> * scale (fp16) -> 128-point Hadamard (fp32) -> / sqrt(128) -> fp16: ExLlamaV3's
// had_hf_r_128_inner<pre_scale = true, post_scale = false>, op for op (had_x32 above). Lane l holds
// elements 4l..4l+3.
__device__ __forceinline__ uint2 had_hf_pre(half2 x01, half2 x23, half4 s, int lane)
{
    half2 p0 = __hmul2(x01, s.x);
    half2 p1 = __hmul2(x23, s.y);
    float v0 = __low2float(p0);
    float v1 = __high2float(p0);
    float v2 = __low2float(p1);
    float v3 = __high2float(p1);
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    had_x32(h0, h1, h2, h3, lane);
    half2_uint32 o0(__floats2half2_rn(h0 * kRScale, h1 * kRScale));
    half2_uint32 o1(__floats2half2_rn(h2 * kRScale, h3 * kRScale));
    return make_uint2(o0.as_uint32, o1.as_uint32);
}

// fp32 in -> 128-point Hadamard -> / sqrt(128) -> * scale (fp16 -> fp32): ExLlamaV3's
// had_ff_r_128_inner<pre_scale = false, post_scale = true> (the fat path's output rotation), op for op.
__device__ __forceinline__ float4 had_ff_post(float4 v, const half* __restrict__ scale4, int lane)
{
    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;
    had_x32(v.x, v.y, v.z, v.w, lane);
    v.x *= kRScale;
    v.y *= kRScale;
    v.z *= kRScale;
    v.w *= kRScale;
    uint2 raw = __ldg(reinterpret_cast<const uint2*>(scale4));
    half2_uint32 sa(raw.x), sb(raw.y);
    v.x *= __low2float(sa.as_half2);
    v.y *= __high2float(sa.as_half2);
    v.z *= __low2float(sb.as_half2);
    v.w *= __high2float(sb.as_half2);
    return v;
}

__device__ __forceinline__ half4 ld_half4(const half* __restrict__ p)
{
    uint2 raw = __ldg(reinterpret_cast<const uint2*>(p));
    half2_uint32 a(raw.x), b(raw.y);
    return half4(a.as_half2, b.as_half2);
}

// torch.clamp semantics (NaN propagates), as the fat path's clamp_ calls.
__device__ __forceinline__ float clamp_hi(float v, float hi) { return isnan(v) ? v : fminf(v, hi); }
__device__ __forceinline__ float clamp_lohi(float v, float lo, float hi) { return isnan(v) ? v : fminf(fmaxf(v, lo), hi); }

// The fat path: act = sigmoid(g) * g * u in fp32 (torch.sigmoid = 1 / (1 + exp(-g))), act.half().
__device__ __forceinline__ float silu_mul(float g, float u, float limit)
{
    if (limit > 0.0f) {
        g = clamp_hi(g, limit);
        u = clamp_lohi(u, -limit, limit);
    }
    float sg = 1.0f / (1.0f + expf(-g));
    float a = sg * g;
    return a * u;
}

__device__ __forceinline__ void atomic_add4(float* p, float4 v)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    atomicAdd(reinterpret_cast<float4*>(p), v);
#else
    atomicAdd(p + 0, v.x);
    atomicAdd(p + 1, v.y);
    atomicAdd(p + 2, v.z);
    atomicAdd(p + 3, v.w);
#endif
}

template <bool BF16>
__device__ __forceinline__ void to_half(uint2 raw, half2& x01, half2& x23)
{
    if constexpr (BF16) {
        // torch's bf16 -> fp16 (x.half()): exact widen to fp32, round to nearest fp16.
        __nv_bfloat162 b01 = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
        __nv_bfloat162 b23 = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
        float2 f01 = __bfloat1622float2(b01);
        float2 f23 = __bfloat1622float2(b23);
        x01 = __floats2half2_rn(f01.x, f01.y);
        x23 = __floats2half2_rn(f23.x, f23.y);
    } else {
        half2_uint32 a(raw.x), b(raw.y);
        x01 = a.as_half2;
        x23 = b.as_half2;
    }
}

// ----------------------------------------------------------------------------------------------- work list
// One block of 1024 threads, n_exp <= 1024. counts[e] = routed rows of local expert e (counts[n_exp] = the
// invalid-id sentinel, excluded). Rows of expert e sit at sorted positions [off_e, off_e + n_e) (stable
// argsort of the local ids). Writes items (expert, first sorted row, rows, 0): kMt1-row chunks for the
// gate/up kernel, kMt2-row chunks for the down kernel, the item counts and zeroed tickets.
__global__ void __launch_bounds__(1024) worklist_kernel(const int64_t* __restrict__ counts, int n_exp,
                                                        int4* __restrict__ items1, int4* __restrict__ items2,
                                                        int* __restrict__ ctrl)
{
    __shared__ int s_warp[3][32];
    const int e = threadIdx.x;
    const int lane = e & 31;
    const int warp = e >> 5;
    const int n = e < n_exp ? (int) counts[e] : 0;
    int v[3] = {n, (n + kMt1 - 1) / kMt1, (n + kMt2 - 1) / kMt2};
    int inc[3] = {v[0], v[1], v[2]};
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            int o = __shfl_up_sync(0xffffffffu, inc[j], d);
            if (lane >= d) inc[j] += o;
        }
    }
    if (lane == 31) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) s_warp[j][warp] = inc[j];
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            int t = lane < nw ? s_warp[j][lane] : 0;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int o = __shfl_up_sync(0xffffffffu, t, d);
                if (lane >= d) t += o;
            }
            s_warp[j][lane] = t;  // inclusive scan of warp totals
        }
    }
    __syncthreads();
    int ex[3];
    #pragma unroll
    for (int j = 0; j < 3; ++j) ex[j] = inc[j] - v[j] + (warp ? s_warp[j][warp - 1] : 0);
    for (int c = 0; c < v[1]; ++c) items1[ex[1] + c] = make_int4(e, ex[0] + c * kMt1, min(kMt1, n - c * kMt1), 0);
    for (int c = 0; c < v[2]; ++c) items2[ex[2] + c] = make_int4(e, ex[0] + c * kMt2, min(kMt2, n - c * kMt2), 0);
    if (e == blockDim.x - 1) {
        ctrl[kCtrlItems1] = ex[1] + v[1];
        ctrl[kCtrlItems2] = ex[2] + v[2];
        ctrl[kCtrlTicket1] = 0;
        ctrl[kCtrlTicket2] = 0;
    }
}

// ----------------------------------------------------------------------------------------------- gate/up
// One unit: rows [row0, row0 + nrows) of the sorted list (expert e), intermediate columns
// [col0, col0 + 128 * NB) of gate and of up. Warp w owns 16-column strips w and w + 8 (NB = 2).
template <int KG, int NB, bool BF16>
__device__ __forceinline__ void gu_unit(const void* __restrict__ x, int hidden, int inter, const Tables& tb,
                                        half* __restrict__ a2, float limit, char* smem, const int* s_tok,
                                        int e, int row0, int nrows, int col0)
{
    constexpr int kTileB = 32 * KG;              // bytes of one 16x16 trellis tile
    constexpr int kRowB = 16 * kTileB;           // smem bytes per (matrix, k16) row of tiles
    constexpr int kStageB = 8 * kRowB;           // 2 matrices x 4 k16 rows
    constexpr int kTilesN = 8 * NB;              // tiles loaded per (matrix, k16) row
    constexpr int kChunksRow = kTilesN * kTileB / 16;
    constexpr int kChunks = 8 * kChunksRow;
    static_assert(kStageB == gu_stage_bytes(KG), "stage layout");

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int kblocks = hidden / kKb;
    const int n16 = inter >> 4;
    const int mblocks = (nrows + 15) >> 4;
    const uint16_t* __restrict__ wg = tb.gate_trellis[e];
    const uint16_t* __restrict__ wu = tb.up_trellis[e];
    const half* __restrict__ suh_g = tb.gate_suh[e];
    const half* __restrict__ suh_u = tb.up_suh[e];
    char* sA = smem;
    char* sB = smem + kGuA;

    // Half k-block h of k-block kb (k16 rows kb * 8 + h * 4 .. + 3), both matrices, into stage `stage`.
    auto load_b = [&](int kb, int h, int stage) {
        char* dst = sB + stage * kStageB;
        for (int q = threadIdx.x; q < kChunks; q += kThreads) {
            const int mk = q / kChunksRow;
            const int j = q - mk * kChunksRow;
            const uint16_t* w = (mk >> 2) ? wu : wg;
            const uint16_t* src = w + ((size_t) (kb * 8 + h * 4 + (mk & 3)) * n16 + (col0 >> 4)) * (16 * KG) + j * 8;
            cp_async(dst + mk * kRowB + j * 16, src);
        }
    };

    // Warp w rotates rows 8 w .. 8 w + 7. A warp whose first row is past nrows skips its rows (a
    // warp-uniform test: no divergence around the shuffles); inside the last partial warp the rows
    // past nrows read token 0 (s_tok is 0 there) and only feed output rows that are never stored.
    const bool wrows = __any_sync(0xffffffffu, 8 * warp < nrows);  // vote: warp-uniform to the compiler

    // Raw 4 elements (8 bytes) of row (8 warp + i), k-block kb. Four rolling slots per warp: slot i & 3 holds
    // row i of the next block for the 4 k16 steps between its load and its rotation.
    auto ldg_x = [&](int kb, int i) -> uint2 {
        const int r = 8 * warp + i;
        const char* p = reinterpret_cast<const char*>(x) + ((size_t) s_tok[r] * hidden + kb * kKb + 4 * lane) * 2;
        return __ldg(reinterpret_cast<const uint2*>(p));
    };

    // Rotate slot i into A buffer `buf` for gate (suh sg) and up (suh su).
    auto xform = [&](uint2 raw, half4 sg, half4 su, int buf, int i) {
        const int r = 8 * warp + i;
        half2 x01, x23;
        to_half<BF16>(raw, x01, x23);
        const int off = a_off(r, lane >> 1) + ((lane & 1) << 3);
        *reinterpret_cast<uint2*>(sA + (buf * 2 + 0) * (kMt1 * 256) + off) = had_hf_pre(x01, x23, sg, lane);
        *reinterpret_cast<uint2*>(sA + (buf * 2 + 1) * (kMt1 * 256) + off) = had_hf_pre(x01, x23, su, lane);
    };

    FragC cg[4][NB][2];
    FragC cu[4][NB][2];
    #pragma unroll
    for (int mb = 0; mb < 4; ++mb)
        #pragma unroll
        for (int s = 0; s < NB; ++s)
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                cg[mb][s][n] = {};
                cu[mb][s][n] = {};
            }

    // k16 step `ks` (0..7) of k-block with A buffer p, B stage ks >> 2.
    auto mma_step = [&](int p, int ks) {
        const char* bst = sB + (ks >> 2) * kStageB;
        const int kk = ks & 3;
        FragB bg[NB][2];
        FragB bu[NB][2];
        #pragma unroll
        for (int s = 0; s < NB; ++s) {
            dq_dispatch<KG, kCbMcg>(reinterpret_cast<const uint32_t*>(bst + kk * kRowB + (warp + 8 * s) * kTileB),
                                    lane << 3, bg[s][0], bg[s][1]);
            dq_dispatch<KG, kCbMcg>(reinterpret_cast<const uint32_t*>(bst + (4 + kk) * kRowB + (warp + 8 * s) * kTileB),
                                    lane << 3, bu[s][0], bu[s][1]);
        }
        const char* ag = sA + (p * 2 + 0) * (kMt1 * 256);
        const char* au = sA + (p * 2 + 1) * (kMt1 * 256);
        #pragma unroll
        for (int mb = 0; mb < 4; ++mb) {
            if (mb < mblocks) {
                const int off = a_off(mb * 16 + (lane & 15), 2 * ks + (lane >> 4));
                FragA fa;
                FragA fu;
                ldsm4(fa, ag + off);
                ldsm4(fu, au + off);
                #pragma unroll
                for (int s = 0; s < NB; ++s)
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        ptx_mma_m16n8k16(fa, bg[s][n], cg[mb][s][n]);
                        ptx_mma_m16n8k16(fu, bu[s][n], cu[mb][s][n]);
                    }
            }
        }
    };

    // Prologue: B for k-block 0 (both halves), block 0 rotated into A buffer 0, slots <- block 1.
    load_b(0, 0, 0);
    cp_async_fence();
    load_b(0, 1, 1);
    cp_async_fence();
    uint2 xs[4];
    half4 sg = ld_half4(suh_g + 4 * lane);
    half4 su = ld_half4(suh_u + 4 * lane);
    if (wrows) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) xs[i] = ldg_x(0, 4 * h + i);
            #pragma unroll
            for (int i = 0; i < 4; ++i) xform(xs[i], sg, su, 0, 4 * h + i);
        }
        if (kblocks > 1) {
            sg = ld_half4(suh_g + kKb + 4 * lane);
            su = ld_half4(suh_u + kKb + 4 * lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) xs[i] = ldg_x(1, i);
        }
    }
    cp_async_wait<1>();
    __syncthreads();

    for (int kb = 0; kb < kblocks; ++kb) {
        const int p = kb & 1;
        const bool next = kb + 1 < kblocks;
        const bool next2 = kb + 2 < kblocks;
        half4 sg2 = sg;
        half4 su2 = su;
        if (next2 && wrows) {
            sg2 = ld_half4(suh_g + (kb + 2) * kKb + 4 * lane);
            su2 = ld_half4(suh_u + (kb + 2) * kKb + 4 * lane);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i == 4) {
                // Second half of block kb: its B stage landed; stage 0 is free for block kb + 1.
                cp_async_wait<0>();
                __syncthreads();
                if (next) load_b(kb + 1, 0, 0);
                cp_async_fence();
            }
            mma_step(p, i);
            if (next && wrows) {
                // Row (8 warp + i) of block kb + 1 into the other A buffer. Its slot refills with row i + 4 of
                // block kb + 1 (steps 0..3) or row i - 4 of block kb + 2 (steps 4..7).
                xform(xs[i & 3], sg, su, p ^ 1, i);
                if (i < 4)
                    xs[i & 3] = ldg_x(kb + 1, i + 4);
                else if (next2)
                    xs[i & 3] = ldg_x(kb + 2, i - 4);
            }
        }
        // Block kb done on every warp: stage 1 is free, A buffer p ^ 1 is complete, stage 0 holds kb + 1.
        cp_async_wait<0>();
        __syncthreads();
        if (next) load_b(kb + 1, 1, 1);
        cp_async_fence();
        sg = sg2;
        su = su2;
    }

    // Epilogue, one 128-column Hadamard block at a time. The smem of the K loop is free (the last
    // __syncthreads above came after the last MMA; no copies are pending).
    float* Cg = reinterpret_cast<float*>(smem);
    float* Cu = Cg + kMt1 * kCpitch;
    const half* __restrict__ svh_g = tb.gate_svh[e];
    const half* __restrict__ svh_u = tb.up_svh[e];
    const half* __restrict__ suh_d = tb.down_suh[e];
    #pragma unroll
    for (int sb = 0; sb < NB; ++sb) {
        if (sb) __syncthreads();
        #pragma unroll
        for (int mb = 0; mb < 4; ++mb) {
            if (mb < mblocks) {
                const int r0 = mb * 16 + (lane >> 2);
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int c = warp * 16 + n * 8 + (lane & 3) * 2;
                    *reinterpret_cast<float2*>(Cg + r0 * kCpitch + c) = make_float2(cg[mb][sb][n][0], cg[mb][sb][n][1]);
                    *reinterpret_cast<float2*>(Cg + (r0 + 8) * kCpitch + c) = make_float2(cg[mb][sb][n][2], cg[mb][sb][n][3]);
                    *reinterpret_cast<float2*>(Cu + r0 * kCpitch + c) = make_float2(cu[mb][sb][n][0], cu[mb][sb][n][1]);
                    *reinterpret_cast<float2*>(Cu + (r0 + 8) * kCpitch + c) = make_float2(cu[mb][sb][n][2], cu[mb][sb][n][3]);
                }
            }
        }
        __syncthreads();
        // Warp w: rows 8 w .. 8 w + 7, skipped as a whole past nrows (warp-uniform); inside the last
        // partial warp rows past nrows are transformed from stale values and not stored.
        const int cb = col0 + sb * 128 + 4 * lane;
        const half4 sd = ld_half4(suh_d + cb);
        #pragma unroll 2
        for (int i = 0; i < (wrows ? 8 : 0); ++i) {
            const int r = 8 * warp + i;
            float4 g = had_ff_post(*reinterpret_cast<const float4*>(Cg + r * kCpitch + 4 * lane), svh_g + cb, lane);
            float4 u = had_ff_post(*reinterpret_cast<const float4*>(Cu + r * kCpitch + 4 * lane), svh_u + cb, lane);
            half2 a01 = __floats2half2_rn(silu_mul(g.x, u.x, limit), silu_mul(g.y, u.y, limit));
            half2 a23 = __floats2half2_rn(silu_mul(g.z, u.z, limit), silu_mul(g.w, u.w, limit));
            const uint2 o = had_hf_pre(a01, a23, sd, lane);
            if (r < nrows) *reinterpret_cast<uint2*>(a2 + (size_t) (row0 + r) * inter + cb) = o;
        }
    }
    __syncthreads();
}

template <int KG, bool BF16>
__global__ void __launch_bounds__(kThreads, 1)
gu_kernel(const void* __restrict__ x, int hidden, int inter, Tables tb, const int* __restrict__ token_sorted,
          const int4* __restrict__ items, int* __restrict__ ctrl, half* __restrict__ a2, float limit)
{
    extern __shared__ __align__(16) char smem[];
    int* s_tok = reinterpret_cast<int*>(smem + kGuA + 2 * gu_stage_bytes(KG));
    int* s_unit = s_tok + kMt1;
    const int nt2 = inter >> 8;                       // 256-column units
    const int nt = nt2 + ((inter & 255) ? 1 : 0);     // + one 128-column unit when inter % 256 == 128
    const int units = ctrl[kCtrlItems1] * nt;
    while (true) {
        if (threadIdx.x == 0) *s_unit = atomicAdd(&ctrl[kCtrlTicket1], 1);
        __syncthreads();
        const int u = *s_unit;
        if (u >= units) break;
        const int item = u / nt;
        const int j = u - item * nt;
        const int4 it = items[item];
        if (threadIdx.x < kMt1) s_tok[threadIdx.x] = threadIdx.x < it.z ? token_sorted[it.y + threadIdx.x] : 0;
        __syncthreads();
        if (j < nt2)
            gu_unit<KG, 2, BF16>(x, hidden, inter, tb, a2, limit, smem, s_tok, it.x, it.y, it.z, j * 256);
        else
            gu_unit<KG, 1, BF16>(x, hidden, inter, tb, a2, limit, smem, s_tok, it.x, it.y, it.z, nt2 * 256);
    }
}

// ----------------------------------------------------------------------------------------------- down
template <int KD>
__global__ void __launch_bounds__(kThreads, 2)
dn_kernel(int hidden, int inter, Tables tb, const int* __restrict__ token_sorted,
          const half* __restrict__ weight_sorted, const int4* __restrict__ items, int* __restrict__ ctrl,
          const half* __restrict__ a2, float* __restrict__ out)
{
    constexpr int kTileB = 32 * KD;
    constexpr int kRowB = 8 * kTileB;              // 8 tiles = 128 hidden columns
    constexpr int kStageB = 8 * kRowB;             // 8 k16 rows = one 128-block of the intermediate dim
    constexpr int kChunksRow = kRowB / 16;
    static_assert(kStageB == dn_stage_bytes(KD), "stage layout");

    extern __shared__ __align__(16) char smem[];
    char* sA = smem;
    char* sB = smem + kDnA;
    int* s_tok = reinterpret_cast<int*>(smem + kDnA + 2 * kStageB);
    half* s_w = reinterpret_cast<half*>(s_tok + kMt2);
    int* s_unit = reinterpret_cast<int*>(s_w + kMt2);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nt = hidden >> 7;
    const int kblocks = inter / kKb;
    const int n16 = hidden >> 4;
    const int units = ctrl[kCtrlItems2] * nt;

    while (true) {
        if (threadIdx.x == 0) *s_unit = atomicAdd(&ctrl[kCtrlTicket2], 1);
        __syncthreads();
        const int u = *s_unit;
        if (u >= units) break;
        const int item = u / nt;
        const int col0 = (u - item * nt) * 128;
        const int4 it = items[item];
        const int e = it.x;
        const int row0 = it.y;
        const int nrows = it.z;
        const int mblocks = (nrows + 15) >> 4;
        if (threadIdx.x < kMt2) {
            const bool ok = threadIdx.x < nrows;
            s_tok[threadIdx.x] = ok ? token_sorted[row0 + threadIdx.x] : 0;
            s_w[threadIdx.x] = ok ? weight_sorted[row0 + threadIdx.x] : __float2half(0.0f);
        }
        const uint16_t* __restrict__ wd = tb.down_trellis[e];

        auto load = [&](int kb, int stage) {
            char* a_dst = sA + stage * (kMt2 * 256);
            for (int q = threadIdx.x; q < kMt2 * 16; q += kThreads) {
                const int r = q >> 4;
                const int c = q & 15;
                if (r < nrows) cp_async(a_dst + a_off(r, c), a2 + (size_t) (row0 + r) * inter + kb * kKb + c * 8);
            }
            char* b_dst = sB + stage * kStageB;
            for (int q = threadIdx.x; q < 8 * kChunksRow; q += kThreads) {
                const int kk = q / kChunksRow;
                const int j = q - kk * kChunksRow;
                cp_async(b_dst + kk * kRowB + j * 16, wd + ((size_t) (kb * 8 + kk) * n16 + (col0 >> 4)) * (16 * KD) + j * 8);
            }
        };

        FragC acc[kMt2 / 16][2];
        #pragma unroll
        for (int mb = 0; mb < kMt2 / 16; ++mb) {
            acc[mb][0] = {};
            acc[mb][1] = {};
        }
        load(0, 0);
        cp_async_fence();
        if (kblocks > 1) load(1, 1);
        cp_async_fence();
        for (int kb = 0; kb < kblocks; ++kb) {
            const int st = kb & 1;
            cp_async_wait<1>();
            __syncthreads();
            const char* a = sA + st * (kMt2 * 256);
            const char* b = sB + st * kStageB;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                FragB fb0;
                FragB fb1;
                dq_dispatch<KD, kCbMcg>(reinterpret_cast<const uint32_t*>(b + kk * kRowB + warp * kTileB), lane << 3, fb0, fb1);
                #pragma unroll
                for (int mb = 0; mb < kMt2 / 16; ++mb) {
                    if (mb < mblocks) {
                        FragA fa;
                        ldsm4(fa, a + a_off(mb * 16 + (lane & 15), 2 * kk + (lane >> 4)));
                        ptx_mma_m16n8k16(fa, fb0, acc[mb][0]);
                        ptx_mma_m16n8k16(fa, fb1, acc[mb][1]);
                    }
                }
            }
            __syncthreads();
            if (kb + 2 < kblocks) load(kb + 2, st);
            cp_async_fence();
        }
        cp_async_wait<0>();

        // Epilogue: [rows][128] fp32 tile, then per row: Hadamard * svh * routing weight -> atomic add.
        float* C = reinterpret_cast<float*>(smem);
        #pragma unroll
        for (int mb = 0; mb < kMt2 / 16; ++mb) {
            if (mb < mblocks) {
                const int r0 = mb * 16 + (lane >> 2);
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const int c = warp * 16 + n * 8 + (lane & 3) * 2;
                    *reinterpret_cast<float2*>(C + r0 * kCpitch + c) = make_float2(acc[mb][n][0], acc[mb][n][1]);
                    *reinterpret_cast<float2*>(C + (r0 + 8) * kCpitch + c) = make_float2(acc[mb][n][2], acc[mb][n][3]);
                }
            }
        }
        __syncthreads();
        // Warp w: rows kRowsW2 w .. + kRowsW2 - 1, skipped as a whole past nrows (warp-uniform); inside the
        // last partial warp rows past nrows are transformed from stale values and not added.
        const half* __restrict__ svh = tb.down_svh[e] + col0 + 4 * lane;
        const int wr = __any_sync(0xffffffffu, kRowsW2 * warp < nrows) ? kRowsW2 : 0;  // vote: warp-uniform
        #pragma unroll 2
        for (int i = 0; i < wr; ++i) {
            const int r = kRowsW2 * warp + i;
            float4 v = had_ff_post(*reinterpret_cast<const float4*>(C + r * kCpitch + 4 * lane), svh, lane);
            const float w = __half2float(s_w[r]);
            v.x *= w;
            v.y *= w;
            v.z *= w;
            v.w *= w;
            if (r < nrows) atomic_add4(out + (size_t) s_tok[r] * hidden + col0 + 4 * lane, v);
        }
        __syncthreads();
    }
}

}  // namespace dsv41_tc
