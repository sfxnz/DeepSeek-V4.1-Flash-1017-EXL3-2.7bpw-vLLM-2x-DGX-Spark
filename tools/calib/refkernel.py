"""Torch implementations of the DeepSeek-V4.1-Flash-1017 reference `inference/kernel.py` API.

Drop-in for the TileLang module: same function names and signatures, same numerics where torch can
express them exactly, so `sys.modules["kernel"] = refkernel` lets the official model.py run on any GPU.

Exactness notes (verified against TileLang on sm_121 by tests/test_refkernel.py when tilelang loads):
- act_quant / fp4_act_quant: bit-exact (power-of-two UE8M0 scales via the same exponent-bit trick,
  RNE casts to e4m3 / e2m1, same clamps and amax floors).
- fp8_gemm / fp4_gemm: weights and activations are dequantized to bf16, which is exact for e4m3 and
  e2m1 values times a power-of-two scale, then one cuBLAS bf16 GEMM with fp32 accumulation and a bf16
  output (bf16 reduced-precision split-K reductions are disabled). Only the fp32 summation order differs
  from the per-32-K-block accumulation of the TileLang kernel.
- sparse_attn: exact gather semantics (-1 = empty slot, finite -1e30 running-max floor so all-empty rows
  give zeros, attn_sink only in the denominator, probabilities rounded to bf16 before P@V). It uses one
  global max per row instead of the kernel's 64-wide online softmax, so the bf16 rounding of P differs
  slightly (relative ~1e-3 on individual probabilities, far below bf16 output resolution on average).
- hc_split_sinkhorn: same formulas and iteration order in fp32.
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_FP8_MAX_INV = torch.tensor(1.0 / 448.0, dtype=torch.float32).item()  # fp32 constant, as in the kernel
_FP4_MAX_INV = torch.tensor(1.0 / 6.0, dtype=torch.float32).item()

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def pow2_ceil_scale(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive fp32 x via exponent bits (kernel.py fast_log2_ceil/fast_pow2)."""
    bits = x.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values already clamped to [-6, 6] to the nearest e2m1 value, ties to even code."""
    a = v.abs()
    q = torch.where(
        a <= 0.25, 0.0,
        torch.where(a < 0.75, 0.5,
        torch.where(a <= 1.25, 1.0,
        torch.where(a < 1.75, 1.5,
        torch.where(a <= 2.5, 2.0,
        torch.where(a < 3.5, 3.0,
        torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.copysign(q, v)


def _ue8m0_scales(xb: torch.Tensor, max_inv: float, floor: float) -> torch.Tensor:
    amax = xb.abs().amax(-1, keepdim=True).float().clamp_min(floor)
    return pow2_ceil_scale(amax * max_inv)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None, scale_dtype=torch.float32, inplace: bool = False):
    """Block-wise FP8 (e4m3) quantization along the last dim. inplace=True: quant+dequant written back to x."""
    N = x.size(-1)
    assert N % block_size == 0
    xb = x.float().unflatten(-1, (N // block_size, block_size))
    if scale_fmt is not None:
        s = _ue8m0_scales(xb, _FP8_MAX_INV, 1e-4)
    else:
        s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4) * _FP8_MAX_INV
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def act_quant_dequant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Out-of-place fp8 quant+dequant with UE8M0 scales (what `linear()` feeds the GEMM), as bf16."""
    N = x.size(-1)
    xb = x.float().unflatten(-1, (N // block_size, block_size))
    s = _ue8m0_scales(xb, _FP8_MAX_INV, 1e-4)
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * s).flatten(-2).to(torch.bfloat16)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False, scale_dtype=torch.float8_e8m0fnu):
    """FP4 (e2m1) with E8M0 power-of-two scales (indexer) or E4M3 scales (compressed KV)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    xb = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = pow2_ceil_scale(amax * _FP4_MAX_INV)
    q = round_fp4((xb / s).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((q * s).flatten(-2).to(x.dtype))
        return x
    # packed output is only used by the reference for KV caches we never build; keep the API honest
    codes = torch.bucketize(q.abs(), torch.tensor(FP4_VALUES[:8], device=q.device)).to(torch.uint8)
    codes = codes | ((q < 0) | ((q == 0) & torch.signbit(q))).to(torch.uint8) << 3
    codes = codes.flatten(-2)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)
    return packed, s.squeeze(-1).to(scale_dtype)


def dequant_fp8_weight(w: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """MXFP8 weight [N, K] e4m3 with one E8M0 scale per (block x block) -> bf16 (exact)."""
    N, K = w.shape
    s = scale.float()
    s = s.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return (w.float() * s).to(torch.bfloat16)


_LUTS: dict = {}


def _fp4_byte_lut(device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _LUTS:
        tbl = torch.tensor(FP4_VALUES, dtype=torch.float32)
        b = torch.arange(256)
        _LUTS[key] = torch.stack([tbl[b & 15], tbl[b >> 4]], dim=-1).to(device=device, dtype=dtype)
    return _LUTS[key]


def dequant_fp4_weight(w: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """MXFP4 weight [N, K//2] (int8/uint8/float4_e2m1fn_x2, low nibble = even column) with E8M0 scale
    [N, K//32] -> dtype [N, K] (exact in bf16/fp16/fp32)."""
    b = w.contiguous().view(torch.uint8)
    N = b.size(0)
    v = _fp4_byte_lut(b.device, dtype)[b.int()].view(N, -1)  # byte -> (low nibble, high nibble) values
    K = v.size(1)
    s = scale.contiguous().view(torch.uint8).int()
    s = ((s << 23).view(torch.float32)).to(dtype)  # 2**(e-127); e=0 -> 0 (never used by real weights)
    return (v.view(N, K // 32, 32) * s.unsqueeze(-1)).view(N, K)


def _scale_to_f32(s: torch.Tensor) -> torch.Tensor:
    return s.float()


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size: int = 128):
    """C[M,N] = A[M,K] @ B[N,K]^T with per-block FP8 scaling; output in the default dtype."""
    K = a.size(-1)
    ad = (a.float().unflatten(-1, (K // block_size, block_size)) * _scale_to_f32(a_s).unsqueeze(-1)).flatten(-2)
    bd = dequant_fp8_weight(b, b_s, block_size)
    out = torch.matmul(ad.to(torch.bfloat16), bd.t())
    return out.to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size: int = 128):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T."""
    K = a.size(-1)
    ad = (a.float().unflatten(-1, (K // act_block_size, act_block_size)) * _scale_to_f32(a_s).unsqueeze(-1)).flatten(-2)
    bd = dequant_fp4_weight(b, b_s)
    out = torch.matmul(ad.to(torch.bfloat16), bd.t())
    return out.to(torch.get_default_dtype())


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float, chunk: int = 256) -> torch.Tensor:
    """q [b,m,h,d] bf16, kv [b,n,d] bf16 (shared K=V), topk_idxs [b,m,k] int32 (-1 empty) -> [b,m,h,d] bf16."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, h, 1)
    for bi in range(b):
        for c0 in range(0, m, chunk):
            c1 = min(m, c0 + chunk)
            idx = topk_idxs[bi, c0:c1].long()
            valid = idx >= 0
            g = kv[bi][idx.clamp_min(0)]  # [c, k, d]
            g = torch.where(valid.unsqueeze(-1), g, torch.zeros((), dtype=g.dtype, device=g.device))
            qc = q[bi, c0:c1]  # [c, h, d]
            s = torch.bmm(qc, g.transpose(1, 2), out_dtype=torch.float32) * softmax_scale  # [c, h, k]
            s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = s.amax(-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(s - mx)
            den = p.sum(-1, keepdim=True) + torch.exp(sink - mx)
            o = torch.bmm(p.to(torch.bfloat16), g, out_dtype=torch.float32) / den
            out[bi, c0:c1] = o.to(out.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    """mixes [..., (2+hc)*hc] fp32 -> pre [..., hc], post [..., hc], comb [..., hc, hc]."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
