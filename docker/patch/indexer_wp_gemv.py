"""Indexer weights_proj as a Triton GEMV, bit-identical to cuBLAS (DSV41_INDEXER_WP_GEMV=1).

The V4.1 indexer's weights_proj is a bf16 ReplicatedLinear [32, 5120]. At
decode sizes cuBLAS runs it as cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_
16x16_128x1 on two one-warp CTAs: 70-114 us per call in the r3 trace, on aux
stream 1 beside qkv_a and the compressor, and the last branch to finish at
that join in all 8 indexer layers (the join waits 24-29 us past qkv_a).

The Triton kernel computes each 16x16 output tile with the same MMA chain
(bf16 mma, fp32 accumulate, k ascending, one bf16 rounding at the end), so the
output is bit-for-bit cuBLAS's; it only keeps more loads in flight. It serves
2-D bf16 inputs with MIN_M..MAX_M rows and a contiguous bf16 [N, K] weight
with K a multiple of the k block; anything else calls the stock forward. At
one row cuBLAS switches to another kernel (10 us, a different reduction
order), so M = 1 stays stock. The
first DSV41_INDEXER_WP_VERIFY eager calls (default 8, minimum 1; never during
CUDA graph capture) also run the stock forward and compare raw bits; a
mismatch prints one LOG_DISARMED line and every later call is stock. Decode
batches replay CUDA graphs, so eager decode-size calls may never come: the
first eager call of any size (the profile run's) also self-tests the GEMV on
random rows at every M in MIN_M..MAX_M against the stock forward, and prints
LOG_ENGAGED or disarms. Rows MAX_M+1..WIDE_MAX_M (c >= 3 verify batches) have
their own self-test after that one: a mismatch there leaves only those rows on
cuBLAS (no LOG_DISARMED line), so M <= MAX_M cannot regress. Above BM rows the
GEMV runs one launch per 16-row tile (the same kernel and MMA chain per row).

Top-level imports are stdlib only. decode_levers.install() calls install()
when the env is on.
"""

from __future__ import annotations

import os
import types

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: indexer weights_proj GEMV self-check bit-exact"
LOG_DISARMED = "dsv41: indexer weights_proj GEMV DISABLED ->"

MIN_M, MAX_M = 2, 8  # rows checked bit-exact vs cuBLAS (decode captures 3, 4, 6, 8)
WIDE_MAX_M = 12  # c4-kernels: own self-test; cuBLAS keeps the 16x16 wmma chain only up to M = 12 (GPU check
                 # 2026-10-19: GEMV != cuBLAS bits from M = 13, where cuBLAS runs a faster kernel: 12 us at M 16)
BM = BN = 16
BK, WARPS, STAGES = 512, 2, 3  # kernel_study/fusion_host/wp_gemv_bench.py
DEFAULT_VERIFY = 8
_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "engaged": False, "wide": False}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_INDEXER_WP_GEMV", "0") == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get("DSV41_INDEXER_WP_VERIFY", "") or DEFAULT_VERIFY))


def _disarm(exc: BaseException) -> None:
    if _STATE["armed"]:
        _STATE["armed"] = False
        print("dsv41: indexer weights_proj GEMV DISABLED -> stock GEMM: %r" % (exc,), flush=True)


def _build_kernel(tl, triton):
    @triton.jit
    def _wp_gemv_kernel(
        x_ptr, w_ptr, o_ptr, M, N, K, sxm, swn, som,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
            b = tl.load(w_ptr + rn[None, :] * swn + (k0 + rk)[:, None], mask=rn[None, :] < N, other=0.0)
            acc = tl.dot(a, b, acc)
        tl.store(
            o_ptr + rm[:, None] * som + rn[None, :],
            acc.to(tl.bfloat16),
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )

    return _wp_gemv_kernel


def launch(kernel, triton, x, w, o, bk=BK, warps=WARPS, stages=STAGES) -> None:
    m, k = x.shape
    n = w.shape[0]
    for r0 in range(0, m, BM):  # one launch per BM-row tile; m <= BM is the one launch it always was
        xr, orr = x[r0 : r0 + BM], o[r0 : r0 + BM]
        kernel[(triton.cdiv(n, BN),)](
            xr, w, orr, xr.shape[0], n, k, xr.stride(0), w.stride(0), orr.stride(0),
            BM=BM, BN=BN, BK=bk, num_warps=warps, num_stages=stages,
        )


def usable(torch, x, w, bias, max_m=None) -> bool:
    max_m = (WIDE_MAX_M if _STATE["wide"] else MAX_M) if max_m is None else max_m
    return (
        _STATE["armed"]
        and bias is None
        and x.dim() == 2
        and MIN_M <= x.shape[0] <= max_m
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_cuda
        and x.stride(1) == 1
        and w.is_contiguous()
        and w.dim() == 2
        and x.shape[1] == w.shape[1]
        and w.shape[1] % BK == 0
    )


def selftest(torch, triton, kernel, layer, stock_forward, device, lo=MIN_M, hi=MAX_M) -> None:
    """GEMV vs the stock forward on random rows, every M in lo..hi (default MIN_M..MAX_M)."""
    g = torch.Generator(device=device).manual_seed(0 if lo == MIN_M else lo)
    k = layer.weight.shape[1]
    for m in range(lo, hi + 1):
        x = torch.randn(m, k, device=device, generator=g).to(torch.bfloat16)
        if not usable(torch, x, layer.weight, getattr(layer, "bias", None), max_m=hi):
            raise RuntimeError(f"self-test shape not served (m={m}, k={k})")
        o = torch.empty((m, layer.weight.shape[0]), dtype=torch.bfloat16, device=device)
        launch(kernel, triton, x, layer.weight, o)
        ref = stock_forward(x)
        ref = ref[0] if isinstance(ref, tuple) else ref
        if not torch.equal(o.view(torch.int16), ref.view(torch.int16)):
            raise RuntimeError(f"GEMV != stock in the self-test (m={m})")


def make_forward(torch, triton, kernel, stock_forward):
    def forward(self, x):
        bias = getattr(self, "bias", None)
        if _STATE["armed"] and not _STATE["engaged"] and not torch.cuda.is_current_stream_capturing():
            try:
                selftest(torch, triton, kernel, self, stock_forward, x.device)
                _STATE["engaged"] = True
                print(
                    "dsv41: indexer weights_proj GEMV self-check bit-exact (m=%d..%d, n=%d, k=%d)"
                    % (MIN_M, MAX_M, self.weight.shape[0], self.weight.shape[1]),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                _disarm(exc)
            if _STATE["engaged"]:
                try:  # c4-kernels: rows MAX_M+1..WIDE_MAX_M; a mismatch keeps only them on cuBLAS
                    selftest(torch, triton, kernel, self, stock_forward, x.device, MAX_M + 1, WIDE_MAX_M)
                    _STATE["wide"] = True
                    print(
                        "dsv41: indexer weights_proj GEMV wide rows self-check bit-exact (m=%d..%d)"
                        % (MAX_M + 1, WIDE_MAX_M),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        "dsv41: indexer weights_proj GEMV wide rows (m=%d..%d) stay on cuBLAS: %r"
                        % (MAX_M + 1, WIDE_MAX_M, exc),
                        flush=True,
                    )
        if not usable(torch, x, self.weight, bias):
            return stock_forward(x)
        o = torch.empty((x.shape[0], self.weight.shape[0]), dtype=torch.bfloat16, device=x.device)
        launch(kernel, triton, x, self.weight, o)
        if _STATE["verify_left"] > 0 and not torch.cuda.is_current_stream_capturing():
            ref = stock_forward(x)
            ref_t = ref[0] if isinstance(ref, tuple) else ref
            if not torch.equal(o.view(torch.int16), ref_t.view(torch.int16)):
                _disarm(RuntimeError(f"GEMV != stock (m={x.shape[0]})"))
                return ref
            _STATE["verify_left"] -= 1
        return (o, None) if getattr(self, "return_bias", True) else o

    return forward


def install() -> str:
    import torch
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Indexer
    from vllm.triton_utils import tl, triton

    if getattr(DeepseekV4Indexer.__init__, "_dsv41_wp_gemv", False):
        return "already installed"
    _STATE["verify_left"] = verify_calls()
    kernel = _build_kernel(tl, triton)
    orig_init = DeepseekV4Indexer.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        wp = self.weights_proj
        wp.forward = types.MethodType(make_forward(torch, triton, kernel, wp.forward), wp)

    __init__._dsv41_wp_gemv = True
    DeepseekV4Indexer.__init__ = __init__
    return (
        f"DeepseekV4Indexer.weights_proj GEMV armed ({MIN_M}<=m<={MAX_M}, "
        f"{MAX_M + 1}<=m<={WIDE_MAX_M} after its self-test, verify={_STATE['verify_left']})"
    )
