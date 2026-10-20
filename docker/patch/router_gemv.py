"""bf16 x bf16 -> fp32 small-M GEMMs (router gates, compressor kv_score) as one Triton kernel that
reproduces cuBLAS's split-K arithmetic bit for bit (DSV41_ROUTER_GEMV=1, default off).

What it replaces. On sm_121 GateLinear takes its cuBLAS tier, torch.mm(x, W.T, out_dtype=fp32), and
the V4.1 compressor's kv_score is the same call. cuBLAS runs these as cutlass_80_wmma_tensorop_
s161616gemm_bf16_16x16_* on one-warp CTAs (perf2 B0 traces, 1017, e15-kbits):
- target router [384 x 5120], 40 a step: m=4 grid (8,3,16) split-K 16, 47.0 us + splitKreduce 9.3 us
  (c=1); m=16 (8,3,16), 20.2 + 4.8 us (c=4). Its 384 CTAs x 16 KiB smem fill the SMs for ~13 us, so
  the shared-expert gate_up GEMV it overlaps starts late, and the cooperative p2b MoE waits for both.
- draft router [128 x 5120], 3 a step: m=3 grid (8,1,1), no split, 103.6 us (12.7 GB/s) on the draft
  critical path; m=12 (8,1,9) 21.4 us + 10.0 us reduce.
- compressor kv_score [1024 or 512 x 5120], layers 2/8/14/20: m=4 (8,8,16) 64-91 us + reduce; at m=16
  its 576 CTAs hold the qkv_a GEMM back ~45 us in layers 2/8/14.

Arithmetic. cuBLAS splits K into S slices of KS elements (a multiple of 8); each slice is one MMA
chain (bf16 m16n8k16, fp32 accumulate, k ascending in 16-element steps, starting from 0), and the
splitKreduce kernel sums the S fp32 partials in slice order. The Triton kernel computes the same
chains (tl.dot over k-ascending blocks; a slice's tail block is masked to zeros, which adds exact
zeros) and sums them in the same order inside one launch, so the fp32 output equals cuBLAS's bit for
bit. S = 1 is the unsplit chain (the indexer_wp_gemv proof covers that kernel family). Which (KS, S)
cuBLAS uses depends on M (and N, K): it is not guessed. At the first eager call per weight shape the
self-test runs the stock op and the kernel for every candidate KS at every M in 1..MAX_M on random
rows and keeps, per M, the first candidate that matches bitwise on two inputs; an M with no match
stays on cuBLAS (M = 1 usually: cuBLAS takes a GEMV kernel there). The serving launch config is then
checked once per kept M, bitwise, which also compiles it before any CUDA-graph capture. The first
DSV41_ROUTER_GEMV_VERIFY eager calls (default 8, minimum 1; never during capture) also run the stock
op and compare raw bits; a mismatch prints one LOG_DISARMED line and every later call is stock.

Numerics: bit-exact by construction and by the self-test; the routing (top-6 of the fp32 logits)
cannot change. The one value class the self-test cannot see is the sign of an exactly-zero output,
which no consumer distinguishes (sqrtsoftplus, RMSNorm).

Wiring: GateLinear.forward (target and DSpark draft gates) when the stock forward would take the
cuBLAS tier; DeepseekV4Attention._run_parallel_input_projections is rebuilt from its own source with
the kv_score torch.mm swapped for mm_f32 (refused unless the anchor matches once). Top-level imports
are stdlib only; decode_levers.install() calls install() when the env is on.
"""

from __future__ import annotations

import os

# Boot-log markers for tools/engagement_audit.py.
LOG_ENGAGED = "dsv41: router gemv self-check bit-exact"
LOG_DISARMED = "dsv41: router gemv DISABLED ->"

ENV = "DSV41_ROUTER_GEMV"
VERIFY_ENV = "DSV41_ROUTER_GEMV_VERIFY"
# Weight rows N to serve, comma list (384 target gate, 128 draft gate, 1024/512 compressor); empty = all.
SHAPES_ENV = "DSV41_ROUTER_GEMV_N"
DEFAULT_VERIFY = 8
MAX_M = 24  # c=4 at k=5 verifies 24 rows
BN = 16  # cuBLAS's 16-wide tiles; tl.dot's minimum
# Launch configs (BK, warps, stages); the bits do not depend on them (a slice tail is masked to zeros).
DISCOVERY = (64, 2, 2)  # small: also the fallback when a serving config does not launch
# One slice: indexer_wp_gemv's tuned config (2 CTAs streamed 13 GB/s each there); split: 128-wide
# blocks, deeper ring. 32-row tiles (M > 16) cap BK so the ring fits 99 KiB of shared memory.
# kernel_study/router_gemv/bench.py --configs sweeps others.
SERVE_UNSPLIT = (512, 2, 3)
SERVE_SPLIT = (128, 2, 4)
SPLIT_SEARCH = tuple(range(1, 33))
ALIGNS = (8, 16, 32, 64, 128, 256)
# Seen in the B0 traces for K = 5120: S = 16 (KS 320), S = 9 (KS 576), S = 1 (KS 5120).
LIKELY_KS = (320, 576, 5120)
SEQ, REV = 0, 1  # order of the slice-partial sum

# The compressor call the attention rebuild replaces (attention.py, e15 image, 16-space indent).
KV_ANCHOR = (
    "                return torch.mm(\n"
    "                    hidden_states,\n"
    "                    compressor.fused_wkv_wgate.weight.T,\n"
    "                    out_dtype=torch.float32,\n"
    "                )\n"
)
KV_REPLACEMENT = (  # same line count as the anchor
    "                return _dsv41_mm_f32(\n"
    "                    hidden_states,\n"
    "                    compressor.fused_wkv_wgate.weight,\n"
    "                    # dsv41 router_gemv: torch.mm(x, w.T, out_dtype=fp32), bitwise\n"
    "                )\n"
)

_STATE = {"armed": True, "verify_left": DEFAULT_VERIFY, "tables": {}, "kernel": None, "mods": None,
          "shapes": None}


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get(ENV, "0") or "0") == "1"


def verify_calls(env=None) -> int:
    env = os.environ if env is None else env
    return max(1, int(env.get(VERIFY_ENV, "") or DEFAULT_VERIFY))


def served_n(env=None) -> frozenset[int] | None:
    """N values the lever serves (None = every shape)."""
    env = os.environ if env is None else env
    raw = (env.get(SHAPES_ENV, "") or "").replace(" ", "")
    return frozenset(int(v) for v in raw.split(",") if v) or None


def cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def split_candidates(k: int, first=()) -> list[int]:
    """Slice lengths KS to try for a K-long reduction: `first`, then round_up(ceil(K/S), align)."""
    out: list[int] = []

    def add(ks: int) -> None:
        if 8 <= ks <= k and ks % 8 == 0 and ks not in out:
            out.append(ks)

    for ks in first:
        add(int(ks))
    for s in SPLIT_SEARCH:
        for a in ALIGNS:
            add(cdiv(cdiv(k, s), a) * a)
    return out


def bm_for(m: int) -> int:
    return 16 if m <= 16 else 32


def serve_config(k: int, ks: int, m: int = 16) -> tuple[int, int, int]:
    """(BK, warps, stages) to serve M rows with slice length KS; any BK gives the same bits."""
    bk, warps, stages = SERVE_UNSPLIT if cdiv(k, ks) == 1 else SERVE_SPLIT
    if ks < 256:
        bk = 64
    if bm_for(m) == 32:
        bk = min(bk, 256 if stages <= 3 else 128)
    return bk, warps, stages


def schedule(k: int, ks: int, bk: int, order: int = SEQ) -> list[tuple[int, int, int, bool]]:
    """(slice, k0, kend, first) per kernel iteration, in kernel order (host mirror of the loop)."""
    s_count, nb = cdiv(k, ks), cdiv(ks, bk)
    out = []
    for j in range(s_count * nb):
        q, b = divmod(j, nb)
        s = s_count - 1 - q if order == REV else q
        k0 = s * ks + b * bk
        out.append((s, k0, min(s * ks + ks, k), b == 0))
    return out


def discover(ms, run_stock, run_mine, equal, k: int, first=()):
    """Per M, the first (KS, order) whose output equals the stock op's on every probe input.

    run_stock(m) -> list of reference outputs (one per probe input); run_mine(m, ks, order) -> list of
    kernel outputs on the same inputs; equal(a, b) -> bool (bitwise). Returns ({m: (ks, order)},
    [m without a match]). The last match is tried first: cuBLAS keeps one config over a range of M.
    """
    table, stock_ms = {}, []
    last = None
    for m in ms:
        refs = run_stock(m)
        cands = split_candidates(k, ((last[0],) if last else ()) + tuple(first))
        orders = (last[1],) if last else ()
        orders += tuple(o for o in (SEQ, REV) if o not in orders)
        hit = None
        for order in orders:
            for ks in cands:
                if order == REV and cdiv(k, ks) == 1:
                    continue  # one slice: the order is moot, SEQ already tried it
                got = run_mine(m, ks, order)
                if len(got) == len(refs) and all(equal(a, b) for a, b in zip(got, refs)):
                    hit = (ks, order)
                    break
            if hit:
                break
        if hit:
            table[m] = hit
            last = hit
        else:
            stock_ms.append(m)
    return table, stock_ms


def _build_kernel(tl, triton):
    @triton.jit
    def _mm_f32_split_kernel(
        x_ptr, w_ptr, o_ptr, M, N, K, KS, S, NB, sxm, swn, som,
        REV: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mrow = rm < M
        ncol = rn < N
        total = tl.zeros((BM, BN), dtype=tl.float32)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for j in range(0, S * NB):
            q = j // NB
            b = j - q * NB
            if REV:
                s = S - 1 - q
            else:
                s = q
            k0 = s * KS + b * BK
            kend = tl.minimum(s * KS + KS, K)
            kk = k0 + rk
            kin = kk < kend
            first = b == 0
            # a slice boundary: the finished slice's chain joins the ordered sum, a new chain starts at 0
            total = total + tl.where(first, acc, 0.0)
            acc = tl.where(first, 0.0, acc)
            a = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :], mask=mrow[:, None] & kin[None, :], other=0.0)
            w = tl.load(w_ptr + rn[None, :] * swn + kk[:, None], mask=ncol[None, :] & kin[:, None], other=0.0)
            acc = tl.dot(a, w, acc)
        total = total + acc
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], total, mask=mrow[:, None] & ncol[None, :])

    return _mm_f32_split_kernel


def launch(kernel, triton, x, w, o, ks: int, order: int, bk: int, warps: int, stages: int) -> None:
    m, k = x.shape
    n = w.shape[0]
    bm = bm_for(m)
    kernel[(triton.cdiv(n, BN), triton.cdiv(m, bm))](
        x, w, o, m, n, k, ks, cdiv(k, ks), cdiv(ks, bk), x.stride(0), w.stride(0), o.stride(0),
        REV=order == REV, BM=bm, BN=BN, BK=bk, num_warps=warps, num_stages=stages,
    )


def usable(torch, x, w) -> bool:
    """Shapes the kernel serves (the M table is checked separately)."""
    return (
        x.dim() == 2
        and w.dim() == 2
        and 1 <= x.shape[0] <= MAX_M
        and x.shape[1] == w.shape[1]
        and x.shape[1] % 8 == 0
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_cuda
        and w.is_cuda
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 16 == 0
        and w.data_ptr() % 16 == 0
    )


def _stock_mm(torch, x, w):
    return torch.mm(x, w.T, out_dtype=torch.float32)


def _disarm(exc) -> None:
    """A serving-time mismatch: every shape goes back to cuBLAS for the process lifetime."""
    if _STATE["armed"]:
        _STATE["armed"] = False
        print(f"dsv41: router gemv DISABLED -> cuBLAS stays everywhere: {exc!r}", flush=True)


def _mods():
    """(torch, tl, triton), imported once."""
    if _STATE["mods"] is None:
        import torch
        from vllm.triton_utils import tl, triton

        _STATE["mods"] = (torch, tl, triton)
    return _STATE["mods"]


def _kernel(tl, triton):
    if _STATE["kernel"] is None:
        _STATE["kernel"] = _build_kernel(tl, triton)
    return _STATE["kernel"]


def selftest(torch, triton, kernel, w, stock, device):
    """Per-M (KS, order) table for this weight shape (see discover), then the serving config checked."""
    n, k = w.shape
    g = torch.Generator(device=device).manual_seed(0)
    probes = {}

    def inputs(m):
        if m not in probes:
            base = torch.randn(m, k, device=device, generator=g)
            probes[m] = [base.to(torch.bfloat16), (base * 8.0).to(torch.bfloat16)]
        return probes[m]

    def run_stock(m):
        return [stock(x) for x in inputs(m)]

    def run_mine(m, ks, order, cfg=DISCOVERY):
        outs = []
        for x in inputs(m):
            o = torch.empty((m, n), dtype=torch.float32, device=device)
            launch(kernel, triton, x, w, o, ks, order, *cfg)
            outs.append(o)
        return outs

    def equal(a, b):
        return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))

    found, stock_ms = discover(range(1, MAX_M + 1), run_stock, run_mine, equal, k, LIKELY_KS)
    table = {}
    for m, (ks, order) in found.items():
        # the serving config, checked bitwise (and compiled) here; else the discovery config
        for cfg in (serve_config(k, ks, m), DISCOVERY):
            try:
                got = run_mine(m, ks, order, cfg)
            except Exception:  # noqa: BLE001 - e.g. out of shared memory: try the next config
                continue
            if all(equal(a, b) for a, b in zip(got, run_stock(m))):
                table[m] = (ks, order, cfg)
                break
        else:
            stock_ms.append(m)
    return table, sorted(stock_ms)


def weight_usable(torch, w) -> bool:
    return (
        w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_cuda and w.is_contiguous()
        and w.shape[1] % 8 == 0 and w.data_ptr() % 16 == 0
    )


def _table(torch, triton, tl, w, stock):
    """The (N, K) shape's M table; the first eager call of the shape (any M) builds it (self-test)."""
    key = (int(w.shape[0]), int(w.shape[1]))
    tables = _STATE["tables"]
    if key in tables:
        return tables[key]
    if torch.cuda.is_current_stream_capturing() or not weight_usable(torch, w):
        return None  # never self-test inside a capture; that call stays stock
    if _STATE["shapes"] is not None and key[0] not in _STATE["shapes"]:
        tables[key] = {}
        print(f"dsv41: router gemv: n={key[0]} k={key[1]} left to cuBLAS ({SHAPES_ENV})", flush=True)
        return tables[key]
    try:
        table, stock_ms = selftest(torch, triton, _kernel(tl, triton), w, stock, w.device)
    except Exception as exc:  # noqa: BLE001 - this shape stays on cuBLAS, the others keep theirs
        tables[key] = {}
        print(f"dsv41: router gemv DISABLED -> cuBLAS stays for n={key[0]} k={key[1]}: {exc!r}", flush=True)
        return tables[key]
    tables[key] = table
    if not table:
        print(
            f"dsv41: router gemv DISABLED -> cuBLAS stays for n={key[0]} k={key[1]}: no M matched it bitwise",
            flush=True,
        )
        return table
    splits = sorted({(ks, cdiv(key[1], ks), order) for ks, order, _ in table.values()})
    print(
        f"dsv41: router gemv self-check bit-exact (n={key[0]} k={key[1]}): m {sorted(table)} on the kernel, "
        f"(ks, slices, order) {splits}; cuBLAS keeps m {stock_ms}",
        flush=True,
    )
    return table


def mm_f32(x, w, stock=None):
    """torch.mm(x, w.T, out_dtype=fp32) for bf16 x [M, K], w [N, K]; bitwise cuBLAS's result."""
    torch, tl, triton = _mods()

    def ref(t):
        return stock(t) if stock is not None else _stock_mm(torch, t, w)

    if not _STATE["armed"]:
        return ref(x)
    table = _table(torch, triton, tl, w, ref)
    if not _STATE["armed"] or not table or x.shape[0] not in table or not usable(torch, x, w):
        return ref(x)
    ks, order, cfg = table[x.shape[0]]
    o = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device=x.device)
    launch(_kernel(tl, triton), triton, x, w, o, ks, order, *cfg)
    if _STATE["verify_left"] > 0 and not torch.cuda.is_current_stream_capturing():
        r = ref(x)
        if not torch.equal(o.view(torch.int32), r.view(torch.int32)):
            _disarm(RuntimeError(f"kernel != stock (n={w.shape[0]} m={x.shape[0]})"))
            return r
        _STATE["verify_left"] -= 1
    return o


def gate_takes_cublas(gate, x, torch) -> bool:
    """True when GateLinear.forward (e15) would run its torch.mm(..., out_dtype=fp32) tier."""
    if getattr(gate, "allow_ll_bf16_gemm", False) and x.shape[0] <= 16 and x.dtype == torch.bfloat16:
        return False
    if getattr(gate, "allow_fp32_router_gemm", False) and x.dtype in (torch.float32, torch.bfloat16):
        return False
    if getattr(gate, "allow_bf16x3_router_gemm", False) and x.dtype == torch.bfloat16:
        return False
    return bool(getattr(gate, "allow_cublas_router_gemm", False)) and x.dtype == torch.bfloat16


def make_gate_forward(torch, stock_forward):
    def forward(self, x):
        if not _STATE["armed"] or not gate_takes_cublas(self, x, torch):
            return stock_forward(self, x)
        return mm_f32(x, self.weight, lambda t: stock_forward(self, t)[0]), None

    forward._dsv41_router_gemv = True
    return forward


def rebuild_kv_projection(cls, helper) -> str:
    """Rebuild cls._run_parallel_input_projections from its source with the kv_score mm swapped."""
    import __future__
    import inspect
    import sys
    import textwrap

    fn = cls._run_parallel_input_projections
    if getattr(fn, "_dsv41_router_gemv", False):
        return "already rebuilt"
    src, first_line = inspect.getsourcelines(fn)
    src = "".join(src)
    if src.count(KV_ANCHOR) != 1:
        raise RuntimeError("compressor kv_score anchor not found exactly once")
    # Same line count, so tracebacks point at the image's source lines.
    new_src = "\n" * (first_line - 1) + textwrap.dedent(src.replace(KV_ANCHOR, KV_REPLACEMENT))
    module = sys.modules[fn.__module__]
    module.__dict__["_dsv41_mm_f32"] = helper
    code = compile(new_src, inspect.getsourcefile(fn) or "<router_gemv>", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    local_ns: dict = {}
    exec(code, module.__dict__, local_ns)  # noqa: S102 - the image's own method text, one call swapped
    new_fn = local_ns[fn.__name__]
    new_fn.__qualname__ = fn.__qualname__
    new_fn._dsv41_router_gemv = True
    cls._run_parallel_input_projections = new_fn
    return "compressor kv_score rebuilt"


def install() -> str:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    torch = _mods()[0]
    _STATE["verify_left"] = verify_calls()
    _STATE["shapes"] = served_n()
    parts = []
    if getattr(GateLinear.forward, "_dsv41_router_gemv", False):
        parts.append("GateLinear already patched")
    else:
        GateLinear.forward = make_gate_forward(torch, GateLinear.forward)
        parts.append("GateLinear cuBLAS tier armed")
    try:
        from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention

        parts.append(rebuild_kv_projection(DeepseekV4Attention, mm_f32))
    except Exception as exc:  # noqa: BLE001 - the gates stay armed without the compressor
        parts.append(f"compressor kv_score stays cuBLAS ({exc!r})")
    shapes = "all" if _STATE["shapes"] is None else sorted(_STATE["shapes"])
    return f"{'; '.join(parts)} (m<={MAX_M}, n {shapes}, verify={_STATE['verify_left']})"
