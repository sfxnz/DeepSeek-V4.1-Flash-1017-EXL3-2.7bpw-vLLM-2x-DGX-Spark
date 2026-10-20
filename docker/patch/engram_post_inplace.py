"""Engram layers 1 and 14 in eager prefill steps: post-wkv output in place, optional wkv row chunks.

Engram.forward (models/deepseek_v4_1/common/engram.py) runs, per rank at T = 8190:
  kv = self.wkv(self.embed(hash_ids).flatten(-2))    # [T, 25600] bf16, 400 MiB (input [T, 6144], 96 MiB)
  output = torch.empty_like(hidden_states)            # [T, 4, 5120] bf16, 320 MiB
  _fused_engram_post_wkv_kernel[(T * hc_mult,)](hidden_states, kv, ..., output, ...)
so its peak holds the input stream, kv and a second stream at once.

DSV41_ENGRAM_POST_INPLACE=1: the kernel writes its output into hidden_states. Bit-identical by
construction: program (t, h) loads row hidden[t, h, :] whole (BLOCK_SIZE >= DIM), reduces it,
and only then stores output[t, h, :]; no program reads another program's row, and the store
depends on the loaded values, so writing in place changes no value. The caller's only reference
to the input stream is the name it rebinds to the result (`residual = self.engram(residual, ...)`;
Engram layers sit before decoder replay's cut and are not DSpark taps). -320 MiB at T = 8190.

DSV41_ENGRAM_WKV_ROWS=N (default 0 = off): the wkv GEMM and the post-wkv kernel run per N-row
slice of the step (the embedding gather stays one call, so both ranks run the same collectives).
kv shrinks to [N, 25600] (-300 MiB at N = 2048). Each output row depends only on its own input
row, but the GEMM's kernel and split may depend on M, so this is bit-identical only if the
kernel is M-independent: the first VERIFY_CALLS eager calls compute the full-M wkv too and
compare every slice bitwise; a mismatch disarms the row chunks (LOG_DISARMED) and keeps the full
result. The in-place output is checked the same way on those calls (stock kernel into a fresh
buffer first, then in place, bitwise compare; a mismatch disarms it and returns the stock result).

Only eager calls with more than MIN_ROWS tokens and no sequence parallelism change; decode steps and
CUDA-graph capture run the stock forward. Runtime patch (sitecustomize); refuses (LOG_DISARMED)
unless the image's Engram.forward matches EXPECTED_FORWARD (ast.unparse, docstring dropped).
Top-level imports are stdlib only.
"""

from __future__ import annotations

import os

MARK = "dsv41: engram post"
LOG_ENGAGED = "dsv41: engram post engaged"
LOG_DISARMED = (
    "dsv41: engram post DISABLED",
    "dsv41: engram post in place DISABLED",
    "dsv41: engram wkv rows DISABLED",
)
MIN_ROWS = 64
VERIFY_CALLS = 2

EXPECTED_FORWARD = """def forward(self, hidden_states: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None=None) -> torch.Tensor:
    kv = self.wkv(self.embed(hash_ids).flatten(-2))
    num_kv_tokens = hash_ids.shape[0]
    assert token_mask is None or token_mask.shape == (num_kv_tokens,)
    if self.use_sequence_parallel:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = (num_kv_tokens + tp_size - 1) // tp_size
        assert hidden_states.shape[0] == shard_size
        start = min(tp_rank * shard_size, num_kv_tokens)
        num_kv_tokens = min(shard_size, num_kv_tokens - start)
        if token_mask is not None:
            token_mask = token_mask[start:start + num_kv_tokens]
    num_tokens, hc_mult, dim = hidden_states.shape
    assert hc_mult == self.hc_mult and dim == self.dim
    assert kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim
    output = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return output
    block_size = triton.next_power_of_2(dim)
    num_warps = 8 if block_size >= 2048 else 4
    mask = token_mask if token_mask is not None else hidden_states
    _fused_engram_post_wkv_kernel[num_tokens * hc_mult,](hidden_states, kv, self.q_weight, self.k_weight, mask, output, num_kv_tokens, hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), kv.stride(0), kv.stride(1), self.q_weight.stride(0), self.q_weight.stride(1), self.k_weight.stride(0), self.k_weight.stride(1), token_mask.stride(0) if token_mask is not None else 0, output.stride(0), output.stride(1), output.stride(2), self.eps, self.clamp_value, DIM=dim, HC_MULT=hc_mult, BLOCK_SIZE=block_size, HAS_MASK=token_mask is not None, num_warps=num_warps)
    return output"""


def settings(env=None) -> tuple[bool, int]:
    """(in place on, wkv row chunk or 0)."""
    env = os.environ if env is None else env
    inplace = env.get("DSV41_ENGRAM_POST_INPLACE", "0") == "1"
    try:
        rows = max(0, int(env.get("DSV41_ENGRAM_WKV_ROWS", "0") or 0))
    except ValueError:
        rows = 0
    return inplace, rows


def enabled(env=None) -> bool:
    inplace, rows = settings(env)
    return inplace or rows > 0


def bits_equal(torch, a, b) -> bool:
    """Bitwise equality (NaN payloads included)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.bfloat16, torch.float16):
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    if a.dtype == torch.float32:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    return bool(torch.equal(a, b))


def make(cls, eng, torch, inplace: bool, rows: int, say=print):
    """Engram.forward replacement; `eng` is the engram module (triton, kernel)."""
    from attn_out_defer import capturing

    stock = cls.forward
    state = {"inplace": inplace, "rows": rows, "verify": VERIFY_CALLS, "disarmed": False}

    def launch(self, hidden, kv, mask_t, output, num_kv):
        hc_mult, dim = hidden.shape[1], hidden.shape[2]
        block_size = eng.triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4
        mask = mask_t if mask_t is not None else hidden
        eng._fused_engram_post_wkv_kernel[(hidden.shape[0] * hc_mult,)](
            hidden, kv, self.q_weight, self.k_weight, mask, output, num_kv,
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            kv.stride(0), kv.stride(1),
            self.q_weight.stride(0), self.q_weight.stride(1),
            self.k_weight.stride(0), self.k_weight.stride(1),
            mask_t.stride(0) if mask_t is not None else 0,
            output.stride(0), output.stride(1), output.stride(2),
            self.eps, self.clamp_value,
            DIM=dim, HC_MULT=hc_mult, BLOCK_SIZE=block_size, HAS_MASK=mask_t is not None, num_warps=num_warps,
        )

    def check_shapes(self, hidden_states, hash_ids, token_mask, kv=None):
        num_kv_tokens = hash_ids.shape[0]
        assert token_mask is None or token_mask.shape == (num_kv_tokens,)
        _, hc_mult, dim = hidden_states.shape
        assert hc_mult == self.hc_mult and dim == self.dim
        assert kv is None or (kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim)
        return num_kv_tokens

    def verify(self, hidden_states, hash_ids, token_mask):
        """First eager calls: stock kernel into a fresh buffer, then the lever's path; bitwise compare."""
        x = self.embed(hash_ids).flatten(-2)
        kv = self.wkv(x)
        n = check_shapes(self, hidden_states, hash_ids, token_mask, kv)
        ref = torch.empty_like(hidden_states)
        launch(self, hidden_states, kv, token_mask, ref, n)
        step = state["rows"]
        if 0 < step < n:
            bad = None
            for s in range(0, n, step):  # every slice, so all ranks make the same wkv calls
                e = min(s + step, n)
                if not bits_equal(torch, self.wkv(x[s:e]), kv[s:e]) and bad is None:
                    bad = (s, e)
            if bad is not None:
                state["rows"], state["disarmed"] = 0, True
                say(f"dsv41: engram wkv rows DISABLED: wkv on rows {bad[0]}:{bad[1]} of {n} differs bitwise "
                    f"from the full-M GEMM; the step runs the full GEMM")
        if state["inplace"]:
            launch(self, hidden_states, kv, token_mask, hidden_states, n)
            if not bits_equal(torch, hidden_states, ref):
                state["inplace"], state["disarmed"] = False, True
                say("dsv41: engram post in place DISABLED: in-place output differs bitwise from the stock "
                    "kernel's; the stock buffer is used")
                return ref
            return hidden_states
        return ref

    def forward(self, hidden_states, hash_ids, token_mask=None):
        num_tokens = hidden_states.shape[0]
        if (self.use_sequence_parallel or num_tokens <= MIN_ROWS or capturing(torch)
                or not (state["inplace"] or state["rows"])):
            return stock(self, hidden_states, hash_ids, token_mask)
        if state["verify"] > 0:
            state["verify"] -= 1
            out = verify(self, hidden_states, hash_ids, token_mask)
            if state["verify"] == 0 and not state["disarmed"]:
                say(f"dsv41: engram post self-check bit-exact on {VERIFY_CALLS} eager calls "
                    f"(in place {'on' if state['inplace'] else 'off'}, wkv rows {state['rows'] or 'full'})")
            return out
        output = hidden_states if state["inplace"] else torch.empty_like(hidden_states)
        step = state["rows"]
        if not 0 < step < num_tokens:
            kv = self.wkv(self.embed(hash_ids).flatten(-2))
            n = check_shapes(self, hidden_states, hash_ids, token_mask, kv)
            launch(self, hidden_states, kv, token_mask, output, n)
            return output
        x = self.embed(hash_ids).flatten(-2)
        n = check_shapes(self, hidden_states, hash_ids, token_mask)
        for s in range(0, n, step):
            e = min(s + step, n)
            kv = self.wkv(x[s:e])
            assert kv.ndim == 2 and kv.shape[1] == (self.hc_mult + 1) * self.dim
            launch(self, hidden_states[s:e], kv, None if token_mask is None else token_mask[s:e], output[s:e], e - s)
            kv = None
        return output

    forward._dsv41_engram_post = True
    forward.__wrapped__ = stock
    return forward, state


def install(cls=None, eng=None, env=None, torch=None) -> str:
    """Replace Engram.forward. 'engaged', 'already installed' or 'disabled: ...'."""
    import inspect

    if torch is None:
        import torch

    inplace, rows = settings(env)
    if eng is None:
        from vllm.models.deepseek_v4_1.common import engram as eng
    if cls is None:
        cls = eng.Engram
    if getattr(cls.forward, "_dsv41_engram_post", False):
        return "already installed"
    from attn_out_defer import fn_text

    why = None
    try:
        if fn_text(inspect.getsource(cls.forward)) != EXPECTED_FORWARD:
            why = "Engram.forward differs from the pinned e15 source"
        elif not hasattr(eng, "_fused_engram_post_wkv_kernel") or not hasattr(eng, "triton"):
            why = "engram module has no _fused_engram_post_wkv_kernel / triton"
    except (OSError, TypeError, SyntaxError, StopIteration) as exc:
        why = f"cannot read Engram.forward ({exc!r})"
    if why is not None:
        print(f"dsv41: engram post DISABLED: {why}; the stock forward stays", flush=True)
        return f"disabled: {why}"
    cls.forward, _ = make(cls, eng, torch, inplace, rows, say=lambda m: print(m, flush=True))
    print(f"dsv41: engram post engaged (in place {'on' if inplace else 'off'}, wkv rows "
          f"{rows or 'full'}, above {MIN_ROWS} eager rows, self-check {VERIFY_CALLS} calls)", flush=True)
    return "engaged"
