"""Attention output buffer allocated after the indexer in eager prefill steps (DSV41_ATTN_OUT_DEFER=1). Bit-identical.

DeepseekV4Attention.forward (models/deepseek_v4_1/attention.py) allocates the attention output
o_padded [T, 32, 512] bf16 (256 MiB per rank at T = 8190) before the input projections, so it is
live through the wq_b output (256 MiB), the fused q-norm/RoPE that returns a new padded q
(256 MiB) and the indexer's logits (up to VLLM_SPARSE_INDEXER_MAX_LOGITS_MB). It also keeps the
projection outputs (qr_kv, qr, kv, kv_score, indexer weights; ~110 MiB at T = 8190) to the end,
through o_proj, although nothing reads them after the attention kernel.

For an eager call with more than MIN_ROWS tokens (prefill; decode steps and CUDA-graph capture keep
the stock forward, whose eager break needs the caller-provided buffer), the replacement:
- runs the same projections and `_prepare_and_attn_fn` with a Deferred placeholder for o_padded;
- `_sparse_indexer_and_attn` runs the indexer, then allocates o_padded (the same torch.empty) right
  before forward_mqa, which writes all of it as in the stock path;
- drops the projection outputs before o_proj.
Same calls in the same order on the same inputs; only when two buffers are allocated and freed
changes. Estimated -256 MiB at the q-norm and indexer moments and -110 MiB at o_proj (T = 8190).

Runtime patch (sitecustomize). Refuses (LOG_DISARMED) unless the image's forward and
_sparse_indexer_and_attn match EXPECTED_FORWARD / EXPECTED_SIA (ast.unparse, docstring and
decorators dropped). DSV41_ATTN_OUT_DEFER=0 keeps the stock forward. Top-level imports are stdlib only.
"""

from __future__ import annotations

import os

MARK = "dsv41: attention out defer"
LOG_ENGAGED = "dsv41: attention out defer engaged"
LOG_DISARMED = "dsv41: attention out defer DISABLED"
MIN_ROWS = 64  # eager steps above this many tokens are prefill chunks; decode batches are <= 16

EXPECTED_FORWARD = """def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, llama_4_scaling: torch.Tensor | None=None) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    o_padded = torch.empty((num_tokens, self.padded_heads, self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
    qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
    self._prepare_and_attn_fn(hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, o_padded)
    o = o_padded[:, :self.n_local_heads, :]
    return self._o_proj(o, positions)"""
EXPECTED_SIA = """def _sparse_indexer_and_attn(self, hidden_states: torch.Tensor, index_q: torch.Tensor | None, index_q_scale: torch.Tensor | None, index_weights: torch.Tensor | None, q: torch.Tensor, kv: torch.Tensor, positions: torch.Tensor, out: torch.Tensor) -> None:
    if self.indexer is not None and index_q is not None:
        assert index_weights is not None
        q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
        self.indexer.indexer_op(hidden_states, q_quant, None, index_weights)
    self.forward_mqa(q, kv, positions, out)"""


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_ATTN_OUT_DEFER", "0") == "1"


def fn_text(src: str) -> str:
    """ast.unparse of the first function in `src`, docstring and decorators dropped."""
    import ast
    import textwrap

    node = next(n for n in ast.walk(ast.parse(textwrap.dedent(src))) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    node.body, node.decorator_list = body, []
    return ast.unparse(node)


class Deferred:
    """Stands in for o_padded until _sparse_indexer_and_attn allocates it."""

    __slots__ = ("args", "tensor")

    def __init__(self, torch, shape, dtype, device) -> None:
        self.args = (torch, shape, dtype, device)
        self.tensor = None

    def materialize(self):
        if self.tensor is None:
            torch, shape, dtype, device = self.args
            self.tensor = torch.empty(shape, dtype=dtype, device=device)
        return self.tensor


def capturing(torch) -> bool:
    """A CUDA stream capture or a breakable-cudagraph capture is in progress."""
    if torch.cuda.is_current_stream_capturing():
        return True
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        cap = BreakableCUDAGraphCapture.current()
        return cap is not None and bool(getattr(cap, "_capturing", False))
    except Exception:
        return False


def make(cls, torch):
    """(forward, _sparse_indexer_and_attn) replacements for DeepseekV4Attention `cls`."""
    stock_forward = cls.forward
    stock_sia = cls._sparse_indexer_and_attn

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        num_tokens = hidden_states.shape[0]
        if num_tokens <= MIN_ROWS or capturing(torch):
            return stock_forward(self, positions, hidden_states, llama_4_scaling)
        qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        del qr_kv
        out = Deferred(torch, (num_tokens, self.padded_heads, self.head_dim), hidden_states.dtype, hidden_states.device)
        self._prepare_and_attn_fn(hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, out)
        del qr, qr_scale, kv, kv_score, indexer_weights
        o_padded = out.materialize()
        out = None
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    def _sparse_indexer_and_attn(self, hidden_states, index_q, index_q_scale, index_weights, q, kv, positions, out):
        if not isinstance(out, Deferred):
            return stock_sia(self, hidden_states, index_q, index_q_scale, index_weights, q, kv, positions, out)
        if self.indexer is not None and index_q is not None:
            assert index_weights is not None
            q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
            self.indexer.indexer_op(hidden_states, q_quant, None, index_weights)
        self.forward_mqa(q, kv, positions, out.materialize())

    forward._dsv41_attn_out_defer = True
    _sparse_indexer_and_attn._dsv41_attn_out_defer = True
    forward.__wrapped__ = stock_forward
    _sparse_indexer_and_attn.__wrapped__ = stock_sia
    return forward, _sparse_indexer_and_attn


def install(cls=None, torch=None) -> str:
    """Replace DeepseekV4Attention.forward / _sparse_indexer_and_attn. 'engaged', 'already installed' or 'disabled: ...'."""
    import inspect

    if torch is None:
        import torch

    if cls is None:
        from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention as cls
    if getattr(cls.forward, "_dsv41_attn_out_defer", False):
        return "already installed"
    why = None
    try:
        if fn_text(inspect.getsource(cls.forward)) != EXPECTED_FORWARD:
            why = "DeepseekV4Attention.forward differs from the pinned e15 source"
        elif fn_text(inspect.getsource(inspect.unwrap(cls._sparse_indexer_and_attn))) != EXPECTED_SIA:
            why = "DeepseekV4Attention._sparse_indexer_and_attn differs from the pinned e15 source"
    except (OSError, TypeError, SyntaxError, StopIteration) as exc:
        why = f"cannot read the attention source ({exc!r})"
    if why is not None:
        print(f"dsv41: attention out defer DISABLED: {why}; the stock forward stays", flush=True)
        return f"disabled: {why}"
    cls.forward, cls._sparse_indexer_and_attn = make(cls, torch)
    print(f"dsv41: attention out defer engaged (o_padded after the indexer above {MIN_ROWS} eager rows)", flush=True)
    return "engaged"
