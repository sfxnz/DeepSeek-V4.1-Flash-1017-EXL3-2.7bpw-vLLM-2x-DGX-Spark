"""Ragged decode batches in the sparse indexer: vLLM 0.32rc1's guard on e15 (required, always on).

e15 vllm/model_executor/layers/sparse_attn_indexer.py, decode branch: when the
metadata's requires_padding is False it reshapes
q_quant[:num_decode_tokens] to (num_decodes, -1, ...) and unpacks nothing.
requires_padding is min(decode_lens_cpu) != max(decode_lens_cpu) over the
decode requests (v1/attention/backends/mla/indexer.py _prepare_decode_tensors,
native path), while num_decode_tokens is the metadata's token count
(split_decodes_and_prefills returns num_actual_tokens in its fast path). A
batch whose decode lengths look uniform but whose token count is not a multiple
of its decode-request count makes that reshape raise ("shape '[N, -1, ...]' is
invalid for input of size ...") and kills the worker. Upstream saw it on SM120
TP=2 as 8 tokens over 6 sequences in a warmup/mixed batch; the fix landed in
0.32rc1 (sparse_attn_indexer.py:795-808): needs_padded_path = requires_padding
or num_decode_tokens % num_decodes != 0, used by both the pack and the matching
unpack branch. Not reproduced here. In the e15 paths read for the backport,
eager/PIECEWISE metadata carries the real token count and FULL graphs pad with
zero-length requests (which already makes the lengths ragged), so a crashing
batch needs a token count the CPU decode lengths do not account for. Where the
count divides, the guard is False and the code path is the stock one; it only
turns the reshape error into the pack/unpack path ragged batches already take.
MAX_NUM_SEQS=4 adds batch shapes (12/16 verify rows, capture sizes 9/12/16)
that no 1017 boot has served.

Source rewrite of the image's file from sitecustomize (required: an anchor miss
fails the boot under DSV41_PATCH_STRICT=1). Each anchor must match once.
Idempotent (MARK). Top-level imports are stdlib only.
"""

from __future__ import annotations

MARK = "dsv41-indexer-ragged-decode"
REL = "model_executor/layers/sparse_attn_indexer.py"

GUARD_OLD = """        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
"""
GUARD_NEW = """        decode_lens = decode_metadata.decode_lens
        # dsv41-indexer-ragged-decode (vLLM 0.32rc1 sparse_attn_indexer.py:795-808):
        # uniform decode_lens with a token count that is not a multiple of the
        # decode requests would crash the uniform reshape below; pad instead.
        # Shared by the pack branch and the matching unpack branch.
        needs_padded_path = decode_metadata.requires_padding or (
            num_decode_tokens % decode_lens.shape[0] != 0
        )
        if num_decode_tokens == 0:
"""
PACK_OLD = """        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
"""
PACK_NEW = """        elif needs_padded_path:
            # pad in edge case where we have short chunked prefill length <
"""
UNPACK_OLD = """        if decode_metadata.requires_padding:
            # if padded, we need to unpack
"""
UNPACK_NEW = """        if needs_padded_path:
            # if padded, we need to unpack
"""
HUNKS = (
    (GUARD_OLD, GUARD_NEW, "decode needs_padded_path guard"),
    (PACK_OLD, PACK_NEW, "decode pack branch"),
    (UNPACK_OLD, UNPACK_NEW, "decode unpack branch"),
)


def patch_source(src: str) -> str:
    """vllm/model_executor/layers/sparse_attn_indexer.py -> ragged decode guard. Idempotent."""
    if MARK in src:
        return src
    for old, new, label in HUNKS:
        if src.count(old) != 1:
            raise ValueError(f"{MARK}: {label}: expected 1 match, found {src.count(old)}")
        src = src.replace(old, new, 1)
    return src
