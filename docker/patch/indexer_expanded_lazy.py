"""Indexer expanded block table on first use (DSV41_INDEXER_EXPANDED_TABLE_LAZY=1).

e15 DeepseekV32IndexerMetadataBuilder.__init__ (v1/attention/backends/mla/indexer.py:792) allocates
expanded_block_table_buffer = zeros(max_num_batched_tokens, block_table_width) int32 for every indexer
builder: B x (blocks of one max-length request) x 4 bytes. At max-model-len 500,000 and B=8192 that is
122 MiB per ratio-2 builder (width 3,907) and 244 MiB for the ratio-1 builder (width 7,814): 366 MiB per
rank with the image's KV groups, 611 MiB with packed groups (3 ratio-2 builders).

Its only readers (grep of the whole e15 vllm tree, the EXL3 plugin and docker/patch) are the flattened
branch of _prepare_decode_tensors (indexer.py:873-938, `if not use_native:`: the uniform-decode kernel
and the variable-length repeat_interleave). No class derives from the builder and nothing reads its
attributes dynamically. build() is the only caller (cudagraph capture goes through build too). On SM120
with DeepGEMM that branch does not run: use_native = not (use_flattening or supports_varlen) and
max_decode_len <= next_n and step_next_n_ok, where supports_varlen is SM100-only, use_flattening needs
_supports_native_decode(next_n) False (native_next_n_supported is True off SM90) or adaptive verification
with SM90 flattening, max_decode_len <= next_n holds by the builder's own decode split
(decode_threshold = next_n), and step_next_n_ok is True for the same reason. Prefill, the DSpark drafter
(no indexer layers), the ragged-decode guard and the replay / model_specific_attn_metadata path never
touch the buffer.

With the lever the builder no longer allocates it in __init__. A class property of the same name
allocates the stock zeros buffer on first read and logs one WARNING per builder (LOG_ALLOCATED): the
flattened path ran and its memory is back. Consumers are unchanged (they read the attribute). The
engagement audit treats LOG_ALLOCATED as a problem line, so a strict audit shows when the saving did
not hold.

Source rewrite of the image's file from sitecustomize, before anything imports vLLM (required when on:
an anchor miss fails the boot under DSV41_PATCH_STRICT=1). Each anchor must match once. Idempotent
(MARK). DSV41_INDEXER_EXPANDED_TABLE_LAZY=0 keeps the stock allocation; the boot refuses a container
whose file was already rewritten then. Top-level imports are stdlib only.
"""

from __future__ import annotations

import os
import sys

MARK = "dsv41-indexer-expanded-lazy"
LOG_ENGAGED = "dsv41: indexer expanded block table lazy engaged"
LOG_ALLOCATED = "dsv41: indexer expanded block table allocated on first use"
LOG_DISARMED = (
    "dsv41: indexer expanded block table lazy FAILED",
    "dsv41: indexer expanded block table allocated on first use",
)
REL = "v1/attention/backends/mla/indexer.py"
MODULE = "vllm.v1.attention.backends.mla.indexer"

PROP_OLD = """    reorder_batch_threshold: int | None = None
    requires_block_table_width = True
"""
PROP_NEW = """    reorder_batch_threshold: int | None = None
    requires_block_table_width = True

    # dsv41-indexer-expanded-lazy: only the flattened decode branch of _prepare_decode_tensors reads
    # this; allocate the stock zeros buffer on first read instead of in __init__.
    @property
    def expanded_block_table_buffer(self) -> torch.Tensor:
        buf = self._dsv41_expanded_block_table
        if buf is None:
            rows, cols = self._dsv41_expanded_shape
            logger.warning(
                "dsv41: indexer expanded block table allocated on first use "
                "(%d x %d int32, %.1f MiB, %s): the flattened decode path ran",
                rows,
                cols,
                rows * cols * 4 / 2**20,
                self.layer_names[0] if self.layer_names else "?",
            )
            buf = torch.zeros((rows, cols), dtype=torch.int32, device=self.device)
            self._dsv41_expanded_block_table = buf
        return buf
"""
ALLOC_OLD = """        self.expanded_block_table_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens, block_table_width),
            dtype=torch.int32,
            device=self.device,
        )
"""
ALLOC_NEW = """        # dsv41-indexer-expanded-lazy: allocated by the expanded_block_table_buffer property on first read.
        self._dsv41_expanded_shape = (scheduler_config.max_num_batched_tokens, block_table_width)
        self._dsv41_expanded_block_table = None
"""
HUNKS = (
    (PROP_OLD, PROP_NEW, "builder class attributes"),
    (ALLOC_OLD, ALLOC_NEW, "__init__ expanded_block_table_buffer allocation"),
)


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_INDEXER_EXPANDED_TABLE_LAZY", "0") or "0") == "1"


def patch_source(src: str) -> str:
    """vllm/v1/attention/backends/mla/indexer.py -> lazy expanded block table. Idempotent."""
    if MARK in src:
        return src
    for old, new, label in HUNKS:
        if src.count(old) != 1:
            raise ValueError(f"{MARK}: {label}: expected 1 match, found {src.count(old)}")
        src = src.replace(old, new, 1)
    return src


def install(vllm_root, rewrite, modules=None) -> str:
    """Rewrite the indexer with sitecustomize's `rewrite(path, fn)`; print the engagement line."""
    modules = sys.modules if modules is None else modules
    try:
        done = rewrite(os.path.join(str(vllm_root), REL), patch_source)
        if done == "rewritten" and MODULE in modules:
            raise RuntimeError(f"{MODULE} was imported before the rewrite; this process allocates eagerly")
    except BaseException as exc:
        print(f"dsv41: indexer expanded block table lazy FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
    print(f"dsv41: indexer expanded block table lazy engaged (indexer {done})", file=sys.stderr, flush=True)
    return f"indexer {done}"


def refuse_stale(vllm_root) -> None:
    """DSV41_INDEXER_EXPANDED_TABLE_LAZY=0 on a container whose indexer is already rewritten."""
    path = os.path.join(str(vllm_root), REL)
    if os.path.isfile(path):
        with open(path) as fh:
            if MARK in fh.read():
                raise RuntimeError(f"{path} carries {MARK} but DSV41_INDEXER_EXPANDED_TABLE_LAZY is not 1")
