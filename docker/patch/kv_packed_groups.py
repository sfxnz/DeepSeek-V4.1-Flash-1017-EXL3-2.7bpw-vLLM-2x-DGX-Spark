"""Packed KV group sizing: vLLM PR #60996 on the image's planner (DSV41_KV_PACKED_GROUPS=1).

With the block-outermost (BLHNC) layout every pool block is as wide as the
widest KV-cache group. e15's _get_packed_kv_cache_groups splits the layer
buckets with _approximate_gcd([43, 3, 3, 1], lower_bound=3) = 3 pattern repeats
per group, so with the recipe's page pins (SWA 64, ratio-2 manager 128 via
sm120_page.extra_page_bump) the ratio-2 group (layers 2, 8, 14: MLA + indexer)
sets a 138,240-byte block and the ratio-1 group (layer 20, 46,080 bytes of
pages) pays a full block per 64 tokens: 18 groups, 3,240 pool bytes per context
token for 1,800 bytes of content.

Upstream PR #60996 ("Size packed groups by pooled bytes per request, not layer
padding", open, vllm-project/vllm) replaces _approximate_gcd with a search over
1..max repeats per group that minimises _max_memory_usage_bytes_from_groups, the
bytes a max-length request draws from the shared pool. On this model it picks
one repeat per group: 50 groups (43 SWA of one layer, 3 ratio-2 MLA+indexer, 3
compressor rings, 1 ratio-1 MLA+indexer), 46,080-byte blocks. Per-layer page
sizes, kernels and block sizes are unchanged; only the grouping and the pool
block width move. CPU, the image's own planner with the recipe pins
(perf2/upstream/evidence/kv_sim_pr60996.txt): 3.875 GiB holds 1,817,276 tokens at
max-model-len 1M (1.73x; stock 1,108,848, 1.06x), 1,497,155 at 524,288 (2.86x; stock
975,641, 1.86x) and 1,471,832 at 500,000 (2.94x; stock 964,308, 1.93x).

The rewrite is the PR's v1/core/kv_cache_utils.py change, hunk for hunk (H1-H4,
cut from the PR applied to e15's file; tests/test_kv_packed_groups.py rebuilds
the PR text from the e15 pin), plus one INFO line in the EngineCore log after
the split (LOG_PLANNER: group count and repeats per group). The PR's second
commit (batched sliding-window metadata) is swa_meta_batched.py here.

Applied from sitecustomize at interpreter start (required: an anchor miss is
image drift and fails the boot under DSV41_PATCH_STRICT=1). Every hunk must
match exactly once. Idempotent (MARK). DSV41_KV_PACKED_GROUPS=0 keeps the
image's planner; the boot refuses a container whose file was already rewritten
then. Top-level imports are stdlib only.
"""

from __future__ import annotations

import os
import sys

MARK = "dsv41-kv-packed-groups"
LOG_ENGAGED = "dsv41: kv packed groups engaged"
LOG_DISARMED = "dsv41: kv packed groups FAILED"
LOG_PLANNER = "dsv41-kv-packed-groups: %d KV cache groups, %s pattern repeats per group"
REL = "v1/core/kv_cache_utils.py"
MODULE = "vllm.v1.core.kv_cache_utils"

# PR #60996 kv_cache_utils.py hunks, cut from the PR applied to e15's file (perf2/upstream/notes/pr60996/e15).
H1_OLD = '''def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


'''
H1_NEW = ""

H2_OLD = '''    repeats (picked by ``_approximate_gcd`` to minimize padding), so all
    groups pack into the same per-block layout. Mamba buckets are additionally
    split to fit the block the attention buckets already need.
    Returns None when the layout is not block-outermost or all layers already
    share one page size.
'''
H2_NEW = '''    repeats, picked to minimize the bytes a max-length request draws from the
    shared pool, so all groups pack into the same per-block layout. Mamba
    buckets are additionally split to fit the block the attention buckets
    already need. Returns None when the layout is not block-outermost or all
    layers already share one page size.
'''

H3_OLD = '''    # Balanced buckets that mix page sizes must stay whole, so the largest one
    # sets a floor on the repeats per group; larger single-size buckets are
    # split down toward it. No such bucket means nothing needs packing.
    min_repeats_per_group = max(
        (
            spec.get_max_layers_per_page_size()
            for spec, page_size_layers, balanced in bucketed
            if balanced and len(page_size_layers) > 1
        ),
        default=0,
    )
    repeats_per_group = (
        _approximate_gcd(
            [
                spec.get_max_layers_per_page_size()
                for spec, _, balanced in bucketed
                if balanced
            ],
            lower_bound=min_repeats_per_group,
        )
        if min_repeats_per_group
        else None
    )

    def num_groups_for(spec: UniformTypeKVCacheSpecs, balanced: bool) -> int:
        if balanced and repeats_per_group is not None:
            return cdiv(spec.get_max_layers_per_page_size(), repeats_per_group)
'''
H3_NEW = '''    def num_groups_for(
        spec: UniformTypeKVCacheSpecs, balanced: bool, repeats: int | None
    ) -> int:
        if balanced and repeats is not None:
            return cdiv(spec.get_max_layers_per_page_size(), repeats)
'''

H4_OLD = '''    # Bytes a block must hold however the state buckets end up split: mamba,
    # circular-buffer and unsplit sliding-window buckets can go down to one
    # state per group, every other bucket's split is fixed by the repeat pattern.
    def is_state_bucket(spec: UniformTypeKVCacheSpecs) -> bool:
        if isinstance(spec.first_spec, (MambaSpec, CircularBufferSpec)):
            return True
        return repeats_per_group is None and isinstance(
            spec.first_spec, SlidingWindowSpec
        )

    anchor_bytes = max(
        (
            widest_group_bytes(
                page_size_layers,
                len(spec.kv_cache_specs)
                if is_state_bucket(spec)
                else num_groups_for(spec, balanced),
            )
            for spec, page_size_layers, balanced in bucketed
        ),
        default=0,
    )

    groups = []
    for spec, page_size_layers, balanced in bucketed:
        num_groups = num_groups_for(spec, balanced)
        # Cap a state group at the states a block already fits rather than let
        # it widen the block.
        if anchor_bytes and is_state_bucket(spec):
            states_per_block = max(anchor_bytes // spec.first_spec.page_size_bytes, 1)
            num_groups = max(
                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
            )
        if num_groups == 1:
            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))
            continue

        pattern_repeats = list(zip(*page_size_layers.values()))
        for i in range(num_groups):
            group_layer_names = [
                name for repeat in pattern_repeats[i::num_groups] for name in repeat
            ]
            group_layer_specs = {
                name: spec.kv_cache_specs[name] for name in group_layer_names
            }
            group_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert group_spec is not None
            groups.append(KVCacheGroupSpec(group_layer_names, group_spec))
'''
H4_NEW = '''    # A recycling bucket that the repeat pattern does not split may still be
    # split so that one of its states fills a block.
    def is_state_bucket(spec: UniformTypeKVCacheSpecs, repeats: int | None) -> bool:
        if isinstance(spec.first_spec, (MambaSpec, CircularBufferSpec)):
            return True
        return repeats is None and isinstance(spec.first_spec, SlidingWindowSpec)

    def anchor_bytes(repeats: int | None) -> int:
        """Widest block the state buckets could need under a split choice.

        Mamba, circular-buffer and unsplit sliding-window buckets can go down to
        one state per group; every other bucket's width is fixed by the repeat
        pattern.
        """
        return max(
            (
                widest_group_bytes(
                    page_size_layers,
                    len(spec.kv_cache_specs)
                    if is_state_bucket(spec, repeats)
                    else num_groups_for(spec, balanced, repeats),
                )
                for spec, page_size_layers, balanced in bucketed
            ),
            default=0,
        )

    def split_plan(repeats: int | None) -> list[int]:
        """The number of groups each bucket splits into."""
        anchor = anchor_bytes(repeats)
        plan = []
        for spec, _, balanced in bucketed:
            num_groups = num_groups_for(spec, balanced, repeats)
            # Cap a state group at the states a block already fits rather than
            # let it widen the block.
            if anchor and is_state_bucket(spec, repeats):
                states_per_block = max(anchor // spec.first_spec.page_size_bytes, 1)
                num_groups = max(
                    num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
                )
            plan.append(num_groups)
        return plan

    # Every group claims its blocks from one shared pool, so a block is as wide
    # as the widest group's page and the narrower groups only pad out the rest
    # of it. Splitting a bucket into more, narrower groups shrinks that block,
    # but every group claims its own blocks for a request - and a recycling
    # group (sliding window, circular buffer, mamba) reserves as many blocks for
    # one layer as for many - so the block count grows with the group count.
    # Price both against the bytes a max-length request draws from the pool.
    max_repeats_per_group = max(
        (
            spec.get_max_layers_per_page_size()
            for spec, _, balanced in bucketed
            if balanced
        ),
        default=0,
    )
    needs_packing = any(
        balanced and len(page_size_layers) > 1
        for _, page_size_layers, balanced in bucketed
    )

    def build_groups(repeats: int | None) -> list[KVCacheGroupSpec]:
        """The groups a split choice produces."""
        groups: list[KVCacheGroupSpec] = []
        for (spec, page_size_layers, _), num_groups in zip(
            bucketed, split_plan(repeats)
        ):
            if num_groups == 1:
                groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))
                continue

            pattern_repeats = list(zip(*page_size_layers.values()))
            for i in range(num_groups):
                group_layer_names = [
                    name for repeat in pattern_repeats[i::num_groups] for name in repeat
                ]
                group_layer_specs = {
                    name: spec.kv_cache_specs[name] for name in group_layer_names
                }
                group_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
                assert group_spec is not None
                groups.append(KVCacheGroupSpec(group_layer_names, group_spec))
        return groups

    # Price a split by what the allocator actually charges for it: every group
    # claims its own blocks from the shared pool, and a block is rounded up to
    # the layers' stride alignment, which is what
    # ``_max_memory_usage_bytes_from_groups`` computes.
    repeats_per_group = (
        min(
            range(1, max_repeats_per_group + 1),
            key=lambda repeats: (
                _max_memory_usage_bytes_from_groups(vllm_config, build_groups(repeats)),
                -repeats,
            ),
        )
        if needs_packing
        else None
    )

    groups = build_groups(repeats_per_group)
'''

# One INFO line in the EngineCore log after the split (not part of the PR).
LOG_OLD = """    groups = build_groups(repeats_per_group)
"""
LOG_NEW = """    groups = build_groups(repeats_per_group)
    logger.info_once(  # dsv41-kv-packed-groups
        "dsv41-kv-packed-groups: %d KV cache groups, %s pattern repeats per group",
        len(groups),
        repeats_per_group,
    )
"""

PR_HUNKS = (
    (H1_OLD, H1_NEW, "_approximate_gcd removed"),
    (H2_OLD, H2_NEW, "_get_packed_kv_cache_groups docstring"),
    (H3_OLD, H3_NEW, "repeats_per_group / num_groups_for"),
    (H4_OLD, H4_NEW, "pooled-bytes split search"),
)


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("DSV41_KV_PACKED_GROUPS", "0") or "0") == "1"


def _sub(src: str, old: str, new: str, label: str) -> str:
    if src.count(old) != 1:
        raise ValueError(f"{MARK}: {label}: expected 1 match, found {src.count(old)}")
    return src.replace(old, new, 1)


def apply_hunks(src: str, hunks) -> str:
    for old, new, label in hunks:
        src = _sub(src, old, new, label)
    return src


def patch_source(src: str) -> str:
    """vllm/v1/core/kv_cache_utils.py -> PR #60996 packed grouping. Idempotent."""
    if MARK in src:
        return src
    return _sub(apply_hunks(src, PR_HUNKS), LOG_OLD, LOG_NEW, "planner log line")


def install(vllm_root, rewrite, modules=None) -> str:
    """Rewrite the planner with sitecustomize's `rewrite(path, fn)`; print the engagement line."""
    modules = sys.modules if modules is None else modules
    try:
        done = rewrite(os.path.join(str(vllm_root), REL), patch_source)
        if done == "rewritten" and MODULE in modules:
            raise RuntimeError(f"{MODULE} was imported before the rewrite; this process plans with the stock code")
    except BaseException as exc:
        print(f"dsv41: kv packed groups FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
    print(f"dsv41: kv packed groups engaged (kv_cache_utils {done})", file=sys.stderr, flush=True)
    return f"kv_cache_utils {done}"


def refuse_stale(vllm_root) -> None:
    """DSV41_KV_PACKED_GROUPS=0 on a container whose planner is already rewritten."""
    path = os.path.join(str(vllm_root), REL)
    if os.path.isfile(path):
        with open(path) as fh:
            if MARK in fh.read():
                raise RuntimeError(f"{path} carries {MARK} but DSV41_KV_PACKED_GROUPS is not 1")
