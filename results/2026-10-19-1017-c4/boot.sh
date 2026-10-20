#!/usr/bin/env bash
# 1017 promoted boot of the perf2 stack: the serve env written out, then ./run.sh.
# Every value equals the recipe.yaml serve.env default, so a plain ./run.sh boots the same serve, and every
# DSV41_* lever default is exported here (tests/test_recipe_ops.py test_defaults_match_promoted_boot).
# Against the D1/r1 release boot (results/2026-10-18-1017/boot.sh):
#   - perf2 kv-pack: MAX_NUM_SEQS 2 -> 4, MAX_MODEL_LEN 1048576 -> 500000, packed KV groups
#     (DSV41_KV_PACKED_GROUPS=1, vLLM PR #60996), batched SWA metadata (DSV41_SWA_META_BATCHED=1), the lazy
#     indexer expanded block table (DSV41_INDEXER_EXPANDED_TABLE_LAZY=1), --mm-processor-cache-gb 1,
#     --prefix-cache-retention-interval 4096. 500,000 because 4 x 500K fits the pool at B=8192 without
#     preemption and 4 x 524,288 does not (CPU: 3.82 vs 3.99 GiB); B=4096 would allow 4 x 524,288. KV pool
#     unchanged (3.875 GiB).
#   - perf2 c4-kernels: IMAGE e15-kbits -> e16-rows (native decode MoE for 9..24 rows).
#   - perf2 prefill-moe: DSV41_MOE_PREFILL_TC=1 (tensor-core routed-MoE prefill kernel, prebuilt in e16-rows).
#   - perf2 engram-io: DSV41_ENGRAM_PREFILL_AHEAD=1 (Engram prefill read-ahead, _DONTNEED=1).
#   - perf2 decoder-skip: DSV41_DECODER_REPLAY_ROWS=2670 (exact CED decoder replay), DSV41_PREFIX_END_FALLBACK=1,
#     DSV41_PREFIX_REPLY_RETAIN=1.
#   - perf2 prefill-mem: DSV41_PREFILL_EMPTY_CACHE_HOLD=1, DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB=3,
#     DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP=1 (MemAvailable + the kernel's per-CPU free page lists),
#     DSV41_PREFILL_EMPTY_CACHE_GROWTH_MIB=150, DSV41_FAT_SCRATCH_SHARED=1, DSV41_MODEL_LOOP_RELEASE=1,
#     DSV41_ATTN_OUT_DEFER=1, DSV41_ENGRAM_POST_INPLACE=1 (spark1 headroom with decoder replay off is not
#     solved: flags.md).
#   - perf2 dspark-adaptive: NUM_SPECULATIVE_TOKENS_PER_BATCH=1-2:5,3-4:3 (DSpark block 5; verify k=5 at 1-2
#     requests, k=3 at 3-4) and DSV41_DSPARK_ADAPTIVE=1 (TAU 0.15). NUM_SPECULATIVE_TOKENS=3 applies only
#     with NUM_SPECULATIVE_TOKENS_PER_BATCH=0.
# GPU evidence: results/2026-10-19-1017-perf2/ (boot I3 ran the kv-pack defaults with the later perf2 levers
# off). Boot FINAL ran the full stack's defaults (the last of these PRs) with the 4-session resident soak.
# Run with: SKIP_DOWNLOAD=1 AUDIT=strict results/2026-10-19-1017-c4/boot.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

export IMAGE=dsv41-flash-exl3-sm121:e16-rows
export SNAPSHOT_SHA=2.7bpw-mcg-hq
export KV_CACHE_MEMORY=4160749568
export DSV41_KV_PACKED_GROUPS=1
export MAX_MODEL_LEN=500000
export MAX_NUM_SEQS=4
export UTIL=0.75
export MAX_NUM_BATCHED_TOKENS=8192
export NUM_SPECULATIVE_TOKENS_PER_BATCH=1-2:5,3-4:3
export MM_PROCESSOR_CACHE_GB=1
export PREFIX_CACHE_RETENTION_INTERVAL=4096

export NCCL_BUFFSIZE=1048576
export NCCL_LL128_BUFFSIZE=262144
export NCCL_PROTO='^LL128'
export NCCL_MAX_NCHANNELS=8

export DSV41_DROP_PAGE_CACHE=1
export DSV41_INDEXER_PREFILL_FACTOR=1
export DSV41_INDEXER_EXPANDED_TABLE_LAZY=1
export DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192
export DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=4
export DSV41_PREFILL_EMPTY_CACHE_HOLD=1
export DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB=3
export DSV41_PREFILL_EMPTY_CACHE_GROWTH_MIB=150
export DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP=1
export DSV41_FAT_SCRATCH_SHARED=1
export DSV41_MODEL_LOOP_RELEASE=1
export DSV41_ATTN_OUT_DEFER=1
export DSV41_ENGRAM_POST_INPLACE=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256

export DSV41_ENGRAM_DISK=1
export DSV41_ENGRAM_PREFETCH=1
export DSV41_ENGRAM_CENSUS=1
export DSV41_ENGRAM_GATHER_V2=1
export DSV41_ENGRAM_WILLNEED=1
export DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_LMHEAD_MXFP8=1

# Round-34 levers.
export DSV41_STREAM_FEED=1
export DSV41_WOA_PREPACK=1
export DSV41_DSPARK_SPARSE_MARKOV=1
# Round-35 levers.
export DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024
export DSV41_P2B_COOP=2
export DSV41_DENSE_GEMV=1
export DSV41_MHC_DET_SPLITS=16
export DSV41_ENGRAM_NATIVE_STAGE=1
export DSV41_ENGRAM_EARLY_HASH=1
export DSV41_ATTN_T2R_DEDUP=1
export DSV41_SWA_META_FUSED=1
export DSV41_MOE_PREP_FUSED=1
export DSV41_CANDIDATE_MASK_BOUNDED=1
export DSV41_INDEXER_WP_GEMV=1
# perf2 kv-pack.
export DSV41_SWA_META_BATCHED=1
# perf2 prefill-moe.
export DSV41_MOE_PREFILL_TC=1
# perf2 decoder-skip.
export DSV41_DECODER_REPLAY_ROWS=2670
export DSV41_PREFIX_END_FALLBACK=1
export DSV41_PREFIX_REPLY_RETAIN=1
# perf2 dspark-adaptive.
export DSV41_DSPARK_ADAPTIVE=1

exec ./run.sh
