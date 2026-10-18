#!/usr/bin/env bash
# 1017 promoted boot: the release serve env written out, then ./run.sh.
# Every value equals the recipe.yaml serve.env default, so a plain ./run.sh
# boots the same serve (tests/test_recipe_ops.py test_defaults_match_promoted_boot).
# Boot D1 (2026-10-18, evidence boot-d1/) ran
#   SKIP_DOWNLOAD=1 AUDIT=strict KV_CACHE_MEMORY=4160749568 ./run.sh
# Match it with: SKIP_DOWNLOAD=1 AUDIT=strict results/2026-10-18-1017/boot.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

export IMAGE=dsv41-flash-exl3-sm121:e15-kbits
export SNAPSHOT_SHA=2.7bpw-mcg-hq
export KV_CACHE_MEMORY=4160749568
export MAX_MODEL_LEN=1048576
export MAX_NUM_SEQS=2
export UTIL=0.75
export MAX_NUM_BATCHED_TOKENS=8192
export NUM_SPECULATIVE_TOKENS=3

export NCCL_BUFFSIZE=1048576
export NCCL_LL128_BUFFSIZE=262144
export NCCL_PROTO='^LL128'
export NCCL_MAX_NCHANNELS=8

export DSV41_DROP_PAGE_CACHE=1
export DSV41_INDEXER_PREFILL_FACTOR=1
export DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192
export DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=4
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256

export DSV41_ENGRAM_DISK=1
export DSV41_ENGRAM_PREFETCH=1
export DSV41_ENGRAM_CENSUS=1
export DSV41_ENGRAM_GATHER_V2=1
export DSV41_ENGRAM_WILLNEED=1
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

exec ./run.sh
