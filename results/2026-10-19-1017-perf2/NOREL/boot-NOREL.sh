#!/usr/bin/env bash
# perf2 boot NOREL (main, 2026-10-20 ~02:56Z): D4ENV exactly + DSV41_PREFILL_EMPTY_CACHE_TOKENS=100000000.
# prefill_empty_cache.py: the hook still installs (threshold > 0, per-step log kept by DSV41_PREFILL_MEM_LOG=1) but every
# action needs `longest >= threshold`, which no prompt reaches (max_model_len 500000), so no step ever calls empty_cache.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_ROUTER_GEMV=0 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=0 DSV41_PREFILL_MEM_LOG=1
export NUM_SPECULATIVE_TOKENS_PER_BATCH=1-2:5,3-4:3 DSV41_DSPARK_ADAPTIVE=1 DSV41_SCHED_PREFILL_CAP=2048 DSV41_SCHED_DECODE_STEPS=1
export DSV41_PREFILL_EMPTY_CACHE_TOKENS=100000000
exec ./run.sh
