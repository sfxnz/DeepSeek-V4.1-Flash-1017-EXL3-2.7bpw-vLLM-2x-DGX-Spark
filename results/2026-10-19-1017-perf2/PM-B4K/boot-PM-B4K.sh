#!/usr/bin/env bash
# perf2 phase 3 boot PM-B4K (main go 2026-10-20T01:3xZ, item 1): D4ENV exactly + MAX_NUM_BATCHED_TOKENS=4096.
# KV unchanged (run.sh default KV_CACHE_MEMORY=4160749568) to isolate the chunk-size effect. FLOOR default 3 on this tree.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_ROUTER_GEMV=0 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=0 DSV41_PREFILL_MEM_LOG=1
export NUM_SPECULATIVE_TOKENS_PER_BATCH=1-2:5,3-4:3 DSV41_DSPARK_ADAPTIVE=1 DSV41_SCHED_PREFILL_CAP=2048 DSV41_SCHED_DECODE_STEPS=1
export MAX_NUM_BATCHED_TOKENS=4096
exec ./run.sh
