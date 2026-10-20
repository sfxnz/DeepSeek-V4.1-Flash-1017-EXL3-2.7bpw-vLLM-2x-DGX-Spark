#!/usr/bin/env bash
# perf2 integration phase-2 boot D2: BASE (I4 minus router gemv / main_proj overlap / native poll; FLOOR 3) + arm knobs. integration3 worktree.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration3
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_ROUTER_GEMV=0 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=0 DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB=3 DSV41_PREFILL_MEM_LOG=1
export NUM_SPECULATIVE_TOKENS=5 DSV41_DSPARK_ADAPTIVE=1
exec ./run.sh
