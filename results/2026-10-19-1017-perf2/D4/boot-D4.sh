#!/usr/bin/env bash
# perf2 phase 3 boot D4 (RUNSHEET "D4 (FIRST boot of phase 3 ...)"): integration4 @746ed86; per-batch DSpark depth
# (k=5 at batch 1-2, k=3 at 3-4), adaptive verify tau 0.15, C1 scheduling cap. FLOOR default 3 on this tree.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_ROUTER_GEMV=0 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=0 DSV41_PREFILL_MEM_LOG=1
export NUM_SPECULATIVE_TOKENS_PER_BATCH=1-2:5,3-4:3 DSV41_DSPARK_ADAPTIVE=1 DSV41_SCHED_PREFILL_CAP=2048 DSV41_SCHED_DECODE_STEPS=1
exec ./run.sh
