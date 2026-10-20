#!/usr/bin/env bash
# perf2 PROBE arm (main, 2026-10-20 ~03:2xZ): worktree dsv41-1017-probe (perf2/prefill-mem-int4 @d1fa19d = integration4 746ed86
# + dormant prefill-mem probe). D4ENV + DSV41_DECODER_REPLAY_ROWS=0 (reproduces the chunk-1 loss) + DSV41_PREFILL_MEM_PROBE=1.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-probe
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_ROUTER_GEMV=0 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=0 DSV41_PREFILL_MEM_LOG=1
export NUM_SPECULATIVE_TOKENS_PER_BATCH=1-2:5,3-4:3 DSV41_DSPARK_ADAPTIVE=1 DSV41_SCHED_PREFILL_CAP=2048 DSV41_SCHED_DECODE_STEPS=1
export DSV41_DECODER_REPLAY_ROWS=0 DSV41_PREFILL_MEM_PROBE=1
exec ./run.sh
