#!/usr/bin/env bash
# perf2 integration phase-2 boot I5 third attempt: main_proj overlap OFF after its verify DISABLED on rank 1 (fail2). Run from integration3 worktree (= integration2 + c4-sched, knobs 0).
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration3
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_ROUTER_GEMV=1 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=1 DSV41_PREFILL_MEM_LOG=1 DSV41_DECODER_REPLAY_ROWS=0
exec ./run.sh
