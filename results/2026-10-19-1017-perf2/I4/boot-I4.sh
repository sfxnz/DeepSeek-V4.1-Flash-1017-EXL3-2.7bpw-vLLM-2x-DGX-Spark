#!/usr/bin/env bash
# perf2 integration phase-2 boot I4 (RUNSHEET.md). Run from integration2 worktree.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration2
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_ROUTER_GEMV=1 DSV41_MAIN_PROJ_OVERLAP=1 DSV41_ENGRAM_NATIVE_POLL=1 DSV41_PREFILL_MEM_LOG=1
exec ./run.sh
