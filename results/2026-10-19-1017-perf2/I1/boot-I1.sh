#!/usr/bin/env bash
# perf2 integration boot I1 (RUNSHEET.md): everything on except dspark adaptive.
# + torch profiler endpoint for the sheet's PROFILE cells (stacks off, dump off: the only allowed profiler config;
#   idle-cost-free per Sept P3 / B0). Run from the integration worktree.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces","torch_profiler_dump_cuda_time_total":false,"torch_profiler_with_stack":false}'
exec ./run.sh
