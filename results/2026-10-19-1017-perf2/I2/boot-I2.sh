#!/usr/bin/env bash
# perf2 integration boot I2 (RUNSHEET.md): I1 + DSV41_DECODER_REPLAY_ROWS=0 (decoder replay off: A arm). No profiler.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1 DSV41_DECODER_REPLAY_ROWS=0
exec ./run.sh
