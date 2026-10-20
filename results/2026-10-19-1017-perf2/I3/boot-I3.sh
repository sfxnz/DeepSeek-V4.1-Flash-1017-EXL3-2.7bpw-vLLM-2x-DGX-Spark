#!/usr/bin/env bash
# perf2 integration boot I3 (RUNSHEET.md): I1 + replay off + moe tc off + p2b wide off + engram prefill-ahead off
# (kv-pack-only stack on e16-rows). No profiler.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration
export SKIP_DOWNLOAD=1 AUDIT=strict IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1
export DSV41_DECODER_REPLAY_ROWS=0 DSV41_MOE_PREFILL_TC=0 DSV41_P2B_WIDE=0 DSV41_ENGRAM_PREFILL_AHEAD=0
exec ./run.sh
