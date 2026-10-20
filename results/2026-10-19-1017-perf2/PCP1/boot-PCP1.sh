#!/usr/bin/env bash
# PCP1 validation boot (prefill-mem round 4, after FINAL): FINAL's tree (release tip 4ae29a6, pr/10-evidence)
# + the PR 6 change d98ee22 (the empty-cache hook counts the kernel's per-CPU free page lists)
# = branch perf2/validate-pcp @2c8e9ed. Plain defaults as FINAL; DSV41_PREFILL_MEM_LOG=1 only adds one log
# line per prefill step (raw MemAvailable, the lists, available, the action).
#   W=<worktree checked out at perf2/validate-pcp> setsid -f bash boot-PCP1.sh > <ARM>/boot.txt 2>&1
set -euo pipefail
W=${W:?worktree checked out at perf2/validate-pcp}
cd "$W"
[ "$(git rev-parse --short=7 HEAD)" = 2c8e9ed ] || { echo "HEAD is $(git rev-parse --short=7 HEAD), not 2c8e9ed" >&2; exit 1; }
export SKIP_DOWNLOAD=1 AUDIT=strict
export DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP=1 DSV41_PREFILL_MEM_LOG=1
exec ./run.sh
