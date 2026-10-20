#!/usr/bin/env bash
# PCP1 cells against the boot-PCP1.sh serve: samplers (memsample, procmem, pcp_watch, live logs; both nodes)
# and the smokes, FINAL's micro block (benches/micro.sh --contexts 4096 16384 65536 --runs 2), then three
# novel 64K prompts back to back (pp_chunks.py), then logs (docker logs both ranks, NVRM lines, samplers
# stopped). Run memwatch2.sh alongside, as in FINAL.
#   W=<worktree> setsid -f bash cells-PCP1.sh <ARM_DIR> > <ARM_DIR>/cells.log 2>&1
set -u
W=${W:?worktree checked out at perf2/validate-pcp}
O=${1:?arm dir}
T=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
mkdir -p "$O"
mark() { echo "$(date -u +%FT%TZ) $1" | tee -a "$O/phases.txt"; }
DSV41_RECIPE=$W bash "$T/pm_cells.sh" "$O" sample smoke
mark "micro start"
(cd "$W" && bash benches/micro.sh --contexts 4096 16384 65536 --runs 2 > "$O/micro.txt" 2>&1)
mark "micro end"
DSV41_RECIPE=$W bash "$T/pm_cells.sh" "$O" pp64kx3 logs
