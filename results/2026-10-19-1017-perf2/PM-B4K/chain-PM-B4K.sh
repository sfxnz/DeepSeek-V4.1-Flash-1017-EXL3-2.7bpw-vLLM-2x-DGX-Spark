#!/usr/bin/env bash
# PM-B4K cells (main go, phase 3 item 1), D4 order for the decode cells, micro after them, gen102k LAST (no retry after a trip).
# memsample (1 Hz) + procmem + live logs run from `sample` to `logs` so every cell has per-node minima.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/PM-B4K; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
PMT=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
bash $PMT/pm_cells.sh $O sample
bash $I/std.sh PM-B4K smoke c1short
up && { mark "lail5 start"; $T/run_phase.sh $O/lail.txt python3 tools/measure_lail_prose.py --runs 5; mark "lail5 end"; }
up && bash $I/std.sh PM-B4K c4distinct micro
if up; then bash $PMT/pm_cells.sh $O gen102k logs; else mark "serve down before gen102k: not run"; fi
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "PM-B4K chain done"
