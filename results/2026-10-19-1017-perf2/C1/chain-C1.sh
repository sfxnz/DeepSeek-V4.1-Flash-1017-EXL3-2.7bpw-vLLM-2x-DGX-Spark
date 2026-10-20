#!/usr/bin/env bash
# C1 per RUNSHEET c4-sched: cells.sh C1 <integration3>; quick quality; compares vs C0.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/C1; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration3; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W; R=$T/run_phase.sh
CS=/home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/runs
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
mark "C1 cells start"
bash /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/cells.sh C1 $W > $CS/C1-cells.log 2>&1
mark "C1 cells end rc=$?"
bash $I/std.sh C1 quality
$R $O/compare-solo.txt python3 tools/c4_mixed.py compare $CS/C0-solo.json $CS/C1-solo.json
$R $O/compare-dec.txt python3 tools/c4_mixed.py compare $CS/C0-dec.json $CS/C1-dec.json
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "C1 chain done"
