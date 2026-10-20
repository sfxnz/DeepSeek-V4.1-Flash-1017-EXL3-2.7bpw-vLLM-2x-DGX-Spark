#!/usr/bin/env bash
# KP-C cells (main): the KP-B cells (LAIL x10, prose/structured c=1 x9, c=4 distinct x5, quick quality); identity re-gate vs D4.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/KP-C; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
bash $I/std.sh KP-C smoke lail10 c1p9 c4distinct quality
cd $W && [ -s $O/quick.json ] && { python3 tests/quality_eval.py --result $O/quick.json --baseline $I/D4/quick.json > $O/identity-vs-D4.txt 2>&1; echo "rc=$?" >> $O/identity-vs-D4.txt; }
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "KP-C chain done"
