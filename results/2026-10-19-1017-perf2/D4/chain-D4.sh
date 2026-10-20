#!/usr/bin/env bash
# D4 cells: D-arm cells + micro 4K/16K/64K runs 2 + grammar check; identity re-gate vs I4.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/D4; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
bash $I/std.sh D4 smoke lail10 c12 code c4distinct
mark "grammar start"; $T/run_phase.sh $O/grammar.txt python3 $T/grammar_check.py --out $O/grammar.json; mark "grammar end"
bash $I/std.sh D4 quality micro
cd $W && python3 tests/quality_eval.py --result $O/quick.json --baseline $I/I4/quick.json > $O/identity-vs-I4.txt 2>&1; echo "rc=$?" >> $O/identity-vs-I4.txt
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "D4 chain done"
