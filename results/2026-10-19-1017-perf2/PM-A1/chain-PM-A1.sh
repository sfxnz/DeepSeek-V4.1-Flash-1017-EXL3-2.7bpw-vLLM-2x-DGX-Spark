#!/usr/bin/env bash
# PM-A1 cells (main go, item 2) in the given order; gen (8K/32K/102K) and gen102k are never retried after a trip.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/PM-A1; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
PMT=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
bash $PMT/pm_cells.sh $O sample smoke gen pp16k pp64k gen102k soak quality logs
cd $W
[ -s $O/gen.json ] && { python3 tools/replay_ab.py compare $I/I5/A-gen.json $O/gen.json > $O/compare-gen-vs-I5c.txt 2>&1; echo "rc=$?" >> $O/compare-gen-vs-I5c.txt
  python3 tools/replay_ab.py compare $I/I2/A-gen.json $O/gen.json > $O/compare-gen-vs-I2.txt 2>&1; echo "rc=$?" >> $O/compare-gen-vs-I2.txt; }
[ -s $O/quick.json ] && { python3 tests/quality_eval.py --result $O/quick.json --baseline $I/D4/quick.json > $O/identity-vs-D4.txt 2>&1; echo "rc=$?" >> $O/identity-vs-D4.txt; }
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "PM-A1 chain done"
