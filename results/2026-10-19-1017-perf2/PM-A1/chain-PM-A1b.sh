#!/usr/bin/env bash
# PM-A1 re-boot cells (RUNSHEET failure handling: re-boot once, skip the cell that tripped). Boot 1 tripped memwatch in gen's
# 102,400-token prompt (PM-A1/trip1/). Skipped: gen 102,400 and gen102k (the same replay-off 102K prompt). gen runs 8192 + 32768
# only (as I5c) so the compare vs I5c has token data. quality runs before soak, so a soak trip cannot lose it.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/PM-A1; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
PMT=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
bash $PMT/pm_cells.sh $O sample smoke
if up; then mark "gen start"; (cd $W && python3 tools/replay_ab.py gen --lengths 8192 32768 --out $O/gen.json > $O/gen.txt 2>&1); mark "gen end"; fi
up && bash $PMT/pm_cells.sh $O pp16k pp64k quality soak
grep -q " logs$" $O/phases.txt || bash $PMT/pm_cells.sh $O logs
cd $W
[ -s $O/gen.json ] && { python3 tools/replay_ab.py compare $I/I5/A-gen.json $O/gen.json > $O/compare-gen-vs-I5c.txt 2>&1; echo "rc=$?" >> $O/compare-gen-vs-I5c.txt
  python3 tools/replay_ab.py compare $I/I2/A-gen.json $O/gen.json > $O/compare-gen-vs-I2.txt 2>&1; echo "rc=$?" >> $O/compare-gen-vs-I2.txt; }
[ -s $O/quick.json ] && { python3 tests/quality_eval.py --result $O/quick.json --baseline $I/D4/quick.json > $O/identity-vs-D4.txt 2>&1; echo "rc=$?" >> $O/identity-vs-D4.txt; }
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "PM-A1 chain done"
