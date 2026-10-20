#!/usr/bin/env bash
# I4 per RUNSHEET: STD (c1 runs 9, LAIL 10); replay_ab prefix arm B (+gen/turns B); micro 64K x2; logs/census extracted after.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/I4; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration2; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W; R=$T/run_phase.sh
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
bash $I/std.sh I4 smoke c1p9 lail10 c2 c4 micro prefix quality
up || exit 1
mark "prefixB start"
$R $O/prefix-B.txt python3 tools/replay_ab.py prefix --replay-rows 2670 --retention 4096 --out $O/prefix-B.json
$R $O/B-gen.txt python3 tools/replay_ab.py gen --out $O/B-gen.json
$R $O/B-turns.txt python3 tools/replay_ab.py turns --out $O/B-turns.json
mark "prefixB end"
up || exit 1
bash $I/std.sh I4 micro64
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
up || exit 1
# c4-sched C0 (stock scheduling), client-only. tools/c4_mixed.py exists only in integration3 (= integration2 + c4-sched
# with knobs 0); the serve itself runs from integration2 @2844fce.
mark "C0 cells start"
bash /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/cells.sh C0 /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration3 > /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/runs/C0-cells.log 2>&1
mark "C0 cells end rc=$?"
$T/memsnap.sh mem-end-C0 > $O/mem-end-C0.txt 2>&1
mark "I4 chain done"
