#!/usr/bin/env bash
# I2 cells per RUNSHEET: smokes; micro; replay_ab gen/turns (A arm); compare vs I1 B arm; warm_prefix 20000; + extra: G0 prefix tests.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/I2; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration; export RECIPE_DIR=$W DSV41_RECIPE=$W; R=$T/run_phase.sh
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
bash $I/std.sh I2 smoke micro
up || exit 1
mark "A-arm start"
$R $O/A-gen.txt python3 tools/replay_ab.py gen --out $O/A-gen.json
$R $O/A-turns.txt python3 tools/replay_ab.py turns --out $O/A-turns.json
$R $O/warm_prefix.txt python3 tools/warm_prefix.py --tokens 20000
mark "A-arm end"
up || exit 1
$R $O/compare-gen.txt python3 tools/replay_ab.py compare $O/A-gen.json $I/I1/B-gen.json
$R $O/compare-turns.txt python3 tools/replay_ab.py compare $O/A-turns.json $I/I1/B-turns.json
bash $I/std.sh I2 prefix
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "I2 chain done"
