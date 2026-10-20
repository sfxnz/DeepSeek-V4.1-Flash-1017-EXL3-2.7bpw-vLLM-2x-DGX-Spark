#!/usr/bin/env bash
# I5 per RUNSHEET (+ main agent's A/A request): smokes; replay_ab gen A twice back to back (A/A); turns A; prefix A;
# compares vs I4 (B arm); c=1 prose runs 5; micro 4K/16K; micro 64K last.
# Re-boot after trip 1: gen 102,400 skipped (tripped memwatch with replay off + HOLD).
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/I5; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration3; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W; R=$T/run_phase.sh
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
bash $I/std.sh I5 smoke
up || exit 1
mark "A-arm start"
$R $O/A-gen.txt python3 tools/replay_ab.py gen --lengths 8192 32768 --out $O/A-gen.json
$R $O/A2-gen.txt python3 tools/replay_ab.py gen --lengths 8192 32768 --out $O/A2-gen.json
$R $O/A-turns.txt python3 tools/replay_ab.py turns --out $O/A-turns.json
$R $O/prefix-A.txt python3 tools/replay_ab.py prefix --replay-rows 0 --retention 4096 --out $O/prefix-A.json
mark "A-arm end"
$R $O/compare-AA-gen.txt python3 tools/replay_ab.py compare $O/A-gen.json $O/A2-gen.json
$R $O/compare-gen.txt python3 tools/replay_ab.py compare $O/A-gen.json $I/I4/B-gen.json
$R $O/compare-turns.txt python3 tools/replay_ab.py compare $O/A-turns.json $I/I4/B-turns.json
$R $O/compare-prefix.txt python3 tools/replay_ab.py compare $O/prefix-A.json $I/I4/prefix-B.json
up || exit 1
bash $I/std.sh I5 c1short
up || exit 1
mark "micro416 start"; $R $O/micro.txt benches/micro.sh --contexts 4096 16384 --runs 2; mark "micro416 end"
up || exit 1
mark "micro64 start"; $R $O/micro64.txt benches/micro.sh --contexts 65536 --runs 2; mark "micro64 end"
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "I5 chain done"
