#!/usr/bin/env bash
# D2 per RUNSHEET DSpark adaptive arms: smokes; LAIL 10; bench_decode prose|structured c=1 2 runs 9; code 400 tok runs 5;
# bench_distinct c=4 runs 5; spec_decode per-position snapshots around decode cells; quick quality (includes selfcons, decode);
# then greedy identity: quality_eval --result <arm>/quick.json --baseline I4/quick.json (no traffic).
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration3
export WT=$W
bash $I/std.sh D2 smoke lail10 c12 code c4distinct quality
cd $W && python3 tests/quality_eval.py --result $I/D2/quick.json --baseline $I/I4/quick.json > $I/D2/identity-vs-I4.txt 2>&1; echo "rc=$?" >> $I/D2/identity-vs-I4.txt
echo "$(date -u +%FT%TZ) D2 chain done" >> $I/D2/phases.txt
