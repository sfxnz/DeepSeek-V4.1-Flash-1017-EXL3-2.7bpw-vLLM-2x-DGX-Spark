#!/usr/bin/env bash
# I6 per RUNSHEET: smokes; c=1 prose/structured runs 9; LAIL 10; c=4 distinct 5; micro (tg cells); Cached before/after micro.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/I6; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration3; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
cached() { { echo "## $1 $(date -u +%FT%TZ) spark1"; grep -E '^(MemAvailable|Cached|MemFree|SwapFree):' /proc/meminfo; echo "## spark2"; ssh -o BatchMode=yes spark2 "grep -E '^(MemAvailable|Cached|MemFree|SwapFree):' /proc/meminfo"; } >> $O/cached.txt; }
bash $I/std.sh I6 smoke c1p9 lail10 c4distinct
cached before-micro
bash $I/std.sh I6 micro
cached after-micro
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "I6 chain done"
