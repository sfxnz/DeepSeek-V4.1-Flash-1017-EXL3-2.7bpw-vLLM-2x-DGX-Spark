#!/usr/bin/env bash
# FINAL block 2, decode headline: prose/structured c=1 x9, c=2 x5; LAIL x10; code c=1 x5 (400 tok); bench_distinct c=4 x5; bench_decode c=4 same-prompt x3.
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
mark "block2 start"
specsnap c12-before; bash $I/std.sh FINAL c1p9 c2; specsnap c12-after
up && bash $I/std.sh FINAL lail10 code c4distinct
up && { mark "c4same start"; $T/run_phase.sh $O/c4_same.txt python3 bench_decode.py --concurrency 4 --runs 3; mark "c4same end"; }
mark "block2 end"
