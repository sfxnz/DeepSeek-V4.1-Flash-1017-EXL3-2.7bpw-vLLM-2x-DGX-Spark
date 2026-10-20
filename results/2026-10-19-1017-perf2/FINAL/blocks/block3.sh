#!/usr/bin/env bash
# FINAL block 3, prefill: micro --contexts 4096 16384 65536 --runs 2.
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
mark "block3 start"; bash $I/std.sh FINAL micro; mark "block3 end"
