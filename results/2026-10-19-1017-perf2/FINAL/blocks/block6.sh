#!/usr/bin/env bash
# FINAL block 6, c4-sched cells (solo + dec + dec-aa + turns) from the release tree -> perf2/c4-sched/runs/FINAL-*.
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
mark "block6 start"
bash /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/cells.sh FINAL $W > /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/runs/FINAL-cells.log 2>&1
echo "cells rc=$?" >> /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/runs/FINAL-cells.log
mark "block6 end"
