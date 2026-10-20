#!/usr/bin/env bash
# FINAL block 7, needles: phased/tools/long_needle.py at 256K (262144), then 500K (500000 = max-model-len; run_needle budget L-64).
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
mark "block7 start"
up && { mark "needle256k start"; $T/run_phase.sh $O/needle-256k.txt python3 /home/sfxnz/projects/data/dsv41-1017/phased/tools/long_needle.py $O/needle-256k.json 262144; mark "needle256k end"; }
up && { mark "needle500k start"; $T/run_phase.sh $O/needle-500k.txt python3 /home/sfxnz/projects/data/dsv41-1017/phased/tools/long_needle.py $O/needle-500k.json 500000; mark "needle500k end"; }
mark "block7 end"
