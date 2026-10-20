#!/usr/bin/env bash
# FINAL block 8, soak (last): 4 resident ~450K sessions, 4096 tokens each, 0 preemptions expected.
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
mark "block8 start"
up && { mark "soak start"; $T/run_phase.sh $O/soak.txt python3 tools/resident_soak.py $O/soak.json --sessions 4 --tokens 450000 --max-tokens 4096; mark "soak end"; }
mark "block8 end"
