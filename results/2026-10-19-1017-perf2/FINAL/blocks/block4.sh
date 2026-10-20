#!/usr/bin/env bash
# FINAL block 4, quality: quick vs quick.json; --full vs full.json; served KL in-boot (main: --max-len 2048 = 61 of 72 rows), last.
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
mark "block4 start"
bash $I/std.sh FINAL quality
up && { mark "full start"; $T/run_phase.sh $O/quality-full.txt python3 tests/quality_eval.py --full --baseline results/2026-10-18-1017/quality-baseline/full.json --out $O/full.json; mark "full end"; }
up && { mark "servedkl start"; $T/run_phase.sh $O/served_kl.txt python3 tools/served_kl.py --panel /home/sfxnz/projects/data/dsv41-1017/calib/panel.jsonl \
          --native /home/sfxnz/projects/data/dsv41-1017/calib-harness/panel_native.topk.safetensors --max-len 2048 --out $O/served_kl.json; mark "servedkl end"; }
mark "block4 end"
