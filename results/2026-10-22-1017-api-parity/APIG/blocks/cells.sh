#!/usr/bin/env bash
# APIG cells, in order: smoke+headers, api eval (+A/A), quick direct + via gateway, decode c=1/c=4, full via gateway.
source /home/sfxnz/projects/data/dsv41-1017/api-parity/boot/APIG/blocks/common.sh
M=deepseek-ai/DeepSeek-V4.1-Flash-1017
mark "cells start"; gwstats boot
up && { mark "smoke start"; $R $O/smoke_chat.txt python3 smoke_chat.py; $R $O/smoke_vision.txt python3 smoke_vision.py
  $R $O/headers.txt bash -c "curl -s -D - -o /dev/null $GW/v1/chat/completions -H 'content-type: application/json' -d '{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8}'; curl -s -D - -o /dev/null $GW/v1/chat/completions -H 'content-type: application/json' -d '{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"echo\":true}'; curl -s $GW/v1/models"
  gwstats smoke; mark "smoke end"; }
up && { mark "api start"; $R $O/api.txt python3 tests/quality_eval.py --api --url $GW --out $O/api.json
  $R $O/api-aa.txt python3 tests/quality_eval.py --api --url $GW --baseline $O/api.json --out $O/api-aa.json; gwstats api; mark "api end"; }
up && { mark "quick start"; $R $O/quality-quick-direct.txt python3 tests/quality_eval.py --quick --url $VL --baseline results/2026-10-18-1017/quality-baseline/quick.json --out $O/quick-direct.json
  $R $O/quality-quick-gateway.txt python3 tests/quality_eval.py --quick --url $GW --baseline results/2026-10-18-1017/quality-baseline/quick.json --out $O/quick-gateway.json; gwstats quick; mark "quick end"; }
up && { mark "decode start"
  for p in prose structured; do
    $R $O/c1_${p}_gateway.txt python3 bench_decode.py --phase $p --concurrency 1 --runs 9 --url $GW/v1/chat/completions
    $R $O/c1_${p}_direct.txt python3 bench_decode.py --phase $p --concurrency 1 --runs 9 --url $VL/v1/chat/completions
  done
  $R $O/c4_distinct_gateway.txt python3 $T/bench_distinct.py --concurrency 4 --runs 5 --url $GW/v1/chat/completions
  gwstats decode; mark "decode end"; }
up && { mark "full start"; $R $O/quality-full-gateway.txt python3 tests/quality_eval.py --full --url $GW --baseline results/2026-10-18-1017/quality-baseline/full.json --out $O/full-gateway.json; gwstats full; mark "full end"; }
up || mark "SERVE DOWN"
mark "cells done"
