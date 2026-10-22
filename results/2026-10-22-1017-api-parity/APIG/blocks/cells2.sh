#!/usr/bin/env bash
# APIG cells after the gateway restart (final config incl. MALLOC_MMAP_THRESHOLD_): smoke, api vs api.json, full via gateway.
source /home/sfxnz/projects/data/dsv41-1017/api-parity/boot/APIG/blocks/common.sh
mark "cells2 start"; gwstats restart
up && { mark "smoke2 start"; $R $O/smoke_chat-2.txt python3 smoke_chat.py; $R $O/smoke_vision-2.txt python3 smoke_vision.py; mark "smoke2 end"; }
up && { mark "api2 start"; $R $O/api-2.txt python3 tests/quality_eval.py --api --url $GW --baseline $O/api.json --out $O/api-2.json; gwstats api2; mark "api2 end"; }
up && { mark "full2 start"; $R $O/quality-full-gateway.txt python3 tests/quality_eval.py --full --url $GW --baseline results/2026-10-18-1017/quality-baseline/full.json --out $O/full-gateway.json; gwstats full2; mark "full2 end"; }
up || mark "SERVE DOWN"
mark "cells2 done"
