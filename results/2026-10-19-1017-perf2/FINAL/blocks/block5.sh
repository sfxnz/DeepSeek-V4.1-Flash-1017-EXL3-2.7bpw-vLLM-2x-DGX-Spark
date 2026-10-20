#!/usr/bin/env bash
# FINAL block 5, replay A/A prefix noise floor: replay_ab prefix twice in this boot, each with its own cache_salt
# (replay_ab has no salt option; baseline/tools/replay_ab_salted.py adds it to chat requests without touching the tree), then compare.
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
mark "block5 start"
for b in B1 B2; do up && { mark "prefix-$b start"; $T/run_phase.sh $O/prefix-$b.txt python3 $T/replay_ab_salted.py FINAL-$b-$(date +%s) prefix --replay-rows 2670 --retention 4096 --out $O/prefix-$b.json; mark "prefix-$b end"; }; done
[ -s $O/prefix-B1.json ] && [ -s $O/prefix-B2.json ] && $T/run_phase.sh $O/prefix-compare.txt python3 tools/replay_ab.py compare $O/prefix-B1.json $O/prefix-B2.json
mark "block5 end"
