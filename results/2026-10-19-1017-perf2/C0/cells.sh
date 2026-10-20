#!/usr/bin/env bash
# c4-sched GPU cells for one boot (one arm) against the live TP=2 serve. HTTP only, no containers.
#
#   setsid -f bash /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/cells.sh <ARM> <RECIPE_DIR> \
#       > /home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/runs/<ARM>-cells.log 2>&1
#
# RECIPE_DIR: the worktree the serve was started from (tools/c4_mixed.py must be in it).
# Cells (~20-25 min with 64K + 128K prefills):
#   solo   every prompt alone (identity reference; lone requests are scheduled as stock)
#   dec    3 pure decode streams; 64K then 128K novel prefill from a 4th session
#   dec-aa the same again on this boot (fresh cache_salt): this boot's mixed-run noise floor
#   turns  2 decode streams + 1 turn-taking stream (1K-token turns) during a 128K prefill
# then logs, scheduler/read-ahead/empty_cache greps, memory.
set -euo pipefail
ARM="${1:?arm label}"
RECIPE="${2:?recipe worktree the serve runs from}"
OUT="${OUT:-/home/sfxnz/projects/data/dsv41-1017/perf2/c4-sched/runs}"
CONTAINER="${CONTAINER:-dsv41-flash-exl3}"
mkdir -p "$OUT"
cd "$RECIPE"
stamp() { date -u +%FT%TZ; }
phase() { echo "$(stamp) $ARM $*" | tee -a "$OUT/phases.txt"; }
mem() { { echo "## spark1 $(stamp) $1"; free -m; echo "## spark2"; ssh -o BatchMode=yes spark2 free -m; } >> "$OUT/$ARM-mem.txt"; }

phase "start recipe=$RECIPE HEAD=$(git rev-parse --short HEAD) dirty='$(git status --short | tr '\n' ' ')'"
set +e  # diagnostics: a grep without a match is not a failure
{
  echo "# env of the serve (head container)"
  docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep -E '^DSV41_(SCHED|DECODER_REPLAY|ENGRAM_PREFILL_AHEAD|MOE_PREFILL_TC|PREFILL_EMPTY_CACHE)' || true
  echo "# lever lines (head, counted)"
  docker logs "$CONTAINER" 2>&1 | grep -E 'sched prefill cap|decoder replay (armed|prefix-cache guard)|engram prefill ahead armed' \
    | sed -E 's/^\([A-Za-z_]+[^)]*\) //' | sort | uniq -c
  echo "# lever lines (worker, counted)"
  ssh -o BatchMode=yes spark2 "docker logs $CONTAINER 2>&1" | grep -E 'sched prefill cap|decoder replay armed' \
    | sed -E 's/^\([A-Za-z_]+[^)]*\) //' | sort | uniq -c
} > "$OUT/$ARM-armed.txt" 2>&1
set -e
mem before

phase "solo start"
python3 tools/c4_mixed.py solo --seed 4300 --prefill 65536 131072 --out "$OUT/$ARM-solo.json" > "$OUT/$ARM-solo.txt" 2>&1
phase "dec start"
python3 tools/c4_mixed.py run --seed 4300 --turn-streams 0 --prefill 65536 131072 --out "$OUT/$ARM-dec.json" \
  > "$OUT/$ARM-dec.txt" 2>&1
mem after-dec
phase "dec-aa start"
python3 tools/c4_mixed.py run --seed 4300 --turn-streams 0 --prefill 65536 131072 --out "$OUT/$ARM-dec-aa.json" \
  > "$OUT/$ARM-dec-aa.txt" 2>&1
phase "turns start"
python3 tools/c4_mixed.py run --seed 4400 --turn-streams 1 --prefill 131072 --out "$OUT/$ARM-turns.json" \
  > "$OUT/$ARM-turns.txt" 2>&1
mem after-turns
phase "cells done"

python3 tools/c4_mixed.py compare "$OUT/$ARM-dec.json" "$OUT/$ARM-dec-aa.json" > "$OUT/$ARM-dec-vs-aa.json"
python3 tools/c4_mixed.py compare "$OUT/$ARM-solo.json" "$OUT/$ARM-dec.json" > "$OUT/$ARM-solo-vs-dec.json"
docker logs "$CONTAINER" > "$OUT/$ARM-head.log" 2>&1
ssh -o BatchMode=yes spark2 "docker logs $CONTAINER" > "$OUT/$ARM-worker.log" 2>&1
set +e
{
  echo "# empty_cache hook lines (head / worker)"
  grep -h 'dsv41-prefill-empty-cache' "$OUT/$ARM-head.log" | tail -5
  grep -h 'dsv41-prefill-empty-cache' "$OUT/$ARM-worker.log" | tail -5
  echo "# prefill read-ahead census: lines, waits > 100 ms"
  grep -c '\[prefill-ahead\]' "$OUT/$ARM-head.log" || true
  grep -h '\[prefill-ahead\]' "$OUT/$ARM-head.log" | awk '{for(i=1;i<=NF;i++) if($i ~ /^wait=/){w=substr($i,6)+0; if(w>100) n++}} END {print n+0, "waits > 100 ms"}'
  echo "# iteration lines (only with EXTRA_ARGS=--enable-logging-iteration-details): context-token histogram"
  grep -ho 'context requests, [0-9]* context tokens' "$OUT/$ARM-head.log" | awk '{print $3}' | sort -n | uniq -c | sort -rn | head -12
  echo "# disarm / failure lines"
  grep -hE 'DISABLED|dsv41-patch FAIL|FAILED|Traceback' "$OUT/$ARM-head.log" "$OUT/$ARM-worker.log" | sort | uniq -c | head -20
} > "$OUT/$ARM-greps.txt" 2>&1
set -e
phase "done"
