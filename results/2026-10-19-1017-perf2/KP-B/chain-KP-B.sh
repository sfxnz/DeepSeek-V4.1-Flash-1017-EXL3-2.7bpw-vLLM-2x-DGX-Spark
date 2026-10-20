#!/usr/bin/env bash
# KP-B cells (main go, item 3): LAIL x10, prose/structured c=1 x9, c=4 distinct x5, quick quality; then (main, 02:2xZ) a read-only
# 1 Hz /proc probe on both nodes (procstat_sampler.py: full meminfo + vLLM processes' VmRSS/VmSwap/RssAnon/RssFile/RssShmem/VmPin)
# around ONE 64K pp_novel prompt (pp_chunks.py, prefill only); identity re-gate vs D4. No nvidia-smi (AGENTS.md rule).
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/KP-B; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
PMT=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
bash $I/std.sh KP-B smoke lail10 c1p9 c4distinct quality
if up; then
  mark "probe start"
  setsid -f python3 $T/procstat_sampler.py $O/procstat-spark1.tsv
  ssh -o BatchMode=yes spark2 mkdir -p $O && scp -q $T/procstat_sampler.py spark2:$O/procstat_sampler.py
  setsid -f ssh -o BatchMode=yes spark2 "python3 $O/procstat_sampler.py $O/procstat-spark2.tsv"
  setsid -f bash $PMT/memsample.sh $O/memsample.tsv
  sleep 4
  (cd $W && python3 $PMT/pp_chunks.py --tokens 65536 --runs 1 --out $O/pp64k-probe.jsonl > $O/pp64k-probe.txt 2>&1)
  sleep 4
  kill "$(cat $O/procstat-spark1.tsv.pid)"; ssh -o BatchMode=yes spark2 "kill \$(cat $O/procstat-spark2.tsv.pid)"
  scp -q spark2:$O/procstat-spark2.tsv $O/procstat-spark2.tsv
  sp=$(cat $O/memsample.tsv.pid); pkill -P "$sp"; kill "$sp"
  docker logs -t dsv41-flash-exl3 > $O/head-t.log 2>&1; ssh -o BatchMode=yes spark2 docker logs -t dsv41-flash-exl3 > $O/worker-t.log 2>&1
  mark "probe end"
fi
cd $W && [ -s $O/quick.json ] && { python3 tests/quality_eval.py --result $O/quick.json --baseline $I/D4/quick.json > $O/identity-vs-D4.txt 2>&1; echo "rc=$?" >> $O/identity-vs-D4.txt; }
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "KP-B chain done"
