#!/usr/bin/env bash
# PROBE cells (main): /proc sampler on both nodes for the whole run; pm_cells sample smoke pp16kx3 pp32kx3 logs from the probe tree.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/PROBE; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
PMT=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-probe; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
setsid -f python3 $T/procstat_sampler.py $O/procstat-spark1.tsv
ssh -o BatchMode=yes spark2 mkdir -p $O && scp -q $T/procstat_sampler.py spark2:$O/procstat_sampler.py
setsid -f ssh -o BatchMode=yes spark2 "python3 $O/procstat_sampler.py $O/procstat-spark2.tsv"
bash $PMT/pm_cells.sh $O sample smoke pp16kx3 pp32kx3 logs
grep -q " logs$" $O/phases.txt || bash $PMT/pm_cells.sh $O logs
grep -h "\] probe \(step\|flush\|release\) " $O/head.live.log $O/worker.live.log > $O/probe-lines-live.txt 2>/dev/null
kill "$(cat $O/procstat-spark1.tsv.pid)" 2>/dev/null; ssh -o BatchMode=yes spark2 "kill \$(cat $O/procstat-spark2.tsv.pid) 2>/dev/null"
scp -q spark2:$O/procstat-spark2.tsv $O/procstat-spark2.tsv
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "PROBE chain done"
