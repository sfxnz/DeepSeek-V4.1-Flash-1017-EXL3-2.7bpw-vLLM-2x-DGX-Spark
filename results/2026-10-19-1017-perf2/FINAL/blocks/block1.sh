#!/usr/bin/env bash
# FINAL block 1: samplers for the whole run (procstat both nodes; pm_cells sample = memsample + procmem + live logs), smokes, free -h.
source /home/sfxnz/projects/data/dsv41-1017/perf2/integration/FINAL/blocks/common.sh
setsid -f python3 $T/procstat_sampler.py $O/procstat-spark1.tsv
ssh -o BatchMode=yes spark2 mkdir -p $O && scp -q $T/procstat_sampler.py spark2:$O/procstat_sampler.py
setsid -f ssh -o BatchMode=yes spark2 "python3 $O/procstat_sampler.py $O/procstat-spark2.tsv"
bash $PMT/pm_cells.sh $O sample
mark "block1 start"
bash $I/std.sh FINAL smoke
{ echo "## free -h spark1 $(date -u +%FT%TZ)"; free -h; echo "## free -h spark2"; ssh -o BatchMode=yes spark2 free -h; } > $O/free-block1.txt 2>&1
mark "block1 end"
