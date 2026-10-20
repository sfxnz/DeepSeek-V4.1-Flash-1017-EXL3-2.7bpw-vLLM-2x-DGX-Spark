#!/usr/bin/env bash
# NOREL cells (main): samplers on for the whole run (memsample + procmem + live logs via pm_cells `sample`, plus
# procstat_sampler.py on both nodes); smoke; 3 x 32K pp_novel, 3 x 64K pp_novel, replay-on 102K gen back to back (no pause);
# a trip stops the cells; then c1 prose x5 and bench_distinct c=4 x5; logs.
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/NOREL; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
PMT=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-integration4; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
setsid -f python3 $T/procstat_sampler.py $O/procstat-spark1.tsv
ssh -o BatchMode=yes spark2 mkdir -p $O && scp -q $T/procstat_sampler.py spark2:$O/procstat_sampler.py
setsid -f ssh -o BatchMode=yes spark2 "python3 $O/procstat_sampler.py $O/procstat-spark2.tsv"
bash $PMT/pm_cells.sh $O sample smoke pp32kx3 pp64kx3 gen102k
up && bash $I/std.sh NOREL c1short c4distinct
bash $PMT/pm_cells.sh $O logs
kill "$(cat $O/procstat-spark1.tsv.pid)" 2>/dev/null; ssh -o BatchMode=yes spark2 "kill \$(cat $O/procstat-spark2.tsv.pid) 2>/dev/null"
scp -q spark2:$O/procstat-spark2.tsv $O/procstat-spark2.tsv
$T/memsnap.sh mem-end > $O/mem-end.txt 2>&1
mark "NOREL chain done"
