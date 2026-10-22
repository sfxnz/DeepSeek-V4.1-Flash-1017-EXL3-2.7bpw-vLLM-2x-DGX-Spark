# sourced by the APIG block scripts
O=/home/sfxnz/projects/data/dsv41-1017/api-parity/boot/APIG; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-eval; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W PYTHONDONTWRITEBYTECODE=1
R=$T/run_phase.sh; GW=http://127.0.0.1:8000; VL=http://127.0.0.1:8001
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3 && docker ps --format '{{.Names}}' | grep -qx dsv41-api-gateway; }
gwstats() { docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.CPUPerc}}' dsv41-api-gateway > $O/gw-stats-$1.txt 2>&1; $T/memsnap.sh mem-$1 > $O/mem-$1.txt 2>&1; }
