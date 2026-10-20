# sourced by the FINAL block scripts
I=/home/sfxnz/projects/data/dsv41-1017/perf2/integration; O=$I/FINAL; T=/home/sfxnz/projects/data/dsv41-1017/perf2/baseline/tools
PMT=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-mem/tools
W=/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-release; export WT=$W RECIPE_DIR=$W DSV41_RECIPE=$W PYTHONDONTWRITEBYTECODE=1
mark() { echo "$(date -u +%FT%TZ) $1" >> $O/phases.txt; }
up() { docker ps --format '{{.Names}}' | grep -qx dsv41-flash-exl3; }
specsnap() { curl -s http://127.0.0.1:8000/metrics | grep -E '^vllm:spec_decode' > $O/spec-$1.prom; }
