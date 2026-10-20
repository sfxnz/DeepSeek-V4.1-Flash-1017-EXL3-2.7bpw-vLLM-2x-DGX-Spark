#!/usr/bin/env bash
# Run one router_gemv GPU script (bench.py | prologue_bench.py) on spark1 or spark2 under the campaign GPU lock, in a memory-capped,
# network-less --rm container of the serving image, with this worktree at /repo and the 1017 pack at /hf
# (read-only). Raw output: /home/sfxnz/projects/data/dsv41-1017/perf2/decode-glue/gpu/<tag>.{log,json,clocks.csv}
# (synced back to spark1 when run on spark2).
#   kernel_study/router_gemv/run_gpu.sh <spark1|spark2> <tag> <bench.py|prologue_bench.py> [args...]
# Env: IMG (default dsv41-flash-exl3-sm121:e15-kbits), MEM (default 16g).
# Only when the main agent has assigned that GPU: nothing else may run on it (no serve container).
set -euo pipefail
host=$1; tag=$2; script=$3; shift 3
here=$(cd "$(dirname "$0")/../.." && pwd)
img=${IMG:-dsv41-flash-exl3-sm121:e15-kbits}
out=/home/sfxnz/projects/data/dsv41-1017/perf2/decode-glue/gpu
pack=/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3
lock=/home/sfxnz/projects/data/dsv41-1017/$host-gpu.lock
dst=$here
if [[ $host == spark2 ]]; then
  dst=/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2/$(basename "$here")
  ssh spark2 "mkdir -p $dst $out"
  rsync -a --delete --exclude .git --exclude '__pycache__' "$here/" "spark2:$dst/"
fi
mkdir -p "$out"
step() {
  local dst=$1 out=$2 lock=$3 tag=$4 img=$5 mem=$6 pack=$7 script=$8; shift 8
  exec 9>"$lock"
  echo "$(date -u +%FT%TZ) waiting for $lock" >> "$out/$tag.log"
  flock -w 14400 9 || { echo "lock timeout" | tee -a "$out/$tag.log"; return 4; }
  if docker ps --format '{{.Image}} {{.Names}}' | grep -q dsv41-flash-exl3; then
    echo "a dsv41-flash-exl3 container is up on $(hostname): refusing" | tee -a "$out/$tag.log"; return 5
  fi
  { echo "# $(date -u +%FT%TZ) host $(hostname) tag $tag image $img mem $mem"
    echo "# cmd: python3 /repo/kernel_study/router_gemv/$script $*"
    nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv; free -g; } >> "$out/$tag.log"
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
      --format=csv -lms 500 > "$out/$tag.clocks.csv" 2>&1 &
  local smi=$!
  docker run --rm --name "dsv41-1017-rg-$tag" --gpus all --network none --memory "$mem" --memory-swap "$mem" \
    --user 1000:1000 -e HOME=/tmp -v "$dst:/repo:ro" -v "$pack:/hf:ro" -w /repo \
    -e TORCH_CUDA_ARCH_LIST=12.1a --entrypoint python3 "$img" "/repo/kernel_study/router_gemv/$script" "$@" \
    2> >(grep --line-buffered -v -e '^dsv41-patch' >> "$out/$tag.log") > "$out/$tag.json"
  local rc=$?
  kill $smi 2>/dev/null || true
  echo "# $(date -u +%FT%TZ) rc $rc" >> "$out/$tag.log"
  return $rc
}
set +e
if [[ $host == spark2 ]]; then
  ssh spark2 "$(declare -f step); step $(printf '%q ' "$dst" "$out" "$lock" "$tag" "$img" "${MEM:-16g}" "$pack" "$script" "$@")"
  rc=$?
  rsync -a "spark2:$out/" "$out/"
else
  step "$dst" "$out" "$lock" "$tag" "$img" "${MEM:-16g}" "$pack" "$script" "$@"
  rc=$?
fi
exit $rc
