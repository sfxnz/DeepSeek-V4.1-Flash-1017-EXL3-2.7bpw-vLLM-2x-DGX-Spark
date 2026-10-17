#!/usr/bin/env bash
# From spark1: sync this worktree to spark2, run one GPU step there under the campaign GPU lock,
# sync the raw logs back.
#   kernel_study/p2b_kbits/spark2.sh <tag> <image> <python3 args...>
# The step runs `python3 <args>` in a --rm container dsv41-1017-kb-<tag> (memory-capped, no network)
# with the repo at /repo, the Sept source snapshot at /src (ro), the served K=2 pack at /hf (ro)
# and the campaign kernels data dir at /data. Log: /home/sfxnz/projects/data/dsv41-1017/kernels/runs/<tag>.log
# (on spark2, rsynced back to the same path on spark1). MEM (default 24g) caps the container.
set -euo pipefail
tag=$1; img=$2; shift 2
here=$(cd "$(dirname "$0")/../.." && pwd)
dst=/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2/$(basename "$here")
data=/home/sfxnz/projects/data/dsv41-1017/kernels
lock=/home/sfxnz/projects/data/dsv41-1017/spark2-gpu.lock
ssh spark2 "mkdir -p $dst $data/runs"
rsync -a --delete --exclude .git --exclude '__pycache__' --exclude 'kernel_study/*/build*/' "$here/" "spark2:$dst/"
set +e
ssh spark2 "bash -s -- $(printf '%q ' "$dst" "$data" "$lock" "$tag" "$img" "${MEM:-24g}" "$@")" <<'EOF'
set -uo pipefail
dst=$1; data=$2; lock=$3; tag=$4; img=$5; mem=$6; shift 6
log=$data/runs/$tag.log
src=/home/sfxnz/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
pack=/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-EXL3
exec 9>"$lock"
echo "$(date -u +%FT%TZ) waiting for $lock" >> "$log"
flock -w 14400 9 || { echo "lock timeout" | tee -a "$log"; exit 4; }
{
  echo "# $(date -u +%FT%TZ) host $(hostname) tag $tag image $img mem $mem"
  echo "# cmd: python3 $*"
  echo "# pmon before:"; nvidia-smi pmon -c 1
  echo "# free -g:"; free -g
} >> "$log"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -lms 500 > "$data/runs/$tag.clocks.csv" 2>&1 &
smi=$!
docker run --rm --name "dsv41-1017-kb-$tag" --gpus all --network none --memory "$mem" --memory-swap "$mem" \
  --user 1000:1000 -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e TORCH_EXTENSIONS_DIR=/repo/kernel_study/p2b_kbits/build/torch_ext \
  -v "$dst:/repo" -v "$src:/srcrepo:ro" -v "$pack:/hf:ro" -v "$data:/data" -w /repo \
  --entrypoint python3 "$img" "$@" 2>&1 \
  | grep --line-buffered -v -e '^dsv41-patch' -e '^dsv41-indexer' -e ' \[importing.py' -e ' \[interface.py' -e 'cpp_extension.py:' \
  | tee -a "$log"
rc=${PIPESTATUS[0]}
kill $smi 2>/dev/null
{ echo "# $(date -u +%FT%TZ) rc $rc"; echo "# pmon after:"; nvidia-smi pmon -c 1; } >> "$log"
exit $rc
EOF
rc=$?
set -e
mkdir -p "$data/runs"
rsync -a "spark2:$data/runs/" "$data/runs/"
exit $rc
