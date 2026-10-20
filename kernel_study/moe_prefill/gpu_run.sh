#!/usr/bin/env bash
# One GPU step on THIS node (spark1 by default) under the campaign GPU lock, in a --rm container.
#   kernel_study/moe_prefill/gpu_run.sh <tag> <image> <python3 args...>
# The worktree is mounted at /repo (rw: the JIT build cache lands in kernel_study/moe_prefill/build/),
# the served 1017 pack snapshot at /pack (ro), the s10 census at /census (ro) and the run dir at /data.
# Log: /home/sfxnz/projects/data/dsv41-1017/perf2/prefill-moe/runs/<tag>.log (+ .clocks.csv). MEM caps the
# container (default 24g). LOCK overrides the lock file (spark2: .../spark2-gpu.lock).
set -euo pipefail
tag=$1; img=$2; shift 2
here=$(cd "$(dirname "$0")/../.." && pwd)
runs=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-moe/runs
lock=${LOCK:-/home/sfxnz/projects/data/dsv41-1017/spark1-gpu.lock}
pack=${PACK:-/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq}
census=${CENSUS:-/home/sfxnz/projects/data/dsv41-1017/kernels/census_npy}
mkdir -p "$runs" "$here/kernel_study/moe_prefill/build"
log=$runs/$tag.log
exec 9>"$lock"
echo "$(date -u +%FT%TZ) waiting for $lock" >> "$log"
flock -w 14400 9 || { echo "lock timeout" | tee -a "$log"; exit 4; }
{
  echo "# $(date -u +%FT%TZ) host $(hostname) tag $tag image $img mem ${MEM:-24g} git $(git -C "$here" rev-parse --short HEAD)"
  echo "# cmd: python3 $*"
  echo "# docker ps:"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  echo "# free -g:"; free -g
} >> "$log"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -lms 500 > "$runs/$tag.clocks.csv" 2>&1 &
smi=$!
set +e
docker run --rm --name "dsv41-1017-pm-$tag" --gpus all --network none --memory "${MEM:-24g}" --memory-swap "${MEM:-24g}" \
  --user 1000:1000 -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e TORCH_EXTENSIONS_DIR=/repo/kernel_study/moe_prefill/build/torch_ext \
  -v "$here:/repo" -v "$pack:/pack:ro" -v "$census:/census:ro" -v "$runs:/data" -w /repo \
  --entrypoint python3 "$img" "$@" 2>&1 \
  | grep --line-buffered -v -e '^dsv41-patch' -e '^dsv41-indexer' -e ' \[importing.py' -e ' \[interface.py' -e 'cpp_extension.py:' \
  | tee -a "$log"
rc=${PIPESTATUS[0]}
set -e
kill $smi 2>/dev/null || true
echo "# $(date -u +%FT%TZ) rc $rc" >> "$log"
exit $rc
