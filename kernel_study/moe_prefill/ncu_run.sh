#!/usr/bin/env bash
# Nsight Compute (the host's /opt/nvidia/nsight-compute/2025.3.1, mounted at /ncu) on THIS node's GPU under
# the campaign lock, as root with SYS_ADMIN in a --rm container; <args> are ncu's (target command last).
#   kernel_study/moe_prefill/ncu_run.sh <tag> <ncu args...>
# Log and report: /home/sfxnz/projects/data/dsv41-1017/perf2/prefill-moe/runs/<tag>.{log,ncu-rep}.
set -euo pipefail
tag=$1; shift
here=$(cd "$(dirname "$0")/../.." && pwd)
runs=/home/sfxnz/projects/data/dsv41-1017/perf2/prefill-moe/runs
lock=${LOCK:-/home/sfxnz/projects/data/dsv41-1017/spark1-gpu.lock}
pack=${PACK:-/home/sfxnz/.cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq}
img=${IMG:-dsv41-flash-exl3-sm121:e15-kbits}
mkdir -p "$runs"
log=$runs/$tag.log
exec 9>"$lock"
flock -w 14400 9 || { echo "lock timeout" | tee -a "$log"; exit 4; }
{ echo "# $(date -u +%FT%TZ) host $(hostname) tag $tag image $img"; echo "# ncu $*"; docker ps --format '{{.Names}} {{.Image}}'; } >> "$log"
set +e
docker run --rm --name "dsv41-1017-pm-$tag" --gpus all --network none --memory 24g --memory-swap 24g \
  --cap-add SYS_ADMIN -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=12.1a \
  -v /opt/nvidia/nsight-compute/2025.3.1:/ncu:ro -v "$here:/repo" -v "$pack:/pack:ro" -v "$runs:/data" -w /repo \
  --entrypoint /ncu/ncu "$img" "$@" 2>&1 | grep --line-buffered -v -e '^dsv41-patch' -e '^dsv41-indexer' | tee -a "$log"
rc=${PIPESTATUS[0]}
set -e
echo "# $(date -u +%FT%TZ) rc $rc" >> "$log"
exit $rc
