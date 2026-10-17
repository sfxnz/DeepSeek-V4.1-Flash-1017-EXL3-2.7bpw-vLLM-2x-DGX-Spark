#!/bin/bash
# Run a command from tools/calib inside the dsv41-quant151 image on THIS host's GPU, serialized by the
# campaign GPU lock. Use on spark2 (spark1's GPU is reserved).  Usage:
#   tools/calib/run_gpu.sh <name-suffix> python3 tools/calib/cli.py nll ...
# Env: CALIB_MEM (default 110g), CALIB_IMAGE (default dsv41-quant151), CALIB_LOCK (lock file).
set -euo pipefail
NAME="dsv41-1017-calib-$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${CALIB_LOCK:-/home/sfxnz/projects/data/dsv41-1017/spark2-gpu.lock}"
IMAGE="${CALIB_IMAGE:-dsv41-quant151}"
mkdir -p "$(dirname "$LOCK")"
exec flock "$LOCK" docker run --rm --name "$NAME" --gpus all --memory "${CALIB_MEM:-110g}" --memory-swap "${CALIB_MEM:-110g}" \
  --network none --ipc host --ulimit memlock=-1 --user "$(id -u):$(id -g)" -e HOME=/home/sfxnz \
  -v /home/sfxnz:/home/sfxnz -w "$HERE" \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TILELANG_CACHE_DIR=/home/sfxnz/projects/data/dsv41-1017/calib-harness/tilelang-cache \
  --entrypoint "$1" "$IMAGE" "${@:2}"
