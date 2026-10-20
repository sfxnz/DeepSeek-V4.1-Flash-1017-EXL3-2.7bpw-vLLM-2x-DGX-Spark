#!/usr/bin/env bash
# From spark1: sync the worktree to spark2 and JIT-build the bench_rows module there in a no-GPU container
# (CPU nvcc, ~3-4 min), so the locked GPU window does not compile. Same paths as kernel_study/p2b_kbits/spark2.sh.
#   kernel_study/p2b_rows/prebuild_spark2.sh [image]
set -euo pipefail
IMG="${1:-dsv41-flash-exl3-sm121:e15-kbits}"
here=$(cd "$(dirname "$0")/../.." && pwd)
dst=/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2/$(basename "$here")
rsync -a --delete --exclude .git --exclude '__pycache__' --exclude 'kernel_study/*/build*/' "$here/" "spark2:$dst/"
ssh spark2 "docker run --rm --name dsv41-1017-rows-prebuild --network none --memory 16g --cpus 8 --user 1000:1000 \
  -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST=12.1a -e TORCH_EXTENSIONS_DIR=/repo/kernel_study/p2b_kbits/build/torch_ext \
  -v $dst:/repo -w /repo --entrypoint python3 $IMG -c '
import sys, time; sys.path.insert(0, \"kernel_study/p2b_rows\")
import bench_rows
t = time.time(); m = bench_rows.build_ext(); print(\"built\", m.__file__, round(time.time() - t, 1), \"s\")
'" 2>&1 | grep -v -e '^dsv41-patch' -e '^dsv41-indexer' -e 'importing.py' -e 'interface.py' -e 'cpp_extension.py:' -e '^unchanged'
