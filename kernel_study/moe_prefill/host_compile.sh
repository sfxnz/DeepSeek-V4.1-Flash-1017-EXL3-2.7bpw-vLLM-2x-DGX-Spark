#!/usr/bin/env bash
# CPU only: compile docker/patch/moe_prefill_tc.cuh (every kernel instance the binding launches) for sm_121a
# with the host CUDA toolkit, print ptxas registers / spills per kernel and the hot-loop instruction mix.
#   EXL3_EXT=<dir with util.cuh, ptx.cuh and quant/> kernel_study/moe_prefill/host_compile.sh [out_dir]
# EXL3_EXT is the image's exllamav3/exllamav3_ext (ExLlamaV3 1.4.9 @ 5be8865), e.g. extracted with
#   docker create --name x dsv41-flash-exl3-sm121:e15-kbits && docker cp \
#     x:/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext ./exl3ext && docker rm x
# ~0.5 GB RSS, ~15 s. Fails if any kernel spills.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
ext=${EXL3_EXT:?set EXL3_EXT to an exllamav3_ext directory}
out=${1:-$here/kernel_study/moe_prefill/build/host}
mkdir -p "$out"
nvcc -std=c++20 -O3 -gencode=arch=compute_121a,code=sm_121a -Xptxas -v -diag-suppress 20012 \
  -I"$here/docker/patch" -I"$ext" -I"$ext/quant" \
  -c "$here/kernel_study/moe_prefill/host_compile.cu" -o "$out/host_compile.o" > "$out/ptxas.log" 2>&1
cuobjdump -sass "$out/host_compile.o" > "$out/host_compile.sass"
python3 "$here/kernel_study/moe_prefill/sass_stats.py" "$out/ptxas.log" "$out/host_compile.sass"
