#!/usr/bin/env bash
# perf2 G0 boot B0c (= B0 with torch_profiler_with_stack=false; B0b export of a stacks-on trace tripped memwatch): target 4-session config on the release stack (main 20ee8ba, e15-kbits, 2.7bpw-mcg-hq).
# Task command + the torch profiler endpoint (idle-cost-free per Sept P3 K_p: profiler-idle L.A.I.L 48.84 ms/step
# vs 48.33/48.91 on the same config without it). torch_profiler_dump_cuda_time_total=false skips the
# key_averages() table on stop_profile (lazy torch 2.13 parse -> no FunctionEvent build in the worker; the
# trace file is unchanged). FORCE_UNSAFE_CTX=1 for this campaign boot only (MAX_NUM_SEQS=4 > release guard 2).
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-1017-2X-DGX-SPARKS-EXL3
export SKIP_DOWNLOAD=1 AUDIT=strict FORCE_UNSAFE_CTX=1 MAX_NUM_SEQS=4 MAX_MODEL_LEN=524288
export EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/dsv41-traces","torch_profiler_dump_cuda_time_total":false,"torch_profiler_with_stack":false}'
exec ./run.sh
