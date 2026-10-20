#!/usr/bin/env bash
# perf2 FINAL (c): served-KL eval boot, apples to apples with results/2026-10-18-1017/boot-eval (MAX_NUM_BATCHED_TOKENS=512
# LANGUAGE_MODEL_ONLY=1), from the release tree @4ae29a6 on image e16-rows df157a2d8afe; then tools/served_kl.py on all rows.
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-release
export SKIP_DOWNLOAD=1 AUDIT=strict MAX_NUM_BATCHED_TOKENS=512 LANGUAGE_MODEL_ONLY=1
exec ./run.sh
