#!/usr/bin/env bash
# perf2 FINAL (main go 2026-10-20 ~04:40Z): release worktree @4ae29a6 (pr/10-evidence), image e16-rows sha256:df157a2d8afe (both nodes).
# PLAIN defaults: no env beyond SKIP_DOWNLOAD=1 AUDIT=strict (the deliverable test).
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-release
export SKIP_DOWNLOAD=1 AUDIT=strict
exec ./run.sh
