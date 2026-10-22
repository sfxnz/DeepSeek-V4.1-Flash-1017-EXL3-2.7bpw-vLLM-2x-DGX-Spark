#!/usr/bin/env bash
# api-parity APIG: stack tip (api/05-api-eval worktree on api/04-gateway), image e16-rows (both nodes).
# PLAIN defaults: no env beyond SKIP_DOWNLOAD=1 AUDIT=strict (the deliverable test; the wheel is already provisioned).
set -euo pipefail
cd /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-eval
export SKIP_DOWNLOAD=1 AUDIT=strict
exec ./run.sh
