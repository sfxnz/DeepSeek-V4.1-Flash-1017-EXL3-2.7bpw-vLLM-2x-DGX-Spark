#!/usr/bin/env bash
# 14:51:17Z a unit-test run in the api/04-gateway worktree ran the real stop.sh (tests/test_recipe_ops.py
# test_stop_sh_orchestrate_zero_is_local_only, GATEWAY_CONTAINER_NAME unset) and removed the live gateway
# (docker events: kill, die 137, destroy). The vLLM ranks stayed up. This restarts the gateway with run.sh's
# start_gateway command plus MALLOC_MMAP_THRESHOLD_=131072 (the final PR 4/5 config); the gateway/ code is the serve's.
docker run -d --name dsv41-api-gateway --restart on-failure:3 --network host --memory 3g --cpus 4 \
  -v /dev/null:/usr/lib/python3.12/sitecustomize.py:ro -v /home/sfxnz/.cache/dsv41-gateway/deepseek_recipe-0.2.0:/opt/dsr:ro -v /home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-eval/gateway:/opt/dsv41-gateway:ro \
  -v /home/sfxnz/.cache/huggingface/hub:/cache/huggingface/hub:ro \
  -e PYTHONPATH=/opt/dsr -e PYTHONDONTWRITEBYTECODE=1 -e MALLOC_MMAP_THRESHOLD_=131072 \
  -e DSV41_GATEWAY_UPSTREAM=http://127.0.0.1:8001 -e DSV41_GATEWAY_MAX_MODEL_LEN=500000 \
  -e DSV41_GATEWAY_STRICT=0 -e DSV41_GATEWAY_BLOCK_PRIVATE_URLS=0 \
  --entrypoint python3 dsv41-flash-exl3-sm121:e16-rows /opt/dsv41-gateway/dsv41_gateway.py --host 0.0.0.0 --port 8000 \
  --tokenizer /cache/huggingface/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq/tokenizer.json \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017
