---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: vllm
tags:
  - exl3
  - quantized
  - vllm
pipeline_tag: image-text-to-text
---

# DeepSeek-V4.1-Flash-1017 EXL3 2.7 bpw MCG (mixed K)

EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Only the routed experts are re-encoded: EXL3 MCG trellis with an integer K in {2, 3, 4} per layer and projection group, exactly 2.70 bpw on average, encoded with real activation Hessians. K follows the measured sensitivity of the official 1017 model. The lm_head is MXFP8. The DSpark/MTP drafter experts and every other tensor keep the source dtypes. Engram tables stay on NVMe at serve time.

This pack exists so 2× NVIDIA DGX Spark (GB10) can serve the model at tensor-parallel 2. Native MXFP4/MXFP8 is about 510 GB and does not fit 2× Spark UMA.

## Download

```bash
hf download sfxnz/DeepSeek-V4.1-Flash-1017-EXL3 --revision 2.7bpw-mcg-hq
```

The two Engram shards are about 95 GiB each. Keep Hugging Face xet enabled. Do not set `HF_HUB_DISABLE_XET`.

## Serve

The 2× DGX Spark recipe is [sfxnz/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-vLLM-2x-DGX-Spark). Clone it, download this revision on both nodes, build the image `dsv41-flash-exl3-sm121:e15-kbits` on both nodes, then run `./run.sh` (and `./stop.sh` when done). Stock vLLM and ExLlamaV3 wheels do not load this model, and the earlier `canonical-e14` image cannot load mixed K. Each node needs about 450 GB free disk (pack plus image).

## Results

Measured on 2× DGX Spark at the recipe's release defaults (DSpark k=3, KV pool 3.875 GiB = 1,108,848 tokens, 1.06× one 1,048,576-token request), boot D1, 2026-10-18:

| Cell | Result |
|---|---|
| L.A.I.L prose c=1 (512 tokens, t=0.2) | 39.6 tok/s |
| Greedy prose c=1 / structured c=1 (200 tokens, `ignore_eos`; the prose prompt stops naturally at 83) | 49.6 / 77.0 tok/s |
| c=2 aggregate, prose / structured | 75.8 / 140.3 tok/s |
| Prefill, novel text, 4k / 16k / 64k | 795 / 755 / 733 tok/s |
| Teacher-forced NLL, 40 passages | 0.1445 nats/token |
| GSM8K-100 thinking off / GSM8K-40 thinking on | 97/100 / 39/40 |
| MMLU 4×57 | 208/228 (91.2%) |
| Tool calls | 22/22 exact args, 8/8 no-call |
| Needle 8k / 32k / 128k × 3 depths | 9/9 |
| Needle 256k / 1M (3 depths each) | 3/3 / 3/3 |
| Served KL vs native unquantized 1017 (72 rows, 92,818 tokens) | 0.025 nats/token, ΔNLL +0.63%, top-1 95.3% (encyclopedic worst: 0.114, 88.1%) |
| 60-minute c=2 soak, mixed checked traffic | 4,296 requests, 0 errors, 4,295 correct |

The recipe's README has the full table, the memory notes and the quality baseline.

## Pack

| Field | Value |
|---|---|
| Source | `deepseek-ai/DeepSeek-V4.1-Flash-1017` commit `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Revision | `2.7bpw-mcg-hq` |
| Quant | `quant_method=exl3`, codebook MCG. Routed experts (40 layers × 384, source MXFP4) at 2.70 bpw, K per layer and group in `quantization_config.expert_bits` = `{"<layer>": [K_gu, K_d]}` (GU = w1+w3, D = w2; layer bpw = (2·K_gu + K_d)/3) |
| Allocation | Layers 0-6 2/2; 7-9 2/3; 10 3/3; 11 2/3; 12 3/3; 13 2/3; 14-15 3/3; 16 2/3; 17-18 3/3; 19 3/4; 20-33 3/3; 34 2/3; 35 3/3; 36 2/3; 37-39 3/3 |
| Encoder | ExLlamaV3 1.5.1 tail-biting Viterbi + scale refit with real activation Hessians (`dsv41-hessian-v1`) |
| Non-routed | Dense, attention, shared-expert and indexer weights: source MXFP8. lm_head: MXFP8 (source shard 43 converted). Embeddings and vision: source BF16. DSpark/MTP drafter experts: source MXFP4, never quantized. Engram tables: source FP8 (shards 47-48) |
| Shards | `model-00001-of-00048.safetensors` through `model-00048-of-00048.safetensors`, plus sidecars |
| Size on disk | 405.1 GB (377.2 GiB) |
| Model memory | 97.73 GiB per TP=2 rank |

Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0. The recipe image clones it at build.

## How it was made

The allocation comes from a native (unquantized) layer-streaming harness of the official 1017 model, in lockstep with the official `model.py` (NLL 0.64707 vs 0.64705). Per projection group it measured the KL from injected noise on a held-out panel of 8 domains (including images, tools, code, math and multilingual), with the noise calibrated to the real per-K error of real-Hessian encodes. An exact dynamic program chose K under the 2.70 bpw budget from an ensemble of 6 estimators (regret ≤ 2.6%; ≤ 4.6% under ±14% calibration jitter); two panels agree (Spearman 0.94).

Validated with every group perturbed at once (2 seeds), KL_eq was 0.050 / 0.049 for this allocation, 0.066 / 0.061 for a code-optimised one, 0.088 / 0.087 for uniform 3/2 (2.67 bpw) and 0.137 / 0.135 for uniform 2/2 (2.0 bpw).

The Hessians were captured from the native 1017 stream on 410,542 calibration tokens (218 sequences of chat, code, math, tools, multilingual and images, disjoint from the panel): per expert, routed-weight-weighted over that expert's tokens and mixed with an all-token prior for cold experts. The encode took 13.9 h and 14.3 h on two Sparks in parallel. All 41 per-unit gates passed with no fallback. Mean relative weight error is 0.32-0.34 at K=2, 0.160-0.172 at K=3 and 0.087 at K=4 (down projection); the lm_head MXFP8 relative error is 0.027.

The recipe ships the tools to rebuild it: `tools/calib/` (Hessian capture and sensitivity allocation) and `tools/build_pack_1017.py` (`plan`, `run --alloc --hessian-mode real`, `assemble`), driven per node by `tools/phase_c.py`.
