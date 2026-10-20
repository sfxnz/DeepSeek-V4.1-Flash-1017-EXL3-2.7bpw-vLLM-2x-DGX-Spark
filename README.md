# Serve DeepSeek-V4.1-Flash-1017 EXL3 2.7 bpw on 2× DGX Spark

Serve an EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (source commit `1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2 with vLLM. The pack is [sfxnz/DeepSeek-V4.1-Flash-1017-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3) at revision `2.7bpw-mcg-hq`. The image is `dsv41-flash-exl3-sm121:e15-kbits`. The served model name is `deepseek-ai/DeepSeek-V4.1-Flash-1017`.

Native MXFP4 and MXFP8 weights are about 510 GB and do not fit 2× Spark UMA. The pack re-encodes only the routed experts (40 layers × 384, source MXFP4) as EXL3 MCG trellis with an integer K in {2, 3, 4} per layer and projection group (GU = gate+up, D = down). The routed mean is exactly 2.70 bpw. K is not uniform: the allocation puts the bits where the measured sensitivity of the official 1017 model is highest. Jointly perturbed, the chosen allocation scores KL_eq 0.050, against 0.088 for uniform 3/2 at 2.67 bpw and 0.137 for uniform 2/2 at 2.0 bpw ([How the pack was made](#how-the-pack-was-made)). The experts are encoded with real activation Hessians. Everything else keeps the source dtypes, the lm_head is MXFP8, and the Engram tables stay on NVMe (`DSV41_ENGRAM_DISK=1`).

On boot D1 the pack serves 39.6 tok/s on the L.A.I.L prose cell and 49.6 tok/s greedy prose at c=1, holds one 1,048,576-token request in its KV pool, and passes the full quality eval (GSM8K 97/100, MMLU 208/228, tools 22/22, needle 9/9). It finds a needle at 256k and at 1M tokens (3/3 depths each), stays within 0.025 nats/token KL of the native unquantized model, and ran a 60-minute mixed c=2 soak with 0 errors. Those numbers were measured with `MAX_NUM_SEQS=2` and a 1,048,576-token window. The defaults are now four sessions at a 500,000-token window from the same 3.875 GiB KV pool, with packed KV groups ([Memory and context](#memory-and-context)). The perf2 boots measured that configuration (`results/2026-10-19-1017-perf2/`), and boot FINAL ran the full quality eval, the long needles and the 4-session soak on the full perf2 stack (`results/2026-10-19-1017-perf2/FINAL/`).

This recipe replaces the September recipe for `deepseek-ai/DeepSeek-V4.1-Flash` (2.0 bpw pack, rounds 1-36): [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) @ `3de9146`. Its numbers are history, not this pack's.

Default thinking is off (`thinking=false`, `reasoning_effort=low`). Turn it on per request with `chat_template_kwargs`. The efforts are `low` (50), `high` (75) and `max` (100). `xhigh` is not a 1017 effort and is rejected. With thinking on, a small `max_tokens` can return empty `content`.

## Prerequisites

You need:

- Two DGX Spark nodes
- Passwordless SSH from the head to `WORKER_HOST` (default `spark2`)
- Exclusive GPUs on both nodes. Do not start the serve while another `--gpus all` container is up.
- About 450 GB free disk per node: the pack is 405.1 GB (377.2 GiB) and the image about 24 GB. The source model is not needed to serve.
- Hugging Face `hf` (or `huggingface-cli`) and Docker on both nodes

- The two Sparks' direct ConnectX link up, with an IPv4 address on the same subnet on `IFACE` on both nodes

The recipe defaults are `HEAD_IP=10.100.8.1` (the head's address on `IFACE`), `WORKER_HOST=spark2`, `IFACE=enp1s0f1np1`, and `HCA=rocep1s0f1`. If your fabric differs, export `HEAD_IP`, `WORKER_HOST`, `IFACE`, and `HCA` before `./run.sh`. `run.sh` pins `NCCL_IB_HCA` from `HCA` and the socket interfaces from `IFACE`; exporting `NCCL_IB_HCA` directly has no effect. GB10 exposes four HCAs and two are DOWN: pick the one `ibdev2netdev` maps to `IFACE` with `(Up)`. Read unified memory with `free -h`. Never read VRAM from `nvidia-smi`.

In this README spark1 is the head and spark2 the worker.

## Clone the recipe

Clone on both nodes at the same commit (the worker needs `docker/` to build the image). `./run.sh` and `./stop.sh` run from the head's checkout; `run.sh` copies its patch directory to the worker. The default branch is `main`.

```bash
git clone https://github.com/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-vLLM-2x-DGX-Spark.git
cd DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-vLLM-2x-DGX-Spark
```

## Download the pack

Run the download on both nodes. Keep Hugging Face xet enabled. Do not set `HF_HUB_DISABLE_XET`. The revision is 405.1 GB (48 shards plus sidecars).

```bash
unset HF_HUB_DISABLE_XET
export HF_XET_HIGH_PERFORMANCE=1
hf download sfxnz/DeepSeek-V4.1-Flash-1017-EXL3 --revision 2.7bpw-mcg-hq
```

`hf download` writes `refs/2.7bpw-mcg-hq` and `snapshots/<commit>/`. `./run.sh` resolves that layout under `HF_CACHE` (default `~/.cache/huggingface`, the same path on both nodes); it does not read `HF_HOME`. If you download elsewhere, `export HF_CACHE=<that HF_HOME>` before `./run.sh`. A pack assembled locally at `snapshots/2.7bpw-mcg-hq` (Rebuild the pack) also works. If both exist, `run.sh` uses the Hub commit snapshot. If the pack is already in the Hub cache on a node, skip the download on that node.

## Build the image

On both nodes, from this repo:

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58
docker build -f docker/Dockerfile -t dsv41-flash-exl3-sm121:e16-rows docker
```

If the image `dsv41-flash-exl3-sm121:e16-rows` is already present, skip the pull and the build on that node. A node that already has `dsv41-flash-exl3-sm121:e15-kbits` can layer the rows stage on it instead of a full build: `docker build -f docker/Dockerfile.e16 -t dsv41-flash-exl3-sm121:e16-rows docker` (`e15-kbits` itself is `docker/Dockerfile.e15` layered on `canonical-e14`). `canonical-e14` alone cannot serve the mixed-K pack. `e15-kbits` serves it, but runs decode batches of 9..24 rows (three or four sessions) on the slower exllamav3 `exl3_moe` path.

Stock `vllm/vllm-openai` wheels do not load `DeepseekV41ForCausalLM`. The image starts from the pinned `deepseekv41-flash-0909` digest and overlays Engram-on-disk plus `vllm-exl3`. The K-bits stage (`docker/patch/widen_p2b_kbits.py`) builds the coop and dataflow MoE kernels at K 2/3/4 with K_gu ≠ K_d, reads `quantization_config.expert_bits` in the plugin, and runs mixed-K prefill on `exl3_moe_kmix`. Packs without `expert_bits` load as before. The rows stage (`docker/patch/widen_p2b_rows.py`) adds the dataflow MoE kernel for 9..24 decode rows, which three or four sessions verify at DSpark k=3 (`DSV41_P2B_WIDE=0` keeps the 8-row contract). The image also prebuilds the tensor-core prefill MoE extension (`docker/patch/moe_prefill_tc.py`, `DSV41_MOE_PREFILL_TC=1`); with edited kernel sources it compiles at first use (~1-2 min per rank). At boot each rank logs one `routed experts K_gu/K_d` line per layer, and the engagement audit checks them against the pack. The image carries a `dsv41.recipe.patches` label. `run.sh` warns, but still boots, when `IMAGE` lacks it.

## Run

If another `--gpus all` container is up, stop it first.

On the head node:

```bash
./run.sh
python3 smoke_chat.py
python3 smoke_vision.py
python3 bench_decode.py --phase prose --concurrency 1
python3 tools/measure_lail_prose.py
```

`./run.sh` starts the worker, then the head, and returns when the serve is ready. It prints `==> Ready → http://127.0.0.1:8000/v1  (context=500000)`, then `==> audit ok`, the warmup replies, and `==> Stop with: ./stop.sh`. Run the smokes after that. Follow the head with `docker logs -f dsv41-flash-exl3`.

The API is `http://127.0.0.1:8000/v1`. The served model is `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Cap is `MAX_NUM_SEQS=4`. Do not send a fifth stream. Boot D1 took about 11 minutes (model load 508 s on the head, 305 s on the worker).

The `smoke_chat.py` default prompt is `What is 17*19? Return only the integer.` Thinking is off. The pass is `content` matching `323` (`--expect ''` accepts any non-empty content). `smoke_vision.py` sends a 64x64 solid-red PNG as OpenAI `image_url`. It must not return HTTP 400 `is not a multimodal model`, and the answer must contain `red`. `bench_decode.py` is streamed greedy, thinking off, 200 completion tokens, 3-run median by default (the headline cell uses `--runs 9`).

If `ORCHESTRATE=auto` (the default) and SSH to `WORKER_HOST` fails, `run.sh` exits 1. It does not start a TP=2 head rank alone.

When you are done:

```bash
./stop.sh
```

Run it on the head. It removes `dsv41-flash-exl3` on both nodes (the worker over ssh). If it reports `Cannot SSH to <worker>`, run `docker rm -f dsv41-flash-exl3` on the worker.

The serve levers are the ones the September recipe promoted, all default-on: DSpark k=3 (`NUM_SPECULATIVE_TOKENS=3`) with cudagraph capture sizes derived from k and `MAX_NUM_SEQS` (`[1,3,4,6,8,9,12,16]` at 4 sessions), `MAX_NUM_BATCHED_TOKENS=8192` (`2048` is the override for long-prompt-heavy workloads), the NCCL AR-tail set, the mem-hygiene bundle, Engram prefetch + gather v2 + census + `DSV41_ENGRAM_WILLNEED`, lm_head MXFP8 (`DSV41_LMHEAD_MXFP8=1`), `DSV41_STREAM_FEED`, `DSV41_WOA_PREPACK`, `DSV41_DSPARK_SPARSE_MARKOV` with `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024`, and the round-3 kernel bundle (`DSV41_P2B_COOP=2`, `DSV41_DENSE_GEMV=1`, `DSV41_MHC_DET_SPLITS=16`, `DSV41_ENGRAM_NATIVE_STAGE=1`, `DSV41_ENGRAM_EARLY_HASH=1`, `DSV41_ATTN_T2R_DEDUP=1`, `DSV41_SWA_META_FUSED=1`, `DSV41_MOE_PREP_FUSED=1`, `DSV41_CANDIDATE_MASK_BOUNDED=1`, `DSV41_INDEXER_WP_GEMV=1`). Set any `DSV41_*` lever to `0` to turn it off (`DSV41_DSPARK_SPARSE_MARKOV_TOPK=256` restores the round-34 top-k; `SPEC=none` turns speculative decoding off). `flags.md` has the per-flag record; the September A/B evidence is in the September recipe.

The 4-session defaults add packed KV groups (`DSV41_KV_PACKED_GROUPS=1`, vLLM PR #60996 on the image's planner), batched SWA decode metadata (`DSV41_SWA_META_BATCHED=1`, needs `DSV41_SWA_META_FUSED=1`), a ragged-batch guard in the sparse indexer (always on, from vLLM 0.32rc1), the indexer's expanded block table on first use only (`DSV41_INDEXER_EXPANDED_TABLE_LAZY=1`), `--mm-processor-cache-gb 1` and `--prefix-cache-retention-interval 4096`. Boot I3 ran them with the later perf2 levers off: c=1 prose 50.3 tok/s at 52.7 ms/step (the config-only 4-session baseline B0: 49.0 at 52.7), c=4 distinct prose 62.3 tok/s aggregate (B0 61.4), quick quality PASS (`results/2026-10-19-1017-perf2/I3/`).

## Memory and context

The model takes 97.73 GiB per TP rank. The KV pool is `KV_CACHE_MEMORY=4160749568` (3.875 GiB). The image's planner splits the 43 sliding-window layers and 4 compressed-KV layers into 18 KV groups on 138,240-byte blocks, and the ratio-1 group pays a full block for 46,080 bytes of pages. Packed groups (`DSV41_KV_PACKED_GROUPS=1`) give 50 groups on 46,080-byte blocks. Per-layer page sizes and kernels do not change. The image's own planner and KV manager, run on CPU, give these numbers (the perf2 boots log the 500,000 / 8192 one):

- at the 500,000-token window: 1,471,832 tokens (2.94×) at `MAX_NUM_BATCHED_TOKENS=8192`, 1,793,682 (3.59×) at 4096. The image's 18 groups give 964,308 (1.93×) at 8192 (the perf2 baseline boot logged 975,641, 1.86×, at 524,288).
- four sessions growing to 500,000 tokens at once, with no preemption, need 3.82 GiB at 8192 and 3.59 GiB at 4096. That is why the window is 500,000. Four full 524,288-token sessions need 3.99 GiB at 8192 (more than the pool) and 3.75 GiB at 4096, so `MAX_MODEL_LEN=524288 MAX_NUM_BATCHED_TOKENS=4096` allows four sessions at 2^19.

Packing also adds attention-metadata buffers outside the KV pool (28 more SWA builders, 2 more indexer builders). Every indexer builder used to allocate an expanded block table (B × blocks of one max-length request × 4 bytes) that only the flattened decode path reads; SM120 decode is native, so `DSV41_INDEXER_EXPANDED_TABLE_LAZY=1` allocates it on first use only. Estimated from the builders' allocations, per rank at 500,000 and 8192: 513 MiB with the image's grouping and allocation, 985 MiB packed, 374 MiB packed with the lazy table (4096: 256 / 492 / 187 MiB).

A 4.5 GiB pool would leave spark1 about 3.7 GiB `MemAvailable` after the smokes (estimate from boot D1), below the 4 GiB floor target.

`MemAvailable` on boot D1 (`MAX_NUM_SEQS=2`, 1M window, 18 KV groups; spark1 / spark2):

- minimum during boot: 4.6 / 5.4 GiB
- after the smokes: 4.3 / 5.9 GiB
- normal serving, including the 128k needle: ≥ 3.8 GiB
- minimum under the quality eval: 3.17 GiB on spark1 for about 10 s, during `quality_eval.py`'s NLL component (full-vocab `prompt_logprobs` logits on the head)
- 256k and 1M needles: spark1 dips to 2.1–2.8 GiB for 2 s or less at prefill boundaries, and to 1.8 GiB for 2 s or less while the API server tokenizes a 1M-token prompt (the engine is idle then)
- release boot r1: 4.4 / 5.2 GiB after the smokes and 4.1 / 5.5 GiB after the 60-minute soak; spark2's lowest sample over all tests was 3.6 GiB

These short dips use page cache and swap headroom and did not OOM. The default window is 500,000 tokens. A 1M-token request needs `MAX_MODEL_LEN=1048576`; with packed groups the pool holds 1.73× of one such request (CPU planner), and two 1M-token requests do not fit at once. With the 4-session defaults, MemAvailable minima per phase are in each perf2 boot's `numbers.txt` (`results/2026-10-19-1017-perf2/`); boot I3's lowest was 2.0 GiB on spark1 during the prefill micro bench.

The memory watchdog (stop at under 1.5 GiB on either node for two 2 s samples) never tripped in any test. After a prefill step of 8192 tokens or more (`DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192`), the allocator releases its unused cached blocks while `MemAvailable` is below 4 GiB (`DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=4`).

## Quality eval

`tests/quality_eval.py` measures output quality against a running serve. It is stdlib-only and uses HTTP only. `--quick` covers:

- teacher-forced NLL on 40 public-domain passages (`prompt_logprobs`, with BOS)
- a decode-vs-prefill logprob probe
- 30 tool-call items (JSON-valid, exact-args and no-call rates)
- needle recall at 8k/32k × 3 depths, on generated filler
- a 12×2 greedy self-consistency control
- a c=2 sanity check
- the red-PNG vision check

`--full` adds GSM8K-100 with thinking off, GSM8K-40 with thinking on, MMLU 4×57, and the needle at 128k. The run exits 1 when a gate fails.

```bash
python3 tests/quality_eval.py --full --out /tmp/q.json \
  --baseline results/2026-10-18-1017/quality-baseline/full.json
```

With `--baseline`, it gates on the following:

| Metric | Gate |
|--------|------|
| NLL | ≤ baseline + max(0.01, 3× repeat noise) nats |
| Decode probe | median \|Δlogprob\| ≤ baseline + 0.05, and greedy-text NLL ≤ baseline + 0.15 |
| Rates | Wilson 95% upper bound with one item of slack ((k+1)/n) ≥ baseline rate |
| Needle | found ≥ baseline |
| Golden flip hazard | ≤ 2× max(baseline A/A hazard, 0.005); the run's own A/A hazard has the same limit |

The golden flip hazard compares greedy text with the baseline's own greedy run, so it applies only when the baseline ran the same pack (weights). It gates kernel and numerics levers. A pack change is judged on paired NLL over the fixed passages, paired task scores and A/A self-consistency, and then the baseline is re-recorded on the new pack (ARMS.md step 6). Vision and c=2 must always pass. `--result saved.json --baseline other.json` re-gates two saved runs offline with no traffic, which is how an A/B compares two boots. Run it serialized: never next to a bench, and never with a third stream. The vendored data and licenses are in `tests/quality/README.md`.

The 1017 baseline for `2.7bpw-mcg-hq` is `results/2026-10-18-1017/quality-baseline/`, recorded on boot D1. `full.json` passed:

| Component | Result |
|---|---|
| NLL (40 passages) | 0.1445 nats/token (repeat delta 0.0018) |
| Decode probe | median \|Δlogprob\| 0.00069 |
| Tools | 22/22 exact args, 8/8 no-call |
| Needle | 9/9 (8k / 32k / 128k × 3 depths) |
| Self-consistency | 5/12 identical |
| c=2, vision | pass |
| GSM8K-100, thinking off | 97/100 |
| GSM8K-40, thinking on | 39/40 |
| MMLU 4×57 | 208/228 (91.2%) |

The quick baseline (`quality-baseline/quick.json`) was recorded on release boot r1: PASS, NLL 0.1430, decode median |Δlogprob| 0.00096, tools exact, needle 6/6, c=2 and vision pass.

**Long context.** Needle at 256k: 3/3 depths (10% / 50% / 90%), about 400 s per request end to end, roughly 650 tok/s prefill. Needle at 1,047,432 prompt tokens: 3/3 depths, 1908–1999 s per request, roughly 530 tok/s prefill. One request at a time; the KV pool held 1.06× a 1M request on those boot D1 settings (`results/2026-10-18-1017/boot-d1/long-needle*.txt`).

**Served KL against the native model.** `tools/served_kl.py` scores the live serve's `prompt_logprobs` (top 20) against the native unquantized 1017 model's top-128 logits on 72 held-out text rows (92,818 tokens). It measures the whole served stack, so engine numerics count against it as well as the quantization. Eval boot: `MAX_NUM_BATCHED_TOKENS=512 LANGUAGE_MODEL_ONLY=1`, because `prompt_logprobs` materialises full-vocab logits per chunk on the head.

| Domain | Tokens | KL (nats/token) | ΔNLL | Top-1 agreement |
|---|---:|---:|---:|---:|
| All | 92,818 | 0.025 | +0.0059 (+0.63%) | 95.3% |
| Chat | 25,682 | 0.022 | +0.0042 | 95.0% |
| Code | 17,116 | 0.008 | +0.0042 | 97.3% |
| Math | 21,969 | 0.010 | +0.0052 | 96.6% |
| Tool calls | 6,825 | 0.009 | −0.0006 | 96.9% |
| Multilingual | 13,436 | 0.036 | +0.0044 | 94.4% |
| Encyclopedic | 7,790 | 0.114 | +0.0251 | 88.1% |

Encyclopedic text loses the most. Rare-fact recall lives in the routed experts that 2.70 bpw compresses hardest, and the native model is already least certain there (NLL 1.86 against 0.93 overall). Code, math and tool calls are close to native. Evidence: `results/2026-10-18-1017/boot-eval/`.

**Soak.** A 60-minute c=2 run of mixed, answer-checked traffic on release boot r1 (`results/2026-10-18-1017/boot-r1/soak.json`): 4,296 requests, 0 HTTP errors, 4,295 correct. Tool calls 891/891, vision 862/862, prose 844/844, 32k needle 6/6, arithmetic 1,692/1,693. The one miss was a model arithmetic error (561×54 answered 30300, expected 30294; thinking off). Latency p50: arithmetic 0.39 s, vision 0.44 s, tool 1.62 s, 300-token prose 4.64 s.

**Second boot.** Release boot r1 reproduced boot D1: L.A.I.L prose 39.6 tok/s, greedy prose 50.0 tok/s, structured 75.8 tok/s at c=1 (`results/2026-10-18-1017/boot-r1/`).

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `dsv41-flash-exl3-sm121:e16-rows` |
| Model | `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` revision `2.7bpw-mcg-hq` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 500000 |
| `--max-num-seqs` | 4 |
| `--max-num-batched-tokens` | 8192 |
| `--kv-cache-dtype` | `fp8` |
| `--kv-cache-memory` | 4160749568 |
| KV cache groups | packed, one layer-pattern repeat per group (`DSV41_KV_PACKED_GROUPS=1`, vLLM PR #60996): 50 groups on 46,080-byte blocks; `0` restores the image's 18 groups on 138,240-byte blocks |
| `--quantization` | `exl3` |
| Engram | disk (`DSV41_ENGRAM_DISK=1`) |
| `--block-size` | 64 |
| `--prefix-cache-retention-interval` | 4096 (sliding-window prefix-cache checkpoint every 4096 tokens; vLLM's default 0 keeps only the replay boundary) |
| Speculative | DSpark-3 (`SPEC=dspark`) |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes {1} ∪ {s·k, s·(k+1)} for s ≤ `--max-num-seqs` (`[1,3,4,6,8,9,12,16]` at k=3) (`ENFORCE_EAGER=0`, `DSV41_ALLOW_CUDA_GRAPHS=1`) |
| Engram prefetch / census / gather | `DSV41_ENGRAM_PREFETCH=1` `DSV41_ENGRAM_CENSUS=1` `DSV41_ENGRAM_GATHER_V2=1` |
| Engram prefill read-ahead | `DSV41_ENGRAM_WILLNEED=1` (gv2 calls with >= 512 rows fadvise their pages before the preadv loop) `DSV41_ENGRAM_PREFILL_AHEAD=1` (the next chunk's rows are read on 8 threads while the current chunk computes, and used only where they match the GPU's hash ids; `_DONTNEED=1` drops the pages a prefill read brought in) |
| lm_head | MXFP8 (`DSV41_LMHEAD_MXFP8=1`), needs an lm_head-MXFP8 pack (`2.7bpw-mcg-hq` is one: `tools/build_pack_1017.py` converts source shard 43); self-disarms on a bf16-head pack |
| Weight-load stream feed | `DSV41_STREAM_FEED=1` |
| Decode levers | `DSV41_WOA_PREPACK=1` (needs the `dsv41-flash-exl3-sm121:e16-rows` o_proj stage) `DSV41_DSPARK_SPARSE_MARKOV=1` `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024` |
| Round-3 kernel bundle | `DSV41_P2B_COOP=2` (coop dataflow MoE kernel, needs the `dsv41-flash-exl3-sm121:e16-rows` build) `DSV41_DENSE_GEMV=1` `DSV41_MHC_DET_SPLITS=16` `DSV41_ENGRAM_NATIVE_STAGE=1` `DSV41_ENGRAM_EARLY_HASH=1` `DSV41_ATTN_T2R_DEDUP=1` `DSV41_SWA_META_FUSED=1` `DSV41_SWA_META_BATCHED=1` `DSV41_MOE_PREP_FUSED=1` `DSV41_CANDIDATE_MASK_BOUNDED=1` `DSV41_INDEXER_WP_GEMV=1` (set one to `0` to turn it off) |
| Prefill MoE | `DSV41_MOE_PREFILL_TC=1` (tensor-core EXL3 routed-MoE kernel for calls of >= 9 rows, prebuilt in `dsv41-flash-exl3-sm121:e16-rows`; self-checked and verified against the stock path at boot; `0` = stock `exl3_moe` / fat path) |
| Decoder replay (CED prefill skip) | `DSV41_DECODER_REPLAY_ROWS=2670`: in eager prefill steps layer 20 writes its KV for every row and layers 20-39 run on kept rows only: W >= 2414 keeps each prompt's last W rows (exact), 128 keeps each chunk's last 128 rows (the upstream/DeepSeek approximation), 0 is off; SWA blocks of skipped rows stay out of the prefix cache, so a later prompt reuses a trimmed prompt's KV only if it diverges no earlier than 64 tokens before the prompt's last 128-token boundary (with the prefix-end fallback), or at a shared-prefix junction: the second request sharing a system prompt keeps up to 2542 more decoder rows ending there and the third hits it, as with stock e15 |
| Prefix-end fallback | `DSV41_PREFIX_END_FALLBACK=1`: SWA groups also keep the prefix-cache tail one 128-token hit unit below each prompt end (and shared-prefix junction), so every resend or follow-up that diverges at the prompt end hits within 192 tokens; e15 alone misses for about half of all prompt lengths (0 = stock) |
| Prefix reply retention | `DSV41_PREFIX_REPLY_RETAIN=1`: SWA blocks filled during decode enter the prefix cache as they fill, so a follow-up that repeats the assistant reply verbatim hits within 192 tokens of the shared end instead of prefilling the reply again (0 = stock) |
| NCCL AR-tail set | `NCCL_BUFFSIZE=1048576` `NCCL_LL128_BUFFSIZE=262144` `NCCL_PROTO=^LL128` `NCCL_MAX_NCHANNELS=8` |
| Memory hygiene | `DSV41_DROP_PAGE_CACHE=1` `DSV41_INDEXER_PREFILL_FACTOR=1` `DSV41_INDEXER_EXPANDED_TABLE_LAZY=1` `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192` `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=4` `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` |
| Tokenizers / tools / reasoning | `deepseek_v41` |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) |
| `--mm-encoder-tp-mode` | data |
| `--mm-processor-cache-gb` | 1 (vLLM default 4, mirrored in the API server and EngineCore on the head) |
| Default thinking | `thinking=false`, `reasoning_effort=low` |
| Post-ready | engagement audit of both ranks' logs (`AUDIT=warn`; `strict` fails the boot, `off` skips), then greedy, t=0.7, ~3k/~1k/~300-token nonce prefill and small-image warmup requests (`WARMUP=1`) |
| API | `http://<head>:8000/v1` |
| Container | `dsv41-flash-exl3` |
| Master port | 29524 |
<!-- END generated defaults -->

## Measured on 2× DGX Spark

Boot D1 (2026-10-18, both nodes, `AUDIT=strict`, `KV_CACHE_MEMORY=4160749568`, the image and pack above, every other setting at its default). Evidence: `results/2026-10-18-1017/boot-d1/`. The right column is the September recipe's round-36 pack on the September model, kept only as history: a different model, not a paired comparison.

| Cell | 1017 2.70 bpw pack | Sept 2.0 bpw pack, Sept model (previous recipe, history only) |
|---|---|---|
| L.A.I.L prose c=1 (512 tokens, t=0.2, median of 3) | 39.6 tok/s, 52.5 ms/step, acceptance 2.09 | 42.6 tok/s, 48.8 ms/step, acceptance 2.11 |
| `bench_decode.py` prose c=1 (9 runs, 200 tokens) | 49.6 tok/s, 52.2 ms/step, acceptance 2.58 | 50.7 tok/s, 48.5 ms/step, acceptance 2.45 |
| `bench_decode.py` structured c=1 (9 runs) | 77.0 tok/s, 51.4 ms/step, acceptance 3.96 | 84.3 tok/s, 47.5 ms/step |
| `bench_decode.py` prose c=2 aggregate (5 runs) | 75.8 tok/s | 81.0 tok/s |
| `bench_decode.py` structured c=2 aggregate (5 runs) | 140.3 tok/s | 156.3 tok/s |
| Prefill, novel text, 4k / 16k / 64k | 795 / 755 / 733 tok/s | 814 (8k) / 812 (32k) tok/s |
| Prefill, warm, 4k / 16k / 64k | 707 / 689 / 688 tok/s | 773 (8k) / 775 (32k) tok/s |
| Decode after 64k context | 48.1 tok/s | - |
| TTFT, short prompt | ~0.24 s | - |
| `benches/e2e.sh` | 3/3 pass; coding agent 53.9 tok/s decode; 64k document recall hit (TTFT 90.6 s) | - |

`tools/measure_lail_prose.py` matches L.A.I.L's streamed prose (512 tokens, temperature 0.2): it is the real-use cell. `bench_decode.py` is streamed greedy with thinking off. Its prose prompt stops naturally at 83 tokens, so the 200-token cell forces `ignore_eos` and runs on past EOS; the structured prompt runs to 200 tokens. Reproduce with `python3 bench_decode.py --runs 9 --concurrency 1`, `python3 bench_decode.py --runs 5 --concurrency 2`, `benches/micro.sh --contexts 512 4096 16384 65536 --runs 2` and `benches/e2e.sh`. The quality gate is `tests/quality_eval.py --full --baseline results/2026-10-18-1017/quality-baseline/full.json` (`tests/correctness.sh --full` is a separate fixed correctness check).

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 49.6 | 49.6 | 0.24 s |
| prose | 2 | 38.4 | 75.8 | 0.28 s |
| structured | 1 | 77.0 | 77.0 | 0.19 s |
| structured | 2 | 70.2 | 140.3 | 0.23 s |
| lail_prose | 1 | 39.6 | 39.6 | 0.34 s |
<!-- END generated measured -->

## How the pack was made

**Allocation.** A native (unquantized) layer-streaming harness of the official 1017 model (`tools/calib/`) runs in lockstep with the official `model.py` (NLL 0.64707 vs 0.64705). In smooth mode (fp32 activations, frozen routing) it injects noise into one projection group at a time and measures the KL against the native output on a held-out panel of 8 domains (including images, tools, code, math and multilingual). The noise level per K is calibrated to the real per-K output error of real-Hessian encodes (r_eff). An exact dynamic program then picks K per (layer, group) under the 2.70 bpw budget. The choice is an ensemble of 6 sensitivity estimators (regret ≤ 2.6% against each estimator's own optimum; ≤ 4.6% under ±14% r_eff jitter), and two panels agree (Spearman 0.94).

The allocation is `tools/calib/alloc/expert_bits_2.70.json` (K_gu/K_d; layer bpw = (2·K_gu + K_d)/3):

| Layers | K_gu/K_d |
|---|---|
| 0-6 | 2/2 |
| 7-9, 11, 13, 16, 34, 36 | 2/3 |
| 10, 12, 14-15, 17-18, 20-33, 35, 37-39 | 3/3 |
| 19 | 3/4 |

That is 7 layers at 2/2, 8 at 2/3, 24 at 3/3 and 1 at 3/4. The pack's `config.json` carries it as `quantization_config.expert_bits` = `{"<layer>": [K_gu, K_d]}`.

**Joint validation.** With every group perturbed at once (smooth mode, 2 seeds), KL_eq was:

| Allocation | Routed bpw | KL_eq (seed 3 / seed 4: `joint_s3.json` / `joint_s4.json`) |
|---|---|---|
| Chosen | 2.70 | 0.050 / 0.049 |
| Code-optimised | | 0.066 / 0.061 |
| Uniform 3/2 | 2.67 | 0.088 / 0.087 |
| Uniform 2/2 | 2.0 | 0.137 / 0.135 |

The ranking matched the prediction in every domain except tools on one seed. Evidence: `results/2026-10-18-1017/allocation/`.

**Encode.** ExLlamaV3 1.5.1 tail-biting Viterbi plus scale refit, with real activation Hessians in the `dsv41-hessian-v1` format: per expert, the routed-weight-weighted H over that expert's tokens, mixed with an all-token prior for cold experts (the w2 prior comes from 512 random tokens). The Hessians were captured from the native 1017 stream on 410,542 calibration tokens (218 sequences of chat, code, math, tools, multilingual and images, disjoint from the panel). Wall time was 13.9 h and 14.3 h on the two Sparks in parallel.

**Gates.** All 41 per-unit gates passed (40 expert shards plus the lm_head), with no fallback. Mean relative weight error:

| K | GU | D |
|---|---|---|
| 2 | 0.32-0.33 | 0.33-0.34 |
| 3 | 0.160-0.166 | 0.167-0.172 |
| 4 | | 0.087 |

The lm_head MXFP8 relative error is 0.027. Evidence: `results/2026-10-18-1017/encode/`.

**Non-routed tensors.** Dense, attention, shared-expert and indexer weights are the source MXFP8. The lm_head is MXFP8 (source shard 43 converted). Embeddings and vision are the source BF16. The DSpark/MTP drafter experts stay at the source MXFP4 and are never quantized. The Engram tables are the source FP8 on NVMe (shards 47-48).

## Rebuild the pack

You do not need this to serve: the published revision is the serve path. Rebuilding needs the GPUs of both Sparks for about 14 hours and exclusive GPUs. Stop the serve (`./stop.sh`) and any quant container before you start, and before `./run.sh` again.

The capture and encode run inside the quant image `dsv41-quant151` (ExLlamaV3 1.5.1): `docker build -f docker/Dockerfile.quant151 -t dsv41-quant151 docker/`, on both nodes. Never serve from that image. It builds `FROM dsv41-flash-exl3-sm121:canonical-e12`, which this repo does not build: that is the September recipe's `docker/Dockerfile` output at commit `f67434b1fa816cc5bcca876eb51cbef1d89d5f48` (`docker build -f docker/Dockerfile -t dsv41-flash-exl3-sm121:canonical-e12 docker` in that checkout).

The tools default to the author's paths under `/home/sfxnz/projects/data/dsv41-1017`. Pass your own: `--out` (builder state and shards, same path on both nodes), `--work`, `--calib`, `--ssh-dir ~/.ssh`.

1. Download the official snapshot on both nodes.

```bash
python3 tools/download_official.py
```

The commit is `1bae0e1088993b9b82de08f59de5be5a34e42170` (`deepseek-ai/DeepSeek-V4.1-Flash-1017`). The two Engram shards are about 95 GiB each and need `hf_xet`. Do not set `HF_HUB_DISABLE_XET`.

2. Build the calibration set (required: the Hessian capture reads it). `python3 tools/calib/data/build.py --out <data dir>/calib` writes `calib.jsonl` (410,542 tokens) and the disjoint `panel.jsonl`, encoded exactly as the 1017 model sees them (record schema: the `tools/calib/data/load.py` docstring).

   Redoing the allocation is optional. The shipped `tools/calib/alloc/expert_bits_2.70.json` is the allocation above. To redo it: `tools/calib/cli.py` (run through `tools/calib/run_gpu.sh`) is the native harness (`nll`, `panel`, `sens`, `swap`), `tools/calib/sens_sweep.py` plans the sensitivity passes, `tools/calib/r_eff.py` measures r_eff on real-Hessian encodes, and `tools/calib/alloc/alloc.py` with `tools/calib/alloc/ensemble.py` solves the allocation.

3. Check the plan: sizes, per-node disk, per-rank GiB and ETA (host, no torch).

```bash
python3 tools/build_pack_1017.py plan --alloc tools/calib/alloc/expert_bits_2.70.json --hessian-mode real
```

4. Capture Hessians and encode, one driver per node. `tools/phase_c.py` holds the node's GPU lock and loops: it encodes every layer whose Hessians are complete (`build_pack_1017.py run --layers <ready> --hessian-mode real`), deletes those Hessians, then captures the next layers that fit on disk (`tools/calib/capture_h.py`, resuming the native stream from its checkpoint). Hessians are about 48.4 GB per layer, so they are streamed, not kept. The builder state lives on spark1; spark2 points at it with `--state-host spark1`. A crash or reboot resumes at unit granularity.

```bash
python3 tools/phase_c.py split --alloc tools/calib/alloc/expert_bits_2.70.json   # balanced layer sets

# spark1
python3 tools/phase_c.py run --node spark1 --layers <spark1 layers> --lmhead \
  --alloc tools/calib/alloc/expert_bits_2.70.json --gates results/2026-10-18-1017/encode/gates.json \
  --calib <data dir>/calib/calib.jsonl --out <out dir> --work <work dir> --ssh-dir ~/.ssh
# spark2
python3 tools/phase_c.py run --node spark2 --layers <spark2 layers> --state-host spark1 \
  --alloc tools/calib/alloc/expert_bits_2.70.json --gates results/2026-10-18-1017/encode/gates.json \
  --calib <data dir>/calib/calib.jsonl --out <out dir> --work <work dir> --ssh-dir ~/.ssh

python3 tools/phase_c.py status --node spark1 --layers <spark1 layers> \
  --alloc tools/calib/alloc/expert_bits_2.70.json --work <work dir>
python3 tools/build_pack_1017.py status
```

5. Assemble on spark1. `assemble` pulls spark2's shards and refuses any whose size or sha256 differs from its done record, hard-links the source shards that are not re-encoded (vision, embed, DSpark/MTP, Engram), and writes `config.json` with `expert_bits`, the index and the manifest to `snapshots/2.7bpw-mcg-hq` in the Hub cache.

```bash
python3 tools/build_pack_1017.py assemble --node spark1 --out <out dir> \
  --src ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
```

spark2 needs the same snapshot: copy spark1's built shards to `<out dir>/shards/` on spark2 and run the same `assemble --node spark2 --state-host spark1` there (it hard-links spark2's own source copy), or copy the snapshot directory. Then `./run.sh`. `--hessian-mode identity` is the uncalibrated fallback (H = I, no Hessians needed). The release pack used `real`. Pick K from a measured allocation only (`--alloc`).

## License

MIT for the scripts. Model weights are MIT from DeepSeek. `vllm-exl3` is AGPL-3.0.
