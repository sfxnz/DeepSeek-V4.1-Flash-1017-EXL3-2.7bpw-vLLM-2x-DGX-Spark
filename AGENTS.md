# AGENTS.md — DeepSeek-V4.1-Flash-1017 EXL3 2.7bpw · 2× DGX Spark

Serve an EXL3 pack of `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170` at TP=2, served as `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Pack `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` @ `2.7bpw-mcg-hq`. Local image `dsv41-flash-exl3-sm121:e16-rows`, built by `docker/Dockerfile` from the pinned `vllm/vllm-openai:deepseekv41-flash-0909` digest, or by `docker/Dockerfile.e16` layered on `e15-kbits` (`docker/Dockerfile.e15` on `canonical-e14`). Engram stays on NVMe. Stock ExLlamaV3 and stock vLLM wheels do not load `deepseek_v41` / CED / CSA2 / Engram.

Humans read [README.md](README.md).

Pack: the routed experts (40 layers × 384, source MXFP4) are EXL3 MCG with integer K in {2,3,4} per layer and projection group (GU = w1+w3, D = w2), in `quantization_config.expert_bits` = `{"<layer>": [K_gu, K_d]}`. The routed mean is exactly 2.70 bpw. The allocation is `tools/calib/alloc/expert_bits_2.70.json`, chosen from measured sensitivity (`results/2026-10-18-1017/allocation/`). `tools/build_pack_1017.py` encodes it with real activation Hessians (`plan` / `run --dry-run`, `run`, `encode`, `status`, `launch-cmd`, `assemble`; Hessian format `dsv41-hessian-v1`, see its docstring). lm_head is MXFP8. Dense, attention, shared and indexer weights stay source MXFP8, and the DSpark/MTP drafter experts stay source MXFP4. Model memory is 97.73 GiB per TP rank. `docker/patch/chat_encoder_1017.py` brings the image's V4.1 chat encoder to 1017 parity. The audit is K-aware (`flags.md`, 1017 retarget).

The public path: clone `https://github.com/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-vLLM-2x-DGX-Spark`, run `hf download sfxnz/DeepSeek-V4.1-Flash-1017-EXL3 --revision 2.7bpw-mcg-hq` and build `e16-rows` on both nodes, then `./run.sh`. `./stop.sh` stops both nodes. `run.sh` resolves Hub `refs/<rev>` to `snapshots/<commit>/`. It also accepts an assembled pack at `snapshots/<rev>` (`tools/build_pack_1017.py assemble`). The Sept 2.0 bpw pack (`sfxnz/DeepSeek-V4.1-Flash-EXL3`, the pre-1017 model) belongs to the Sept recipe, `https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark` @ `3de9146`. `SNAPSHOT_SHA=2.0bpw-*` resolves against this recipe's `MODEL` and fails.

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up. Unload that serve first (`./stop.sh` in its recipe). Stop the pack builder too (`dsv41-1017-build-<node>`, image `dsv41-quant151`; older converters ran as `dsv41-quant`): it holds the GPU while it encodes.
- `NCCL_IB_HCA` is pinned by `run.sh` from `HCA` (default `rocep1s0f1`) and the socket interfaces from `IFACE` (default `enp1s0f1np1`). Change `HCA`/`IFACE`, not `NCCL_IB_HCA` (it is not in FORWARD_ENVS). GB10 exposes four HCAs and two are DOWN. A second rail, `roceP2p1s0f1` / `enP2p1s0f1np1`, is also ACTIVE. The recipe does not use it, and it is unmeasured.
- Default thinking is off. `chat_template_kwargs`: `thinking=false`, `reasoning_effort=low`. Tokenizer/tool/reasoning parsers are `deepseek_v41`.
- DeepJIT is a CUDA/Ascend kernel JIT library, not a serving stack. Official V4.1 kernels in the HF `inference/` tree are TileLang. Do not vendor DeepJIT into this image.
- K comes from the measured allocation only: `build_pack_1017.py --alloc tools/calib/alloc/expert_bits_2.70.json --hessian-mode real`. `--hessian-mode identity` is the uncalibrated fallback. `MAX_NUM_SEQS=4` at `MAX_MODEL_LEN=500000` rests on the perf2 occupancy row in flags.md (CPU planner and KV manager; the perf2 boots log the predicted 1,471,832 tokens): 4 × 500K fits the pool at `MAX_NUM_BATCHED_TOKENS=8192`, 4 × 524,288 needs 4096. Do not raise it past 4 before a new occupancy row.
- DSpark k=3 (`NUM_SPECULATIVE_TOKENS=3`). The k sweep is a Sept 2.0 bpw measurement (Sept R15/R17, Sept model and drafter): L.A.I.L k5 26.32 / k4 25.87 / k3 28.76 / k2 26.89. The 1017 drafter weights changed and no 1017 k sweep is recorded; boot D1 measured acceptance 2.09 (L.A.I.L) at k=3. `run.sh` derives cudagraph capture sizes from k and `MAX_NUM_SEQS` as {1} ∪ {s·k, s·(k+1)}, which gives `[1,3,4,6,8,9,12,16]` (`[1,3,4,6,8]` at 2 sessions). Matched captures measured +5.5% in Sept R16. k must stay 1..5 (the DSpark block is 5). Sept E6 k=10 collapsed to 12.1.
- `--max-num-batched-tokens` defaults to 8192. Sept-pack measurements: E1 measured 2048 at pp@16k −28%; PR 6 measured 8192 at 12,712-token prompts as 755 vs 797 tok/s (−5%) against 2048. `MAX_NUM_BATCHED_TOKENS=2048` is the override for that workload. No 1017 re-check is recorded.
- The serve pin is `SNAPSHOT_SHA=2.7bpw-mcg-hq`. Its quality baseline and goldens are `results/2026-10-18-1017/quality-baseline/`: `full.json` (recorded on boot D1, PASS) and `quick.json` (recorded on release boot r1, PASS). Sept baselines were recorded on a different model and do not gate 1017 boots.
- Lever defaults carried over from Sept rounds 34 and 35 (evidence in the Sept repo @ `3de9146`, `results/RESULTS.md` and `flags.md`). Round 34: `DSV41_ENGRAM_WILLNEED=1`, `DSV41_STREAM_FEED=1`, `DSV41_WOA_PREPACK=1` and `DSV41_DSPARK_SPARSE_MARKOV=1`. To turn a lever off, set it to `0`; an empty value keeps the default. `DSV41_MHC_DECODE_SPLITS` stays 0 (quick quality failed at 40). `DSV41_P2B_SRC_SORT`, `DSV41_DENSE_DG_SMALLM` and dual-rail NCCL stay off (flags.md).
- Round 35: `DSV41_P2B_COOP=2`, `DSV41_DENSE_GEMV=1`, `DSV41_MHC_DET_SPLITS=16`, `DSV41_ENGRAM_NATIVE_STAGE=1`, `DSV41_ENGRAM_EARLY_HASH=1`, `DSV41_ATTN_T2R_DEDUP=1`, `DSV41_SWA_META_FUSED=1`, `DSV41_MOE_PREP_FUSED=1`, `DSV41_CANDIDATE_MASK_BOUNDED=1`, `DSV41_INDEXER_WP_GEMV=1` and `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024`. `DSV41_ENGRAM_EARLY_HASH=1` needs `DSV41_ENGRAM_NATIVE_STAGE=1`, and the native stage refuses to arm under `DSV41_ENGRAM_CPU_HASH=1` or `DSV41_ENGRAM_DEFER=1`. Set a lever to `0` to turn it off (`DSV41_DSPARK_SPARSE_MARKOV_TOPK=256` restores the round-34 top-k). The round-34 configuration is all ten set to `0` plus `TOPK=256`. The KP add-ons (`DSV41_MHC_DET_OVERLAP`, `DSV41_ENGRAM_WKV_TP`, `DSV41_PM_QOS_US`, `DSV41_NCCL_EAGER_TWIN`, `DSV41_AR_L2_PREFETCH`) stay off: their ABAB did not pass step 6 (flags.md).
- The mixed-K pack and `DSV41_P2B_COOP=2` on it need `e15-kbits` or newer; the default `e16-rows` adds the native decode MoE for the 9..24-row batches of three or four sessions (on `e15-kbits` they run the slower `exl3_moe` path). `canonical-e14` and older images have no `expert_bits` support: the plugin ignores it, and the K-aware audit fails `AUDIT=strict` (flags.md, 1017 retarget). Roll a lever back by setting it to `0` on `e16-rows` (`DSV41_P2B_WIDE=0` for the wide decode MoE), not by switching image.
- Never pull, merge or check out in the checkout the serve was started from (here `~/projects/ai-lab/recipes/DeepSeek-V4.1-Flash-1017-2X-DGX-SPARKS-EXL3`) while the serve is up. The head container bind-mounts its `docker/patch` and `sitecustomize.py` read-only, so a pull changes head patch code under a running serve and skews it from the worker's `~/.cache/dsv41-patch` copy. Work in a worktree.
- KV layout: `DSV41_KV_PACKED_GROUPS=1` (`docker/patch/kv_packed_groups.py`, vLLM PR #60996, sitecustomize rewrite of `v1/core/kv_cache_utils.py`) packs the pool into 50 KV groups on 46,080-byte blocks; `0` restores the image's 18. `DSV41_SWA_META_BATCHED=1` builds the 43 SWA groups' decode rows in one launch and needs `DSV41_SWA_META_FUSED=1`. `DSV41_INDEXER_EXPANDED_TABLE_LAZY=1` (`docker/patch/indexer_expanded_lazy.py`) allocates the indexer builders' expanded block table on first use only (only the flattened decode path reads it; a WARNING and an audit line say when it happens). `--mm-processor-cache-gb 1` and `--prefix-cache-retention-interval 4096` are `run.sh` CLI flags (`MM_PROCESSOR_CACHE_GB`, `PREFIX_CACHE_RETENTION_INTERVAL`).
- Prefill MoE: `DSV41_MOE_PREFILL_TC=1` (`docker/patch/moe_prefill_tc.py`) runs routed-MoE calls of >= 9 rows on the tensor-core kernel that `e16-rows` prebuilds (edited sources compile at first use). It is not bit-exact against stock `exl3_moe` (fp32 accumulate vs fp16 split-K partials): it self-checks and verifies its first calls against the stock path (max_rel <= 2e-2, rms_rel <= 5e-3) and disarms on a miss. With it on, exllamav3's 294 MiB `exl3_moe` scratch is allocated only if a stock call needs it. `0` restores the stock path.
- Engram prefill read-ahead: `DSV41_ENGRAM_PREFILL_AHEAD=1` (`docker/patch/engram_prefill_ahead.py`) reads each prefilling request's next chunk of Engram rows on 8 threads while the current chunk computes. The next stage still runs the stock GPU hash and uses a read-ahead row only where its hash ids match (a mismatch disarms); the first 4 calls are checked bit for bit against the stock gather. `DSV41_ENGRAM_PREFILL_AHEAD_DONTNEED=1` (default) drops the pages a prefill read brought in. `0` restores the stock stage.
- Decoder replay: `DSV41_DECODER_REPLAY_ROWS=2670` (`docker/patch/decoder_replay.py`, the CED prefill skip). In eager prefill steps layer 20 writes its KV for every row; its attention, wo and MoE and layers 21-39 then run on each prompt's last W rows only. Decode rows run every layer, graph steps never trim, and prompt-logprob requests keep every row. W >= 2414 (128 + 18 × 127) is exact for the last row, every layer's decode-window KV and the DSpark taps; at 2670 the last 384 prompt rows are exact in every layer, which covers the rows the prefix-end fallback's hits read. `128` (each chunk's last 128 rows, the upstream approximation) is not exact; `0` is off. SWA blocks of skipped rows stay out of the prefix cache. `DSV41_PREFIX_END_FALLBACK=1` and `DSV41_PREFIX_REPLY_RETAIN=1` (`0` = stock e15) let resends, follow-ups that diverge at the prompt end and follow-ups that repeat the reply hit within 192 tokens of the shared end; a prompt that diverges from a trimmed one more than about 128 tokens before its end, away from a shared-prefix junction, recomputes from the start or from a boundary an untrimmed request cached (flags.md).
- Every env var a `docker/patch/*.py` file reads goes in `FORWARD_ENVS` in `run.sh`. That one list feeds `docker run -e` on the head and the worker ssh line. `tests/test_recipe_ops.py` enforces it. `python3 tests/run_sh_harness.py` prints both `docker run` commands with docker and ssh stubbed.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Refuse-guards (`run.sh`)

Exits unless the matching `FORCE_UNSAFE_*=1`. All three default to 0. Campaign boots set one for that boot only and record it. Never commit one as a default.

- `QUANTIZATION` not `exl3` (`FORCE_UNSAFE_QUANT`)
- `DSV41_ENGRAM_DISK` not `1` (`FORCE_UNSAFE_ENGRAM`)
- `--max-model-len` above 1048576 (`FORCE_UNSAFE_CTX`)
- `MAX_NUM_SEQS` above 4 (`FORCE_UNSAFE_CTX`)
- `KV_CACHE_MEMORY` above 8 GiB (`FORCE_UNSAFE_CTX`; the 1017 default is 3.875 GiB, `4160749568`)
- `NUM_SPECULATIVE_TOKENS` not an integer 1..5 when `SPEC=dspark` (`FORCE_UNSAFE_CTX`)

Native MXFP4 experts are ~130 GiB per TP=2 rank before Engram. That does not fit 121 GiB UMA. Engram pinned in host RAM is the same pool.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
```

Torch-gated tests skip on the host. Run them CPU-only in the image, from `tests/` with plain module names: the image's dist-packages has its own `tests` package, which shadows `tests.X`.

```bash
docker run --rm --network none --memory 8g --cpus 4 -v "$PWD":/w -w /w/tests --entrypoint python3 \
  dsv41-flash-exl3-sm121:e16-rows -m unittest test_lmhead_mxfp8 test_engram_gather_v2 test_fix_o_proj_woa_fp8 -q
```

The pack builder's torch tests and the chat-encoder image check (no GPU):

```bash
docker run --rm --network none --memory 8g --user 1000:1000 -e HOME=/tmp -v "$PWD":/repo:ro -w /repo \
  --entrypoint python3 dsv41-quant151 -m unittest discover -s tests -p test_build_pack_1017.py -q
DSV41_IMAGE_TESTS=1 DSV41_IMAGE=dsv41-flash-exl3-sm121:e16-rows python3 -m unittest tests.test_chat_encoder_1017 -q
```

After `./run.sh` is up:

```bash
python3 smoke_chat.py
python3 smoke_vision.py
```

`smoke_chat.py` fails unless `choices[0].message.content` matches `323` for `17*19` (`--expect ''` accepts any non-empty content). `smoke_vision.py` must not return HTTP 400 `is not a multimodal model`, and its answer for the solid-red PNG must contain `red`.

## Never touch

- Live HF tokens
- Floating `:latest` on the dsv41 base. Digest is pinned in `recipe.yaml` and the Dockerfile
- Hand-edited generated README / `run.sh` blocks
