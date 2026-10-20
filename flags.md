# flags.md — every flag, env var, and patch in this recipe

What / why / measured effect for everything `run.sh` sets and everything the
image patches at import time. Sources: `recipe.yaml` (source of truth),
`run.sh`, `docker/patch/sitecustomize.py`, `results/`.

**Pack history.** Rows and rounds from before the 1017 retarget measured the
Sept 2.0 bpw pack (`sfxnz/DeepSeek-V4.1-Flash-EXL3`) on the Sept model. Their
evidence (`evidence/`, `results/RESULTS.md`, `results/2026-09-*`) lives in the
Sept recipe repo (https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ 3de9146), not in this one. A number is a 1017
measurement only where the row says 1017; 1017 evidence is under
`results/2026-10-18-1017/` ("1017 retarget" below, and the "1017 release" round
at the end of the round log).

Effect column key: **measured** = number recorded in the cited evidence;
**required** = the serve fails or regresses without it; **guard** =
refuse-guard default, not a tuning knob.

## vLLM engine flags (run.sh)

| Flag / value | What | Why | Effect |
|---|---|---|---|
| `--tensor-parallel-size 2` `--nnodes 2` | One GB10 rank per Spark | The 1017 2.70 bpw pack loads 97.73 GiB per rank (1017 boot D1); one UMA cannot hold it | **required** |
| `--distributed-executor-backend mp` | Multiproc within node | vLLM TP across 2 nodes via master-addr | **required** |
| `--max-model-len 500000` | Window per session (4-session default, perf2) | The 3.875 GiB pool holds four 500K sessions without preemption at `MAX_NUM_BATCHED_TOKENS=8192` with packed KV groups; four 524,288-token sessions need 3.99 GiB there, so the window is 500,000, not 2^19 (`MAX_MODEL_LEN=524288` fits four sessions at `MAX_NUM_BATCHED_TOKENS=4096`; `--max-num-seqs` row). 500,000 is not a multiple of 64: the e15 paths read handle it (cdiv block counts, block tables padded to 128 tokens, compressed lengths 250,000 / 500,000); DeepGEMM's paged-MQA logits width is not checked, so the first boot and needle are the check (500,096 = 3,907 x 128 is the fallback). V4.1 Flash ships CSA2 1M context: `MAX_MODEL_LEN=1048576` still boots (packed: 1.73x of one 1M request, CPU planner; D1 ran it with the image's grouping) | guard at 1M (`FORCE_UNSAFE_CTX`) |
| `--max-num-seqs 4` | Running requests | perf2 occupancy row (kv-pack, 2026-10-19, CPU: e15's planner and KVCacheManager with the recipe page pins; the boots log the same token count): with `DSV41_KV_PACKED_GROUPS=1` four sessions growing to 500,000 tokens at once fit without preemption in 3.82 GiB (`MAX_NUM_BATCHED_TOKENS=8192`) / 3.59 GiB (4096) of the 3.875 GiB pool; four full 524,288-token sessions need 3.99 / 3.75 GiB. With the image's 18 groups the pool holds 4 x ~282K (8192 at 524,288). Sept pack: occupancy measured for 2 | guard >4 (`FORCE_UNSAFE_CTX`); GPU: boots I1-I4 (`results/2026-10-19-1017-perf2/`) serve 4 sessions at 500,000 with `GPU KV cache size: 1,471,832 tokens` (2.94x, the CPU prediction) under strict audit; MemAvailable minima per phase in each boot's `numbers.txt` (lowest: spark1 1,626 MiB during I4's prefill micro bench); the 4-session resident soak ran on the full stack's defaults (FINAL: 4 x ~447K tokens resident, 0 preemptions, KV usage peak 0.876, `results/2026-10-19-1017-perf2/FINAL/soak.txt`) |
| `--max-num-batched-tokens 8192` | Chunked-prefill chunk | **measured local optimum** (2026-09-18): 2048 → pp@16k −28%, 16384 → pp@64k −12% and needs +1 GiB KV to even boot; see `results/RESULTS.md` E1/E2/E2b |
| `--kv-cache-dtype fp8` | KV cache storage | KV bytes/token halved vs bf16; CSA2 + indexer pages | measured vs auto in early packs |
| `--kv-cache-memory 4160749568` (1017 default) | 3.875 GiB KV pool | The 2.70 bpw 1017 pack loads 97.73 GiB/rank (Sept 2.0 pack: 75.58), so 8 GiB no longer leaves a MemAvailable floor; 4.5 GiB would leave spark1 ~3.7 GiB MemAvailable after the smokes, below the 4 GiB floor target | **measured (1017 boot D1)**: "GPU KV cache size: 1,108,848 tokens", 1.06x of one 1,048,576-token request. At the 524,288 window: 975,641 tokens, 1.86x with the image's 18 groups (logged, perf2 baseline boot B0). At the 500,000 default: 1,471,832, 2.94x with packed groups (CPU planner, not yet logged; image's groups 964,308, 1.93x). MemAvailable (spark1 / spark2): min during boot 4.6 / 5.4 GiB, after smoke 4.3 / 5.9 GiB; normal serving incl. the 128k needle stayed >= 3.8 GiB; min under all D1 tests 3.17 GiB (spark1, ~10 s, during quality_eval's nll component: prompt_logprobs full-vocab logits on the head); the memory watchdog never tripped; `results/2026-10-18-1017/boot-d1/` |
| `--kv-cache-memory 8589934592` (Sept 2.0 pack, history) | 8 GiB KV pool (Sept default) | **measured keep on the Sept pack (E5, 2026-09-19)**: pool grows to 2,289,205 tokens = 2.18× concurrency at 1M ctx (4 GiB held exactly 2×1M with zero prefix-cache slack); pp/tg/correctness/e2e all unchanged; 16 GiB still free for Engram staging | guard at 8 GiB (`FORCE_UNSAFE_CTX`) |
| `--gpu-memory-utilization 0.75` | UMA fraction | Left ~25 GiB for Engram staging + OS on the Sept pack (75.58 GiB/rank); on 1017 (97.73 GiB/rank) the headroom is the MemAvailable floors in the KV row | **required** (0.9 OOMs boot) |
| `--block-size 64` | KV pages | FlashInfer SM120 DSV4 decode kernel is compiled for page 64 | **required** (32 breaks decode kernel) |
| `--quantization exl3` | Loader | Native MXFP4 experts ≈130 GiB/rank do not fit 121 GiB UMA | guard (`FORCE_UNSAFE_QUANT`) |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"greedy"}'` | DSpark MTP drafting | Checkpoint ships 3 nextn layers + Markov head; k=3 is the Sept-pack R15/R17 sweep optimum (k5 26.32 / k4 25.87 / k3 28.76 / k2 26.89 L.A.I.L); k must stay 1..5 (block size 5) | **measured (Sept)**: 12.75→21.2 tok/s (1.67×) at k=5, `evidence/h1-dspark`; k=3 R15/R17. 1017 boot D1 at k=3: acceptance 2.58 prose / 3.96 structured (bench_decode c=1), 2.09 L.A.I.L; no 1017 k sweep is recorded |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,3,4,6,8,9,12,16],"custom_ops":["all"]}'` | CUDA graphs incl. draft; run.sh derives the sizes as {1} ∪ {s·k, s·(k+1)} for s ≤ `MAX_NUM_SEQS` when `COMPILATION_CONFIG` is unset (`[1,3,4,6,8]` at 2 sessions) | Eager decode wastes ~4 µs/launch × ~11 kernels/step; matched sizes avoid padding the verify batch | **measured**: see `evidence/c1-graphs234`, `evidence/vision-on-lail`; matched captures +5.5% (R16, 27.27→28.76) |
| `--mm-processor-cache-gb 1` (`MM_PROCESSOR_CACHE_GB`, vision on only) | MM processor cache cap | vLLM's default 4 GiB is mirrored in the API server and EngineCore, both on the head: up to +8 GiB with many distinct images (perf2 memory report) | on since perf2 kv-pack: boots I1, I3, I4 pass the quick-quality vision check with it (`results/2026-10-19-1017-perf2/I4/quick.json`); the head-growth cap itself is not measured (no many-image test) |
| `--prefix-cache-retention-interval 4096` (`PREFIX_CACHE_RETENTION_INTERVAL`) | Sliding-window prefix-cache checkpoints | e15 defaults to 0: SWA blocks keep their hash only at the replay boundary (`num_prompt - 1`) and shared-prefix junctions (`single_type_kv_cache_manager.py` `reachable_block_mask`); 4096 also keeps a checkpoint tail per 4096 tokens, so a follow-up that diverges inside the old prompt recomputes at most ~4096 tokens. Retained blocks are freed blocks that keep their hash (evictable), not extra pool. Must be a multiple of the scheduler block (128). External report: Mia kit 4096 fixed zero-hit prefix caching for hybrid SWA + DSpark (identical resend 63 s -> 0.46 s), not reproduced here | **measured (perf2 boots)**: the repeat of a request that shares a 16K prefix with an earlier one (`f_identical_to_e`) hits 28,672 tokens on I2 (4096, decoder replay off: the last 4096 checkpoint) against 16,384 on B0 (interval 0), and 32,128 on I4 (4096 with decoder replay, prefix-end fallback and reply retention) (`results/2026-10-19-1017-perf2/I2/numbers.txt`, `results/2026-10-19-1017-perf2/I4/prefix_cache.json`) |
| `--kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}'` | Disable SM120 sparse-MLA autotune | Autotune feeds DeepGEMM paged-MQA which asserts `block_kv∈{32,64}` and crashes warmup | **required** for boot with graphs |
| `--tokenizer-mode/--tool-call-parser/--reasoning-parser deepseek_v41` | Native parsers | V4.1 chat template + tool format | **required** for tool-call eval |
| `--default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}'` | Thinking off by default | Recipe ships fast-answer default; caller can override per request | quality tradeoff — see `results/RESULTS.md` baseline notes |
| `--language-model-only` NOT set; `--mm-encoder-tp-mode data` | Vision on | V4.1 Flash is multimodal; data-parallel encoder | measured `evidence/vision-on-lail` (14.66 vs 15.1 LMO ≈ noise, kept vision) |
| `--trust-remote-code`, `--served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017` | Identity | Clients send the official 1017 id | — |

## Container / fabric env (run.sh)

| Env | Value | Why | Effect |
|---|---|---|---|
| `NCCL_IB_HCA=rocep1s0f1` (+`NCCL_NET=IB`, `NCCL_CROSS_NIC=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`) | Pin one HCA | GB10 exposes 4 HCAs, 2 are DOWN; default selection hangs TP | **required** |
| `NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME=enp1s0f1np1` | Pin TCP iface | Avoids routing over the management NIC | **required** |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Allocator | Fragmentation with the 405 GB pack + graphs | avoids late-boot OOM |
| `TORCH_CUDA_ARCH_LIST=12.1a` `FLASHINFER_CUDA_ARCH_LIST=12.1a` | JIT arch | GB10 is SM121; wheels default misses some kernels | **required** for exl3 CUDA JIT |
| `VLLM_PLUGINS=vllm_exl3` + sitecustomize `load_general_plugins()` | Plugin load in every proc | EngineCore resolves `--quantization exl3` before plugins load | **required** |
| `VLLM_EXL3_MOE_KERNEL=native` | Fused p2b MoE | Native 5120×1152 CFG=1 tile vs generic `exl3_moe` | **measured**: ~23 vs 15 tok/s, `evidence/p2b-cfg1` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | Breakable graphs | Graph capture survives spec-decode shapes on SM121 | **required** for graphs+DSpark |
| `DSV41_ALLOW_CUDA_GRAPHS=1` | Un-stub warmup | Re-enables `compile_or_warm_up_model` after DeepGEMM warmup stub | **measured**: graphs cell in RESULTS |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | Boot patience | 405 GB pack load + graph capture (1017 boot D1: ~11 min, load 508 s head / 305 s worker) | — |
| `DSV41_ENGRAM_DISK=1` | Engram on NVMe | 196B Engram pinned in UMA OOMs a Spark | guard (`FORCE_UNSAFE_ENGRAM`) |
| `LANGUAGE_MODEL_ONLY=0` | Vision weights loaded | See flag row above | — |
| `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=256` | Adaptive verify profile | Spec verification profile context | default kept |
| `HF_XET_HIGH_PERFORMANCE=1` (download path) | Xet on | Engram shards ~95 GiB need xet | — |
| `NCCL_BUFFSIZE=1048576` `NCCL_LL128_BUFFSIZE=262144` `NCCL_PROTO=^LL128` `NCCL_MAX_NCHANNELS=8` | NCCL AR-tail set | Smaller pinned buffers; LL128 off (isolated AR: LL 43.0 µs vs LL128 81.8 µs, E10) | **KEEP (R12)**: prose flat, prefill32k +2.1%, L.A.I.L above band, MemAvail +3.5/+2.8 GiB; `results/2026-09-20-nccl/` |
| `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=4` (1017 default) | Release cached allocator blocks after a >= 8192-token prefill step while MemAvailable is below 4 GiB (Sept: 2.5) | At 97.73 GiB/rank the post-smoke MemAvailable was 4.3 / 5.9 GiB on boot D1 (spark1 / spark2), so 2.5 would let long prefills eat the last headroom before releasing; 4 matches the 4 GiB floor target. Costs prefill throughput when it fires (~20% at 50k on mia's box, every chunk) | **1017 boot D1** (with it on): pp_novel 4k / 16k / 64k 795 / 755 / 733 tok/s; normal serving incl. the 128k needle stayed >= 3.8 GiB MemAvailable. No pp_novel A/B against 2.5 is recorded. Long-context trace: 256k needle 3/3 and 1M needle 3/3, spark1 dips to 2.1–2.8 GiB at prefill boundaries and 1.8 GiB during 1M tokenization (each 2 s or less), watchdog never tripped; `results/2026-10-18-1017/boot-d1/` |
| `DSV41_DROP_PAGE_CACHE=1` `DSV41_INDEXER_PREFILL_FACTOR=1` `DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192` `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5` `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` | Mem-hygiene bundle (Sept values) | Room for Engram staging and long prefills | **KEEP (2026-09-20)**: perf flat, MemAvail post-32k +5.82/+6.33 GiB; `results/2026-09-20-memhygiene/VERDICT.md` |
| `DSV41_ENGRAM_PREFETCH=1` + `DSV41_ENGRAM_CENSUS=1` | Engram prefetch v3 (fadvise next-step rows) + census | Hides NVMe reads behind the step | **KEEP (R21)**: L.A.I.L 28.76→30.10 (+4.6%); pf_hit 100% both ranks measured at R21 only (unobservable R23-2026-09-24: gv2 returned before the census body; a628e7b restores it in `[gv2-census]`). Self-disabled at runtime on both ranks on 2026-09-24 (c=2 IndexError in `_pf_next_chunk`), so the s0 baseline is prefetch-off: results/RESULTS.md "2026-09-24 review s0". Fix 2ebb78d held on GPU in R34: after FULL/DECODE + C2-STRESS (12 c=2 requests) in s3, s5, s6, s7 and s8, prefetch v3 stayed armed on both ranks with 0 IndexError lines; `results/2026-09-21-pfrearm/VERDICT.md`, `results/2026-09-24-review/campaign/` |
| `DSV41_ENGRAM_GATHER_V2=1` | Engram gather v2 (preadv run batching, bit-exact dequant) | Removes the stage pool hop per gather | **KEEP (R23)**: L.A.I.L 30.10→31.37; `results/2026-09-21-gatherv2/` |
| `DSV41_LMHEAD_MXFP8=1` + an lm_head-MXFP8 pack | lm_head on the b12x MXFP8 path. The 1017 `2.7bpw-mcg-hq` is one: `tools/build_pack_1017.py` converts source shard 43 (relerr 0.027, `results/2026-10-18-1017/encode/`); the Sept one was `2.0bpw-mcg-lmhead-mxfp8` | Halves the vocab-head weight stream | **KEEP (R33, Sept pack)**: L.A.I.L 31.37→33.23 (+5.9%); self-disarms on stock `2.0bpw-mcg`; `results/2026-09-22-close/VERDICT.md`. R34 quality A/B vs stock bf16 head (s12): ΔNLL (stock − mxfp8) −0.00045, GSM8K 94/100 both, thinking-on 38/40 both, MMLU 197/228 both, cross-boot flip hazard 0.01973 vs 0.01855 mxfp8-vs-mxfp8 control; stock head costs −6.2% L.A.I.L (32.38 vs 34.51); `results/2026-09-24-review/campaign/s12-lmhead-quality-ab/` |
| `DSV41_ENGRAM_WILLNEED=1` (+`DSV41_ENGRAM_WILLNEED_MIN_ROWS=512`) | gv2 prefill read-ahead: one `posix_fadvise(WILLNEED)` per merged 4 KiB page span of a gather's unique rows before the serial preadv loop, for calls with >= 512 rows (decode is 48-96) | The serial preadv loop is queue depth 1 on a cold prefill | **KEEP (R34 s4)**: micro novel 8k 188.8 → 798.3 tok/s (4.23x), novel 32k 234.7 → 805.7 (3.43x), warm 8k 747.2 vs 761.9 (−1.9%), L.A.I.L 33.81 (5 runs) vs 34.11 (10 runs), decode census fadv/call 0.0; prefill Engram read ms per 1k rows −83% (TP0) / −87% (TP1); `results/2026-09-24-review/campaign/s4-E1-willneed/` |
| `DSV41_STREAM_FEED=1` | Drain the VL wrapper's sorted weight list while it loads (`g8_stream_feed.py` on stock packs) | Shorter load | **KEEP (R34 s4, hb-2 gate ≥5% on both ranks)**: 'Loading weights took' TP0 334.25 → 313.24 s (−6.3%), TP1 190.68 → 151.42 s (−20.6%) vs s3; one boot per arm; min MemAvail during load not compared (s3 has no trace); `results/2026-09-24-review/campaign/s4-E1-willneed/` |
| `DSV41_WOA_PREPACK=1` | Pack the fp32 wo_a scale to UE8M0 once per layer (bitwise self-test, else the fp32 scale stays). Needs `canonical-e13` (`fix_o_proj_woa_fp8.py` stage 2); on `canonical-e12` it logs `the lever is OFF` | `fp8_einsum` repacked the scale on every call (~43 per step) | **KEEP (R34)**: bitwise equal at M=3/4/8; 31.75 → 25.0 us/call at M=3 (s1); 43/43 layers engaged per rank, 0 REJECTED (s5, s7); trace −0.71 / −0.81 ms/step r0/r1 (s5, vs trace3). ABAB (bundled with MHC split-K 40): prose c=1 63.795 vs 64.395 ms/step (−0.60, B spread 0.27), tok/s flat (39.485 vs 39.61); `results/2026-09-24-review/campaign/s1-microbench/`, `results/2026-09-24-review/campaign/s5-C1-decode-bundle/`, `results/2026-09-24-review/campaign/s7-C2-decode-bundle/abab.json` |
| `DSV41_DSPARK_SPARSE_MARKOV=1` (+`_TOPK`: in-code default 256, run.sh default 1024 since R35) | Gathered top-k Markov bias via the speculator's `_sample_sequential_topk` (drops the 3 x 66 MB Markov GEMM; argmax stays). It refuses `DSV41_DSPARK_MARKOV_SCALE` != 1, `CONF_GATE`, `DRAFT_TOPK` (lever OFF, strict audit fails): set it to 0 to test those | Greedy verify keeps output exact; only acceptance can move | **KEEP (R34 s8, one boot, Sept pack; the 1017 drafter changed)**: ms/step vs the B mean: prose c=1 −0.825 (B spread 0.27), structured −0.45 (0.04), L.A.I.L fresh −0.245 (0.07), post-stress −1.35 (0.30). L.A.I.L 34.51 at acceptance 2.172 (B 2.160 / 2.183); prose pooled acceptance 2.504 is below B (2.566 / 2.621). Quick quality PASS. Confounded with WOA_PREPACK (no WOA-only boot); no trace. s13, the second S boot: L.A.I.L acceptance 2.071, the lowest of the campaign. Pooled over 4 L.A.I.L batches, S vs B is acceptance 2.129 vs 2.159 (−1.4%), 62.67 vs 63.50 ms/step, tok/s 33.72 vs 33.95 (−0.7%). The acceptance cost is open: next is an S-vs-WOA-only ABAB or the TOPK=1024 retry; `results/2026-09-24-review/campaign/s8-S-sparse-markov/compare.json`, results/RESULTS.md round 34 R35 P6 (W = `SPARSE_MARKOV=0` vs S, W-1, S-1, W-2, S-2 on the K bundle): W does not beat S beyond the noise on either gate cell (L.A.I.L tok/s +0.431 vs noise 0.81; prose_long c=1 +0.189 vs noise 1.403), so S stays. S relative to W: L.A.I.L acceptance −2.74% (beyond noise), L.A.I.L ms/step −0.735 (noise 0.07), structured c=1 −0.576 ms/step (noise 0.284) and +1.003 tok/s (noise 0.5); `results/2026-09-26-serve-r3/decision-markov.json` |
| `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024` | Top-k width of the gathered Markov bias (in-code default 256; 1..4096) | A draft winner outside the top k gets no bias; a wider k recovers part of SPARSE_MARKOV's acceptance cost | **KEEP (R35 P6, Sept pack)**: ABAB S#3, S1024#1, S#4, S1024#2 (S-3, S1024-1, S-4, S1024-2) on the K bundle. L.A.I.L tok/s S 42.36/42.871 (median 42.6155) vs S1024 43.167/43.615, margins +0.552/+1.000 vs noise 0.511; prose_long c=1 S 42.788/42.195 (median 42.4915) vs S1024 44.174/43.728, margins +1.683/+1.237 vs noise 0.593: every S1024 boot beats the S median by more than the noise on both gate cells. The gain is acceptance (prose_long c=1 median-run +0.1062, noise 0.0214; L.A.I.L +0.0528, noise 0.0525); the gather costs +0.473 ms/step on L.A.I.L (noise 0.219), the other c=1 ms/step cells are inside the noise. Non-inferiority holds against the larger spread (closest: pp_warm 8k −3.75 vs 3.8). Quick quality PASS on all four boots. Caveat: S-3/S-4 are the two lowest of the 10 S-config boots on prose_long c=1; against S-1..S-4 (same env, not interleaved) and against all 10 S-config boots neither gate cell meets the rule. `results/2026-09-26-serve-r3/decision-markov.json` |
| `DSV41_P2B_COOP=2` `DSV41_DENSE_GEMV=1` `DSV41_MHC_DET_SPLITS=16` `DSV41_ENGRAM_NATIVE_STAGE=1` `DSV41_ENGRAM_EARLY_HASH=1` `DSV41_ATTN_T2R_DEDUP=1` `DSV41_SWA_META_FUSED=1` `DSV41_MOE_PREP_FUSED=1` `DSV41_CANDIDATE_MASK_BOUNDED=1` `DSV41_INDEXER_WP_GEMV=1` | k3 round-3 kernel bundle K (rows below: coop dataflow MoE, dense GEMV, det mHC, native Engram stage + early hash, five fusion-host levers). `DSV41_P2B_COOP=2` needs `canonical-e14` on the Sept K=2 pack and `e15-kbits` on the 1017 mixed-K pack (on `canonical-e14` the K-aware strict audit fails) | Bit-exact against the stock path, or within 1 fp16 ulp for COOP (fp32 slot-sum order); none changes drafting | **KEEP (R35 P2, one unit, Sept K=2 pack)**: A, K, KC, A, K, KC on review-e14. K vs A: four prose c=1 63.866/63.083 → 47.156/48.245 ms/step (−15.774, noise 1.089), prose_long c=1 63.47/63.216 → 49.01/48.702 (−14.487, noise 0.308), acceptance matched; tok/s medians L.A.I.L 33.95 → 43.049 (+26.8%), prose_long c=1 33.001 → 43.806 (+32.7%), c=2 aggregates +43.8% to +56.6%; every K boot beats the A median on every row. K vs KC (COOP) −8.483 / −8.175 ms/step; KC vs A −7.291 / −6.312. Non-inferiority holds against the larger spread. Quality quick PASS K-1, K-2, KC-1, KC-2; full PASS K-2 (GSM8K 94/100, thinking 39/40, MMLU 199/228, needle 9/9). Every measured saving exceeds the s11 projection (K vs A c=1 11.9-13.1 projected, 14.172-16.657 measured). `results/2026-09-26-serve-r3/decision-bundle.json`, per-lever: `P3/attribution.txt` |
| `SNAPSHOT_SHA=2.0bpw-mcg-viterbi-lmhead-mxfp8` (Sept pack, history; the 1017 default is `SNAPSHOT_SHA=2.7bpw-mcg-hq` on `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3`) | Sept default pack R36 (on `sfxnz/DeepSeek-V4.1-Flash-EXL3`, for the Sept model): the routed experts of `2.0bpw-mcg-lmhead-mxfp8` re-encoded from the MXFP4 source (exllamav3 tail-biting Viterbi + `refit_scales`, same 2.0 bpw MCG K=2 format and bytes, relerr 0.2616 vs 0.3775), lm_head MXFP8 shard unchanged | Weight-space error 0.69x of stock per tensor; decode kernels and bytes per step unchanged, so speed moves only through DSpark acceptance | **ADOPTED (R36, protocol owner's decision)** over the R35 P4 REJECT as written (golden hazard 0.094-0.102, a pack change fails it by construction; L.A.I.L −1.024 tok/s beyond noise, prose_long c=1 +3.108 beyond noise). Quality, paired vs the old pack: NLL −42% (P4 −0.10489, R36 −0.10477 nats/token, 39/40 passages), GSM8K 97/97 vs 95/94, MMLU 206/204 vs 196/198, tools exact 88/88 vs 84/88. R36 V-1/V-2 vs R35 final config (not interleaved): L.A.I.L 43.17 → 42.58 (noise 0.88), prose_long c=1 43.73 → 43.80, four prose c=1 52.52 → 50.51 (acceptance 2.513 → 2.416), structured c=2 agg 138.34 → 156.29, c=1 ms/step and prefill within noise; V-2 quality quick + full PASS against the new baseline (`results/2026-09-27-viterbi-adopt/quality-baseline/`). `results/2026-09-27-viterbi-adopt/` (Sept repo) |

## 1017 retarget (2026-10-17; release pins 2026-10-18)

| Item | Value | Notes |
|---|---|---|
| Source | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170` | `tools/pack_meta.py` SOURCE_*; every shard changed vs Sept (vision, embed, drafter and Engram included) |
| Pack | `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` @ `2.7bpw-mcg-hq` (`SNAPSHOT_SHA=2.7bpw-mcg-hq`) | `tools/build_pack_1017.py`: MCG, K in {2,3,4} per (layer, w1+w3 / w2) from `quantization_config.expert_bits`; routed mean exactly 2.70 bpw, allocation `tools/calib/alloc/expert_bits_2.70.json` (measured sensitivity, `results/2026-10-18-1017/allocation/`); Viterbi + scale refit with real activation Hessians (`dsv41-hessian-v1`), 41/41 per-unit gates passed (`results/2026-10-18-1017/encode/`). Dense/attention/shared/indexer source MXFP8, lm_head MXFP8, embed + vision BF16, DSpark/MTP drafter experts source MXFP4, Engram source FP8 on NVMe. 405.1 GB (377.2 GiB) on disk per node; 97.73 GiB per TP rank loaded |
| Served name | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | client `--model` defaults follow |
| Image | `dsv41-flash-exl3-sm121:e16-rows` since perf2 c4-kernels (the release boots ran `e15-kbits`, image ID sha256:7e2cec199bed) | K-bits p2b kernels (`widen_p2b_kbits.py`: coop/dataflow at K 2/3/4 and K_gu≠K_d) plus the 9..24-row dataflow kernel (`widen_p2b_rows.py`) and the prebuilt tensor-core prefill MoE extension (`moe_prefill_tc.py --prebuild`). Built by `docker/Dockerfile` (from scratch) or `docker/Dockerfile.e16` (layered on `e15-kbits`, which `docker/Dockerfile.e15` layers on `canonical-e14`). `DSV41_P2B_COOP=2` on this pack needs `e15-kbits` or newer: on `canonical-e14` the K-aware strict audit fails |
| `chat_encoder_1017` (sitecustomize, required) | 1017 `encoding.py` parity for the image's V4.1 tokenizer: tool-result-only user turns do not start a turn, `content_blocks` passthrough, placeholder check before merging, reasoning-effort budgets low 50 / high 75 / max 100 (the image had 25/50/xhigh 75/100; `xhigh` is now rejected like the reference) | `docker/patch/chat_encoder_1017.py`; 5/5 1017 golden vectors and 1017 behaviours exact in the e14 image (`tests/chat_encoder_1017_image_check.py`, 26/26; unpatched control 12/26) |
| K-aware audit | `tools/engagement_audit.py --pack-config` (run.sh passes the snapshot's config.json), `tools/disarm_scan.sh` | With `DSV41_P2B_COOP=2`, every routed (K_gu, K_d) of the pack must be declared by the patch set (`DATAFLOW_K`, `DATAFLOW_MIXED_K`, optional per-K `LOG_ENGAGED_K`); otherwise strict fails even without a log line. A dataflow "lever is OFF" line stays a disarm for any K |
| Memory | `KV_CACHE_MEMORY=4160749568` (3.875 GiB), `MAX_NUM_SEQS=2` at the release (4 since perf2 kv-pack, with `MAX_MODEL_LEN=500000` and packed KV groups), `MAX_NUM_BATCHED_TOKENS=8192`, `DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=4` | 1017 boot D1: 97.73 GiB/rank model memory, KV pool 1,108,848 tokens (1.06x of one 1M request), MemAvailable floors in the KV row; `results/2026-10-18-1017/boot-d1/` |

## Refuse-guards (run.sh, all overridable)

`QUANTIZATION!=exl3` (`FORCE_UNSAFE_QUANT`), `DSV41_ENGRAM_DISK!=1`
(`FORCE_UNSAFE_ENGRAM`), `MAX_MODEL_LEN>1048576` / `MAX_NUM_SEQS>4` /
`KV_CACHE_MEMORY>8 GiB` / DSpark tokens not an integer 1..5 (`FORCE_UNSAFE_CTX`;
Sept-pack E6: k=10 collapsed to 12.1). These encode measured occupancy/memory limits, not
style rules. All `FORCE_UNSAFE_*` default to 0.

## Image patches — always on (docker/patch/sitecustomize.py)

| Patch | What | Why | Effect |
|---|---|---|---|
| `apply_engram_disk.py` + `engram_disk.py` | DiskEngramTable | Engram rows stay on NVMe, staged per step | **required** (memory) |
| `apply_engram_prestage.py` | Prestage rows in `prepare_inputs` | Hides NVMe pread behind compute; un-escapes lookback markers | **measured** decode cell |
| `prefer_b12x_mxfp8.py` | `mm_mxfp8 backend=auto` | SM120 b12x small-M tiles beat cutlass on Q/O GEMMs | **measured**: 23.44 vs ~23 (L.A.I.L wave 22.37), `evidence/b12x-mxfp8` |
| `widen_p2b_shapes.py` + `widen_p2b_cfg1.py` (in image) | Native p2b 5120×1152 CFG=1 | Stock kernel hardcodes hidden=4096 | **measured**: `evidence/p2b-cfg1` |
| `widen_p2b_kbits.py` (in `e15-kbits`) | p2b / coop / dataflow MoE kernels at K 2/3/4 and K_gu≠K_d; the plugin reads `quantization_config.expert_bits` | The 1017 pack stores routed experts at mixed K; the round-3 kernels were K=2 only | **required** for the 1017 pack; boot logs one `routed experts K_gu/K_d` line per layer per rank (1017 boot D1: 40 per rank, matching the allocation) |
| `widen_p2b_rows.py` (in `e16-rows`) | Native p2b decode MoE for 9..24 rows: `p2b_coop_dfw_kernel`, the dataflow kernel's own text with a 144-pair sort cap (24 rows x top-6), launched under `DSV41_P2B_COOP=2` when m x top_k is in (64, 144]; the plugin sends those rows native unless `DSV41_P2B_WIDE=0`. The 42 e15 p2b kernels keep byte-identical machine code. With it the JIT levers take more rows behind their own self-tests: indexer weights_proj GEMV M 9..12, mHC det T 17..24 | Four sessions at DSpark k=3 verify 16 rows; the 8-row native contract sent them to exllamav3 `exl3_moe` (B0 c=4 distinct profile: routed MoE 98.4 of 149.4 ms/step) | **measured (perf2 c4-kernels)**: bitwise per row against the m <= 8 kernel at m 1..24 on spark2 (graph replays, 500-call stress per m > 8: 0 mismatches); routed MoE per step at m=16 112.7 -> 59.0 ms in graph replays. Serve: c=4 distinct prose 91.1 tok/s aggregate at 101.2 ms/step on I1 and 89.0 at 100.8 on I4, against 62.3 at 154.5 on I3 (`DSV41_P2B_WIDE=0`) and 61.4 at 158.0 on B0 (`results/2026-10-19-1017-perf2/I1`, `I3`, `I4`, `B0`). At 9..24 rows the output moves from exl3_moe numerics to the dataflow kernel's: a quality check at c=4 is pending |
| `sm120_page.py` (page-64 coercion, indexer block sizes, extra-page 128, vision clamp, native indexer decode) | SM120 page-size fixes | FlashInfer DSV4 decode compiled for page 64; upstream reports 128 | **required** |
| `sm120_page.patch_persistent_topk` | Exclude family 120 from cooperative topk | persistent_topk oversubscribes at 2 rows TopK=512 | **required** (boot) |
| `kv_packed_groups.py` (`DSV41_KV_PACKED_GROUPS=1`, required when on) | vLLM PR #60996's `v1/core/kv_cache_utils.py` hunks: packed groups sized by the pooled bytes a max-length request draws, not `_approximate_gcd` layer padding; plus one EngineCore log line (`dsv41-kv-packed-groups: N KV cache groups`) | The image's grouping makes every block 138,240 B and the ratio-1 group pays a full block: 3,240 pool bytes per context token for 1,800 of content | CPU (perf2 upstream and memory runs of the same hunk on e15's planner): 50 groups, 46,080 B blocks; 1,471,832 tokens at 500,000 (2.94x), 1,497,155 at 524,288 (2.86x); per-builder buffers +28 SWA and +2 ratio-2 indexer builders (~+0.46 GiB/rank at 500,000 / 8192, ~+0.23 at 4096, estimate; the lazy indexer table row below takes back more than that); GPU: boots I1-I4 log `dsv41-kv-packed-groups: 50 KV cache groups` and `GPU KV cache size: 1,471,832 tokens, Maximum concurrency for 500,000 tokens per request: 2.94x` (`results/2026-10-19-1017-perf2/I3/boot-grep-boot.txt`) |
| `indexer_expanded_lazy.py` (`DSV41_INDEXER_EXPANDED_TABLE_LAZY=1`, required when on) | DeepseekV32IndexerMetadataBuilder stops allocating `expanded_block_table_buffer` (B x blocks of one max-length request x 4 B) in `__init__`; a property of that name allocates the stock zeros buffer on first read and logs one WARNING per builder (`dsv41: indexer expanded block table allocated on first use`, also an audit problem line) | Its only readers are the flattened branch of `_prepare_decode_tensors` (`if not use_native:`); on SM120 with DeepGEMM `use_native` is always true (`native_next_n_supported` is True off SM90, varlen is SM100-only, decodes are <= next_n by the builder's own split). No subclass, no other reader in e15, the EXL3 plugin or docker/patch (pinned and checked in `tests/test_indexer_expanded_lazy.py`) | estimate (`perf2/kv-pack/builder_mem.txt`), per rank at 500,000: -366 MiB with the image's groups, -611 MiB packed at 8192 (-183 / -305 at 4096); GPU: engaged on boots I1-I4 with no `allocated on first use` line (strict audit ok, `results/2026-10-19-1017-perf2/I4/boot-grep-final.txt`); the saving itself is not measured |
| `indexer_ragged_decode.py` (required) | vLLM 0.32rc1's `needs_padded_path` (`sparse_attn_indexer.py:795-808`): uniform decode lengths with a token count that is not a multiple of the decode requests take the pack/unpack path | e15's uniform reshape crashes on such a batch (upstream: SM120 TP=2, 8 tokens over 6 seqs); `MAX_NUM_SEQS=4` adds unserved batch shapes | no-op on divisible batches; not reproduced |
| warmup stubs (`kernel_warmup = None`, `deepseek_v4_sparse_mla_attention_warmup = None`) | Skip DeepGEMM paged-MQA dummy forward | asserts block_kv∈{32,64} on this pack | **required** for boot |
| Exl3Config `weight_block_size` copy | Mapper picks `weight_scale` vs `weight_scale_inv` | Exl3Config keeps the field in `non_routed_quantization` | **required** (weights load) |
| `DSV41_DSPARK_MARKOV_SCALE` wrap | Scale Markov bias | experiment knob (default 1 = stock) | measured: ≠1 rejected (`evidence/` decode waves) |

## Compile / kernel experiments — verdicts (2026-09-19)

| Change | What | Verdict | Evidence |
|---|---|---|---|
| `--compilation-config mode=VLLM_COMPILE` | inductor compile of the model graph | **rejected**: pp@64k 699 vs 703 (flat), pp@16k −5%; nothing outside the opaque custom kernels to fuse | `results/RESULTS.md` E7 |
| p2b mma-over-m rows (`docker/Dockerfile.mma`, image `:mma8`) | batch ≤8 same-expert rows per mma tile | **rejected for serving**: pp@64k 709 vs 703 (flat), decode halves (m_loc≈1 pays the batching overhead) | `results/RESULTS.md` E8 |
| Chunk size 2048 / 16384 | chunked-prefill chunk | **rejected**: both directions lose; 8192 is the local optimum | E1/E2/E2b |
| `VLLM_EXL3_FAT_THRESHOLD` 96 / 2048 | fat-expert GEMM loop cutoff | **rejected**: pp@64k ≈ 700±10 at 96/256/2048 | E3/E4 |
| `NUM_SPECULATIVE_TOKENS=10` | deeper DSpark draft | **rejected**: L.A.I.L 12.1 vs 22.6; acceptance falls to 1.87 | E6 |

Kernel conclusion (measured, not guessed): prefill MoE time is invariant to
row batching, chunking, kernel mix, and compile mode — the wall is inside
the p2b weight-decode path itself (trellis `dq8` + shuffle stream at ~23 GB/s
effective vs ~273 GB/s UMA). The next kernel step is a vectorized multi-
weight trellis decode: a new inner loop, gated on the E8 image pipeline.
Do not retry row batching or threshold tuning; those axes are closed.

## Image/kernel env (vllm_exl3, read at import)

| Env | Default | What | Verdict |
|---|---|---|---|
| `VLLM_EXL3_MOE_KERNEL` | `native` (run.sh sets) | Native p2b fused MoE vs generic | **measured** ~23 vs 15 tok/s decode |
| `VLLM_EXL3_FAT_THRESHOLD` | 256 | Rows above which an expert takes the per-expert 128×128 fat GEMM loop instead of the standard kernel | **measured** (E3): 96 → pp@16k −11%, pp@64k noise; keep 256 |
| `VLLM_EXL3_PREFILL_SYNC` | unset | CPU/device sync workaround for 33..144-row prefill wedges | not needed at 8192 chunks; leave unset |
| `DSV41_P2B_SRC_SORT` | unset (off) | `=1`: p2b gate/up + down work items walk a src-sorted pair order so verify-row duplicate experts stream together (`widen_p2b_srcsort.py`); bit-exact, off = byte-identical SASS. Needs an image built with the patch: on an older `vllm_exl3_c` the boot logs `decode lever p2b_src_sort: ... the lever is OFF` (audit disarm) | **rejected (R34 s1)**: bitwise equal, SORT=0 SASS identical (6/6); cold p2b time −0.37 / +1.41 / +0.83 % at 0 / 25 / 50 % duplicate experts, below the 3 % gate. The s10 census measured 0.2987 duplicate fraction per layer, so the gain needs a kernel that reads each expert once (coop), not L2 reuse; `results/2026-09-24-review/campaign/s1-microbench/`, `results/2026-09-24-review/campaign/s10-diag-census-skew/` |
| `DSV41_P2B_COOP` | `2` (run.sh default since R35; unset or `0` = off) | `=1`: K=2 MCG p2b launches the SORT=2 kernel (`widen_p2b_coop.py`): each run of equal expert across the verify rows (chunks of <= 8) becomes MMA rows 0..7 of one tile, so a unique expert's trellis streams once per call; the slot sum is fixed-order with no atomics (deterministic). One-hot routing weights: same bits as p2b; full weights: fp32 slot-sum order only. Needs `docker/Dockerfile.e14` (`review-e14`). `=2`: the dataflow kernel `p2b_coop_df_kernel<2,1>` (`widen_p2b_dataflow.py`; built by `docker/Dockerfile.e14` / `review-e14`, into which the round-3 integration folded the Sept-era e15 recipe, and by `docker/Dockerfile`; `docker/Dockerfile.e15` now names the 1017 K-bits layer): the same coop tiles as one atomic task list (gate/up tiles, then down tiles), last-finisher epilogues and per-unit readiness counts instead of 5 of the 7 grid barriers, 3 blocks/SM without spills, L2 + A-fragment prefetch; bit-identical to `=1` for any routing weights. Engaged: `dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)` (audited); an image without it logs `decode lever p2b_coop_dataflow: ... the lever is OFF`, a call it cannot take `dsv41: p2b coop dataflow lever is OFF for this call` (audit disarm) | **`=2` default on since R35 (k3 serve campaign, bundle K; measurements in this row are Sept K=2)**. On the 1017 mixed-K pack `=2` needs `e15-kbits` (`docker/Dockerfile` or `docker/Dockerfile.e15`, `widen_p2b_kbits.py`: K 2/3/4 and K_gu≠K_d); 1017 boot D1 audit ok on both ranks. Sept: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): routed MoE −6.109 / −5.909 ms/step at c=1 (projection −6.76..−6.81) and −16.508 / −16.406 at c=2 (projection −15.17..−15.34). Leave-one-out K vs KC (the bundle without COOP): four prose c=1 −8.483 ms/step (noise 1.089), prose_long c=1 −8.175 (noise 0.308), every K boot below the KC median, acceptance matched. K vs KC pp_novel 32k −12.7 tok/s is 0.8 beyond its noise of 11.9 (runs overlap; COOP does not run in prefill). Before the serve campaign, **`=2` microbench KEEP**: real layer-20 weights, layer-20 census routing, 800 cold iterations, median per call p2b / `=1` / `=2`: m=4 521.3 / 378.8 / 336.3 us (-35.5% / -11.2%), m=8 1013.7 / 669.6 / 606.8 us (-40.1% / -9.4%), no-sharing m=4 526.3 / 537.5 / 488.4; `=2` streams at 89.5% (m=4) / 93.2% (m=8) of a flat 16-B read of the same bytes (235 / 236 GB/s, the fastest read measured on spark2; the coop tile pattern itself reads 226 / 231, the gap is a layout question, Round 9); one-hot bit-exact vs p2b, bitwise = `=1` at m 1..8 on layer 20 rank 0 and layer 33 rank 1, fp64-reference error = p2b's, 2000-call race hunts and 48 graph replays per m bitwise, the real e15 `vllm_exl3_c` = the bench bitwise; SORT=0/1/2 machine code byte-identical 13/13. Projected per step from the census of all 40 routed layers (layer-stratified, mean paired per-call saving x 40, three independent 800-iteration runs): -6.8 ms at c=1 (m=4, 6.76-6.81 +- 0.16; ~+12% tok/s at matched acceptance; -1.8 to -1.9 vs `=1`), -15.2 to -15.3 ms at c=2 (m=8); layer 20 alone reads -7.5 to -7.6 / -16.4 to -16.8 (it shares more experts than the average layer). Round 4 (k3 fix pass) kernel variants, all bitwise, none promoted (bar: >= 0.5% at m=4 and m=8): 32-column tail tiles (slower), whole-K tail prefetch and a deeper prologue prefetch (no change), early input-Hadamard loads (+0.74% m=4, +0.36% m=8). Next lever after the serve ABAB: fold the wrapper glue (plan in `kernel_study/p2b_coop/README.txt`). `results/2026-09-25-kernels/coop-moe/` |
| `TEMP_ROWS_FUSED` | 2048 (const) | Fused-kernel per-expert row cap; chunks with a hotter expert are re-sliced | **measured** (E1): avoiding re-slice via 2048 chunks does not improve pp — not a bottleneck |

## Experiment knobs — present, default OFF, with verdicts

All are env vars consumed by sitecustomize; every one was measured on the
frozen decode bench and rejected unless noted. Keep them documented so the
same ideas are not retried blind.

| Env | What | Verdict | Evidence |
|---|---|---|---|
| `DSV41_STEP_CENSUS=1` | Per-step census (draft/target/Engram timings, plus per-rank `pre_draft` = target end to draft start; compare ranks for the draft-start AR skew) | diagnostic only | `evidence/b12x-census` |
| `DSV41_STREAM_FEED=1` | Drain the VL wrapper's sorted weight list while it loads (`g8_stream_feed.py` on stock packs) | **default on since R34** (Container env table). The R31 drain kept every item alive until the 2026-09-24 fix, so R32's G8 "post-load balloon" verdict is still unproven (hb-3 not run) | `docker/patch/g8_stream_feed.py`, `tests/test_stream_feed.py` |
| `DSV41_INDEX_TOPK` | Clamp indexer topk | rejected as default | `evidence/extra-topk-128`, `evidence/indexer-native` |
| `DSV41_MHC_DECODE_SPLITS=1` | Collapse MHC prenorm split-K to 1 | rejected | `evidence/mhc-decode-splits` |
| `DSV41_MHC_DECODE_SPLITS=N` (N >= 2) | Force MHC prenorm split-K to N at num_tokens <= 64 (stock clamps 48 -> 16); `docker/patch/decode_levers.py` | **rejected at 40 (R34)**: kernel 18.08 → 11.17 us/call at T=4 (s1), trace −0.47 / −0.58 ms/step r0/r1 (s5). Not bit-exact (fp32 reduction order, maxabs 6e-6). Quick quality FAILED in C1 (selfcons A/A hazard 0.03409 > 0.03206); all four C A/A samples (0.0341 / 0.0194 / 0.0216 / 0.0220) sit above both B boots (0.0174 / 0.0146); L.A.I.L acceptance 2.147 / 2.136 vs B 2.160 / 2.183. Re-open only with an MHC-only ABAB and a larger selfcons sample | `results/2026-09-24-review/campaign/s1-microbench/`, `results/2026-09-24-review/campaign/s5-C1-decode-bundle/`, `results/2026-09-24-review/campaign/s7-C2-decode-bundle/abab.json` |
| `DSV41_MHC_DET_SPLITS=16` | Bitwise-stock decode mHC on faster kernels, T <= 16 (`docker/patch/mhc_det.py` + `mhc_det.cu`, NVRTC at load): det post (TileLang `mhc_post` FMA order), det prenorm GEMM (the same 16 splits and TF32 mma chain per (split, 8-col n-tile), fn packed once as its RNE-tf32 bits, 48 one-warp CTAs) and det fused norm (TileLang op order, sinkhorn sums with three independent shuffles per butterfly; its layer_input half runs during the GEMM). 16 is the only value (the stock split count it reproduces). Self-test at load, bitwise vs DeepGEMM/TileLang per fn, else OFF | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): elapsed mHC chain −0.984 / −0.976 ms/step (projection −1.0..−1.4). Before the serve campaign: **default off, serve ABAB pending (k3)**. Stock split-K is already deterministic (1000-run bitwise, 16 and 40 splits: the R34 A/A rise was not a racy reduction). Bitwise equal to stock: 4656 GEMM cases (87 real fn, T 1..16), whole 43-layer recurrence T {1,3,4,6,8,16}, 1000-run det, graph replay. Serve-like graph, us per 86-sublayer pass (one decode step's mHC; final suite on 8d1afbd): T=4 2141.1 -> 1110.5 (AR proxy a plain copy) and 1955.0 -> 997.1 (AR proxy a PDL-aware copy); T=8 2010.7 -> 1034.4 / 1956.1 -> 1017.5 (GEMM 16.0 -> 7.65 us/call, post 2.9 -> 1.4, fused-norm tail after the GEMM 4.8 -> ~3.3); ~101 MiB per rank for the packed fn; serve plan `results/2026-09-25-kernels/mhc-det/serve_validation_plan.txt` | `results/2026-09-25-kernels/mhc-det/` |
| `DSV41_MHC_DET_OVERLAP=1` | With `DSV41_MHC_DET_SPLITS=16`: each decode pre launches only layer_input (`mhc_det_norm_li`) right after the post; its det prenorm GEMM + sinkhorn (`mhc_det_norm_coef`) are deferred and forked onto a side stream at the next TP all-reduce (wrapped `GroupCoordinator.all_reduce`; attention wo_b and the MoE final reduce both go through it), joined before the next post; launched in place when no all-reduce comes first. Same kernels, same bits (`docker/patch/mhc_det_overlap.py`). Self-test at load, else the fused det pre | **not promoted, default off (R35 P5)**: KP = K + all five add-ons, ABAB K#3, KP#1, K#4, KP#2 (Kp-1, KP-1, Kp-2, KP-2): ms/step four prose c=1 −1.325 (noise 0.561), prose_long c=1 −1.81 (noise 0.333), L.A.I.L −1.732 (noise 0.082), every KP boot below the K median, against the s11 projection −2.4..−7.1. Per-boot median-run acceptance does not overlap on the two primary cells (four prose K 2.5897/2.5844 vs KP 2.525/2.5769; prose_long c=1 K 2.1505/2.117 vs KP 2.0729/2.0938), so step 6 gates on tok/s: prose_long c=1 +0.198 (noise 0.718, KP-2 below the K median) and L.A.I.L +1.459 (noise 1.594), not met. Quick quality PASS on both KP boots; greedy smokes byte-identical to K. No KP_p profile, so no per-add-on attribution (`results/2026-09-26-serve-r3/decision-addons.json`). Engaged on target and draft on both ranks; its mechanism was not measured. Before the serve campaign: **default off, serve ABAB pending (k3)**, after `DSV41_MHC_DET_SPLITS=16`. One-GPU `overlap_probe` (86 sublayers, real weights; AR stand-in = 5 CTAs spinning 22.7 us without memory traffic; work stand-in = 4 MB streamed per sublayer): mHC per pass det -> overlap -592..-756 us at T 1..8 (12-22.7 us window), -432..-738 with a 6 us window; mHC path ~2040-2180 (stock) -> ~1040-1120 (det) -> ~360-460 us. L2-prefetching only the next fn in the window: -84..-180 us (subsumed, not built). Bitwise on every output; changing-input race check 0 mismatches with 4 negative controls detected; edge inputs, compute-sanitizer clean. Unknown until the serve: NCCL RING_LL (polling host memory = the same LPDDR5x) with 1.23 MB read beside it | `results/2026-09-25-kernels/mhc-det/` (iterations 16-19) |
| `DSV41_ENGRAM_CACHE=1` | Host LRU for Engram rows | rejected (staging already prestage-hidden) | `evidence/engram-cache` |
| `DSV41_ENGRAM_FAST_STAGE=1` (default) | Parallel per-table Engram disk gathers | **KEEP**: kills ~13 ms/step of GPU idle; L.A.I.L 22.0-23.5 -> 25.2-26.3 | `results/2026-09-19-faststage` |
| `DSV41_ENGRAM_STAGE_THREADS=16` | Worker pool for the parallel stage | default measured good | round 11 |
| `DSV41_ENGRAM_PREFETCH=1` | fadvise next-step Engram rows from CPU-hash at postprocess | round 11: off, pf_hit ~0%. Superseded: v3 is **default on** since R21 (pf_hit 100% at R21 only; see the Container env row for the 2026-09-24 runtime disarm) | round 11, R21 |
| `DSV41_ENGRAM_CENSUS=1` | Per-gather read/dequant timing + pf_hit | diagnostic; **default on** with prefetch v3 | round 11, R21 |
| `DSV41_ENGRAM_WILLNEED=1` (+`DSV41_ENGRAM_WILLNEED_MIN_ROWS`, default 512) | gv2 prefill pre-pass: one `posix_fadvise(WILLNEED)` per merged 4 KiB page span of a gather's unique rows (w and s files) before the serial preadv loop, for calls with >= MIN_ROWS rows (decode is 48-96) | **default on since R34** (Container env table; E1 gate passed in s4). CPU read bench: cold 2000 rows 844.7 -> 18.9 ms/call, hot 7.1 -> 7.6 | `results/2026-09-24-review/engram-prefill/read_bench.txt`, `results/2026-09-24-review/campaign/s4-E1-willneed/` |
| `DSV41_ENGRAM_GATHER_V2_MAX_ROWS=N` | gv2 calls with more than N rows take the stock 32-thread pool (prefill fallback arm; unset/0 = off) | untested and not needed: WILLNEED gave 4.23x on novel 8k, above the 1.3x trigger for the E2 (MAX_ROWS=256) boot | `docker/patch/engram_gather_v2.py` |
| `DSV41_ENGRAM_NATIVE_STAGE=1` (+`DSV41_ENGRAM_NATIVE_MAX_TOKENS`, default 64; `DSV41_ENGRAM_NATIVE_VERIFY`, default 8) | Decode-size Engram stage gather in C with the GIL released (`engram_native.c` via ctypes, gcc at first use): preadv2(RWF_NOWAIT) per owned row, misses WILLNEED together, fp8 x ue8m0 -> bf16 straight into the pinned staging buffer; hash, sync, H2D and prefill stay stock. Refuses to arm (one LOG_DISARMED line, nothing patched) when `DSV41_ENGRAM_CPU_HASH=1` or `DSV41_ENGRAM_DEFER=1` has rewritten stage() | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): Engram stage span (hash DtoH end → rows HtoD start) −1.507 / −1.161 ms/step; it halves but stays at 1.368 / 1.558 ms against the expected 0.2-0.35 (page-cache misses, P3 top-3 item 2). Before the serve campaign: **microbench KEEP, serve untested (k3 fusion-host)**: bit-exact (all 65536 byte pairs, real-table calls, first 8 serve calls re-verified; sabotage run refills the step from the stock gather and disarms). GPU idle the stage causes, spark2, real stage() both arms, CUDA-event span behind a 4 ms GPU sleep: n=4 1438 -> 215 µs (rank 0 head span, prefetched rows), 1768 -> 197 (rank 1), 2116 -> 224 (hot), 1967 -> 222 (cold L2); n=8 2494 -> 361 (r0), 2322 -> 384 (r1). The stock arm is noisy (p10-p90 ~0.8-1.2 ms). Under a pure-Python GIL spinner (worst case, CPU bench) 274.9 ms stock vs 18.7 ms native per call. Projected −1.2..−1.6 ms/step c=1, −1.9..−2.1 c=2; Round 24 (DEFER, gather off-thread, flat L.A.I.L at n=5) means only a serve ABAB with enough repeats can confirm | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `DSV41_ENGRAM_EARLY_HASH=1` (+`DSV41_ENGRAM_EARLY_VERIFY`, default 8; needs `DSV41_ENGRAM_NATIVE_STAGE=1`) | Launch the Engram lookback + hash + DtoH at the start of GPUModelRunner.prepare_attn instead of as the last eager launch, so the native gather overlaps the queued block-table / slot-mapping / attention-metadata kernels (`engram_early_hash.py`) | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): eager chain before the hash −1.018 / −1.001 ms/step (projection −0.18). Before the serve campaign: **microbench KEEP, serve untested (k3 fusion-host)**: first 8 steps compare the early hash with the stock late hash int32 for int32 (mismatch disarms). The hide window is the eager chain between the early point and stage(): 241 ops stock (alone 529 µs), 122 with `DSV41_ATTN_T2R_DEDUP=1` (253 µs), 77 with it and `DSV41_SWA_META_FUSED=1` (178 µs). spark2, chain + stage GPU span native -> native+early: stock chain n=4 817.0 -> 603.0 µs, n=8 968.8 -> 582.0; ALL chain n=4 404.6 -> 223.5, n=8 567.7 -> 369.2; the gather is fully hidden in every case. Projected a further −0.21 ms/step c=1, −0.39 c=2 with the stock chain (B1), −0.18 / −0.20 inside ALL | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `DSV41_ENGRAM_NATIVE_POLL=1` (+`DSV41_ENGRAM_NATIVE_POLL_US`, default 2000; needs `DSV41_ENGRAM_NATIVE_STAGE=1`) | The native stage gathers all Engram tables of a decode step in one `eng_gather_tables_bf16` call (`engram_poll.c`, its own .so that includes `engram_native.c`; `engram_native_poll.py`): the RWF_NOWAIT probe pass over every row of every table (on this 6.17 kernel a probe that misses starts the page's read), no WILLNEED pass (it submitted nothing new), then each missed row re-probed with a cpu yield between tries for up to POLL_US, then WILLNEED for whatever is still missing and a blocking pread per row (today's sequence: a kernel whose probe does not start the read loses nothing), the same dequant. Replaces, table by table, WILLNEED + a blocking pread per missed row that sleeps on every page still in flight. Same buffered reads: same bytes, same pages cached. Not io_uring (Docker's default seccomp blocks it) and not AIO + O_DIRECT (a second device read per miss, since the probe already started one). No RWF_NOWAIT or any error prints one LOG_DISARMED line and that step and the rest go per table. Off: the per-table path is unchanged and the new code is not built | **default off, not promoted**. GPU: boot I4 (on): armed (`engram native poll armed (2 tables, 1536 rows max, spin 2000 us)`) and native-stage self-check bit-exact on both ranks, 0 DISABLED; c=4 distinct window census rank 0 1,166 us/call (47.8 misses/call), rank 1 1,827 us/call (64.0) (`results/2026-10-19-1017-perf2/I4/native-census-c4.txt`). Not an A/B: B0c's 2,641 / 7,447 us/call came with 247 / 201 misses per call, and I4's decode cells are within noise of I1 (poll off) with other levers changed too. Boot I6 (poll, router GEMV and main_proj overlap off) against I4 (on): c=1 prose 49.2 vs 48.5 tok/s, L.A.I.L 38.3 vs 37.1, c=4 distinct prose 94.7 vs 89.0, all within run noise: no decode gain (`results/2026-10-19-1017-perf2/I6/numbers-vs-I4.txt`). Motivation: B0c census at c~4, 334 rows/call: rank 1 (spark2) 7,447 us/call with 201 misses (~37 us/miss), rank 0 2,641 us with 247 (~11 us/miss); decode-glue: rank 0 waits ~7.1 ms at the first target all-reduce. Host: probe-started reads measured (64 RWF_NOWAIT probes on a dropped file: ~63 pages cached within 200 ms); tests/test_engram_native_poll.py (bytes vs eng_gather_bf16 and a numpy reference: hot, cold, mixed, unowned, padding heads, rows straddling pages, the last row before EOF, 0-64 tokens, two files; same pages cached as the per-table path; POLL_US 0; no-RWF_NOWAIT, failing-call and attach-failure fallbacks re-gather per table with one LOG_DISARMED line) | `perf2/engram-io/`, `kernel_study/fusion_host/engram_native_check.py`, `results/2026-10-19-1017-perf2/I4/`, `results/2026-10-19-1017-perf2/I6/` |
| `DSV41_ENGRAM_PREFILL_AHEAD=1` (+`_TOKENS`, default and cap = `--max-num-batched-tokens`; `_THREADS`, default 8; `_VERIFY`, default 4; `_DONTNEED`, default 1; `_CPUS`, default unpinned) | Prefill-size Engram stage calls (more tokens than `DSV41_ENGRAM_NATIVE_MAX_TOKENS`) take their rows from a read-ahead: after a stage whose batch holds a request still prefilling, the next chunk's rows (hash ids from the prompt tokens in the V2 runner's UVA `all_token_ids`, numpy transcription of `_hash_ids_kernel`) are read and dequantized by `engram_native.c` on a thread pool (GIL released) into a pinned buffer while this chunk computes. The next stage keeps the stock GPU hash + DtoH + sync and uses a read-ahead row only where its hash ids equal the GPU's for every table (a mismatch disarms); the other rows (a request's first chunk, decode rows of a mixed batch) take the same parallel native gather. `_DONTNEED=1` drops only the pages a prefill read brought into the page cache (RWF_NOWAIT miss); decode rows are never dropped. Memory: 12 KiB pinned per read-ahead token (96 MiB at 8192, 48 MiB at 4096, at first use) + ~7 MiB (`engram_prefill_ahead.py`) | **default on since perf2 engram-io**: phase A on spark2 (e15-kbits): the hash mirror against the real Triton `_hash_ids_kernel` 0 segment mismatches in 40 trials; staged device rows bitwise equal to the stock stage (DONTNEED 1 and 0) in 7 scenarios x both ranks; sabotage gives one DISABLED line; real 1017 tables 144/144 calls bit-equal. Serve: armed and self-check bit-exact on both ranks in I1, I2, I4, 0 DISABLED; every later prefill chunk was read ahead (I1 93/93; I4 218/218, 59 of them mixed steps whose 12 decode rows took the sync gather), wait median 0.0 ms, max 0.4 ms; first chunks (I1 37, I4 52) take the parallel C gather; I3 -> I2 (read-ahead and TC on, replay off) pp_novel x1.94 / x2.36 / x2.31 at 4K / 16K / 64K, not separated from TC (`results/2026-10-19-1017-perf2/I1/prefill-ahead-census-final.summary.txt`, `results/2026-10-19-1017-perf2/I4/prefill-ahead-census-final.summary.txt`). `_DONTNEED` stays 1: boot I6 (`_DONTNEED=0`) brought no decode gain, its page cache grew 3.4 GiB on spark1 over the prefill micro bench (Cached 3.1 -> 6.4 GiB; spark2 +2.4 GiB), and its 64K pp_novel was 2,563 tok/s against 2,849 on I4 (`results/2026-10-19-1017-perf2/I6/cached.txt`, `results/2026-10-19-1017-perf2/I6/numbers-vs-I4.txt`). Host: 26 tests (mirror vs a scalar `_hash_ids_kernel` transcription incl. dead tokens and positions 0..2, chunking invariance, scheduler simulation through the real C gather: chunked prefill, 4-sequence mixed batches, prefix-cache start, preemption, abort, two prefills, shorter chunk, wrong read-ahead, both TP ranks, bit-exact rows; DONTNEED leaves the page cache as found, mincore). Motivation: B0c prefill profile, rank 0 chunk 2 (8,001 tok) 2,985 ms of 11,456 ms GPU idle between hash DtoH and rows HtoD; rank 1 chunks 2,568 / 1,636 ms (`perf2/baseline/report.md`); 1017 [gv2-census] 16-19 µs/row read time | `perf2/engram-io/`, `kernel_study/fusion_host/engram_prefill_ahead_check.py`, `results/2026-10-19-1017-perf2/I1/`, `results/2026-10-19-1017-perf2/I4/`, `results/2026-10-19-1017-perf2/I6/` |
| `DSV41_MOE_PREP_FUSED=1` (+`DSV41_MOE_PREP_VERIFY`, default 16) | vllm_exl3 `_apply_native_fused_moe`'s p2b input prep (14 single-block torch kernels per MoE layer on the routing path: id mapping/clamp/valid mask, weights and x to fp16) as one Triton kernel, and p2b's fp16 output cast to bf16 once instead of via fp32 (apply_exl3_experts only); the image's functions with only those blocks rewritten (`moe_prep_fused.py`) | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): router gate end → routed start −2.944 / −3.000 ms/step (projection −0.19..−0.34). Before the serve campaign: **microbench KEEP, serve untested (k3 fusion-host)**: bit-exact (576 sweep cases incl. invalid ids / NaN / inf / fp16 overflow, 100 graph replays, the rewritten function vs stock with a stand-in p2b; sabotage run: verify returns stock and disarms, self-test disarms). Graph replay m=4, launch hidden: 20.35 -> 3.97 µs warm, 26.56 -> 4.22 cold, 26.43 -> 4.13 beside a DRAM-saturating copy (launch included: 19.78 -> 6.38); m 1-8 alike. The routing path ended ~7.4 µs after the shared expert in ~98% of layers: projected −0.15..−0.30 ms/step, and it stops capping shared-expert speedups. Output cast: all 65536 fp16 patterns identical; 40 casts 122.7 -> 85.9 µs/step | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `DSV41_MOE_PREFILL_TC=1` (+`DSV41_MOE_PREFILL_TC_MIN_ROWS`, default 9; `DSV41_MOE_PREFILL_TC_VERIFY`, default 2) | Routed-MoE calls of >= MIN_ROWS token rows (prefill) on two persistent mma.sync kernels (`moe_prefill_tc.cuh`; `moe_prefill_tc.py` wraps vllm_exl3 `apply_exl3_fused_moe`): rows sorted by expert on the device (no host sync), each EXL3 trellis tile decoded once per 64 rows of one expert (gate/up and down) instead of once per 16 rows (exl3_moe) or a full fp16 reconstruct (fat path); ExLlamaV3's decode and rotations, the fat path's activation, fp32 accumulate, fp32 atomic combine. Not bit-exact vs stock (stock's exl3_moe keeps fp16 split-K partials): self-check + first VERIFY calls against stock, max_rel <= 2e-2 / rms_rel <= 5e-3, else disarm. exllamav3 exl3_moe's scratch (294 MiB per rank) is no longer allocated at load (build_exl3_fused_state rewritten); a stock call allocates it once (`exl3_moe scratch allocated at the first stock call`), the references borrow it. The default image `e16-rows` (`docker/Dockerfile`, `docker/Dockerfile.e16`) prebuilds it; another image, or edited kernel sources, compile it at first use (~32-36 s per rank measured) | **default on since perf2 prefill-moe**: kernel checks 28/28 on spark1 (K pairs 2/2, 2/3, 3/3, 3/4; uniform and census routings, edge cases): ~4.7x closer to an fp32 reference than stock, deterministic; median per layer per rank, weighted over the 2.70 allocation, stock -> TC 40.4-42.5 -> 15.9-16.7 ms at m=2048 (x2.5), 132.4-142.0 -> 44.7-45.5 ms at m=8192 (x3.0-3.1); 0 host syncs per call (stock 3); transient peak 269 MiB at 8192 rows, nothing held (stock census calls hold 202-637 MiB of fat scratch); the lazy exl3_moe scratch saves 294 MiB per rank. Serve: self-check ok and verify passed on both ranks in boots I1, I2, I4 (JIT); I3 -> I2 (TC and Engram read-ahead on, replay off) pp_novel 817 / 654 / 676 -> 1,583 / 1,541 / 1,561 tok/s at 4K / 16K / 64K (not separated from the read-ahead); quick quality PASS on I1 and I4 (TC on) and I3 (off) (`results/2026-10-19-1017-perf2/I1`, `I2`, `I3`, `I4`). Quality --full on the full stack's defaults: PASS (FINAL, `results/2026-10-19-1017-perf2/FINAL/quality-full.txt`) | `kernel_study/moe_prefill/`, `results/2026-10-19-1017-perf2/` |
| `DSV41_CANDIDATE_MASK_BOUNDED=1` (+`DSV41_CANDIDATE_MASK_VERIFY`, default 4) | V4.1 decode candidate mask (4 indexer layers/step) stops at each row's end instead of zeroing 131k flag slots and writing -inf across the whole max_model_len-wide logits row (`candidate_mask_bounded.py`) | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): mask + flags kernels −0.300 / −0.296 ms/step (projection −0.27..−0.43). Before the serve campaign: **microbench KEEP, serve untested (k3 fusion-host)**: bit-exact below every row end (54 cases); decode top-k with garbage past the end returns the same index sets (40 cases; the stock top-k's order is itself nondeterministic); 20 graph replays; sabotage run: verify restores the stock result and disarms, self-test disarms. Graph replay at the serve's 1M width, launch hidden, typical ends: rows 4 98.1 -> 8.1 µs warm, 116.7 -> 10.2 cold; rows 8 186.6 -> 12.1 warm, 205.8 -> 12.3 cold. The bounded cost grows with the row end (cold, end 128k: rows 4 119.0 -> 32.8, rows 8 207.8 -> 48.1). Projected −0.27..−0.43 ms/step c=1, −0.70..−0.77 c=2 | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `DSV41_INDEXER_WP_GEMV=1` (+`DSV41_INDEXER_WP_VERIFY`, default 8) | The indexer's bf16 weights_proj [32, 5120] (cuBLAS: a cutlass wmma kernel on 2 one-warp CTAs, the last branch at the qkv_a join in all 8 indexer layers) as a Triton GEMV with the same k-ordered MMA chain, M 2..8 (`indexer_wp_gemv.py`) | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): qkv join wait on the 8 weights_proj layers −0.123 / −0.122 ms/step (projection −0.10..−0.17). Before the serve campaign: **microbench KEEP, serve untested (k3 fusion-host)**: bit-identical to cuBLAS on all 8 layers' real weights at M 2..8 (M = 1: cuBLAS uses another kernel, stays stock), eager self-test + verify, 20 graph replays; sabotage run: verify returns stock and disarms, self-test disarms. Graph replay m=4, launch hidden: 38.94 -> 10.69 µs cold, 26.46 -> 10.08 warm, 38.94 -> 10.43 beside a DRAM-saturating copy; m 3-8 alike (latency-bound: 0.33 MB). Projected ~−0.10..−0.17 ms/step | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `DSV41_ATTN_T2R_DEDUP=1` (+`DSV41_ATTN_T2R_VERIFY`, default 8) | One token -> request map per decode-step metadata build instead of one per KV-cache group (21/step: 3 draft + 18 target): later groups sharing the same query_start_loc tensor objects copy the first group's result into their own buffer; the first 8 reuses are checked against the stock method, a mismatch restores the stock values and disarms (`attn_t2r_dedup.py`) | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): t2r scan/map kernels −0.070 / −0.068 ms/step; no step gain on the critical path in this bundle (masked). Before the serve campaign: **microbench KEEP, serve untested (k3 fusion-host)**: bit-exact (own buffers, same values, graph addresses unchanged; every buffer checked every iteration; sabotage run on the real CommonAttentionMetadata disarms and restores the stock map). GPU-bound (calibrated 6 ms GPU sleep ahead, host enqueue checked below it): ops per step 168 -> 35; 510.8 -> 83.8 µs warm, 518.9 -> 104.3 µs cold, c=1 = c=2. Projected −0.41..−0.43 ms/step (the earlier −0.52 came from a host-bound bench) | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `DSV41_SWA_META_FUSED=1` (+`DSV41_SWA_META_VERIFY`, default 8) | A causal pure-decode DeepseekSparseSWAMetadataBuilder.build (15 per target step) writes is_valid_token, the lens tail and the SWA indices/lens with one Triton kernel instead of compare + copy + fill + the stock kernel; the image's build with two blocks rewritten; prefill, mixed, non-causal DSpark draft builds and update_draft_decode_metadata stay stock (`swa_meta_fused.py`) | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): SWA metadata kernel time −0.001 / +0.004 ms/step; step gain masked. Before the serve campaign: **microbench KEEP, serve untested (k3 fusion-host)**: integer math, bit-exact (48 builds of the rewritten vs the image's build incl. padded requests/tokens and short sequences, 20 graph replays with new inputs, sabotage test disarms and keeps the stock values). GPU ops per build 12 -> 9 (2 with `DSV41_ATTN_T2R_DEDUP=1`); 15 builds GPU-bound behind a calibrated sleep: 521.1 -> 426.8 µs warm, 617.1 -> 461.6 µs cold, c=1 = c=2. Projected −0.09..−0.16 ms/step | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `DSV41_SWA_META_BATCHED=1` (+`DSV41_SWA_META_BATCHED_VERIFY`, default 8; needs `DSV41_SWA_META_FUSED=1`) | The causal SWA groups of one target `build_attn_metadata` call (43 with packed KV groups, 15 without) get their decode window rows from one launch over (tokens, groups) instead of one fused launch each: the first group's build launches for all, the others skip theirs; every group keeps its own buffers. Off for calls with `model_specific_attn_metadata`, non-bool causal, graph capture, or until the fused self-check is done. The kernel body is the fused kernel's line for line (`swa_meta_batched.py`; e15 counterpart of PR #60996's second commit) | **default on since perf2 kv-pack**: in-serve self-check bit-exact (43 groups) on boots I1-I4 (`results/2026-10-19-1017-perf2/I3/boot-grep-boot.txt`); boot I3 (packed groups and batching, the later perf2 levers off) ran c=1 prose at 52.7 ms/step (B0, 18 groups: 52.7) and c=4 distinct at 154.5 ms/step (B0 158.0); there is no batched-vs-fused A/B, so the batching's own ms/step is not measured. Bit-exact check against the image's stock ops in `kernel_study/fusion_host/swa_meta_batched_check.py` (GPU, or CPU under TRITON_INTERPRET=1); in-serve check of the first 8 launches against the per-group fused launches | `kernel_study/fusion_host/swa_meta_batched_check.py`, `results/2026-10-19-1017-perf2/I3/` |
| `DSV41_ENGRAM_WKV_TP=1` | The Engram wkv (ReplicatedLinear [25600, 6144] MXFP8, read whole on both ranks: 758 µs per call, 2 a step in r3) becomes a ColumnParallelLinear with the output all-gathered: each rank loads and reads its 12800 rows (`engram_wkv_tp.py`); only with TP > 1, no sequence parallelism, no Engram DP | **not promoted, default off (R35 P5)**: KP = K + all five add-ons, ABAB K#3, KP#1, K#4, KP#2 (Kp-1, KP-1, Kp-2, KP-2): ms/step four prose c=1 −1.325 (noise 0.561), prose_long c=1 −1.81 (noise 0.333), L.A.I.L −1.732 (noise 0.082), every KP boot below the K median, against the s11 projection −2.4..−7.1. Per-boot median-run acceptance does not overlap on the two primary cells (four prose K 2.5897/2.5844 vs KP 2.525/2.5769; prose_long c=1 K 2.1505/2.117 vs KP 2.0729/2.0938), so step 6 gates on tok/s: prose_long c=1 +0.198 (noise 0.718, KP-2 below the K median) and L.A.I.L +1.459 (noise 1.594), not met. Quick quality PASS on both KP boots; greedy smokes byte-identical to K. No KP_p profile, so no per-add-on attribution (`results/2026-09-26-serve-r3/decision-addons.json`). Booted on 2 ranks 3 times (KPsafe-1, KP-1, KP-2): armed, self-check bit-exact on layers 1 and 14, no hang; model memory 75.43 vs 75.58 GiB per rank; dense GEMV armed drops from 9 to 8 per rank (ARMS-r3 s12). Timing not measured alone. Before the serve campaign: **microbench KEEP, serve untested, never booted on 2 ranks (k3 fusion-host)**: bit-identical (the production MXFP8 GEMM gives each output column the same bits whatever N: 128 cases, layers 1 and 14, M 1..2048; with the image's own classes and loaders, TP=2 emulated in one process, the processed shards equal the stock processed weight and swizzled scale byte for byte, the gathered output equals the replicated layer in 128 more cases and 20 graph replays). A rank-symmetric self-check after loading compares the sharded GEMM + all-gather with the stock kernel on the gathered weight (verdict all-reduced; sabotage disarms to the gathered replicated weight, equal to stock). Per rank, m=4: 805.6 -> 444.4 µs cold, 754.5 -> 384.9 warm; the all-gather of [4, 12800] bf16 is modeled at ~20-30 µs (the r3 Engram embed gather, 24 KiB/rank: 18.4 µs). Projected −0.66..−0.70 ms/step c=1, −0.64..−0.68 c=2, −157 MB weight per rank; prefill pays a [chunk, 12800] gather per Engram layer (serve plan section 5) | `results/2026-09-25-kernels/fusion-host/iterations.txt` (s10), `.../fusion-host/fix/` |
| `--async-scheduling` | V1 async scheduler | neutral-negative on L.A.I.L (21.8/21.9 vs 22.0-23.5). This build turns async scheduling on when it is unset (vllm/config/vllm.py:1297-1346; dspark is exempt from the spec-decode disable), so R11 compared on vs on: noise, not a verdict on the lever | round 11 |
| `NCCL_MIN/MAX_NCHANNELS=1` | Force single NCCL channel | null (steps/s 9.65 vs 9.68-10.26); AR p50 already 41-54 us | round 11 |
| `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` `NCCL_CROSS_NIC=0` (± `NCCL_IB_MERGE_NICS=0`) | Dual-rail NCCL over both ACTIVE f1 HCAs (`tools/nccl_dualrail.sh`, serve down) | **rejected (R34 s1)**: busbw +64 / +71 / +77 % at 32 / 64 / 128 MiB (single ~13.1 GB/s), but small-message latency on the median of 3 reps is +30.0 % (dual_keep) and +9.5 % (nomerge), over the +5 % limit; decode all-reduces are small messages | `results/2026-09-24-review/campaign/s1-microbench/` |
| `NCCL_GRAPH_MIXING_SUPPORT=0` | Drop NCCL's support for mixing captured and non-captured calls on one communicator | **microbench only, not a serve lever yet (k3 profile)**: exact decode sizes in a CUDA graph, 2x-L2 flush before each call, profiler off: steady critical-path cost of an AR m=4 57.6 → 22.5 µs (m=8 63.2 → 29.1, logits AG m=4 100.9 → 61.4); with mixing on, each captured network collective slows the neighbouring bandwidth-bound kernel 13.5% (32% in a replay's first slot). Modeled −3.08 ms/step at c=1 and −3.06 at c=2. **Unsafe as a plain env var**: the docs do not support a non-captured collective launched during an outstanding graph on the same communicator, and the serve launches the eager logits AG 45 ms before the target graph ends in 100% of steps; needs eager collectives on a second communicator, then a serve ABAB | `results/2026-09-25-kernels/profile/nccl-sweep.json`, `nccl-graph-diag.txt` |
| `DSV41_NCCL_EAGER_TWIN=1` | NCCL graph mixing off, made safe: the patch writes `NCCL_GRAPH_MIXING_SUPPORT=0` and the TP group gets an eager-only twin communicator. A collective whose stream is capturing stays on the stock communicator; every other one goes to the twin (`docker/patch/nccl_eager_twin.py`). Other groups get an eager-only guard. Fails closed: disarms unless the NCCL that vLLM loads is in `VALIDATED_NCCL` (2.30.7, the only release whose graphUsageMode handling was read). A group with >1 rank and no working PyNccl raises REFUSED | **not promoted, default off (R35 P5)**: KP = K + all five add-ons, ABAB K#3, KP#1, K#4, KP#2 (Kp-1, KP-1, Kp-2, KP-2): ms/step four prose c=1 −1.325 (noise 0.561), prose_long c=1 −1.81 (noise 0.333), L.A.I.L −1.732 (noise 0.082), every KP boot below the K median, against the s11 projection −2.4..−7.1. Per-boot median-run acceptance does not overlap on the two primary cells (four prose K 2.5897/2.5844 vs KP 2.525/2.5769; prose_long c=1 K 2.1505/2.117 vs KP 2.0729/2.0938), so step 6 gates on tok/s: prose_long c=1 +0.198 (noise 0.718, KP-2 below the K median) and L.A.I.L +1.459 (noise 1.594), not met. Quick quality PASS on both KP boots; greedy smokes byte-identical to K. No KP_p profile, so no per-add-on attribution (`results/2026-09-26-serve-r3/decision-addons.json`). P1 device check passed on both ranks (0 bad steps; two-node harness c=1 twin −6.619 ms/step vs keep, not a serve). Engaged in the serve with guard and routing lines. Open: NCCL's graphUsageMode line appeared 0 times per rank against the 434 ARMS-r3 s3 expects. Before the serve campaign: **implemented, default off; serve ABAB pending (k3 comm)**: CPU tests on pinned e13 sources; in-image check: e13 reports 2.30.7, a 2.31.0 reader disarms. NCCL 2.30.7 logs "graphUsageMode is set to 0 but the user is capturing graphs" once per captured collective (434 lines per rank at boot); expected. Two ranks on one GB10 (MPS, NET/IB loopback): `nccl_twin_check.py --device-check` checked 330/330 pipelined steps on both ranks in keep/twin/keep_qos/twin_qos with 0 bad steps; an injected fault at step 37 was flagged exactly. Modeled −3.08 ms/step at c=1 from the clean two-node sweep; in the one-GPU emulation the only consistent gain is the draft phase (eager collective during an outstanding graph, −0.22 ms/step, 3/3 reps, review) | `results/2026-09-25-kernels/comm/` (`b-hardening/`, `twin-devcheck/`) |
| `DSV41_PM_QOS_US=20` | Hold a PM QoS CPU wake-up latency request (`/dev/cpu_dma_latency`, host-wide) while the serve runs. run.sh passes the device to both containers only when set (`docker/patch/pm_qos.py`) | **not promoted, default off (R35 P5)**: KP = K + all five add-ons, ABAB K#3, KP#1, K#4, KP#2 (Kp-1, KP-1, Kp-2, KP-2): ms/step four prose c=1 −1.325 (noise 0.561), prose_long c=1 −1.81 (noise 0.333), L.A.I.L −1.732 (noise 0.082), every KP boot below the K median, against the s11 projection −2.4..−7.1. Per-boot median-run acceptance does not overlap on the two primary cells (four prose K 2.5897/2.5844 vs KP 2.525/2.5769; prose_long c=1 K 2.1505/2.117 vs KP 2.0729/2.0938), so step 6 gates on tok/s: prose_long c=1 +0.198 (noise 0.718, KP-2 below the K median) and L.A.I.L +1.459 (noise 1.594), not met. Quick quality PASS on both KP boots; greedy smokes byte-identical to K. No KP_p profile, so no per-add-on attribution (`results/2026-09-26-serve-r3/decision-addons.json`). Engaged in every serve process (head 4, worker 3). Not measured alone. Before the serve campaign: **implemented, default off; serve ABAB pending (k3 comm)**: GB10 cpuidle LPI-1/2/3 advertise 42/231/433 us exit latency, and LPI-2 is never entered (usage 0 on all 20 cores of both hosts: same target residency as LPI-3). A CUDA graph host node (NCCL's per-collective proxy-op upload) after 20 ms of host idle waits 481-586 us median with no request and 3.4-14.7 us with a 0/20/40 us request. A 50/100 us request still waits 384-608 us: LPI-1 must be excluded, and its real exit cost is far above 42 us. No GPU cost measured: 1 GiB read 232.3 vs 232.9 GB/s, bf16 GEMM 89.7 vs 90.0 TFLOPS, SM 2190 MHz and ~29 W either way (alternating 3 s blocks, spark2). The r3 trace books 1.13 ms/step of graph startup to this wait | `results/2026-09-25-kernels/comm/host-node-qos-sweep/`, `c-scope/`, `c-cost-gdr/` |
| `DSV41_AR_L2_PREFETCH=1` (+`DSV41_AR_L2_PREFETCH_MIB`, default 10 = all of the next `fused_wqa_wkv`) | A side stream forked right before each target MoE all-reduce issues a TMA L2 prefetch (`cp.async.bulk.prefetch.L2`, 1-CTA Triton kernel) of the next layer's `fused_wqa_wkv` weight and scale. It is joined in the next layer's `DeepseekV4Attention._split_qkv_and_norm`, after its qkv_a and before the breakable graph's eager break (`docker/patch/ar_l2_prefetch.py`, `l2pf_kernel.py`); 37/40 layers (none before Engram layers or after the last) | **not promoted, default off (R35 P5)**: KP = K + all five add-ons, ABAB K#3, KP#1, K#4, KP#2 (Kp-1, KP-1, Kp-2, KP-2): ms/step four prose c=1 −1.325 (noise 0.561), prose_long c=1 −1.81 (noise 0.333), L.A.I.L −1.732 (noise 0.082), every KP boot below the K median, against the s11 projection −2.4..−7.1. Per-boot median-run acceptance does not overlap on the two primary cells (four prose K 2.5897/2.5844 vs KP 2.525/2.5769; prose_long c=1 K 2.1505/2.117 vs KP 2.0729/2.0938), so step 6 gates on tok/s: prose_long c=1 +0.198 (noise 0.718, KP-2 below the K median) and L.A.I.L +1.459 (noise 1.594), not met. Quick quality PASS on both KP boots; greedy smokes byte-identical to K. No KP_p profile, so no per-add-on attribution (`results/2026-09-26-serve-r3/decision-addons.json`). Engaged on 37/40 layers at 9.02 MiB per window. Not measured alone. Before the serve campaign: **implemented, default off; a range, not a number (k3 comm fix)**: vs a clock-spin AR stand-in, through the lever's own hooks, −15.7..−17.3 us per window (qkv_a 47.1 → 16.4-18.7 us; −0.6 ms/step, optimistic). Vs a real NCCL AR (two ranks on one GB10, NET/IB loopback, p2b stand-in, the lever's fork and join points) the late rank's AR slows +23..+27 us and qkv_a saves 28..32 us: net −1.9..−8.1 us at m=4, +2.4..−4.3 at m=8 (−0.07..−0.30 ms/step c=1, pessimistic). Placement, issue and budget come from that data: a fork after the AR costs mHC +18..+34 us; paced / ring / multi-CTA issue lose; 5.5 MiB beats the whole qkv_a only against the spin. A prefetch before p2b loses as a separate kernel (+2..+18 us, serialized: p2b fills every SM's RF) and is zero-sum inside p2b (real p2b +8..+18 us for qkv_a −9..−17). The earlier join at the next layer's entry is 3.5..5.7 us per window worse against the spin (lever's own hooks, whole qkv_a) and equal with the real AR. E fails in the serve if the late rank's MoE AR slows by ≥ the qkv_a saving | `results/2026-09-25-kernels/comm/ar-window-nccl/`, `final/`, `p2b-window-*` |
| Decode-size NCCL knob sweep (`NCCL_PROTO` LL / Simple / LL128, `NCCL_ALGO` Ring / `allreduce:tree`, `NCCL_MAX/MIN_NCHANNELS` 1 / 2 / 4 / min 8, `NCCL_NTHREADS` 64-512, `NCCL_IB_USE_INLINE`, `NCCL_PROXY_CPUSET`, `NCCL_IGNORE_CPU_AFFINITY`, `NCCL_NET_GDR_C2C`) | `tools/nccl_decode_sweep.sh`, serve down, both nodes | **no knob beats KEEP (k3 profile)**: clean runs, c=1 modeled collective path KEEP 6.19 ms/step; LL +0.46, Simple +1.03, LL128 +1.67, Ring +0.18, 1 channel +0.58, 2 / 4 / min-8 channels −0.19 / −0.15 / −0.14 at c=1 but +0.84 / +0.29 / +0.44 at c=2, NTHREADS 128 / 256 +0.19 / +0.05. A global `NCCL_ALGO=Tree` fails (AllGather has no Tree). Rep 2 and the rest ran while another session's chromium shared spark1's GPU and are tagged, not pooled with the clean runs | `results/2026-09-25-kernels/profile/nccl-sweep.json` |
| `NCCL_CUMEM_ENABLE=1` | cuMem-backed NCCL buffers, the prerequisite for DMA-BUF GPUDirect RDMA (run.sh sets 0) | **rejected (k3 comm fix, two ranks on one GB10, NET/IB loopback, serve NCCL env)**: GDR stays 0. NCCL logs `cuMemGdrSupport 0`, and the image's libmlx5 has no `mlx5dv_reg_dmabuf_mr`. cuMem alone costs time: AR m=4 eager 35.6 → 50.2 us, back to back in a graph 29.7 → 34.5, after a 2x-L2 read 30.4 → 37.8; m=8 62.3 → 72.7 / 51.9 → 54.0 / 54.0 → 60.6 (rank 0 medians) | `results/2026-09-25-kernels/comm/c-cost-gdr/` |
| `DSV41_MHC_NO_DEEPGEMM=1` | TileLang GEMM for MHC prenorm | rejected | `evidence/mhc-tilelang-gemm` |
| `DSV41_DSPARK_DRAFT_TOPK=k` | Topk-mask draft logits | rejected | `evidence/dspark-draft-topk` |
| `DSV41_DSPARK_SPARSE_MARKOV=1` (+`_TOPK`, default 256) | Gathered top-k Markov bias via the speculator's `_sample_sequential_topk` (drops the 3 x 66 MB Markov GEMM; argmax stays); refuses MARKOV_SCALE != 1, CONF_GATE, DRAFT_TOPK | **default on since R34; `_TOPK=1024` default since R35** (Container env table) | `kernel_study/decode_levers` |
| `DSV41_WOA_PREPACK=1` | Pack the fp32 wo_a scale to UE8M0 once (bitwise self-test, else fp32 stays); needs an image with `fix_o_proj_woa_fp8.py` stage 2 (`canonical-e13`) | **default on since R34** (Container env table) | `kernel_study/decode_levers` |
| `DSV41_DSPARK_TAIL_NGRAM=1` (+`_POS`) | Prompt-lookup tail on drafts | rejected | `evidence/dspark-tail-ngram`, `evidence/ngram-overlay` |
| `DSV41_DSPARK_SOFTMAX_VERIFY=1` | Greedy propose, softmax verify | rejected | `evidence/dspark-softmax-verify` |
| `DSV41_DSPARK_REFINE_PASS=1` | Second draft pass on pass-1 fills | rejected | `evidence/dspark-refine-pass` |
| `DSV41_DSPARK_CONF_GATE=1` | Confidence-gated Markov bias | rejected | `evidence/dspark-conf-gate` |
| `DSV41_MLA_IO_WARPS=2` | 2 IO warps in DSV4 MLA decode | rejected (races mbarriers at 4) | `evidence/mla-io2` |
| `DSV41_MLA_CHUNKS_PER_BLOCK=k` | Bake MLA chunks_per_block | rejected | `evidence/mla-chunks-per-block` |
| `DSV41_DENSE_DG_SMALLM=1` (+`DSV41_DENSE_DG_SHAPES=KxN,...`) | Dense MXFP8 GEMMs at M<=8 on deep_gemm `fp8_gemm_nt` (recipe (1,1,32), scales packed once at load; heuristic split-K unconfirmed until `DG_PRINT_CONFIGS=1`) instead of b12x; per-layer load self-test (M=1..8 vs b12x, graph capture checked at M=4 once per shape), falls back to b12x. wq_b is not wireable (fused q/kv RMSNorm+quant hands it a QuantizedActivation) | **rejected (R34 s1)**: deep_gemm is slower than b12x on every shape at decode M: qkv_a −3.8 %, wq_b −7.1 %, wo_b −12.0 %, shared_gate_up −3.1 %, shared_down −9.4 %, main_proj −9.2 %; split-K sk2/sk4 slower too. No shape reached the 10 % serve-arm bar, so `DSV41_DENSE_DG_SHAPES` stays empty and there was no serve arm | `docker/patch/dense_mxfp8_deepgemm.py`, `results/2026-09-24-review/campaign/s1-microbench/` |
| `DSV41_DENSE_GEMV=1` (+`DSV41_DENSE_GEMV_SHAPES=KxN,...,wo_a`, empty = every tuned shape plus `wo_a`) | Purpose-made M<=8 MXFP8 decode GEMV for sm_121a (`docker/patch/dense_gemv_kernel.cu`, nvcc JIT on first use). It replaces b12x for qkv_a, wq_b, wo_b, shared gate_up/down, Engram wkv, the draft main_proj (M<=4) and the mxfp8 lm_head, and o_proj's wo_a `fp8_einsum`. Bit-exact by construction: FlashInfer's activation-quant PTX and the same block-scaled m16n8k32 MMA chained over k (swap-AB, no split-K). Self-tests: each layer at load (M=1..8 x 3 distributions bitwise vs b12x, graph replay once per shape); the pre-quantized wq_b input once vs b12x. wo_a arms at a layer's first eager decode-size call (vLLM runs one eager warmup per capture size), with a self-test vs `fp8_einsum` once and an in-situ check per layer. Any failure keeps the stock path and logs `; b12x stays` / `; fp8_einsum stays` (a rejected launch's CUDA error is cleared first, so the fallback survives). Shapes with different K share a kernel instantiation and the serve arms every layer before the first call, so the dynamic-smem attribute is only ever raised. The dense shapes stand down, logged, when `DSV41_DENSE_DG_SMALLM=1` claims the same call | **default on since R35 (k3 serve campaign, bundle K)**: K vs A, 2+2 boots interleaved A, K, KC, A, K, KC: four prose c=1 −15.774 ms/step (noise 1.089), prose_long c=1 −14.487 (noise 0.308), acceptance matched, every K boot below the A median; quality quick PASS on K-1/K-2/KC-1/KC-2, full PASS on K-2 (`results/2026-09-26-serve-r3/decision-bundle.json`). In-serve attribution, K_p vs A_p, c=1, rank 0 / rank 1 (`results/2026-09-26-serve-r3/P3/attribution.txt`): dense MXFP8 GEMM kernel time −1.690 / −1.683 ms/step (projection −1.60) plus −0.46..−0.51 of activation-quant kernels; the router gate next to the shared gate_up GEMV slows from 19.6 to 47.0 us per layer, +1.096 / +1.136 ms/step back on the FFN-prologue path (P3 top-3 item 3). Before the serve campaign: **microbench KEEP, serve arm pending (k3 dense-gemv, review fix pass done)**: the review found that with the flag on the serve crashed at its first decode-size forward: arming a smaller-K shape lowered a shared instantiation's dynamic-smem attribute, so after load-order arming the Engram, qkv_a and main_proj launches failed. Fixed; `kernel_study/dense_gemv/serve_hook_test.py` now arms 21 layers in the image's load order before any call: 325/325 bitwise (the old kernel failed 38). Cold medians at M=4 from the review's fixed-config graph chain (production configs, no best-of), stock → GEMV: qkv_a 45.98 → 41.57 µs (+10.6 %), wq_b 97.33 → 92.25 (+5.5 %), wo_b 100.96 → 91.91 (+9.8 %), gate_up 57.28 → 52.22 (+9.7 %), down 29.98 → 27.64 (+8.5 %), Engram wkv 738.1 → 680.7 (+8.4 %), main_proj 373.4 → 338.5 (+10.3 %), lm_head 1516.7 → 1455.7 (+4.2 %), wo_a 79.33 → 74.39 (+6.6 %). That is 86-94 % of 250 GB/s, and all 78 cells are bitwise. The serve-like 8-layer chain went 351.0 → 325.5 µs/layer, bitwise equal. Modeled saving ≈1.60 ms/step at c=1 (2.5 %); correcting the shared expert for the concurrent router GEMM (7.36 µs/layer) leaves it unchanged. It stacks with `DSV41_AR_L2_PREFETCH=1`: with 61 % of qkv_a prefetched, qkv_a goes 24.5 → 20.4-22.5 µs at every M. With fully L2-resident weights (not the serve case) it loses at qkv_a M=8, gate_up M≥6 and wo_a M≥3. Only main_proj clears the 10 % per-shape bar; the case for a serve arm is the sum. The indexer wq_b (K1280xN4096) gained nothing and is not wired | `docker/patch/dense_gemv.py`, `kernel_study/dense_gemv/`, `results/2026-09-25-kernels/dense-gemv/iterations.txt` |
| p2b kernel variants (`widen_p2b_mma.py`, `attic/widen_p2b_{fma,cp16,cpasync,ldg,pf4,nocoop}.py`) | Alternate p2b kernels | all rejected or reverted; mma→mrow regression 13.24. `widen_p2b_mrow.py` is live (docker/Dockerfile) | `evidence/p2b-*`, `evidence/mma-revert` |
| `widen_mla_{tile32,io2}.py`, `attic/{widen_mla_kv_buf,sm120_wo_a,c1_graph_safe_adaptive}.py` | MLA/KV/graph variants | unwired (failed L.A.I.L or no win). Index of `docker/patch/attic/`: its README | `evidence/mla-*`, `evidence/c1-*`, `evidence/sm120-wo-a*` |

## Bench/eval settings (frozen for comparability)

- Decode prose: `bench_decode.py --phase prose --concurrency 1 --runs 3 --max-tokens 200`, greedy, thinking off.
- L.A.I.L prose: `tools/measure_lail_prose.py` — 512 tokens, temperature 0.2, c=1.
- Correctness: `tests/correctness.sh [--full]` — temperature 0, seed 20260918, thinking off, effort low.
- Micro: `benches/micro.sh` — fresh docs per invocation (prefix-cache-proof), pp = prompt_tokens/TTFT, tg32 post-first-token.
- E2E: `benches/e2e.sh` — coding-agent turn, 64k doc recall+summary, tool/JSON, then L.A.I.L prose.
- `reasoning_effort` and `thinking` stay locked in every harness; never bench with the default-on thinking.

## Round 5 — decode campaign axes (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| p2b prefetch ring depth (PF 2/4/8) | **REJECT** | e=30 warm: 672 → 727/761/1359 us; register pressure |
| Full codebook LUT in smem | **CLOSED (hw)** | 128 KiB table > 99 KiB smem/block opt-in on sm_121 |
| vdec1 funnelshift extraction | **KEEP-candidate** | −6.1% bit-exact warm (670.7→629.7 us), 0 cold; ship with next kernel rebuild |
| vdec2 warp-smem staging | reject-ish | −2.5% warm, +3% cold |
| MUL1 pack as decode lever | **CLOSED** | same window format; −16% prose was acceptance-driven; kernel-time parity at best |
| decode-instruction rewrite as main lever | **DEPRIORITIZED** | cold-L2 serving case is memory-stream-bound at 74% UMA peak; ceiling ≤1.35x on 35% of step |

New open axes (from live profile): dense-projection stream efficiency
(b12x grids starve 48 SMs; o_proj wo_a on sm_80 WMMA), NCCL 2-rank AR
latency. See results/RESULTS.md round 5.

## E10 (2026-09-19)

| change | verdict | evidence |
|---|---|---|
| widen_p2b_fshift (funnelshift window merge) | **KEEP** | bit-exact; -6.1% p2b warm; correctness 8/8 |
| widen_b12x_smalls ((16,64) tiles m<=8 n<=8192) | **KEEP (harmless, no effect)** | prefill flat 754/700; gates pass. The same-session prose 25.1->31.4 is not this patch: the R7 trace shows NO EFFECT (dense GEMM total 990→1038 ms) |
| probe_wo_a (diagnostic) | keep | wo_a runtime dtype = bf16 (pack F8) -> sm_80 WMMA bmm: next target |
| NCCL_PROTO forcing | **CLOSED** | isolated 2-rank AR: default=LL 43.0us; LL128 81.8us |

## E11 (2026-09-19)

| change | verdict | evidence |
|---|---|---|
| fix_o_proj_woa_fp8 (exact requant, self-guarded) | **KEEP** | 43/43 layers engaged; prose 31.4->33.6; 8/8 + 3/3; pp band intact |
| wo_a bf16-bmm (emulation dequant) | root-caused | MXFP8 emulation kernel dequants at load; scales retained -> exact roundtrip |

## Round 7 axes (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| p2b cp.async 4-buffer ring (DEC3, bit-exact) | **REJECT** | cold e=30: 672.4 us vs stock 642.2 (−4.7%); kernel not load-starved |
| p2b load-mechanics axis | **CLOSED** | PF depth, smem staging, cp.async all rejected; kernel at ~76% UMA peak cold — further gains need pack-layout work |
| b12x (16,64) small-m tiles | **NO EFFECT** (kept, harmless) | trace-flat: dense GEMM total 990→1038 ms over same steps; those GEMMs are latency-bound, not parallelism-starved |
| wo_a fp8 einsum | **VERIFIED IN-TRACE** | WMMA 288 us x40/step gone; deep_gemm einsum 80.6 us x40/step |

## Round 8 (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| flashinfer autotune (serve flag) | **NO EFFECT / stays off** | prose 30.2 vs 31.2, L.A.I.L 22.3 vs 21.9 — bands; GEMMs latency-bound |
| async-TP / comm overlap | **NOT AVAILABLE** in build | no flag/config; structural patch required (hand-off) |

## Round 9 (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| WNT=8 wide tile (CFG=2) | **REJECT** | 1798 vs 655 us cold — registers kill occupancy |
| group-major trellis layout | **CONFIRMED +6% kernel, not yet integrated** | bit-exact; cold 627.4 vs 667.6 us; needs loader permute + all readers re-indexed (prefill risk) |
| pair-codebook requant | **DEPRIORITIZED→closed** | decode is memory-bound cold; instruction cuts provably don't move it |

## Round 10 (2026-09-19)

| axis | verdict | evidence |
|---|---|---|
| N-split warp decomposition (DEC6) | **REJECT** | 758.0 vs 643.3 us cold (−18%); K-split latency spreading wins |
| p2b variant space | **EXHAUSTED** | K-split+stock layout = local optimum; group-major permute is the sole remaining lever (+6%, needs prefill harness) |

## Round 14 — Engram prefetch v2 (2026-09-20): NO-GO, REVERTED

v2 patch (per-table immediate publish, gen-stamped union, exact fadvise spans,
paired census) wired at `befa570` (dormant while PREFETCH=0; also fixes the
run.sh:475 worker omission of PF_DUMP + adds PREFETCH_DEBUG fwd on both ranks).
Booted with PREFETCH=1 on the nccl-set baseline. Pairing self-check worked as
designed and pinpointed the failure: `[pf-pub]` lands before `[pf-pair]` every
gen (publish timing FIXED — v1's publish-too-late bug is dead), yet
table_pred_id ≠ consumed_id on ~88% of 80 pairs, intersect ≈0 (sum 88/72 rows
over 40 pairs/rank); census pf_hit ~0% (98/105 windows 0%), read_w already
0.06–0.10 ms cold. L.A.I.L 25.23 cold / 26.79 warm / job 26.05 = parity, not
≥28. Doc's "pairing still wrong" branch — but the divergence is now on the
PREDICTION side (rows hashed ≠ rows the next gather reads), not delivery;
direct-pread next lever NOT indicated. Serve reverted to the exact NCCL-arm
config (boot-arm.sh), wiring commit stays dormant. Evidence:
results/2026-09-20-engrampf/ (VERDICT.md, pf-evidence-spark{1,2}.log).

## Round 12 — NCCL AR-tail set (2026-09-20)

| axis | verdict | evidence |
|---|---|---|
| NCCL_BUFFSIZE=1M + LL128_BUFFSIZE=256K + PROTO='^LL128' + MAX_NCHANNELS=8 (on mem-hygiene baseline) | **KEEP** | prose flat (34.69 vs 34.67 re-run; first pass 31.59 = cold-cache artifact), prefill32k +2.1% (733.0), L.A.I.L 26.55/26.12 both above 25.2–26.3 band, MemAvail +3.5/+2.8 GiB/rank (pinned-buffer shrink 4.7→0.14 GiB realized), zero NCCL WARN lines both nodes. results/2026-09-20-nccl/ |

## Round 13 — mul1 lane A/B on 2x Spark (2026-09-20)

Mia-AiLab 2.9bpw pack on her stock GHCR image, our topology (spark2 CX7=f1/GID1,
NFS weights+engram over QSFP), k=3 dspark vs spec off:

| arm | prose c=1 | prefill 8k / 32k | LAIL t0.2 | MemAvail s1/s2 |
|---|---|---|---|---|
| k=3 dspark | 16.35 (acc 1.32, draft 0.108) | 593.7 / 711.7 | 25.63 | 5.3 / 6.7 |
| spec off | **23.64** | 587.9 / **845.6** | 23.46 | 8.8 / 10.0 |
| MCG lane ref | 34.69 | 690 / 733 | 26.39 | 22.3 / 23.8 |

Verdicts: pack+kernels healthy (spec-off matches her ~23; mul1 32k prefill
+15% over MCG). k=3 drafting actively harmful in our boots — draft/verify
pathology (her draft-wo-a-slices patch skipped). Mul1 lane parked at
SPEC_METHOD=none until a draft fix; coop Boot B deferred (its 40.2 ceiling
assumes healthy acceptance). Evidence: results/2026-09-20-mul1-lane/.

### Round 13a — mul1 A3: draft_sample_method=greedy (REFUTED)

Forcing greedy drafting moved greedy-verify acceptance 1.32 → 1.36 (noise);
prose 16.80. Sampling method is not the pathology. Signature stands: drafter
agrees with sampled decoding (2.08 @ t=0.2) but is systematically wrong under
greedy verify. Lane parked at SPEC_METHOD=none; Boot B (coop) blocked until a
draft fix. Evidence: results/2026-09-20-mul1-lane/BOOT-A3-VERDICT.md.

### Round 13b — mul1 A4: LANGUAGE_MODEL_ONLY=1 text-only arm — lane PARKED PERMANENTLY

Text-only server (her documented measurement condition) moved greedy prose by
+0.1 tok/s (16.35→16.45) and acceptance 1.32→1.33 — identical, within noise.
Her exact 400-token budget: 20.30 tok/s @ acceptance 1.69 (acceptance grows
with generation length, not with text-only). Her 28 tok/s k=3 is not
reproducible on this pack under any condition tested (A/A3/A4 × vision on/off
× 200/400 tok × default/greedy draft sampling). Root cause confirmed as pack
property: 4-bit EXL3 MTP drafter (`mtp_bits: 4`) — MCG's source-precision
drafter shows 2.77/2.87 with the same harness. Decision rule (prose ≥24 AND
acc ≥2.0) FAILED on both. mul1 lane PARKED PERMANENTLY at SPEC_METHOD=none
(23.64 / 845.6 @32k); Boot B stays blocked. Re-open mechanism: requantize the
pack with a source-precision drafter (`mtp_experts: source`, MCG-style).
A4 side-observations: text-only k=3 floors 6.2/7.7 GiB; micro tg 8k/32k
36.8/33.9 (acc 2.89/2.92 — best of the lane); LAIL t=0.2 26.79 (parity with
MCG lane). MCG restored + verified; LAIL real job c33b06334e36 = 25.34.
Evidence: results/2026-09-20-mul1-lane/BOOT-A4-VERDICT.md.

### Round 13c — p2b group-major prefill gate: FAIL (bit-exactness)

PREFILL-GATE.md harness (commit eb78a8b) run in the GPU window per spec:
`--mode check` MISMATCH at ALL 20 (matrix, m∈{64,128,256,512}, shape) combos;
control proves both layouts individually deterministic — the PF-G8 B-load
remap genuinely computes wrong outputs (max abs diff 415–678, ~99.998% of
elements differ). Advisory bench (wrong-result timing): G8 within ±1% of
stock nearly everywhere; layer estimate −0.1%..−2.4%. Verdict FAIL — do NOT
fold PF-G8 into quantize_experts_exl3.py; keep p2b stock layout. Lever
re-openable only via harness fix + fresh GPU gate re-run (the harness was
CPU-validated for compile only, never numerically on GPU). Evidence:
kernel_study/gemv_bench/PREFILL-GATE-RESULT-2026-09-20.log.

### Round 15 — MCG spec-k sweep: k=3 WINS the L.A.I.L cell (single-lever, same-day n=3)

First sweep of DSpark `num_speculative_tokens` on the MCG lane (k=5 was
inherited; %5 guard bypassed with FORCE_UNSAFE_CTX=1 — k=10 already rejected
in E6). Same day, same image/env, only k varied; L.A.I.L app bench n=3 per arm
+ CLI twin (512 tok, t=0.2) + greedy prose 9×:
k3 27.27 / 28.57 @ acc 2.25 / prose 35.55 @ 2.62 — k5 26.32 / 25.57 @ 2.30 /
33.03 @ 2.90 — k4 25.87 / 25.86 @ 2.18. k3 wins all three cells (+3.6% app,
+11.7% twin, +7.6% prose); k4 worst (deeper than t=0.2 acceptance justifies
AND pads verify 5→6); k7 skipped (flat acc-vs-depth + E6 collapse at k=10).
Mechanism: at t=0.2 acc≈2.3, draft slots 4–5 are dead weight; k3 verifies
4 tok/step and still pads 4→5 on the k5-era capture sizes [1,5,6,10,12] —
matched sizes [1,3,4,6,8] is the queued follow-up. Serve left UP at k=3
(boot-k3.sh in results/2026-09-20-ksweep/, smoke 323 ✓, confirm job
65933ed0ddd7 = 27.0). L.A.I.L job ids in VERDICT.md; contaminated job
1b2fa6c0fc52 discarded (ran concurrently with its warmup — app bench queue
bug in my driver script, fixed). Gap to 35: 7.7 tok/s. Evidence:
results/2026-09-20-ksweep/VERDICT.md.

### Round 16 — capture-size match KEEP (+5.5%), engram prefetch v3 NO-GO

Two single-lever arms on the k=3 serve (Round 15). ARM 1: cudagraph capture
sizes [1,5,6,10,12] → [1,3,4,6,8] via `COMPILATION_CONFIG` env (k3 verifies
4 tok/step; 4 padded to 5 before). Verified in EngineCore
compilation_config; boot ~12 min. L.A.I.L n=3 median **28.76** (26.20/28.76/
28.94) vs 27.27 = +5.5% ≥ +3% gate → **KEEP**, now the serve
(`results/2026-09-21-capture-pf/boot-k3c.sh`). Quick prose 34.89. ARM 2:
engram prefetch v3 (commit e70c4b1, post-propose publish + bonus+draft
anchored prediction) on the ARM-1 config — [pf-pub]/[pf-pair] on both ranks
(40 pair-rows/rank, |pred|=48 not the hoped 72) but intersect recall still
short: mean 0.325/0.292 (s1/s2), ≥0.5× gate only 20%/15%, L.A.I.L 28.69
(−0.2% vs ARM 1) → **NO-GO**, serve reverted to PREFETCH=0 (patch committed,
inactive by default). ARM 3 skipped (no flag cleared the bar; window spent
on restore). Final serve smoke 323 ✓, MemAvail 25/26 GiB, all boots
117/117 pre / ≥25 GiB post, zero abort lines. Gap to 35: 6.24. Evidence:
results/2026-09-21-capture-pf/VERDICT.md.

### Round 17 — draftk2: sampled drafter + k=2 BOTH REVERT; best stays 28.76

Two arms on the k3c serve. ARM A: `draft_sample_method=probabilistic` via
SPEC_CONFIG override (boot-k3c-dsampled.sh). NOTE: the original plan said
"remove draft_sample_method" for engine-default sampling — that is a provable
no-op (vLLM default IS "greedy", config/speculative.py:588; removing the field
reproduces the pinned value), so the arm was run as "probabilistic", which is
what actually implements the sampled-drafter theory (allocates draft_logits,
gumbel-samples drafts, full-logit rejection ratios). Verified in engine
non-default args. L.A.I.L n=3 median 28.50 (28.80/28.50/26.92) = −0.9% vs
28.76, gate ≥29.62 failed → REVERT. Sampled drafter is functionally healthy
(CLI t=0 acc 3.28, draft_accept 0.76 — no collapse, unlike mul1's 4-bit
drafter) but doesn't convert to L.A.I.L speed at t=0.2. ARM B: k=2 with
captures resized to the k2 formula [1,2,3,4,6] (boot-k2c.sh; verify batch 3).
L.A.I.L n=3 median 26.89 (26.08/27.57/26.89) = −6.5% → REVERT; theory refuted
— acc 2.57 nearly saturates k2's max 3.0, but the cheaper verify does not pay
for the lost per-step accepted tokens at c=1. k-lever now swept end-to-end:
k5 26.32 / k4 25.87 / **k3 28.76** / k2 26.89 — k=3 is the optimum. Both arms
reverted → abort-early rule: ARM C skipped, serve restored byte-identical to
boot-k3c.sh (engine log re-verified k3/[1,3,4,6,8]/greedy, smoke 323 ✓,
confirm job a132c12230a2 = 28.69 ≈ baseline). Gap to 35: still 6.24 (82.2%).
Evidence: results/2026-09-21-draftk2/VERDICT.md.

### Round 18 — trace attribution: gap = ONE engram hard-sync; SOFTMAX_VERIFY REVERT

BOOT 1 (trace on stock k3c + --profiler-config torch replica): ONE L.A.I.L
prose request profiled (20.57 s window, 5.02M events). Pure GPU idle
~14.2 ms/step; **99.9% owned by a single stack**: EngramDiskStager.stage →
hashes_ready.synchronize() — a hard cudaEventSynchronize in prepare_inputs
waiting on a tiny D2H hash copy queued behind the previous step's graph
(engram.py image lines 1302-1339). Classes #2/#4/#5 closed (<0.15% of
idle). Winning patch class (named, NOT implemented): **#1 host critical
path — CPU-side hash** (verified CPU port of _hash_ids_kernel exists in
the prefetch-v3 work) → delete the sync; expected ≈ full 13-14 ms/step
pool ⇒ ~34.5-37 tok/s at acc 2.26. Profiler method pinned: the torch
profiler mounts /start_profile//stop_profile ONLY when --profiler-config
profiler=torch is passed at serve (openapi-verified); ignore_frontend=true
crashes start_profile (AsyncLLM.profiler AttributeError) — do not use.
BOOT 2 (SOFTMAX_VERIFY=1, wrap engaged at boot, knob verified boot-fixed):
L.A.I.L warmup + n=3 = 27.14/26.46/28.56 → median 27.14 (−5.6% vs 28.76,
gate ≥29.62 failed); repo harness acc_len 2.20 (vs 2.26 stock) — the k5
acceptance gain does not transfer to k3. **REVERT** (default 0 stays).
BOOT 3: k3c restored and left UP (smoke 323 ✓, MemAvail 25/27, best 28.76).
All boots 116/117 pre, ≥25 post, zero aborts; trace export dropped s1
MemAvail to 3 GiB once → serve stopped before parsing, restored 117/117
(lesson: no big host work while serve resident after profiler export).
Gap to 35: 6.24 — the CPU-hash patch spans it alone. Evidence:
results/2026-09-21-trace-attribution/{ATTRIBUTION.md,VERDICT.md}; trace
kept locally (gitignored). New parsers: parse_gaps{,_stacks,_memcpy}.py.

### Round 19 — CPU-hash REVERT (prediction rule wrong, mirror proven bit-exact)

Built the Round-18 fix as DSV41_ENGRAM_CPU_HASH=1 (commits 03c59e1, ea8bfe8,
dormant default-off): post-propose snapshot → off-thread reconstruct next
chunk + numpy mirror of _hash_ids_kernel + pread/dequant; stage() = join+H2D
only (no GPU hash, no D2H, no hashes_ready.synchronize on the main thread).
PREMISE CORRECTION: input_ids is a GPU buffer (combine_sampled_and_draft_
tokens ← GPU-only last_sampled/draft) — "ids host-resident" is false; the
fix must predict. Offline: 300 fuzz cases bit-exact vs independent scalar
kernel transcription; chain-apply + real-image dry-run clean; 225 tests OK.
BOOT 1: armed both ranks, self-disarmed at warmup (single-stage check).
Disarmed parity 28.87 median ≈ 28.76 (fallback free). BOOT 2 (one retry,
two-stage warmup): stage A MIRROR PASSED LIVE (CPU hash of ACTUAL ids
bit-exact vs GPU kernel); stage B PREDICT FAILED — bonus row (row 0)
matches, draft rows 1-3 differ (positions match, ids differ): the next
step's draft ids are NOT propose()'s return. This also explains prefetch-v3's
~33% intersect (bonus-only rows). → REVERT; patch stays dormant in repo.
Restore k3c UP (smoke 323 ✓, confirm median 27.40, boot spread). Gap to 35:
6.24. NEXT LEVER: find where draft ids change post-propose (set_draft_tokens
writes req_states.draft_tokens — next combine may read that, not the
speculator buffer); snapshot at the commit hook instead. Mirror is ready.
Evidence: results/2026-09-21-cpuhash/VERDICT.md.

### Round 20 — CPU-hash stream-ordering fix: ENGAGED end-to-end, REVERT on L.A.I.L

Root cause of the Round-19 PREDICT failure found by source audit (no
diagnostic boot needed): OUR OWN patch called
side.wait_stream(torch.cuda.current_stream()) INSIDE
'with torch.cuda.stream(side):' — a self-wait NO-OP — in all 3 cpu-hash
side-stream blocks and prefetch-v3's enqueue. The pinned D2H snapshots were
unordered vs the main stream and raced the dspark draft CUDA-graph replay
(writes speculator.draft_tokens, dflash/speculator.py:484): bonus row fresh
(sampler lands earlier), draft rows = previous generation. Refuted the
Round-19 candidate from source: model_runner.py:1980 scatters propose()'s
return into req_states.draft_tokens (same tensor the hook reads), combine
(input_batch.py:445) reads req_states verbatim, set_draft_tokens is
structured-output-only (draft_tokens_np=None otherwise); commit hook
(line 1690) runs BEFORE prepare_inputs→stage() (1733) so commit-time
snapshot would be TOO LATE — enqueue point was right, ordering was wrong.
Fix (commit 35e05fd): hoist wait_stream(main) before the with-block.
BOOT (2 of 3 cap used): ENGAGED BOTH RANKS — "cpu-hash ACTIVE (mirror+
predict bit-exact x4; D2H+event sync removed)", fast-path steps=500+,
zero canary hits. The Round-19 rule was always correct.
L.A.I.L n=5 + independent n=5: medians 28.19 / 27.22, all-10 27.46 vs
k3c 28.76 → sync removal does NOT convert to wall time (off-thread
pread/dequant gather + per-step canary hash ≈ the sync cost). REVERT the
env lever; code fix stays dormant (makes a future PREFETCH=1 re-arm
testable — its ordering bug is also fixed). Restore k3c UP: smoke 323 ✓,
confirm 30.04/29.37/29.38 median 29.38. Gap to 35: 6.24 — pool attribution
stands but "delete the sync via CPU-side prediction" is now MEASURED
END-TO-END and does not recover it; next lever must attack the pool
differently (overlap/defer the gather, not re-predict the hashes).
Evidence: results/2026-09-21-cpuhash2/VERDICT.md.

## Round 21 — 2026-09-21: prefetch v3 re-arm (ordering-fixed) → KEEP (+4.6%)

Single lever: DSV41_ENGRAM_PREFETCH=1 on k3c (v3 code, Round-20 ordering
fix 35e05fd); CENSUS=1 for evidence, PF_DUMP/DEBUG off. Round-16 NO-GO
SUPERSEDED: pf_hit 100% both ranks entire run (was <50%, pair ratio
0.29–0.33) — the stale-id race was the whole failure, overlap thesis was
right. read_w 4.0ms warmup → 0.05–0.07ms steady both ranks. L.A.I.L
c=1 prose: batch medians 29.32 / 30.99 (independent n=5 each), pooled
n=10 = 30.10 vs 28.76 = +4.6% (KEEP gate ≥+3%). 35 NOT claimed. Serve
left UP on boot-k3c-pf.sh (prefetch+census ON) — new best config.
Gap to 35: 4.90 — residual gather (dequant/H2D serialization) + acceptance.
Evidence: results/2026-09-21-pfrearm/VERDICT.md.

## Round 22 — 2026-09-21: re-trace of the ENGAGED 30.10 serve — idle moved INSIDE stage, post-sync

Re-ran Round-18 trace attribution on the live best (k3c + PREFETCH=1 +
CENSUS=1 + torch profiler; results/2026-09-21-trace2/). Window 17.9 s ≈ 238
steps: device-busy 65.3 ms/step, pure idle ~9.9 ms/step (was 14.2). STILL
99.9% one stack: engram.py stage <- prepare_inputs (241/241 gaps ≥2 ms).
NEW: parse_gap_split.py shows the CPU is blocked inside
hashes_ready.synchronize() for 0.7 ms TOTAL — the event fires promptly now.
1802 ms (7.5 ms/step) elapses AFTER the sync returns: host EXECUTION of the
stage gather loop (51 rows × pread+dequant+H2D staging). Round 18's
wait-behind-work is dead; what's left is Python/host loop time. No env
lever attacks it (CPU_HASH=1 already measured end-to-end → Round-20 REVERT;
read_w 0.1 ms proves data is in page cache). NO A/B per gate; 3 boots spent
(1 lost to cp-after-stop — trace was in container tmpfs; protocol note
added). Best serve restored UP on boot-k3c-pf.sh (smoke 323 ✓, L.A.I.L
confirm 32.49/30.51/33.85). Gap to 35: 4.90 — next lever is CODE-level
(batch pread→preadv, vectorize dequant, single pinned H2D, or off-thread
gather for N+1 with event-wait at replay), not env.

## Round 23 — 2026-09-21: engram gather v2 (preadv run batching) — KEEP, new best 31.37

Round-22's code-level lever, implemented: DSV41_ENGRAM_GATHER_V2=1 (default
off, docker/patch/engram_gather_v2.py, commit 5507311). One preadv per
contiguous row run (scatter iovecs, dup-row read-once+memcpy, row-aligned
partial resume), dequant VERBATIM from stock (bit-exact by construction +
live self-check compares raw bf16 BITS — float equal lies on fp8 NaN
bytes), H2D unchanged (one pinned copy per table per step, inside the
round-11 parallel stage workers). Any error → one DISABLED line → stock
path, never crashes the serve.

Key finding (pre-implementation probe, spark1, hot cache, real 256B/8B
geometry, R=51): the stock chunk-pool _read_rows costs 0.64 ms/call of
which pool submit/result overhead alone is 0.105 ms; the SAME 102
per-row preads INLINE cost 0.05 ms. The pool dispatch was the loop cost,
not the syscalls (hash rows are uniform over 384M rows — contiguous runs
are length 1, so syscall count barely drops; the win is removing the
pool hop + per-call alloc/dequant chaining).

Offline validation ALL PASS (validate_gather_v2.py in e12 image): chain
apply, bit-exact vs stock on real-geometry safetensors packs (fuzz
R=51/300, contig+dups, all-unowned, single), partial-preadv resume
(caught + fixed a REAL resume bug: row_bytes-based skip corrupted
follower rows after the first partial — now consumes n bytes off the iov
front), forced-failure fallback, micro-bench stock 0.82-0.85 vs v2
0.12-0.19 ms/call = 4.2-6.9x. Gates: bash -n, py_compile, render --check,
225 unittests OK.

Boot 1 of cap 3: install line + self-check bit-exact (r=120) BOTH ranks,
smoke 323, steady [gv2-census] rows/call=48 runs/call=96=preads/call=96
BOTH ranks (one preadv/run, zero pool futures), zero DISABLED lines.
L.A.I.L: b1 median 32.11, b2 (independent) 30.63, pooled n=10 median
31.37 vs 30.10 = +4.2% → KEEP (gate +3%). 35 NOT crossed (two-medians
rule: 32.11/30.63). Repo prose 9-run median 36.80 agg (acc 2.55), L.A.I.L
prose 512tok 30.71 (acc 2.12). Cold prefill 32k warm runs 713-786;
MemAvail post-32k 21/23 GiB. Serve left UP on
results/2026-09-21-gatherv2/boot-k3c-pf-gv2.sh. Gap to 35: 3.63 —
remaining idle is the ~2.4 ms sub-1-ms fragments + residual stage path;
35 still needs acceptance or device-side gains.

## Round 24 — 2026-09-21: engram stage defer (off-thread next-step gather) — REVERT env, code dormant

Round-18 option 2 implemented (safe since 35e05fd fixed side-stream
ordering): DSV41_ENGRAM_DEFER=1 (default off, docker/patch/engram_defer.py,
commits ea871aa/b2c8ad0/dc548ef). ONE persistent worker per rank predicts
the next chunk (bit-exact v3 rule), CPU-hashes (cpu-hash numpy mirror),
gathers all tables via v2 preadv into double-buffered pinned slots, H2Ds
on a side stream (wait_stream BEFORE the with-block), and stage()
consumes with ONE DtoD per table at replay — zero gather work on the
critical path; ANY anomaly disarms to the sync v2 path with one line
(bounded 2s join, queried events, 4-step warm bit-verify on live data).

Boot-2 finding (fixed dc548ef): adaptive-verification micro-steps call
the post-propose hook with MALFORMED snapshots (sum(qsl[1:]) !=
num_tokens) — enqueueing them churned the generation counter and
zeroed the hit rate SILENTLY (census only printed on hits). Now: enqueue
validates qsl; census prints on attempt cadence; seen-gen guard.

Live (boot 3/3): ACTIVE warm-verified x4, defer-hit 65.6-100% (typical
~88) both ranks, late=0 always, worker 4.6-7.7 ms/pred vs ~72 ms step,
zero DISABLED. Smoke 323. L.A.I.L b1 median 30.28 (28.73-30.55) vs
31.37 baseline — flat (inside the 29.5-33.7 band), gate ≥32.3 FAIL →
REVERT env lever per the engaged-but-flat rule.

CONCLUSION (the round's real result): with the stage() gather fully off
the critical path AND verified ~88-100% consumed, tok/s does not move —
**the stage() host gather was NOT the remaining wall.** The Round-22
residual idle attribution must be re-examined: remaining idle lives in
the ~2.4 ms sub-1-ms fragments + step/scheduler overheads + kernel
launch tails, not the gather. 35 needs acceptance or device-side gains.
Code stays dormant; serve restored UP on boot-k3c-pf-gv2.sh (Round-23
best 31.37).

## Round 25 — 2026-09-21: arbitration trace (trace3) — DEVICE-BOUND verdict

Trace of the 31.37 best config (k3c + PREFETCH + CENSUS + GATHER_V2) with the
Round-22 protocol (one L.A.I.L prose request, cp-before-stop, idle-host parse).
S=249 steps (SSE chunks; acc 2.056 in-window). Results/2026-09-21-trace3/.

- Device-busy (kernel-timestamp union) 65.25 ms/step; profiled wall 72.40;
  live-implied unprofiled step 65.5-69.3 ⇒ profiler inflation ~3-5 ms (~5%),
  all on the CPU launch path. Kernel composition/step: p2b MoE 22.4, dense
  GEMM 17.6, AR 4.6, draft/aux ~13, prefill residue ~2.6.
- Idle NOW 7.15 ms/step total: >=0.5ms pool 4.87 (73x5-10ms + 173x2-5ms,
  median 4.69) + sub-0.5ms dust 2.29. The R22 7.5 ms main-thread gather
  owner is GONE (sync-blocked 0.7 ms/window; gather now in the 2 parallel
  stage workers). Largest coherent residual: 191 posix_fadvise(WILLNEED)/step
  on the OFF-thread prefetch pool worker (~3.1 ms of gap time; pf_hit 100%
  ⇒ pure waste) — code-gated (cap the fan-out), NOT env-gated.
- ARBITRATION: 35 @ acc 2.3 needs step <=65.7 ms. Device-busy alone 65.25.
  Perfect recovery of all 7.15 ms idle lands exactly at the bar with zero
  margin, and no single env-reachable owner >=4 ms exists ⇒ device-bound;
  NO A/B run. 35 at this acceptance requires kernel work (p2b/dense GEMM)
  or acceptance gains, not idle recovery.
- Serve restored + verified on boot-k3c-pf-gv2.sh (boot 2 of 3): smoke 323,
  gv2 self-check both ranks, env levers present, MemAvail 25/27.

## Round 26 — 2026-09-21: PF-G8 prefill gate REPAIR + re-run — PASS; fold+loader staged dormant

- Root cause of the 2026-09-20 FAIL (20/20 mismatch): the eb78a8b G8 remap
  set `gl_b_stride_k = 0`, believing load_b_gl fully addresses global B. It
  is WINDOW-RELATIVE; the stock window base advances one k-tile per
  iteration (cb2/exl3_gemm_inner.cuh L168). Zeroed stride ⇒ shapes 2/3
  re-read k-tile 0 per window, shape 4 pinned at base+0 for all 144
  k-blocks. The earlier numpy reference was derived from the same wrong
  model — it validated the harness against itself. Fix (9f71b6e): G8
  k-stride = TILEBLOCKS_K * 8 * 16 * bits.
- Phase A (CPU): proof_prefill_bitexact.py — stock reference asserted
  against cb2 source lines, full tile-walk sim on real shapes
  gate/up [320,144,32] + down [144,320,32], shapes 2/3/4: 921,600
  predicated B loads, 0 mismatches, 100% coverage.
- Phase B (GPU window, serves stopped both ranks): check 20/20 BIT-EXACT;
  bench reps 5: at veto m∈{128,256,512} worst -0.1% (g8 slower), best
  +7.7% (gate/up m=128 shape3 g8 faster); layer estimate +0.3%/-0.0%/+0.2%
  — PASS on all three criteria (PREFILL-GATE-RESULT-2026-09-21.log).
- Phase C staged dormant (no rebuild this session): quantize fold
  DSV41_PACK_PF_G8=1 (tools/quantize_experts_exl3.py _pfg8_fold, both fast
  and LDLQ pack paths, roundtrip-verified in the recipe image) + loader
  re-index docker/patch/pfg8_loader_reindex.py (DSV41_LOAD_PF_G8=1, TP
  narrow-dim swap; 9 groups/rank gate/up, 72 k-tiles/rank down — both
  whole). REBUILD-PLAN.md: measured 16.4 min/shard (mul1 build), ~5.5-6 h
  both ranks parallel, 358.1 GB new pack vs 608-652 GB free (archive
  2.0bpw-mul1 first). Serving kernels G8-aware + image rebuild remain the
  gating work for the next dispatch.
- Rider staged: DSV41_ENGRAM_FADVISE_CAP (dormant, unset = exact legacy
  per-row fadvise; set = coalesce to one fadvise per contiguous page run,
  capped) in engram_prefetch_v3 — Round-25's 191 fadvise/step ~3.1 ms
  off-thread at pf_hit 100%. py_compile + chain-apply + functional matrix.
- Serve restored on boot-k3c-pf-gv2.sh: smoke '17 * 19 = ? Step by step,
  then answer.' → 323, env levers PREFETCH=1 CENSUS=1 GATHER_V2=1 live,
  gv2 self-check bit-exact, MemAvail 25/27 GiB post-smoke.

## Round 28 — 2026-09-22 endgame: G8 lane ABORTED at the GPU bit-exact gate

- Rebuild finished clean both ranks (Exited 0, 20/20 shards each, 46,080
  trellis tensors, zero error lines). Pack assembled on spark1: 48/48,
  334 GB, index 188,245 tensors; layout verified 46080/46080 folded shapes
  (18,320,256)/(40,144,256) via shard-header scan.
- Staged pfg8_bitexact_gate.py was WRONG for the shipped image on three
  counts: (1) pybind exl3_gemm/gemv signatures differ (C-out + A_had +
  force_num_sms); (2) pf_g8_set drives only the DEVICE constants — host
  TORCH_CHECKs and LinearEXL3.K read env-cached pfg8::env_value(), so the
  A/B must be two processes (ref env-unset vs g8 DSV41_LOAD_PF_G8=1);
  (3) m>144 silently routed to reconstruct_hgemm (AUTO_RECONSTRUCT_
  THRESHOLD=144) — the staged script would have compared gemm-vs-hgemm.
  Rewritten (two-process, real serving entry make_linear_exl3.forward,
  determinism control 16/16 SAME) — kept in results/2026-09-21-pfg8/.
- GATE VERDICT (4 independent runs): exl3_gemm ALL PASS (both shapes,
  m∈{64,128,256,512}); exl3_gemv gate_up ALL PASS; exl3_gemv DOWN
  NON-DETERMINISTIC MISMATCH m∈{1,2,4} (m=4 failed 4/4 runs; m=1,2 failed
  2/4) — same inputs, different outputs ⇒ race/edge in the ported QTIP
  small-m reader at NT=144 (gate_up NT=320 is exact; suspect the DEC5
  8-tile pack-group addressing at non-warp-multiple NT). p2b_fused_moe
  harness leg crashed with CUDA illegal memory access (stock-flag ref run)
  ⇒ decode fused path UNPROVEN in-image. ANY mismatch = ABORT per mandate.
- Abort executed: spark2 push killed (~8%), rebuild containers removed,
  rollback pack verified 48/48 both ranks, serve restored via
  boot-k3c-pf-gv2.sh (canonical-e12 + 2.0bpw-mcg + all four levers), health
  green, smoke 323, MemAvail 25/27 GiB post-smoke.
- NOT measured: G8, lm_head stack, fadvise cap — 35 tok/s not reached;
  baseline 31.37 stands. G8 pack intact on spark1; resume = fix gemv reader
  (down NT=144, m≤4) + prove p2b in-image + re-run gate + assemble_pack.sh
  (resumable) then Phase 3+. Details: results/2026-09-22-endgame/VERDICT.md.

## Round 29 — 2026-09-22 endgame2 (G8-repair + resume)

- R28's "NT=144 nondeterministic gemv race" was a GATE HARNESS ARTIFACT, not
  a kernel defect. v2 passed mcg=None → host cb=0 → K=2 QTIP gemv ineligible
  → m≤8 points silently ran the AUTOTUNED regular GEMM, whose candidate
  timing on real B differs between stock and G8 layouts → different split-k
  order → sub-ULP nondeterministic diffs (autotune key MAX(size_m,2) explains
  the m=1/m=2 coupling). p2b crash = wrong-length scale tables (gu/uu/dv need
  hidden=5120, got 2304) + 1-entry pointer tables with ids=[0,1] → OOB.
  CPU ownership-map proof: results/2026-09-22-endgame2/ownership_map_g8.py —
  G8 branch addressing exact for BOTH shapes and BOTH readers (no race
  possible). Fixes: results/2026-09-21-pfg8/pfg8_bitexact_gate.py v3 (mcg
  marker + shared EXLLAMAV3_TUNE_CACHE + correct p2b tables + NaN-aware eq).
- GATE: 24/24 PASS ×2 consecutive (canonical-g8 image unchanged). The G8
  kernel port is proven bit-exact including down m=1,2,4 and p2b_fused_moe.
- G8 BOOT lane dead at the floor: loader phase-2 dest alloc fix landed
  (2bddbc3, 06670cf) but spark2 worker OOM-killed ×3 during weight load
  (anon ~45.5GB; G8 load path balloons host memory on the worker rank).
  Kernels fine; loader needs a memory-lean G8 load. New dispatch required.
- lm_head fallback BLOCKED in-image: checkpoint key lm_head.weight does not
  route to language_model.lm_head.weight (ValueError at load). Import-path
  fixes landed (ba0675d: vllm.models.* + direct flashinfer.mm_mxfp8;
  install() now True in-image). Needs key-routing fix + REAL-image test —
  its tests used a fake vllm package.
- Serve restored: boot-k3c-pf-gv2.sh, health green, smoke 323,
  correctness 8/8 (--full incl recall_64k), quality probes 3/3.
  Baseline 31.37 stands; 35 NOT reached (no candidate booted).
- Ops notes: spark2 pack push must use parallel streams (single-stream TCP
  collapses to 22MB/s at 11ms RTT; 4 streams ≈ 400MB/s); snapshot-dir
  symlink copies must use RELATIVE links (absolute /home paths break inside
  the container's /cache/huggingface mount).

## Round 30 — 2026-09-22 lmhead: lm_head MXFP8 BOOTED (+2.5%, sub-gate → reverted)
- Key routing FIXED (two stacked bugs): (1) the model root is the VL wrapper
  DeepseekV41ForCausalLM (vl_model.py); its suffix rule renames via rsplit
  (prefix preserved), so staged lm_head.weight → lm_language_model.lm_head
  .weight (garbage) and lm_head.weight_scale matched no rule (\.scale$ wants
  a dot) → bare-key ValueError at the wrapper root. Fix: install() adds
  REGEX rules (run before suffixes) ^lm_head\.weight$→head.weight and
  ^lm_head\.weight_scale$→language_model.lm_head.weight_scale to both
  mapper makers; inert for stock packs (791ef2b). (2) head_dtype is a
  PROPERTY always returning the model dtype — the "is not None" disarm
  always tripped; disarm now only on genuine override (1a9def4).
- REAL-image integration test now exists: results/2026-09-22-lmhead/
  integration_test.py (CPU-only docker run, canonical-e12): real mapper ×
  both real pack indexes + real AutoWeightsLoader; 15/15 PASS. It caught
  bug (1)'s weight-key corruption that code-reading missed. Lesson: mapper
  behavior must be tested against rsplit semantics, never assumed.
- Boot 2 (boot 1 = bug 2): both ranks 'lm_head mxfp8 enabled (b12x,
  (64640, 5120))', smoke 323, correctness 7/7 quick, live argmax-flip
  ≈1–2%/step (3/12 greedy completions identical; offline bound 4.5%).
- L.A.I.L: real1 median 32.45 (n=5), real2 31.89 (n=5), pooled 32.17 (n=10)
  = +2.5% vs 31.37 — below the +3% KEEP gate (32.31) → REVERTED. CLI prose
  n=9 median 32.35 (twin). 35 NOT claimed (needs two medians ≥35; best 32.45).
- Serve restored: boot-k3c-pf-gv2.sh (stock 31.37 config), smoke 323
  post-restore. lm lever remains one flag away and boot-clean.
- Boot cap: 3 of 4 used (boot1 disarm-bug, boot2 measure, boot3 restore).

## Round 31 — 2026-09-22 g8final: root cause of the G8 boot OOM found + fixed (load phase)
- Wrapper-in-loop diag (the layer every prior diag skipped): diag_v4 CPU
  (real VL mapper.apply + sorted feed + real shard fns + G8 consume) —
  REFUTES the prime hypothesis: per-step new anon storage 0 on all legs,
  sharded views CONTIGUOUS in both stock and G8; the tracer's "2 tensors per
  key" are both file-backed mmap views (mapper rename = second view on the
  same storage, reclaimable). Feed exonerated on CPU.
- diag_v5 FULL GPU topology (spark2, 40 layers / 184,320 keys, real CUDA
  dests + H2D + pwaf): the balloon is the CONSUME phase with CUDA dests —
  each dest.copy_ H2D pins the pageable mmap source pages until tensor free;
  the wrapper retains the whole sorted mapped list → pins accumulate: stock
  peaks 17.9 GiB anon + 9.6 swap (survives), G8 39.3 + 14.0 (dies at 59.4
  in the real boot). Site: vl_model.py DeepseekV41ForCausalLM.load_weights
  `mapped = sorted(...)` retained; AutoWeightsLoader itself is fully
  streaming (verified).
- FIX (de51315): docker/patch/g8_stream_feed.py — env-gated
  DSV41_LOAD_PF_G8, drain-in-place generator (identical order, each slot
  None'd as yielded). diag_v6 A/B: G8 consume anon peak 39.3 → 1.9 GiB
  FLAT, swap 14.0 → 0.6, MemAvail flat 51.9 GiB. Stock path untouched
  (marker absent without env, verified on e12 + g8 images).

## Round 32 — 2026-09-22 g8final: G8 LANE CLOSED; stock restored
- boot-g8-6 (fix live, both ranks 'g8 stream feed installed'): THE LOAD-
  PHASE OOM IS GONE — both ranks loaded 48/48 shards for the first time
  (previously killed ×3 at 45.5 GiB anon mid-load). ~7-8 min AFTER load
  completion, during post-load engine init (no log line between 48/48 and
  engine teardown), worker TP0 died with NO traceback (spark1 avail 0.6 GiB
  at death) — a SECOND, distinct host balloon in the real-engine post-load
  path (pwaf was measured lean at 2.2 GiB single-process in diag_v5; the
  residual is under NCCL/EngineCore/cudagraph, not reachable within boot
  budget). Per the one-fix-iteration rule: G8 LANE CLOSED with the
  evidence chain complete.
- Ops: watcher v2's abort on rss+swap TOTAL was wrong (counts ~79 GiB/rank
  reclaimable pack mmap pages — a normal stock load too); it killed
  boot-g8-5 at 43 GiB, not an OOM. watcher v3 (anon-only smaps_rollup)
  can't read root-owned container procs; MemAvail floors are the usable
  signal. Boot budget 5/5 used.
- lm_head: stays REVERTED (round 30: +2.5% pooled < +3% gate). No
  re-measure spent — same pack, same gate, nothing could change.
- Serve restored: boot-k3c-pf-gv2.sh, health green, smoke 323, MemAvail
  26.3/28.3 GB. L.A.I.L stock confirm n=5 pooled 31.89 (consistent with
  the 31.37 baseline; noise band ±1). 35 NOT reached (best-ever 32.45).
- Best serve = stock 31.37 config. Full story + artifact index:
  results/2026-09-22-g8final/VERDICT.md.

## Round 33 — 2026-09-22 close: lm_head high-n KEEP (+5.9%, 33.23) — CAMPAIGN FINAL BEST; serve UP
- Fresh n=10 re-measure (warmup 32.82 discarded; batches 33.78 / 32.67
  independent n=5): pooled median 33.23 ≥ 32.31 gate → **KEEP**, reversing
  Round-30's 0.14-miss revert — the round-30 sample sat in the
  acceptance-lottery low band. CLI twin 33.56 (acc 2.21, no collapse).
  Serve LEFT UP: results/2026-09-22-endgame2/boot-lm.sh (k3c-pf-gv2 base
  + SNAPSHOT 2.0bpw-mcg-lmhead-mxfp8 + DSV41_LMHEAD_MXFP8=1).
- Four numbers (fresh, final serve): prose 9-run 39.60 (acc 2.61), cold
  prefill 8k 240.1 / 32k 589.6, MemAvail post-32k 21.6/23.2 GiB.
- G8 post-load site CLOSED with evidence (no boot spent): saved boot-g8-6
  log has no worker-side stack (died with container); watcher v3 anon
  unreadable (root procs); source-read of all DSV41-reachable post-load
  allocs (pwaf .contiguous() no-ops, ngram .cpu().tolist() small) finds
  no named ≥30GiB alloc with an env/flag mitigation — blocker is host
  memory physics under the real engine (NCCL/EngineCore/cudagraph);
  fix = vLLM-build-level = out of scope per stop rule.
- 35 NOT crossed. Ceiling ~33-34 at acc 2.26 with lm_head banked; 35
  needs p2b 22.4ms (G8 stack blocked) + dense 17.6ms CUDA work. Full
  table rounds 15-33 + artifacts: results/2026-09-22-close/VERDICT.md.
- Scoreboard: 26.4 → 27.27 → 28.76 → 30.10 → 31.37 → 33.23. Boots 2/4.

## Round 34 — 2026-09-24 review campaign: 4 levers promoted, canonical-e13, serve UP
- Promoted to defaults (rows above; af90bbb): `DSV41_ENGRAM_WILLNEED=1`,
  `DSV41_STREAM_FEED=1`, `DSV41_WOA_PREPACK=1`, `DSV41_DSPARK_SPARSE_MARKOV=1`.
  The IMAGE default is `canonical-e13` (review-e13, sha256:c81762335a12,
  tagged on both nodes), which carries the woa-prepack o_proj stage.
- Rejected, still off: `DSV41_MHC_DECODE_SPLITS=40` (C1 quick quality FAIL,
  every C A/A hazard sample above both B boots), `DSV41_P2B_SRC_SORT` (cold
  −0.37/+1.41/+0.83%, under the 3% gate), `DSV41_DENSE_DG_SMALLM` (deep_gemm
  slower on every shape), dual-rail NCCL (small-message latency +9.5-30%).
  `DSV41_ENGRAM_GATHER_V2_MAX_ROWS` was not run and not needed.
- Kept: lm_head mxfp8 (s12 quality A/B vs the stock bf16 head: ΔNLL −0.00045,
  GSM8K 94/94; the stock head costs −6.2% L.A.I.L).
- Final boot s13 vs s2 (fresh R33 config, same protocol):
  - prose c=1 37.07 → 39.18 (67.77 → 63.28 ms/step)
  - L.A.I.L fresh 32.44 → 32.44 (66.38 → 62.57 ms/step; acceptance 2.167 → 2.071)
  - L.A.I.L after C2-STRESS 31.91 → 33.63
  - micro pass 1 novel 8k 184.2 → 838.0, 32k 231.1 → 811.5; warm 8k 211.9 → 716.3, 32k 430.0 → 761.1
  - quick quality PASS
  - Evidence: `results/2026-09-24-review/campaign/s13-promote-final/final.json`
- Prefetch v3 c=2 self-disable (IndexError in `_prefetch_worker`, disabled
  after 3 errors per rank on the aged R33 serve): fixed in 2ebb78d, and it
  held through C2-STRESS in s3, s5-s8 and s13. Full account and numbers:
  results/RESULTS.md round 34.

## Round 35 — 2026-09-25 k3: full Viterbi re-encode of the routed experts RUNNING (pack lever, not promoted)
- `tools/requant_full.py` re-encodes all 40 expert shards (46,080 tensors) from the MXFP4 source.
  Encoder: exllamav3 1.5.1 Viterbi + refit_scales (H = I). Format: 2.0bpw MCG K=2. Resumable across both
  Sparks: spark1 claims ascending, spark2 claims descending over ssh. Target revision
  `2.0bpw-mcg-viterbi-lmhead-mxfp8`. Plan, commands and ETA:
  `/home/sfxnz/projects/data/dsv41-requant-viterbi/PLAN.txt`.
- Unit 00003 (1152 tensors) against the served 2.0bpw-mcg shard:
  - relerr 0.26156 vs 0.37754, MSE ratio 0.480; every tensor at 0.691-0.694x of stock.
  - safetensors header byte-identical; non-expert tensors and `.mcg` markers byte-equal.
  - tile_chunk 4608 bitwise equal to 256.
  - Evidence: `results/2026-09-25-kernels/requant/`.
- Format note for kernel work: refit makes `svh` a real per-column scale (0.925-1.072, stock is exactly
  ±1), and `suh` moves slightly. p2b (`had_hf_r_128_inner(sv)`), coop (`scale_h4(svh)`) and exllamav3
  reconstruct already multiply by the fp16 values. Do not add a kernel shortcut that assumes |svh| == 1.
- Serve pin unchanged. The pack goes to an ARMS step-6 ABAB after assembly: smoke 323, correctness --full,
  quality quick+full. Any speed effect comes through acceptance only.

## Round 35 — 2026-09-26/27 k3 serve campaign: kernel bundle + TOPK=1024 promoted, canonical-e14, Viterbi pack rejected, serve UP
- Plan `results/2026-09-25-kernels/ARMS-r3.txt`; evidence `results/2026-09-26-serve-r3/`; full account in
  results/RESULTS.md round 35. Every counted boot: gpu_guard 0 FOREIGN, disarm_scan rc 0, honest cells, quick
  quality PASS except the Viterbi boots (golden hazard) and the A-1 A/A control rep1 (aa hazard, no lever armed).
- Promoted to defaults (rows above; 5da6f21): the round-3 kernel bundle as one unit (`DSV41_P2B_COOP=2`,
  `DSV41_DENSE_GEMV=1`, `DSV41_MHC_DET_SPLITS=16`, `DSV41_ENGRAM_NATIVE_STAGE=1`, `DSV41_ENGRAM_EARLY_HASH=1`,
  `DSV41_ATTN_T2R_DEDUP=1`, `DSV41_SWA_META_FUSED=1`, `DSV41_MOE_PREP_FUSED=1`, `DSV41_CANDIDATE_MASK_BOUNDED=1`,
  `DSV41_INDEXER_WP_GEMV=1`) and `DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024`. The IMAGE default is `canonical-e14`
  (review-e14, sha256:3a002b55c9bc, tagged on both nodes), which carries the coop/dataflow p2b build.
- Not promoted, still off: the five KP add-ons (ms/step −1.3..−1.9 beyond noise, but acceptance unmatched and the
  tok/s fallback not met); the revert to `DSV41_DSPARK_SPARSE_MARKOV=0` (W does not beat S beyond noise).
- Rejected: the Viterbi pack `2.0bpw-mcg-viterbi-lmhead-mxfp8` (golden hazard 0.094-0.102 vs limits 0.032/0.043 on
  both boots; L.A.I.L −1.024 tok/s vs noise 0.345; NLL −42.1% and prose_long c=1 +3.108 tok/s in its favour). Serve
  pin unchanged; the pack is local only (publish command and local run: results/RESULTS.md round 35, ARMS.md).
- Before (A-1, A-2, round-34 defaults) → after (S1024-1, S1024-2, final-1, the final config), arm medians:
  - L.A.I.L 33.95 → 43.17 tok/s (+27.1%), 62.79 → 49.00 ms/step; acceptance 2.138 → 2.133
  - four prose c=1 39.16 → 52.52 (+34.1%); prose_long c=1 33.00 → 43.73 (+32.5%), 63.34 → 48.98 ms/step
  - structured c=1 62.19 → 83.81 (+34.8%); c=2 aggregates: prose 56.70 → 81.69, structured 89.04 → 138.34,
    prose_long 47.40 → 65.62
  - prefill within noise (pp_novel 8k/32k 819.2/813.0 → 821.4/806.3)
  - final-1 quality full PASS (NLL 0.242247, GSM8K 94/100, thinking 39/40, MMLU 199/228, needle 9/9)
  - Evidence: `results/2026-09-26-serve-r3/round35-headline.json`
- Serve left UP on final-1 (canonical-e14, round-35 defaults, `AUDIT=strict ./run.sh` from the kernels-r3 worktree,
  started 2026-09-27T02:35:28Z).

## Round 36 — 2026-09-27 Viterbi pack adopted (protocol owner's decision), serve UP
- Default pack `SNAPSHOT_SHA=2.0bpw-mcg-viterbi-lmhead-mxfp8` (row above; 8f94827). It goes to the Hub as a new
  revision in the Publish stage. `2.0bpw-mcg-lmhead-mxfp8` and `2.0bpw-mcg` stay documented as previous revisions.
- Protocol amendment (ARMS.md step 6, "Pack changes"; aac194a). The golden-hazard gate compares greedy text with
  goldens recorded on the same pack, so it gates kernel/numerics levers only. A pack change is judged on paired NLL,
  paired task scores and A/A self-consistency, and the baseline plus goldens are re-recorded on the new pack.
  Decision record: `results/2026-09-27-viterbi-adopt/decision-viterbi-addendum.json` (P4 numbers).
- New quality baseline (V-1, no --baseline): `results/2026-09-27-viterbi-adopt/quality-baseline/{quick,full}.json`.
  quick NLL 0.138564, A/A 0.0122. full NLL 0.139323, A/A 0.00968, GSM8K 97/100, thinking 39/40, MMLU 205/228. The
  09-24 baseline stays the reference for `2.0bpw-mcg-lmhead-mxfp8`.
- V-1 and V-2 booted the new defaults with no overrides (AUDIT=strict), full ARMS-r3 procedure, every gate clean.
  V-2 quick + full PASS against the new baseline: golden 0.01104 / 0.01128 against limits 0.0244 / 0.01936; GSM8K
  96/100; MMLU 202/228.
- Pooled V-1 + V-2 (recipe.yaml measured rows): L.A.I.L 42.58 tok/s (n=20; R35 42.98, n=30), prose c=1 50.41
  (52.97), structured c=1 84.20 (83.73), prose c=2 agg 81.02 (81.73), structured c=2 agg 156.29 (138.34).
  Evidence: `results/2026-09-27-viterbi-adopt/round36-headline.json`; full account in results/RESULTS.md round 36.
- Serve left UP on V-2 (canonical-e14, round-36 defaults, `AUDIT=strict ./run.sh` from the kernels-r3 worktree,
  started 2026-09-27T06:55:34Z).

## 1017 release — 2026-10-18 boot D1: 2.7bpw-mcg-hq on e15-kbits, KV 3.875 GiB
- Rounds 1-36 above measured the Sept 2.0 bpw pack on the Sept model; their evidence is in the Sept recipe repo
  (https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ 3de9146). This round is the first 1017 serve measurement.
- Pins: source `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, served name
  `deepseek-ai/DeepSeek-V4.1-Flash-1017`, pack `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` @ `2.7bpw-mcg-hq` (2.70 bpw
  routed mean, mixed K 2/3/4), image `dsv41-flash-exl3-sm121:e15-kbits` (sha256:7e2cec199bed). Default
  `KV_CACHE_MEMORY` 4831838208 → 4160749568 (3.875 GiB); the other serve defaults are unchanged (round-34/35 levers on).
- Boot D1 (2026-10-18 18:23Z, 2 nodes, `SKIP_DOWNLOAD=1 AUDIT=strict KV_CACHE_MEMORY=4160749568 ./run.sh`): rc 0,
  audit ok head + worker, 40 per-rank `routed experts K_gu/K_d` lines matching the allocation, smokes 323 / Red,
  warmup ok. Boot ~11 min (load 508 s head / 305 s worker). Model memory 97.73 GiB/rank; KV 1,108,848 tokens
  (1.06x of one 1M request).
- MemAvailable (spark1 / spark2): min during boot 4.6 / 5.4 GiB, after smoke 4.3 / 5.9 GiB; normal serving incl. the
  128k needle >= 3.8 GiB; min under all D1 tests 3.17 GiB (spark1, ~10 s, quality_eval nll prompt_logprobs on the
  head). The memory watchdog never tripped.
- Cells (one boot; recipe.yaml measured rows):
  - L.A.I.L prose c=1 (512 tok, t=0.2, median of 3): 39.6 tok/s, 52.5 ms/step, acceptance 2.09
  - bench_decode prose c=1 (x9, 200 tok): 49.6 tok/s, 52.2 ms/step, acceptance 2.58; natural stop at 83 tokens
  - bench_decode structured c=1 (x9): 77.0 tok/s, 51.4 ms/step, acceptance 3.96
  - c=2 aggregates (x5): prose 75.8, structured 140.3 tok/s
  - prefill pp_novel 4k / 16k / 64k 795 / 755 / 733 tok/s; pp_warm 707 / 689 / 688; tg after 64k context 48.1
  - TTFT short prompt ~0.24 s; e2e 3/3 pass (coding-agent 53.9 tok/s decode, 64k doc recall hit, TTFT 90.6 s)
- Quality: `quality_eval --full` PASS; `results/2026-10-18-1017/quality-baseline/full.json` is the 1017 baseline.
  nll 0.1445 (repeat delta 0.0018), decode median |dlogprob| 0.00069, tools 22/22 exact + 8/8 no-call, needle 9/9
  (8k/32k/128k x 3 depths), selfcons 5/12 identical, c2 pass, vision pass, GSM8K 97/100, GSM8K thinking 39/40,
  MMLU 208/228 (91.2%). The Sept pack's numbers are on a different model and are not a paired comparison.
- Release boot r1 (`results/2026-10-18-1017/boot-r1/`): quick baseline PASS (`quality-baseline/quick.json`); L.A.I.L 39.6,
  prose c=1 50.0, structured c=1 75.8 tok/s; 60-minute c=2 soak 4,296 requests, 0 errors, 4,295 correct (one model
  arithmetic miss). Long context (boot D1): needle 256k 3/3, 1M 3/3. Served KL vs native 1017 (eval boot,
  `results/2026-10-18-1017/boot-eval/`): 0.025 nats/token, dNLL +0.63%, top-1 95.3%; encyclopedic worst (0.114).
- Evidence: `results/2026-10-18-1017/boot-d1/` (command, boot, head/worker logs, bench_decode, micro, e2e + L.A.I.L,
  quality, MemAvailable after smoke); allocation and encode: `results/2026-10-18-1017/allocation/`,
  `results/2026-10-18-1017/encode/`.

## perf2 kv-pack — 2026-10-19: 4-session defaults, packed KV groups

- Defaults: `MAX_NUM_SEQS` 2 -> 4 (guard >4), `MAX_MODEL_LEN` 1048576 -> 500000, `DSV41_KV_PACKED_GROUPS=1`
  (vLLM PR #60996 on e15's planner), `DSV41_SWA_META_BATCHED=1`, `DSV41_INDEXER_EXPANDED_TABLE_LAZY=1`,
  `--mm-processor-cache-gb 1`,
  `--prefix-cache-retention-interval 4096`, the 0.32rc1 ragged-indexer guard (required). KV pool unchanged
  (4160749568). Promoted boot: `results/2026-10-19-1017-c4/boot.sh`.
- Occupancy row (CPU, e15 planner + KVCacheManager, recipe page pins; perf2/memory and perf2/kv-pack): packed groups
  50 / 46,080 B per block (image: 18 / 138,240). At 500,000: 1,471,832 tokens (2.94x) at B=8192, 1,793,682 (3.59x)
  at 4096 (image: 964,308 / 1,099,590). Four sessions growing to 500,000 tokens at once, zero preemptions: 3.82 GiB
  (8192) / 3.59 (4096); four full 524,288-token sessions: 3.99 / 3.75 GiB, hence the 500,000 window at B=8192
  (B=4096 allows 4 x 524,288).
- Memory not in the KV pool (estimate from the e15 builder code, `perf2/kv-pack/builder_mem.txt`): packing adds 28
  SWA metadata builders (B x 1037 B each: 8.1 MiB at 8192) and 2 ratio-2 indexer builders. Builders per rank at
  500,000 / 8192: 513 MiB image's groups, 985 MiB packed, 374 MiB packed with the lazy indexer table (147 MiB image's
  groups + lazy); at 4096: 256 / 492 / 187 MiB.
- GPU (`results/2026-10-19-1017-perf2/`): strict audit ok and `GPU KV cache size: 1,471,832 tokens` (2.94x) on boots I1-I4.
  Boot I3 (these defaults, image e16-rows with `DSV41_P2B_WIDE=0`, the other perf2 levers off): c=1 prose 50.3 tok/s
  at 52.7 ms/step (B0, 18 groups at 524,288: 49.0 at 52.7), c=4 distinct prose 62.3 tok/s aggregate at 154.5 ms/step
  (B0 61.4 at 158.0), quick quality PASS (nll 0.1433); MemAvailable minimum spark1 2,038 / spark2 4,367 MiB.
- Full stack (boot FINAL, plain defaults, `results/2026-10-19-1017-perf2/FINAL/`): the 4-session resident soak (4 x ~447K tokens,
  0 preemptions, KV usage peak 0.876) and needles at 256K and 500K (3/3 each).
