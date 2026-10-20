# perf2 evidence (2026-10-19): four sessions on 2× DGX Spark

Pack `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` @ `2.7bpw-mcg-hq`, TP=2: spark1 is the head and spark2 the worker. These boots measured the perf2 changes: packed KV groups and 4-session defaults, the decode kernels past 8 rows, the tensor-core prefill MoE, the Engram read-ahead and poll, the CED decoder replay with its prefix-cache fixes, the allocator hold and shared fat scratch, the router GEMV and main_proj overlap, the DSpark adaptive verify depth and per-batch k, the prefill cap, three of the Sept KP add-ons, and the spark1 memory arms; FINAL is the stack tip with plain defaults. `flags.md` and the perf2 PR descriptions cite these folders; the tools named here that are not in the tree yet (`docker/Dockerfile.e16`, `tools/replay_ab.py`, `tools/c4_mixed.py`) come with the perf2 code PRs, which merge after this folder. Every folder has the boot command (`command.txt`, `boot-*.sh`) and the run.sh output (`boot.txt`). Each row below is the median of the runs in that boot's cell files; `numbers.txt` has the same numbers next to B0 (I1-I6) or the arm's reference boot (D1-D3), and `numbers-vs-<boot>.txt` against the boot named.

## Boots

| Boot | Code | Image | Env against the code's defaults | Cells |
|---|---|---|---|---|
| B0 | `main` @20ee8ba (release stack, the image's 18 KV groups) | `e15-kbits` | `FORCE_UNSAFE_CTX=1 MAX_NUM_SEQS=4 MAX_MODEL_LEN=524288`, profiler endpoint (B0b/B0c are profile boots of the same env) | decode c=1/2/4, prefill micro, prefix cache, prefill profile breakdown (`analysis/`) |
| I1 | perf2/integration @ff424e8: kv-pack, c4-kernels, prefill-moe, dspark-adaptive, decoder replay W=2606 (before the prefix-end fallback and junction anchors), engram-io read-ahead | `e16-rows` (built from c4-kernels @d365c92; TC kernel JIT at boot) | `DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1`, profiler endpoint | decode, prefill, prefix cache, quick quality, replay B arm (`B-gen`, `B-turns`), prefill16k profile (`profiles/`, `analysis/`) |
| I2 | as I1 | as I1 | I1 + `DSV41_DECODER_REPLAY_ROWS=0` | prefill, prefix cache, replay A arm and compare (`A-*`, `compare-*`) |
| I3 | as I1 | as I1 | I1 + `DSV41_DECODER_REPLAY_ROWS=0 DSV41_MOE_PREFILL_TC=0 DSV41_P2B_WIDE=0 DSV41_ENGRAM_PREFILL_AHEAD=0` (the kv-pack defaults alone) | decode c=1 and c=4 distinct, prefill, quick quality |
| I4 | perf2/integration2 @2844fce: I1's code + prefill-mem (round 1), decode-glue, decoder replay W=2670 with the fallback, anchors and reply retention, Engram native poll | as I1 | `DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1 DSV41_ROUTER_GEMV=1 DSV41_MAIN_PROJ_OVERLAP=1 DSV41_ENGRAM_NATIVE_POLL=1 DSV41_PREFILL_MEM_LOG=1` | decode, prefill (+64K twice), prefix cache, replay prefix checks (`prefix-B`), quick quality, census summaries |
| I5 | perf2/integration3 @ca42873: I4's code + c4-sched (off) | as I1 | I4 + `DSV41_DECODER_REPLAY_ROWS=0`; the re-run (I5c) also `DSV41_MAIN_PROJ_OVERLAP=0` | replay A arm on I4's code (`A-*`, `prefix-A`, `compare-*`), prefill, c=1 prose; `trip1/` and `fail2/` are the two failed attempts |
| I6 | as I5 | as I1 | `DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1 DSV41_ROUTER_GEMV=0 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=0 DSV41_ENGRAM_PREFILL_AHEAD_DONTNEED=0 DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB=3 DSV41_PREFILL_MEM_LOG=1` | decode c=1 / c=4 distinct, prefill, page cache (`cached.txt`), acceptance per draft position (`spec-c4-*.prom`) |
| C0 | as I4: cells on boot I4's serve, after its own cells | as I1 | I4's (stock scheduler: I4's code has no c4-sched) | `c4-sched/cells.sh` (`C0/cells.sh`): solo, dec, dec-aa, turns (`C0-*`) |
| C1 | perf2/integration3 @ca42873 | as I1 | BASE + `DSV41_SCHED_PREFILL_CAP=2048 DSV41_SCHED_DECODE_STEPS=1`, `EXTRA_ARGS=--enable-logging-iteration-details` | the same cells (`C1-*`), compares against C0 (`compare-*.txt`), quick quality, the solo prompts' allocator steps (`prefill-mem-steps-solo.txt`) |
| D1 | as C1 | as I1 | BASE + `DSV41_DSPARK_ADAPTIVE=1` (k=3, TAU 0.15) | decode c=1/2 prose and structured, code c=1, c=4 distinct prose and structured, L.A.I.L, acceptance counters around the cells (`spec-*.prom`), quick quality, greedy identity against I4's goldens (`identity-vs-I4.txt`) |
| D2 | as C1 | as I1 | BASE + `NUM_SPECULATIVE_TOKENS=5 DSV41_DSPARK_ADAPTIVE=1` (TAU 0.15) | as D1, plus CUDA graph memory (`graphs.txt`) |
| D3 | as C1 | as I1 | BASE + `NUM_SPECULATIVE_TOKENS=5 DSV41_DSPARK_ADAPTIVE=1 DSV41_DSPARK_TAU=0.3` | as D2 |
| D4 | perf2/integration4 @746ed86: integration3 + prefill-mem round 2 + dspark-adaptive round 2 | as I1 | BASE + `NUM_SPECULATIVE_TOKENS_PER_BATCH=1-2:5,3-4:3 DSV41_DSPARK_ADAPTIVE=1 DSV41_SCHED_PREFILL_CAP=2048 DSV41_SCHED_DECODE_STEPS=1` (D4's env; the floor is 3 GiB by default there) | as D1, plus grammar (json_schema, `grammar.*`), prefill micro and the allocator's per-prompt memory (`prompt-mem-*.txt`) |
| KP-B | as D4 | as I1 | D4's env + `DSV41_NCCL_EAGER_TWIN=1 DSV41_PM_QOS_US=20 DSV41_ENGRAM_WKV_TP=1 DSV41_MHC_DET_OVERLAP=1 DSV41_AR_L2_PREFETCH=1 DSV41_ATTN_OUT_DEFER=0` | decode c=1 / c=4 distinct, L.A.I.L, quick quality, identity against D4, one 64K prompt with a /proc probe (`probe-report.txt`, `memsample.tsv`) |
| KP-C | as D4 | as I1 | D4's env + `DSV41_PM_QOS_US=20 DSV41_ENGRAM_WKV_TP=1 DSV41_MHC_DET_OVERLAP=1` | as KP-B without the probe; `numbers-vs-KP-B.txt` |
| PM-B4K | as D4 | as I1 | D4's env + `MAX_NUM_BATCHED_TOKENS=4096` | decode c=1 / c=4 distinct, prefill micro (memwatch trip) |
| PM-A1 | as D4 | as I1 | D4's env + `DSV41_DECODER_REPLAY_ROWS=0` | `replay_ab` gen against I5c / I2, 16K / 64K prompts; both boots tripped (`trip1/`, `trip2.txt`) |
| NOREL | as D4 | as I1 | D4's env + `DSV41_PREFILL_EMPTY_CACHE_TOKENS=100000000` | 32K prompts with /proc probes (`probe-report-32k-run*.txt`); tripped |
| PROBE | perf2/prefill-mem-int4 @d1fa19d: integration4 + the memory probe | as I1 | D4's env + `DSV41_DECODER_REPLAY_ROWS=0 DSV41_PREFILL_MEM_PROBE=1` | 16K / 32K prompts with the in-worker probe (`probe-lines.txt`); tripped |
| FINAL | the release stack tip @4ae29a6 (pr/10-evidence) | `e16-rows` sha256:df157a2d8afe (Chain A from @156a06a; TC prebuilt) | none: `SKIP_DOWNLOAD=1 AUDIT=strict ./run.sh` (`boot-FINAL.sh`, `env-head.txt`, `resolved-env-harness.txt`) | blocks 1-8 (`blocks/`, `phases.txt`): smoke; decode c=1 / 2 / 4 (distinct and same prompt), L.A.I.L, code; prefill micro; quick and full quality, served KL; replay prefix A/A; c4-sched cells; needles 256K / 500K; the 4-session resident soak |
| FINAL-EVAL | as FINAL | as FINAL | `MAX_NUM_BATCHED_TOKENS=512 LANGUAGE_MODEL_ONLY=1`, the release eval boot's settings (`results/2026-10-18-1017/boot-eval/`) | served KL on all 72 panel rows (`served_kl.*`) |
| PCP1 | perf2/validate-pcp @2c8e9ed: FINAL's tree + the per-CPU list counting (perf2/prefill-mem-pcp-pr06 @d98ee22) | as FINAL | none (`DSV41_PREFILL_EMPTY_CACHE_COUNT_PCP=1` is that tree's default) | smoke, FINAL's prefill micro bench, three 64K prompts (`pp64kx3.*`), memory samplers |

### Decode (aggregate tok/s, ms/step in brackets)

| Cell | B0 | I1 | I4 | I6 | I3 |
|---|---|---|---|---|---|
| L.A.I.L prose c=1 | 38.5 (53.7) | 37.9 (53.6) | 37.1 (53.4) | 38.3 (53.7) | - |
| prose c=1 | 49.0 (52.7) | 47.6 (53.2) | 48.5 (52.9) | 49.2 (52.3) | 50.3 (52.7) |
| structured c=1 | 75.6 (52.6) | 74.4 (53.2) | 75.1 (52.7) | 75.2 (52.7) | - |
| prose c=2 | 73.2 (68.1) | 78.2 (66.7) | 80.4 (65.1) | - | - |
| structured c=2 | 128.7 (61.3) | 122.0 (63.8) | 121.5 (63.3) | - | - |
| prose c=4, distinct prompts | 61.4 (158.0) | 91.1 (101.2) | 89.0 (100.8) | 94.7 (100.0) | 62.3 (154.5) |
| structured c=4, distinct prompts | 140.3 (113.8) | 197.2 (80.9) | 202.3 (77.4) | 199.5 (78.5) | 145.6 (109.3) |
| prose c=4, same prompt | 83.8 (122.8) | 117.3 (87.0) | 120.1 (84.3) | - | - |
| structured c=4, same prompt | 159.9 (99.2) | 219.8 (72.0) | 223.4 (70.9) | - | - |

### Prefill and decode after prefill (benches/micro.sh, median tok/s)

| Cell | B0 | I1 | I4 | I6 | I5 | I2 | I3 |
|---|---|---|---|---|---|---|---|
| 4K pp_novel | 783.3 | 1,888.2 | 1,851.4 | 1,916.0 | 1,595.4 | 1,583.5 | 817.3 |
| 4K pp_warm | 686.9 | 1,781.3 | 1,708.0 | 1,680.0 | 1,477.3 | 1,361.9 | 739.4 |
| 4K tg | 47.2 | 44.2 | 44.5 | 40.3 | 46.8 | 37.9 | 41.8 |
| 16K pp_novel | 642.8 | 1,497.8 | 2,544.1 | 2,560.7 | 1,539.7 | 1,541.3 | 654.0 |
| 16K pp_warm | 610.6 | 2,013.3 | 2,110.2 | 1,891.0 | 1,237.7 | 1,300.0 | 479.4 |
| 16K tg | 38.0 | 38.8 | 36.2 | 42.4 | 39.8 | 104.0 | 35.8 |
| 64K pp_novel | 542.5 | 2,806.8 | 2,867.2 | 2,562.9 | 1,580.4 | 1,560.8 | 675.5 |
| 64K pp_warm | 601.0 | 2,722.1 | 2,447.4 | 2,791.3 | 1,387.4 | 1,421.3 | 680.5 |
| 64K tg | 44.0 | 41.0 | 40.4 | 43.5 | 44.3 | 32.5 | 36.8 |

### Prefix cache (prompt tokens / hit tokens)

| Cell | B0 | I1 | I4 | I2 |
|---|---|---|---|---|
| `a_novel` | 32,614 / 0 | 32,591 / 0 | 32,486 / 0 | 32,766 / 0 |
| `b_identical` | 32,614 / 32,512 | 32,591 / 32,512 | 32,486 / 32,384 | 32,766 / 32,640 |
| `c_multiturn` | 32,696 / 32,512 | 32,675 / 32,512 | 32,570 / 32,384 | 32,850 / 32,640 |
| `d_same_doc_new_question` | 32,614 / 32,512 | 32,591 / 0 | 32,486 / 32,384 | 32,766 / 32,640 |
| `e_shared_16k_prefix_new_tail` | 32,184 / 0 | 32,387 / 0 | 32,296 / 0 | 32,419 / 16,384 |
| `f_identical_to_e` | 32,184 / 16,384 | 32,387 / 0 | 32,296 / 32,128 | 32,419 / 28,672 |

### Quick quality

| Boot | pass | nll | golden hazard |
|---|---|---|---|
| I1 | True | 0.14465 | 0.01693 |
| I3 | True | 0.143297 | 0.00903 |
| I4 | True | 0.143728 | 0.01308 |

### Attribution: prefill pp_novel (tok/s)

| Context | I3 | I2 | I2 / I3 | I1 | I1 / I2 | I4 |
|---|---|---|---|---|---|---|
| 4K | 817.3 | 1,583.5 | x1.94 | 1,888.2 | x1.19 | 1,851.4 |
| 16K | 654.0 | 1,541.3 | x2.36 | 1,497.8 | x0.97 | 2,544.1 |
| 64K | 675.5 | 1,560.8 | x2.31 | 2,806.8 | x1.80 | 2,867.2 |

### Attribution: decoder replay at W=2670 on the code of I4 and I5, prefill pp_novel (tok/s)

| Context | I5 (replay off) | I4 (W=2670) | I4 / I5 |
|---|---|---|---|
| 4K | 1,595.4 | 1,851.4 | x1.16 |
| 16K | 1,539.7 | 2,544.1 | x1.65 |
| 64K | 1,580.4 | 2,867.2 | x1.81 |

### MemAvailable minimum over the boot's cells (MiB, spark1 / spark2)

| Boot | spark1 | spark2 |
|---|---|---|
| I1 | 2,681 | 5,500 |
| I2 | 1,740 | 6,873 |
| I3 | 2,038 | 4,367 |
| I4 | 1,626 | 5,872 |
| I5 | 1,278 | 6,658 |
| I6 | 3,163 | 7,281 |

### DSpark: adaptive verify and per-batch k (aggregate tok/s; ms/step and mean tokens per decode step in brackets)

| Cell | I6 (k=3, off) | D1 (k=3, TAU 0.15) | D2 (k=5, TAU 0.15) | D3 (k=5, TAU 0.3) | D4 (`1-2:5,3-4:3`, TAU 0.15) |
|---|---|---|---|---|---|
| L.A.I.L prose c=1 | 38.3 (53.7; 2.07) | 40.2 (49.8; 2.00) | 39.7 (52.0; 2.04) | 37.9 (49.9; 1.89) | 38.9 (52.0; 2.02) |
| prose c=1 | 49.2 (52.3; 2.58) | 49.6 (50.7; 2.56) | 49.6 (54.3; 2.70) | 48.2 (52.1; 2.50) | 49.6 (54.5; 2.71) |
| structured c=1 | 75.2 (52.7; 3.96) | 74.9 (52.9; 3.96) | 99.2 (59.3; 5.88) | 100.5 (58.5; 5.88) | 100.5 (58.5; 5.88) |
| code c=1 | - | 61.4 (53.6; 3.28) | 68.1 (59.3; 4.07) | 67.4 (58.5; 3.91) | 68.0 (60.1; 4.04) |
| prose c=2 | - | 80.6 (62.3; 2.56) | 78.4 (69.6; 2.75) | 72.6 (66.6; 2.50) | 76.0 (70.4; 2.71) |
| structured c=2 | - | 122.0 (64.5; 3.96) | 180.2 (65.3; 5.88) | 180.3 (65.2; 5.88) | 181.1 (64.9; 5.88) |
| prose c=4, distinct prompts | 94.7 (100.0; 2.47) | 98.7 (90.8; 2.40) | 91.5 (104.9; 2.58) | 93.4 (96.5; 2.39) | 94.8 (95.6; 2.42) |
| structured c=4, distinct prompts | 199.5 (78.5; 3.98) | 198.5 (79.7; 3.98) | 261.7 (90.8; 5.94) | 263.0 (90.3; 5.94) | 198.1 (80.1; 3.99) |

| Boot | quick quality | nll | golden hazard (release goldens) | golden hazard (I4's goldens) | spark1 / spark2 MemAvailable min (MiB) |
|---|---|---|---|---|---|
| D1 | PASS | 0.143497 | 0.01589 | 0.00976 | 3,073 / 5,320 |
| D2 | PASS | 0.145042 | 0.01641 | 0.01111 | 3,486 / 6,086 |
| D3 | PASS | 0.145091 | 0.01995 | 0.01411 | 2,565 / 6,724 |
| D4 | PASS | 0.143105 | 0.0152 | 0.00797 | 1,551 / 6,543 |

### c4-sched cells: three decode streams while a fourth session prefills (C0 stock, C1 `CAP=2048` `STEPS=1`)

| Window | Metric | C0 | C0 A/A | C1 | C1 A/A |
|---|---|---|---|---|---|
| 64K prefill | ITL p50 (s) | 2.74 | 2.73 | 0.14 | 0.12 |
| 64K prefill | ITL p90 (s) | 3.06 | 3.03 | 0.74 | 0.73 |
| 64K prefill | max stall (s) | 3.50 | 3.51 | 1.21 | 0.81 |
| 64K prefill | slowest stream (tok/s) | 1.10 | 0.89 | 4.06 | 4.25 |
| 64K prefill | prefill tok/s (TTFT s) | 2,772 (23.6) | 2,788 (23.5) | 2,249 (29.1) | 2,337 (28.0) |
| 128K prefill | ITL p50 (s) | 2.75 | 2.77 | 0.13 | 0.11 |
| 128K prefill | ITL p90 (s) | 2.85 | 2.84 | 0.77 | 0.74 |
| 128K prefill | max stall (s) | 3.70 | 3.70 | 1.04 | 0.86 |
| 128K prefill | slowest stream (tok/s) | 0.97 | 0.86 | 4.78 | 4.30 |
| 128K prefill | prefill tok/s (TTFT s) | 2,819 (46.4) | 2,804 (46.7) | 2,307 (56.7) | 2,363 (55.4) |
| before the prefills | ITL p50 (s) | 0.089 | 0.088 | 0.089 | 0.089 |
| between the prefills | ITL p50 (s) | 0.090 | 0.090 | 0.093 | 0.090 |
| after the prefills | ITL p50 (s) | 0.091 | 0.090 | 0.090 | 0.092 |
| whole cell | MemAvailable min, spark1 / spark2 (MiB) | 2,387 / 6,419 | 3,172 / 6,571 | 2,819 / 6,869 | 3,463 / 6,395 |

### c4-sched cells: lone requests and a turn-taking session

| Cell | C0 | C1 |
|---|---|---|
| solo: 64K prompt alone, TTFT (s) | 22.52 | 30.35 |
| solo: 128K prompt alone, TTFT (s) | 45.11 | 46.99 |
| solo: short prompts alone, TTFT (s) | 0.24 / 0.24 / 0.17 | 1.98 / 0.23 / 0.17 |
| solo: greedy outputs, C1 against C0 | - | 3 diverged, 5 near-tie |
| turns: new-turn TTFT during the 128K prefill (s) | none | 1.88, 2.00 |
| turns: ITL p50 (s) / slowest stream (tok/s) during the 128K prefill | 2.81 / 0.66 | 0.46 / 4.60 |
| turns: new-turn TTFT outside the prefill (s) | 0.93-0.95 | 0.90-0.97 |

### Round-35 KP add-ons on the D4 env (aggregate tok/s; ms/step and mean tokens per decode step in brackets)

| Cell | D4 | KP-C (PM QoS 20, Engram wkv TP, mHC det overlap) | KP-B (all five add-ons, attention out defer off) |
|---|---|---|---|
| L.A.I.L prose c=1 | 38.9 (52.0; 2.02) | 40.4 (49.8; 2.03) | 41.6 (49.7; 2.09) |
| prose c=1 | 49.6 (54.5; 2.71) | 51.6 (52.0; 2.69) | 51.1 (51.9; 2.72) |
| structured c=1 | 100.5 (58.5; 5.88) | 103.7 (56.7; 5.88) | 104.0 (56.6; 5.88) |
| prose c=4, distinct prompts | 94.8 (95.6; 2.42) | 98.5 (89.9; 2.45) | 103.0 (91.4; 2.45) |
| structured c=4, distinct prompts | 198.1 (80.1; 3.99) | 201.3 (78.7; 3.99) | 202.9 (77.9; 3.99) |

| Boot | quick quality | nll | golden hazard against D4's goldens (`identity-vs-D4.txt`) | spark1 / spark2 MemAvailable min (MiB) |
|---|---|---|---|---|
| KP-C | PASS | 0.144193 | 0.00892 | 2,783 / 6,491 |
| KP-B | PASS | 0.144284 | 0.01267 | 2,058 / 5,409 |

### spark1 memory arms (memwatch: 2 s samples, stops the serve below 1,536 MiB for two samples)

| Boot | Env against D4 | Long prompts | memwatch | Where it tripped |
|---|---|---|---|---|
| D4 | - (decoder replay on, B=8192) | prefill micro 4K / 16K / 64K, two runs each; quick quality needles (8K, 32K) | no trip; lowest spark1 sample 1,551 MiB | - (lowest at the start of its first 64K prompt) |
| KP-B | five KP add-ons, `DSV41_ATTN_OUT_DEFER=0` | one 64K prompt; quick quality needles | no trip; lowest 2,058 MiB | - |
| KP-C | three KP add-ons | quick quality needles (8K, 32K) | no trip; lowest 2,783 MiB | - |
| PM-B4K | `MAX_NUM_BATCHED_TOKENS=4096` | micro 4K-64K | trip 01:50:07Z: spark1 6,406 -> 1,305 MiB in 14 s | the sixth consecutive 64K prompt (`trip.txt`) |
| PM-A1, boot 1 (`trip1/`) | `DSV41_DECODER_REPLAY_ROWS=0` | gen 8K, 32K, 102K | trip 02:08:36Z: 5,078 -> 1,102 MiB | the 102K prompt's second chunk |
| PM-A1, boot 2 | the same | gen 8K, 32K; 16K twice; 64K | trip 02:24:48Z: 5,277 -> 1,167 MiB | the first 64K prompt's second chunk (`trip2.txt`) |
| NOREL | `DSV41_PREFILL_EMPTY_CACHE_TOKENS=100000000` (no releases) | 32K, three planned | trip 03:12:02Z: 4,920 -> 617 MiB, pool flat at 103.71 GiB | the second 32K prompt |
| PROBE | `DSV41_DECODER_REPLAY_ROWS=0 DSV41_PREFILL_MEM_PROBE=1` | 16K three times, 32K three planned | trip 03:49:21Z: 1,140 MiB | the second 32K prompt's second chunk |

### FINAL: decode against B0 (aggregate tok/s; ms/step and mean tokens per decode step in brackets)

| Cell | B0 | FINAL | FINAL / B0 |
|---|---|---|---|
| L.A.I.L prose c=1 | 38.5 (53.7) | 40.5 (50.4; 2.02) | x1.05 |
| prose c=1 | 49.0 (52.7) | 51.3 (52.2; 2.74) | x1.05 |
| structured c=1 | 75.6 (52.6) | 103.6 (56.8; 5.88) | x1.37 |
| code c=1 | - | 70.6 (58.6; 4.15) | - |
| prose c=2 | 73.2 (68.1) | 74.6 (69.7; 2.73) | x1.02 |
| structured c=2 | 128.7 (61.3) | 183.4 (64.1; 5.88) | x1.43 |
| prose c=4, distinct prompts | 61.4 (158.0) | 96.4 (90.8; 2.40) | x1.57 |
| structured c=4, distinct prompts | 140.3 (113.8) | 201.1 (78.6; 3.99) | x1.43 |
| prose c=4, same prompt | 83.8 (122.8) | 123.5 (81.4; 2.68) | x1.47 |
| structured c=4, same prompt | 159.9 (99.2) | 221.9 (71.6; 3.97) | x1.39 |

### FINAL: prefill and decode after prefill (benches/micro.sh, median tok/s)

| Cell | B0 | FINAL |
|---|---|---|
| 4K pp_novel | 783.3 | 1,830.3 |
| 4K pp_warm | 686.9 | 1,594.5 |
| 4K tg | 47.2 | 44.6 |
| 16K pp_novel | 642.8 | 2,235.8 |
| 16K pp_warm | 610.6 | 2,036.5 |
| 16K tg | 38.0 | 44.8 |
| 64K pp_novel | 542.5 | 2,932.0 |
| 64K pp_warm | 601.0 | 2,773.3 |
| 64K tg | 44.0 | 45.1 |

### FINAL: gates

| Gate | Result | Files |
|---|---|---|
| Quality, quick | PASS (nll 0.144659) | `quality-quick.txt`, `quick.json` |
| Quality, full, against `results/2026-10-18-1017/quality-baseline/full.json` | PASS on every gate: nll 0.145776 (limit 0.15447), decode median abs dlogprob 0.00171, tools 22/22 json and exact, 8/8 no-call, needle 9/9, golden / A/A hazard 0.0196 / 0.0124 (limit 0.0285), GSM8K 98/100 (base 97), MMLU 211/228 = 0.9254 (base 0.9123), GSM8K thinking 38/40 = 0.95 (base 39/40 = 0.975: passes the gate's Wilson rule, below the base point value) | `quality-full.txt`, `full.json` |
| Served KL against the native unquantized model, the 57 panel rows of at most 2,048 tokens (in FINAL) | KL 0.0316 nats/token, dNLL +0.0067, top-1 94.65% (the release reference on the same 57 rows: 0.0319, +0.0059, 94.63%); encyclopedic worst, KL 0.104 | `served_kl.txt`, `served_kl.json` |
| Served KL, all 72 panel rows (92,818 tokens), on the release eval boot's settings (FINAL-EVAL) | KL 0.0253 nats/token, dNLL +0.0054, top-1 95.24% (the release reference: 0.0254, +0.0059, 95.27%); encyclopedic worst, KL 0.112, top-1 87.5% | `FINAL-EVAL/served_kl.txt`, `FINAL-EVAL/served_kl.json` |
| Needles, 3 depths each | 256K 3/3 (260,703-token prompts, about 89 s each), 500K 3/3 (499,342-token prompts, about 181 s each) | `needle-*.txt`, `needle-*.json` |
| Resident soak | 4 sessions of 446,057-449,180 prompt tokens + 4,096 completion tokens each, all 4 found their needle, 0 preemptions, at most 4 running and 0 waiting, peak KV usage 0.876 | `soak.txt`, `soak.json` |
| Decoder replay prefix cases, A/A on one boot (salted) | hit checks 28/28 in both runs; greedy identical 15/29, every divergence a near-tie swap | `prefix-B1.txt`, `prefix-B2.txt`, `prefix-compare.txt` |
| Image CPU tests on `e16-rows` (sha256:df157a2d8afe) | 273 OK (3 skipped); chat encoder 8 OK | `image-tests.txt` |
| Strict audit, levers | audit ok; every default lever engaged on both ranks (TC prebuilt, no JIT); KV 1,471,832 tokens (2.94x one 500K session); model 97.29 GiB per rank | `boot.txt`, `boot-grep-boot.txt`, `expected-markers.txt` |

### FINAL: c4-sched cells (same procedure as C0 / C1; `FINAL-*`)

| Cell | FINAL | FINAL A/A | C1 | C0 |
|---|---|---|---|---|
| solo 64K / 128K prompt, TTFT (s) | 21.95 / 43.28 | - | 30.35 / 46.99 | 22.52 / 45.11 |
| ITL p50 during the 64K / 128K prefill (s) | 0.107 / 0.110 | 0.106 / 0.107 | 0.14 / 0.13 | 2.74 / 2.75 |
| max stall during the 64K / 128K prefill (s) | 1.56 / 0.86 | 0.83 / 0.89 | 1.21 / 1.04 | 3.50 / 3.70 |
| slowest stream during the 64K / 128K prefill (tok/s) | 4.00 / 4.52 | 4.03 / 4.28 | 4.06 / 4.78 | 1.10 / 0.97 |
| prefill beside three decoders (tok/s), and against FINAL's solo | 2,255 / 2,348 (0.76 / 0.78) | 2,332 / 2,354 | 2,249 / 2,307 | 2,772 / 2,819 |
| new-turn TTFT during a 128K prefill (s) | 2.00, 1.89, 2.23 | - | 1.88, 2.00 | none |
| ITL p50 before the prefills (s) | 0.080 | 0.080 | 0.089 | 0.089 |

### PCP1: the per-CPU free page lists counted (against FINAL)

| Cell | FINAL | PCP1 |
|---|---|---|
| pp_novel 4K / 16K / 64K (tok/s) | 1,830 / 2,236 / 2,932 | 1,830 / 2,527 / 2,924 |
| three 64K prompts (tok/s) | - | 2,924 / 2,974 / 2,903 |
| allocator releases in the micro bench | 10 or more (the floor among them) | 1 (`grown`), 0 at the floor |
| spark1 MemAvailable + lists, minimum (MiB) | 4,047 (micro; samplers joined by second) | 4,093 (micro), 5,453 (64K prompts; memwatch2, one read) |
| spark1 raw MemAvailable, minimum (MiB) | 1,724, with 4,123 on the lists | 748, with 3,403 on the lists |
| spark1 over the micro bench: pgscan_direct / allocstall / swap-out | 15,634 / 3,810 / 1.98 GiB | 137,967 / 3,369 / 1.77 GiB |
| spark2 over the micro bench: pgscan_direct / allocstall / swap-out | 18,832 / 68 / 0.02 GiB | 0 / 0 / 0 |

### Image builds, 2026-10-20 (`image-builds/`)

| Chain | Build | Node | Result |
|---|---|---|---|
| A | `docker/Dockerfile.e16` on `e15-kbits`, the stack at @156a06a | spark1 | `e16-rows` sha256:df157a2d8afe in 2m54s: `vllm_exl3 rows cuda ok`, `moe_prefill_tc prebuilt ok 022a960b5338` (`build-e16-spark1-156a06a.txt`, `checks-spark1.txt`) |
| A | the same | spark2 | stopped at the `git clone` of vllm-exl3: spark2 has no default route and cannot resolve github.com (`build-e16-spark2-156a06a.txt`); spark1's image was loaded there with `docker save \| ssh spark2 docker load` (`transfer-e16-spark2.*`, `checks-spark2.txt`: the same image ID) |
| B | `docker/Dockerfile` from scratch, a clean detached worktree at @4ae29a6 | spark1 | `e16-rows-scratch` sha256:05122e36e015 in 4m41s with the base layers in the local cache: `exllamav3 ok`, `vllm_exl3 rows cuda ok`, `moe_prefill_tc prebuilt /opt/dsv41-moe-prefill-tc/022a960b5338`, `moe_prefill_tc prebuilt ok 022a960b5338`, the patches label, and `cuobjdump -sass` of `vllm_exl3_c` equal to the serving `e16-rows`' (d1fe0963...) (`build-scratch-spark1-4ae29a6.txt`, `checks-scratch-spark1.txt`) |

The image CPU tests on the new `e16-rows` found three test-harness problems on the round-4 tip (`image-tests-spark1.txt`, `image-tests-spark1-triage.txt`); round 5 fixed them, and FINAL's run passed 273 (`FINAL/image-tests.txt`). The spark2 transfer is this lab's network; the recipe's path is still to build the image on each node.

## Notes

- B0 is the release stack configured for four sessions, not a perf2 change. KV 975,641 tokens (1.86x at 524,288) against 1,471,832 (2.94x at 500,000) with packed groups on I1-I6.
- c=4 cells: `bench_distinct` sends four different prompts; the same-prompt cells (`bench_decode --concurrency 4`) overstate prose throughput (B0: 1.36x) and are kept only for comparison with B0.
- Attribution, prefill: I3 -> I2 turns on the TC prefill MoE and the Engram read-ahead together (the wide decode kernel does not run in prefill). I2 -> I1 turns on the decoder replay at W=2606. I1's 16K cell is out of line: its profile run prefilled 16,046 tokens in 6.49 s (`profiles/prefill16k.run.txt`), while micro took ~10.8 s per prompt. On I4's code, I5 -> I4 is the decoder replay at W=2670. I4 also carries the allocator hold (I1 does not).
- Attribution, decode: I3 -> I1 at c=4 distinct is the wide decode MoE (I3 has `DSV41_P2B_WIDE=0`; the other levers I3 turns off run in prefill only). c=1 does not use it.
- I6 against I4: the router GEMV, main_proj overlap and native poll off, plus `DSV41_ENGRAM_PREFILL_AHEAD_DONTNEED=0`. Decode is within run noise of I4 (c=1 prose 49.2 vs 48.5, L.A.I.L 38.3 vs 37.1, c=4 distinct 94.7 vs 89.0; `I6/numbers-vs-I4.txt`). With `_DONTNEED=0`, spark1's Cached rose from 3.1 to 6.4 GiB over the prefill micro bench (`I6/cached.txt`). Its 64K pp_novel was 2,563 tok/s. `numbers-vs-I4.txt` pools I4's four 64K runs (micro and micro64) into a median of 2,849.
- Greedy identity, I5 (replay off) against I4 (W=2670), same code (`I5/compare-*.txt`):
  - gen at 8K and 32K: identical;
  - prefix cases: 17/29 identical, with the hit checks passing in both arms;
  - turns: `continue` diverges at token 9 and `diverge` at token 6.
- I2's 16K `tg` cell is the median of 168.8 and 39.2 tok/s (`I2/micro.txt`).
- I4 ran while host unit tests ran on spark1. Nine runs of `python3 -m unittest discover` (nice 19, idle I/O class, ~50 s each) between 21:24:51Z and 21:42:30Z overlapped I4's cells from `c1p9` to `micro64` (`I4/phases.txt`). Later test runs (22:19Z-22:59Z) overlapped I5c's boot and cells.
- I5 first attempt (`I5/trip1/`): memwatch stopped the serve when spark1 MemAvailable fell from 4,270 to 1,323 MiB in about 12 s, on the 102,400-token `replay_ab gen` prompt (allocator hold on, decoder replay off). I2 ran the same cell without the hold and bottomed at 3,275 MiB. The re-run dropped that cell; its lowest spark1 sample was 1,278 MiB at the start of the 32K gen cell.
- I5 second attempt (`I5/fail2/`): rank 1 passed the main_proj overlap self-check (rows=20), then disarmed on a verify with `RuntimeError('side-stream combine != stock (rows=6)')`, and the strict audit failed the boot.
- BASE (C1, D1-D3) is perf2/integration3 @ca42873 booted with `IMAGE=dsv41-flash-exl3-sm121:e16-rows DSV41_MOE_PREFILL_TC=1 DSV41_ENGRAM_PREFILL_AHEAD=1 DSV41_ROUTER_GEMV=0 DSV41_MAIN_PROJ_OVERLAP=0 DSV41_ENGRAM_NATIVE_POLL=0 DSV41_PREFILL_EMPTY_CACHE_FLOOR_GIB=3 DSV41_PREFILL_MEM_LOG=1` (`command.txt` in each folder): I6's levers with `_DONTNEED` back at 1.
- DSpark arms: one boot each. Acceptance is the mean number of tokens a decode step emits. D2's structured and code cells gain from k=5; its prose cells do not, and c=4 distinct prose loses 7% against D1. TAU 0.3 (D3) did not beat TAU 0.15. Greedy identity: `tests/quality_eval.py --result <arm>/quick.json --baseline I4/quick.json` re-scores the arm's quick run against I4's goldens (`identity-vs-I4.txt`).
- c4-sched cells (`tools/c4_mixed.py`, `C0/cells.sh`): dec runs three DSpark-3 decode streams and sends a 64K and then a 128K novel prompt from a fourth session; dec-aa repeats it on the same boot; turns runs two decode streams and a session taking 1K-token turns during a 128K prefill; solo sends every prompt alone. C0 ran on boot I4's serve right after I4's cells (21:42:44Z-21:50:18Z, `I4/phases.txt`), with I4's levers (router GEMV, main_proj overlap and Engram native poll on, floor 2 GiB); C1 ran on its own BASE boot, the solo cell first. The solo compare is therefore across two boots with different levers and a warm against a fresh serve, and no boot ran the solo cell twice.
- C1 ran while host unit tests ran on spark1: seven runs of `python3 -m unittest discover` (release builds; 4 cores, nice 19, idle I/O class, 54-70 s each) between about 23:24:00Z and 23:33:12Z overlapped the solo, dec and dec-aa cells and the first 40 s of turns (cells from 23:25:56Z, `C1/phases.txt`). The solo cell (23:25:57Z-23:27:45Z, with the 64K prompt at 30.35 s against C0's 22.52 s) ran mostly inside one of those runs (23:25:51Z-23:27:01Z), and C1's lowest spark1 sample, 1,338 MiB (`C1/mem-minima.txt`), is at 23:27:01Z, the second that run ended (`C1/memwatch-cells.txt`). The head's two floor releases in the solo prompts fall in the same run: after the 64K prompt's third chunk (MemAvailable 2.51 GiB, regrown by 1,646 MiB in the fourth) and after the 128K prompt's second (1.32 GiB) (`C1/prefill-mem-steps-solo.txt`, steps 10 and 17).
- D4 is the stack's DSpark default; the DSpark table's D4 column ran on integration4 with the prefill cap on, the other columns on the integration3 BASE. D4 against D1: structured +34 / +48% at c=1 / 2 and code +11% (k=5 there), prose c=1 flat, prose c=2 -6%, L.A.I.L -3%; c=4 (k=3 there) within run noise. Its grammar check passed 7/7 (`D4/grammar.txt`).
- KP-B ran all five Sept KP add-ons with `DSV41_ATTN_OUT_DEFER=0` (`DSV41_AR_L2_PREFETCH=1` makes `attn_out_defer.py` refuse, which fails the strict audit). KP-C ran three of them. Their ms/step match within ±1.6% on every cell, so the twin communicator and the L2 prefetch add nothing measurable; against D4, KP-C's ms/step is 2-6% lower on every cell. The KP boots' spark1 minima are below D4's, and post-boot MemAvailable on spark1 varied by about 1.1 GiB across these boots (D4 5,783, PM-B4K 6,011, PM-A1 5,590, KP-B 5,530, NOREL 4,932, KP-C 5,478 MiB, `boot-mem.txt`).
- spark1 memory: with the defaults (decoder replay on, B=8192) D4, KP-B and KP-C did not trip memwatch, but D4 came within 15 MiB of its floor. Every arm that changed the prefill path tripped. NOREL rules out the allocator releases as the cause: without them spark1 fell faster. PROBE (rank 0 only, replay off) loses 0.8-2.8 GiB of MemFree outside the torch pool in each long prompt's first chunk; sync, gc, the pinned host cache, mempool and graph memory trims, the stack limit and a 2 s idle wait return none of it, and the next forward step returns it. That hold is under investigation (perf2 prefill-mem).
- FINAL ran the stack tip with plain defaults and no memwatch trip in eight blocks (04:56Z-06:04Z). The watchdog switched to MemAvailable + per-CPU lists (memwatch2) at 05:24:15Z. In the prefill micro block spark1's raw MemAvailable bottomed at 1,724 MiB while 4,123 MiB of free pages sat on the kernel's per-CPU lists, and MemAvailable + lists never fell below 4,047 MiB (`FINAL/memory-micro-spark1.txt`); over the soak that sum stayed at or above 5,130 MiB.
- FINAL against D4 and KP-C: decode within about 1% ms/step of KP-C; prefill 4K 1,830 and 64K 2,932 tok/s match D4, 16K 2,236 is the median of 2,441 and 2,030 (one slow run).
- FINAL's c4-sched cells ran without host load; their solo TTFT (21.95 s at 64K) is back at C0's, which settles C1's solo caveat.
- FINAL-EVAL ran the release eval boot's settings (`MAX_NUM_BATCHED_TOKENS=512`, no vision) on FINAL's tree, so its served KL on all 72 rows is the like-for-like comparison with the release's 0.0254.
- PCP1 counted the kernel's per-CPU free page lists in the allocator hook (the PR 7 change). The hook stopped releasing at the floor and the 16K prefill rose 13% (2,527 against 2,236; FINAL's 16K had one slow run), but raw MemAvailable on spark1 fell lower than in FINAL (748 MiB, with 3,403 MiB on the lists) because the pool stays held, and spark1 still went into direct reclaim (137,967 pages scanned against FINAL's 15,634; allocstall 3,369 against 3,810) and swapped out about 1.8 GiB (FINAL about 2.0). spark2 stayed clean. The host sysctl `vm.percpu_pagelist_high_fraction=512` (flags.md) is the user's choice, needs root and has not been measured. Sources: `PCP1/memwatch2-cells.txt` (one read of both values every 2 s) and `PCP1/memsample.tsv` (1 Hz, with the reclaim counters).

## Files

| Path | Contents |
|---|---|
| `B0/` | Boot B0 and the profile boots B0b/B0c: commands, `boot*.txt`, `boot-facts.txt`, smokes, `bench_decode_c{1,2,4}.txt`, `bench_distinct_c4.txt`, `lail.txt`, `micro*.txt`, `tg64k_retry.txt`, `prefix_cache.*`, `prefix_junction.*`, MemAvailable snapshots, `memwatch-trips.txt`, `nvrm-oom-count-B0.txt`; `analysis/`: B0c prefill16k chunk breakdowns per rank (`prefill16k_rank*.txt`, `prefill16k_buckets_rank*.txt`) and the decode skew summaries |
| `I1/` .. `I6/` | `command.txt`, `boot-I*.sh`, `boot.txt`, `boot-grep-{boot,final}.txt` (lever lines and the must-not-appear scan), `numbers.{txt,json}` (and `numbers-vs-I1.txt` / `numbers-vs-I4.txt`), the cell files (`c1_*`, `c2_*`, `c4_*`, `lail.txt`, `micro*.txt`), `prefix_cache.*`, `prefix_junction.*`, `quick.json` + `quality-quick.txt`, `tools/replay_ab.py` outputs (`A-*` / `B-*` / `compare-*` / `prefix-*`), census summaries (`prefill-ahead-census-*.summary.txt`, `prefill-mem-steps-final.summary-*.txt`, `native-census-c4.txt`), MemAvailable snapshots (`mem-*.txt`, `phases.txt`) |
| `I1/profiles/`, `I1/analysis/` | I1's prefill16k and decode_c4 profile runs and the prefill16k chunk breakdown |
| `I5/trip1/`, `I5/fail2/` | I5's first attempt (boot, memwatch trace, the cells done before the trip) and second attempt (boot, lever lines with the disarm, stop) |
| `I6/cached.txt`, `I6/spec-c4-*.prom` | Page cache before and after the prefill micro bench on both nodes; spec-decode acceptance counters around the c=4 cell |
| `C0/` | The c4-sched cells on boot I4's serve (`C0-*`: serve env and lever lines, solo / dec / dec-aa / turns summaries, the dec A/A compare, greps, memory) and `cells.sh`, the cell script both arms ran |
| `C1/` | C1's boot files (as for I1-I6), its cells (`C1-*`), the compares against C0 (`compare-dec.txt`, `compare-solo.txt`), `mem-minima.txt`, quick quality, the solo prompts' allocator steps (`prefill-mem-steps-solo.txt`) and memwatch's 2 s samples over the cells (`memwatch-cells.txt`) |
| `D1/` .. `D3/` | Boot files (as for I1-I6), the cells (`c12_*`, `code.txt`, `c4_distinct.txt`, `lail.txt`), acceptance counters around them (`spec-*-{before,after}.prom`), `numbers*.txt`, quick quality and `identity-vs-I4.txt`; `graphs.txt` (CUDA graph memory) on D2 and D3 |
| `D4/` | As D1-D3, plus `grammar.{txt,json}`, `micro.txt`, `numbers-vs-{D1,I4}.txt`, `hold-steps-final.txt` and the allocator's per-prompt memory (`prompt-mem-*.txt`) |
| `KP-B/`, `KP-C/` | Boot files, the decode cells, `numbers-vs-D4.txt` (KP-C also `numbers-vs-KP-B.txt`, `cell-mem.txt`), quick quality and `identity-vs-D4.txt`; KP-B's 64K /proc probe (`probe-report.txt`, `pp64k-probe.txt`, `memsample.tsv`) |
| `PM-B4K/`, `PM-A1/`, `NOREL/`, `PROBE/` | The memory arms: boot files, `numbers.txt` (what ran, the trip, the step lines), `trip*.txt` (memwatch around the trip), `prompt-mem.txt` and `phase-min-memsample.txt` (per-prompt and per-phase minima), `policy-lines.txt` (the armed lines), `nvrm-spark1.txt` (NVRM lines), `memsample.tsv` (1 Hz /proc sampler), the cells that ran; PM-A1's first boot in `trip1/`; NOREL's /proc reports; PROBE's probe lines |
| `FINAL/` | The stack tip with plain defaults: boot files and the container env of both ranks (`env-*.txt`, `cmd-head.txt`), the block scripts (`blocks/`), the decode and prefill cells (`numbers-block2-vs-B0.txt`, `numbers-block{2,3}-vs-D4.txt`), quick and full quality, served KL, needles, the soak, the replay A/A prefix compare, the image CPU tests, the c4-sched cells (`FINAL-*`), NVRM lines and the micro block's MemAvailable next to the per-CPU lists (`memory-micro-spark1.txt`) |
| `FINAL-EVAL/` | The served-KL eval boot: command, run.sh output, `served_kl.txt` / `served_kl.json` (72 rows) |
| `PCP1/` | The per-CPU list validation boot: command and boot scripts (`boot-PCP1.sh`, `cells-PCP1.sh`), boot files, the micro bench and the three 64K prompts, the policy lines, memwatch2's samples over the cells (`memwatch2-cells.txt`), the 1 Hz meminfo / reclaim sampler (`memsample.tsv`) and the per-CPU list samplers of both nodes (`pcp-spark*.tsv`), NVRM lines |
| `image-builds/` | The 2026-10-20 image builds: Chain A (spark1 build, spark2 load) and Chain B (from scratch) logs and checks, the transfer script, the preconditions and the first image CPU test run with its triage |
