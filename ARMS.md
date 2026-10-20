# ARMS.md — campaign arm discipline

One boot = one lever. Every arm produces the four numbers via
`tools/four_numbers.sh --arm NAME` and a verdict appended to `flags.md`,
the arm's evidence directory under `results/` (1017 evidence starts at
`results/2026-10-18-1017/`), and the `recipes` skill ledger.

Rounds R1-R36 and the `results/2026-09-*` paths cited below are Sept
history (Sept model, 2.0 bpw pack). Their evidence is in the Sept recipe,
`https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark` @
`3de9146`. They explain the protocol; they are not reference numbers for
1017 arms.

## Arm loop

1. **Pre-flight (both nodes)** — `docker ps --format '{{.Names}} {{.Image}}'`
   must show only `dsv41-flash-exl3`; `ssh spark2 docker ps` same.
2. **Stop both nodes**: `./stop.sh` (reads `.run-state/worker_host`, ssh's
   to spark2).
3. **Boot with ONE env lever** (one knob changed vs baseline; full argv/env
   recorded — paste the exact `FOO=1 ./run.sh` line into the arm notes):
   ```bash
   DSV41_P2B_SRC_SORT=1 AUDIT=strict ./run.sh     # example lever (off by default)
   ```
   run.sh refuses unsafe configs (`FORCE_UNSAFE_QUANT/ENGRAM/CTX` guards);
   if an A/B legitimately needs an override, set the guard for that boot
   only and record it — never commit it as a default.
4. **Boot floors** (read `free -h`, NEVER nvidia-smi):
   - abort if MemAvailable **< 4 GiB** on either node during boot or
     **after smoke** (`smoke_chat.py` + `smoke_vision.py`). The 2.70 bpw
     pack is 97.73 GiB per TP rank (Sept 2.0 pack: 75.58), so the Sept
     floors (12 GiB at boot, 8 GiB after smoke) cannot hold on 1017.
     Boot D1 (KV 3.875 GiB, 2 seqs, 1M window, `results/2026-10-18-1017/boot-d1/`) read a
     boot minimum of 4.6 / 5.4 GiB and 4.3 / 5.9 GiB after smoke
     (spark1 / spark2);
   - known dip, not an abort: `quality_eval.py`'s nll component
     (prompt_logprobs, full-vocab logits on the head) took spark1 to
     3.17 GiB for ~10 s on D1. Normal serving, the 128k needle included,
     stayed at or above 3.8 GiB, and the memory watchdog never tripped.
5. **Four numbers** (serialized, ~20-25 min estimated; re-measure on the
   first campaign run):
   ```bash
   tools/four_numbers.sh --arm <name>          # prose 9x median, pp_warm/pp_novel 8k/32k,
                                               # MemAvailable both nodes, L.A.I.L 3x,
                                               # prose_long c=1/c=2, warm-prefix,
                                               # env digest per rank, page cache
   ```
   Number 3 (MoE/attention ms/layer at shipped chunk) has no live probe —
   fallback is boot knobs `DSV41_STEP_CENSUS=1` / `DSV41_ENGRAM_CENSUS=1`
   (decode-side only; E0 prefill-flush caveat); real profiling is a
   separate gated step. `four_numbers.sh` does not run e2e: run
   `benches/e2e.py` separately on the same boot (step 6 requires exit 0).

   **Order on a fresh boot** (the Round 34 FULL and DECODE protocols; keep
   it the same on every arm so the cells compare like with like):
   `smoke_chat.py` + `smoke_vision.py`, then the fresh L.A.I.L
   (`tools/measure_lail_prose.py --runs 10`), then `bench_decode.py`, then
   `benches/micro.py`, quality, and C2-STRESS last. The L.A.I.L goes before
   `bench_decode.py` because bench_decode's default `--concurrency 1 2`
   sends c=2 traffic. A c=1 cell that has to run before the L.A.I.L passes
   `--concurrency 1`.
6. **Promote rule**. The old text here ("9-run prose median beats
   baseline") was never what decided an arm. Rounds R16-R33 used +3% on
   the pooled L.A.I.L t=0.2 median (flags.md sections Round 16, Round 21,
   Round 23, Round 30, Round 32 and Round 33). That gate sits inside boot-to-boot
   noise: the identical config measured 27.40/26.13/28.01
   (`results/2026-09-21-cpuhash/VERDICT.md`), and lm_head went from REVERT
   at +2.5% (R30) to KEEP at +5.9% (R33). An arm promotes only when all of
   these hold:
   - **ABAB boots**: baseline A and arm B each boot at least twice,
     interleaved A, B, A, B, with `four_numbers.sh` on every boot. No
     comparison against a baseline from another day or boot sequence.
   - **Primary metric**: `prose_median_ms_per_step`, and
     `median_ms_per_step` of the prose_long c=1 cell (ms per verify step;
     lower is better; the reference is the A boots of the same ABAB
     sequence, not a number from another day). Use it only at matched acceptance: the A and B
     per-boot `median_run_acceptance_len` must overlap. If they do not,
     the arm changed drafting; gate on prose_long c=1 tok/s instead with
     the same noise rule.
   - **Noise-aware gate**: noise = the larger boot-to-boot spread (max-min
     of per-boot medians) of A and of B. Promote when the B-vs-A median
     improvement exceeds that noise AND every B boot beats the A median.
     A fixed +3% is not a gate.
   - **Non-inferiority**: prefill `pp_warm` and `pp_novel` 8k/32k,
     prose_long c=2 aggregate, and warm-prefix `hit_fraction_of_expected`
     not worse than A by more than their own boot-to-boot spread; boot
     floors (step 4) hold; `benches/e2e.py` exits 0.
   - **Honest cells**: prose_long `natural_finish_reason` is `length`
     (post-EOS fraction 0), `serve_env_ranks_match` is true, and
     `lever_disarmed` is false (`tools/disarm_scan.sh` found no
     LOG_DISARMED line in either rank's docker logs; a lever can turn
     itself off at runtime, after the post-ready audit ran).

   **Quality gate (required before any KEEP)**: after the four numbers,
   serialized and never interleaved with perf capture, the quick eval must
   pass against the stored baseline of the pack under test:
   ```bash
   python3 tests/quality_eval.py --quick \
     --baseline results/2026-10-18-1017/quality-baseline/quick.json \
     --out results/<arm-dir>/quality_quick.json      # exit 0 = pass
   ```
   The baseline for `2.7bpw-mcg-hq` is
   `results/2026-10-18-1017/quality-baseline/`: `full.json` was recorded
   on boot D1 (PASS); `quick.json` was recorded on release
   boot r1 (PASS). Sept baselines were recorded on a different model
   and fail by construction against 1017 weights.
   It gates prefill numerics (NLL), decode numerics (decode-vs-prefill
   probe and the golden flip hazard against the A/A control), tool calls,
   needle 8k/32k, c=2 and vision. A lever that changes numerics by design
   (a new pack, lm_head, a kernel format) also runs `--full` against
   `full.json`. A failed gate means no KEEP, even when the perf win is real.

   **Pack changes (amendment, round 36, 2026-09-27; decided by the
   protocol owner).** The golden flip hazard (`selfcons.golden_hazard`)
   compares the arm's greedy text with run A of the baseline's
   `selfcons.runs` (the goldens). That only means something when the
   baseline was recorded on the SAME pack (the same weights). So the golden
   gate gates kernel and numerics levers on the pack the baseline ran:
   kernels, formats, fused paths, lm_head paths. It does not apply to a pack
   (weights) change. A new pack fails it by construction: round 35 P4 read
   golden hazard 0.094-0.102 on the Viterbi pack while that pack's own A/A
   hazard was 0.008-0.016 (`results/2026-09-26-serve-r3/decision-viterbi.json`).
   A pack change is judged on these, against the current pack, in the same
   ABAB:
   - **paired NLL** over the 40 fixed passages: the per-passage delta
     against the current pack, with its 95% CI and Wilcoxon test. It must
     not be worse, and the same-arm cross-boot per-passage deltas are the
     noise;
   - **paired benchmark tests** over the fixed items: GSM8K, GSM8K-think,
     MMLU and tools exact_args, with McNemar per interleaved pair and a
     pooled sign test. No significant loss is allowed;
   - **A/A self-consistency**: the new pack's own `aa_hazard`, and the
     cross-boot hazard between its boots, inside the current pack's A/A
     band;
   - every other quick/full component in its band: decode probe, needle,
     tools json/no_call, c=2, vision, `tests/correctness.sh --full`.
   Speed follows the rest of step 6, as for any arm. Adopting a pack means
   re-recording the quality baseline and goldens on it. Run quick and full
   with no `--baseline` on the first boot of the new defaults, store them
   as the new baseline, and gate every later boot against it (including
   the second boot of the new defaults, where the golden hazard applies
   again, within the A/A band). The old baseline stays as the reference for
   the previous pack.

   **1017 clause (2026-10-18).** The 1017 retarget is not a pack change
   in this sense. It is a new source model: every shard changed (vision,
   embed, drafter and Engram included), it is served under a different
   `MODEL` and served name, and it uses the 1017 chat encoder. There is
   no current pack on the same weights to pair against, so the paired
   NLL and McNemar tests above do not apply to it. The 1017 serve rests on
   absolute results on 1017: `quality_eval.py --full` PASS on boot D1,
   which is also its baseline and goldens
   (`results/2026-10-18-1017/quality-baseline/full.json`; quick baseline
   `quick.json`, release boot r1), plus its own A/A band from later 1017
   boots. A later pack built from the same 1017 source (a new allocation
   or encode) is a pack change and follows the paired rule above against
   `2.7bpw-mcg-hq`.
7. **Free the box between arms**: `./stop.sh`, confirm no GPU containers on
   either node, then boot the next arm. Never stack levers on a promoted
   arm's boot without re-running the full four numbers.
8. **Verdicts**: append to `flags.md` (per-round note), the arm's
   evidence directory (under `results/2026-10-18-1017/` or a new dated
   `results/` directory, with the evidence path in the flags.md note), and
   the `recipes` skill ledger. Record per-boot medians, the spread and the
   ms/step deltas, not only the pooled number.
   1017 reference cells (boot D1, one boot, the release defaults,
   `results/2026-10-18-1017/boot-d1/`; `bench_decode.py`, `benches/micro.py`
   and the fresh L.A.I.L, not `four_numbers.sh`): L.A.I.L prose c=1 (512
   tokens, t=0.2, median of 3) 39.6 tok/s at 52.5 ms/step, acceptance 2.09;
   bench_decode prose c=1 (x9, 200 tokens) 49.6 tok/s at 52.2 ms/step,
   acceptance 2.58; structured c=1 (x9) 77.0 tok/s at 51.4 ms/step,
   acceptance 3.96; prose c=2 aggregate (x5) 75.8; structured c=2 aggregate
   (x5) 140.3; pp_novel 4k/16k/64k 795/755/733 tok/s; pp_warm 4k/16k/64k
   707/689/688 tok/s; tg after 64k context 48.1 tok/s. The bench_decode
   prose prompt stops naturally at 83 tokens, so most of that cell is
   post-EOS text; the structured prompt ran to 200. D1 has no
   `four_numbers.sh` capture (pp 8k/32k, prose_long c=1/c=2, warm-prefix):
   the first 1017 ABAB records those on its A boots. A new ABAB always
   needs its own A boots. The Sept R33-R36 reference cells are history
   (Sept repo, `flags.md` and `results/RESULTS.md`).
   Warm-prefix hits depend on the prompt length: a repeat of N tokens
   missed entirely when N ran only 10-58 tokens past the last 128-token
   boundary, and hit at 68-127 (Sept `results/RESULTS.md` round 34).
   `tools/warm_prefix.py` now pads the prompt so that tail is at least 80
   tokens (`tail_tokens` in its summary). Read a 0.0 in older captures with
   that in mind.

## Exact restore sequence

Current serve (1017) = `dsv41-flash-exl3-sm121:e16-rows` on both nodes,
pack `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` @ `2.7bpw-mcg-hq`, served as
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, with the 4-session defaults:
`KV_CACHE_MEMORY=4160749568` (3.875 GiB), `MAX_MODEL_LEN=500000`,
`MAX_NUM_SEQS=4`, `DSV41_KV_PACKED_GROUPS=1`, `DSV41_SWA_META_BATCHED=1`,
`DSV41_INDEXER_EXPANDED_TABLE_LAZY=1`,
`MAX_NUM_BATCHED_TOKENS=8192`, DSpark k=3, `UTIL=0.75`,
`DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=4`, `--mm-processor-cache-gb 1`,
`--prefix-cache-retention-interval 4096`, and every round-34 and round-35
lever on (AGENTS.md, working rules). Boot I3 ran these defaults
(`results/2026-10-19-1017-perf2/I3/`); the D1/r1 release boots ran
`MAX_NUM_SEQS=2` at `MAX_MODEL_LEN=1048576` with the image's KV grouping.
Launched from this checkout:

```bash
./stop.sh                       # both nodes (stops spark2 via ssh)
AUDIT=strict ./run.sh
# readiness: python3 smoke_chat.py && python3 smoke_vision.py
# then: tools/four_numbers.sh --arm <label> to re-confirm baseline cells
```

Boot D1 ran `SKIP_DOWNLOAD=1 AUDIT=strict KV_CACHE_MEMORY=4160749568
./run.sh` (`results/2026-10-18-1017/boot-d1/command.txt`).
`results/2026-10-18-1017/boot.sh` exports the D1/r1 release env and then runs
`exec ./run.sh`; `results/2026-10-19-1017-c4/boot.sh` does the same for the
4-session defaults.

The mixed-K pack needs `e15-kbits` or newer (the default `e16-rows` adds the
9..24-row decode MoE). `canonical-e14` and older images have
no `expert_bits` support (the plugin ignores it), and the K-aware audit
fails `AUDIT=strict`. Do not roll back by image on 1017.

- `serve.sh` → `exec ./run.sh`; on the head with `ORCHESTRATE=auto` (default)
  and `NNODES=2`, run.sh starts the **worker on spark2 first** (scps itself to
  /tmp and `docker/patch` to `~/.cache/dsv41-patch` there, launches the worker
  container with the full env forwarded), waits for the worker's `headless
  multiproc executor` line (up to 120 s), then starts head rank 0 and `wait_ready`
  (`/health` + `/v1/models`; allow up to 60 min, plan ≥20). D1 booted in
  ~11 min (load 508 s head / 305 s worker).
- Lever rollbacks stay on `e16-rows`: set the lever to `0` (an empty value
  keeps the default). Round-35 kernel levers off (the round-34 config):
  ```bash
  DSV41_P2B_COOP=0 DSV41_DENSE_GEMV=0 DSV41_MHC_DET_SPLITS=0 \
    DSV41_ENGRAM_NATIVE_STAGE=0 DSV41_ENGRAM_EARLY_HASH=0 DSV41_ATTN_T2R_DEDUP=0 \
    DSV41_SWA_META_FUSED=0 DSV41_MOE_PREP_FUSED=0 DSV41_CANDIDATE_MASK_BOUNDED=0 \
    DSV41_INDEXER_WP_GEMV=0 DSV41_DSPARK_SPARSE_MARKOV_TOPK=256 AUDIT=strict ./run.sh
  ```
  Round-34 levers off as well: add `DSV41_ENGRAM_WILLNEED=0
  DSV41_STREAM_FEED=0 DSV41_WOA_PREPACK=0 DSV41_DSPARK_SPARSE_MARKOV=0`.
- Before the Hub revision is on a node, `tools/build_pack_1017.py assemble`
  writes the pack to `snapshots/2.7bpw-mcg-hq`; `run.sh` resolves
  `snapshots/<rev>/` when no `refs/<rev>` exists. If both exist, it uses the
  Hub commit snapshot.
- Sept restores (the Sept model, the 2.0 bpw packs, `canonical-e12`..`e14`,
  `boot-lm.sh`) run from the Sept recipe @ `3de9146`, not from this
  checkout: this repo's `MODEL`, served name, KV default and the required
  `chat_encoder_1017` patch are 1017-only.
- Patch-script changes (`docker/patch/engram_*`) take effect on restart
  without an image rebuild (patch dir is volume-mounted read-only).
