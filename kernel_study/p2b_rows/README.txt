Decode kernels past 8 rows (perf2 c4-kernels; image e16-rows = docker/Dockerfile.e16 on e15-kbits)
=============================================================================================

Why
---
Four sessions at DSpark k = 3 verify 16 rows (4 x (k + 1)) and draft 12; adaptive depth up to k = 5 verifies
24. Every fast decode path stopped at 8 rows (perf2/profile-synthesis/report.md 1.6):
  routed MoE   vllm_exl3 native contract 8 rows; dataflow kernel m * top_k <= 64   -> exl3_moe / exl3_moe_kmix
  dense MXFP8  DENSE_GEMV M <= 8                                                   -> b12x (32, 128) tiles (as fast)
  indexer wp   INDEXER_WP_GEMV M 2..8                                              -> cuBLAS
  mHC          MHC_DET T <= 16                                                      -> stock TileLang + DeepGEMM
  cand. mask   no row limit (row-parallel kernels); self-test covered rows 1/4/8 only

What changed (all bit-exact per row against the m <= 8 kernels, existing kernels untouched)
----------------------------------------------------------------------------------------
docker/patch/widen_p2b_rows.py (build time, e16): p2b_coop_dfw_kernel<BITS, 1> = p2b_coop_df_kernel's text
  with the sort / chunk arrays, unit count and control words sized by 144 pairs (24 rows x top-6) and its own
  shared arrays; the same tiles (p2b_df_tile), epilogue, slot sum and 2 grid barriers. Host: 1..24 rows;
  DSV41_P2B_COOP=2 and m * top_k in (64, 144] launches it (m 9..10 still fit the e15 dataflow kernel's 64
  pairs). Plugin: rows up to vllm_exl3_c.P2B_WIDE_ROWS go native for MCG layers under DSV41_P2B_COOP=2
  unless DSV41_P2B_WIDE=0 (run.sh forwards it; unset = on). Chunks keep <= 8 rows (MMA rows 0..7): a
  16-row chunk tile would need the MMA's upper half and 32 KiB of reduction smem per block for +0.4 % B
  at m = 16 (census below).
docker/patch/indexer_wp_gemv.py (runtime): rows 9..12 behind their own self-test (cuBLAS keeps the GEMV's 16x16
  wmma chain only up to M 12; from M 13 it runs a faster kernel with other bits, 12.3 us at M 16, and stays stock).
docker/patch/mhc_det.{cu,py} (runtime): T 17..24 via mhc_det_gemm_t24 (rows 16..23 in a second m16 tile), behind
  their own self-tests (a mismatch keeps only T > 16 on stock);
  t8 / t16 instantiate gemm_body as before (ROW2 = false code is compile-time dead).
docker/patch/candidate_mask_bounded.py, moe_prep_fused.py (runtime): self-test rows / m up to 24.
Dense: no change. Measured 2026-10-19 (dense_gemv/rows16_check.py, 1017 weights, cold): b12x at its e15 (32, 128)
  tile runs 9..24 rows as fast as m <= 8 (qkv_a 48.2 us at M 16 vs 48.0 at M 8; wo_b 103.2 vs 103.4; gate_up 60.4 vs
  59.4), and keeping the 16-row decode tiles up to m = 16 changed nothing (-2.4..+2.0 %, bitwise): that patch was
  dropped. The dense GEMV is 3.6-8.4 % faster than b12x at M 8 only; it stages the whole M x K activation per CTA
  (80 KiB at M 16, K 5120), and two <= 8-row launches lose 19-65 %. An MR = 16 GEMV for K <= 4096 (wq_b, wo_b /
  wo_a, shared down) would gain ~0.8 ms/step at c = 4. Marlin MXFP8 exists in e15 (mxfp8/marlin.py) but is W8A16,
  not bit-exact with the served W8A8 path.

Census (kernel_study/p2b_rows/census_rows.py, s10 decode census, 6 captures, S sequences x T tokens)
  m  4 (1 x 4)  distinct 16.7  chunks<=8 16.7
  m  8 (2 x 4)  distinct 31.2  chunks<=8 31.2
  m 12 (3 x 4)  distinct 44.7  chunks<=8 44.8  (+0.2 % B)
  m 16 (4 x 4)  distinct 56.1  chunks<=8 56.3  (+0.4 % B)   0.23 experts/layer with > 8 rows
  m 24 (4 x 6)  distinct 72.5  chunks<=8 73.7  (+1.7 % B)   1.17 experts/layer with > 8 rows

Measured 2026-10-19 on spark2 (image e16-rows, layer-20 experts at each (K_gu, K_d), c4_all routing, CUDA-graph
replays, cold L2, 200 iterations, SM 2190 MHz; perf2/c4-kernels/runs/rows-time*.json). Per step = 40 layers weighted
by the 2.70 allocation (7 x 2/2, 8 x 2/3, 24 x 3/3, 1 x 3/4), ms:
       m (seqs x rows)   dataflow (df)   plugin native   fallback (exl3_moe/kmix)   fallback - df
       4  (1 x 4)            18.6            19.6             36.7                    18.1   (m <= 8: df is today's path)
       8  (2 x 4)            33.7            34.7             68.7                    35.0
      12  (3 x 4)            48.3            49.0             91.3                    43.0
      12  (2 x 6)            43.8            44.5             81.3                    37.5   (c = 2 at DSpark k = 5)
      16  (4 x 4)            59.0            59.9            112.7                    53.7
      24  (4 x 6)            77.4            78.1            146.3                    69.0
  The wide kernel streams 227-233 GB/s (91-93 % of 250) at m 12..24, as the m <= 8 kernel; the fallback 93-145 GB/s.
  The model above m = 8 (dataflow fit to m 4/6/8, fallback per active-expert tile) predicted 59.3 / 95-108 at m = 16.
  Serve check: the c = 4 baseline (perf2/baseline, B0) measured routed_moe_exl3 at 98.4 ms/step (2.46 ms/call), 0.88x
  this bench's fallback; at c = 1 the served p2b was 1.09x the bench's df. Projected c = 4 (k = 3) step: 149.4 ms minus
  37..50 ms -> ~100-112 ms.

GPU plan (spark2 assigned; every GPU step under flock spark2-gpu.lock via kernel_study/p2b_kbits/spark2.sh,
--rm, memory-capped, network none; logs in /home/sfxnz/projects/data/dsv41-1017/kernels/runs/, copy the
verdicts to perf2/c4-kernels/). No serve may run on spark2 meanwhile.
 0  CPU, no GPU (any node): SASS identity, ~2 min
      kernel_study/p2b_rows/sass_identity_rows.sh dsv41-flash-exl3-sm121:e15-kbits      # 42 identical, 9 new, regs/spills
      git show 20ee8ba:docker/patch/mhc_det.cu > /tmp/mhc_det.e15.cu
      docker run --rm --network none --memory 4g --user 1000:1000 -v "$PWD":/repo -v /tmp/mhc_det.e15.cu:/old.cu:ro \
        -w /repo --entrypoint python3 dsv41-flash-exl3-sm121:e15-kbits \
        kernel_study/mhc_det/sass_identity_t24.py /old.cu docker/patch/mhc_det.cu    # t8/t16/post/norm identical, t24 new
 1  build e16 on spark2 (CPU, ~4 min; repeat on spark1 before a TP=2 boot):
      rsync the worktree to spark2:/home/sfxnz/projects/ai-lab/recipes/.worktrees-spark2/dsv41-1017-c4-kernels, then there
      docker build --memory 24g --memory-swap 24g -f docker/Dockerfile.e16 -t dsv41-flash-exl3-sm121:e16-rows \
        --build-arg DSV41_GIT_REV=$(git rev-parse HEAD) \
        --build-arg DSV41_TREE_DIFF=$(git diff HEAD --binary -- docker | sha256sum | cut -c1-64) docker
      (the RUN asserts the 9 wide kernels, P2B_WIDE_ROWS == 24 and the plugin helper)
 2  kernel_study/p2b_rows/prebuild_spark2.sh dsv41-flash-exl3-sm121:e16-rows     # CPU JIT of bench_rows, ~3-4 min
 3  MoE gates, ~15 min:
      kernel_study/p2b_kbits/spark2.sh rows-validate dsv41-flash-exl3-sm121:e16-rows kernel_study/p2b_rows/bench_rows.py \
        --pairs 2:2,2:3,3:3,3:4 --mode check,capture,stress --ref --check-routings 8 --replays 32 --stress-calls 500 \
        --out /data/runs/rows-validate.json
      pass = batch invariance (m rows == 8- and 4-row slices, bitwise), one-hot == p2b, real <= 1e-3, repeat,
      graph replay == eager (bench v2 and the installed plugin's native path, which must also equal the bench
      module bitwise), 500-call stress bitwise; the fallback's distance to v2 and both fp64 errors are reported.
 4  MoE timing, ~6 min:
      kernel_study/p2b_kbits/spark2.sh rows-time dsv41-flash-exl3-sm121:e16-rows kernel_study/p2b_rows/bench_rows.py \
        --pairs 2:2,2:3,3:3,3:4 --mode time --time-ms 4,8,12,16,24 --source c4_all --iters 200 \
        --out /data/runs/rows-time.json
      (CUDA-graph replays of p2b, df, fallback, native; "projection" = ms/step per arm and m for the allocation)
 5  dense b12x rows16, ~10 min (e15-kbits is enough: both tilings are produced in-process):
      kernel_study/p2b_kbits/spark2.sh rows16 dsv41-flash-exl3-sm121:e16-rows kernel_study/dense_gemv/rows16_check.py \
        --pack /hf1017/snapshots/2.7bpw-mcg-hq --out /data/runs/rows16.json
 6  runtime levers, ~10 min total:
      kernel_study/p2b_kbits/spark2.sh wp-wide dsv41-flash-exl3-sm121:e16-rows kernel_study/fusion_host/wp_gemv_bench.py \
        --wide --snap /hf1017/snapshots/2.7bpw-mcg-hq
      kernel_study/p2b_kbits/spark2.sh cmask-rows dsv41-flash-exl3-sm121:e16-rows kernel_study/fusion_host/candidate_mask_check.py
      kernel_study/p2b_kbits/spark2.sh prep-rows dsv41-flash-exl3-sm121:e16-rows kernel_study/fusion_host/moe_prep_check.py
      kernel_study/p2b_kbits/spark2.sh mhc-t24 dsv41-flash-exl3-sm121:e16-rows kernel_study/mhc_det/bench_gemm.py --mode check
      kernel_study/p2b_kbits/spark2.sh mhc-edge dsv41-flash-exl3-sm121:e16-rows kernel_study/mhc_det/edge_check.py
      kernel_study/p2b_kbits/spark2.sh mhc-smoke dsv41-flash-exl3-sm121:e16-rows kernel_study/mhc_det/lever_smoke.py
 7  serve (main agent, both nodes, IMAGE=dsv41-flash-exl3-sm121:e16-rows, MAX_NUM_SEQS=4 under FORCE_UNSAFE_CTX):
      boot logs: "p2b coop dataflow wide kernel engaged", "indexer weights_proj GEMV wide rows self-check
      bit-exact", "mhc det engaged ... T <= 24"; quick quality baseline; ABAB DSV41_P2B_WIDE=0 vs unset at c = 4.
Files: make_bench_rows.py (CPU: e15 / e16 TUs + bench module), sass_identity_rows.sh, bench_rows.py,
census_rows.py, prebuild_spark2.sh; tests/test_widen_p2b_rows.py (host).
