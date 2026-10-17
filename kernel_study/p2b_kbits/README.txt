p2b K-bits (1017 pack: routed experts at K in {2, 3, 4} per (layer, gate/up | down))
====================================================================================

What
----
docker/patch/widen_p2b_kbits.py (after widen_p2b_dataflow.py; docker/Dockerfile.e15 FROM canonical-e14,
image dsv41-flash-exl3-sm121:e15-kbits):
- kernels: BITS = K (2, 3, 4) or 10 * K_gu + K_d (23 24 32 34 42 43). The p2b (SORT=0/1), coop
  (DSV41_P2B_COOP=1, SORT=2) and dataflow (=2) kernels run gate/up tiles at K_gu and down tiles at K_d;
  p2b_df_tile / p2b_df_prefetch_tile are templated on K (run_gemv_tile_coop's K lines). Coop and dataflow
  are selected for every MCG K (was K = 2 only). Host: K_gate == K_up, K_d free, mixed needs MCG.
  The 14 kernels of canonical-e14 are byte-identical (shared arrays are per BITS so the K = 2 kernels keep
  their kernel-local shared layout). 28 new kernels. The dataflow kernel runs K != 2 at 2 blocks/SM
  (spill-free MMA loops; ab_lb.py: 0.4-10.6 % faster than 3 blocks/SM, bitwise equal).
- plugin: quantization_config.expert_bits {"<layer>": [K_gu, K_d]} -> w13 at K_gu, w2 at K_d words;
  native p2b gets (K_gu, K_gu, K_d); exl3_moe gets the same (its K = 0 instance switches K at runtime);
  the fat prefill path uses each projection's own K (exl3_fat_gemm only where that K is 4).

Files
-----
make_bench_kbits.py     CPU. build/chain_r3.cu (canonical-e14 TU), build/chain_kbits.cu (e15 TU),
                        build/bench_kbits.cu (+ make_bench_r3 runtime variant switch), bench_kbits_lb3.cu
                        (bench-only: dataflow at __launch_bounds__(256, 3) for every K; ab_lb.py compares).
sass_identity_kbits.sh  CPU. e14 TU vs e15 TU machine code (14 identical, 28 new), ptxas regs / spills /
                        stack of every p2b kernel, inner MMA-loop LDL/STL (hot_loops.py).
encode_layer.py         GPU, dsv41-quant151. Layer 20 TP-rank-0 experts encoded at K (MCG, H = I,
                        requant_full's per-tensor path) -> /data/tensors/L20_tp0_K<K>.safetensors.
bench_kbits.py          GPU. check / capture / stress / time per (K_gu, K_d) (bench_r3's gates).
prefill_kbits.py        GPU, e15 image. exl3_moe + fat path through the installed plugin, m = 512..8192.
real_ext_kbits.py       GPU, e15 image. installed vllm_exl3_c == bench module, per DSV41_P2B_COOP value.
plugin_check.py         CPU, e15 image. expert_bits -> create_weights shapes -> _load_exl3.
spark2.sh               sync to spark2, run one step under /home/sfxnz/projects/data/dsv41-1017/spark2-gpu.lock
                        in a --rm dsv41-1017-kb-<tag> container; raw logs in
                        /home/sfxnz/projects/data/dsv41-1017/kernels/runs/.
prebuild_spark2.sh      CPU on spark2: JIT-build the bench modules outside the GPU window.

Reproduce
---------
  kernel_study/p2b_kbits/sass_identity_kbits.sh dsv41-flash-exl3-sm121:canonical-e14
  for K in 2 3 4; do kernel_study/p2b_kbits/spark2.sh enc-L20r0-K$K dsv41-quant151 kernel_study/p2b_kbits/encode_layer.py \
      --src /srcrepo/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 --layer 20 --tp-rank 0 --bits $K --out /data/tensors; done
  kernel_study/p2b_kbits/prebuild_spark2.sh
  IMG=dsv41-flash-exl3-sm121:e15-kbits
  kernel_study/p2b_kbits/spark2.sh kb-validate $IMG kernel_study/p2b_kbits/bench_kbits.py --mode check,capture,stress \
      --check-routings 8 --ref --replays 32 --stress-calls 2000 --out /data/runs/kb-validate.json
  kernel_study/p2b_kbits/spark2.sh kb-time-lb3 $IMG kernel_study/p2b_kbits/bench_kbits.py --mode time --iters 300 \
      --out /data/runs/kb-time-lb3.json
  kernel_study/p2b_kbits/spark2.sh kb-prefill $IMG kernel_study/p2b_kbits/prefill_kbits.py --out /data/runs/kb-prefill.json
  kernel_study/p2b_kbits/spark2.sh kb-real-ext $IMG kernel_study/p2b_kbits/real_ext_kbits.py --out /data/runs/kb-real-ext.json
