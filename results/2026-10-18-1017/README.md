# 1017 release evidence (2026-10-18)

Pack `sfxnz/DeepSeek-V4.1-Flash-1017-EXL3` @ `2.7bpw-mcg-hq`, image `dsv41-flash-exl3-sm121:e15-kbits`. The README and `flags.md` cite these files.

| Path | Contents |
|---|---|
| `boot.sh` | Promoted boot: exports the release serve env (equal to the `recipe.yaml` defaults), then `exec ./run.sh`. `tests/test_recipe_ops.py` checks it against a plain `./run.sh`. |
| `boot-d1/` | Release boot D1 (`command.txt`): boot and audit (`boot.txt`, `head.log`, `worker.log`), smokes, `mem-after-smoke.txt`, `bench_decode.txt` (c=1), `bench_decode_c2.txt` (c=2), `micro.txt` (prefill / tg cells), `e2e.txt` (e2e and L.A.I.L prose), `quality-full.{json,txt}`. |
| `allocation/` | K allocation: per-group sensitivity curves, r_eff calibration, the ensemble allocation (`alloc_2.70_merged.json`, `ALLOC_FINAL_2.70.json`) and the joint validation (`joint_eval.json`, `joint_s3.json`, `joint_s4.json`). |
| `encode/` | Real-Hessian encode: build manifest, per-unit records (`units.json`) and per-K gate limits (`gates.json`). |
| `quality-baseline/` | `full.json`: the `quality_eval.py --full` baseline for `2.7bpw-mcg-hq`, recorded on boot D1. `quick.json`: TBD-PHASED(quick baseline). |

The September recipe's evidence (Sept model, 2.0 bpw packs, rounds 1-36) is in https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ `3de9146`.
