#!/usr/bin/env python3
"""CPU check of the installed (e15-kbits) vllm_exl3 plugin: expert_bits -> Exl3MoEMethod -> param shapes.

  docker run --rm --network none --entrypoint python3 -v $PWD:/repo -w /repo dsv41-flash-exl3-sm121:e15-kbits \
      kernel_study/p2b_kbits/plugin_check.py

Builds Exl3Config.from_config with expert_bits, the method get_quant_method would build for a few layer
prefixes, runs create_weights on a bare nn.Module at the V4.1 TP2 shapes, and loads one gate and one down
trellis of the right K through _load_exl3 (and checks a wrong-K down trellis is refused).
"""

from __future__ import annotations

import torch

import vllm_exl3.exl3 as P

HIDDEN, INTER_LOCAL, E = 5120, 1152, 4


def main() -> int:
    cfg = P.Exl3Config.from_config({"quant_method": "exl3", "bits": 2, "codebook": "mcg", "layer_bits": {"9": 4},
                                    "expert_bits": {"0": [3, 2], "1": [2, 4], "2": [4, 3]}})
    ok = True
    for prefix, want in (("model.layers.0.ffn.experts", (3, 2)), ("model.layers.1.ffn.experts", (2, 4)),
                         ("model.layers.2.ffn.experts", (4, 3)), ("model.layers.9.ffn.experts", (4, 4)),
                         ("model.layers.5.ffn.experts", (2, 2))):
        kgu, kd = cfg.expert_bits_for_prefix(prefix)
        m = P.Exl3MoEMethod.__new__(P.Exl3MoEMethod)  # FusedMoEMethodBase.__init__ needs a vLLM moe config
        m.quant_config, m.bits, m.bits_d, m._logged = cfg, kgu, kd, False
        layer = torch.nn.Module()
        m.create_weights(layer, E, HIDDEN, INTER_LOCAL, torch.bfloat16)
        shapes = (tuple(layer.w13_trellis.shape), tuple(layer.w2_trellis.shape))
        exp = ((E, 2, HIDDEN // 16, INTER_LOCAL // 16, 16 * kgu), (E, INTER_LOCAL // 16, HIDDEN // 16, 16 * kd))
        good = (kgu, kd) == want and shapes == exp and (layer._exl3_bits, layer._exl3_bits_d) == (kgu, kd)
        # one expert's gate (full tensor: in_tiles x out_tiles_full x 16K) and down through the loader
        g = torch.zeros(HIDDEN // 16, 2 * INTER_LOCAL // 16, 16 * kgu, dtype=torch.int16)
        d = torch.zeros(2 * INTER_LOCAL // 16, HIDDEN // 16, 16 * kd, dtype=torch.int16)
        for param, t, sid, name in ((layer.w13_trellis, g, "w1", "experts.w13_trellis"),
                                    (layer.w2_trellis, d, "w2", "experts.w2_trellis")):
            param._exl3_owner = None
            param.tp_rank, param.tp_size = 0, 2
            m._load_exl3(param, t, name, shard_id=sid, expert_id=1)
        refused = False
        bad = torch.zeros(2 * INTER_LOCAL // 16, HIDDEN // 16, 16 * (kd + 1 if kd < 4 else 2), dtype=torch.int16)
        try:
            m._load_exl3(layer.w2_trellis, bad, "experts.w2_trellis", shard_id="w2", expert_id=1)
        except RuntimeError as exc:
            refused = "shape mismatch" in str(exc)
        good &= refused
        ok &= good
        print(f"{prefix}: K_gu={kgu} K_d={kd} w13 {shapes[0]} w2 {shapes[1]} wrong-K down refused={refused} "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("plugin_check:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
