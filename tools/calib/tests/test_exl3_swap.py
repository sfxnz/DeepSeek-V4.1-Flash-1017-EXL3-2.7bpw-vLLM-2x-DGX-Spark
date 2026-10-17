"""EXL3 substitution path: reconstruct routed experts from an EXL3 pack and compare with the MXFP4
source they were encoded from. With the September 2.0bpw-mcg-viterbi pack vs the September source,
the recorded relerr is ~0.26 (results manifest), so this checks orientation, K detection, Hadamard and
scale handling of Exl3Pack.reconstruct. Needs GPU + exllamav3 + both checkpoints (spark2 has layers 20-39
of the September source); skipped otherwise. Env: CALIB_PACK, CALIB_PACK_SRC, CALIB_SWAP_LAYER.
"""

import os
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

HUB = Path.home() / ".cache/huggingface/hub"
PACK = Path(os.environ.get("CALIB_PACK", HUB / "models--sfxnz--DeepSeek-V4.1-Flash-EXL3/snapshots/2.0bpw-mcg-viterbi-lmhead-mxfp8"))
SRC = Path(os.environ.get("CALIB_PACK_SRC", HUB / "models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"))
LAYER = int(os.environ.get("CALIB_SWAP_LAYER", "20"))


def have():
    if not torch.cuda.is_available() or not (PACK / "config.json").exists():
        return False
    try:
        import exllamav3.ext  # noqa: F401
    except Exception:
        return False
    return (SRC / f"model-{LAYER + 3:05d}-of-00048.safetensors").exists()


@unittest.skipUnless(have(), "needs GPU, exllamav3, the Sept EXL3 pack and Sept source shard")
class Exl3Reconstruct(unittest.TestCase):
    def test_relerr_vs_source(self):
        import refkernel as rk
        from stio import Checkpoint, Exl3Pack
        pack, src = Exl3Pack(PACK), Checkpoint(SRC)
        cb = pack.codebook(LAYER)
        errs = []
        for e in (0, 1, 2, 191, 383):
            for p in ("w1", "w2", "w3"):
                stem = f"layers.{LAYER}.ffn.experts.{e}.{p}"
                T = pack.ckpt.read([f"{stem}.trellis", f"{stem}.suh", f"{stem}.svh"])
                Wq = pack.reconstruct(T[f"{stem}.trellis"], T[f"{stem}.suh"], T[f"{stem}.svh"], cb).float()
                S = src.read([f"{stem}.weight", f"{stem}.scale"])
                W = rk.dequant_fp4_weight(S[f"{stem}.weight"], S[f"{stem}.scale"], torch.float32)
                self.assertEqual(tuple(Wq.shape), tuple(W.shape))
                err = ((Wq - W).norm() / W.norm()).item()
                errs.append(err)
                print(f"{stem} K={pack.k_of(LAYER, e, p)} cb={cb} relerr={err:.4f}")
        mean = sum(errs) / len(errs)
        print(f"mean relerr {mean:.4f}")
        self.assertTrue(0.2 < mean < 0.32, mean)


if __name__ == "__main__":
    unittest.main()
