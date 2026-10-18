#!/usr/bin/env python3
"""EXL3 mixed-pack metadata for DeepSeek-V4.1-Flash.

Routed experts are EXL3 trellis. Everything else stays in the official
MXFP8/MXFP4/BF16 layout, including Engram tables (read from disk at serve).
DSpark draft experts stay source-format from backbone layer 40.

Codebook is a named pair (marker suffix, p2b cb). MCG is cb=1. MUL1 is cb=2.
The Sept rebuild tools default to MUL1 and p2b cb=2 exists so a later pack
can use it. spark1+spark2 measured MUL1 + cb=2 at 23.52 vs 27.98 prose
decode. The 1017 pack (tools/build_pack_1017.py) is MCG.
A pack-only swap without cb=2 drops native p2b onto generic exl3_moe.

1017 mixed-K routed experts: quantization_config["expert_bits"] maps
"<layer 0..39>" -> [K_gu, K_d]; K_gu is the K of w1 and w3, K_d of w2, each
in {2, 3, 4}. The serving plugin honors it and falls back to layer_bits /
bits when it is absent. DSpark/MTP experts stay source MXFP4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BACKBONE_LAYERS = 40
DEFAULT_BITS = 2


@dataclass(frozen=True)
class Codebook:
    name: str
    cb: int
    suffix: str
    quant_key: str


CODEBOOKS = {
    "mcg": Codebook("mcg", 1, "mcg", "mcg"),
    "mul1": Codebook("mul1", 2, "mul1", "mul1"),
}
DEFAULT_CODEBOOK = "mul1"
# 1017 pins. SERVE_REVISION is the published serve revision.
SOURCE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SOURCE_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SOURCE_SNAPSHOT_REL = (
    "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/" + SOURCE_REVISION
)
PACK_REPO = "sfxnz/DeepSeek-V4.1-Flash-1017-EXL3"
SERVE_REVISION = "2.7bpw-mcg-hq"
EXPERT_K = (2, 3, 4)


def get_codebook(name: str) -> Codebook:
    key = str(name).strip().lower()
    found = CODEBOOKS.get(key)
    if found is None:
        raise ValueError(f"unsupported codebook={name}")
    return found


def revision_for(bits: int = DEFAULT_BITS, codebook: str = DEFAULT_CODEBOOK) -> str:
    get_codebook(codebook)
    return f"{int(bits)}.0bpw-{codebook}"


def is_routed_expert_tensor(name: str) -> bool:
    """True for backbone routed-expert w1/w2/w3 weight or scale tensors."""
    if ".ffn.shared_experts." in name or ".engram." in name:
        return False
    if ".ffn.experts." not in name:
        return False
    if name.startswith("mtp.") or ".mtp." in name:
        return False
    return name.endswith(
        (".w1.weight", ".w2.weight", ".w3.weight", ".w1.scale", ".w2.scale", ".w3.scale")
    )


def is_routed_expert_weight(name: str) -> bool:
    return is_routed_expert_tensor(name) and name.endswith(".weight")


def validate_expert_bits(expert_bits: dict) -> dict[str, list[int]]:
    """Normalize {"<layer>": [K_gu, K_d]} over backbone layers 0..39 (all required)."""
    out: dict[str, list[int]] = {}
    for key, pair in expert_bits.items():
        layer = int(key)
        if not 0 <= layer < BACKBONE_LAYERS or str(layer) != str(key).strip():
            raise ValueError(f"expert_bits: bad layer key {key!r}")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"expert_bits[{key}]: want [K_gu, K_d], got {pair!r}")
        ks = [int(k) for k in pair]
        if any(k != v or k not in EXPERT_K for k, v in zip(ks, pair)):
            raise ValueError(f"expert_bits[{key}]: K must be in {EXPERT_K}, got {pair!r}")
        out[str(layer)] = ks
    missing = [str(n) for n in range(BACKBONE_LAYERS) if str(n) not in out]
    if missing:
        raise ValueError(f"expert_bits: missing layers {missing[:8]}")
    return {str(n): out[str(n)] for n in range(BACKBONE_LAYERS)}


def layer_bpw(k_gu: int, k_d: int) -> float:
    """Trellis bits per routed-expert weight of a layer: w1, w3 at K_gu and w2 at K_d."""
    return (2 * k_gu + k_d) / 3


def routed_bpw(expert_bits: dict) -> float:
    """Param-weighted mean trellis bpw over the backbone's routed experts."""
    eb = validate_expert_bits(expert_bits)
    return sum(layer_bpw(*eb[str(n)]) for n in range(BACKBONE_LAYERS)) / BACKBONE_LAYERS


def build_quantization_config(
    *,
    bits: int = DEFAULT_BITS,
    codebook: str = DEFAULT_CODEBOOK,
    layer_bits: dict[str, int] | None = None,
    expert_bits: dict | None = None,
) -> dict[str, Any]:
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError(f"unsupported EXL3 bits={bits}")
    cb = get_codebook(codebook)
    cfg: dict[str, Any] = {
        "quant_method": "exl3",
        "bits": int(bits),
        "codebook": cb.name,
        "mtp_experts": "source",
        "mtp_experts_start_layer": BACKBONE_LAYERS,
        # Copied onto Exl3Config so DSV4.1's scale mapper uses MXFP8
        # ``weight_scale`` (not block-FP8 ``weight_scale_inv``).
        "weight_block_size": [32, 32],
        # Not bf16_as_stored: attention/shared-expert tensors stay official
        # MXFP8 (ue8m0, 32x32). vllm-exl3 delegates those to DeepseekV4FP8Config.
        "non_routed_quantization": {
            "quant_method": "deepseek_v4_fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32],
            "expert_dtype": "fp4",
        },
    }
    if layer_bits:
        cfg["layer_bits"] = {str(k): int(v) for k, v in layer_bits.items()}
    if expert_bits is not None:
        eb = validate_expert_bits(expert_bits)
        cfg["expert_bits"] = eb
        # Fallback for a loader without expert_bits: uniform layers only. A
        # K_gu != K_d layer is left to `bits`, which then fails the loader's
        # shape check loudly instead of serving the wrong K.
        lb = dict(cfg.get("layer_bits") or {})
        lb.update({k: v[0] for k, v in eb.items() if v[0] == v[1]})
        if lb:
            cfg["layer_bits"] = {k: lb[k] for k in sorted(lb, key=int)}
        cfg["routed_bpw"] = round(routed_bpw(eb), 6)
    return cfg


def apply_pack_config(config: dict[str, Any], quant: dict[str, Any]) -> dict[str, Any]:
    """Write EXL3 quantization_config onto a DeepSeek-V4.1 config.json object."""
    out = dict(config)
    out["quantization_config"] = quant
    # Keep nested text_config's architecture fields; do not flatten.
    return out
