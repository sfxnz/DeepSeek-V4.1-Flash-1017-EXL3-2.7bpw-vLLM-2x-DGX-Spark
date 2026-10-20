#!/usr/bin/env python3
"""Source contracts of DSV41_ROUTER_GEMV and DSV41_MAIN_PROJ_OVERLAP against a vLLM tree (CPU, no imports).

  source_contract.py [VLLM_DIR]   (default: the image's /usr/local/lib/python3.12/dist-packages/vllm)
Checks, by reading the source text (ast), that the code the two levers wrap still has the shape they
assume. Prints one JSON line; exit 1 on any failed contract.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker" / "patch"))
import router_gemv as rg  # noqa: E402

DEFAULT = "/usr/local/lib/python3.12/dist-packages/vllm"


def method(path: Path, cls: str, name: str) -> str:
    src = path.read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return "".join(src.splitlines(keepends=True)[f.lineno - 1 : f.end_lineno])
    raise KeyError(f"{cls}.{name} not in {path}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    checks = {}

    def check(name, fn):
        try:
            checks[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001
            checks[name] = f"error: {exc!r}"

    gl = root / "model_executor/layers/fused_moe/router/gate_linear.py"
    fwd = lambda: method(gl, "GateLinear", "forward")  # noqa: E731
    check("gate: cuBLAS tier is torch.mm(x, self.weight.T, out_dtype=torch.float32)",
          lambda: "output = torch.mm(x, self.weight.T, out_dtype=torch.float32)" in fwd())
    check("gate: cuBLAS tier gated by allow_cublas_router_gemm and bf16 x",
          lambda: "if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:" in fwd())
    check("gate: tiers 1-3 use the flags router_gemv checks",
          lambda: all(f in fwd() for f in ("self.allow_ll_bf16_gemm and x.shape[0] <= 16",
                                           "if self.allow_fp32_router_gemm", "if self.allow_bf16x3_router_gemm")))
    attn = root / "models/deepseek_v4_1/attention.py"
    check("attention: kv_score anchor once in _run_parallel_input_projections",
          lambda: method(attn, "DeepseekV4Attention", "_run_parallel_input_projections").count(rg.KV_ANCHOR) == 1)
    moe = root / "models/deepseek_v4/nvidia/model.py"
    check("moe: the gate is GateLinear with out_dtype fp32",
          lambda: "self.gate = GateLinear(" in moe.read_text() and "out_dtype=torch.float32," in moe.read_text())
    runner = root / "v1/worker/gpu/model_runner.py"
    st = lambda: method(runner, "GPUModelRunner", "sample_tokens")  # noqa: E731
    check("runner: sample_tokens reads aux_hidden_states from execute_model_state, then samples, then proposes",
          lambda: (st().index("aux_hidden_states = self.execute_model_state.aux_hidden_states")
                   < st().index("self.sample(") < st().index("self.speculator.propose(")))
    check("runner: propose gets that aux_hidden_states list",
          lambda: "aux_hidden_states,\n" in st()[st().index("self.speculator.propose("):])
    lp = root / "model_executor/layers/logits_processor.py"
    gl_src = lambda: method(lp, "LogitsProcessor", "_get_logits")  # noqa: E731
    check("logits: _get_logits applies the head, then _gather_logits",
          lambda: gl_src().index("self._apply_head(") < gl_src().index("self._gather_logits(logits)"))
    dfl = root / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    check("dflash propose: combine_hidden_states(torch.cat(aux_hidden_states, dim=-1)) on the model",
          lambda: "hidden_states = self.model.combine_hidden_states(\n                torch.cat(aux_hidden_states, dim=-1)"
          in method(dfl, "DFlashSpeculator", "propose"))
    ds = root / "models/deepseek_v4_1/nvidia/dspark.py"
    check("dspark: DSparkDeepseekV4ForCausalLM.combine_hidden_states delegates to main_norm(main_proj(.))",
          lambda: "self.model.combine_hidden_states(aux_hidden_states)"
          in method(ds, "DSparkDeepseekV4ForCausalLM", "combine_hidden_states")
          and "self.main_norm(self.main_proj(aux_hidden_states))"
          in method(ds, "DSparkDeepseekV4Model", "combine_hidden_states"))
    out = {"vllm": str(root), "checks": checks, "ok": all(v is True for v in checks.values())}
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
