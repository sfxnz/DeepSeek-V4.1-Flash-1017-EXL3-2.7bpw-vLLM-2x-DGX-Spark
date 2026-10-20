"""DeepseekV4Model.forward stops keeping tensors it no longer needs (DSV41_MODEL_LOOP_RELEASE=1). Bit-identical.

The layer loop of models/deepseek_v4_1/nvidia/model.py calls every decoder layer as
`layer(hidden_states, ..., residual, ...)`. For the whole call the loop's own names and
nn.Module.__call__'s argument tuples keep the previous layer's outputs alive: the hc residual
[T, 4, 5120] bf16 (320 MiB per rank at T = 8190) and the FFN output [T, 5120] bf16 (80 MiB).
The layer consumes them in its first mHC post and pre, so at its Engram and attention peaks they
are dead. Rewrite H1 hands them over in a list the call empties and calls layer.forward directly
when the layer and torch have no module hooks and the layer is not compiled in place
(nn.Module._call_impl adds nothing then; otherwise the stock call runs). CPython 3.11+ moves call arguments into the callee's frame, so
the layer's own rebinding frees them: -400 MiB at every layer's peak after layer 0, -80 MiB at
layer 0 (the embedding). Rewrite H2: after a DSpark tap (layers 37-39) the loop kept the tap's
[T, 4, 5120] recon in `final_aux_recon` (and `aux_recon`) to the end of the forward, but only
reuses it when the end layer is a tap; it keeps it only then (-320 MiB from layer 37 on at
T = 8190 with decoder replay off; with replay the taps of long prompts run on the kept rows).
Same calls, same order, same values; only lifetimes change.

H1 and H2 sit inside the loop body; decoder_replay.py's anchors (the loop header and the
`if layer is not None:` tail) are outside them, and it reads this file's text, so it composes.
Applied from sitecustomize at interpreter start, before anything imports vLLM. Each hunk must
match exactly once. Idempotent (MARK). DSV41_MODEL_LOOP_RELEASE=0 keeps the image's loop and puts
it back in a container whose file was already rewritten. Top-level imports are stdlib only.
"""

from __future__ import annotations

import os
import sys

MARK = "dsv41-model-loop-release"
LOG_ENGAGED = "dsv41: model loop release engaged"
LOG_DISARMED = "dsv41: model loop release FAILED"
REL = "models/deepseek_v4_1/nvidia/model.py"
MODULE = "vllm.models.deepseek_v4_1.nvidia.model"

H1_OLD = """            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
            )
"""
H1_NEW = """            # dsv41-model-loop-release: the layer gets the previous layer's outputs from a list the
            # call empties, so its first mHC post / pre free them instead of this frame keeping them.
            _dsv41_io = [hidden_states, residual]
            hidden_states = residual = None
            hidden_states, residual, post_mix, res_mix, pre_mix = _dsv41_plain_call(layer)(
                _dsv41_io.pop(0),
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                _dsv41_io.pop(),
                engram_hashes,
                engram_mask,
            )
"""
H2_OLD = """                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
"""
H2_NEW = """                aux_hidden_states.append(aux_hidden_state)
                # dsv41-model-loop-release: the tail reuses the recon only when the end layer is a tap.
                final_aux_recon = aux_recon if self.end_layer in self.aux_hidden_state_layers else None
                aux_recon = None
"""
HELPER = '''

# dsv41-model-loop-release: layer.forward when nothing hooks or compiles the module (nn.Module._call_impl
# then only forwards the call, keeping its argument tuples alive meanwhile), else the module itself.
def _dsv41_plain_call(module):
    import torch.nn.modules.module as _mm

    if (
        getattr(module, "_compiled_call_impl", None) is not None
        or module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or _mm._global_forward_hooks
        or _mm._global_forward_pre_hooks
        or _mm._global_backward_hooks
        or _mm._global_backward_pre_hooks
    ):
        return module
    return module.forward
'''
HUNKS = ((H1_OLD, H1_NEW, "H1 layer call"), (H2_OLD, H2_NEW, "H2 aux recon"))


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("DSV41_MODEL_LOOP_RELEASE", "0") == "1"


def patch_source(src: str) -> str:
    """The image's model.py with H1 and H2 applied and the helper appended (idempotent)."""
    if MARK in src:
        return src
    for old, new, label in HUNKS:
        if src.count(old) != 1:
            raise ValueError(f"{label}: anchor found {src.count(old)} times in {REL}, need 1")
        src = src.replace(old, new)
    if "def _dsv41_plain_call(" in src:
        raise ValueError("_dsv41_plain_call already defined")
    return src + HELPER


def install(vllm_root, rewrite, modules=None) -> str:
    """Rewrite model.py with sitecustomize's `rewrite(path, fn)`; print the engagement line."""
    modules = sys.modules if modules is None else modules
    try:
        done = rewrite(os.path.join(str(vllm_root), REL), patch_source)
        if done == "rewritten" and MODULE in modules:
            raise RuntimeError(f"{MODULE} was imported before the rewrite; this process runs the stock loop")
    except BaseException as exc:
        print(f"dsv41: model loop release FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
    print(f"dsv41: model loop release engaged (model.py {done})", file=sys.stderr, flush=True)
    return f"model.py {done}"


def unpatch_source(src: str) -> str:
    """The image's text back from a rewritten model.py (inverse of patch_source)."""
    if MARK not in src:
        return src
    if not src.endswith(HELPER):
        raise ValueError(f"{REL} carries {MARK} but does not end with the helper")
    src = src[: -len(HELPER)]
    for old, new, label in HUNKS:
        if src.count(new) != 1:
            raise ValueError(f"{label}: rewritten text found {src.count(new)} times in {REL}, need 1")
        src = src.replace(new, old)
    return src


def restore(vllm_root, rewrite) -> str:
    """DSV41_MODEL_LOOP_RELEASE=0: put the image's loop back if this container's model.py was rewritten."""
    return rewrite(os.path.join(str(vllm_root), REL), unpatch_source)
