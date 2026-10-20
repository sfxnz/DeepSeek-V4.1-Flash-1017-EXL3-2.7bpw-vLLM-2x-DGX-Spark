"""docker/patch/model_loop_release.py on the pinned e15 model.py (CPU only, no torch).

tests/fixtures/dsv41_model_e13.pin.py is the image's models/deepseek_v4_1/nvidia/model.py (the e15 file is
the same text). Besides the anchors, the loop the rewrite produces is executed with stand-in layers whose
forward rebinds its arguments the way DeepseekV4DecoderLayer.forward does, and weak references show when
the previous layer's outputs die: during the layer call with the rewrite, only after it without.
"""

from __future__ import annotations

import contextlib
import gc
import io
import itertools
import os
import sys
import tempfile
import textwrap
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch"
PIN = ROOT / "tests/fixtures/dsv41_model_e13.pin.py"
sys.path.insert(0, str(PATCH))

import model_loop_release as mlr  # noqa: E402

# decoder_replay.py's anchors in the same forward (its loop header and tail must stay verbatim).
REPLAY_LOOP = """        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
"""
REPLAY_TAIL = """        if layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
"""


def forward_source(src: str) -> str:
    start = src.index("    def forward(\n        self,\n        input_ids: torch.Tensor,")
    end = src.index("\n    def load_weights(", start)
    return src[start:end]


class Obj:
    """A stand-in tensor: weak-referenceable, `.mean(dim=1)` gives a new one."""

    def __init__(self, tag):
        self.tag = tag

    def mean(self, dim=None):
        return Obj(f"mean({self.tag})")


class FakeLayer:
    """Rebinds `residual` then `x` like DeepseekV4DecoderLayer.forward's mHC post / pre, and records whether
    the objects it was given are already dead at its 'attention' (after the rebinding)."""

    def __init__(self, i, record, hooks=False, compiled=False):
        self.i, self.record = i, record
        self._forward_hooks = {1: object()} if hooks else {}
        if compiled:  # nn.Module.compile() sets it; __call__ then runs the compiled module
            self._compiled_call_impl = self.__call__
        self._forward_pre_hooks, self._backward_hooks, self._backward_pre_hooks = {}, {}, {}

    def __call__(self, *args, **kwargs):  # nn.Module.__call__ keeps its argument tuple meanwhile
        return self.forward(*args, **kwargs)

    def forward(self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask):
        wx = weakref.ref(x)
        wr = weakref.ref(residual) if residual is not None else None
        residual = Obj(f"r{self.i}")  # mHC post: a new stream; the old one is dead from here
        x = Obj(f"x{self.i}")  # mHC pre: the new layer input
        gc.collect()
        self.record.append((self.i, wx() is None, wr is None or wr() is None))
        return Obj(f"h{self.i}"), residual, post_mix, res_mix, pre_mix


def harness(src: str):
    """The forward's layer loop (exactly as `src` has it) in a method with stand-in globals."""
    fwd = forward_source(src)
    loop = fwd[fwd.index(REPLAY_LOOP):fwd.index(REPLAY_TAIL)]
    body = (
        "class Harness:\n"
        "    def run(self, hidden_states, residual, positions, input_ids, pre_mix, post_mix, res_mix,\n"
        "            engram_hashes, engram_mask, full_num_tokens):\n"
        "        aux_hidden_states = []\n"
        "        final_aux_recon = None\n"
        + loop
        + "        return hidden_states, residual, aux_hidden_states, final_aux_recon\n"
    )
    mm = types.ModuleType("torch.nn.modules.module")
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                 "_global_backward_pre_hooks"):
        setattr(mm, name, {})
    ns = {"islice": itertools.islice, "mhc_post_tilelang": lambda h, r, pm, rm: Obj("recon"),
          "sp_all_gather": lambda t: t}
    stubs = {"torch": types.ModuleType("torch"), "torch.nn": types.ModuleType("torch.nn"),
             "torch.nn.modules": types.ModuleType("torch.nn.modules"), "torch.nn.modules.module": mm}
    exec(compile(body + (mlr.HELPER if "_dsv41_plain_call" in loop else ""), "loop", "exec"), ns)
    return ns["Harness"], stubs


def run_loop(src, n_layers=5, taps=(), hooks=False, compiled=False):
    record = []
    cls, stubs = harness(src)
    h = cls()
    h.layers = [FakeLayer(i, record, hooks, compiled) for i in range(n_layers)]
    h.start_layer, h.end_layer, h.aux_hidden_state_layers = 0, n_layers, tuple(taps)
    h.use_sequence_parallel = False
    with mock.patch.dict(sys.modules, stubs):
        out = h.run(Obj("embed"), None, None, None, None, None, None, None, None, 0)
    return record, out


class RewriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.src = PIN.read_text()

    def test_hunks_apply_once_compile_and_round_trip(self) -> None:
        out = mlr.patch_source(self.src)
        for old, new, label in mlr.HUNKS:
            self.assertEqual(self.src.count(old), 1, label)
            self.assertEqual(out.count(new), 1, label)
            self.assertNotIn(old, out, label)
        compile(out, "model.py", "exec")
        self.assertEqual(mlr.patch_source(out), out)
        self.assertEqual(mlr.unpatch_source(out), self.src)
        self.assertEqual(mlr.unpatch_source(self.src), self.src)

    def test_decoder_replay_anchors_survive(self) -> None:
        fwd = forward_source(mlr.patch_source(self.src))
        self.assertEqual(fwd.count(REPLAY_LOOP), 1)
        self.assertEqual(fwd.count(REPLAY_TAIL), 1)
        if (PATCH / "decoder_replay.py").is_file():  # integration branches: its own rewrite must still apply
            import decoder_replay

            # its install rewrites inspect.getsource (method indent), then dedents
            patched = textwrap.dedent(decoder_replay.patch_forward_source(fwd))
            compile(patched, "forward", "exec")
            self.assertIn("_dsv41_decoder_replay.stop_layer(self)", patched)
            self.assertIn("_dsv41_plain_call(layer)", patched)

    def test_a_changed_loop_is_refused(self) -> None:
        bad = self.src.replace("                residual,\n                engram_hashes,", "                residual,\n                engram_hashes,  # x", 1)
        with self.assertRaises(ValueError):
            mlr.patch_source(bad)


class LifetimeTests(unittest.TestCase):
    """CPython 3.11+ moves call arguments into the callee frame; the rewrite relies on it."""

    def test_rewrite_frees_the_previous_outputs_inside_the_layer(self) -> None:
        record, _ = run_loop(mlr.patch_source(PIN.read_text()))
        # layer 0 gets the embedding and no residual; later layers get the previous layer's outputs
        self.assertEqual([r[1:] for r in record], [(True, True)] * 5, record)

    def test_stock_loop_keeps_them_alive_control(self) -> None:
        record, _ = run_loop(PIN.read_text())
        self.assertEqual([r[1:] for r in record], [(False, True)] + [(False, False)] * 4, record)

    def test_hooks_or_an_in_place_compile_fall_back_to_the_module_call(self) -> None:
        record, _ = run_loop(mlr.patch_source(PIN.read_text()), hooks=True)
        self.assertEqual([r[1:] for r in record], [(False, True)] + [(False, False)] * 4, record)
        record, _ = run_loop(mlr.patch_source(PIN.read_text()), compiled=True)
        self.assertEqual([r[1:] for r in record], [(False, True)] + [(False, False)] * 4, record)

    def test_tap_recon_kept_only_when_the_end_layer_is_a_tap(self) -> None:
        src = mlr.patch_source(PIN.read_text())
        _, (h, r, aux, final) = run_loop(src, n_layers=5, taps=(3, 4))
        self.assertEqual(len(aux), 2)
        self.assertIsNone(final)
        _, (h, r, aux, final) = run_loop(src, n_layers=5, taps=(4, 5))
        self.assertEqual(final.tag, "recon")
        _, (h, r, aux, final) = run_loop(PIN.read_text(), n_layers=5, taps=(3, 4))
        self.assertEqual(final.tag, "recon")  # stock keeps it either way


class InstallTests(unittest.TestCase):
    def _root(self, d: str) -> Path:
        path = Path(d) / mlr.REL
        path.parent.mkdir(parents=True)
        path.write_text(PIN.read_text())
        return Path(d)

    def _rewrite(self, path, fn):
        path = Path(path)
        src = path.read_text()
        out = fn(src)
        if out == src:
            return "already applied"
        path.write_text(out)
        return "rewritten"

    def test_install_restore_and_import_order(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = self._root(d)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mlr.install(root, self._rewrite, modules={}), "model.py rewritten")
                self.assertEqual(mlr.install(root, self._rewrite, modules={mlr.MODULE: object()}), "model.py already applied")
            self.assertIn(mlr.LOG_ENGAGED, err.getvalue())
            self.assertIn(mlr.MARK, (root / mlr.REL).read_text())
            self.assertEqual(mlr.restore(root, self._rewrite), "rewritten")
            self.assertEqual((root / mlr.REL).read_text(), PIN.read_text())
            self.assertEqual(mlr.restore(root, self._rewrite), "already applied")
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(RuntimeError):
                mlr.install(root, self._rewrite, modules={mlr.MODULE: object()})
            self.assertIn(mlr.LOG_DISARMED, err.getvalue())

    def test_env_gate(self) -> None:
        self.assertTrue(mlr.enabled({"DSV41_MODEL_LOOP_RELEASE": "1"}))
        self.assertFalse(mlr.enabled({"DSV41_MODEL_LOOP_RELEASE": "0"}))
        self.assertFalse(mlr.enabled({}))

    def test_sitecustomize_rewrites_before_any_vllm_import(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        ours = site.index('_patch("model_loop_release", _p_model_loop_release)')
        first_patch = site.index("\n_patch(")
        self.assertEqual(site.index("\n_patch(", first_patch), site.index('\n_patch("model_loop_release"'))
        self.assertNotIn("import vllm", site[:ours])
        self.assertNotIn("from vllm", site[:ours])

    def test_defaults_on_in_recipe_and_forwarded(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn('DSV41_MODEL_LOOP_RELEASE="${DSV41_MODEL_LOOP_RELEASE:-1}"', run)
        self.assertIn("  DSV41_MODEL_LOOP_RELEASE=\n", run)
        self.assertIn("    DSV41_MODEL_LOOP_RELEASE: 1\n", (ROOT / "recipe.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
