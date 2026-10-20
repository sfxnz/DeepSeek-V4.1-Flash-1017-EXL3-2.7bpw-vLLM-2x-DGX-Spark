"""docker/patch/attn_out_defer.py on the pinned e15 DeepseekV4Attention (CPU only, no torch).

The class under test is built from the fixture's own forward and _sparse_indexer_and_attn source
(tests/fixtures/dsv41_attention_e13.pin.py, the e15 text) in a temporary module, so inspect.getsource
works and the stock path is the image's code. A subclass supplies stand-ins for the projections, the
query/indexer preparation, the indexer, forward_mqa and o_proj that record the call order and which
tensors are alive at each of them (weak references).
"""

from __future__ import annotations

import ast
import contextlib
import gc
import importlib.util
import io
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch"
PIN = ROOT / "tests/fixtures/dsv41_attention_e13.pin.py"
sys.path.insert(0, str(PATCH))

import attn_out_defer as aod  # noqa: E402
import sm120_page  # noqa: E402

PROJECTIONS = ("qr_kv", "kv_score", "indexer_weights", "qr", "kv")
HEAD = '''from __future__ import annotations

import functools

torch = None


def eager_break_during_capture(fn):  # e15: a plain call outside a breakable-cudagraph capture
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class DeepseekV4Attention:
'''


def pinned_methods(src: str) -> dict[str, str]:
    """forward and _sparse_indexer_and_attn of DeepseekV4Attention, decorators included, class indent kept."""
    lines = src.splitlines(keepends=True)
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Attention")
    out = {}
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("forward", "_sparse_indexer_and_attn"):
            start = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
            out[fn.name] = "".join(lines[start:fn.end_lineno])
    return out


class FT:
    """A stand-in tensor: shape, dtype, device, a payload, and the view base."""

    def __init__(self, label, shape, data=None, base=None):
        self.label, self.shape, self.data, self.base = label, tuple(shape), data, base
        self.dtype, self.device = "bf16", "cuda:0"

    def __getitem__(self, idx):  # o_padded[:, : n_local_heads, :]
        return FT(self.label + "[:, :h]", (self.shape[0], idx[1].stop, self.shape[2]), base=self)


class Rec:
    def __init__(self):
        self.events, self.refs = [], {}

    def tensor(self, label, shape=(1,)):
        t = FT(label, shape)
        self.refs[label] = weakref.ref(t)
        self.events.append(("alloc", label))
        return t

    def mark(self, name):
        gc.collect()
        self.events.append((name, tuple(sorted(k for k, r in self.refs.items() if r() is not None))))

    def at(self, name):
        return next(alive for ev, alive in self.events if ev == name)

    def calls(self):
        return [ev for ev, _ in self.events if ev != "alloc"]

    def index(self, event):
        return self.events.index(event) if isinstance(event, tuple) else [e for e, _ in self.events].index(event)


class FakeTorch:
    def __init__(self, rec, capturing=False):
        self.rec = rec
        self.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: capturing)

    def empty(self, shape, dtype=None, device=None):
        assert dtype == "bf16" and device == "cuda:0"
        return self.rec.tensor("o_padded", shape)


def build(src_methods: dict[str, str] | None = None):
    """(module, Attention subclass) from the pinned methods (or the given texts) in a temp module file."""
    methods = src_methods or pinned_methods(PIN.read_text())
    d = tempfile.mkdtemp()
    path = Path(d) / "pinned_attention.py"
    path.write_text(HEAD + methods["forward"] + "\n" + methods["_sparse_indexer_and_attn"])
    spec = importlib.util.spec_from_file_location(f"pinned_attention_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Attention(mod.DeepseekV4Attention):
        n_local_heads, padded_heads, head_dim = 32, 64, 512

        def __init__(self, rec):
            self.rec = rec
            self.indexer = types.SimpleNamespace(indexer_op=self._indexer_op)

        def _run_parallel_input_projections(self, hidden_states):
            self.rec.mark("proj")
            return self.rec.tensor("qr_kv"), self.rec.tensor("kv_score"), self.rec.tensor("indexer_weights")

        def _split_qkv_and_norm(self, qr_kv):
            self.rec.mark("split")
            return self.rec.tensor("qr"), None, self.rec.tensor("kv")

        def _prepare_and_attn_fn(self, hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, o_padded):
            # e15 _prepare_and_attn: wq_b + q-norm/RoPE (a new padded q), indexer prep, then the eager break
            self.rec.mark("prepare")
            q, index_q = self.rec.tensor("q"), self.rec.tensor("index_q")
            self._sparse_indexer_and_attn(hidden_states, index_q, None, indexer_weights, q, kv, positions, o_padded)

        def _indexer_op(self, hidden_states, q_quant, _, weights):
            assert q_quant.label == "index_q" and weights.label == "indexer_weights"
            self.rec.mark("indexer")

        def forward_mqa(self, q, kv, positions, output):
            self.rec.mark("mqa")
            output.data = ("attn", q.label, kv.label)

        def _o_proj(self, o, positions):
            self.rec.mark("o_proj")
            return ("o_proj", o.base.data, o.shape)

    return mod, Attention


def run(attention_cls, mod, tokens, capturing=False):
    """One forward; the module's torch (shared with an installed patch) records into a fresh Rec."""
    rec = Rec()
    if mod.torch is None:
        mod.torch = FakeTorch(rec, capturing)
    mod.torch.rec = rec
    hidden = rec.tensor("hidden", (tokens, 4096))
    out = attention_cls(rec).forward("positions", hidden)
    return rec, out


def installed(capturing=False):
    mod, cls = build()
    mod.torch = FakeTorch(Rec(), capturing)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        detail = aod.install(mod.DeepseekV4Attention, torch=mod.torch)
    return mod, cls, detail, log.getvalue()


class PinTests(unittest.TestCase):
    def test_pinned_texts_match_the_e15_source_before_and_after_the_sm120_rewrites(self) -> None:
        src = PIN.read_text()
        rewritten = sm120_page.patch_indexer_short_context_source(sm120_page.patch_attention_image_width_source(src))
        for text in (src, rewritten):
            methods = pinned_methods(text)
            self.assertEqual(aod.fn_text(methods["forward"]), aod.EXPECTED_FORWARD)
            self.assertEqual(aod.fn_text(methods["_sparse_indexer_and_attn"]), aod.EXPECTED_SIA)


class DeferTests(unittest.TestCase):
    def test_stock_order_control(self) -> None:
        mod, cls = build()
        rec, out = run(cls, mod, 8190)
        self.assertEqual(rec.calls(), ["proj", "split", "prepare", "indexer", "mqa", "o_proj"])
        self.assertLess(rec.index(("alloc", "o_padded")), rec.index("proj"))
        self.assertIn("o_padded", rec.at("indexer"))
        for name in PROJECTIONS:
            self.assertIn(name, rec.at("o_proj"), name)
        self.assertEqual(out, ("o_proj", ("attn", "q", "kv"), (8190, 32, 512)))

    def test_prefill_allocates_the_output_after_the_indexer_and_drops_projections_before_o_proj(self) -> None:
        mod, cls, detail, log = installed()
        self.assertEqual(detail, "engaged")
        self.assertIn(aod.LOG_ENGAGED, log)
        rec, out = run(cls, mod, 8190)
        self.assertEqual(rec.calls(), ["proj", "split", "prepare", "indexer", "mqa", "o_proj"])
        alloc = rec.index(("alloc", "o_padded"))
        self.assertTrue(rec.index("indexer") < alloc < rec.index("mqa"), rec.events)
        self.assertNotIn("o_padded", rec.at("indexer"))
        self.assertNotIn("qr_kv", rec.at("prepare"))
        self.assertEqual(set(rec.at("o_proj")) & set(PROJECTIONS), set())
        self.assertEqual(rec.at("o_proj"), ("hidden", "o_padded"))
        self.assertEqual(out, ("o_proj", ("attn", "q", "kv"), (8190, 32, 512)))  # same as the stock control

    def test_decode_sized_steps_and_capture_keep_the_stock_order(self) -> None:
        for tokens, capturing in ((aod.MIN_ROWS, False), (8, False), (8190, True)):
            mod, cls, _, _ = installed(capturing)
            rec, out = run(cls, mod, tokens, capturing)
            self.assertLess(rec.index(("alloc", "o_padded")), rec.index("proj"), (tokens, capturing))
            for name in PROJECTIONS:
                self.assertIn(name, rec.at("o_proj"), (tokens, capturing, name))
            self.assertEqual(out, ("o_proj", ("attn", "q", "kv"), (tokens, 32, 512)))

    def test_breakable_capture_counts_as_capturing(self) -> None:
        cap = types.SimpleNamespace(_capturing=True)
        bc = types.ModuleType("vllm.compilation.breakable_cudagraph")
        bc.BreakableCUDAGraphCapture = types.SimpleNamespace(current=lambda: cap)
        mods = {"vllm": types.ModuleType("vllm"), "vllm.compilation": types.ModuleType("vllm.compilation"),
                "vllm.compilation.breakable_cudagraph": bc}
        torch = FakeTorch(Rec())
        with mock.patch.dict(sys.modules, mods):
            self.assertTrue(aod.capturing(torch))
            cap._capturing = False
            self.assertFalse(aod.capturing(torch))
        self.assertFalse(aod.capturing(torch))  # no vLLM here: only the stream check


class InstallTests(unittest.TestCase):
    def test_install_is_idempotent(self) -> None:
        mod, _, detail, _ = installed()
        self.assertEqual(detail, "engaged")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(aod.install(mod.DeepseekV4Attention, torch=FakeTorch(Rec())), "already installed")

    def test_changed_sources_are_refused(self) -> None:
        methods = pinned_methods(PIN.read_text())
        changed = [
            dict(methods, forward=methods["forward"].replace("num_tokens = hidden_states.shape[0]",
                                                             "num_tokens = hidden_states.shape[0] + 0")),
            dict(methods, _sparse_indexer_and_attn=methods["_sparse_indexer_and_attn"].replace(
                "self.forward_mqa(q, kv, positions, out)", "self.forward_mqa(q, kv, positions, out=out)")),
        ]
        for text in changed:
            mod, _ = build(text)
            stock = (mod.DeepseekV4Attention.forward, mod.DeepseekV4Attention._sparse_indexer_and_attn)
            with contextlib.redirect_stdout(io.StringIO()) as log:
                detail = aod.install(mod.DeepseekV4Attention, torch=FakeTorch(Rec()))
            self.assertTrue(detail.startswith("disabled: DeepseekV4Attention."), detail)
            self.assertIn(aod.LOG_DISARMED, log.getvalue())
            self.assertEqual((mod.DeepseekV4Attention.forward, mod.DeepseekV4Attention._sparse_indexer_and_attn), stock)

    def test_env_gate_sitecustomize_and_recipe(self) -> None:
        self.assertTrue(aod.enabled({"DSV41_ATTN_OUT_DEFER": "1"}))
        self.assertFalse(aod.enabled({"DSV41_ATTN_OUT_DEFER": "0"}))
        self.assertFalse(aod.enabled({}))
        self.assertIn('_patch("attn_out_defer", _p_attn_out_defer)', (PATCH / "sitecustomize.py").read_text())
        run_sh = (ROOT / "run.sh").read_text()
        self.assertIn('DSV41_ATTN_OUT_DEFER="${DSV41_ATTN_OUT_DEFER:-1}"', run_sh)
        self.assertIn("  DSV41_ATTN_OUT_DEFER=\n", run_sh)
        self.assertIn("    DSV41_ATTN_OUT_DEFER: 1\n", (ROOT / "recipe.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
