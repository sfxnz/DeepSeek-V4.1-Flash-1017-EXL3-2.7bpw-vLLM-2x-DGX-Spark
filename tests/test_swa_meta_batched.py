"""swa_meta_batched (DSV41_SWA_META_BATCHED): kernel mirror, call plan, leader/follower routing, wiring.

Host tests use stand-ins; nothing imports torch or vLLM. The kernel itself is checked bit for bit
against the image's stock ops by kernel_study/fusion_host/swa_meta_batched_check.py (GPU, or CPU
under TRITON_INTERPRET=1); TorchGatedTests runs that script in the image and skips on the host.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import swa_meta_batched as smb  # noqa: E402
import swa_meta_fused as smf  # noqa: E402

CHECK = ROOT / "kernel_study" / "fusion_host" / "swa_meta_batched_check.py"


def kernel_body(path: Path, name: str) -> str:
    """The kernel's source from `pid = tl.program_id(0)` to its end."""
    src = path.read_text()
    start = src.index(f"    def {name}(")
    body = src.index("        pid = tl.program_id(0)\n", start)
    end = src.index(f"\n    return {name}\n", body)
    return src[body:end].rstrip() + "\n"


class P:
    """A stand-in tensor: data_ptr, shape, stride, element size."""

    def __init__(self, ptr, shape=(16,), stride=(1,), esize=4):
        self.ptr, self.shape, self._stride, self.esize = ptr, tuple(shape), tuple(stride), esize

    def data_ptr(self):
        return self.ptr

    def stride(self, dim):
        return self._stride[dim]

    def element_size(self):
        return self.esize

    def __getitem__(self, row):
        return P(self.ptr + row * self._stride[0] * self.esize, self.shape[1:], self._stride[1:], self.esize)


class SWASpec:
    pass


class OtherSpec:
    pass


class Builder:
    def __init__(self, base):
        self.window_size, self.block_size = 128, 64
        self.decode_swa_indices = P(base, (8192, 1, 128), (128, 128, 1))
        self.decode_swa_indices.dtype = "int32"
        self.decode_swa_lens = P(base + 1, (8192,))
        self.decode_swa_lens.dtype = "int32"
        self.is_valid_token = P(base + 2, (8192,), esize=1)
        self.is_valid_token.dtype = "bool"


def kv_config(kinds):
    groups = []
    for kind in kinds:
        spec = types.SimpleNamespace(first_spec=SWASpec() if kind == "swa" else OtherSpec())
        groups.append(types.SimpleNamespace(kv_cache_spec=spec))
    return types.SimpleNamespace(kv_cache_groups=groups)


def attn(builder):
    return types.SimpleNamespace(get_metadata_builder=lambda i: builder)


def plan(n=3, other_first=True):
    kinds = (["mla"] if other_first else []) + ["swa"] * n
    builders = [Builder(1000 * (i + 1)) for i in range(n)]
    attn_groups = ([[attn(object())]] if other_first else []) + [[attn(b)] for b in builders]
    members = smb.swa_groups(attn_groups, kv_config(kinds), SWASpec, Builder)
    tables = tuple(P(50_000 + 4096 * i, (4, 8192), (8192, 1)) for i in range(len(kinds)))
    slots = P(900_000, (len(kinds), 16), (8192, 1), esize=8)
    return smb.Plan(members, smb.static_pointers([b for _, b in members]), tables, slots), builders, tables, slots


class KernelMirrorTests(unittest.TestCase):
    def test_batched_body_is_the_fused_body_line_for_line(self) -> None:
        fused = kernel_body(PATCH / "swa_meta_fused.py", "_swa_decode_fused_kernel")
        batched = kernel_body(PATCH / "swa_meta_batched.py", "_swa_decode_batched_kernel")
        self.assertGreater(len(fused.splitlines()), 30)
        self.assertEqual(batched, fused, "mirror swa_meta_fused's kernel change into swa_meta_batched")

    def test_only_the_five_group_buffers_come_from_pointer_arrays(self) -> None:
        src = (PATCH / "swa_meta_batched.py").read_text()
        pre = src[src.index("        group = tl.program_id(1)\n"):src.index("        pid = tl.program_id(0)\n")]
        loads = [line.split(" = ")[0].strip() for line in pre.splitlines() if "tl.cast(tl.load(" in line]
        self.assertEqual(sorted(loads), sorted(["swa_indices", "swa_lens", "is_valid", "slot_mapping", "block_table"]))
        self.assertIn("tl.pointer_type(tl.int1)", pre)  # bool buffers, as the fused kernel's *i1 argument
        self.assertIn("tl.pointer_type(tl.int64)", pre)  # slot mappings are int64

    def test_launch_argument_order_matches_the_kernel(self) -> None:
        tree = ast.parse((PATCH / "swa_meta_batched.py").read_text())
        kernel = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_swa_decode_batched_kernel")
        params = [a.arg for a in kernel.args.args]
        self.assertEqual(params[:16], [
            "swa_indices_ptrs", "swa_stride", "swa_lens_ptrs", "window_size", "index_width",
            "query_start_loc", "seq_lens", "token_to_req", "slot_mapping_ptrs", "is_valid_ptrs",
            "block_table_ptrs", "bt_stride", "block_size", "num_decode", "num_slots", "max_tokens"])
        launch = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "launch")
        call = next(n for n in ast.walk(launch) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript))
        args = [ast.unparse(a) for a in call.args]
        # pointer rows: 0 indices, 1 lens, 2 valid, 3 block table, 4 slot row (Plan.pointers)
        self.assertEqual([args[0], args[2], args[8], args[9], args[10]],
                         ["ptrs[0]", "ptrs[1]", "ptrs[4]", "ptrs[2]", "ptrs[3]"])


class PlanTests(unittest.TestCase):
    def test_swa_groups_skip_other_specs_and_empty_groups(self) -> None:
        b1, b2 = Builder(1), Builder(2)
        cfg = kv_config(["mla", "swa", "swa", "swa"])
        members = smb.swa_groups([[attn(object())], [attn(b1)], [], [attn(b2)]], cfg, SWASpec, Builder)
        self.assertEqual(members, [(1, b1), (3, b2)])

    def test_swa_groups_refuse_two_attention_groups_or_foreign_builders(self) -> None:
        cfg = kv_config(["swa", "swa"])
        b = Builder(1)
        self.assertEqual(smb.swa_groups([[attn(b)], [attn(b), attn(b)]], cfg, SWASpec, Builder), [])
        self.assertEqual(smb.swa_groups([[attn(b)], [attn(object())]], cfg, SWASpec, Builder), [])

    def test_shapes_must_agree(self) -> None:
        a, b = Builder(1), Builder(2)
        self.assertTrue(smb.shapes_agree([a, b]))
        b.window_size = 64
        self.assertFalse(smb.shapes_agree([a, b]))

    def test_slot_rows_and_pointer_layout(self) -> None:
        p, builders, tables, slots = plan(3)
        self.assertEqual([gid for gid, _ in p.members], [1, 2, 3])
        self.assertEqual(p.n_slots, 16)
        self.assertEqual(p.slot_ptr(0), 900_000 + 1 * 8192 * 8)
        self.assertEqual(p.slot_row(2).data_ptr(), slots[3].data_ptr())
        rows = p.pointers()
        self.assertEqual(rows[0], [b.decode_swa_indices.data_ptr() for b in builders])
        self.assertEqual(rows[1], [b.decode_swa_lens.data_ptr() for b in builders])
        self.assertEqual(rows[2], [b.is_valid_token.data_ptr() for b in builders])
        self.assertEqual(rows[3], [tables[g].data_ptr() for g in (1, 2, 3)])
        self.assertEqual(rows[4], [p.slot_ptr(i) for i in range(3)])
        self.assertIs(p.block_tables[0], tables[1])  # the call's own objects, for the identity check
        self.assertEqual(p.key()[3], 8192)


class EligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        smb._STATE.update(armed=True, verify_left=0, engaged=True)
        self.kw = dict(attn_groups=[], kv_cache_config=object(), block_tables=(), slot_mappings=object())

    def test_plain_target_call_is_eligible(self) -> None:
        self.assertTrue(smb.call_eligible(self.kw))
        self.assertTrue(smb.call_eligible(dict(self.kw, causal=True, for_cudagraph_capture=True)))

    def test_replay_metadata_non_causal_or_missing_inputs_are_not(self) -> None:
        self.assertFalse(smb.call_eligible(dict(self.kw, model_specific_attn_metadata=object())))
        self.assertFalse(smb.call_eligible(dict(self.kw, causal=False)))
        self.assertFalse(smb.call_eligible(dict(self.kw, causal={0: True})))
        self.assertFalse(smb.call_eligible(dict(self.kw, block_tables=None)))
        smb._STATE["armed"] = False
        self.assertFalse(smb.call_eligible(self.kw))


class RowsWrittenTests(unittest.TestCase):
    def setUp(self) -> None:
        smb._STATE.update(armed=True, verify_left=0, engaged=True)
        self.p, self.builders, self.tables, self.slots = plan(3)
        self.h = smb.SwaBatched(None, None, None, None)
        self.calls = []
        self.h.launch = lambda *a: self.calls.append(("launch", a[-1]))
        self.h.verify = lambda *a: self.calls.append(("verify", a[-1]))
        self.qsl, self.seq = object(), object()
        smb._CALL.update(plan=self.p, batch=None)

    def tearDown(self) -> None:
        smb._CALL.update(plan=None, batch=None)

    def ask(self, i, nd=4, qsl=None, seq=None, table=None, slots=None):
        b = self.builders[i]
        gid = i + 1
        return self.h.rows_written(b, b.decode_swa_indices, qsl or self.qsl, seq or self.seq, "t2r",
                                   slots or self.slots[gid], b.is_valid_token, table or self.tables[gid], nd)

    def test_leader_launches_once_followers_skip(self) -> None:
        self.assertTrue(self.ask(1))
        self.assertTrue(self.ask(0))
        self.assertTrue(self.ask(2))
        self.assertEqual(self.calls, [("launch", 4)])

    def test_first_launches_verify(self) -> None:
        smb._STATE["verify_left"] = 3
        self.assertTrue(self.ask(0))
        self.assertTrue(self.ask(1))
        self.assertEqual(self.calls, [("verify", 4)])

    def test_follower_with_other_inputs_takes_its_own_launch(self) -> None:
        self.assertTrue(self.ask(0))
        self.assertFalse(self.ask(1, nd=8))
        self.assertFalse(self.ask(1, qsl=object()))
        self.assertFalse(self.ask(1, seq=object()))
        self.assertTrue(self.ask(1))

    def test_unknown_builder_or_foreign_tables_are_per_group(self) -> None:
        other = Builder(7)
        self.assertFalse(self.h.rows_written(other, other.decode_swa_indices, self.qsl, self.seq, "t2r",
                                             self.slots[1], other.is_valid_token, self.tables[1], 4))
        self.assertFalse(self.ask(0, table=P(self.tables[1].data_ptr(), (4, 8192), (8192, 1))))  # not the call's object
        self.assertFalse(self.ask(0, slots=self.slots[2]))  # another group's slot row
        self.assertFalse(self.ask(0, slots=P(self.slots[1].data_ptr(), (15,))))
        self.assertEqual(self.calls, [])

    def test_no_call_in_flight_or_disarmed_is_per_group(self) -> None:
        smb._CALL["plan"] = None
        self.assertFalse(self.ask(0))
        smb._CALL["plan"] = self.p
        smb._STATE["armed"] = False
        self.assertFalse(self.ask(0))
        self.assertEqual(self.calls, [])

    def test_launch_error_disarms_and_falls_back(self) -> None:
        self.h.launch = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        with redirect_stdout(StringIO()) as out:
            self.assertFalse(self.ask(0))
        self.assertFalse(smb._STATE["armed"])
        self.assertIn(smb.LOG_DISARMED, out.getvalue())
        self.assertFalse(self.ask(1))


class WrapTests(unittest.TestCase):
    def setUp(self) -> None:
        smb._STATE.update(armed=True, verify_left=0, engaged=True)
        smb._CALL.update(plan=None, batch=None)
        self.kw = dict(attn_groups=[], kv_cache_config=object(), block_tables=(), slot_mappings=object())

    def test_plan_is_published_only_while_the_call_runs(self) -> None:
        seen = []
        stock = lambda **kw: seen.append(smb._CALL["plan"]) or "md"  # noqa: E731
        fn = smb.wrap(stock, lambda kw: "PLAN")
        self.assertEqual(fn(**self.kw), "md")
        self.assertEqual(seen, ["PLAN"])
        self.assertIsNone(smb._CALL["plan"])
        self.assertTrue(fn._dsv41_swa_batched)
        self.assertIs(fn.__wrapped__, stock)

    def test_cleared_even_when_the_call_raises(self) -> None:
        def stock(**kw):
            raise ValueError("x")

        with self.assertRaises(ValueError):
            smb.wrap(stock, lambda kw: "PLAN")(**self.kw)
        self.assertIsNone(smb._CALL["plan"])

    def test_planner_error_disarms_but_the_call_runs(self) -> None:
        def bad(kw):
            raise KeyError("attn_groups")

        with redirect_stdout(StringIO()) as out:
            self.assertEqual(smb.wrap(lambda **kw: "md", bad)(**self.kw), "md")
        self.assertFalse(smb._STATE["armed"])
        self.assertIn(smb.LOG_DISARMED, out.getvalue())

    def test_positional_or_ineligible_calls_skip_the_planner(self) -> None:
        planned = []
        fn = smb.wrap(lambda *a, **kw: "md", lambda kw: planned.append(1) or "PLAN")
        fn(1, 2)
        fn(**dict(self.kw, causal=False))
        self.assertEqual(planned, [])


class FusedHookTests(unittest.TestCase):
    def test_fused_decode_asks_the_hook_first(self) -> None:
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: False))
        h = smf.SwaFused(torch, None, None, None)
        calls = []
        h._launch = lambda *a: calls.append("fused")
        smf._STATE.update(armed=True, verify_left=0, engaged=True)
        try:
            smf._BATCH_HOOK = lambda *a: calls.append(("hook", len(a))) or True
            h.decode(*range(9))
            self.assertEqual(calls, [("hook", 9)])
            smf._BATCH_HOOK = lambda *a: False
            h.decode(*range(9))
            self.assertEqual(calls, [("hook", 9), "fused"])
        finally:
            smf._BATCH_HOOK = None

    def test_fused_change_is_the_hook_only(self) -> None:
        src = (PATCH / "swa_meta_fused.py").read_text()
        self.assertEqual(src.count("_BATCH_HOOK"), 3)  # declaration, check, call
        self.assertIn("        if _BATCH_HOOK is not None and _BATCH_HOOK(*args):\n            return\n", src)


class WiringTests(unittest.TestCase):
    def test_env(self) -> None:
        self.assertFalse(smb.enabled({}))
        self.assertFalse(smb.enabled({"DSV41_SWA_META_BATCHED": "1"}))
        self.assertTrue(smb.enabled({"DSV41_SWA_META_BATCHED": "1", "DSV41_SWA_META_FUSED": "1"}))
        self.assertEqual(smb.verify_calls({}), 8)
        self.assertEqual(smb.verify_calls({"DSV41_SWA_META_BATCHED_VERIFY": "0"}), 1)

    def test_decode_levers_step_runs_after_fused(self) -> None:
        src = (PATCH / "decode_levers.py").read_text()
        self.assertLess(src.index('("swa-meta-fused", _install_swa_meta_fused)'),
                        src.index('("swa-meta-batched", _install_swa_meta_batched)'))
        calls = []
        fake = types.ModuleType("swa_meta_batched")
        fake.install = lambda: calls.append(1) or "batched"
        fake.enabled = smb.enabled
        with mock.patch.dict(sys.modules, {"swa_meta_batched": fake}), redirect_stdout(StringIO()) as out:
            dl._install_swa_meta_batched({})
            self.assertEqual(calls, [])
            dl._install_swa_meta_batched({"DSV41_SWA_META_BATCHED": "1"})  # fused off: batched off, quietly
            self.assertEqual(calls, [])
            dl._install_swa_meta_batched({"DSV41_SWA_META_BATCHED": "1", "DSV41_SWA_META_FUSED": "1"})
        self.assertEqual(calls, [1])
        self.assertIn("swa metadata batched: batched", out.getvalue())
        self.assertIn("swa metadata batched: off (needs DSV41_SWA_META_FUSED=1)", out.getvalue())
        self.assertNotIn(dl.LOG_DISARMED, out.getvalue())  # not an audit disarm

    def test_forwarded_and_default_on(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_SWA_META_BATCHED=\n", run)
        self.assertIn("  DSV41_SWA_META_BATCHED_VERIFY=\n", run)
        self.assertIn('DSV41_SWA_META_BATCHED="${DSV41_SWA_META_BATCHED:-1}"\n', run)

    def test_audit_gate(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import engagement_audit as ea

        self.assertEqual(ea.PATCHES["swa_meta_batched.py"],
                         {"DSV41_SWA_META_FUSED": "1", "DSV41_SWA_META_BATCHED": "1"})

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse((PATCH / "swa_meta_batched.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os"})


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("vllm"), "needs the image")
class TorchGatedTests(unittest.TestCase):
    def test_kernel_matches_stock_ops_cpu_interpreter(self) -> None:
        env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
        proc = subprocess.run([sys.executable, str(CHECK), "--patch-dir", str(PATCH), "--cases", "12"],
                              env=env, capture_output=True, text=True, timeout=1800)
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-4000:])
        self.assertIn('"ok": true', proc.stdout)


if __name__ == "__main__":
    unittest.main()
