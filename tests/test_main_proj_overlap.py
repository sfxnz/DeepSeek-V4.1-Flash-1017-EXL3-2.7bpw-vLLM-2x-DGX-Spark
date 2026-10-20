"""main_proj_overlap (DSV41_MAIN_PROJ_OVERLAP): step registration, fork, consume, verify, host only.

A fake torch records the stream operations so the ordering contract is checked: the side stream waits
for an event recorded at the logits gather (after the lm_head GEMM was enqueued), the combine runs on
the side stream, and the consumer waits for the side stream's done event before using the result.
The GPU check is a serve A/B (the lever's own verify compares the first steps bitwise).
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
import decode_levers as dl  # noqa: E402
import main_proj_overlap as mpo  # noqa: E402

LOG = []


class Tensor:
    def __init__(self, shape, val=0, stream="main"):
        self.shape = shape
        self.val = val
        self.stream = stream
        self.recorded = []

    def record_stream(self, s):
        self.recorded.append(s.name)

    def view(self, dtype):
        return self


class Stream:
    def __init__(self, name="side"):
        self.name = name

    def wait_event(self, ev):
        LOG.append(("wait", self.name, ev.name))


class Event:
    n = 0

    def __init__(self):
        Event.n += 1
        self.name = f"ev{Event.n}"

    def record(self, s=None):
        LOG.append(("record", (s or FT.cuda._cur).name, self.name))


class _Cuda:
    def __init__(self):
        self._cur = Stream("main")
        self.capturing = False

    def current_stream(self):
        return self._cur

    def Stream(self):  # noqa: N802
        return Stream("side")

    def Event(self):  # noqa: N802
        return Event()

    @contextmanager
    def stream(self, s):
        prev, self._cur = self._cur, s
        try:
            yield
        finally:
            self._cur = prev

    def is_current_stream_capturing(self):
        return self.capturing


FT = types.SimpleNamespace(cuda=_Cuda(), int16="i16")
FT.cat = lambda ts, dim=-1: Tensor((ts[0].shape[0], sum(t.shape[-1] for t in ts)), sum(t.val for t in ts), FT.cuda._cur.name)
FT.equal = lambda a, b: a.val == b.val


class DSpark:
    """Stand-in for DSparkDeepseekV4ForCausalLM: combine = sum of the inputs' values, on the current stream."""

    def combine_hidden_states(self, x):
        LOG.append(("combine", FT.cuda._cur.name))
        return Tensor((x.shape[0], 5120), x.val * 10, FT.cuda._cur.name)


ORIG_COMBINE = DSpark.combine_hidden_states


def reset(verify=2):
    LOG.clear()
    FT.cuda._cur = Stream("main")
    FT.cuda.capturing = False
    mpo._STATE.update(armed=True, verify_left=verify, pending=None, ready=None, side=None, engaged=False,
                      combine=ORIG_COMBINE, cls=DSpark)


def runner(aux=None, last=True, spec_model=None):
    st = types.SimpleNamespace(aux_hidden_states=aux)
    return types.SimpleNamespace(execute_model_state=st, is_last_pp_rank=last,
                                 speculator=types.SimpleNamespace(model=spec_model if spec_model is not None else DSpark()))


class StepAuxTests(unittest.TestCase):
    def test_registered_only_for_dspark_with_aux(self):
        aux = [Tensor((4, 5120), 1)] * 3
        self.assertIsNotNone(mpo.step_aux(runner(aux), DSpark))
        self.assertIsNone(mpo.step_aux(runner(None), DSpark))
        self.assertIsNone(mpo.step_aux(runner([]), DSpark))
        self.assertIsNone(mpo.step_aux(runner(aux, last=False), DSpark))
        self.assertIsNone(mpo.step_aux(runner(aux, spec_model=object()), DSpark))
        self.assertIsNone(mpo.step_aux(types.SimpleNamespace(execute_model_state=None, speculator=None), DSpark))
        self.assertIsNone(mpo.step_aux(runner(aux), None))


class FlowTests(unittest.TestCase):
    def setUp(self):
        reset()
        self.aux = [Tensor((4, 5120), 1), Tensor((4, 5120), 2), Tensor((4, 5120), 3)]
        self.gather = mpo.wrap_gather(FT, lambda self, logits: ("gathered", logits))
        self.combine = mpo.wrap_combine(FT, ORIG_COMBINE)

    def step(self, body):
        st = mpo.wrap_sample_tokens(lambda self: body())
        return st(runner(self.aux))

    def test_fork_at_gather_then_consume(self):
        model = DSpark()

        def body():
            self.assertEqual(self.gather(None, "L"), ("gathered", "L"))
            self.gather(None, "L2")  # a second gather in the step does not fork again
            x = FT.cat(self.aux)
            return self.combine(model, x)

        with redirect_stdout(StringIO()) as out:
            res = self.step(body)
        self.assertEqual(res.val, 60)
        self.assertEqual(res.stream, "side")
        self.assertIn("main", res.recorded)
        self.assertTrue(all("side" in t.recorded for t in self.aux))
        # order: record start on main -> side waits for it -> combine on side -> done recorded on side ->
        # main waits for done -> the stock combine (verify) on main
        kinds = [(e[0], e[1]) for e in LOG]
        self.assertEqual(kinds[:4], [("record", "main"), ("wait", "side"), ("combine", "side"), ("record", "side")])
        start, done = LOG[0][2], LOG[3][2]
        self.assertEqual(LOG[1][2], start)
        self.assertIn(("wait", "main", done), LOG)
        self.assertLess(LOG.index(("wait", "main", done)), LOG.index(("combine", "main")))
        self.assertEqual(sum(1 for e in LOG if e[0] == "combine" and e[1] == "side"), 1)
        self.assertIn(mpo.LOG_ENGAGED, out.getvalue())
        self.assertIsNone(mpo._STATE["pending"])
        self.assertIsNone(mpo._STATE["ready"])

    def test_after_verify_no_stock_combine(self):
        reset(verify=1)
        model = DSpark()
        with redirect_stdout(StringIO()):
            self.step(lambda: (self.gather(None, "L"), self.combine(model, FT.cat(self.aux)))[1])
            LOG.clear()
            res = self.step(lambda: (self.gather(None, "L"), self.combine(model, FT.cat(self.aux)))[1])
        self.assertEqual(res.stream, "side")
        self.assertNotIn(("combine", "main"), LOG)

    def test_mismatch_disarms_and_returns_stock(self):
        bad = mpo.wrap_combine(FT, ORIG_COMBINE)
        model = DSpark()

        def body():
            self.gather(None, "L")
            mpo._STATE["ready"][0].val += 1  # the side result is off by one
            return bad(model, FT.cat(self.aux))

        with redirect_stdout(StringIO()) as out:
            res = self.step(body)
        self.assertEqual(res.stream, "main")
        self.assertFalse(mpo._STATE["armed"])
        self.assertIn(mpo.LOG_DISARMED, out.getvalue())
        LOG.clear()
        self.step(lambda: self.gather(None, "L"))
        self.assertFalse(any(e[0] == "combine" for e in LOG))  # disarmed: nothing forks

    def test_no_fork_while_capturing_or_without_registration(self):
        FT.cuda.capturing = True
        self.step(lambda: self.gather(None, "L"))
        self.assertEqual(LOG, [])
        FT.cuda.capturing = False
        self.gather(None, "L")  # outside sample_tokens: nothing registered
        self.assertEqual(LOG, [])

    def test_wrong_shape_computes_stock(self):
        model = DSpark()

        def body():
            self.gather(None, "L")
            return self.combine(model, Tensor((8, 15360), 6))

        res = self.step(body)
        self.assertEqual(res.stream, "main")
        self.assertTrue(mpo._STATE["armed"])

    def test_state_cleared_on_exception(self):
        def body():
            self.gather(None, "L")
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            self.step(body)
        self.assertIsNone(mpo._STATE["pending"])
        self.assertIsNone(mpo._STATE["ready"])


class WiringTests(unittest.TestCase):
    def test_env(self):
        self.assertFalse(mpo.enabled({}))
        self.assertTrue(mpo.enabled({"DSV41_MAIN_PROJ_OVERLAP": "1"}))
        self.assertEqual(mpo.verify_calls({}), 8)
        self.assertEqual(mpo.verify_calls({"DSV41_MAIN_PROJ_OVERLAP_VERIFY": "0"}), 1)

    def test_decode_levers_step(self):
        self.assertIn('("main-proj-overlap", _install_main_proj_overlap)', (PATCH / "decode_levers.py").read_text())
        calls = []
        fake = types.ModuleType("main_proj_overlap")
        fake.install = lambda: calls.append(1) or "wrapped"
        with mock.patch.dict(sys.modules, {"main_proj_overlap": fake}), redirect_stdout(StringIO()) as out:
            dl._install_main_proj_overlap({})
            self.assertEqual(calls, [])
            dl._install_main_proj_overlap({"DSV41_MAIN_PROJ_OVERLAP": "1"})
        self.assertEqual(calls, [1])
        self.assertIn("main_proj overlap: wrapped", out.getvalue())

    def test_forwarded_default_off(self):
        run = (ROOT / "run.sh").read_text()
        self.assertIn("  DSV41_MAIN_PROJ_OVERLAP=0\n", run)
        self.assertIn("  DSV41_MAIN_PROJ_OVERLAP_VERIFY=\n", run)

    def test_top_level_imports_are_stdlib_only(self):
        tree = ast.parse((PATCH / "main_proj_overlap.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in top if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "os"})


if __name__ == "__main__":
    unittest.main()
