"""dspark_adaptive (DSV41_DSPARK_ADAPTIVE): adaptive DSpark verify depth at fixed shapes.

Host, stdlib only: settings, conflicts, install gating, decode_levers and
run.sh wiring, the engagement-audit marker, and a reference model of the e15
rejection rule (greedy and one-hot sampled) showing that masking the drafts
past m emits exactly what k' = m speculation emits.

With a vLLM source tree (DSV41_VLLM_SRC=<dir holding vllm's package files>,
default: the image's dist-packages vllm when present): the source contract the
hooks rely on (perf2/src/vllm-e15 is the e15-kbits tree on the host).

With torch (the image, CPU only): the module's torch code on synthetic batches.
With torch + triton + the source tree: the e15 layout and rejection kernels
run under the Triton CPU interpreter, masked k=5 vs truncated k' = m.

  python3 -m unittest tests.test_dspark_adaptive -q
  DSV41_VLLM_SRC=/home/sfxnz/projects/data/dsv41-1017/perf2/src/vllm-e15 python3 -m unittest tests.test_dspark_adaptive -q
  docker run --rm --network none --memory 8g --cpus 4 -v "$PWD":/w -w /w/tests --entrypoint python3 \\
    dsv41-flash-exl3-sm121:e15-kbits -m unittest test_dspark_adaptive -q
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker" / "patch"
sys.path.insert(0, str(PATCH))
sys.path.insert(0, str(ROOT / "tools"))
import engagement_audit as ea  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# By path: in the image the baked sitecustomize already imported older copies under these names.
da = _load("dspark_adaptive", PATCH / "dspark_adaptive.py")
dl = _load("decode_levers", PATCH / "decode_levers.py")

IMAGE_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SRC = Path(os.environ.get("DSV41_VLLM_SRC") or IMAGE_VLLM)
HAVE_SRC = (SRC / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py").is_file()
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_TRITON = importlib.util.find_spec("triton") is not None
K = 5  # the adaptive config drafts the full DSpark block


# ---------------------------------------------------------------------------
# Settings, gating, wiring (host)
# ---------------------------------------------------------------------------
class SettingsTests(unittest.TestCase):
    def test_off_unless_exactly_1(self) -> None:
        for value in (None, "", "0", "2", "true"):
            env = {} if value is None else {"DSV41_DSPARK_ADAPTIVE": value}
            self.assertFalse(da.enabled(env), value)
        self.assertTrue(da.enabled({"DSV41_DSPARK_ADAPTIVE": "1"}))

    def test_defaults_tau_015_and_cap_k(self) -> None:
        self.assertEqual(da.settings({}, 5), ((0.15,), 5))
        self.assertEqual(da.settings({"DSV41_DSPARK_TAU": "", "DSV41_DSPARK_VERIFY_MAX": ""}, 3), ((0.15,), 3))

    def test_values(self) -> None:
        env = {"DSV41_DSPARK_TAU": "0.25", "DSV41_DSPARK_VERIFY_MAX": "3"}
        self.assertEqual(da.settings(env, 5), ((0.25,), 3))
        self.assertEqual(da.settings({"DSV41_DSPARK_TAU": "0"}, 5), ((0.0,), 5))
        self.assertEqual(da.settings({"DSV41_DSPARK_TAU": "1"}, 5), ((1.0,), 5))

    def test_tau_per_batch_size(self) -> None:
        env = {"DSV41_DSPARK_TAU": "0.15,0.15,0.3,0.3"}
        self.assertEqual(da.settings(env, 5), ((0.15, 0.15, 0.3, 0.3), 5))
        self.assertEqual(da.settings({"DSV41_DSPARK_TAU": " 0.1 , 0.2 "}, 3), ((0.1, 0.2), 3))

    def test_bad_values_raise(self) -> None:
        for env in (
            {"DSV41_DSPARK_TAU": "-0.1"},
            {"DSV41_DSPARK_TAU": "1.5"},
            {"DSV41_DSPARK_TAU": "nan"},
            {"DSV41_DSPARK_TAU": "x"},
            {"DSV41_DSPARK_TAU": "0.15,,0.3"},
            {"DSV41_DSPARK_TAU": "0.15,0.3,2"},
            {"DSV41_DSPARK_TAU": "0.15;0.3"},
            {"DSV41_DSPARK_VERIFY_MAX": "0"},
            {"DSV41_DSPARK_VERIFY_MAX": "6"},
            {"DSV41_DSPARK_VERIFY_MAX": "2.5"},
        ):
            with self.assertRaises(ValueError, msg=env):
                da.settings(env, 5)

    def test_conflicts(self) -> None:
        self.assertEqual(da.conflicts({}), [])
        self.assertEqual(da.conflicts({"DSV41_DSPARK_CONF_GATE": "0", "DSV41_DSPARK_TAIL_NGRAM": ""}), [])
        env = {"DSV41_DSPARK_CONF_GATE": "1", "DSV41_DSPARK_TAIL_NGRAM": "1"}
        self.assertEqual(da.conflicts(env), ["DSV41_DSPARK_CONF_GATE", "DSV41_DSPARK_TAIL_NGRAM"])


class InstallGatingTests(unittest.TestCase):
    def test_off_imports_nothing(self) -> None:
        # The host has neither torch nor vllm: any import would raise.
        self.assertEqual(da.install({}), "off")
        self.assertEqual(da.install({"DSV41_DSPARK_ADAPTIVE": "0"}), "off")
        self.assertIsNone(da._STATE)

    def test_conflict_refuses_before_any_import(self) -> None:
        with self.assertRaises(ValueError):
            da.install({"DSV41_DSPARK_ADAPTIVE": "1", "DSV41_DSPARK_CONF_GATE": "1"})

    def test_decode_levers_off_does_not_import_the_module(self) -> None:
        sys.modules.pop("dspark_adaptive", None)
        try:
            out = StringIO()
            with redirect_stdout(out):
                dl._install_dspark_adaptive({})
            self.assertNotIn("dspark_adaptive", sys.modules)
            self.assertEqual(out.getvalue(), "")
        finally:
            sys.modules["dspark_adaptive"] = da

    @unittest.skipIf(HAVE_TORCH, "the failure path needs a host without vllm")
    def test_decode_levers_on_without_vllm_prints_an_audited_off_line(self) -> None:
        out = StringIO()
        with redirect_stdout(out):
            dl.install({"DSV41_DSPARK_ADAPTIVE": "1"})
        line = next(x for x in out.getvalue().splitlines() if "dspark-adaptive" in x)
        self.assertIn("FAILED, lever is OFF", line)
        self.assertIn(dl.LOG_DISARMED, line)

    def test_decode_levers_lists_the_lever(self) -> None:
        src = (PATCH / "decode_levers.py").read_text()
        self.assertIn('("dspark-adaptive", _install_dspark_adaptive)', src)


class WiringTests(unittest.TestCase):
    def test_run_sh_forwards_the_knobs_default_on(self) -> None:
        run = (ROOT / "run.sh").read_text()
        block = re.search(r"^FORWARD_ENVS=\((.*?)^\)", run, re.M | re.S).group(1).split()
        fwd = dict(item.split("=", 1) for item in block)
        self.assertEqual(fwd["DSV41_DSPARK_ADAPTIVE"], "")  # recipe.yaml default, rendered into run.sh
        self.assertIn('DSV41_DSPARK_ADAPTIVE="${DSV41_DSPARK_ADAPTIVE:-1}"', run)
        self.assertEqual(fwd["DSV41_DSPARK_TAU"], "")
        self.assertEqual(fwd["DSV41_DSPARK_VERIFY_MAX"], "")

    def test_bench_has_the_c5_code_cell(self) -> None:
        spec = importlib.util.spec_from_file_location("bench_decode", ROOT / "bench_decode.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        self.assertIn("LRU cache", bench.PHASES["code"])
        self.assertIn("code", bench.PHASE_GROUPS["all"])
        self.assertEqual(bench.PHASE_GROUPS["both"], ["prose", "structured"])  # frozen cells unchanged

    def test_audit_expects_the_armed_line_on_every_rank_when_on(self) -> None:
        self.assertEqual(ea.PATCHES["dspark_adaptive.py"], {"DSV41_DSPARK_ADAPTIVE": "1"})
        env = {"DSV41_DSPARK_ADAPTIVE": "1"}
        armed = "(Worker_TP0 pid=1) dsv41: dspark adaptive verify armed (tau=0.15 verify_max=5 k=5, masking; ...)\n"
        mine = lambda problems: [p for p in problems if "dspark_adaptive" in p or "adaptive verify" in p]  # noqa: E731
        self.assertEqual(len(mine(ea.audit({"head": "", "worker": ""}, env))), 2)  # missing on both ranks
        self.assertEqual(mine(ea.audit({"head": armed, "worker": armed}, env)), [])
        off = "dsv41: dspark adaptive verify DISABLED -> stock verify: RuntimeError('x')\n"
        self.assertEqual(len(mine(ea.audit({"head": armed + off, "worker": armed}, env))), 1)
        self.assertEqual(mine(ea.audit({"head": "", "worker": ""}, {})), [])


# ---------------------------------------------------------------------------
# Reference model of the e15 rejection rule (host)
# ---------------------------------------------------------------------------
def _argmax(p: list[float]) -> int:
    return max(range(len(p)), key=lambda t: (p[t], -t))  # leftmost maximum


def _sample(p: list[float], rng: random.Random) -> int:
    total = sum(p)
    x = rng.random() * total
    for t, w in enumerate(p):
        x -= w
        if x < 0:
            return t
    return len(p) - 1


def reference_rejection(rows: list[list[float]], drafts: list[int], greedy: bool, seed: int) -> list[int]:
    """One request, one-hot drafts, as e15's rejection_sampler_utils.py does it.

    rows[i] is the target distribution after input i; row i tests drafts[i]
    (= d_{i+1}). _rejection_kernel: greedy accepts iff the draft is the argmax
    and stores the argmax at the first miss (a -1 placeholder always misses);
    sampled stops at a -1 before testing and accepts iff u < p(d). The
    resample/bonus draw (_resample_kernel) uses the residual (p without d) for a
    rejected draft and the raw p for a bonus or a -1. Every draw is keyed by its
    row (the kernels key by (seed, position)), so equal rows give equal draws.
    """
    out = []
    for i, d in enumerate(drafts):
        p = rows[i]
        if greedy:
            best = _argmax(p)
            if d >= 0 and d == best:
                out.append(d)
                continue
            out.append(best)
            return out
        if d < 0:
            break
        if random.Random(seed * 7919 + i).random() < p[d]:
            out.append(d)
            continue
        residual = [0.0 if t == d else w for t, w in enumerate(p)]
        out.append(_sample(residual, random.Random(seed * 104729 + i)))
        return out
    i = len(out)
    out.append(_argmax(rows[i]) if greedy else _sample(rows[i], random.Random(seed * 104729 + i)))
    return out


def toy_row(prefix: tuple[int, ...], vocab: int) -> list[float]:
    """A causal toy target: the row after a prefix depends on that prefix only."""
    rng = random.Random(hash(prefix) & 0xFFFFFFFF)
    w = [rng.random() ** 3 for _ in range(vocab)]
    w[vocab - 1] = 0.0  # the sentinel: only a garbage row picks it
    s = sum(w)
    return [x / s for x in w]


class ReferenceRejectionTests(unittest.TestCase):
    VOCAB = 12
    GARBAGE = [0.0] * (VOCAB - 1) + [1.0]

    def rows_for(self, ctx, anchor, drafts, live_upto):
        """Logit rows of the verify inputs [anchor, d_1..d_n]; inputs past live_upto are masked."""
        inputs = [anchor] + list(drafts)
        return [
            toy_row(tuple(ctx + inputs[: i + 1]), self.VOCAB) if i <= live_upto else self.GARBAGE
            for i in range(len(inputs))
        ]

    def trial(self, rng, greedy, seed, draft_shift=0):
        ctx = [rng.randrange(self.VOCAB - 1) for _ in range(rng.randrange(2, 6))]
        anchor = rng.randrange(self.VOCAB - 1)
        drafts = []
        for _ in range(K):  # mostly the target's own choice, like a good drafter
            row = toy_row(tuple(ctx + [anchor] + drafts), self.VOCAB)
            drafts.append(_argmax(row) if rng.random() < 0.75 else rng.randrange(self.VOCAB - 1))
        m = rng.randrange(1, K + 1)
        # Masked k=5: input rows past m are masked (garbage logits) and drafts past m
        # (+ draft_shift, a mutation) become -1.
        masked_drafts = [d if j + 1 <= m + draft_shift else -1 for j, d in enumerate(drafts)]
        got = reference_rejection(self.rows_for(ctx, anchor, drafts, m), masked_drafts, greedy, seed)
        # Truncated k' = m: the scheduler gives the request m drafts.
        want = reference_rejection(self.rows_for(ctx, anchor, drafts[:m], m), drafts[:m], greedy, seed)
        return got, want, m

    def test_masked_k5_emits_what_truncated_k_prime_emits(self) -> None:
        rng = random.Random(0)
        seen = set()
        for t in range(400):
            for greedy in (True, False):
                got, want, m = self.trial(rng, greedy, seed=t)
                self.assertEqual(got, want, (t, greedy, m))
                self.assertNotIn(self.VOCAB - 1, got, "a masked row was read")
                seen.add((greedy, len(got) == m + 1))
        self.assertEqual(seen, {(True, True), (True, False), (False, True), (False, False)})

    def test_a_draft_mask_one_row_late_reads_a_masked_row(self) -> None:
        # The MoE remap without the matching rejection mask is the dangerous half.
        rng = random.Random(1)
        read = 0
        for t in range(400):
            for greedy in (True, False):
                got, _, _ = self.trial(rng, greedy, seed=t, draft_shift=1)
                read += self.VOCAB - 1 in got
        self.assertGreater(read, 0)


# ---------------------------------------------------------------------------
# Source contract of the hooks (needs a vLLM tree)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "set DSV41_VLLM_SRC to a vllm/ package dir (e.g. perf2/src/vllm-e15)")
class SourceContractTests(unittest.TestCase):
    def read(self, rel: str) -> str:
        return (SRC / rel).read_text()

    def test_rejection_sampler_calls_the_module_global_after_sampling_params(self) -> None:
        src = self.read("v1/worker/gpu/spec_decode/rejection_sampler.py")
        self.assertIn("from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (\n    rejection_sample,\n)", src)
        verify = src[src.index("    def _verify(") : src.index("    def _verify_in_chunks(")]
        self.assertLess(verify.index("apply_sampling_params("), verify.index("rejection_sample("))
        args = verify[verify.index("rejection_sample(") :].split(")")[0]
        names = [a.strip() for a in args.split("(", 1)[1].split(",")][:8]
        self.assertEqual(names[6:8], ["expanded_idx_mapping", "expanded_local_pos"])

    def test_kernels_reject_a_minus_one_draft(self) -> None:
        src = self.read("v1/worker/gpu/spec_decode/rejection_sampler_utils.py")
        for needle in (
            "is_valid_draft = draft_sampled >= 0",
            "verifying &= is_valid_draft",
            "accepted &= is_valid_draft",
            "if is_bonus or not is_valid_rejected_draft:",
        ):
            self.assertIn(needle, src)

    def test_dspark_writes_confidences_under_the_instance_flag(self) -> None:
        src = self.read("v1/worker/gpu/spec_decode/dspark/speculator.py")
        self.assertEqual(src.count("self.draft_token_confidence_probs[:num_reqs] = confidence.view("), 2)
        self.assertEqual(src.count("if self.enable_adaptive_verification:"), 4)
        self.assertIn("return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))",
                      self.read("models/deepseek_v4_1/nvidia/dspark.py"))

    def test_runner_reads_the_flag_once_inside_initialize_kv_cache(self) -> None:
        src = self.read("v1/worker/gpu/model_runner.py")
        read = 'self.speculator, "enable_adaptive_verification", False'
        self.assertEqual(src.count(read), 1)
        init = src[src.index("    def initialize_kv_cache(") : src.index("    def _init_kv_zero_meta(")]
        self.assertIn(read, init)
        self.assertIn("maybe_create_adaptive_verification_manager(", init)
        self.assertIn("    def prepare_inputs(", src)
        self.assertIn("draft_tokens = self.speculator.propose(", src)

    def test_every_router_goes_through_one_select_experts(self) -> None:
        router_dir = SRC / "model_executor/layers/fused_moe/router"
        owners = [p.name for p in sorted(router_dir.glob("*.py")) if "def select_experts(" in p.read_text()]
        self.assertEqual(owners, ["fused_moe_router.py"])
        runner = self.read("model_executor/layers/fused_moe/runner/moe_runner.py")
        self.assertIn("topk_weights, topk_ids = self.router.select_experts(", runner)

    def test_logits_rows_are_the_query_tail_anchor_first(self) -> None:
        src = self.read("v1/worker/gpu/input_batch.py")
        self.assertIn("logits_start = query_end - num_logits", src)
        self.assertIn("tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)", src)
        fields = src[src.index("class InputBatch:") :]
        for name in ("idx_mapping", "expanded_idx_mapping", "expanded_local_pos", "num_tokens_after_padding",
                     "num_draft_tokens", "logits_indices"):
            self.assertRegex(fields, rf"\n    {name}: ", name)

    def test_compressor_ring_holds_a_k5_step(self) -> None:
        src = self.read("models/deepseek_v4_1/compressor.py")
        self.assertIn("rows_per_step = get_current_vllm_config().num_speculative_tokens + 2", src)
        self.assertIn("self.block_size = max(8, 1 << (rows_per_step - 1).bit_length())", src)
        self.assertGreaterEqual(max(8, 1 << (K + 2 - 1).bit_length()), K + 2)


# ---------------------------------------------------------------------------
# The module's torch code (image, CPU)
# ---------------------------------------------------------------------------
def synthetic_batch(torch, reqs, pad=0):
    """reqs: [(req_state, query_len, num_logits)]; logits rows are the last num_logits query rows."""
    logits, local, idx, start = [], [], [], 0
    for state, qlen, nlog in reqs:
        for j in range(nlog):
            logits.append(start + qlen - nlog + j)
            local.append(j)
            idx.append(state)
        start += qlen
    return SimpleNamespace(
        num_tokens=start,
        num_tokens_after_padding=start + pad,
        num_draft_tokens=sum(n - 1 for _, _, n in reqs),
        logits_indices=torch.tensor(logits, dtype=torch.int64),
        expanded_local_pos=torch.tensor(local, dtype=torch.int32),
        expanded_idx_mapping=torch.tensor(idx, dtype=torch.int64),
        idx_mapping=torch.tensor([s for s, _, _ in reqs], dtype=torch.int64),
    )


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run in the image)")
class ModuleTorchTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch

    def make(self, tau=0.15, cap=K, reqs=4, tokens=48):
        return da.AdaptiveVerify(tau, cap, K, reqs, tokens, "cpu")

    def test_verify_width(self) -> None:
        torch = self.torch
        conf = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.9]])  # survival .9 .63 .315 .0945 .085
        w = lambda tau, cap=K: da.verify_width(conf, tau, cap).tolist()  # noqa: E731
        self.assertEqual(w(0.15), [3])
        self.assertEqual(w(0.5), [2])
        self.assertEqual(w(0.95), [1])  # never below 1
        self.assertEqual(w(0.0), [K])
        self.assertEqual(w(0.0, 3), [3])  # the VERIFY_MAX cap
        self.assertEqual(da.verify_width(torch.full((2, K), float("nan")), 0.15, K).tolist(), [1, 1])
        self.assertEqual(da.verify_width(conf, 0.15, K).dtype, torch.int32)

    def test_prepare_marks_rows_past_the_width(self) -> None:
        torch = self.torch
        av = self.make()
        av.verify_len[torch.tensor([2, 0])] = torch.tensor([3, K], dtype=torch.int32)
        # c=2 verify (states 2 and 0: 1 + 5 rows each), a 5-token prefill chunk, 3 padding rows.
        batch = synthetic_batch(torch, [(2, 6, 6), (0, 6, 6), (1, 5, 1)], pad=3)
        av.prepare(batch)
        self.assertEqual(torch.nonzero(av.masked[:20]).flatten().tolist(), [4, 5])
        self.assertEqual(av.src[:20].tolist(), [0, 1, 2, 3, 0, 0] + list(range(6, 20)))
        # Next step, smaller batch, request 0 at width 1: stale rows are cleared.
        av.verify_len[0] = 1
        av.prepare(synthetic_batch(torch, [(0, 6, 6)]))
        self.assertEqual(torch.nonzero(av.masked[:20]).flatten().tolist(), [2, 3, 4, 5])
        self.assertEqual(av.src[:6].tolist(), [0, 1, 0, 0, 0, 0])
        self.assertTrue(bool((av.src <= av.iota).all()))  # the remap gather stays in the batch

    def test_no_drafts_masks_nothing(self) -> None:
        torch = self.torch
        av = self.make()
        av.masked.fill_(True)
        av.src.zero_()
        av.prepare(synthetic_batch(torch, [(0, 30, 1)], pad=2))
        self.assertFalse(bool(av.masked[:32].any()))
        self.assertEqual(av.src[:32].tolist(), list(range(32)))

    def test_remap_reuses_anchor_experts_with_zero_weight(self) -> None:
        torch = self.torch
        g = torch.Generator().manual_seed(0)
        av = self.make()
        av.verify_len[torch.tensor([2, 0])] = torch.tensor([2, 4], dtype=torch.int32)
        batch = synthetic_batch(torch, [(2, 6, 6), (0, 6, 6)])
        av.prepare(batch)
        n = batch.num_tokens
        ids = torch.stack([torch.randperm(384, generator=g)[:6] for _ in range(n)]).int()
        w = torch.rand(n, 6, generator=g)
        w2, ids2 = av.remap(w, ids)
        masked = av.masked[:n]
        live = ~masked
        self.assertTrue(torch.equal(ids2[live], ids[live]))
        self.assertTrue(torch.equal(w2[live], w[live]))  # live rows bit-identical
        self.assertTrue(bool((w2[masked] == 0).all()))
        self.assertTrue(torch.equal(ids2[3:6], ids[0].expand(3, 6)))  # request 0: m=2
        self.assertTrue(torch.equal(ids2[11:12], ids[6].expand(1, 6)))  # request 1: m=4
        for start, m in ((0, 2), (6, 4)):  # no expert beyond the live rows' union
            block = set(ids2[start : start + 6].flatten().tolist())
            self.assertEqual(block, set(ids[start : start + m + 1].flatten().tolist()))
        self.assertEqual((ids2.dtype, w2.dtype), (ids.dtype, w.dtype))

    def test_draft_mask_is_the_input_mask(self) -> None:
        torch = self.torch
        av = self.make()
        av.verify_len[torch.tensor([3, 1])] = torch.tensor([4, 1], dtype=torch.int32)
        batch = synthetic_batch(torch, [(3, 6, 6), (1, 6, 6), (0, 3, 1)])
        av.prepare(batch)
        draft = torch.arange(100, 100 + batch.logits_indices.numel(), dtype=torch.int32)
        out = av.mask_drafts(draft, batch.expanded_idx_mapping, batch.expanded_local_pos)
        self.assertEqual(
            batch.logits_indices[out < 0].tolist(),
            torch.nonzero(av.masked[: batch.num_tokens]).flatten().tolist(),
        )
        self.assertEqual(out.tolist(), [100, 101, 102, 103, 104, -1, 106, 107, -1, -1, -1, -1, 112])
        self.assertEqual(out.dtype, draft.dtype)

    def test_a_record_between_prepare_and_rejection_keeps_the_steps_mask(self) -> None:
        torch = self.torch
        av = self.make()
        av.verify_len[0] = 2
        batch = synthetic_batch(torch, [(0, 6, 6)])
        av.prepare(batch)
        av.record(torch.full((1, K), 0.99), torch.tensor([0], dtype=torch.int64))  # next step: width 5
        out = av.mask_drafts(torch.arange(6, dtype=torch.int32), batch.expanded_idx_mapping, batch.expanded_local_pos)
        self.assertEqual(out.tolist(), [0, 1, 2, -1, -1, -1])
        self.assertEqual(int(av.verify_len[0]), K)

    def test_record_uses_the_batch_sizes_tau(self) -> None:
        torch = self.torch
        av = da.AdaptiveVerify((0.15, 0.15, 0.5), K, K, 8, 48, "cpu")
        conf = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.9]] * 6)  # survival .9 .63 .315 .0945 .085
        av.record(conf, torch.tensor([0], dtype=torch.int64))  # batch of 1: tau 0.15
        self.assertEqual(int(av.verify_len[0]), 3)
        av.record(conf, torch.tensor([1, 2, 3], dtype=torch.int64))  # batch of 3: tau 0.5
        self.assertEqual(av.verify_len[1:4].tolist(), [2, 2, 2])
        av.record(conf, torch.arange(4, 8))  # batch of 4: past the list, the last tau holds
        self.assertEqual(av.verify_len[4:8].tolist(), [2, 2, 2, 2])
        self.assertEqual(da.AdaptiveVerify(0.3, K, K, 4, 8, "cpu").taus, (0.3,))

    def test_record_writes_widths_by_request_state(self) -> None:
        torch = self.torch
        av = self.make(tau=0.15)
        conf = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.9], [0.99] * 5, [0.1] * 5])
        av.record(conf, torch.tensor([3, 1], dtype=torch.int64))  # two requests in this batch
        self.assertEqual(av.verify_len.tolist(), [K, K, K, 3])
        cap = self.make(tau=0.0, cap=3)
        cap.record(conf, torch.tensor([0, 2], dtype=torch.int64))
        self.assertEqual(cap.verify_len.tolist(), [3, K, 3, K])

    def test_aa_mode_is_the_identity(self) -> None:
        torch = self.torch
        av = self.make(tau=0.0, cap=K)
        av.record(torch.rand(2, K), torch.tensor([0, 1], dtype=torch.int64))
        batch = synthetic_batch(torch, [(0, 6, 6), (1, 6, 6)])
        av.prepare(batch)
        self.assertFalse(bool(av.masked[:12].any()))
        ids, w = torch.randint(0, 384, (12, 6), dtype=torch.int32), torch.rand(12, 6)
        w2, ids2 = av.remap(w, ids)
        self.assertTrue(torch.equal(ids2, ids) and torch.equal(w2, w))
        draft = torch.arange(12, dtype=torch.int32)
        self.assertTrue(torch.equal(av.mask_drafts(draft, batch.expanded_idx_mapping, batch.expanded_local_pos), draft))


def fake_vllm(torch):
    """The five e15 names the lever hooks, as stand-ins with e15's signatures."""
    calls = {}

    class FusedMoERouter:
        def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            return calls["routing"]

    class RejectionSampler:
        pass

    def rejection_sample(target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
                         expanded_idx_mapping, expanded_local_pos, temperature, seed, num_speculative_steps,
                         synthetic_conditional_rates=None, use_fp64=False, use_block_verification=False):
        calls["draft_sampled"] = draft_sampled
        return "sampled", "num_sampled"

    class DSparkSpeculator:
        def __init__(self, k):
            self.num_speculative_steps = k
            self.enable_adaptive_verification = False
            self.draft_token_confidence_probs = torch.full((4, k), 0.99)
            self.model = SimpleNamespace(model=SimpleNamespace(confidence_head=object()))

        def propose(self, input_batch, *args, **kwargs):
            return "draft_tokens"

    class GPUModelRunner:
        def initialize_kv_cache(self, kv_cache_config, is_profiling=False):
            calls["init_kv"] = kv_cache_config

        def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
            return calls["batch"]

    def package(name, **attrs):
        mod = type(sys)(name)
        mod.__path__ = []
        mod.__dict__.update(attrs)
        return mod

    mods = {
        "vllm.model_executor.layers.fused_moe.router.fused_moe_router": package("fmr", FusedMoERouter=FusedMoERouter),
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler": package(
            "rs", RejectionSampler=RejectionSampler, rejection_sample=rejection_sample),
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator": package("sp", DSparkSpeculator=DSparkSpeculator),
        "vllm.v1.worker.gpu.model_runner": package("mr", GPUModelRunner=GPUModelRunner),
    }
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
                 "vllm.model_executor.layers.fused_moe.router", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
                 "vllm.v1.worker.gpu.spec_decode", "vllm.v1.worker.gpu.spec_decode.dspark"):
        mods[name] = package(name)
    for name, mod in list(mods.items()):  # `from parent import child` reads the parent's attribute
        parent, _, child = name.rpartition(".")
        if parent in mods:
            setattr(mods[parent], child, mod)
    return mods, calls


def fake_runner(mods, spec, routers, rejection_sampler=None):
    rs = mods["vllm.v1.worker.gpu.spec_decode.rejection_sampler"]
    runner = mods["vllm.v1.worker.gpu.model_runner"].GPUModelRunner()
    runner.speculator = spec
    runner.num_speculative_steps = spec.num_speculative_steps
    runner.rejection_sampler = rejection_sampler or rs.RejectionSampler()
    runner.adaptive_verification = None
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1,
                                             enable_expert_parallel=False)
    layers = [SimpleNamespace(router=r) for r in routers]
    runner.model = SimpleNamespace(modules=lambda: iter([SimpleNamespace()] + layers))
    runner.max_num_reqs, runner.max_num_tokens, runner.device = 4, 48, "cpu"
    return runner


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run in the image)")
class ArmWiringTests(unittest.TestCase):
    """install() -> speculator init -> initialize_kv_cache -> one step, on stand-ins."""

    def setUp(self) -> None:
        import torch
        from unittest import mock

        self.torch = torch
        self.mods, self.calls = fake_vllm(torch)
        patcher = mock.patch.dict(sys.modules, self.mods)
        patcher.start()
        self.addCleanup(patcher.stop)
        da._STATE = None
        self.addCleanup(setattr, da, "_STATE", None)
        self.env = {"DSV41_DSPARK_ADAPTIVE": "1", "DSV41_DSPARK_TAU": "0.5"}

    def boot(self, **runner_kwargs):
        sp = self.mods["vllm.v1.worker.gpu.spec_decode.dspark.speculator"]
        fmr = self.mods["vllm.model_executor.layers.fused_moe.router.fused_moe_router"]
        self.assertTrue(da.install(self.env).startswith("hooks bind"))
        self.assertEqual(da.install(self.env), "already installed")
        spec = sp.DSparkSpeculator(K)  # binds the runner hooks
        routers = [fmr.FusedMoERouter() for _ in range(3)]
        runner = fake_runner(self.mods, spec, routers, **runner_kwargs)
        out = StringIO()
        with redirect_stdout(out):
            runner.initialize_kv_cache("kv-config")
        self.assertEqual(self.calls["init_kv"], "kv-config")
        return runner, spec, routers, out.getvalue()

    def test_arms_and_one_step_masks_both_sides(self) -> None:
        torch = self.torch
        runner, spec, routers, log = self.boot()
        self.assertIn("dsv41: dspark adaptive verify armed (tau=0.5 verify_max=5 k=5, masking; ", log)
        self.assertIn("in 3 MoE layers", log)
        self.assertIsNotNone(da._STATE)
        self.assertTrue(spec.enable_adaptive_verification)
        # Step t: the drafter records widths (survival .9 .45 ... -> width 1 at tau 0.5).
        spec.draft_token_confidence_probs[:2] = torch.tensor([[0.9, 0.5, 0.9, 0.9, 0.9], [0.99] * 5])
        batch = synthetic_batch(torch, [(2, 6, 6), (0, 6, 6)])
        self.assertEqual(spec.propose(batch), "draft_tokens")
        self.assertEqual(da._STATE.verify_len[[2, 0]].tolist(), [1, K])
        # Step t+1: prepare, route, verify.
        self.calls["batch"] = batch
        self.assertIs(runner.prepare_inputs("so", "brs", "desc"), batch)
        ids = torch.arange(12 * 6, dtype=torch.int32).view(12, 6)
        w = torch.ones(12, 6)
        self.calls["routing"] = (w, ids)
        w2, ids2 = routers[0].select_experts("h", "logits")
        self.assertTrue(torch.equal(ids2[2:6], ids[0].expand(4, 6)))
        self.assertEqual(w2[2:6].sum().item(), 0.0)
        self.assertTrue(torch.equal(ids2[6:], ids[6:]))
        rs = self.mods["vllm.v1.worker.gpu.spec_decode.rejection_sampler"]
        draft = torch.arange(100, 112, dtype=torch.int32)
        out = rs.rejection_sample("lg", None, draft, "cu", "pos", "idx", batch.expanded_idx_mapping,
                                  batch.expanded_local_pos, "t", "s", K, None, use_fp64=False)
        self.assertEqual(out, ("sampled", "num_sampled"))
        self.assertEqual(self.calls["draft_sampled"].tolist(), [100, 101, -1, -1, -1, -1] + list(range(106, 112)))
        # A second initialize_kv_cache does not re-arm.
        self.assertEqual(da.arm(runner, self.env), "already armed")

    def test_an_unbound_router_is_untouched(self) -> None:
        fmr = self.mods["vllm.model_executor.layers.fused_moe.router.fused_moe_router"]
        self.boot()
        draft_router = fmr.FusedMoERouter()  # e.g. the drafter's MoE: never bound
        self.calls["routing"] = ("w", "ids")
        self.assertEqual(draft_router.select_experts("h", "logits"), ("w", "ids"))

    def test_arms_with_a_tau_per_batch_size(self) -> None:
        self.env["DSV41_DSPARK_TAU"] = "0.15,0.15,0.3,0.3"
        _, _, _, log = self.boot()
        self.assertIn("dsv41: dspark adaptive verify armed (tau=0.15,0.15,0.3,0.3 verify_max=5 k=5, masking; ", log)
        self.assertEqual(da._STATE.taus, (0.15, 0.15, 0.3, 0.3))

    def test_refusal_binds_nothing(self) -> None:
        rs = self.mods["vllm.v1.worker.gpu.spec_decode.rejection_sampler"]
        custom = type("Custom", (rs.RejectionSampler,), {})()
        runner, spec, routers, log = self.boot(rejection_sampler=custom)
        self.assertIn("dsv41: dspark adaptive verify DISABLED -> stock verify: RuntimeError('needs the stock V2 RejectionSampler')", log)
        self.assertIsNone(da._STATE)
        self.assertFalse(spec.enable_adaptive_verification)
        self.assertFalse(any(hasattr(r, "_dsv41_av_remap") for r in routers))
        self.calls["routing"] = ("w", "ids")
        self.assertEqual(routers[0].select_experts("h", "logits"), ("w", "ids"))


# ---------------------------------------------------------------------------
# End to end on e15's own kernels (Triton CPU interpreter, image)
# ---------------------------------------------------------------------------
STUBS = {
    "vllm/__init__.py": "",
    "vllm/utils/__init__.py": "import uuid\n\n\ndef random_uuid():\n    return uuid.uuid4().hex\n",
    "vllm/utils/math_utils.py": "def cdiv(a, b):\n    return -(-a // b)\n",
    # The interpreter does not patch libdevice; a jit-wrapped log1p resolves.
    "vllm/triton_utils/__init__.py": textwrap.dedent(
        """\
        import types

        import triton
        import triton.language as tl

        HAS_TRITON = True


        @triton.jit
        def _log1p(x):
            return tl.where(tl.abs(x) < 1e-4, x * (1.0 - 0.5 * x), tl.log(1.0 + x))


        tldevice = types.SimpleNamespace(log1p=_log1p)
        """
    ),
    "vllm/v1/__init__.py": "",
    "vllm/v1/worker/__init__.py": "",
    "vllm/v1/worker/gpu/__init__.py": "",
    "vllm/v1/worker/gpu/sample/__init__.py": "",
    "vllm/v1/worker/gpu/spec_decode/__init__.py": "",
}
E15_FILES = (
    "v1/worker/gpu/input_batch.py",
    "v1/worker/gpu/sample/gumbel.py",
    "v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
)

E2E_SCRIPT = textwrap.dedent(
    r'''
    import sys

    # The image's sitecustomize may have imported the real vllm (and a baked patch dir):
    # drop them so the stub tree and this checkout's module are the ones imported.
    for _name in [n for n in sys.modules if n in ("vllm", "dspark_adaptive") or n.startswith("vllm.")]:
        del sys.modules[_name]
    sys.path[:0] = [sys.argv[1], sys.argv[2]]

    import zlib
    from collections import Counter
    from types import SimpleNamespace

    import numpy as np
    import torch

    from vllm.v1.worker.gpu import input_batch as ib
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    from dspark_adaptive import AdaptiveVerify

    K, V, MAX_REQS, MAX_TOKENS, MAX_LEN, PAD, E, TOPK = 5, 64, 4, 40, 64, 3, 384, 6
    SENTINEL = V - 1  # the toy target never picks it; a masked row picks nothing else
    GARBAGE = np.full(V, -30.0, np.float32)
    GARBAGE[SENTINEL] = 30.0


    def target_logits(prefix):
        seed = zlib.crc32(np.asarray(prefix, np.int64).tobytes())
        x = np.random.default_rng(seed).standard_normal(V).astype(np.float32) * 4.0
        x[SENTINEL] = -30.0
        return x


    def experts(prefix):
        seed = zlib.crc32(np.asarray(prefix, np.int64).tobytes()) ^ 0x5EED
        return np.random.default_rng(seed).choice(E, TOPK, replace=False)


    def step(reqs, n_drafts, temperature, block, draft_logits, av=None):
        """One step laid out by e15's kernels, 'forwarded', then verified by e15's sampler."""
        R = len(reqs)
        idx_mapping = torch.tensor([r["state"] for r in reqs], dtype=torch.int64)
        qlens = [1 + n if r["verify"] else len(r["chunk"]) for r, n in zip(reqs, n_drafts)]
        qsl = np.zeros(MAX_REQS + 1, np.int32)
        qsl[1 : R + 1] = np.cumsum(qlens)
        qsl[R + 1 :] = qsl[R]
        query_start_loc = torch.from_numpy(qsl)
        num_tokens = int(qsl[R])
        num_computed = torch.zeros(MAX_REQS, dtype=torch.int32)
        prefill_len = torch.zeros(MAX_REQS, dtype=torch.int32)
        last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64)
        draft_tokens = torch.zeros(MAX_REQS, K, dtype=torch.int64)
        all_token_ids = torch.zeros(MAX_REQS, MAX_LEN, dtype=torch.int32)
        for r, n in zip(reqs, n_drafts):
            s, ctx = r["state"], r["context"]
            num_computed[s] = len(ctx)
            if r["verify"]:
                prefill_len[s] = r["prompt_len"]
                last_sampled[s, 0] = r["anchor"]
                draft_tokens[s, :n] = torch.tensor(r["drafts"][:n])
            else:
                toks = ctx + r["chunk"]
                prefill_len[s] = len(toks)
                all_token_ids[s, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
        input_ids = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        positions = torch.zeros(MAX_TOKENS, dtype=torch.int64)
        seq_lens = torch.zeros(MAX_REQS, dtype=torch.int32)
        next_prefill = torch.zeros(1, MAX_REQS, dtype=torch.int32)
        ib.prepare_prefill_inputs(
            input_ids, next_prefill, idx_mapping, query_start_loc, all_token_ids,
            prefill_len, num_computed,
        )
        ib.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed, positions, seq_lens)
        cu = np.zeros(R + 1, np.int32)
        cu[1:] = np.cumsum([1 + n for n in n_drafts])
        cu_num_logits = torch.from_numpy(cu)
        num_logits = int(cu[-1])
        logits_indices = ib.combine_sampled_and_draft_tokens(
            input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens,
            prefill_len, draft_tokens, cu_num_logits, num_logits,
        )
        expanded_idx_mapping, expanded_local_pos = ib.expand_idx_mapping(
            idx_mapping, num_logits, cu_num_logits, K + 1
        )
        batch = SimpleNamespace(
            num_tokens_after_padding=num_tokens + PAD,
            num_draft_tokens=sum(n_drafts),
            logits_indices=logits_indices,
            expanded_local_pos=expanded_local_pos,
            expanded_idx_mapping=expanded_idx_mapping,
        )
        masked = torch.zeros(num_tokens + PAD, dtype=torch.bool)
        if av is not None:
            av.prepare(batch)
            masked = av.masked[: num_tokens + PAD].clone()
        assert not masked[num_tokens:].any(), "padding rows masked"

        # 'Forward': row t sees its request's context and its query up to t. A masked
        # row outputs garbage, which shows up in any token it feeds.
        req_of = np.searchsorted(qsl[1 : R + 1], np.arange(num_tokens), side="right")
        prefix = [reqs[b]["context"] + input_ids[qsl[b] : t + 1].tolist() for t, b in enumerate(req_of)]
        rows = logits_indices.tolist()
        logits = torch.from_numpy(np.stack([GARBAGE if masked[t] else target_logits(prefix[t]) for t in rows]))
        if av is not None:  # routing remap on the real layout
            ids = torch.from_numpy(np.stack([experts(p) for p in prefix])).int()
            w = torch.rand(num_tokens, TOPK)
            w2, ids2 = av.remap(w, ids)
            live = ~masked[:num_tokens]
            assert torch.equal(ids2[live], ids[live]) and torch.equal(w2[live], w[live])
            assert bool((w2[~live] == 0).all())
            for b in range(R):
                q0, q1 = int(qsl[b]), int(qsl[b + 1])
                got = set(ids2[q0:q1].flatten().tolist())
                want = set(ids[q0:q1][live[q0:q1]].flatten().tolist())
                assert got == want, "a masked row added an expert"

        draft_sampled = input_ids[logits_indices]
        pos = positions[logits_indices]
        if av is not None:
            draft_sampled = av.mask_drafts(draft_sampled, expanded_idx_mapping, expanded_local_pos)
        temp = torch.full((MAX_REQS,), temperature, dtype=torch.float32)
        seeds = torch.tensor([11, 22, 33, 44], dtype=torch.int64)
        sampled, num_sampled = rejection_sample(
            logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
            expanded_idx_mapping, expanded_local_pos, temp, seeds, K, None,
            use_fp64=False, use_block_verification=block,
        )
        out = [sampled[b, : int(num_sampled[b])].tolist() for b in range(R)]
        assert all(SENTINEL not in o for o in out), "a masked row was read"
        return out


    def make_requests(rng):
        states = rng.permutation(MAX_REQS)[:3].tolist()
        reqs = []
        for s in states[:2]:
            ctx = rng.integers(0, V - 1, int(rng.integers(3, 9))).tolist()
            anchor = int(rng.integers(0, V - 1))
            drafts = []
            for _ in range(K):
                best = int(np.argmax(target_logits(ctx + [anchor] + drafts)))
                drafts.append(best if rng.random() < 0.8 else int(rng.integers(0, V - 1)))
            reqs.append(dict(verify=True, state=s, context=ctx, prompt_len=2, anchor=anchor, drafts=drafts))
        ctx = rng.integers(0, V - 1, 5).tolist()
        reqs.append(dict(verify=False, state=states[2], context=ctx, chunk=rng.integers(0, V - 1, 4).tolist()))
        return reqs


    rng = np.random.default_rng(0)
    modes = {
        "greedy": (0.0, False, False),
        "sampled": (1.0, False, False),
        "sampled+block": (1.0, True, False),
        "sampled+draft_logits": (1.0, False, True),
        "sampled+block+draft_logits": (1.0, True, True),
    }
    seen = Counter()
    for trial in range(int(sys.argv[3])):
        reqs = make_requests(rng)
        widths = [int(rng.integers(1, K + 1)) for _ in range(2)]
        av = AdaptiveVerify(0.15, K, K, MAX_REQS, MAX_TOKENS, "cpu")
        for r, m in zip(reqs, widths):
            av.verify_len[r["state"]] = m
        dl = torch.randn(MAX_REQS, K, V) * 2
        for name, (temperature, block, with_dl) in modes.items():
            draft_logits = dl if with_dl else None
            got = step(reqs, [K, K, 0], temperature, block, draft_logits, av)
            want = step(reqs, widths + [0], temperature, block, draft_logits)
            assert got == want, (trial, name, widths, got, want)
            for out, m in zip(got, widths):
                seen[(name, "all m accepted" if len(out) == m + 1 else "rejected early")] += 1
    for key in sorted(seen):
        print("%-28s %-15s %d" % (*key, seen[key]))
    both = {name: {k[1] for k in seen if k[0] == name} for name in modes}
    assert all(v == {"all m accepted", "rejected early"} for v in both.values()), both
    print("masked k=5 == truncated k'=m on every trial and mode")
    '''
)


# ---------------------------------------------------------------------------
# run.sh: DSpark depth per batch size (NUM_SPECULATIVE_TOKENS_PER_BATCH, host)
# ---------------------------------------------------------------------------
_RUN_SH_KNOBS = ("SPEC", "SPEC_CONFIG", "COMPILATION_CONFIG", "NUM_SPECULATIVE_TOKENS",
                 "NUM_SPECULATIVE_TOKENS_PER_BATCH", "MAX_NUM_SEQS", "FORCE_UNSAFE_CTX")


def _validate(**extra: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _RUN_SH_KNOBS}
    env.update(extra, VALIDATE_ONLY="1")
    return subprocess.run([str(ROOT / "run.sh")], cwd=str(ROOT), env=env, capture_output=True, text=True,
                          check=False)


def _field(stdout: str, name: str) -> str:
    """A validate-only field (name=value, values contain no spaces)."""
    return re.search(rf"(?:^| ){name}=(\S*)", stdout).group(1)


def tier_k(spec: str) -> dict[int, int]:
    out = {}
    for tier in spec.split(","):
        sizes, k = tier.split(":")
        lo, _, hi = sizes.partition("-")
        out.update({s: int(k) for s in range(int(lo), int(hi or lo) + 1)})
    return out


class SpecTierRunShTests(unittest.TestCase):
    def test_tiers_build_spec_config_and_capture_every_reachable_shape(self) -> None:
        import json

        proc = _validate(NUM_SPECULATIVE_TOKENS_PER_BATCH="1-2:5,3-4:3")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_field(proc.stdout, "spec_tokens"), "5")  # the DSpark block is the largest k
        self.assertEqual(json.loads(_field(proc.stdout, "spec_config")), {
            "method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "greedy",
            "num_speculative_tokens_per_batch_size": [[1, 2, 5], [3, 4, 3]],
        })
        sizes = json.loads(_field(proc.stdout, "compilation_config"))["cudagraph_capture_sizes"]
        self.assertEqual(sizes, [1, 4, 5, 6, 8, 10, 12, 15, 16, 20])
        ks = tier_k("1-2:5,3-4:3")
        for c in range(1, 5):
            for s in range(1, c + 1):  # width picked at batch c, decoded by s <= c requests
                self.assertIn(s * (ks[c] + 1), sizes, (s, c))
            self.assertIn(c * 5, sizes)  # the drafter's full block for c requests

    def test_capture_sets_for_other_tiers(self) -> None:
        import json

        for spec, seqs, k, want in (
            ("1-2:3,3-4:1", "4", "3", [1, 2, 3, 4, 6, 8, 9, 12]),
            ("1:5,2:4,3-4:3", "4", "5", [1, 4, 5, 6, 8, 10, 12, 15, 16, 20]),
            ("1:5,2:3", "2", "5", [1, 4, 5, 6, 8, 10]),
            ("1-4:3", "4", "3", [1, 3, 4, 6, 8, 9, 12, 16]),  # one tier = the fixed-k set
            (" 1-2:5 , 3-4:3 ", "4", "5", [1, 4, 5, 6, 8, 10, 12, 15, 16, 20]),
        ):
            proc = _validate(NUM_SPECULATIVE_TOKENS_PER_BATCH=spec, MAX_NUM_SEQS=seqs)
            self.assertEqual(proc.returncode, 0, (spec, proc.stderr))
            self.assertEqual(_field(proc.stdout, "spec_tokens"), k, spec)
            sizes = json.loads(_field(proc.stdout, "compilation_config"))["cudagraph_capture_sizes"]
            self.assertEqual(sizes, want, spec)

    def test_default_is_the_d4_tiers(self) -> None:
        import json

        proc = _validate()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(_field(proc.stdout, "spec_tokens"), "5")
        spec = json.loads(_field(proc.stdout, "spec_config"))
        self.assertEqual(spec["num_speculative_tokens_per_batch_size"], [[1, 2, 5], [3, 4, 3]])
        self.assertIn('"cudagraph_capture_sizes":[1,4,5,6,8,10,12,15,16,20]', proc.stdout)

    def test_zero_is_one_k(self) -> None:
        for extra in ({}, {"MAX_NUM_SEQS": "2"}, {"SPEC_CONFIG": '{"method":"dspark","num_speculative_tokens":3}'}):
            proc = _validate(NUM_SPECULATIVE_TOKENS_PER_BATCH="0", **extra)
            self.assertEqual(proc.returncode, 0, (extra, proc.stderr))
            self.assertNotIn("per_batch_size", _field(proc.stdout, "spec_config"), extra)
        proc = _validate(NUM_SPECULATIVE_TOKENS_PER_BATCH="0")
        self.assertEqual(_field(proc.stdout, "spec_tokens"), "3")
        self.assertIn('"cudagraph_capture_sizes":[1,3,4,6,8,9,12,16]', proc.stdout)
        self.assertEqual(_validate(SPEC="none", NUM_SPECULATIVE_TOKENS_PER_BATCH="0").returncode, 0)
        proc = _validate(MAX_NUM_SEQS="2")  # the default tiers run to batch size 4
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("0 = one k (NUM_SPECULATIVE_TOKENS)", proc.stderr)

    def test_malformed_specs_refuse(self) -> None:
        for spec, extra, why in (
            ("1-2:6,3-4:3", {}, "malformed tier '1-2:6'"),
            ("1-2:5,3-4:0", {}, "malformed tier '3-4:0'"),
            ("01-2:5,3-4:3", {}, "malformed tier '01-2:5'"),
            ("1-2:5;3-4:3", {}, "malformed tier"),
            ("1-2:5,,3-4:3", {}, "malformed tier ''"),
            ("x", {}, "malformed tier 'x'"),
            ("2-4:3", {}, "should start at batch size 1"),
            ("1-2:5,4:3", {}, "should start at batch size 3"),
            ("1-3:5,3-4:3", {}, "should start at batch size 4"),
            ("2-1:5", {}, "should start at batch size 1"),
            ("1-2:5", {}, "the tiers stop at batch size 2"),
            ("1-2:5,3-5:3", {}, "runs past MAX_NUM_SEQS=4"),
            ("1-4:3", {"SPEC": "none"}, "needs SPEC=dspark"),
            ("1-4:3", {"SPEC_CONFIG": '{"method":"dspark"}'}, "SPEC_CONFIG is set as well"),
        ):
            proc = _validate(NUM_SPECULATIVE_TOKENS_PER_BATCH=spec, **extra)
            self.assertNotEqual(proc.returncode, 0, spec)
            self.assertIn(why, proc.stderr, spec)
            self.assertIn("e.g. 1-2:5,3-4:3", proc.stderr, spec)
            self.assertNotIn("validate-only", proc.stdout, spec)

    @unittest.skipIf(hasattr(sys.modules.get("sitecustomize"), "_patch"),
                     "host only: in the image, sitecustomize prints into the docker stub's output, and run.sh's "
                     "preflight reads that as a running container")
    def test_dry_run_gives_both_ranks_the_same_tiered_configs(self) -> None:
        import json

        sys.path.insert(0, str(ROOT / "tests"))
        import run_sh_harness as h

        res = h.dry_run(NUM_SPECULATIVE_TOKENS_PER_BATCH="1-2:5,3-4:3", DSV41_DSPARK_ADAPTIVE="1")
        self.assertEqual(res["returncode"], 0, res["stdout"] + res["stderr"])
        got = {}
        for role in ("head", "worker"):
            _, args = h.image_and_args(res[role])
            got[role] = (h.flag_value(args, "--speculative-config"), h.flag_value(args, "--compilation-config"))
            self.assertEqual(h.container_env(res[role])["DSV41_DSPARK_ADAPTIVE"], "1", role)
        self.assertEqual(got["head"], got["worker"])
        spec, comp = (json.loads(x) for x in got["head"])
        self.assertEqual(spec["num_speculative_tokens"], 5)
        self.assertEqual(spec["num_speculative_tokens_per_batch_size"], [[1, 2, 5], [3, 4, 3]])
        self.assertEqual(comp["cudagraph_capture_sizes"], [1, 4, 5, 6, 8, 10, 12, 15, 16, 20])


class EndToEndScriptTests(unittest.TestCase):
    def test_script_compiles_on_the_host(self) -> None:
        compile(E2E_SCRIPT, "dspark_adaptive_e2e", "exec")

    @unittest.skipUnless(HAVE_SRC, "needs a vLLM tree")
    def test_the_kernel_files_it_copies_exist(self) -> None:
        for rel in E15_FILES:
            self.assertTrue((SRC / rel).is_file(), rel)


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs a vLLM tree, torch and triton (image)")
class EndToEndInterpreterTest(unittest.TestCase):
    TRIALS = 12

    def test_masked_k5_emits_what_truncated_k_prime_emits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dspark-adaptive-e2e-") as tmp:
            tmp = Path(tmp)
            for rel, text in STUBS.items():
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                (tmp / rel).write_text(text)
            for rel in E15_FILES:
                shutil.copy2(SRC / rel, tmp / "vllm" / rel)
            res = subprocess.run(
                [sys.executable, "-c", E2E_SCRIPT, str(tmp), str(PATCH), str(self.TRIALS)],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "TRITON_INTERPRET": "1"},
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            print("\n  " + res.stdout.strip().replace("\n", "\n  "))


if __name__ == "__main__":
    unittest.main()
