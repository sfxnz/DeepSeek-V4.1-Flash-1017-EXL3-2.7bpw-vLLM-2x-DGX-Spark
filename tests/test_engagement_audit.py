#!/usr/bin/env python3
"""tools/engagement_audit.py on canned boot logs (CPU only)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import engagement_audit as ea  # noqa: E402

DEFAULT_ENV = {
    "DSV41_LMHEAD_MXFP8": "1",
    "DSV41_ENGRAM_PREFETCH": "1",
    "DSV41_ENGRAM_GATHER_V2": "1",
    "DSV41_DROP_PAGE_CACHE": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
    "ENFORCE_EAGER": "0",
}
# Lines as they appear in the 2026-09-22 live boot logs (both ranks).
ENGAGED = """\
(Worker_TP0 pid=618) dsv41: lm_head mxfp8 enabled (b12x, (64640, 5120))
(Worker_TP0 pid=618) dsv41: engram prefetch v3 armed (ngram=4 heads=8 depth=3 layers=[0, 1])
(Worker_TP0 pid=618) [dsv41-drop-page-cache] dropped page cache of 48 shard files: MemFree 15.78GiB -> 37.55GiB
(Worker_TP0 pid=618) [woa-requant] fp8 einsum engaged: (4, 1024, 4096)
(Worker_TP0 pid=618) [woa-requant] fp8 einsum engaged: (4, 1024, 4096)
(Worker_TP0 pid=618) INFO 09-22 12:03:56 [breakable_cudagraph.py:290] Breakable CUDA graph enabled
(Worker_TP0 pid=618) dsv41: engram gather v2 self-check bit-exact (r=120)
[dsv41-prefill-empty-cache] skipped #1 (longest seq 9000, MemAvailable 3.1 GiB)
dsv41: chat encoder 1017 parity engaged (encoding rewritten, tokenizer rewritten)
"""


class MarkerSourceTests(unittest.TestCase):
    def test_every_marker_is_printed_by_its_patch(self) -> None:
        for name in ea.PATCHES:
            src = (ea.PATCH_DIR / name).read_text()
            found = ea.markers(ea.PATCH_DIR / name)
            self.assertIn("LOG_DISARMED", found, name)
            for marker in found.get("LOG_ENGAGED", ()) + found["LOG_DISARMED"]:
                # once in the constant, at least once where it is printed
                self.assertGreaterEqual(src.count(marker), 2, f"{name}: {marker!r}")

    def test_gated_patches_define_an_engaged_marker(self) -> None:
        for name, gate in ea.PATCHES.items():
            if gate is not None:
                self.assertIn("LOG_ENGAGED", ea.markers(ea.PATCH_DIR / name), name)

    def test_gates_are_forwarded_env_names(self) -> None:
        run = (ROOT / "run.sh").read_text()
        for gate in list(ea.PATCHES.values()) + [ea.BREAKABLE_GATE, ea.GATEWAY_GATE]:
            for env in gate or {}:
                self.assertIn(f"{env}=", run, env)


GATEWAY_ENGAGED = ("dsv41-gateway: engaged deepseek-recipe 0.2.0 upstream=http://127.0.0.1:8001 "
                   "model=deepseek-ai/DeepSeek-V4.1-Flash-1017\n")


class GatewayAuditTests(unittest.TestCase):
    """API_GATEWAY=1: the gateway's own log holds its startup line; no patch marker is expected there."""

    def test_the_marker_is_the_gateways_startup_line(self) -> None:
        self.assertEqual(ea.markers(ea.GATEWAY_SRC)["LOG_ENGAGED"], ("dsv41-gateway: engaged deepseek-recipe",))
        self.assertIn('print(f"{LOG_ENGAGED} {d.__version__} upstream=', ea.GATEWAY_SRC.read_text())
        self.assertIn('logs+=("gateway=$RUN_STATE/audit-gateway-$ts.log")', (ROOT / "run.sh").read_text())

    def test_gate_on_needs_the_line_in_the_gateway_log_only(self) -> None:
        env = dict(DEFAULT_ENV, API_GATEWAY="1")
        ranks = {"head": ENGAGED, "worker": ENGAGED}
        self.assertEqual(ea.audit({**ranks, "gateway": GATEWAY_ENGAGED}, env), [])
        missing = ["gateway: missing 'dsv41-gateway: engaged deepseek-recipe' (gateway/dsv41_gateway.py)"]
        self.assertEqual(ea.audit({**ranks, "gateway": "INFO:     Uvicorn running on http://0.0.0.0:8000\n"}, env), missing)
        self.assertEqual(ea.audit(ranks, env), missing)  # no gateway log at all
        # A vLLM rank needs no gateway line, and the gateway log needs no patch marker.
        self.assertEqual(ea.audit({"head": ENGAGED, "gateway": GATEWAY_ENGAGED}, env), [])

    def test_gate_off_expects_no_gateway(self) -> None:
        for env in (DEFAULT_ENV, dict(DEFAULT_ENV, API_GATEWAY="0")):
            self.assertEqual(ea.audit({"head": ENGAGED}, env), [])
            self.assertEqual(ea.audit({"head": ENGAGED, "gateway": ""}, env), [])

    def test_cli_strict_fails_without_the_line(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            head, gateway = Path(d) / "head.log", Path(d) / "gateway.log"
            head.write_text(ENGAGED)
            gateway.write_text(GATEWAY_ENGAGED)
            argv = [sys.executable, str(ROOT / "tools/engagement_audit.py"), "--mode", "strict",
                    f"head={head}", f"gateway={gateway}"]
            env = dict(DEFAULT_ENV, API_GATEWAY="1", PATH="/usr/bin:/bin")
            proc = subprocess.run(argv, env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("audit ok: every expected patch engaged on head, gateway", proc.stdout)
            gateway.write_text("ModuleNotFoundError: No module named 'deepseek_recipe'\n")
            proc = subprocess.run(argv, env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("WARNING audit gateway: missing 'dsv41-gateway: engaged deepseek-recipe'", proc.stderr)


class AuditTests(unittest.TestCase):
    def test_live_like_logs_pass_on_both_ranks(self) -> None:
        self.assertEqual(ea.audit({"head": ENGAGED, "worker": ENGAGED}, DEFAULT_ENV), [])

    def test_benign_prefill_skip_is_not_a_disarm(self) -> None:
        self.assertEqual(ea.audit({"head": ENGAGED}, DEFAULT_ENV), [])

    def test_missing_marker_names_rank_and_patch(self) -> None:
        log = ENGAGED.replace("dsv41: lm_head mxfp8 enabled", "")
        problems = ea.audit({"head": ENGAGED, "worker": log}, DEFAULT_ENV)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith("worker: missing"))
        self.assertIn("lmhead_mxfp8.py", problems[0])

    def test_disarm_lines_are_reported_once(self) -> None:
        bad = ENGAGED + (
            "dsv41: lm_head mxfp8 self-disarmed (no lm_head.weight_scale under /snap)\n"
            "(Worker_TP1 pid=1) dsv41: engram prefetch disabled after 3 errors: IndexError()\n"
            "dsv41: engram gather v2 DISABLED -> stock path: OSError()\n"
            "dsv41-patch FAIL swa_page_coerce: ImportError()\n"
            "dsv41-patch FAIL swa_page_coerce: ImportError()\n"
            "dsv41: drop-page-cache install skipped: ImportError()\n"
        )
        problems = ea.audit({"head": bad}, DEFAULT_ENV)
        self.assertEqual(len(problems), 5, problems)

    def test_decode_lever_off_lines_are_disarms(self) -> None:
        # Printed by docker/patch/decode_levers.py (image dry-run, scenario B).
        bad = ENGAGED + (
            "dsv41: decode lever sparse-markov FAILED, lever is OFF: ValueError('x')\n"
            "dsv41: WARNING DSV41_WOA_PREPACK=1 but this image's o_proj.py has no "
            "prepack stage; the lever is OFF. Build docker/Dockerfile.woa-prepack.\n"
        )
        problems = ea.audit({"head": bad}, DEFAULT_ENV)
        self.assertEqual(len(problems), 2, problems)

    def test_opt_in_lever_self_rejects_are_disarms(self) -> None:
        # A lever whose load self-test rejects every layer must not read "audit ok".
        bad = ENGAGED + (
            "[woa-prepack] REJECTED (ValueError('self-test not bitwise equal')); this layer "
            "keeps the fp32 scale (per-call repack)\n"
            "[dense-dg] rank0 K1024xN1792 rejected (RuntimeError('DeepGEMM E8M0 not in use')); b12x stays\n"
            "[dense-dg] rank1 K4096xN2304 armed but got QuantizedActivation input; b12x runs\n"
            "[dense-dg] rank0 K1024xN1792 armed (M<=8 via deep_gemm fp8_gemm_nt; graph capture ok at M=4)\n"
        )
        problems = ea.audit({"head": bad}, DEFAULT_ENV)
        self.assertEqual(len(problems), 3, problems)
        self.assertFalse(any("graph capture ok" in p for p in problems))

    def test_gates_off_expect_nothing(self) -> None:
        env = dict(DEFAULT_ENV, DSV41_LMHEAD_MXFP8="0", ENFORCE_EAGER="1")
        log = ENGAGED.replace("dsv41: lm_head mxfp8 enabled", "").replace("Breakable CUDA graph enabled", "")
        self.assertEqual(ea.audit({"head": log}, env), [])
        expected, _ = ea.expectations({})
        self.assertEqual([label for label, _ in expected], ["fix_o_proj_woa_fp8.py", "chat_encoder_1017.py"])

    def test_cli_warn_exits_0_strict_exits_1(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "head.log"
            log.write_text("dsv41: lm_head mxfp8 self-disarmed (no key)\n")
            argv = [sys.executable, str(ROOT / "tools/engagement_audit.py")]
            env = dict(DEFAULT_ENV, PATH="/usr/bin:/bin")
            for mode, rc in (("warn", 0), ("strict", 1), ("off", 0)):
                proc = subprocess.run(argv + ["--mode", mode, f"head={log}"], env=env, capture_output=True, text=True)
                self.assertEqual(proc.returncode, rc, (mode, proc.stderr))
                if mode != "off":
                    self.assertIn("WARNING audit head: dsv41: lm_head mxfp8 self-disarmed", proc.stderr)
            log.write_text(ENGAGED)
            proc = subprocess.run(argv + ["--mode", "strict", f"head={log}"], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("audit ok", proc.stdout)
            proc = subprocess.run(argv + ["--mode", "strict", f"worker={d}/missing.log"], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("cannot read", proc.stderr)


def _pack(**q) -> dict:
    return {"quantization_config": {"quant_method": "exl3", "bits": 2, **q}}


K3_PACK = _pack(expert_bits={str(n): ([3, 3] if n < 28 else [2, 2]) for n in range(40)})
COOP = dict(DEFAULT_ENV, DSV41_P2B_COOP="2")
KBITS_PATCH = (
    'DATAFLOW_K = (2, 3, 4)\n'
    'DATAFLOW_MIXED_K = False\n'
    'LOG_ENGAGED_K = "dsv41: p2b dataflow engaged K={kgu}/{kd}"\n'
)


class KAwareTests(unittest.TestCase):
    """1017 mixed-K packs: the dataflow lever must cover every routed (K_gu, K_d) of the pack."""

    def test_pack_k_layers_prefers_expert_bits_then_layer_bits_then_bits(self) -> None:
        self.assertEqual(ea.pack_k_layers(_pack()), {(2, 2): list(range(40))})
        lb = ea.pack_k_layers(_pack(layer_bits={"0": 3, "1": 4}))
        self.assertEqual(lb[(3, 3)], [0])
        self.assertEqual(lb[(4, 4)], [1])
        self.assertEqual(len(lb[(2, 2)]), 38)
        eb = ea.pack_k_layers(_pack(layer_bits={"0": 4}, expert_bits={str(n): [3, 2] for n in range(40)}))
        self.assertEqual(eb, {(3, 2): list(range(40))})

    def test_k2_pack_on_the_k2_patch_set_is_covered(self) -> None:
        # widen_p2b_kbits.py's per-layer plugin line (LOG_ENGAGED_K) is the only per-K expectation.
        boot = [("p2b dataflow K", t.format(kgu=2, kd=2)) for t in ea.dataflow_support()["engaged_k"]]
        self.assertEqual(ea.coverage(_pack(), COOP), ([], boot))
        self.assertEqual(ea.audit({"head": ENGAGED}, COOP | {"DSV41_P2B_COOP": "0"}, K3_PACK), [])

    def test_k3_pack_without_a_k3_kernel_fails_even_with_clean_logs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "widen_p2b_dataflow.py").write_text("x = 1\n")
            problems, _ = ea.coverage(K3_PACK, COOP, Path(d))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("K_gu/K_d=3/3 (layers 0-27)", problems[0])

    def test_the_kbits_patch_file_declares_k234_and_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "widen_p2b_kbits.py").write_text("x = 1\n")
            sup = ea.dataflow_support(Path(d))
        self.assertEqual((sup["k"], sup["mixed"]), ({2, 3, 4}, True))
        mixed = _pack(expert_bits={str(n): [3, 2] for n in range(40)})
        if (ea.PATCH_DIR / "widen_p2b_kbits.py").is_file():
            self.assertEqual(ea.coverage(mixed, COOP)[0], [], "this tree's patch set covers the 1017 pack")

    def test_declared_constants_cover_k3_and_expect_per_k_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "some_kernel_patch.py").write_text(KBITS_PATCH)
            sup = ea.dataflow_support(Path(d))
            self.assertEqual(sup["k"], {2, 3, 4})
            problems, expected = ea.coverage(K3_PACK, COOP, Path(d))
            self.assertEqual(problems, [])
            self.assertEqual(sorted(m for _, m in expected),
                             ["dsv41: p2b dataflow engaged K=2/2", "dsv41: p2b dataflow engaged K=3/3"])
            mixed, _ = ea.coverage(_pack(expert_bits={str(n): [3, 2] for n in range(40)}), COOP, Path(d))
            self.assertEqual(len(mixed), 1)
            self.assertIn("3/2", mixed[0])
            (Path(d) / "some_kernel_patch.py").write_text(KBITS_PATCH.replace("False", "True"))
            self.assertEqual(ea.coverage(_pack(expert_bits={str(n): [3, 2] for n in range(40)}), COOP, Path(d))[0], [])

    def test_strict_boot_on_the_kbits_kernel_passes_and_a_real_disarm_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "widen_p2b_kbits.py").write_text(KBITS_PATCH)
            orig = ea.PATCH_DIR
            try:
                for name in list(ea.PATCHES) + ["widen_p2b_kbits.py"]:
                    src = orig / name
                    if src.is_file():
                        (Path(d) / name).write_text(src.read_text())
                (Path(d) / "widen_p2b_kbits.py").write_text(KBITS_PATCH)
                ea.PATCH_DIR = Path(d)
                env = dict(COOP, DSV41_DENSE_GEMV="0", DSV41_MHC_DET_SPLITS="0")
                good = ENGAGED + (
                    "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)\n"
                    "dsv41: p2b dataflow engaged K=3/3\n"
                    "dsv41: p2b dataflow engaged K=2/2\n")
                self.assertEqual(ea.audit({"head": good, "worker": good}, env, K3_PACK), [])
                off = good + "dsv41: p2b coop dataflow lever is OFF for this call (K=3 mcg=1 m=4 top_k=6)\n"
                problems = ea.audit({"head": good, "worker": off}, env, K3_PACK)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn("worker: dsv41: p2b coop dataflow lever is OFF", problems[0])
                missing = good.replace("dsv41: p2b dataflow engaged K=3/3\n", "")
                problems = ea.audit({"head": missing}, env, K3_PACK)
                self.assertEqual(problems, ["head: missing 'dsv41: p2b dataflow engaged K=3/3' (p2b dataflow K)"])
            finally:
                ea.PATCH_DIR = orig

    def test_cli_coverage_only_and_pack_config(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            k5 = _pack(expert_bits={str(n): ([5, 5] if n < 28 else [2, 2]) for n in range(40)})  # no kernel takes K=5
            cfg.write_text(json.dumps(k5))
            argv = [sys.executable, str(ROOT / "tools/engagement_audit.py")]
            env = dict(COOP, PATH="/usr/bin:/bin")
            proc = subprocess.run(argv + ["--coverage-only", "--pack-config", str(cfg)], env=env,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertTrue(proc.stdout.startswith("pack: K_gu/K_d=5/5 (layers 0-27)"), proc.stdout)
            log = Path(d) / "head.log"
            log.write_text(ENGAGED + "dsv41: p2b coop dataflow kernel engaged (DSV41_P2B_COOP=2)\n")
            env = dict(env, DSV41_DENSE_GEMV="0", DSV41_MHC_DET_SPLITS="0")
            proc = subprocess.run(argv + ["--mode", "strict", "--pack-config", str(cfg), f"head={log}"], env=env,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertIn("WARNING audit pack: K_gu/K_d=5/5", proc.stderr)
            cfg.write_text(json.dumps(_pack()))
            log.write_text(log.read_text() + "dsv41: exl3 routed experts K_gu/K_d=2/2 layer 0 (bits)\n")
            proc = subprocess.run(argv + ["--mode", "strict", "--pack-config", str(cfg), f"head={log}"], env=env,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
