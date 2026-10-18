#!/usr/bin/env python3
"""tools/build_pack_1017.py: host logic on tiny synthetic shards (CPU, stdlib).

Torch-gated tests (verify / Hessian loader / proxy / lm_head unit) run where torch and
safetensors import, e.g. inside dsv41-quant151 without a GPU:
  docker run --rm --network none --memory 8g --user 1000:1000 -e HOME=/tmp -v $PWD:/repo -w /repo \\
    --entrypoint python3 dsv41-quant151 -m unittest tests.test_build_pack_1017 -v
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_pack_1017 as bp  # noqa: E402
import pack_meta  # noqa: E402

try:
    import torch  # noqa: F401
    import safetensors  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

HEADERS_1017 = Path("/home/sfxnz/projects/data/dsv41-1017/recipe-tooling/src-headers-1017")
ALLOC_K2 = {str(n): [2, 2] for n in range(40)}
ALLOC_270 = {str(n): ([3, 3] if n < 28 else [2, 2]) for n in range(40)}
NBYTES = {"I8": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "F8_E4M3": 1, "F8_E8M0": 1}


def write_st(path: Path, tensors: dict) -> None:
    """Minimal safetensors writer: {name: (dtype, shape, fill byte)}."""
    hdr, off, payload = {}, 0, bytearray()
    for name, (dtype, shape, fill) in tensors.items():
        n = NBYTES[dtype] * math.prod(shape)
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        payload += bytes([fill]) * n
        off += n
    raw = json.dumps(hdr).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(payload))


def tiny_source(root: Path, layers: int = 40, experts: int = 2) -> dict:
    """1017-shaped source at toy size: shard 1 vision, 2 embed, 3..42 layers, 43 head, 44 mtp, 47 engram."""
    shards: dict[str, dict] = {"model-00001-of-00048.safetensors": {"vision.blocks.0.attn.wqkv.weight": ("BF16", [4, 4], 1)},
                               "model-00002-of-00048.safetensors": {"embed.weight": ("BF16", [8, 4], 2)}}
    for layer in range(layers):
        t = {f"layers.{layer}.attn.wq_a.weight": ("F8_E4M3", [32, 32], 3),
             f"layers.{layer}.ffn.gate.weight": ("BF16", [experts, 32], 4)}
        for e in range(experts):
            for w, (out_f, in_f) in (("w1", (32, 32)), ("w3", (32, 32)), ("w2", (32, 32))):
                t[f"layers.{layer}.ffn.experts.{e}.{w}.weight"] = ("I8", [out_f, in_f // 2], 5)
                t[f"layers.{layer}.ffn.experts.{e}.{w}.scale"] = ("F8_E8M0", [out_f, in_f // 32], 6)
        t[f"layers.{layer}.ffn.shared_experts.w1.weight"] = ("F8_E4M3", [32, 32], 7)
        shards[f"model-{layer + 3:05d}-of-00048.safetensors"] = t
    shards["model-00043-of-00048.safetensors"] = {"head.weight": ("BF16", [64, 32], 8), "norm.weight": ("BF16", [32], 9)}
    shards["model-00044-of-00048.safetensors"] = {"mtp.0.ffn.experts.0.w1.weight": ("I8", [32, 16], 10),
                                                  "mtp.0.ffn.experts.0.w1.scale": ("F8_E8M0", [32, 1], 11)}
    shards["model-00047-of-00048.safetensors"] = {"layers.1.engram.embed.weight": ("F8_E4M3", [16, 8], 12)}
    root.mkdir(parents=True, exist_ok=True)
    wm = {}
    for fname, t in shards.items():
        write_st(root / fname, t)
        wm.update({k: fname for k in t})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (root / "config.json").write_text(json.dumps({"architectures": ["DeepseekV41ForCausalLM"],
                                                  "quantization_config": {"quant_method": "fp8"}}))
    (root / "tokenizer.json").write_text("{}")
    return wm


class AllocTests(unittest.TestCase):
    def test_validate_and_bpw(self) -> None:
        self.assertEqual(pack_meta.validate_expert_bits(ALLOC_K2)["39"], [2, 2])
        self.assertAlmostEqual(pack_meta.routed_bpw(ALLOC_270), 2.70)
        self.assertAlmostEqual(pack_meta.layer_bpw(3, 2), 8 / 3)
        for bad in ({**ALLOC_K2, "5": [5, 2]}, {k: v for k, v in ALLOC_K2.items() if k != "7"},
                    {**ALLOC_K2, "40": [2, 2]}, {**ALLOC_K2, "3": [3]}, {**ALLOC_K2, "3": [2.5, 2]}):
            with self.assertRaises(ValueError):
                pack_meta.validate_expert_bits(bad)

    def test_quantization_config_carries_expert_bits_and_uniform_fallback(self) -> None:
        eb = {**ALLOC_270, "39": [3, 2]}
        q = pack_meta.build_quantization_config(bits=2, codebook="mcg", expert_bits=eb)
        self.assertEqual(q["expert_bits"]["0"], [3, 3])
        self.assertEqual(q["expert_bits"]["39"], [3, 2])
        self.assertEqual(q["layer_bits"]["0"], 3)
        self.assertNotIn("39", q["layer_bits"], "a K_gu != K_d layer must not get a wrong uniform K")
        self.assertEqual((q["bits"], q["codebook"], q["mtp_experts"]), (2, "mcg", "source"))
        self.assertAlmostEqual(q["routed_bpw"], pack_meta.routed_bpw(eb), places=6)
        self.assertNotIn("expert_bits", pack_meta.build_quantization_config())

    def test_pins(self) -> None:
        self.assertEqual(pack_meta.SOURCE_REVISION, "1bae0e1088993b9b82de08f59de5be5a34e42170")
        self.assertEqual(pack_meta.PACK_REPO, "sfxnz/DeepSeek-V4.1-Flash-1017-EXL3")
        self.assertTrue(pack_meta.SOURCE_SNAPSHOT_REL.startswith("models--deepseek-ai--DeepSeek-V4.1-Flash-1017/"))
        self.assertEqual(bp.PACK_DIR, "models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3")


class PlanTests(unittest.TestCase):
    def test_names_and_k(self) -> None:
        self.assertEqual(bp.parse_expert("layers.12.ffn.experts.383.w2.weight"), (12, 383, "w2"))
        with self.assertRaises(ValueError):
            bp.parse_expert("mtp.0.ffn.experts.0.w1.weight")
        alloc = {**ALLOC_K2, "12": [4, 3]}
        self.assertEqual(bp.expert_k("layers.12.ffn.experts.0.w1.weight", alloc), 4)
        self.assertEqual(bp.expert_k("layers.12.ffn.experts.0.w3.weight", alloc), 4)
        self.assertEqual(bp.expert_k("layers.12.ffn.experts.0.w2.weight", alloc), 3)

    def test_entries_match_the_served_layout(self) -> None:
        w13 = bp.expert_entries("layers.0.ffn.experts.0.w1.weight", {"dtype": "I8", "shape": [2304, 2560]}, 3)
        self.assertEqual(w13["layers.0.ffn.experts.0.w1.trellis"], ("I16", [320, 144, 48]))
        self.assertEqual(w13["layers.0.ffn.experts.0.w1.suh"], ("F16", [5120]))
        self.assertEqual(w13["layers.0.ffn.experts.0.w1.svh"], ("F16", [2304]))
        self.assertEqual(w13["layers.0.ffn.experts.0.w1.mcg"], ("I32", []))
        w2 = bp.expert_entries("layers.0.ffn.experts.0.w2.weight", {"dtype": "I8", "shape": [5120, 1152]}, 2)
        self.assertEqual(w2["layers.0.ffn.experts.0.w2.trellis"], ("I16", [144, 320, 32]))

    def test_units(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            wm = tiny_source(Path(d))
        units = bp.plan_units(wm)
        self.assertEqual(len(units), 41)
        self.assertEqual(units["00003"], {"file": "model-00003-of-00048.safetensors", "kind": "experts", "layer": 0})
        self.assertEqual(units["00043"]["kind"], "lmhead")
        bad = dict(wm, **{"layers.1.ffn.experts.0.w1.weight": "model-00003-of-00048.safetensors"})
        with self.assertRaises(ValueError):
            bp.plan_units(bad)

    @unittest.skipUnless((HEADERS_1017 / "model-00003-of-00048.safetensors.json").is_file(), "1017 headers not saved")
    def test_real_headers_k2_shard_size_and_rank_gib(self) -> None:
        hdrs = {p.name[:-5]: json.loads(p.read_text()) for p in HEADERS_1017.glob("model-*.json")}
        wm = json.loads((HEADERS_1017 / "model.safetensors.index.json").read_text())["weight_map"]
        plan = bp.build_plan(hdrs, wm, ALLOC_K2, 1.92, "real")
        # Sept 2.0bpw-mcg shard 3 (same tensor layout): 3,585,021,280 bytes.
        self.assertLess(abs(plan["units"]["00003"]["bytes"] - 3_585_021_280), 65536)
        self.assertAlmostEqual(plan["rank"]["total_gib"], bp.SEPT_RANK_GIB, places=6)
        plan = bp.build_plan(hdrs, wm, ALLOC_270, 1.92, "real")
        self.assertAlmostEqual(plan["rank"]["total_gib"], 97.73, delta=0.05)
        self.assertAlmostEqual(plan["hessian_bytes_per_layer"] / 1e9, 48.4, delta=0.1)
        self.assertEqual(set(plan["linked_files"]), {f"model-000{n:02d}-of-00048.safetensors" for n in (1, 2, 44, 45, 46, 47, 48)})
        self.assertEqual(len(plan["nodes"]["spark1"]["units"]) + len(plan["nodes"]["spark2"]["units"]), 41)

    def test_plan_cli_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            tiny_source(src)
            alloc = Path(d) / "alloc.json"
            alloc.write_text(json.dumps(ALLOC_270))
            for cmd in (["plan"], ["run", "--dry-run"]):
                proc = subprocess.run([sys.executable, str(ROOT / "tools/build_pack_1017.py"), *cmd, "--src", str(src),
                                       "--alloc", str(alloc), "--out", str(Path(d) / "out")],
                                      capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("routed bpw (trellis, param-weighted): 2.7000", proc.stdout)
                self.assertIn("per rank (TP=2)", proc.stdout)
                self.assertIn("spark2:", proc.stdout)
                self.assertFalse((Path(d) / "out").exists(), "a dry run writes nothing")

    def test_node_split_meets_in_the_middle(self) -> None:
        units = {f"{n:05d}": {} for n in range(3, 44)}
        cost = {u: 1.0 for u in units}
        split = bp.node_split(units, cost)
        self.assertEqual(split["spark1"] + split["spark2"], sorted(units))
        self.assertLessEqual(abs(len(split["spark1"]) - len(split["spark2"])), 1)


def _rows(group: str, k: int, rel: float, proxy=None, n: int = 4) -> list[dict]:
    return [{"group": group, "k": k, "relerr": rel, **({"proxy": proxy} if proxy is not None else {})} for _ in range(n)]


class GateTests(unittest.TestCase):
    def test_k_specific_limits(self) -> None:
        g = bp.GATES
        ok = bp.gate_unit(_rows("gu", 3, 0.15, 0.001) + _rows("d", 2, 0.30, 0.004), [3, 2], g, True)
        self.assertTrue(ok["ok"], ok)
        bad = bp.gate_unit(_rows("gu", 3, 0.30, 0.001) + _rows("d", 2, 0.30, 0.004), [3, 2], g, True)
        self.assertFalse(bad["ok"])
        self.assertFalse(bad["groups"]["gu"]["ok"])
        self.assertTrue(bad["groups"]["d"]["ok"])

    def test_proxy_gate_and_missing_proxy(self) -> None:
        g = bp.GATES
        self.assertFalse(bp.gate_unit(_rows("gu", 4, 0.07, 0.05) + _rows("d", 4, 0.07, 0.001), [4, 4], g, True)["ok"])
        self.assertFalse(bp.gate_unit(_rows("gu", 4, 0.07) + _rows("d", 4, 0.07), [4, 4], g, True)["ok"])
        self.assertTrue(bp.gate_unit(_rows("gu", 4, 0.07) + _rows("d", 4, 0.07), [4, 4], g, False)["ok"])

    def test_non_finite_and_empty_fail(self) -> None:
        g = bp.GATES
        self.assertFalse(bp.gate_unit(_rows("gu", 2, float("nan")) + _rows("d", 2, 0.26), [2, 2], g, False)["ok"])
        self.assertFalse(bp.gate_unit([], [2, 2], g, False)["ok"])
        self.assertFalse(bp.gate_unit(_rows("gu", 2, 0.26), [2, 2], g, False)["ok"], "a group with no rows fails")

    def test_gate_override_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "g.json"
            f.write_text(json.dumps({"relerr_mean": {"3": 0.5}}))
            g = bp._gates(type("A", (), {"gates": str(f)})())
        self.assertEqual(g["relerr_mean"][3], 0.5)
        self.assertEqual(g["relerr_mean"][2], bp.GATES["relerr_mean"][2])


class VerifyTests(unittest.TestCase):
    def test_layout_and_copied_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "src.st", Path(d) / "out.st"
            write_st(src, {"a": ("BF16", [4], 1), "layers.0.ffn.experts.0.w1.weight": ("I8", [32, 16], 2),
                           "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [32, 1], 3)})
            write_st(out, {"layers.0.ffn.experts.0.w1.trellis": ("I16", [2, 2, 48], 0),
                           "layers.0.ffn.experts.0.w1.suh": ("F16", [32], 0), "a": ("BF16", [4], 1),
                           "layers.0.ffn.experts.0.w1.svh": ("F16", [32], 0),
                           "layers.0.ffn.experts.0.w1.mcg": ("I32", [], 0)})
            copied, encoded = bp.expected_layout(bp.read_header(src)[1], {**ALLOC_K2, "0": [3, 2]})
            self.assertEqual(bp.layout_problems(bp.read_header(out)[1], {**copied, **encoded}), [])
            self.assertEqual(bp.copied_bytes_problems(out, src, copied), [])
            write_st(out, {"layers.0.ffn.experts.0.w1.trellis": ("I16", [2, 2, 32], 0), "a": ("BF16", [4], 7)})
            probs = bp.layout_problems(bp.read_header(out)[1], {**copied, **encoded})
            self.assertTrue(any("missing 3 keys" in p for p in probs), probs)
            self.assertTrue(any("trellis" in p and "48" in p for p in probs), probs)
            self.assertEqual(bp.copied_bytes_problems(out, src, ["a"]), ["a: bytes differ from the 1017 source"])

    def test_expected_pack_names_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            wm = tiny_source(Path(d))
            names = bp.expected_pack_names(wm)
            self.assertIn("layers.0.ffn.experts.1.w2.trellis", names)
            self.assertNotIn("layers.0.ffn.experts.1.w2.scale", names)
            self.assertIn("mtp.0.ffn.experts.0.w1.weight", names, "drafter experts stay source MXFP4")
            self.assertIn("lm_head.weight_scale", names)
            self.assertNotIn("head.weight", names)
            self.assertIn("layers.1.engram.embed.weight", names)
            idx = bp.build_index(Path(d))
            self.assertEqual(set(idx["weight_map"]), set(wm))
            self.assertGreater(idx["metadata"]["total_size"], 0)


class StateTests(unittest.TestCase):
    def test_hessian_ready(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertTrue(bp.hessian_ready(None, 3), "identity mode needs no Hessians")
            self.assertFalse(bp.hessian_ready(root, 3))
            (root / "layer_03").mkdir()
            meta = {"format": bp.HESSIAN_FORMAT, "layer": 3, "complete": False}
            (root / "layer_03" / "meta.json").write_text(json.dumps(meta))
            self.assertFalse(bp.hessian_ready(root, 3))
            (root / "layer_03" / "meta.json").write_text(json.dumps(meta | {"complete": True}))
            self.assertTrue(bp.hessian_ready(root, 3))
            self.assertFalse(bp.hessian_ready(root, 4))
            (root / "layer_03" / "meta.json").write_text("{not json")
            self.assertFalse(bp.hessian_ready(root, 3))

    def test_run_layer_selection(self) -> None:
        self.assertEqual(bp.parse_layer_list("1,3, 5-7"), {1, 3, 5, 6, 7})
        exp, head = {"kind": "experts", "layer": 3}, {"kind": "lmhead", "layer": None}
        self.assertTrue(bp.unit_selected(exp, None, False, None))
        self.assertTrue(bp.unit_selected(head, None, False, None))
        self.assertTrue(bp.unit_selected(exp, {1, 3}, False, None))
        self.assertFalse(bp.unit_selected(exp, {0, 2}, True, None))
        self.assertFalse(bp.unit_selected(head, {1, 3}, False, None), "--layers without --lmhead skips lm_head")
        self.assertTrue(bp.unit_selected(head, {0, 2}, True, None))
        self.assertFalse(bp.unit_selected(head, None, True, 8), "a --limit test run never takes lm_head")
        self.assertTrue(bp.unit_selected(exp, {3}, False, 8))

    def test_pin_state_refuses_a_different_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store = bp.LocalStore(Path(d))
            bp.pin_state(store, "alloc.json", ALLOC_K2)
            bp.pin_state(store, "alloc.json", ALLOC_K2)
            with self.assertRaises(SystemExit):
                bp.pin_state(store, "alloc.json", ALLOC_270)

    def test_manifest_done(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store = bp.LocalStore(Path(d))
            store.mkdir("done")
            units = {"00003": {}, "00043": {}}
            gate = {"ok": True, "groups": {"gu": {"relerr_mean": 0.15}, "d": {"relerr_mean": 0.3}}}
            store.write("done/00003.json", json.dumps({"kind": "experts", "gate": gate, "rows": [{"x": 1}]}))
            man = bp.refresh_manifest(store, units, ALLOC_270, {"revision": "r"})
            self.assertFalse(man["complete"])
            self.assertNotIn("rows", man["units"]["00003"])
            store.write("done/00043.json", json.dumps({"kind": "lmhead", "gate": {"ok": True}}))
            man = bp.refresh_manifest(store, units, ALLOC_270, {"revision": "r"})
            self.assertTrue(man["complete"])
            self.assertAlmostEqual(man["relerr_mean_by_group"]["gu"], 0.15)
            self.assertIsNotNone(store.read("DONE"))

    def test_launch_argv_is_campaign_safe(self) -> None:
        argv = bp.launch_argv("spark2", "desc", alloc="/a.json", hessians="/h", state_host="spark1",
                              gpu_lock="/home/sfxnz/projects/data/dsv41-1017/spark2-gpu.lock")
        self.assertIn("--rm", argv)
        self.assertEqual(argv[argv.index("--name") + 1], "dsv41-1017-build-spark2")
        self.assertEqual(argv[argv.index("--memory") + 1], "32g")
        self.assertIn("/h:/h:ro", argv)
        self.assertIn("--wait-hessians", argv)
        self.assertEqual(argv[argv.index("--hessian-mode") + 1], "real")
        ident = bp.launch_argv("spark1", "asc", alloc="/a.json", hessians=None, hessian_mode="identity")
        self.assertNotIn("--hessians", ident)
        self.assertEqual(ident[ident.index("--network") + 1], "none")


class AssembleTests(unittest.TestCase):
    def test_assemble_links_source_shards_and_writes_config(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src, out, hub = d / "src", d / "out", d / "hub"
            wm = tiny_source(src, layers=40, experts=1)
            units = bp.plan_units(wm)
            store = bp.LocalStore(out / "state")
            (out / "state").mkdir(parents=True)
            (out / "shards").mkdir()
            bp.init_state(store, units)
            alloc = {**ALLOC_270, "39": [3, 2]}
            store.write("alloc.json", json.dumps(alloc))
            store.write("build.json", json.dumps({"revision": "test-rev"}))
            src_hdrs = {f: bp.read_header(src / f)[1] for f in set(wm.values())}
            for u, info in units.items():
                hdr = src_hdrs[info["file"]]
                if info["kind"] == "lmhead":
                    lay = bp.lmhead_layout(hdr)
                else:
                    copied, enc = bp.expected_layout(hdr, alloc)
                    lay = {**copied, **enc}
                write_st(out / "shards" / info["file"], {k: (dt, sh, 0) for k, (dt, sh) in lay.items()})
                f = out / "shards" / info["file"]
                gate = {"ok": True, "groups": {"gu": {"relerr_mean": 0.1}, "d": {"relerr_mean": 0.2}}}
                store.write(f"done/{u}.json", json.dumps({"kind": info["kind"], "file": info["file"], "node": "spark1",
                                                          "size": f.stat().st_size, "sha256": bp.sha256_file(f),
                                                          "gate": gate, "rows": [{"tensor": "t", "relerr": 0.1}]}))
            bp.refresh_manifest(store, units, alloc, {"revision": "test-rev"})
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = bp.main(["assemble", "--out", str(out), "--src", str(src), "--node", "spark1", "--hub", str(hub)])
            self.assertEqual(rc, 0)
            snap = hub / bp.PACK_DIR / "snapshots" / "test-rev"
            cfg = json.loads((snap / "config.json").read_text())["quantization_config"]
            self.assertEqual(cfg["quant_method"], "exl3")
            self.assertEqual(cfg["expert_bits"]["39"], [3, 2])
            self.assertEqual(cfg["mtp_experts"], "source")
            self.assertEqual((snap / "model-00047-of-00048.safetensors").stat().st_ino,
                             (src / "model-00047-of-00048.safetensors").stat().st_ino, "Engram shard hard-linked")
            self.assertEqual((snap / "model-00010-of-00048.safetensors").stat().st_ino,
                             (out / "shards" / "model-00010-of-00048.safetensors").stat().st_ino)
            idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(idx["lm_head.weight"], "model-00043-of-00048.safetensors")
            self.assertTrue((snap / "tokenizer.json").is_file())
            man = json.loads((snap / "build-1017-manifest.json").read_text())
            self.assertEqual(man["rows"]["00003"][0]["relerr"], 0.1)
            # a shard whose bytes no longer match its done record stops the assembly
            os.unlink(snap / "model-00003-of-00048.safetensors")
            (out / "shards" / "model-00003-of-00048.safetensors").write_bytes(b"x")
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                bp.main(["assemble", "--out", str(out), "--src", str(src), "--node", "spark1", "--hub", str(hub)])


@unittest.skipUnless(HAVE_TORCH, "torch + safetensors (run in dsv41-quant151)")
class TorchTests(unittest.TestCase):
    def test_load_hessian_checks(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as d:
            ld = Path(d) / "layer_02"
            ld.mkdir()
            a = torch.randn(64, 32)
            h = (a @ a.T / 32).float()
            save_file({"w13_in": h.contiguous(), "w2_in": torch.eye(32)}, str(ld / "expert_005.safetensors"))
            self.assertTrue(torch.equal(bp.load_hessian(Path(d), 2, 5, "gu", 64), h))
            self.assertEqual(tuple(bp.load_hessian(Path(d), 2, 5, "d", 32).shape), (32, 32))
            with self.assertRaises(ValueError):
                bp.load_hessian(Path(d), 2, 5, "gu", 32)
            bad = h.clone()
            bad[0, 1] += 1.0
            save_file({"w13_in": bad.contiguous(), "w2_in": torch.eye(32)}, str(ld / "expert_006.safetensors"))
            with self.assertRaises(ValueError):
                bp.load_hessian(Path(d), 2, 6, "gu", 64)
            save_file({"w13_in": h.double().contiguous(), "w2_in": torch.eye(32)}, str(ld / "expert_007.safetensors"))
            with self.assertRaises(ValueError):
                bp.load_hessian(Path(d), 2, 7, "gu", 64)

    def test_proxy_error_identity_is_relerr_squared(self) -> None:
        import torch

        w = torch.randn(64, 48, dtype=torch.float64)
        q = w + 0.1 * torch.randn(64, 48, dtype=torch.float64)
        rel = ((w - q).norm() / w.norm()).item()
        self.assertAlmostEqual(bp.proxy_error(w, q, torch.eye(64, dtype=torch.float64)), rel ** 2, places=10)
        h = torch.diag(torch.tensor([1.0] + [0.0] * 63, dtype=torch.float64))
        self.assertAlmostEqual(bp.proxy_error(w, q, h), ((w - q)[0].square().sum() / w[0].square().sum()).item(), places=10)

    def test_h_data_is_a_copy_normalized_once(self) -> None:
        import torch

        h = torch.eye(16)
        rec = bp.h_data(h, "cpu", "k")
        self.assertEqual(rec["count"], 1)
        rec["H"].mul_(2)
        self.assertTrue(torch.equal(h, torch.eye(16)), "exllamav3 mutates H in place; the proxy keeps the original")

    def test_finish_unit_writes_verifies_and_rejects(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "src.safetensors"
            from safetensors.torch import save_file

            save_file({"a": torch.ones(4, dtype=torch.bfloat16),
                       "layers.0.ffn.experts.0.w1.weight": torch.zeros(32, 16, dtype=torch.int8),
                       "layers.0.ffn.experts.0.w1.scale": torch.zeros(32, 1, dtype=torch.uint8)}, str(src))
            copied, enc = bp.expected_layout(bp.read_header(src)[1], {**ALLOC_K2, "0": [3, 3]})
            tensors = {"a": torch.ones(4, dtype=torch.bfloat16),
                       "layers.0.ffn.experts.0.w1.trellis": torch.arange(2 * 2 * 48, dtype=torch.int16).view(2, 2, 48),
                       "layers.0.ffn.experts.0.w1.suh": torch.ones(32, dtype=torch.half),
                       "layers.0.ffn.experts.0.w1.svh": torch.ones(32, dtype=torch.half),
                       "layers.0.ffn.experts.0.w1.mcg": torch.tensor(-878961683, dtype=torch.int32)}
            rec = {"gate": {"ok": True}, "started": bp.utcnow()}
            out = bp._finish_unit("00003", "model-00003-of-00048.safetensors", src, dict(tensors), {**copied, **enc},
                                  list(copied), [{"relerr": 0.1}], dict(rec), d, partial=False, expert_keys=list(enc))
            self.assertTrue(out["written"])
            self.assertTrue((d / "shards" / "model-00003-of-00048.safetensors").is_file())
            self.assertEqual(bp.reuse_local(d, "00003", "model-00003-of-00048.safetensors")["sha256"], out["sha256"])
            wrong = dict(tensors, a=torch.zeros(4, dtype=torch.bfloat16))
            with self.assertRaises(RuntimeError):
                bp._finish_unit("00004", "model-00004-of-00048.safetensors", src, wrong, {**copied, **enc},
                                list(copied), [], dict(rec), d, partial=False, expert_keys=list(enc))
            self.assertTrue((d / "units" / "00004.failed.json").is_file())
            self.assertFalse((d / "shards" / "model-00004-of-00048.safetensors").exists())
            with self.assertRaises(RuntimeError):
                bp._finish_unit("00005", "model-00005-of-00048.safetensors", src, dict(tensors), {**copied, **enc},
                                list(copied), [], {"gate": {"ok": False}, "started": bp.utcnow()}, d, partial=False,
                                expert_keys=list(enc))

    def test_lmhead_unit_end_to_end(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "src"
            src.mkdir()
            head = torch.randn(64, 128, dtype=torch.bfloat16)
            save_file({"head.weight": head, "norm.weight": torch.ones(128, dtype=torch.bfloat16)},
                      str(src / "model-00043-of-00048.safetensors"))
            info = {"file": "model-00043-of-00048.safetensors", "kind": "lmhead", "layer": None}
            rec = bp.encode_lmhead_unit("00043", info, [src], d / "out", log=lambda m: None)
            self.assertTrue(rec["written"], rec)
            self.assertLess(rec["gate"]["relerr"], 0.05)
            hdr = bp.read_header(d / "out" / "shards" / info["file"])[1]
            self.assertEqual(hdr["lm_head.weight"]["dtype"], "F8_E4M3")
            self.assertEqual(hdr["lm_head.weight_scale"]["shape"], [64, 4])
            self.assertNotIn("head.weight", hdr)


if __name__ == "__main__":
    unittest.main()
