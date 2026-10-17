"""python3 -m unittest test_calib_data   (from tools/calib/data; needs tokenizers + Pillow and the pinned
reference files in DSV41_1017_REF). The reference-parity test also needs torch + transformers (run it in
the dsv41-quant151 image); it is skipped otherwise."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import encode
import load

REF = encode.REF_DIR
DATA = load.DEFAULT_ROOT


def _have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


class EncodingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = encode.Encoder()

    def test_golden_vectors(self):
        enc = self.E.enc
        n = 0
        for inp in sorted((REF / "encoding" / "tests").glob("test_input_*.json")):
            case = enc.load_cases(str(inp))[0]
            want = (REF / "encoding" / "tests" / inp.name.replace("input", "output").replace(".json", ".txt")).read_text()
            prompt, _ = enc.encode_case(case, thinking_mode="chat")
            self.assertEqual(prompt, want, inp.name)
            # span rendering reproduces the reference prompt (asserted inside render) and is token-aligned
            p, _, spans = self.E.render(case["messages"], case.get("thinking_mode") or "chat", case.get("reasoning_effort"))
            self.assertEqual(p, want)
            self.E.tokenize(p, spans)
            n += 1
        self.assertEqual(n, 5)

    def test_loss_mask_is_exactly_the_assistant_text(self):
        msgs = [{"role": "system", "content": "Be brief."},
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4.", "reasoning_content": "old reasoning, dropped"},
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "6.", "reasoning_content": "Add three and three."}]
        r = self.E.encode(msgs, "thinking", 100)
        tok = self.E.tok
        cur, runs = [], []
        for t, m in zip(r["tokens"] + [None], r["loss_mask"] + [0]):
            if m:
                cur.append(t)
            elif cur:
                runs.append(tok.decode(cur, skip_special_tokens=False))
                cur = []
        self.assertEqual(runs, ["4.<｜end▁of▁sentence｜>", "Add three and three.</think>6.<｜end▁of▁sentence｜>"])
        self.assertTrue(r["prompt"].startswith("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 "))
        self.assertIn("<｜User｜>And 3+3?<｜Assistant｜><think>Add three", r["prompt"])
        self.assertNotIn("old reasoning", r["prompt"])

    def test_tool_call_spans(self):
        tools = [{"type": "function", "function": {"name": "f", "description": "d",
                                                   "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}]
        msgs = [{"role": "system", "content": "s", "tools": tools}, {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "reasoning_content": "call f",
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 3}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "{\"y\": 9}"},
                {"role": "assistant", "content": "y is 9", "reasoning_content": "got it"}]
        r = self.E.encode(msgs, "thinking", 75)
        p = r["prompt"]
        self.assertIn('<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="x" string="false">3</｜DSML｜ parameter>', p)
        self.assertIn("<｜User｜><tool_result>{\"y\": 9}</tool_result><｜Assistant｜><think>got it</think>", p)
        dec = self.E.tok.decode([t for t, m in zip(r["tokens"], r["loss_mask"]) if m], skip_special_tokens=False)
        self.assertTrue(dec.startswith("call f</think>\n\n<｜DSML｜ calls>"))
        self.assertTrue(dec.endswith("got it</think>y is 9<｜end▁of▁sentence｜>"))

    def test_vision_example_layout(self):
        case = self.E.enc.load_cases(str(REF / "inference" / "examples" / "example_harmony.json"))[0]
        ex = REF / "inference"
        imgs = {}
        for m in case["messages"]:
            for b in m["content"]:
                if b.get("type") == "image_url":
                    imgs[b["image_url"]["url"]] = (ex / b["image_url"]["url"]).read_bytes()
        r = self.E.encode(case["messages"], "chat", None, imgs)
        self.assertEqual(len(r["images"]), 2)
        types = encode.token_types(len(r["tokens"]), r["images"])
        for im in r["images"]:
            s, k = im["start"], im["n_tokens"]
            self.assertTrue(all(t == encode.IMAGE_TOKEN_ID for t in r["tokens"][s:s + k]))
            self.assertEqual(types[s], encode.IMAGE_START)
            self.assertEqual(types[s + k - 1], encode.IMAGE_END)
            self.assertEqual(types[s + 1:s + k - 1].count(encode.IMAGE_NEW_LINE), im["n_llm_h"])
            self.assertLessEqual(k, 1024)
        self.assertEqual(sum(t == encode.IMAGE_TOKEN_ID for t in r["tokens"]), sum(i["n_tokens"] for i in r["images"]))
        rec = dict(id="t", n_tokens=len(r["tokens"]), tokens=r["tokens"], loss_spans=[],
                   images=[dict(path=i["url"], **i) for i in r["images"]])
        self.assertEqual(load.token_types(rec), types)
        load.validate(rec)

    @unittest.skipUnless(_have("torch") and _have("transformers"), "needs torch + transformers (dsv41-quant151 image)")
    def test_reference_parity(self):
        """Our ids/types/grid == reference AutoTokenizer.encode + image_processor.prepare_vl_inputs."""
        import types as _t
        from transformers import AutoTokenizer
        ip = self.E.ip
        tk = AutoTokenizer.from_pretrained(str(REF))
        cfg = json.loads((REF / "inference" / "config.json").read_text())
        args = _t.SimpleNamespace(**cfg, vision_enabled=cfg["vision_n_layers"] > 0)
        cases = self.E.enc.load_cases(str(REF / "inference" / "examples" / "example_harmony.json"))
        old = os.getcwd()
        os.chdir(REF / "inference")  # example image paths are relative to inference/
        try:
            for case in cases:
                prompt, recs = self.E.enc.encode_case(case, "chat")
                tokens, ttypes, image_inputs = ip.prepare_vl_inputs(prompt, recs, tk, args)
                imgs = {rec["url"]: Path(rec["url"]).read_bytes() for rec in recs}
                r = self.E.encode(case["messages"], "chat", None, imgs)
                self.assertEqual(r["tokens"], tokens)
                self.assertEqual(encode.token_types(len(r["tokens"]), r["images"]), ttypes)
                for mine, ref in zip(r["images"], image_inputs or []):
                    self.assertEqual((mine["start"], mine["n_vit_h"], mine["n_vit_w"]), (ref.start, ref.n_vit_h, ref.n_vit_w))
                    self.assertEqual(mine["n_tokens"], ref.types.numel())
        finally:
            os.chdir(old)


class LoaderTest(unittest.TestCase):
    def test_roundtrip(self):
        E = encode.Encoder()
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        r = E.encode(msgs, "chat")
        spans = [[i, i + 1] for i, m in enumerate(r["loss_mask"]) if m]
        merged = []
        for a, b in spans:
            if merged and merged[-1][1] == a:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        rec = dict(id="calib-00000", set="calib", domain="chat", n_tokens=len(r["tokens"]), tokens=r["tokens"],
                   loss_spans=merged, images=[])
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "calib.jsonl").write_text(json.dumps(rec) + "\n")
            got = load.load_set("calib", d)
        self.assertEqual(len(got), 1)
        load.validate(got[0])
        self.assertEqual(load.loss_mask(got[0]), r["loss_mask"])
        self.assertEqual(load.token_types(got[0]), [-1] * len(r["tokens"]))


@unittest.skipUnless((DATA / "calib.jsonl").exists() and (DATA / "panel.jsonl").exists(), "sets not built")
class OutputSetsTest(unittest.TestCase):
    def test_disjoint_and_valid(self):
        seen = {}
        for name in ("calib", "panel"):
            keys, shas, imgs = set(), set(), set()
            for rec in load.iter_set(name):
                load.validate(rec)
                keys.update(rec["key"].split("+"))
                shas.add(rec["tokens_sha256"])
                imgs.update(i["sha256"] for i in rec["images"])
            seen[name] = (keys, shas, imgs)
        for a, b in zip(seen["calib"], seen["panel"]):
            self.assertFalse(a & b)


if __name__ == "__main__":
    unittest.main()
