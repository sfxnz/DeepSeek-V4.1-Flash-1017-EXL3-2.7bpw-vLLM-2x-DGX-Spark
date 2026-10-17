#!/usr/bin/env python3
"""docker/patch/chat_encoder_1017.py: 1017 chat-encoder parity for the image's V4.1 tokenizer.

Host (CPU, stdlib): the rewrite on the e14 sources pinned in tests/fixtures, then the 1017
encoding/tests golden vectors and test_encoding.py behaviours through the patched encoder
(exact match). The vectors come from the 1017 snapshot in the HF cache (skipped when absent).

Image (opt-in, DSV41_IMAGE_TESTS=1, docker, no GPU): the same checks plus the serving path
(apply_chat_template) inside the serve image, with docker/patch mounted the way run.sh mounts
it, so sitecustomize applies the patch at interpreter start.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docker/patch"))
sys.path.insert(0, str(ROOT / "tests"))

import chat_encoder_1017 as ce  # noqa: E402
import chat_encoder_1017_image_check as chk  # noqa: E402

FIX = ROOT / "tests/fixtures"
ENC_PIN = FIX / "dsv41_tokenizer_encoding_e14.pin.py"
TOK_PIN = FIX / "dsv41_tokenizer_e14.pin.py"
VECTORS = (Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots"
           / "1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/tests")
IMAGE = os.environ.get("DSV41_IMAGE", "dsv41-flash-exl3-sm121:canonical-e14")


def _module(src: str, name: str):
    mod = types.ModuleType(name)
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


class RewriteTests(unittest.TestCase):
    def test_rewrites_apply_once_and_compile(self) -> None:
        for path, fn in ((ENC_PIN, ce.patch_encoding_source), (TOK_PIN, ce.patch_tokenizer_source)):
            src = path.read_text()
            out = fn(src)
            self.assertNotEqual(out, src, path.name)
            self.assertIn(ce.MARK, out)
            self.assertEqual(fn(out), out, f"{path.name}: not idempotent")
            compile(out, path.name, "exec")

    def test_anchor_miss_raises(self) -> None:
        with self.assertRaises(ValueError):
            ce.patch_encoding_source("def merge_tool_messages(): pass\n")
        with self.assertRaises(ValueError):
            ce.patch_tokenizer_source("x = 1\n")

    def test_tokenizer_validates_before_flattening(self) -> None:
        out = ce.patch_tokenizer_source(TOK_PIN.read_text())
        self.assertLess(out.index("_validate_no_image_sp_tokens(message)"), out.index("parts.append(IMAGE_PLACEHOLDER)"))
        self.assertIn("reasoning_effort must be low, high, max", out)

    def test_install_uses_the_given_rewrite_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tokenizers").mkdir()
            shutil.copy(ENC_PIN, root / ce.ENCODING_REL)
            shutil.copy(TOK_PIN, root / ce.TOKENIZER_REL)

            def rewrite(path, fn):
                src = Path(path).read_text()
                out = fn(src)
                Path(path).write_text(out)
                return "rewritten" if out != src else "already applied"

            import contextlib
            import io

            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertIn("rewritten", ce.install(root, rewrite))
                self.assertIn("already applied", ce.install(root, rewrite))
            self.assertIn(ce.LOG_ENGAGED, err.getvalue())
            (root / ce.TOKENIZER_REL).write_text("drifted\n")
            (root / ce.ENCODING_REL).write_text(ENC_PIN.read_text())
            with contextlib.redirect_stderr(err), self.assertRaises(ValueError):
                ce.install(root, rewrite)
            self.assertIn(ce.LOG_DISARMED, err.getvalue())

    def test_sitecustomize_wires_it_as_required(self) -> None:
        site = (ROOT / "docker/patch/sitecustomize.py").read_text()
        self.assertIn('_patch("chat_encoder_1017", lambda: __import__("chat_encoder_1017").install(_VLLM, _rewrite), '
                      "required=True)", site)


@unittest.skipUnless(VECTORS.is_dir(), f"1017 encoding/tests not in the HF cache ({VECTORS})")
class VectorTests(unittest.TestCase):
    def test_patched_encoder_matches_every_1017_vector_and_behaviour(self) -> None:
        enc = _module(ce.patch_encoding_source(ENC_PIN.read_text()), "dsv41_enc_patched")
        results = chk.check_encoder(enc, VECTORS)
        self.assertEqual(len(chk.vector_ids(VECTORS)), 5)
        failed = [(n, d) for n, ok, d in results if not ok]
        self.assertEqual(failed, [])

    def test_unpatched_encoder_shows_the_gap(self) -> None:
        enc = _module(ENC_PIN.read_text(), "dsv41_enc_pristine")
        failed = {n for n, ok, _ in chk.check_encoder(enc, VECTORS) if not ok}
        self.assertIn("encoder vector 1", failed)  # default effort "high" renders 50, the 1017 golden says 75
        self.assertIn("encoder drop_thinking keeps the current turn across tool results", failed)


@unittest.skipUnless(os.environ.get("DSV41_IMAGE_TESTS") == "1" and shutil.which("docker") and VECTORS.is_dir(),
                     "set DSV41_IMAGE_TESTS=1 (needs docker, the serve image and the 1017 vectors)")
class ImageTests(unittest.TestCase):
    def test_vectors_and_serving_path_inside_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for p in VECTORS.iterdir():
                shutil.copy(p.resolve(), Path(d) / p.name)
            os.chmod(d, 0o755)
            proc = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--memory", "4g", "--name", f"dsv41-1017-chatenc-{os.getpid()}",
                 "-v", f"{ROOT}/docker/patch:/opt/dsv41-patch:ro",
                 "-v", f"{ROOT}/docker/patch/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro",
                 "-v", f"{ROOT}/tests:/checks:ro", "-v", f"{d}:/vectors:ro",
                 "--entrypoint", "python3", IMAGE, "/checks/chat_encoder_1017_image_check.py", "/vectors"],
                capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:])
        self.assertIn(ce.LOG_ENGAGED, proc.stderr + proc.stdout)
        self.assertIn("PASS serving vector 1", proc.stdout)
        self.assertNotIn("FAIL", proc.stdout)


if __name__ == "__main__":
    unittest.main()
