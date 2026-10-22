"""docker/patch/image_16bit_gray.py on the pinned e16-rows sources (CPU only).

tests/fixtures/multimodal_image_e16.pin.py and dsv41_mm_preprocess_e16.pin.py are the image's
vllm/multimodal/image.py and models/deepseek_v4_1/common/mm_preprocess.py. The rewritten convert_image_mode
runs here from the pinned text with Pillow and numpy (skipped without them). load_image needs torch and vLLM,
so the serve decode paths, the processor included, are checked in the image: ImageTests (DSV41_IMAGE_TESTS=1)
runs tests/image_16bit_gray_image_check.py with the lever on and off.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docker/patch"
CONVERT_PIN = ROOT / "tests/fixtures/multimodal_image_e16.pin.py"
LOAD_PIN = ROOT / "tests/fixtures/dsv41_mm_preprocess_e16.pin.py"
IMAGE = os.environ.get("DSV41_IMAGE", "dsv41-flash-exl3-sm121:e16-rows")
sys.path.insert(0, str(PATCH))

import image_16bit_gray as g16  # noqa: E402

HAVE_PIL = all(importlib.util.find_spec(m) for m in ("numpy", "PIL"))
BG = (0, 0, 255)


def _module(src: str, name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


class RewriteTests(unittest.TestCase):
    def test_rewrites_apply_once_compile_and_round_trip(self) -> None:
        for pin, (rel, _, patch, unpatch), old, new in (
            (CONVERT_PIN, g16.REWRITES[0], g16.CONVERT_OLD, g16.CONVERT_NEW),
            (LOAD_PIN, g16.REWRITES[1], g16.LOAD_OLD, g16.LOAD_NEW),
        ):
            src = pin.read_text()
            out = patch(src)
            self.assertEqual(src.count(old), 1, rel)
            self.assertEqual(out.count(new), 1, rel)
            self.assertNotIn(old, out, rel)
            self.assertIn(g16.MARK, out, rel)
            compile(out, rel, "exec")
            self.assertEqual(patch(out), out, rel)
            self.assertEqual(unpatch(out), src, rel)
            self.assertEqual(unpatch(src), src, rel)

    def test_a_changed_anchor_is_refused(self) -> None:
        bad = CONVERT_PIN.read_text().replace('    if to_mode == "RGB" and _has_transparency(image):',
                                              '    if to_mode in ("RGB",) and _has_transparency(image):')
        with self.assertRaises(ValueError):
            g16.patch_convert_source(bad)
        with self.assertRaises(ValueError):
            g16.patch_load_source(LOAD_PIN.read_text().replace('image = image.convert("RGB")', "image = image"))


@unittest.skipUnless(HAVE_PIL, "needs numpy and Pillow")
class ConvertTests(unittest.TestCase):
    """The pinned convert_image_mode, stock and rewritten, on images Pillow makes from arrays."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.stock = _module(CONVERT_PIN.read_text(), "dsv41_test_image_stock")
        cls.fixed = _module(g16.patch_convert_source(CONVERT_PIN.read_text()), "dsv41_test_image_fixed")

    def test_every_16bit_value_scales_like_the_api(self) -> None:
        import numpy as np
        from PIL import Image

        v = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        want = np.rint(v / 257.0).astype(np.uint8)  # the API: 16U * 1/257
        for arr in (v, v.astype(">u2")):  # I;16 and I;16B
            img = Image.fromarray(arr)
            self.assertTrue(img.mode.startswith("I;16"), img.mode)
            out = np.asarray(self.fixed.convert_image_mode(img, "RGB"))
            for c in range(3):
                np.testing.assert_array_equal(out[..., c], want)
            np.testing.assert_array_equal(np.asarray(self.fixed.convert_image_mode(img, "L")), want)
            self.assertFalse(np.array_equal(np.asarray(self.stock.convert_image_mode(img, "RGB"))[..., 0], want))

    def test_trns_key_still_composites_onto_the_background(self) -> None:
        import numpy as np
        from PIL import Image

        v = (np.arange(40 * 30, dtype=np.uint32).reshape(30, 40) * 55).astype(np.uint16)
        key = int(v[4, 9])
        img = Image.fromarray(v)
        img.info["transparency"] = key  # what PngImagePlugin sets for a gray tRNS chunk
        out = np.asarray(self.fixed.convert_image_mode(img, "RGB", BG))
        want = np.repeat(np.rint(v / 257.0).astype(np.uint8)[..., None], 3, axis=2)
        want[v == key] = BG
        np.testing.assert_array_equal(out, want)

    def test_wide_targets_and_other_modes_are_unchanged(self) -> None:
        import numpy as np
        from PIL import Image

        v = (np.arange(24 * 16, dtype=np.uint32).reshape(16, 24) * 170).astype(np.uint16)
        for to_mode in ("I", "F", "I;16"):
            img = Image.fromarray(v)
            np.testing.assert_array_equal(np.asarray(self.fixed.convert_image_mode(img, to_mode)),
                                          np.asarray(self.stock.convert_image_mode(img, to_mode)), to_mode)
        g = (np.arange(24 * 16).reshape(16, 24) % 256).astype(np.uint8)
        a = g[::-1].copy()
        pal = Image.fromarray(np.dstack([g, a, g])).quantize(colors=8)
        pal.info["transparency"] = 3
        l_trns = Image.fromarray(g)
        l_trns.info["transparency"] = 40
        for img in (Image.fromarray(g), Image.fromarray(np.dstack([g, a])), Image.fromarray(np.dstack([g, a, g])),
                    Image.fromarray(np.dstack([g, a, g, a])), pal, l_trns):
            np.testing.assert_array_equal(np.asarray(self.fixed.convert_image_mode(img, "RGB", BG)),
                                          np.asarray(self.stock.convert_image_mode(img, "RGB", BG)), img.mode)


class InstallTests(unittest.TestCase):
    def _root(self, d: str) -> Path:
        for pin, rel in ((CONVERT_PIN, g16.CONVERT_REL), (LOAD_PIN, g16.LOAD_REL)):
            path = Path(d) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(pin.read_text())
        return Path(d)

    @staticmethod
    def _rewrite(path, fn):
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
                self.assertEqual(g16.install(root, self._rewrite, modules={}),
                                 "image.py rewritten, mm_preprocess.py rewritten")
                self.assertEqual(g16.install(root, self._rewrite, modules={g16.CONVERT_MODULE: object()}),
                                 "image.py already applied, mm_preprocess.py already applied")
            self.assertIn(g16.LOG_ENGAGED, err.getvalue())
            self.assertEqual(g16.restore(root, self._rewrite), ["rewritten", "rewritten"])
            self.assertEqual((root / g16.CONVERT_REL).read_text(), CONVERT_PIN.read_text())
            self.assertEqual((root / g16.LOAD_REL).read_text(), LOAD_PIN.read_text())
            self.assertEqual(g16.restore(root, self._rewrite), ["already applied", "already applied"])
            for module in (g16.CONVERT_MODULE, g16.LOAD_MODULE):
                with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(RuntimeError):
                    g16.install(root, self._rewrite, modules={module: object()})
                self.assertIn(g16.LOG_DISARMED, err.getvalue())
                g16.restore(root, self._rewrite)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(g16.restore(Path(d), self._rewrite), [])  # files absent: nothing to put back

    def test_env_gate(self) -> None:
        self.assertTrue(g16.enabled({g16.ENV: "1"}))
        self.assertFalse(g16.enabled({g16.ENV: "0"}))
        self.assertFalse(g16.enabled({}))


class WiringTests(unittest.TestCase):
    def test_sitecustomize_rewrites_before_any_vllm_import(self) -> None:
        site = (PATCH / "sitecustomize.py").read_text()
        ours = site.index('_patch("image_16bit_gray", _p_image_16bit_gray, required=True)')
        self.assertNotIn("import vllm", site[:ours])
        self.assertNotIn("from vllm", site[:ours])

    def test_default_on_forwarded_audited_and_in_the_promoted_boot(self) -> None:
        run = (ROOT / "run.sh").read_text()
        self.assertIn('DSV41_IMAGE_16BIT_GRAY="${DSV41_IMAGE_16BIT_GRAY:-1}"', run)
        self.assertIn("  DSV41_IMAGE_16BIT_GRAY=\n", run)
        self.assertIn("    DSV41_IMAGE_16BIT_GRAY: 1\n", (ROOT / "recipe.yaml").read_text())
        self.assertIn('"image_16bit_gray.py": {"DSV41_IMAGE_16BIT_GRAY": "1"}', (ROOT / "tools/engagement_audit.py").read_text())
        self.assertIn("export DSV41_IMAGE_16BIT_GRAY=1\n", (ROOT / "results/2026-10-19-1017-c4/boot.sh").read_text())


@unittest.skipUnless(os.environ.get("DSV41_IMAGE_TESTS") == "1" and shutil.which("docker"),
                     "set DSV41_IMAGE_TESTS=1 (needs docker and the serve image)")
class ImageTests(unittest.TestCase):
    """tests/image_16bit_gray_image_check.py in the image, docker/patch mounted the way run.sh mounts it."""

    def _run(self, shell: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--memory", "4g", "--name", f"dsv41-img16-{os.getpid()}",
             "-e", "DSV41_PATCH_STRICT=1",
             "-v", f"{PATCH}:/opt/dsv41-patch:ro",
             "-v", f"{PATCH}/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro",
             "-v", f"{ROOT}/tests:/checks:ro", "--entrypoint", "sh", IMAGE, "-c", shell],
            capture_output=True, text=True, timeout=900)

    def test_serve_decode_paths_lever_on(self) -> None:
        proc = self._run("DSV41_IMAGE_16BIT_GRAY=1 python3 /checks/image_16bit_gray_image_check.py")
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:])
        self.assertIn(g16.LOG_ENGAGED, proc.stderr)
        self.assertIn("PASS 16-bit gray, all 65536 values via ImageMediaIO.load_bytes: round(v / 257)", proc.stdout)
        self.assertNotIn("FAIL", proc.stdout)

    def test_lever_off_puts_the_image_text_back(self) -> None:
        proc = self._run("DSV41_IMAGE_16BIT_GRAY=1 python3 -c pass && "
                         "DSV41_IMAGE_16BIT_GRAY=0 python3 /checks/image_16bit_gray_image_check.py")
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:])
        self.assertIn("dsv41: image 16-bit gray off: the image's conversion restored", proc.stderr)
        self.assertIn("PASS DSV41_IMAGE_16BIT_GRAY=0: both files stock", proc.stdout)
        self.assertNotIn("FAIL", proc.stdout)


if __name__ == "__main__":
    unittest.main()
