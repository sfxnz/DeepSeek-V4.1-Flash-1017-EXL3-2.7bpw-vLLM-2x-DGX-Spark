#!/usr/bin/env python3
"""Run tests/test_api_gateway.py (host tests) against harness/review/patched/dsv41_gateway.py."""
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
WT = Path("/home/sfxnz/projects/ai-lab/worktrees/dsv41-1017-api-gateway")
sys.path.insert(0, str(WT / "tests"))
import test_api_gateway as T  # noqa: E402

T.GATEWAY = Path(__file__).resolve().parent / "patched/dsv41_gateway.py"
suite = unittest.defaultTestLoader.loadTestsFromModule(T)
sys.exit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
