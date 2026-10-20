"""tools/resident_soak.py: /metrics parsing and the pass/fail verdict (host only, no serve)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resident_soak", ROOT / "tools" / "resident_soak.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

PAGE = """# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="m"} 4.0
vllm:num_requests_waiting{engine="0",model_name="m"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.93
vllm:num_preemptions_total{engine="0",model_name="m"} 2.0
vllm:num_preemptions_created{engine="0",model_name="m"} 1.7e9
"""


class ResidentSoakTests(unittest.TestCase):
    def test_metric_values_sum_label_sets_and_skip_lookalikes(self) -> None:
        got = rs.metric_values(PAGE + 'vllm:num_requests_running{engine="1",model_name="m"} 1.0\n')
        self.assertEqual(got, {"running": 5.0, "waiting": 0.0, "kv_usage": 0.93, "preemptions": 2.0})
        self.assertEqual(rs.metric_values("")["running"], None)

    def test_verdict(self) -> None:
        ok = [{"i": i, "code": "A", "found": True} for i in range(4)]
        samples = [{"running": 0, "preemptions": 3.0}, {"running": 4, "preemptions": 3.0}]
        self.assertEqual(rs.verdict(ok, samples, 4), [])
        self.assertIn("preemptions +1", rs.verdict(ok, samples + [{"running": 4, "preemptions": 4.0}], 4))
        self.assertIn("max running 3 < 4", rs.verdict(ok, [{"running": 3, "preemptions": 0.0}], 4))
        bad = ok[:3] + [{"i": 3, "code": "A", "found": False, "answer": "B"}]
        self.assertEqual(len(rs.verdict(bad, samples, 4)), 1)
        err = ok[:3] + [{"i": 3, "code": "A", "error": "HTTP 400"}]
        self.assertIn("session 3: HTTP 400", rs.verdict(err, samples, 4))
        self.assertIn("no vllm:num_preemptions_total samples", rs.verdict(ok, [{"running": 4}], 4))


if __name__ == "__main__":
    unittest.main()
