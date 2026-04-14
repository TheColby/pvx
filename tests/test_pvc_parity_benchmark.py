# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Tests for PVC parity benchmark runner and gate logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_pvc_parity import _gate_failures, main as run_pvc_parity_main  # noqa: E402


class TestPVCParityBenchmark(unittest.TestCase):
    def test_quick_run_and_gate_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-pvc-bench-") as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            baseline = root / "baseline.json"
            code = run_pvc_parity_main(
                [
                    "--quick",
                    "--out-dir",
                    str(out_dir),
                    "--refresh-baseline",
                    str(baseline),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "pvc_parity_report.json").exists())
            self.assertTrue((out_dir / "pvc_parity_report.md").exists())
            self.assertTrue(baseline.exists())

            code_gate = run_pvc_parity_main(
                [
                    "--quick",
                    "--out-dir",
                    str(out_dir),
                    "--baseline",
                    str(baseline),
                    "--gate",
                    "--gate-tolerance",
                    "0.5",
                ]
            )
            self.assertEqual(code_gate, 0)

    def test_gate_detects_drift(self) -> None:
        baseline = {
            "rows": [
                {
                    "name": "case_a",
                    "snr_db": 40.0,
                    "log_spectral_distance": 0.1,
                    "modulation_spectrum_distance": 0.1,
                    "envelope_correlation": 0.99,
                    "peak_abs": 0.6,
                    "rms_out": 0.2,
                }
            ]
        }
        current = {
            "rows": [
                {
                    "name": "case_a",
                    "snr_db": 37.0,
                    "log_spectral_distance": 0.6,
                    "modulation_spectrum_distance": 0.1,
                    "envelope_correlation": 0.5,
                    "peak_abs": 0.9,
                    "rms_out": 0.2,
                }
            ]
        }
        failures = _gate_failures(current, baseline, tolerance=0.05)
        self.assertGreaterEqual(len(failures), 2)
        self.assertTrue(any("log_spectral_distance" in msg for msg in failures))

    def test_report_json_shape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-pvc-bench-shape-") as tmp:
            out_dir = Path(tmp) / "out"
            code = run_pvc_parity_main(["--quick", "--out-dir", str(out_dir)])
            self.assertEqual(code, 0)
            payload = json.loads((out_dir / "pvc_parity_report.json").read_text(encoding="utf-8"))
            self.assertIn("rows", payload)
            self.assertIn("aggregate", payload)
            self.assertTrue(payload["rows"])

    def test_full_run_includes_analysis_response_function_case(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-pvc-bench-full-") as tmp:
            out_dir = Path(tmp) / "out"
            code = run_pvc_parity_main(["--out-dir", str(out_dir)])
            self.assertEqual(code, 0)
            payload = json.loads((out_dir / "pvc_parity_report.json").read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
            case_names = {str(row.get("name", "")) for row in rows if isinstance(row, dict)}
            self.assertIn("analysis_response_function_chain", case_names)


if __name__ == "__main__":
    unittest.main()
