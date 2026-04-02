# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for PVC-inspired Phase 6 function-stream utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from pvx.cli.pvxenvelope import main as envelope_main  # noqa: E402
from pvx.cli.pvx import main as pvx_main  # noqa: E402
from pvx.cli.pvxreshape import main as reshape_main  # noqa: E402
from pvx.core.pvc_functions import (  # noqa: E402
    dump_control_points_csv,
    generate_envelope_points,
    parse_control_points_payload,
    reshape_control_points,
)
from pvx.core.pvc_ops import evaluate_scalar_control  # noqa: E402


class TestPVCPhase6Utilities(unittest.TestCase):
    def test_generate_adsr_points_have_expected_bounds(self) -> None:
        times, values = generate_envelope_points(
            duration_sec=2.0,
            rate_hz=10.0,
            mode="adsr",
            start=0.0,
            peak=1.0,
            sustain=0.6,
            end=0.0,
            attack_sec=0.2,
            decay_sec=0.4,
            release_sec=0.4,
        )
        self.assertGreaterEqual(times.size, 2)
        self.assertAlmostEqual(float(times[0]), 0.0, places=8)
        self.assertAlmostEqual(float(times[-1]), 2.0, places=8)
        self.assertAlmostEqual(float(values[0]), 0.0, places=6)
        self.assertAlmostEqual(float(values[-1]), 0.0, places=5)
        self.assertGreater(float(np.max(values)), 0.95)

    def test_reshape_resample_then_scale(self) -> None:
        t = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        v = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
        rt, rv = reshape_control_points(
            t,
            v,
            operation="resample",
            resample_rate_hz=20.0,
            interp="linear",
        )
        self.assertGreater(rt.size, t.size)
        st, sv = reshape_control_points(rt, rv, operation="scale", factor=0.5)
        self.assertEqual(st.size, rt.size)
        self.assertAlmostEqual(float(np.max(sv)), 1.0, places=5)

    def test_csv_roundtrip_parser(self) -> None:
        t = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        v = np.asarray([1.2, 0.8, 1.0], dtype=np.float64)
        payload = dump_control_points_csv(t, v, key="stretch")
        out_t, out_v = parse_control_points_payload(
            payload,
            key="stretch",
            source_label="test",
            fmt="csv",
        )
        self.assertEqual(out_t.size, 3)
        self.assertTrue(np.allclose(out_v, v, atol=1e-9))

    def test_generate_triangle_lfo_bounds(self) -> None:
        times, values = generate_envelope_points(
            duration_sec=2.0,
            rate_hz=20.0,
            mode="triangle",
            start=1.0,
            peak=0.25,
            sine_cycles=2.0,
        )
        self.assertGreaterEqual(times.size, 2)
        self.assertGreaterEqual(float(np.min(values)), 0.75 - 1e-6)
        self.assertLessEqual(float(np.max(values)), 1.25 + 1e-6)

    def test_new_interpolation_modes_are_supported(self) -> None:
        frame_t = np.linspace(0.0, 1.0, num=21, dtype=np.float64)
        ctrl_t = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        ctrl_v = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
        for mode in ("exponential", "s_curve", "smootherstep"):
            out = evaluate_scalar_control(frame_t, ctrl_t, ctrl_v, mode=mode, order=3)
            self.assertEqual(out.shape, frame_t.shape)
            self.assertTrue(np.isfinite(out).all())
            self.assertAlmostEqual(float(out[0]), 1.0, places=6)
            self.assertAlmostEqual(float(out[-1]), 1.0, places=6)


class TestPVCPhase6Cli(unittest.TestCase):
    def test_envelope_cli_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-p6-env-") as tmp:
            out = Path(tmp) / "env.csv"
            code = envelope_main(
                [
                    "--mode",
                    "ramp",
                    "--duration",
                    "1.0",
                    "--rate",
                    "10",
                    "--start",
                    "1.0",
                    "--end",
                    "2.0",
                    "--key",
                    "stretch",
                    "--output",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("time_sec,stretch", text)

    def test_reshape_cli_scales_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-p6-shape-") as tmp:
            src = Path(tmp) / "src.csv"
            dst = Path(tmp) / "dst.csv"
            src.write_text("time_sec,stretch\n0.0,1.0\n1.0,2.0\n", encoding="utf-8")
            code = reshape_main(
                [
                    str(src),
                    "--key",
                    "stretch",
                    "--operation",
                    "scale",
                    "--factor",
                    "0.5",
                    "--output",
                    str(dst),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            text = dst.read_text(encoding="utf-8")
            self.assertIn("0.500000000", text)
            self.assertIn("1.000000000", text)

    def test_envelope_cli_lfo_alias_args_write_csv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-p6-lfo-") as tmp:
            out = Path(tmp) / "lfo.csv"
            code = envelope_main(
                [
                    "--wave",
                    "triangle",
                    "--duration",
                    "2.0",
                    "--rate",
                    "10",
                    "--frequency-hz",
                    "0.5",
                    "--center",
                    "1.0",
                    "--amplitude",
                    "0.2",
                    "--key",
                    "stretch",
                    "--output",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("time_sec,stretch", text)

    def test_envelope_cli_rejects_cycles_and_frequency_together(self) -> None:
        with self.assertRaises(SystemExit):
            envelope_main(
                [
                    "--wave",
                    "sine",
                    "--duration",
                    "1.0",
                    "--cycles",
                    "2",
                    "--frequency-hz",
                    "2.0",
                    "--quiet",
                ]
            )

    def test_pvx_lfo_alias_dispatches_envelope_tool(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-p6-lfo-dispatch-") as tmp:
            out = Path(tmp) / "dispatch.csv"
            code = pvx_main(
                [
                    "lfo",
                    "--wave",
                    "sine",
                    "--duration",
                    "1.0",
                    "--cycles",
                    "2",
                    "--center",
                    "1.0",
                    "--amplitude",
                    "0.1",
                    "--key",
                    "stretch",
                    "--output",
                    str(out),
                    "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
