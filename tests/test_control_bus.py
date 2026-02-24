# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
# ruff: noqa: E402

"""Unit tests for control-bus routing helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.core.control_bus import (
    apply_control_routes_csv,
    parse_control_route,
    parse_control_routes,
)


class TestControlBus(unittest.TestCase):
    def test_parse_control_route_alias_and_const(self) -> None:
        route = parse_control_route("time_stretch=const(1.5)")
        self.assertEqual(route.target, "stretch")
        self.assertEqual(route.op, "const")
        self.assertAlmostEqual(float(route.value or 0.0), 1.5, places=9)

    def test_parse_control_route_pow(self) -> None:
        route = parse_control_route("pitch_ratio=pow(f0_hz,0.5)")
        self.assertEqual(route.target, "pitch_ratio")
        self.assertEqual(route.op, "pow")
        self.assertEqual(route.source, "f0_hz")
        self.assertAlmostEqual(float(route.exponent or 0.0), 0.5, places=9)

    def test_parse_control_route_affine_and_clip(self) -> None:
        route = parse_control_route("pitch_ratio=affine(mfcc_01,0.25,1.0)")
        self.assertEqual(route.op, "affine")
        self.assertEqual(route.source, "mfcc_01")
        self.assertEqual(tuple(route.params or ()), (0.25, 1.0))
        route2 = parse_control_route("stretch=clip(spectral_flux,0.8,1.2)")
        self.assertEqual(route2.op, "clip")
        self.assertEqual(route2.source, "spectral_flux")
        self.assertEqual(tuple(route2.params or ()), (0.8, 1.2))

    def test_apply_control_route_missing_source_column_raises(self) -> None:
        payload = (
            "start_sec,end_sec,stretch,pitch_ratio,confidence\n0.0,0.1,1.0,1.0,1.0\n"
        )
        with self.assertRaises(ValueError):
            apply_control_routes_csv(
                payload,
                routes=parse_control_routes(["stretch=unknown_signal"]),
                source_label="unit",
            )

    def test_apply_control_routes_csv_maps_pitch_ratio_to_stretch(self) -> None:
        payload = (
            "start_sec,end_sec,stretch,pitch_ratio,confidence,f0_hz\n"
            "0.0,0.1,1.0,1.25,0.9,220.0\n"
            "0.1,0.2,1.0,0.75,0.8,180.0\n"
        )
        routed = apply_control_routes_csv(
            payload,
            routes=parse_control_routes(
                ["stretch=pitch_ratio", "pitch_ratio=const(1.0)"]
            ),
            source_label="unit",
        )
        lines = [line.strip() for line in routed.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertIn("stretch", lines[0])
        self.assertIn("pitch_ratio", lines[0])
        self.assertIn(",1.250000000,1.000000000,", lines[1])
        self.assertIn(",0.750000000,1.000000000,", lines[2])

    def test_apply_control_routes_csv_pitch_ratio_from_cents(self) -> None:
        payload = (
            "start_sec,end_sec,stretch,pitch_cents,confidence\n0.0,0.1,1.0,1200,1.0\n"
        )
        routed = apply_control_routes_csv(
            payload,
            routes=parse_control_routes(["stretch=pitch_ratio"]),
            source_label="unit",
        )
        lines = [line.strip() for line in routed.splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn(",2.000000000,", lines[1])

    def test_apply_control_routes_csv_affine_feature_source(self) -> None:
        payload = (
            "start_sec,end_sec,stretch,pitch_ratio,confidence,mfcc_01\n"
            "0.0,0.1,1.0,1.0,1.0,0.2\n"
            "0.1,0.2,1.0,1.0,1.0,0.6\n"
        )
        routed = apply_control_routes_csv(
            payload,
            routes=parse_control_routes(["pitch_ratio=affine(mfcc_01,0.5,1.0)"]),
            source_label="unit",
        )
        lines = [line.strip() for line in routed.splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn(",1.100000000,", lines[1])
        self.assertIn(",1.300000000,", lines[2])

    def test_apply_control_routes_csv_inv_zero_raises(self) -> None:
        payload = (
            "start_sec,end_sec,stretch,pitch_ratio,confidence,f0_hz\n"
            "0.0,0.1,1.0,1.0,0.0,0.0\n"
        )
        with self.assertRaises(ValueError):
            apply_control_routes_csv(
                payload,
                routes=parse_control_routes(["stretch=inv(f0_hz)"]),
                source_label="unit",
            )


if __name__ == "__main__":
    unittest.main()
