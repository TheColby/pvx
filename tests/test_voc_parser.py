"""Regression tests for the extracted pvx voc parser contract."""

from __future__ import annotations

import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path

from pvx.core import voc, voc_parser


class TestVocParser(unittest.TestCase):
    def test_core_wrapper_build_parser_matches_extracted_module(self) -> None:
        self.assertIsInstance(voc.build_parser(), ArgumentParser)
        self.assertEqual(voc.build_parser().prog, voc_parser.build_parser().prog)

    def test_validate_args_accepts_dynamic_time_stretch_control_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            control_path = Path(tmpdir) / "stretch.csv"
            control_path.write_text("time_sec,value\n0.0,1.0\n1.0,1.2\n", encoding="utf-8")

            parser = voc_parser.build_parser()
            args = parser.parse_args(
                ["input.wav", "--time-stretch", str(control_path), "--interp", "cubic"]
            )

            voc_parser.validate_args(args, parser)

            dynamic_refs = dict(getattr(args, "_dynamic_control_refs", {}) or {})
            self.assertIn("time_stretch", dynamic_refs)
            self.assertEqual(dynamic_refs["time_stretch"].path, control_path)
            self.assertEqual(dynamic_refs["time_stretch"].interpolation, "cubic")

    def test_validate_args_requires_checkpoint_id_for_segmented_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = voc_parser.build_parser()
            args = parser.parse_args(
                [
                    "-",
                    "--checkpoint-dir",
                    str(Path(tmpdir) / "checkpoints"),
                    "--auto-segment-seconds",
                    "1.0",
                ]
            )

            with self.assertRaises(SystemExit):
                voc_parser.validate_args(args, parser)

    def test_validate_args_rejects_gpu_and_cpu_together(self) -> None:
        parser = voc_parser.build_parser()
        args = parser.parse_args(["input.wav", "--gpu", "--cpu"])
        with self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

    def test_validate_args_rejects_dynamic_control_bad_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = voc_parser.build_parser()
            bad_path = Path(tmpdir) / "stretch.txt"
            bad_path.write_text("1.0", encoding="utf-8")
            args = parser.parse_args(["input.wav", "--time-stretch", str(bad_path)])
            with self.assertRaises(SystemExit):
                voc_parser.validate_args(args, parser)

    def test_validate_args_accepts_dynamic_pitch_ratio_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            control_path = Path(tmpdir) / "ratio.json"
            control_path.write_text(
                '[{"time_sec": 0.0, "value": 1.0}, {"time_sec": 1.0, "value": 1.5}]',
                encoding="utf-8",
            )
            parser = voc_parser.build_parser()
            args = parser.parse_args(["input.wav", "--ratio", str(control_path)])
            voc_parser.validate_args(args, parser)
            dynamic_refs = dict(getattr(args, "_dynamic_control_refs", {}) or {})
            self.assertIn("pitch_ratio", dynamic_refs)
            self.assertEqual(dynamic_refs["pitch_ratio"].value_kind, "pitch_ratio")

    def test_validate_args_rejects_route_without_control_map(self) -> None:
        parser = voc_parser.build_parser()
        args = parser.parse_args(["input.wav", "--route", "stretch=const(1.2)"])
        with self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

    def test_validate_args_rejects_target_duration_with_dynamic_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            control_path = Path(tmpdir) / "stretch.csv"
            control_path.write_text("time_sec,value\n0.0,1.0\n", encoding="utf-8")
            parser = voc_parser.build_parser()
            args = parser.parse_args(
                [
                    "input.wav",
                    "--time-stretch",
                    str(control_path),
                    "--target-duration",
                    "2.0",
                ]
            )
            with self.assertRaises(SystemExit):
                voc_parser.validate_args(args, parser)

    def test_validate_args_rejects_multires_weights_without_fusion(self) -> None:
        parser = voc_parser.build_parser()
        args = parser.parse_args(["input.wav", "--multires-weights", "1,1,1"])
        with self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

    def test_validate_args_rejects_multires_weight_count_mismatch(self) -> None:
        parser = voc_parser.build_parser()
        args = parser.parse_args(
            ["input.wav", "--multires-fusion", "--multires-ffts", "1024,2048", "--multires-weights", "1"]
        )
        with self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)


if __name__ == "__main__":
    unittest.main()
