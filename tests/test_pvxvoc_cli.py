"""Regression tests for the dedicated pvxvoc CLI module."""

import argparse
import importlib
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(ROOT / "src"))

from pvx.cli import pvxvoc
from pvx.core import voc


class TestPvxvocCli(unittest.TestCase):
    def test_expand_inputs_deduplicates_glob_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root / "a.wav"
            b = root / "b.wav"
            a.write_bytes(b"wav")
            b.write_bytes(b"wav")

            resolved = pvxvoc.expand_inputs([str(a), str(root / "*.wav"), "-", "-"])

            self.assertEqual(resolved[0], a.resolve())
            self.assertEqual(resolved[1], b.resolve())
            self.assertEqual(str(resolved[2]), "-")
            self.assertEqual(len(resolved), 3)

    def test_core_voc_main_delegates_to_cli_module(self) -> None:
        with patch("pvx.voc_cli.main", return_value=17) as cli_main:
            result = voc.main(["--example", "basic"])
        cli_main.assert_called_once_with(["--example", "basic"])
        self.assertEqual(result, 17)

    def test_core_voc_expand_inputs_uses_shared_module(self) -> None:
        with patch("pvx.voc_cli.expand_inputs", return_value=[Path("/tmp/in.wav")]) as expand:
            result = voc.expand_inputs(["in.wav"])
        expand.assert_called_once_with(["in.wav"])
        self.assertEqual(result, [Path("/tmp/in.wav")])

    def test_guided_mode_available_through_wrapper(self) -> None:
        args = argparse.Namespace(device="cpu")
        with patch("pvx.voc_cli.run_guided_mode", return_value=args) as guided:
            result = voc.run_guided_mode(args)
        guided.assert_called_once_with(args)
        self.assertIs(result, args)

    def test_guided_mode_interactive_flow(self) -> None:
        args = argparse.Namespace(
            inputs=[],
            output=None,
            stdout=False,
            time_stretch=1.0,
            pitch_shift_cents=None,
            pitch_shift_ratio=None,
            target_f0=None,
            pitch_shift_semitones=None,
            preset="none",
            device="auto",
        )
        answers = iter(["voice.wav", "out.wav", "both", "1.5", "2", "default", "cpu", "yes"])
        with patch("sys.stdin.isatty", return_value=True), patch(
            "builtins.input", side_effect=lambda _prompt: next(answers)
        ):
            result = pvxvoc.run_guided_mode(args)

        self.assertEqual(result.inputs, ["voice.wav"])
        self.assertEqual(result.time_stretch, 1.5)
        self.assertEqual(result.pitch_shift_semitones, 2.0)
        self.assertEqual(result.preset, "default")
        self.assertEqual(result.device, "cpu")
        self.assertTrue(result.stdout)
        self.assertIsNone(result.output)

    def test_guided_mode_rejects_non_tty(self) -> None:
        args = argparse.Namespace(
            inputs=[],
            output=None,
            stdout=False,
            time_stretch=1.0,
            pitch_shift_cents=None,
            pitch_shift_ratio=None,
            target_f0=None,
            pitch_shift_semitones=None,
            preset="none",
            device="auto",
        )
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(ValueError):
                pvxvoc.run_guided_mode(args)

    def test_legacy_registry_alias_emits_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            module = importlib.import_module("pvxalgorithms.registry")
            importlib.reload(module)

        self.assertTrue(
            any(
                item.category is DeprecationWarning
                and "pvx.algorithms.registry" in str(item.message)
                and "v0.2.0" in str(item.message)
                for item in caught
            )
        )


if __name__ == "__main__":
    unittest.main()
