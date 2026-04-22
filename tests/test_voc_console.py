"""Focused regression tests for the extracted voc console helpers."""

from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout

from pvx.core.voc_console import (
    apply_named_preset,
    collect_cli_flags,
    console_level,
    print_cli_examples,
)


class TestVocConsole(unittest.TestCase):
    def test_collect_cli_flags_handles_equals_form(self) -> None:
        flags = collect_cli_flags(["--preset=ambient", "--quiet", "input.wav", "-v"])
        self.assertEqual(flags, {"--preset", "--quiet"})

    def test_apply_named_preset_skips_user_supplied_fields(self) -> None:
        args = argparse.Namespace(
            quality_profile="neutral",
            transient_mode="off",
            stereo_mode="independent",
            coherence_strength=0.0,
        )

        changed = apply_named_preset(
            args,
            preset="stereo_coherent",
            provided_flags={"--coherence-strength"},
        )

        self.assertIn("quality_profile", changed)
        self.assertNotIn("coherence_strength", changed)
        self.assertEqual(args.quality_profile, "music")
        self.assertEqual(args.coherence_strength, 0.0)

    def test_console_level_respects_no_progress_and_silent(self) -> None:
        quiet_args = argparse.Namespace(
            verbosity="debug",
            verbose=1,
            no_progress=True,
            quiet=False,
            silent=False,
        )
        silent_args = argparse.Namespace(
            verbosity="normal",
            verbose=0,
            no_progress=False,
            quiet=False,
            silent=True,
        )

        self.assertEqual(console_level(quiet_args), 1)
        self.assertEqual(console_level(silent_args), 0)

    def test_print_cli_examples_emits_example_body(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_cli_examples("basic")
        output = buf.getvalue()
        self.assertIn("Basic time stretch", output)
        self.assertIn("pvx voc input.wav", output)


if __name__ == "__main__":
    unittest.main()
