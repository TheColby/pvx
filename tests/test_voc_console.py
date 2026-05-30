"""Focused regression tests for the extracted voc console helpers."""

from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from pvx.core.voc_console import (
    ProgressBar,
    apply_named_preset,
    collect_cli_flags,
    console_level,
    is_quiet,
    is_silent,
    log_error,
    log_message,
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

    def test_print_cli_examples_all_and_unknown(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_cli_examples("all")
        output = buf.getvalue()
        self.assertIn("[basic]", output)
        with self.assertRaises(ValueError):
            print_cli_examples("mystery")

    def test_progress_bar_disabled_and_throttled_paths(self) -> None:
        bar = ProgressBar("demo", enabled=False)
        bar.set(0.5, "half")
        self.assertFalse(bar._finished)

        enabled = ProgressBar("demo", enabled=True)
        stream = io.StringIO()
        with redirect_stderr(stream), patch("pvx.core.voc_console.time.time", side_effect=[0.0, 0.01, 0.02]):
            enabled.set(0.0, "start")
            last_fraction = enabled._last_fraction
            enabled.set(0.001, "tiny")
            self.assertEqual(enabled._last_fraction, last_fraction)
            enabled.finish()
        self.assertEqual(enabled._last_fraction, 1.0)
        self.assertTrue(enabled._finished)
        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_log_helpers_respect_silence_and_stdout_routing(self) -> None:
        silent_args = argparse.Namespace(
            verbosity="normal",
            verbose=0,
            no_progress=False,
            quiet=False,
            silent=True,
            stdout=False,
        )
        loud_args = argparse.Namespace(
            verbosity="verbose",
            verbose=0,
            no_progress=False,
            quiet=False,
            silent=False,
            stdout=True,
        )
        quiet_args = argparse.Namespace(
            verbosity="quiet",
            verbose=0,
            no_progress=False,
            quiet=False,
            silent=False,
            stdout=False,
        )

        self.assertTrue(is_silent(silent_args))
        self.assertTrue(is_quiet(quiet_args))

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            log_message(loud_args, "hello", min_level="verbose")
            log_error(silent_args, "hidden")

        self.assertEqual(out.getvalue(), "")
        self.assertIn("hello", err.getvalue())

    def test_apply_named_preset_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            apply_named_preset(argparse.Namespace(), preset="wat", provided_flags=set())


if __name__ == "__main__":
    unittest.main()
