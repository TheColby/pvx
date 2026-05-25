"""Direct branch coverage tests for the extracted pvx voc parser and CLI."""

from __future__ import annotations

import argparse
import io
import json
import runpy
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pvx import voc_cli
from pvx.core import voc_parser


def _job_result(path: Path, output_path: Path | None = None):
    from pvx.core.voc import JobResult

    return JobResult(
        input_path=path,
        output_path=output_path or path.with_name("out.wav"),
        in_sr=16000,
        out_sr=16000,
        in_samples=16,
        out_samples=16,
        channels=1,
        stretch=1.0,
        pitch_ratio=1.0,
    )


class TestVocParserCoverage(unittest.TestCase):
    def _parse(self, argv: list[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
        parser = voc_parser.build_parser()
        args = parser.parse_args(argv)
        return parser, args

    def _assert_validate_error(self, argv: list[str]) -> None:
        parser, args = self._parse(argv)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

    def test_validate_args_covers_scalar_alias_and_dynamic_restore(self) -> None:
        parser, args = self._parse(["input.wav", "--stretch", "2.5"])
        args._dynamic_control_raw_values = {"time_stretch": "3.5"}
        voc_parser.validate_args(args, parser)
        self.assertEqual(args.time_stretch, 2.5)
        self.assertNotIn("time_stretch", args._dynamic_control_raw_values)

    def test_validate_args_covers_dynamic_numeric_int_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp) / "nfft.csv"
            control.write_text("time_sec,value\n0.0,2048\n", encoding="utf-8")
            parser, args = self._parse(["input.wav", "--n-fft", str(control)])
            voc_parser.validate_args(args, parser)
            self.assertEqual(args.n_fft, 2048)
            refs = dict(args._dynamic_control_refs)
            self.assertEqual(refs["n_fft"].path, control)

    def test_validate_args_covers_pitch_scalar_and_dynamic_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ratio = Path(tmp) / "ratio.csv"
            semi = Path(tmp) / "semi.csv"
            cents = Path(tmp) / "cents.json"
            ratio.write_text("time_sec,value\n0.0,1.0\n", encoding="utf-8")
            semi.write_text("time_sec,value\n0.0,2.0\n", encoding="utf-8")
            cents.write_text('[{"time_sec": 0.0, "value": 5.0}]', encoding="utf-8")

            parser, args = self._parse(["input.wav", "--ratio", "3/2"])
            voc_parser.validate_args(args, parser)
            self.assertAlmostEqual(args.pitch_shift_ratio, 1.5)

            parser, args = self._parse(["input.wav", "--pitch", str(semi)])
            voc_parser.validate_args(args, parser)
            self.assertIn("pitch_ratio", dict(args._dynamic_control_refs))

            parser, args = self._parse(["input.wav", "--pitch-shift-cents", str(cents)])
            voc_parser.validate_args(args, parser)
            self.assertIn("pitch_ratio", dict(args._dynamic_control_refs))

            for argv in (
                ["input.wav", "--ratio", "bogus"],
                ["input.wav", "--pitch", "bogus"],
                ["input.wav", "--pitch-shift-cents", "bogus"],
            ):
                self._assert_validate_error(argv)

            for attr, raw in (
                ("pitch_shift_ratio", "-"),
                ("pitch_shift_ratio", str(Path(tmp) / "ratio.txt")),
                ("pitch_shift_semitones", "-"),
                ("pitch_shift_semitones", str(Path(tmp) / "semi.txt")),
                ("pitch_shift_cents", "-"),
                ("pitch_shift_cents", str(Path(tmp) / "cents.txt")),
            ):
                parser, args = self._parse(["input.wav"])
                setattr(args, attr, raw)
                with (
                    patch("pvx.core.voc._looks_like_control_signal_reference", return_value=True),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    voc_parser.validate_args(args, parser)

    def test_validate_args_covers_value_and_range_errors(self) -> None:
        cases = [
            ["input.wav", "--order", "0"],
            ["input.wav", "--route", "not-a-route"],
            ["input.wav", "--n-fft", "0"],
            ["input.wav", "--win-length", "0"],
            ["input.wav", "--win-length", "4096", "--n-fft", "2048"],
            ["input.wav", "--hop-size", "0"],
            ["input.wav", "--hop-size", "4096", "--win-length", "2048"],
            ["input.wav", "--time-stretch", "0"],
            ["input.wav", "--extreme-stretch-threshold", "1.0"],
            ["input.wav", "--max-stage-stretch", "1.0"],
            ["input.wav", "--output", "out.wav", "--output-dir", "dir"],
            ["input.wav", "--output", "out.wav", "--stdout"],
            ["input.wav", "--target-duration", "0"],
            ["input.wav", "--pitch-conf-min", "-0.1"],
            ["input.wav", "--pitch-map-smooth-ms", "-1"],
            ["input.wav", "--pitch-map-crossfade-ms", "-1"],
            ["input.wav", "--target-f0", "0"],
            ["input.wav", "--f0-min", "200", "--f0-max", "100"],
            ["input.wav", "--target-sample-rate", "0"],
            ["input.wav", "--transient-threshold", "0"],
            ["input.wav", "--transient-sensitivity", "1.5"],
            ["input.wav", "--transient-protect-ms", "0"],
            ["input.wav", "--transient-crossfade-ms", "-1"],
            ["input.wav", "--ref-channel", "-1"],
            ["input.wav", "--coherence-strength", "2"],
            ["input.wav", "--ambient-phase-mix", "2"],
            ["input.wav", "--onset-credit-pull", "2"],
            ["input.wav", "--onset-credit-max", "-1"],
            ["input.wav", "--formant-lifter", "-1"],
            ["input.wav", "--formant-strength", "2"],
            ["input.wav", "--formant-max-gain-db", "0"],
            ["input.wav", "--fourier-sync-min-fft", "8"],
            ["input.wav", "--fourier-sync-min-fft", "32", "--fourier-sync-max-fft", "16"],
            ["input.wav", "--fourier-sync-smooth", "0"],
            ["input.wav", "--kaiser-beta", "-1"],
            ["input.wav", "--cuda-device", "-1"],
            ["input.wav", "--auto-profile-lookahead-seconds", "0"],
            ["input.wav", "--auto-segment-seconds", "-1"],
            ["input.wav", "--resume"],
            ["input.wav", "--manifest-append"],
            ["input.wav", "--multires-fusion", "--multires-ffts", ""],
            ["input.wav", "--multires-fusion", "--multires-ffts", "8"],
            ["input.wav", "--multires-fusion", "--multires-weights", "wat"],
            ["input.wav", "--bit-depth", "32f", "--dither", "tpdf"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self._assert_validate_error(argv)

    def test_validate_args_covers_mutated_choice_guards_and_valid_aliases(self) -> None:
        parser, args = self._parse(["input.wav", "--gpu"])
        voc_parser.validate_args(args, parser)
        self.assertEqual(args.device, "cuda")

        parser, args = self._parse(["input.wav", "--cpu", "--pitch-follow-stdin"])
        voc_parser.validate_args(args, parser)
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.pitch_map_stdin)

        parser, args = self._parse(["input.wav", "--control-stdin"])
        voc_parser.validate_args(args, parser)
        self.assertTrue(args.pitch_map_stdin)

        parser, args = self._parse(["input.wav"])
        args.transient_mode = "mystery"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

        parser, args = self._parse(["input.wav"])
        args.stereo_mode = "mystery"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

        parser, args = self._parse(["input.wav"])
        args.phase_engine = "mystery"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

        parser, args = self._parse(["input.wav"])
        args.quality_profile = "mystery"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

        parser, args = self._parse(["input.wav"])
        args.preset = "mystery"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

    def test_validate_args_covers_file_and_conflict_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            stretch = tmpdir / "stretch.csv"
            stretch.write_text("time_sec,value\n0.0,1.1\n", encoding="utf-8")
            pitch_map = tmpdir / "map.csv"
            pitch_map.write_text("time_sec,value\n0.0,1.0\n", encoding="utf-8")

            self._assert_validate_error(
                ["input.wav", "--time-stretch", str(stretch), "--pitch-map", str(pitch_map)]
            )
            self._assert_validate_error(
                ["input.wav", "--time-stretch", str(tmpdir / "missing.csv")]
            )
            self._assert_validate_error(
                ["input.wav", "--pitch-map-stdin", "--pitch-map", str(pitch_map)]
            )
            self._assert_validate_error(
                ["input.wav", "--pitch-map", str(tmpdir / "missing-map.csv")]
            )
            self._assert_validate_error(
                [
                    "input.wav",
                    "--checkpoint-dir",
                    str(tmpdir / "ckpt"),
                    "--pitch-map",
                    "-",
                    "--auto-segment-seconds",
                    "1.0",
                ]
            )

    def test_validate_args_covers_dynamic_stdin_and_bad_extension_guards(self) -> None:
        cases = [
            ("time_stretch", "-", "-"),
            ("time_stretch", "control.txt", "control.txt"),
            ("pitch_shift_ratio", "-", "-"),
            ("pitch_shift_ratio", "ratio.txt", "ratio.txt"),
            ("pitch_shift_semitones", "-", "-"),
            ("pitch_shift_semitones", "semi.txt", "semi.txt"),
            ("pitch_shift_cents", "-", "-"),
            ("pitch_shift_cents", "cents.txt", "cents.txt"),
        ]
        for attr, raw, trigger in cases:
            parser, args = self._parse(["input.wav"])
            setattr(args, attr, raw)
            with (
                patch(
                    "pvx.core.voc._looks_like_control_signal_reference",
                    side_effect=lambda value, trigger=trigger: value == trigger,
                ),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                voc_parser.validate_args(args, parser)

    def test_validate_args_covers_multires_defaults_and_reset_transient(self) -> None:
        parser, args = self._parse(["input.wav", "--multires-fusion", "--multires-ffts", "1024,2048"])
        voc_parser.validate_args(args, parser)
        self.assertEqual(args._multires_ffts, [1024, 2048])
        self.assertEqual(args._multires_weights, [1.0, 1.0])

        parser, args = self._parse(
            [
                "input.wav",
                "--multires-fusion",
                "--multires-ffts",
                "1024,2048",
                "--multires-weights",
                "0,0",
            ]
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

        parser, args = self._parse(
            [
                "input.wav",
                "--multires-fusion",
                "--multires-ffts",
                "1024,2048",
                "--multires-weights",
                "-1,1",
            ]
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

        parser, args = self._parse(
            [
                "input.wav",
                "--multires-fusion",
                "--multires-ffts",
                "1024,2048",
                "--multires-weights",
                "1,2",
            ]
        )
        voc_parser.validate_args(args, parser)
        self.assertEqual(args._multires_weights, [1.0, 2.0])

        parser, args = self._parse(["input.wav", "--transient-mode", "reset"])
        voc_parser.validate_args(args, parser)
        self.assertTrue(args.transient_preserve)

    def test_validate_args_covers_auto_profile_preset_conflict_and_empty_multires(self) -> None:
        parser, args = self._parse(["input.wav", "--auto-profile", "--preset", "ambient"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            voc_parser.validate_args(args, parser)

        parser, args = self._parse(["input.wav", "--multires-fusion", "--multires-ffts", "1024"])
        with (
            patch("pvx.core.voc.parse_int_list", return_value=[]),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            voc_parser.validate_args(args, parser)


class TestVocCliCoverage(unittest.TestCase):
    def test_prompt_helpers_and_explain_plan(self) -> None:
        with patch("builtins.input", return_value="bad"):
            with self.assertRaises(ValueError):
                voc_cli._prompt_choice("Mode", ("yes", "no"), "yes")

        args = argparse.Namespace(
            _active_quality_profile="ambient",
            transient_mode="hybrid",
            transient_sensitivity=0.5,
            transient_protect_ms=10.0,
            transient_crossfade_ms=5.0,
            stereo_mode="independent",
            ref_channel=0,
            coherence_strength=0.0,
            multires_fusion=False,
            _multires_ffts=[2048],
            _multires_weights=[1.0],
            device="cpu",
            cuda_device=0,
            output_dir=None,
            stdout=False,
            manifest_json=None,
            checkpoint_dir=None,
            subtype=None,
            bit_depth="inherit",
            dither="none",
            dither_seed=None,
            true_peak_max_dbtp=None,
            metadata_policy="none",
            _dynamic_control_refs={},
        )
        config = SimpleNamespace(
            n_fft=2048,
            win_length=2048,
            hop_size=512,
            window="hann",
            transform="fft",
            phase_locking="identity",
            phase_engine="propagate",
        )
        with patch("pvx.voc_cli.runtime_config", return_value=SimpleNamespace(active_device="cpu")):
            plan = voc_cli._build_explain_plan(args, [Path("input.wav")], config, ["window"], None)
        self.assertEqual(plan["active_profile"], "ambient")
        self.assertEqual(plan["config"]["transform"], "fft")

    def test_manifest_entries_and_preflight_invalid_append(self) -> None:
        args = argparse.Namespace(
            _active_quality_profile="neutral",
            transient_mode="off",
            stereo_mode="independent",
            coherence_strength=0.0,
            subtype=None,
            bit_depth="inherit",
            dither="none",
            dither_seed=None,
            true_peak_max_dbtp=None,
            metadata_policy="none",
        )
        config = SimpleNamespace(transform="fft", window="hann", phase_engine="propagate")
        with patch("pvx.voc_cli.runtime_config", return_value=SimpleNamespace(active_device="cpu")):
            entries = voc_cli._manifest_entries(
                args,
                config,
                [_job_result(Path("in.wav"), Path("out.wav"))],
                [(Path("bad.wav"), "boom")],
            )
        self.assertEqual(entries[1]["status"], "error")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("{broken", encoding="utf-8")
            parser = voc_parser.build_parser()
            args = parser.parse_args(["input.wav", "--manifest-json", str(path), "--manifest-append"])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                voc_cli._preflight_manifest_target(args, parser)

    def test_main_example_guided_and_error_guards(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = voc_cli.main(["--example", "basic"])
        self.assertEqual(rc, 0)
        self.assertIn("pvx voc input.wav", out.getvalue())

        with patch("pvx.voc_cli.print_cli_examples", side_effect=ValueError("bad example")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                voc_cli.main(["--example", "basic"])

        with patch("pvx.voc_cli.run_guided_mode", side_effect=ValueError("bad")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                voc_cli.main(["input.wav", "--guided"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "input.wav"
            other = root / "other.wav"
            audio.write_bytes(b"wav")
            other.write_bytes(b"wav")
            missing = root / "missing.wav"

            cases = [
                [],
                [str(missing)],
                ["-", str(audio)],
                [str(audio), str(other), "--stdout"],
                [str(audio), "--stdout", "--output-dir", str(root)],
                [str(audio), str(other), "--output", str(root / "out.wav")],
                [str(audio), str(other), "--pitch-map-stdin"],
                ["-", "--pitch-map-stdin"],
                ["-", "--auto-profile"],
            ]
            for argv in cases:
                with self.subTest(argv=argv):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        voc_cli.main(argv)

            with patch("pvx.voc_cli.expand_inputs", return_value=[Path("-"), Path("-")]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    voc_cli.main(["input.wav"])

            with patch("pvx.voc_cli.expand_inputs", return_value=[Path("-"), audio.resolve()]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    voc_cli.main(["input.wav"])

    def test_main_covers_resolve_and_transient_alias_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"entries": []}), encoding="utf-8")
            captured: dict[str, object] = {}
            config = SimpleNamespace(transform="fft", window="hann", phase_engine="propagate")

            def fake_process(path, args, config, file_index=0, file_total=1):
                captured["output_dir"] = args.output_dir
                captured["checkpoint_dir"] = args.checkpoint_dir
                captured["manifest_json"] = args.manifest_json
                captured["transient_mode"] = args.transient_mode
                return _job_result(path, root / "out.wav")

            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.build_vocoder_config_from_args", return_value=config),
                patch("pvx.voc_cli.configure_runtime_from_args"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.process_file", side_effect=fake_process),
                patch("pvx.voc_cli.write_manifest"),
            ):
                rc = voc_cli.main(
                    [
                        str(input_path),
                        "--transient-preserve",
                        "--output-dir",
                        str(root / "outdir"),
                        "--checkpoint-dir",
                        str(root / "ckpt"),
                        "--manifest-json",
                        str(manifest),
                        "--manifest-append",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(captured["transient_mode"], "reset")
            self.assertTrue(Path(captured["output_dir"]).is_absolute())
            self.assertTrue(Path(captured["checkpoint_dir"]).is_absolute())
            self.assertTrue(Path(captured["manifest_json"]).is_absolute())

    def test_main_covers_dynamic_refs_auto_transform_ambient_verbose_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            control = root / "nfft.csv"
            control.write_text("time_sec,value\n0.0,2048\n", encoding="utf-8")

            config = SimpleNamespace(
                n_fft=2048,
                win_length=2048,
                hop_size=512,
                window="hann",
                transform="fft",
                phase_locking="identity",
                phase_engine="random",
            )
            out = io.StringIO()
            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.resolve_transform_auto", return_value="dct"),
                patch("pvx.voc_cli.build_vocoder_config_from_args", return_value=config),
                patch("pvx.voc_cli.configure_runtime_from_args"),
                patch("pvx.voc_cli.runtime_config", return_value=SimpleNamespace(active_device="cpu")),
                patch("pvx.voc_cli.log_message"),
                redirect_stdout(out),
            ):
                rc = voc_cli.main(
                    [
                        str(input_path),
                        "--n-fft",
                        str(control),
                        "--auto-transform",
                        "--ambient-preset",
                        "--verbosity",
                        "verbose",
                        "--explain-plan",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["config"]["transform"], "fft")

    def test_main_covers_output_and_pitch_map_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            pitch_map = root / "map.csv"
            pitch_map.write_text("time_sec,value\n0.0,1.0\n", encoding="utf-8")
            output = root / "out.wav"
            captured: dict[str, object] = {}
            config = SimpleNamespace(transform="fft", window="hann", phase_engine="propagate")

            def fake_process(path, args, config, file_index=0, file_total=1):
                captured["output"] = args.output
                captured["pitch_map"] = args.pitch_map
                return _job_result(path, output)

            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.build_vocoder_config_from_args", return_value=config),
                patch("pvx.voc_cli.configure_runtime_from_args"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.process_file", side_effect=fake_process),
            ):
                rc = voc_cli.main(
                    [str(input_path), "--output", str(output), "--pitch-map", str(pitch_map)]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(Path(captured["output"]).is_absolute())
            self.assertTrue(Path(captured["pitch_map"]).is_absolute())

    def test_main_covers_auto_transform_parse_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            config = SimpleNamespace(
                n_fft=2048,
                win_length=2048,
                hop_size=512,
                window="hann",
                transform="fft",
                phase_locking="identity",
                phase_engine="propagate",
            )
            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.looks_like_control_signal_reference", return_value=False),
                patch("pvx.voc_cli.parse_int_cli_value", side_effect=ValueError("bad int")),
                patch("pvx.voc_cli.resolve_transform_auto", return_value="fft"),
                patch("pvx.voc_cli.build_vocoder_config_from_args", return_value=config),
                patch("pvx.voc_cli.configure_runtime_from_args"),
                patch("pvx.voc_cli.process_file", return_value=_job_result(input_path)),
            ):
                rc = voc_cli.main([str(input_path), "--auto-transform"])
            self.assertEqual(rc, 0)

    def test_main_covers_auto_profile_empty_input_and_failure_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")

            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.read_audio_input", return_value=(np.zeros((0, 1)), 16000)),
            ):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    voc_cli.main([str(input_path), "--auto-profile"])

            err = io.StringIO()
            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.build_vocoder_config_from_args", return_value=SimpleNamespace()),
                patch("pvx.voc_cli.configure_runtime_from_args"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.process_file", side_effect=ValueError("boom")),
                redirect_stderr(err),
            ):
                rc = voc_cli.main([str(input_path), "--quiet"])
            self.assertEqual(rc, 1)
            self.assertIn("[error]", err.getvalue())

    def test_main_covers_manifest_write_error_and_module_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            manifest = root / "manifest.json"

            err = io.StringIO()
            config = SimpleNamespace(transform="fft", window="hann", phase_engine="propagate")
            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.build_vocoder_config_from_args", return_value=config),
                patch("pvx.voc_cli.configure_runtime_from_args"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.process_file", return_value=_job_result(input_path)),
                patch("pvx.voc_cli.write_manifest", side_effect=ValueError("bad manifest")),
                redirect_stderr(err),
            ):
                rc = voc_cli.main([str(input_path), "--manifest-json", str(manifest)])
            self.assertEqual(rc, 1)
            self.assertIn("bad manifest", err.getvalue())

            with (
                patch("pvx.voc_cli.main", return_value=0),
                patch("sys.argv", ["pvxvoc.py"]),
            ):
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(Path("src/pvx/cli/pvxvoc.py")), run_name="__main__")

            with (
                patch("pvx.voc_cli.main", return_value=0),
                patch("sys.argv", ["voc_cli.py"]),
            ):
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(Path("src/pvx/voc_cli.py")), run_name="__main__")

            tap = root / "homebrew-pvx"
            formula = root / "pvx.rb"
            formula.write_text("class Pvx < Formula\nend\n", encoding="utf-8")
            with (
                patch("sys.argv", ["scripts_sync_homebrew_tap_formula.py", str(tap), "--formula", str(formula)]),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    runpy.run_path(
                        str(Path("scripts/scripts_sync_homebrew_tap_formula.py")),
                        run_name="__main__",
                    )


if __name__ == "__main__":
    unittest.main()
