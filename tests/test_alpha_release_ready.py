"""Direct seam tests for alpha-release refactors."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"
MAKEFILE = ROOT / "Makefile"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.scripts_sync_homebrew_tap_formula import main as sync_homebrew_formula_main

from pvx import voc_cli
from pvx.cli.pvx import _consume_lucky_options, _infer_output_format, _parse_size_bytes
from pvx.core import voc
from pvx.core.voc import DynamicControlRef
from pvx.core.voc_console import (
    apply_named_preset as console_apply_named_preset,
)
from pvx.core.voc_console import (
    collect_cli_flags as console_collect_cli_flags,
)
from pvx.core.voc_jobs import _checkpoint_job_id


class TestCLIHelperSeams(unittest.TestCase):
    def test_makefile_enforces_supported_slice_coverage_at_100(self) -> None:
        makefile = MAKEFILE.read_text(encoding="utf-8")
        self.assertIn("--fail-under=100", makefile)

    def test_release_workflow_runs_alpha_gate_and_formula_sync(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python scripts/scripts_alpha_check.py", workflow)
        self.assertIn("./scripts/refresh_homebrew_formula.sh", workflow)
        self.assertIn("python scripts/scripts_sync_homebrew_tap_formula.py", workflow)
        self.assertIn("ruby -c Formula/pvx.rb", workflow)

    def test_consume_lucky_options_preserves_non_lucky_args(self) -> None:
        clean, lucky_count, lucky_seed = _consume_lucky_options(
            ["clip.wav", "--lucky", "3", "--quiet", "--lucky-seed=17"]
        )
        self.assertEqual(clean, ["clip.wav", "--quiet"])
        self.assertEqual(lucky_count, 3)
        self.assertEqual(lucky_seed, 17)

    def test_parse_size_bytes_supports_decimal_units(self) -> None:
        self.assertEqual(_parse_size_bytes("2.5GB"), 2_500_000_000.0)

    def test_infer_output_format_normalizes_aliases(self) -> None:
        self.assertEqual(_infer_output_format(Path("clip.aif"), "auto"), "aiff")
        self.assertEqual(_infer_output_format(Path("clip.wav"), ".oga"), "ogg")

    def test_sync_homebrew_formula_script_copies_formula_into_tap_checkout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-homebrew-tap-") as tmp:
            root = Path(tmp)
            tap = root / "homebrew-pvx"
            formula_dir = tap / "Formula"
            formula_dir.mkdir(parents=True)
            source_formula = root / "pvx.rb"
            source_formula.write_text("class Pvx < Formula\nend\n", encoding="utf-8")

            rc = sync_homebrew_formula_main([str(tap), "--formula", str(source_formula)])

            self.assertEqual(rc, 0)
            self.assertEqual(
                (formula_dir / "pvx.rb").read_text(encoding="utf-8"),
                source_formula.read_text(encoding="utf-8"),
            )

    def test_sync_homebrew_formula_script_rejects_missing_formula(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-homebrew-tap-") as tmp:
            tap = Path(tmp) / "homebrew-pvx"
            tap.mkdir()
            with self.assertRaises(SystemExit):
                sync_homebrew_formula_main([str(tap), "--formula", str(Path(tmp) / "missing.rb")])

    def test_sync_homebrew_formula_script_rejects_missing_or_bad_tap_checkout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-homebrew-tap-") as tmp:
            root = Path(tmp)
            source_formula = root / "pvx.rb"
            source_formula.write_text("class Pvx < Formula\nend\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                sync_homebrew_formula_main(
                    [str(root / "missing-tap"), "--formula", str(source_formula)]
                )
            bad_tap = root / "pvx"
            bad_tap.mkdir()
            with self.assertRaises(SystemExit):
                sync_homebrew_formula_main([str(bad_tap), "--formula", str(source_formula)])

    def test_supported_surface_doc_matches_stable_pyproject_scripts(self) -> None:
        payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        scripts = payload["project"]["scripts"]
        supported_surface = (ROOT / "docs" / "SUPPORTED_SURFACE.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        stable_expected = [
            "pvx",
            "pvxvoc",
            "pvxfreeze",
            "pvxwarp",
            "pvxformant",
            "pvxfilter",
            "pvxretune",
            "pvxanalysis",
        ]
        for name in stable_expected:
            self.assertIn(name, scripts)
            self.assertIn(f"`{name}`", supported_surface)
            self.assertIn(f"`{name}`", readme)


class TestVocCompatibilitySeams(unittest.TestCase):
    def test_collect_cli_flags_wrapper_matches_console_helper(self) -> None:
        argv = ["input.wav", "--stretch=1.25", "--preset", "default", "--quiet"]
        self.assertEqual(voc.collect_cli_flags(argv), console_collect_cli_flags(argv))

    def test_apply_named_preset_wrapper_matches_console_helper(self) -> None:
        args_voc = argparse.Namespace(
            quality_profile="neutral",
            transient_mode="off",
            stereo_mode="independent",
            coherence_strength=0.0,
        )
        args_console = argparse.Namespace(
            quality_profile="neutral",
            transient_mode="off",
            stereo_mode="independent",
            coherence_strength=0.0,
        )
        changed_voc = voc.apply_named_preset(args_voc, preset="default", provided_flags=set())
        changed_console = console_apply_named_preset(
            args_console, preset="default", provided_flags=set()
        )
        self.assertEqual(changed_voc, changed_console)
        self.assertEqual(vars(args_voc), vars(args_console))

    def test_compute_output_path_wrapper_uses_suffix_and_format(self) -> None:
        out = voc.compute_output_path(
            Path("/tmp/input.wav"),
            Path("/tmp/out"),
            "_alpha",
            "flac",
        )
        self.assertEqual(out, Path("/tmp/out/input_alpha.flac"))

    def test_write_manifest_wrapper_persists_entries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-manifest-") as tmp:
            path = Path(tmp) / "manifest.json"
            entries = [{"status": "ok", "input_path": "in.wav", "output_path": "out.wav"}]
            voc.write_manifest(path, entries, append=False)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["entry_count"], 1)
            self.assertEqual(payload["entries"][0]["status"], "ok")

    def test_write_manifest_append_rejects_invalid_existing_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-manifest-") as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                voc.write_manifest(path, [{"status": "ok"}], append=True)

    def test_checkpoint_job_id_changes_when_control_map_content_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-checkpoint-") as tmp:
            root = Path(tmp)
            control = root / "map.csv"
            control.write_text("start_sec,end_sec,stretch\n0,1,1.0\n", encoding="utf-8")
            args = argparse.Namespace(
                pitch_map=control,
                route=[],
                _dynamic_control_refs={},
                target_duration=None,
                n_fft=2048,
                win_length=2048,
                hop_size=512,
                window="hann",
                phase_engine="propagate",
                transform="fft",
                _active_quality_profile="neutral",
                phase_locking="identity",
                stretch_mode="standard",
                pitch_mode="standard",
                transient_mode="off",
                transient_sensitivity=0.5,
                transient_protect_ms=30.0,
                transient_crossfade_ms=10.0,
                stereo_mode="independent",
                ref_channel=0,
                coherence_strength=0.0,
                auto_segment_seconds=0.0,
                multires_fusion=False,
                _multires_ffts=[],
                _multires_weights=[],
                ambient_phase_mix=0.5,
                formant_lifter=32,
                formant_strength=1.0,
                formant_max_gain_db=12.0,
            )
            before = _checkpoint_job_id(
                input_path=root / "input.wav",
                args=args,
                base_stretch=1.0,
                pitch_ratio=1.0,
            )
            control.write_text("start_sec,end_sec,stretch\n0,1,1.5\n", encoding="utf-8")
            after = _checkpoint_job_id(
                input_path=root / "input.wav",
                args=args,
                base_stretch=1.0,
                pitch_ratio=1.0,
            )
            self.assertNotEqual(before, after)

    def test_checkpoint_job_id_changes_when_input_audio_changes_in_place(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-checkpoint-") as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"first")
            args = argparse.Namespace(
                pitch_map=None,
                route=[],
                _dynamic_control_refs={},
                target_duration=None,
                n_fft=2048,
                win_length=2048,
                hop_size=512,
                window="hann",
                phase_engine="propagate",
                transform="fft",
                _active_quality_profile="neutral",
                phase_locking="identity",
                stretch_mode="standard",
                pitch_mode="standard",
                transient_mode="off",
                transient_sensitivity=0.5,
                transient_protect_ms=30.0,
                transient_crossfade_ms=10.0,
                stereo_mode="independent",
                ref_channel=0,
                coherence_strength=0.0,
                auto_segment_seconds=0.0,
                multires_fusion=False,
                _multires_ffts=[],
                _multires_weights=[],
                ambient_phase_mix=0.5,
                formant_lifter=32,
                formant_strength=1.0,
                formant_max_gain_db=12.0,
                pitch_conf_min=0.0,
                pitch_lowconf_mode="hold",
                pitch_map_smooth_ms=0.0,
                pitch_map_crossfade_ms=0.0,
            )
            before = _checkpoint_job_id(
                input_path=input_path,
                args=args,
                base_stretch=1.0,
                pitch_ratio=1.0,
            )
            input_path.write_bytes(b"second")
            after = _checkpoint_job_id(
                input_path=input_path,
                args=args,
                base_stretch=1.0,
                pitch_ratio=1.0,
            )
            self.assertNotEqual(before, after)

    def test_checkpoint_job_id_changes_when_dynamic_control_content_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-checkpoint-") as tmp:
            root = Path(tmp)
            dyn = root / "stretch.csv"
            dyn.write_text("time_sec,value\n0,1.0\n", encoding="utf-8")
            args = argparse.Namespace(
                pitch_map=None,
                route=[],
                _dynamic_control_refs={
                    "time_stretch": DynamicControlRef(
                        parameter="time_stretch",
                        path=dyn,
                        value_kind="float",
                        interpolation="linear",
                        order=3,
                    )
                },
                target_duration=None,
                n_fft=2048,
                win_length=2048,
                hop_size=512,
                window="hann",
                phase_engine="propagate",
                transform="fft",
                _active_quality_profile="neutral",
                phase_locking="identity",
                stretch_mode="standard",
                pitch_mode="standard",
                transient_mode="off",
                transient_sensitivity=0.5,
                transient_protect_ms=30.0,
                transient_crossfade_ms=10.0,
                stereo_mode="independent",
                ref_channel=0,
                coherence_strength=0.0,
                auto_segment_seconds=0.0,
                multires_fusion=False,
                _multires_ffts=[],
                _multires_weights=[],
                ambient_phase_mix=0.5,
                formant_lifter=32,
                formant_strength=1.0,
                formant_max_gain_db=12.0,
            )
            before = _checkpoint_job_id(
                input_path=root / "input.wav",
                args=args,
                base_stretch=1.0,
                pitch_ratio=1.0,
            )
            dyn.write_text("time_sec,value\n0,1.2\n", encoding="utf-8")
            after = _checkpoint_job_id(
                input_path=root / "input.wav",
                args=args,
                base_stretch=1.0,
                pitch_ratio=1.0,
            )
            self.assertNotEqual(before, after)

    def test_checkpoint_job_id_changes_when_control_policy_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-checkpoint-") as tmp:
            root = Path(tmp)
            control = root / "map.csv"
            control.write_text(
                "start_sec,end_sec,stretch,confidence\n0,1,1.0,0.5\n", encoding="utf-8"
            )
            args = argparse.Namespace(
                pitch_map=control,
                route=[],
                _dynamic_control_refs={},
                target_duration=None,
                n_fft=2048,
                win_length=2048,
                hop_size=512,
                window="hann",
                phase_engine="propagate",
                transform="fft",
                _active_quality_profile="neutral",
                phase_locking="identity",
                stretch_mode="standard",
                pitch_mode="standard",
                transient_mode="off",
                transient_sensitivity=0.5,
                transient_protect_ms=30.0,
                transient_crossfade_ms=10.0,
                stereo_mode="independent",
                ref_channel=0,
                coherence_strength=0.0,
                auto_segment_seconds=0.0,
                multires_fusion=False,
                _multires_ffts=[],
                _multires_weights=[],
                ambient_phase_mix=0.5,
                formant_lifter=32,
                formant_strength=1.0,
                formant_max_gain_db=12.0,
                pitch_conf_min=0.0,
                pitch_lowconf_mode="hold",
                pitch_map_smooth_ms=0.0,
                pitch_map_crossfade_ms=0.0,
            )
            before = _checkpoint_job_id(
                input_path=root / "input.wav", args=args, base_stretch=1.0, pitch_ratio=1.0
            )
            args.pitch_conf_min = 0.7
            after = _checkpoint_job_id(
                input_path=root / "input.wav", args=args, base_stretch=1.0, pitch_ratio=1.0
            )
            self.assertNotEqual(before, after)

    def test_validate_args_rejects_implicit_checkpoint_id_with_stdin_sources(self) -> None:
        parser = voc.build_parser()
        args = parser.parse_args(
            ["-", "--checkpoint-dir", "/tmp/checkpoints", "--auto-segment-seconds", "1.0"]
        )
        with self.assertRaises(SystemExit):
            voc.validate_args(args, parser)

    def test_validate_args_allows_checkpoint_dir_without_segment_writes_for_stdin(self) -> None:
        parser = voc.build_parser()
        args = parser.parse_args(
            ["-", "--checkpoint-dir", "/tmp/checkpoints", "--time-stretch", "1.1"]
        )
        with patch.object(parser, "error", side_effect=AssertionError("unexpected parser.error")):
            voc.validate_args(args, parser)

    def test_force_alias_sets_overwrite(self) -> None:
        parser = voc.build_parser()
        args = parser.parse_args(["input.wav", "--force"])
        self.assertTrue(args.overwrite)

    def test_voc_cli_manifest_preflight_fails_before_processing_when_target_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-manifest-") as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.build_vocoder_config_from_args"),
                patch("pvx.voc_cli.process_file", side_effect=AssertionError("should not process")),
                self.assertRaises(SystemExit),
            ):
                voc_cli.main([str(input_path), "--manifest-json", str(manifest)])

    def test_voc_cli_manifest_append_allows_existing_valid_manifest_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-manifest-") as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"entries": []}), encoding="utf-8")
            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch(
                    "pvx.voc_cli.read_audio_input",
                    return_value=(__import__("numpy").ones((16, 1)), 16000),
                ),
                patch("pvx.voc_cli.resolve_base_stretch", return_value=1.0),
                patch(
                    "pvx.voc_cli.estimate_content_features", return_value={"spectral_flatness": 0.1}
                ),
                patch("pvx.voc_cli.suggest_quality_profile", return_value="neutral"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.build_vocoder_config_from_args"),
                patch(
                    "pvx.voc_cli.process_file",
                    return_value=voc.JobResult(
                        input_path=input_path,
                        output_path=root / "out.wav",
                        in_sr=16000,
                        out_sr=16000,
                        in_samples=16,
                        out_samples=16,
                        channels=1,
                        stretch=1.0,
                        pitch_ratio=1.0,
                    ),
                ),
                patch("pvx.voc_cli.write_manifest"),
            ):
                result = voc_cli.main(
                    [str(input_path), "--manifest-json", str(manifest), "--manifest-append"]
                )
            self.assertEqual(result, 0)

    def test_voc_cli_auto_profile_prefetches_first_input_audio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-alpha-prefetch-") as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            input_path.write_bytes(b"wav")
            fake_audio = __import__("numpy").ones((16, 1))
            captured_args: list[argparse.Namespace] = []

            def capture_process_file(path, args, config, file_index=0, file_total=1):
                captured_args.append(args)
                return voc.JobResult(
                    input_path=path,
                    output_path=root / "out.wav",
                    in_sr=16000,
                    out_sr=16000,
                    in_samples=16,
                    out_samples=16,
                    channels=1,
                    stretch=1.0,
                    pitch_ratio=1.0,
                )

            with (
                patch("pvx.voc_cli.ensure_runtime_dependencies"),
                patch("pvx.voc_cli.read_audio_input", return_value=(fake_audio, 16000)),
                patch("pvx.voc_cli.resolve_base_stretch", return_value=1.0),
                patch(
                    "pvx.voc_cli.estimate_content_features", return_value={"spectral_flatness": 0.1}
                ),
                patch("pvx.voc_cli.suggest_quality_profile", return_value="neutral"),
                patch("pvx.voc_cli.apply_quality_profile_overrides", return_value=[]),
                patch("pvx.voc_cli.build_vocoder_config_from_args"),
                patch("pvx.voc_cli.process_file", side_effect=capture_process_file),
            ):
                result = voc_cli.main([str(input_path), "--auto-profile"])

            self.assertEqual(result, 0)
            self.assertEqual(len(captured_args), 1)
            self.assertIn(str(input_path.resolve()), captured_args[0]._prefetched_audio_cache)


if __name__ == "__main__":
    unittest.main()
