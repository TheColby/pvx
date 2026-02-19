"""CLI regression tests for pvxvoc end-to-end workflows.

Coverage includes:
- baseline multi-channel pitch/time behavior
- dry-run behavior with existing outputs
- microtonal cents-shift CLI path
- non-power-of-two Fourier-sync mode
- a numeric DSP snapshot metric for drift detection
"""

import subprocess
import sys
import tempfile
import unittest
import csv
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "pvxvoc.py"
UNIFIED_CLI = ROOT / "pvx.py"
HPS_CLI = ROOT / "HPS-pitch-track.py"


def write_stereo_tone(path: Path, sr: int = 24000, duration: float = 0.5) -> tuple[np.ndarray, int]:
    t = np.arange(int(sr * duration)) / sr
    left = 0.35 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.30 * np.sin(2 * np.pi * 330.0 * t)
    audio = np.stack([left, right], axis=1)
    sf.write(path, audio, sr)
    return audio, sr


def write_mono_tone(path: Path, sr: int = 24000, duration: float = 0.5, freq_hz: float = 220.0) -> tuple[np.ndarray, int]:
    t = np.arange(int(sr * duration)) / sr
    audio = 0.35 * np.sin(2 * np.pi * freq_hz * t)
    sf.write(path, audio, sr)
    return audio.astype(np.float64), sr


class TestCLIRegression(unittest.TestCase):
    def test_unified_cli_lists_tools(self) -> None:
        cmd = [sys.executable, str(UNIFIED_CLI), "list"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("voc", proc.stdout)
        self.assertIn("freeze", proc.stdout)
        self.assertIn("pitch-track", proc.stdout)

    def test_unified_cli_dispatches_voc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "unified.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.3)
            out_path = tmp_path / "unified_out.wav"

            cmd = [
                sys.executable,
                str(UNIFIED_CLI),
                "voc",
                str(in_path),
                "--time-stretch",
                "1.15",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            expected_len = int(round(input_audio.shape[0] * 1.15))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=8)

    def test_unified_cli_path_shortcut_defaults_to_voc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "shortcut.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.3)
            out_path = tmp_path / "shortcut_out.wav"

            cmd = [
                sys.executable,
                str(UNIFIED_CLI),
                str(in_path),
                "--stretch",
                "1.10",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            expected_len = int(round(input_audio.shape[0] * 1.10))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=8)

    def test_unified_cli_chain_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "chain.wav"
            write_stereo_tone(in_path, duration=0.22)
            out_path = tmp_path / "chain_out.wav"

            cmd = [
                sys.executable,
                str(UNIFIED_CLI),
                "chain",
                str(in_path),
                "--pipeline",
                "voc --time-stretch 1.10 | formant --formant-shift-ratio 1.02",
                "--output",
                str(out_path),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            out_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, 24000)
            self.assertEqual(out_audio.shape[1], 2)
            self.assertGreater(out_audio.shape[0], 0)

    def test_unified_cli_stream_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "stream.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.24)
            out_path = tmp_path / "stream_out.wav"

            cmd = [
                sys.executable,
                str(UNIFIED_CLI),
                "stream",
                str(in_path),
                "--output",
                str(out_path),
                "--chunk-seconds",
                "0.08",
                "--time-stretch",
                "1.2",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            expected_len = int(round(input_audio.shape[0] * 1.2))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=24)

    def test_unified_cli_stream_wrapper_mode_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "stream_wrapper.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.24)
            out_path = tmp_path / "stream_wrapper_out.wav"

            cmd = [
                sys.executable,
                str(UNIFIED_CLI),
                "stream",
                str(in_path),
                "--mode",
                "wrapper",
                "--output",
                str(out_path),
                "--chunk-seconds",
                "0.08",
                "--time-stretch",
                "1.2",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            expected_len = int(round(input_audio.shape[0] * 1.2))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=24)

    def test_common_tool_accepts_explicit_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "freeze_in.wav"
            write_mono_tone(in_path, duration=0.22, freq_hz=330.0)
            out_path = tmp_path / "freeze_out.wav"

            cmd = [
                sys.executable,
                str(UNIFIED_CLI),
                "freeze",
                str(in_path),
                "--duration",
                "1.0",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

    def test_hps_pitch_tracker_emits_control_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "pitch_src.wav"
            write_mono_tone(in_path, duration=0.55, freq_hz=245.0)

            cmd = [
                sys.executable,
                str(HPS_CLI),
                str(in_path),
                "--backend",
                "acf",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            rows = list(csv.DictReader(io.StringIO(proc.stdout)))
            self.assertGreater(len(rows), 10)
            first = rows[0]
            self.assertIn("start_sec", first)
            self.assertIn("end_sec", first)
            self.assertIn("stretch", first)
            self.assertIn("pitch_ratio", first)
            self.assertIn("confidence", first)
            self.assertGreater(float(first["pitch_ratio"]), 0.0)

    def test_cli_pitch_map_stdin_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pitch_src = tmp_path / "a.wav"
            proc_src = tmp_path / "b.wav"
            write_mono_tone(pitch_src, duration=0.65, freq_hz=196.0)
            input_audio, sr = write_stereo_tone(proc_src, duration=0.65)

            track_cmd = [
                sys.executable,
                str(HPS_CLI),
                str(pitch_src),
                "--backend",
                "acf",
                "--quiet",
            ]
            track = subprocess.run(track_cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(track.returncode, 0, msg=track.stderr)
            self.assertIn("pitch_ratio", track.stdout)

            out_path = tmp_path / "follow.wav"
            voc_cmd = [
                sys.executable,
                str(CLI),
                str(proc_src),
                "--pitch-follow-stdin",
                "--pitch-conf-min",
                "0.1",
                "--time-stretch-factor",
                "1.0",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            voc = subprocess.run(voc_cmd, cwd=ROOT, input=track.stdout, capture_output=True, text=True)
            self.assertEqual(voc.returncode, 0, msg=voc.stderr)
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            self.assertGreater(output_audio.shape[0], 0)
            self.assertNotEqual(output_audio.shape[0], input_audio.shape[0])

    def test_cli_output_policy_sidecar_and_bit_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "policy_in.wav"
            write_stereo_tone(in_path, duration=0.22)
            out_path = tmp_path / "policy_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--time-stretch",
                "1.0",
                "--bit-depth",
                "16",
                "--dither",
                "tpdf",
                "--dither-seed",
                "7",
                "--metadata-policy",
                "sidecar",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            info = sf.info(out_path)
            self.assertEqual(info.subtype, "PCM_16")

            sidecar = Path(str(out_path) + ".metadata.json")
            self.assertTrue(sidecar.exists())
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata_policy"], "sidecar")
            self.assertEqual(payload["output"]["subtype"], "PCM_16")
            self.assertEqual(payload["output_policy"]["dither"], "tpdf")
            self.assertEqual(payload["output_policy"]["dither_seed"], 7)

    def test_cli_multi_channel_pitch_and_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "in.wav"
            input_audio, sr = write_stereo_tone(in_path)

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--time-stretch",
                "1.3",
                "--pitch-shift-semitones",
                "4",
                "--phase-locking",
                "identity",
                "--transient-preserve",
                "--pitch-mode",
                "formant-preserving",
                "--device",
                "cpu",
                "--overwrite",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            out_path = tmp_path / "in_pv.wav"
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)

            expected_len = int(round(input_audio.shape[0] * 1.3))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=4)

    def test_cli_dry_run_allows_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "tone.wav"
            write_stereo_tone(in_path)

            out_path = tmp_path / "tone_pv.wav"
            sf.write(out_path, np.zeros((128, 2), dtype=np.float64), 24000)

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--target-f0",
                "440",
                "--f0-min",
                "80",
                "--f0-max",
                "600",
                "--dry-run",
                "--verbose",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("[ok]", proc.stdout)

    def test_cli_microtonal_pitch_shift_cents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "micro.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.4)

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--pitch-shift-cents",
                "50",
                "--phase-locking",
                "identity",
                "--overwrite",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            out_path = tmp_path / "micro_pv.wav"
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            self.assertAlmostEqual(output_audio.shape[0], input_audio.shape[0], delta=4)

    def test_cli_time_stretch_factor_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "alias.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.45)
            out_path = tmp_path / "alias_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--time-stretch-factor",
                "1.12",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            expected_len = int(round(input_audio.shape[0] * 1.12))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=6)

    def test_cli_extreme_multistage_time_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "extreme.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.22)
            out_path = tmp_path / "extreme_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--time-stretch",
                "4.0",
                "--stretch-mode",
                "multistage",
                "--max-stage-stretch",
                "1.5",
                "--n-fft",
                "512",
                "--win-length",
                "512",
                "--hop-size",
                "128",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            expected_len = int(round(input_audio.shape[0] * 4.0))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=10)

    def test_cli_pitch_shift_ratio_expression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "ratio_expr.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.35)

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--pitch-shift-ratio",
                "2^(1/12)",
                "--overwrite",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            out_path = tmp_path / "ratio_expr_pv.wav"
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            self.assertAlmostEqual(output_audio.shape[0], input_audio.shape[0], delta=4)

    def test_cli_transform_switch_dct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "dct.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.3)

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--transform",
                "dct",
                "--time-stretch",
                "1.05",
                "--overwrite",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            out_path = tmp_path / "dct_pv.wav"
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            expected_len = int(round(input_audio.shape[0] * 1.05))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=4)

    def test_cli_multires_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "mr.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.35)
            out_path = tmp_path / "mr_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--multires-fusion",
                "--multires-ffts",
                "256,512",
                "--multires-weights",
                "0.45,0.55",
                "--time-stretch",
                "1.10",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            expected_len = int(round(input_audio.shape[0] * 1.10))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=10)

    def test_cli_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "cp.wav"
            write_stereo_tone(in_path, duration=0.55)
            checkpoint_dir = tmp_path / "ckpt"
            out_path_a = tmp_path / "cp_a.wav"
            out_path_b = tmp_path / "cp_b.wav"

            base_cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--auto-segment-seconds",
                "0.10",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--time-stretch",
                "1.25",
                "--overwrite",
                "--quiet",
            ]

            first = subprocess.run(base_cmd + ["--output", str(out_path_a)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertTrue(out_path_a.exists())
            self.assertTrue(any(checkpoint_dir.rglob("segment_*.npy")))

            second = subprocess.run(
                base_cmd + ["--resume", "--output", str(out_path_b)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, msg=second.stderr)
            self.assertTrue(out_path_b.exists())

    def test_cli_explain_plan_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "plan.wav"
            write_stereo_tone(in_path, duration=0.25)

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--auto-profile",
                "--auto-transform",
                "--explain-plan",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertIn("active_profile", payload)
            self.assertIn("config", payload)
            self.assertIn("runtime", payload)

    def test_cli_example_mode_outputs_command(self) -> None:
        cmd = [
            sys.executable,
            str(CLI),
            "--example",
            "basic",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("pvx voc input.wav", proc.stdout)

    def test_unified_stream_help_includes_mode(self) -> None:
        cmd = [sys.executable, str(UNIFIED_CLI), "stream", "--help"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--mode {stateful,wrapper}", proc.stdout)

    def test_cli_help_contains_grouped_sections(self) -> None:
        cmd = [sys.executable, str(CLI), "--help"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        help_text = proc.stdout
        for heading in (
            "I/O:",
            "Performance:",
            "Quality/Phase:",
            "Time/Pitch:",
            "Transients:",
            "Stereo:",
            "Output/Mastering:",
            "Debug:",
        ):
            self.assertIn(heading, help_text)
        self.assertEqual(help_text.count("Time/Pitch:"), 1)
        self.assertIn("--bit-depth", help_text)
        self.assertIn("--dither", help_text)
        self.assertIn("--true-peak-max-dbtp", help_text)
        self.assertIn("--metadata-policy", help_text)

    def test_cli_transient_preserve_maps_to_reset_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "legacy_transient.wav"
            write_stereo_tone(in_path, duration=0.2)
            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--transient-preserve",
                "--explain-plan",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            plan = json.loads(proc.stdout)
            self.assertEqual(plan["config"]["transient_mode"], "reset")

    def test_cli_preset_and_beginner_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "alias_preset.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.35)
            out_path = tmp_path / "alias_preset_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--preset",
                "vocal",
                "--stretch",
                "1.08",
                "--pitch",
                "-1",
                "--out",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            expected_len = int(round(input_audio.shape[0] * 1.08))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=8)

    def test_cli_fourier_sync_non_power_of_two_fft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "harmonic.wav"
            input_audio, sr = write_stereo_tone(in_path, sr=22050, duration=0.6)

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--fourier-sync",
                "--n-fft",
                "1500",
                "--win-length",
                "1500",
                "--hop-size",
                "375",
                "--f0-min",
                "70",
                "--f0-max",
                "500",
                "--time-stretch",
                "1.2",
                "--overwrite",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            out_path = tmp_path / "harmonic_pv.wav"
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)

            expected_len = int(round(input_audio.shape[0] * 1.2))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=8)

    def test_cli_hybrid_transient_mode_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "hybrid_in.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.35)
            out_path = tmp_path / "hybrid_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--transient-mode",
                "hybrid",
                "--transient-sensitivity",
                "0.6",
                "--transient-protect-ms",
                "30",
                "--transient-crossfade-ms",
                "10",
                "--time-stretch",
                "1.12",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            expected_len = int(round(input_audio.shape[0] * 1.12))
            self.assertAlmostEqual(output_audio.shape[0], expected_len, delta=8)

    def test_cli_stereo_coherence_mode_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "stereo_lock_in.wav"
            _, sr = write_stereo_tone(in_path, duration=0.35)
            out_path = tmp_path / "stereo_lock_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--stereo-mode",
                "ref_channel_lock",
                "--ref-channel",
                "0",
                "--coherence-strength",
                "0.9",
                "--time-stretch",
                "1.1",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)

    def test_cli_quiet_prints_audio_metrics_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "metrics_in.wav"
            write_stereo_tone(in_path, duration=0.25)
            out_path = tmp_path / "metrics_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--time-stretch",
                "1.05",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            combined = proc.stdout + "\n" + proc.stderr
            self.assertIn("Audio Metrics", combined)
            self.assertIn("Audio Compare Metrics", combined)
            self.assertIn("SNR dB", combined)
            self.assertIn("delta(last-first)", combined)

    def test_cli_silent_hides_audio_metrics_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "silent_metrics_in.wav"
            write_stereo_tone(in_path, duration=0.25)
            out_path = tmp_path / "silent_metrics_out.wav"

            cmd = [
                sys.executable,
                str(CLI),
                str(in_path),
                "--time-stretch",
                "1.05",
                "--output",
                str(out_path),
                "--overwrite",
                "--silent",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            combined = proc.stdout + "\n" + proc.stderr
            self.assertNotIn("Audio Metrics", combined)

    def test_regression_metrics_snapshot(self) -> None:
        from pvxvoc import VocoderConfig, phase_vocoder_time_stretch, resample_1d

        sr = 24000
        n = int(sr * 0.7)
        t = np.arange(n) / sr
        rng = np.random.default_rng(42)
        x = (
            0.4 * np.sin(2 * np.pi * 220.0 * t)
            + 0.2 * np.sin(2 * np.pi * 440.0 * t)
            + 0.05 * rng.normal(size=n)
        )

        ratio = 2 ** (3.0 / 12.0)
        stretch = 1.15
        internal = stretch * ratio

        cfg = VocoderConfig(
            n_fft=1024,
            win_length=1024,
            hop_size=256,
            window="hann",
            center=True,
            phase_locking="identity",
            transient_preserve=True,
            transient_threshold=2.0,
        )

        y = phase_vocoder_time_stretch(x, internal, cfg)
        y = resample_1d(y, int(round(y.size / ratio)), mode="linear")

        rms = float(np.sqrt(np.mean(y * y)))
        peak = float(np.max(np.abs(y)))
        spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
        freqs = np.fft.rfftfreq(y.size, d=1.0 / sr)
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))

        self.assertEqual(y.size, 19320)
        self.assertAlmostEqual(rms, 0.3158, delta=0.015)
        self.assertAlmostEqual(peak, 0.6728, delta=0.06)
        self.assertAlmostEqual(centroid, 4691.8, delta=300.0)


if __name__ == "__main__":
    unittest.main()
