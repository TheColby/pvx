# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

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
import os
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "pvx.core.voc"]
UNIFIED_CLI = [sys.executable, "-m", "pvx.cli.pvx"]
HPS_CLI = [sys.executable, "-m", "pvx.cli.hps_pitch_track"]
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")


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


def write_mono_glide(path: Path, sr: int = 24000, duration: float = 0.6, f0_start: float = 180.0, f0_end: float = 360.0) -> tuple[np.ndarray, int]:
    t = np.arange(int(sr * duration), dtype=np.float64) / float(sr)
    freq = np.linspace(float(f0_start), float(f0_end), num=t.size, dtype=np.float64)
    phase = 2.0 * np.pi * np.cumsum(freq) / float(sr)
    audio = 0.35 * np.sin(phase)
    sf.write(path, audio, sr)
    return audio.astype(np.float64), sr


def write_mono_complex(path: Path, sr: int = 24000, duration: float = 0.5) -> tuple[np.ndarray, int]:
    t = np.arange(int(sr * duration), dtype=np.float64) / float(sr)
    audio = (
        0.22 * np.sin(2 * np.pi * 110.0 * t)
        + 0.18 * np.sin(2 * np.pi * 330.0 * t)
        + 0.12 * np.sin(2 * np.pi * 880.0 * t)
    )
    env = np.linspace(0.3, 1.0, num=t.size, dtype=np.float64)
    audio = 0.35 * env * audio
    sf.write(path, audio, sr)
    return audio.astype(np.float64), sr


def write_multichannel_ir(path: Path, sr: int = 24000, channels: int = 4, duration: float = 0.12) -> tuple[np.ndarray, int]:
    n = max(8, int(round(sr * duration)))
    ir = np.zeros((n, channels), dtype=np.float64)
    taps = [0, int(0.006 * sr), int(0.012 * sr), int(0.021 * sr)]
    for ch in range(channels):
        for idx, tap in enumerate(taps):
            if tap < n:
                ir[tap, ch] += float((0.75 / (idx + 1)) * (1.0 - (0.08 * ch)))
    # Tiny diffuse tail to avoid perfectly sparse impulse artifacts in tests.
    tail = np.linspace(0.15, 0.0, num=n, dtype=np.float64)
    for ch in range(channels):
        ir[:, ch] += 0.002 * tail
    sf.write(path, ir, sr)
    return ir, sr


class TestCLIRegression(unittest.TestCase):
    def test_unified_cli_lists_tools(self) -> None:
        cmd = [*UNIFIED_CLI, "list"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("voc", proc.stdout)
        self.assertIn("freeze", proc.stdout)
        self.assertIn("pitch-track", proc.stdout)
        self.assertIn("follow", proc.stdout)
        self.assertIn("stretch-budget", proc.stdout)

    def test_unified_cli_stretch_budget_json_reports_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "budget_in.wav"
            write_mono_tone(in_path, duration=0.4, freq_hz=220.0)

            cmd = [
                *UNIFIED_CLI,
                "stretch-budget",
                str(in_path),
                "--disk-budget",
                "40MB",
                "--bit-depth",
                "16",
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertIn("max_safe_stretch", payload)
            self.assertGreater(float(payload["max_safe_stretch"]), 1000.0)
            self.assertEqual(payload["output_format_assumed"], "wav")
            self.assertIsNone(payload["requested_fits_budget"])

    def test_unified_cli_stretch_budget_requested_fail_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "budget_req.wav"
            write_mono_tone(in_path, duration=0.25, freq_hz=250.0)

            cmd = [
                *UNIFIED_CLI,
                "stretch-budget",
                str(in_path),
                "--disk-budget",
                "1MB",
                "--bit-depth",
                "16",
                "--requested-stretch",
                "1000000",
                "--fail-if-exceeds",
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            self.assertEqual(float(payload["requested_stretch"]), 1_000_000.0)
            self.assertFalse(bool(payload["requested_fits_budget"]))
            self.assertIn("exceeds usable budget", proc.stderr)

    def test_unified_cli_dispatches_voc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "unified.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.3)
            out_path = tmp_path / "unified_out.wav"

            cmd = [
                *UNIFIED_CLI,
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
                *UNIFIED_CLI,
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
                *UNIFIED_CLI,
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

    def test_unified_cli_chain_lucky_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "chain_lucky.wav"
            write_stereo_tone(in_path, duration=0.18)
            out_path = tmp_path / "chain_out.wav"

            cmd = [
                *UNIFIED_CLI,
                "chain",
                str(in_path),
                "--pipeline",
                "voc --time-stretch 1.03",
                "--output",
                str(out_path),
                "--lucky",
                "2",
                "--lucky-seed",
                "7",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            lucky_1 = tmp_path / "chain_out_lucky_001.wav"
            lucky_2 = tmp_path / "chain_out_lucky_002.wav"
            self.assertTrue(lucky_1.exists())
            self.assertTrue(lucky_2.exists())

    def test_unified_cli_stream_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "stream.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.24)
            out_path = tmp_path / "stream_out.wav"

            cmd = [
                *UNIFIED_CLI,
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
                *UNIFIED_CLI,
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

    def test_unified_cli_follow_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            guide_path = tmp_path / "follow_guide.wav"
            target_path = tmp_path / "follow_target.wav"
            write_mono_glide(guide_path, duration=0.65, f0_start=140.0, f0_end=360.0)
            input_audio, sr = write_stereo_tone(target_path, duration=0.65)
            out_path = tmp_path / "follow_out.wav"

            cmd = [
                *UNIFIED_CLI,
                "follow",
                str(guide_path),
                str(target_path),
                "--backend",
                "acf",
                "--emit",
                "pitch_to_stretch",
                "--stretch-scale",
                "2.0",
                "--stretch-min",
                "1.5",
                "--stretch-max",
                "2.5",
                "--pitch-conf-min",
                "0.1",
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
            self.assertGreater(abs(output_audio.shape[0] - input_audio.shape[0]), 32)

    def test_unified_cli_follow_rejects_output_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            guide_path = tmp_path / "follow_err_guide.wav"
            target_path = tmp_path / "follow_err_target.wav"
            write_mono_tone(guide_path, duration=0.2, freq_hz=220.0)
            write_mono_tone(target_path, duration=0.2, freq_hz=330.0)
            out_path = tmp_path / "follow_err_out.wav"

            cmd = [
                *UNIFIED_CLI,
                "follow",
                str(guide_path),
                str(target_path),
                "--output",
                str(out_path),
                "--stdout",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Do not pass", proc.stderr)

    def test_unified_cli_help_follow_target(self) -> None:
        cmd = [*UNIFIED_CLI, "help", "follow"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("pvx follow --help", proc.stdout)

    def test_unified_cli_help_stretch_budget_target(self) -> None:
        cmd = [*UNIFIED_CLI, "help", "stretch-budget"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("pvx stretch-budget --help", proc.stdout)

    def test_unified_cli_retune_help_includes_a4_reference_flag(self) -> None:
        cmd = [*UNIFIED_CLI, "retune", "--help"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--a4-reference-hz", proc.stdout)
        self.assertIn("--root-hz", proc.stdout)
        self.assertIn("--recommend-root", proc.stdout)

    def test_unified_cli_retune_rejects_root_hz_with_recommend_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "retune_root_conflict.wav"
            write_mono_tone(in_path, duration=0.25, freq_hz=220.0)

            cmd = [
                *UNIFIED_CLI,
                "retune",
                str(in_path),
                "--root-hz",
                "261.6256",
                "--recommend-root",
                "--output",
                str(tmp_path / "retune_root_conflict_out.wav"),
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--recommend-root and --root-hz are mutually exclusive", proc.stderr)

    def test_unified_cli_follow_example_basic(self) -> None:
        cmd = [*UNIFIED_CLI, "follow", "--example"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("[basic]", proc.stdout)
        self.assertIn("pvx follow guide.wav target.wav", proc.stdout)

    def test_unified_cli_follow_example_all(self) -> None:
        cmd = [*UNIFIED_CLI, "follow", "--example", "all"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("pvx follow example commands", proc.stdout)
        self.assertIn("[mfcc_flux]", proc.stdout)
        self.assertIn("[noise_aware]", proc.stdout)

    def test_common_tool_accepts_explicit_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "freeze_in.wav"
            write_mono_tone(in_path, duration=0.22, freq_hz=330.0)
            out_path = tmp_path / "freeze_out.wav"

            cmd = [
                *UNIFIED_CLI,
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

    def test_freeze_phase_mode_instantaneous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "freeze_phase_in.wav"
            write_mono_tone(in_path, duration=0.22, freq_hz=210.0)
            out_path = tmp_path / "freeze_phase_out.wav"

            cmd = [
                *UNIFIED_CLI,
                "freeze",
                str(in_path),
                "--freeze-time",
                "0.11",
                "--duration",
                "1.1",
                "--phase-mode",
                "instantaneous",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            out_audio, _ = sf.read(out_path, always_2d=True)
            self.assertGreater(float(np.sqrt(np.mean(out_audio[:, 0] * out_audio[:, 0]))), 1e-4)

    def test_unified_cli_lucky_voc_generates_multiple_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "lucky_voc_in.wav"
            write_stereo_tone(in_path, duration=0.22)
            out_dir = tmp_path / "out"

            cmd = [
                *UNIFIED_CLI,
                "voc",
                str(in_path),
                "--output-dir",
                str(out_dir),
                "--lucky",
                "2",
                "--lucky-seed",
                "17",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            lucky_outputs = sorted(out_dir.glob("*_lucky_*.wav"))
            self.assertGreaterEqual(len(lucky_outputs), 2)

    def test_unified_cli_trajectory_reverb_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_path = tmp_path / "traj_src.wav"
            ir_path = tmp_path / "traj_ir_4ch.wav"
            out_path = tmp_path / "traj_out.wav"
            write_mono_tone(src_path, duration=0.24, freq_hz=180.0)
            write_multichannel_ir(ir_path, channels=4)

            cmd = [
                *UNIFIED_CLI,
                "trajectory-reverb",
                str(src_path),
                "--ir",
                str(ir_path),
                "--coord-system",
                "cartesian",
                "--start=-1,0,1",
                "--end",
                "1,0,1",
                "--trajectory-shape",
                "ease-in-out",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            out_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, 24000)
            self.assertEqual(out_audio.shape[1], 4)
            self.assertGreater(out_audio.shape[0], 0)

    def test_pvxmorph_carrier_envelope_mode_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a_path = tmp_path / "morph_a.wav"
            b_path = tmp_path / "morph_b.wav"
            write_mono_tone(a_path, duration=0.35, freq_hz=220.0)
            write_mono_complex(b_path, duration=0.35)
            out_path = tmp_path / "morph_env.wav"

            cmd = [
                *UNIFIED_CLI,
                "morph",
                str(a_path),
                str(b_path),
                "--alpha",
                "0.75",
                "--blend-mode",
                "carrier_a_envelope_b",
                "--envelope-lifter",
                "28",
                "--normalize-energy",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, 24000)
            self.assertEqual(output_audio.shape[1], 1)
            self.assertGreater(output_audio.shape[0], 0)

    def test_pvxmorph_blend_modes_produce_different_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a_path = tmp_path / "morph_cmp_a.wav"
            b_path = tmp_path / "morph_cmp_b.wav"
            write_mono_complex(a_path, duration=0.4)
            write_mono_tone(b_path, duration=0.4, freq_hz=540.0)
            out_linear = tmp_path / "morph_linear.wav"
            out_mask = tmp_path / "morph_mask.wav"

            cmd_linear = [
                *UNIFIED_CLI,
                "morph",
                str(a_path),
                str(b_path),
                "--alpha",
                "0.6",
                "--blend-mode",
                "linear",
                "--output",
                str(out_linear),
                "--overwrite",
                "--quiet",
            ]
            cmd_mask = [
                *UNIFIED_CLI,
                "morph",
                str(a_path),
                str(b_path),
                "--alpha",
                "0.6",
                "--blend-mode",
                "carrier_a_mask_b",
                "--mask-exponent",
                "1.4",
                "--output",
                str(out_mask),
                "--overwrite",
                "--quiet",
            ]
            proc_linear = subprocess.run(cmd_linear, cwd=ROOT, capture_output=True, text=True)
            proc_mask = subprocess.run(cmd_mask, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc_linear.returncode, 0, msg=proc_linear.stderr)
            self.assertEqual(proc_mask.returncode, 0, msg=proc_mask.stderr)
            self.assertTrue(out_linear.exists())
            self.assertTrue(out_mask.exists())

            y_linear, sr_linear = sf.read(out_linear, always_2d=True)
            y_mask, sr_mask = sf.read(out_mask, always_2d=True)
            self.assertEqual(sr_linear, sr_mask)
            n = min(y_linear.shape[0], y_mask.shape[0])
            diff = float(np.mean(np.abs(y_linear[:n, 0] - y_mask[:n, 0])))
            self.assertGreater(diff, 1e-4)

    def test_pvxmorph_alpha_trajectory_control_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a_path = tmp_path / "morph_traj_a.wav"
            b_path = tmp_path / "morph_traj_b.wav"
            a_audio, sr = write_mono_tone(a_path, duration=0.5, freq_hz=220.0)
            b_audio, _ = write_mono_tone(b_path, duration=0.5, freq_hz=660.0)
            alpha_path = tmp_path / "alpha_curve.csv"
            alpha_path.write_text(
                "time_sec,value\n"
                "0.0,0.0\n"
                "0.25,0.5\n"
                "0.5,1.0\n",
                encoding="utf-8",
            )
            out_path = tmp_path / "morph_traj.wav"

            cmd = [
                *UNIFIED_CLI,
                "morph",
                str(a_path),
                str(b_path),
                "--alpha",
                str(alpha_path),
                "--interp",
                "linear",
                "--blend-mode",
                "linear",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())

            y_out, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            a_ref = np.asarray(a_audio[: y_out.shape[0]], dtype=np.float64)
            b_ref = np.asarray(b_audio[: y_out.shape[0]], dtype=np.float64)
            out_mono = np.asarray(y_out[:, 0], dtype=np.float64)

            win = max(1, out_mono.shape[0] // 4)
            start_out = out_mono[:win]
            end_out = out_mono[-win:]
            start_a = a_ref[:win]
            start_b = b_ref[:win]
            end_a = a_ref[-win:]
            end_b = b_ref[-win:]

            start_err_a = float(np.mean(np.abs(start_out - start_a)))
            start_err_b = float(np.mean(np.abs(start_out - start_b)))
            end_err_a = float(np.mean(np.abs(end_out - end_a)))
            end_err_b = float(np.mean(np.abs(end_out - end_b)))

            self.assertLess(start_err_a, start_err_b)
            self.assertLess(end_err_b, end_err_a)

    def test_hps_pitch_tracker_emits_control_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "pitch_src.wav"
            write_mono_tone(in_path, duration=0.55, freq_hz=245.0)

            cmd = [
                *HPS_CLI,
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

    def test_hps_pitch_tracker_emit_pitch_to_stretch_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "pitch_glide.wav"
            write_mono_glide(in_path, duration=0.7, f0_start=160.0, f0_end=360.0)

            cmd = [
                *HPS_CLI,
                str(in_path),
                "--backend",
                "acf",
                "--emit",
                "pitch_to_stretch",
                "--stretch-scale",
                "1.0",
                "--stretch-min",
                "0.4",
                "--stretch-max",
                "2.5",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            rows = list(csv.DictReader(io.StringIO(proc.stdout)))
            self.assertGreater(len(rows), 10)
            stretch_values = np.asarray([float(row["stretch"]) for row in rows], dtype=np.float64)
            pitch_values = np.asarray([float(row["pitch_ratio"]) for row in rows], dtype=np.float64)
            self.assertGreater(float(np.std(stretch_values)), 1e-3)
            self.assertTrue(np.allclose(pitch_values, 1.0, atol=1e-9))

    def test_hps_pitch_tracker_emits_feature_vector_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "pitch_features.wav"
            write_mono_glide(in_path, duration=0.5, f0_start=150.0, f0_end=340.0)

            cmd = [
                *HPS_CLI,
                str(in_path),
                "--backend",
                "acf",
                "--feature-set",
                "all",
                "--mfcc-count",
                "8",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            rows = list(csv.DictReader(io.StringIO(proc.stdout)))
            self.assertGreater(len(rows), 8)
            first = rows[0]
            for key in (
                "rms_db",
                "spectral_flux",
                "onset_strength",
                "voicing_prob",
                "pitch_stability",
                "formant_f1_hz",
                "tempo_bpm",
                "speech_prob",
                "music_prob",
                "mpeg7_spectral_centroid_hz",
                "mpeg7_audio_spectrum_envelope_01",
                "mfcc_01",
                "mfcc_08",
            ):
                self.assertIn(key, first)

    def test_cli_pitch_map_stdin_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pitch_src = tmp_path / "a.wav"
            proc_src = tmp_path / "b.wav"
            write_mono_tone(pitch_src, duration=0.65, freq_hz=196.0)
            input_audio, sr = write_stereo_tone(proc_src, duration=0.65)

            track_cmd = [
                *HPS_CLI,
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
                *CLI,
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

    def test_cli_control_route_affine_feature_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            guide_path = tmp_path / "guide_feature.wav"
            target_path = tmp_path / "target_feature.wav"
            write_mono_glide(guide_path, duration=0.65, f0_start=170.0, f0_end=300.0)
            input_audio, sr = write_stereo_tone(target_path, duration=0.65)

            track_cmd = [
                *HPS_CLI,
                str(guide_path),
                "--backend",
                "acf",
                "--feature-set",
                "all",
                "--mfcc-count",
                "6",
                "--quiet",
            ]
            track = subprocess.run(track_cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(track.returncode, 0, msg=track.stderr)
            self.assertIn("mfcc_01", track.stdout)

            out_path = tmp_path / "feature_route_out.wav"
            # `clip(affine(...))` is expressed as chained routes.
            voc_cmd = [
                *CLI,
                str(target_path),
                "--control-stdin",
                "--route",
                "pitch_ratio=affine(mfcc_01,0.002,1.0)",
                "--route",
                "pitch_ratio=clip(pitch_ratio,0.5,2.0)",
                "--route",
                "stretch=affine(rms,2.0,0.8)",
                "--route",
                "stretch=clip(stretch,0.9,1.8)",
                "--pitch-conf-min",
                "0.0",
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
            self.assertGreater(abs(output_audio.shape[0] - input_audio.shape[0]), 24)

    def test_cli_control_stdin_route_pipeline_without_awk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            guide_path = tmp_path / "guide.wav"
            target_path = tmp_path / "target.wav"
            write_mono_glide(guide_path, duration=0.7, f0_start=180.0, f0_end=320.0)
            input_audio, sr = write_stereo_tone(target_path, duration=0.7)

            track_cmd = [
                *HPS_CLI,
                str(guide_path),
                "--backend",
                "acf",
                "--ratio-reference",
                "hz",
                "--reference-hz",
                "440",
                "--quiet",
            ]
            track = subprocess.run(track_cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(track.returncode, 0, msg=track.stderr)
            self.assertIn("pitch_ratio", track.stdout)

            out_path = tmp_path / "route_follow.wav"
            voc_cmd = [
                *CLI,
                str(target_path),
                "--control-stdin",
                "--route",
                "stretch=pitch_ratio",
                "--route",
                "pitch_ratio=const(1.0)",
                "--pitch-conf-min",
                "0.1",
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
            self.assertLess(output_audio.shape[0], int(round(input_audio.shape[0] * 0.9)))

    def test_cli_route_requires_control_map_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "route_err.wav"
            write_mono_tone(in_path, duration=0.3, freq_hz=220.0)
            cmd = [
                *CLI,
                str(in_path),
                "--route",
                "stretch=pitch_ratio",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--route requires", proc.stderr)

    def test_cli_dynamic_stretch_csv_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "dyn_stretch_in.wav"
            input_audio, sr = write_mono_tone(in_path, duration=0.45, freq_hz=240.0)
            map_path = tmp_path / "stretch.csv"
            map_path.write_text(
                "time_sec,value\n"
                "0.0,1.0\n"
                "0.22,1.6\n"
                "0.45,2.0\n",
                encoding="utf-8",
            )
            out_path = tmp_path / "dyn_stretch_out.wav"
            cmd = [
                *CLI,
                str(in_path),
                "--stretch",
                str(map_path),
                "--interp",
                "linear",
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
            self.assertEqual(output_audio.shape[1], 1)
            self.assertGreater(output_audio.shape[0], input_audio.shape[0])

    def test_cli_dynamic_pitch_ratio_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "dyn_pitch_in.wav"
            input_audio, sr = write_mono_tone(in_path, duration=0.45, freq_hz=220.0)
            map_path = tmp_path / "pitch.json"
            payload = {
                "interpolation": "polynomial",
                "order": 3,
                "points": [
                    {"time_sec": 0.0, "value": 1.0},
                    {"time_sec": 0.2, "value": 2 ** (2 / 12)},
                    {"time_sec": 0.45, "value": 2 ** (5 / 12)},
                ],
            }
            map_path.write_text(json.dumps(payload), encoding="utf-8")
            out_path = tmp_path / "dyn_pitch_out.wav"
            cmd = [
                *CLI,
                str(in_path),
                "--time-stretch",
                "1.0",
                "--pitch-shift-ratio",
                str(map_path),
                "--interp",
                "linear",
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
            self.assertEqual(output_audio.shape[1], 1)
            self.assertGreater(output_audio.shape[0], 0)
            self.assertAlmostEqual(output_audio.shape[0], input_audio.shape[0], delta=16)

    def test_cli_dynamic_pitch_ratio_persists_across_validation_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "dyn_pitch_plan_in.wav"
            write_mono_tone(in_path, duration=0.35, freq_hz=220.0)
            map_path = tmp_path / "pitch_plan.json"
            payload = {
                "points": [
                    {"time_sec": 0.0, "value": 1.0},
                    {"time_sec": 0.35, "value": 2 ** (3 / 12)},
                ],
            }
            map_path.write_text(json.dumps(payload), encoding="utf-8")
            cmd = [
                *CLI,
                str(in_path),
                "--pitch-shift-ratio",
                str(map_path),
                "--interp",
                "linear",
                "--explain-plan",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            plan = json.loads(proc.stdout)
            controls = plan.get("io", {}).get("dynamic_controls", [])
            self.assertTrue(any(item.get("parameter") == "pitch_ratio" for item in controls))

    def test_cli_dynamic_nfft_persists_across_validation_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "dyn_nfft_plan_in.wav"
            write_mono_tone(in_path, duration=0.35, freq_hz=220.0)
            map_path = tmp_path / "nfft_plan.csv"
            map_path.write_text(
                "time_sec,value\n"
                "0.0,1024\n"
                "0.35,4096\n",
                encoding="utf-8",
            )
            cmd = [
                *CLI,
                str(in_path),
                "--n-fft",
                str(map_path),
                "--interp",
                "linear",
                "--explain-plan",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            plan = json.loads(proc.stdout)
            controls = plan.get("io", {}).get("dynamic_controls", [])
            self.assertTrue(any(item.get("parameter") == "n_fft" for item in controls))

    def test_cli_dynamic_formant_lifter_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "dyn_formant_in.wav"
            write_mono_tone(in_path, duration=0.4, freq_hz=220.0)
            lifter_path = tmp_path / "lifter.csv"
            lifter_path.write_text(
                "time_sec,value\n"
                "0.0,16\n"
                "0.2,32\n"
                "0.4,64\n",
                encoding="utf-8",
            )
            out_path = tmp_path / "dyn_formant_out.wav"
            cmd = [
                *CLI,
                str(in_path),
                "--pitch-mode",
                "formant-preserving",
                "--pitch-shift-semitones",
                "3",
                "--formant-lifter",
                str(lifter_path),
                "--interp",
                "linear",
                "--output",
                str(out_path),
                "--overwrite",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, 24000)
            self.assertEqual(output_audio.shape[1], 1)
            self.assertGreater(output_audio.shape[0], 0)

    def test_unified_cli_stream_dynamic_stretch_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "stream_dyn.wav"
            input_audio, sr = write_stereo_tone(in_path, duration=0.3)
            stretch_path = tmp_path / "stream_stretch.csv"
            stretch_path.write_text(
                "time_sec,value\n"
                "0.0,1.0\n"
                "0.15,1.6\n"
                "0.3,2.0\n",
                encoding="utf-8",
            )
            out_path = tmp_path / "stream_dyn_out.wav"
            cmd = [
                *UNIFIED_CLI,
                "stream",
                str(in_path),
                "--output",
                str(out_path),
                "--chunk-seconds",
                "0.1",
                "--stretch",
                str(stretch_path),
                "--interp",
                "linear",
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_path.exists())
            output_audio, out_sr = sf.read(out_path, always_2d=True)
            self.assertEqual(out_sr, sr)
            self.assertEqual(output_audio.shape[1], 2)
            self.assertGreater(output_audio.shape[0], input_audio.shape[0])

    def test_cli_dynamic_control_rejects_legacy_pitch_map_combination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "dyn_conflict_in.wav"
            write_mono_tone(in_path, duration=0.3, freq_hz=220.0)
            stretch_path = tmp_path / "stretch.csv"
            stretch_path.write_text("time_sec,value\n0.0,1.0\n0.3,1.4\n", encoding="utf-8")
            pitch_map = tmp_path / "legacy_map.csv"
            pitch_map.write_text("start_sec,end_sec,stretch,pitch_ratio\n0.0,0.3,1.0,1.0\n", encoding="utf-8")
            cmd = [
                *CLI,
                str(in_path),
                "--stretch",
                str(stretch_path),
                "--pitch-map",
                str(pitch_map),
                "--quiet",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Dynamic per-parameter control files cannot be combined", proc.stderr)

    def test_cli_output_policy_sidecar_and_bit_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "policy_in.wav"
            write_stereo_tone(in_path, duration=0.22)
            out_path = tmp_path / "policy_out.wav"

            cmd = [
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
            "--example",
            "basic",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("pvx voc input.wav", proc.stdout)

    def test_unified_stream_help_includes_mode(self) -> None:
        cmd = [*UNIFIED_CLI, "stream", "--help"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--mode {stateful,wrapper}", proc.stdout)

    def test_cli_help_contains_grouped_sections(self) -> None:
        cmd = [*CLI, "--help"]
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
                *CLI,
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
        from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch, resample_1d

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
