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
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "pvxvoc.py"
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
