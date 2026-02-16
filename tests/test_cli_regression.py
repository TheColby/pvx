import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "pvocode.py"


def write_stereo_tone(path: Path, sr: int = 24000, duration: float = 0.5) -> tuple[np.ndarray, int]:
    t = np.arange(int(sr * duration)) / sr
    left = 0.35 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.30 * np.sin(2 * np.pi * 330.0 * t)
    audio = np.stack([left, right], axis=1)
    sf.write(path, audio, sr)
    return audio, sr


class TestCLIRegression(unittest.TestCase):
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

    def test_regression_metrics_snapshot(self) -> None:
        from pvocode import VocoderConfig, phase_vocoder_time_stretch, resample_1d

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
