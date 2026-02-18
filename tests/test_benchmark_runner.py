"""Tests for benchmark runner profile selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.run_bench import TaskSpec, _pvx_bench_args


class TestBenchmarkRunnerProfiles(unittest.TestCase):
    def _write_wav(self, path: Path, channels: int) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.1)) / sr
        tone = 0.2 * np.sin(2 * np.pi * 220.0 * t)
        if channels == 1:
            audio = tone[:, None]
        else:
            audio = np.stack([tone, np.roll(tone, 3)], axis=1)
        sf.write(path, audio, sr)

    def test_tuned_profile_mono_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mono.wav"
            self._write_wav(path, channels=1)
            args = _pvx_bench_args(path, TaskSpec("stretch", "stretch", 1.8), tuned=True)
            self.assertIn("--transient-mode", args)
            self.assertIn("off", args)
            self.assertIn("--n-fft", args)
            self.assertIn("1024", args)
            self.assertNotIn("--stereo-mode", args)

    def test_tuned_profile_stereo_pitch_adds_stereo_and_formant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            self._write_wav(path, channels=2)
            args = _pvx_bench_args(path, TaskSpec("pitch", "pitch", 4.0), tuned=True)
            self.assertIn("--stereo-mode", args)
            self.assertIn("mid_side_lock", args)
            self.assertIn("--coherence-strength", args)
            self.assertIn("--pitch-mode", args)
            self.assertIn("formant-preserving", args)

    def test_legacy_profile_keeps_hybrid_reference_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mono.wav"
            self._write_wav(path, channels=1)
            args = _pvx_bench_args(path, TaskSpec("stretch", "stretch", 1.8), tuned=False)
            self.assertIn("--transient-mode", args)
            self.assertIn("hybrid", args)
            self.assertIn("--stereo-mode", args)
            self.assertIn("ref_channel_lock", args)


if __name__ == "__main__":
    unittest.main()
