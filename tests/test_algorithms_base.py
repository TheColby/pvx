# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for shared DSP algorithms base functions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.algorithms.base import granular_time_stretch


class TestAlgorithmsBase(unittest.TestCase):
    def test_granular_time_stretch_mono(self) -> None:
        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr
        x = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        audio = x[:, None]  # Mono input

        stretch_factor = 1.5
        y = granular_time_stretch(audio, stretch=stretch_factor, grain=2048, hop=512)

        expected_len = int(round(audio.shape[0] * stretch_factor))
        self.assertEqual(y.shape[0], expected_len)
        self.assertEqual(y.shape[1], 1)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_granular_time_stretch_stereo(self) -> None:
        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr
        x1 = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        x2 = 0.5 * np.sin(2 * np.pi * 880.0 * t)
        audio = np.stack([x1, x2], axis=1)  # Stereo input

        stretch_factor = 0.8
        y = granular_time_stretch(audio, stretch=stretch_factor, grain=1024, hop=256)

        expected_len = int(round(audio.shape[0] * stretch_factor))
        self.assertEqual(y.shape[0], expected_len)
        self.assertEqual(y.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_granular_time_stretch_short_audio(self) -> None:
        # Audio length smaller than grain size
        rng = np.random.default_rng(42)
        audio = rng.standard_normal((1000, 1), dtype=np.float64)
        grain_size = 2048
        stretch_factor = 1.2

        y = granular_time_stretch(audio, stretch=stretch_factor, grain=grain_size, hop=512)

        expected_len = int(round(audio.shape[0] * stretch_factor))
        self.assertEqual(y.shape[0], expected_len)
        self.assertEqual(y.shape[1], 1)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_granular_time_stretch_edge_cases(self) -> None:
        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t))[:, None]

        # Extremely small stretch factor (should be clamped to 1e-4 in the function)
        y = granular_time_stretch(audio, stretch=1e-10)
        self.assertGreaterEqual(y.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
