# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for the granular_time_stretch algorithm."""

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.algorithms.base import granular_time_stretch

class TestGranularTimeStretch(unittest.TestCase):
    def setUp(self) -> None:
        self.sr = 16000
        # 1 second of audio
        t = np.arange(self.sr) / self.sr
        self.x_mono = np.sin(2 * np.pi * 440 * t)[:, None]
        self.x_stereo = np.stack([
            np.sin(2 * np.pi * 440 * t),
            np.sin(2 * np.pi * 880 * t)
        ], axis=1)

    def test_identity_stretch(self) -> None:
        """Test with stretch=1.0"""
        stretch = 1.0
        grain = 1024
        hop = 256
        y = granular_time_stretch(self.x_mono, stretch=stretch, grain=grain, hop=hop)
        expected_len = int(round(self.x_mono.shape[0] * stretch))
        # Since it uses overlap-add, the output length depends on frames, grain, and hop.
        # It's an approximation.
        self.assertTrue(abs(y.shape[0] - expected_len) <= grain)
        self.assertEqual(y.shape[1], self.x_mono.shape[1])

    def test_time_stretching(self) -> None:
        """Test with stretch > 1.0"""
        stretch = 1.5
        grain = 1024
        hop = 256
        y = granular_time_stretch(self.x_mono, stretch=stretch, grain=grain, hop=hop)
        expected_len = int(round(self.x_mono.shape[0] * stretch))
        self.assertTrue(abs(y.shape[0] - expected_len) <= grain * stretch)
        self.assertEqual(y.shape[1], self.x_mono.shape[1])

    def test_time_compressing(self) -> None:
        """Test with stretch < 1.0"""
        stretch = 0.5
        grain = 1024
        hop = 256
        y = granular_time_stretch(self.x_mono, stretch=stretch, grain=grain, hop=hop)
        expected_len = int(round(self.x_mono.shape[0] * stretch))
        self.assertTrue(abs(y.shape[0] - expected_len) <= grain)
        self.assertEqual(y.shape[1], self.x_mono.shape[1])

    def test_stereo_input(self) -> None:
        """Test with multi-channel (stereo) input"""
        stretch = 1.25
        grain = 1024
        hop = 256
        y = granular_time_stretch(self.x_stereo, stretch=stretch, grain=grain, hop=hop)
        expected_len = int(round(self.x_stereo.shape[0] * stretch))
        self.assertTrue(abs(y.shape[0] - expected_len) <= grain * stretch)
        self.assertEqual(y.shape[1], self.x_stereo.shape[1])

    def test_short_signal_edge_case(self) -> None:
        """Test with signal shorter than the grain size"""
        short_x = self.x_mono[:512, :]
        stretch = 1.0
        grain = 1024
        hop = 256
        # Grain is 1024, signal is 512
        y = granular_time_stretch(short_x, stretch=stretch, grain=grain, hop=hop)
        expected_len = int(round(short_x.shape[0] * stretch))
        self.assertTrue(abs(y.shape[0] - expected_len) <= grain)
        self.assertEqual(y.shape[1], short_x.shape[1])

    def test_zero_or_negative_stretch(self) -> None:
        """Test with extreme stretch values that get clamped to 1e-4"""
        stretch = 0.0
        grain = 1024
        hop = 256
        y = granular_time_stretch(self.x_mono, stretch=stretch, grain=grain, hop=hop)
        # Should be clamped to max(1e-4, stretch)
        expected_len = max(1, int(round(self.x_mono.shape[0] * 1e-4)))
        self.assertTrue(abs(y.shape[0] - expected_len) <= grain)
        self.assertEqual(y.shape[1], self.x_mono.shape[1])

    def test_output_energy(self) -> None:
        """Test that output is not completely silent for identity stretch."""
        stretch = 1.0
        grain = 1024
        hop = 256
        y = granular_time_stretch(self.x_mono, stretch=stretch, grain=grain, hop=hop)
        input_energy = np.sum(self.x_mono ** 2)
        output_energy = np.sum(y ** 2)
        self.assertGreater(output_energy, 0.0)
        # For a sine wave, energy should be roughly preserved
        self.assertTrue(output_energy > input_energy * 0.1)

if __name__ == "__main__":
    unittest.main()
