# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

import unittest
import sys
import numpy as np
from pathlib import Path

# Add src to sys.path to allow importing pvx.core.wsola
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pvx.core.wsola import wsola_time_stretch


class TestWSOLA(unittest.TestCase):
    def test_identity_stretch(self):
        """Test that stretch=1.0 returns the input signal unchanged."""
        sr = 44100
        signal = np.random.rand(1000).astype(np.float64)
        result = wsola_time_stretch(signal, 1.0, sr)
        np.testing.assert_array_equal(result, signal)

    def test_empty_input(self):
        """Test that an empty input returns an empty array."""
        sr = 44100
        signal = np.array([], dtype=np.float64)
        result = wsola_time_stretch(signal, 1.5, sr)
        self.assertEqual(result.size, 0)
        np.testing.assert_array_equal(result, signal)

    def test_invalid_stretch(self):
        """Test that stretch <= 0 raises ValueError."""
        sr = 44100
        signal = np.random.rand(100).astype(np.float64)
        with self.assertRaises(ValueError):
            wsola_time_stretch(signal, 0.0, sr)
        with self.assertRaises(ValueError):
            wsola_time_stretch(signal, -0.5, sr)

    def test_short_signal(self):
        """Test that a signal shorter than frame length is interpolated."""
        sr = 44100
        signal = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float64)
        stretch = 2.0

        result = wsola_time_stretch(signal, stretch, sr)
        expected_size = int(round(signal.size * stretch))
        self.assertEqual(result.size, expected_size)

        # Manually verify expected output based on implementation detail (linear interpolation)
        x = signal
        target = expected_size
        expected = np.interp(
            np.linspace(0.0, 1.0, target, endpoint=True),
            np.linspace(0.0, 1.0, x.size, endpoint=True),
            x,
        ).astype(np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_stretch_length(self):
        """Test that stretching a normal signal produces approximately correct length."""
        sr = 44100
        duration = 0.5 # seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 440 * t)

        stretch = 1.5
        result = wsola_time_stretch(signal, stretch, sr)
        expected_len = int(round(signal.size * stretch))

        self.assertEqual(result.size, expected_len)

if __name__ == '__main__':
    unittest.main()
