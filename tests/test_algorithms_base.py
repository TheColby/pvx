# ruff: noqa: E402
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unittest
from unittest.mock import patch

import numpy as np

from pvx.algorithms.base import estimate_f0_track


class TestAlgorithmsBase(unittest.TestCase):
    def test_estimate_f0_track_librosa(self) -> None:
        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr
        freq = 440.0
        audio = np.sin(2.0 * np.pi * freq * t)

        f0 = estimate_f0_track(audio, sr, fmin=50.0, fmax=1200.0, hop=256)

        valid_f0 = f0[f0 > 0]
        self.assertGreater(len(valid_f0), 0)
        self.assertAlmostEqual(float(np.median(valid_f0)), freq, delta=5.0)

    @patch("pvx.algorithms.base.maybe_librosa", return_value=None)
    def test_estimate_f0_track_fallback(self, mock_maybe_librosa) -> None:
        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr
        freq = 440.0
        audio = np.sin(2.0 * np.pi * freq * t)

        f0 = estimate_f0_track(audio, sr, fmin=50.0, fmax=1200.0, hop=256)

        mock_maybe_librosa.assert_called_once()
        valid_f0 = f0[f0 > 0]
        self.assertGreater(len(valid_f0), 0)
        self.assertAlmostEqual(float(np.median(valid_f0)), freq, delta=15.0)

    def test_estimate_f0_track_silence(self) -> None:
        sr = 16000
        audio = np.zeros(sr, dtype=np.float64)
        f0 = estimate_f0_track(audio, sr, fmin=50.0, fmax=1200.0, hop=256)
        self.assertTrue(np.all(f0 == 0.0))

    @patch("pvx.algorithms.base.maybe_librosa", return_value=None)
    def test_estimate_f0_track_silence_fallback(self, mock_maybe_librosa) -> None:
        sr = 16000
        audio = np.zeros(sr, dtype=np.float64)
        f0 = estimate_f0_track(audio, sr, fmin=50.0, fmax=1200.0, hop=256)
        # Since it short-circuits on silence before maybe_librosa() is called,
        # it is expected to not be called.
        mock_maybe_librosa.assert_not_called()
        self.assertTrue(np.all(f0 == 0.0))


if __name__ == "__main__":
    unittest.main()
