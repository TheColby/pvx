# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
# ruff: noqa: E402

"""Tests for stereo coherence metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.metrics.coherence import stereo_coherence_drift_score


class TestCoherence(unittest.TestCase):
    def test_stereo_coherence_drift_score_identical(self) -> None:
        sr = 44100
        t = np.arange(sr) / sr
        ref = np.zeros((len(t), 2))
        ref[:, 0] = np.sin(2 * np.pi * 440 * t)
        ref[:, 1] = np.sin(2 * np.pi * 440 * t)  # Both channels identical

        # Candidate is identical to reference
        cand = ref.copy()

        score = stereo_coherence_drift_score(ref, cand)
        self.assertAlmostEqual(score, 0.0, places=7)

    def test_stereo_coherence_drift_score_constant_shift(self) -> None:
        sr = 44100
        t = np.arange(sr) / sr
        ref = np.zeros((len(t), 2))
        ref[:, 0] = np.sin(2 * np.pi * 440 * t)
        ref[:, 1] = np.sin(2 * np.pi * 440 * t)  # In phase

        cand = np.zeros((len(t), 2))
        cand[:, 0] = np.sin(2 * np.pi * 440 * t)
        cand[:, 1] = np.sin(2 * np.pi * 440 * t + np.pi / 2)  # 90 degree shift

        score = stereo_coherence_drift_score(ref, cand)
        # Expected drift is pi/2 ~ 1.57
        # Using a slightly larger tolerance because of windowing effects
        self.assertAlmostEqual(score, np.pi / 2, places=1)

    def test_stereo_coherence_drift_score_mono_input(self) -> None:
        ref = np.zeros((100, 1))
        cand = np.zeros((100, 1))
        score = stereo_coherence_drift_score(ref, cand)
        self.assertEqual(score, 0.0)

        # Also test 1D array if passed (though type hint says ndarray, usually we expect 2D)
        ref_1d = np.zeros(100)
        cand_1d = np.zeros(100)
        score = stereo_coherence_drift_score(ref_1d, cand_1d)
        self.assertEqual(score, 0.0)

    def test_stereo_coherence_drift_score_shape_mismatch(self) -> None:
        rng = np.random.RandomState(42)
        ref = rng.rand(1000, 2)
        cand = rng.rand(500, 2)
        # Should not raise error, but truncate to min length
        try:
            stereo_coherence_drift_score(ref, cand)
        except Exception as e:
            self.fail(f"stereo_coherence_drift_score raised {e} unexpectedly!")

    def test_stereo_coherence_drift_score_silence(self) -> None:
        ref = np.zeros((1000, 2))
        cand = np.zeros((1000, 2))
        score = stereo_coherence_drift_score(ref, cand)
        self.assertEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
