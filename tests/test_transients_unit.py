# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
# ruff: noqa: E402

"""Unit tests for transient detection and segmentation logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.core.transients import TransientFeatures, pick_onset_frames


class TestPickOnsetFrames(unittest.TestCase):
    def _make_features(self, score: np.ndarray) -> TransientFeatures:
        n = score.size
        zeros = np.zeros(n, dtype=np.float64)
        return TransientFeatures(
            flux=zeros,
            hfc=zeros,
            broadbandness=zeros,
            score=score,
            frame_times_s=zeros,
            hop_size=256,
            n_fft=1024,
        )

    def test_empty_score_returns_empty(self) -> None:
        features = self._make_features(np.array([], dtype=np.float64))
        onsets = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=5)
        self.assertEqual(onsets.size, 0)

    def test_small_score_returns_argmax(self) -> None:
        # Case size=1
        features = self._make_features(np.array([0.5], dtype=np.float64))
        onsets = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=5)
        self.assertEqual(onsets.tolist(), [0])

        # Case size=2
        features = self._make_features(np.array([0.2, 0.8], dtype=np.float64))
        onsets = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=5)
        self.assertEqual(onsets.tolist(), [1])

    def test_basic_peak_picking(self) -> None:
        # Create a score array with clear peaks
        score = np.zeros(100, dtype=np.float64)
        score[20] = 0.9  # Peak 1
        score[60] = 0.8  # Peak 2
        # Add some noise below threshold (0.05 < 0.08 min threshold)
        score[40] = 0.05

        features = self._make_features(score)

        onsets = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=10)
        self.assertEqual(sorted(onsets.tolist()), [20, 60])

    def test_sensitivity_affects_threshold(self) -> None:
        # Use a linear ramp to ensure a predictable distribution of scores
        # Linspace from 0.1 to 0.5. q=0.92 ~ 0.47. q=0.42 ~ 0.27.
        score = np.linspace(0.1, 0.5, 100)

        # Add peaks (ramp itself has no local maxima)
        # 10: 0.9
        # 30: 0.8
        # 50: 0.6
        # 70: 0.4
        peaks = [(10, 0.9), (30, 0.8), (50, 0.6), (70, 0.4)]
        for idx, val in peaks:
            score[idx] = val
            # Neighbors on a ramp are lower/higher, but we set peaks much higher than local ramp value
            # e.g., at 70, ramp is ~0.38. Peak is 0.4.
            # score[71] is ~0.384. 0.4 > 0.384. Local max holds.

        features = self._make_features(score)

        # High sensitivity (s=1.0) -> threshold ~ 0.27
        # Low sensitivity (s=0.0) -> threshold ~ 0.47

        onsets_high = pick_onset_frames(features, sensitivity=1.0, min_separation_frames=5)
        onsets_low = pick_onset_frames(features, sensitivity=0.0, min_separation_frames=5)

        # Peak at 70 (0.4) is > 0.27 but < 0.47
        self.assertIn(70, onsets_high)
        self.assertNotIn(70, onsets_low)

        # Check other peaks
        self.assertIn(10, onsets_low)
        self.assertIn(30, onsets_low)
        self.assertIn(50, onsets_low)

        self.assertGreater(onsets_high.size, onsets_low.size)

    def test_separation_logic(self) -> None:
        score = np.zeros(100, dtype=np.float64)
        # Two peaks close to each other
        score[20] = 0.8
        score[22] = 0.9 # Higher, should replace 20 if within separation

        features = self._make_features(score)

        # Separation > 2
        onsets = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=5)
        self.assertEqual(onsets.tolist(), [22])

        # Separation < 2 -> both picked
        onsets_close = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=1)
        self.assertEqual(sorted(onsets_close.tolist()), [20, 22])

    def test_separation_logic_keep_first_if_higher(self) -> None:
        score = np.zeros(100, dtype=np.float64)
        # First peak is higher
        score[20] = 0.9
        score[22] = 0.8

        features = self._make_features(score)
        onsets = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=5)
        self.assertEqual(onsets.tolist(), [20])

    def test_no_candidates(self) -> None:
        score = np.zeros(100, dtype=np.float64) + 0.05
        features = self._make_features(score)
        onsets = pick_onset_frames(features, sensitivity=0.5, min_separation_frames=5)
        self.assertEqual(onsets.size, 0)

if __name__ == "__main__":
    unittest.main()
