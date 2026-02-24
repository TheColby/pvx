# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
# ruff: noqa: E402

"""Unit tests for core transient analysis functions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Inject src into sys.path to allow imports without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.core.transients import TransientFeatures, compute_transient_features


class TestComputeTransientFeatures(unittest.TestCase):
    def test_compute_transient_features_shapes(self) -> None:
        sr = 16000
        n_fft = 1024
        hop_size = 256
        t = np.arange(sr) / sr
        signal = np.sin(2 * np.pi * 440 * t)

        features = compute_transient_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_size=hop_size,
            center=True,
        )

        self.assertIsInstance(features, TransientFeatures)
        self.assertEqual(features.n_fft, n_fft)
        self.assertEqual(features.hop_size, hop_size)

        frame_count = features.score.shape[0]
        self.assertEqual(features.flux.shape[0], frame_count)
        self.assertEqual(features.hfc.shape[0], frame_count)
        self.assertEqual(features.broadbandness.shape[0], frame_count)
        self.assertEqual(features.frame_times_s.shape[0], frame_count)

    def test_compute_transient_features_empty(self) -> None:
        signal = np.array([], dtype=np.float64)
        sr = 16000
        n_fft = 1024
        hop_size = 256

        features = compute_transient_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_size=hop_size,
        )

        self.assertEqual(features.score.shape[0], 0)
        self.assertEqual(features.flux.shape[0], 0)
        self.assertEqual(features.hfc.shape[0], 0)
        self.assertEqual(features.broadbandness.shape[0], 0)

    def test_compute_transient_features_silence(self) -> None:
        sr = 16000
        signal = np.zeros(sr, dtype=np.float64)
        n_fft = 1024
        hop_size = 256

        features = compute_transient_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_size=hop_size,
        )

        # Expect zero flux and hfc for silence
        self.assertTrue(np.all(features.flux == 0.0))
        self.assertTrue(np.all(features.hfc == 0.0))

        # After robust normalization, if values are constant, they become 0.
        self.assertTrue(np.all(features.score == 0.0))

    def test_compute_transient_features_impulse(self) -> None:
        sr = 16000
        signal = np.zeros(2048, dtype=np.float64)
        impulse_loc = 1024
        signal[impulse_loc] = 1.0
        n_fft = 512
        hop_size = 128

        features = compute_transient_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_size=hop_size,
        )

        # Impulse should create a spike in the score around the impulse location
        expected_frame = impulse_loc // hop_size
        start_frame = max(0, expected_frame - 2)
        end_frame = min(features.score.size, expected_frame + 2)

        max_score_around_impulse = np.max(features.score[start_frame:end_frame])
        self.assertGreater(max_score_around_impulse, 0.5)

    def test_compute_transient_features_contrast(self) -> None:
        # Create a signal with both steady state and a transient.
        # This ensures the normalization doesn't amplify noise in a pure sine wave.
        sr = 16000
        n_samples = 4096
        t = np.arange(n_samples) / sr
        signal = 0.1 * np.sin(2 * np.pi * 440.0 * t)

        # Add a transient
        impulse_loc = 2048
        signal[impulse_loc] = 1.0

        n_fft = 512
        hop_size = 128

        features = compute_transient_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_size=hop_size,
        )

        impulse_frame = impulse_loc // hop_size

        # Score at impulse should be high
        # Score at steady state should be low relative to the impulse

        impulse_score = features.score[impulse_frame]

        # Take a region away from the impulse
        steady_scores = np.concatenate([
            features.score[:impulse_frame-5],
            features.score[impulse_frame+5:]
        ])
        median_steady = np.median(steady_scores)

        self.assertGreater(impulse_score, 0.8)
        self.assertLess(median_steady, 0.3)


if __name__ == "__main__":
    unittest.main()
