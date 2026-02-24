# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
# ruff: noqa: E402

"""Tests for feature tracking extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.core.feature_tracking import (
    as_serializable_columns,
    extract_feature_tracks,
    feature_subset,
)


class TestFeatureTracking(unittest.TestCase):
    def setUp(self) -> None:
        self.sr = 24000
        self.frame_length = 1024
        self.hop_size = 256
        self.fmin = 50.0
        self.fmax = 1200.0

    def test_extract_features_silence(self) -> None:
        n_frames = 10
        n_samples = n_frames * self.hop_size
        audio = np.zeros(n_samples, dtype=np.float64)
        f0_hz = np.zeros(n_frames, dtype=np.float64)
        confidence = np.zeros(n_frames, dtype=np.float64)

        features = extract_feature_tracks(
            audio,
            self.sr,
            frame_length=self.frame_length,
            hop_size=self.hop_size,
            f0_hz=f0_hz,
            confidence=confidence,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        self.assertTrue(np.allclose(features["rms"], 0.0, atol=1e-9))
        self.assertTrue(np.allclose(features["zcr"], 0.0, atol=1e-9))
        self.assertTrue(np.allclose(features["spectral_centroid_hz"], 0.0, atol=1e-9))
        self.assertEqual(len(features["rms"]), n_frames)

    def test_extract_features_sine_wave(self) -> None:
        n_frames = 20
        n_samples = n_frames * self.hop_size + self.frame_length
        t = np.arange(n_samples) / self.sr
        freq = 440.0
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        f0_hz = np.full(n_frames, freq, dtype=np.float64)
        confidence = np.full(n_frames, 1.0, dtype=np.float64)

        features = extract_feature_tracks(
            audio,
            self.sr,
            frame_length=self.frame_length,
            hop_size=self.hop_size,
            f0_hz=f0_hz,
            confidence=confidence,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # Centroid of a pure sine should be close to its frequency
        # Allowing some tolerance due to windowing and finite resolution
        centroid = features["spectral_centroid_hz"]
        # Filter out frames where signal hasn't fully started or ended
        valid_frames = centroid[5:-5]
        self.assertTrue(np.allclose(valid_frames, freq, rtol=0.1))

        # RMS should be approx 0.5 / sqrt(2) * window_factor
        # Window is Hanning, RMS of Hanning window is sqrt(3/8) approx 0.612
        # RMS of windowed sine = (A/sqrt(2)) * RMS_window
        # Wait, the implementation calculates RMS on raw frame before windowing?
        # Let's check implementation:
        # fr = _frame(mono, start, n_fft)
        # rms_i = float(np.sqrt(np.mean(fr * fr)))
        # So it's raw frame RMS.
        # RMS of sine amplitude 0.5 is 0.5 / sqrt(2) ~= 0.3535
        rms = features["rms"][5:-5]
        self.assertTrue(np.allclose(rms, 0.5 / np.sqrt(2), rtol=0.05))

    def test_extract_features_stereo(self) -> None:
        n_frames = 10
        n_samples = n_frames * self.hop_size + self.frame_length
        # Left channel louder
        left = np.ones(n_samples)
        right = 0.5 * np.ones(n_samples)
        audio = np.stack([left, right], axis=1)

        f0_hz = np.zeros(n_frames, dtype=np.float64)
        confidence = np.zeros(n_frames, dtype=np.float64)

        features = extract_feature_tracks(
            audio,
            self.sr,
            frame_length=self.frame_length,
            hop_size=self.hop_size,
            f0_hz=f0_hz,
            confidence=confidence,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # ILD dB = 20 * log10(L_rms / R_rms)
        # L_rms = 1.0, R_rms = 0.5. Ratio = 2.0. 20 * log10(2) ~= 6.02 dB
        ild = features["ild_db"][2:-2]
        self.assertTrue(np.allclose(ild, 20 * np.log10(2.0), atol=0.1))

    def test_feature_subset(self) -> None:
        dummy_features = {
            "rms": np.array([1.0]),
            "rms_db": np.array([0.0]),
            "zcr": np.array([0.1]),
            "spectral_centroid_hz": np.array([100.0]),
            "extra_feature": np.array([99.9]),
        }

        none_subset = feature_subset(dummy_features, subset="none")
        self.assertEqual(len(none_subset), 0)

        basic_subset = feature_subset(dummy_features, subset="basic")
        self.assertIn("rms", basic_subset)
        self.assertNotIn("extra_feature", basic_subset)

        all_subset = feature_subset(dummy_features, subset="all")
        self.assertEqual(len(all_subset), len(dummy_features))
        self.assertIn("extra_feature", all_subset)

    def test_as_serializable_columns(self) -> None:
        n_rows = 5
        features = {
            "short": np.zeros(3),
            "exact": np.zeros(5),
            "long": np.zeros(8),
            "empty": np.array([]),
        }

        cols = as_serializable_columns(features, n_rows=n_rows)

        for k, v in cols.items():
            self.assertEqual(v.size, n_rows, msg=f"Key {k} has wrong size")

        # Check padding
        # "short" should be padded with last value (0.0)
        self.assertEqual(cols["short"].size, 5)

        # "long" should be truncated
        self.assertEqual(cols["long"].size, 5)

        # "empty" should be zeros
        self.assertEqual(cols["empty"].size, 5)
        self.assertTrue(np.all(cols["empty"] == 0.0))

    def test_extract_features_short_audio(self) -> None:
        # Audio shorter than frame length
        n_samples = self.frame_length // 2
        audio = np.random.randn(n_samples)
        f0_hz = np.array([440.0]) # Request 1 frame
        confidence = np.array([1.0])

        features = extract_feature_tracks(
            audio,
            self.sr,
            frame_length=self.frame_length,
            hop_size=self.hop_size,
            f0_hz=f0_hz,
            confidence=confidence,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        self.assertEqual(len(features["rms"]), 1)
        # Should not crash

    def test_extract_features_empty_audio(self) -> None:
        audio = np.array([])
        f0_hz = np.array([]) # 0 frames
        confidence = np.array([])

        features = extract_feature_tracks(
            audio,
            self.sr,
            frame_length=self.frame_length,
            hop_size=self.hop_size,
            f0_hz=f0_hz,
            confidence=confidence,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        self.assertEqual(len(features), 0)


if __name__ == "__main__":
    unittest.main()
