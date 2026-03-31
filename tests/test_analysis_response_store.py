# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for PVXAN/PVXRF artifact storage and determinism."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from pvx.core.analysis_store import (  # noqa: E402
    AnalysisArtifact,
    analysis_digest,
    analyze_audio,
    load_analysis_artifact,
    save_analysis_artifact,
)
from pvx.core.response_store import (  # noqa: E402
    ResponseArtifact,
    load_response_artifact,
    response_digest,
    response_from_analysis,
    save_response_artifact,
)
from pvx.core.voc import VocoderConfig, configure_runtime  # noqa: E402


class TestAnalysisResponseStore(unittest.TestCase):
    def setUp(self) -> None:
        configure_runtime("cpu")
        self.config = VocoderConfig(
            n_fft=256,
            win_length=256,
            hop_size=64,
            window="hann",
            center=True,
            phase_locking="off",
            transient_preserve=False,
            transient_threshold=2.0,
            transform="fft",
        )
        sr = 16000
        t = np.arange(int(sr * 0.3), dtype=np.float64) / sr
        self.audio = np.stack(
            [
                0.4 * np.sin(2.0 * np.pi * 220.0 * t),
                0.25 * np.sin(2.0 * np.pi * 330.0 * t + 0.5),
            ],
            axis=1,
        )
        self.sample_rate = sr

    def test_analysis_round_trip_and_digest_stability(self) -> None:
        artifact = analyze_audio(self.audio, self.sample_rate, self.config, source_path="unit.wav")
        digest_before = analysis_digest(artifact)

        with tempfile.TemporaryDirectory(prefix="pvxan-test-") as tmp:
            path = Path(tmp) / "unit.pvxan.npz"
            save_analysis_artifact(path, artifact)
            loaded = load_analysis_artifact(path)

        self.assertIsInstance(loaded, AnalysisArtifact)
        self.assertEqual(loaded.sample_rate, artifact.sample_rate)
        self.assertEqual(loaded.channels, artifact.channels)
        self.assertEqual(loaded.frames, artifact.frames)
        self.assertEqual(loaded.bins, artifact.bins)
        self.assertTrue(np.allclose(loaded.spectrum, artifact.spectrum))
        self.assertEqual(digest_before, analysis_digest(loaded))

    def test_response_round_trip_and_digest_stability(self) -> None:
        analysis = analyze_audio(self.audio, self.sample_rate, self.config, source_path="unit.wav")
        response = response_from_analysis(
            analysis,
            method="median",
            phase_mode="mean",
            normalize="peak",
            smoothing_bins=3,
        )
        digest_before = response_digest(response)

        with tempfile.TemporaryDirectory(prefix="pvxrf-test-") as tmp:
            path = Path(tmp) / "unit.pvxrf.npz"
            save_response_artifact(path, response)
            loaded = load_response_artifact(path)

        self.assertIsInstance(loaded, ResponseArtifact)
        self.assertEqual(loaded.sample_rate, response.sample_rate)
        self.assertEqual(loaded.channels, response.channels)
        self.assertEqual(loaded.bins, response.bins)
        self.assertTrue(np.allclose(loaded.frequencies_hz, response.frequencies_hz))
        self.assertTrue(np.allclose(loaded.magnitude, response.magnitude))
        self.assertTrue(np.allclose(loaded.phase, response.phase))
        self.assertEqual(digest_before, response_digest(loaded))


if __name__ == "__main__":
    unittest.main()
