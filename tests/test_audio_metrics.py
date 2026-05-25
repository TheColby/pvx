"""Tests for shared audio metric table utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.core.audio_metrics import render_audio_metrics_table, summarize_audio_metrics


class TestAudioMetrics(unittest.TestCase):
    def test_summarize_audio_metrics_basic(self) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.2)) / sr
        x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
        m = summarize_audio_metrics(x[:, None], sr)
        self.assertEqual(m.sample_rate, sr)
        self.assertEqual(m.channels, 1)
        self.assertGreater(m.duration_s, 0.19)
        self.assertTrue(np.isfinite(m.peak_dbfs))
        self.assertTrue(np.isfinite(m.rms_dbfs))

    def test_render_table_contains_headers_and_delta(self) -> None:
        sr = 16000
        x = np.linspace(-0.2, 0.2, sr // 10, dtype=np.float64)
        y = np.clip(1.2 * x, -1.0, 1.0)
        table = render_audio_metrics_table(
            [
                ("in:test", summarize_audio_metrics(x[:, None], sr)),
                ("out:test", summarize_audio_metrics(y[:, None], sr)),
            ],
            title="Audio Metrics",
            include_delta_from_first=True,
        )
        self.assertIn("Audio Metrics", table)
        self.assertIn("peak_dbfs", table)
        self.assertIn("delta(last-first)", table)


if __name__ == "__main__":
    unittest.main()
