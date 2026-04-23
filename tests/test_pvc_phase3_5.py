# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for PVC-inspired Phase 3-5 core operators."""

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

from pvx.core.pvc_harmony import chord_mapper_mask, process_harmony_operator  # noqa: E402
from pvx.core.pvc_ops import process_response_operator  # noqa: E402
from pvx.core.pvc_resonators import process_ring_operator  # noqa: E402
from pvx.core.response_store import ResponseArtifact  # noqa: E402
from pvx.core.voc import VocoderConfig, configure_runtime  # noqa: E402


class TestPVCPhase3To5(unittest.TestCase):
    def setUp(self) -> None:
        configure_runtime("cpu")
        self.sr = 16000
        t = np.arange(int(self.sr * 0.4), dtype=np.float64) / float(self.sr)
        self.audio = (0.45 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float64)
        self.cfg = VocoderConfig(
            n_fft=512,
            win_length=512,
            hop_size=128,
            window="hann",
            center=True,
            phase_locking="off",
            transient_preserve=False,
            transient_threshold=2.0,
            transform="fft",
        )

    def _flat_response(self, bins: int) -> ResponseArtifact:
        freqs = np.linspace(0.0, self.sr * 0.5, bins, dtype=np.float64)
        mag = np.ones((1, bins), dtype=np.float64)
        pha = np.zeros((1, bins), dtype=np.float64)
        return ResponseArtifact(
            sample_rate=self.sr,
            bins=bins,
            channels=1,
            frequencies_hz=freqs,
            magnitude=mag,
            phase=pha,
            method="median",
            phase_mode="mean",
            normalize="none",
            smoothing_bins=1,
        )

    def test_filter_identity_with_flat_response(self) -> None:
        response = self._flat_response(bins=self.cfg.n_fft // 2 + 1)
        out = process_response_operator(
            self.audio,
            self.sr,
            self.cfg,
            response,
            operator="filter",
            response_mix=1.0,
            dry_mix=0.0,
        )
        self.assertEqual(out.shape[0], self.audio.shape[0])
        self.assertTrue(np.all(np.isfinite(out)))
        err = float(np.mean(np.abs(out[:, 0] - self.audio)))
        self.assertLess(err, 2e-3)

    def test_ringtvfilter_processes_control_map(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-ringtv-") as tmp:
            map_path = Path(tmp) / "ring_map.csv"
            map_path.write_text(
                "time_sec,frequency_hz,depth,mix\n"
                "0.0,20,0.2,0.4\n"
                "0.2,120,1.0,1.0\n",
                encoding="utf-8",
            )
            out = process_ring_operator(
                self.audio,
                self.sr,
                operator="ringtvfilter",
                tv_map_path=map_path,
                resonance_hz=1400.0,
                resonance_q=6.0,
                resonance_mix=0.3,
            )
        self.assertEqual(out.shape[0], self.audio.shape[0])
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(float(np.mean(np.abs(out[:, 0] - self.audio))), 1e-4)

    def test_chordmapper_and_inharmonator_finite(self) -> None:
        mapped = process_harmony_operator(
            self.audio,
            self.sr,
            self.cfg,
            operator="chordmapper",
            root_hz=220.0,
            chord="major",
            strength=0.8,
        )
        warped = process_harmony_operator(
            self.audio,
            self.sr,
            self.cfg,
            operator="inharmonator",
            inharmonic_f0_hz=220.0,
            inharmonicity=2e-4,
            inharmonic_mix=1.0,
        )
        self.assertEqual(mapped.shape[0], self.audio.shape[0])
        self.assertEqual(warped.shape[0], self.audio.shape[0])
        self.assertTrue(np.all(np.isfinite(mapped)))
        self.assertTrue(np.all(np.isfinite(warped)))

    def test_chord_mask_prefers_chord_tones(self) -> None:
        freqs = np.asarray([220.0, 261.63, 277.18], dtype=np.float64)  # A3, C4-ish, C#4-ish
        mask = chord_mapper_mask(freqs, root_hz=220.0, chord="major", tolerance_cents=25.0)
        self.assertGreater(mask[0], 0.9)  # root
        self.assertGreater(mask[2], mask[1])  # major third (C#) closer than non-chord neighbor (C)


if __name__ == "__main__":
    unittest.main()
