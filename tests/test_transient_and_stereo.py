"""Tests for hybrid transient processing and stereo coherence modes."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.core.transients import detect_transient_regions, smooth_binary_mask
from pvx.metrics.coherence import interchannel_coherence_drift
from pvxvoc import VocoderConfig, configure_runtime, process_audio_block


def _build_args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "stretch_mode": "standard",
        "extreme_stretch_threshold": 2.0,
        "extreme_time_stretch": False,
        "max_stage_stretch": 1.8,
        "multires_fusion": False,
        "_multires_ffts": [2048],
        "_multires_weights": [1.0],
        "fourier_sync": False,
        "analysis_channel": "mix",
        "f0_min": 50.0,
        "f0_max": 1000.0,
        "fourier_sync_min_fft": 256,
        "fourier_sync_max_fft": 8192,
        "fourier_sync_smooth": 5,
        "resample_mode": "linear",
        "pitch_mode": "standard",
        "formant_lifter": 32,
        "formant_strength": 1.0,
        "formant_max_gain_db": 12.0,
        "transient_mode": "off",
        "transient_sensitivity": 0.5,
        "transient_protect_ms": 30.0,
        "transient_crossfade_ms": 10.0,
        "transient_preserve": False,
        "stereo_mode": "independent",
        "ref_channel": 0,
        "coherence_strength": 0.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _phase_drift_internal(stereo: np.ndarray, *, n_fft: int = 1024, hop: int = 256) -> float:
    if stereo.shape[1] < 2:
        return 0.0
    x = np.asarray(stereo[:, 0], dtype=np.float64)
    y = np.asarray(stereo[:, 1], dtype=np.float64)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
        y = np.pad(y, (0, n_fft - y.size))
    rem = (x.size - n_fft) % hop
    if rem:
        x = np.pad(x, (0, hop - rem))
        y = np.pad(y, (0, hop - rem))
    win = np.hanning(n_fft)
    frames = 1 + (x.size - n_fft) // hop
    sx = np.empty((n_fft // 2 + 1, frames), dtype=np.complex128)
    sy = np.empty((n_fft // 2 + 1, frames), dtype=np.complex128)
    for idx in range(frames):
        start = idx * hop
        sx[:, idx] = np.fft.rfft(x[start : start + n_fft] * win)
        sy[:, idx] = np.fft.rfft(y[start : start + n_fft] * win)
    delta = np.angle(sy * np.conj(sx))
    drift = (delta - delta[:, :1] + np.pi) % (2.0 * np.pi) - np.pi
    weight = np.abs(sx) * np.abs(sy)
    return float(np.sum(np.abs(drift) * weight) / (np.sum(weight) + 1e-12))


class TestTransientHybridAndStereo(unittest.TestCase):
    def setUp(self) -> None:
        configure_runtime("cpu")
        self.cfg = VocoderConfig(
            n_fft=1024,
            win_length=1024,
            hop_size=256,
            window="hann",
            center=True,
            phase_locking="identity",
            transient_preserve=True,
            transient_threshold=1.8,
        )

    def test_transient_detection_is_deterministic(self) -> None:
        sr = 16000
        n = int(sr * 0.9)
        t = np.arange(n) / sr
        x = 0.15 * np.sin(2 * np.pi * 210.0 * t)
        x[2400] += 1.0
        x[9200] += 0.8

        _, mask_a, regions_a = detect_transient_regions(
            x,
            sr,
            n_fft=1024,
            hop_size=256,
            sensitivity=0.6,
            protect_ms=30.0,
            crossfade_ms=10.0,
            center=True,
        )
        _, mask_b, regions_b = detect_transient_regions(
            x,
            sr,
            n_fft=1024,
            hop_size=256,
            sensitivity=0.6,
            protect_ms=30.0,
            crossfade_ms=10.0,
            center=True,
        )
        self.assertTrue(np.array_equal(mask_a, mask_b))
        self.assertEqual(len(regions_a), len(regions_b))
        self.assertTrue(mask_a[2400])
        self.assertTrue(mask_a[9200])

    def test_hybrid_mode_identity_stretch_is_near_perfect(self) -> None:
        sr = 22050
        t = np.arange(int(sr * 0.6)) / sr
        x = 0.28 * np.sin(2 * np.pi * 190.0 * t) + 0.14 * np.sin(2 * np.pi * 410.0 * t)
        x[1800] += 0.8
        x[6200] += 0.7
        audio = x[:, None]

        args = _build_args(
            transient_mode="hybrid",
            transient_sensitivity=0.62,
            transient_protect_ms=28.0,
            transient_crossfade_ms=8.0,
            transient_preserve=True,
            stereo_mode="independent",
            coherence_strength=0.0,
        )
        block = process_audio_block(audio, sr, args, self.cfg, stretch=1.0, pitch_ratio=1.0)
        y = np.asarray(block.audio[:, 0], dtype=np.float64)
        y = y[: x.size]
        noise = np.mean((x - y) ** 2) + 1e-12
        signal = np.mean(x * x) + 1e-12
        snr = 10.0 * np.log10(signal / noise)
        self.assertGreater(snr, 24.0)

    def test_hybrid_mode_is_deterministic_in_cpu(self) -> None:
        sr = 22050
        t = np.arange(int(sr * 0.75)) / sr
        x = 0.25 * np.sin(2 * np.pi * 160.0 * t) + 0.11 * np.sin(2 * np.pi * 510.0 * t)
        x[2100] += 0.9
        x[10400] += 0.7
        audio = x[:, None]

        args = _build_args(
            transient_mode="hybrid",
            transient_sensitivity=0.58,
            transient_protect_ms=30.0,
            transient_crossfade_ms=10.0,
            transient_preserve=True,
            stereo_mode="independent",
            coherence_strength=0.0,
        )
        out_a = process_audio_block(audio, sr, args, self.cfg, stretch=1.28, pitch_ratio=1.0).audio
        out_b = process_audio_block(audio, sr, args, self.cfg, stretch=1.28, pitch_ratio=1.0).audio
        self.assertEqual(out_a.shape, out_b.shape)
        self.assertTrue(np.allclose(out_a, out_b, atol=1e-12, rtol=0.0))

    def test_mask_smoothing_creates_crossfade_ramp(self) -> None:
        mask = np.zeros(200, dtype=bool)
        mask[80:120] = True
        smoothed = smooth_binary_mask(mask, 12)
        self.assertGreater(np.max(smoothed), 0.9)
        self.assertLess(np.min(smoothed), 0.1)
        edge = smoothed[74:86]
        self.assertTrue(np.any((edge > 0.0) & (edge < 1.0)))

    def test_ref_channel_lock_reduces_phase_drift(self) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.7)) / sr
        phase_offset = 0.9
        left = 0.35 * np.sin(2 * np.pi * 330.0 * t)
        right = 0.35 * np.sin(2 * np.pi * 330.0 * t + phase_offset)
        stereo = np.stack([left, right], axis=1)

        args_ind = _build_args(
            transient_mode="reset",
            transient_preserve=True,
            stereo_mode="independent",
            coherence_strength=0.0,
        )
        args_lock = _build_args(
            transient_mode="reset",
            transient_preserve=True,
            stereo_mode="ref_channel_lock",
            ref_channel=0,
            coherence_strength=0.9,
        )
        out_ind = process_audio_block(stereo, sr, args_ind, self.cfg, stretch=1.4, pitch_ratio=1.0).audio
        out_lock = process_audio_block(stereo, sr, args_lock, self.cfg, stretch=1.4, pitch_ratio=1.0).audio

        drift_ind = _phase_drift_internal(out_ind)
        drift_lock = _phase_drift_internal(out_lock)
        self.assertLess(drift_lock, drift_ind)

        report = interchannel_coherence_drift(stereo, out_lock[: stereo.shape[0], :], n_fft=1024, hop_size=256)
        self.assertIn("overall_drift_rad", report)

    def test_mid_side_lock_mode_runs_and_preserves_stereo_shape(self) -> None:
        sr = 22050
        t = np.arange(int(sr * 0.5)) / sr
        left = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.08 * np.sin(2 * np.pi * 880.0 * t)
        right = 0.3 * np.sin(2 * np.pi * 220.0 * t + 0.4) + 0.06 * np.sin(2 * np.pi * 880.0 * t + 0.3)
        stereo = np.stack([left, right], axis=1)

        args = _build_args(
            transient_mode="hybrid",
            transient_sensitivity=0.55,
            transient_protect_ms=30.0,
            transient_crossfade_ms=10.0,
            stereo_mode="mid_side_lock",
            coherence_strength=0.85,
        )
        out = process_audio_block(stereo, sr, args, self.cfg, stretch=1.2, pitch_ratio=1.0).audio
        self.assertEqual(out.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
