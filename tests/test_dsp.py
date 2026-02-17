"""DSP unit tests for core vocoder and analysis primitives.

These tests validate transform length behavior, F0 estimation, transient
handling, formant-preserving correction, Fourier-sync operation, runtime
selection, and support for all registered analysis windows.
"""

import unittest

import numpy as np

from pvxvoc import (
    WINDOW_CHOICES,
    VocoderConfig,
    apply_formant_preservation,
    build_fourier_sync_plan,
    configure_runtime,
    estimate_f0_autocorrelation,
    phase_vocoder_time_stretch,
    phase_vocoder_time_stretch_fourier_sync,
    resample_1d,
    runtime_config,
)


def spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    denom = float(np.sum(spectrum))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(freqs * spectrum) / denom)


class TestPhaseVocoderDSP(unittest.TestCase):
    def setUp(self) -> None:
        configure_runtime("cpu")
        self.cfg_off = VocoderConfig(
            n_fft=1024,
            win_length=1024,
            hop_size=256,
            window="hann",
            center=True,
            phase_locking="off",
            transient_preserve=False,
            transient_threshold=2.0,
        )
        self.cfg_lock = VocoderConfig(
            n_fft=1024,
            win_length=1024,
            hop_size=256,
            window="hann",
            center=True,
            phase_locking="identity",
            transient_preserve=True,
            transient_threshold=1.5,
        )

    def test_time_stretch_length(self) -> None:
        sr = 16000
        t = np.arange(sr) / sr
        x = 0.5 * np.sin(2 * np.pi * 440 * t)
        stretch = 1.75
        y = phase_vocoder_time_stretch(x, stretch, self.cfg_off)
        self.assertEqual(y.size, int(round(x.size * stretch)))

    def test_f0_estimation_for_sine(self) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.8)) / sr
        x = 0.6 * np.sin(2 * np.pi * 220.0 * t)
        f0 = estimate_f0_autocorrelation(x, sr, f0_min_hz=80.0, f0_max_hz=400.0)
        self.assertAlmostEqual(f0, 220.0, delta=1.0)

    def test_transient_phase_locking_improves_attack_concentration(self) -> None:
        sr = 16000
        n = sr
        x = np.zeros(n, dtype=np.float64)
        impulse_idx = 400
        x[impulse_idx] = 1.0
        x += 0.15 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)

        stretch = 1.8
        y_off = phase_vocoder_time_stretch(x, stretch, self.cfg_off)
        y_on = phase_vocoder_time_stretch(x, stretch, self.cfg_lock)

        out_idx = int(round(impulse_idx * stretch))
        half = 80

        def local_energy_ratio(sig: np.ndarray) -> float:
            start = max(0, out_idx - half)
            end = min(sig.size, out_idx + half)
            total = float(np.sum(sig * sig))
            local = float(np.sum(sig[start:end] * sig[start:end]))
            return 0.0 if total <= 1e-12 else local / total

        ratio_off = local_energy_ratio(y_off)
        ratio_on = local_energy_ratio(y_on)
        self.assertGreater(ratio_on, ratio_off * 1.2)

    def test_formant_preserving_mode_keeps_spectral_shape_closer(self) -> None:
        sr = 22050
        t = np.arange(int(sr * 1.0)) / sr
        f0 = 120.0

        x = np.zeros_like(t)
        for k in range(1, 50):
            fk = k * f0
            if fk >= sr / 2:
                break
            env = np.exp(-0.5 * ((fk - 800.0) / 180.0) ** 2) + 0.8 * np.exp(
                -0.5 * ((fk - 1800.0) / 260.0) ** 2
            )
            x += env * np.sin(2 * np.pi * fk * t)
        x /= np.max(np.abs(x))

        ratio = 1.5
        stretched = phase_vocoder_time_stretch(x, ratio, self.cfg_lock)
        shifted = resample_1d(stretched, int(round(stretched.size / ratio)), mode="linear")
        preserved = apply_formant_preservation(
            reference=x,
            shifted=shifted,
            config=self.cfg_lock,
            lifter=28,
            strength=1.0,
            max_gain_db=12.0,
        )

        c_src = spectral_centroid(x, sr)
        c_shift = spectral_centroid(shifted, sr)
        c_pres = spectral_centroid(preserved, sr)

        self.assertLess(abs(c_pres - c_src), abs(c_shift - c_src))

    def test_fourier_sync_non_power_of_two_frame_locking(self) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.9)) / sr
        x = 0.45 * np.sin(2 * np.pi * 180.0 * t) + 0.15 * np.sin(2 * np.pi * 360.0 * t)

        cfg_sync = VocoderConfig(
            n_fft=1500,
            win_length=1500,
            hop_size=375,
            window="hann",
            center=True,
            phase_locking="identity",
            transient_preserve=True,
            transient_threshold=2.0,
        )
        plan = build_fourier_sync_plan(
            signal=x,
            sample_rate=sr,
            config=cfg_sync,
            f0_min_hz=80.0,
            f0_max_hz=400.0,
            min_fft=512,
            max_fft=4096,
            smooth_span=5,
        )
        self.assertGreater(plan.frame_lengths.size, 4)
        self.assertTrue(np.any(plan.frame_lengths != cfg_sync.n_fft))

        y = phase_vocoder_time_stretch_fourier_sync(x, 1.2, cfg_sync, plan)
        self.assertEqual(y.size, int(round(x.size * 1.2)))

    def test_runtime_cpu_configuration(self) -> None:
        cfg = configure_runtime("cpu")
        self.assertEqual(cfg.active_device, "cpu")
        self.assertEqual(runtime_config().active_device, "cpu")

    def test_all_window_types_supported(self) -> None:
        sr = 12000
        t = np.arange(int(sr * 0.25)) / sr
        x = 0.35 * np.sin(2 * np.pi * 330.0 * t)
        stretch = 1.12

        for window in WINDOW_CHOICES:
            cfg = VocoderConfig(
                n_fft=512,
                win_length=512,
                hop_size=128,
                window=window,
                center=True,
                phase_locking="off",
                transient_preserve=False,
                transient_threshold=2.0,
            )
            y = phase_vocoder_time_stretch(x, stretch, cfg)
            self.assertEqual(y.size, int(round(x.size * stretch)))
            self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
