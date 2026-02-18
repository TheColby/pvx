"""Unit tests for benchmark metric primitives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.metrics import (  # noqa: E402
    attack_time_error_ms,
    bandwidth_95_delta_hz,
    clipping_ratio_delta,
    crest_factor_delta_db,
    dc_offset_delta,
    envelope_correlation,
    f0_rmse_cents,
    harmonic_to_noise_ratio_drift_db,
    ild_drift_db,
    integrated_lufs_delta_lu,
    interchannel_phase_deviation_by_band,
    itd_drift_ms,
    log_spectral_distance,
    loudness_range_delta_lu,
    modulation_spectrum_distance,
    musical_noise_index,
    onset_precision_recall_f1,
    peaq_odg,
    pesq_mos_lqo,
    phasiness_index,
    polqa_mos_lqo,
    rms_level_delta_db,
    signal_to_noise_ratio_db,
    si_sdr_db,
    short_term_lufs_delta_lu,
    spectral_convergence,
    stereo_coherence_drift,
    stoi_score,
    true_peak_delta_dbtp,
    transient_smear_score,
    visqol_mos_lqo,
    voicing_f1_score,
    zero_crossing_rate_delta,
)


class TestBenchmarkMetrics(unittest.TestCase):
    def test_identity_metrics_are_near_zero(self) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.8)) / sr
        x = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.12 * np.sin(2 * np.pi * 660.0 * t)
        x[3200] += 0.8
        x[9200] += 0.7

        lsd = log_spectral_distance(x, x)
        mod = modulation_spectrum_distance(x, x)
        smear = transient_smear_score(x, x)
        self.assertLess(lsd, 1e-9)
        self.assertLess(mod, 1e-9)
        self.assertLess(smear, 1e-9)
        self.assertGreater(signal_to_noise_ratio_db(x, x), 100.0)
        self.assertGreater(si_sdr_db(x, x), 100.0)
        self.assertLess(spectral_convergence(x, x), 1e-9)
        self.assertGreater(envelope_correlation(x, x), 0.999)
        self.assertAlmostEqual(rms_level_delta_db(x, x), 0.0, places=9)
        self.assertAlmostEqual(crest_factor_delta_db(x, x, sample_rate=sr), 0.0, places=9)
        self.assertAlmostEqual(bandwidth_95_delta_hz(x, x, sample_rate=sr), 0.0, places=6)
        self.assertAlmostEqual(zero_crossing_rate_delta(x, x), 0.0, places=9)
        self.assertAlmostEqual(dc_offset_delta(x, x), 0.0, places=9)
        self.assertAlmostEqual(clipping_ratio_delta(x, x), 0.0, places=9)
        self.assertAlmostEqual(integrated_lufs_delta_lu(x, x, sr), 0.0, places=7)
        self.assertAlmostEqual(short_term_lufs_delta_lu(x, x, sr), 0.0, places=7)
        self.assertAlmostEqual(loudness_range_delta_lu(x, x, sr), 0.0, places=7)
        self.assertAlmostEqual(true_peak_delta_dbtp(x, x, sr), 0.0, places=7)
        self.assertAlmostEqual(f0_rmse_cents(x, x, sr), 0.0, places=7)
        self.assertGreaterEqual(voicing_f1_score(x, x, sr), 0.99)
        self.assertAlmostEqual(harmonic_to_noise_ratio_drift_db(x, x, sr), 0.0, places=6)
        on_p, on_r, on_f1 = onset_precision_recall_f1(x, x, sample_rate=sr)
        self.assertGreaterEqual(on_p, 0.99)
        self.assertGreaterEqual(on_r, 0.99)
        self.assertGreaterEqual(on_f1, 0.99)
        self.assertAlmostEqual(attack_time_error_ms(x, x, sample_rate=sr), 0.0, places=6)
        self.assertLess(phasiness_index(x, x), 1e-9)
        self.assertLess(musical_noise_index(x, x), 1e-9)

        pesq = pesq_mos_lqo(x, x, sample_rate=sr)
        stoi = stoi_score(x, x, sample_rate=sr)
        estoi = stoi_score(x, x, sample_rate=sr, extended=True)
        visqol = visqol_mos_lqo(x, x, sample_rate=sr)
        polqa = polqa_mos_lqo(x, x, sample_rate=sr)
        peaq = peaq_odg(x, x, sample_rate=sr)
        self.assertTrue(np.isfinite(pesq.value))
        self.assertTrue(np.isfinite(stoi.value))
        self.assertTrue(np.isfinite(estoi.value))
        self.assertTrue(np.isfinite(visqol.value))
        self.assertTrue(np.isfinite(polqa.value))
        self.assertTrue(np.isfinite(peaq.value))
        self.assertGreater(pesq.value, 2.0)
        self.assertGreater(stoi.value, 0.6)
        self.assertGreater(estoi.value, 0.6)
        self.assertGreater(visqol.value, 2.0)
        self.assertGreater(polqa.value, 2.0)
        self.assertGreater(peaq.value, -1.0)

    def test_transient_smear_detects_smoothing(self) -> None:
        sr = 22050
        t = np.arange(int(sr * 0.6)) / sr
        x = 0.22 * np.sin(2 * np.pi * 180.0 * t)
        x[2200] += 1.0
        x[7800] += 0.9

        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        y = np.convolve(x, kernel, mode="same")
        smear = transient_smear_score(x, y)
        self.assertGreater(smear, 0.003)

    def test_stereo_coherence_is_zero_for_identical_stereo(self) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.7)) / sr
        left = 0.3 * np.sin(2 * np.pi * 330.0 * t)
        right = 0.3 * np.sin(2 * np.pi * 330.0 * t + 0.75)
        stereo = np.stack([left, right], axis=1)

        drift = stereo_coherence_drift(stereo, stereo)
        self.assertLess(drift, 1e-9)
        self.assertAlmostEqual(ild_drift_db(stereo, stereo), 0.0, places=7)
        self.assertAlmostEqual(itd_drift_ms(stereo, stereo, sample_rate=sr), 0.0, places=7)
        phase = interchannel_phase_deviation_by_band(stereo, stereo, sample_rate=sr)
        self.assertLess(abs(phase["phase_deviation_mean_rad"]), 1e-9)

    def test_metric_functions_handle_length_mismatch(self) -> None:
        x = np.linspace(-0.4, 0.4, 2048, dtype=np.float64)
        y = np.linspace(-0.4, 0.4, 1536, dtype=np.float64)

        self.assertTrue(np.isfinite(log_spectral_distance(x, y)))
        self.assertTrue(np.isfinite(modulation_spectrum_distance(x, y)))
        self.assertTrue(np.isfinite(transient_smear_score(x, y)))
        self.assertTrue(np.isfinite(signal_to_noise_ratio_db(x, y)))
        self.assertTrue(np.isfinite(si_sdr_db(x, y)))
        self.assertTrue(np.isfinite(spectral_convergence(x, y)))
        self.assertTrue(np.isfinite(envelope_correlation(x, y)))
        self.assertTrue(np.isfinite(rms_level_delta_db(x, y)))
        self.assertTrue(np.isfinite(crest_factor_delta_db(x, y)))
        self.assertTrue(np.isfinite(bandwidth_95_delta_hz(x, y)))
        self.assertTrue(np.isfinite(zero_crossing_rate_delta(x, y)))
        self.assertTrue(np.isfinite(dc_offset_delta(x, y)))
        self.assertTrue(np.isfinite(clipping_ratio_delta(x, y)))
        self.assertTrue(np.isfinite(integrated_lufs_delta_lu(x, y, sample_rate=24000)))
        self.assertTrue(np.isfinite(short_term_lufs_delta_lu(x, y, sample_rate=24000)))
        self.assertTrue(np.isfinite(loudness_range_delta_lu(x, y, sample_rate=24000)))
        self.assertTrue(np.isfinite(true_peak_delta_dbtp(x, y, sample_rate=24000)))
        self.assertTrue(np.isfinite(f0_rmse_cents(x, y, sample_rate=24000)))
        self.assertTrue(np.isfinite(voicing_f1_score(x, y, sample_rate=24000)))
        self.assertTrue(np.isfinite(harmonic_to_noise_ratio_drift_db(x, y, sample_rate=24000)))
        p, r, f1 = onset_precision_recall_f1(x, y, sample_rate=24000)
        self.assertTrue(np.isfinite(p))
        self.assertTrue(np.isfinite(r))
        self.assertTrue(np.isfinite(f1))
        self.assertTrue(np.isfinite(attack_time_error_ms(x, y, sample_rate=24000)))
        self.assertTrue(np.isfinite(phasiness_index(x, y)))
        self.assertTrue(np.isfinite(musical_noise_index(x, y)))
        self.assertTrue(np.isfinite(pesq_mos_lqo(x, y, sample_rate=24000).value))
        self.assertTrue(np.isfinite(stoi_score(x, y, sample_rate=24000).value))
        self.assertTrue(np.isfinite(stoi_score(x, y, sample_rate=24000, extended=True).value))
        self.assertTrue(np.isfinite(visqol_mos_lqo(x, y, sample_rate=24000).value))
        self.assertTrue(np.isfinite(polqa_mos_lqo(x, y, sample_rate=24000).value))
        self.assertTrue(np.isfinite(peaq_odg(x, y, sample_rate=24000).value))


if __name__ == "__main__":
    unittest.main()
