# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Tests for the spectral_gate algorithm in pvx.algorithms.base."""

import unittest

import numpy as np

from pvx.algorithms.base import spectral_gate


class TestSpectralGate(unittest.TestCase):
    def test_spectral_gate_noise_reduction(self) -> None:
        sr = 22050
        t = np.arange(sr * 2) / sr

        # Part 1: Signal with noise
        s_part = 0.8 * np.sin(2 * np.pi * 440 * t[:sr])
        n_part_1 = 0.05 * np.random.randn(sr)
        part1 = s_part + n_part_1

        # Part 2: Pure noise
        n_part_2 = 0.05 * np.random.randn(sr)
        part2 = n_part_2

        # Concatenate and reshape to (samples, channels)
        x = np.concatenate([part1, part2]).reshape(-1, 1)

        y = spectral_gate(x, strength=2.0, floor=0.01)

        y_part1 = y[:sr, 0]
        y_part2 = y[sr:, 0]

        noise_energy_in = np.sum(part2**2)
        noise_energy_out = np.sum(y_part2**2)

        signal_energy_in = np.sum(part1**2)
        signal_energy_out = np.sum(y_part1**2)

        # The pure noise part should be significantly attenuated
        self.assertLess(noise_energy_out, noise_energy_in * 0.1)
        # The signal part should be mostly preserved
        self.assertGreater(signal_energy_out, signal_energy_in * 0.5)

    def test_spectral_gate_preserves_shape_and_multichannel(self) -> None:
        sr = 22050
        t = np.arange(sr) / sr

        # Create a 2-channel signal
        ch1 = 0.8 * np.sin(2 * np.pi * 440 * t)
        ch2 = 0.5 * np.sin(2 * np.pi * 880 * t)

        x = np.stack([ch1, ch2], axis=1)
        self.assertEqual(x.shape, (sr, 2))

        y = spectral_gate(x, strength=0.1, floor=0.1)

        # Output shape should match input shape
        self.assertEqual(y.shape, x.shape)

        # Both channels should have signal remaining
        self.assertGreater(np.max(np.abs(y[:, 0])), 0.5)
        self.assertGreater(np.max(np.abs(y[:, 1])), 0.3)

    def test_spectral_gate_pure_silence(self) -> None:
        # Check that it behaves gracefully with pure silence
        x = np.zeros((44100, 1), dtype=np.float64)
        y = spectral_gate(x, strength=1.2, floor=0.05)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.abs(y) < 1e-10))


if __name__ == "__main__":
    unittest.main()
