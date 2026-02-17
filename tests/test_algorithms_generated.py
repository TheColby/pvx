"""Regression smoke tests for all generated pvx algorithm modules.

This test verifies that every algorithm listed in `pvxalgorithms.registry`
is importable and can process a synthetic stereo signal while returning
finite 2D output and implemented metadata status.
"""

import importlib
import unittest

import numpy as np

from pvxalgorithms.registry import ALGORITHM_REGISTRY


class TestGeneratedAlgorithms(unittest.TestCase):
    def test_every_algorithm_runs(self) -> None:
        sr = 16000
        n = 4096
        t = np.arange(n, dtype=np.float64) / sr
        audio = np.stack(
            [
                0.30 * np.sin(2.0 * np.pi * 220.0 * t) + 0.03 * np.sin(2.0 * np.pi * 55.0 * t),
                0.28 * np.sin(2.0 * np.pi * 330.0 * t),
            ],
            axis=1,
        )

        for algorithm_id, entry in ALGORITHM_REGISTRY.items():
            module = importlib.import_module(entry["module"])
            result = module.process(audio, sr)
            self.assertEqual(result.sample_rate, sr, msg=algorithm_id)
            self.assertEqual(result.audio.ndim, 2, msg=algorithm_id)
            self.assertEqual(result.metadata.get("status"), "implemented", msg=algorithm_id)
            self.assertTrue(np.all(np.isfinite(result.audio)), msg=algorithm_id)


if __name__ == "__main__":
    unittest.main()
