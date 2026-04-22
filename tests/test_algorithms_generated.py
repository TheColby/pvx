"""Regression smoke tests for generated pvx algorithm wrappers.

This test verifies that every algorithm listed in `pvx.algorithms.registry`
is importable and can process a synthetic stereo signal while returning
finite 2D output and honest dispatch metadata.
"""

import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(ROOT / "src"))

from pvx.algorithms.base import run_algorithm
from pvx.algorithms.registry import ALGORITHM_REGISTRY


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
            self.assertEqual(
                result.metadata.get("implementation_style"),
                "shared_dispatch",
                msg=algorithm_id,
            )
            self.assertFalse(bool(result.metadata.get("is_fallback")), msg=algorithm_id)
            self.assertTrue(np.all(np.isfinite(result.audio)), msg=algorithm_id)

    def test_unknown_algorithm_is_not_reported_as_implemented(self) -> None:
        audio = np.zeros((32, 2), dtype=np.float64)
        result = run_algorithm(
            algorithm_id="unknown_theme.made_up_algorithm",
            algorithm_name="Made up algorithm",
            theme="Unknown Theme",
            audio=audio,
            sample_rate=16000,
            params={},
        )

        self.assertEqual(result.metadata.get("status"), "unsupported")
        self.assertEqual(result.metadata.get("fallback_reason"), "unknown_algorithm_id")
        self.assertEqual(result.metadata.get("implementation_style"), "shared_dispatch")
        self.assertTrue(bool(result.metadata.get("is_fallback")))

    def test_legacy_registry_alias_points_to_canonical_registry(self) -> None:
        legacy_registry = importlib.import_module("pvxalgorithms.registry")
        self.assertIs(legacy_registry.ALGORITHM_REGISTRY, ALGORITHM_REGISTRY)


if __name__ == "__main__":
    unittest.main()
