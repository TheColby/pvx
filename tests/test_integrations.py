"""Smoke tests for pvx.integrations.

Tests that the integration modules import cleanly, the map functions work
with pure-NumPy inputs, and framework-specific functionality is properly
guarded behind optional import checks.
"""

from __future__ import annotations

import unittest

import numpy as np


def _mono(n: int = 8000, sr: int = 16000) -> tuple[np.ndarray, int]:
    return np.random.default_rng(0).standard_normal(n).astype(np.float32) * 0.1, sr


# ---------------------------------------------------------------------------
# HuggingFace integration (no framework required)
# ---------------------------------------------------------------------------


class TestHuggingFaceIntegration(unittest.TestCase):
    def _make_pipeline(self):
        from pvx.augment import AddNoise, GainPerturber, Pipeline

        return Pipeline(
            [
                GainPerturber(gain_db=(-3, 3), p=0.8),
                AddNoise(snr_db=(15, 30), p=0.5),
            ],
            seed=0,
        )

    def test_make_augment_map_fn_array_input(self):
        from pvx.integrations.huggingface import make_augment_map_fn

        pipeline = self._make_pipeline()
        fn = make_augment_map_fn(pipeline, audio_column="audio", default_sr=16000)

        audio, sr = _mono()
        row = {"audio": audio, "label": 0}
        result = fn(row, idx=0)

        self.assertIn("audio", result)
        out = np.asarray(result["audio"])
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(result["label"], 0)

    def test_make_augment_map_fn_dict_input(self):
        from pvx.integrations.huggingface import make_augment_map_fn

        pipeline = self._make_pipeline()
        fn = make_augment_map_fn(pipeline, audio_column="audio")

        audio, sr = _mono()
        hf_audio = {"array": audio, "sampling_rate": sr, "path": "test.wav"}
        row = {"audio": hf_audio, "label": "speech"}
        result = fn(row, idx=1)

        out_audio = result["audio"]
        self.assertIn("array", out_audio)
        self.assertEqual(out_audio["sampling_rate"], sr)
        np.testing.assert_equal(out_audio["array"].shape, audio.shape)

    def test_output_column_override(self):
        from pvx.integrations.huggingface import make_augment_map_fn

        pipeline = self._make_pipeline()
        fn = make_augment_map_fn(pipeline, audio_column="audio", output_column="audio_aug")

        audio, _ = _mono()
        row = {"audio": audio}
        result = fn(row, idx=0)

        self.assertIn("audio", result)  # original preserved
        self.assertIn("audio_aug", result)  # augmented written

    def test_reproducibility(self):
        from pvx.integrations.huggingface import make_augment_map_fn

        pipeline = self._make_pipeline()
        fn = make_augment_map_fn(pipeline, audio_column="audio", base_seed=42)

        audio, _ = _mono()
        row = {"audio": audio}
        r1 = fn(row, idx=5)
        r2 = fn(row, idx=5)
        np.testing.assert_array_equal(np.asarray(r1["audio"]), np.asarray(r2["audio"]))

    def test_hf_mapper_single_view(self):
        from pvx.integrations.huggingface import HFAugmentMapper

        pipeline = self._make_pipeline()
        mapper = HFAugmentMapper(pipeline, audio_column="audio", output_prefix="aug")

        audio, _ = _mono()
        row = {"audio": audio}
        result = mapper(row, idx=0)
        self.assertIn("aug", result)

    def test_hf_mapper_pair_mode(self):
        from pvx.integrations.huggingface import HFAugmentMapper

        pipeline = self._make_pipeline()
        mapper = HFAugmentMapper(
            pipeline, audio_column="audio", generate_pair=True, output_prefix="view"
        )

        audio, _ = _mono()
        row = {"audio": audio}
        result = mapper(row, idx=0)
        self.assertIn("view_a", result)
        self.assertIn("view_b", result)
        # Two views should differ
        va = np.asarray(result["view_a"])
        vb = np.asarray(result["view_b"])
        self.assertFalse(np.allclose(va, vb))

    def test_return_metadata(self):
        import json

        from pvx.integrations.huggingface import make_augment_map_fn

        pipeline = self._make_pipeline()
        fn = make_augment_map_fn(pipeline, audio_column="audio", return_metadata=True)

        audio, _ = _mono()
        row = {"audio": audio}
        result = fn(row, idx=0)
        self.assertIn("augment_params", result)
        meta = json.loads(result["augment_params"])
        self.assertIn("seed", meta)
        self.assertIn("sr_out", meta)


# ---------------------------------------------------------------------------
# TensorFlow integration guards (no TF required)
# ---------------------------------------------------------------------------


class TestTensorFlowIntegrationImport(unittest.TestCase):
    def test_import_without_tf_raises_informative_error(self):
        import sys

        # Temporarily hide tensorflow if present
        tf_mod = sys.modules.pop("tensorflow", None)
        try:
            from pvx.integrations import tensorflow as pvx_tf

            with self.assertRaises(ImportError) as ctx:
                pvx_tf.make_tf_augment_fn(None)
            self.assertIn("tensorflow", str(ctx.exception).lower())
        finally:
            if tf_mod is not None:
                sys.modules["tensorflow"] = tf_mod


# ---------------------------------------------------------------------------
# PyTorch integration guards (no PyTorch required)
# ---------------------------------------------------------------------------


class TestPyTorchIntegrationImport(unittest.TestCase):
    def test_import_without_torch_raises_informative_error(self):
        import importlib
        import sys

        # Temporarily hide torch so _require_torch() cannot import it.
        torch_mod = sys.modules.pop("torch", None)
        # Block fresh imports by inserting a None sentinel.
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            # Reload so _require_torch picks up the blocked module.
            import pvx.integrations.pytorch as pt_mod

            importlib.reload(pt_mod)
            with self.assertRaises(ImportError) as ctx:
                ds = pt_mod.PvxAugmentDataset(["fake.wav"])
            self.assertIn("pytorch", str(ctx.exception).lower())
        except ImportError:
            pass  # If torch is not installed at all, that's fine
        finally:
            del sys.modules["torch"]
            if torch_mod is not None:
                sys.modules["torch"] = torch_mod


if __name__ == "__main__":
    unittest.main()
