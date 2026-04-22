"""Unit tests for pvx.augment transform library.

Tests cover: shape preservation, reproducibility, per-transform correctness,
pipeline composition, and combinators.
"""

from __future__ import annotations

import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mono(n: int = 16000, sr: int = 16000) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(0)
    return rng.standard_normal(n).astype(np.float32) * 0.1, sr


def _stereo(n: int = 16000, sr: int = 16000) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(1)
    return rng.standard_normal((2, n)).astype(np.float32) * 0.1, sr


def _sine(freq: float = 440.0, sr: int = 16000, duration: float = 1.0) -> tuple[np.ndarray, int]:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t) * 0.5, sr


# ---------------------------------------------------------------------------
# Core / Pipeline
# ---------------------------------------------------------------------------


class TestPipelineCore(unittest.TestCase):
    def test_identity_passthrough(self):
        from pvx.augment import Identity

        audio, sr = _mono()
        aug = Identity()
        out, sr_out = aug(audio, sr, seed=0)
        np.testing.assert_array_equal(out, audio)
        self.assertEqual(sr_out, sr)

    def test_pipeline_empty(self):
        from pvx.augment import Pipeline

        audio, sr = _mono()
        p = Pipeline([])
        out, _sr_out = p(audio, sr, seed=0)
        np.testing.assert_array_equal(out, audio)

    def test_pipeline_reproducibility(self):
        from pvx.augment import AddNoise, Pipeline

        audio, sr = _mono()
        p = Pipeline([AddNoise(snr_db=20)], seed=42)
        out1, _ = p(audio, sr, seed=42)
        out2, _ = p(audio, sr, seed=42)
        np.testing.assert_array_equal(out1, out2)

    def test_pipeline_different_seeds(self):
        from pvx.augment import AddNoise, Pipeline

        audio, sr = _mono()
        p = Pipeline([AddNoise(snr_db=20)])
        out1, _ = p(audio, sr, seed=0)
        out2, _ = p(audio, sr, seed=1)
        self.assertFalse(np.allclose(out1, out2))

    def test_pipeline_p_zero_passthrough(self):
        from pvx.augment import AddNoise, Pipeline

        audio, sr = _mono()
        p = Pipeline([AddNoise(snr_db=20)], p=0.0)
        out, _ = p(audio, sr, seed=0)
        np.testing.assert_array_equal(out, audio)

    def test_one_of_selects_one(self):
        from pvx.augment import AddNoise, Identity, OneOf

        audio, sr = _mono()
        aug = OneOf([AddNoise(snr_db=20), Identity()])
        out, sr_out = aug(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(sr_out, sr)

    def test_some_of(self):
        from pvx.augment import AddNoise, GainPerturber, Identity, SomeOf

        audio, sr = _mono()
        aug = SomeOf([AddNoise(snr_db=20), GainPerturber(gain_db=3), Identity()], k=2)
        out, _ = aug(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)

    def test_random_apply_p_zero(self):
        from pvx.augment import AddNoise, RandomApply

        audio, sr = _mono()
        aug = RandomApply(AddNoise(snr_db=5), p=0.0)
        out, _ = aug(audio, sr, seed=0)
        np.testing.assert_array_equal(out, audio)


# ---------------------------------------------------------------------------
# Noise transforms
# ---------------------------------------------------------------------------


class TestAddNoise(unittest.TestCase):
    def test_shape_preserved_mono(self):
        from pvx.augment import AddNoise

        audio, sr = _mono()
        out, sr_out = AddNoise(snr_db=20)(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(sr_out, sr)

    def test_shape_preserved_stereo(self):
        from pvx.augment import AddNoise

        audio, sr = _stereo()
        out, _sr_out = AddNoise(snr_db=20)(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)

    def test_noise_types(self):
        from pvx.augment import AddNoise

        audio, sr = _mono(n=8000)
        for ntype in ("white", "pink", "brown", "gaussian", "bandlimited"):
            with self.subTest(noise_type=ntype):
                out, _ = AddNoise(snr_db=20, noise_type=ntype)(audio, sr, seed=0)
                self.assertEqual(out.shape, audio.shape)
                self.assertFalse(np.allclose(out, audio))

    def test_snr_range(self):
        from pvx.augment import AddNoise

        audio, sr = _sine()
        # High SNR should be close to original
        out_high, _ = AddNoise(snr_db=(40, 40))(audio, sr, seed=0)
        # Low SNR should differ significantly
        out_low, _ = AddNoise(snr_db=(0, 0))(audio, sr, seed=0)
        diff_high = float(np.mean((out_high - audio) ** 2))
        diff_low = float(np.mean((out_low - audio) ** 2))
        self.assertGreater(diff_low, diff_high)

    def test_invalid_noise_type(self):
        from pvx.augment import AddNoise

        with self.assertRaises(ValueError):
            AddNoise(noise_type="laser_beam")


class TestImpulseNoise(unittest.TestCase):
    def test_shape_preserved(self):
        from pvx.augment import ImpulseNoise

        audio, sr = _mono()
        out, _ = ImpulseNoise(rate=5.0)(audio, sr, seed=7)
        self.assertEqual(out.shape, audio.shape)

    def test_clipping_range(self):
        from pvx.augment import ImpulseNoise

        audio, sr = _mono()
        out, _ = ImpulseNoise(rate=20.0, amplitude_range=(0.8, 0.9))(audio, sr, seed=0)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)


# ---------------------------------------------------------------------------
# Room / RIR
# ---------------------------------------------------------------------------


class TestRoomSimulator(unittest.TestCase):
    def test_shape_preserved(self):
        from pvx.augment import RoomSimulator

        audio, sr = _mono(n=8000)
        out, sr_out = RoomSimulator(rt60_range=(0.2, 0.5))(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(sr_out, sr)

    def test_stereo_shape_preserved(self):
        from pvx.augment import RoomSimulator

        audio, sr = _stereo(n=8000)
        out, _ = RoomSimulator(rt60_range=(0.2, 0.5))(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)

    def test_output_differs_from_input(self):
        from pvx.augment import RoomSimulator

        audio, sr = _mono(n=8000)
        out, _ = RoomSimulator(rt60_range=(0.5, 0.5), wet_range=(0.9, 0.9))(audio, sr, seed=0)
        self.assertFalse(np.allclose(out, audio))

    def test_wet_zero_is_passthrough(self):
        from pvx.augment import RoomSimulator

        audio, sr = _mono(n=4096)
        out, _ = RoomSimulator(rt60_range=(0.5, 0.5), wet_range=(0.0, 0.0))(audio, sr, seed=0)
        np.testing.assert_allclose(out, audio, atol=1e-5)


# ---------------------------------------------------------------------------
# Codec / degradation
# ---------------------------------------------------------------------------


class TestCodecDegradation(unittest.TestCase):
    def test_all_presets_run(self):
        from pvx.augment import CodecDegradation

        audio, sr = _mono(n=8000, sr=16000)
        for codec in (
            "mp3_low",
            "mp3_medium",
            "telephone",
            "voip_narrow",
            "voip_wide",
            "am_radio",
            "lo_fi",
        ):
            with self.subTest(codec=codec):
                out, _ = CodecDegradation(codec=codec)(audio, sr, seed=0)
                self.assertEqual(out.shape, audio.shape)

    def test_random_codec_runs(self):
        from pvx.augment import CodecDegradation

        audio, sr = _mono(n=8000)
        out, _ = CodecDegradation(codec="random")(audio, sr, seed=42)
        self.assertEqual(out.shape, audio.shape)


class TestBitCrusher(unittest.TestCase):
    def test_8bit_reduces_resolution(self):
        from pvx.augment import BitCrusher

        audio, sr = _sine()
        out, _ = BitCrusher(bits=(8, 8))(audio, sr, seed=0)
        # Values should be quantized
        unique = len(np.unique(np.round(out * 256).astype(int)))
        self.assertLessEqual(unique, 256)

    def test_shape_preserved(self):
        from pvx.augment import BitCrusher

        audio, sr = _stereo(n=4096)
        out, _ = BitCrusher(bits=(8, 12))(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------


class TestSpecAugment(unittest.TestCase):
    def test_shape_preserved_mono(self):
        from pvx.augment import SpecAugment

        audio, sr = _mono()
        out, sr_out = SpecAugment(freq_mask_param=20, time_mask_param=30)(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(sr_out, sr)

    def test_shape_preserved_stereo(self):
        from pvx.augment import SpecAugment

        audio, sr = _stereo()
        out, _ = SpecAugment()(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)

    def test_reproducibility(self):
        from pvx.augment import SpecAugment

        audio, sr = _mono()
        aug = SpecAugment(freq_mask_param=15, time_mask_param=20)
        out1, _ = aug(audio, sr, seed=99)
        out2, _ = aug(audio, sr, seed=99)
        np.testing.assert_array_equal(out1, out2)

    def test_masking_reduces_energy(self):
        from pvx.augment import SpecAugment

        audio, sr = _sine()
        # Fill with zero should reduce RMS
        out, _ = SpecAugment(
            freq_mask_param=100,
            time_mask_param=200,
            num_freq_masks=5,
            num_time_masks=5,
            fill_value=0.0,
        )(audio, sr, seed=0)
        rms_in = float(np.sqrt(np.mean(audio**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        self.assertLess(rms_out, rms_in)


class TestEQPerturber(unittest.TestCase):
    def test_shape_preserved(self):
        from pvx.augment import EQPerturber

        audio, sr = _sine(sr=44100)
        out, sr_out = EQPerturber(n_bands=4)(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(sr_out, sr)

    def test_zero_gain_passthrough(self):
        from pvx.augment import EQPerturber

        audio, sr = _mono()
        out, _ = EQPerturber(n_bands=2, gain_db_range=(0.0, 0.0))(audio, sr, seed=0)
        np.testing.assert_allclose(out, audio, atol=1e-4)


# ---------------------------------------------------------------------------
# Time domain
# ---------------------------------------------------------------------------


class TestGainPerturber(unittest.TestCase):
    def test_positive_gain_louder(self):
        from pvx.augment import GainPerturber

        audio, sr = _mono()
        out, _ = GainPerturber(gain_db=(6.0, 6.0))(audio, sr, seed=0)
        self.assertGreater(float(np.max(np.abs(out))), float(np.max(np.abs(audio))))

    def test_negative_gain_quieter(self):
        from pvx.augment import GainPerturber

        audio, sr = _mono()
        out, _ = GainPerturber(gain_db=(-6.0, -6.0))(audio, sr, seed=0)
        self.assertLess(float(np.max(np.abs(out))), float(np.max(np.abs(audio))))


class TestNormalizer(unittest.TestCase):
    def test_peak_normalization(self):
        from pvx.augment import Normalizer

        audio, sr = _mono()
        out, _ = Normalizer(mode="peak", target_db=-3.0)(audio, sr, seed=0)
        peak_out = float(np.max(np.abs(out)))
        target = 10 ** (-3.0 / 20.0)
        self.assertAlmostEqual(peak_out, target, places=4)

    def test_rms_normalization(self):
        from pvx.augment import Normalizer

        audio, sr = _sine()
        out, _ = Normalizer(mode="rms", target_db=-20.0)(audio, sr, seed=0)
        rms_out = float(np.sqrt(np.mean(out**2)))
        target = 10 ** (-20.0 / 20.0)
        self.assertAlmostEqual(rms_out, target, places=3)


class TestClippingSimulator(unittest.TestCase):
    def test_output_bounded(self):
        from pvx.augment import ClippingSimulator

        audio = np.ones(1000, dtype=np.float32) * 0.9
        out, _ = ClippingSimulator(percentile=(50.0, 50.0))(audio, 16000, seed=0)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_soft_mode(self):
        from pvx.augment import ClippingSimulator

        audio, sr = _sine()
        out, _ = ClippingSimulator(percentile=(90.0, 90.0), mode="soft")(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)


class TestTimeShift(unittest.TestCase):
    def test_shape_preserved(self):
        from pvx.augment import TimeShift

        audio, sr = _mono()
        out, _ = TimeShift(shift=(-0.1, 0.1))(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)

    def test_wrap_mode(self):
        from pvx.augment import TimeShift

        audio, sr = _mono(n=1000)
        out, _ = TimeShift(shift=(0.05, 0.05), mode="wrap")(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)


class TestReverse(unittest.TestCase):
    def test_double_reverse_is_identity(self):
        from pvx.augment import Reverse

        audio, sr = _mono()
        aug = Reverse(p=1.0)
        out, _ = aug(audio, sr, seed=0)
        out2, _ = aug(out, sr, seed=0)
        np.testing.assert_array_equal(out2, audio)


class TestFade(unittest.TestCase):
    def test_fade_in_first_sample_near_zero(self):
        from pvx.augment import Fade

        audio = np.ones(16000, dtype=np.float32)
        aug = Fade(fade_in=(0.5, 0.5), fade_out=0.0)
        out, _ = aug(audio, 16000, seed=0)
        self.assertAlmostEqual(float(out[0]), 0.0, places=3)

    def test_fade_out_last_sample_near_zero(self):
        from pvx.augment import Fade

        audio = np.ones(16000, dtype=np.float32)
        aug = Fade(fade_in=0.0, fade_out=(0.5, 0.5))
        out, _ = aug(audio, 16000, seed=0)
        self.assertAlmostEqual(float(out[-1]), 0.0, places=3)


class TestFixedLengthCrop(unittest.TestCase):
    def test_crop_longer(self):
        from pvx.augment import FixedLengthCrop

        audio, sr = _mono(n=32000)
        out, _ = FixedLengthCrop(duration_s=0.5)(audio, sr, seed=0)
        self.assertEqual(out.shape[-1], sr // 2)

    def test_pad_shorter(self):
        from pvx.augment import FixedLengthCrop

        audio, sr = _mono(n=4000)
        out, _ = FixedLengthCrop(duration_s=1.0)(audio, sr, seed=0)
        self.assertEqual(out.shape[-1], sr)


# ---------------------------------------------------------------------------
# Intent preset factories
# ---------------------------------------------------------------------------


class TestPresetFactories(unittest.TestCase):
    def test_asr_pipeline_runs(self):
        from pvx.augment import asr_pipeline

        audio, sr = _mono(n=8000)
        pipeline = asr_pipeline(seed=0)
        out, sr_out = pipeline(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)
        self.assertEqual(sr_out, sr)

    def test_music_pipeline_runs(self):
        from pvx.augment import music_pipeline

        audio, sr = _mono(n=8000)
        pipeline = music_pipeline(seed=0)
        out, _ = pipeline(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)

    def test_speech_enhancement_pipeline_runs(self):
        from pvx.augment import speech_enhancement_pipeline

        audio, sr = _mono(n=8000)
        pipeline = speech_enhancement_pipeline(seed=0)
        out, _ = pipeline(audio, sr, seed=0)
        self.assertEqual(out.shape, audio.shape)

    def test_contrastive_pipeline_returns_two(self):
        from pvx.augment import contrastive_pipeline

        pipelines = contrastive_pipeline(seed=0)
        self.assertEqual(len(pipelines), 2)
        audio, sr = _mono(n=8000)
        out_a, _ = pipelines[0](audio, sr, seed=0)
        out_b, _ = pipelines[1](audio, sr, seed=1)
        # Both should differ from input and from each other
        self.assertFalse(np.allclose(out_a, audio))
        self.assertFalse(np.allclose(out_b, audio))


# ---------------------------------------------------------------------------
# Engine selection for TimeStretch / PitchShift (requires PyTorch)
# ---------------------------------------------------------------------------


class TestEngineSelection(unittest.TestCase):
    """Test the engine='auto'/'pytorch'/'pvx-cli' parameter."""

    def setUp(self):
        try:
            import torch  # noqa: F401

            self._has_torch = True
        except ImportError:
            self._has_torch = False

    @unittest.skipUnless(True, "always runs — tests resolve logic")
    def test_resolve_engine_auto_returns_string(self):
        from pvx.augment.time_domain import _resolve_engine

        result = _resolve_engine("auto")
        self.assertIn(result, ("pytorch", "pvx-cli"))

    @unittest.skipUnless(True, "always runs")
    def test_resolve_engine_pvx_cli(self):
        from pvx.augment.time_domain import _resolve_engine

        self.assertEqual(_resolve_engine("pvx-cli"), "pvx-cli")

    def test_time_stretch_accepts_engine_param(self):
        from pvx.augment import TimeStretch

        ts = TimeStretch(rate=(0.9, 1.1), engine="auto")
        self.assertEqual(ts.engine, "auto")

    def test_pitch_shift_accepts_engine_param(self):
        from pvx.augment import PitchShift

        ps = PitchShift(semitones=(-1, 1), engine="auto")
        self.assertEqual(ps.engine, "auto")

    def test_time_stretch_pytorch_engine(self):
        if not self._has_torch:
            self.skipTest("PyTorch not installed")
        from pvx.augment import TimeStretch

        ts = TimeStretch(rate=(1.1, 1.1), engine="pytorch", p=1.0)
        audio, sr = _sine(freq=440.0, sr=16000, duration=0.5)
        out, out_sr = ts(audio, sr, seed=42)
        self.assertEqual(out_sr, sr)
        # Stretched by 1.1 → ~10% longer
        self.assertGreater(len(out), len(audio) * 1.05)

    def test_time_stretch_pytorch_length_accuracy(self):
        """Regression: _torch_phase_vocoder must honor target length across stretch factors."""
        if not self._has_torch:
            self.skipTest("PyTorch not installed")
        from pvx.augment import TimeStretch

        audio, sr = _sine(freq=440.0, sr=16000, duration=0.5)
        input_len = len(audio)
        # Test a wide range of stretch factors including the one that triggered the bug
        for rate in (0.5, 0.75, 0.9, 1.05, 1.1, 1.25, 1.5, 2.0):
            ts = TimeStretch(rate=(rate, rate), engine="pytorch", p=1.0)
            out, _ = ts(audio, sr, seed=42)
            expected = int(round(input_len * rate))
            # Allow a small window around expected (STFT framing rounding)
            tolerance = max(int(0.02 * expected), 256)
            self.assertGreaterEqual(
                len(out),
                expected - tolerance,
                f"rate={rate}: got {len(out)}, expected ~{expected}",
            )
            self.assertLessEqual(
                len(out),
                expected + tolerance,
                f"rate={rate}: got {len(out)}, expected ~{expected}",
            )

    def test_pitch_shift_pytorch_engine(self):
        if not self._has_torch:
            self.skipTest("PyTorch not installed")
        from pvx.augment import PitchShift

        ps = PitchShift(semitones=(2, 2), engine="pytorch", p=1.0)
        audio, sr = _sine(freq=440.0, sr=16000, duration=0.5)
        out, out_sr = ps(audio, sr, seed=42)
        self.assertEqual(out_sr, sr)
        # Duration should be preserved (approximately)
        self.assertAlmostEqual(len(out) / sr, len(audio) / sr, delta=0.05)

    def test_pytorch_engine_raises_without_torch(self):
        """engine='pytorch' should raise RuntimeError when torch is missing."""
        from pvx.augment.time_domain import _resolve_engine

        if self._has_torch:
            # Can only truly test this without torch; just verify it doesn't crash
            result = _resolve_engine("pytorch")
            self.assertEqual(result, "pytorch")
        else:
            with self.assertRaises(RuntimeError):
                _resolve_engine("pytorch")


class TestPluginSystem(unittest.TestCase):
    """Test the transform plugin registry."""

    def test_register_and_get_transform(self):
        from pvx.augment.core import (
            Transform,
            get_transform,
            list_transforms,
            register_transform,
        )

        @register_transform("test_custom_gain")
        class TestCustomGain(Transform):
            def apply(self, audio, sr, rng):
                return audio * 0.5, sr

        # Should be findable
        cls = get_transform("test_custom_gain")
        self.assertIs(cls, TestCustomGain)

        # Should appear in listing
        registry = list_transforms()
        self.assertIn("test_custom_gain", registry)

    def test_get_unknown_transform_raises(self):
        from pvx.augment.core import get_transform

        with self.assertRaises(KeyError):
            get_transform("nonexistent_transform_xyz")

    def test_register_non_transform_raises(self):
        from pvx.augment.core import register_transform

        with self.assertRaises(TypeError):

            @register_transform("bad")
            class NotATransform:
                pass

    def test_auto_snake_case_name(self):
        from pvx.augment.core import (
            Transform,
            get_transform,
            register_transform,
        )

        @register_transform()
        class MyCustomReverb(Transform):
            def apply(self, audio, sr, rng):
                return audio, sr

        cls = get_transform("my_custom_reverb")
        self.assertIs(cls, MyCustomReverb)

    def test_registered_transform_works_in_pipeline(self):
        from pvx.augment import Pipeline
        from pvx.augment.core import (
            Transform,
            get_transform,
            register_transform,
        )

        @register_transform("test_double")
        class DoubleAudio(Transform):
            def apply(self, audio, sr, rng):
                return audio * 2.0, sr

        cls = get_transform("test_double")
        pipeline = Pipeline([cls(p=1.0)], seed=42)
        audio, sr = _mono()
        out, _out_sr = pipeline(audio, sr)
        np.testing.assert_allclose(out, audio * 2.0, atol=1e-7)


class TestPipelineConfig(unittest.TestCase):
    """Test YAML/JSON manifest-driven pipeline construction."""

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, content):
        from pathlib import Path

        p = Path(self.tmpdir) / name
        p.write_text(content)
        return p

    def test_load_pipeline_from_json(self):
        from pvx.augment import Pipeline, load_pipeline

        manifest = """
        {
          "seed": 42,
          "p": 1.0,
          "transforms": [
            {"name": "gain_perturber", "params": {"gain_db": [-3, 3], "p": 0.9}},
            {"name": "add_noise", "params": {"snr_db": [20, 30], "p": 0.5}}
          ]
        }
        """
        path = self._write("pipeline.json", manifest)
        pipeline = load_pipeline(path)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline), 2)
        audio, sr = _mono()
        out, _ = pipeline(audio, sr, seed=42)
        self.assertEqual(out.shape, audio.shape)

    def test_load_pipeline_from_yaml_if_available(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        from pvx.augment import load_pipeline

        manifest = (
            "seed: 7\n"
            "transforms:\n"
            "  - name: gain_perturber\n"
            "    params:\n"
            "      gain_db: [-3, 3]\n"
            "      p: 0.5\n"
        )
        path = self._write("pipeline.yaml", manifest)
        pipeline = load_pipeline(path)
        self.assertEqual(len(pipeline), 1)

    def test_load_pipeline_uses_plugin_registry(self):
        from pvx.augment import load_pipeline
        from pvx.augment.core import Transform, register_transform

        @register_transform("config_test_passthrough")
        class ConfigTestPassthrough(Transform):
            def apply(self, audio, sr, rng):
                return audio, sr

        manifest = '{"transforms": [{"name": "config_test_passthrough", "params": {"p": 1.0}}]}'
        path = self._write("pipeline.json", manifest)
        pipeline = load_pipeline(path)
        self.assertEqual(len(pipeline), 1)
        self.assertIsInstance(pipeline.transforms[0], ConfigTestPassthrough)

    def test_load_pipeline_unknown_transform_raises(self):
        from pvx.augment import load_pipeline

        path = self._write(
            "pipeline.json",
            '{"transforms": [{"name": "totally_made_up_xyz"}]}',
        )
        with self.assertRaises(KeyError):
            load_pipeline(path)

    def test_load_pipeline_missing_file(self):
        from pvx.augment import load_pipeline

        with self.assertRaises(FileNotFoundError):
            load_pipeline("/tmp/does_not_exist_zzzz.yaml")

    def test_load_pipeline_empty_transforms_list(self):
        from pvx.augment import load_pipeline

        path = self._write("pipeline.json", '{"transforms": []}')
        with self.assertRaises(ValueError):
            load_pipeline(path)

    def test_recipes_asr_robust_loads(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        from pathlib import Path

        from pvx.augment import load_pipeline

        recipe = Path(__file__).resolve().parents[1] / "recipes" / "asr_robust.yaml"
        if not recipe.exists():
            self.skipTest("recipes/asr_robust.yaml not present")
        pipeline = load_pipeline(recipe)
        self.assertGreater(len(pipeline), 0)
        audio, sr = _mono()
        out, _ = pipeline(audio, sr, seed=42)
        self.assertEqual(out.shape, audio.shape)


if __name__ == "__main__":
    unittest.main()
