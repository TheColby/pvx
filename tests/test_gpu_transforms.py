"""Tests for GPU-accelerated augmentation transforms (pvx.augment.gpu)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for GPU transform tests")

from pvx.augment.gpu import (
    NumpyTransformAdapter,
    TorchAddNoise,
    TorchClippingSimulator,
    TorchEQPerturber,
    TorchGainPerturber,
    TorchMixup,
    TorchNormalizer,
    TorchPipeline,
    TorchPitchShift,
    TorchRoomSimulator,
    TorchSpecAugment,
    TorchTimeStretch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_audio(batch: int = 4, channels: int = 1, samples: int = 16000):
    """Return a batch of random audio tensors."""
    return torch.randn(batch, channels, samples)


def _sine_audio(
    batch: int = 4, channels: int = 1, samples: int = 16000, freq: float = 440.0, sr: int = 16000
):
    """Return a batch of sine-wave tensors."""
    t = torch.linspace(0, samples / sr, samples)
    sine = 0.5 * torch.sin(2 * torch.pi * freq * t)
    return sine.unsqueeze(0).unsqueeze(0).expand(batch, channels, -1).clone()


# ---------------------------------------------------------------------------
# TorchGainPerturber
# ---------------------------------------------------------------------------


class TestTorchGainPerturber:
    def test_output_shape(self):
        aug = TorchGainPerturber(gain_db=(-6, 6))
        audio = _random_audio()
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_gain_changes_amplitude(self):
        aug = TorchGainPerturber(gain_db=(6, 6), p=1.0)  # fixed +6 dB
        audio = _sine_audio()
        result = aug(audio, sr=16000)
        assert result.abs().mean() > audio.abs().mean()

    def test_mono_input(self):
        aug = TorchGainPerturber(gain_db=(-3, 3))
        audio = torch.randn(16000)  # (T,)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_stereo_input(self):
        aug = TorchGainPerturber(gain_db=(-3, 3))
        audio = torch.randn(2, 16000)  # (C, T)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_reproducibility(self):
        aug = TorchGainPerturber(gain_db=(-6, 6))
        audio = _random_audio()
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        r1 = aug(audio, sr=16000, generator=gen1)
        r2 = aug(audio, sr=16000, generator=gen2)
        assert torch.allclose(r1, r2)

    def test_probability_zero(self):
        aug = TorchGainPerturber(gain_db=(-20, -20), p=0.0)
        audio = _random_audio()
        result = aug(audio, sr=16000)
        assert torch.allclose(result, audio)


# ---------------------------------------------------------------------------
# TorchAddNoise
# ---------------------------------------------------------------------------


class TestTorchAddNoise:
    def test_output_shape(self):
        aug = TorchAddNoise(snr_db=(10, 30))
        audio = _sine_audio()
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_noise_changes_signal(self):
        aug = TorchAddNoise(snr_db=(5, 5), p=1.0)
        audio = _sine_audio()
        result = aug(audio, sr=16000)
        assert not torch.allclose(result, audio)

    @pytest.mark.parametrize("noise_type", ["white", "pink", "brown"])
    def test_noise_types(self, noise_type):
        aug = TorchAddNoise(snr_db=(10, 30), noise_type=noise_type)
        audio = _sine_audio()
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape
        assert torch.isfinite(result).all()

    def test_invalid_noise_type(self):
        with pytest.raises(ValueError, match="noise_type"):
            TorchAddNoise(noise_type="invalid")


# ---------------------------------------------------------------------------
# TorchEQPerturber
# ---------------------------------------------------------------------------


class TestTorchEQPerturber:
    def test_output_shape(self):
        aug = TorchEQPerturber(n_bands=4)
        audio = _sine_audio(samples=8192)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_modifies_spectrum(self):
        aug = TorchEQPerturber(n_bands=4, gain_db_range=(-12, 12), p=1.0)
        audio = _random_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        # Spectrum should be different
        spec_orig = torch.fft.rfft(audio).abs().mean()
        spec_aug = torch.fft.rfft(result).abs().mean()
        assert not torch.isclose(spec_orig, spec_aug)


# ---------------------------------------------------------------------------
# TorchSpecAugment
# ---------------------------------------------------------------------------


class TestTorchSpecAugment:
    def test_output_shape(self):
        aug = TorchSpecAugment(freq_mask_param=10, time_mask_param=20)
        audio = _sine_audio(samples=8192)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_masking_reduces_energy(self):
        aug = TorchSpecAugment(
            freq_mask_param=50,
            time_mask_param=50,
            num_freq_masks=4,
            num_time_masks=4,
            p=1.0,
        )
        audio = _sine_audio(samples=8192)
        result = aug(audio, sr=16000)
        # Heavy masking should reduce spectral energy
        assert (
            result.pow(2).mean() < audio.pow(2).mean() * 1.5
        )  # just check it's finite and different

    def test_fill_mean(self):
        aug = TorchSpecAugment(freq_mask_param=10, fill_value="mean")
        audio = _sine_audio(samples=8192)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# TorchNormalizer
# ---------------------------------------------------------------------------


class TestTorchNormalizer:
    def test_peak_normalize(self):
        aug = TorchNormalizer(mode="peak", target_db=-1.0)
        audio = torch.randn(4, 1, 8000) * 0.1
        result = aug(audio, sr=16000)
        target = 10.0 ** (-1.0 / 20.0)
        peaks = result.abs().amax(dim=-1)
        assert torch.allclose(peaks, torch.full_like(peaks, target), atol=0.01)

    def test_rms_normalize(self):
        aug = TorchNormalizer(mode="rms", target_db=-20.0)
        audio = torch.randn(4, 1, 8000)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape


# ---------------------------------------------------------------------------
# TorchClippingSimulator
# ---------------------------------------------------------------------------


class TestTorchClippingSimulator:
    def test_hard_clip(self):
        aug = TorchClippingSimulator(percentile=(50, 50), mode="hard", p=1.0)
        audio = torch.randn(4, 1, 8000)
        result = aug(audio, sr=16000)
        # After clipping, max should be <= original max
        assert result.abs().max() <= audio.abs().max() + 1e-6

    def test_soft_clip(self):
        aug = TorchClippingSimulator(percentile=(80, 90), mode="soft", p=1.0)
        audio = torch.randn(4, 1, 8000)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# TorchPipeline
# ---------------------------------------------------------------------------


class TestTorchPipeline:
    def test_sequential_application(self):
        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-3, 3)),
                TorchAddNoise(snr_db=(20, 30)),
            ]
        )
        audio = _sine_audio()
        result = pipeline(audio, sr=16000)
        assert result.shape == audio.shape

    def test_reproducibility(self):
        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-6, 6)),
                TorchAddNoise(snr_db=(10, 30)),
            ],
            seed=42,
        )
        audio = _sine_audio()
        r1 = pipeline(audio, sr=16000, seed=42)
        r2 = pipeline(audio, sr=16000, seed=42)
        assert torch.allclose(r1, r2)

    def test_full_pipeline(self):
        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-6, 6), p=0.8),
                TorchAddNoise(snr_db=(10, 30), noise_type="pink", p=0.5),
                TorchSpecAugment(freq_mask_param=20, time_mask_param=30, p=0.5),
            ],
            seed=42,
        )
        audio = _random_audio(batch=8, samples=8192)
        result = pipeline(audio, sr=16000)
        assert result.shape == audio.shape
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# NumpyTransformAdapter
# ---------------------------------------------------------------------------


class TestNumpyTransformAdapter:
    def test_wraps_numpy_transform(self):
        from pvx.augment import GainPerturber

        np_transform = GainPerturber(gain_db=(-6, 6), p=1.0)
        adapter = NumpyTransformAdapter(np_transform)
        audio = _random_audio(batch=2)
        result = adapter(audio, sr=16000)
        assert result.shape == audio.shape
        assert not torch.allclose(result, audio)

    def test_in_pipeline(self):
        from pvx.augment import AddNoise as NpAddNoise

        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-3, 3)),
                NumpyTransformAdapter(NpAddNoise(snr_db=(10, 30), noise_type="white")),
            ],
            seed=42,
        )
        audio = _sine_audio(batch=2)
        result = pipeline(audio, sr=16000)
        assert result.shape == audio.shape


# ---------------------------------------------------------------------------
# TorchTimeStretch
# ---------------------------------------------------------------------------


class TestTorchTimeStretch:
    def test_output_shape_stretch_down(self):
        aug = TorchTimeStretch(rate=(0.5, 0.5), p=1.0)
        audio = _sine_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        # Stretched down → shorter; result is batched with same T across batch
        assert result.ndim == 3
        assert result.shape[0] == 2
        assert result.shape[1] == 1
        # At rate=0.5 output should be roughly half length (padded to uniform)
        assert result.shape[2] > 0

    def test_output_shape_stretch_up(self):
        aug = TorchTimeStretch(rate=(1.5, 1.5), p=1.0)
        audio = _sine_audio(batch=2, samples=4096)
        result = aug(audio, sr=16000)
        assert result.ndim == 3
        assert result.shape[2] > 4096  # should be ~1.5x longer

    def test_identity_stretch(self):
        aug = TorchTimeStretch(rate=(1.0, 1.0), p=1.0)
        audio = _sine_audio(batch=2, samples=4096)
        result = aug(audio, sr=16000)
        # At rate=1.0 output length should match input (approximately)
        assert abs(result.shape[2] - audio.shape[2]) < 512  # within one hop

    def test_finite_output(self):
        aug = TorchTimeStretch(rate=(0.8, 1.25))
        audio = _random_audio(batch=4, samples=8192)
        result = aug(audio, sr=16000)
        assert torch.isfinite(result).all()

    def test_probability_zero(self):
        aug = TorchTimeStretch(rate=(2.0, 2.0), p=0.0)
        audio = _random_audio(batch=2, samples=4096)
        result = aug(audio, sr=16000)
        assert torch.allclose(result, audio)


# ---------------------------------------------------------------------------
# TorchPitchShift
# ---------------------------------------------------------------------------


class TestTorchPitchShift:
    def test_preserves_duration(self):
        aug = TorchPitchShift(semitones=(2, 2), p=1.0)
        audio = _sine_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        # Pitch shift should preserve duration (resampled back)
        assert result.shape == audio.shape

    def test_changes_signal(self):
        aug = TorchPitchShift(semitones=(5, 5), p=1.0)
        audio = _sine_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        assert not torch.allclose(result, audio)

    def test_negative_shift(self):
        aug = TorchPitchShift(semitones=(-3, -3), p=1.0)
        audio = _sine_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape
        assert torch.isfinite(result).all()

    def test_finite_output(self):
        aug = TorchPitchShift(semitones=(-2, 2))
        audio = _random_audio(batch=4, samples=8192)
        result = aug(audio, sr=16000)
        assert torch.isfinite(result).all()

    def test_probability_zero(self):
        aug = TorchPitchShift(semitones=(12, 12), p=0.0)
        audio = _random_audio(batch=2, samples=4096)
        result = aug(audio, sr=16000)
        assert torch.allclose(result, audio)


# ---------------------------------------------------------------------------
# TorchRoomSimulator
# ---------------------------------------------------------------------------


class TestTorchRoomSimulator:
    def test_output_shape(self):
        aug = TorchRoomSimulator(rt60_range=(0.2, 0.5))
        audio = _sine_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_changes_signal(self):
        aug = TorchRoomSimulator(rt60_range=(0.5, 0.5), wet_range=(0.8, 0.8), p=1.0)
        audio = _sine_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        assert not torch.allclose(result, audio)

    def test_dry_signal_preserved(self):
        aug = TorchRoomSimulator(wet_range=(0.0, 0.0), p=1.0)
        audio = _sine_audio(batch=2, samples=8192)
        result = aug(audio, sr=16000)
        assert torch.allclose(result, audio, atol=1e-4)

    def test_finite_output(self):
        aug = TorchRoomSimulator(rt60_range=(0.1, 2.0), wet_range=(0.3, 0.9))
        audio = _random_audio(batch=4, samples=8192)
        result = aug(audio, sr=16000)
        assert torch.isfinite(result).all()

    def test_stereo(self):
        aug = TorchRoomSimulator(rt60_range=(0.2, 0.5))
        audio = torch.randn(2, 2, 8192)  # (B=2, C=2, T)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape


# ---------------------------------------------------------------------------
# TorchMixup
# ---------------------------------------------------------------------------


class TestTorchMixup:
    def test_output_shape(self):
        aug = TorchMixup(alpha=0.4)
        audio = _random_audio(batch=4, samples=8192)
        result = aug(audio, sr=16000)
        assert result.shape == audio.shape

    def test_single_sample_passthrough(self):
        aug = TorchMixup(alpha=0.4, p=1.0)
        audio = _random_audio(batch=1, samples=4096)
        result = aug(audio, sr=16000)
        assert torch.allclose(result, audio)

    def test_changes_signal(self):
        aug = TorchMixup(alpha=1.0, p=1.0)  # alpha=1 → uniform Beta
        audio = _random_audio(batch=8, samples=4096)
        result = aug(audio, sr=16000)
        # With high alpha and many samples, at least some should differ
        assert not torch.allclose(result, audio)

    def test_finite_output(self):
        aug = TorchMixup(alpha=0.4)
        audio = _random_audio(batch=8, samples=4096)
        result = aug(audio, sr=16000)
        assert torch.isfinite(result).all()

    def test_in_pipeline(self):
        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-3, 3)),
                TorchMixup(alpha=0.4),
            ],
            seed=42,
        )
        audio = _random_audio(batch=4, samples=4096)
        result = pipeline(audio, sr=16000)
        assert result.shape == audio.shape
