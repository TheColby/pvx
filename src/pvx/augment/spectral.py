"""Spectral-domain augmentation transforms.

Implements SpecAugment (frequency and time masking), random EQ perturbation,
spectral noise injection, and harmonic distortion — all operating in the
STFT domain with overlap-add resynthesis.
"""

from __future__ import annotations

import numpy as np

from .core import Transform, _from_2d, _to_2d

# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------


def _stft(
    audio: np.ndarray,
    n_fft: int = 1024,
    hop: int = 256,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Return complex STFT with shape ``(n_bins, n_frames)``."""
    if window is None:
        window = np.hanning(n_fft).astype(np.float32)
    len(audio)
    pad = n_fft // 2
    padded = np.pad(audio.astype(np.float32), pad)
    n_frames = 1 + (len(padded) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
        writeable=False,
    ).copy()
    frames *= window
    return np.fft.rfft(frames, n=n_fft, axis=-1).T  # (bins, frames)


def _istft(
    spec: np.ndarray,
    hop: int = 256,
    n_fft: int | None = None,
    window: np.ndarray | None = None,
    length: int | None = None,
) -> np.ndarray:
    """Inverse STFT via overlap-add."""
    n_bins, n_frames = spec.shape
    if n_fft is None:
        n_fft = (n_bins - 1) * 2
    if window is None:
        window = np.hanning(n_fft).astype(np.float32)

    frames = np.fft.irfft(spec.T, n=n_fft, axis=-1)  # (frames, n_fft)
    frames *= window

    pad = n_fft // 2
    out_len = (n_frames - 1) * hop + n_fft
    output = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    win2 = window**2

    for i in range(n_frames):
        start = i * hop
        output[start : start + n_fft] += frames[i]
        norm[start : start + n_fft] += win2

    # Avoid divide-by-zero
    norm = np.where(norm < 1e-8, 1.0, norm)
    output /= norm
    # Remove padding
    output = output[pad : pad + (n_frames * hop)]
    if length is not None:
        if len(output) >= length:
            output = output[:length]
        else:
            output = np.pad(output, (0, length - len(output)))
    return output


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------


class SpecAugment(Transform):
    """SpecAugment: frequency and time masking in the STFT domain.

    Implements the policy from Park et al. (2019), "SpecAugment: A Simple
    Data Augmentation Method for Automatic Speech Recognition."

    Parameters
    ----------
    freq_mask_param:
        Maximum width *F* of each frequency mask in bins.
    time_mask_param:
        Maximum width *T* of each time mask in frames.
    num_freq_masks:
        Number of frequency masks to apply per channel.
    num_time_masks:
        Number of time masks to apply per channel.
    fill_value:
        Value to substitute in masked regions (0 = silence, ``"mean"``
        uses the spectrogram mean).
    n_fft:
        FFT size used internally.
    hop_length:
        Hop size used internally.
    p:
        Probability of applying this transform.

    References
    ----------
    Park, D. S., et al. (2019). SpecAugment. Interspeech.
    https://arxiv.org/abs/1904.08779

    Examples
    --------
    >>> from pvx.augment import SpecAugment
    >>> aug = SpecAugment(freq_mask_param=30, time_mask_param=40, num_freq_masks=2)
    >>> audio_out, sr = aug(audio, sr=16000, seed=0)
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        fill_value: float | str = 0.0,
        n_fft: int = 512,
        hop_length: int = 128,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)
        self.fill_value = fill_value
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape
        out_channels = []

        for ch in range(n_ch):
            sig = arr[ch]
            spec = _stft(sig, n_fft=self.n_fft, hop=self.hop_length)
            n_bins, n_frames = spec.shape

            if isinstance(self.fill_value, str) and self.fill_value == "mean":
                fill = float(np.mean(np.abs(spec)))
            else:
                fill = float(self.fill_value)

            # Frequency masks
            for _ in range(self.num_freq_masks):
                f = int(rng.integers(0, self.freq_mask_param + 1))
                f0 = int(rng.integers(0, max(1, n_bins - f + 1)))
                spec[f0 : f0 + f, :] = fill

            # Time masks
            for _ in range(self.num_time_masks):
                t = int(rng.integers(0, self.time_mask_param + 1))
                t0 = int(rng.integers(0, max(1, n_frames - t + 1)))
                spec[:, t0 : t0 + t] = fill

            reconstructed = _istft(spec, hop=self.hop_length, n_fft=self.n_fft, length=n_samp)
            out_channels.append(reconstructed.astype(arr.dtype))

        result = np.stack(out_channels, axis=0)
        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# EQPerturber
# ---------------------------------------------------------------------------


class EQPerturber(Transform):
    """Randomly perturb audio EQ with parametric peaking filters.

    Applies *n_bands* randomised peak/notch filters to simulate
    room acoustics, microphone colouration, or intentional tonal shaping.

    Parameters
    ----------
    n_bands:
        Number of EQ bands.
    gain_db_range:
        ``(min, max)`` gain per band in dB.
    freq_range:
        ``(min_hz, max_hz)`` centre frequency range.
    q_range:
        ``(min, max)`` quality factor range.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import EQPerturber
    >>> aug = EQPerturber(n_bands=5, gain_db_range=(-8, 8))
    >>> audio_out, sr = aug(audio, sr=44100, seed=3)
    """

    def __init__(
        self,
        n_bands: int = 4,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        freq_range: tuple[float, float] = (80.0, 12000.0),
        q_range: tuple[float, float] = (0.5, 4.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.n_bands = int(n_bands)
        self.gain_db_range = gain_db_range
        self.freq_range = freq_range
        self.q_range = q_range

    @staticmethod
    def _peak_sos(
        fc: float,
        gain_db: float,
        Q: float,
        sr: int,
    ) -> np.ndarray:
        """Second-order peaking EQ filter in SOS form."""
        # Bilinear-transform peaking EQ (Audio EQ Cookbook)
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * fc / sr
        alpha = np.sin(w0) / (2.0 * Q)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A

        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])
        return np.concatenate([[b[0], b[1], b[2], 1.0, a[1], a[2]]]).reshape(1, 6)

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        import numpy as np
        from scipy.signal import sosfiltfilt

        arr, was_mono = _to_2d(audio)
        n_ch, _n_samp = arr.shape
        nyq = sr / 2.0

        out_channels = []
        for ch in range(n_ch):
            sig = arr[ch].copy().astype(np.float64)
            for _ in range(self.n_bands):
                fc = float(rng.uniform(self.freq_range[0], min(self.freq_range[1], nyq * 0.99)))
                gain = float(rng.uniform(self.gain_db_range[0], self.gain_db_range[1]))
                Q = float(rng.uniform(self.q_range[0], self.q_range[1]))
                sos = self._peak_sos(fc, gain, Q, sr)
                sig = sosfiltfilt(sos, sig)
            out_channels.append(sig.astype(arr.dtype))

        result = np.stack(out_channels, axis=0)
        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# SpectralNoise
# ---------------------------------------------------------------------------


class SpectralNoise(Transform):
    """Add noise directly to STFT magnitude bins.

    Unlike waveform noise injection, spectral noise distributes energy more
    uniformly across the frequency axis regardless of signal energy, making
    it useful for training robust spectral feature extractors.

    Parameters
    ----------
    noise_std_range:
        ``(min, max)`` standard deviation of Gaussian noise added to
        magnitude spectrum (as a fraction of the mean magnitude).
    n_fft:
        FFT size used internally.
    hop_length:
        Hop size used internally.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        noise_std_range: tuple[float, float] = (0.01, 0.05),
        n_fft: int = 512,
        hop_length: int = 128,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.noise_std_range = noise_std_range
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape
        std_frac = float(rng.uniform(self.noise_std_range[0], self.noise_std_range[1]))
        out_channels = []
        for ch in range(n_ch):
            spec = _stft(arr[ch], n_fft=self.n_fft, hop=self.hop_length)
            mag = np.abs(spec)
            phase = np.angle(spec)
            mean_mag = float(np.mean(mag))
            noise = rng.standard_normal(mag.shape).astype(np.float32) * std_frac * mean_mag
            mag_noisy = np.maximum(mag + noise, 0.0)
            spec_noisy = mag_noisy * np.exp(1j * phase)
            reconstructed = _istft(spec_noisy, hop=self.hop_length, n_fft=self.n_fft, length=n_samp)
            out_channels.append(reconstructed.astype(arr.dtype))
        result = np.stack(out_channels, axis=0)
        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# PitchShiftSimple
# ---------------------------------------------------------------------------


class PitchShiftSimple(Transform):
    """Lightweight pitch shift via frequency-bin interpolation in the STFT.

    This is a simple phase-vocoder-free approximation suitable for small
    shifts (±1-3 semitones) where computational cost matters more than
    artifact quality.  For production-quality pitch shifting use
    :class:`pvx.augment.time_domain.PitchShift` which calls ``pvx voc``.

    Parameters
    ----------
    semitones:
        Fixed shift or ``(min, max)`` range in semitones.
    n_fft:
        FFT size.
    hop_length:
        Hop size.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        semitones: float | tuple[float, float] = (-2.0, 2.0),
        n_fft: int = 2048,
        hop_length: int = 512,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(semitones, (int, float)):
            self.semitones_range: tuple[float, float] = (float(semitones), float(semitones))
        else:
            self.semitones_range = (float(semitones[0]), float(semitones[1]))
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        from scipy.interpolate import interp1d

        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape
        semitones = float(rng.uniform(self.semitones_range[0], self.semitones_range[1]))
        ratio = 2.0 ** (semitones / 12.0)

        out_channels = []
        for ch in range(n_ch):
            spec = _stft(arr[ch], n_fft=self.n_fft, hop=self.hop_length)
            n_bins = spec.shape[0]
            old_bins = np.arange(n_bins, dtype=float)
            new_bins = old_bins / ratio

            mag = np.abs(spec)
            phase = np.angle(spec)

            # Interpolate magnitude
            interp_mag = interp1d(
                old_bins,
                mag,
                axis=0,
                bounds_error=False,
                fill_value=0.0,
            )(new_bins)
            new_spec = interp_mag * np.exp(1j * phase)
            reconstructed = _istft(new_spec, hop=self.hop_length, n_fft=self.n_fft, length=n_samp)
            out_channels.append(reconstructed.astype(arr.dtype))

        result = np.stack(out_channels, axis=0)
        return _from_2d(result, was_mono), sr
