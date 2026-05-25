"""Noise-based audio augmentation transforms.

Provides additive noise injection at controlled SNR levels, background
audio mixing, and impulse noise (click/pop) simulation.  All transforms
are NumPy-native and require only ``scipy`` as an optional dependency for
spectral-shaped noise generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .core import Transform, _from_2d, _to_2d, load_audio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _linear_to_db(linear: float) -> float:
    return 20.0 * np.log10(max(float(linear), 1e-12))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _scale_to_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Scale *noise* so that it is *snr_db* below *signal* in RMS."""
    sig_rms = _rms(signal)
    noise_rms = _rms(noise)
    target_noise_rms = sig_rms / _db_to_linear(snr_db)
    scale = target_noise_rms / (noise_rms + 1e-12)
    return noise * scale


def _generate_white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _generate_pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink (1/f) noise using the Voss-McCartney algorithm."""
    # Use spectral shaping via FFT
    white = rng.standard_normal(n).astype(np.float64)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    S = 1.0 / np.sqrt(freqs)
    S[0] = 0.0  # zero DC
    spectrum = np.fft.rfft(white) * S
    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
    # Normalize to unit std
    pink /= np.std(pink) + 1e-12
    return pink


def _generate_brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate brown (1/f²) Brownian noise via cumulative sum of white."""
    white = rng.standard_normal(n).astype(np.float64)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    S = 1.0 / freqs
    S[0] = 0.0
    spectrum = np.fft.rfft(white) * S
    brown = np.fft.irfft(spectrum, n=n).astype(np.float32)
    brown -= brown.mean()
    brown /= np.std(brown) + 1e-12
    return brown


def _generate_bandlimited_noise(
    n: int,
    sr: int,
    low_hz: float,
    high_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Band-limited Gaussian noise via spectral masking."""
    from scipy.signal import butter, sosfiltfilt

    white = rng.standard_normal(n).astype(np.float32)
    nyq = sr / 2.0
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 1.0 - 1e-4)
    if lo >= hi:
        return white
    sos = butter(4, [lo, hi], btype="band", output="sos")
    out = sosfiltfilt(sos, white).astype(np.float32)
    out /= np.std(out) + 1e-12
    return out


# ---------------------------------------------------------------------------
# AddNoise
# ---------------------------------------------------------------------------


class AddNoise(Transform):
    """Add synthetic noise to audio at a specified signal-to-noise ratio.

    Parameters
    ----------
    snr_db:
        Target SNR in dB.  Either a fixed float or a ``(min, max)`` range
        from which SNR is sampled uniformly.
    noise_type:
        One of ``"white"``, ``"pink"``, ``"brown"``, ``"gaussian"``
        (alias for white), or ``"bandlimited"``.
    band_hz:
        ``(low_hz, high_hz)`` frequency band used when
        *noise_type* is ``"bandlimited"``.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import AddNoise
    >>> aug = AddNoise(snr_db=(10, 30), noise_type="pink")
    >>> audio_out, sr = aug(audio, sr=16000, seed=42)
    """

    NOISE_TYPES = ("white", "gaussian", "pink", "brown", "bandlimited")

    def __init__(
        self,
        snr_db: float | tuple[float, float] = (15.0, 35.0),
        noise_type: str = "white",
        band_hz: tuple[float, float] = (300.0, 4000.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(snr_db, (int, float)):
            self.snr_db_range: tuple[float, float] = (float(snr_db), float(snr_db))
        else:
            self.snr_db_range = (float(snr_db[0]), float(snr_db[1]))
        if noise_type not in self.NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {self.NOISE_TYPES}, got {noise_type!r}")
        self.noise_type = noise_type
        self.band_hz = band_hz

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        snr_db = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
        n_ch, n_samp = arr.shape

        noisy_channels = []
        for ch in range(n_ch):
            sig = arr[ch]
            if self.noise_type in ("white", "gaussian"):
                noise = _generate_white_noise(n_samp, rng)
            elif self.noise_type == "pink":
                noise = _generate_pink_noise(n_samp, rng)
            elif self.noise_type == "brown":
                noise = _generate_brown_noise(n_samp, rng)
            else:  # bandlimited
                noise = _generate_bandlimited_noise(
                    n_samp, sr, self.band_hz[0], self.band_hz[1], rng
                )
            scaled_noise = _scale_to_snr(sig, noise, snr_db)
            noisy_channels.append((sig + scaled_noise).astype(arr.dtype))

        result = np.stack(noisy_channels, axis=0)
        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# BackgroundMixer
# ---------------------------------------------------------------------------


class BackgroundMixer(Transform):
    """Mix a background audio clip into the signal at a target SNR.

    Background clips are randomly selected from a list of files or a
    directory.  If the background is shorter than the signal it is looped;
    if longer it is randomly cropped.

    Parameters
    ----------
    background_sources:
        List of file paths or a single directory path.
    snr_db:
        Target SNR in dB (fixed float or ``(min, max)`` range).
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import BackgroundMixer
    >>> aug = BackgroundMixer("backgrounds/", snr_db=(5, 20))
    >>> audio_out, sr = aug(audio, sr=16000, seed=0)
    """

    def __init__(
        self,
        background_sources: str | Path | Sequence[str | Path],
        snr_db: float | tuple[float, float] = (10.0, 25.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(background_sources, (str, Path)):
            src = Path(background_sources)
            if src.is_dir():
                exts = {".wav", ".flac", ".aiff", ".ogg", ".mp3", ".caf"}
                self.background_files = [p for p in src.rglob("*") if p.suffix.lower() in exts]
            else:
                self.background_files = [src]
        else:
            self.background_files = [Path(p) for p in background_sources]
        if not self.background_files:
            raise ValueError("No background audio files found")
        if isinstance(snr_db, (int, float)):
            self.snr_db_range: tuple[float, float] = (float(snr_db), float(snr_db))
        else:
            self.snr_db_range = (float(snr_db[0]), float(snr_db[1]))

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape
        snr_db = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))

        bg_path = self.background_files[int(rng.integers(len(self.background_files)))]
        bg, _bg_sr = load_audio(bg_path, target_sr=sr, mono=(n_ch == 1))
        bg2d, _ = _to_2d(bg)

        # Match channel count
        if bg2d.shape[0] < n_ch:
            bg2d = np.tile(bg2d, (int(np.ceil(n_ch / bg2d.shape[0])), 1))[:n_ch]
        elif bg2d.shape[0] > n_ch:
            bg2d = bg2d[:n_ch]

        # Match length: loop or crop
        bg_len = bg2d.shape[1]
        if bg_len < n_samp:
            reps = int(np.ceil(n_samp / bg_len))
            bg2d = np.tile(bg2d, (1, reps))[:, :n_samp]
        else:
            start = int(rng.integers(0, bg_len - n_samp + 1))
            bg2d = bg2d[:, start : start + n_samp]

        scaled_bg = _scale_to_snr(arr, bg2d, snr_db)
        mixed = (arr + scaled_bg).astype(arr.dtype)
        return _from_2d(mixed, was_mono), sr


# ---------------------------------------------------------------------------
# ImpulseNoise  (click / pop simulation)
# ---------------------------------------------------------------------------


class ImpulseNoise(Transform):
    """Insert random click/pop impulses into audio.

    Parameters
    ----------
    rate:
        Expected number of impulses per second.
    amplitude_range:
        ``(min, max)`` amplitude for each impulse (linear).
    duration_samples:
        Duration of each impulse in samples (1 = single-sample click).
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> aug = ImpulseNoise(rate=2.0, amplitude_range=(0.1, 0.5))
    >>> audio_out, sr = aug(audio, sr=44100, seed=7)
    """

    def __init__(
        self,
        rate: float = 2.0,
        amplitude_range: tuple[float, float] = (0.05, 0.3),
        duration_samples: int = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.rate = float(rate)
        self.amplitude_range = amplitude_range
        self.duration_samples = max(1, int(duration_samples))

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape
        result = arr.copy()
        duration_s = n_samp / sr
        n_impulses = int(rng.poisson(self.rate * duration_s))
        for _ in range(n_impulses):
            pos = int(rng.integers(0, max(1, n_samp - self.duration_samples)))
            amp = float(rng.uniform(self.amplitude_range[0], self.amplitude_range[1]))
            sign = rng.choice([-1.0, 1.0])
            ch = int(rng.integers(0, n_ch))
            end = min(pos + self.duration_samples, n_samp)
            result[ch, pos:end] += sign * amp
        result = np.clip(result, -1.0, 1.0).astype(arr.dtype)
        return _from_2d(result, was_mono), sr
