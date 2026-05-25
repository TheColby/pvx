"""Room acoustics augmentation transforms.

Provides synthetic room impulse response (RIR) approximation via an
exponentially-decaying noise model and convolution with arbitrary
user-supplied impulse response files.

The synthetic RIR generator is a lightweight statistical approximation
suitable for data augmentation — it does *not* perform physical room
modelling (image-source or ray-tracing).  For physics-based simulation,
use external tools such as ``pyroomacoustics`` or ``gpuRIR`` and load
the resulting IR files with :class:`ImpulseResponseConvolver`.

No external neural-network dependencies required.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .core import Transform, _from_2d, _to_2d, load_audio

# ---------------------------------------------------------------------------
# Synthetic RIR generation
# ---------------------------------------------------------------------------


def _generate_synthetic_rir(
    rt60_s: float,
    sr: int,
    rng: np.random.Generator,
    drr_db: float = 6.0,
    pre_delay_ms: float = 5.0,
) -> np.ndarray:
    """Generate an approximate synthetic RIR using the exponential decay model.

    This is a *statistical approximation* — shaped noise with an exponential
    decay envelope — not a physics-based room model.  It produces
    perceptually plausible reverberation for data augmentation purposes.
    For accurate acoustic modelling, use measured IRs via
    :class:`ImpulseResponseConvolver`.

    The RIR consists of a direct-path impulse followed by a noise tail that
    decays at a rate matching the requested RT60.

    Parameters
    ----------
    rt60_s:
        Reverberation time (RT60) in seconds.
    sr:
        Sample rate in Hz.
    rng:
        Random number generator.
    drr_db:
        Direct-to-reverb ratio in dB.
    pre_delay_ms:
        Pre-delay before the reverberant tail in milliseconds.

    Returns
    -------
    np.ndarray
        1-D float32 RIR normalised to unit peak.
    """
    n_samples = int(rt60_s * sr)
    if n_samples < 64:
        n_samples = 64

    # Noise tail
    noise = rng.standard_normal(n_samples).astype(np.float32)

    # Decay envelope: -60 dB in rt60_s → exp(-3 * ln(10) / rt60_s * t)
    decay_rate = 3.0 * np.log(10) / rt60_s
    t = np.arange(n_samples) / sr
    envelope = np.exp(-decay_rate * t).astype(np.float32)
    tail = noise * envelope

    # Direct impulse
    direct_amp = float(10.0 ** (drr_db / 20.0))
    pre_samples = int(pre_delay_ms * sr / 1000.0)
    rir = np.zeros(n_samples + pre_samples, dtype=np.float32)
    rir[pre_samples] = direct_amp
    rir[pre_samples:] += tail

    # Normalize
    peak = np.max(np.abs(rir))
    if peak > 1e-8:
        rir /= peak
    return rir


# ---------------------------------------------------------------------------
# RoomSimulator
# ---------------------------------------------------------------------------


class RoomSimulator(Transform):
    """Approximate room acoustics by convolving audio with a synthetic RIR.

    Uses a lightweight exponential-decay noise model to generate plausible
    reverberation for data augmentation.  This is *not* a physics-based
    room simulator — for accurate acoustic modelling, use measured impulse
    responses via :class:`ImpulseResponseConvolver`.

    Parameters
    ----------
    rt60_range:
        ``(min_s, max_s)`` reverberation time.  Typical values:
        small room ≈ 0.2-0.4 s, large room ≈ 0.6-1.5 s, hall ≈ 1.5-4 s.
    drr_db_range:
        ``(min, max)`` direct-to-reverb ratio in dB.
    wet_range:
        ``(min, max)`` wet/dry mix (0 = dry, 1 = fully wet).
    pre_delay_ms:
        Fixed pre-delay for the reverberant tail in milliseconds.
    preserve_length:
        If ``True`` trim the output to the input length.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import RoomSimulator
    >>> aug = RoomSimulator(rt60_range=(0.2, 1.5), wet_range=(0.3, 0.8))
    >>> audio_out, sr = aug(audio, sr=16000, seed=42)
    """

    def __init__(
        self,
        rt60_range: tuple[float, float] = (0.2, 1.5),
        drr_db_range: tuple[float, float] = (3.0, 12.0),
        wet_range: tuple[float, float] = (0.3, 0.8),
        pre_delay_ms: float = 5.0,
        preserve_length: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.rt60_range = rt60_range
        self.drr_db_range = drr_db_range
        self.wet_range = wet_range
        self.pre_delay_ms = float(pre_delay_ms)
        self.preserve_length = preserve_length

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        from scipy.signal import fftconvolve

        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape

        rt60 = float(rng.uniform(self.rt60_range[0], self.rt60_range[1]))
        drr_db = float(rng.uniform(self.drr_db_range[0], self.drr_db_range[1]))
        wet = float(rng.uniform(self.wet_range[0], self.wet_range[1]))

        rir = _generate_synthetic_rir(rt60, sr, rng, drr_db=drr_db, pre_delay_ms=self.pre_delay_ms)

        out_channels = []
        for ch in range(n_ch):
            # Different RIR per channel for naturalness (slight de-correlation)
            if ch > 0:
                rir = _generate_synthetic_rir(
                    rt60 * float(rng.uniform(0.95, 1.05)),
                    sr,
                    rng,
                    drr_db=drr_db,
                    pre_delay_ms=self.pre_delay_ms,
                )
            wet_sig = fftconvolve(arr[ch], rir).astype(arr.dtype)
            if self.preserve_length:
                wet_sig = wet_sig[:n_samp]
            mixed = (1.0 - wet) * arr[ch] + wet * wet_sig[: len(arr[ch])]
            out_channels.append(mixed.astype(arr.dtype))

        result = np.stack(out_channels, axis=0)
        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# ImpulseResponseConvolver
# ---------------------------------------------------------------------------


class ImpulseResponseConvolver(Transform):
    """Convolve audio with a user-provided impulse response file.

    Parameters
    ----------
    ir_sources:
        List of IR file paths or a single directory.  One file is randomly
        selected per call.
    wet_range:
        ``(min, max)`` wet/dry mix.
    normalize_ir:
        If ``True`` normalize the IR to unit RMS before convolution.
    preserve_length:
        If ``True`` trim output to input length.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import ImpulseResponseConvolver
    >>> aug = ImpulseResponseConvolver("irs/", wet_range=(0.4, 1.0))
    >>> audio_out, sr = aug(audio, sr=44100, seed=1)
    """

    def __init__(
        self,
        ir_sources: str | Path | Sequence[str | Path],
        wet_range: tuple[float, float] = (0.5, 1.0),
        normalize_ir: bool = True,
        preserve_length: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(ir_sources, (str, Path)):
            src = Path(ir_sources)
            if src.is_dir():
                exts = {".wav", ".flac", ".aiff"}
                self.ir_files = [p for p in src.rglob("*") if p.suffix.lower() in exts]
            else:
                self.ir_files = [src]
        else:
            self.ir_files = [Path(p) for p in ir_sources]
        if not self.ir_files:
            raise ValueError("No IR files found")
        self.wet_range = wet_range
        self.normalize_ir = normalize_ir
        self.preserve_length = preserve_length

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        from scipy.signal import fftconvolve

        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape

        ir_path = self.ir_files[int(rng.integers(len(self.ir_files)))]
        ir_audio, _ir_sr = load_audio(ir_path, target_sr=sr, mono=True)
        ir = ir_audio if ir_audio.ndim == 1 else ir_audio[0]

        if self.normalize_ir:
            rms = float(np.sqrt(np.mean(ir**2)) + 1e-12)
            ir = ir / rms

        wet = float(rng.uniform(self.wet_range[0], self.wet_range[1]))

        out_channels = []
        for ch in range(n_ch):
            wet_sig = fftconvolve(arr[ch], ir).astype(arr.dtype)
            if self.preserve_length:
                wet_sig = wet_sig[:n_samp]
            mixed = (1.0 - wet) * arr[ch] + wet * wet_sig[:n_samp]
            out_channels.append(mixed.astype(arr.dtype))

        result = np.stack(out_channels, axis=0)
        return _from_2d(result, was_mono), sr
