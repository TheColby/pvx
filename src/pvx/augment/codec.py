"""Codec and transmission degradation transforms.

Simulates lossy audio compression (mp3-like, telephone, VoIP), bit-depth
reduction (bit-crushing), and low-fidelity recording artifacts — all using
only NumPy and SciPy (no external codec binaries required).
"""

from __future__ import annotations

import numpy as np

from .core import Transform, _from_2d, _to_2d

# ---------------------------------------------------------------------------
# Codec presets
# ---------------------------------------------------------------------------

_CODEC_PRESETS: dict[str, dict[str, object]] = {
    # Simulate a ~32 kbps MP3 artifact: 22 kHz bandwidth limit + mild quantization
    "mp3_low": {"bandwidth_hz": 11000, "bit_depth": 12, "resample_ratio": None},
    # Simulate a ~128 kbps MP3: 16 kHz bandwidth, light quantization
    "mp3_medium": {"bandwidth_hz": 16000, "bit_depth": 14, "resample_ratio": None},
    # Telephone: 300-3400 Hz bandpass, 8 kHz effective sample rate
    "telephone": {"bandwidth_hz": (300.0, 3400.0), "bit_depth": 8, "resample_ratio": 8000},
    # VoIP (narrow band): 50-4000 Hz, 8-kHz resample
    "voip_narrow": {"bandwidth_hz": (50.0, 4000.0), "bit_depth": 8, "resample_ratio": 8000},
    # VoIP (wide band): up to 7 kHz, 16-kHz resample
    "voip_wide": {"bandwidth_hz": (50.0, 7000.0), "bit_depth": 12, "resample_ratio": 16000},
    # AM radio: ~200 Hz-5 kHz bandpass
    "am_radio": {"bandwidth_hz": (200.0, 5000.0), "bit_depth": 8, "resample_ratio": None},
    # Lo-fi: heavy bit-crush + low-pass
    "lo_fi": {"bandwidth_hz": 8000, "bit_depth": 6, "resample_ratio": None},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lowpass(audio: np.ndarray, cutoff_hz: float, sr: int, order: int = 6) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    wn = min(cutoff_hz / nyq, 1.0 - 1e-6)
    sos = butter(order, wn, btype="low", output="sos")
    return sosfiltfilt(sos, audio).astype(audio.dtype)


def _bandpass(
    audio: np.ndarray,
    low_hz: float,
    high_hz: float,
    sr: int,
    order: int = 4,
) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 1.0 - 1e-4)
    if lo >= hi:
        return audio
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, audio).astype(audio.dtype)


def _quantize(audio: np.ndarray, bits: int) -> np.ndarray:
    """Simulate bit-depth reduction by quantizing to *bits* bits."""
    levels = 2**bits
    # Quantize in [-1, 1] space
    clipped = np.clip(audio, -1.0, 1.0 - 1e-6)
    quantized = np.round(clipped * (levels / 2)) / (levels / 2)
    return quantized.astype(audio.dtype)


def _resample_simulate(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Downsample then upsample to simulate codec sample-rate reduction."""
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(target_sr, sr)
    down_up = target_sr // g, sr // g
    up_down = sr // g, target_sr // g
    downsampled = resample_poly(audio, down_up[0], down_up[1])
    upsampled = resample_poly(downsampled, up_down[0], up_down[1])
    n = len(audio)
    if len(upsampled) >= n:
        return upsampled[:n].astype(audio.dtype)
    return np.pad(upsampled, (0, n - len(upsampled))).astype(audio.dtype)


# ---------------------------------------------------------------------------
# CodecDegradation
# ---------------------------------------------------------------------------


class CodecDegradation(Transform):
    """Simulate lossy codec artifacts without external codec binaries.

    Uses a combination of band-limiting, bit-depth reduction, and
    sample-rate decimation to produce perceptually similar degradation to
    real codecs.

    Parameters
    ----------
    codec:
        Preset name: ``"mp3_low"``, ``"mp3_medium"``, ``"telephone"``,
        ``"voip_narrow"``, ``"voip_wide"``, ``"am_radio"``, ``"lo_fi"``,
        or ``"random"`` (pick randomly each call).
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import CodecDegradation
    >>> aug = CodecDegradation(codec="telephone")
    >>> audio_out, sr = aug(audio, sr=16000, seed=0)
    """

    CODEC_NAMES = (*_CODEC_PRESETS.keys(), "random")

    def __init__(self, codec: str = "random", p: float = 1.0) -> None:
        super().__init__(p=p)
        if codec not in self.CODEC_NAMES:
            raise ValueError(f"codec must be one of {self.CODEC_NAMES}, got {codec!r}")
        self.codec = codec

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        n_ch, _n_samp = arr.shape

        codec_name = self.codec
        if codec_name == "random":
            codec_name = list(_CODEC_PRESETS.keys())[int(rng.integers(len(_CODEC_PRESETS)))]
        preset = _CODEC_PRESETS[codec_name]
        bandwidth = preset["bandwidth_hz"]
        bit_depth = int(preset["bit_depth"])
        resample_target = preset.get("resample_ratio")

        out_channels = []
        for ch in range(n_ch):
            sig = arr[ch].copy()

            # 1. Bandwidth limiting
            if isinstance(bandwidth, (list, tuple)):
                sig = _bandpass(sig, float(bandwidth[0]), float(bandwidth[1]), sr)
            elif bandwidth is not None:
                sig = _lowpass(sig, float(bandwidth), sr)

            # 2. Sample-rate simulation
            if resample_target is not None and int(resample_target) < sr:
                sig = _resample_simulate(sig, sr, int(resample_target))

            # 3. Bit-depth quantization
            sig = _quantize(sig, bit_depth)

            out_channels.append(sig.astype(arr.dtype))

        result = np.stack(out_channels, axis=0)
        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# BitCrusher
# ---------------------------------------------------------------------------


class BitCrusher(Transform):
    """Reduce effective bit depth to simulate lo-fi / ADC artifacts.

    Parameters
    ----------
    bits:
        Fixed bit depth or ``(min, max)`` range sampled uniformly.
        Range 4-16 covers everything from extreme lo-fi to near-lossless.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import BitCrusher
    >>> aug = BitCrusher(bits=(6, 12))
    >>> audio_out, sr = aug(audio, sr=44100, seed=5)
    """

    def __init__(
        self,
        bits: int | tuple[int, int] = (8, 16),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(bits, int):
            self.bits_range: tuple[int, int] = (bits, bits)
        else:
            self.bits_range = (int(bits[0]), int(bits[1]))

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        bits = int(rng.integers(self.bits_range[0], self.bits_range[1] + 1))
        result = _quantize(arr, bits).astype(arr.dtype)
        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# BandwidthLimiter
# ---------------------------------------------------------------------------


class BandwidthLimiter(Transform):
    """Randomly limit the audio bandwidth with a low-pass filter.

    Useful for simulating degraded recording equipment, telephone
    channels, or low-bitrate streaming.

    Parameters
    ----------
    cutoff_hz:
        Fixed cutoff or ``(min_hz, max_hz)`` range.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        cutoff_hz: float | tuple[float, float] = (4000.0, 16000.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(cutoff_hz, (int, float)):
            self.cutoff_range: tuple[float, float] = (float(cutoff_hz), float(cutoff_hz))
        else:
            self.cutoff_range = (float(cutoff_hz[0]), float(cutoff_hz[1]))

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        cutoff = float(rng.uniform(self.cutoff_range[0], self.cutoff_range[1]))
        out = np.stack(
            [_lowpass(arr[ch], cutoff, sr) for ch in range(arr.shape[0])],
            axis=0,
        ).astype(arr.dtype)
        return _from_2d(out, was_mono), sr
