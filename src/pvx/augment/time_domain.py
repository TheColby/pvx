"""Time-domain audio augmentation transforms.

Provides transforms that operate on the waveform directly: gain
perturbation, clipping, time shifting, fade in/out, silence trimming, and
reverse.  Production-quality time-stretch and pitch-shift wrappers that
call ``pvx voc`` are also provided here.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .core import Transform, _from_2d, _to_2d, load_audio, save_audio

# ---------------------------------------------------------------------------
# GainPerturber
# ---------------------------------------------------------------------------


class GainPerturber(Transform):
    """Randomly adjust the gain of audio.

    Parameters
    ----------
    gain_db:
        Fixed gain or ``(min_db, max_db)`` range.  Positive = louder,
        negative = quieter.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import GainPerturber
    >>> aug = GainPerturber(gain_db=(-6, 6))
    >>> audio_out, sr = aug(audio, sr=16000, seed=0)
    """

    def __init__(
        self,
        gain_db: float | tuple[float, float] = (-6.0, 6.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(gain_db, (int, float)):
            self.gain_db_range: tuple[float, float] = (float(gain_db), float(gain_db))
        else:
            self.gain_db_range = (float(gain_db[0]), float(gain_db[1]))

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        gain_db = float(rng.uniform(self.gain_db_range[0], self.gain_db_range[1]))
        scale = 10.0 ** (gain_db / 20.0)
        return (audio * scale).astype(audio.dtype), sr


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class Normalizer(Transform):
    """Normalize audio to a target peak or RMS level.

    Parameters
    ----------
    mode:
        ``"peak"`` normalizes to peak amplitude, ``"rms"`` to RMS.
    target_db:
        Target level in dBFS (default -1.0 for peak, -20.0 for RMS).
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        mode: str = "peak",
        target_db: float = -1.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if mode not in ("peak", "rms"):
            raise ValueError(f"mode must be 'peak' or 'rms', got {mode!r}")
        self.mode = mode
        self.target_db = float(target_db)

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        if self.mode == "peak":
            current = float(np.max(np.abs(audio)) + 1e-12)
        else:
            current = float(np.sqrt(np.mean(audio**2)) + 1e-12)
        target = 10.0 ** (self.target_db / 20.0)
        scale = target / current
        return (audio * scale).astype(audio.dtype), sr


# ---------------------------------------------------------------------------
# ClippingSimulator
# ---------------------------------------------------------------------------


class ClippingSimulator(Transform):
    """Simulate ADC clipping or amplifier saturation.

    Parameters
    ----------
    percentile:
        Samples above this percentile of absolute amplitude are clipped.
        Range (80, 100) — lower values = more aggressive clipping.
        Either a fixed value or a ``(min, max)`` range.
    mode:
        ``"hard"`` = rectangular clipping; ``"soft"`` = tanh saturation.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import ClippingSimulator
    >>> aug = ClippingSimulator(percentile=(85, 99), mode="soft")
    >>> audio_out, sr = aug(audio, sr=44100, seed=2)
    """

    def __init__(
        self,
        percentile: float | tuple[float, float] = (90.0, 99.5),
        mode: str = "hard",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(percentile, (int, float)):
            self.percentile_range: tuple[float, float] = (float(percentile), float(percentile))
        else:
            self.percentile_range = (float(percentile[0]), float(percentile[1]))
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self.mode = mode

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        pct = float(rng.uniform(self.percentile_range[0], self.percentile_range[1]))
        threshold = float(np.percentile(np.abs(audio), pct))
        if self.mode == "hard":
            clipped = np.clip(audio, -threshold, threshold)
        else:
            # Soft clipping: scale to threshold, apply tanh, scale back
            scaled = audio / (threshold + 1e-8)
            clipped = np.tanh(scaled) * threshold
        return clipped.astype(audio.dtype), sr


# ---------------------------------------------------------------------------
# TimeShift
# ---------------------------------------------------------------------------


class TimeShift(Transform):
    """Randomly shift audio in time with zero-padding or wrap-around.

    Parameters
    ----------
    shift:
        Fixed shift in seconds or ``(min_s, max_s)`` range.  Negative
        values shift left (cut the start), positive shift right (add silence).
    mode:
        ``"zero"`` = pad with zeros; ``"wrap"`` = wrap the audio.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import TimeShift
    >>> aug = TimeShift(shift=(-0.2, 0.2), mode="zero")
    >>> audio_out, sr = aug(audio, sr=16000, seed=9)
    """

    def __init__(
        self,
        shift: float | tuple[float, float] = (-0.1, 0.1),
        mode: str = "zero",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(shift, (int, float)):
            self.shift_range: tuple[float, float] = (float(shift), float(shift))
        else:
            self.shift_range = (float(shift[0]), float(shift[1]))
        if mode not in ("zero", "wrap"):
            raise ValueError(f"mode must be 'zero' or 'wrap', got {mode!r}")
        self.mode = mode

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        shift_s = float(rng.uniform(self.shift_range[0], self.shift_range[1]))
        shift_samples = int(shift_s * sr)
        arr, was_mono = _to_2d(audio)
        _n_ch, n_samp = arr.shape

        if shift_samples == 0:
            return _from_2d(arr, was_mono), sr

        if self.mode == "zero":
            result = np.zeros_like(arr)
            if shift_samples > 0:
                result[:, shift_samples:] = arr[:, : n_samp - shift_samples]
            else:
                result[:, : n_samp + shift_samples] = arr[:, -shift_samples:]
        else:
            result = np.roll(arr, shift_samples, axis=1)

        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------


class Reverse(Transform):
    """Reverse the audio waveform.

    Parameters
    ----------
    p:
        Probability of applying this transform.
    """

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        return np.flip(audio, axis=-1).copy(), sr


# ---------------------------------------------------------------------------
# Fade
# ---------------------------------------------------------------------------


class Fade(Transform):
    """Apply fade-in and/or fade-out to audio.

    Parameters
    ----------
    fade_in:
        Fixed fade-in duration in seconds or ``(min_s, max_s)`` range.
    fade_out:
        Fixed fade-out duration in seconds or ``(min_s, max_s)`` range.
    curve:
        Fade curve type: ``"linear"``, ``"logarithmic"``, ``"exponential"``.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import Fade
    >>> aug = Fade(fade_in=(0.01, 0.1), fade_out=(0.05, 0.3))
    >>> audio_out, sr = aug(audio, sr=44100, seed=0)
    """

    def __init__(
        self,
        fade_in: float | tuple[float, float] = 0.01,
        fade_out: float | tuple[float, float] = 0.05,
        curve: str = "linear",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(fade_in, (int, float)):
            self.fade_in_range: tuple[float, float] = (float(fade_in), float(fade_in))
        else:
            self.fade_in_range = (float(fade_in[0]), float(fade_in[1]))
        if isinstance(fade_out, (int, float)):
            self.fade_out_range: tuple[float, float] = (float(fade_out), float(fade_out))
        else:
            self.fade_out_range = (float(fade_out[0]), float(fade_out[1]))
        if curve not in ("linear", "logarithmic", "exponential"):
            raise ValueError(f"curve must be linear/logarithmic/exponential, got {curve!r}")
        self.curve = curve

    @staticmethod
    def _make_envelope(n: int, curve: str) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n)
        if curve == "linear":
            return t.astype(np.float32)
        if curve == "logarithmic":
            return (np.log1p(t * (np.e - 1))).astype(np.float32)
        # exponential
        return ((np.exp(t) - 1.0) / (np.e - 1.0)).astype(np.float32)

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        _n_ch, n_samp = arr.shape
        result = arr.copy()

        fi_s = float(rng.uniform(self.fade_in_range[0], self.fade_in_range[1]))
        fo_s = float(rng.uniform(self.fade_out_range[0], self.fade_out_range[1]))

        fi_samp = min(int(fi_s * sr), n_samp)
        fo_samp = min(int(fo_s * sr), n_samp - fi_samp)

        if fi_samp > 0:
            env = self._make_envelope(fi_samp, self.curve)
            result[:, :fi_samp] *= env

        if fo_samp > 0:
            env = self._make_envelope(fo_samp, self.curve)[::-1]
            result[:, n_samp - fo_samp :] *= env

        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# TrimSilence
# ---------------------------------------------------------------------------


class TrimSilence(Transform):
    """Trim leading and/or trailing silence.

    Parameters
    ----------
    threshold_db:
        Amplitude threshold below which samples are considered silence.
    frame_length:
        Frame size for silence detection in samples.
    trim_start:
        Whether to trim silence from the start.
    trim_end:
        Whether to trim silence from the end.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        threshold_db: float = -60.0,
        frame_length: int = 512,
        trim_start: bool = True,
        trim_end: bool = True,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.threshold_db = float(threshold_db)
        self.frame_length = int(frame_length)
        self.trim_start = trim_start
        self.trim_end = trim_end

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        # Use first channel to detect silence positions
        mono = arr.mean(axis=0)
        threshold = 10.0 ** (self.threshold_db / 20.0)
        is_active = np.abs(mono) > threshold

        if not np.any(is_active):
            return _from_2d(arr, was_mono), sr

        first = int(np.argmax(is_active)) if self.trim_start else 0
        last = int(len(is_active) - np.argmax(is_active[::-1])) if self.trim_end else arr.shape[1]

        trimmed = arr[:, first:last]
        return _from_2d(trimmed, was_mono), sr


# ---------------------------------------------------------------------------
# FixedLengthCrop
# ---------------------------------------------------------------------------


class FixedLengthCrop(Transform):
    """Crop or pad audio to a fixed duration.

    Parameters
    ----------
    duration_s:
        Target duration in seconds.
    crop_mode:
        ``"random"`` crop starting position, ``"start"`` always from the
        beginning, ``"end"`` always from the end.
    pad_mode:
        If audio is shorter than *duration_s*: ``"zero"`` pads with silence,
        ``"wrap"`` loops the audio.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import FixedLengthCrop
    >>> aug = FixedLengthCrop(duration_s=3.0)
    >>> audio_out, sr = aug(audio, sr=16000, seed=0)
    """

    def __init__(
        self,
        duration_s: float = 3.0,
        crop_mode: str = "random",
        pad_mode: str = "zero",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.duration_s = float(duration_s)
        if crop_mode not in ("random", "start", "end"):
            raise ValueError(f"crop_mode must be random/start/end, got {crop_mode!r}")
        self.crop_mode = crop_mode
        if pad_mode not in ("zero", "wrap"):
            raise ValueError(f"pad_mode must be zero/wrap, got {pad_mode!r}")
        self.pad_mode = pad_mode

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        arr, was_mono = _to_2d(audio)
        n_ch, n_samp = arr.shape
        target = int(self.duration_s * sr)

        if n_samp >= target:
            if self.crop_mode == "random":
                start = int(rng.integers(0, n_samp - target + 1))
            elif self.crop_mode == "start":
                start = 0
            else:
                start = n_samp - target
            result = arr[:, start : start + target]
        else:
            if self.pad_mode == "zero":
                result = np.zeros((n_ch, target), dtype=arr.dtype)
                result[:, :n_samp] = arr
            else:
                reps = int(np.ceil(target / n_samp))
                result = np.tile(arr, (1, reps))[:, :target]

        return _from_2d(result, was_mono), sr


# ---------------------------------------------------------------------------
# Engine detection helper
# ---------------------------------------------------------------------------


def _has_torch() -> bool:
    """Return True if PyTorch is importable."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _has_torchaudio() -> bool:
    """Return True if torchaudio is importable."""
    try:
        import torchaudio  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_engine(engine: str) -> str:
    """Resolve ``"auto"`` to the best available engine.

    Returns ``"torchaudio"``, ``"pytorch"``, or ``"pvx-cli"``.

    Priority for ``"auto"``: torchaudio > pytorch > pvx-cli.
    """
    if engine == "torchaudio":
        if not _has_torchaudio():
            raise RuntimeError(
                "engine='torchaudio' requested but torchaudio is not installed. "
                "Install with: pip install 'pvx[torch]'"
            )
        return "torchaudio"
    if engine == "pytorch":
        if not _has_torch():
            raise RuntimeError(
                "engine='pytorch' requested but PyTorch is not installed. "
                "Install with: pip install 'pvx[torch]'"
            )
        return "pytorch"
    if engine == "pvx-cli":
        return "pvx-cli"
    # auto: prefer torchaudio > pytorch > pvx-cli
    if _has_torchaudio():
        return "torchaudio"
    if _has_torch():
        return "pytorch"
    return "pvx-cli"


def _torchaudio_time_stretch(audio: np.ndarray, sr: int, rate: float) -> tuple[np.ndarray, int]:
    """Time-stretch using torchaudio's built-in phase vocoder.

    torchaudio.transforms.TimeStretch uses an optimized C++ phase-vocoder
    kernel that is typically faster than the pure-Python loop in
    ``_torch_phase_vocoder``.
    """
    import torch
    import torchaudio

    was_mono = audio.ndim == 1
    if was_mono:
        arr = audio[np.newaxis, :]
    else:
        arr = audio

    # torchaudio.transforms.TimeStretch expects (batch, freq, time) complex STFT
    # but the functional API is more direct:
    n_fft = 2048
    hop_length = 512
    tensor = torch.from_numpy(arr.astype(np.float32))  # (C, T)
    window = torch.hann_window(n_fft)

    channels = []
    for ch in range(tensor.shape[0]):
        spec = torch.stft(
            tensor[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )  # (n_bins, n_frames)
        # Use torchaudio's phase_vocoder
        stretched_spec = torchaudio.functional.phase_vocoder(
            spec.unsqueeze(0),  # (1, n_bins, n_frames)
            rate=rate,
            phase_advance=torch.linspace(
                0, torch.pi * hop_length, spec.shape[0], dtype=torch.float32
            )[..., None],
        ).squeeze(0)  # (n_bins, n_out_frames)
        # iSTFT
        stretched = torch.istft(
            stretched_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
        )
        channels.append(stretched.numpy())

    result = np.stack(channels, axis=0)
    if was_mono:
        result = result[0]
    return result, sr


def _torchaudio_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> tuple[np.ndarray, int]:
    """Pitch-shift using torchaudio's built-in phase vocoder + resampling."""
    from scipy.signal import resample

    pitch_ratio = 2.0 ** (semitones / 12.0)
    stretched, _ = _torchaudio_time_stretch(audio, sr, rate=pitch_ratio)

    # Resample back to original length
    orig_len = audio.shape[-1]
    if stretched.ndim == 1:
        result = resample(stretched, orig_len).astype(np.float32)
    else:
        channels = []
        for ch in range(stretched.shape[0]):
            channels.append(resample(stretched[ch], orig_len).astype(np.float32))
        result = np.stack(channels, axis=0)

    return result, sr


def _pytorch_time_stretch(audio: np.ndarray, sr: int, rate: float) -> tuple[np.ndarray, int]:
    """Time-stretch using the native PyTorch phase vocoder."""
    import torch

    from .gpu import _torch_phase_vocoder

    was_mono = audio.ndim == 1
    if was_mono:
        arr = audio[np.newaxis, :]
    else:
        arr = audio

    channels = []
    for ch in range(arr.shape[0]):
        t = torch.from_numpy(arr[ch].astype(np.float32))
        stretched = _torch_phase_vocoder(t, stretch=rate)
        channels.append(stretched.numpy())

    result = np.stack(channels, axis=0)
    if was_mono:
        result = result[0]
    return result, sr


def _pytorch_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> tuple[np.ndarray, int]:
    """Pitch-shift using the native PyTorch phase vocoder + resampling."""
    from scipy.signal import resample

    pitch_ratio = 2.0 ** (semitones / 12.0)

    # Step 1: time-stretch by pitch_ratio
    stretched, _ = _pytorch_time_stretch(audio, sr, rate=pitch_ratio)

    # Step 2: resample back to original length
    orig_len = audio.shape[-1]
    if stretched.ndim == 1:
        result = resample(stretched, orig_len).astype(np.float32)
    else:
        channels = []
        for ch in range(stretched.shape[0]):
            channels.append(resample(stretched[ch], orig_len).astype(np.float32))
        result = np.stack(channels, axis=0)

    return result, sr


# ---------------------------------------------------------------------------
# TimeStretch — phase-vocoder time stretching
# ---------------------------------------------------------------------------


class TimeStretch(Transform):
    """High-quality time-stretch via a phase-vocoder engine.

    By default (``engine="auto"``), uses the native PyTorch phase vocoder
    when PyTorch is installed, falling back to ``pvx voc`` (subprocess) if
    not.  You can force a specific engine with ``engine="pytorch"`` or
    ``engine="pvx-cli"``.

    The ``pvx-cli`` engine leverages the full pvx DSP stack (transient
    handling, stereo coherence, formant preservation) but requires the
    pvx CLI to be installed.  The ``pytorch`` engine runs entirely in
    Python with no subprocess overhead.

    Parameters
    ----------
    rate:
        Stretch factor (> 1.0 = slower, < 1.0 = faster) or
        ``(min, max)`` range.
    preserve_pitch:
        If ``True`` pitch is locked at 0 semitones during stretching.
        Only used with ``engine="pvx-cli"``.
    preset:
        pvx preset name (e.g. ``"vocal_studio"``, ``"drums_safe"``).
        Only used with ``engine="pvx-cli"``.
    engine:
        ``"auto"`` (default — prefer pytorch), ``"pytorch"``, or ``"pvx-cli"``.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import TimeStretch
    >>> aug = TimeStretch(rate=(0.8, 1.25))
    >>> audio_out, sr = aug(audio, sr=16000, seed=0)

    >>> # Force pvx CLI for production-quality transient handling
    >>> aug = TimeStretch(rate=(0.8, 1.25), engine="pvx-cli", preset="drums_safe")
    """

    def __init__(
        self,
        rate: float | tuple[float, float] = (0.8, 1.25),
        preserve_pitch: bool = True,
        preset: str = "default",
        engine: str = "auto",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(rate, (int, float)):
            self.rate_range: tuple[float, float] = (float(rate), float(rate))
        else:
            self.rate_range = (float(rate[0]), float(rate[1]))
        self.preserve_pitch = preserve_pitch
        self.preset = preset
        if engine not in ("auto", "pytorch", "torchaudio", "pvx-cli"):
            raise ValueError(f"engine must be auto/pytorch/torchaudio/pvx-cli, got {engine!r}")
        self.engine = engine

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        rate = float(rng.uniform(self.rate_range[0], self.rate_range[1]))
        engine = _resolve_engine(self.engine)

        if engine == "torchaudio":
            return _torchaudio_time_stretch(audio, sr, rate)
        if engine == "pytorch":
            return _pytorch_time_stretch(audio, sr, rate)
        return _call_pvx_voc(
            audio,
            sr,
            rng,
            stretch=rate,
            pitch=0.0 if self.preserve_pitch else None,
            preset=self.preset,
        )


# ---------------------------------------------------------------------------
# PitchShift — phase-vocoder pitch shifting
# ---------------------------------------------------------------------------


class PitchShift(Transform):
    """High-quality pitch shift via a phase-vocoder engine.

    By default (``engine="auto"``), uses the native PyTorch phase vocoder
    + resampling when PyTorch is installed, falling back to ``pvx voc``
    (subprocess) if not.

    The ``pvx-cli`` engine supports formant preservation and advanced
    presets.  The ``pytorch`` engine runs entirely in Python with no
    subprocess overhead.

    Parameters
    ----------
    semitones:
        Fixed shift or ``(min, max)`` range in semitones.
    preserve_duration:
        If ``True`` stretch is locked at 1.0.
    formant_mode:
        ``"formant-preserving"`` or ``"standard"``.
        Only used with ``engine="pvx-cli"``.
    preset:
        pvx preset name.
        Only used with ``engine="pvx-cli"``.
    engine:
        ``"auto"`` (default — prefer pytorch), ``"pytorch"``, or ``"pvx-cli"``.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> from pvx.augment import PitchShift
    >>> aug = PitchShift(semitones=(-2, 2))
    >>> audio_out, sr = aug(audio, sr=16000, seed=1)

    >>> # Force pvx CLI for formant-preserving pitch shift
    >>> aug = PitchShift(semitones=(-2, 2), engine="pvx-cli", formant_mode="formant-preserving")
    """

    def __init__(
        self,
        semitones: float | tuple[float, float] = (-2.0, 2.0),
        preserve_duration: bool = True,
        formant_mode: str = "formant-preserving",
        preset: str = "vocal_studio",
        engine: str = "auto",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(semitones, (int, float)):
            self.semitones_range: tuple[float, float] = (float(semitones), float(semitones))
        else:
            self.semitones_range = (float(semitones[0]), float(semitones[1]))
        self.preserve_duration = preserve_duration
        self.formant_mode = formant_mode
        self.preset = preset
        if engine not in ("auto", "pytorch", "torchaudio", "pvx-cli"):
            raise ValueError(f"engine must be auto/pytorch/torchaudio/pvx-cli, got {engine!r}")
        self.engine = engine

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        semitones = float(rng.uniform(self.semitones_range[0], self.semitones_range[1]))
        engine = _resolve_engine(self.engine)

        if engine == "torchaudio":
            return _torchaudio_pitch_shift(audio, sr, semitones)
        if engine == "pytorch":
            return _pytorch_pitch_shift(audio, sr, semitones)
        stretch = 1.0 if self.preserve_duration else None
        return _call_pvx_voc(
            audio,
            sr,
            rng,
            stretch=stretch,
            pitch=semitones,
            preset=self.preset,
            formant_mode=self.formant_mode,
        )


# ---------------------------------------------------------------------------
# Internal helper: call pvx voc as subprocess
# ---------------------------------------------------------------------------


def _call_pvx_voc(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    stretch: float | None = 1.0,
    pitch: float | None = None,
    preset: str = "default",
    formant_mode: str | None = None,
) -> tuple[np.ndarray, int]:
    """Write audio to a temp WAV, call ``pvx voc``, read back result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "in.wav"
        tmp_out = Path(tmpdir) / "out.wav"
        save_audio(tmp_in, audio, sr)

        cmd = [sys.executable, "-m", "pvx.cli.pvx", "voc", str(tmp_in), "--output", str(tmp_out)]
        if stretch is not None:
            cmd += ["--stretch", f"{stretch:.6f}"]
        if pitch is not None:
            cmd += ["--pitch", f"{pitch:.6f}"]
        cmd += ["--preset", preset]
        if formant_mode:
            cmd += ["--pitch-mode", formant_mode]
        cmd += ["--verbosity", "silent"]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0 or not tmp_out.exists():
                stderr_msg = result.stderr.decode(errors="replace").strip() if result.stderr else ""
                raise RuntimeError(
                    f"pvx voc failed (exit code {result.returncode}). "
                    f"TimeStretch and PitchShift require the pvx CLI to be installed "
                    f"and working. Install with: pip install pvx\n"
                    f"stderr: {stderr_msg}"
                )
            return load_audio(tmp_out)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "pvx voc timed out after 120 seconds. The input audio may be "
                "too long for real-time processing."
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pvx CLI not found. TimeStretch and PitchShift require the pvx "
                "CLI to be installed and accessible. Install with: pip install pvx"
            ) from exc
