#!/usr/bin/env python3
"""Multi-channel phase vocoder CLI for time and pitch manipulation."""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency guard
    np = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - dependency guard
    sf = None

try:
    from scipy.signal import resample as scipy_resample
except Exception:  # pragma: no cover - optional dependency
    scipy_resample = None

WindowType = Literal["hann", "hamming", "blackman", "rect"]
ResampleMode = Literal["auto", "fft", "linear"]
PhaseLockMode = Literal["off", "identity"]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class VocoderConfig:
    n_fft: int
    win_length: int
    hop_size: int
    window: WindowType
    center: bool
    phase_locking: PhaseLockMode
    transient_preserve: bool
    transient_threshold: float


@dataclass(frozen=True)
class PitchConfig:
    ratio: float
    source_f0_hz: float | None = None


@dataclass(frozen=True)
class JobResult:
    input_path: Path
    output_path: Path
    in_sr: int
    out_sr: int
    in_samples: int
    out_samples: int
    channels: int
    stretch: float
    pitch_ratio: float


@dataclass(frozen=True)
class FourierSyncPlan:
    frame_lengths: np.ndarray
    f0_track_hz: np.ndarray
    reference_n_fft: int


class ProgressBar:
    def __init__(self, label: str, enabled: bool, width: int = 32) -> None:
        self.label = label
        self.enabled = enabled
        self.width = max(10, width)
        self._last_fraction = -1.0
        self._last_ts = 0.0
        self._finished = False
        if self.enabled:
            self.set(0.0, "start")

    def set(self, fraction: float, detail: str = "") -> None:
        if not self.enabled or self._finished:
            return

        now = time.time()
        frac = min(1.0, max(0.0, fraction))
        should_render = (
            frac >= 1.0
            or self._last_fraction < 0.0
            or (frac - self._last_fraction) >= 0.005
            or (now - self._last_ts) >= 0.15
        )
        if not should_render:
            return

        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {detail}" if detail else ""
        sys.stderr.write(f"\r[{bar}] {frac * 100:6.2f}% {self.label}{suffix}")
        sys.stderr.flush()
        self._last_fraction = frac
        self._last_ts = now
        if frac >= 1.0:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._finished = True

    def finish(self, detail: str = "done") -> None:
        self.set(1.0, detail)


def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def ensure_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if sf is None:
        missing.append("soundfile")
    if missing:
        print(
            "Missing required dependencies: " + ", ".join(missing) + ". "
            "Install them with: pip install numpy soundfile",
            file=sys.stderr,
        )
        raise SystemExit(2)


def principal_angle(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def make_window(kind: WindowType, n_fft: int, win_length: int) -> np.ndarray:
    if kind == "hann":
        base = np.hanning(win_length)
    elif kind == "hamming":
        base = np.hamming(win_length)
    elif kind == "blackman":
        base = np.blackman(win_length)
    elif kind == "rect":
        base = np.ones(win_length, dtype=np.float64)
    else:  # pragma: no cover - parser blocks this
        raise ValueError(f"Unsupported window: {kind}")

    if win_length == n_fft:
        return base.astype(np.float64, copy=False)

    window = np.zeros(n_fft, dtype=np.float64)
    offset = (n_fft - win_length) // 2
    window[offset : offset + win_length] = base
    return window


def pad_for_framing(signal: np.ndarray, n_fft: int, hop: int, center: bool) -> tuple[np.ndarray, int]:
    if center:
        signal = np.pad(signal, (n_fft // 2, n_fft // 2), mode="constant")

    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size), mode="constant")

    remainder = (signal.size - n_fft) % hop
    pad_end = (hop - remainder) % hop
    if pad_end:
        signal = np.pad(signal, (0, pad_end), mode="constant")

    frame_count = 1 + (signal.size - n_fft) // hop
    return signal, frame_count


def stft(signal: np.ndarray, config: VocoderConfig) -> np.ndarray:
    signal, frame_count = pad_for_framing(signal, config.n_fft, config.hop_size, config.center)
    window = make_window(config.window, config.n_fft, config.win_length)
    spectrum = np.empty((config.n_fft // 2 + 1, frame_count), dtype=np.complex128)

    for frame_idx in range(frame_count):
        start = frame_idx * config.hop_size
        frame = signal[start : start + config.n_fft]
        spectrum[:, frame_idx] = np.fft.rfft(frame * window)

    return spectrum


def istft(
    spectrum: np.ndarray,
    config: VocoderConfig,
    expected_length: int | None = None,
) -> np.ndarray:
    n_frames = spectrum.shape[1]
    output_len = config.n_fft + config.hop_size * max(0, n_frames - 1)
    output = np.zeros(output_len, dtype=np.float64)
    weight = np.zeros(output_len, dtype=np.float64)
    window = make_window(config.window, config.n_fft, config.win_length)

    for frame_idx in range(n_frames):
        start = frame_idx * config.hop_size
        frame = np.fft.irfft(spectrum[:, frame_idx], n=config.n_fft)
        output[start : start + config.n_fft] += frame * window
        weight[start : start + config.n_fft] += window * window

    nz = weight > 1e-12
    output[nz] /= weight[nz]

    if config.center:
        trim = config.n_fft // 2
        if output.size > 2 * trim:
            output = output[trim:-trim]
        else:
            output = np.zeros(0, dtype=np.float64)

    if expected_length is not None:
        output = force_length(output, expected_length)

    return output


def scaled_win_length(base_win: int, base_fft: int, frame_len: int) -> int:
    if base_fft <= 0:
        return max(2, frame_len)
    scaled = int(round(base_win * frame_len / base_fft))
    return max(2, min(frame_len, scaled))


def resize_spectrum_bins(spectrum: np.ndarray, target_bins: int) -> np.ndarray:
    if target_bins <= 0:
        raise ValueError("target_bins must be > 0")
    if spectrum.size == target_bins:
        return spectrum.astype(np.complex128, copy=True)
    if spectrum.size == 0:
        return np.zeros(target_bins, dtype=np.complex128)

    x_old = np.linspace(0.0, 1.0, num=spectrum.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_bins, endpoint=True)
    real_new = np.interp(x_new, x_old, spectrum.real)
    imag_new = np.interp(x_new, x_old, spectrum.imag)
    return (real_new + 1j * imag_new).astype(np.complex128)


def smooth_series(values: np.ndarray, span: int) -> np.ndarray:
    if span <= 1 or values.size <= 2:
        return values
    kernel = np.ones(span, dtype=np.float64) / span
    pad = span // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def regularize_frame_lengths(frame_lengths: np.ndarray, max_step: int) -> np.ndarray:
    if frame_lengths.size <= 1:
        return frame_lengths

    out = frame_lengths.astype(np.int64, copy=True)
    step = max(1, int(max_step))

    for idx in range(1, out.size):
        lo = out[idx - 1] - step
        hi = out[idx - 1] + step
        out[idx] = int(np.clip(out[idx], lo, hi))

    for idx in range(out.size - 2, -1, -1):
        lo = out[idx + 1] - step
        hi = out[idx + 1] + step
        out[idx] = int(np.clip(out[idx], lo, hi))

    return out


def fill_nan_with_nearest(values: np.ndarray, fallback: float) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if out.size == 0:
        return out
    out[np.isnan(out)] = fallback
    for idx in range(1, out.size):
        if not np.isfinite(out[idx]):
            out[idx] = out[idx - 1]
    for idx in range(out.size - 2, -1, -1):
        if not np.isfinite(out[idx]):
            out[idx] = out[idx + 1]
    out[~np.isfinite(out)] = fallback
    return out


def lock_fft_length_to_f0(
    f0_hz: float,
    sample_rate: int,
    harmonic_bin: int,
    min_fft: int,
    max_fft: int,
) -> int:
    safe_f0 = max(float(f0_hz), 1e-6)
    locked = int(round(max(1, harmonic_bin) * sample_rate / safe_f0))
    return int(np.clip(max(16, locked), min_fft, max_fft))


def build_fourier_sync_plan(
    signal: np.ndarray,
    sample_rate: int,
    config: VocoderConfig,
    f0_min_hz: float,
    f0_max_hz: float,
    min_fft: int,
    max_fft: int,
    smooth_span: int,
    progress_callback: ProgressCallback | None = None,
) -> FourierSyncPlan:
    framed, frame_count = pad_for_framing(signal, config.n_fft, config.hop_size, config.center)
    if frame_count <= 0:
        return FourierSyncPlan(
            frame_lengths=np.array([config.n_fft], dtype=np.int64),
            f0_track_hz=np.array([(f0_min_hz + f0_max_hz) * 0.5], dtype=np.float64),
            reference_n_fft=max(config.n_fft, min_fft),
        )

    f0_track = np.full(frame_count, np.nan, dtype=np.float64)
    for frame_idx in range(frame_count):
        start = frame_idx * config.hop_size
        frame = framed[start : start + config.n_fft]
        if frame.size >= 4 and float(np.sqrt(np.mean(frame * frame))) >= 1e-6:
            try:
                f0_track[frame_idx] = estimate_f0_autocorrelation(frame, sample_rate, f0_min_hz, f0_max_hz)
            except Exception:
                pass
        if progress_callback is not None:
            progress_callback(frame_idx + 1, frame_count)

    finite_f0 = f0_track[np.isfinite(f0_track)]
    fallback_f0 = float(np.median(finite_f0)) if finite_f0.size else (f0_min_hz + f0_max_hz) * 0.5
    f0_track = fill_nan_with_nearest(f0_track, fallback=fallback_f0)
    f0_track = smooth_series(f0_track, smooth_span)
    f0_track = np.clip(f0_track, f0_min_hz, f0_max_hz)

    min_fft = max(16, min_fft)
    max_fft = max(min_fft, max_fft)
    reference_f0 = float(np.median(f0_track)) if f0_track.size else fallback_f0
    target_bin = max(1, int(round(reference_f0 * config.n_fft / sample_rate)))
    frame_lengths = np.array(
        [
            lock_fft_length_to_f0(
                f0_hz=f0,
                sample_rate=sample_rate,
                harmonic_bin=target_bin,
                min_fft=min_fft,
                max_fft=max_fft,
            )
            for f0 in f0_track
        ],
        dtype=np.int64,
    )
    frame_lengths = np.rint(smooth_series(frame_lengths.astype(np.float64), smooth_span)).astype(np.int64)
    frame_lengths = np.clip(frame_lengths, min_fft, max_fft)
    frame_lengths = regularize_frame_lengths(
        frame_lengths,
        max_step=max(2, int(round(config.n_fft * 0.015))),
    )
    reference_n_fft = int(max(config.n_fft, int(np.max(frame_lengths))))
    return FourierSyncPlan(frame_lengths=frame_lengths, f0_track_hz=f0_track, reference_n_fft=reference_n_fft)


def compute_transient_flags(magnitude: np.ndarray, threshold_scale: float) -> np.ndarray:
    if magnitude.shape[1] <= 1:
        return np.zeros(magnitude.shape[1], dtype=bool)

    flux = np.zeros(magnitude.shape[1], dtype=np.float64)
    positive_delta = np.maximum(0.0, np.diff(magnitude, axis=1))
    flux[1:] = np.sqrt(np.sum(positive_delta * positive_delta, axis=0))

    baseline = float(np.median(flux[1:])) if flux.size > 1 else 0.0
    if baseline <= 1e-12:
        baseline = float(np.mean(flux[1:])) if flux.size > 1 else 0.0
    if baseline <= 1e-12:
        return np.zeros_like(flux, dtype=bool)

    flags = flux >= (baseline * threshold_scale)
    flags[0] = False
    return flags


def find_spectral_peaks(magnitude: np.ndarray) -> np.ndarray:
    if magnitude.size < 3:
        return np.array([int(np.argmax(magnitude))], dtype=np.int64)

    interior = (
        (magnitude[1:-1] > magnitude[:-2])
        & (magnitude[1:-1] >= magnitude[2:])
    )
    peak_bins = np.where(interior)[0] + 1
    if peak_bins.size == 0:
        peak_bins = np.array([int(np.argmax(magnitude))], dtype=np.int64)
    return peak_bins.astype(np.int64, copy=False)


def apply_identity_phase_locking(
    synth_phase: np.ndarray,
    analysis_phase: np.ndarray,
    magnitude: np.ndarray,
) -> np.ndarray:
    peaks = find_spectral_peaks(magnitude)
    if peaks.size == 0:
        return synth_phase

    bins = np.arange(synth_phase.size, dtype=np.int64)[:, None]
    nearest_peak_idx = np.argmin(np.abs(bins - peaks[None, :]), axis=1)
    nearest_peaks = peaks[nearest_peak_idx]

    locked = synth_phase.copy()
    rel = principal_angle(analysis_phase - analysis_phase[nearest_peaks])
    locked[:] = synth_phase[nearest_peaks] + rel
    return locked


def phase_vocoder_time_stretch(
    signal: np.ndarray,
    stretch: float,
    config: VocoderConfig,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    if stretch <= 0:
        raise ValueError("Stretch factor must be > 0")
    if signal.size == 0:
        return signal

    input_stft = stft(signal, config)
    n_bins, n_frames = input_stft.shape
    if n_frames < 2:
        target_len = max(1, int(round(signal.size * stretch)))
        return force_length(signal.copy(), target_len)

    out_frames = max(2, int(round(n_frames * stretch)))
    time_steps = np.arange(out_frames, dtype=np.float64) / stretch
    time_steps = np.clip(time_steps, 0.0, n_frames - 1.000001)

    input_phase = np.angle(input_stft)
    input_mag = np.abs(input_stft)
    transient_flags = (
        compute_transient_flags(input_mag, config.transient_threshold)
        if config.transient_preserve
        else np.zeros(n_frames, dtype=bool)
    )

    phase = input_phase[:, 0].copy()
    omega = 2.0 * np.pi * config.hop_size * np.arange(n_bins, dtype=np.float64) / config.n_fft
    output_stft = np.zeros((n_bins, out_frames), dtype=np.complex128)
    if progress_callback is not None:
        progress_callback(0, out_frames)

    for out_idx, t in enumerate(time_steps):
        frame_idx = int(math.floor(t))
        frac = t - frame_idx

        left = input_stft[:, frame_idx]
        right = input_stft[:, min(frame_idx + 1, n_frames - 1)]
        left_phase = input_phase[:, frame_idx]
        right_phase = input_phase[:, min(frame_idx + 1, n_frames - 1)]

        mag = (1.0 - frac) * np.abs(left) + frac * np.abs(right)

        delta = right_phase - left_phase - omega
        delta = principal_angle(delta)
        synth_phase = phase + omega + delta

        if config.transient_preserve:
            transient_idx = min(frame_idx + (1 if frac >= 0.5 else 0), n_frames - 1)
            if transient_flags[transient_idx]:
                phase_blend = (1.0 - frac) * np.exp(1j * left_phase) + frac * np.exp(1j * right_phase)
                synth_phase = np.angle(phase_blend)

        if config.phase_locking == "identity":
            analysis_phase = np.angle(
                (1.0 - frac) * np.exp(1j * left_phase) + frac * np.exp(1j * right_phase)
            )
            synth_phase = apply_identity_phase_locking(synth_phase, analysis_phase, mag)

        phase = synth_phase
        output_stft[:, out_idx] = mag * np.exp(1j * phase)
        if progress_callback is not None and (out_idx == out_frames - 1 or (out_idx % 8) == 0):
            progress_callback(out_idx + 1, out_frames)

    target_length = max(1, int(round(signal.size * stretch)))
    return istft(output_stft, config, expected_length=target_length)


def phase_vocoder_time_stretch_fourier_sync(
    signal: np.ndarray,
    stretch: float,
    config: VocoderConfig,
    sync_plan: FourierSyncPlan,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    if stretch <= 0:
        raise ValueError("Stretch factor must be > 0")
    if signal.size == 0:
        return signal

    framed, frame_count = pad_for_framing(signal, config.n_fft, config.hop_size, config.center)
    if frame_count < 2:
        target_len = max(1, int(round(signal.size * stretch)))
        return force_length(signal.copy(), target_len)

    if sync_plan.frame_lengths.size != frame_count:
        src = np.linspace(0.0, 1.0, num=sync_plan.frame_lengths.size, endpoint=True)
        dst = np.linspace(0.0, 1.0, num=frame_count, endpoint=True)
        frame_lengths = np.interp(dst, src, sync_plan.frame_lengths.astype(np.float64))
        frame_lengths = np.clip(np.round(frame_lengths), 16, None).astype(np.int64)
    else:
        frame_lengths = sync_plan.frame_lengths.astype(np.int64, copy=False)

    out_frames = max(2, int(round(frame_count * stretch)))
    total_steps = frame_count + out_frames + out_frames

    ref_n_fft = int(max(sync_plan.reference_n_fft, config.n_fft, int(np.max(frame_lengths))))
    ref_bins = ref_n_fft // 2 + 1
    input_stft = np.empty((ref_bins, frame_count), dtype=np.complex128)
    if progress_callback is not None:
        progress_callback(0, total_steps)

    for frame_idx in range(frame_count):
        n_fft_i = int(frame_lengths[frame_idx])
        start = frame_idx * config.hop_size
        frame = force_length(framed[start : start + n_fft_i], n_fft_i)
        win_length_i = scaled_win_length(config.win_length, config.n_fft, n_fft_i)
        window = make_window(config.window, n_fft_i, win_length_i)
        spectrum = np.fft.rfft(frame * window, n=n_fft_i)
        input_stft[:, frame_idx] = resize_spectrum_bins(spectrum, ref_bins)
        if progress_callback is not None and (frame_idx == frame_count - 1 or (frame_idx % 8) == 0):
            progress_callback(frame_idx + 1, total_steps)

    time_steps = np.arange(out_frames, dtype=np.float64) / stretch
    time_steps = np.clip(time_steps, 0.0, frame_count - 1.000001)

    input_phase = np.angle(input_stft)
    input_mag = np.abs(input_stft)
    transient_flags = (
        compute_transient_flags(input_mag, config.transient_threshold)
        if config.transient_preserve
        else np.zeros(frame_count, dtype=bool)
    )

    phase = input_phase[:, 0].copy()
    omega = 2.0 * np.pi * config.hop_size * np.arange(ref_bins, dtype=np.float64) / ref_n_fft
    output_stft = np.zeros((ref_bins, out_frames), dtype=np.complex128)
    output_lengths = np.zeros(out_frames, dtype=np.int64)
    completed_steps = 0

    for out_idx, t in enumerate(time_steps):
        frame_idx = int(math.floor(t))
        frac = t - frame_idx
        right_idx = min(frame_idx + 1, frame_count - 1)

        left = input_stft[:, frame_idx]
        right = input_stft[:, right_idx]
        left_phase = input_phase[:, frame_idx]
        right_phase = input_phase[:, right_idx]

        mag = (1.0 - frac) * np.abs(left) + frac * np.abs(right)
        delta = principal_angle(right_phase - left_phase - omega)
        synth_phase = phase + omega + delta

        if config.transient_preserve:
            transient_idx = min(frame_idx + (1 if frac >= 0.5 else 0), frame_count - 1)
            if transient_flags[transient_idx]:
                phase_blend = (1.0 - frac) * np.exp(1j * left_phase) + frac * np.exp(1j * right_phase)
                synth_phase = np.angle(phase_blend)

        if config.phase_locking == "identity":
            analysis_phase = np.angle(
                (1.0 - frac) * np.exp(1j * left_phase) + frac * np.exp(1j * right_phase)
            )
            synth_phase = apply_identity_phase_locking(synth_phase, analysis_phase, mag)

        phase = synth_phase
        output_stft[:, out_idx] = mag * np.exp(1j * phase)

        n_left = int(frame_lengths[frame_idx])
        n_right = int(frame_lengths[right_idx])
        output_lengths[out_idx] = max(16, int(round((1.0 - frac) * n_left + frac * n_right)))
        completed_steps += 1
        if progress_callback is not None and (out_idx == out_frames - 1 or (out_idx % 8) == 0):
            progress_callback(frame_count + completed_steps, total_steps)

    output_len = config.hop_size * max(0, out_frames - 1) + int(np.max(output_lengths))
    output = np.zeros(output_len, dtype=np.float64)
    weight = np.zeros(output_len, dtype=np.float64)

    for frame_idx in range(out_frames):
        n_fft_i = int(output_lengths[frame_idx])
        bins_i = n_fft_i // 2 + 1
        spec_i = resize_spectrum_bins(output_stft[:, frame_idx], bins_i)
        frame = np.fft.irfft(spec_i, n=n_fft_i)
        win_length_i = scaled_win_length(config.win_length, config.n_fft, n_fft_i)
        window = make_window(config.window, n_fft_i, win_length_i)

        start = frame_idx * config.hop_size
        output[start : start + n_fft_i] += frame * window
        weight[start : start + n_fft_i] += window * window
        if progress_callback is not None and (frame_idx == out_frames - 1 or (frame_idx % 8) == 0):
            progress_callback(frame_count + out_frames + frame_idx + 1, total_steps)

    nz = weight > 1e-12
    output[nz] /= weight[nz]

    if config.center:
        trim = config.n_fft // 2
        if output.size > 2 * trim:
            output = output[trim:-trim]
        else:
            output = np.zeros(0, dtype=np.float64)

    target_length = max(1, int(round(signal.size * stretch)))
    return force_length(output, target_length)


def linear_resample_1d(signal: np.ndarray, output_samples: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros(output_samples, dtype=np.float64)
    if output_samples <= 1:
        return np.array([signal[0]], dtype=np.float64) if output_samples == 1 else np.zeros(0, dtype=np.float64)
    if signal.size == 1:
        return np.full(output_samples, signal[0], dtype=np.float64)

    x_old = np.linspace(0.0, 1.0, num=signal.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=output_samples, endpoint=True)
    return np.interp(x_new, x_old, signal).astype(np.float64)


def resample_1d(signal: np.ndarray, output_samples: int, mode: ResampleMode) -> np.ndarray:
    if output_samples < 0:
        raise ValueError("output_samples must be non-negative")
    if output_samples == signal.size:
        return signal.copy()

    use_fft = mode == "fft" or (mode == "auto" and scipy_resample is not None)
    if use_fft and scipy_resample is not None:
        return scipy_resample(signal, output_samples).astype(np.float64)

    return linear_resample_1d(signal, output_samples)


def force_length(signal: np.ndarray, length: int) -> np.ndarray:
    if length < 0:
        raise ValueError("Target length must be non-negative")
    if signal.size == length:
        return signal
    if signal.size > length:
        return signal[:length]
    return np.pad(signal, (0, length - signal.size), mode="constant")


def estimate_f0_autocorrelation(
    samples: np.ndarray,
    sample_rate: int,
    f0_min_hz: float,
    f0_max_hz: float,
) -> float:
    if samples.size < 4:
        raise ValueError("Signal is too short to estimate F0")

    centered = samples.astype(np.float64, copy=False)
    centered = centered - np.mean(centered)
    if np.allclose(centered, 0.0):
        raise ValueError("Signal appears silent; cannot estimate F0")

    analysis_len = min(centered.size, sample_rate * 3)
    frame = centered[:analysis_len]
    frame = frame * np.hanning(frame.size)

    corr = np.correlate(frame, frame, mode="full")
    corr = corr[corr.size // 2 :]

    min_lag = max(1, int(sample_rate / f0_max_hz))
    max_lag = min(corr.size - 1, int(sample_rate / f0_min_hz))
    if max_lag <= min_lag:
        raise ValueError("Invalid F0 search bounds")

    segment = corr[min_lag : max_lag + 1]
    peak_rel = int(np.argmax(segment))
    peak_val = segment[peak_rel]
    if not np.isfinite(peak_val) or peak_val <= 0:
        raise ValueError("No valid periodic peak found for F0 estimation")

    lag = min_lag + peak_rel

    # Optional parabolic refinement around the autocorrelation peak.
    if 1 <= lag < corr.size - 1:
        y0, y1, y2 = corr[lag - 1], corr[lag], corr[lag + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-12:
            lag = lag + 0.5 * (y0 - y2) / denom

    if lag <= 0:
        raise ValueError("Estimated lag is not positive")

    return float(sample_rate / lag)


def normalize_audio(
    audio: np.ndarray,
    mode: str,
    peak_dbfs: float,
    rms_dbfs: float,
) -> np.ndarray:
    if mode == "none":
        return audio

    out = audio.astype(np.float64, copy=True)
    if mode == "peak":
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.0:
            out *= db_to_amplitude(peak_dbfs) / peak
        return out

    if mode == "rms":
        rms = float(np.sqrt(np.mean(out * out))) if out.size else 0.0
        if rms > 0.0:
            out *= db_to_amplitude(rms_dbfs) / rms
        return out

    raise ValueError(f"Unknown normalization mode: {mode}")


def cepstral_envelope(magnitude: np.ndarray, lifter: int) -> np.ndarray:
    n_bins = magnitude.size
    n_fft = max(2, (n_bins - 1) * 2)
    log_mag = np.log(np.maximum(magnitude.astype(np.float64, copy=False), 1e-12))
    cep = np.fft.irfft(log_mag, n=n_fft)

    if lifter > 0 and lifter < n_fft // 2:
        lifted = np.zeros_like(cep)
        lifted[: lifter + 1] = cep[: lifter + 1]
        lifted[-lifter:] = cep[-lifter:]
        cep = lifted

    env_log = np.fft.rfft(cep, n=n_fft).real
    return np.exp(env_log)


def apply_formant_preservation(
    reference: np.ndarray,
    shifted: np.ndarray,
    config: VocoderConfig,
    lifter: int,
    strength: float,
    max_gain_db: float,
) -> np.ndarray:
    if reference.size == 0 or shifted.size == 0 or strength <= 0.0:
        return shifted

    ref_spec = stft(reference, config)
    tgt_spec = stft(shifted, config)
    ref_mag = np.abs(ref_spec)
    tgt_mag = np.abs(tgt_spec)
    tgt_phase = np.angle(tgt_spec)

    ref_frames = ref_mag.shape[1]
    tgt_frames = tgt_mag.shape[1]
    if ref_frames == 0 or tgt_frames == 0:
        return shifted

    ref_env = np.empty_like(ref_mag)
    for idx in range(ref_frames):
        ref_env[:, idx] = cepstral_envelope(ref_mag[:, idx], lifter)

    gain_limit = db_to_amplitude(max_gain_db)
    min_gain = 1.0 / gain_limit
    max_gain = gain_limit

    corrected = np.empty_like(tgt_spec)
    for idx in range(tgt_frames):
        ref_idx = (
            0
            if tgt_frames == 1
            else int(round(idx * (ref_frames - 1) / max(1, tgt_frames - 1)))
        )
        tgt_env = cepstral_envelope(tgt_mag[:, idx], lifter)
        gain = ref_env[:, ref_idx] / np.maximum(tgt_env, 1e-12)
        gain = np.clip(gain, min_gain, max_gain)
        if strength < 1.0:
            gain = np.power(gain, strength)
        corrected[:, idx] = (tgt_mag[:, idx] * gain) * np.exp(1j * tgt_phase[:, idx])

    return istft(corrected, config, expected_length=shifted.size)


def choose_pitch_ratio(args: argparse.Namespace, signal: np.ndarray, sr: int) -> PitchConfig:
    if args.pitch_shift_ratio is not None:
        return PitchConfig(ratio=args.pitch_shift_ratio)

    if args.pitch_shift_semitones is not None:
        return PitchConfig(ratio=2.0 ** (args.pitch_shift_semitones / 12.0))

    if args.target_f0 is None:
        return PitchConfig(ratio=1.0)

    if args.analysis_channel == "first":
        f0_source = signal[:, 0]
    else:
        f0_source = np.mean(signal, axis=1)

    detected_f0 = estimate_f0_autocorrelation(f0_source, sr, args.f0_min, args.f0_max)
    ratio = args.target_f0 / detected_f0
    if ratio <= 0:
        raise ValueError("Computed pitch ratio from target F0 is not positive")
    return PitchConfig(ratio=ratio, source_f0_hz=detected_f0)


def resolve_base_stretch(args: argparse.Namespace, in_samples: int, sr: int) -> float:
    if args.target_duration is not None:
        return args.target_duration * sr / max(in_samples, 1)
    return args.time_stretch


def compute_output_path(
    input_path: Path,
    output_dir: Path | None,
    suffix: str,
    output_format: str | None,
) -> Path:
    base_dir = output_dir if output_dir is not None else input_path.parent
    ext = output_format.lower().lstrip(".") if output_format else input_path.suffix.lstrip(".")
    if not ext:
        ext = "wav"
    return base_dir / f"{input_path.stem}{suffix}.{ext}"


def process_file(
    input_path: Path,
    args: argparse.Namespace,
    config: VocoderConfig,
    file_index: int = 0,
    file_total: int = 1,
) -> JobResult:
    progress_enabled = (not args.no_progress) and sys.stderr.isatty()
    progress = ProgressBar(
        label=f"{input_path.name} [{file_index + 1}/{file_total}]",
        enabled=progress_enabled,
    )

    def make_progress_callback(start: float, end: float, detail: str) -> ProgressCallback | None:
        if not progress_enabled:
            return None

        span = max(0.0, end - start)

        def _callback(done: int, total: int) -> None:
            denom = max(1, total)
            progress.set(start + span * (done / denom), detail)

        return _callback

    progress.set(0.02, "read")
    audio, sr = sf.read(str(input_path), always_2d=True)
    audio = audio.astype(np.float64, copy=False)

    if audio.shape[0] == 0:
        raise ValueError("Input file has no audio samples")

    progress.set(0.08, "analyze")
    pitch = choose_pitch_ratio(args, audio, sr)
    base_stretch = resolve_base_stretch(args, audio.shape[0], sr)
    internal_stretch = base_stretch * pitch.ratio

    if internal_stretch <= 0.0:
        raise ValueError("Computed internal stretch must be > 0")

    sync_plan: FourierSyncPlan | None = None
    if args.fourier_sync:
        progress.set(0.10, "f0 prescan")
        sync_source = audio[:, 0] if args.analysis_channel == "first" else np.mean(audio, axis=1)
        sync_plan = build_fourier_sync_plan(
            signal=sync_source,
            sample_rate=sr,
            config=config,
            f0_min_hz=args.f0_min,
            f0_max_hz=args.f0_max,
            min_fft=args.fourier_sync_min_fft,
            max_fft=args.fourier_sync_max_fft,
            smooth_span=args.fourier_sync_smooth,
            progress_callback=make_progress_callback(0.10, 0.25, "f0 prescan"),
        )

    channel_start = 0.25 if args.fourier_sync else 0.10
    channel_end = 0.88
    processed_channels: list[np.ndarray] = []
    for ch in range(audio.shape[1]):
        sub_start = channel_start + (channel_end - channel_start) * (ch / audio.shape[1])
        sub_end = channel_start + (channel_end - channel_start) * ((ch + 1) / audio.shape[1])
        detail = f"channel {ch + 1}/{audio.shape[1]}"
        source_ch = audio[:, ch]
        if args.fourier_sync:
            if sync_plan is None:  # pragma: no cover - defensive
                raise RuntimeError("Fourier-sync plan is unavailable")
            stretched = phase_vocoder_time_stretch_fourier_sync(
                audio[:, ch],
                internal_stretch,
                config,
                sync_plan,
                progress_callback=make_progress_callback(sub_start, sub_end, detail),
            )
        else:
            stretched = phase_vocoder_time_stretch(
                audio[:, ch],
                internal_stretch,
                config,
                progress_callback=make_progress_callback(sub_start, sub_end, detail),
            )
        if abs(pitch.ratio - 1.0) > 1e-10:
            pitch_len = max(1, int(round(stretched.size / pitch.ratio)))
            shifted = resample_1d(stretched, pitch_len, args.resample_mode)
        else:
            shifted = stretched

        if args.pitch_mode == "formant-preserving" and abs(pitch.ratio - 1.0) > 1e-10:
            shifted = apply_formant_preservation(
                source_ch,
                shifted,
                config,
                lifter=args.formant_lifter,
                strength=args.formant_strength,
                max_gain_db=args.formant_max_gain_db,
            )
        processed_channels.append(shifted)

    progress.set(0.90, "mix")
    out_len = max(ch_data.size for ch_data in processed_channels)
    out_audio = np.zeros((out_len, len(processed_channels)), dtype=np.float64)
    for ch, ch_data in enumerate(processed_channels):
        out_audio[: ch_data.size, ch] = ch_data

    if args.target_duration is not None:
        exact_len = max(1, int(round(args.target_duration * sr)))
        out_audio = force_length_multi(out_audio, exact_len)

    out_sr = sr
    if args.target_sample_rate is not None and args.target_sample_rate != sr:
        new_len = max(1, int(round(out_audio.shape[0] * args.target_sample_rate / sr)))
        out_audio = resample_multi(out_audio, new_len, args.resample_mode)
        out_sr = args.target_sample_rate

    out_audio = normalize_audio(out_audio, args.normalize, args.peak_dbfs, args.rms_dbfs)
    if args.clip:
        out_audio = np.clip(out_audio, -1.0, 1.0)

    output_path = compute_output_path(input_path, args.output_dir, args.suffix, args.output_format)
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --overwrite to replace it."
        )

    if not args.dry_run:
        progress.set(0.96, "write")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), out_audio, out_sr, subtype=args.subtype)

    if args.verbose:
        msg = (
            f"[info] {input_path.name}: channels={audio.shape[1]}, sr={sr}, "
            f"stretch={base_stretch:.6f}, pitch_ratio={pitch.ratio:.6f}, "
            f"internal_stretch={internal_stretch:.6f}, "
            f"phase_locking={config.phase_locking}, pitch_mode={args.pitch_mode}, "
            f"fourier_sync={'on' if args.fourier_sync else 'off'}"
        )
        if pitch.source_f0_hz is not None:
            msg += f", detected_f0={pitch.source_f0_hz:.3f}Hz"
        if sync_plan is not None and sync_plan.f0_track_hz.size:
            msg += (
                f", sync_f0_med={float(np.median(sync_plan.f0_track_hz)):.3f}Hz"
                f", sync_fft_med={int(np.median(sync_plan.frame_lengths))}"
            )
        print(msg)

    progress.finish("done")

    return JobResult(
        input_path=input_path,
        output_path=output_path,
        in_sr=sr,
        out_sr=out_sr,
        in_samples=audio.shape[0],
        out_samples=out_audio.shape[0],
        channels=audio.shape[1],
        stretch=base_stretch,
        pitch_ratio=pitch.ratio,
    )


def force_length_multi(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        return audio[:length, :]
    pad = np.zeros((length - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.vstack([audio, pad])


def resample_multi(audio: np.ndarray, output_samples: int, mode: ResampleMode) -> np.ndarray:
    out = np.zeros((output_samples, audio.shape[1]), dtype=np.float64)
    for ch in range(audio.shape[1]):
        out[:, ch] = resample_1d(audio[:, ch], output_samples, mode)
    return out


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.n_fft <= 0:
        parser.error("--n-fft must be > 0")
    if args.win_length <= 0:
        parser.error("--win-length must be > 0")
    if args.win_length > args.n_fft:
        parser.error("--win-length must be <= --n-fft")
    if args.hop_size <= 0:
        parser.error("--hop-size must be > 0")
    if args.hop_size > args.win_length:
        parser.error("--hop-size should be <= --win-length")
    if args.time_stretch <= 0:
        parser.error("--time-stretch must be > 0")
    if args.target_duration is not None and args.target_duration <= 0:
        parser.error("--target-duration must be > 0")
    if args.pitch_shift_ratio is not None and args.pitch_shift_ratio <= 0:
        parser.error("--pitch-shift-ratio must be > 0")
    if args.target_f0 is not None and args.target_f0 <= 0:
        parser.error("--target-f0 must be > 0")
    if args.f0_min <= 0 or args.f0_max <= 0 or args.f0_min >= args.f0_max:
        parser.error("--f0-min and --f0-max must satisfy 0 < f0-min < f0-max")
    if args.target_sample_rate is not None and args.target_sample_rate <= 0:
        parser.error("--target-sample-rate must be > 0")
    if args.transient_threshold <= 0:
        parser.error("--transient-threshold must be > 0")
    if args.formant_lifter < 0:
        parser.error("--formant-lifter must be >= 0")
    if not (0.0 <= args.formant_strength <= 1.0):
        parser.error("--formant-strength must be between 0.0 and 1.0")
    if args.formant_max_gain_db <= 0:
        parser.error("--formant-max-gain-db must be > 0")
    if args.fourier_sync_min_fft < 16:
        parser.error("--fourier-sync-min-fft must be >= 16")
    if args.fourier_sync_max_fft < args.fourier_sync_min_fft:
        parser.error("--fourier-sync-max-fft must be >= --fourier-sync-min-fft")
    if args.fourier_sync_smooth <= 0:
        parser.error("--fourier-sync-smooth must be > 0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-vocoder CLI for multi-file, multi-channel time stretching and pitch shifting."
        )
    )

    parser.add_argument("inputs", nargs="+", help="Input audio files (wav/flac/aiff/ogg/etc)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: same directory as each input)",
    )
    parser.add_argument(
        "--suffix",
        default="_pv",
        help="Suffix appended to output filename stem (default: _pv)",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="Output format/extension (e.g. wav, flac, aiff). Default: keep input extension.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Resolve settings without writing files")
    parser.add_argument("--verbose", action="store_true", help="Print per-file processing diagnostics")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")

    stft_group = parser.add_argument_group("STFT / vocoder parameters")
    stft_group.add_argument("--n-fft", type=int, default=2048, help="FFT size (default: 2048)")
    stft_group.add_argument(
        "--win-length",
        type=int,
        default=2048,
        help="Window length in samples (default: 2048)",
    )
    stft_group.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="Hop size in samples (default: 512)",
    )
    stft_group.add_argument(
        "--window",
        choices=["hann", "hamming", "blackman", "rect"],
        default="hann",
        help="Window type (default: hann)",
    )
    stft_group.add_argument(
        "--no-center",
        action="store_true",
        help="Disable center padding in STFT/ISTFT",
    )
    stft_group.add_argument(
        "--phase-locking",
        choices=["off", "identity"],
        default="identity",
        help="Inter-bin phase locking mode for transient fidelity (default: identity)",
    )
    stft_group.add_argument(
        "--transient-preserve",
        action="store_true",
        help="Enable transient phase resets based on spectral flux",
    )
    stft_group.add_argument(
        "--transient-threshold",
        type=float,
        default=2.0,
        help="Spectral-flux multiplier for transient detection (default: 2.0)",
    )
    stft_group.add_argument(
        "--fourier-sync",
        action="store_true",
        help=(
            "Enable fundamental frame locking. Uses generic short-time Fourier "
            "transforms with per-frame FFT sizes locked to detected F0."
        ),
    )
    stft_group.add_argument(
        "--fourier-sync-min-fft",
        type=int,
        default=256,
        help="Minimum frame FFT size for --fourier-sync (default: 256)",
    )
    stft_group.add_argument(
        "--fourier-sync-max-fft",
        type=int,
        default=8192,
        help="Maximum frame FFT size for --fourier-sync (default: 8192)",
    )
    stft_group.add_argument(
        "--fourier-sync-smooth",
        type=int,
        default=5,
        help="Smoothing span (frames) for prescanned F0 track in --fourier-sync (default: 5)",
    )

    time_group = parser.add_argument_group("Timing controls")
    time_group.add_argument(
        "--time-stretch",
        type=float,
        default=1.0,
        help="Final duration multiplier (1.0=unchanged, 2.0=2x longer)",
    )
    time_group.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Absolute target duration in seconds (overrides --time-stretch)",
    )

    pitch_group = parser.add_argument_group("Pitch controls")
    pitch_mutex = pitch_group.add_mutually_exclusive_group()
    pitch_mutex.add_argument(
        "--pitch-shift-semitones",
        "--target-pitch-shift-semitones",
        type=float,
        default=None,
        help="Pitch shift in semitones (+12 is one octave up)",
    )
    pitch_mutex.add_argument(
        "--pitch-shift-ratio",
        type=float,
        default=None,
        help="Pitch ratio (>1 up, <1 down)",
    )
    pitch_mutex.add_argument(
        "--target-f0",
        type=float,
        default=None,
        help="Target fundamental frequency in Hz. Auto-estimates source F0 per file.",
    )
    pitch_group.add_argument(
        "--analysis-channel",
        choices=["first", "mix"],
        default="mix",
        help="Channel strategy for F0 estimation with --target-f0 (default: mix)",
    )
    pitch_group.add_argument(
        "--f0-min",
        type=float,
        default=50.0,
        help="Minimum F0 search bound in Hz (default: 50)",
    )
    pitch_group.add_argument(
        "--f0-max",
        type=float,
        default=1000.0,
        help="Maximum F0 search bound in Hz (default: 1000)",
    )
    pitch_group.add_argument(
        "--pitch-mode",
        choices=["standard", "formant-preserving"],
        default="standard",
        help="Pitch mode: standard shift or formant-preserving correction (default: standard)",
    )
    pitch_group.add_argument(
        "--formant-lifter",
        type=int,
        default=32,
        help="Cepstral lifter cutoff for formant envelope extraction (default: 32)",
    )
    pitch_group.add_argument(
        "--formant-strength",
        type=float,
        default=1.0,
        help="Formant correction blend 0..1 when pitch mode is formant-preserving (default: 1.0)",
    )
    pitch_group.add_argument(
        "--formant-max-gain-db",
        type=float,
        default=12.0,
        help="Max per-bin formant correction gain in dB (default: 12)",
    )

    io_group = parser.add_argument_group("Resampling / output")
    io_group.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="Output sample rate in Hz (default: keep input rate)",
    )
    io_group.add_argument(
        "--resample-mode",
        choices=["auto", "fft", "linear"],
        default="auto",
        help="Resampling engine (auto=fft if scipy available, else linear)",
    )
    io_group.add_argument(
        "--normalize",
        choices=["none", "peak", "rms"],
        default="none",
        help="Normalize output amplitude (default: none)",
    )
    io_group.add_argument(
        "--peak-dbfs",
        type=float,
        default=-1.0,
        help="Target peak dBFS when --normalize peak (default: -1.0)",
    )
    io_group.add_argument(
        "--rms-dbfs",
        type=float,
        default=-18.0,
        help="Target RMS dBFS when --normalize rms (default: -18.0)",
    )
    io_group.add_argument(
        "--clip",
        action="store_true",
        help="Clip output to [-1, 1] after processing",
    )
    io_group.add_argument(
        "--subtype",
        default=None,
        help="Output file subtype for soundfile (e.g. PCM_16, PCM_24, FLOAT)",
    )

    return parser


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = [Path(match) for match in glob.glob(pattern, recursive=True)]
        else:
            matches = [Path(pattern)]
        for match in matches:
            if match.is_file():
                paths.append(match)
    # Keep stable ordering and remove duplicates while preserving sequence.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    ensure_runtime_dependencies()

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        parser.error("No readable input files matched the provided paths/patterns")

    config = VocoderConfig(
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_size=args.hop_size,
        window=args.window,
        center=not args.no_center,
        phase_locking=args.phase_locking,
        transient_preserve=args.transient_preserve,
        transient_threshold=args.transient_threshold,
    )

    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()

    results: list[JobResult] = []
    failures: list[tuple[Path, Exception]] = []

    for idx, path in enumerate(input_paths):
        try:
            result = process_file(path, args, config, file_index=idx, file_total=len(input_paths))
            results.append(result)
        except Exception as exc:  # pragma: no cover - runtime I/O errors
            failures.append((path, exc))

    for result in results:
        in_dur = result.in_samples / result.in_sr
        out_dur = result.out_samples / result.out_sr
        print(
            f"[ok] {result.input_path} -> {result.output_path} | "
            f"ch={result.channels}, sr={result.in_sr}->{result.out_sr}, "
            f"dur={in_dur:.3f}s->{out_dur:.3f}s, "
            f"stretch={result.stretch:.6f}, pitch_ratio={result.pitch_ratio:.6f}"
        )

    for path, exc in failures:
        print(f"[error] {path}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
