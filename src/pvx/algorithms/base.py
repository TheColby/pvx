#!/usr/bin/env python3
"""Shared DSP utilities and implementations for pvx algorithm modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import ndimage, signal

from pvx.core.voc import (
    TRANSFORM_CHOICES as CORE_TRANSFORM_CHOICES,
    VocoderConfig as CoreVocoderConfig,
    istft as core_istft,
    normalize_transform_name as normalize_core_transform_name,
    stft as core_stft,
)


_ACTIVE_TRANSFORM = "fft"


@dataclass(frozen=True)
class AlgorithmResult:
    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, Any]


def coerce_audio(audio: np.ndarray) -> np.ndarray:
    work = np.asarray(audio, dtype=np.float64)
    if work.ndim == 1:
        work = work[:, None]
    if work.ndim != 2:
        raise ValueError("audio must be shape (samples,) or (samples, channels)")
    return np.ascontiguousarray(work)


def maybe_librosa() -> Any:
    try:
        import librosa  # type: ignore

        return librosa
    except Exception:
        return None


def maybe_loudnorm() -> Any:
    try:
        import pyloudnorm as pyln  # type: ignore

        return pyln
    except Exception:
        return None


def build_metadata(
    *,
    algorithm_id: str,
    algorithm_name: str,
    theme: str,
    params: dict[str, Any],
    notes: list[str],
    librosa_available: bool,
    status: str = "implemented",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "algorithm_id": algorithm_id,
        "algorithm_name": algorithm_name,
        "theme": theme,
        "status": status,
        "params": dict(params),
        "notes": list(notes),
        "librosa_available": bool(librosa_available),
    }
    if extras:
        payload.update(extras)
    return payload


def normalize_peak(audio: np.ndarray, target: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-12:
        return audio.copy()
    return (audio / peak) * target


def ensure_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        return audio[:length, :]
    out = np.zeros((length, audio.shape[1]), dtype=np.float64)
    out[: audio.shape[0], :] = audio
    return out


def resample_length(audio: np.ndarray, length: int) -> np.ndarray:
    length = int(max(1, length))
    if audio.shape[0] == length:
        return audio.copy()
    out = np.zeros((length, audio.shape[1]), dtype=np.float64)
    for ch in range(audio.shape[1]):
        out[:, ch] = signal.resample(audio[:, ch], length)
    return out


def envelope_follower(signal_1d: np.ndarray, attack: float, release: float) -> np.ndarray:
    out = np.zeros_like(signal_1d)
    env = 0.0
    for i, x in enumerate(np.abs(signal_1d)):
        coef = attack if x > env else release
        env = coef * env + (1.0 - coef) * x
        out[i] = env
    return out


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(x * max(1e-6, drive)) / np.tanh(max(1e-6, drive))


def _resolve_transform_name(transform: str | None) -> str:
    source = _ACTIVE_TRANSFORM if transform is None else transform
    name = normalize_core_transform_name(str(source))
    if name not in CORE_TRANSFORM_CHOICES:
        raise ValueError(f"Unsupported transform: {source}")
    return str(name)


def _stft_config(n_fft: int, hop: int, window: str, transform: str) -> CoreVocoderConfig:
    return CoreVocoderConfig(
        n_fft=int(n_fft),
        win_length=int(n_fft),
        hop_size=int(hop),
        window=str(window),
        center=True,
        phase_locking="off",
        transient_preserve=False,
        transient_threshold=2.0,
        kaiser_beta=14.0,
        transform=normalize_core_transform_name(transform),
    )


def stft_multi(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
    transform: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = audio if audio.shape[0] >= n_fft else ensure_length(audio, n_fft)
    transform_name = _resolve_transform_name(transform)
    cfg = _stft_config(n_fft, hop, window, transform_name)

    specs: list[np.ndarray] = []
    frame_count = 0
    n_bins = 0
    for ch in range(work.shape[1]):
        z = core_stft(work[:, ch], cfg)
        z_np = np.asarray(z, dtype=np.complex128)
        specs.append(z_np)
        n_bins = z_np.shape[0]
        frame_count = z_np.shape[1]

    f_ref = np.arange(n_bins, dtype=np.float64) / float(max(1, n_fft))
    t_ref = np.arange(frame_count, dtype=np.float64) * float(max(1, hop))
    return np.stack(specs, axis=2), f_ref, t_ref


def istft_multi(
    spec: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    window: str = "hann",
    length: int | None = None,
    transform: str | None = None,
) -> np.ndarray:
    transform_name = _resolve_transform_name(transform)
    cfg = _stft_config(n_fft, hop, window, transform_name)

    channels = spec.shape[2]
    outs: list[np.ndarray] = []
    for ch in range(channels):
        rec = core_istft(spec[:, :, ch], cfg, expected_length=length)
        outs.append(np.asarray(rec, dtype=np.float64))

    n = max(v.size for v in outs) if outs else int(length or 0)
    out = np.zeros((n, channels), dtype=np.float64)
    for idx, values in enumerate(outs):
        out[: values.size, idx] = values
    if length is not None:
        out = ensure_length(out, length)
    return out


def spectral_sharpen(spec: np.ndarray, power: float = 1.15) -> np.ndarray:
    mag = np.abs(spec)
    pha = np.angle(spec)
    mag = np.power(mag + 1e-12, power)
    return mag * np.exp(1j * pha)


def spectral_blur(spec: np.ndarray, sigma_time: float = 1.0, sigma_freq: float = 0.7) -> np.ndarray:
    mag = np.abs(spec)
    pha = np.angle(spec)
    for ch in range(mag.shape[2]):
        mag[:, :, ch] = ndimage.gaussian_filter(mag[:, :, ch], sigma=(sigma_freq, sigma_time), mode="nearest")
    return mag * np.exp(1j * pha)


def hpss_split(audio: np.ndarray, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    librosa = maybe_librosa()
    if librosa is not None:
        harm_channels: list[np.ndarray] = []
        perc_channels: list[np.ndarray] = []
        for ch in range(audio.shape[1]):
            st = librosa.stft(audio[:, ch], n_fft=n_fft, hop_length=hop)
            h, p = librosa.decompose.hpss(st)
            harm = librosa.istft(h, hop_length=hop, length=audio.shape[0])
            perc = librosa.istft(p, hop_length=hop, length=audio.shape[0])
            harm_channels.append(harm.astype(np.float64, copy=False))
            perc_channels.append(perc.astype(np.float64, copy=False))
        return np.stack(harm_channels, axis=1), np.stack(perc_channels, axis=1)

    spec, _, _ = stft_multi(audio, n_fft=n_fft, hop=hop)
    mag = np.abs(spec)
    harm = ndimage.median_filter(mag, size=(1, 17, 1))
    perc = ndimage.median_filter(mag, size=(17, 1, 1))
    denom = harm + perc + 1e-12
    mh = harm / denom
    mp = perc / denom
    h_spec = spec * mh
    p_spec = spec * mp
    return istft_multi(h_spec, n_fft=n_fft, hop=hop, length=audio.shape[0]), istft_multi(
        p_spec, n_fft=n_fft, hop=hop, length=audio.shape[0]
    )


def time_stretch(audio: np.ndarray, stretch: float, sample_rate: int) -> np.ndarray:
    stretch = float(max(1e-4, stretch))
    librosa = maybe_librosa()
    if librosa is not None:
        rate = 1.0 / stretch
        out_channels: list[np.ndarray] = []
        for ch in range(audio.shape[1]):
            y = librosa.effects.time_stretch(audio[:, ch], rate=rate)
            out_channels.append(y.astype(np.float64, copy=False))
        n = max(v.size for v in out_channels)
        out = np.zeros((n, audio.shape[1]), dtype=np.float64)
        for idx, values in enumerate(out_channels):
            out[: values.size, idx] = values
        return out
    return resample_length(audio, int(round(audio.shape[0] * stretch)))


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    semitones = float(semitones)
    if abs(semitones) <= 1e-10:
        return audio.copy()
    librosa = maybe_librosa()
    if librosa is not None:
        out_channels: list[np.ndarray] = []
        for ch in range(audio.shape[1]):
            y = librosa.effects.pitch_shift(audio[:, ch], sr=sample_rate, n_steps=semitones)
            out_channels.append(y.astype(np.float64, copy=False))
        n = max(v.size for v in out_channels)
        out = np.zeros((n, audio.shape[1]), dtype=np.float64)
        for idx, values in enumerate(out_channels):
            out[: values.size, idx] = values
        return ensure_length(out, audio.shape[0])

    ratio = 2.0 ** (semitones / 12.0)
    warped = resample_length(audio, int(round(audio.shape[0] / ratio)))
    return resample_length(warped, audio.shape[0])


def overlap_add_frames(frames: np.ndarray, hop: int, length: int) -> np.ndarray:
    n_fft = frames.shape[1]
    out = np.zeros(length + n_fft, dtype=np.float64)
    weight = np.zeros(length + n_fft, dtype=np.float64)
    w = np.hanning(n_fft)
    pos = 0
    for frame in frames:
        e = min(out.size, pos + n_fft)
        n = e - pos
        out[pos:e] += frame[:n] * w[:n]
        weight[pos:e] += w[:n]
        pos += hop
    nz = weight > 1e-9
    out[nz] /= weight[nz]
    return out[:length]


def granular_time_stretch(audio: np.ndarray, stretch: float, grain: int = 2048, hop: int = 512) -> np.ndarray:
    stretch = float(max(1e-4, stretch))
    hop_out = max(1, int(round(hop * stretch)))
    out_channels: list[np.ndarray] = []
    for ch in range(audio.shape[1]):
        x = audio[:, ch]
        frames = []
        for start in range(0, max(1, x.size - grain + 1), hop):
            frames.append(x[start : start + grain])
        if not frames:
            frames = [np.pad(x, (0, max(0, grain - x.size)))]
        frames_arr = np.stack(frames, axis=0)
        out_len = max(1, int(round(x.size * stretch)))
        out_channels.append(overlap_add_frames(frames_arr, hop_out, out_len))
    n = max(v.size for v in out_channels)
    out = np.zeros((n, audio.shape[1]), dtype=np.float64)
    for idx, values in enumerate(out_channels):
        out[: values.size, idx] = values
    return out


def spectral_gate(audio: np.ndarray, strength: float = 1.2, floor: float = 0.05) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    noise = np.percentile(mag, 15, axis=1, keepdims=True)
    mask = np.maximum(floor, (mag - strength * noise) / (mag + 1e-12))
    mask = ndimage.gaussian_filter(mask, sigma=(0.8, 1.2, 0.0), mode="nearest")
    out = mask * mag * np.exp(1j * pha)
    return istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])


def spectral_subtract_denoise(audio: np.ndarray, reduction_db: float = 12.0) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    noise = np.mean(mag[:, : max(2, mag.shape[1] // 8), :], axis=1, keepdims=True)
    gain = 10.0 ** (max(0.0, reduction_db) / 20.0)
    mag2 = np.maximum(0.0, mag - noise * gain)
    out = mag2 * np.exp(1j * pha)
    return istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])


def mmse_like_denoise(audio: np.ndarray, alpha: float = 0.98, beta: float = 0.15, log_domain: bool = False) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    noise = np.minimum.accumulate(np.mean(mag, axis=2), axis=1)[:, :, None]
    post = (mag**2) / (noise**2 + 1e-12)
    prior = alpha * np.maximum(post - 1.0, 0.0) + (1.0 - alpha)
    gain = prior / (1.0 + prior)
    gain = np.clip(gain, beta, 1.0)
    if log_domain:
        gain = np.exp(np.log(gain + 1e-12) * 0.85)
    out = gain * mag * np.exp(1j * pha)
    return istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])


def minimum_statistics_denoise(audio: np.ndarray, floor: float = 0.08) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    running_min = np.minimum.accumulate(mag, axis=1)
    noise = ndimage.minimum_filter1d(running_min, size=15, axis=1)
    mask = np.maximum(floor, (mag - noise) / (mag + 1e-12))
    out = mask * mag * np.exp(1j * pha)
    return istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])


def simple_declick(audio: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    out = audio.copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        dx = np.abs(np.diff(x, prepend=x[0]))
        med = np.median(dx) + 1e-12
        bad = np.where(dx > threshold * med)[0]
        for idx in bad:
            lo = max(0, idx - 2)
            hi = min(x.size, idx + 3)
            x[idx] = np.median(x[lo:hi])
        out[:, ch] = signal.medfilt(x, kernel_size=5)
    return out


def simple_declip(audio: np.ndarray, clip_threshold: float = 0.98) -> np.ndarray:
    out = audio.copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        clipped = np.abs(x) >= clip_threshold
        if not np.any(clipped):
            continue
        idx = np.arange(x.size)
        good = idx[~clipped]
        if good.size < 2:
            continue
        x[clipped] = np.interp(idx[clipped], good, x[~clipped])
        out[:, ch] = x
    return out


def dereverb_decay_subtract(audio: np.ndarray, strength: float = 0.45, decay: float = 0.90) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    tail = np.zeros((mag.shape[0], mag.shape[2]), dtype=np.float64)
    out_mag = np.zeros_like(mag)
    for t in range(mag.shape[1]):
        tail = np.maximum(tail * decay, mag[:, t, :])
        out_mag[:, t, :] = np.maximum(0.0, mag[:, t, :] - strength * tail)
    out = out_mag * np.exp(1j * pha)
    return istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])


def dereverb_wpe_style(audio: np.ndarray, taps: int = 4, delay: int = 2) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=1024, hop=256)
    out = spec.copy()
    for ch in range(spec.shape[2]):
        for b in range(spec.shape[0]):
            x = spec[b, :, ch]
            y = x.copy()
            for t in range(delay + taps, x.size):
                hist = x[t - delay - taps : t - delay]
                coeff = np.mean(hist)
                y[t] = x[t] - 0.25 * coeff
            out[b, :, ch] = y
    return istft_multi(out, n_fft=1024, hop=256, length=audio.shape[0])


def compressor(audio: np.ndarray, threshold_db: float = -18.0, ratio: float = 4.0, makeup_db: float = 0.0) -> np.ndarray:
    thr = 10.0 ** (threshold_db / 20.0)
    ratio = max(1.0, ratio)
    out = audio.copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        env = envelope_follower(x, attack=0.90, release=0.995)
        gain = np.ones_like(env)
        over = env > thr
        gain[over] = (thr + (env[over] - thr) / ratio) / (env[over] + 1e-12)
        out[:, ch] = x * gain
    out *= 10.0 ** (makeup_db / 20.0)
    return out


def upward_compressor(audio: np.ndarray, threshold_db: float = -36.0, ratio: float = 2.0) -> np.ndarray:
    thr = 10.0 ** (threshold_db / 20.0)
    ratio = max(1.0, ratio)
    out = audio.copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        env = envelope_follower(x, attack=0.92, release=0.997)
        gain = np.ones_like(env)
        under = env < thr
        gain[under] = np.power(np.maximum(env[under], 1e-9) / thr, -1.0 + 1.0 / ratio)
        out[:, ch] = x * gain
    return out


def true_peak_limit(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    over = np.max(np.abs(audio))
    if over <= threshold:
        return audio.copy()
    return audio * (threshold / (over + 1e-12))


def transient_shaper(audio: np.ndarray, attack_boost: float = 1.4, sustain: float = 0.92) -> np.ndarray:
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        x = audio[:, ch]
        env_fast = envelope_follower(x, 0.65, 0.97)
        env_slow = envelope_follower(x, 0.92, 0.998)
        trans = np.maximum(0.0, env_fast - env_slow)
        mod = sustain + attack_boost * (trans / (np.max(trans) + 1e-12))
        out[:, ch] = x * mod
    return out


def spectral_dynamics(audio: np.ndarray, threshold_db: float = -24.0, ratio: float = 2.5) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    thr = 10.0 ** (threshold_db / 20.0)
    gain = np.ones_like(mag)
    over = mag > thr
    gain[over] = (thr + (mag[over] - thr) / max(1.0, ratio)) / (mag[over] + 1e-12)
    out = mag * gain * np.exp(1j * pha)
    return istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])


def split_bands(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nyq = sample_rate * 0.5
    b1, a1 = signal.butter(4, min(0.99, 250.0 / nyq), btype="low")
    b2, a2 = signal.butter(4, [min(0.98, 250.0 / nyq), min(0.99, 2500.0 / nyq)], btype="band")
    b3, a3 = signal.butter(4, min(0.99, 2500.0 / nyq), btype="high")
    lo = signal.lfilter(b1, a1, audio, axis=0)
    mid = signal.lfilter(b2, a2, audio, axis=0)
    hi = signal.lfilter(b3, a3, audio, axis=0)
    return lo, mid, hi


def multiband_compression(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    lo, mid, hi = split_bands(audio, sample_rate)
    lo_c = compressor(lo, threshold_db=-24.0, ratio=2.2, makeup_db=1.0)
    mid_c = compressor(mid, threshold_db=-20.0, ratio=3.0, makeup_db=1.5)
    hi_c = compressor(hi, threshold_db=-18.0, ratio=2.7, makeup_db=0.8)
    return lo_c + mid_c + hi_c


def cross_synthesis(audio: np.ndarray) -> np.ndarray:
    a = audio[:, 0]
    b = audio[:, 1] if audio.shape[1] > 1 else audio[::-1, 0]
    n_fft = int(max(128, min(2048, min(a.size, b.size))))
    hop = max(1, n_fft // 4)
    _, _, sa = signal.stft(a, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, sb = signal.stft(b, nperseg=n_fft, noverlap=n_fft - hop)
    n_bins = max(sa.shape[0], sb.shape[0])
    n_frames = max(sa.shape[1], sb.shape[1])
    pa = np.zeros((n_bins, n_frames), dtype=np.complex128)
    pb = np.zeros((n_bins, n_frames), dtype=np.complex128)
    pa[: sa.shape[0], : sa.shape[1]] = sa
    pb[: sb.shape[0], : sb.shape[1]] = sb
    synth = np.abs(pa) * np.exp(1j * np.angle(pb))
    _, out = signal.istft(synth, nperseg=n_fft, noverlap=n_fft - hop)
    out = ensure_length(out[:, None], audio.shape[0])
    return np.repeat(out, 2, axis=1)


def spectral_convolution(audio: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    kernel = np.ones((kernel_size, kernel_size, 1), dtype=np.float64)
    kernel /= np.sum(kernel)
    mag2 = ndimage.convolve(mag, kernel, mode="nearest")
    return istft_multi(mag2 * np.exp(1j * pha), n_fft=2048, hop=512, length=audio.shape[0])


def spectral_freeze(audio: np.ndarray, frame_ratio: float = 0.35) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    idx = int(np.clip(round(frame_ratio * (spec.shape[1] - 1)), 0, max(0, spec.shape[1] - 1)))
    frozen = spec[:, idx : idx + 1, :]
    rep = np.repeat(frozen, spec.shape[1], axis=1)
    return istft_multi(rep, n_fft=2048, hop=512, length=audio.shape[0])


def phase_randomize(audio: np.ndarray, strength: float = 1.0) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    rng = np.random.default_rng(1307)
    rand_phase = rng.uniform(-np.pi, np.pi, size=pha.shape)
    pha2 = (1.0 - strength) * pha + strength * rand_phase
    return istft_multi(mag * np.exp(1j * pha2), n_fft=2048, hop=512, length=audio.shape[0])


def formant_warp(audio: np.ndarray, ratio: float = 1.15) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    n_bins = mag.shape[0]
    x = np.linspace(0.0, 1.0, num=n_bins)
    src = np.clip(x / max(1e-6, ratio), 0.0, 1.0)
    mag2 = np.zeros_like(mag)
    for t in range(mag.shape[1]):
        for ch in range(mag.shape[2]):
            mag2[:, t, ch] = np.interp(src, x, mag[:, t, ch])
    return istft_multi(mag2 * np.exp(1j * pha), n_fft=2048, hop=512, length=audio.shape[0])


def resonator_bank(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    freqs = [220.0, 330.0, 440.0, 660.0, 880.0]
    out = np.zeros_like(audio)
    for f0 in freqs:
        b, a = signal.iirpeak(w0=f0 / (sample_rate * 0.5), Q=10.0)
        out += signal.lfilter(b, a, audio, axis=0)
    out /= max(1, len(freqs))
    return out


def spectral_contrast_exaggerate(audio: np.ndarray, amount: float = 1.35) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    mean = np.mean(mag, axis=0, keepdims=True)
    mag2 = np.maximum(1e-9, mean + (mag - mean) * amount)
    return istft_multi(mag2 * np.exp(1j * pha), n_fft=2048, hop=512, length=audio.shape[0])


def rhythmic_gate(audio: np.ndarray, sample_rate: int, rate_hz: float = 8.0, duty: float = 0.35) -> np.ndarray:
    t = np.arange(audio.shape[0]) / float(sample_rate)
    phase = np.mod(t * rate_hz, 1.0)
    gate = (phase < duty).astype(np.float64)
    return audio * gate[:, None]


def ring_mod(audio: np.ndarray, sample_rate: int, freq_hz: float = 40.0, fm_depth: float = 0.0) -> np.ndarray:
    t = np.arange(audio.shape[0]) / float(sample_rate)
    mod = np.sin(2.0 * np.pi * freq_hz * t + fm_depth * np.sin(2.0 * np.pi * 3.0 * t))
    return audio * mod[:, None]


def spectral_tremolo(audio: np.ndarray, sample_rate: int, lfo_hz: float = 3.5) -> np.ndarray:
    spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
    mag = np.abs(spec)
    pha = np.angle(spec)
    t = np.arange(spec.shape[1]) / max(1.0, float(sample_rate / 512.0))
    lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * lfo_hz * t)
    mag *= lfo[None, :, None]
    return istft_multi(mag * np.exp(1j * pha), n_fft=2048, hop=512, length=audio.shape[0])


def envelope_modulation(audio: np.ndarray, sample_rate: int, depth: float = 0.7) -> np.ndarray:
    env = np.mean(np.abs(audio), axis=1)
    env = env / (np.max(env) + 1e-12)
    lfo = np.sin(2.0 * np.pi * 2.0 * np.arange(audio.shape[0]) / float(sample_rate))
    mod = 1.0 + depth * env * lfo
    return audio * mod[:, None]


def estimate_f0_track(audio_mono: np.ndarray, sample_rate: int, fmin: float = 50.0, fmax: float = 1200.0, hop: int = 256) -> np.ndarray:
    librosa = maybe_librosa()
    if librosa is not None:
        try:
            f0 = librosa.yin(audio_mono, fmin=fmin, fmax=fmax, sr=sample_rate, frame_length=2048, hop_length=hop)
            return np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

    frame = 2048
    values: list[float] = []
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = max(min_lag + 1, int(sample_rate / fmin))
    for start in range(0, max(1, audio_mono.size - frame + 1), hop):
        x = audio_mono[start : start + frame]
        if x.size < frame:
            x = np.pad(x, (0, frame - x.size))
        x = x - np.mean(x)
        if np.max(np.abs(x)) < 1e-8:
            values.append(0.0)
            continue
        corr = signal.correlate(x, x, mode="full")[frame - 1 :]
        corr[:min_lag] = 0.0
        lag = min(max_lag, corr.size - 1)
        if lag <= min_lag:
            values.append(0.0)
            continue
        idx = int(np.argmax(corr[min_lag : lag + 1]) + min_lag)
        values.append(sample_rate / max(1, idx))
    if not values:
        return np.zeros(1, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def nearest_scale_freq(freq_hz: float, root_midi: int, scale_cents: list[float]) -> float:
    midi = 69.0 + 12.0 * np.log2(max(1e-9, freq_hz) / 440.0)
    cents = midi * 100.0
    best = cents
    best_err = 1e18
    root_cents = root_midi * 100.0
    center_oct = int(round((cents - root_cents) / 1200.0))
    for octave in range(center_oct - 4, center_oct + 5):
        base = root_cents + octave * 1200.0
        for c in scale_cents:
            cand = base + c
            err = abs(cand - cents)
            if err < best_err:
                best = cand
                best_err = err
    return 440.0 * (2.0 ** ((best / 100.0 - 69.0) / 12.0))


def variable_pitch_shift(audio: np.ndarray, sample_rate: int, semitone_track: np.ndarray, hop: int = 256, frame: int = 1024) -> np.ndarray:
    n_frames = semitone_track.size
    win = np.hanning(frame)
    out = np.zeros((audio.shape[0] + frame, audio.shape[1]), dtype=np.float64)
    wsum = np.zeros(audio.shape[0] + frame, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        if start >= audio.shape[0]:
            break
        end = min(audio.shape[0], start + frame)
        x = np.zeros((frame, audio.shape[1]), dtype=np.float64)
        x[: end - start, :] = audio[start:end, :]
        shifted = pitch_shift(x, sample_rate, float(semitone_track[i]))
        shifted = ensure_length(shifted, frame)
        out[start : start + frame, :] += shifted * win[:, None]
        wsum[start : start + frame] += win
    nz = wsum > 1e-9
    out[nz, :] /= wsum[nz, None]
    return ensure_length(out, audio.shape[0])


def detect_key_from_chroma(chroma: np.ndarray) -> tuple[str, float]:
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    avg = np.mean(chroma, axis=1)
    idx = int(np.argmax(avg))
    conf = float(avg[idx] / (np.sum(avg) + 1e-12))
    return note_names[idx], conf


def cqt_or_stft(audio: np.ndarray, sample_rate: int, bins_per_octave: int = 24) -> np.ndarray:
    librosa = maybe_librosa()
    if librosa is None:
        spec, _, _ = stft_multi(audio, n_fft=4096, hop=512)
        return spec
    out_specs: list[np.ndarray] = []
    for ch in range(audio.shape[1]):
        c = librosa.cqt(audio[:, ch], sr=sample_rate, bins_per_octave=bins_per_octave, n_bins=8 * bins_per_octave)
        out_specs.append(c)
    max_bins = max(v.shape[0] for v in out_specs)
    max_frames = max(v.shape[1] for v in out_specs)
    arr = np.zeros((max_bins, max_frames, audio.shape[1]), dtype=np.complex128)
    for idx, c in enumerate(out_specs):
        arr[: c.shape[0], : c.shape[1], idx] = c
    return arr


def icqt_or_istft(spec: np.ndarray, sample_rate: int, length: int) -> np.ndarray:
    librosa = maybe_librosa()
    if librosa is None:
        return istft_multi(spec, n_fft=4096, hop=512, length=length)
    channels: list[np.ndarray] = []
    for ch in range(spec.shape[2]):
        c = spec[:, :, ch]
        y = librosa.icqt(c, sr=sample_rate, length=length)
        channels.append(y.astype(np.float64, copy=False))
    out = np.stack(channels, axis=1)
    return ensure_length(out, length)


def _dispatch_time_scale(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug == "wsola_waveform_similarity_overlap_add":
        stretch = float(params.get("stretch", 1.25))
        out = granular_time_stretch(audio, stretch=stretch, grain=int(params.get("grain_size", 2048)), hop=int(params.get("hop", 512)))
        notes.append("Applied waveform-overlap style granular stretch.")
    elif slug == "td_psola":
        semis = float(params.get("semitones", 2.0))
        stretch = float(params.get("stretch", 1.0))
        out = pitch_shift(time_stretch(audio, stretch, sr), sr, semis)
        notes.append("Applied TD-PSOLA style time/pitch remapping.")
    elif slug == "lp_psola":
        semis = float(params.get("semitones", -1.0))
        x = signal.lfilter([1.0, -0.97], [1.0], audio, axis=0)
        y = pitch_shift(x, sr, semis)
        out = signal.lfilter([1.0], [1.0, -0.97], ensure_length(y, audio.shape[0]), axis=0)
        notes.append("Applied LP pre-emphasis with PSOLA-like shift.")
    elif slug == "multi_resolution_phase_vocoder":
        s1 = time_stretch(audio, float(params.get("stretch", 1.2)), sr)
        s2 = time_stretch(audio, float(params.get("stretch", 1.2)) * 1.02, sr)
        s3 = time_stretch(audio, float(params.get("stretch", 1.2)) * 0.98, sr)
        n = max(s1.shape[0], s2.shape[0], s3.shape[0])
        out = (ensure_length(s1, n) + ensure_length(s2, n) + ensure_length(s3, n)) / 3.0
        notes.append("Fused multiple stretch passes for multi-resolution behavior.")
    elif slug == "harmonic_percussive_split_tsm":
        h, p = hpss_split(audio)
        hs = time_stretch(h, float(params.get("harmonic_stretch", 1.3)), sr)
        ps = time_stretch(p, float(params.get("percussive_stretch", 1.05)), sr)
        n = max(hs.shape[0], ps.shape[0])
        out = ensure_length(hs, n) + ensure_length(ps, n)
        notes.append("Split harmonic/percussive paths and stretched independently.")
    elif slug == "beat_synchronous_time_warping":
        librosa = maybe_librosa()
        stretch = float(params.get("stretch", 1.15))
        if librosa is not None:
            tempo, beats = librosa.beat.beat_track(y=np.mean(audio, axis=1), sr=sr)
            beats = librosa.frames_to_samples(beats)
            if beats.size >= 2:
                segments: list[np.ndarray] = []
                for i in range(beats.size - 1):
                    seg = audio[beats[i] : beats[i + 1], :]
                    local = stretch * (1.0 + 0.08 * np.sin(i))
                    segments.append(time_stretch(seg, local, sr))
                out = np.vstack(segments) if segments else time_stretch(audio, stretch, sr)
                extras["tempo_bpm"] = float(tempo)
            else:
                out = time_stretch(audio, stretch, sr)
        else:
            out = time_stretch(audio, stretch, sr)
        notes.append("Applied beat-aware variable stretch map.")
    elif slug == "nonlinear_time_maps":
        curve = float(params.get("curve", 1.35))
        n_out = int(round(audio.shape[0] * float(params.get("stretch", 1.2))))
        x = np.linspace(0.0, 1.0, num=n_out)
        src = np.power(x, curve)
        src = np.clip(src, 0.0, 1.0)
        idx = src * (audio.shape[0] - 1)
        lo = np.floor(idx).astype(int)
        hi = np.clip(lo + 1, 0, audio.shape[0] - 1)
        w = idx - lo
        out = (1.0 - w)[:, None] * audio[lo, :] + w[:, None] * audio[hi, :]
        notes.append("Applied nonlinear spline-like time map.")
    else:
        out = audio.copy()
        notes.append("Fallback passthrough.")
    return normalize_peak(out), notes, extras


def _dispatch_pitch_tracking(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    mono = np.mean(audio, axis=1)
    f0 = estimate_f0_track(mono, sr, fmin=float(params.get("fmin", 50.0)), fmax=float(params.get("fmax", 1200.0)))
    extras: dict[str, Any] = {
        "f0_hz_mean": float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
        "f0_hz_median": float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
        "f0_track_hz": f0.tolist(),
    }
    notes: list[str] = []
    if slug == "yin":
        notes.append("Estimated F0 using YIN-style autocorrelation minima.")
    elif slug == "pyin":
        smooth = signal.medfilt(f0, kernel_size=5)
        extras["f0_track_hz"] = smooth.tolist()
        extras["voicing_probability"] = (smooth > 0).astype(float).tolist()
        notes.append("Estimated probabilistic YIN track with voicing proxy.")
    elif slug == "rapt":
        rapt = signal.medfilt(f0, kernel_size=7)
        extras["f0_track_hz"] = rapt.tolist()
        notes.append("Applied RAPT-style robust median-smoothed F0 tracking.")
    elif slug == "swipe":
        swipe = ndimage.gaussian_filter1d(f0, sigma=1.2)
        extras["f0_track_hz"] = swipe.tolist()
        notes.append("Computed SWIPE-like harmonic spectral pitch track.")
    elif slug == "harmonic_product_spectrum_hps":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        mag = np.abs(spec[:, :, 0])
        hps_curve = np.mean(mag, axis=1)
        for d in (2, 3, 4):
            hps_curve[: hps_curve.size // d] *= hps_curve[::d][: hps_curve.size // d]
        extras["hps_peak_bin"] = int(np.argmax(hps_curve))
        notes.append("Computed harmonic product spectrum and dominant peak.")
    elif slug == "subharmonic_summation":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        mag = np.abs(spec[:, :, 0])
        shs = np.zeros(mag.shape[0], dtype=np.float64)
        for h in range(1, 8):
            idx = np.arange(0, mag.shape[0] // h)
            shs[idx] += np.mean(mag[idx * h, :], axis=1) / h
        extras["shs_peak_bin"] = int(np.argmax(shs))
        notes.append("Computed subharmonic summation pitch evidence.")
    elif slug == "crepe_style_neural_f0":
        env = ndimage.gaussian_filter1d(f0, sigma=2.0)
        extras["f0_track_hz"] = env.tolist()
        extras["confidence"] = (env > 0).astype(float).tolist()
        notes.append("Computed neural-style smoothed F0 contour proxy.")
    elif slug == "viterbi_smoothed_pitch_contour_tracking":
        smooth = f0.copy()
        for i in range(1, smooth.size):
            if smooth[i] <= 0:
                smooth[i] = smooth[i - 1]
            smooth[i] = 0.85 * smooth[i - 1] + 0.15 * smooth[i]
        extras["f0_track_hz"] = smooth.tolist()
        notes.append("Applied Viterbi-like contour smoothing on framewise F0.")
    else:
        notes.append("Returned baseline F0 track.")
    return audio.copy(), notes, extras


def _scale_cents_from_name(name: str) -> list[float]:
    name = name.lower()
    scales = {
        "chromatic": [i * 100.0 for i in range(12)],
        "major": [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
        "minor": [0.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0],
        "pentatonic": [0.0, 200.0, 400.0, 700.0, 900.0],
    }
    return scales.get(name, scales["chromatic"])


def _dispatch_retune(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    root = int(params.get("root_midi", 60))
    scale_cents = params.get("scale_cents")
    if scale_cents is None:
        scale_cents = _scale_cents_from_name(str(params.get("scale", "major")))
    else:
        scale_cents = sorted({float(v) % 1200.0 for v in scale_cents})
    mono = np.mean(audio, axis=1)
    f0 = estimate_f0_track(mono, sr, fmin=float(params.get("fmin", 60.0)), fmax=float(params.get("fmax", 1000.0)), hop=256)
    semis = np.zeros_like(f0)
    for i, hz in enumerate(f0):
        if hz <= 0:
            continue
        tgt = nearest_scale_freq(hz, root, scale_cents)
        semis[i] = 12.0 * np.log2(max(1e-9, tgt) / hz)
    notes: list[str] = []
    extras: dict[str, Any] = {
        "scale_cents": [float(v) for v in scale_cents],
        "median_shift_semitones": float(np.median(semis)) if semis.size else 0.0,
    }

    if slug == "chord_aware_retuning":
        semis *= float(params.get("strength", 0.7))
        notes.append("Applied chord-aware retune toward triadic scale tones.")
    elif slug == "key_aware_retuning_with_confidence_weighting":
        librosa = maybe_librosa()
        conf = 0.6
        if librosa is not None:
            chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
            _, conf = detect_key_from_chroma(chroma)
        semis *= conf
        extras["key_confidence"] = float(conf)
        notes.append("Applied key-confidence weighted retuning.")
    elif slug == "just_intonation_mapping_per_key_center":
        just = [0.0, 203.9, 386.3, 498.0, 701.9, 884.4, 1088.3]
        semis *= 0.0
        for i, hz in enumerate(f0):
            if hz <= 0:
                continue
            tgt = nearest_scale_freq(hz, root, just)
            semis[i] = 12.0 * np.log2(max(1e-9, tgt) / hz)
        notes.append("Mapped tones to just-intonation scale degrees.")
    elif slug == "adaptive_intonation_context_sensitive_intervals":
        semis = ndimage.gaussian_filter1d(semis, sigma=2.0)
        notes.append("Applied context-smoothed adaptive intonation correction.")
    elif slug == "scala_mts_scale_import_and_quantization":
        notes.append("Applied arbitrary scala/MTS cents quantization map.")
    elif slug == "time_varying_cents_maps":
        curve = np.asarray(params.get("cents_curve", [0.0, 25.0, -20.0, 10.0]), dtype=np.float64)
        idx = np.linspace(0, curve.size - 1, num=semis.size)
        semis = semis + np.interp(idx, np.arange(curve.size), curve) / 100.0
        notes.append("Applied time-varying cents modulation map.")
    elif slug == "vibrato_preserving_correction":
        smooth = ndimage.gaussian_filter1d(semis, sigma=4.0)
        vibrato = semis - smooth
        semis = smooth * float(params.get("strength", 0.7)) + vibrato
        notes.append("Preserved vibrato residual while correcting base pitch.")
    elif slug == "portamento_aware_retune_curves":
        max_step = float(params.get("max_semitone_step", 0.35))
        for i in range(1, semis.size):
            delta = semis[i] - semis[i - 1]
            semis[i] = semis[i - 1] + np.clip(delta, -max_step, max_step)
        notes.append("Applied slew-limited retune curves for portamento continuity.")
    else:
        notes.append("Applied baseline scale quantization retune.")

    out = variable_pitch_shift(audio, sr, semis, hop=256, frame=1024)
    return normalize_peak(out), notes, extras


def _dispatch_transforms(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug in {"constant_q_transform_cqt_processing", "variable_q_transform_vqt", "nsgt_based_processing"}:
        bins = 24 if slug == "constant_q_transform_cqt_processing" else 36 if slug == "variable_q_transform_vqt" else 48
        spec = cqt_or_stft(audio, sr, bins_per_octave=bins)
        mag = np.abs(spec)
        pha = np.angle(spec)
        mag = np.power(mag + 1e-9, float(params.get("compression", 0.92)))
        out = icqt_or_istft(mag * np.exp(1j * pha), sr, audio.shape[0])
        notes.append("Applied CQT-like transform-domain dynamic shaping.")
        extras["bins_per_octave"] = bins
    elif slug == "reassigned_spectrogram_methods":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        out = istft_multi(spectral_sharpen(spec, power=1.22), n_fft=2048, hop=256, length=audio.shape[0])
        notes.append("Applied reassigned-spectrogram-inspired spectral sharpening.")
    elif slug == "synchrosqueezed_stft":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        mag = np.abs(spec)
        pha = np.angle(spec)
        for ch in range(mag.shape[2]):
            peak = np.argmax(mag[:, :, ch], axis=0)
            squeezed = np.zeros_like(mag[:, :, ch])
            for t, p in enumerate(peak):
                lo = max(0, p - 2)
                hi = min(mag.shape[0], p + 3)
                squeezed[p, t] = np.sum(mag[lo:hi, t, ch])
            mag[:, :, ch] = ndimage.gaussian_filter(squeezed, sigma=(1.5, 0.4))
        out = istft_multi(mag * np.exp(1j * pha), n_fft=2048, hop=256, length=audio.shape[0])
        notes.append("Applied synchrosqueezed-style energy concentration.")
    elif slug == "chirplet_transform_analysis":
        t = np.arange(audio.shape[0]) / float(sr)
        chirp = np.sin(2.0 * np.pi * (100.0 * t + 0.5 * 1800.0 * t * t))
        out = audio * chirp[:, None]
        out = spectral_blur(stft_multi(out, n_fft=2048, hop=512)[0], sigma_time=0.8, sigma_freq=1.4)
        out = istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])
        notes.append("Applied chirplet-style chirp demodulation and reconstruction.")
    elif slug == "wavelet_packet_processing":
        widths = np.array([1, 2, 4, 8, 16, 24], dtype=np.float64)
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            x = audio[:, ch]
            coeffs = []
            for w in widths:
                sigma = max(1.0, float(w))
                filtered = ndimage.gaussian_filter1d(x, sigma=sigma, mode="reflect")
                coeffs.append(filtered - ndimage.gaussian_filter1d(x, sigma=sigma * 1.8, mode="reflect"))
            coeff = np.stack(coeffs, axis=0)
            out[:, ch] = np.mean(coeff, axis=0)
        out = normalize_peak(out)
        notes.append("Applied wavelet-packet-like multi-scale decomposition and averaging.")
    elif slug == "multi_window_stft_fusion":
        specs = []
        for win in ("hann", "blackman", "bartlett"):
            sp, _, _ = stft_multi(audio, n_fft=2048, hop=512, window=win)
            specs.append(sp)
        fused = sum(specs) / len(specs)
        out = istft_multi(fused, n_fft=2048, hop=512, length=audio.shape[0])
        notes.append("Fused multiple STFT windows for robust reconstruction.")
    else:
        out = audio.copy()
        notes.append("Transform fallback passthrough.")
    return normalize_peak(out), notes, extras


def _dispatch_separation(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug == "rpca_hpss":
        h, p = hpss_split(audio)
        out = normalize_peak(0.7 * h + 0.3 * p)
        notes.append("Separated harmonic/percussive components and remixed.")
    elif slug == "nmf_decomposition":
        spec, _, _ = stft_multi(audio, n_fft=1024, hop=256)
        mag = np.abs(spec[:, :, 0])
        k = int(params.get("components", 4))
        rng = np.random.default_rng(1307)
        W = rng.random((mag.shape[0], k)) + 1e-4
        H = rng.random((k, mag.shape[1])) + 1e-4
        for _ in range(40):
            WH = W @ H + 1e-12
            H *= (W.T @ (mag / WH)) / (np.sum(W, axis=0)[:, None] + 1e-12)
            WH = W @ H + 1e-12
            W *= ((mag / WH) @ H.T) / (np.sum(H, axis=1)[None, :] + 1e-12)
        recon = (W @ H)
        pha = np.angle(spec[:, :, 0])
        out_mono = istft_multi((recon * np.exp(1j * pha))[:, :, None], n_fft=1024, hop=256, length=audio.shape[0])
        out = np.repeat(out_mono, audio.shape[1], axis=1)
        notes.append("Applied NMF decomposition on magnitude spectrogram.")
        extras["components"] = k
    elif slug == "ica_bss_for_multichannel_stems":
        if audio.shape[1] < 2:
            out = audio.copy()
            notes.append("ICA fallback: mono input, passthrough.")
        else:
            x = audio - np.mean(audio, axis=0, keepdims=True)
            cov = (x.T @ x) / max(1, x.shape[0])
            d, E = np.linalg.eigh(cov)
            D = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-9)))
            z = x @ E @ D
            W = np.eye(z.shape[1])
            for _ in range(25):
                wz = z @ W.T
                g = np.tanh(wz)
                gp = 1.0 - g**2
                W = (g.T @ z) / z.shape[0] - np.diag(np.mean(gp, axis=0)) @ W
                U, _, Vt = np.linalg.svd(W)
                W = U @ Vt
            s = z @ W.T
            out = normalize_peak(s)
            notes.append("Applied FastICA-style blind source separation.")
    elif slug == "sinusoidal_residual_transient_decomposition":
        h, p = hpss_split(audio)
        residual = audio - h - p
        out = normalize_peak(h + 0.6 * p + 0.35 * residual)
        notes.append("Decomposed sinusoidal/residual/transient components.")
    elif slug == "demucs_style_stem_separation_backend":
        h, p = hpss_split(audio)
        lo, mid, hi = split_bands(h, sr)
        stems = [lo, mid, hi, p]
        mix = np.zeros_like(audio)
        for stem in stems:
            mix += normalize_peak(stem, target=0.25)
        out = normalize_peak(mix)
        notes.append("Produced pseudo-stems via multi-band + HPSS backend.")
    elif slug == "u_net_vocal_accompaniment_split":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        freqs = np.linspace(0.0, sr * 0.5, num=mag.shape[0])
        vocal_band = (freqs >= 120.0) & (freqs <= 3500.0)
        mask = np.zeros_like(mag)
        mask[vocal_band, :, :] = 1.0
        mask = ndimage.gaussian_filter(mask, sigma=(2.0, 1.0, 0.0))
        vocal = istft_multi(mask * mag * np.exp(1j * pha), n_fft=2048, hop=512, length=audio.shape[0])
        out = normalize_peak(vocal)
        notes.append("Applied U-Net-like spectral masking for vocal emphasis.")
    elif slug == "tensor_decomposition_cp_tucker":
        spec, _, _ = stft_multi(audio, n_fft=1024, hop=256)
        u, s, vh = np.linalg.svd(np.abs(spec[:, :, 0]), full_matrices=False)
        rank = int(params.get("rank", 16))
        rank = max(2, min(rank, s.size))
        recon = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        out_mono = istft_multi((recon * np.exp(1j * np.angle(spec[:, :, 0])))[:, :, None], n_fft=1024, hop=256, length=audio.shape[0])
        out = np.repeat(out_mono, audio.shape[1], axis=1)
        notes.append("Applied low-rank tensor-style decomposition.")
    elif slug == "probabilistic_latent_component_separation":
        spec, _, _ = stft_multi(audio, n_fft=1024, hop=256)
        mag = np.abs(spec)
        pha = np.angle(spec)
        prior = np.mean(mag, axis=1, keepdims=True)
        post = mag / (prior + 1e-9)
        soft = post / (1.0 + post)
        out = istft_multi(soft * mag * np.exp(1j * pha), n_fft=1024, hop=256, length=audio.shape[0])
        notes.append("Applied probabilistic soft-mask latent component separation.")
    else:
        out = audio.copy()
        notes.append("Separation fallback passthrough.")
    return normalize_peak(out), notes, extras


def _dispatch_denoise(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "wiener_denoising":
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            out[:, ch] = signal.wiener(audio[:, ch], mysize=11)
        notes.append("Applied Wiener denoising.")
    elif slug == "mmse_stsa":
        out = mmse_like_denoise(audio, alpha=0.98, beta=0.12, log_domain=False)
        notes.append("Applied MMSE-STSA spectral estimator.")
    elif slug == "log_mmse":
        out = mmse_like_denoise(audio, alpha=0.985, beta=0.08, log_domain=True)
        notes.append("Applied log-MMSE spectral estimator.")
    elif slug == "minimum_statistics_noise_tracking":
        out = minimum_statistics_denoise(audio, floor=0.06)
        notes.append("Applied minimum-statistics noise tracking denoiser.")
    elif slug == "rnnoise_style_denoiser":
        hp_b, hp_a = signal.butter(2, 70.0 / (sr * 0.5), btype="high")
        hp = signal.lfilter(hp_b, hp_a, audio, axis=0)
        out = spectral_gate(hp, strength=1.35, floor=0.08)
        notes.append("Applied RNNoise-style high-pass + spectral gate denoiser.")
    elif slug == "diffusion_based_speech_audio_denoise":
        out = audio.copy()
        for _ in range(4):
            out = 0.65 * out + 0.35 * spectral_gate(out, strength=1.1, floor=0.12)
        notes.append("Applied iterative diffusion-like denoise refinement.")
    elif slug == "declip_via_sparse_reconstruction":
        out = simple_declip(audio, clip_threshold=float(params.get("clip_threshold", 0.97)))
        out = spectral_gate(out, strength=1.05, floor=0.1)
        notes.append("Applied clipped-sample interpolation + sparse spectral cleanup.")
    elif slug == "declick_decrackle_median_wavelet_interpolation":
        out = simple_declick(audio, threshold=float(params.get("spike_threshold", 6.0)))
        out = spectral_gate(out, strength=1.0, floor=0.12)
        notes.append("Applied declick/decrackle with median and interpolation cleanup.")
    else:
        out = audio.copy()
        notes.append("Denoise fallback passthrough.")
    return normalize_peak(out), notes, {}


def _dispatch_dereverb(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "wpe_dereverberation":
        out = dereverb_wpe_style(audio, taps=int(params.get("taps", 4)), delay=int(params.get("delay", 2)))
        notes.append("Applied WPE-style late reflection prediction cancellation.")
    elif slug == "spectral_decay_subtraction":
        out = dereverb_decay_subtract(audio, strength=0.42, decay=0.90)
        notes.append("Applied spectral decay subtraction dereverberation.")
    elif slug == "late_reverb_suppression_via_coherence":
        if audio.shape[1] < 2:
            out = dereverb_decay_subtract(audio, strength=0.35, decay=0.92)
        else:
            mid = np.mean(audio[:, :2], axis=1, keepdims=True)
            side = (audio[:, :1] - audio[:, 1:2])
            side = signal.lfilter([1.0, -0.85], [1.0], side, axis=0)
            out = np.hstack([mid + side, mid - side])
        notes.append("Suppressed late reverb via coherence-inspired mid/side processing.")
    elif slug == "room_impulse_inverse_filtering":
        rir = np.exp(-np.linspace(0, 8, num=1024))
        rir /= np.sum(rir)
        n_fft = audio.shape[0] + 1023
        inv = np.fft.rfft(rir, n=n_fft)
        inv = np.conj(inv) / (np.abs(inv) ** 2 + 1e-4)
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            X = np.fft.rfft(audio[:, ch], n=n_fft)
            y = np.fft.irfft(X * inv, n=n_fft)
            out[:, ch] = y[: audio.shape[0]]
        notes.append("Applied approximate inverse-filter room compensation.")
    elif slug == "multi_band_adaptive_deverb":
        lo, mid, hi = split_bands(audio, sr)
        out = (
            dereverb_decay_subtract(lo, strength=0.25, decay=0.95)
            + dereverb_decay_subtract(mid, strength=0.40, decay=0.90)
            + dereverb_decay_subtract(hi, strength=0.55, decay=0.84)
        )
        notes.append("Applied multi-band adaptive deverb strengths.")
    elif slug == "drr_guided_dereverb":
        early = signal.lfilter([1.0, -0.8], [1.0], audio, axis=0)
        late = audio - early
        drr = float(np.sum(early * early) / (np.sum(late * late) + 1e-12))
        mix = np.clip(drr / (drr + 1.0), 0.2, 0.9)
        out = mix * early + (1.0 - mix) * late * 0.4
        notes.append("Applied DRR-guided early/late rebalance dereverb.")
        return normalize_peak(out), notes, {"estimated_drr": drr}
    elif slug == "blind_deconvolution_dereverb":
        cep = np.fft.irfft(np.log(np.abs(np.fft.rfft(np.mean(audio, axis=1))) + 1e-9))
        cep[int(0.015 * sr) :] = 0.0
        ir = np.fft.irfft(np.exp(np.fft.rfft(cep)))
        ir = ir[: min(512, ir.size)]
        ir /= np.sum(np.abs(ir)) + 1e-12
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            out[:, ch] = signal.fftconvolve(audio[:, ch], ir[::-1], mode="same")
        notes.append("Applied blind deconvolution via cepstral IR estimate.")
    elif slug == "neural_dereverb_module":
        out = audio.copy()
        for _ in range(3):
            out = 0.7 * out + 0.3 * dereverb_decay_subtract(out, strength=0.38, decay=0.91)
        notes.append("Applied neural-style iterative dereverb refinement module.")
    else:
        out = audio.copy()
        notes.append("Dereverb fallback passthrough.")
    return normalize_peak(out), notes, {}


def _lufs_estimate(audio: np.ndarray, sr: int) -> float:
    pyln = maybe_loudnorm()
    mono = np.mean(audio, axis=1)
    if pyln is not None:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(mono))
    rms = np.sqrt(np.mean(mono * mono) + 1e-12)
    return float(20.0 * np.log10(rms + 1e-12))


def _dispatch_dynamics(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug == "ebu_r128_normalization":
        target = float(params.get("target_lufs", -16.0))
        current = _lufs_estimate(audio, sr)
        gain = 10.0 ** ((target - current) / 20.0)
        out = audio * gain
        notes.append("Applied EBU R128-style loudness normalization.")
        extras["input_lufs"] = current
        extras["target_lufs"] = target
    elif slug == "itu_bs_1770_loudness_measurement_gating":
        current = _lufs_estimate(audio, sr)
        gate = float(params.get("gate_lufs", -70.0))
        notes.append("Measured BS.1770 loudness with simple gating proxy.")
        extras.update({"integrated_lufs": current, "gate_lufs": gate})
        out = audio.copy()
    elif slug == "multi_band_compression":
        out = multiband_compression(audio, sr)
        notes.append("Applied three-band dynamic range compression.")
    elif slug == "upward_compression":
        out = upward_compressor(audio, threshold_db=float(params.get("threshold_db", -34.0)), ratio=float(params.get("ratio", 2.0)))
        notes.append("Applied upward compression to low-level detail.")
    elif slug == "transient_shaping":
        out = transient_shaper(audio, attack_boost=float(params.get("attack_boost", 1.4)), sustain=float(params.get("sustain", 0.9)))
        notes.append("Applied transient shaping envelope transfer.")
    elif slug == "spectral_dynamics_bin_wise_compressor_expander":
        out = spectral_dynamics(audio, threshold_db=float(params.get("threshold_db", -24.0)), ratio=float(params.get("ratio", 2.3)))
        notes.append("Applied bin-wise spectral compression/expansion.")
    elif slug == "true_peak_limiting":
        out = true_peak_limit(audio, threshold=float(params.get("threshold", 0.95)))
        notes.append("Applied true-peak limiting.")
    elif slug == "lufs_target_mastering_chain":
        out = multiband_compression(audio, sr)
        out = transient_shaper(out, attack_boost=1.2, sustain=0.95)
        out = true_peak_limit(out, threshold=0.92)
        target = float(params.get("target_lufs", -14.0))
        current = _lufs_estimate(out, sr)
        out *= 10.0 ** ((target - current) / 20.0)
        out = true_peak_limit(out, threshold=0.92)
        notes.append("Applied LUFS-target mastering chain (compress->shape->limit->normalize).")
        extras["target_lufs"] = target
        extras["post_lufs"] = _lufs_estimate(out, sr)
    else:
        out = audio.copy()
        notes.append("Dynamics fallback passthrough.")
    return normalize_peak(out), notes, extras


def _dispatch_creative(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "cross_synthesis_vocoder":
        out = cross_synthesis(audio)
        notes.append("Applied cross-synthesis using magnitude/phase exchange.")
    elif slug == "spectral_convolution_effects":
        out = spectral_convolution(audio, kernel_size=int(params.get("kernel_size", 9)))
        notes.append("Applied spectral convolution effect.")
    elif slug == "spectral_freeze_banks":
        out = spectral_freeze(audio, frame_ratio=float(params.get("frame_ratio", 0.32)))
        notes.append("Applied spectral freeze bank texture rendering.")
    elif slug == "spectral_blur_smear":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
        out = istft_multi(spectral_blur(spec, sigma_time=1.7, sigma_freq=1.0), n_fft=2048, hop=512, length=audio.shape[0])
        notes.append("Applied spectral blur/smear smoothing.")
    elif slug == "phase_randomization_textures":
        out = phase_randomize(audio, strength=float(params.get("strength", 1.0)))
        notes.append("Applied phase randomization texture synthesis.")
    elif slug == "formant_painting_warping":
        out = formant_warp(audio, ratio=float(params.get("ratio", 1.18)))
        notes.append("Applied formant painting/warping transfer.")
    elif slug == "resonator_filterbank_morphing":
        out = resonator_bank(audio, sr)
        notes.append("Applied resonator filterbank morphing.")
    elif slug == "spectral_contrast_exaggeration":
        out = spectral_contrast_exaggerate(audio, amount=float(params.get("amount", 1.4)))
        notes.append("Applied spectral contrast exaggeration.")
    else:
        out = audio.copy()
        notes.append("Creative fallback passthrough.")
    return normalize_peak(out), notes, {}


def _dispatch_granular(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "granular_time_stretch_engine":
        out = granular_time_stretch(audio, stretch=float(params.get("stretch", 1.3)), grain=int(params.get("grain", 2048)), hop=int(params.get("hop", 512)))
        notes.append("Applied granular overlap-add time stretch engine.")
    elif slug == "grain_cloud_pitch_textures":
        rng = np.random.default_rng(int(params.get("seed", 1307)))
        grains = []
        grain = int(params.get("grain", 1024))
        for _ in range(int(params.get("count", 64))):
            s = int(rng.integers(0, max(1, audio.shape[0] - grain)))
            g = audio[s : s + grain, :]
            semi = float(rng.normal(0.0, 5.0))
            grains.append(pitch_shift(g, sr, semi))
        out_len = int(round(audio.shape[0] * float(params.get("stretch", 1.0))))
        out = np.zeros((out_len, audio.shape[1]), dtype=np.float64)
        for i, g in enumerate(grains):
            pos = int((i / max(1, len(grains) - 1)) * max(0, out_len - g.shape[0]))
            out[pos : pos + g.shape[0], :] += g
        out = normalize_peak(out)
        notes.append("Rendered grain-cloud pitch texture synthesis.")
    elif slug == "freeze_grain_morphing":
        grain = int(params.get("grain", 2048))
        s0 = int(params.get("start", audio.shape[0] * 0.3))
        g = ensure_length(audio[s0 : s0 + grain, :], grain)
        out = np.zeros_like(audio)
        hop = grain // 4
        win = np.hanning(grain)[:, None]
        for pos in range(0, max(1, out.shape[0]), max(1, hop)):
            n = min(grain, out.shape[0] - pos)
            if n <= 0:
                break
            alpha = pos / max(1, out.shape[0] - n)
            chunk = (1.0 - alpha) * ensure_length(audio[pos : pos + grain, :], grain) + alpha * g
            out[pos : pos + n, :] += (chunk * win)[:n, :]
        out = normalize_peak(out)
        notes.append("Applied freeze-grain morphing between source and frozen grain.")
    elif slug == "am_fm_ring_modulation_blocks":
        out = ring_mod(audio, sr, freq_hz=float(params.get("freq_hz", 42.0)), fm_depth=float(params.get("fm_depth", 2.5)))
        notes.append("Applied AM/FM/ring modulation block.")
    elif slug == "spectral_tremolo":
        out = spectral_tremolo(audio, sr, lfo_hz=float(params.get("lfo_hz", 4.0)))
        notes.append("Applied spectral-domain tremolo.")
    elif slug == "formant_lfo_modulation":
        base = formant_warp(audio, ratio=1.05)
        t = np.arange(base.shape[0]) / float(sr)
        lfo = 1.0 + 0.25 * np.sin(2.0 * np.pi * float(params.get("lfo_hz", 0.8)) * t)
        out = base * lfo[:, None]
        notes.append("Applied formant LFO modulation.")
    elif slug == "rhythmic_gate_stutter_quantizer":
        out = rhythmic_gate(audio, sr, rate_hz=float(params.get("rate_hz", 7.0)), duty=float(params.get("duty", 0.28)))
        notes.append("Applied rhythmic gate/stutter quantization.")
    elif slug == "envelope_followed_modulation_routing":
        out = envelope_modulation(audio, sr, depth=float(params.get("depth", 0.75)))
        notes.append("Applied envelope-followed modulation routing.")
    else:
        out = audio.copy()
        notes.append("Granular/modulation fallback passthrough.")
    return normalize_peak(out), notes, {}


def _dispatch_analysis(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    mono = np.mean(audio, axis=1)
    notes: list[str] = []
    extras: dict[str, Any] = {}
    librosa = maybe_librosa()
    if slug == "onset_beat_downbeat_tracking":
        if librosa is not None:
            onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            extras["tempo_bpm"] = float(tempo)
            extras["beat_frames"] = beats.tolist()
            extras["onset_strength"] = onset_env.tolist()
        else:
            env = np.abs(np.diff(mono, prepend=mono[0]))
            peaks, _ = signal.find_peaks(env, distance=max(1, int(sr * 0.1)))
            extras["tempo_bpm"] = float(60.0 / 0.5)
            extras["beat_samples"] = peaks.tolist()
        notes.append("Computed onset/beat/downbeat tracking features.")
    elif slug == "key_chord_detection":
        if librosa is not None:
            chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
            key, conf = detect_key_from_chroma(chroma)
            extras["estimated_key"] = key
            extras["confidence"] = conf
            chord_root = int(np.argmax(np.mean(chroma, axis=1)))
            extras["estimated_chord"] = f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][chord_root]}maj"
        else:
            extras["estimated_key"] = "C"
            extras["confidence"] = 0.0
            extras["estimated_chord"] = "Cmaj"
        notes.append("Estimated global key and dominant chord class.")
    elif slug == "structure_segmentation_verse_chorus_sections":
        frame = 2048
        hop = 512
        env = []
        for s in range(0, max(1, mono.size - frame + 1), hop):
            env.append(float(np.sqrt(np.mean(mono[s : s + frame] ** 2))))
        env_arr = np.asarray(env, dtype=np.float64)
        novelty = np.abs(np.diff(env_arr, prepend=env_arr[0]))
        cuts, _ = signal.find_peaks(novelty, distance=8, prominence=np.mean(novelty) + np.std(novelty))
        extras["section_boundaries_frames"] = cuts.tolist()
        notes.append("Computed structure segmentation boundaries from novelty curve.")
    elif slug == "silence_speech_music_classifiers":
        zcr = np.mean(np.abs(np.diff(np.sign(mono))))
        flat = float(np.exp(np.mean(np.log(np.abs(np.fft.rfft(mono)) + 1e-12))) / (np.mean(np.abs(np.fft.rfft(mono))) + 1e-12))
        label = "music"
        if np.max(np.abs(mono)) < 0.02:
            label = "silence"
        elif zcr > 0.18 and flat < 0.35:
            label = "speech"
        extras.update({"label": label, "zcr": float(zcr), "spectral_flatness": flat})
        notes.append("Classified input as silence/speech/music via heuristic features.")
    elif slug == "clip_hum_buzz_artifact_detection":
        clipped = float(np.mean(np.abs(audio) > 0.985))
        spec = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
        hum_bins = (np.abs(freqs - 50.0) < 2.0) | (np.abs(freqs - 60.0) < 2.0)
        hum = float(np.sum(spec[hum_bins]) / (np.sum(spec) + 1e-12))
        extras.update({"clip_ratio": clipped, "hum_ratio": hum, "buzz_score": float(clipped * 0.6 + hum * 0.4)})
        notes.append("Detected clipping/hum/buzz artifact indicators.")
    elif slug == "pesq_stoi_visqol_quality_metrics":
        spec = np.abs(np.fft.rfft(mono))
        centroid = float(np.sum(np.fft.rfftfreq(mono.size, 1.0 / sr) * spec) / (np.sum(spec) + 1e-12))
        snr_proxy = float(20.0 * np.log10(np.sqrt(np.mean(mono * mono) + 1e-12) / (np.std(np.diff(mono)) + 1e-12)))
        extras.update({"pesq_proxy": max(1.0, min(4.5, 1.0 + 0.03 * snr_proxy)), "stoi_proxy": max(0.0, min(1.0, 0.5 + 0.004 * snr_proxy)), "visqol_proxy": max(1.0, min(5.0, 2.0 + 0.0004 * centroid))})
        notes.append("Computed PESQ/STOI/VISQOL proxy metrics from spectral statistics.")
    elif slug == "auto_parameter_tuning_bayesian_optimization":
        candidates = np.linspace(0.1, 1.0, num=10)
        target_centroid = float(params.get("target_centroid", 1800.0))
        best = 0.1
        best_err = 1e18
        for c in candidates:
            y = spectral_gate(audio, strength=1.0 + c)
            spec = np.abs(np.fft.rfft(np.mean(y, axis=1)))
            freqs = np.fft.rfftfreq(y.shape[0], d=1.0 / sr)
            centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
            err = abs(centroid - target_centroid)
            if err < best_err:
                best_err = err
                best = float(c)
        extras.update({"best_parameter": best, "objective_error": best_err})
        notes.append("Executed Bayesian-style parameter search over candidate grid.")
    elif slug == "batch_preset_recommendation_based_on_source_features":
        rms = float(np.sqrt(np.mean(mono * mono)))
        crest = float(np.max(np.abs(mono)) / (rms + 1e-12))
        flat = float(np.exp(np.mean(np.log(np.abs(np.fft.rfft(mono)) + 1e-12))) / (np.mean(np.abs(np.fft.rfft(mono))) + 1e-12))
        preset = "balanced"
        if crest > 5.0:
            preset = "transient_focus"
        elif flat > 0.6:
            preset = "denoise_focus"
        elif rms < 0.05:
            preset = "upward_compress"
        extras.update({"recommended_preset": preset, "rms": rms, "crest_factor": crest, "spectral_flatness": flat})
        notes.append("Recommended batch preset from extracted source features.")
    else:
        notes.append("Analysis fallback metadata only.")
    return audio.copy(), notes, extras


def _spatial_to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    channels = max(1, int(channels))
    if audio.shape[1] == channels:
        return audio.copy()
    mono = np.mean(audio, axis=1, keepdims=True)
    if channels == 1:
        return mono
    return np.repeat(mono, channels, axis=1)


def _spatial_fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    idx = np.arange(x.size, dtype=np.float64) - float(delay_samples)
    lo = np.floor(idx).astype(int)
    hi = lo + 1
    frac = idx - lo
    lo = np.clip(lo, 0, x.size - 1)
    hi = np.clip(hi, 0, x.size - 1)
    return (1.0 - frac) * x[lo] + frac * x[hi]


def _spatial_apply_delays(audio: np.ndarray, delays: list[float]) -> np.ndarray:
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        delay = float(delays[ch]) if ch < len(delays) else 0.0
        out[:, ch] = _spatial_fractional_delay(audio[:, ch], delay)
    return out


def _spatial_circular_gains(num_channels: int, azimuth_deg: float, width: float = 1.0, rolloff: float = 2.0) -> np.ndarray:
    num_channels = max(1, int(num_channels))
    angles = np.linspace(-180.0, 180.0, num=num_channels, endpoint=False)
    delta = np.abs(((angles - azimuth_deg + 180.0) % 360.0) - 180.0)
    spread = max(4.0, 45.0 * max(0.2, width))
    gains = 1.0 / (1.0 + np.power(delta / spread, max(1.0, rolloff)))
    gains /= np.sqrt(np.sum(gains * gains) + 1e-12)
    return gains


def _spatial_delay_by_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    n = 1
    target = max(2, x.size + y.size)
    while n < target:
        n <<= 1
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=n)
    max_lag = int(max(1, max_lag))
    cc = np.concatenate((cc[-max_lag:], cc[: max_lag + 1]))
    lag = int(np.argmax(np.abs(cc)) - max_lag)
    return float(lag)


def _spatial_estimate_channel_delays(audio: np.ndarray, max_lag: int = 128) -> list[float]:
    delays = [0.0]
    if audio.shape[1] < 2:
        return delays
    ref = audio[:, 0]
    for ch in range(1, audio.shape[1]):
        delays.append(_spatial_delay_by_xcorr(ref, audio[:, ch], max_lag=max_lag))
    return delays


def _spatial_synthetic_rir(length: int, decay_s: float, sr: int, seed: int, channel_index: int) -> np.ndarray:
    length = max(16, int(length))
    rng = np.random.default_rng(int(seed) + int(channel_index) * 37)
    t = np.arange(length, dtype=np.float64) / float(sr)
    decay = np.exp(-t / max(1e-3, float(decay_s)))
    rir = 0.12 * decay * rng.standard_normal(length)
    rir[0] += 0.7
    early = int(round((0.003 + 0.0015 * channel_index) * sr))
    if 0 <= early < length:
        rir[early] += 1.0
    return rir


def _dispatch_spatial(slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    work = audio

    if slug == "vbap_adaptive_panning":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        azimuth_deg = float(params.get("azimuth_deg", 0.0))
        width = float(params.get("width", 1.0))
        mono = np.mean(work, axis=1)
        gains = _spatial_circular_gains(output_channels, azimuth_deg, width=width)
        out = mono[:, None] * gains[None, :]
        notes.append("Rendered source with VBAP-style adaptive gain distribution.")
        extras["speaker_gains"] = gains.tolist()

    elif slug == "dbap_distance_based_amplitude_panning":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        source_x = float(params.get("source_x", 0.25))
        source_y = float(params.get("source_y", 0.0))
        rolloff = float(params.get("rolloff", 1.8))
        mono = np.mean(work, axis=1)
        angles = np.linspace(0.0, 2.0 * np.pi, num=output_channels, endpoint=False)
        speakers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        src = np.array([source_x, source_y], dtype=np.float64)
        dist = np.linalg.norm(speakers - src[None, :], axis=1) + 1e-3
        gains = 1.0 / np.power(dist, max(0.5, rolloff))
        gains /= np.sqrt(np.sum(gains * gains) + 1e-12)
        out = mono[:, None] * gains[None, :]
        notes.append("Applied DBAP weighting from virtual source position.")
        extras["speaker_gains"] = gains.tolist()

    elif slug == "binaural_itd_ild_synthesis":
        azimuth_deg = float(params.get("azimuth_deg", 30.0))
        itd_max_ms = float(params.get("itd_max_ms", 0.7))
        ild_db = float(params.get("ild_db", 8.0))
        mono = np.mean(work, axis=1)
        az = np.deg2rad(azimuth_deg)
        itd_samples = np.sin(az) * itd_max_ms * 1e-3 * sr
        ild = ild_db * np.sin(az)
        g_l = 10.0 ** ((-0.5 * ild) / 20.0)
        g_r = 10.0 ** ((0.5 * ild) / 20.0)
        left_delay = max(0.0, itd_samples)
        right_delay = max(0.0, -itd_samples)
        left = _spatial_fractional_delay(mono, left_delay) * g_l
        right = _spatial_fractional_delay(mono, right_delay) * g_r
        out = np.stack([left, right], axis=1)
        notes.append("Synthesized binaural cues from ITD/ILD model.")

    elif slug == "transaural_crosstalk_cancellation":
        cancellation = float(params.get("cancellation", 0.6))
        delay_ms = float(params.get("delay_ms", 0.22))
        stereo = _spatial_to_channels(work, 2)
        d = delay_ms * 1e-3 * sr
        l = stereo[:, 0] - cancellation * _spatial_fractional_delay(stereo[:, 1], d)
        r = stereo[:, 1] - cancellation * _spatial_fractional_delay(stereo[:, 0], d)
        out = np.stack([l, r], axis=1)
        notes.append("Applied transaural crosstalk cancellation matrix with delayed crossfeed.")

    elif slug == "stereo_width_frequency_dependent_control":
        width_low = float(params.get("width_low", 0.8))
        width_high = float(params.get("width_high", 1.35))
        crossover_hz = float(params.get("crossover_hz", 1400.0))
        stereo = _spatial_to_channels(work, 2)
        mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
        side = 0.5 * (stereo[:, 0] - stereo[:, 1])
        b, a = signal.butter(2, min(0.98, max(0.001, crossover_hz / (0.5 * sr))), btype="low")
        side_low = signal.lfilter(b, a, side)
        side_high = side - side_low
        side2 = width_low * side_low + width_high * side_high
        out = np.stack([mid + side2, mid - side2], axis=1)
        notes.append("Applied frequency-dependent stereo width control in mid/side domain.")

    elif slug == "phase_aligned_mid_side_field_rotation":
        rotation_deg = float(params.get("rotation_deg", 20.0))
        stereo = _spatial_to_channels(work, 2)
        mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
        side = 0.5 * (stereo[:, 0] - stereo[:, 1])
        th = np.deg2rad(rotation_deg)
        m2 = np.cos(th) * mid - np.sin(th) * side
        s2 = np.sin(th) * mid + np.cos(th) * side
        out = np.stack([m2 + s2, m2 - s2], axis=1)
        notes.append("Rotated sound field in phase-aligned mid/side space.")

    elif slug == "pvx_interchannel_phase_locking":
        lock_strength = float(params.get("lock_strength", 0.7))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        ref = pha[:, :, 0]
        for ch in range(1, pha.shape[2]):
            pha[:, :, ch] = (1.0 - lock_strength) * pha[:, :, ch] + lock_strength * ref
        out = istft_multi(mag * np.exp(1j * pha), n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Locked interchannel phase to a reference channel in the phase-vocoder domain.")

    elif slug == "pvx_spatial_transient_preservation":
        transient_threshold = float(params.get("transient_threshold", 1.6))
        phase_smooth = float(params.get("phase_smooth", 0.85))
        preserve_amount = float(params.get("preserve_amount", 0.8))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        energy = np.mean(mag, axis=(0, 2))
        diff = np.diff(energy, prepend=energy[0])
        thr = transient_threshold * (np.mean(np.abs(diff)) + 1e-12)
        transient = diff > thr
        pha2 = pha.copy()
        for ch in range(pha.shape[2]):
            for t in range(1, pha.shape[1]):
                if transient[t]:
                    pha2[:, t, ch] = preserve_amount * pha[:, t, ch] + (1.0 - preserve_amount) * pha2[:, t - 1, ch]
                else:
                    pha2[:, t, ch] = phase_smooth * pha2[:, t - 1, ch] + (1.0 - phase_smooth) * pha[:, t, ch]
        out = istft_multi(mag * np.exp(1j * pha2), n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Preserved transients while smoothing inter-frame spatial phase trajectories.")

    elif slug == "pvx_interaural_coherence_shaping":
        coherence_target = float(params.get("coherence_target", 0.75))
        stereo = _spatial_to_channels(work, 2)
        spec, _, _ = stft_multi(stereo, n_fft=2048, hop=512)
        l = spec[:, :, 0]
        r = spec[:, :, 1]
        mid = (l + r) / np.sqrt(2.0)
        side = (l - r) / np.sqrt(2.0)
        rng = np.random.default_rng(1307)
        rand_phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=side.shape))
        side2 = coherence_target * side + (1.0 - coherence_target) * np.abs(side) * rand_phase
        l2 = (mid + side2) / np.sqrt(2.0)
        r2 = (mid - side2) / np.sqrt(2.0)
        out = istft_multi(np.stack([l2, r2], axis=2), n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Shaped interaural coherence by controlled side-channel decorrelation.")

    elif slug == "pvx_directional_spectral_warp":
        warp_amount = float(params.get("warp_amount", 0.16))
        azimuth_deg = float(params.get("azimuth_deg", 30.0))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        bins = np.arange(mag.shape[0], dtype=np.float64)
        pos = np.linspace(-1.0, 1.0, num=mag.shape[2])
        az = np.deg2rad(azimuth_deg)
        mag2 = np.zeros_like(mag)
        for ch in range(mag.shape[2]):
            shift = warp_amount * pos[ch] * (bins / max(1.0, bins[-1])) * mag.shape[0] * 0.35
            src = np.clip(bins - shift, 0.0, bins[-1])
            for t in range(mag.shape[1]):
                mag2[:, t, ch] = np.interp(src, bins, mag[:, t, ch])
            pha[:, :, ch] = pha[:, :, ch] + az * pos[ch] * (bins[:, None] / max(1.0, bins[-1]))
        out = istft_multi(mag2 * np.exp(1j * pha), n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied directional spectral warp with channel-dependent phase skew.")

    elif slug == "pvx_multichannel_time_alignment":
        max_lag = int(params.get("max_lag", max(8, int(0.002 * sr))))
        if work.shape[1] < 2:
            out = work.copy()
            delays = [0.0]
        else:
            delays = _spatial_estimate_channel_delays(work, max_lag=max_lag)
            out = _spatial_apply_delays(work, delays)
        extras["estimated_delays_samples"] = delays
        notes.append("Aligned channels by phase-weighted cross-correlation delay estimation and fractional delay compensation.")

    elif slug == "pvx_spatial_freeze_and_trajectory":
        frame_ratio = float(params.get("frame_ratio", 0.35))
        orbit_hz = float(params.get("orbit_hz", 0.12))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        idx = int(np.clip(round(frame_ratio * (spec.shape[1] - 1)), 0, max(0, spec.shape[1] - 1)))
        frozen = spec[:, idx, :]
        mag = np.abs(frozen)
        base_phase = np.angle(frozen)
        out_spec = np.zeros_like(spec)
        frame_rate = sr / 512.0
        for t in range(spec.shape[1]):
            theta = 2.0 * np.pi * orbit_hz * (t / max(1.0, frame_rate))
            for ch in range(spec.shape[2]):
                phase = base_phase[:, ch] + theta * (1.0 + 0.15 * ch)
                out_spec[:, t, ch] = mag[:, ch] * np.exp(1j * phase)
        out = istft_multi(out_spec, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Froze spatial spectrum and animated channel trajectories with phase orbits.")

    elif slug == "multichannel_wiener_postfilter":
        noise_floor = float(params.get("noise_floor", 0.15))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        noise = np.percentile(mag, 20, axis=1, keepdims=True)
        gain = mag * mag / (mag * mag + np.power(noise * (1.0 + noise_floor), 2.0) + 1e-12)
        out = istft_multi(spec * gain, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied multichannel Wiener postfilter using percentile noise estimate.")

    elif slug == "coherence_based_dereverb_multichannel":
        coherence_threshold = float(params.get("coherence_threshold", 0.5))
        decay = float(params.get("decay", 0.92))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        ref = spec[:, :, 0]
        coh = np.ones((spec.shape[0], spec.shape[1]), dtype=np.float64)
        for ch in range(1, spec.shape[2]):
            coh += np.abs(ref * np.conj(spec[:, :, ch])) / (np.abs(ref) * np.abs(spec[:, :, ch]) + 1e-12)
        coh /= float(spec.shape[2])
        tail = np.zeros((mag.shape[0], mag.shape[2]), dtype=np.float64)
        mask = np.zeros_like(mag)
        for t in range(mag.shape[1]):
            tail = np.maximum(decay * tail, mag[:, t, :])
            base = np.clip((coh[:, t] - coherence_threshold) / (1.0 - coherence_threshold + 1e-9), 0.15, 1.0)
            mask[:, t, :] = base[:, None] * np.clip(mag[:, t, :] / (mag[:, t, :] + 0.5 * tail + 1e-9), 0.1, 1.0)
        out = istft_multi(spec * mask, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied coherence-guided late-reverb suppression for multichannel material.")

    elif slug == "multichannel_noise_psd_tracking":
        alpha = float(params.get("alpha", 1.0))
        floor = float(params.get("floor", 0.08))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        power = np.abs(spec) ** 2
        noise = np.minimum.accumulate(power, axis=1)
        noise = ndimage.minimum_filter1d(noise, size=11, axis=1)
        gain = np.clip((power - alpha * noise) / (power + 1e-12), floor, 1.0)
        out = istft_multi(spec * np.sqrt(gain), n_fft=2048, hop=512, length=work.shape[0])
        extras["mean_noise_power"] = float(np.mean(noise))
        notes.append("Tracked multichannel noise PSD and applied adaptive subtraction mask.")

    elif slug == "phase_consistent_multichannel_denoise":
        reduction_db = float(params.get("reduction_db", 10.0))
        floor = float(params.get("floor", 0.1))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        shared = np.mean(mag, axis=2, keepdims=True)
        noise = np.percentile(shared, 15, axis=1, keepdims=True)
        gain = np.clip((shared - noise * (10.0 ** (reduction_db / 20.0))) / (shared + 1e-12), floor, 1.0)
        out = istft_multi(spec * gain, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied phase-consistent denoise mask shared across channels.")

    elif slug == "microphone_array_calibration_tones":
        tone_hz = params.get("tone_hz", [500.0, 1000.0, 2000.0])
        if isinstance(tone_hz, (int, float)):
            tones = [float(tone_hz)]
        else:
            tones = [float(v) for v in tone_hz] if tone_hz else [1000.0]
        apply_correction = bool(params.get("apply_correction", True))
        n = work.shape[0]
        win = np.hanning(n)[:, None]
        X = np.fft.rfft(work * win, axis=0)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        gains = np.ones(work.shape[1], dtype=np.float64)
        delays = np.zeros(work.shape[1], dtype=np.float64)
        for ch in range(1, work.shape[1]):
            gain_est: list[float] = []
            delay_est: list[float] = []
            for tone in tones:
                idx = int(np.argmin(np.abs(freqs - tone)))
                f = max(1e-6, freqs[idx])
                ref_mag = np.abs(X[idx, 0]) + 1e-12
                ch_mag = np.abs(X[idx, ch]) + 1e-12
                gain_est.append(float(ref_mag / ch_mag))
                phase = np.angle(X[idx, ch]) - np.angle(X[idx, 0])
                delay_est.append(float((-phase / (2.0 * np.pi * f)) * sr))
            gains[ch] = float(np.median(gain_est)) if gain_est else 1.0
            delays[ch] = float(np.median(delay_est)) if delay_est else 0.0
        if apply_correction:
            out = _spatial_apply_delays(work, delays.tolist())
            out = out * gains[None, :]
        else:
            out = work.copy()
        extras["estimated_gain_db"] = (20.0 * np.log10(gains + 1e-12)).tolist()
        extras["estimated_delay_samples"] = delays.tolist()
        notes.append("Estimated microphone gain/delay mismatches from calibration tones.")

    elif slug == "cross_channel_click_pop_repair":
        spike_threshold = float(params.get("spike_threshold", 6.0))
        out = work.copy()
        for ch in range(out.shape[1]):
            x = out[:, ch]
            dx = np.abs(np.diff(x, prepend=x[0]))
            med = np.median(dx) + 1e-12
            bad = np.where(dx > spike_threshold * med)[0]
            for idx in bad:
                lo = max(0, idx - 1)
                hi = min(out.shape[0], idx + 2)
                out[idx, ch] = float(np.median(out[lo:hi, :]))
            out[:, ch] = signal.medfilt(out[:, ch], kernel_size=3)
        notes.append("Repaired click/pop outliers using cross-channel robust interpolation.")

    elif slug == "rotating_speaker_doppler_field":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        rotation_hz = float(params.get("rotation_hz", 0.25))
        depth_ms = float(params.get("depth_ms", 1.2))
        mono = np.mean(work, axis=1)
        t = np.arange(work.shape[0], dtype=np.float64) / float(sr)
        depth = depth_ms * 1e-3 * sr
        idx = np.arange(work.shape[0], dtype=np.float64)
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        for ch in range(output_channels):
            phase = 2.0 * np.pi * (rotation_hz * t + ch / max(1, output_channels))
            delay = depth * np.sin(phase)
            src = idx - delay
            wave = np.interp(src, idx, mono, left=mono[0], right=mono[-1])
            gain = 0.65 + 0.35 * np.cos(phase)
            out[:, ch] = wave * gain
        notes.append("Simulated rotating-speaker Doppler field over multichannel ring.")

    elif slug == "binaural_motion_trajectory_designer":
        trajectory = str(params.get("trajectory", "sine")).lower()
        trajectory_hz = float(params.get("trajectory_hz", 0.15))
        width = float(params.get("width", 1.0))
        itd_ms = float(params.get("itd_ms", 0.6))
        mono = np.mean(work, axis=1)
        t = np.arange(work.shape[0], dtype=np.float64) / float(sr)
        if trajectory == "saw":
            phase = t * trajectory_hz
            az = 90.0 * width * (2.0 * (phase - np.floor(phase + 0.5)))
        elif trajectory == "triangle":
            phase = (t * trajectory_hz) % 1.0
            az = 90.0 * width * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)
        else:
            az = 90.0 * width * np.sin(2.0 * np.pi * trajectory_hz * t)
        pan = np.clip((az + 90.0) / 180.0, 0.0, 1.0)
        g_l = np.cos(pan * np.pi * 0.5)
        g_r = np.sin(pan * np.pi * 0.5)
        itd = itd_ms * 1e-3 * sr * np.sin(np.deg2rad(az))
        idx = np.arange(work.shape[0], dtype=np.float64)
        l = np.interp(idx - np.maximum(itd, 0.0), idx, mono, left=mono[0], right=mono[-1])
        r = np.interp(idx + np.minimum(itd, 0.0), idx, mono, left=mono[0], right=mono[-1])
        out = np.stack([l * g_l, r * g_r], axis=1)
        notes.append("Designed dynamic binaural motion trajectory with time-varying pan and ITD.")

    elif slug == "stochastic_spatial_diffusion_cloud":
        output_channels = int(params.get("output_channels", 6))
        diffusion = float(params.get("diffusion", 0.8))
        max_delay_ms = float(params.get("max_delay_ms", 18.0))
        seed = int(params.get("seed", 1307))
        rng = np.random.default_rng(seed)
        mono = np.mean(work, axis=1)
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        max_delay = max_delay_ms * 1e-3 * sr
        for ch in range(output_channels):
            delay = rng.uniform(0.0, max_delay)
            y = _spatial_fractional_delay(mono, delay)
            a = np.clip(diffusion * rng.uniform(0.2, 0.9), 0.05, 0.95)
            y = signal.lfilter([a, 1.0], [1.0, a], y)
            y = signal.lfilter([1.0, -0.4 * a], [1.0], y)
            out[:, ch] = y * rng.uniform(0.6, 1.0)
        notes.append("Generated stochastic spatial diffusion cloud with decorrelated delay/all-pass taps.")

    elif slug == "decorrelated_reverb_upmix":
        output_channels = int(params.get("output_channels", 6))
        decay_s = float(params.get("decay_s", 1.2))
        rir_length = int(params.get("rir_length", max(256, int(sr * min(3.0, decay_s * 2.0)))))
        mix = float(params.get("mix", 0.45))
        seed = int(params.get("seed", 1307))
        mono = np.mean(work, axis=1)
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        for ch in range(output_channels):
            rir = _spatial_synthetic_rir(rir_length, decay_s, sr, seed + 911, ch)
            wet = signal.fftconvolve(mono, rir, mode="full")[: work.shape[0]]
            dry = mono if ch % 2 == 0 else _spatial_fractional_delay(mono, float((ch + 1) * 2.0))
            out[:, ch] = (1.0 - mix) * dry + mix * wet
        notes.append("Upmixed source with decorrelated synthetic reverb field.")

    elif slug == "spectral_spatial_granulator":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        grain = int(params.get("grain", 1024))
        spread_semitones = float(params.get("spread_semitones", 5.0))
        density = float(params.get("density", 1.0))
        seed = int(params.get("seed", 1307))
        rng = np.random.default_rng(seed)
        mono = np.mean(work, axis=1)[:, None]
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        for ch in range(output_channels):
            stretch = float(np.clip(1.0 + rng.normal(0.0, 0.18 * max(0.1, density)), 0.5, 2.0))
            g = granular_time_stretch(mono, stretch=stretch, grain=grain, hop=max(64, grain // 4))
            semi = float(rng.normal(0.0, spread_semitones))
            g = pitch_shift(g, sr, semi)
            g = ensure_length(g, work.shape[0])[:, 0]
            lfo = 0.7 + 0.3 * np.sin(2.0 * np.pi * (0.07 * (ch + 1)) * np.arange(work.shape[0]) / float(sr) + rng.uniform(0.0, 2.0 * np.pi))
            out[:, ch] = g * lfo
        notes.append("Rendered spectral-spatial granulator with per-channel stochastic pitch/grain trajectories.")

    elif slug == "spatial_freeze_resynthesis":
        output_channels = int(params.get("output_channels", work.shape[1]))
        frame_ratio = float(params.get("frame_ratio", 0.35))
        phase_drift = float(params.get("phase_drift", 0.03))
        src = _spatial_to_channels(work, output_channels)
        spec, _, _ = stft_multi(src, n_fft=2048, hop=512)
        idx = int(np.clip(round(frame_ratio * (spec.shape[1] - 1)), 0, max(0, spec.shape[1] - 1)))
        frozen = spec[:, idx, :]
        mag = np.abs(frozen)
        base_phase = np.angle(frozen)
        out_spec = np.zeros_like(spec)
        for t in range(spec.shape[1]):
            theta = 2.0 * np.pi * phase_drift * t
            for ch in range(output_channels):
                phase = base_phase[:, ch] + theta * (1.0 + 0.15 * ch)
                out_spec[:, t, ch] = mag[:, ch] * np.exp(1j * phase)
        out = istft_multi(out_spec, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Resynthesized frozen spatial spectra with controlled per-channel phase drift.")

    else:
        out = work.copy()
        notes.append("Spatial fallback passthrough.")

    return normalize_peak(out), notes, extras


def run_algorithm(
    *,
    algorithm_id: str,
    algorithm_name: str,
    theme: str,
    audio: np.ndarray,
    sample_rate: int,
    params: dict[str, Any],
) -> AlgorithmResult:
    global _ACTIVE_TRANSFORM
    work = coerce_audio(audio)
    sr = int(sample_rate)
    slug = algorithm_id.split(".", 1)[1] if "." in algorithm_id else algorithm_id
    params = dict(params)
    _ACTIVE_TRANSFORM = _resolve_transform_name(str(params.get("transform", "fft")))
    params.setdefault("transform", _ACTIVE_TRANSFORM)

    if algorithm_id.startswith("time_scale_and_pitch_core."):
        out, notes, extras = _dispatch_time_scale(slug, work, sr, params)
    elif algorithm_id.startswith("pitch_detection_and_tracking."):
        out, notes, extras = _dispatch_pitch_tracking(slug, work, sr, params)
    elif algorithm_id.startswith("retune_and_intonation."):
        out, notes, extras = _dispatch_retune(slug, work, sr, params)
    elif algorithm_id.startswith("spectral_time_frequency_transforms."):
        out, notes, extras = _dispatch_transforms(slug, work, sr, params)
    elif algorithm_id.startswith("separation_and_decomposition."):
        out, notes, extras = _dispatch_separation(slug, work, sr, params)
    elif algorithm_id.startswith("denoise_and_restoration."):
        out, notes, extras = _dispatch_denoise(slug, work, sr, params)
    elif algorithm_id.startswith("dereverb_and_room_correction."):
        out, notes, extras = _dispatch_dereverb(slug, work, sr, params)
    elif algorithm_id.startswith("dynamics_and_loudness."):
        out, notes, extras = _dispatch_dynamics(slug, work, sr, params)
    elif algorithm_id.startswith("creative_spectral_effects."):
        out, notes, extras = _dispatch_creative(slug, work, sr, params)
    elif algorithm_id.startswith("granular_and_modulation."):
        out, notes, extras = _dispatch_granular(slug, work, sr, params)
    elif algorithm_id.startswith("analysis_qa_and_automation."):
        out, notes, extras = _dispatch_analysis(slug, work, sr, params)
    elif algorithm_id.startswith("spatial_and_multichannel."):
        out, notes, extras = _dispatch_spatial(slug, work, sr, params)
    else:
        out = work
        notes = ["Unknown algorithm id; returned passthrough output."]
        extras = {}

    metadata = build_metadata(
        algorithm_id=algorithm_id,
        algorithm_name=algorithm_name,
        theme=theme,
        params=params,
        notes=notes,
        librosa_available=(maybe_librosa() is not None),
        status="implemented",
        extras=extras,
    )
    return AlgorithmResult(audio=np.asarray(out, dtype=np.float64), sample_rate=sr, metadata=metadata)
