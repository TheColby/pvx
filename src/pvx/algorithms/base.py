#!/usr/bin/env python3

"""Shared DSP utilities and implementations for pvx algorithm modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage, signal

from pvx.core.voc import (
    TRANSFORM_CHOICES as CORE_TRANSFORM_CHOICES,
)
from pvx.core.voc import (
    VocoderConfig as CoreVocoderConfig,
)
from pvx.core.voc import (
    istft as core_istft,
)
from pvx.core.voc import (
    normalize_transform_name as normalize_core_transform_name,
)
from pvx.core.voc import (
    stft as core_stft,
)

_ACTIVE_TRANSFORM = "fft"
_IMPLEMENTATION_STYLE_SHARED_DISPATCH = "shared_dispatch"


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
    except (ImportError, ModuleNotFoundError):
        return None


def maybe_loudnorm() -> Any:
    try:
        import pyloudnorm as pyln  # type: ignore

        return pyln
    except (ImportError, ModuleNotFoundError):
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


def _resolve_metadata_status(notes: list[str]) -> str:
    note_blob = " ".join(note.lower() for note in notes)
    if "unknown algorithm id" in note_blob:
        return "unsupported"
    if "fallback metadata only" in note_blob:
        return "metadata_only"
    if "fallback" in note_blob:
        return "fallback"
    return "implemented"


def _annotate_dispatch_metadata(
    *, algorithm_id: str, notes: list[str], extras: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    status = _resolve_metadata_status(notes)
    payload = dict(extras)
    payload.setdefault("implementation_style", _IMPLEMENTATION_STYLE_SHARED_DISPATCH)
    payload.setdefault("dispatch_family", algorithm_id.split(".", 1)[0])
    payload["is_fallback"] = status != "implemented"
    if status == "unsupported":
        payload.setdefault("fallback_reason", "unknown_algorithm_id")
    elif status == "metadata_only":
        payload.setdefault("fallback_reason", "metadata_only")
    elif status == "fallback":
        payload.setdefault("fallback_reason", "theme_fallback_passthrough")
    return status, payload


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
        mag[:, :, ch] = ndimage.gaussian_filter(
            mag[:, :, ch], sigma=(sigma_freq, sigma_time), mode="nearest"
        )
    return mag * np.exp(1j * pha)


def hpss_split(
    audio: np.ndarray, n_fft: int = 2048, hop: int = 512
) -> tuple[np.ndarray, np.ndarray]:
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


def granular_time_stretch(
    audio: np.ndarray, stretch: float, grain: int = 2048, hop: int = 512
) -> np.ndarray:
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


def mmse_like_denoise(
    audio: np.ndarray, alpha: float = 0.98, beta: float = 0.15, log_domain: bool = False
) -> np.ndarray:
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


def dereverb_decay_subtract(
    audio: np.ndarray, strength: float = 0.45, decay: float = 0.90
) -> np.ndarray:
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


def compressor(
    audio: np.ndarray, threshold_db: float = -18.0, ratio: float = 4.0, makeup_db: float = 0.0
) -> np.ndarray:
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


def upward_compressor(
    audio: np.ndarray, threshold_db: float = -36.0, ratio: float = 2.0
) -> np.ndarray:
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


def transient_shaper(
    audio: np.ndarray, attack_boost: float = 1.4, sustain: float = 0.92
) -> np.ndarray:
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        x = audio[:, ch]
        env_fast = envelope_follower(x, 0.65, 0.97)
        env_slow = envelope_follower(x, 0.92, 0.998)
        trans = np.maximum(0.0, env_fast - env_slow)
        mod = sustain + attack_boost * (trans / (np.max(trans) + 1e-12))
        out[:, ch] = x * mod
    return out


def spectral_dynamics(
    audio: np.ndarray, threshold_db: float = -24.0, ratio: float = 2.5
) -> np.ndarray:
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


def rhythmic_gate(
    audio: np.ndarray, sample_rate: int, rate_hz: float = 8.0, duty: float = 0.35
) -> np.ndarray:
    t = np.arange(audio.shape[0]) / float(sample_rate)
    phase = np.mod(t * rate_hz, 1.0)
    gate = (phase < duty).astype(np.float64)
    return audio * gate[:, None]


def ring_mod(
    audio: np.ndarray, sample_rate: int, freq_hz: float = 40.0, fm_depth: float = 0.0
) -> np.ndarray:
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


def estimate_f0_track(
    audio_mono: np.ndarray,
    sample_rate: int,
    fmin: float = 50.0,
    fmax: float = 1200.0,
    hop: int = 256,
) -> np.ndarray:
    librosa = maybe_librosa()
    if librosa is not None:
        try:
            f0 = librosa.yin(
                audio_mono, fmin=fmin, fmax=fmax, sr=sample_rate, frame_length=2048, hop_length=hop
            )
            return np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        except (ValueError, RuntimeError, ArithmeticError):
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


def variable_pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    semitone_track: np.ndarray,
    hop: int = 256,
    frame: int = 1024,
) -> np.ndarray:
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


def cqt_or_stft(
    audio: np.ndarray, sample_rate: int, bins_per_octave: int = 24
) -> tuple[np.ndarray, dict[str, Any]]:
    librosa = maybe_librosa()
    if librosa is None:
        spec, _, _ = stft_multi(audio, n_fft=4096, hop=512)
        return spec, {"mode": "stft", "n_fft": 4096, "hop": 512}
    fmin = float(librosa.note_to_hz("C1"))
    nyquist = 0.5 * float(sample_rate)
    max_n_bins = int(
        np.floor(bins_per_octave * np.log2(max((nyquist * 0.98) / max(fmin, 1e-12), 1e-12)))
    )
    target_n_bins = 8 * bins_per_octave
    n_bins = min(target_n_bins, max_n_bins)
    if n_bins < bins_per_octave:
        spec, _, _ = stft_multi(audio, n_fft=4096, hop=512)
        return spec, {"mode": "stft", "n_fft": 4096, "hop": 512}
    out_specs: list[np.ndarray] = []
    for ch in range(audio.shape[1]):
        c = librosa.cqt(
            audio[:, ch],
            sr=sample_rate,
            bins_per_octave=bins_per_octave,
            n_bins=n_bins,
            fmin=fmin,
        )
        out_specs.append(c)
    max_bins = max(v.shape[0] for v in out_specs)
    max_frames = max(v.shape[1] for v in out_specs)
    arr = np.zeros((max_bins, max_frames, audio.shape[1]), dtype=np.complex128)
    for idx, c in enumerate(out_specs):
        arr[: c.shape[0], : c.shape[1], idx] = c
    return arr, {"mode": "cqt", "bins_per_octave": bins_per_octave, "n_bins": n_bins, "fmin": fmin}


def icqt_or_istft(
    spec: np.ndarray,
    sample_rate: int,
    length: int,
    transform_meta: dict[str, Any] | None = None,
) -> np.ndarray:
    meta = transform_meta or {}
    librosa = maybe_librosa()
    if librosa is None or str(meta.get("mode", "stft")) != "cqt":
        n_fft = int(meta.get("n_fft", 4096))
        hop = int(meta.get("hop", 512))
        return istft_multi(spec, n_fft=n_fft, hop=hop, length=length)
    channels: list[np.ndarray] = []
    bins_per_octave = int(meta.get("bins_per_octave", 24))
    fmin = float(meta.get("fmin", librosa.note_to_hz("C1")))
    try:
        for ch in range(spec.shape[2]):
            c = spec[:, :, ch]
            y = librosa.icqt(
                c,
                sr=sample_rate,
                length=length,
                bins_per_octave=bins_per_octave,
                fmin=fmin,
            )
            channels.append(y.astype(np.float64, copy=False))
    except (ImportError, ModuleNotFoundError, ValueError, RuntimeError):
        n_fft = int(meta.get("n_fft", 4096))
        hop = int(meta.get("hop", 512))
        return istft_multi(spec, n_fft=n_fft, hop=hop, length=length)
    out = np.stack(channels, axis=1)
    return ensure_length(out, length)


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
        from pvx.algorithms.pitch_dispatch import dispatch_time_scale

        out, notes, extras = dispatch_time_scale(slug, work, sr, params)
    elif algorithm_id.startswith("pitch_detection_and_tracking."):
        from pvx.algorithms.pitch_dispatch import dispatch_pitch_tracking

        out, notes, extras = dispatch_pitch_tracking(slug, work, sr, params)
    elif algorithm_id.startswith("retune_and_intonation."):
        from pvx.algorithms.pitch_dispatch import dispatch_retune

        out, notes, extras = dispatch_retune(slug, work, sr, params)
    elif algorithm_id.startswith("spectral_time_frequency_transforms."):
        from pvx.algorithms.pitch_dispatch import dispatch_transforms

        out, notes, extras = dispatch_transforms(slug, work, sr, params)
    elif algorithm_id.startswith("separation_and_decomposition."):
        from pvx.algorithms.audio_dispatch import dispatch_separation

        out, notes, extras = dispatch_separation(slug, work, sr, params)
    elif algorithm_id.startswith("denoise_and_restoration."):
        from pvx.algorithms.audio_dispatch import dispatch_denoise

        out, notes, extras = dispatch_denoise(slug, work, sr, params)
    elif algorithm_id.startswith("dereverb_and_room_correction."):
        from pvx.algorithms.audio_dispatch import dispatch_dereverb

        out, notes, extras = dispatch_dereverb(slug, work, sr, params)
    elif algorithm_id.startswith("dynamics_and_loudness."):
        from pvx.algorithms.audio_dispatch import dispatch_dynamics

        out, notes, extras = dispatch_dynamics(slug, work, sr, params)
    elif algorithm_id.startswith("creative_spectral_effects."):
        from pvx.algorithms.effects_dispatch import dispatch_creative

        out, notes, extras = dispatch_creative(slug, work, sr, params)
    elif algorithm_id.startswith("granular_and_modulation."):
        from pvx.algorithms.effects_dispatch import dispatch_granular

        out, notes, extras = dispatch_granular(slug, work, sr, params)
    elif algorithm_id.startswith("analysis_qa_and_automation."):
        from pvx.algorithms.effects_dispatch import dispatch_analysis

        out, notes, extras = dispatch_analysis(slug, work, sr, params)
    elif algorithm_id.startswith("spatial_and_multichannel."):
        from pvx.algorithms.spatial_dispatch import dispatch_spatial

        out, notes, extras = dispatch_spatial(slug, work, sr, params)
    else:
        out = work
        notes = ["Unknown algorithm id; returned passthrough output."]
        extras = {}

    status, extras = _annotate_dispatch_metadata(
        algorithm_id=algorithm_id,
        notes=notes,
        extras=extras,
    )
    metadata = build_metadata(
        algorithm_id=algorithm_id,
        algorithm_name=algorithm_name,
        theme=theme,
        params=params,
        notes=notes,
        librosa_available=(maybe_librosa() is not None),
        status=status,
        extras=extras,
    )
    return AlgorithmResult(
        audio=np.asarray(out, dtype=np.float64), sample_rate=sr, metadata=metadata
    )
