from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from pvx.algorithms.base import (
    istft_multi,
    maybe_loudnorm,
    normalize_peak,
    stft_multi,
)


def envelope_follower(signal_1d: np.ndarray, attack: float, release: float) -> np.ndarray:
    out = np.zeros_like(signal_1d)
    env = 0.0
    for i, x in enumerate(np.abs(signal_1d)):
        coef = attack if x > env else release
        env = coef * env + (1.0 - coef) * x
        out[i] = env
    return out


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


def _lufs_estimate(audio: np.ndarray, sr: int) -> float:
    pyln = maybe_loudnorm()
    mono = np.mean(audio, axis=1)
    if pyln is not None:
        meter = pyln.Meter(sr)
        try:
            min_samples = int(np.ceil(float(meter.block_size) * float(sr))) + 1
        except Exception:
            min_samples = int(0.4 * float(sr)) + 1
        if mono.size <= min_samples:
            pad = max(1, min_samples - int(mono.size) + 1)
            mono_eval = np.pad(mono, (0, pad), mode="edge")
        else:
            mono_eval = mono
        try:
            return float(meter.integrated_loudness(mono_eval))
        except Exception:
            pass
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
