#!/usr/bin/env python3

"""Auto-profile helpers for `pvx voc` and related CLI surfaces."""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

QUALITY_PROFILE_CHOICES: tuple[str, ...] = (
    "neutral",
    "speech",
    "music",
    "percussion",
    "ambient",
    "extreme",
)

QUALITY_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "neutral": {},
    "speech": {
        "phase_engine": "propagate",
        "phase_locking": "identity",
        "transient_preserve": True,
        "transient_mode": "reset",
        "window": "hann",
        "n_fft": 4096,
        "win_length": 4096,
        "hop_size": 256,
        "stretch_mode": "standard",
        "pitch_mode": "formant-preserving",
        "resample_mode": "linear",
    },
    "music": {
        "phase_engine": "propagate",
        "phase_locking": "identity",
        "transient_preserve": True,
        "transient_mode": "reset",
        "window": "blackmanharris",
        "n_fft": 4096,
        "win_length": 4096,
        "hop_size": 512,
        "stretch_mode": "auto",
        "pitch_mode": "formant-preserving",
    },
    "percussion": {
        "phase_engine": "propagate",
        "phase_locking": "identity",
        "transient_preserve": True,
        "transient_mode": "wsola",
        "transient_sensitivity": 0.68,
        "transient_protect_ms": 24.0,
        "transient_crossfade_ms": 6.0,
        "window": "kaiser",
        "kaiser_beta": 16.0,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_size": 128,
        "stretch_mode": "standard",
        "pitch_mode": "standard",
    },
    "ambient": {
        "phase_engine": "random",
        "phase_locking": "off",
        "transient_preserve": True,
        "transient_mode": "hybrid",
        "transient_sensitivity": 0.46,
        "transient_protect_ms": 36.0,
        "transient_crossfade_ms": 14.0,
        "window": "kaiser",
        "kaiser_beta": 18.0,
        "n_fft": 16384,
        "win_length": 16384,
        "hop_size": 2048,
        "stretch_mode": "multistage",
        "max_stage_stretch": 1.35,
        "onset_time_credit": True,
        "onset_credit_pull": 0.65,
        "onset_credit_max": 12.0,
        "pitch_mode": "standard",
    },
    "extreme": {
        "phase_engine": "hybrid",
        "ambient_phase_mix": 0.35,
        "phase_locking": "identity",
        "transient_preserve": True,
        "transient_mode": "hybrid",
        "transient_sensitivity": 0.54,
        "transient_protect_ms": 40.0,
        "transient_crossfade_ms": 16.0,
        "window": "kaiser",
        "kaiser_beta": 20.0,
        "n_fft": 16384,
        "win_length": 16384,
        "hop_size": 1024,
        "stretch_mode": "multistage",
        "max_stage_stretch": 1.25,
        "onset_credit_pull": 0.75,
        "onset_credit_max": 16.0,
        "pitch_mode": "formant-preserving",
        "onset_time_credit": True,
    },
}


def estimate_content_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    channel_mode: str = "mix",
    lookahead_seconds: float = 6.0,
) -> dict[str, float]:
    work = np.asarray(audio, dtype=np.float64)
    if work.ndim == 2 and work.shape[1] > 1:
        if channel_mode == "first":
            mono = work[:, 0]
        else:
            mono = np.mean(work, axis=1)
    else:
        mono = work.reshape(-1)

    max_samples = int(round(max(0.01, lookahead_seconds) * sample_rate))
    segment = mono[:max_samples] if mono.size > max_samples else mono
    if segment.size <= 8:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "crest": 1.0,
            "zcr": 0.0,
            "centroid_hz": 0.0,
            "flatness": 1.0,
            "transient_density": 0.0,
        }

    rms = float(np.sqrt(np.mean(segment * segment) + 1e-12))
    peak = float(np.max(np.abs(segment)))
    crest = peak / max(rms, 1e-12)

    signs = np.signbit(segment)
    zcr = float(np.mean(signs[1:] != signs[:-1]))

    n_fft = min(8192, max(512, int(2 ** round(math.log2(min(segment.size, 8192))))))
    win = np.hanning(n_fft)
    padded = np.zeros(n_fft, dtype=np.float64)
    padded[: min(n_fft, segment.size)] = segment[: min(n_fft, segment.size)]
    spec = np.abs(np.fft.rfft(padded * win))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    spec_sum = float(np.sum(spec))
    centroid_hz = float(np.sum(freqs * spec) / spec_sum) if spec_sum > 1e-12 else 0.0
    flatness = float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12))

    frame = 1024
    hop = 256
    if segment.size < frame:
        transient_density = 0.0
    else:
        frames = 1 + (segment.size - frame) // hop
        prev_mag = None
        flux: list[float] = []
        window = np.hanning(frame)
        for idx in range(frames):
            start = idx * hop
            chunk = segment[start : start + frame] * window
            mag = np.abs(np.fft.rfft(chunk))
            if prev_mag is not None:
                delta = np.maximum(0.0, mag - prev_mag)
                flux.append(float(np.sqrt(np.mean(delta * delta))))
            prev_mag = mag
        if flux:
            flux_np = np.asarray(flux, dtype=np.float64)
            threshold = float(np.median(flux_np) * 2.0)
            transient_density = float(np.mean(flux_np >= threshold))
        else:
            transient_density = 0.0

    return {
        "rms": rms,
        "peak": peak,
        "crest": crest,
        "zcr": zcr,
        "centroid_hz": centroid_hz,
        "flatness": flatness,
        "transient_density": transient_density,
    }


def suggest_quality_profile(*, stretch_ratio: float, features: dict[str, float]) -> str:
    ratio = max(1e-9, float(stretch_ratio))
    ratio_mag = max(ratio, 1.0 / ratio)
    if ratio_mag >= 40.0:
        return "extreme"
    if ratio_mag >= 8.0:
        return "ambient"

    zcr = float(features.get("zcr", 0.0))
    crest = float(features.get("crest", 1.0))
    flatness = float(features.get("flatness", 1.0))
    centroid = float(features.get("centroid_hz", 0.0))
    transient_density = float(features.get("transient_density", 0.0))

    if transient_density > 0.28 and zcr > 0.09 and crest > 5.0:
        return "percussion"
    if centroid < 1700.0 and zcr < 0.10 and flatness < 0.45:
        return "speech"
    return "music"


def apply_quality_profile_overrides(
    args: argparse.Namespace,
    *,
    profile: str,
    provided_flags: set[str],
) -> list[str]:
    overrides = QUALITY_PROFILE_OVERRIDES.get(profile, {})
    changed: list[str] = []
    if not overrides:
        return changed

    for key, value in overrides.items():
        cli_flag = f"--{key.replace('_', '-')}"
        if cli_flag in provided_flags:
            continue
        if not hasattr(args, key):
            continue
        setattr(args, key, value)
        changed.append(key)
    return changed
