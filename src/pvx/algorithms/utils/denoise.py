#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Dispatch logic for denoise and restoration algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def _dispatch_denoise(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    # Local imports to avoid circular dependencies
    from pvx.algorithms.base import (
        mmse_like_denoise,
        minimum_statistics_denoise,
        normalize_peak,
        simple_declick,
        simple_declip,
        spectral_gate,
    )

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
