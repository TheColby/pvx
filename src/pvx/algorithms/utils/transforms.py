#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

from __future__ import annotations

import numpy as np

from pvx.core.voc import (
    TRANSFORM_CHOICES as CORE_TRANSFORM_CHOICES,
    VocoderConfig as CoreVocoderConfig,
    istft as core_istft,
    normalize_transform_name as normalize_core_transform_name,
    stft as core_stft,
)

_ACTIVE_TRANSFORM = "fft"

def _get_ensure_length():
    from pvx.algorithms.base import ensure_length
    return ensure_length

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
    ensure_length = _get_ensure_length()
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
    ensure_length = _get_ensure_length()
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
