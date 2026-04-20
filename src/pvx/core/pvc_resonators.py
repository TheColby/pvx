#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""PVC-inspired ring/resonator operators for pvx.

Phase 4 coverage:
- ring
- ringfilter
- ringtvfilter
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from pvx.core.common import coerce_audio
from pvx.core.pvc_ops import evaluate_scalar_control, load_scalar_control_points

try:
    from scipy.signal import iirpeak, lfilter
except Exception:  # pragma: no cover - optional at import-time
    iirpeak = None
    lfilter = None

RingOperatorName = Literal["ring", "ringfilter", "ringtvfilter"]
InterpMode = Literal[
    "none",
    "stairstep",
    "nearest",
    "linear",
    "cubic",
    "polynomial",
    "exponential",
    "s_curve",
    "smootherstep",
]


def _sample_times(num_samples: int, sample_rate: int) -> np.ndarray:
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.arange(num_samples, dtype=np.float64) / float(sample_rate)


def _ring_modulate(
    signal: np.ndarray,
    sample_rate: int,
    freq_track_hz: np.ndarray,
    depth_track: np.ndarray,
    mix_track: np.ndarray,
    *,
    feedback: float = 0.0,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0:
        return x.copy()

    freq = np.clip(np.asarray(freq_track_hz, dtype=np.float64).reshape(-1), 0.0, 0.5 * float(sample_rate))
    depth = np.clip(np.asarray(depth_track, dtype=np.float64).reshape(-1), 0.0, 1.0)
    mix = np.clip(np.asarray(mix_track, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if freq.size != n or depth.size != n or mix.size != n:
        raise ValueError("Ring tracks must match signal length")

    # Integrate instantaneous frequency into carrier phase for sample-accurate modulation.
    phase = np.cumsum(2.0 * np.pi * freq / float(sample_rate))
    carrier = np.sin(phase)
    mod = x * ((1.0 - depth) + depth * carrier)
    y = (1.0 - mix) * x + mix * mod

    fb = float(np.clip(feedback, 0.0, 0.999))
    if fb > 0.0:
        out = y.copy()
        # Single-pole feedback path gives controlled "ring memory" without instability.
        for idx in range(1, out.size):
            out[idx] += fb * out[idx - 1]
        y = out
    return y


def _resonant_peak_filter(
    signal: np.ndarray,
    sample_rate: int,
    *,
    center_hz: float,
    q: float,
    mix: float,
    decay: float,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()

    if iirpeak is None or lfilter is None:
        return x.copy()

    sr = float(sample_rate)
    center = float(np.clip(center_hz, 1.0, 0.5 * sr - 1.0))
    if center <= 0.0:
        return x.copy()
    q_val = max(0.1, float(q))
    w0 = center / (0.5 * sr)
    if not (0.0 < w0 < 1.0):
        return x.copy()

    # Peak IIR approximates a narrow resonator around center_hz.
    b, a = iirpeak(w0, Q=q_val)
    y = lfilter(b, a, x)

    alpha = float(np.clip(decay, 0.0, 0.999))
    if alpha > 0.0:
        mem = np.zeros_like(y)
        # Exponential tail memory controls resonator ring-down time.
        for idx in range(1, y.size):
            mem[idx] = y[idx] + alpha * mem[idx - 1]
        y = (1.0 - alpha) * mem

    wet_mix = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - wet_mix) * x + wet_mix * y


def process_ring_operator(
    audio: np.ndarray,
    sample_rate: int,
    *,
    operator: RingOperatorName,
    frequency_hz: float = 40.0,
    depth: float = 1.0,
    mix: float = 1.0,
    feedback: float = 0.0,
    resonance_hz: float = 1200.0,
    resonance_q: float = 8.0,
    resonance_mix: float = 0.35,
    resonance_decay: float = 0.2,
    tv_map_path: str | Path | None = None,
    tv_interp: InterpMode = "linear",
    tv_order: int = 3,
) -> np.ndarray:
    """Apply ring/resonator operator to mono/multichannel audio."""
    work = coerce_audio(audio)
    n_samples, channels = work.shape
    times = _sample_times(n_samples, sample_rate)

    freq_points_t, freq_points_v = load_scalar_control_points(
        tv_map_path, key="frequency_hz", default_value=float(frequency_hz)
    )
    depth_points_t, depth_points_v = load_scalar_control_points(
        tv_map_path, key="depth", default_value=float(depth)
    )
    mix_points_t, mix_points_v = load_scalar_control_points(
        tv_map_path, key="mix", default_value=float(mix)
    )

    freq_track = evaluate_scalar_control(
        times,
        freq_points_t,
        freq_points_v,
        mode=tv_interp,
        order=tv_order,
        default_value=float(frequency_hz),
    )
    depth_track = evaluate_scalar_control(
        times,
        depth_points_t,
        depth_points_v,
        mode=tv_interp,
        order=tv_order,
        default_value=float(depth),
    )
    mix_track = evaluate_scalar_control(
        times,
        mix_points_t,
        mix_points_v,
        mode=tv_interp,
        order=tv_order,
        default_value=float(mix),
    )

    if operator != "ringtvfilter":
        # Non-tv variants intentionally ignore control tracks and stay static.
        freq_track = np.full_like(freq_track, float(frequency_hz))
        depth_track = np.full_like(depth_track, float(depth))
        mix_track = np.full_like(mix_track, float(mix))

    out = np.zeros_like(work)
    for ch in range(channels):
        ringed = _ring_modulate(
            work[:, ch],
            sample_rate,
            freq_track,
            depth_track,
            mix_track,
            feedback=float(feedback),
        )
        if operator in {"ringfilter", "ringtvfilter"}:
            ringed = _resonant_peak_filter(
                ringed,
                sample_rate,
                center_hz=float(resonance_hz),
                q=float(resonance_q),
                mix=float(resonance_mix),
                decay=float(resonance_decay),
            )
        out[:, ch] = ringed
    return out
