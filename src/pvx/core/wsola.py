# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Deterministic WSOLA time-stretch primitives for transient handling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pvx.core.device import _array_module, _is_cupy_array, _to_numpy, _to_runtime_array, runtime_config


def _safe_window(length: int, xp: Any = np) -> Any:
    n = max(2, int(length))
    if hasattr(xp, "hanning"):
        return xp.hanning(n).astype(xp.float64)
    # Fallback for older CuPy or other backends
    return xp.asarray(np.hanning(n)).astype(xp.float64)


def wsola_time_stretch(
    signal: np.ndarray,
    stretch: float,
    sample_rate: int,
    *,
    frame_ms: float = 40.0,
    analysis_hop_ms: float = 10.0,
    search_ms: float = 12.0,
) -> np.ndarray:
    """Time-stretch a mono signal using deterministic WSOLA.

    This implementation is designed for reproducible CPU behavior:
    - deterministic candidate scan order
    - no random tie-breaking
    """

    bridge_to_cuda = runtime_config().active_device == "cuda" and not _is_cupy_array(signal)
    work_signal = _to_runtime_array(signal) if bridge_to_cuda else signal
    xp = _array_module(work_signal)

    x = xp.asarray(work_signal, dtype=xp.float64).reshape(-1)
    if x.size == 0:
        out = x.copy()
        return _to_numpy(out) if bridge_to_cuda else out

    ratio = float(stretch)
    if ratio <= 0.0:
        raise ValueError("stretch must be > 0")
    if abs(ratio - 1.0) <= 1e-12:
        out = x.copy()
        return _to_numpy(out) if bridge_to_cuda else out

    frame_len = max(64, int(round(max(8.0, frame_ms) * sample_rate / 1000.0)))
    analysis_hop = max(1, int(round(max(1.0, analysis_hop_ms) * sample_rate / 1000.0)))
    synthesis_hop = max(1, int(round(analysis_hop * ratio)))
    search_radius = max(0, int(round(max(0.0, search_ms) * sample_rate / 1000.0)))

    if frame_len > x.size:
        frame_len = int(x.size)
    if frame_len < 8:
        target = max(1, int(round(x.size * ratio)))
        out = xp.interp(
            xp.linspace(0.0, 1.0, target, endpoint=True),
            xp.linspace(0.0, 1.0, x.size, endpoint=True),
            x,
        ).astype(xp.float64)
        return _to_numpy(out) if bridge_to_cuda else out

    window = _safe_window(frame_len, xp=xp)
    target_len = max(1, int(round(x.size * ratio)))
    max_out = target_len + frame_len + 2 * max(analysis_hop, synthesis_hop)
    y = xp.zeros(max_out, dtype=xp.float64)
    weight = xp.zeros(max_out, dtype=xp.float64)

    # Seed first frame.
    y[:frame_len] += x[:frame_len] * window
    weight[:frame_len] += window * window

    overlap = max(16, min(frame_len - 1, frame_len - synthesis_hop))
    search_step = max(1, analysis_hop // 8)

    prev_in = 0
    frame_idx = 1
    while True:
        out_pos = frame_idx * synthesis_hop
        if out_pos >= target_len:
            break

        expected_in = int(round(prev_in + analysis_hop))
        lo = max(0, expected_in - search_radius)
        hi = min(max(0, x.size - frame_len), expected_in + search_radius)
        if hi < lo:
            hi = lo

        out_overlap = y[out_pos : out_pos + overlap]
        out_w = weight[out_pos : out_pos + overlap]
        out_norm_arr = out_overlap.copy()
        nz = out_w > 1e-9
        out_norm_arr[nz] = out_overlap[nz] / out_w[nz]

        best_pos = int(xp.clip(expected_in, 0, max(0, x.size - frame_len)))
        best_score = -math.inf

        if out_norm_arr.size >= 4 and xp.any(xp.abs(out_norm_arr) > 1e-10):
            ref_norm = xp.linalg.norm(out_norm_arr) + 1e-12
            cands = xp.arange(lo, hi + 1, search_step)
            cands = cands[cands + overlap <= x.size]
            if cands.size > 0:
                shape = (cands.size, overlap)
                strides = (x.strides[0] * search_step, x.strides[0])
                segments = xp.lib.stride_tricks.as_strided(x[lo:], shape=shape, strides=strides)

                dots = segments @ out_norm_arr
                seg_norms = xp.linalg.norm(segments, axis=1)
                scores = dots / ((seg_norms + 1e-12) * ref_norm)

                best_idx = int(xp.argmax(scores))
                best_score = float(scores[best_idx])
                best_pos = int(cands[best_idx])

        frame = x[best_pos : best_pos + frame_len]
        if frame.size < frame_len:
            frame = xp.pad(frame, (0, frame_len - frame.size), mode="constant")
        y[out_pos : out_pos + frame_len] += frame * window
        weight[out_pos : out_pos + frame_len] += window * window

        prev_in = best_pos
        frame_idx += 1

    nz = weight > 1e-9
    y[nz] /= weight[nz]
    if xp.any(~nz):
        y[~nz] = 0.0
    if y.size < target_len:
        y = xp.pad(y, (0, target_len - y.size), mode="constant")

    out = y[:target_len].astype(xp.float64, copy=False)
    return _to_numpy(out) if bridge_to_cuda else out
