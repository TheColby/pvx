"""Deterministic WSOLA time-stretch primitives for transient handling."""

from __future__ import annotations

import math

import numpy as np


def _safe_window(length: int) -> np.ndarray:
    n = max(2, int(length))
    return np.hanning(n).astype(np.float64)


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

    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()
    ratio = float(stretch)
    if ratio <= 0.0:
        raise ValueError("stretch must be > 0")
    if abs(ratio - 1.0) <= 1e-12:
        return x.copy()

    frame_len = max(64, int(round(max(8.0, frame_ms) * sample_rate / 1000.0)))
    analysis_hop = max(1, int(round(max(1.0, analysis_hop_ms) * sample_rate / 1000.0)))
    synthesis_hop = max(1, int(round(analysis_hop * ratio)))
    search_radius = max(0, int(round(max(0.0, search_ms) * sample_rate / 1000.0)))

    if frame_len > x.size:
        frame_len = int(x.size)
    if frame_len < 8:
        target = max(1, int(round(x.size * ratio)))
        return np.interp(
            np.linspace(0.0, 1.0, target, endpoint=True),
            np.linspace(0.0, 1.0, x.size, endpoint=True),
            x,
        ).astype(np.float64)

    window = _safe_window(frame_len)
    target_len = max(1, int(round(x.size * ratio)))
    max_out = target_len + frame_len + 2 * max(analysis_hop, synthesis_hop)
    y = np.zeros(max_out, dtype=np.float64)
    weight = np.zeros(max_out, dtype=np.float64)

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
        out_norm = out_overlap.copy()
        nz = out_w > 1e-9
        out_norm[nz] = out_overlap[nz] / out_w[nz]

        best_pos = int(np.clip(expected_in, 0, max(0, x.size - frame_len)))
        best_score = -math.inf
        if out_norm.size >= 4 and np.any(np.abs(out_norm) > 1e-10):
            ref_norm = np.linalg.norm(out_norm) + 1e-12
            for cand in range(lo, hi + 1, search_step):
                seg = x[cand : cand + overlap]
                if seg.size != overlap:
                    continue
                denom = (np.linalg.norm(seg) + 1e-12) * ref_norm
                score = float(np.dot(seg, out_norm) / denom)
                if score > best_score:
                    best_score = score
                    best_pos = cand

        frame = x[best_pos : best_pos + frame_len]
        if frame.size < frame_len:
            frame = np.pad(frame, (0, frame_len - frame.size), mode="constant")
        y[out_pos : out_pos + frame_len] += frame * window
        weight[out_pos : out_pos + frame_len] += window * window

        prev_in = best_pos
        frame_idx += 1

    nz = weight > 1e-9
    y[nz] /= weight[nz]
    if np.any(~nz):
        y[~nz] = 0.0
    if y.size < target_len:
        y = np.pad(y, (0, target_len - y.size), mode="constant")
    return y[:target_len].astype(np.float64, copy=False)
