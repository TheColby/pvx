"""Inter-channel coherence drift metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _principal(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex128)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size), mode="constant")
    rem = (x.size - n_fft) % hop
    if rem:
        x = np.pad(x, (0, hop - rem), mode="constant")
    frames = 1 + (x.size - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float64)
    out = np.empty((n_fft // 2 + 1, frames), dtype=np.complex128)
    for idx in range(frames):
        start = idx * hop
        frame = x[start : start + n_fft]
        out[:, idx] = np.fft.rfft(frame * win, n=n_fft)
    return out


def interchannel_coherence_drift(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    n_fft: int = 1024,
    hop_size: int = 256,
) -> dict[str, Any]:
    """Measure drift in inter-channel phase relationships.

    Returns a dict with:
    - `overall_drift_rad` (lower is better)
    - `per_channel_drift_rad` (relative to channel 0)
    """

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2:
        raise ValueError("reference and candidate must be 2D arrays (samples, channels)")
    if ref.shape[1] != cand.shape[1]:
        raise ValueError("reference/candidate must have matching channel count")
    if ref.shape[1] < 2:
        return {
            "overall_drift_rad": 0.0,
            "per_channel_drift_rad": [],
        }

    ch = ref.shape[1]
    min_len = min(ref.shape[0], cand.shape[0])
    ref = ref[:min_len, :]
    cand = cand[:min_len, :]

    ref_specs = [_stft(ref[:, idx], n_fft=n_fft, hop=hop_size) for idx in range(ch)]
    cand_specs = [_stft(cand[:, idx], n_fft=n_fft, hop=hop_size) for idx in range(ch)]
    min_frames = min(spec.shape[1] for spec in ref_specs + cand_specs)
    if min_frames <= 0:
        return {
            "overall_drift_rad": 0.0,
            "per_channel_drift_rad": [0.0 for _ in range(ch - 1)],
        }
    ref_specs = [spec[:, :min_frames] for spec in ref_specs]
    cand_specs = [spec[:, :min_frames] for spec in cand_specs]

    per_channel: list[float] = []
    ref0 = ref_specs[0]
    cand0 = cand_specs[0]
    for idx in range(1, ch):
        ref_delta = np.angle(ref_specs[idx] * np.conj(ref0))
        cand_delta = np.angle(cand_specs[idx] * np.conj(cand0))
        drift = _principal(cand_delta - ref_delta)
        weight = np.abs(ref_specs[idx]) * np.abs(ref0)
        score = float(np.sum(np.abs(drift) * weight) / (np.sum(weight) + 1e-12))
        per_channel.append(score)

    overall = float(np.mean(per_channel)) if per_channel else 0.0
    return {
        "overall_drift_rad": overall,
        "per_channel_drift_rad": per_channel,
    }


def stereo_coherence_drift_score(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    n_fft: int = 1024,
    hop_size: int = 256,
) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2:
        return 0.0
    if ref.shape[1] < 2 or cand.shape[1] < 2:
        return 0.0
    report = interchannel_coherence_drift(ref[:, :2], cand[:, :2], n_fft=n_fft, hop_size=hop_size)
    return float(report["overall_drift_rad"])

