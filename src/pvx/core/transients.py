# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Transient analysis and segmentation helpers for hybrid pvx modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy import fft as scipy_fft
except Exception:  # pragma: no cover - optional dependency
    scipy_fft = None

from pvx.core.device import _array_module, _is_cupy_array, _to_numpy, _to_runtime_array, runtime_config


@dataclass(frozen=True)
class TransientFeatures:
    flux: np.ndarray
    hfc: np.ndarray
    broadbandness: np.ndarray
    score: np.ndarray
    frame_times_s: np.ndarray
    hop_size: int
    n_fft: int


@dataclass(frozen=True)
class TransientRegion:
    start_sample: int
    end_sample: int
    is_transient: bool


def _principal(x: Any, xp: Any = np) -> Any:
    return (x + xp.pi) % (2.0 * xp.pi) - xp.pi


def _normalize_robust(values: Any) -> Any:
    xp = _array_module(values)
    if values.size == 0:
        return values.astype(xp.float64)
    x = xp.asarray(values, dtype=xp.float64)
    if hasattr(xp, "percentile"):
        lo = float(xp.percentile(x, 10.0))
        hi = float(xp.percentile(x, 90.0))
    else:
        # Fallback if percentile is missing (e.g. some CuPy versions)
        lo = float(xp.min(x))
        hi = float(xp.max(x))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        hi = float(xp.max(x))
        lo = float(xp.min(x))
    span = max(1e-12, hi - lo)
    return xp.clip((x - lo) / span, 0.0, 1.0)


def _frame_signal(signal: np.ndarray, n_fft: int, hop_size: int, *, center: bool = True) -> Any:
    xp = _array_module(signal)
    x = xp.asarray(signal, dtype=xp.float64).reshape(-1)
    if x.size == 0:
        return xp.zeros((n_fft, 0), dtype=xp.float64)

    if center:
        pad = n_fft // 2
        x = xp.pad(x, (pad, pad), mode="constant")
    if x.size < n_fft:
        x = xp.pad(x, (0, n_fft - x.size), mode="constant")
    rem = (x.size - n_fft) % hop_size
    if rem:
        x = xp.pad(x, (0, hop_size - rem), mode="constant")

    frame_count = 1 + (x.size - n_fft) // hop_size
    shape = (frame_count, n_fft)
    strides = (x.strides[0] * hop_size, x.strides[0])
    frames = xp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return frames.T.copy()


def compute_transient_features(
    signal: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_size: int,
    center: bool = True,
) -> TransientFeatures:
    bridge_to_cuda = runtime_config().active_device == "cuda" and not _is_cupy_array(signal)
    work_signal = _to_runtime_array(signal) if bridge_to_cuda else signal
    xp = _array_module(work_signal)

    frames = _frame_signal(work_signal, n_fft, hop_size, center=center)
    frame_count = frames.shape[1]
    if frame_count == 0:
        z = np.zeros(0, dtype=np.float64)
        return TransientFeatures(
            flux=z,
            hfc=z,
            broadbandness=z,
            score=z,
            frame_times_s=z,
            hop_size=hop_size,
            n_fft=n_fft,
        )

    if hasattr(xp, "hanning"):
        window = xp.hanning(n_fft).astype(xp.float64)
    else:
        window = xp.asarray(np.hanning(n_fft)).astype(xp.float64)

    if xp is np and scipy_fft is not None:
        spec = scipy_fft.rfft(frames * window[:, None], axis=0)
    else:
        spec = xp.fft.rfft(frames * window[:, None], axis=0)

    mag = xp.abs(spec).astype(xp.float64)
    bins = mag.shape[0]

    # Spectral flux (positive differences only).
    flux = xp.zeros(frame_count, dtype=xp.float64)
    if frame_count > 1:
        delta = xp.maximum(0.0, mag[:, 1:] - mag[:, :-1])
        flux[1:] = xp.sqrt(xp.sum(delta * delta, axis=0))

    # High-frequency content.
    freq_idx = xp.arange(bins, dtype=xp.float64) + 1.0
    hfc = xp.sum(mag * freq_idx[:, None], axis=0)

    # Broadbandness = blend of spectral flatness and HF energy ratio.
    flatness = xp.exp(xp.mean(xp.log(mag + 1e-12), axis=0)) / (xp.mean(mag + 1e-12, axis=0))
    hf_start = int(round(0.35 * (bins - 1)))
    hf_energy = xp.sum(mag[hf_start:, :], axis=0)
    total_energy = xp.sum(mag, axis=0) + 1e-12
    hf_ratio = hf_energy / total_energy
    broadbandness = 0.6 * flatness + 0.4 * hf_ratio

    flux_n = _normalize_robust(flux)
    hfc_n = _normalize_robust(hfc)
    broad_n = _normalize_robust(broadbandness)
    score = xp.clip(0.5 * flux_n + 0.3 * hfc_n + 0.2 * broad_n, 0.0, 1.0)

    frame_times_s = (xp.arange(frame_count, dtype=xp.float64) * hop_size) / float(max(1, sample_rate))

    res = TransientFeatures(
        flux=_to_numpy(flux),
        hfc=_to_numpy(hfc),
        broadbandness=_to_numpy(broadbandness),
        score=_to_numpy(score),
        frame_times_s=_to_numpy(frame_times_s),
        hop_size=hop_size,
        n_fft=n_fft,
    )
    return res


def pick_onset_frames(
    features: TransientFeatures,
    *,
    sensitivity: float,
    min_separation_frames: int,
) -> np.ndarray:
    xp = _array_module(features.score)
    score = xp.asarray(features.score, dtype=xp.float64)
    if score.size == 0:
        return np.zeros(0, dtype=np.int64)
    if score.size <= 2:
        return np.array([int(xp.argmax(score))], dtype=np.int64) if score.size else np.zeros(0, dtype=np.int64)

    s = float(np.clip(sensitivity, 0.0, 1.0))
    if hasattr(xp, "quantile"):
        threshold = float(xp.quantile(score, 0.92 - 0.50 * s))
    else:
        # Fallback for old CuPy
        score_np = _to_numpy(score)
        threshold = float(np.quantile(score_np, 0.92 - 0.50 * s))

    threshold = float(np.clip(threshold, 0.08, 0.95))

    local_max = xp.zeros(score.size, dtype=bool)
    local_max[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:])
    candidates = xp.flatnonzero(local_max & (score >= threshold))
    if candidates.size == 0:
        return np.zeros(0, dtype=np.int64)

    sep = max(1, int(min_separation_frames))
    picked: list[int] = []
    candidates_np = _to_numpy(candidates)
    score_np = _to_numpy(score)
    for idx in candidates_np:
        if not picked or idx - picked[-1] >= sep:
            picked.append(int(idx))
        elif score_np[idx] > score_np[picked[-1]]:
            picked[-1] = int(idx)
    return np.asarray(picked, dtype=np.int64)


def _mask_to_regions(mask: np.ndarray) -> list[TransientRegion]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []

    regions: list[TransientRegion] = []
    start = 0
    current = bool(values[0])
    for idx in range(1, values.size):
        state = bool(values[idx])
        if state != current:
            regions.append(TransientRegion(start_sample=start, end_sample=idx, is_transient=current))
            start = idx
            current = state
    regions.append(TransientRegion(start_sample=start, end_sample=values.size, is_transient=current))
    return regions


def _enforce_min_region_samples(mask: np.ndarray, min_region_samples: int) -> np.ndarray:
    if min_region_samples <= 1:
        return np.asarray(mask, dtype=bool)
    out = np.asarray(mask, dtype=bool).copy()
    for region in _mask_to_regions(out):
        if (region.end_sample - region.start_sample) < min_region_samples:
            out[region.start_sample : region.end_sample] = not region.is_transient
    return out


def build_transient_mask(
    signal_samples: int,
    onset_samples: np.ndarray,
    *,
    protect_samples: int,
    merge_gap_samples: int,
    min_region_samples: int,
) -> np.ndarray:
    n = max(0, int(signal_samples))
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    if onset_samples.size == 0:
        return mask

    protect = max(1, int(protect_samples))
    pre = max(1, int(round(0.35 * protect)))
    post = max(1, int(round(0.65 * protect)))
    for onset in np.asarray(onset_samples, dtype=np.int64):
        center = int(np.clip(onset, 0, n - 1))
        start = max(0, center - pre)
        end = min(n, center + post)
        mask[start:end] = True

    # Merge close transient islands to avoid unstable toggling.
    gap = max(0, int(merge_gap_samples))
    if gap > 0:
        true_idx = np.flatnonzero(mask)
        if true_idx.size:
            # Vectorized gap filling
            diffs = np.diff(true_idx)
            fill_starts = true_idx[:-1][diffs <= gap]
            fill_ends = true_idx[1:][diffs <= gap]
            for s, e in zip(fill_starts, fill_ends):
                mask[s : e + 1] = True

    return _enforce_min_region_samples(mask, int(max(1, min_region_samples)))


def map_mask_to_output(mask_in: np.ndarray, stretch: float, output_samples: int) -> np.ndarray:
    src = np.asarray(mask_in, dtype=bool).reshape(-1)
    out_len = max(0, int(output_samples))
    if out_len == 0:
        return np.zeros(0, dtype=bool)
    if src.size == 0:
        return np.zeros(out_len, dtype=bool)
    ratio = max(1e-9, float(stretch))
    idx = np.rint(np.arange(out_len, dtype=np.float64) / ratio).astype(np.int64)
    idx = np.clip(idx, 0, src.size - 1)
    return src[idx]


def smooth_binary_mask(mask: np.ndarray, fade_samples: int) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float64).reshape(-1)
    width = max(1, int(fade_samples))
    if values.size == 0 or width <= 1:
        return np.clip(values, 0.0, 1.0)
    kernel = np.ones(width, dtype=np.float64) / float(width)
    smoothed = np.convolve(values, kernel, mode="same")
    smoothed = np.convolve(smoothed, kernel, mode="same")
    return np.clip(smoothed, 0.0, 1.0)


def detect_transient_regions(
    signal: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_size: int,
    sensitivity: float,
    protect_ms: float,
    crossfade_ms: float,
    center: bool = True,
) -> tuple[TransientFeatures, np.ndarray, list[TransientRegion]]:
    features = compute_transient_features(
        signal,
        sample_rate,
        n_fft=max(64, int(n_fft)),
        hop_size=max(1, int(hop_size)),
        center=center,
    )
    protect_samples = int(round(max(2.0, float(protect_ms)) * sample_rate / 1000.0))
    crossfade_samples = int(round(max(0.0, float(crossfade_ms)) * sample_rate / 1000.0))
    min_sep_frames = max(1, int(round((0.5 * protect_samples) / float(max(1, hop_size)))))
    onset_frames = pick_onset_frames(
        features,
        sensitivity=float(np.clip(sensitivity, 0.0, 1.0)),
        min_separation_frames=min_sep_frames,
    )
    onset_samples = onset_frames.astype(np.int64) * max(1, int(hop_size))
    mask = build_transient_mask(
        signal_samples=int(np.asarray(signal).size),
        onset_samples=onset_samples,
        protect_samples=protect_samples,
        merge_gap_samples=max(crossfade_samples, protect_samples // 3),
        min_region_samples=max(8, crossfade_samples),
    )
    regions = _mask_to_regions(mask)
    return features, mask, regions
