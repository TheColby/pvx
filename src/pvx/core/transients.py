"""Transient analysis and segmentation helpers for hybrid pvx modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _principal(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _normalize_robust(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    x = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(x, 10.0))
    hi = float(np.percentile(x, 90.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        hi = float(np.max(x))
        lo = float(np.min(x))
    span = max(1e-12, hi - lo)
    return np.clip((x - lo) / span, 0.0, 1.0)


def _frame_signal(signal: np.ndarray, n_fft: int, hop_size: int, *, center: bool = True) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((n_fft, 0), dtype=np.float64)

    if center:
        pad = n_fft // 2
        x = np.pad(x, (pad, pad), mode="constant")
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size), mode="constant")
    rem = (x.size - n_fft) % hop_size
    if rem:
        x = np.pad(x, (0, hop_size - rem), mode="constant")

    frame_count = 1 + (x.size - n_fft) // hop_size
    out = np.empty((n_fft, frame_count), dtype=np.float64)
    for idx in range(frame_count):
        start = idx * hop_size
        out[:, idx] = x[start : start + n_fft]
    return out


def compute_transient_features(
    signal: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_size: int,
    center: bool = True,
) -> TransientFeatures:
    frames = _frame_signal(signal, n_fft, hop_size, center=center)
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

    window = np.hanning(n_fft).astype(np.float64)
    spec = np.fft.rfft(frames * window[:, None], axis=0)
    mag = np.abs(spec).astype(np.float64)
    bins = mag.shape[0]

    # Spectral flux (positive differences only).
    flux = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        delta = np.maximum(0.0, mag[:, 1:] - mag[:, :-1])
        flux[1:] = np.sqrt(np.sum(delta * delta, axis=0))

    # High-frequency content.
    freq_idx = np.arange(bins, dtype=np.float64) + 1.0
    hfc = np.sum(mag * freq_idx[:, None], axis=0)

    # Broadbandness = blend of spectral flatness and HF energy ratio.
    flatness = np.exp(np.mean(np.log(mag + 1e-12), axis=0)) / (np.mean(mag + 1e-12, axis=0))
    hf_start = int(round(0.35 * (bins - 1)))
    hf_energy = np.sum(mag[hf_start:, :], axis=0)
    total_energy = np.sum(mag, axis=0) + 1e-12
    hf_ratio = hf_energy / total_energy
    broadbandness = 0.6 * flatness + 0.4 * hf_ratio

    flux_n = _normalize_robust(flux)
    hfc_n = _normalize_robust(hfc)
    broad_n = _normalize_robust(broadbandness)
    score = np.clip(0.5 * flux_n + 0.3 * hfc_n + 0.2 * broad_n, 0.0, 1.0)

    frame_times_s = (np.arange(frame_count, dtype=np.float64) * hop_size) / float(max(1, sample_rate))
    return TransientFeatures(
        flux=flux,
        hfc=hfc,
        broadbandness=broadbandness,
        score=score,
        frame_times_s=frame_times_s,
        hop_size=hop_size,
        n_fft=n_fft,
    )


def pick_onset_frames(
    features: TransientFeatures,
    *,
    sensitivity: float,
    min_separation_frames: int,
) -> np.ndarray:
    score = np.asarray(features.score, dtype=np.float64)
    if score.size == 0:
        return np.zeros(0, dtype=np.int64)
    if score.size <= 2:
        return np.array([int(np.argmax(score))], dtype=np.int64) if score.size else np.zeros(0, dtype=np.int64)

    s = float(np.clip(sensitivity, 0.0, 1.0))
    threshold = float(np.quantile(score, 0.92 - 0.50 * s))
    threshold = float(np.clip(threshold, 0.08, 0.95))

    local_max = np.zeros(score.size, dtype=bool)
    local_max[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:])
    candidates = np.flatnonzero(local_max & (score >= threshold))
    if candidates.size == 0:
        return np.zeros(0, dtype=np.int64)

    sep = max(1, int(min_separation_frames))
    picked: list[int] = []
    for idx in candidates:
        if not picked or idx - picked[-1] >= sep:
            picked.append(int(idx))
        elif score[idx] > score[picked[-1]]:
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
            for idx in range(1, true_idx.size):
                if (true_idx[idx] - true_idx[idx - 1]) <= gap:
                    mask[true_idx[idx - 1] : true_idx[idx] + 1] = True

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

