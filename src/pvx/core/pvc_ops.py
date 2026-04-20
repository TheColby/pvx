#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""PVC-inspired response-driven spectral operators for pvx.

Phase 3 coverage:
- filter
- tvfilter
- noisefilter
- bandamp
- spec-compander
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np

from pvx.core.common import coerce_audio
from pvx.core.response_store import ResponseArtifact
from pvx.core.voc import VocoderConfig, istft, stft

OperatorName = Literal["filter", "tvfilter", "noisefilter", "bandamp", "spec-compander"]
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


def _piecewise_segment_fraction(
    query_t: np.ndarray,
    control_t: np.ndarray,
    *,
    side: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map query times to control-point segments and normalized local fractions."""
    idx = np.searchsorted(control_t, query_t, side=side) - 1
    idx = np.clip(idx, 0, control_t.size - 2)
    t0 = control_t[idx]
    t1 = control_t[idx + 1]
    span = np.maximum(1e-12, t1 - t0)
    u = (query_t - t0) / span
    u = np.clip(u, 0.0, 1.0)
    return idx, t0, u


def _smoothstep(u: np.ndarray) -> np.ndarray:
    # Cubic Hermite easing in [0,1] with zero first derivative at boundaries.
    return (u * u) * (3.0 - 2.0 * u)


def _smootherstep(u: np.ndarray) -> np.ndarray:
    # Quintic Hermite easing with zero first and second derivatives at boundaries.
    return (u * u * u) * (u * (u * 6.0 - 15.0) + 10.0)


def _exp_ease(u: np.ndarray, *, strength: float) -> np.ndarray:
    # Normalized exponential ease-in curve; strength>=0 controls convexity.
    k = max(0.0, float(strength))
    if k <= 1e-9:
        return u
    num = np.expm1(k * u)
    den = np.expm1(k)
    if abs(den) <= 1e-12:
        return u
    return num / den


def db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _resize_curve(values: np.ndarray, target_bins: int) -> np.ndarray:
    src = np.asarray(values, dtype=np.float64).reshape(-1)
    if target_bins <= 0:
        raise ValueError("target_bins must be > 0")
    if src.size == target_bins:
        return src.copy()
    if src.size == 0:
        return np.ones(target_bins, dtype=np.float64)
    # Normalize to [0,1] so the mapping is shape-preserving regardless of FFT size.
    x_old = np.linspace(0.0, 1.0, num=src.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_bins, endpoint=True)
    return np.interp(x_new, x_old, src)


def _shift_response_curve(curve: np.ndarray, *, shift_bins: int, transpose_semitones: float) -> np.ndarray:
    src = np.asarray(curve, dtype=np.float64).reshape(-1)
    n = src.size
    if n == 0:
        return src
    ratio = float(2.0 ** (float(transpose_semitones) / 12.0))
    if ratio <= 0.0:
        ratio = 1.0
    # Bin shift and pitch transpose are applied in one interpolation step.
    indices = (np.arange(n, dtype=np.float64) - float(shift_bins)) / ratio
    x_old = np.arange(n, dtype=np.float64)
    out = np.interp(indices, x_old, src, left=src[0], right=src[-1])
    return np.maximum(out, 1e-9)


def _read_rows_from_map(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            points = payload.get("points", [])
            if not isinstance(points, list):
                raise ValueError(f"{path}: JSON 'points' must be a list")
            rows = [p for p in points if isinstance(p, dict)]
        elif isinstance(payload, list):
            rows = [p for p in payload if isinstance(p, dict)]
        else:
            raise ValueError(f"{path}: unsupported JSON control-map structure")
        return [dict(row) for row in rows]

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: v for k, v in row.items()})
    return rows


def load_scalar_control_points(
    path: str | Path | None,
    *,
    key: str,
    default_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load scalar control points from CSV/JSON.

    Supports:
    - point rows: `time_sec,key`
    - segment rows: `start_sec,end_sec,key`
    """
    if path is None:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"Control map not found: {p}")

    rows = _read_rows_from_map(p)
    if not rows:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    points: list[tuple[float, float]] = []
    for row in rows:
        if "time_sec" in row and key in row:
            t = float(row["time_sec"])
            v = float(row[key])
            points.append((t, v))
            continue
        if "start_sec" in row and "end_sec" in row and key in row:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            value = float(row[key])
            if end < start:
                start, end = end, start
            points.append((start, value))
            points.append((end, value))
            continue
        if "time_sec" in row and "value" in row and key == "value":
            points.append((float(row["time_sec"]), float(row["value"])))

    if not points:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    points.sort(key=lambda item: item[0])
    dedup_t: list[float] = []
    dedup_v: list[float] = []
    for t, v in points:
        if dedup_t and abs(t - dedup_t[-1]) <= 1e-12:
            dedup_v[-1] = v
        else:
            dedup_t.append(t)
            dedup_v.append(v)

    t_arr = np.asarray(dedup_t, dtype=np.float64)
    v_arr = np.asarray(dedup_v, dtype=np.float64)
    finite = np.isfinite(t_arr) & np.isfinite(v_arr)
    t_arr = t_arr[finite]
    v_arr = v_arr[finite]
    if t_arr.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    v_arr = np.where(np.isfinite(v_arr), v_arr, float(default_value))
    return t_arr, v_arr


def evaluate_scalar_control(
    frame_times_sec: np.ndarray,
    points_t: np.ndarray,
    points_v: np.ndarray,
    *,
    mode: InterpMode = "linear",
    order: int = 3,
    default_value: float = 1.0,
) -> np.ndarray:
    times = np.asarray(frame_times_sec, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return np.zeros(0, dtype=np.float64)

    t = np.asarray(points_t, dtype=np.float64).reshape(-1)
    v = np.asarray(points_v, dtype=np.float64).reshape(-1)
    if t.size == 0 or v.size == 0:
        return np.full(times.shape, float(default_value), dtype=np.float64)
    if t.size != v.size:
        raise ValueError("Control points mismatch: times and values lengths differ")
    if t.size == 1:
        return np.full(times.shape, float(v[0]), dtype=np.float64)

    interp = str(mode).lower()
    if interp in {"none", "stairstep"}:
        idx = np.searchsorted(t, times, side="right") - 1
        idx = np.clip(idx, 0, v.size - 1)
        return v[idx]
    if interp == "nearest":
        idx = np.searchsorted(t, times, side="left")
        idx = np.clip(idx, 0, v.size - 1)
        prev = np.clip(idx - 1, 0, v.size - 1)
        use_prev = np.abs(times - t[prev]) <= np.abs(t[idx] - times)
        out_idx = np.where(use_prev, prev, idx)
        return v[out_idx]
    if interp == "linear":
        return np.interp(times, t, v, left=float(v[0]), right=float(v[-1]))
    if interp == "cubic":
        # Global cubic fit keeps runtime tiny and deterministic for short control maps.
        deg = min(3, v.size - 1)
        coef = np.polyfit(t, v, deg=deg)
        out = np.polyval(coef, times)
        return np.asarray(out, dtype=np.float64)
    if interp == "exponential":
        seg_idx, _, u = _piecewise_segment_fraction(times, t, side="right")
        eased = _exp_ease(u, strength=4.0)
        return v[seg_idx] + (v[seg_idx + 1] - v[seg_idx]) * eased
    if interp == "s_curve":
        seg_idx, _, u = _piecewise_segment_fraction(times, t, side="right")
        eased = _smoothstep(u)
        return v[seg_idx] + (v[seg_idx + 1] - v[seg_idx]) * eased
    if interp == "smootherstep":
        seg_idx, _, u = _piecewise_segment_fraction(times, t, side="right")
        eased = _smootherstep(u)
        return v[seg_idx] + (v[seg_idx + 1] - v[seg_idx]) * eased
    if interp == "polynomial":
        deg = max(1, min(int(order), v.size - 1))
        coef = np.polyfit(t, v, deg=deg)
        out = np.polyval(coef, times)
        return np.asarray(out, dtype=np.float64)
    raise ValueError(f"Unsupported interpolation mode: {mode}")


def _frame_times(n_frames: int, hop_size: int, sample_rate: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    return (np.arange(n_frames, dtype=np.float64) * float(hop_size)) / float(sample_rate)


def _blend_dry_wet(dry: np.ndarray, wet: np.ndarray, dry_mix: float) -> np.ndarray:
    alpha = float(np.clip(dry_mix, 0.0, 1.0))
    return (alpha * dry) + ((1.0 - alpha) * wet)


def _compute_band_shape(curve: np.ndarray, *, peak_count: int, width_bins: int) -> np.ndarray:
    src = np.asarray(curve, dtype=np.float64).reshape(-1)
    n = src.size
    if n == 0:
        return src
    count = int(np.clip(peak_count, 1, n))
    idx = np.argpartition(src, -count)[-count:]
    idx = np.sort(idx)
    sigma = max(1.0, float(width_bins))
    x = np.arange(n, dtype=np.float64)
    band = np.zeros(n, dtype=np.float64)
    for peak in idx:
        band += np.exp(-0.5 * np.square((x - float(peak)) / sigma))
    mx = float(np.max(band))
    if mx > 1e-12:
        band /= mx
    return band


def process_response_operator(
    audio: np.ndarray,
    sample_rate: int,
    config: VocoderConfig,
    response: ResponseArtifact,
    *,
    operator: OperatorName,
    response_mix: float = 1.0,
    dry_mix: float = 0.0,
    response_gain_db: float = 0.0,
    transpose_semitones: float = 0.0,
    shift_bins: int = 0,
    tv_points_t: np.ndarray | None = None,
    tv_points_v: np.ndarray | None = None,
    tv_interp: InterpMode = "linear",
    tv_order: int = 3,
    noise_floor: float = 1.0,
    band_gain_db: float = 6.0,
    band_width_bins: int = 6,
    peak_count: int = 8,
    comp_threshold_db: float = -18.0,
    comp_ratio: float = 2.0,
    expand_ratio: float = 1.2,
) -> np.ndarray:
    """Apply PVC-inspired response operator to mono/multichannel audio."""
    work = coerce_audio(audio)
    out = np.zeros_like(work)
    channels = work.shape[1]

    for ch in range(channels):
        signal = work[:, ch]
        spectrum = stft(signal, config)
        mag = np.abs(spectrum)
        pha = np.angle(spectrum)
        n_bins, n_frames = mag.shape
        if n_bins <= 0 or n_frames <= 0:
            out[:, ch] = signal
            continue

        src_idx = min(ch, int(response.channels) - 1)
        response_curve = _resize_curve(response.magnitude[src_idx], n_bins)
        response_curve = _shift_response_curve(
            response_curve,
            shift_bins=int(shift_bins),
            transpose_semitones=float(transpose_semitones),
        )
        response_curve *= db_to_amp(float(response_gain_db))
        response_curve = np.maximum(response_curve, 1e-9)

        frame_times = _frame_times(n_frames, config.hop_size, sample_rate)
        # mix_vec is per-frame modulation depth; tvfilter consumes map dynamics here.
        mix_vec = evaluate_scalar_control(
            frame_times,
            np.asarray(tv_points_t if tv_points_t is not None else [], dtype=np.float64),
            np.asarray(tv_points_v if tv_points_v is not None else [], dtype=np.float64),
            mode=tv_interp,
            order=tv_order,
            default_value=float(response_mix),
        )
        mix_vec = np.clip(mix_vec, 0.0, 4.0)

        if operator in {"filter", "tvfilter"}:
            # Response curve is interpreted as multiplicative magnitude shaping.
            gain = 1.0 + (response_curve[:, None] - 1.0) * mix_vec[None, :]
            out_mag = mag * np.maximum(gain, 1e-9)
        elif operator == "noisefilter":
            floor = max(1e-9, float(noise_floor))
            threshold = floor * response_curve[:, None]
            atten = np.clip(mag / np.maximum(threshold, 1e-12), 0.0, 1.0)
            den = mag * atten
            out_mag = (1.0 - mix_vec[None, :]) * mag + mix_vec[None, :] * den
        elif operator == "bandamp":
            # bandamp builds broad Gaussian-like boosts around dominant response peaks.
            band_shape = _compute_band_shape(
                response_curve,
                peak_count=int(peak_count),
                width_bins=max(1, int(band_width_bins)),
            )
            boost = db_to_amp(float(band_gain_db))
            gain_curve = 1.0 + (boost - 1.0) * band_shape
            gain = 1.0 + (gain_curve[:, None] - 1.0) * mix_vec[None, :]
            out_mag = mag * np.maximum(gain, 1e-9)
        elif operator == "spec-compander":
            # spec-compander works in "relative-to-response" space, then maps back to absolute.
            ref = np.maximum(response_curve[:, None], 1e-9)
            rel = mag / ref
            threshold = max(1e-6, db_to_amp(float(comp_threshold_db)))
            comp = max(1.0, float(comp_ratio))
            expn = max(1.0, float(expand_ratio))

            rel_out = rel.copy()
            over = rel > threshold
            under = rel < threshold
            rel_out[over] = threshold + (rel[over] - threshold) / comp
            rel_out[under] = threshold * np.power(np.maximum(rel[under], 1e-12) / threshold, expn)
            shaped = rel_out * ref
            out_mag = (1.0 - mix_vec[None, :]) * mag + mix_vec[None, :] * shaped
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        out_spec = out_mag * np.exp(1j * pha)
        wet = istft(out_spec, config, expected_length=signal.size)
        out[:, ch] = _blend_dry_wet(signal, wet, dry_mix=float(dry_mix))

    return out
